//! Stack-based bytecode execution mode.
//!
//! The tree-walking interpreter in the crate root re-traverses the AST on
//! every call. This module offers a second execution mode: [`compile_fn`]
//! lowers a function's body once into a flat array of [`Instr`]uctions, and
//! [`run`] executes that array on a virtual operand stack.
//!
//! Both modes coexist in a single [`Runtime`]. A compiled function is wrapped
//! into an ordinary `Any::Fn` value ([`CompiledFn::into_function`]) speaking
//! the same `(result, args_consumed)` protocol as AST-defined and
//! host-registered functions, so compiled code can call interpreted code and
//! vice versa — including currying/partial application. Enable per-runtime
//! with [`Runtime::set_compile_fns`].
//!
//! The compiler produces [`Instr`] values (a plain Rust enum) and then
//! [`encode`]s them into [`Code`], a flat byte array executed directly. The
//! opcode byte determines the instruction's exact shape: operand widths are
//! packed into the opcode (u8/u16/u32 variants), the smallest slot values
//! and [`SMALL_INTS`] integers ride in the opcode itself, `True`/`False`/
//! `Nil` and operator kinds have dedicated opcodes, `x ± small-int` fuses
//! into one `ADD_INT`/`SUB_INT` instruction, larger integer immediates
//! carry a sign- or zero-extended payload (never the const pool), and jump
//! targets are fixed 4-byte little-endian byte offsets. [`Code::disassemble`]
//! decodes back to `Instr` for inspection. Operands are indices into pools shared by all of
//! a runtime's compiled functions — constants, names and patterns are interned once into the
//! [`VmContext`] living on the [`Runtime`] — or absolute jump targets into
//! the function's own `code`. The context also owns the operand stack all
//! compiled frames execute on; frames address it relative to a base
//! recorded on entry and truncate back on exit.
//!
//! Locals are compile-time-resolved frame slots on that same stack (see
//! [`SlotTable`]): the arguments a caller leaves on the stack are the
//! frame's first slots, body-introduced names extend the frame, and reads
//! of a not-yet-assigned local fall back to the global of the same name —
//! so compiled functions never touch the runtime's name-keyed scopes.

use alloc::{
    boxed::Box,
    collections::BTreeMap,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};
use core::fmt;
use std::collections::{HashMap, HashSet};

use raft_ast::{BinOpKind, Expr, ExprKind, Lit, Pat, PatKind, Span, Stmt, StmtKind, UnOpKind};

use crate::{
    Any, Atom, FnValue, Function, Number, ObjectKind, Runtime, RuntimeError, assign_field,
    assign_index, eval_binary, eval_unary, field_of, index_of, is_falsey, literal_value,
};

/// One virtual-machine instruction.
///
/// The machine state is: an operand stack of [`Any`] values, an iterator
/// stack driving active `for` loops, and a program counter. Raft's "a
/// function yields the value of its body's last statement" rule is resolved
/// statically: the compiler flags tail-position statements, and only those
/// leave their value on the operand stack for `Return` to pick up.
///
/// Stack effects below are written `before → after`, top of stack rightmost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Instr {
    /// `→ const` — push `consts[i]`.
    Const(u32),
    /// `→ Nil` — push the `Nil` atom.
    Nil,
    /// `→ True` — push the `True` atom.
    True,
    /// `→ False` — push the `False` atom.
    False,
    /// `→ n` — push an integer immediate. Values in [`SMALL_INTS`] ride
    /// entirely in the opcode; anything else is encoded as the smallest
    /// signed or unsigned payload that holds it (`INT_8`..`INT_64`,
    /// `UINT_8`..`UINT_32`). Integers never touch the const pool.
    Int(i64),
    /// `v →` — pop and discard (an expression statement whose value is
    /// not the function's result).
    Pop,
    /// `→ v` — push a copy of frame slot `slot`. Only emitted for names
    /// the compiler proved always-initialized (parameters and names bound
    /// by the parameter-destructuring prologue).
    LoadSlot(u32),
    /// `→ v` — push a copy of frame slot `slot`; if the slot is still
    /// unassigned, fall back to reading global `names[name]` (a body-local
    /// name may be read before its first assignment, which reaches the
    /// global of the same name).
    LoadLocal { slot: u32, name: u32 },
    /// `→ v` — push global variable `names[i]`; error if unbound. Used for
    /// names never assigned in the function — they can only be globals.
    LoadGlobal(u32),
    /// `v →` — pop and store into frame slot `slot` (assignments inside a
    /// function always target locals).
    StoreLocal(u32),
    /// `v →` — pop and bind against pattern `pats[i]`, which may bind
    /// several frame slots (destructuring) or fail the match with an error.
    Bind(u32),
    /// `v1 .. vn → list` — pop `n` values, push a new list of them.
    MakeList(u32),
    /// `k1 v1 .. kn vn → record` — pop `n` key/value pairs (keys are
    /// string constants pushed by the compiler), push a new record.
    MakeRecord(u32),
    /// `a → op(a)` — apply a unary operator.
    Unary(UnOpKind),
    /// `a b → a op b` — apply a binary operator.
    Binary(BinOpKind),
    /// `a → a op n` — apply a binary operator whose right operand is an
    /// integer carried by the opcode. `Add`/`Sub` take values from
    /// [`SMALL_INTS`], `BitAnd`/`BitOr`/`BitXor` from [`TINY_MASKS`], and
    /// the six comparisons from [`TINY_INTS`].
    BinaryInt(BinOpKind, i16),
    /// `→` — add/subtract 1 to frame slot `slot` in place (no stack
    /// traffic). Falls back to the global `names[name]` when the slot is
    /// still unassigned, exactly like `LoadLocal`.
    IncSlot { slot: u32, name: u32 },
    DecSlot { slot: u32, name: u32 },
    /// `obj → obj.names[i]` — read a record field.
    GetField(u32),
    /// `obj idx → obj[idx]` — read a list element.
    GetIndex,
    /// `obj v →` — write record field `names[i]`.
    SetField(u32),
    /// `obj idx v →` — write a list element.
    SetIndex,
    /// `f a1 .. an → ret` — apply `f` to `n` arguments with the language's
    /// currying rules: a callee consuming fewer than `n` arguments has the
    /// leftovers re-applied to its result.
    Call(u32),
    /// `v → ret` — a value in call position with no arguments: if `v` is a
    /// zero-argument function it is invoked, otherwise it passes through.
    CallBare,
    /// `→` — unconditional jump to code index `t`.
    Jump(u32),
    /// `c →` — pop; jump to `t` if the value is falsey.
    JumpIfFalse(u32),
    /// `iterable →` — pop, open an iterator over it and push it on the
    /// iterator stack.
    IterInit,
    /// `→ item` *or* jump — advance the innermost iterator: push the next
    /// item, or (when exhausted) close the iterator and jump to `t`.
    IterNext(u32),
    /// `→` — close the innermost iterator (used by `break` in a `for`).
    IterPop,
    /// `v →` — pop the return value and leave the function.
    Return,
}

/// One byte per instruction, with the operand width — or the operand
/// itself — packed into the opcode.
///
/// Every single-operand instruction owns a block of 11 consecutive
/// opcodes: `base+0..=base+7` carry the operand value inline (the whole
/// instruction is one byte), and `base+8`/`base+9`/`base+10` are followed
/// by a `u8`/`u16`/`u32` little-endian operand. The opcode alone therefore
/// determines the instruction's length — no continuation bits to chase.
/// Unary/binary operators get one opcode per kind, jump targets are fixed
/// 4-byte little-endian **byte offsets** (so patching never changes
/// instruction widths).
mod opcode {
    /// Frame-local operands (slots) and counts are small by nature, so
    /// these blocks spend 11 opcodes each: `base+0..=base+7` carry the
    /// operand inline, `base+8/9/10` take a u8/u16/u32 payload.
    pub const LOAD_SLOT: u8 = 0;
    pub const LOAD_SLOT_END: u8 = LOAD_SLOT + 10;
    pub const STORE_LOCAL: u8 = 11;
    pub const STORE_LOCAL_END: u8 = STORE_LOCAL + 10;
    pub const MAKE_LIST: u8 = 22;
    pub const MAKE_LIST_END: u8 = MAKE_LIST + 10;
    pub const MAKE_RECORD: u8 = 33;
    pub const MAKE_RECORD_END: u8 = MAKE_RECORD + 10;
    pub const CALL: u8 = 44;
    pub const CALL_END: u8 = CALL + 10;

    /// Operands indexing the shared `VmContext` pools grow with the whole
    /// program, so inline values would be wasted opcodes — these blocks
    /// only encode the operand's byte width: `base+0/1/2` = u8/u16/u32.
    pub const CONST: u8 = 55;
    pub const CONST_END: u8 = CONST + 2;
    pub const LOAD_GLOBAL: u8 = 58;
    pub const LOAD_GLOBAL_END: u8 = LOAD_GLOBAL + 2;
    pub const BIND: u8 = 61;
    pub const BIND_END: u8 = BIND + 2;
    pub const GET_FIELD: u8 = 64;
    pub const GET_FIELD_END: u8 = GET_FIELD + 2;
    pub const SET_FIELD: u8 = 67;
    pub const SET_FIELD_END: u8 = SET_FIELD + 2;

    /// Two operands: a block of 11 × 3 opcodes,
    /// `LOAD_LOCAL + slot_variant * 3 + name_width`. The slot follows the
    /// same principle as LOAD_SLOT (8 inline values, then u8/u16/u32); the
    /// name is a pool index and carries only its byte width.
    pub const LOAD_LOCAL: u8 = 70;
    pub const LOAD_LOCAL_END: u8 = LOAD_LOCAL + 32;

    /// One opcode per operator kind — no operand bytes at all.
    pub const UNARY: u8 = 103; // + unop_to_byte(kind), 4 kinds
    pub const UNARY_END: u8 = UNARY + 3;
    pub const BINARY: u8 = 107; // + binop_to_byte(kind), 16 kinds
    pub const BINARY_END: u8 = BINARY + 15;

    /// No operands.
    pub const NIL: u8 = 123;
    pub const POP: u8 = 124;
    pub const GET_INDEX: u8 = 125;
    pub const SET_INDEX: u8 = 126;
    pub const CALL_BARE: u8 = 127;
    pub const ITER_INIT: u8 = 128;
    pub const ITER_POP: u8 = 129;
    pub const RETURN: u8 = 130;

    /// Fixed 4-byte little-endian byte-offset operand.
    pub const JUMP: u8 = 131;
    pub const JUMP_IF_FALSE: u8 = 132;
    pub const ITER_NEXT: u8 = 133;

    /// Immediate values: the singleton atoms, and one opcode per entry of
    /// [`super::SMALL_INTS`] (`INT + index`). No operand bytes and no
    /// const-pool access at runtime.
    pub const TRUE: u8 = 134;
    pub const FALSE: u8 = 135;
    pub const INT: u8 = 136;
    pub const INT_END: u8 = INT + super::SMALL_INTS.len() as u8 - 1;

    /// Fused binary ops with a small-integer right operand: `base + index`
    /// into [`super::SMALL_INTS`]. One instruction, no operand bytes, no
    /// const-pool access.
    pub const ADD_INT: u8 = 148;
    pub const ADD_INT_END: u8 = ADD_INT + super::SMALL_INTS.len() as u8 - 1;
    pub const SUB_INT: u8 = 160;
    pub const SUB_INT_END: u8 = SUB_INT + super::SMALL_INTS.len() as u8 - 1;

    /// Integer immediates too large for [`super::SMALL_INTS`] but fitting
    /// 16 bits: the value follows as a little-endian payload,
    /// sign-extended (`INT_*`) or zero-extended (`UINT_*`) to `i64`.
    /// Anything larger lives in the const pool.
    pub const INT_8: u8 = 172;
    pub const INT_16: u8 = 173;
    pub const UINT_8: u8 = 174;
    pub const UINT_16: u8 = 175;

    /// Fused bitwise ops with a mask from [`super::TINY_MASKS`]
    /// (`base + index`), no operand bytes.
    pub const AND_MASK: u8 = 176;
    pub const AND_MASK_END: u8 = AND_MASK + super::TINY_MASKS.len() as u8 - 1;
    pub const OR_MASK: u8 = 180;
    pub const OR_MASK_END: u8 = OR_MASK + super::TINY_MASKS.len() as u8 - 1;
    pub const XOR_MASK: u8 = 184;
    pub const XOR_MASK_END: u8 = XOR_MASK + super::TINY_MASKS.len() as u8 - 1;

    /// Fused comparisons with an integer from [`super::TINY_INTS`]
    /// (`base + index`), no operand bytes.
    pub const INT_EQ: u8 = 188;
    pub const INT_EQ_END: u8 = INT_EQ + super::TINY_INTS.len() as u8 - 1;
    pub const INT_NE: u8 = 194;
    pub const INT_NE_END: u8 = INT_NE + super::TINY_INTS.len() as u8 - 1;
    pub const INT_LT: u8 = 200;
    pub const INT_LT_END: u8 = INT_LT + super::TINY_INTS.len() as u8 - 1;
    pub const INT_GT: u8 = 206;
    pub const INT_GT_END: u8 = INT_GT + super::TINY_INTS.len() as u8 - 1;
    pub const INT_LE: u8 = 212;
    pub const INT_LE_END: u8 = INT_LE + super::TINY_INTS.len() as u8 - 1;
    pub const INT_GE: u8 = 218;
    pub const INT_GE_END: u8 = INT_GE + super::TINY_INTS.len() as u8 - 1;

    /// Whole-statement fusions of `x = x + 1` / `x = x - 1`: 11-opcode
    /// blocks where the slot follows the LOAD_SLOT principle (8 inline
    /// values + u8/u16/u32 payload), followed by a fixed `u16` name index
    /// for the unassigned-slot global fallback. The compiler falls back to
    /// the unfused sequence when the name index does not fit.
    pub const INC_SLOT: u8 = 224;
    pub const INC_SLOT_END: u8 = INC_SLOT + 10;
    pub const DEC_SLOT: u8 = 235;
    pub const DEC_SLOT_END: u8 = DEC_SLOT + 10;
}

/// Bitmasks common enough to deserve their own opcodes for bit operations.
pub const TINY_MASKS: [i64; 4] = [1, 3, 7, 15];

/// Integer values common enough to deserve their own opcodes for many operations.
pub const TINY_INTS: [i64; 6] = [-1, 0, 1, 2, 3, 4];

/// Integer values common enough to deserve their own opcodes.
pub const SMALL_INTS: [i64; 12] = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

/// Whether an integer can be an immediate at all (larger values live in
/// the const pool).
pub(crate) fn int_fits_immediate(n: i64) -> bool {
    SMALL_INTS.contains(&n) || u16::try_from(n).is_ok() || i16::try_from(n).is_ok()
}

/// Payload bytes an integer immediate needs (0 = rides in the opcode).
fn int_payload_len(n: i64) -> usize {
    if SMALL_INTS.contains(&n) {
        0
    } else if u8::try_from(n).is_ok() || i8::try_from(n).is_ok() {
        1
    } else {
        2
    }
}

/// Emit an integer immediate with the smallest encoding: SMALL_INTS ride
/// in the opcode, everything else prefers unsigned then signed payloads.
fn push_int(out: &mut Vec<u8>, n: i64) {
    if let Some(idx) = SMALL_INTS.iter().position(|&s| s == n) {
        out.push(opcode::INT + idx as u8);
    } else if let Ok(v) = u8::try_from(n) {
        out.push(opcode::UINT_8);
        out.push(v);
    } else if let Ok(v) = i8::try_from(n) {
        out.push(opcode::INT_8);
        out.push(v as u8);
    } else if let Ok(v) = u16::try_from(n) {
        out.push(opcode::UINT_16);
        out.extend_from_slice(&v.to_le_bytes());
    } else if let Ok(v) = i16::try_from(n) {
        out.push(opcode::INT_16);
        out.extend_from_slice(&v.to_le_bytes());
    } else {
        unreachable!("Instr::Int value outside the immediate range");
    }
}

/// Extra bytes a single-operand instruction needs for `v` (0 when the
/// value rides inline in the opcode).
#[inline]
fn operand_len(v: u32) -> usize {
    match v {
        0..=7 => 0,
        8..=0xff => 1,
        0x100..=0xffff => 2,
        _ => 4,
    }
}

/// Variant selector for a slot-style operand: `0..=7` mean "the value is
/// this variant" (inline), `8`/`9`/`10` mean a `u8`/`u16`/`u32` payload.
#[inline]
fn operand_variant(v: u32) -> u8 {
    match v {
        0..=7 => v as u8,
        8..=0xff => 8,
        0x100..=0xffff => 9,
        _ => 10,
    }
}

/// Emit the payload bytes for `operand_variant(v)` (nothing when inline).
fn push_payload(out: &mut Vec<u8>, variant: u8, v: u32) {
    match variant {
        0..=7 => {}
        8 => out.push(v as u8),
        9 => out.extend_from_slice(&(v as u16).to_le_bytes()),
        _ => out.extend_from_slice(&v.to_le_bytes()),
    }
}

/// Emit a single-operand instruction from its block `base`, choosing the
/// smallest encoding.
fn push_op(out: &mut Vec<u8>, base: u8, v: u32) {
    let variant = operand_variant(v);
    out.push(base + variant);
    push_payload(out, variant, v);
}

/// Decode a single-operand instruction's operand, given the opcode's
/// offset within its block.
#[inline]
fn decode_operand(rel: u8, code: &[u8], pc: &mut usize) -> Result<u32, RuntimeError> {
    match rel {
        0..=7 => Ok(rel as u32),
        8 => Ok(read_u8(code, pc)? as u32),
        9 => Ok(read_u16(code, pc)? as u32),
        _ => read_u32(code, pc),
    }
}

/// Width index for an operand encoded as raw bytes only (no inline forms):
/// `0`/`1`/`2` = `u8`/`u16`/`u32`.
#[inline]
fn width_idx(v: u32) -> u8 {
    match v {
        0..=0xff => 0,
        0x100..=0xffff => 1,
        _ => 2,
    }
}

/// Payload size in bytes for a `width_idx` value.
#[inline]
fn width_len(idx: u8) -> usize {
    match idx {
        0 => 1,
        1 => 2,
        _ => 4,
    }
}

/// Emit a width-only instruction (pool-index operand) from its block
/// `base`, choosing the smallest payload width.
fn push_wop(out: &mut Vec<u8>, base: u8, v: u32) {
    match width_idx(v) {
        0 => {
            out.push(base);
            out.push(v as u8);
        }
        1 => {
            out.push(base + 1);
            out.extend_from_slice(&(v as u16).to_le_bytes());
        }
        _ => {
            out.push(base + 2);
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
}

/// Decode a width-only instruction's operand, given the opcode's offset
/// within its block.
#[inline]
fn decode_wop(rel: u8, code: &[u8], pc: &mut usize) -> Result<u32, RuntimeError> {
    match rel {
        0 => Ok(read_u8(code, pc)? as u32),
        1 => Ok(read_u16(code, pc)? as u32),
        _ => read_u32(code, pc),
    }
}

#[inline]
fn read_u8(code: &[u8], pc: &mut usize) -> Result<u8, RuntimeError> {
    let byte = *code.get(*pc).ok_or_else(truncated)?;
    *pc += 1;
    Ok(byte)
}

#[inline]
fn read_u16(code: &[u8], pc: &mut usize) -> Result<u16, RuntimeError> {
    let bytes = code.get(*pc..*pc + 2).ok_or_else(truncated)?;
    *pc += 2;
    Ok(u16::from_le_bytes(bytes.try_into().unwrap()))
}

#[inline]
fn read_u32(code: &[u8], pc: &mut usize) -> Result<u32, RuntimeError> {
    let bytes = code.get(*pc..*pc + 4).ok_or_else(truncated)?;
    *pc += 4;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}


fn truncated() -> RuntimeError {
    RuntimeError::Other("vm: truncated bytecode".to_string())
}

fn unop_to_byte(k: UnOpKind) -> u8 {
    match k {
        UnOpKind::Not => 0,
        UnOpKind::BitNot => 1,
        UnOpKind::Pos => 2,
        UnOpKind::Neg => 3,
    }
}

fn byte_to_unop(b: u8) -> Result<UnOpKind, RuntimeError> {
    Ok(match b {
        0 => UnOpKind::Not,
        1 => UnOpKind::BitNot,
        2 => UnOpKind::Pos,
        3 => UnOpKind::Neg,
        _ => return Err(RuntimeError::Other("vm: bad unary op".to_string())),
    })
}

fn binop_to_byte(k: BinOpKind) -> u8 {
    match k {
        BinOpKind::BitAnd => 0,
        BinOpKind::BitOr => 1,
        BinOpKind::BitXor => 2,
        BinOpKind::Shl => 3,
        BinOpKind::Shr => 4,
        BinOpKind::Pow => 5,
        BinOpKind::Mul => 6,
        BinOpKind::Div => 7,
        BinOpKind::Add => 8,
        BinOpKind::Sub => 9,
        BinOpKind::Eq => 10,
        BinOpKind::Ne => 11,
        BinOpKind::Lt => 12,
        BinOpKind::Gt => 13,
        BinOpKind::Le => 14,
        BinOpKind::Ge => 15,
    }
}

fn byte_to_binop(b: u8) -> Result<BinOpKind, RuntimeError> {
    Ok(match b {
        0 => BinOpKind::BitAnd,
        1 => BinOpKind::BitOr,
        2 => BinOpKind::BitXor,
        3 => BinOpKind::Shl,
        4 => BinOpKind::Shr,
        5 => BinOpKind::Pow,
        6 => BinOpKind::Mul,
        7 => BinOpKind::Div,
        8 => BinOpKind::Add,
        9 => BinOpKind::Sub,
        10 => BinOpKind::Eq,
        11 => BinOpKind::Ne,
        12 => BinOpKind::Lt,
        13 => BinOpKind::Gt,
        14 => BinOpKind::Le,
        15 => BinOpKind::Ge,
        _ => return Err(RuntimeError::Other("vm: bad binary op".to_string())),
    })
}

/// Encoded size of one instruction: opcode byte + operands.
fn instr_len(i: &Instr) -> usize {
    1 + match i {
        Instr::LoadSlot(v)
        | Instr::StoreLocal(v)
        | Instr::MakeList(v)
        | Instr::MakeRecord(v)
        | Instr::Call(v) => operand_len(*v),
        Instr::Const(v)
        | Instr::LoadGlobal(v)
        | Instr::Bind(v)
        | Instr::GetField(v)
        | Instr::SetField(v) => width_len(width_idx(*v)),
        Instr::LoadLocal { slot, name } => {
            operand_len(*slot) + width_len(width_idx(*name))
        }
        Instr::Unary(_) | Instr::Binary(_) | Instr::BinaryInt(..) => 0,
        Instr::IncSlot { slot, .. } | Instr::DecSlot { slot, .. } => operand_len(*slot) + 2,
        Instr::Jump(_) | Instr::JumpIfFalse(_) | Instr::IterNext(_) => 4,
        Instr::Int(n) => int_payload_len(*n),
        Instr::Nil
        | Instr::True
        | Instr::False
        | Instr::Pop
        | Instr::GetIndex
        | Instr::SetIndex
        | Instr::CallBare
        | Instr::IterInit
        | Instr::IterPop
        | Instr::Return => 0,
    }
}

/// A function's instructions in their executable form: a flat byte array.
/// Jump operands inside are byte offsets into this array. Decode back to
/// [`Instr`] with [`Code::disassemble`] (the `Debug` impl prints a
/// listing).
pub struct Code {
    bytes: Box<[u8]>,
}

/// Encode instructions into bytes. Jump operands come in as instruction
/// indices (the compiler's view) and leave as byte offsets: pass one lays
/// out every instruction's offset — operand widths depend only on values
/// known up front — and pass two emits with targets mapped through that
/// layout.
pub fn encode(instrs: &[Instr]) -> Code {
    let mut offsets = Vec::with_capacity(instrs.len() + 1);
    let mut off: u32 = 0;
    for i in instrs {
        offsets.push(off);
        off += instr_len(i) as u32;
    }
    // a jump may target one past the final instruction
    offsets.push(off);

    let mut bytes = Vec::with_capacity(off as usize);
    for i in instrs {
        match *i {
            Instr::Const(v) => push_wop(&mut bytes, opcode::CONST, v),
            Instr::Nil => bytes.push(opcode::NIL),
            Instr::True => bytes.push(opcode::TRUE),
            Instr::False => bytes.push(opcode::FALSE),
            Instr::Int(n) => push_int(&mut bytes, n),
            Instr::Pop => bytes.push(opcode::POP),
            Instr::LoadSlot(v) => push_op(&mut bytes, opcode::LOAD_SLOT, v),
            Instr::LoadLocal { slot, name } => {
                let sv = operand_variant(slot);
                let nw = width_idx(name);
                bytes.push(opcode::LOAD_LOCAL + sv * 3 + nw);
                push_payload(&mut bytes, sv, slot);
                match nw {
                    0 => bytes.push(name as u8),
                    1 => bytes.extend_from_slice(&(name as u16).to_le_bytes()),
                    _ => bytes.extend_from_slice(&name.to_le_bytes()),
                }
            }
            Instr::LoadGlobal(v) => push_wop(&mut bytes, opcode::LOAD_GLOBAL, v),
            Instr::StoreLocal(v) => push_op(&mut bytes, opcode::STORE_LOCAL, v),
            Instr::Bind(v) => push_wop(&mut bytes, opcode::BIND, v),
            Instr::MakeList(v) => push_op(&mut bytes, opcode::MAKE_LIST, v),
            Instr::MakeRecord(v) => push_op(&mut bytes, opcode::MAKE_RECORD, v),
            Instr::Unary(k) => bytes.push(opcode::UNARY + unop_to_byte(k)),
            Instr::Binary(k) => bytes.push(opcode::BINARY + binop_to_byte(k)),
            Instr::BinaryInt(k, v) => {
                let (base, table): (u8, &[i64]) = match k {
                    BinOpKind::Add => (opcode::ADD_INT, &SMALL_INTS),
                    BinOpKind::Sub => (opcode::SUB_INT, &SMALL_INTS),
                    BinOpKind::BitAnd => (opcode::AND_MASK, &TINY_MASKS),
                    BinOpKind::BitOr => (opcode::OR_MASK, &TINY_MASKS),
                    BinOpKind::BitXor => (opcode::XOR_MASK, &TINY_MASKS),
                    BinOpKind::Eq => (opcode::INT_EQ, &TINY_INTS),
                    BinOpKind::Ne => (opcode::INT_NE, &TINY_INTS),
                    BinOpKind::Lt => (opcode::INT_LT, &TINY_INTS),
                    BinOpKind::Gt => (opcode::INT_GT, &TINY_INTS),
                    BinOpKind::Le => (opcode::INT_LE, &TINY_INTS),
                    BinOpKind::Ge => (opcode::INT_GE, &TINY_INTS),
                    _ => unreachable!("BinaryInt kind has no encoding"),
                };
                let idx = table
                    .iter()
                    .position(|&n| n == v as i64)
                    .expect("BinaryInt value not in its kind's table");
                bytes.push(base + idx as u8);
            }
            Instr::IncSlot { slot, name } | Instr::DecSlot { slot, name } => {
                let base = if matches!(*i, Instr::IncSlot { .. }) {
                    opcode::INC_SLOT
                } else {
                    opcode::DEC_SLOT
                };
                let variant = operand_variant(slot);
                bytes.push(base + variant);
                push_payload(&mut bytes, variant, slot);
                bytes.extend_from_slice(
                    &u16::try_from(name)
                        .expect("IncSlot/DecSlot name must fit u16")
                        .to_le_bytes(),
                );
            }
            Instr::GetField(v) => push_wop(&mut bytes, opcode::GET_FIELD, v),
            Instr::GetIndex => bytes.push(opcode::GET_INDEX),
            Instr::SetField(v) => push_wop(&mut bytes, opcode::SET_FIELD, v),
            Instr::SetIndex => bytes.push(opcode::SET_INDEX),
            Instr::Call(v) => push_op(&mut bytes, opcode::CALL, v),
            Instr::CallBare => bytes.push(opcode::CALL_BARE),
            Instr::Jump(t) => {
                bytes.push(opcode::JUMP);
                bytes.extend_from_slice(&offsets[t as usize].to_le_bytes());
            }
            Instr::JumpIfFalse(t) => {
                bytes.push(opcode::JUMP_IF_FALSE);
                bytes.extend_from_slice(&offsets[t as usize].to_le_bytes());
            }
            Instr::IterInit => bytes.push(opcode::ITER_INIT),
            Instr::IterNext(t) => {
                bytes.push(opcode::ITER_NEXT);
                bytes.extend_from_slice(&offsets[t as usize].to_le_bytes());
            }
            Instr::IterPop => bytes.push(opcode::ITER_POP),
            Instr::Return => bytes.push(opcode::RETURN),
        }
    }

    Code {
        bytes: bytes.into_boxed_slice(),
    }
}

impl Code {
    #[inline]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Decode the instruction at byte offset `pc`; returns it together
    /// with the offset of the next instruction. Jump operands come back as
    /// byte offsets.
    pub fn decode_at(&self, mut pc: usize) -> Result<(Instr, usize), RuntimeError> {
        let code = &self.bytes[..];
        let op = read_u8(code, &mut pc)?;
        let instr = match op {
            opcode::CONST..=opcode::CONST_END => {
                Instr::Const(decode_wop(op - opcode::CONST, code, &mut pc)?)
            }
            opcode::LOAD_SLOT..=opcode::LOAD_SLOT_END => {
                Instr::LoadSlot(decode_operand(op - opcode::LOAD_SLOT, code, &mut pc)?)
            }
            opcode::LOAD_GLOBAL..=opcode::LOAD_GLOBAL_END => {
                Instr::LoadGlobal(decode_wop(op - opcode::LOAD_GLOBAL, code, &mut pc)?)
            }
            opcode::STORE_LOCAL..=opcode::STORE_LOCAL_END => {
                Instr::StoreLocal(decode_operand(op - opcode::STORE_LOCAL, code, &mut pc)?)
            }
            opcode::BIND..=opcode::BIND_END => {
                Instr::Bind(decode_wop(op - opcode::BIND, code, &mut pc)?)
            }
            opcode::MAKE_LIST..=opcode::MAKE_LIST_END => {
                Instr::MakeList(decode_operand(op - opcode::MAKE_LIST, code, &mut pc)?)
            }
            opcode::MAKE_RECORD..=opcode::MAKE_RECORD_END => {
                Instr::MakeRecord(decode_operand(op - opcode::MAKE_RECORD, code, &mut pc)?)
            }
            opcode::GET_FIELD..=opcode::GET_FIELD_END => {
                Instr::GetField(decode_wop(op - opcode::GET_FIELD, code, &mut pc)?)
            }
            opcode::SET_FIELD..=opcode::SET_FIELD_END => {
                Instr::SetField(decode_wop(op - opcode::SET_FIELD, code, &mut pc)?)
            }
            opcode::CALL..=opcode::CALL_END => {
                Instr::Call(decode_operand(op - opcode::CALL, code, &mut pc)?)
            }
            opcode::LOAD_LOCAL..=opcode::LOAD_LOCAL_END => {
                let rel = op - opcode::LOAD_LOCAL;
                let slot = decode_operand(rel / 3, code, &mut pc)?;
                let name = match rel % 3 {
                    0 => read_u8(code, &mut pc)? as u32,
                    1 => read_u16(code, &mut pc)? as u32,
                    _ => read_u32(code, &mut pc)?,
                };
                Instr::LoadLocal { slot, name }
            }
            opcode::UNARY..=opcode::UNARY_END => Instr::Unary(byte_to_unop(op - opcode::UNARY)?),
            opcode::BINARY..=opcode::BINARY_END => {
                Instr::Binary(byte_to_binop(op - opcode::BINARY)?)
            }
            opcode::ADD_INT..=opcode::ADD_INT_END => Instr::BinaryInt(
                BinOpKind::Add,
                SMALL_INTS[(op - opcode::ADD_INT) as usize] as i16,
            ),
            opcode::SUB_INT..=opcode::SUB_INT_END => Instr::BinaryInt(
                BinOpKind::Sub,
                SMALL_INTS[(op - opcode::SUB_INT) as usize] as i16,
            ),
            opcode::AND_MASK..=opcode::AND_MASK_END => Instr::BinaryInt(
                BinOpKind::BitAnd,
                TINY_MASKS[(op - opcode::AND_MASK) as usize] as i16,
            ),
            opcode::OR_MASK..=opcode::OR_MASK_END => Instr::BinaryInt(
                BinOpKind::BitOr,
                TINY_MASKS[(op - opcode::OR_MASK) as usize] as i16,
            ),
            opcode::XOR_MASK..=opcode::XOR_MASK_END => Instr::BinaryInt(
                BinOpKind::BitXor,
                TINY_MASKS[(op - opcode::XOR_MASK) as usize] as i16,
            ),
            opcode::INT_EQ..=opcode::INT_EQ_END => Instr::BinaryInt(
                BinOpKind::Eq,
                TINY_INTS[(op - opcode::INT_EQ) as usize] as i16,
            ),
            opcode::INT_NE..=opcode::INT_NE_END => Instr::BinaryInt(
                BinOpKind::Ne,
                TINY_INTS[(op - opcode::INT_NE) as usize] as i16,
            ),
            opcode::INT_LT..=opcode::INT_LT_END => Instr::BinaryInt(
                BinOpKind::Lt,
                TINY_INTS[(op - opcode::INT_LT) as usize] as i16,
            ),
            opcode::INT_GT..=opcode::INT_GT_END => Instr::BinaryInt(
                BinOpKind::Gt,
                TINY_INTS[(op - opcode::INT_GT) as usize] as i16,
            ),
            opcode::INT_LE..=opcode::INT_LE_END => Instr::BinaryInt(
                BinOpKind::Le,
                TINY_INTS[(op - opcode::INT_LE) as usize] as i16,
            ),
            opcode::INT_GE..=opcode::INT_GE_END => Instr::BinaryInt(
                BinOpKind::Ge,
                TINY_INTS[(op - opcode::INT_GE) as usize] as i16,
            ),
            opcode::INC_SLOT..=opcode::INC_SLOT_END => Instr::IncSlot {
                slot: decode_operand(op - opcode::INC_SLOT, code, &mut pc)?,
                name: read_u16(code, &mut pc)? as u32,
            },
            opcode::DEC_SLOT..=opcode::DEC_SLOT_END => Instr::DecSlot {
                slot: decode_operand(op - opcode::DEC_SLOT, code, &mut pc)?,
                name: read_u16(code, &mut pc)? as u32,
            },
            opcode::NIL => Instr::Nil,
            opcode::TRUE => Instr::True,
            opcode::FALSE => Instr::False,
            opcode::INT..=opcode::INT_END => {
                Instr::Int(SMALL_INTS[(op - opcode::INT) as usize])
            }
            opcode::INT_8 => Instr::Int(read_u8(code, &mut pc)? as i8 as i64),
            opcode::INT_16 => Instr::Int(read_u16(code, &mut pc)? as i16 as i64),
            opcode::UINT_8 => Instr::Int(read_u8(code, &mut pc)? as i64),
            opcode::UINT_16 => Instr::Int(read_u16(code, &mut pc)? as i64),
            opcode::POP => Instr::Pop,
            opcode::GET_INDEX => Instr::GetIndex,
            opcode::SET_INDEX => Instr::SetIndex,
            opcode::CALL_BARE => Instr::CallBare,
            opcode::ITER_INIT => Instr::IterInit,
            opcode::ITER_POP => Instr::IterPop,
            opcode::RETURN => Instr::Return,
            opcode::JUMP => Instr::Jump(read_u32(code, &mut pc)?),
            opcode::JUMP_IF_FALSE => Instr::JumpIfFalse(read_u32(code, &mut pc)?),
            opcode::ITER_NEXT => Instr::IterNext(read_u32(code, &mut pc)?),
            _ => return Err(RuntimeError::Other("vm: unknown opcode".to_string())),
        };
        Ok((instr, pc))
    }

    /// Iterate `(byte_offset, instruction)` pairs from start to end.
    pub fn disassemble(&self) -> impl Iterator<Item = Result<(usize, Instr), RuntimeError>> + '_ {
        let mut pc = 0;
        core::iter::from_fn(move || {
            if pc >= self.bytes.len() {
                return None;
            }
            Some(self.decode_at(pc).map(|(instr, next)| {
                let at = pc;
                pc = next;
                (at, instr)
            }))
        })
    }
}

impl fmt::Debug for Code {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Code ({} bytes)", self.bytes.len())?;
        for entry in self.disassemble() {
            match entry {
                Ok((at, instr)) => writeln!(f, "  {at:4}: {instr:?}")?,
                Err(e) => return writeln!(f, "  <decode error: {e}>"),
            }
        }
        Ok(())
    }
}

/// A pattern lowered for execution. The AST [`Pat`] is built for parsing —
/// spans everywhere, literals still in source text; this form does all of
/// that work once at compile time: number literals are pre-parsed, string
/// and char literals pre-unescaped, record-shorthand fields normalized to
/// explicit `key: Var(key)` bindings, and bound names resolved to frame
/// slot indices of the function the pattern was compiled in.
///
/// `bind` mirrors `Runtime::bind_pattern` (the tree walker's reference
/// implementation) — keep their semantics in sync.
#[derive(Clone, Debug)]
pub enum CompiledPat {
    Ignore,
    /// Bind the whole value to frame slot `slot`.
    Var(u32),
    /// Match an atom by name.
    Atom(Atom),
    /// Match a number literal (see [`NumberPat`] for the semantics the
    /// suffix selects).
    Number(NumberPat),
    /// Match a string literal (pre-unescaped).
    String(Rc<str>),
    /// Match a char literal (pre-unescaped).
    Char(char),
    /// Destructure a list of exactly this shape.
    List(Box<[CompiledPat]>),
    /// Destructure record fields by key.
    Record(Box<[(Rc<str>, CompiledPat)]>),
}

/// How a number literal matches in a pattern. The suffix chooses the
/// strictness: an unsuffixed integer literal matches numerically, while a
/// suffix pins the type — making number patterns usable as type
/// discriminators.
#[derive(Clone, Copy, Debug)]
pub enum NumberPat {
    /// `1i` — matches the integer `1` only, never a float.
    Integer(i64),
    /// `1f`, `1.0`, `1e3` — matches exactly this float (IEEE equality, so
    /// `+inf`/`-inf` match by sign and `-0.0` matches `0.0`; NaN matches
    /// NaN and only NaN). Never matches an integer.
    Float(f64),
    /// `1` — matches the integer `1`, or a float that *is* that integer
    /// (finite, nothing after the dot, exactly convertible).
    Numeric(i64),
    /// An out-of-range literal: matches nothing at all.
    Never,
}

impl NumberPat {
    /// Interpret a number literal as a pattern. Bad suffixes never get
    /// past the parser, so only the spelling and range matter here.
    pub fn from_literal(n: &raft_ast::LitNum) -> NumberPat {
        if n.has_dot() || n.has_exponent() || n.suffix() == Some("f") {
            match n.value().parse::<f64>() {
                Ok(f) => NumberPat::Float(f),
                Err(_) => NumberPat::Never,
            }
        } else {
            match n.value().parse::<i64>() {
                Ok(i) if n.suffix() == Some("i") => NumberPat::Integer(i),
                Ok(i) => NumberPat::Numeric(i),
                Err(_) => NumberPat::Never,
            }
        }
    }

    pub fn matches(&self, actual: Number) -> bool {
        match (self, actual) {
            (NumberPat::Integer(p), Number::Integer(i)) => *p == i,
            (NumberPat::Integer(_), Number::Float(_)) => false,
            (NumberPat::Float(p), Number::Float(f)) => *p == f || (p.is_nan() && f.is_nan()),
            (NumberPat::Float(_), Number::Integer(_)) => false,
            (NumberPat::Numeric(p), Number::Integer(i)) => *p == i,
            (NumberPat::Numeric(p), Number::Float(f)) => {
                // integral f64 values in [-2^63, 2^63) convert to i64
                // exactly; the range guard keeps `as`-cast saturation from
                // faking equality (2^63 is not i64::MAX)
                const LO: f64 = i64::MIN as f64; // -2^63, exactly representable
                const HI: f64 = -(i64::MIN as f64); // 2^63
                f.is_finite() && f.fract() == 0.0 && f >= LO && f < HI && (f as i64) == *p
            }
            (NumberPat::Never, _) => false,
        }
    }
}

/// Compile-time map of a function's variable names to frame slot indices.
/// Parameters occupy the argument slots (positions `0..arity`, numbered so
/// a plain-ident parameter's slot is exactly where its argument lands on
/// the stack — no moves); names introduced by the body extend the same
/// index space from `arity` upward.
pub struct SlotTable {
    map: HashMap<Rc<str>, Slot>,
    reads: HashSet<Rc<str>>,
    total: u32,
}

#[derive(Clone, Copy)]
struct Slot {
    index: u32,
    /// Whether the slot is provably assigned before any load (parameters
    /// and names bound by the parameter prologue); such loads skip the
    /// unset-check and global fallback.
    definite: bool,
}

impl SlotTable {
    fn get(&self, name: &str) -> Option<Slot> {
        self.map.get(name).copied()
    }

    fn add_read(&mut self, name: Rc<str>) {
        self.reads.insert(name);
    }

    fn add_local(&mut self, name: Rc<str>, is_uniform: bool) {
        let definite = is_uniform && !self.reads.contains(&name);

        // first allocation wins: a body re-assignment to a parameter name
        // reuses the parameter's slot
        if !self.map.contains_key(&name) {
            let index = self.total;
            self.map.insert(name, Slot { index, definite });
            self.total += 1;
        }
    }
}

/// Scan parameters and body for every name the function can bind, and
/// assign each a frame slot. Nested `fn` bodies are separate frames — only
/// the nested function's *name* is a local here.
fn collect_slots(params: &[Pat], body: &[Stmt]) -> SlotTable {
    let arity = params.len() as u32;
    let mut table = SlotTable {
        map: HashMap::new(),
        reads: HashSet::new(),
        total: arity,
    };

    // arguments arrive first-on-top, so parameter i's value sits in slot
    // arity-1-i; a later duplicate parameter name shadows an earlier one
    for (i, p) in params.iter().enumerate() {
        if let PatKind::Ident(id) = p.kind() {
            // a `_` parameter still occupies its argument slot, but binds
            // nothing — reads of `_` in the body reach the global scope
            if id.name() == "_" {
                continue;
            }
            table.map.insert(
                id.rc_name(),
                Slot {
                    index: arity - 1 - i as u32,
                    definite: true,
                },
            );
        } else {
            // names inside destructuring params are bound by the prologue,
            // so they are definitely assigned before the body runs
            collect_pat_names(p, &mut table, true);
        }
    }

    collect_stmt_names(body, &mut table, true);
    table
}

fn collect_expr_names(expr: &Expr, table: &mut SlotTable) {
    match expr.kind() {
        ExprKind::Atom(_) | ExprKind::Literal(_) => {}
        ExprKind::Ident(ident) => table.add_read(ident.rc_name()),
        ExprKind::Unary(_, operand) => collect_expr_names(operand, table),
        ExprKind::Binary(lhs, _, rhs) => {
            collect_expr_names(lhs, table);
            collect_expr_names(rhs, table);
        }
        ExprKind::Apply(f, args) => {
            collect_expr_names(f, table);
            for a in args.iter() {
                collect_expr_names(a, table);
            }
        }
        ExprKind::Field(obj, _) => collect_expr_names(obj, table),
        ExprKind::Index(obj, idx) => {
            collect_expr_names(obj, table);
            collect_expr_names(idx, table);
        }
        ExprKind::List(items) => {
            for e in items.iter() {
                collect_expr_names(e, table);
            }
        }
        ExprKind::Record(fields) => {
            for f in fields.iter() {
                if let Some(e) = f.value() {
                    collect_expr_names(e, table);
                } else {
                    // shorthand `{ x }` requires the field but binds nothing
                    table.add_read(f.key().rc_name());
                }
            }
        }
        ExprKind::Parenthesized(expr) => collect_expr_names(expr, table),
    }
}

fn collect_pat_names(pattern: &Pat, table: &mut SlotTable, is_uniform: bool) {
    match pattern.kind() {
        PatKind::Ident(id) if id.name() == "_" => {}
        PatKind::Ident(id) => table.add_local(id.rc_name(), is_uniform),
        PatKind::Atom(_) | PatKind::Literal(_) => {}
        PatKind::List(items) => {
            for p in items.iter() {
                collect_pat_names(p, table, is_uniform);
            }
        }
        PatKind::Record(fields) => {
            for f in fields.iter() {
                match f.pattern() {
                    Some(p) => collect_pat_names(p, table, is_uniform),
                    // shorthand `{ _ }` requires the field but binds nothing
                    None => table.add_local(f.key().rc_name(), is_uniform),
                }
            }
        }
    }
}

fn collect_stmt_names(stmts: &[Stmt], table: &mut SlotTable, is_uniform: bool) {
    for stmt in stmts {
        match stmt.kind() {
            StmtKind::AssignPat { target, value } => {
                collect_expr_names(value, table);
                collect_pat_names(target, table, is_uniform);
            }
            StmtKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                collect_expr_names(cond, table);
                collect_stmt_names(then_branch, table, false);
                if let Some(eb) = else_branch {
                    collect_stmt_names(eb, table, false);
                }
            }
            StmtKind::While {
                cond, body, else_branch,
            } => {
                collect_expr_names(cond, table);
                collect_stmt_names(body, table, false);
                if let Some(eb) = else_branch {
                    collect_stmt_names(eb, table, false);
                }
            }
            StmtKind::For {
                target,
                iterable,
                body,
                else_branch,
            } => {
                collect_expr_names(iterable, table);
                collect_pat_names(target, table, false);
                collect_stmt_names(body, table, false);
                if let Some(eb) = else_branch {
                    collect_stmt_names(eb, table, false);
                }
            }
            // the nested function's body is its own frame
            StmtKind::Fn { name, .. } => {
                table.add_local(name.rc_name(), is_uniform);
            }
            StmtKind::Expr(expr) => {
                collect_expr_names(expr, table);
            }
            StmtKind::AssignField { target, value, .. } => {
                collect_expr_names(target, table);
                collect_expr_names(value, table);
            }
            StmtKind::AssignIndex { target, index, value } => {
                collect_expr_names(target, table);
                collect_expr_names(index, table);
                collect_expr_names(value, table);
            }
            StmtKind::Return(expr) => {
                if let Some(e) = expr {
                    collect_expr_names(e, table);
                }
            }
            StmtKind::Break | StmtKind::Continue => {}
        }
    }
}

/// Lower an AST pattern, resolving bound names to frame slots through
/// `slots`. Infallible: bad suffixes are rejected by the parser before
/// patterns exist, and an out-of-range number literal compiles to a
/// pattern that matches nothing (`NumberPat::Never`) — the same non-match
/// the tree walker produces for it.
pub fn compile_pat(pattern: &Pat, slots: &SlotTable) -> CompiledPat {
    fn slot_of(slots: &SlotTable, name: &str) -> u32 {
        slots
            .get(name)
            .expect("collect_slots missed a pattern name")
            .index
    }

    match pattern.kind() {
        PatKind::Ident(id) if id.name() == "_" => CompiledPat::Ignore,
        PatKind::Ident(id) => CompiledPat::Var(slot_of(slots, id.name())),
        PatKind::Atom(a) => CompiledPat::Atom(Atom::new(a.rc_name())),
        PatKind::Literal(lit) => match lit {
            Lit::Num(n) => CompiledPat::Number(NumberPat::from_literal(n)),
            Lit::Str(s) => CompiledPat::String(Rc::from(s.unescape())),
            Lit::Char(c) => CompiledPat::Char(c.unescape()),
        },
        PatKind::List(items) => {
            CompiledPat::List(items.iter().map(|p| compile_pat(p, slots)).collect())
        }
        PatKind::Record(fields) => {
            let fields = fields
                .iter()
                .filter_map(|f| {
                    let key = f.key().rc_name();
                    let pattern = match f.pattern() {
                        Some(p) => compile_pat(p, slots),
                        // shorthand `{ x }` binds the field to its own name
                        None => CompiledPat::Var(slot_of(slots, &key)),
                    };

                    if let CompiledPat::Ignore = pattern {
                        None
                    } else {
                        Some((key, pattern))
                    }
                })
                .collect::<Box<[_]>>();

            CompiledPat::Record(fields)
        }
    }
}

impl CompiledPat {
    /// Match `val` against this pattern, binding variables into the slots
    /// of the frame based at `base`. Mirrors `Runtime::bind_pattern` —
    /// including its failure behavior of leaving earlier bindings in place.
    pub fn bind(&self, rt: &mut Runtime, base: usize, val: &Any) -> Result<(), RuntimeError> {
        fn fail() -> RuntimeError {
            RuntimeError::Other("pattern match failed".into())
        }

        match self {
            CompiledPat::Ignore => Ok(()),
            CompiledPat::Var(slot) => {
                rt.vm.set_slot(base, *slot, val.clone());
                Ok(())
            }
            CompiledPat::Atom(a) => match val {
                Any::Atom(av) if av == a => Ok(()),
                _ => Err(fail()),
            },
            CompiledPat::Number(expected) => match val {
                Any::Number(actual) if expected.matches(*actual) => Ok(()),
                _ => Err(fail()),
            },
            CompiledPat::String(s) => match val {
                Any::String(v) if v == s => Ok(()),
                _ => Err(fail()),
            },
            CompiledPat::Char(c) => match val {
                Any::Char(v) if v == c => Ok(()),
                _ => Err(fail()),
            },
            CompiledPat::List(items) => match val {
                Any::Object(o) => match &o.borrow().kind {
                    ObjectKind::List(vec) if vec.len() == items.len() => {
                        for (p, v) in items.iter().zip(vec.iter()) {
                            p.bind(rt, base, v)?;
                        }
                        Ok(())
                    }
                    _ => Err(fail()),
                },
                _ => Err(fail()),
            },
            CompiledPat::Record(fields) => match val {
                Any::Object(o) => match &o.borrow().kind {
                    ObjectKind::Record(map) => {
                        for (key, pattern) in fields.iter() {
                            match map.get(key) {
                                Some(v) => pattern.bind(rt, base, v)?,
                                None => return Err(fail()),
                            }
                        }
                        Ok(())
                    }
                    _ => Err(fail()),
                },
                _ => Err(fail()),
            },
        }
    }
}

/// Shared compilation and execution context for all of a runtime's
/// compiled functions, living at `Runtime::vm`. Compilation interns
/// constants, variable names and patterns here (deduplicated, so `fib` and
/// `add` share one copy of the name `"n"` or the constant `1`), and
/// execution runs every compiled frame on the one operand `stack`.
pub struct VmContext {
    consts: Vec<Any>,
    names: Vec<Rc<str>>,
    pats: Vec<Rc<CompiledPat>>,
    /// The operand stack shared by all compiled-function frames. Public for
    /// inspection — a host function called from compiled code can watch the
    /// caller's temporaries live. Each frame works relative to the stack
    /// height at its entry and restores it on exit; pushing extra values
    /// from a host function mid-call is at your own peril.
    stack: Vec<Any>,
}

impl VmContext {
    pub fn new() -> Self {
        VmContext {
            consts: Vec::new(),
            names: Vec::new(),
            pats: Vec::new(),
            stack: Vec::new(),
        }
    }

    /// Intern a constant. Only immutable scalar values are deduplicated —
    /// and never across numeric kinds, since `Any`'s equality treats `1`
    /// and `1.0` as equal but the program must observe distinct values.
    fn const_(&mut self, v: Any) -> u32 {
        fn same(a: &Any, b: &Any) -> bool {
            match (a, b) {
                (Any::Number(Number::Integer(x)), Any::Number(Number::Integer(y))) => x == y,
                (Any::Number(Number::Float(x)), Any::Number(Number::Float(y))) => {
                    x.to_bits() == y.to_bits()
                }
                (Any::String(x), Any::String(y)) => x == y,
                (Any::Char(x), Any::Char(y)) => x == y,
                (Any::Atom(x), Any::Atom(y)) => x == y,
                _ => false,
            }
        }

        if let Some(i) = self.consts.iter().position(|c| same(c, &v)) {
            return i as u32;
        }
        self.consts.push(v);
        (self.consts.len() - 1) as u32
    }

    fn name(&mut self, n: Rc<str>) -> u32 {
        if let Some(i) = self.names.iter().position(|m| *m == n) {
            return i as u32;
        }
        self.names.push(n);
        (self.names.len() - 1) as u32
    }

    fn pattern(&mut self, p: &Pat, slots: &SlotTable) -> u32 {
        self.pats.push(Rc::new(compile_pat(p, slots)));
        (self.pats.len() - 1) as u32
    }

    /// Read frame slot `slot` of the frame based at `base`.
    #[inline]
    pub fn slot(&self, base: usize, slot: u32) -> &Any {
        &self.stack[base + slot as usize]
    }

    /// Write frame slot `slot` of the frame based at `base`.
    #[inline]
    pub fn set_slot(&mut self, base: usize, slot: u32, v: Any) {
        self.stack[base + slot as usize] = v;
    }

    /// Reserve `n` not-yet-assigned locals on top of the stack.
    #[inline]
    pub fn extend_uninit(&mut self, n: usize) {
        self.stack.resize_with(self.stack.len() + n, || Any::Uninit);
    }

    #[inline]
    pub fn push_stack(&mut self, v: Any) {
        self.stack.push(v);
    }

    #[inline]
    pub fn stack_len(&self) -> usize {
        self.stack.len()
    }

    #[inline]
    pub fn pop_stack(&mut self) -> Any {
        match self.stack.pop() {
            Some(v) => v,
            None => unreachable!("Attempted to pop from an empty VM stack"),
        }
    }

    #[inline]
    pub fn extend_stack(&mut self, values: impl IntoIterator<Item = Any>) {
        self.stack.extend(values);
    }

    #[inline]
    pub fn reverse_stack(&mut self, count: usize) {
        let at = self.stack.len() - count;
        self.stack[at..].reverse();
    }

    #[inline]
    pub fn drain_off_stack(&mut self, count: usize) -> impl DoubleEndedIterator<Item = Any> {
        let at = self.stack.len() - count;
        self.stack.drain(at..)
    }

    #[inline]
    pub fn truncate_stack(&mut self, len: usize) {
        self.stack.truncate(len);
    }
}

impl Default for VmContext {
    fn default() -> Self {
        Self::new()
    }
}

/// A function lowered to instructions. Its operands index the pools of the
/// [`VmContext`] it was compiled against — running it on a different
/// runtime's context yields wrong constants or errors.
///
/// Locals are frame slots on the shared operand stack: the `arity`
/// arguments the caller left on the stack are the first slots (first
/// argument on top ⇒ parameter `i` is slot `arity-1-i`, so plain-ident
/// parameters never move), and `slots - arity` additional slots for
/// body-introduced names are reserved on frame entry.
#[derive(Debug)]
pub struct CompiledFn {
    /// Number of arguments a full application consumes.
    pub arity: u32,
    /// Total frame slots, arguments included.
    pub slots: u32,
    /// Variable-length-encoded instructions (see [`Code`]).
    pub code: Code,
}

impl CompiledFn {
    #[inline]
    pub fn arity(&self) -> usize {
        self.arity as usize
    }

    /// Wrap into a first-class `Any::Fn` value that speaks the same calling
    /// convention as AST-defined and host functions — partial application
    /// and currying are handled by the runtime through the arity hint.
    #[inline]
    pub fn into_function(self) -> Any {
        Any::Fn(FnValue::new(self))
    }
}

impl Function for CompiledFn {
    #[inline]
    fn min_args(&self) -> usize {
        self.arity()
    }

    #[inline]
    fn max_args(&self) -> Option<usize> {
        Some(self.arity())
    }

    /// Compiled functions keep their locals in frame slots and never touch
    /// the runtime's name-keyed local scope.
    #[inline]
    fn wants_local_scope(&self) -> bool {
        false
    }

    #[inline]
    fn call(&self, rt: &mut Runtime, args: usize) -> Any {
        debug_assert_eq!(args, self.arity());

        // the arguments stay on the stack: they are the frame's first
        // slots (destructuring parameters are unpacked by the compiled
        // prologue)
        match run(rt, self) {
            Ok(v) => v,
            Err(e) => {
                rt.set_error(e);
                Any::nil()
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct CompileError {
    span: Span,
    msg: String,
}

impl CompileError {
    fn new(span: Span, msg: impl Into<String>) -> Self {
        CompileError {
            span,
            msg: msg.into(),
        }
    }

    pub fn span(&self) -> Span {
        self.span
    }

    pub fn message(&self) -> &str {
        &self.msg
    }
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "compile error: {}", self.msg)
    }
}

/// Compile a function (parameter patterns + body statements) to bytecode.
///
/// The compiler covers the whole statement/expression grammar; it only
/// rejects code that could never run successfully anyway (`break`/`continue`
/// outside a loop, malformed literals). Callers are expected to fall back to
/// the AST walker on error, which reproduces the interpreter's runtime
/// behavior for those cases.
pub fn compile_fn(
    ctx: &mut VmContext,
    params: Rc<[Pat]>,
    body: &[Stmt],
) -> Result<CompiledFn, CompileError> {
    let arity = params.len();
    let slots = collect_slots(&params, body);

    let mut c = Compiler {
        ctx,
        slots,
        code: Vec::new(),
        loops: Vec::new(),
    };

    // prologue: unpack destructuring parameters out of their argument
    // slots into the named slots they bind (plain-ident parameters simply
    // stay where their argument landed)
    for (i, p) in params.iter().enumerate() {
        if !matches!(p.kind(), PatKind::Ident(_)) {
            let arg_slot = (arity - 1 - i) as u32;
            c.emit(Instr::LoadSlot(arg_slot));
            let pattern = c.ctx.pattern(p, &c.slots);
            c.emit(Instr::Bind(pattern));
        }
    }

    // the body is compiled in tail position: exactly one value — the
    // function's result — is on the stack when `Return` is reached
    c.compile_block(body, true)?;
    c.emit(Instr::Return);

    Ok(CompiledFn {
        arity: arity as u32,
        slots: c.slots.total,
        code: encode(&c.code),
    })
}

struct LoopCtx {
    /// Jump target for `continue`: the condition check (`while`) or the
    /// `IterNext` instruction (`for`).
    continue_to: u32,
    /// `Jump` sites emitted by `break`, patched to the loop end (past the
    /// `else` block, which `break` skips).
    breaks: Vec<usize>,
    /// Whether `break` must also close the loop's iterator.
    in_for: bool,
    /// Whether the loop statement itself is in tail position: a `break`
    /// out of it must then push the loop's value (nil) for the function
    /// result.
    tail: bool,
}

/// If `expr` is an integer literal (possibly under unary minus), return
/// its value.
fn int_literal_of(expr: &Expr) -> Option<i64> {
    let (negated, lit) = match expr.kind() {
        ExprKind::Literal(lit) => (false, lit),
        ExprKind::Unary(op, inner) if op.kind() == UnOpKind::Neg => match inner.kind() {
            ExprKind::Literal(lit) => (true, lit),
            _ => return None,
        },
        _ => return None,
    };
    match literal_value(lit) {
        Ok(Any::Number(Number::Integer(n))) => {
            Some(if negated { n.wrapping_neg() } else { n })
        }
        _ => None,
    }
}

struct Compiler<'a> {
    /// Shared pools of the owning runtime; instruction operands index here.
    ctx: &'a mut VmContext,
    /// This function's name→frame-slot resolution.
    slots: SlotTable,
    code: Vec<Instr>,
    loops: Vec<LoopCtx>,
}

impl Compiler<'_> {
    fn emit(&mut self, instr: Instr) -> usize {
        self.code.push(instr);
        self.code.len() - 1
    }

    fn here(&self) -> u32 {
        self.code.len() as u32
    }

    fn patch(&mut self, at: usize, target: u32) {
        match &mut self.code[at] {
            Instr::Jump(t) | Instr::JumpIfFalse(t) | Instr::IterNext(t) => *t = target,
            other => unreachable!("patching non-jump instruction {:?}", other),
        }
    }

    /// Fuse `x = x + 1` / `x = x - 1` into a single in-place slot
    /// increment/decrement. Returns false (emitting nothing) when the
    /// statement doesn't have that shape or the operands don't fit the
    /// instruction's compact widths.
    fn try_inc_dec(&mut self, target: &Pat, value: &Expr) -> bool {
        let PatKind::Ident(id) = target.kind() else {
            return false;
        };
        if id.name() == "_" {
            return false;
        }
        let ExprKind::Binary(lhs, op, rhs) = value.kind() else {
            return false;
        };
        if !matches!(op.kind(), BinOpKind::Add | BinOpKind::Sub) {
            return false;
        }
        let ExprKind::Ident(lid) = lhs.kind() else {
            return false;
        };
        if lid.name() != id.name() || int_literal_of(rhs) != Some(1) {
            return false;
        }
        let Some(slot) = self.slots.get(id.name()) else {
            return false;
        };
        let name = self.ctx.name(id.rc_name());
        if name > u16::MAX as u32 {
            return false;
        }

        self.emit(if op.kind() == BinOpKind::Add {
            Instr::IncSlot {
                slot: slot.index,
                name,
            }
        } else {
            Instr::DecSlot {
                slot: slot.index,
                name,
            }
        });
        true
    }

    /// Bind the value on top of the stack to an assignment/parameter/loop
    /// target. Plain identifiers compile to a store; anything else goes
    /// through full pattern matching.
    fn compile_bind(&mut self, target: &Pat) {
        match target.kind() {
            // `_` matches anything and binds nothing: just drop the value
            PatKind::Ident(id) if id.name() == "_" => {
                self.emit(Instr::Pop);
            }
            PatKind::Ident(id) => {
                let slot = self
                    .slots
                    .get(id.name())
                    .expect("collect_slots missed an assignment target")
                    .index;
                self.emit(Instr::StoreLocal(slot));
            }
            _ => {
                let i = self.ctx.pattern(target, &self.slots);
                self.emit(Instr::Bind(i));
            }
        }
    }

    /// Emit the correct load for a variable name: a definitely-initialized
    /// slot, a maybe-unset slot with global fallback, or a plain global.
    /// Emit a constant value: small integers and the singleton atoms get
    /// immediate opcodes, everything else goes through the const pool.
    fn emit_const_value(&mut self, v: Any) {
        match v {
            Any::Number(Number::Integer(n)) if int_fits_immediate(n) => {
                self.emit(Instr::Int(n));
            }
            Any::Atom(Atom::Nil) => {
                self.emit(Instr::Nil);
            }
            Any::Atom(Atom::True) => {
                self.emit(Instr::True);
            }
            Any::Atom(Atom::False) => {
                self.emit(Instr::False);
            }
            v => {
                let i = self.ctx.const_(v);
                self.emit(Instr::Const(i));
            }
        }
    }

    fn emit_load_name(&mut self, name: Rc<str>) {
        match self.slots.get(&name) {
            Some(Slot {
                index,
                definite: true,
            }) => {
                self.emit(Instr::LoadSlot(index));
            }
            Some(Slot {
                index,
                definite: false,
            }) => {
                let name = self.ctx.name(name);
                self.emit(Instr::LoadLocal { slot: index, name });
            }
            None => {
                let name = self.ctx.name(name);
                self.emit(Instr::LoadGlobal(name));
            }
        }
    }

    /// Compile a block of statements. A block's value is its last
    /// statement's value; `tail` marks blocks whose value is the function
    /// result — only their final statement leaves a value on the stack
    /// (an empty tail block yields nil). Non-tail blocks have no net stack
    /// effect.
    fn compile_block(&mut self, stmts: &[Stmt], tail: bool) -> Result<(), CompileError> {
        match stmts.split_last() {
            None => {
                if tail {
                    self.emit(Instr::Nil);
                }
            }
            Some((last, init)) => {
                for statement in init {
                    self.compile_stmt(statement, false)?;
                }
                self.compile_stmt(last, tail)?;
            }
        }
        Ok(())
    }

    /// Compile one statement. If `tail` is true this statement's value is
    /// the function result and must end up on the stack (control-flow
    /// statements forward tailness into their branches); otherwise the
    /// statement must leave the stack untouched.
    fn compile_stmt(&mut self, statement: &Stmt, tail: bool) -> Result<(), CompileError> {
        match statement.kind() {
            StmtKind::Expr(e) => {
                self.compile_expr_callfn(e)?;
                if !tail {
                    self.emit(Instr::Pop);
                }
            }
            StmtKind::AssignPat { target, value } => {
                if !self.try_inc_dec(target, value) {
                    self.compile_expr(value)?;
                    self.compile_bind(target);
                }
                if tail {
                    self.emit(Instr::Nil); // assignments yield nil
                }
            }
            StmtKind::AssignField {
                target,
                field,
                value,
            } => {
                self.compile_expr(target)?;
                self.compile_expr(value)?;
                let i = self.ctx.name(field.rc_name());
                self.emit(Instr::SetField(i));
                if tail {
                    self.emit(Instr::Nil);
                }
            }
            StmtKind::AssignIndex {
                target,
                index,
                value,
            } => {
                self.compile_expr(target)?;
                self.compile_expr(index)?;
                self.compile_expr(value)?;
                self.emit(Instr::SetIndex);
                if tail {
                    self.emit(Instr::Nil);
                }
            }
            StmtKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.compile_expr(cond)?;
                match else_branch {
                    Some(eb) => {
                        let to_else = self.emit(Instr::JumpIfFalse(0));
                        self.compile_block(then_branch, tail)?;
                        let to_end = self.emit(Instr::Jump(0));

                        let else_at = self.here();
                        self.patch(to_else, else_at);
                        self.compile_block(eb, tail)?;

                        let end = self.here();
                        self.patch(to_end, end);
                    }
                    // in tail position a skipped `if` yields nil
                    None if tail => {
                        let to_else = self.emit(Instr::JumpIfFalse(0));
                        self.compile_block(then_branch, true)?;
                        let to_end = self.emit(Instr::Jump(0));

                        let else_at = self.here();
                        self.patch(to_else, else_at);
                        self.emit(Instr::Nil);

                        let end = self.here();
                        self.patch(to_end, end);
                    }
                    // otherwise the false path just falls through
                    None => {
                        let to_end = self.emit(Instr::JumpIfFalse(0));
                        self.compile_block(then_branch, false)?;

                        let end = self.here();
                        self.patch(to_end, end);
                    }
                }
            }
            StmtKind::While {
                cond,
                body,
                else_branch,
            } => {
                let head = self.here();
                self.compile_expr(cond)?;
                let to_exit = self.emit(Instr::JumpIfFalse(0));

                self.loops.push(LoopCtx {
                    continue_to: head,
                    breaks: Vec::new(),
                    in_for: false,
                    tail,
                });
                self.compile_block(body, false)?;
                self.emit(Instr::Jump(head));
                let ctx = self.loops.pop().unwrap();

                // normal exit runs `else` (which then carries the loop's
                // tailness); it belongs to the enclosing loop as far as
                // break/continue are concerned. `break` jumps past it.
                let exit_at = self.here();
                self.patch(to_exit, exit_at);
                match else_branch {
                    Some(eb) => self.compile_block(eb, tail)?,
                    None if tail => {
                        self.emit(Instr::Nil);
                    }
                    None => {}
                }

                let end = self.here();
                for site in ctx.breaks {
                    self.patch(site, end);
                }
            }
            StmtKind::For {
                target,
                iterable,
                body,
                else_branch,
            } => {
                self.compile_expr(iterable)?;
                self.emit(Instr::IterInit);

                let head = self.here();
                let next = self.emit(Instr::IterNext(0));
                self.compile_bind(target);

                self.loops.push(LoopCtx {
                    continue_to: head,
                    breaks: Vec::new(),
                    in_for: true,
                    tail,
                });
                self.compile_block(body, false)?;
                self.emit(Instr::Jump(head));
                let ctx = self.loops.pop().unwrap();

                let exit_at = self.here();
                self.patch(next, exit_at);
                match else_branch {
                    Some(eb) => self.compile_block(eb, tail)?,
                    None if tail => {
                        self.emit(Instr::Nil);
                    }
                    None => {}
                }

                let end = self.here();
                for site in ctx.breaks {
                    self.patch(site, end);
                }
            }
            StmtKind::Return(value) => {
                match value {
                    Some(e) => self.compile_expr(e)?,
                    None => {
                        self.emit(Instr::Nil);
                    }
                }
                self.emit(Instr::Return);
            }
            StmtKind::Break => {
                let Some((in_for, loop_tail)) = self.loops.last().map(|l| (l.in_for, l.tail))
                else {
                    return Err(CompileError::new(
                        statement.span(),
                        "break statement outside of loop",
                    ));
                };
                if in_for {
                    self.emit(Instr::IterPop);
                }
                // a broken loop skips its `else` block and yields nil —
                // which only materializes if the loop's value is the
                // function result
                if loop_tail {
                    self.emit(Instr::Nil);
                }
                let site = self.emit(Instr::Jump(0));
                self.loops.last_mut().unwrap().breaks.push(site);
            }
            StmtKind::Continue => {
                let Some(continue_to) = self.loops.last().map(|l| l.continue_to) else {
                    return Err(CompileError::new(
                        statement.span(),
                        "continue statement outside of loop",
                    ));
                };
                self.emit(Instr::Jump(continue_to));
            }
            StmtKind::Fn { name, params, body } => {
                // nested definitions are compiled too, becoming constants
                let compiled = compile_fn(self.ctx, params.clone(), body)?;
                self.ctx.consts.push(compiled.into_function());
                let i = (self.ctx.consts.len() - 1) as u32;
                self.emit(Instr::Const(i));
                let slot = self
                    .slots
                    .get(name.name())
                    .expect("collect_slots missed a fn name")
                    .index;
                self.emit(Instr::StoreLocal(slot));
                if tail {
                    self.emit(Instr::Nil); // definitions yield nil
                }
            }
        }
        Ok(())
    }

    fn compile_expr(&mut self, expr: &Expr) -> Result<(), CompileError> {
        match expr.kind() {
            ExprKind::Literal(lit) => {
                let v = literal_value(lit)
                    .map_err(|e| CompileError::new(expr.span(), e.to_string()))?;
                self.emit_const_value(v);
            }
            ExprKind::Ident(id) => {
                self.emit_load_name(id.rc_name());
            }
            ExprKind::Atom(a) => {
                self.emit_const_value(Any::new_atom(a.rc_name()));
            }
            ExprKind::List(elements) => {
                for e in elements.iter() {
                    self.compile_expr(e)?;
                }
                self.emit(Instr::MakeList(elements.len() as u32));
            }
            ExprKind::Record(fields) => {
                for f in fields.iter() {
                    let key = f.key().rc_name();
                    let ki = self.ctx.const_(Any::String(key.clone()));
                    self.emit(Instr::Const(ki));
                    match f.value() {
                        Some(v) => self.compile_expr(v)?,
                        // shorthand field reads the same-named variable
                        None => self.emit_load_name(key),
                    }
                }
                self.emit(Instr::MakeRecord(fields.len() as u32));
            }
            ExprKind::Unary(op, operand) => {
                // fold `-<number literal>` at compile time, mirroring what
                // evaluation would do, so negative small integers reach
                // the immediate opcodes
                if let (UnOpKind::Neg, ExprKind::Literal(lit)) = (op.kind(), operand.kind()) {
                    let v = literal_value(lit)
                        .map_err(|e| CompileError::new(expr.span(), e.to_string()))?;
                    if let Any::Number(n) = v {
                        let negated = n
                            .neg()
                            .map_err(|e| CompileError::new(expr.span(), e.to_string()))?;
                        self.emit_const_value(Any::Number(negated));
                        return Ok(());
                    }
                }
                self.compile_expr(operand)?;
                self.emit(Instr::Unary(op.kind()));
            }
            ExprKind::Binary(lhs, op, rhs) => {
                // fuse `<expr> op n` with an integer-literal right operand
                // into a single instruction, when n is in the op's table
                let table: Option<&[i64]> = match op.kind() {
                    BinOpKind::Add | BinOpKind::Sub => Some(&SMALL_INTS),
                    BinOpKind::BitAnd | BinOpKind::BitOr | BinOpKind::BitXor => {
                        Some(&TINY_MASKS)
                    }
                    BinOpKind::Eq
                    | BinOpKind::Ne
                    | BinOpKind::Lt
                    | BinOpKind::Gt
                    | BinOpKind::Le
                    | BinOpKind::Ge => Some(&TINY_INTS),
                    _ => None,
                };
                if let (Some(table), Some(n)) = (table, int_literal_of(rhs)) {
                    if table.contains(&n) {
                        self.compile_expr(lhs)?;
                        self.emit(Instr::BinaryInt(op.kind(), n as i16));
                        return Ok(());
                    }
                }
                self.compile_expr(lhs)?;
                self.compile_expr(rhs)?;
                self.emit(Instr::Binary(op.kind()));
            }
            ExprKind::Apply(func, args) => {
                self.compile_expr(func)?;
                for a in args.iter() {
                    self.compile_expr(a)?;
                }
                self.emit(Instr::Call(args.len() as u32));
            }
            ExprKind::Field(obj, field) => {
                self.compile_expr(obj)?;
                let i = self.ctx.name(field.rc_name());
                self.emit(Instr::GetField(i));
            }
            ExprKind::Index(obj, index) => {
                self.compile_expr(obj)?;
                self.compile_expr(index)?;
                self.emit(Instr::GetIndex);
            }
            // parentheses put the inner expression in call position
            ExprKind::Parenthesized(inner) => self.compile_expr_callfn(inner)?,
        }
        Ok(())
    }

    /// Compile an expression in call position (statement expressions and
    /// parenthesized expressions): a bare identifier holding a zero-argument
    /// function gets called instead of yielding the function value.
    fn compile_expr_callfn(&mut self, expr: &Expr) -> Result<(), CompileError> {
        match expr.kind() {
            ExprKind::Ident(_) => {
                self.compile_expr(expr)?;
                self.emit(Instr::CallBare);
            }
            ExprKind::Parenthesized(inner) => self.compile_expr_callfn(inner)?,
            _ => self.compile_expr(expr)?,
        }
        Ok(())
    }
}

/// Execute a compiled function's code. Parameters are expected to already be
/// bound in the current (local) scope — [`CompiledFn::into_function`] does
/// that — so `run` itself is just the instruction loop.
///
/// The frame executes on the runtime's shared operand stack
/// (`rt.vm.stack`): it treats the stack height at entry as its floor and
/// restores it on the way out, whether it returns a value or an error, so
/// nested and recursive frames (including mixed-mode reentry through host
/// or AST functions) compose without per-call stack allocations.
pub fn run(rt: &mut Runtime, f: &CompiledFn) -> Result<Any, RuntimeError> {
    // the caller's arguments are already on the stack and become the
    // frame's first slots; reserve the rest for body-introduced locals
    debug_assert!(rt.vm.stack.len() >= f.arity());
    let base = rt.vm.stack.len() - f.arity();
    rt.vm.extend_uninit((f.slots - f.arity) as usize);

    let result = run_frame(rt, f, base);
    debug_assert!(rt.vm.stack.len() >= base);
    rt.vm.stack.truncate(base);
    result
}

fn run_frame(rt: &mut Runtime, f: &CompiledFn, base: usize) -> Result<Any, RuntimeError> {
    let mut iters = Vec::new();
    let mut pc: usize = 0;
    let code = &f.code.bytes[..];

    loop {
        // `pc` is a byte offset; each opcode decodes its own operands
        let Some(&op) = code.get(pc) else {
            return Err(RuntimeError::Other(
                "vm: execution ran past the end of code".to_string(),
            ));
        };
        pc += 1;

        match op {
            opcode::CONST..=opcode::CONST_END => {
                let i = decode_wop(op - opcode::CONST, code, &mut pc)?;
                let v = rt.vm.consts.get(i as usize).cloned().ok_or_else(|| {
                    RuntimeError::Other("vm: constant index out of range".to_string())
                })?;
                rt.vm.stack.push(v);
            }
            opcode::NIL => rt.vm.stack.push(Any::nil()),
            opcode::TRUE => rt.vm.stack.push(Any::true_()),
            opcode::FALSE => rt.vm.stack.push(Any::false_()),
            opcode::INT..=opcode::INT_END => {
                let n = SMALL_INTS[(op - opcode::INT) as usize];
                rt.vm.stack.push(Any::Number(Number::Integer(n)));
            }
            opcode::INT_8 => {
                let n = read_u8(code, &mut pc)? as i8 as i64;
                rt.vm.stack.push(Any::Number(Number::Integer(n)));
            }
            opcode::INT_16 => {
                let n = read_u16(code, &mut pc)? as i16 as i64;
                rt.vm.stack.push(Any::Number(Number::Integer(n)));
            }
            opcode::UINT_8 => {
                let n = read_u8(code, &mut pc)? as i64;
                rt.vm.stack.push(Any::Number(Number::Integer(n)));
            }
            opcode::UINT_16 => {
                let n = read_u16(code, &mut pc)? as i64;
                rt.vm.stack.push(Any::Number(Number::Integer(n)));
            }
            opcode::POP => {
                rt.vm.pop_stack();
            }
            opcode::LOAD_SLOT..=opcode::LOAD_SLOT_END => {
                let slot = decode_operand(op - opcode::LOAD_SLOT, code, &mut pc)?;
                let v = rt.vm.slot(base, slot).clone();
                debug_assert!(
                    !matches!(v, Any::Uninit),
                    "vm: LoadSlot on an unassigned slot"
                );
                rt.vm.stack.push(v);
            }
            opcode::LOAD_LOCAL..=opcode::LOAD_LOCAL_END => {
                let rel = op - opcode::LOAD_LOCAL;
                let slot = decode_operand(rel / 3, code, &mut pc)?;
                let name = match rel % 3 {
                    0 => read_u8(code, &mut pc)? as u32,
                    1 => read_u16(code, &mut pc)? as u32,
                    _ => read_u32(code, &mut pc)?,
                };
                let v = rt.vm.slot(base, slot).clone();
                let v = if matches!(v, Any::Uninit) {
                    // not assigned yet: the name reaches the global scope
                    let name = rt.vm.names.get(name as usize).ok_or_else(|| {
                        RuntimeError::Other("vm: name index out of range".to_string())
                    })?;
                    rt.global
                        .get(&name[..])
                        .cloned()
                        .ok_or_else(|| RuntimeError::UnboundIdentifier(name[..].to_string()))?
                } else {
                    v
                };
                rt.vm.stack.push(v);
            }
            opcode::LOAD_GLOBAL..=opcode::LOAD_GLOBAL_END => {
                let i = decode_wop(op - opcode::LOAD_GLOBAL, code, &mut pc)?;
                let name = rt.vm.names.get(i as usize).ok_or_else(|| {
                    RuntimeError::Other("vm: name index out of range".to_string())
                })?;
                let v = rt
                    .global
                    .get(&name[..])
                    .cloned()
                    .ok_or_else(|| RuntimeError::UnboundIdentifier(name[..].to_string()))?;
                rt.vm.stack.push(v);
            }
            opcode::STORE_LOCAL..=opcode::STORE_LOCAL_END => {
                let slot = decode_operand(op - opcode::STORE_LOCAL, code, &mut pc)?;
                let v = rt.vm.pop_stack();
                rt.vm.set_slot(base, slot, v);
            }
            opcode::BIND..=opcode::BIND_END => {
                let i = decode_wop(op - opcode::BIND, code, &mut pc)?;
                let v = rt.vm.pop_stack();
                let pattern = rt
                    .vm
                    .pats
                    .get(i as usize)
                    .ok_or_else(|| {
                        RuntimeError::Other("vm: pattern index out of range".to_string())
                    })?
                    .clone();
                pattern.bind(rt, base, &v)?;
            }
            opcode::MAKE_LIST..=opcode::MAKE_LIST_END => {
                let n = decode_operand(op - opcode::MAKE_LIST, code, &mut pc)?;
                let elements = rt.vm.drain_off_stack(n as usize);
                let list = Any::new_list(elements.collect());
                rt.vm.push_stack(list);
            }
            opcode::MAKE_RECORD..=opcode::MAKE_RECORD_END => {
                let n = decode_operand(op - opcode::MAKE_RECORD, code, &mut pc)?;
                let mut map = BTreeMap::new();
                {
                    let mut fields = rt.vm.drain_off_stack(n as usize * 2);
                    while let (Some(key), Some(val)) = (fields.next(), fields.next()) {
                        match key {
                            Any::String(key) => {
                                map.insert(key, val);
                            }
                            _ => {
                                return Err(RuntimeError::TypeError(
                                    "vm: record key must be a string".to_string(),
                                ));
                            }
                        }
                    }
                }
                let record = Any::new_record(map);
                rt.vm.push_stack(record);
            }
            opcode::UNARY..=opcode::UNARY_END => {
                let k = byte_to_unop(op - opcode::UNARY)?;
                let a = rt.vm.pop_stack();
                let v = eval_unary(k, &a)?;
                rt.vm.stack.push(v);
            }
            opcode::BINARY..=opcode::BINARY_END => {
                let k = byte_to_binop(op - opcode::BINARY)?;
                let b = rt.vm.pop_stack();
                let a = rt.vm.pop_stack();
                let v = eval_binary(k, &a, &b)?;
                rt.vm.stack.push(v);
            }
            opcode::ADD_INT..=opcode::ADD_INT_END => {
                let n = SMALL_INTS[(op - opcode::ADD_INT) as usize];
                let a = rt.vm.pop_stack();
                let v = match a {
                    // fast path mirroring Number::add's wrapping semantics
                    Any::Number(Number::Integer(x)) => {
                        Any::Number(Number::Integer(x.wrapping_add(n)))
                    }
                    a => eval_binary(BinOpKind::Add, &a, &Any::Number(Number::Integer(n)))?,
                };
                rt.vm.stack.push(v);
            }
            opcode::SUB_INT..=opcode::SUB_INT_END => {
                let n = SMALL_INTS[(op - opcode::SUB_INT) as usize];
                let a = rt.vm.pop_stack();
                let v = match a {
                    Any::Number(Number::Integer(x)) => {
                        Any::Number(Number::Integer(x.wrapping_sub(n)))
                    }
                    a => eval_binary(BinOpKind::Sub, &a, &Any::Number(Number::Integer(n)))?,
                };
                rt.vm.stack.push(v);
            }
            opcode::AND_MASK..=opcode::AND_MASK_END => {
                let n = TINY_MASKS[(op - opcode::AND_MASK) as usize];
                let a = rt.vm.pop_stack();
                let v = match a {
                    Any::Number(Number::Integer(x)) => Any::Number(Number::Integer(x & n)),
                    a => eval_binary(BinOpKind::BitAnd, &a, &Any::Number(Number::Integer(n)))?,
                };
                rt.vm.stack.push(v);
            }
            opcode::OR_MASK..=opcode::OR_MASK_END => {
                let n = TINY_MASKS[(op - opcode::OR_MASK) as usize];
                let a = rt.vm.pop_stack();
                let v = match a {
                    Any::Number(Number::Integer(x)) => Any::Number(Number::Integer(x | n)),
                    a => eval_binary(BinOpKind::BitOr, &a, &Any::Number(Number::Integer(n)))?,
                };
                rt.vm.stack.push(v);
            }
            opcode::XOR_MASK..=opcode::XOR_MASK_END => {
                let n = TINY_MASKS[(op - opcode::XOR_MASK) as usize];
                let a = rt.vm.pop_stack();
                let v = match a {
                    Any::Number(Number::Integer(x)) => Any::Number(Number::Integer(x ^ n)),
                    a => eval_binary(BinOpKind::BitXor, &a, &Any::Number(Number::Integer(n)))?,
                };
                rt.vm.stack.push(v);
            }
            opcode::INT_EQ..=opcode::INT_EQ_END => {
                let n = TINY_INTS[(op - opcode::INT_EQ) as usize];
                let a = rt.vm.pop_stack();
                let v = match a {
                    Any::Number(Number::Integer(x)) => Any::bool_(x == n),
                    a => eval_binary(BinOpKind::Eq, &a, &Any::Number(Number::Integer(n)))?,
                };
                rt.vm.stack.push(v);
            }
            opcode::INT_NE..=opcode::INT_NE_END => {
                let n = TINY_INTS[(op - opcode::INT_NE) as usize];
                let a = rt.vm.pop_stack();
                let v = match a {
                    Any::Number(Number::Integer(x)) => Any::bool_(x != n),
                    a => eval_binary(BinOpKind::Ne, &a, &Any::Number(Number::Integer(n)))?,
                };
                rt.vm.stack.push(v);
            }
            opcode::INT_LT..=opcode::INT_LT_END => {
                let n = TINY_INTS[(op - opcode::INT_LT) as usize];
                let a = rt.vm.pop_stack();
                let v = match a {
                    Any::Number(Number::Integer(x)) => Any::bool_(x < n),
                    a => eval_binary(BinOpKind::Lt, &a, &Any::Number(Number::Integer(n)))?,
                };
                rt.vm.stack.push(v);
            }
            opcode::INT_GT..=opcode::INT_GT_END => {
                let n = TINY_INTS[(op - opcode::INT_GT) as usize];
                let a = rt.vm.pop_stack();
                let v = match a {
                    Any::Number(Number::Integer(x)) => Any::bool_(x > n),
                    a => eval_binary(BinOpKind::Gt, &a, &Any::Number(Number::Integer(n)))?,
                };
                rt.vm.stack.push(v);
            }
            opcode::INT_LE..=opcode::INT_LE_END => {
                let n = TINY_INTS[(op - opcode::INT_LE) as usize];
                let a = rt.vm.pop_stack();
                let v = match a {
                    Any::Number(Number::Integer(x)) => Any::bool_(x <= n),
                    a => eval_binary(BinOpKind::Le, &a, &Any::Number(Number::Integer(n)))?,
                };
                rt.vm.stack.push(v);
            }
            opcode::INT_GE..=opcode::INT_GE_END => {
                let n = TINY_INTS[(op - opcode::INT_GE) as usize];
                let a = rt.vm.pop_stack();
                let v = match a {
                    Any::Number(Number::Integer(x)) => Any::bool_(x >= n),
                    a => eval_binary(BinOpKind::Ge, &a, &Any::Number(Number::Integer(n)))?,
                };
                rt.vm.stack.push(v);
            }
            opcode::INC_SLOT..=opcode::DEC_SLOT_END => {
                let inc = op <= opcode::INC_SLOT_END;
                let rel = if inc {
                    op - opcode::INC_SLOT
                } else {
                    op - opcode::DEC_SLOT
                };
                let slot = decode_operand(rel, code, &mut pc)?;
                let name = read_u16(code, &mut pc)? as u32;
                let cur = rt.vm.slot(base, slot).clone();
                let cur = if matches!(cur, Any::Uninit) {
                    // not assigned yet: the read reaches the global scope
                    let name = rt.vm.names.get(name as usize).ok_or_else(|| {
                        RuntimeError::Other("vm: name index out of range".to_string())
                    })?;
                    rt.global
                        .get(&name[..])
                        .cloned()
                        .ok_or_else(|| RuntimeError::UnboundIdentifier(name[..].to_string()))?
                } else {
                    cur
                };
                let kind = if inc { BinOpKind::Add } else { BinOpKind::Sub };
                let v = match cur {
                    Any::Number(Number::Integer(x)) => {
                        Any::Number(Number::Integer(if inc {
                            x.wrapping_add(1)
                        } else {
                            x.wrapping_sub(1)
                        }))
                    }
                    cur => eval_binary(kind, &cur, &Any::Number(Number::Integer(1)))?,
                };
                rt.vm.set_slot(base, slot, v);
            }
            opcode::GET_FIELD..=opcode::GET_FIELD_END => {
                let i = decode_wop(op - opcode::GET_FIELD, code, &mut pc)?;
                let obj = rt.vm.pop_stack();
                let v = field_of(&obj, &rt.vm.names[i as usize])?;
                rt.vm.stack.push(v);
            }
            opcode::GET_INDEX => {
                let idx = rt.vm.pop_stack();
                let obj = rt.vm.pop_stack();
                let v = index_of(&obj, &idx)?;
                rt.vm.stack.push(v);
            }
            opcode::SET_FIELD..=opcode::SET_FIELD_END => {
                let i = decode_wop(op - opcode::SET_FIELD, code, &mut pc)?;
                let val = rt.vm.pop_stack();
                let obj = rt.vm.pop_stack();
                assign_field(obj, rt.vm.names[i as usize].clone(), val)?;
            }
            opcode::SET_INDEX => {
                let val = rt.vm.pop_stack();
                let idx = rt.vm.pop_stack();
                let obj = rt.vm.pop_stack();
                assign_index(obj, idx, val)?;
            }
            opcode::CALL..=opcode::CALL_END => {
                let n = decode_operand(op - opcode::CALL, code, &mut pc)?;
                rt.vm.reverse_stack(n as usize + 1);
                let fval = rt.vm.pop_stack();
                let ret = rt.apply_value(fval, n as usize)?;
                rt.vm.stack.push(ret);
            }
            opcode::CALL_BARE => {
                let v = rt.vm.pop_stack();
                let ret = rt.call_bare(v)?;
                rt.vm.stack.push(ret);
            }
            opcode::JUMP => {
                pc = read_u32(code, &mut pc)? as usize;
            }
            opcode::JUMP_IF_FALSE => {
                let t = read_u32(code, &mut pc)?;
                let c = rt.vm.pop_stack();
                if is_falsey(&c) {
                    pc = t as usize;
                }
            }
            opcode::ITER_INIT => {
                let v = rt.vm.pop_stack();
                iters.push(v.iter()?.into_iter());
            }
            opcode::ITER_NEXT => {
                let t = read_u32(code, &mut pc)?;
                let iter = iters
                    .last_mut()
                    .ok_or_else(|| RuntimeError::Other("vm: no active iterator".to_string()))?;
                match iter.next() {
                    Some(item) => rt.vm.stack.push(item),
                    None => {
                        iters.pop();
                        pc = t as usize;
                    }
                }
            }
            opcode::ITER_POP => {
                iters.pop();
            }
            opcode::RETURN => return Ok(rt.vm.pop_stack()),
            _ => return Err(RuntimeError::Other("vm: unknown opcode".to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Exec;
    use alloc::{format, vec};
    use core::cell::RefCell;

    fn ast_from_str(s: &str) -> Vec<Stmt> {
        let tokens = raft_ast::lexer::parse_str(s, &raft_ast::lexer::Options::wss()).unwrap();
        let mut stream = raft_ast::parser::TokenStream::new(tokens);
        raft_ast::Stmt::parse_many(&mut stream).unwrap()
    }

    fn run_mode(src: &str, compiled: bool) -> Result<Runtime, RuntimeError> {
        let stmts = ast_from_str(src);
        let mut rt = Runtime::new();
        rt.set_compile_fns(compiled);
        for statement in &stmts {
            rt.exec_stmt(statement)?;
        }
        Ok(rt)
    }

    /// Run `src` through the AST walker and the bytecode VM and assert that
    /// every global variable ends up displaying identically.
    fn assert_modes_agree(src: &str) -> Runtime {
        let walked = run_mode(src, false).expect("AST walker failed");
        let vmed = run_mode(src, true).expect("VM failed");

        let mut walked_keys: Vec<_> = walked.global.keys().collect();
        let mut vmed_keys: Vec<_> = vmed.global.keys().collect();
        walked_keys.sort();
        vmed_keys.sort();
        assert_eq!(walked_keys, vmed_keys, "modes bound different globals");
        for (name, walked_val) in &walked.global {
            let vmed_val = &vmed.global[name];
            assert_eq!(
                format!("{walked_val}"),
                format!("{vmed_val}"),
                "global `{name}` differs between modes"
            );
        }
        vmed
    }

    fn int_var(rt: &Runtime, name: &str) -> i64 {
        match rt.get_var(name) {
            Some(Any::Number(Number::Integer(i))) => i,
            other => panic!("expected integer in `{name}`, got {other:?}"),
        }
    }

    #[test]
    fn functions_actually_compile() {
        let src = "fn add a b:\n    return a + b\n";
        let stmts = ast_from_str(src);
        let StmtKind::Fn { params, body, .. } = stmts[0].kind() else {
            panic!("expected fn statement");
        };
        let mut ctx = VmContext::new();
        let compiled = compile_fn(&mut ctx, params.clone(), body).unwrap();
        let instrs: Vec<Instr> = compiled
            .code
            .disassemble()
            .map(|r| r.unwrap().1)
            .collect();
        assert!(matches!(instrs.last(), Some(Instr::Return)));
        assert_eq!(compiled.arity(), 2);
    }

    #[test]
    fn bytecode_roundtrips_through_the_encoder() {
        // every instruction kind, with operands spanning varint widths
        let instrs = vec![
            Instr::Const(0),
            Instr::Const(127),
            Instr::Const(128),
            Instr::Const(0x4000),
            Instr::Const(u32::MAX),
            Instr::Nil,
            Instr::Pop,
            Instr::LoadSlot(3),
            Instr::LoadLocal { slot: 200, name: 5 },
            Instr::LoadLocal { slot: 3, name: 5 },
            Instr::LoadLocal {
                slot: 300,
                name: 70000,
            },
            Instr::LoadGlobal(70000),
            Instr::StoreLocal(9),
            Instr::Bind(1),
            Instr::MakeList(2),
            Instr::MakeRecord(3),
            Instr::Unary(raft_ast::UnOpKind::Neg),
            Instr::Binary(raft_ast::BinOpKind::Pow),
            Instr::True,
            Instr::False,
            Instr::Int(-4),
            Instr::Int(0),
            Instr::Int(256),
            Instr::Int(200),
            Instr::Int(-100),
            Instr::Int(1000),
            Instr::Int(-1000),
            Instr::Int(60000),
            Instr::Int(-30000),
            Instr::BinaryInt(raft_ast::BinOpKind::Add, 1),
            Instr::BinaryInt(raft_ast::BinOpKind::Sub, -2),
            Instr::BinaryInt(raft_ast::BinOpKind::Sub, 9),
            Instr::BinaryInt(raft_ast::BinOpKind::BitAnd, 1),
            Instr::BinaryInt(raft_ast::BinOpKind::BitOr, 15),
            Instr::BinaryInt(raft_ast::BinOpKind::BitXor, 3),
            Instr::BinaryInt(raft_ast::BinOpKind::Eq, -1),
            Instr::BinaryInt(raft_ast::BinOpKind::Ne, 4),
            Instr::BinaryInt(raft_ast::BinOpKind::Lt, 2),
            Instr::BinaryInt(raft_ast::BinOpKind::Gt, 0),
            Instr::BinaryInt(raft_ast::BinOpKind::Le, 3),
            Instr::BinaryInt(raft_ast::BinOpKind::Ge, 1),
            Instr::IncSlot { slot: 2, name: 9 },
            Instr::DecSlot {
                slot: 250,
                name: 60000,
            },
            Instr::GetField(4),
            Instr::GetIndex,
            Instr::SetField(6),
            Instr::SetIndex,
            Instr::Call(2),
            Instr::CallBare,
            Instr::Jump(0),      // → byte offset of instruction 0
            Instr::JumpIfFalse(5),
            Instr::IterInit,
            Instr::IterNext(24), // → one past the last instruction
            Instr::IterPop,
            Instr::Return,
        ];
        // jump operands hold instruction indices going in and byte offsets
        // coming out; map the expectation through the encoded layout
        let code = encode(&instrs);
        let decoded: Vec<(usize, Instr)> =
            code.disassemble().map(|r| r.unwrap()).collect();
        assert_eq!(decoded.len(), instrs.len());

        let offsets: Vec<u32> = decoded.iter().map(|(at, _)| *at as u32).collect();
        for (expected, (_, got)) in instrs.iter().zip(decoded.iter()) {
            let expected = match *expected {
                Instr::Jump(t) => Instr::Jump(offsets[t as usize]),
                Instr::JumpIfFalse(t) => Instr::JumpIfFalse(offsets[t as usize]),
                Instr::IterNext(t) if (t as usize) < offsets.len() => {
                    Instr::IterNext(offsets[t as usize])
                }
                Instr::IterNext(_) => Instr::IterNext(code.len() as u32),
                other => other,
            };
            assert_eq!(expected, *got);
        }

        // the small forms really are as small as promised
        assert_eq!(encode(&[Instr::LoadSlot(3)]).len(), 1, "inline slot");
        assert_eq!(
            encode(&[Instr::Binary(raft_ast::BinOpKind::Add)]).len(),
            1,
            "operator kind lives in the opcode"
        );
        assert_eq!(encode(&[Instr::LoadSlot(200)]).len(), 2, "u8 slot");
        assert_eq!(
            encode(&[Instr::LoadLocal { slot: 3, name: 5 }]).len(),
            2,
            "inline slot + u8 name"
        );
    }

    #[test]
    fn arithmetic_and_implicit_return() {
        let rt = assert_modes_agree("fn add a b:\n    a + b\nr = add 1 2\n");
        assert_eq!(int_var(&rt, "r"), 3);
    }

    #[test]
    fn explicit_return_and_operators() {
        let rt = assert_modes_agree(
            "fn mix a b:\n    return a * b + (a << 2) - b / 2 + a ** 2\nr = mix 7 4\n",
        );
        assert_eq!(int_var(&rt, "r"), 103);
    }

    #[test]
    fn currying_and_partial_application() {
        let rt = assert_modes_agree(
            "fn add3 a b c:\n    return a + b + c\n\
             add1 = add3 1\nadd12 = add1 2\n\
             r1 = add12 3\nr2 = add3 10 20 30\nr3 = (add3 1) 2 3\n",
        );
        assert_eq!(int_var(&rt, "r1"), 6);
        assert_eq!(int_var(&rt, "r2"), 60);
        assert_eq!(int_var(&rt, "r3"), 6);
    }

    #[test]
    fn over_application_carries_to_returned_function() {
        // `make_adder` returns a function; extra arguments are re-applied
        let rt = assert_modes_agree(
            "fn add a b:\n    return a + b\n\
             fn make_adder a:\n    return add a\n\
             r = make_adder 3 4\n",
        );
        assert_eq!(int_var(&rt, "r"), 7);
    }

    #[test]
    fn recursion() {
        let rt = assert_modes_agree(
            "fn fib n:\n    if n < 2:\n        return n\n    return (fib (n - 1)) + (fib (n - 2))\n\
             r = fib 15\n",
        );
        assert_eq!(int_var(&rt, "r"), 610);
    }

    #[test]
    fn while_loop_and_if_else_chain() {
        let rt = assert_modes_agree(
            "fn collatz n:\n    steps = 0\n    while n != 1:\n        if n & 1 == 0:\n            n = n / 2\n        else:\n            n = 3 * n + 1\n        steps = steps + 1\n    return steps\n\
             r = collatz 27\n",
        );
        assert_eq!(int_var(&rt, "r"), 111);
    }

    #[test]
    fn for_else_and_break() {
        let rt = assert_modes_agree(
            "fn find xs needle:\n    idx = 0\n    for x in xs:\n        if x == needle:\n            break\n        idx = idx + 1\n    else:\n        return -1\n    return idx\n\
             ys = [10, 20, 30]\nr1 = find ys 20\nr2 = find ys 99\n",
        );
        assert_eq!(int_var(&rt, "r1"), 1);
        assert_eq!(int_var(&rt, "r2"), -1);
    }

    #[test]
    fn continue_in_for() {
        let rt = assert_modes_agree(
            "fn sum_odds xs:\n    total = 0\n    for x in xs:\n        if x & 1 == 0:\n            continue\n        total = total + x\n    return total\n\
             ys = [1, 2, 3, 4, 5]\nr = sum_odds ys\n",
        );
        assert_eq!(int_var(&rt, "r"), 9);
    }

    #[test]
    fn nested_for_with_inner_break() {
        let rt = assert_modes_agree(
            "fn count:\n    total = 0\n    for i in [1, 2, 3]:\n        for j in [1, 2, 3]:\n            if j > i:\n                break\n            total = total + 1\n    return total\n\
             r = (count)\n",
        );
        assert_eq!(int_var(&rt, "r"), 6);
    }

    #[test]
    fn while_else_and_break_skips_else() {
        let rt = assert_modes_agree(
            "fn wloop n:\n    while n < 10:\n        if n > 3:\n            break\n        n = n + 1\n    else:\n        return -1\n    return n\n\
             r1 = wloop 0\nr2 = wloop 20\n",
        );
        assert_eq!(int_var(&rt, "r1"), 4);
        assert_eq!(int_var(&rt, "r2"), -1);
    }

    #[test]
    fn record_param_destructuring() {
        let rt = assert_modes_agree(
            "fn dist2 { x, y }:\n    return x * x + y * y\n\
             r = dist2 { x: 3, y: 4 }\n",
        );
        assert_eq!(int_var(&rt, "r"), 25);
    }

    #[test]
    fn list_param_destructuring_and_shorthand_record() {
        let rt = assert_modes_agree(
            "fn swap [a, b]:\n    return [b, a]\n\
             fn wrap x:\n    name = x\n    return { name }\n\
             ys = [1, 2]\nr1 = swap ys\nr2 = wrap \"Ada\"\n",
        );
        assert_eq!(format!("{}", rt.get_var("r2").unwrap()), "{name: Ada}");
    }

    #[test]
    fn field_and_index_mutation_inside_function() {
        let rt = assert_modes_agree(
            "fn setup:\n    o = { a: 1 }\n    o.a = 5\n    xs = [1, 2]\n    xs[1] = 9\n    return [o.a, xs[1]]\n\
             r = (setup)\n",
        );
        assert_eq!(format!("{}", rt.get_var("r").unwrap()), "[5, 9]");
    }

    #[test]
    fn zero_arg_functions_called_bare_and_parenthesized() {
        let rt = assert_modes_agree(
            "fn five:\n    return 5\n\
             fn ten:\n    (five) + (five)\n\
             r1 = (ten)\nr2 = (five)\n",
        );
        assert_eq!(int_var(&rt, "r1"), 10);
        assert_eq!(int_var(&rt, "r2"), 5);
    }

    #[test]
    fn last_statement_value_semantics() {
        // assignments yield nil, so a body ending in one returns nil;
        // an if with a false condition and no else also yields nil
        let rt = assert_modes_agree(
            "fn assigns:\n    x = 5\n\
             fn cond_no_else n:\n    123\n    if n > 100:\n        456\n\
             r1 = (assigns)\nr2 = cond_no_else 1\nr3 = cond_no_else 1000\n",
        );
        assert_eq!(format!("{}", rt.get_var("r1").unwrap()), "Nil");
        assert_eq!(format!("{}", rt.get_var("r2").unwrap()), "Nil");
        assert_eq!(int_var(&rt, "r3"), 456);
    }

    #[test]
    fn loops_in_tail_position() {
        // a loop as the body's final statement: yields its else-block's
        // value on normal exit, nil on break exit or without an else
        let rt = assert_modes_agree(
            "fn tail_while_else n:\n    while n > 0:\n        n = n - 1\n    else:\n        Done\n\
             fn tail_while_break n:\n    while True:\n        if n > 2:\n            break\n        n = n + 1\n    else:\n        Done\n\
             fn tail_while_bare n:\n    while n > 0:\n        n = n - 1\n\
             fn tail_for xs:\n    for x in xs:\n        x\n    else:\n        x\n\
             r1 = tail_while_else 3\nr2 = tail_while_break 0\nr3 = tail_while_bare 3\n\
             ys = [7, 8]\nr4 = tail_for ys\n",
        );
        assert_eq!(format!("{}", rt.get_var("r1").unwrap()), "Done");
        assert_eq!(format!("{}", rt.get_var("r2").unwrap()), "Nil");
        assert_eq!(format!("{}", rt.get_var("r3").unwrap()), "Nil");
        assert_eq!(format!("{}", rt.get_var("r4").unwrap()), "8");
    }

    #[test]
    fn nested_fn_definitions_and_atoms() {
        let rt = assert_modes_agree(
            "fn classify n:\n    fn sign x:\n        if x > 0:\n            return Pos\n        if x < 0:\n            return Neg\n        return Zero\n    return sign n\n\
             r1 = classify 5\nr2 = classify (-5)\nr3 = classify 0\n",
        );
        assert_eq!(format!("{}", rt.get_var("r1").unwrap()), "Pos");
        assert_eq!(format!("{}", rt.get_var("r2").unwrap()), "Neg");
        assert_eq!(format!("{}", rt.get_var("r3").unwrap()), "Zero");
    }

    #[test]
    fn iterating_records_and_strings_of_ops() {
        let rt = assert_modes_agree(
            "fn count_fields rec:\n    n = 0\n    for f in rec:\n        n = n + 1\n    return n\n\
             r = count_fields { a: 1, b: 2, c: 3 }\n",
        );
        assert_eq!(int_var(&rt, "r"), 3);
    }

    #[test]
    fn runtime_errors_agree_between_modes() {
        for src in [
            // unbound identifier inside the body
            "fn bad:\n    return nosuchvar\nr = (bad)\n",
            // calling a non-function
            "fn bad:\n    x = 1\n    return x 2\nr = (bad)\n",
            // pattern match failure in a parameter
            "fn only_one 1:\n    return True\nr = only_one 2\n",
        ] {
            let walked = run_mode(src, false);
            let vmed = run_mode(src, true);
            assert!(walked.is_err(), "walker accepted: {src}");
            assert!(vmed.is_err(), "vm accepted: {src}");
        }
    }

    #[test]
    fn break_outside_loop_falls_back_to_walker() {
        // the compiler rejects this fn, so it silently falls back to the
        // AST closure; the error must still surface at call time
        let src = "fn bad:\n    break\nr = (bad)\n";
        let walked = run_mode(src, false);
        let vmed = run_mode(src, true);
        assert!(walked.is_err());
        assert!(vmed.is_err());
    }

    #[test]
    fn compiled_function_calls_host_function() {
        let src = "fn shout x:\n    emit x\n    emit (x * 2)\n";
        let stmts = ast_from_str(src);

        let seen = Rc::new(RefCell::new(Vec::new()));
        let sink = seen.clone();

        let mut rt = Runtime::new();
        rt.set_compile_fns(true);
        rt.register_function("emit", 0, None, move |rt, args| {
            for a in rt.vm.drain_off_stack(args).rev() {
                sink.borrow_mut().push(format!("{a}"));
            }
            Any::nil()
        });

        for statement in &stmts {
            rt.exec_stmt(statement).unwrap();
        }
        for statement in &ast_from_str("shout 21\n") {
            rt.exec_stmt(statement).unwrap();
        }

        assert_eq!(*seen.borrow(), vec!["21".to_string(), "42".to_string()]);
    }

    #[test]
    fn all_pattern_kinds_bind_identically_in_both_modes() {
        // atom tags, literal matches, and nested destructuring — every
        // CompiledPat variant, checked against the walker's bind_pattern
        let rt = assert_modes_agree(
            "fn area { kind: Circle, radius }:\n    return radius * radius * 3\n\
             fn greet \"hi\" name:\n    return name\n\
             fn is_x 'x':\n    return True\n\
             fn snd [_a, [b, c]]:\n    return b + c\n\
             fn one 1:\n    return One\n\
             fn onef 1.0 x:\n    return x\n\
             fn twoi 2i x:\n    return x\n\
             fn threef 3f x:\n    return x\n\
             c = { kind: Circle, radius: 2 }\n\
             r1 = area c\n\
             r2 = greet \"hi\" \"Ada\"\n\
             r3 = is_x 'x'\n\
             xs = [1, [2, 3]]\n\
             r4 = snd xs\n\
             fn twon 2 x:\n    return x\n\
             r5 = one 1\n\
             r6 = onef 1.0 42\n\
             r7 = twoi 2 43\n\
             r8 = twon 2.0 44\n\
             r9 = threef 3.0 45\n",
        );
        assert_eq!(format!("{}", rt.get_var("r1").unwrap()), "12");
        assert_eq!(format!("{}", rt.get_var("r2").unwrap()), "Ada");
        assert_eq!(format!("{}", rt.get_var("r3").unwrap()), "True");
        assert_eq!(format!("{}", rt.get_var("r4").unwrap()), "5");
        assert_eq!(format!("{}", rt.get_var("r5").unwrap()), "One");
        assert_eq!(format!("{}", rt.get_var("r6").unwrap()), "42");
        assert_eq!(format!("{}", rt.get_var("r7").unwrap()), "43");
        assert_eq!(format!("{}", rt.get_var("r8").unwrap()), "44");
        assert_eq!(format!("{}", rt.get_var("r9").unwrap()), "45");

        // mismatches fail identically too — suffixes pin the type: an
        // unsuffixed `1` matches float 1.0, but `1i` matches integers only
        // and `1.0`/`3f` match floats only
        for src in [
            "fn area { kind: Circle, radius }:\n    return radius\n\
             s = { kind: Square, side: 2 }\nr = area s\n",
            "fn one 1:\n    return One\nr = one 2\n",
            "fn onef 1.0:\n    return One\nr = onef 1\n",
            "fn threef 3f:\n    return One\nr = threef 3\n",
            "fn twoi 2i:\n    return One\nr = twoi 2.0\n",
            "fn is_x 'x':\n    return True\nr = is_x 'y'\n",
        ] {
            assert!(run_mode(src, false).is_err(), "walker accepted: {src}");
            assert!(run_mode(src, true).is_err(), "vm accepted: {src}");
        }

        // `1` as a pattern matches the float 1.0
        let rt = assert_modes_agree("fn one 1 x:\n    return x\nr = one 1.0 7\n");
        assert_eq!(format!("{}", rt.get_var("r").unwrap()), "7");
    }

    #[test]
    fn number_patterns_match_exactly() {
        let rt = assert_modes_agree(
            "fn inf_pat 1e999:\n    return Inf\n\
             fn zero 0f:\n    return Zero\n\
             fn big 9007199254740993:\n    return Big\n\
             r1 = inf_pat 2e999\n\
             r2 = zero (-0f)\n\
             r3 = big 9007199254740993\n",
        );
        // +inf matches +inf (the old |a-b| < ε test gave NaN for these)
        assert_eq!(format!("{}", rt.get_var("r1").unwrap()), "Inf");
        // -0.0 matches 0.0, like the language's own `==`
        assert_eq!(format!("{}", rt.get_var("r2").unwrap()), "Zero");
        assert_eq!(format!("{}", rt.get_var("r3").unwrap()), "Big");

        for src in [
            // exact means exact: no epsilon tolerance
            "fn tiny 1e-20:\n    return T\nr = tiny 2e-20\n",
            // +inf must not match -inf
            "fn inf_pat 1e999:\n    return Inf\nr = inf_pat (-1e999)\n",
            // 2^53+1 must not match the nearest representable float 2^53
            "fn big 9007199254740993:\n    return Big\nr = big 9007199254740992f\n",
            // a float outside i64 range matches no integer literal
            // (2^63 vs i64::MAX — saturation must not fake equality)
            "fn m 9223372036854775807:\n    return M\nr = m 9223372036854775808f\n",
            // an out-of-range integer literal matches nothing at all
            "fn huge 99999999999999999999999:\n    return H\nr = huge 1\n",
        ] {
            assert!(run_mode(src, false).is_err(), "walker accepted: {src}");
            assert!(run_mode(src, true).is_err(), "vm accepted: {src}");
        }
    }

    #[test]
    fn underscore_matches_anything_and_binds_nothing() {
        let rt = assert_modes_agree(
            "fn snd _ b:\n    return b\n\
             fn mid [_, m, _]:\n    return m\n\
             fn tag { kind: _, id }:\n    return id\n\
             fn count xs:\n    n = 0\n    for _ in xs:\n        n = n + 1\n    return n\n\
             fn discard x:\n    _ = x * 100\n    return x\n\
             xs = [7, 8, 9]\n\
             rec = { kind: Circle, id: 5 }\n\
             r1 = snd 1 2\n\
             r2 = mid xs\n\
             r3 = tag rec\n\
             r4 = count xs\n\
             r5 = discard 3\n",
        );
        assert_eq!(format!("{}", rt.get_var("r1").unwrap()), "2");
        assert_eq!(format!("{}", rt.get_var("r2").unwrap()), "8");
        assert_eq!(format!("{}", rt.get_var("r3").unwrap()), "5");
        assert_eq!(format!("{}", rt.get_var("r4").unwrap()), "3");
        assert_eq!(format!("{}", rt.get_var("r5").unwrap()), "3");
        // `_` was never bound anywhere along the way
        assert!(rt.get_var("_").is_none());

        // names merely *starting* with an underscore bind normally
        let rt = assert_modes_agree("fn keep _a:\n    return _a\nr = keep 42\n");
        assert_eq!(format!("{}", rt.get_var("r").unwrap()), "42");

        // `_` never binds, so *reading* it finds nothing (unless a global
        // named `_` exists) — an unbound-identifier error in both modes
        for src in ["fn bad _:\n    return _\nr = bad 1\n"] {
            assert!(run_mode(src, false).is_err(), "walker accepted: {src}");
            assert!(run_mode(src, true).is_err(), "vm accepted: {src}");
        }
    }

    #[test]
    fn locals_in_slots_fall_back_to_globals_before_first_assignment() {
        // `count` is assigned in the body, so it is a local slot — but the
        // read on the right-hand side happens before the store, and must
        // reach the *global* `count`; the global stays untouched after
        let rt = assert_modes_agree(
            "count = 10\n\
             fn bump:\n    count = count + 1\n    return count\n\
             r1 = (bump)\nr2 = (bump)\nr3 = count\n",
        );
        assert_eq!(format!("{}", rt.get_var("r1").unwrap()), "11");
        assert_eq!(format!("{}", rt.get_var("r2").unwrap()), "11");
        assert_eq!(format!("{}", rt.get_var("r3").unwrap()), "10");

        // reassigning a parameter reuses the parameter's own slot
        let rt = assert_modes_agree(
            "fn inc n:\n    n = n + 1\n    return n\n\
             fn swap_halves [a, b]:\n    t = a\n    a = b\n    b = t\n    return [a, b]\n\
             xs = [1, 2]\n\
             r1 = inc 41\nr2 = swap_halves xs\n",
        );
        assert_eq!(format!("{}", rt.get_var("r1").unwrap()), "42");
        assert_eq!(format!("{}", rt.get_var("r2").unwrap()), "[2, 1]");
    }

    #[test]
    fn pools_are_shared_and_deduplicated_across_functions() {
        let mut rt = Runtime::new();
        rt.set_compile_fns(true);
        for statement in &ast_from_str(
            "fn inc n:\n    return (shift n) + 1.5\n\
             fn dec n:\n    return (shift n) - 1.5\n\
             fn one:\n    return 1\n",
        ) {
            rt.exec_stmt(statement).unwrap();
        }

        // local names are frame slots and never touch the name pool...
        assert_eq!(
            rt.vm.names.iter().filter(|m| &m[..] == "n").count(),
            0,
            "slot-resolved local `n` should not be interned"
        );
        // ...while both functions reference the global `shift` and the
        // constant 1.5 through single shared pool entries
        assert_eq!(
            rt.vm.names.iter().filter(|m| &m[..] == "shift").count(),
            1,
            "global name `shift` interned more than once"
        );
        assert_eq!(
            rt.vm
                .consts
                .iter()
                .filter(|c| matches!(c, Any::Number(Number::Float(f)) if *f == 1.5))
                .count(),
            1,
            "constant 1.5 interned more than once"
        );
        // integers ride in opcodes/payloads and never enter the pool
        assert_eq!(
            rt.vm
                .consts
                .iter()
                .filter(|c| matches!(c, Any::Number(Number::Integer(_))))
                .count(),
            0,
            "integers should be immediates, not pool constants"
        );
    }

    #[test]
    fn shared_stack_is_clean_after_calls_and_visible_during_them() {
        let mut rt = Runtime::new();
        rt.set_compile_fns(true);

        // a host function that reports how deep the caller's frame is —
        // reading the runtime's operand stack mid-call, as promised
        rt.register_function("depth", 0, Some(0), |rt, _args| {
            Any::Number(Number::Integer(rt.vm.stack.len() as i64))
        });

        for statement in &ast_from_str(
            "fn probe:\n    return 100 + (depth)\n\
             fn fib n:\n    if n < 2:\n        return n\n    return (fib (n - 1)) + (fib (n - 2))\n",
        ) {
            rt.exec_stmt(statement).unwrap();
        }

        // inside `probe` the temporary `100` sits on the stack when
        // `depth` runs, so it reports 1
        for statement in &ast_from_str("r = (probe)\n") {
            rt.exec_stmt(statement).unwrap();
        }
        assert_eq!(format!("{}", rt.get_var("r").unwrap()), "101");

        // recursion nests frames on the one shared stack and unwinds fully
        for statement in &ast_from_str("f = fib 12\n") {
            rt.exec_stmt(statement).unwrap();
        }
        assert_eq!(format!("{}", rt.get_var("f").unwrap()), "144");
        assert!(rt.vm.stack.is_empty(), "stack not restored after calls");

        // ...and is restored even when a frame errors out mid-expression
        let stmts = ast_from_str("fn bad n:\n    return n + nosuchvar\nfib (bad 1)\n");
        for statement in &stmts[..1] {
            rt.exec_stmt(statement).unwrap();
        }
        assert!(rt.exec_stmt(&stmts[1]).is_err());
        assert!(rt.vm.stack.is_empty(), "stack not restored after error");
    }

    #[test]
    fn modes_mix_in_one_runtime() {
        let mut rt = Runtime::new();

        // `double` walks the AST...
        rt.set_compile_fns(false);
        for statement in &ast_from_str("fn double x:\n    return x * 2\n") {
            rt.exec_stmt(statement).unwrap();
        }

        // ...`quad` runs on the VM and calls `double`...
        rt.set_compile_fns(true);
        for statement in &ast_from_str("fn quad x:\n    return double (double x)\n") {
            rt.exec_stmt(statement).unwrap();
        }

        // ...`octo` walks the AST again and calls compiled `quad`...
        rt.set_compile_fns(false);
        for statement in &ast_from_str("fn octo x:\n    return double (quad x)\n") {
            rt.exec_stmt(statement).unwrap();
        }

        // ...and the top level is interpreted from the AST.
        for statement in &ast_from_str("r = octo 5\n") {
            rt.exec_stmt(statement).unwrap();
        }

        assert_eq!(
            match rt.get_var("r") {
                Some(Any::Number(Number::Integer(i))) => i,
                other => panic!("unexpected {other:?}"),
            },
            40
        );
    }

    #[test]
    fn top_level_expression_sees_compiled_function() {
        let mut rt = Runtime::new();
        rt.set_compile_fns(true);
        for statement in &ast_from_str("fn triple x:\n    return x * 3\n") {
            rt.exec_stmt(statement).unwrap();
        }
        // interpreted expression calling into bytecode
        let stmts = ast_from_str("triple 14\n");
        let Exec::Value(v) = rt.exec_stmt(&stmts[0]).unwrap() else {
            panic!("expected value");
        };
        assert_eq!(format!("{v}"), "42");
    }
}
