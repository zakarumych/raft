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
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};
use core::{cell::RefCell, fmt};
use smallvec::SmallVec;

use raft_ast::{BinOpKind, Expr, ExprKind, Lit, Pat, PatKind, Span, Stmt, StmtKind, UnOpKind};

use crate::{
    Atom, ConstId, Context, DynFn, FixedHashMap, FnVal, Frame, Host, Number, ObjectKind, PatId, Runtime, RuntimeError, SlotId, SlotTable, StringId, Val, assign_field, assign_index, eval_binary, eval_unary, field_of, index_of, is_falsey, literal_value,
};

/// Index into a *defining function's own* `consts`/`templates` arrays
/// (never a global pool — a nested `fn` is only ever referenced from the
/// one `Instr::MakeClosure` site that defines it). Flattened: `0..consts.len()`
/// addresses `consts`, `consts.len()..` addresses `templates`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct FnId(pub u32);

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
    /// `→ v` — push a copy of frame slot `slot`.
    /// If unassigned, loads from outer scope using slot's name index stored in function object.
    LoadSlot(SlotId),
    /// `v →` — pop and store into frame slot `slot` (assignments inside a
    /// function always target locals).
    StoreSlot(SlotId),
    /// `→ v` — push parent's slot `slot` of the executing function's module.
    /// Falls back to the global scope using the slot's name index stored in the module object.
    LoadParent(SlotId),
    /// `→` — add/subtract 1 to frame slot `slot` in place (no stack
    /// traffic). Falls back to the global `names[name]` when the slot is
    /// still unassigned, exactly like `LoadLocal`.
    IncSlot(SlotId),
    DecSlot(SlotId),
    /// `v1 .. vn → list` — pop `n` values, push a new list of them.
    MakeList(u32),
    /// `k1 v1 .. kn vn → record` — pop `n` key/value pairs (keys are
    /// string constants pushed by the compiler), push a new record.
    MakeRecord(u32),
    /// `f a1 .. an → ret` — apply `f` to `n` arguments with the language's
    /// currying rules: a callee consuming fewer than `n` arguments has the
    /// leftovers re-applied to its result.
    Call(u32),
    /// `→ const` — push `consts[i]`.
    Const(ConstId),
    /// `→ v` — push global variable `names[i]`; error if unbound. Used for
    /// names never assigned in the function — they can only be globals.
    LoadGlobal(StringId),
    /// `v →` — pop and bind against pattern `pats[i]`, which may bind
    /// several frame slots (destructuring) or fail the match with an error.
    Bind(PatId),
    /// `obj → obj.names[i]` — read a record field.
    GetField(StringId),
    /// `obj v →` — write record field `names[i]`.
    SetField(StringId),
    /// `a → op(a)` — apply a unary operator.
    Unary(UnOpKind),
    /// `a b → a op b` — apply a binary operator.
    Binary(BinOpKind),
    /// `v →` — pop and discard.
    Pop,
    /// `obj idx → obj[idx]` — read a list element.
    GetIndex,
    /// `obj idx v →` — write a list element.
    SetIndex,
    /// `iterable →` — pop, open an iterator over it and push it on the
    /// iterator stack.
    IterInit,
    /// `→` — close the innermost iterator (used by `break` in a `for`).
    IterPop,
    /// `v →` — pop the return value and leave the function.
    Return,
    /// `→` — unconditional jump to code index `t`.
    Jump(u32),
    /// `c →` — pop; jump to `t` if the value is falsey.
    JumpIfFalse(u32),
    /// `→ item` *or* jump — advance the innermost iterator: push the next
    /// item, or (when exhausted) close the iterator and jump to `t`.
    IterNext(u32),
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
    /// `a → a op n` — apply a binary operator whose right operand is an
    /// integer carried by the opcode. `Add`/`Sub` take values from
    /// [`SMALL_INTS`], `BitAnd`/`BitOr`/`BitXor` from [`TINY_MASKS`], and
    /// the six comparisons from [`TINY_INTS`].
    BinaryInt(BinOpKind, i16),
    /// `→ v` — push a copy of own-frame slot `slot` (a local some nested
    /// `fn` captures). Falls back to the enclosing scope, same as
    /// [`Instr::LoadSlot`], when not yet assigned.
    LoadCap(SlotId),
    /// `v →` — pop and store into own-frame slot `slot`.
    StoreCap(SlotId),
    /// `→ v` — push nested `fn` `i` from the *defining function's own*
    /// `consts`/`templates` arrays (flattened: `i < consts.len()` clones a
    /// ready value, otherwise instantiates `templates[i - consts.len()]`,
    /// attaching the current function's captured-frame — or, if it has
    /// none, its own parent — as the new closure's parent scope). A
    /// nested `fn` is only ever read from the one site that defines it,
    /// so it lives on the defining function rather than a shared pool.
    MakeClosure(FnId),
    /// `→ v` — push `names[i]` read by name from the function's parent
    /// scope, walking as many further ancestors as needed. Used when a
    /// outer name doesn't live in the *immediate* parent's own schema (so
    /// no fixed [`SlotId`] can be resolved at compile time) — [`Instr::LoadParent`]
    /// stays the fast path for the common one-hop case.
    LoadParentByName(StringId),
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
    /// operand inline, `base+8/9` take a u8/u16 payload.
    pub const LOAD_SLOT: u8 = 0;
    pub const LOAD_SLOT_END: u8 = LOAD_SLOT + 10;
    pub const STORE_SLOT: u8 = LOAD_SLOT_END;
    pub const STORE_SLOT_END: u8 = STORE_SLOT + 10;
    pub const LOAD_PARENT: u8 = STORE_SLOT_END;
    pub const LOAD_PARENT_END: u8 = LOAD_PARENT + 10;
    pub const INC_SLOT: u8 = LOAD_PARENT_END;
    pub const INC_SLOT_END: u8 = INC_SLOT + 10;
    pub const DEC_SLOT: u8 = INC_SLOT_END;
    pub const DEC_SLOT_END: u8 = DEC_SLOT + 10;
    /// Own-frame equivalents of LOAD_SLOT/STORE_SLOT, for locals a nested
    /// `fn` captures — same inline/u8/u16/u32 operand shape, just reading
    /// and writing the function's reified [`crate::Frame`] instead of the
    /// shared operand stack.
    pub const LOAD_CAP: u8 = DEC_SLOT_END;
    pub const LOAD_CAP_END: u8 = LOAD_CAP + 10;
    pub const STORE_CAP: u8 = LOAD_CAP_END;
    pub const STORE_CAP_END: u8 = STORE_CAP + 10;
    pub const MAKE_LIST: u8 = STORE_CAP_END;
    pub const MAKE_LIST_END: u8 = MAKE_LIST + 10;
    pub const MAKE_RECORD: u8 = MAKE_LIST_END;
    pub const MAKE_RECORD_END: u8 = MAKE_RECORD + 10;
    pub const CALL: u8 = MAKE_RECORD_END;
    pub const CALL_END: u8 = CALL + 10;

    /// Operands indexing the shared `VmContext` pools grow with the whole
    /// program, so inline values would be wasted opcodes — these blocks
    /// only encode the operand's byte width: `base+0/1/2` = u8/u16/u32.
    pub const CONST: u8 = CALL_END;
    pub const CONST_END: u8 = CONST + 3;
    pub const LOAD_GLOBAL: u8 = CONST_END;
    pub const LOAD_GLOBAL_END: u8 = LOAD_GLOBAL + 3;
    pub const BIND: u8 = LOAD_GLOBAL_END;
    pub const BIND_END: u8 = BIND + 3;
    pub const GET_FIELD: u8 = BIND_END;
    pub const GET_FIELD_END: u8 = GET_FIELD + 3;
    pub const SET_FIELD: u8 = GET_FIELD_END;
    pub const SET_FIELD_END: u8 = SET_FIELD + 3;
    /// `→ closure` — instantiate a `fn` template from the const pool,
    /// attaching the executing function's own captured-frame (or, if it
    /// has none, passing its own parent straight through) as the new
    /// closure's parent scope.
    pub const MAKE_CLOSURE: u8 = SET_FIELD_END;
    pub const MAKE_CLOSURE_END: u8 = MAKE_CLOSURE + 3;
    pub const LOAD_PARENT_BY_NAME: u8 = MAKE_CLOSURE_END;
    pub const LOAD_PARENT_BY_NAME_END: u8 = LOAD_PARENT_BY_NAME + 3;

    /// One opcode per operator kind — no operand bytes at all.
    pub const UNARY: u8 = LOAD_PARENT_BY_NAME_END; // + unop_to_byte(kind), 4 kinds
    pub const UNARY_END: u8 = UNARY + 4;
    pub const BINARY: u8 = UNARY_END; // + binop_to_byte(kind), 16 kinds
    pub const BINARY_END: u8 = BINARY + 16;

    /// No operands.
    pub const POP: u8 = BINARY_END;
    pub const GET_INDEX: u8 = POP + 1;
    pub const SET_INDEX: u8 = GET_INDEX + 1;
    pub const ITER_INIT: u8 = SET_INDEX + 1;
    pub const ITER_POP: u8 = ITER_INIT + 1;
    pub const RETURN: u8 = ITER_POP + 1;

    /// Fixed 4-byte little-endian byte-offset operand.
    pub const JUMP: u8 = RETURN + 1;
    pub const JUMP_IF_FALSE: u8 = JUMP + 1;
    pub const ITER_NEXT: u8 = JUMP_IF_FALSE + 1;

    /// `True`/`False` are common enough to deserve their own opcodes, and
    pub const NIL: u8 = ITER_NEXT + 1;
    pub const TRUE: u8 = NIL + 1;
    pub const FALSE: u8 = TRUE + 1;

    /// Immediate values - one opcode per entry of
    /// [`super::SMALL_INTS`] (`INT + index`). No operand bytes and no
    /// const-pool access at runtime.
    pub const INT: u8 = FALSE + 1;
    pub const INT_END: u8 = INT + super::SMALL_INTS.len() as u8;

    /// Integer immediates too large for [`super::SMALL_INTS`] but fitting
    /// 8/16 bits: the value follows as a little-endian payload,
    /// sign-extended (`INT_*`) or zero-extended (`UINT_*`) to `i64`.
    /// Anything larger lives in the const pool.
    pub const INT_8: u8 = INT_END;
    pub const INT_16: u8 = INT_8 + 1;
    pub const UINT_8: u8 = INT_16 + 1;
    pub const UINT_16: u8 = UINT_8 + 1;

    /// Fused binary ops with a small-integer right operand: `base + index`
    /// into [`super::SMALL_INTS`]. One instruction, no operand bytes, no
    /// const-pool access.
    pub const ADD_INT: u8 = UINT_16 + 1;
    pub const ADD_INT_END: u8 = ADD_INT + super::SMALL_INTS.len() as u8;
    pub const SUB_INT: u8 = ADD_INT_END;
    pub const SUB_INT_END: u8 = SUB_INT + super::SMALL_INTS.len() as u8;

    /// Fused bitwise ops with a mask from [`super::TINY_MASKS`]
    /// (`base + index`), no operand bytes.
    pub const AND_MASK: u8 = SUB_INT_END;
    pub const AND_MASK_END: u8 = AND_MASK + super::TINY_MASKS.len() as u8;
    pub const OR_MASK: u8 = AND_MASK_END;
    pub const OR_MASK_END: u8 = OR_MASK + super::TINY_MASKS.len() as u8;
    pub const XOR_MASK: u8 = OR_MASK_END;
    pub const XOR_MASK_END: u8 = XOR_MASK + super::TINY_MASKS.len() as u8;

    /// Fused comparisons with an integer from [`super::TINY_INTS`]
    /// (`base + index`), no operand bytes.
    pub const INT_EQ: u8 = XOR_MASK_END;
    pub const INT_EQ_END: u8 = INT_EQ + super::TINY_INTS.len() as u8;
    pub const INT_NE: u8 = INT_EQ_END;
    pub const INT_NE_END: u8 = INT_NE + super::TINY_INTS.len() as u8;
    pub const INT_LT: u8 = INT_NE_END;
    pub const INT_LT_END: u8 = INT_LT + super::TINY_INTS.len() as u8;
    pub const INT_GT: u8 = INT_LT_END;
    pub const INT_GT_END: u8 = INT_GT + super::TINY_INTS.len() as u8;
    pub const INT_LE: u8 = INT_GT_END;
    pub const INT_LE_END: u8 = INT_LE + super::TINY_INTS.len() as u8;
    pub const INT_GE: u8 = INT_LE_END;
    pub const INT_GE_END: u8 = INT_GE + super::TINY_INTS.len() as u8;
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
        _ => panic!("operand_variant: value too large for a slot-style operand"),
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
fn width_len(v: u32) -> usize {
    match v {
        0..=0xff => 1,
        0x100..=0xffff => 2,
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
    RuntimeError::Other("vm: truncated bytecode".into())
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
        _ => return Err(RuntimeError::Other("vm: bad unary op".into())),
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
        _ => return Err(RuntimeError::Other("vm: bad binary op".into())),
    })
}

/// Encoded size of one instruction: opcode byte + operands.
fn instr_len(i: &Instr) -> usize {
    1 + match i {
        Instr::LoadSlot(SlotId(v))
        | Instr::StoreSlot(SlotId(v))
        | Instr::LoadParent(SlotId(v))
        | Instr::MakeList(v)
        | Instr::MakeRecord(v)
        | Instr::Call(v)
        | Instr::IncSlot(SlotId(v))
        | Instr::DecSlot(SlotId(v))
        | Instr::LoadCap(SlotId(v))
        | Instr::StoreCap(SlotId(v)) => operand_len(*v),

        Instr::Const(ConstId(v))
        | Instr::LoadGlobal(StringId(v))
        | Instr::Bind(PatId(v))
        | Instr::GetField(StringId(v))
        | Instr::SetField(StringId(v))
        | Instr::MakeClosure(FnId(v))
        | Instr::LoadParentByName(StringId(v)) => width_len(*v),

        Instr::Unary(_) | Instr::Binary(_) | Instr::BinaryInt(..) => 0,

        Instr::Jump(_) | Instr::JumpIfFalse(_) | Instr::IterNext(_) => 4,
        Instr::Int(n) => int_payload_len(*n),
        Instr::Nil
        | Instr::True
        | Instr::False
        | Instr::Pop
        | Instr::GetIndex
        | Instr::SetIndex
        | Instr::IterInit
        | Instr::IterPop
        | Instr::Return => 0,
    }
}

/// A function's instructions in their executable form: a flat byte array.
/// Jump operands inside are byte offsets into this array. Decode back to
/// [`Instr`] with [`Code::disassemble`] (the `Debug` impl prints a
/// listing).
#[derive(Clone)]
pub struct Code {
    bytes: Rc<[u8]>,
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
            Instr::Const(v) => push_wop(&mut bytes, opcode::CONST, v.0),
            Instr::Nil => bytes.push(opcode::NIL),
            Instr::True => bytes.push(opcode::TRUE),
            Instr::False => bytes.push(opcode::FALSE),
            Instr::Int(n) => push_int(&mut bytes, n),
            Instr::Pop => bytes.push(opcode::POP),
            Instr::LoadSlot(v) => push_op(&mut bytes, opcode::LOAD_SLOT, v.0),
            Instr::LoadParent(slot) => push_op(&mut bytes, opcode::LOAD_PARENT, slot.0),
            Instr::LoadGlobal(v) => push_wop(&mut bytes, opcode::LOAD_GLOBAL, v.0),
            Instr::StoreSlot(v) => push_op(&mut bytes, opcode::STORE_SLOT, v.0),
            Instr::Bind(v) => push_wop(&mut bytes, opcode::BIND, v.0),
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
            Instr::IncSlot(slot) | Instr::DecSlot(slot) => push_op(
                &mut bytes,
                if matches!(*i, Instr::IncSlot { .. }) {
                    opcode::INC_SLOT
                } else {
                    opcode::DEC_SLOT
                },
                slot.0,
            ),
            Instr::GetField(v) => push_wop(&mut bytes, opcode::GET_FIELD, v.0),
            Instr::GetIndex => bytes.push(opcode::GET_INDEX),
            Instr::SetField(v) => push_wop(&mut bytes, opcode::SET_FIELD, v.0),
            Instr::SetIndex => bytes.push(opcode::SET_INDEX),
            Instr::Call(v) => push_op(&mut bytes, opcode::CALL, v),
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
            Instr::LoadCap(slot) => push_op(&mut bytes, opcode::LOAD_CAP, slot.0),
            Instr::StoreCap(slot) => push_op(&mut bytes, opcode::STORE_CAP, slot.0),
            Instr::MakeClosure(t) => push_wop(&mut bytes, opcode::MAKE_CLOSURE, t.0),
            Instr::LoadParentByName(v) => {
                push_wop(&mut bytes, opcode::LOAD_PARENT_BY_NAME, v.0)
            }
        }
    }

    Code {
        bytes: Rc::from(bytes),
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
        let instr =
            match op {
                opcode::CONST..opcode::CONST_END => {
                    Instr::Const(ConstId(decode_wop(op - opcode::CONST, code, &mut pc)?))
                }
                opcode::LOAD_SLOT..opcode::LOAD_SLOT_END => Instr::LoadSlot(SlotId(
                    decode_operand(op - opcode::LOAD_SLOT, code, &mut pc)?,
                )),
                opcode::LOAD_PARENT..opcode::LOAD_PARENT_END => {
                    Instr::LoadParent(SlotId(decode_wop(op - opcode::LOAD_PARENT, code, &mut pc)?))
                }
                opcode::LOAD_GLOBAL..opcode::LOAD_GLOBAL_END => Instr::LoadGlobal(StringId(
                    decode_wop(op - opcode::LOAD_GLOBAL, code, &mut pc)?,
                )),
                opcode::STORE_SLOT..opcode::STORE_SLOT_END => Instr::StoreSlot(SlotId(
                    decode_operand(op - opcode::STORE_SLOT, code, &mut pc)?,
                )),
                opcode::BIND..opcode::BIND_END => {
                    Instr::Bind(PatId(decode_wop(op - opcode::BIND, code, &mut pc)?))
                }
                opcode::MAKE_LIST..opcode::MAKE_LIST_END => {
                    Instr::MakeList(decode_operand(op - opcode::MAKE_LIST, code, &mut pc)?)
                }
                opcode::MAKE_RECORD..opcode::MAKE_RECORD_END => {
                    Instr::MakeRecord(decode_operand(op - opcode::MAKE_RECORD, code, &mut pc)?)
                }
                opcode::GET_FIELD..opcode::GET_FIELD_END => {
                    Instr::GetField(StringId(decode_wop(op - opcode::GET_FIELD, code, &mut pc)?))
                }
                opcode::SET_FIELD..opcode::SET_FIELD_END => {
                    Instr::SetField(StringId(decode_wop(op - opcode::SET_FIELD, code, &mut pc)?))
                }
                opcode::CALL..opcode::CALL_END => {
                    Instr::Call(decode_operand(op - opcode::CALL, code, &mut pc)?)
                }
                opcode::UNARY..opcode::UNARY_END => Instr::Unary(byte_to_unop(op - opcode::UNARY)?),
                opcode::BINARY..opcode::BINARY_END => {
                    Instr::Binary(byte_to_binop(op - opcode::BINARY)?)
                }
                opcode::ADD_INT..opcode::ADD_INT_END => Instr::BinaryInt(
                    BinOpKind::Add,
                    SMALL_INTS[(op - opcode::ADD_INT) as usize] as i16,
                ),
                opcode::SUB_INT..opcode::SUB_INT_END => Instr::BinaryInt(
                    BinOpKind::Sub,
                    SMALL_INTS[(op - opcode::SUB_INT) as usize] as i16,
                ),
                opcode::AND_MASK..opcode::AND_MASK_END => Instr::BinaryInt(
                    BinOpKind::BitAnd,
                    TINY_MASKS[(op - opcode::AND_MASK) as usize] as i16,
                ),
                opcode::OR_MASK..opcode::OR_MASK_END => Instr::BinaryInt(
                    BinOpKind::BitOr,
                    TINY_MASKS[(op - opcode::OR_MASK) as usize] as i16,
                ),
                opcode::XOR_MASK..opcode::XOR_MASK_END => Instr::BinaryInt(
                    BinOpKind::BitXor,
                    TINY_MASKS[(op - opcode::XOR_MASK) as usize] as i16,
                ),
                opcode::INT_EQ..opcode::INT_EQ_END => Instr::BinaryInt(
                    BinOpKind::Eq,
                    TINY_INTS[(op - opcode::INT_EQ) as usize] as i16,
                ),
                opcode::INT_NE..opcode::INT_NE_END => Instr::BinaryInt(
                    BinOpKind::Ne,
                    TINY_INTS[(op - opcode::INT_NE) as usize] as i16,
                ),
                opcode::INT_LT..opcode::INT_LT_END => Instr::BinaryInt(
                    BinOpKind::Lt,
                    TINY_INTS[(op - opcode::INT_LT) as usize] as i16,
                ),
                opcode::INT_GT..opcode::INT_GT_END => Instr::BinaryInt(
                    BinOpKind::Gt,
                    TINY_INTS[(op - opcode::INT_GT) as usize] as i16,
                ),
                opcode::INT_LE..opcode::INT_LE_END => Instr::BinaryInt(
                    BinOpKind::Le,
                    TINY_INTS[(op - opcode::INT_LE) as usize] as i16,
                ),
                opcode::INT_GE..opcode::INT_GE_END => Instr::BinaryInt(
                    BinOpKind::Ge,
                    TINY_INTS[(op - opcode::INT_GE) as usize] as i16,
                ),
                opcode::INC_SLOT..opcode::INC_SLOT_END => Instr::IncSlot(SlotId(decode_operand(
                    op - opcode::INC_SLOT,
                    code,
                    &mut pc,
                )?)),
                opcode::DEC_SLOT..opcode::DEC_SLOT_END => Instr::DecSlot(SlotId(decode_operand(
                    op - opcode::DEC_SLOT,
                    code,
                    &mut pc,
                )?)),
                opcode::NIL => Instr::Nil,
                opcode::TRUE => Instr::True,
                opcode::FALSE => Instr::False,
                opcode::INT..opcode::INT_END => Instr::Int(SMALL_INTS[(op - opcode::INT) as usize]),
                opcode::INT_8 => Instr::Int(read_u8(code, &mut pc)? as i8 as i64),
                opcode::INT_16 => Instr::Int(read_u16(code, &mut pc)? as i16 as i64),
                opcode::UINT_8 => Instr::Int(read_u8(code, &mut pc)? as i64),
                opcode::UINT_16 => Instr::Int(read_u16(code, &mut pc)? as i64),
                opcode::POP => Instr::Pop,
                opcode::GET_INDEX => Instr::GetIndex,
                opcode::SET_INDEX => Instr::SetIndex,
                opcode::ITER_INIT => Instr::IterInit,
                opcode::ITER_POP => Instr::IterPop,
                opcode::RETURN => Instr::Return,
                opcode::JUMP => Instr::Jump(read_u32(code, &mut pc)?),
                opcode::JUMP_IF_FALSE => Instr::JumpIfFalse(read_u32(code, &mut pc)?),
                opcode::ITER_NEXT => Instr::IterNext(read_u32(code, &mut pc)?),
                opcode::LOAD_CAP..opcode::LOAD_CAP_END => Instr::LoadCap(SlotId(decode_operand(
                    op - opcode::LOAD_CAP,
                    code,
                    &mut pc,
                )?)),
                opcode::STORE_CAP..opcode::STORE_CAP_END => Instr::StoreCap(SlotId(
                    decode_operand(op - opcode::STORE_CAP, code, &mut pc)?,
                )),
                opcode::MAKE_CLOSURE..opcode::MAKE_CLOSURE_END => Instr::MakeClosure(FnId(
                    decode_wop(op - opcode::MAKE_CLOSURE, code, &mut pc)?,
                )),
                opcode::LOAD_PARENT_BY_NAME..opcode::LOAD_PARENT_BY_NAME_END => {
                    Instr::LoadParentByName(StringId(decode_wop(
                        op - opcode::LOAD_PARENT_BY_NAME,
                        code,
                        &mut pc,
                    )?))
                }
                _ => return Err(RuntimeError::Other("vm: unknown opcode".into())),
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
    Var(SlotId),
    /// Match an atom by name.
    Atom(Atom),
    /// Match a number literal (see [`NumberPat`] for the semantics the
    /// suffix selects).
    Number(NumberPat),
    /// Match a string literal (unescaped).
    String(Rc<str>),
    /// Match a char literal (unescaped).
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

/// Lower an AST pattern, resolving bound names to frame slots through
/// `slots`. Infallible: bad suffixes are rejected by the parser before
/// patterns exist, and an out-of-range number literal compiles to a
/// pattern that matches nothing (`NumberPat::Never`) — the same non-match
/// the tree walker produces for it.
fn compile_pat(pattern: &Pat, slots: &SlotTable, ctx: &mut Context) -> CompiledPat {
    fn slot_of(slots: &SlotTable, name: &str) -> SlotId {
        slots
            .get(name)
            .expect("collect_slots missed a pattern name")
    }

    match pattern.kind() {
        PatKind::Ident(id) if id.name() == "_" => CompiledPat::Ignore,
        PatKind::Ident(id) => CompiledPat::Var(slot_of(slots, id.name())),
        PatKind::Atom(a) => CompiledPat::Atom(Atom::new(a.rc_name())),
        PatKind::Literal(lit) => match lit {
            Lit::Num(n) => CompiledPat::Number(NumberPat::from_literal(n)),
            Lit::Str(s) => CompiledPat::String(s.unescape().into()),
            Lit::Char(c) => CompiledPat::Char(c.unescape()),
        },
        PatKind::List(items) => {
            CompiledPat::List(items.iter().map(|p| compile_pat(p, slots, ctx)).collect())
        }
        PatKind::Record(fields) => {
            let fields = fields
                .iter()
                .filter_map(|f| {
                    let key = f.key().rc_name();

                    let pattern = match f.pattern() {
                        Some(p) => compile_pat(p, slots, ctx),
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
    pub fn bind(&self, rt: &mut Runtime, base: usize, val: &Val) -> Result<(), RuntimeError> {
        fn fail() -> RuntimeError {
            RuntimeError::Other("pattern match failed".into())
        }

        match self {
            CompiledPat::Ignore => Ok(()),
            CompiledPat::Var(slot) => {
                rt.stack.set(base + slot.0 as usize, val.clone());
                Ok(())
            }
            CompiledPat::Atom(a) => match val {
                Val::Atom(av) if av == a => Ok(()),
                _ => Err(fail()),
            },
            CompiledPat::Number(expected) => match val {
                Val::Number(actual) if expected.matches(*actual) => Ok(()),
                _ => Err(fail()),
            },
            CompiledPat::String(s) => match val {
                Val::String(v) if v == s => Ok(()),
                _ => Err(fail()),
            },
            CompiledPat::Char(c) => match val {
                Val::Char(v) if v == c => Ok(()),
                _ => Err(fail()),
            },
            CompiledPat::List(items) => match val {
                Val::Object(o) => match &o.borrow().kind {
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
                Val::Object(o) => match &o.borrow().kind {
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

/// Runtime storage for a compiled scope's reified captured locals. Chains
/// to whatever *its own* nearest used ancestor is (`outer`) — set once,
/// at materialization time, from whichever `CompiledFn`/`CompiledFrame`
/// created it — so a multi-level lookup keeps walking past this frame if
/// the slot it lands on here was itself never assigned. A scope that
/// captures nothing (and whose descendants read nothing from it) never
/// gets one of these at all: `Instr::MakeClosure` skips straight past it,
/// so no hop — nor allocation — is ever spent on a transparent level.
#[derive(Debug)]
pub struct CompiledFrame {
    names: SmallVec<[StringId; 8]>,
    slots: RefCell<SmallVec<[Val; 8]>>,
    /// Per-own-slot fallback for a read-before-assignment: `Some(offset)`
    /// is a flat offset to keep walking from `outer`; `None` falls to
    /// `outer_named` by name, then the global scope.
    fallback: Rc<[Option<u32>]>,
    outer: Option<Rc<CompiledFrame>>,
    outer_named: Option<Rc<Frame>>,
}

impl CompiledFrame {
    fn len(&self) -> usize {
        self.names.len()
    }

    pub(crate) fn get_local(&self, slot: SlotId) -> Val {
        self.slots.borrow()[slot.0 as usize].clone()
    }

    fn set(&self, slot: SlotId, val: Val) {
        self.slots.borrow_mut()[slot.0 as usize] = val;
    }

    /// Read `slot`, falling back through `outer`/`outer_named`/globals if
    /// it was never assigned.
    fn get(&self, slot: SlotId, rt: &mut Runtime) -> Result<Val, RuntimeError> {
        let val = self.get_local(slot);
        if !matches!(val, Val::Uninit) {
            return Ok(val);
        }
        core::hint::cold_path();
        self.resolve_outer(slot, rt)
    }

    fn resolve_outer(&self, slot: SlotId, rt: &mut Runtime) -> Result<Val, RuntimeError> {
        let val = match self.fallback[slot.0 as usize] {
            Some(offset) => match &self.outer {
                Some(o) => o.get_flat(offset, rt)?,
                None => Val::Uninit,
            },
            None => {
                let name = self.names[slot.0 as usize];
                match &self.outer_named {
                    Some(f) => f.get_var(name, rt),
                    None => rt.get_var(name),
                }
            }
        };
        val.init_or_else(|| {
            RuntimeError::UnboundIdentifier(rt.ctx.get_string(self.names[slot.0 as usize]))
        })
    }

    /// Walk a flat offset starting at `self`, subtracting each frame's own
    /// slot count until it lands within one, then reads (with fallback)
    /// there. `Instr::MakeClosure` only ever chains together frames that
    /// actually own something, so every hop here is real work, never a
    /// wasted step through a transparent level.
    fn get_flat(&self, mut offset: u32, rt: &mut Runtime) -> Result<Val, RuntimeError> {
        let mut frame = self;
        loop {
            let size = frame.len() as u32;
            if offset < size {
                return frame.get(SlotId(offset), rt);
            }
            offset -= size;
            match &frame.outer {
                Some(next) => frame = next,
                None => {
                    return Err(RuntimeError::Other(
                        "vm: flat outer slot out of range".into(),
                    ));
                }
            }
        }
    }
}

/// A function's own name schema, fixed at compile time — never holds slot
/// values. Ancestor schemas are threaded separately during compilation as
/// a flat list (see `Compiler::ancestors`/`CompileParent`) rather than a
/// chain living on `Schema` itself — a schema only ever needs to describe
/// its own scope.
#[derive(Debug)]
pub struct Schema {
    names: SmallVec<[StringId; 8]>,
    /// Whether this specific scope actually reifies captured locals into
    /// live storage at runtime (occupies a slot in the flat address space,
    /// and gets a real [`CompiledFrame`] at `Instr::MakeClosure` time) or
    /// is a transparent pass-through (skipped by both the compile-time
    /// walk and the runtime attachment).
    owns_frame: bool,
}

impl Schema {
    fn name_slot(&self, name: StringId) -> Option<SlotId> {
        self.names
            .iter()
            .position(|&n| n == name)
            .map(|i| SlotId(i as u32))
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

    pub frame_size: u32,

    /// Variable-length-encoded instructions (see [`Code`]).
    pub code: Code,
    /// This function's own local names, indexed by [`SlotId`] — the
    /// by-name fallback source for a stack/captured slot read before
    /// assignment when [`CompiledFn::fallback`] has no flat answer. Also
    /// what a fresh [`CompiledFrame`] is built from when this function
    /// owns one.
    pub(crate) own_names: SmallVec<[StringId; 8]>,
    /// Per-own-slot precomputed fallback target for a read-before-assignment:
    /// `Some(offset)` is a flat offset into `outer`; `None` falls to
    /// `outer_named` by name, then the global scope.
    fallback: Rc<[Option<u32>]>,
    /// Whether this function has locals some nested `fn` captures — if so,
    /// each call reifies a fresh [`CompiledFrame`] for just those slots
    /// (`Instr::LoadCap`/`Instr::StoreCap`); otherwise every local stays a
    /// zero-allocation stack slot, exactly like a function with no nested
    /// closures at all.
    pub owns_frame: bool,
    /// The nearest ancestor [`CompiledFrame`] this function (or something
    /// it calls through `Instr::MakeClosure`) actually reads from — using
    /// it by way of a nested closure counts as using it. `None` for a
    /// plain (non-instantiated) compiled function: only ever populated
    /// when `Instr::MakeClosure` builds a real closure over a live
    /// enclosing scope.
    outer: Option<Rc<CompiledFrame>>,
    /// Nearest AST-walking ancestor, if this function (or its compiled
    /// lineage) was ultimately defined inside walked code. `None` inside a
    /// module, which has no walked ancestor by construction.
    outer_named: Option<Rc<Frame>>,
    /// Nested `fn`s defined directly in this function's own body that
    /// don't capture anything — ready `Val::Fn` values, indexed
    /// `0..consts.len()` by `Instr::MakeClosure`. Never referenced from
    /// anywhere but this function's own bytecode, so they live here
    /// instead of a runtime-shared pool.
    consts: Rc<[Val]>,
    /// Nested `fn`s defined directly in this function's own body that DO
    /// capture something — indexed `consts.len()..` by `Instr::MakeClosure`.
    templates: Rc<[Rc<FnTemplate>]>,
}

/// A `fn` compiled once as a reusable template — no ancestor frame
/// attached yet. Instantiated into a real [`CompiledFn`] by
/// `Instr::MakeClosure` each time its defining statement executes, so a
/// function called multiple times produces independent closures over that
/// call's own captured locals. Functions that don't capture anything from
/// an enclosing scope skip this entirely and compile straight to a
/// [`CompiledFn`] constant.
#[derive(Debug)]
pub struct FnTemplate {
    pub arity: u32,
    pub frame_size: u32,
    pub code: Code,
    own_names: SmallVec<[StringId; 8]>,
    fallback: Rc<[Option<u32>]>,
    owns_frame: bool,
    /// Whether this template actually reads something from its defining
    /// function's own reified frame (as opposed to only reading further
    /// out, or nothing at all) — decides whether `Instr::MakeClosure`
    /// attaches that frame directly or skips straight past it to whatever
    /// the defining function's own `outer` already is.
    needs_own_frame: bool,
    /// This function's own nested `fn`s, carried through so the
    /// instantiated [`CompiledFn`] has them too (see `CompiledFn::consts`/
    /// `CompiledFn::templates`) — shared, not copied, across every
    /// instantiation of this template.
    consts: Rc<[Val]>,
    templates: Rc<[Rc<FnTemplate>]>,
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
    pub fn into_function(self) -> Val {
        Val::Fn(FnVal::new_dyn(self))
    }

    /// Resolve slot `slot` against this function's own enclosing scope —
    /// used when a stack slot or captured slot is read before assignment.
    fn resolve_outer(&self, slot: SlotId, rt: &mut Runtime) -> Result<Val, RuntimeError> {
        let val = match self.fallback[slot.0 as usize] {
            Some(offset) => match &self.outer {
                Some(o) => o.get_flat(offset, rt)?,
                None => Val::Uninit,
            },
            None => {
                let name = self.own_names[slot.0 as usize];
                match &self.outer_named {
                    Some(f) => f.get_var(name, rt),
                    None => rt.get_var(name),
                }
            }
        };

        val.init_or_else(|| {
            RuntimeError::UnboundIdentifier(rt.ctx.get_string(self.own_names[slot.0 as usize]))
        })
    }

    fn get_slot(&self, base: usize, slot: SlotId, rt: &mut Runtime) -> Result<Val, RuntimeError> {
        let val = rt.stack.get(base + slot.0 as usize).clone();
        if !matches!(val, Val::Uninit) {
            return Ok(val);
        }

        core::hint::cold_path();
        self.resolve_outer(slot, rt)
    }

    fn get_cap(
        &self,
        own: &CompiledFrame,
        slot: SlotId,
        rt: &mut Runtime,
    ) -> Result<Val, RuntimeError> {
        own.get(slot, rt)
    }

    /// Walk a flat offset from this function's own nearest used ancestor —
    /// no per-instruction list lookup, just a chain of frames that
    /// actually own something.
    fn get_parent(&self, flat: SlotId, rt: &mut Runtime) -> Result<Val, RuntimeError> {
        match &self.outer {
            Some(o) => o.get_flat(flat.0, rt),
            None => Err(RuntimeError::Other(
                "vm: load parent from function with no parent scope".into(),
            )),
        }
    }
}

impl DynFn for CompiledFn {
    #[inline]
    fn min_args(&self) -> usize {
        self.arity()
    }

    #[inline]
    fn max_args(&self) -> Option<usize> {
        Some(self.arity())
    }

    fn dyn_call(self: Rc<Self>, rt: &mut dyn Host, args: usize) -> usize {
        let arity = self.arity();

        // args < arity may still return a partial value without touching
        // any CompiledFn/Runtime machinery beyond the stack, so only
        // downcast once we know we're actually running compiled code.
        if args < arity {
            if args == 0 {
                rt.stack_push(Val::Fn(FnVal::from_rc(self.clone())));
                return 0;
            }
            let partial = FnVal::partial_dyn(self.clone(), rt, args);
            rt.stack_push(Val::Fn(partial));
            return args;
        }

        let rt = rt
            .as_any_mut()
            .downcast_mut::<Runtime>()
            .expect("CompiledFn only runs under raft_runtime::Runtime");

        match run(rt, &self) {
            Ok(v) => v,
            Err(e) => {
                rt.set_error(e);
                rt.stack.push(Val::nil());
            }
        };

        arity
    }

    fn debug_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            writeln!(f, "<fn {}>{{", self.arity())?;

            for r in self.code.disassemble() {
                match r {
                    Ok((at, instr)) => writeln!(f, "  {at:4}: {instr:?}")?,
                    Err(e) => return writeln!(f, "  <decode error: {e}>"),
                }
            }

            write!(f, "}}")
        } else {
            write!(f, "<fn {}>", self.arity())
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
pub enum CompileParent {
    /// Nested inside another compiled function — the flat, nearest-first
    /// list of ancestor schemas visible from here (built by the caller:
    /// the immediately enclosing function's own schema prepended if it
    /// owns a frame, otherwise its own list passed straight through), plus
    /// whether there's a walked boundary somewhere beyond that list.
    Nested {
        schemas: Rc<[Rc<Schema>]>,
        walked_tail: bool,
    },
    /// Compiled directly under AST-walked code (module/REPL root, or an
    /// `AstFn`'s frame) — the live frame to attach to the *returned*
    /// `CompiledFn` directly, since (unlike a nested closure, instantiated
    /// fresh by `Instr::MakeClosure`) it's never re-materialized later.
    /// No compile-time schema list at all — anything not found locally
    /// resolves by name at runtime instead.
    Walked(Rc<Frame>),
}

pub fn compile_fn(
    rt: &mut Runtime,
    params: Rc<[Pat]>,
    body: &[Stmt],
    parent: CompileParent,
    force_captured: &[Rc<str>],
) -> Result<(CompiledFn, Rc<Schema>), CompileError> {
    let arity = params.len() as u32;
    let mut slots = SlotTable::with_params(&params);
    slots.add_stmts(body);

    // which of this function's own slots some nested `fn` reads — those
    // need a per-call reified Frame instead of a stack slot. `force_captured`
    // additionally pins specific names to the reified frame regardless of
    // whether anything nested reads them — used by modules to keep
    // exported bindings alive past the compiled body's own `Return`, which
    // otherwise truncates the stack region ordinary locals live in.
    let mut captured = slots.mark_captured(body);
    for name in force_captured {
        if let Some(slot) = slots.get(name) {
            captured[slot.0 as usize] = true;
        }
    }
    let owns_frame = captured.iter().any(|&c| c);

    if owns_frame && body_has_destructuring(&params, body) {
        // a captured local bound through list/record destructuring would
        // need `Instr::Bind` to write a frame slot instead of a stack
        // slot, which isn't supported yet — fall back to the AST walker
        // for the whole function rather than risk it landing on the stack
        let span = params
            .first()
            .map(|p| p.span())
            .or_else(|| body.first().map(|s| s.span()))
            .unwrap_or(Span::point(0));
        return Err(CompileError::new(
            span,
            "captured locals bound through destructuring aren't supported yet",
        ));
    }

    // this function's own name schema — used both to compile nested `fn`
    // statements (so a grandchild resolves through *this* function's own
    // scope, not straight past it) and as this function's own by-name
    // fallback source
    let names = slots.names(rt);
    let schema = Rc::new(Schema {
        names: names.clone(),
        owns_frame,
    });

    let (ancestors, walked_tail, outer_named): (Rc<[Rc<Schema>]>, bool, Option<Rc<Frame>>) =
        match &parent {
            CompileParent::Nested {
                schemas,
                walked_tail,
            } => (schemas.clone(), *walked_tail, None),
            CompileParent::Walked(f) => (Rc::from([]), true, Some(f.clone())),
        };

    // precompute, for each of this function's own slots, where a
    // read-before-assignment resolves — `Instr::LoadSlot`/`Instr::LoadCap`
    // consult this at runtime instead of re-walking anything
    let fallback: Rc<[Option<u32>]> = names
        .iter()
        .map(|&n| match resolve_outer_name(&ancestors, n) {
            OuterResolution::Flat(offset) => Some(offset),
            OuterResolution::ByName => None,
        })
        .collect::<Vec<_>>()
        .into();

    let mut c = Compiler {
        rt,
        slots,
        code: Vec::new(),
        loops: Vec::new(),
        ancestors,
        walked_tail,
        own_schema: schema.clone(),
        captured,
        nested_consts: Vec::new(),
        nested_templates: Vec::new(),
        template_refs: Vec::new(),
    };

    // prologue: unpack destructuring parameters out of their argument
    // slots into the named slots they bind (plain-ident parameters simply
    // stay where their argument landed), and copy captured plain-ident
    // parameters into the own-frame slot their reads/writes will target
    for (i, p) in params.iter().enumerate() {
        let arg_slot = SlotId(arity - 1 - i as u32);
        match p.kind() {
            PatKind::Ident(id) if id.name() == "_" => {}
            PatKind::Ident(_) => {
                if c.captured[arg_slot.0 as usize] {
                    c.emit(Instr::LoadSlot(arg_slot));
                    c.emit(Instr::StoreCap(arg_slot));
                }
            }
            _ => {
                c.emit(Instr::LoadSlot(arg_slot));
                let pattern = compile_pat(p, &c.slots, &mut c.rt.ctx);
                let pattern = c.rt.ctx.pattern(pattern);
                c.emit(Instr::Bind(pattern));
            }
        }
    }

    // the body is compiled in tail position: exactly one value — the
    // function's result — is on the stack when `Return` is reached
    c.compile_block(body, true)?;
    c.emit(Instr::Return);

    let frame_size = c.slots.next.0 - arity;

    // `nested_templates` occupy the flat range starting right after
    // `nested_consts` — final only now that no more consts can be added —
    // so every `MakeClosure` site recorded against a template gets its
    // local (templates-only) index shifted up by the final consts count
    let consts_len = c.nested_consts.len() as u32;
    for &site in &c.template_refs {
        if let Instr::MakeClosure(FnId(idx)) = &mut c.code[site] {
            *idx += consts_len;
        } else {
            unreachable!("template_refs must only record Instr::MakeClosure sites");
        }
    }

    Ok((
        CompiledFn {
            arity,
            frame_size,
            code: encode(&c.code),
            own_names: names,
            fallback,
            owns_frame,
            // a plain compile_fn result is only ever used directly (never
            // re-instantiated): a top-level walked/module fn, or a nested
            // fn that doesn't capture anything. Either way nothing reads
            // through `outer`.
            outer: None,
            outer_named,
            consts: c.nested_consts.into(),
            templates: c.nested_templates.into(),
        },
        schema,
    ))
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

/// Whether `params`/`body` bind any name through a destructuring pattern
/// (list/record pattern, or a non-plain-ident `for` target) rather than a
/// plain identifier. Doesn't recurse into nested `fn` bodies — their own
/// destructuring is their own concern. Captured *destructured* names aren't
/// supported yet (`Instr::Bind` always writes to a stack slot); a function
/// that both owns a captured-frame and destructures anywhere falls back to
/// the AST walker rather than risk a captured local silently living on the
/// stack.
fn body_has_destructuring(params: &[Pat], body: &[Stmt]) -> bool {
    fn is_destructuring(p: &Pat) -> bool {
        !matches!(p.kind(), PatKind::Ident(_))
    }
    fn stmts_have(stmts: &[Stmt]) -> bool {
        stmts.iter().any(|s| match s.kind() {
            StmtKind::AssignPat { target, .. } => is_destructuring(target),
            StmtKind::For {
                target,
                body,
                else_branch,
                ..
            } => {
                is_destructuring(target)
                    || stmts_have(body)
                    || else_branch.as_deref().is_some_and(stmts_have)
            }
            StmtKind::If {
                then_branch,
                else_branch,
                ..
            } => stmts_have(then_branch) || else_branch.as_deref().is_some_and(stmts_have),
            StmtKind::While {
                body, else_branch, ..
            } => stmts_have(body) || else_branch.as_deref().is_some_and(stmts_have),
            _ => false,
        })
    }
    params.iter().any(is_destructuring) || stmts_have(body)
}

/// Whether `name_id` resolves anywhere in `schema`'s compile-time-known
/// ancestor chain (`schema` itself counts as depth 0). Used to decide
/// between the fast [`Instr::LoadParent`] (a flat offset, however many
/// materialized ancestors it spans) and the by-name [`Instr::LoadParentByName`].
enum OuterResolution {
    /// Found at a flat offset spanning every materialized ancestor nearer
    /// than the one that owns it — matches `Instr::MakeClosure`'s runtime
    /// chain, which only ever links frames that actually own something, so
    /// walking it and subtracting slot counts lands in the same place.
    Flat(u32),
    /// Not resolvable to a fixed slot at compile time — not in the
    /// ancestor list, either because it's genuinely free or because it
    /// lives in a walked ancestor beyond the compiled list (a `Schema`
    /// never represents one, so there's no way to tell which at compile
    /// time). Resolve by name at runtime instead — `Frame::get_var`'s
    /// recursive chain-walk finds it either way, or the true global scope.
    ByName,
}

/// Resolve `name_id` against the flat, nearest-first list of ancestor
/// schemas visible from here, computing the cumulative slot offset across
/// however many of them sit between here and the one that owns it (every
/// entry in `ancestors` owns a frame by construction, so all of them
/// contribute).
fn resolve_outer_name(ancestors: &[Rc<Schema>], name_id: StringId) -> OuterResolution {
    let mut offset = 0u32;
    for schema in ancestors {
        if let Some(slot) = schema.name_slot(name_id) {
            return OuterResolution::Flat(offset + slot.0);
        }
        offset += schema.names.len() as u32;
    }
    OuterResolution::ByName
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
        Ok(Val::Number(Number::Integer(n))) => Some(if negated { n.wrapping_neg() } else { n }),
        _ => None,
    }
}

struct Compiler<'a> {
    /// Shared pools of the owning runtime; instruction operands index here.
    rt: &'a mut Runtime,
    /// This function's name→frame-slot resolution.
    slots: SlotTable,
    code: Vec<Instr>,
    loops: Vec<LoopCtx>,
    /// Flat, nearest-first list of ancestor schemas visible from here —
    /// used to resolve this function's *own* outer names
    /// (`emit_load_name`). Empty when nested directly under AST-walked
    /// code (module/REPL root, an `AstFn`'s frame): anything not locally
    /// bound resolves by name at runtime instead.
    ancestors: Rc<[Rc<Schema>]>,
    /// Whether there's a walked ancestor somewhere beyond `ancestors` —
    /// decides `Instr::LoadParentByName` vs `Instr::LoadGlobal` when a free
    /// name isn't found in `ancestors`.
    walked_tail: bool,
    /// This function's own schema — used to build the ancestor list when
    /// compiling a *nested* `fn` statement, so a grandchild resolves
    /// through this function's own captured slots (previously nested fns
    /// skipped straight to `parent`, missing this function's own scope
    /// entirely).
    own_schema: Rc<Schema>,
    /// Which of this function's own slots (by [`SlotId`]) are captured by
    /// some nested `fn` — those live in a per-call [`CompiledFrame`]
    /// instead of a stack slot.
    captured: Vec<bool>,
    /// Nested `fn`s defined directly in this function's body that don't
    /// capture anything — becomes this `CompiledFn`'s own `consts`.
    /// `Instr::MakeClosure` operands referencing these are final as soon
    /// as they're emitted (they occupy the flat range `0..consts.len()`).
    nested_consts: Vec<Val>,
    /// Nested `fn`s that DO capture something — becomes this `CompiledFn`'s
    /// own `templates`. These occupy the flat range starting at
    /// `nested_consts.len()`, which isn't final until compilation finishes
    /// (more consts can still be added afterward), so `Instr::MakeClosure`
    /// operands emitted for these start as a *local* templates-only index
    /// and get `nested_consts.len()` added on once it's known — see
    /// `template_refs`.
    nested_templates: Vec<Rc<FnTemplate>>,
    /// `c.code` indices of `Instr::MakeClosure` sites referencing
    /// `nested_templates` by local (unshifted) index, patched to the final
    /// flat index once `nested_consts` stops growing.
    template_refs: Vec<usize>,
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
        if self.captured[slot.0 as usize] {
            // IncSlot/DecSlot only ever touch the stack; a captured slot
            // lives in the own-frame instead, so fall through to the
            // general load/add/store path
            return false;
        }

        self.emit(if op.kind() == BinOpKind::Add {
            Instr::IncSlot(slot)
        } else {
            Instr::DecSlot(slot)
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
                    .expect("collect_slots missed an assignment target");
                self.emit(if self.captured[slot.0 as usize] {
                    Instr::StoreCap(slot)
                } else {
                    Instr::StoreSlot(slot)
                });
            }
            _ => {
                let p = compile_pat(target, &self.slots, &mut self.rt.ctx);
                let i = self.rt.ctx.pattern(p);
                self.emit(Instr::Bind(i));
            }
        }
    }

    /// Emit the correct load for a variable name: a definitely-initialized
    /// slot, a maybe-unset slot with global fallback, or a plain global.
    /// Emit a constant value: small integers and the singleton atoms get
    /// immediate opcodes, everything else goes through the const pool.
    fn emit_const_value(&mut self, v: Val) {
        match v {
            Val::Number(Number::Integer(n)) if int_fits_immediate(n) => {
                self.emit(Instr::Int(n));
            }
            Val::Atom(Atom::Nil) => {
                self.emit(Instr::Nil);
            }
            Val::Atom(Atom::True) => {
                self.emit(Instr::True);
            }
            Val::Atom(Atom::False) => {
                self.emit(Instr::False);
            }
            v => {
                let i = self.rt.ctx.const_(v);
                self.emit(Instr::Const(i));
            }
        }
    }

    fn emit_load_name(&mut self, name: Rc<str>) {
        match self.slots.get(&name) {
            Some(index) => {
                self.emit(if self.captured[index.0 as usize] {
                    Instr::LoadCap(index)
                } else {
                    Instr::LoadSlot(index)
                });
            }
            None => {
                let name_id = self.rt.ctx.string(name);

                match resolve_outer_name(&self.ancestors, name_id) {
                    OuterResolution::Flat(offset) => {
                        self.emit(Instr::LoadParent(SlotId(offset)));
                    }
                    OuterResolution::ByName => {
                        self.emit(Instr::LoadParentByName(name_id));
                    }
                }
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
                // Use expression value only in tail position.
                // Otherwise compile only side effects.
                self.compile_expr_callfn(e, tail)?;
            }
            StmtKind::AssignPat { target, value } => {
                if !self.try_inc_dec(target, value) {
                    self.compile_expr(value, true)?;
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
                self.compile_expr(target, true)?;
                self.compile_expr(value, true)?;
                let i = self.rt.ctx.string(field.rc_name());
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
                self.compile_expr(target, true)?;
                self.compile_expr(index, true)?;
                self.compile_expr(value, true)?;
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
                self.compile_expr(cond, true)?;
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
                self.compile_expr(cond, true)?;
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
                self.compile_expr(iterable, true)?;
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
                    Some(e) => self.compile_expr(e, true)?,
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
                // does the nested fn (or something nested *inside* it,
                // transitively — `fn_outer_names` already recurses)
                // actually read one of *this* function's own names? Owning
                // a frame at all doesn't mean every nested fn needs it —
                // only ones that reference it should pay for the hop, so
                // this is decided per-child, not from `self.own_schema.owns_frame`
                // (which only says *something* nested in this function
                // needs it, not that *this particular* one does).
                let free = crate::fn_outer_names(params, body);
                let needs_own_schema = free.iter().any(|n| self.slots.get(n).is_some());
                // invariant: this function's own prepass (`mark_captured`)
                // scans every direct nested fn's outer names the same way
                // — if this specific child needs the schema, that pass
                // must already have flagged the function as owning a frame
                debug_assert!(
                    !needs_own_schema || self.own_schema.owns_frame,
                    "child needs own schema but mark_captured didn't mark this function as owning a frame"
                );

                let child_ancestors: Rc<[Rc<Schema>]> = if needs_own_schema {
                    let mut v = Vec::with_capacity(self.ancestors.len() + 1);
                    v.push(self.own_schema.clone());
                    v.extend(self.ancestors.iter().cloned());
                    Rc::from(v)
                } else {
                    self.ancestors.clone()
                };

                let (compiled, _schema) = compile_fn(
                    self.rt,
                    params.clone(),
                    body,
                    CompileParent::Nested {
                        schemas: child_ancestors,
                        walked_tail: self.walked_tail,
                    },
                    &[],
                )?;

                // does the nested fn read anything not bound within
                // itself? If not, it's fully self-contained and can
                // compile to one shared constant, exactly as before.
                // Otherwise (this also conservatively covers reading a
                // genuine global — a `Schema` list can't tell "nothing
                // out there" from "something in a walked ancestor" apart
                // at compile time, see `resolve_outer_name`) each
                // execution of this statement must produce its own
                // closure over the live enclosing scope.
                let captures_outer = !free.is_empty();

                if captures_outer {
                    let template = FnTemplate {
                        arity: compiled.arity,
                        frame_size: compiled.frame_size,
                        code: compiled.code,
                        own_names: compiled.own_names,
                        fallback: compiled.fallback,
                        owns_frame: compiled.owns_frame,
                        needs_own_frame: needs_own_schema,
                        consts: compiled.consts,
                        templates: compiled.templates,
                    };
                    // local (templates-only) index for now — shifted to
                    // the final flat index once `nested_consts` stops
                    // growing (see the patch pass at the end of compile_fn)
                    let local = self.nested_templates.len() as u32;
                    self.nested_templates.push(Rc::new(template));
                    let site = self.emit(Instr::MakeClosure(FnId(local)));
                    self.template_refs.push(site);
                } else {
                    // consts occupy the flat range 0..consts.len() and
                    // never move, so this index is final immediately
                    let idx = self.nested_consts.len() as u32;
                    self.nested_consts.push(compiled.into_function());
                    self.emit(Instr::MakeClosure(FnId(idx)));
                }

                let slot = self
                    .slots
                    .get(name.name())
                    .expect("collect_slots missed a fn name");
                self.emit(if self.captured[slot.0 as usize] {
                    Instr::StoreCap(slot)
                } else {
                    Instr::StoreSlot(slot)
                });
                if tail {
                    self.emit(Instr::Nil); // definitions yield nil
                }
            }
        }
        Ok(())
    }

    // if `used` is false, the expression's value is discarded
    // thus only expressions that produce side effects need to be compiled.
    fn compile_expr(&mut self, expr: &Expr, used: bool) -> Result<(), CompileError> {
        match expr.kind() {
            ExprKind::Literal(lit) => {
                if used {
                    let v = literal_value(lit)
                        .map_err(|e| CompileError::new(expr.span(), e.to_string()))?;
                    self.emit_const_value(v);
                }
            }
            ExprKind::Ident(id) => {
                if used {
                    self.emit_load_name(id.rc_name());
                }
            }
            ExprKind::Atom(a) => {
                if used {
                    self.emit_const_value(Val::new_atom(a.rc_name()));
                }
            }
            ExprKind::List(elements) => {
                for e in elements.iter() {
                    self.compile_expr(e, used)?;
                }
                if used {
                    self.emit(Instr::MakeList(elements.len() as u32));
                }
            }
            ExprKind::Record(fields) => {
                for f in fields.iter() {
                    let key = f.key().rc_name();
                    let ki = self.rt.ctx.const_(Val::String(key.clone()));
                    self.emit(Instr::Const(ki));
                    match f.value() {
                        Some(v) => self.compile_expr(v, used)?,
                        // shorthand field reads the same-named variable
                        None => {
                            if used {
                                self.emit_load_name(key);
                            }
                        }
                    }
                }
                if used {
                    self.emit(Instr::MakeRecord(fields.len() as u32));
                }
            }
            ExprKind::Unary(op, operand) => {
                // fold `-<number literal>` at compile time, mirroring what
                // evaluation would do, so negative small integers reach
                // the immediate opcodes
                if let (UnOpKind::Neg, ExprKind::Literal(lit)) = (op.kind(), operand.kind()) {
                    let v = literal_value(lit)
                        .map_err(|e| CompileError::new(expr.span(), e.to_string()))?;
                    if let Val::Number(n) = v {
                        let negated = n
                            .neg()
                            .map_err(|e| CompileError::new(expr.span(), e.to_string()))?;
                        self.emit_const_value(Val::Number(negated));
                        return Ok(());
                    }
                }
                self.compile_expr(operand, used)?;
                if used {
                    self.emit(Instr::Unary(op.kind()));
                }
            }
            ExprKind::Binary(lhs, op, rhs) => {
                // fuse `<expr> op n` with an integer-literal right operand
                // into a single instruction, when n is in the op's table
                let table: Option<&[i64]> = match op.kind() {
                    BinOpKind::Add | BinOpKind::Sub => Some(&SMALL_INTS),
                    BinOpKind::BitAnd | BinOpKind::BitOr | BinOpKind::BitXor => Some(&TINY_MASKS),
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
                        self.compile_expr(lhs, used)?;
                        self.emit(Instr::BinaryInt(op.kind(), n as i16));
                        return Ok(());
                    }
                }
                self.compile_expr(lhs, used)?;
                self.compile_expr(rhs, used)?;

                if used {
                    self.emit(Instr::Binary(op.kind()));
                }
            }
            ExprKind::Apply(func, args) => {
                // function calls are always used.
                self.compile_expr(func, true)?;
                for a in args.iter() {
                    self.compile_expr(a, true)?;
                }
                self.emit(Instr::Call(args.len() as u32));
            }
            ExprKind::Field(obj, field) => {
                self.compile_expr(obj, used)?;
                let i = self.rt.ctx.string(field.rc_name());
                if used {
                    self.emit(Instr::GetField(i));
                }
            }
            ExprKind::Index(obj, index) => {
                self.compile_expr(obj, used)?;
                self.compile_expr(index, used)?;
                if used {
                    self.emit(Instr::GetIndex);
                }
            }
            // parentheses put the inner expression in call position
            ExprKind::Parenthesized(inner) => self.compile_expr_callfn(inner, used)?,
        }
        Ok(())
    }

    /// Compile an expression in call position (statement expressions and
    /// parenthesized expressions): a bare identifier holding a zero-argument
    /// function gets called instead of yielding the function value.
    fn compile_expr_callfn(&mut self, expr: &Expr, used: bool) -> Result<(), CompileError> {
        match expr.kind() {
            ExprKind::Ident(id) => {
                self.emit_load_name(id.rc_name());
                self.emit(Instr::Call(0));
                if !used {
                    self.emit(Instr::Pop);
                }
            }
            ExprKind::Parenthesized(inner) => self.compile_expr_callfn(inner, used)?,
            _ => self.compile_expr(expr, used)?,
        }
        Ok(())
    }
}

/// Execute a compiled function's code. Parameters are expected to already be
/// bound in the current (local) scope — [`CompiledFn::into_function`] does
/// that — so `run` itself is just the instruction loop.
///
/// The frame executes on the runtime's shared operand stack
/// (`rt.stack`): it treats the stack height at entry as its floor and
/// restores it on the way out, whether it returns a value or an error, so
/// nested and recursive frames (including mixed-mode reentry through host
/// or AST functions) compose without per-call stack allocations.
/// Build a fresh, all-`Uninit` reified frame for `f`'s own captured
/// locals, sharing `f`'s own fallback/outer/outer_named metadata directly
/// (a captured slot's read-before-assignment behavior is identical to a
/// stack slot's).
fn make_own_frame(f: &CompiledFn) -> Rc<CompiledFrame> {
    Rc::new(CompiledFrame {
        names: f.own_names.clone(),
        slots: RefCell::new(smallvec::smallvec![Val::Uninit; f.own_names.len()]),
        fallback: f.fallback.clone(),
        outer: f.outer.clone(),
        outer_named: f.outer_named.clone(),
    })
}

pub fn run(rt: &mut Runtime, f: &CompiledFn) -> Result<(), RuntimeError> {
    // the caller's arguments are already on the stack and become the
    // frame's first slots; reserve the rest for body-introduced locals
    debug_assert!(rt.stack.len() >= f.arity());
    let base = rt.stack.len() - f.arity();

    rt.stack.extend_uninit(f.frame_size as usize);

    // a fresh reified frame per call, for locals some nested `fn` captures
    // — none of this function's calls share captured state with another
    let own = f.owns_frame.then(|| make_own_frame(f));

    let result = run_frame(rt, &f.code.bytes, base, f, own.as_ref());
    debug_assert!(rt.stack.len() >= base);
    result
}

/// Run a compiled module body (a zero-arg [`CompiledFn`] whose `export`
/// field names were force-captured by [`compile_fn`]) and hand back its
/// reified frame, if it has one, so the caller can read exported values out
/// of it. The body's own tail value (ordinarily `Nil`, since a module's
/// last statement is rarely meaningful on its own) is discarded — `export`
/// is structural, not part of the statement list `f` was compiled from.
pub fn run_module(rt: &mut Runtime, f: &CompiledFn) -> Result<Option<Rc<CompiledFrame>>, RuntimeError> {
    debug_assert_eq!(f.arity(), 0);
    let base = rt.stack.len();
    rt.stack.extend_uninit(f.frame_size as usize);

    let own = f.owns_frame.then(|| make_own_frame(f));

    run_frame(rt, &f.code.bytes, base, f, own.as_ref())?;
    rt.stack.pop();
    debug_assert_eq!(rt.stack.len(), base);
    Ok(own)
}

/// Execute a compiled function's code. Parameters are expected to already be
/// bound in the current (local) scope — [`CompiledFn::into_function`] does
/// that — so `run` itself is just the instruction loop.
///
/// The frame executes on the runtime's shared operand stack
/// (`rt.stack`): it treats the stack height at entry as its floor and
/// restores it on the way out, whether it returns a value or an error, so
/// nested and recursive frames (including mixed-mode reentry through host
/// or AST functions) compose without per-call stack allocations.
pub fn run_recursive(rt: &mut Runtime, code: &[u8], f: &CompiledFn) -> Result<(), RuntimeError> {
    // the caller's arguments are already on the stack and become the
    // frame's first slots; reserve the rest for body-introduced locals
    debug_assert!(rt.stack.len() >= f.arity());
    let base = rt.stack.len() - f.arity();
    rt.stack.extend_uninit(f.frame_size as usize);

    let own = f.owns_frame.then(|| make_own_frame(f));

    let result = run_frame(rt, code, base, f, own.as_ref());
    debug_assert!(rt.stack.len() >= base);
    result
}

fn run_frame(
    rt: &mut Runtime,
    code: &[u8],
    base: usize,
    f: &CompiledFn,
    own: Option<&Rc<CompiledFrame>>,
) -> Result<(), RuntimeError> {
    let mut iters = SmallVec::<[_; 4]>::new();
    let mut pc: usize = 0;

    loop {
        // `pc` is a byte offset; each opcode decodes its own operands
        let Some(&op) = code.get(pc) else {
            return Err(RuntimeError::Other(
                "vm: execution ran past the end of code".into(),
            ));
        };
        pc += 1;

        match op {
            opcode::CONST..opcode::CONST_END => {
                let i = decode_wop(op - opcode::CONST, code, &mut pc)?;
                let v = rt.ctx.get_const(ConstId(i));
                rt.stack.push(v);
            }
            opcode::NIL => rt.stack.push(Val::nil()),
            opcode::TRUE => rt.stack.push(Val::true_()),
            opcode::FALSE => rt.stack.push(Val::false_()),
            opcode::INT..opcode::INT_END => {
                let n = SMALL_INTS[(op - opcode::INT) as usize];
                rt.stack.push(Val::Number(Number::Integer(n)));
            }
            opcode::INT_8 => {
                let n = read_u8(code, &mut pc)? as i8 as i64;
                rt.stack.push(Val::Number(Number::Integer(n)));
            }
            opcode::INT_16 => {
                let n = read_u16(code, &mut pc)? as i16 as i64;
                rt.stack.push(Val::Number(Number::Integer(n)));
            }
            opcode::UINT_8 => {
                let n = read_u8(code, &mut pc)? as i64;
                rt.stack.push(Val::Number(Number::Integer(n)));
            }
            opcode::UINT_16 => {
                let n = read_u16(code, &mut pc)? as i64;
                rt.stack.push(Val::Number(Number::Integer(n)));
            }
            opcode::POP => {
                rt.stack.pop();
            }
            opcode::LOAD_SLOT..opcode::LOAD_SLOT_END => {
                let slot = decode_operand(op - opcode::LOAD_SLOT, code, &mut pc)?;
                let val = f.get_slot(base, SlotId(slot), rt)?;
                rt.stack.push(val);
            }
            opcode::LOAD_PARENT..opcode::LOAD_PARENT_END => {
                let slot = decode_operand(op - opcode::LOAD_PARENT, code, &mut pc)?;
                let v = f.get_parent(SlotId(slot), rt)?;
                rt.stack.push(v);
            }
            opcode::LOAD_PARENT_BY_NAME..opcode::LOAD_PARENT_BY_NAME_END => {
                let i = decode_wop(op - opcode::LOAD_PARENT_BY_NAME, code, &mut pc)?;
                let name = StringId(i);
                let val = match &f.outer_named {
                    Some(named) => named.get_var(name, rt),
                    None => rt.get_var(name),
                };
                let v = val.init_or_else(|| {
                    core::hint::cold_path();
                    RuntimeError::UnboundIdentifier(rt.ctx.get_string(name))
                })?;
                rt.stack.push(v);
            }
            opcode::LOAD_GLOBAL..opcode::LOAD_GLOBAL_END => {
                let i = decode_wop(op - opcode::LOAD_GLOBAL, code, &mut pc)?;
                let name = StringId(i);
                let v = rt.get_var(name).init_or_else(|| {
                    core::hint::cold_path();
                    RuntimeError::UnboundIdentifier(rt.ctx.get_string(name))
                })?;
                rt.stack.push(v);
            }
            opcode::STORE_SLOT..opcode::STORE_SLOT_END => {
                let slot = decode_operand(op - opcode::STORE_SLOT, code, &mut pc)?;
                let v = rt.stack.pop();
                rt.stack.set(base + slot as usize, v);
            }
            opcode::LOAD_CAP..opcode::LOAD_CAP_END => {
                let slot = decode_operand(op - opcode::LOAD_CAP, code, &mut pc)?;
                let frame = own.expect("vm: LOAD_CAP in a function with no own frame");
                let val = f.get_cap(frame, SlotId(slot), rt)?;
                rt.stack.push(val);
            }
            opcode::STORE_CAP..opcode::STORE_CAP_END => {
                let slot = decode_operand(op - opcode::STORE_CAP, code, &mut pc)?;
                let v = rt.stack.pop();
                let frame = own.expect("vm: STORE_CAP in a function with no own frame");
                frame.set(SlotId(slot), v);
            }
            opcode::MAKE_CLOSURE..opcode::MAKE_CLOSURE_END => {
                let i = decode_wop(op - opcode::MAKE_CLOSURE, code, &mut pc)? as usize;
                // flattened: below f.consts.len() is a ready value (no
                // parent to attach, just clone and push); at or above it
                // indexes f.templates (offset back down) and needs a real
                // closure built over the live enclosing scope
                if i < f.consts.len() {
                    rt.stack.push(f.consts[i].clone());
                } else {
                    let template = &f.templates[i - f.consts.len()];
                    // attach this function's own captured-frame as the
                    // child's nearest used ancestor only if the child
                    // actually reads something from it (`needs_own_frame`,
                    // decided per-child at compile time — owning a frame
                    // at all doesn't mean every nested fn needs it);
                    // otherwise skip straight past it to whatever this
                    // function's own `outer` already is — resolved the
                    // same way when this function was itself instantiated,
                    // so by induction it's already the nearest ancestor
                    // any of *this* function's compiled descendants could
                    // need, and a chain of skipped levels collapses to a
                    // single hop at runtime instead of one dead layer per
                    // level
                    let outer = if template.needs_own_frame {
                        match own {
                            Some(o) => Some(o.clone()),
                            None => f.outer.clone(),
                        }
                    } else {
                        f.outer.clone()
                    };
                    let child = CompiledFn {
                        arity: template.arity,
                        frame_size: template.frame_size,
                        code: template.code.clone(),
                        own_names: template.own_names.clone(),
                        fallback: template.fallback.clone(),
                        owns_frame: template.owns_frame,
                        outer,
                        outer_named: f.outer_named.clone(),
                        consts: template.consts.clone(),
                        templates: template.templates.clone(),
                    };
                    rt.stack.push(child.into_function());
                }
            }
            opcode::BIND..opcode::BIND_END => {
                let i = decode_wop(op - opcode::BIND, code, &mut pc)?;
                let v = rt.stack.pop();
                let pattern = rt
                    .ctx
                    .pats
                    .get(i as usize)
                    .ok_or_else(|| {
                        core::hint::cold_path();
                        RuntimeError::Other("vm: pattern index out of range".into())
                    })?
                    .clone();
                pattern.bind(rt, base, &v)?;
            }
            opcode::MAKE_LIST..opcode::MAKE_LIST_END => {
                let n = decode_operand(op - opcode::MAKE_LIST, code, &mut pc)?;
                let elements = rt.stack.drain_top(n as usize);
                let list = Val::new_list(elements.collect());
                rt.stack.push(list);
            }
            opcode::MAKE_RECORD..opcode::MAKE_RECORD_END => {
                let n = decode_operand(op - opcode::MAKE_RECORD, code, &mut pc)?;
                let mut map = FixedHashMap::default();
                {
                    let mut fields = rt.stack.drain_top(n as usize * 2);
                    while let (Some(key), Some(val)) = (fields.next(), fields.next()) {
                        match key {
                            Val::String(key) => {
                                map.insert(key, val);
                            }
                            _ => {
                                core::hint::cold_path();
                                return Err(RuntimeError::TypeError(
                                    "vm: record key must be a string".into(),
                                ));
                            }
                        }
                    }
                }
                let record = Val::new_record(map);
                rt.stack.push(record);
            }
            opcode::UNARY..opcode::UNARY_END => {
                let k = byte_to_unop(op - opcode::UNARY)?;
                let a = rt.stack.pop();
                let v = eval_unary(k, &a)?;
                rt.stack.push(v);
            }
            opcode::BINARY..opcode::BINARY_END => {
                let k = byte_to_binop(op - opcode::BINARY)?;
                let b = rt.stack.pop();
                let a = rt.stack.pop();
                let v = eval_binary(k, &a, &b)?;
                rt.stack.push(v);
            }
            opcode::ADD_INT..opcode::ADD_INT_END => {
                let n = SMALL_INTS[(op - opcode::ADD_INT) as usize];
                let a = rt.stack.pop();
                let v = match a {
                    // fast path mirroring Number::add's wrapping semantics
                    Val::Number(Number::Integer(x)) => {
                        Val::Number(Number::Integer(x.wrapping_add(n)))
                    }
                    a => {
                        core::hint::cold_path();
                        eval_binary(BinOpKind::Add, &a, &Val::Number(Number::Integer(n)))?
                    }
                };
                rt.stack.push(v);
            }
            opcode::SUB_INT..opcode::SUB_INT_END => {
                let n = SMALL_INTS[(op - opcode::SUB_INT) as usize];
                let a = rt.stack.pop();
                let v = match a {
                    Val::Number(Number::Integer(x)) => {
                        Val::Number(Number::Integer(x.wrapping_sub(n)))
                    }
                    a => {
                        core::hint::cold_path();
                        eval_binary(BinOpKind::Sub, &a, &Val::Number(Number::Integer(n)))?
                    }
                };
                rt.stack.push(v);
            }
            opcode::AND_MASK..opcode::AND_MASK_END => {
                let n = TINY_MASKS[(op - opcode::AND_MASK) as usize];
                let a = rt.stack.pop();
                let v = match a {
                    Val::Number(Number::Integer(x)) => Val::Number(Number::Integer(x & n)),
                    a => {
                        core::hint::cold_path();
                        eval_binary(BinOpKind::BitAnd, &a, &Val::Number(Number::Integer(n)))?
                    }
                };
                rt.stack.push(v);
            }
            opcode::OR_MASK..opcode::OR_MASK_END => {
                let n = TINY_MASKS[(op - opcode::OR_MASK) as usize];
                let a = rt.stack.pop();
                let v = match a {
                    Val::Number(Number::Integer(x)) => Val::Number(Number::Integer(x | n)),
                    a => {
                        core::hint::cold_path();
                        eval_binary(BinOpKind::BitOr, &a, &Val::Number(Number::Integer(n)))?
                    }
                };
                rt.stack.push(v);
            }
            opcode::XOR_MASK..opcode::XOR_MASK_END => {
                let n = TINY_MASKS[(op - opcode::XOR_MASK) as usize];
                let a = rt.stack.pop();
                let v = match a {
                    Val::Number(Number::Integer(x)) => Val::Number(Number::Integer(x ^ n)),
                    a => {
                        core::hint::cold_path();
                        eval_binary(BinOpKind::BitXor, &a, &Val::Number(Number::Integer(n)))?
                    }
                };
                rt.stack.push(v);
            }
            opcode::INT_EQ..opcode::INT_EQ_END => {
                let n = TINY_INTS[(op - opcode::INT_EQ) as usize];
                let a = rt.stack.pop();
                let v = match a {
                    Val::Number(Number::Integer(x)) => Val::bool_(x == n),
                    a => {
                        core::hint::cold_path();
                        eval_binary(BinOpKind::Eq, &a, &Val::Number(Number::Integer(n)))?
                    }
                };
                rt.stack.push(v);
            }
            opcode::INT_NE..opcode::INT_NE_END => {
                let n = TINY_INTS[(op - opcode::INT_NE) as usize];
                let a = rt.stack.pop();
                let v = match a {
                    Val::Number(Number::Integer(x)) => Val::bool_(x != n),
                    a => eval_binary(BinOpKind::Ne, &a, &Val::Number(Number::Integer(n)))?,
                };
                rt.stack.push(v);
            }
            opcode::INT_LT..opcode::INT_LT_END => {
                let n = TINY_INTS[(op - opcode::INT_LT) as usize];
                let a = rt.stack.pop();
                let v = match a {
                    Val::Number(Number::Integer(x)) => Val::bool_(x < n),
                    a => {
                        core::hint::cold_path();
                        eval_binary(BinOpKind::Lt, &a, &Val::Number(Number::Integer(n)))?
                    }
                };
                rt.stack.push(v);
            }
            opcode::INT_GT..opcode::INT_GT_END => {
                let n = TINY_INTS[(op - opcode::INT_GT) as usize];
                let a = rt.stack.pop();
                let v = match a {
                    Val::Number(Number::Integer(x)) => Val::bool_(x > n),
                    a => {
                        core::hint::cold_path();
                        eval_binary(BinOpKind::Gt, &a, &Val::Number(Number::Integer(n)))?
                    }
                };
                rt.stack.push(v);
            }
            opcode::INT_LE..opcode::INT_LE_END => {
                let n = TINY_INTS[(op - opcode::INT_LE) as usize];
                let a = rt.stack.pop();
                let v = match a {
                    Val::Number(Number::Integer(x)) => Val::bool_(x <= n),
                    a => {
                        core::hint::cold_path();
                        eval_binary(BinOpKind::Le, &a, &Val::Number(Number::Integer(n)))?
                    }
                };
                rt.stack.push(v);
            }
            opcode::INT_GE..opcode::INT_GE_END => {
                let n = TINY_INTS[(op - opcode::INT_GE) as usize];
                let a = rt.stack.pop();
                let v = match a {
                    Val::Number(Number::Integer(x)) => Val::bool_(x >= n),
                    a => {
                        core::hint::cold_path();
                        eval_binary(BinOpKind::Ge, &a, &Val::Number(Number::Integer(n)))?
                    }
                };
                rt.stack.push(v);
            }
            opcode::INC_SLOT..opcode::DEC_SLOT_END => {
                let inc = op < opcode::INC_SLOT_END;
                let rel = if inc {
                    op - opcode::INC_SLOT
                } else {
                    op - opcode::DEC_SLOT
                };
                let slot = decode_operand(rel, code, &mut pc)?;
                let cur = f.get_slot(base, SlotId(slot), rt)?;
                let v = match cur {
                    Val::Number(Number::Integer(x)) => Val::Number(Number::Integer(if inc {
                        x.wrapping_add(1)
                    } else {
                        x.wrapping_sub(1)
                    })),
                    cur => {
                        core::hint::cold_path();
                        eval_binary(
                            if inc { BinOpKind::Add } else { BinOpKind::Sub },
                            &cur,
                            &Val::Number(Number::Integer(1)),
                        )?
                    }
                };
                rt.stack.set(base + slot as usize, v);
            }
            opcode::GET_FIELD..opcode::GET_FIELD_END => {
                let i = decode_wop(op - opcode::GET_FIELD, code, &mut pc)?;
                let key = rt.ctx.get_string(StringId(i));
                let obj = rt.stack.pop();
                let v = field_of(&obj, &key)?;
                rt.stack.push(v);
            }
            opcode::GET_INDEX => {
                let idx = rt.stack.pop();
                let obj = rt.stack.pop();
                let v = index_of(&obj, &idx)?;
                rt.stack.push(v);
            }
            opcode::SET_FIELD..opcode::SET_FIELD_END => {
                let i = decode_wop(op - opcode::SET_FIELD, code, &mut pc)?;
                let key = rt.ctx.get_string(StringId(i));
                let val = rt.stack.pop();
                let obj = rt.stack.pop();
                assign_field(obj, &key, val)?;
            }
            opcode::SET_INDEX => {
                let val = rt.stack.pop();
                let idx = rt.stack.pop();
                let obj = rt.stack.pop();
                assign_index(obj, idx, val)?;
            }
            opcode::CALL..opcode::CALL_END => {
                let n = decode_operand(op - opcode::CALL, code, &mut pc)?;
                if n > 0 {
                    rt.stack.reverse(n as usize + 1);
                }

                let fval = rt.stack.peek();
                match fval {
                    // compare data addresses only (not the `dyn DynFn` fat
                    // pointer): the callee's Rc<CompiledFn> and `f` may have
                    // been unsize-coerced to `dyn DynFn` at different call
                    // sites, which isn't guaranteed to produce the same
                    // vtable pointer, so comparing the wide pointers
                    // directly is unreliable
                    Val::Fn(fval)
                        if fval.as_ptr()
                            == (f as *const CompiledFn as *const ())
                            && f.arity == n =>
                    {
                        // Calls self: discard the callee we just peeked (we
                        // already know what it is) so the stack holds only
                        // the arguments run_recursive's floor expects.
                        rt.stack.pop();
                        run_recursive(rt, code, f)?;
                    }
                    _ => {
                        rt.call(n as usize)?;
                    }
                }
            }
            opcode::JUMP => {
                pc = read_u32(code, &mut pc)? as usize;
            }
            opcode::JUMP_IF_FALSE => {
                let t = read_u32(code, &mut pc)?;
                let c = rt.stack.pop();
                if is_falsey(&c) {
                    pc = t as usize;
                }
            }
            opcode::ITER_INIT => {
                let v = rt.stack.pop();
                iters.push(v.iter()?.into_iter());
            }
            opcode::ITER_NEXT => {
                let t = read_u32(code, &mut pc)?;
                let iter = iters.last_mut().ok_or_else(|| {
                    core::hint::cold_path();
                    RuntimeError::Other("vm: no active iterator".into())
                })?;
                match iter.next() {
                    Some(item) => rt.stack.push(item),
                    None => {
                        iters.pop();
                        pc = t as usize;
                    }
                }
            }
            opcode::ITER_POP => {
                iters.pop();
            }
            opcode::RETURN => {
                if rt.stack.len() != base + 1 {
                    let ret = rt.stack.pop();
                    rt.stack.truncate(base);
                    rt.stack.push(ret);
                }
                return Ok(());
            }
            _ => return Err(RuntimeError::Other("vm: unknown opcode".into())),
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

    fn run_mode(src: &str, compiled: bool) -> Result<(Runtime, Rc<Frame>), RuntimeError> {
        let stmts = ast_from_str(src);
        let mut rt = Runtime::new();
        let frame = Rc::new(Frame::new());
        rt.set_compile_fns(compiled);
        for statement in &stmts {
            rt.exec_stmt(statement, frame.clone())?;
        }
        Ok((rt, frame))
    }

    /// Run `src` through the AST walker and the bytecode VM and assert that
    /// every global variable ends up displaying identically.
    fn assert_modes_agree(src: &str) -> (Runtime, Rc<Frame>) {
        let (walked_rt, walked_frame) = run_mode(src, false).expect("AST walker failed");
        let (mut vmed_rt, vmed_frame) = run_mode(src, true).expect("VM failed");

        let walked_entries = walked_frame.own_entries();
        let vmed_entries = vmed_frame.own_entries();

        let mut walked_keys: Vec<_> = walked_entries
            .iter()
            .map(|&(k, _)| walked_rt.ctx.get_string(k))
            .collect();
        let mut vmed_keys: Vec<_> = vmed_entries
            .iter()
            .map(|&(k, _)| vmed_rt.ctx.get_string(k))
            .collect();
        walked_keys.sort();
        vmed_keys.sort();
        assert_eq!(walked_keys, vmed_keys, "modes bound different globals");

        for (name_id, walked_val) in walked_entries.iter() {
            let name = walked_rt.ctx.get_string(*name_id);
            let vmed_val = &vmed_frame.get_var(vmed_rt.ctx.string(name.clone()), &mut vmed_rt);
            assert_eq!(
                format!("{walked_val}"),
                format!("{vmed_val}"),
                "global `{name}` differs between modes"
            );
        }
        (vmed_rt, vmed_frame)
    }

    fn int_var(rt: &mut Runtime, frame: &Frame, name: &str) -> i64 {
        match frame.get_var(name, rt) {
            Val::Number(Number::Integer(i)) => i,
            other => panic!("expected integer in `{name}`, got {other:?}"),
        }
    }

    const MATH_MODULE: &str = "pi = 3
fn sq x:
    return x * x
fn dist2 a b:
    return (sq a) + (sq b)
fn area r:
    return pi * (sq r)
export { pi, sq, dist2, area }
";

    fn module_runtime(compiled: bool) -> Runtime {
        let mut rt = Runtime::new();
        rt.set_compile_fns(compiled);
        rt
    }

    #[test]
    fn modules_load_export_and_capture_their_environment() {
        for compiled in [false, true] {
            let mut rt = module_runtime(compiled);
            let frame = Rc::new(Frame::new());
            let module = rt.load_module("math", MATH_MODULE).unwrap();
            rt.set_var("math", module);

            // field access, module-value capture, helper-fn capture — the
            // functions run *after* the load, so they must see the module
            // environment, not the globals
            for st in &ast_from_str(
                "r1 = math.pi
r2 = math.sq 6
r3 = math.dist2 3 4
r4 = math.area 2
",
            ) {
                rt.exec_stmt(st, frame.clone()).unwrap();
            }
            assert_eq!(
                format!("{}", frame.get_var("r1", &mut rt)),
                "3",
                "mode {compiled}"
            );
            assert_eq!(
                format!("{}", frame.get_var("r2", &mut rt)),
                "36",
                "mode {compiled}"
            );
            assert_eq!(
                format!("{}", frame.get_var("r3", &mut rt)),
                "25",
                "mode {compiled}"
            );
            assert_eq!(
                format!("{}", frame.get_var("r4", &mut rt)),
                "12",
                "mode {compiled}"
            );

            // module bindings must not leak into the globals
            assert!(!frame.get_var("pi", &mut rt).is_init());
            assert!(!frame.get_var("sq", &mut rt).is_init());

            // record patterns destructure modules
            for st in &ast_from_str(
                "{ pi, sq } = math
r5 = sq pi
",
            ) {
                rt.exec_stmt(st, frame.clone()).unwrap();
            }
            assert_eq!(format!("{}", frame.get_var("r5", &mut rt)), "9");
        }
    }

    #[test]
    fn modules_are_cached_and_immutable() {
        let mut rt = module_runtime(true);
        let frame = Rc::new(Frame::new());
        let a = rt.load_module("math", MATH_MODULE).unwrap();
        let b = rt
            .load_module("math", "garbage that would not even parse")
            .unwrap();
        // second load must come from the cache: same object
        let (Val::Object(oa), Val::Object(ob)) = (&a, &b) else {
            panic!("modules are objects");
        };
        assert!(Rc::ptr_eq(oa, ob));

        // and the module object rejects mutation
        rt.set_var("math", a);
        let stmts = ast_from_str(
            "math.pi = 4
",
        );
        assert!(rt.exec_stmt(&stmts[0], frame.clone()).is_err());
    }

    #[test]
    fn module_export_rules() {
        let mut rt = module_runtime(true);
        let frame = Rc::new(Frame::new());

        // a module must terminate with an export
        assert!(
            rt.load_module(
                "bad1", "x = 1
"
            )
            .is_err()
        );
        // nothing may follow the export
        assert!(
            rt.load_module(
                "bad2",
                "export { x: 1 }
y = 2
"
            )
            .is_err()
        );
        // export is structural, not a statement: it cannot nest in blocks
        assert!(
            rt.load_module(
                "bad3",
                "if True:
    export { a: 1 }
"
            )
            .is_err()
        );
        // an export referencing an unbound name fails the load cleanly
        // (and the runtime stays usable afterwards)
        assert!(
            rt.load_module(
                "bad4",
                "export { missing }
"
            )
            .is_err()
        );
        let ok = rt
            .load_module(
                "ok",
                "x = 1
export { x }
",
            )
            .unwrap();
        rt.set_var("ok", ok);
        for st in &ast_from_str(
            "r = ok.x
",
        ) {
            rt.exec_stmt(st, frame.clone()).unwrap();
        }
        assert_eq!(
            format!("{}", frame.get_var(rt.ctx.string("r"), &mut rt)),
            "1"
        );
    }

    #[test]
    fn module_level_self_recursion_takes_the_fast_path_correctly() {
        // a module-captured function calling itself resolves the callee
        // via LoadParent every time, so the self-recursive fast path in
        // run_frame's CALL handling actually fires here (unlike a plain
        // global-resolved recursive function) — regression test for a
        // stack-imbalance bug in that path (the peeked callee wasn't
        // popped before computing the recursive call's stack floor)
        let mut rt = module_runtime(true);
        let m = rt
            .load_module(
                "fibonly",
                "fn fib n:
  if n < 2: return n
  (fib (n - 1)) + (fib (n - 2))
export { fib }
",
            )
            .unwrap();
        rt.set_var("m", m);
        let frame = Rc::new(Frame::new());
        for st in &ast_from_str("r = m.fib 10\n") {
            rt.exec_stmt(st, frame.clone()).unwrap();
        }
        assert_eq!(int_var(&mut rt, &frame, "r"), 55);
    }

    #[test]
    fn module_static_call_resolution_respects_reassignment() {
        for compiled in [false, true] {
            let mut rt = module_runtime(compiled);
            // `f` is reassigned after definition — calls must resolve
            // dynamically and see the *final* value; `g` is final and
            // resolves statically; recursion works through both
            let m = rt
                .load_module(
                    "resolve",
                    "fn g x:
  x + 1
fn f x:
  x * 10
f = g
fn use x:
  f (g x)
fn fib n:
  if n < 2: return n
  fib (n - 1) + fib (n - 2)
export { use, fib }
",
                )
                .unwrap();
            rt.set_var("m", m);

            let frame = Rc::new(Frame::new());
            for st in &ast_from_str(
                "r1 = m.use 5
r2 = m.fib 10
",
            ) {
                rt.exec_stmt(st, frame.clone()).unwrap();
            }
            // use 5 → f(g 5) → g(6) → 7 (f is g now, not *10)
            assert_eq!(
                format!("{}", frame.get_var("r1", &mut rt)),
                "7",
                "mode {compiled}"
            );
            assert_eq!(
                format!("{}", frame.get_var("r2", &mut rt)),
                "55",
                "mode {compiled}"
            );
        }
    }

    #[test]
    fn module_cycles_are_detected() {
        let mut rt = module_runtime(true);
        // an `import` host that resolves any name to a module importing it back
        rt.register_function("import_self", 0, Some(0), |rt, _args| {
            match rt.load_module(
                "cycle",
                "m = (import_self)
export { m }
",
            ) {
                Ok(m) => m,
                Err(e) => {
                    rt.set_error(e);
                    Val::nil()
                }
            }
        });
        let err = rt.load_module(
            "cycle",
            "m = (import_self)
export { m }
",
        );
        assert!(err.is_err());
        assert!(format!("{:?}", err.unwrap_err()).contains("circular"));
    }

    #[test]
    fn module_sees_globals_but_prefers_its_own_bindings() {
        for compiled in [false, true] {
            let mut rt = module_runtime(compiled);
            rt.set_var("shift", Val::Number(Number::Integer(100)));
            rt.set_var("pi", Val::Number(Number::Integer(999)));

            // `shift` resolves to the importer's global; `pi` is shadowed
            // by the module's own binding
            let m = rt
                .load_module(
                    "shadow",
                    "pi = 3
fn f x:
    return x + shift + pi
export { f }
",
                )
                .unwrap();
            rt.set_var("m", m);

            let frame = Rc::new(Frame::new());
            for st in &ast_from_str(
                "r = m.f 1
",
            ) {
                rt.exec_stmt(st, frame.clone()).unwrap();
            }
            assert_eq!(
                format!("{}", frame.get_var("r", &mut rt)),
                "104",
                "mode {compiled}"
            );
        }
    }

    #[test]
    fn functions_actually_compile() {
        let src = "fn add a b:\n    return a + b\n";
        let stmts = ast_from_str(src);
        let StmtKind::Fn { params, body, .. } = stmts[0].kind() else {
            panic!("expected fn statement");
        };
        let mut rt = Runtime::new();
        let frame = Rc::new(Frame::new());
        let (compiled, _) =
            compile_fn(&mut rt, params.clone(), body, CompileParent::Walked(frame), &[]).unwrap();
        let instrs: Vec<Instr> = compiled.code.disassemble().map(|r| r.unwrap().1).collect();
        assert!(matches!(instrs.last(), Some(Instr::Return)));
        assert_eq!(compiled.arity(), 2);
    }

    #[test]
    fn bytecode_roundtrips_through_the_encoder() {
        // every instruction kind, with operands spanning varint widths
        let instrs = vec![
            Instr::Const(ConstId(0)),
            Instr::Const(ConstId(127)),
            Instr::Const(ConstId(128)),
            Instr::Const(ConstId(0x4000)),
            Instr::Const(ConstId(u32::MAX)),
            Instr::Nil,
            Instr::Pop,
            Instr::LoadSlot(SlotId(3)),
            Instr::LoadGlobal(StringId(70000)),
            Instr::StoreSlot(SlotId(9)),
            Instr::Bind(PatId(1)),
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
            Instr::IncSlot(SlotId(2)),
            Instr::DecSlot(SlotId(250)),
            Instr::GetField(StringId(4)),
            Instr::GetIndex,
            Instr::SetField(StringId(6)),
            Instr::SetIndex,
            Instr::Call(2),
            Instr::Jump(0), // → byte offset of instruction 0
            Instr::JumpIfFalse(5),
            Instr::IterInit,
            Instr::IterNext(24), // → one past the last instruction
            Instr::IterPop,
            Instr::Return,
        ];
        // jump operands hold instruction indices going in and byte offsets
        // coming out; map the expectation through the encoded layout
        let code = encode(&instrs);
        let decoded: Vec<(usize, Instr)> = code.disassemble().map(|r| r.unwrap()).collect();
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
        assert_eq!(
            encode(&[Instr::LoadSlot(SlotId(3))]).len(),
            1,
            "inline slot"
        );
        assert_eq!(
            encode(&[Instr::Binary(raft_ast::BinOpKind::Add)]).len(),
            1,
            "operator kind lives in the opcode"
        );
        assert_eq!(encode(&[Instr::LoadSlot(SlotId(200))]).len(), 2, "u8 slot");
    }

    #[test]
    fn arithmetic_and_implicit_return() {
        let (mut rt, frame) = assert_modes_agree("fn add a b:\n    a + b\nr = add 1 2\n");
        assert_eq!(int_var(&mut rt, &frame, "r"), 3);
    }

    #[test]
    fn explicit_return_and_operators() {
        let (mut rt, frame) = assert_modes_agree(
            "fn mix a b:\n    return a * b + (a << 2) - b / 2 + a ** 2\nr = mix 7 4\n",
        );
        assert_eq!(int_var(&mut rt, &frame, "r"), 103);
    }

    #[test]
    fn currying_and_partial_application() {
        let (mut rt, frame) = assert_modes_agree(
            "fn add3 a b c:\n    return a + b + c\n\
             add1 = add3 1\nadd12 = add1 2\n\
             r1 = add12 3\nr2 = add3 10 20 30\nr3 = (add3 1) 2 3\n",
        );
        assert_eq!(int_var(&mut rt, &frame, "r1"), 6);
        assert_eq!(int_var(&mut rt, &frame, "r2"), 60);
        assert_eq!(int_var(&mut rt, &frame, "r3"), 6);
    }

    #[test]
    fn over_application_carries_to_returned_function() {
        // `make_adder` returns a function; extra arguments are re-applied
        let (mut rt, frame) = assert_modes_agree(
            "fn add a b:\n    return a + b\n\
             fn make_adder a:\n    return add a\n\
             r = make_adder 3 4\n",
        );
        assert_eq!(int_var(&mut rt, &frame, "r"), 7);
    }

    #[test]
    fn recursion() {
        let (mut rt, frame) = assert_modes_agree(
            "fn fib n:\n    if n < 2:\n        return n\n    return (fib (n - 1)) + (fib (n - 2))\n\
             r = fib 15\n",
        );
        assert_eq!(int_var(&mut rt, &frame, "r"), 610);
    }

    #[test]
    fn while_loop_and_if_else_chain() {
        let (mut rt, frame) = assert_modes_agree(
            "fn collatz n:\n    steps = 0\n    while n != 1:\n        if n & 1 == 0:\n            n = n / 2\n        else:\n            n = 3 * n + 1\n        steps = steps + 1\n    return steps\n\
             r = collatz 27\n",
        );
        assert_eq!(int_var(&mut rt, &frame, "r"), 111);
    }

    #[test]
    fn for_else_and_break() {
        let (mut rt, frame) = assert_modes_agree(
            "fn find xs needle:\n    idx = 0\n    for x in xs:\n        if x == needle:\n            break\n        idx = idx + 1\n    else:\n        return -1\n    return idx\n\
             ys = [10, 20, 30]\nr1 = find ys 20\nr2 = find ys 99\n",
        );
        assert_eq!(int_var(&mut rt, &frame, "r1"), 1);
        assert_eq!(int_var(&mut rt, &frame, "r2"), -1);
    }

    #[test]
    fn continue_in_for() {
        let (mut rt, frame) = assert_modes_agree(
            "fn sum_odds xs:\n    total = 0\n    for x in xs:\n        if x & 1 == 0:\n            continue\n        total = total + x\n    return total\n\
             ys = [1, 2, 3, 4, 5]\nr = sum_odds ys\n",
        );
        assert_eq!(int_var(&mut rt, &frame, "r"), 9);
    }

    #[test]
    fn nested_for_with_inner_break() {
        let (mut rt, frame) = assert_modes_agree(
            "fn count:\n    total = 0\n    for i in [1, 2, 3]:\n        for j in [1, 2, 3]:\n            if j > i:\n                break\n            total = total + 1\n    return total\n\
             r = (count)\n",
        );
        assert_eq!(int_var(&mut rt, &frame, "r"), 6);
    }

    #[test]
    fn while_else_and_break_skips_else() {
        let (mut rt, frame) = assert_modes_agree(
            "fn wloop n:\n    while n < 10:\n        if n > 3:\n            break\n        n = n + 1\n    else:\n        return -1\n    return n\n\
             r1 = wloop 0\nr2 = wloop 20\n",
        );
        assert_eq!(int_var(&mut rt, &frame, "r1"), 4);
        assert_eq!(int_var(&mut rt, &frame, "r2"), -1);
    }

    #[test]
    fn record_param_destructuring() {
        let (mut rt, frame) = assert_modes_agree(
            "fn dist2 { x, y }:\n    return x * x + y * y\n\
             r = dist2 { x: 3, y: 4 }\n",
        );
        assert_eq!(int_var(&mut rt, &frame, "r"), 25);
    }

    #[test]
    fn list_param_destructuring_and_shorthand_record() {
        let (mut rt, frame) = assert_modes_agree(
            "fn swap [a, b]:\n    return [b, a]\n\
             fn wrap x:\n    name = x\n    return { name }\n\
             ys = [1, 2]\nr1 = swap ys\nr2 = wrap \"Ada\"\n",
        );
        assert_eq!(format!("{}", frame.get_var("r2", &mut rt)), "{name: Ada}");
    }

    #[test]
    fn field_and_index_mutation_inside_function() {
        let (mut rt, frame) = assert_modes_agree(
            "fn setup:\n    o = { a: 1 }\n    o.a = 5\n    xs = [1, 2]\n    xs[1] = 9\n    return [o.a, xs[1]]\n\
             r = (setup)\n",
        );
        assert_eq!(format!("{}", frame.get_var("r", &mut rt)), "[5, 9]");
    }

    #[test]
    fn zero_arg_functions_called_bare_and_parenthesized() {
        let (mut rt, frame) = assert_modes_agree(
            "fn five:\n    return 5\n\
             fn ten:\n    (five) + (five)\n\
             r1 = (ten)\nr2 = (five)\n",
        );
        assert_eq!(int_var(&mut rt, &frame, "r1"), 10);
        assert_eq!(int_var(&mut rt, &frame, "r2"), 5);
    }

    #[test]
    fn last_statement_value_semantics() {
        // assignments yield nil, so a body ending in one returns nil;
        // an if with a false condition and no else also yields nil
        let (mut rt, frame) = assert_modes_agree(
            "fn assigns:\n    x = 5\n\
             fn cond_no_else n:\n    123\n    if n > 100:\n        456\n\
             r1 = (assigns)\nr2 = cond_no_else 1\nr3 = cond_no_else 1000\n",
        );
        assert_eq!(format!("{}", frame.get_var("r1", &mut rt)), "Nil");
        assert_eq!(format!("{}", frame.get_var("r2", &mut rt)), "Nil");
        assert_eq!(int_var(&mut rt, &frame, "r3"), 456);
    }

    #[test]
    fn loops_in_tail_position() {
        // a loop as the body's final statement: yields its else-block's
        // value on normal exit, nil on break exit or without an else
        let (mut rt, frame) = assert_modes_agree(
            "fn tail_while_else n:\n    while n > 0:\n        n = n - 1\n    else:\n        Done\n\
             fn tail_while_break n:\n    while True:\n        if n > 2:\n            break\n        n = n + 1\n    else:\n        Done\n\
             fn tail_while_bare n:\n    while n > 0:\n        n = n - 1\n\
             fn tail_for xs:\n    for x in xs:\n        x\n    else:\n        x\n\
             r1 = tail_while_else 3\nr2 = tail_while_break 0\nr3 = tail_while_bare 3\n\
             ys = [7, 8]\nr4 = tail_for ys\n",
        );
        assert_eq!(format!("{}", frame.get_var("r1", &mut rt)), "Done");
        assert_eq!(format!("{}", frame.get_var("r2", &mut rt)), "Nil");
        assert_eq!(format!("{}", frame.get_var("r3", &mut rt)), "Nil");
        assert_eq!(format!("{}", frame.get_var("r4", &mut rt)), "8");
    }

    #[test]
    fn nested_fn_definitions_and_atoms() {
        let (mut rt, frame) = assert_modes_agree(
            "fn classify n:\n    fn sign x:\n        if x > 0:\n            return Pos\n        if x < 0:\n            return Neg\n        return Zero\n    return sign n\n\
             r1 = classify 5\nr2 = classify (-5)\nr3 = classify 0\n",
        );
        assert_eq!(format!("{}", frame.get_var("r1", &mut rt)), "Pos");
        assert_eq!(format!("{}", frame.get_var("r2", &mut rt)), "Neg");
        assert_eq!(format!("{}", frame.get_var("r3", &mut rt)), "Zero");
    }

    #[test]
    fn closure_captures_enclosing_function_local() {
        let (mut rt, frame) = assert_modes_agree(
            "fn make_adder n:\n    fn add x:\n        x + n\n    return add\n\
             f = make_adder 10\nr = f 5\n",
        );
        assert_eq!(int_var(&mut rt, &frame, "r"), 15);
    }

    #[test]
    fn closure_sees_captured_local_mutated_before_it_reads_it() {
        // `n` is captured by `get`, so `n`'s two increments inside
        // `make_thing` must land in the live per-call frame `get` reads
        // from, not a stack slot `get` can't see
        let (mut rt, frame) = assert_modes_agree(
            "fn make_thing base:\n    n = base\n    n = n + 1\n    n = n + 1\n    fn get:\n        return n\n    return get\n\
             g1 = make_thing 10\nr1 = (g1)\n\
             g2 = make_thing 100\nr2 = (g2)\n\
             r1b = (g1)\n",
        );
        assert_eq!(int_var(&mut rt, &frame, "r1"), 12);
        assert_eq!(int_var(&mut rt, &frame, "r2"), 102);
        // g1's captured frame is independent of g2's — reading it again is
        // unaffected by the second call of make_thing
        assert_eq!(int_var(&mut rt, &frame, "r1b"), 12);
    }

    #[test]
    fn closure_captures_through_non_capturing_middle_function() {
        // `middle` doesn't itself reference `g` — only `inner` does — so
        // `middle` must own no captured frame of its own and pass outer's
        // live frame straight through
        let (mut rt, frame) = assert_modes_agree(
            "fn outer g:\n    fn middle:\n        fn inner:\n            return g + 1\n        return inner\n    return middle\n\
             f = outer 100\nmid = (f)\nr = (mid)\n",
        );
        assert_eq!(int_var(&mut rt, &frame, "r"), 101);
    }

    #[test]
    fn sibling_closure_unrelated_to_capture_still_works() {
        // `outer` owns a frame because `user` captures `x` — `skip` is a
        // sibling nested fn that reads nothing from `outer` at all, so it
        // must not get `outer`'s frame attached, yet still has to work
        // correctly (it's still a captures_outer fn, since global-vs-outer
        // isn't distinguished at compile time — see `fn_outer_names`)
        let (mut rt, frame) = assert_modes_agree(
            "fn outer g:\n    x = g + 1\n    fn user:\n        return x\n    fn skip n:\n        return n + 1\n    return [user, skip]\n\
             fns = outer 5\nu = fns[0]\ns = fns[1]\nr1 = (u)\nr2 = s 10\n",
        );
        assert_eq!(int_var(&mut rt, &frame, "r1"), 6);
        assert_eq!(int_var(&mut rt, &frame, "r2"), 11);
    }

    #[test]
    fn three_level_capture_with_siblings_at_every_depth() {
        // level1 owns a frame for `a`,`b` (level3 needs both, transitively
        // through level2). level2 owns a frame for `c`,`d` (level3 needs
        // both). Three siblings at the bottom, each with a different
        // capture shape:
        //   level3         — reads a,b (level1) AND c,d (level2): a
        //                    two-hop chain, level2's frame -> level1's
        //   level3_sibling — reads only c (level2's own): must NOT carry
        //                    level1's frame at all
        //   level3_pure    — reads nothing: plain Const, no MakeClosure
        // Two separate calls of level1/level2 must produce fully
        // independent closures (no cross-talk), and re-reading the same
        // closure twice must be stable.
        let (mut rt, frame) = assert_modes_agree(
            "fn level1 a:\n    b = a + 1\n    fn level2 c:\n        d = b + c\n        fn level3 e:\n            return a + b + c + d + e\n        fn level3_sibling:\n            return c * 2\n        fn level3_pure:\n            return 999\n        return [level3, level3_sibling, level3_pure]\n    return level2\n\
             l2a = level1 10\nl2b = level1 100\n\
             inner1 = l2a 5\ninner2 = l2b 50\n\
             three1 = inner1[0]\nsib1 = inner1[1]\npure1 = inner1[2]\n\
             three2 = inner2[0]\nsib2 = inner2[1]\npure2 = inner2[2]\n\
             r1 = three1 20\nr1_sib = (sib1)\nr1_pure = (pure1)\n\
             r2 = three2 200\nr2_sib = (sib2)\nr2_pure = (pure2)\n\
             r1_again = three1 21\n",
        );
        // inner1: a=10, b=11, c=5, d=16
        assert_eq!(int_var(&mut rt, &frame, "r1"), 62); // 10+11+5+16+20
        assert_eq!(int_var(&mut rt, &frame, "r1_sib"), 10); // c*2
        assert_eq!(int_var(&mut rt, &frame, "r1_pure"), 999);
        // inner2: a=100, b=101, c=50, d=151 — must not see inner1's values
        assert_eq!(int_var(&mut rt, &frame, "r2"), 602); // 100+101+50+151+200
        assert_eq!(int_var(&mut rt, &frame, "r2_sib"), 100); // c*2
        assert_eq!(int_var(&mut rt, &frame, "r2_pure"), 999);
        // re-reading inner1's level3 again is stable, unaffected by inner2
        assert_eq!(int_var(&mut rt, &frame, "r1_again"), 63); // 10+11+5+16+21
    }

    #[test]
    fn recursive_nested_fn_also_captures_outer_local() {
        // `fact` both self-recurses (its own name resolves as a free name,
        // one level out, exercising the same self-recursive fast path as
        // module-level recursion) and captures `base` from `make_fact` at
        // the same time. Two separate `make_fact` calls must stay
        // independent, and calling the same closure with different
        // arguments must not corrupt it.
        let (mut rt, frame) = assert_modes_agree(
            "fn make_fact base:\n    fn fact n:\n        if n < 2:\n            return base\n        return n * (fact (n - 1))\n    return fact\n\
             f5 = make_fact 100\nf1 = make_fact 1\n\
             r_a = f5 5\nr_b = f1 5\nr_a_again = f5 3\n",
        );
        assert_eq!(int_var(&mut rt, &frame, "r_a"), 12000); // 5*4*3*2*100
        assert_eq!(int_var(&mut rt, &frame, "r_b"), 120); // plain 5!
        assert_eq!(int_var(&mut rt, &frame, "r_a_again"), 600); // 3*2*100
    }

    #[test]
    fn nested_fn_sees_module_x_then_shadowing_fn_local_x() {
        // `inner` lexically reads `x` from `middle` (middle also assigns
        // `x`, so `x` resolves to middle's own captured slot, never
        // module's, by static scoping) — but the FIRST call happens
        // before middle's own assignment runs, so it must fall through
        // middle's still-Uninit captured slot to the module's live `x`.
        // The SECOND call, after middle assigns its own `x`, must see
        // middle's value instead.
        for compiled in [false, true] {
            let mut rt = module_runtime(compiled);
            let m = rt
                .load_module(
                    "shadowx",
                    "x = 1
fn middle:
    fn inner:
        return x
    r1 = (inner)
    x = 2
    r2 = (inner)
    return [r1, r2]
export { middle }
",
                )
                .unwrap();
            rt.set_var("m", m);
            let frame = Rc::new(Frame::new());
            for st in &ast_from_str("mid = m.middle\nr = (mid)\n") {
                rt.exec_stmt(st, frame.clone()).unwrap();
            }
            assert_eq!(
                format!("{}", frame.get_var("r", &mut rt)),
                "[1, 2]",
                "mode {compiled}"
            );
        }
    }

    #[test]
    fn iterating_records_and_strings_of_ops() {
        let (mut rt, frame) = assert_modes_agree(
            "fn count_fields rec:\n    n = 0\n    for f in rec:\n        n = n + 1\n    return n\n\
             r = count_fields { a: 1, b: 2, c: 3 }\n",
        );
        assert_eq!(int_var(&mut rt, &frame, "r"), 3);
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
            for a in rt.stack.drain_top(args).rev() {
                sink.borrow_mut().push(format!("{a}"));
            }
            Val::nil()
        });

        let frame = Rc::new(Frame::new());

        for statement in &stmts {
            rt.exec_stmt(statement, frame.clone()).unwrap();
        }
        for statement in &ast_from_str("shout 21\n") {
            rt.exec_stmt(statement, frame.clone()).unwrap();
        }

        assert_eq!(*seen.borrow(), vec!["21".to_string(), "42".to_string()]);
    }

    #[test]
    fn all_pattern_kinds_bind_identically_in_both_modes() {
        // atom tags, literal matches, and nested destructuring — every
        // CompiledPat variant, checked against the walker's bind_pattern
        let (mut rt, frame) = assert_modes_agree(
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
        assert_eq!(format!("{}", frame.get_var("r1", &mut rt)), "12");
        assert_eq!(format!("{}", frame.get_var("r2", &mut rt)), "Ada");
        assert_eq!(format!("{}", frame.get_var("r3", &mut rt)), "True");
        assert_eq!(format!("{}", frame.get_var("r4", &mut rt)), "5");
        assert_eq!(format!("{}", frame.get_var("r5", &mut rt)), "One");
        assert_eq!(format!("{}", frame.get_var("r6", &mut rt)), "42");
        assert_eq!(format!("{}", frame.get_var("r7", &mut rt)), "43");
        assert_eq!(format!("{}", frame.get_var("r8", &mut rt)), "44");
        assert_eq!(format!("{}", frame.get_var("r9", &mut rt)), "45");

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
        let (mut rt, frame) = assert_modes_agree("fn one 1 x:\n    return x\nr = one 1.0 7\n");
        assert_eq!(format!("{}", frame.get_var("r", &mut rt)), "7");
    }

    #[test]
    fn number_patterns_match_exactly() {
        let (mut rt, frame) = assert_modes_agree(
            "fn inf_pat 1e999:\n    return Inf\n\
             fn zero 0f:\n    return Zero\n\
             fn big 9007199254740993:\n    return Big\n\
             r1 = inf_pat 2e999\n\
             r2 = zero (-0f)\n\
             r3 = big 9007199254740993\n",
        );
        // +inf matches +inf (the old |a-b| < ε test gave NaN for these)
        assert_eq!(format!("{}", frame.get_var("r1", &mut rt)), "Inf");
        // -0.0 matches 0.0, like the language's own `==`
        assert_eq!(format!("{}", frame.get_var("r2", &mut rt)), "Zero");
        assert_eq!(format!("{}", frame.get_var("r3", &mut rt)), "Big");

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
        let (mut rt, frame) = assert_modes_agree(
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
        assert_eq!(format!("{}", frame.get_var("r1", &mut rt)), "2");
        assert_eq!(format!("{}", frame.get_var("r2", &mut rt)), "8");
        assert_eq!(format!("{}", frame.get_var("r3", &mut rt)), "5");
        assert_eq!(format!("{}", frame.get_var("r4", &mut rt)), "3");
        assert_eq!(format!("{}", frame.get_var("r5", &mut rt)), "3");
        // `_` was never bound anywhere along the way
        assert!(!frame.get_var("_", &mut rt).is_init());

        // names merely *starting* with an underscore bind normally
        let (mut rt, frame) = assert_modes_agree("fn keep _a:\n    return _a\nr = keep 42\n");
        assert_eq!(format!("{}", frame.get_var("r", &mut rt)), "42");

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
        let (mut rt, frame) = assert_modes_agree(
            "count = 10\n\
             fn bump:\n    count = count + 1\n    return count\n\
             r1 = (bump)\nr2 = (bump)\nr3 = count\n",
        );
        assert_eq!(format!("{}", frame.get_var("r1", &mut rt)), "11");
        assert_eq!(format!("{}", frame.get_var("r2", &mut rt)), "11");
        assert_eq!(format!("{}", frame.get_var("r3", &mut rt)), "10");

        // reassigning a parameter reuses the parameter's own slot
        let (mut rt, frame) = assert_modes_agree(
            "fn inc n:\n    n = n + 1\n    return n\n\
             fn swap_halves [a, b]:\n    t = a\n    a = b\n    b = t\n    return [a, b]\n\
             xs = [1, 2]\n\
             r1 = inc 41\nr2 = swap_halves xs\n",
        );
        assert_eq!(format!("{}", frame.get_var("r1", &mut rt)), "42");
        assert_eq!(format!("{}", frame.get_var("r2", &mut rt)), "[2, 1]");
    }

    #[test]
    fn pools_are_shared_and_deduplicated_across_functions() {
        let mut rt = Runtime::new();
        rt.set_compile_fns(true);
        let frame = Rc::new(Frame::new());
        for statement in &ast_from_str(
            "fn inc n:\n    return (shift n) + 1.5\n\
             fn dec n:\n    return (shift n) - 1.5\n\
             fn one:\n    return 1\n",
        ) {
            rt.exec_stmt(statement, frame.clone()).unwrap();
        }

        // ...while both functions reference the global `shift` and the
        // constant 1.5 through single shared pool entries
        assert_eq!(
            rt.ctx.strings.iter().filter(|m| &m[..] == "shift").count(),
            1,
            "global name `shift` interned more than once"
        );
        assert_eq!(
            rt.ctx
                .consts
                .iter()
                .filter(|c| matches!(c, Val::Number(Number::Float(f)) if *f == 1.5))
                .count(),
            1,
            "constant 1.5 interned more than once"
        );
        // integers ride in opcodes/payloads and never enter the pool
        assert_eq!(
            rt.ctx
                .consts
                .iter()
                .filter(|c| matches!(c, Val::Number(Number::Integer(_))))
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
            Val::Number(Number::Integer(rt.stack.len() as i64))
        });

        let frame = Rc::new(Frame::new());
        for statement in &ast_from_str(
            "fn probe:\n    return 100 + (depth)\n\
             fn fib n:\n    if n < 2:\n        return n\n    return (fib (n - 1)) + (fib (n - 2))\n",
        ) {
            rt.exec_stmt(statement, frame.clone()).unwrap();
        }

        // inside `probe` the temporary `100` sits on the stack when
        // `depth` runs, so it reports 1
        for statement in &ast_from_str("r = (probe)\n") {
            rt.exec_stmt(statement, frame.clone()).unwrap();
        }
        assert_eq!(format!("{}", frame.get_var("r", &mut rt)), "101");

        // recursion nests frames on the one shared stack and unwinds fully
        for statement in &ast_from_str("f = fib 12\n") {
            rt.exec_stmt(statement, frame.clone()).unwrap();
        }
        assert_eq!(format!("{}", frame.get_var("f", &mut rt)), "144");
        assert!(rt.stack.is_empty(), "stack not restored after calls");

        // ...and is restored even when a frame errors out mid-expression
        let stmts = ast_from_str("fn bad n:\n    return n + nosuchvar\nfib (bad 1)\n");
        for statement in &stmts[..1] {
            rt.exec_stmt(statement, frame.clone()).unwrap();
        }
        assert!(rt.exec_stmt(&stmts[1], frame.clone()).is_err());
        assert!(rt.stack.is_empty(), "stack not restored after error");
    }

    #[test]
    fn modes_mix_in_one_runtime() {
        let mut rt = Runtime::new();
        let frame = Rc::new(Frame::new());

        // `double` walks the AST...
        rt.set_compile_fns(false);
        for statement in &ast_from_str("fn double x:\n    return x * 2\n") {
            rt.exec_stmt(statement, frame.clone()).unwrap();
        }

        // ...`quad` runs on the VM and calls `double`...
        rt.set_compile_fns(true);
        for statement in &ast_from_str("fn quad x:\n    return double (double x)\n") {
            rt.exec_stmt(statement, frame.clone()).unwrap();
        }

        // ...`octo` walks the AST again and calls compiled `quad`...
        rt.set_compile_fns(false);
        for statement in &ast_from_str("fn octo x:\n    return double (quad x)\n") {
            rt.exec_stmt(statement, frame.clone()).unwrap();
        }

        // ...and the top level is interpreted from the AST.
        for statement in &ast_from_str("r = octo 5\n") {
            rt.exec_stmt(statement, frame.clone()).unwrap();
        }

        assert_eq!(
            match frame.get_var("r", &mut rt) {
                Val::Number(Number::Integer(i)) => i,
                other => panic!("unexpected {other:?}"),
            },
            40
        );
    }

    #[test]
    fn top_level_expression_sees_compiled_function() {
        let mut rt = Runtime::new();
        let frame = Rc::new(Frame::new());
        rt.set_compile_fns(true);
        for statement in &ast_from_str("fn triple x:\n    return x * 3\n") {
            rt.exec_stmt(statement, frame.clone()).unwrap();
        }
        // interpreted expression calling into bytecode
        let stmts = ast_from_str("triple 14\n");
        let Exec::Value(v) = rt.exec_stmt(&stmts[0], frame.clone()).unwrap() else {
            panic!("expected value");
        };
        assert_eq!(format!("{v}"), "42");
    }
}
