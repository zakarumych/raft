#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::{rc::Rc, vec::Vec};

use core::{cell::RefCell, cmp::Ordering, fmt};

use smallvec::SmallVec;

use raft_ast::{BinOpKind, Expr, ExprKind, Lit, LitNum, Pat, PatKind, Stmt, StmtKind, UnOpKind};

use crate::vm::CompiledPat;

pub mod vm;

// The object model (`Val`, `Object`, `Atom`, `Number`, `RuntimeError`,
// `Function`/`DynFn`/`FnVal`, the `Host` bridge trait, ...) lives in
// `raft-core` now, so it can be shared unmodified by other execution modes.
// Re-exported wholesale so existing call sites (`raft_runtime::Val`, etc.)
// don't need to change.
pub use raft_core::*;

type HashMap<K, V> = hashbrown::HashMap<K, V, foldhash::fast::RandomState>;

/// An `fn`-defined function executed by walking its AST body.
struct AstFn {
    params: Rc<[Pat]>,
    body: Rc<[Stmt]>,
    /// Parent frame.
    parent: Rc<Frame>,
}

impl Function for AstFn {
    fn min_args(&self) -> usize {
        self.params.len()
    }

    fn max_args(&self) -> Option<usize> {
        // consumes exactly its parameter count; the runtime clamps
        // over-application and re-applies the leftovers to the result
        Some(self.params.len())
    }

    fn call(&self, rt: &mut dyn Host, args: usize) {
        let rt = rt
            .as_any_mut()
            .downcast_mut::<Runtime>()
            .expect("AstFn only runs under raft_runtime::Runtime");

        debug_assert!(rt.stack.len() >= args);
        debug_assert_eq!(args, self.params.len());

        // the body sees this function's module environment, not the caller's
        let frame = Rc::new(Frame::new().with_parent(self.parent.clone()));

        // first argument is on top of the stack
        for param in self.params.iter() {
            let arg = rt.stack.pop();
            if let Err(e) = rt.bind_pattern(param, &arg, &frame) {
                rt.set_error(e);
                return;
            }
        }

        let ret = match rt.exec_block(&self.body, frame.clone()) {
            Ok(Exec::Value(v)) => v,
            Ok(Exec::Return(v)) => v,
            Ok(Exec::Break) => {
                rt.set_error(RuntimeError::Other(
                    "break statement outside of loop".into(),
                ));
                return;
            }
            Ok(Exec::Continue) => {
                rt.set_error(RuntimeError::Other(
                    "continue statement outside of loop".into(),
                ));
                return;
            }
            Err(e) => {
                rt.set_error(e);
                return;
            }
        };

        rt.stack.push(ret);
    }

    fn debug_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<fn {:?} {{ ... }}>", self.params)
    }
}

fn fn_from_ast(params: Rc<[Pat]>, body: Rc<[Stmt]>, parent: Rc<Frame>) -> Val {
    Val::Fn(FnVal::new(AstFn {
        params,
        body,
        parent,
    }))
}

/// Identified used to index into function-stack slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct SlotId(pub u32);

/// Collect every identifier *read* reachable from an expression (record
/// shorthand `{ key }` counts as a read of `key`) into `out`. Doesn't
/// distinguish bound from outer — callers filter against a `SlotTable`.
fn collect_reads_expr(expr: &Expr, out: &mut Vec<Rc<str>>) {
    match expr.kind() {
        ExprKind::Ident(id) => out.push(id.rc_name()),
        ExprKind::Atom(_) | ExprKind::Literal(_) => {}
        ExprKind::List(items) => {
            for e in items.iter() {
                collect_reads_expr(e, out);
            }
        }
        ExprKind::Record(fields) => {
            for f in fields.iter() {
                match f.value() {
                    Some(v) => collect_reads_expr(v, out),
                    None => out.push(f.key().rc_name()),
                }
            }
        }
        ExprKind::Unary(_, e) => collect_reads_expr(e, out),
        ExprKind::Binary(a, _, b) => {
            collect_reads_expr(a, out);
            collect_reads_expr(b, out);
        }
        ExprKind::Apply(callee, args) => {
            collect_reads_expr(callee, out);
            for a in args.iter() {
                collect_reads_expr(a, out);
            }
        }
        ExprKind::Field(obj, _) => collect_reads_expr(obj, out),
        ExprKind::Index(obj, idx) => {
            collect_reads_expr(obj, out);
            collect_reads_expr(idx, out);
        }
        ExprKind::Parenthesized(e) => collect_reads_expr(e, out),
    }
}

/// Same, but over a statement — nested `fn` statements contribute their own
/// outer names (recursively computed by [`fn_outer_names`]) as reads at this
/// level, so a name that's only outer several levels deep still propagates
/// outward.
fn collect_reads_stmt(stmt: &Stmt, out: &mut Vec<Rc<str>>) {
    match stmt.kind() {
        StmtKind::Expr(e) => collect_reads_expr(e, out),
        StmtKind::AssignPat { value, .. } => collect_reads_expr(value, out),
        StmtKind::AssignField {
            target,
            field: _,
            value,
        } => {
            collect_reads_expr(target, out);
            collect_reads_expr(value, out);
        }
        StmtKind::AssignIndex {
            target,
            index,
            value,
        } => {
            collect_reads_expr(target, out);
            collect_reads_expr(index, out);
            collect_reads_expr(value, out);
        }
        StmtKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            collect_reads_expr(cond, out);
            for s in then_branch.iter() {
                collect_reads_stmt(s, out);
            }
            if let Some(eb) = else_branch {
                for s in eb.iter() {
                    collect_reads_stmt(s, out);
                }
            }
        }
        StmtKind::While {
            cond,
            body,
            else_branch,
        } => {
            collect_reads_expr(cond, out);
            for s in body.iter() {
                collect_reads_stmt(s, out);
            }
            if let Some(eb) = else_branch {
                for s in eb.iter() {
                    collect_reads_stmt(s, out);
                }
            }
        }
        StmtKind::For {
            target: _,
            iterable,
            body,
            else_branch,
        } => {
            collect_reads_expr(iterable, out);
            for s in body.iter() {
                collect_reads_stmt(s, out);
            }
            if let Some(eb) = else_branch {
                for s in eb.iter() {
                    collect_reads_stmt(s, out);
                }
            }
        }
        StmtKind::Return(Some(e)) => collect_reads_expr(e, out),
        StmtKind::Return(None) | StmtKind::Break | StmtKind::Continue => {}
        StmtKind::Fn { params, body, .. } => {
            out.extend(fn_outer_names(params, body));
        }
    }
}

/// Names read inside `body` (params included as bound) that aren't bound
/// anywhere within it — i.e. must resolve to an enclosing scope. Recurses
/// into nested `fn` bodies, so a name only referenced by a grandchild `fn`
/// still shows up here (propagated up through [`collect_reads_stmt`]'s
/// `StmtKind::Fn` arm).
fn fn_outer_names(params: &[Pat], body: &[Stmt]) -> Vec<Rc<str>> {
    // only params are unconditionally initialized before any possible
    // read — a body-assigned name may still read through to an enclosing
    // scope on its first (pre-assignment) access under the language's
    // shadow-with-fallback rule (`x = x + 1` reads the outer `x`), so it
    // must NOT be excluded here just because it's also assigned locally
    let bound = SlotTable::with_params(params);

    let mut reads = Vec::new();
    for s in body.iter() {
        collect_reads_stmt(s, &mut reads);
    }

    reads.retain(|n| bound.get(n).is_none());
    reads
}

struct SlotTable {
    table: HashMap<Rc<str>, SlotId>,
    next: SlotId,
}

impl SlotTable {
    fn with_params(params: &[Pat]) -> Self {
        let mut next = 0;
        let mut table = HashMap::default();

        for param in params.iter().rev() {
            if let PatKind::Ident(id) = param.kind() {
                if id.name() != "_" {
                    table.insert(id.rc_name(), SlotId(next));
                }
            }
            next += 1;
        }

        let mut me = SlotTable {
            table,
            next: SlotId(next),
        };

        for param in params {
            if let PatKind::Ident(_) = param.kind() {
                continue;
            }
            me.add_pat(param);
        }

        me
    }

    fn add_name(&mut self, name: Rc<str>) {
        self.table.entry(name).or_insert_with(|| {
            let next = self.next;
            self.next = SlotId(next.0 + 1);
            next
        });
    }

    fn add_pat(&mut self, pat: &Pat) {
        match pat.kind() {
            PatKind::Ident(id) if id.name() == "_" => {}
            PatKind::Ident(ident) => self.add_name(ident.rc_name()),
            PatKind::List(list) => {
                for p in list.iter() {
                    self.add_pat(p);
                }
            }
            PatKind::Record(fields) => {
                for f in fields.iter() {
                    match f.pattern() {
                        Some(p) => self.add_pat(p),
                        None => {
                            self.add_name(f.key().rc_name());
                        }
                    }
                }
            }
            PatKind::Atom(_) | PatKind::Literal(_) => {}
        }
    }

    fn add_stmt(&mut self, stmt: &Stmt) {
        match stmt.kind() {
            StmtKind::AssignPat { target, .. } => self.add_pat(target),
            StmtKind::If {
                then_branch,
                else_branch,
                ..
            } => {
                self.add_stmts(then_branch);
                if let Some(eb) = else_branch {
                    self.add_stmts(eb);
                }
            }
            StmtKind::While {
                body, else_branch, ..
            } => {
                self.add_stmts(body);
                if let Some(eb) = else_branch {
                    self.add_stmts(eb);
                }
            }
            StmtKind::For {
                target,
                body,
                else_branch,
                ..
            } => {
                self.add_pat(target);
                self.add_stmts(body);
                if let Some(eb) = else_branch {
                    self.add_stmts(eb);
                }
            }
            StmtKind::Fn { name, .. } => {
                self.add_name(name.rc_name());
            }
            StmtKind::Expr(_)
            | StmtKind::AssignField { .. }
            | StmtKind::AssignIndex { .. }
            | StmtKind::Return(_)
            | StmtKind::Break
            | StmtKind::Continue => {}
        }
    }

    fn add_stmts(&mut self, stmts: &[Stmt]) {
        for stmt in stmts {
            self.add_stmt(stmt);
        }
    }

    fn get(&self, name: &str) -> Option<SlotId> {
        self.table.get(name).copied()
    }

    /// Which of this function's own slots are read by some `fn` nested
    /// (at any depth) inside `body` — those need to live in a per-call
    /// [`Frame`] instead of a stack slot, since a closure escaping this
    /// call must still see them. Everything else stays a stack slot.
    fn mark_captured(&self, body: &[Stmt]) -> Vec<bool> {
        fn collect_nested_fns<'a>(stmts: &'a [Stmt], out: &mut Vec<&'a Stmt>) {
            for stmt in stmts {
                match stmt.kind() {
                    StmtKind::Fn { .. } => out.push(stmt),
                    StmtKind::If {
                        then_branch,
                        else_branch,
                        ..
                    } => {
                        collect_nested_fns(then_branch, out);
                        if let Some(eb) = else_branch {
                            collect_nested_fns(eb, out);
                        }
                    }
                    StmtKind::While {
                        body, else_branch, ..
                    } => {
                        collect_nested_fns(body, out);
                        if let Some(eb) = else_branch {
                            collect_nested_fns(eb, out);
                        }
                    }
                    StmtKind::For {
                        body, else_branch, ..
                    } => {
                        collect_nested_fns(body, out);
                        if let Some(eb) = else_branch {
                            collect_nested_fns(eb, out);
                        }
                    }
                    _ => {}
                }
            }
        }

        let mut captured = alloc::vec![false; self.next.0 as usize];
        let mut nested = Vec::new();
        collect_nested_fns(body, &mut nested);
        for stmt in nested {
            let StmtKind::Fn { params, body, .. } = stmt.kind() else {
                unreachable!()
            };
            for name in fn_outer_names(params, body) {
                if let Some(slot) = self.get(&name) {
                    captured[slot.0 as usize] = true;
                }
            }
        }
        captured
    }

    fn names(&self, rt: &mut Runtime) -> SmallVec<[StringId; 8]> {
        let mut pairs: SmallVec<[(u32, Rc<str>); 8]> = self
            .table
            .iter()
            .map(|(k, idx)| (idx.0, k.clone()))
            .collect();
        pairs.sort_unstable_by_key(|(idx, _)| *idx);

        let mut names = SmallVec::with_capacity(self.next.0 as usize);

        for (idx, name) in pairs {
            if idx > names.len() as u32 {
                for _ in 0..idx {
                    names.push(rt.ctx.string("_"));
                }
            }
            names.push(rt.ctx.string(name));
        }

        names
    }
}

#[derive(Default)]
pub struct Stack {
    array: Vec<Val>,
}

impl Stack {
    /// Reserve `n` not-yet-assigned locals on top of the stack.
    #[inline]
    pub fn extend_uninit(&mut self, n: usize) {
        self.array.resize_with(self.array.len() + n, || Val::Uninit);
    }

    #[inline]
    pub fn push(&mut self, v: Val) {
        self.array.push(v);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.array.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.array.is_empty()
    }

    #[inline]
    pub fn pop(&mut self) -> Val {
        match self.array.pop() {
            Some(v) => v,
            None => unreachable!("Attempted to pop from an empty VM stack"),
        }
    }

    #[inline]
    pub fn peek(&self) -> &Val {
        match self.array.last() {
            Some(v) => v,
            None => unreachable!("Attempted to peek from an empty VM stack"),
        }
    }

    #[inline]
    pub fn extend(&mut self, values: impl IntoIterator<Item = Val>) {
        self.array.extend(values);
    }

    #[inline]
    pub fn reverse(&mut self, count: usize) {
        let at = self.array.len() - count;
        self.array[at..].reverse();
    }

    #[inline]
    pub fn drain_top(&mut self, count: usize) -> impl DoubleEndedIterator<Item = Val> {
        let at = self.array.len() - count;
        self.array.drain(at..)
    }

    #[inline]
    pub fn truncate(&mut self, len: usize) {
        self.array.truncate(len);
    }

    /// Read frame slot `slot` of the frame based at `base`.
    #[inline]
    pub fn get(&self, idx: usize) -> &Val {
        &self.array[idx]
    }

    /// Write frame slot `slot` of the frame based at `base`.
    #[inline]
    pub fn set(&mut self, idx: usize, v: Val) {
        self.array[idx] = v;
    }
}

#[derive(Default)]
pub struct Context {
    /// Interned strings used as identifiers in compiled functions.
    strings: Vec<Rc<str>>,

    /// Contains all constants used within compiled functions.
    consts: Vec<Val>,

    /// Contains compiled patterns used by compiled functions.
    pats: Vec<Rc<CompiledPat>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct StringId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ConstId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct PatId(pub u32);

pub trait IntoStringId {
    fn into_id(self, ctx: &mut Context) -> StringId;
}

impl IntoStringId for StringId {
    #[inline(always)]
    fn into_id(self, _ctx: &mut Context) -> StringId {
        self
    }
}

impl<S> IntoStringId for S
where
    S: AsRef<str> + Into<Rc<str>>,
{
    #[inline(always)]
    fn into_id(self, ctx: &mut Context) -> StringId {
        ctx.string(self)
    }
}

impl Context {
    /// Intern a constant. Only immutable scalar values are deduplicated —
    /// and never across numeric kinds, since `Any`'s equality treats `1`
    /// and `1.0` as equal but the program must observe distinct values.
    pub fn const_(&mut self, v: Val) -> ConstId {
        fn same(a: &Val, b: &Val) -> bool {
            match (a, b) {
                (Val::Number(Number::Integer(x)), Val::Number(Number::Integer(y))) => x == y,
                (Val::Number(Number::Float(x)), Val::Number(Number::Float(y))) => {
                    x.to_bits() == y.to_bits()
                }
                (Val::String(x), Val::String(y)) => x == y,
                (Val::Char(x), Val::Char(y)) => x == y,
                (Val::Atom(x), Val::Atom(y)) => x == y,
                _ => false,
            }
        }

        if let Some(i) = self.consts.iter().position(|c| same(c, &v)) {
            return ConstId(i as u32);
        }
        self.consts.push(v);
        ConstId((self.consts.len() - 1) as u32)
    }

    pub fn string<S>(&mut self, name: S) -> StringId
    where
        S: AsRef<str> + Into<Rc<str>>,
    {
        if let Some(i) = self
            .strings
            .iter()
            .position(|s| s.as_ref() == name.as_ref())
        {
            return StringId(i as u32);
        }
        self.strings.push(name.into());
        StringId((self.strings.len() - 1) as u32)
    }

    pub fn pattern(&mut self, p: CompiledPat) -> PatId {
        self.pats.push(Rc::new(p));
        PatId((self.pats.len() - 1) as u32)
    }

    pub fn get_string(&self, id: StringId) -> Rc<str> {
        self.strings[id.0 as usize].clone()
    }

    pub fn get_const(&self, id: ConstId) -> Val {
        self.consts[id.0 as usize].clone()
    }

    pub fn get_pattern(&self, id: PatId) -> Rc<CompiledPat> {
        self.pats[id.0 as usize].clone()
    }
}

/// The AST walker's dynamic scope. Grows as statements assign new names
/// (no fixed layout — unlike [`vm::CompiledFrame`], which compiled code
/// uses instead), resolved purely by name, chained to whatever frame was
/// live when the enclosing `fn`/module/REPL root started executing.
#[derive(Debug)]
pub struct Frame {
    slots: RefCell<SmallVec<[(StringId, Val); 8]>>,
    parent: Option<Rc<Frame>>,
}

impl Frame {
    pub fn new() -> Self {
        Frame {
            slots: RefCell::new(SmallVec::new()),
            parent: None,
        }
    }

    pub fn with_parent(mut self, parent: Rc<Frame>) -> Self {
        self.parent = Some(parent);
        self
    }

    pub fn set_var(&self, var: StringId, val: Val) {
        let mut slots = self.slots.borrow_mut();
        if let Some(entry) = slots.iter_mut().find(|(n, _)| *n == var) {
            entry.1 = val;
        } else {
            slots.push((var, val));
        }
    }

    pub fn get_var(&self, var: impl IntoStringId, rt: &mut Runtime) -> Val {
        let var = var.into_id(&mut rt.ctx);
        if let Some((_, v)) = self.slots.borrow().iter().find(|(n, _)| *n == var) {
            if !matches!(v, Val::Uninit) {
                return v.clone();
            }
        }
        core::hint::cold_path();
        match &self.parent {
            Some(parent) => parent.get_var(var, rt),
            None => rt.get_var(var),
        }
    }

    /// This frame's own bindings (not the parent chain) — for inspection
    /// (e.g. comparing walker/VM globals in tests), not used by evaluation.
    pub fn own_entries(&self) -> SmallVec<[(StringId, Val); 8]> {
        self.slots.borrow().clone()
    }
}

pub struct Runtime {
    /// Context holding tables with names, constants, and compiled patterns
    /// for all compiled functions to use.
    pub ctx: Context,

    /// The operand stack shared by all compiled-function frames. Public for
    /// inspection — a host function called from compiled code can watch the
    /// caller's temporaries live. Each frame works relative to the stack
    /// height at its entry and restores it on exit; pushing extra values
    /// from a host function mid-call is at your own peril.
    pub stack: Stack,

    /// Global variables, keyed by the name's index.
    global: HashMap<StringId, Val>,

    /// Loaded-module cache, keyed by the name given to [`Runtime::load_module`].
    modules: HashMap<StringId, Val>,

    /// Contexts of modules currently loading, innermost last (cycle detection).
    loading: Vec<StringId>,

    /// Error status.
    status: Result<(), RuntimeError>,

    /// When true, `fn` statements are compiled to stack-based bytecode
    /// (see [`vm`]) instead of being closed over as AST. Both kinds of
    /// function are plain `Any::Fn` values and can call each other freely.
    compile_fns: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Exec {
    /// Stmt executed successfully, no control flow change.
    Value(Val),
    /// Return statement encountered.
    Return(Val),
    /// Continue statement encountered.
    Continue,
    /// Break statement encountered.
    Break,
}

impl Host for Runtime {
    #[inline]
    fn stack_push(&mut self, v: Val) {
        self.stack.push(v);
    }

    #[inline]
    fn stack_pop(&mut self) -> Val {
        self.stack.pop()
    }

    #[inline]
    fn stack_len(&self) -> usize {
        self.stack.len()
    }

    #[inline]
    fn stack_drain_top_into(&mut self, out: &mut [Val]) {
        let count = out.len();
        for (slot, v) in out.iter_mut().zip(self.stack.drain_top(count)) {
            *slot = v;
        }
    }

    #[inline]
    fn stack_extend(&mut self, values: &[Val]) {
        self.stack.extend(values.iter().cloned());
    }

    #[inline]
    fn set_error(&mut self, err: RuntimeError) {
        Runtime::set_error(self, err);
    }

    #[inline]
    fn as_any_mut(&mut self) -> &mut dyn core::any::Any {
        self
    }
}

impl Runtime {
    pub fn new() -> Self {
        Runtime {
            ctx: Context::default(),
            stack: Stack::default(),
            global: HashMap::default(),
            modules: HashMap::default(),
            loading: Vec::new(),
            status: Ok(()),
            compile_fns: false,
        }
    }

    /// Choose how `fn` statements executed from here on are realized:
    /// `true` compiles them to bytecode run by [`vm::run`], `false` (the
    /// default) keeps the tree-walking closure. The modes mix freely within
    /// one runtime — functions defined either way call each other through
    /// the same `Any::Fn` interface.
    pub fn set_compile_fns(&mut self, enabled: bool) {
        self.compile_fns = enabled;
    }

    pub fn compile_fns(&self) -> bool {
        self.compile_fns
    }

    #[cold]
    pub fn set_error(&mut self, err: RuntimeError) {
        self.status = Err(err);
    }

    /// Register a host function in the global scope.
    /// The closure is called with the runtime and the number of arguments it was given;
    /// it can then pop them off the stack and return a value.
    pub fn register_function<F>(
        &mut self,
        name: &str,
        min_args: usize,
        max_args: Option<usize>,
        f: F,
    ) where
        F: Fn(&mut Runtime, usize) -> Val + 'static,
    {
        let wrapped = move |host: &mut dyn Host, args: usize| -> Val {
            let rt = host
                .as_any_mut()
                .downcast_mut::<Runtime>()
                .expect("register_function closures only run under raft_runtime::Runtime");
            f(rt, args)
        };
        let idx = self.ctx.string(name);
        self.global
            .insert(idx, Val::host_function(min_args, max_args, wrapped));
    }

    /// Set variable according to scope rules. If local scope exists, set there; otherwise global.
    pub fn set_var(&mut self, name: impl IntoStringId, val: Val) {
        let name = name.into_id(&mut self.ctx);
        self.global.insert(name, val);
    }

    /// Get variable: check local first, then global.
    pub fn get_var(&mut self, name: impl IntoStringId) -> Val {
        let name = name.into_id(&mut self.ctx);
        self.global.get(&name).cloned().unwrap_or(Val::Uninit)
    }

    pub fn eval(&mut self, expr: &Expr, frame: &Frame) -> Result<Val, RuntimeError> {
        self.eval_impl(expr, frame, false)
    }

    fn eval_impl(
        &mut self,
        expr: &Expr,
        frame: &Frame,
        call_fn: bool,
    ) -> Result<Val, RuntimeError> {
        match expr.kind() {
            ExprKind::Literal(lit) => literal_value(lit),
            ExprKind::Atom(a) => Ok(Val::new_atom(a.rc_name())),
            ExprKind::Ident(i) => {
                let name = self.ctx.string(i.rc_name());

                // Get the variable from the current frame first,
                // then from the global scope if uninitialized.
                let val = frame
                    .get_var(name, self)
                    .init_or_else(|| {
                        RuntimeError::UnboundIdentifier(i.rc_name())
                    })?;

                if call_fn {
                    self.call_ast(val, 0)
                } else {
                    Ok(val)
                }
            }
            ExprKind::List(elements) => {
                let mut vec = Vec::with_capacity(elements.len());
                for e in elements.iter() {
                    vec.push(self.eval(e, frame)?);
                }
                Ok(Val::new_list(vec))
            }
            ExprKind::Record(fields) => {
                let mut map = FixedHashMap::default();
                for f in fields.iter() {
                    let key = f.key().rc_name();

                    let val = match f.value() {
                        None => frame
                            .get_var(key.clone(), self)
                            .init_or_else(|| {
                                RuntimeError::UnboundIdentifier(key.clone())
                            })?,
                        Some(value) => self.eval(value, frame)?,
                    };

                    map.insert(key, val);
                }
                Ok(Val::new_record(map))
            }
            ExprKind::Unary(op, operand) => {
                let v = self.eval(operand, frame)?;
                eval_unary(op.kind(), &v)
            }
            ExprKind::Binary(lhs, op, rhs) => {
                let a = self.eval(lhs, frame)?;
                let b = self.eval(rhs, frame)?;
                eval_binary(op.kind(), &a, &b)
            }
            ExprKind::Apply(func, args) => {
                let fval = self.eval(func, frame)?;

                let base = self.stack.len();
                for a in args.iter() {
                    match self.eval(a, frame) {
                        Ok(arg) => self.stack.push(arg),
                        Err(e) => {
                            // don't strand already-evaluated arguments
                            self.stack.truncate(base);
                            return Err(e);
                        }
                    }
                }
                // calling convention: first argument on top of the stack,
                // same as the reversal Instr::Call performs
                self.stack.reverse(args.len());
                self.stack.push(fval);
                self.apply_value(args.len())?;
                Ok(self.stack.pop())
            }
            ExprKind::Field(obj, field_ident) => {
                let v = self.eval(obj, frame)?;
                field_of(&v, field_ident.name())
            }
            ExprKind::Index(obj, index_expr) => {
                let objv = self.eval(obj, frame)?;
                let idxv = self.eval(index_expr, frame)?;
                index_of(&objv, &idxv)
            }
            ExprKind::Parenthesized(expr) => self.eval_impl(expr, frame, true),
        }
    }

    fn call(&mut self, args: usize) -> Result<(), RuntimeError> {
        if args > 0 {
            self.apply_value(args)
        } else {
            let fval = self.stack.peek();
            let callee = match callee_ref(fval) {
                Some(callee) => callee,
                None => return Ok(()),
            };
            self.stack.pop(); // pop the callee

            callee.invoke(self, &mut 0);
            self.status.clone()
        }
    }

    fn call_ast(&mut self, fval: Val, args: usize) -> Result<Val, RuntimeError> {
        if args > 0 {
            self.apply_value_ast(fval, args)
        } else {
            let callee = match callee(fval) {
                Ok(callee) => callee,
                Err(fval) => return Ok(fval),
            };
            callee.invoke(self, &mut 0);
            self.status.clone()?;
            Ok(self.stack.pop())
        }
    }

    /// Call `fval` with already-evaluated arguments, following the language's
    /// application rules: each callee consumes as many arguments as it wants
    /// (possibly returning a partially-applied function), and leftover
    /// arguments are re-applied to whatever it returned.
    fn apply_value(&mut self, mut args: usize) -> Result<(), RuntimeError> {
        while args > 0 {
            let fval = self.stack.pop();
            let callee = match callee(fval) {
                Ok(callee) => callee,
                Err(fval) => {
                    // don't strand the unconsumed arguments
                    drop(self.stack.drain_top(args));
                    return Err(RuntimeError::NotAFunction(
                        format!("{fval:?} is not callable").into(),
                    ));
                }
            };

            // the callee establishes its own function-local scope (see
            // DynFunction::dyn_call)
            callee.invoke(self, &mut args);
            if self.status.is_err() {
                drop(self.stack.drain_top(args));
                self.status.clone()?;
            }
        }

        Ok(())
    }

    /// Call `fval` with already-evaluated arguments, following the language's
    /// application rules: each callee consumes as many arguments as it wants
    /// (possibly returning a partially-applied function), and leftover
    /// arguments are re-applied to whatever it returned.
    fn apply_value_ast(&mut self, mut fval: Val, mut args: usize) -> Result<Val, RuntimeError> {
        while args > 0 {
            let callee = match callee(fval) {
                Ok(callee) => callee,
                Err(fval) => {
                    // don't strand the unconsumed arguments
                    drop(self.stack.drain_top(args));
                    return Err(RuntimeError::NotAFunction(
                        format!("{fval:?} is not callable").into(),
                    ));
                }
            };

            // the callee establishes its own function-local scope (see
            // DynFunction::dyn_call)
            callee.invoke(self, &mut args);
            fval = self.stack.pop();
            if self.status.is_err() {
                drop(self.stack.drain_top(args));
                self.status.clone()?;
            }
        }

        Ok(fval)
    }

    pub fn exec_stmt(&mut self, stmt: &Stmt, frame: Rc<Frame>) -> Result<Exec, RuntimeError> {
        match stmt.kind() {
            StmtKind::Expr(e) => {
                let val = self.eval_impl(e, &frame, true)?;
                Ok(Exec::Value(val))
            }
            StmtKind::AssignPat { target, value } => {
                let val = self.eval_impl(value, &frame, false)?;
                self.bind_pattern(target, &val, &frame)?;
                Ok(Exec::Value(Val::nil()))
            }
            StmtKind::AssignField {
                target,
                field,
                value,
            } => {
                let objv = self.eval_impl(target, &frame, false)?;
                let val = self.eval_impl(value, &frame, false)?;
                assign_field(objv, field.name(), val)?;
                Ok(Exec::Value(Val::nil()))
            }
            StmtKind::AssignIndex {
                target,
                index,
                value,
            } => {
                let objv = self.eval_impl(target, &frame, false)?;
                let idxv = self.eval_impl(index, &frame, false)?;
                let val = self.eval_impl(value, &frame, false)?;
                assign_index(objv, idxv, val)?;
                Ok(Exec::Value(Val::nil()))
            }
            StmtKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let cv = self.eval_impl(cond, &frame, false)?;
                if !is_falsey(&cv) {
                    self.exec_block(then_branch, frame.clone())
                } else {
                    if let Some(eb) = else_branch {
                        self.exec_block(eb, frame)
                    } else {
                        Ok(Exec::Value(Val::nil()))
                    }
                }
            }
            StmtKind::While {
                cond,
                body,
                else_branch,
            } => loop {
                let cv = self.eval_impl(cond, &frame, false)?;
                if is_falsey(&cv) {
                    if let Some(eb) = else_branch {
                        break self.exec_block(eb, frame.clone());
                    }
                    break Ok(Exec::Value(Val::nil()));
                }
                match self.exec_block(body, frame.clone())? {
                    Exec::Value(_) => continue,
                    Exec::Return(v) => break Ok(Exec::Return(v)),
                    Exec::Continue => continue,
                    Exec::Break => break Ok(Exec::Value(Val::nil())),
                }
            },
            StmtKind::For {
                target,
                iterable,
                body,
                else_branch,
            } => {
                let iter_val = self.eval_impl(iterable, &frame, false)?;
                let values = iter_val.iter()?;

                for value in values {
                    self.bind_pattern(target, &value, &frame)?;

                    match self.exec_block(body, frame.clone())? {
                        Exec::Return(v) => return Ok(Exec::Return(v)),
                        Exec::Break => return Ok(Exec::Value(Val::nil())),
                        Exec::Continue => continue,
                        Exec::Value(_) => continue,
                    }
                }

                if let Some(else_branch) = else_branch {
                    self.exec_block(else_branch, frame.clone())
                } else {
                    Ok(Exec::Value(Val::nil()))
                }
            }
            StmtKind::Return(None) => Ok(Exec::Return(Val::nil())),
            StmtKind::Return(Some(expr)) => {
                let v = self.eval_impl(expr, &frame, false)?;
                Ok(Exec::Return(v))
            }
            StmtKind::Break => Ok(Exec::Break),
            StmtKind::Continue => Ok(Exec::Continue),
            StmtKind::Fn { name, params, body } => {
                let name = self.ctx.string(name.rc_name());
                frame.set_var(name, Val::nil());

                let fval = if self.compile_fns {
                    match vm::compile_fn(
                        self,
                        params.clone(),
                        body,
                        vm::CompileParent::Walked(frame.clone()),
                        &[],
                    ) {
                        Ok((compiled, _schema)) => compiled.into_function(),
                        // constructs the compiler rejects still run on the AST walker
                        Err(_) => fn_from_ast(params.clone(), body.clone(), frame.clone()),
                    }
                } else {
                    fn_from_ast(params.clone(), body.clone(), frame.clone())
                };

                frame.set_var(name, fval);
                Ok(Exec::Value(Val::nil()))
            }
        }
    }

    /// Parse, execute and cache a module. `source` is a block of Raft code
    /// whose tail statement must be `export { .. }` (an `if`/`else` whose
    /// branches all tail-export also qualifies, so a module may export
    /// conditionally). Returns the immutable module object; loading the
    /// same name again returns the cached object without re-executing.
    pub fn load_module(&mut self, name: &str, source: &str) -> Result<Val, RuntimeError> {
        let name_id = self.ctx.string(name);
        if let Some(module) = self.modules.get(&name_id) {
            return Ok(module.clone());
        }
        if self.loading.iter().any(|&module| module == name_id) {
            return Err(RuntimeError::Other(
                format!("circular import of module '{name}'").into(),
            ));
        }

        let tokens =
            raft_ast::lexer::parse_str(source, &raft_ast::lexer::Options::wss()).map_err(|e| {
                RuntimeError::Other(format!("module '{name}': lex error: {e:?}").into())
            })?;
        let mut stream = raft_ast::parser::TokenStream::new(tokens);
        let ast = stream.parse_module().map_err(|e| {
            RuntimeError::Other(format!("module '{name}': parse error: {e:?}").into())
        })?;
        let stmts = ast.rc_stmts();

        // export values are parse-restricted to bare names (shorthand or
        // `key: name`) — this doubles as the set of names the compiled
        // body must keep alive past its own `Return`, which otherwise
        // truncates the stack region ordinary locals live in
        let export_names: Vec<Rc<str>> = ast
            .export()
            .fields()
            .iter()
            .map(|f| match f.value() {
                Some(v) => {
                    let ExprKind::Ident(id) = v.kind() else {
                        unreachable!("export values are parse-restricted to bare identifiers")
                    };
                    id.rc_name()
                }
                None => f.key().rc_name(),
            })
            .collect();

        // the module body runs in a fresh environment: it must not see the
        // importer's locals, and its own bindings must not leak. A module
        // is otherwise an ordinary zero-arg function — no bespoke
        // environment type, just the same compile/walk pipeline every
        // other `fn` goes through.
        self.loading.push(name_id);
        let root = Rc::new(Frame::new());

        let result: Result<FixedHashMap<Rc<str>, Val>, RuntimeError> = 'run: {
            if self.compile_fns {
                if let Ok((compiled, _schema)) = vm::compile_fn(
                    self,
                    Rc::from([]),
                    &stmts[..],
                    vm::CompileParent::Walked(root.clone()),
                    &export_names,
                ) {
                    let own = match vm::run_module(self, &compiled) {
                        Ok(own) => own,
                        Err(e) => break 'run Err(e),
                    };
                    let mut export = FixedHashMap::default();
                    for (f, source) in ast.export().fields().iter().zip(export_names.iter()) {
                        let key = f.key().rc_name();
                        let source_id = self.ctx.string(source.clone());
                        // a name never bound anywhere in the module (a
                        // genuinely unbound export) has no slot at all —
                        // that's an UnboundIdentifier, not a bug
                        let val = compiled
                            .own_names
                            .iter()
                            .position(|&n| n == source_id)
                            .and_then(|slot| own.as_ref().map(|o| o.get_local(SlotId(slot as u32))))
                            .unwrap_or(Val::Uninit);
                        match val.init_or_else(|| RuntimeError::UnboundIdentifier(key.clone())) {
                            Ok(v) => {
                                export.insert(key, v);
                            }
                            Err(e) => break 'run Err(e),
                        }
                    }
                    break 'run Ok(export);
                }
                // compile error: fall back to the AST walker below
            }

            for stmt in stmts.iter() {
                match self.exec_stmt(stmt, root.clone()) {
                    Ok(Exec::Value(_)) => {}
                    Ok(_) => {
                        break 'run Err(RuntimeError::Other(
                            "break/continue/return at module top level".into(),
                        ));
                    }
                    Err(e) => break 'run Err(e),
                }
            }

            let mut export = FixedHashMap::default();
            for (f, source) in ast.export().fields().iter().zip(export_names.iter()) {
                let key = f.key().rc_name();
                let val = root.get_var(source.clone(), self);
                match val.init_or_else(|| RuntimeError::UnboundIdentifier(key.clone())) {
                    Ok(v) => {
                        export.insert(key, v);
                    }
                    Err(e) => break 'run Err(e),
                }
            }
            Ok(export)
        };

        self.loading.pop();

        let export = result?;

        let module = Val::new_module(export);
        self.modules.insert(name_id, module.clone());
        Ok(module)
    }

    /// Execute block of statements. Stops and returns Some(value) if a return happens.
    fn exec_block(&mut self, stmts: &[Stmt], frame: Rc<Frame>) -> Result<Exec, RuntimeError> {
        let mut last_val = Val::nil();
        for s in stmts {
            match self.exec_stmt(s, frame.clone())? {
                Exec::Value(val) => last_val = val,
                Exec::Return(val) => return Ok(Exec::Return(val)),
                Exec::Continue => return Ok(Exec::Continue),
                Exec::Break => return Ok(Exec::Break),
            }
        }
        Ok(Exec::Value(last_val))
    }

    fn bind_pattern(
        &mut self,
        pattern: &Pat,
        val: &Val,
        frame: &Frame,
    ) -> Result<(), RuntimeError> {
        match pattern.kind() {
            PatKind::Ident(id) => {
                if id.name() != "_" {
                    let name = self.ctx.string(id.rc_name());
                    frame.set_var(name, val.clone());
                }
                Ok(())
            }
            PatKind::Atom(a) => match val {
                Val::Atom(av) if av == a.name() => Ok(()),
                _ => Err(RuntimeError::Other("pattern match failed".into())),
            },
            PatKind::Literal(lit) => {
                // compare literal with value
                match (lit, val) {
                    (Lit::Num(nlit), Val::Number(actual)) => {
                        // suffix-aware, exact matching — same rules as the
                        // compiled representation (see vm::NumberPat)
                        if vm::NumberPat::from_literal(nlit).matches(*actual) {
                            Ok(())
                        } else {
                            Err(RuntimeError::Other("pattern match failed".into()))
                        }
                    }
                    (Lit::Str(slit), Val::String(s)) => {
                        if slit.unescape() == &**s {
                            Ok(())
                        } else {
                            Err(RuntimeError::Other("pattern match failed".into()))
                        }
                    }
                    (Lit::Char(clit), Val::Char(c)) => {
                        if clit.unescape() == *c {
                            Ok(())
                        } else {
                            Err(RuntimeError::Other("pattern match failed".into()))
                        }
                    }
                    _ => Err(RuntimeError::Other("pattern match failed".into())),
                }
            }
            PatKind::List(items) => match val {
                Val::Object(o) => match &o.borrow().kind {
                    ObjectKind::List(vec) => {
                        if vec.len() != items.len() {
                            return Err(RuntimeError::Other("pattern match failed".into()));
                        }
                        for (p, v) in items.iter().zip(vec.iter()) {
                            self.bind_pattern(p, v, frame)?;
                        }
                        Ok(())
                    }
                    _ => Err(RuntimeError::Other("pattern match failed".into())),
                },
                _ => Err(RuntimeError::Other("pattern match failed".into())),
            },
            PatKind::Record(fields) => match val {
                Val::Object(o) => match &o.borrow().kind {
                    ObjectKind::Record(map) | ObjectKind::Module(map) => {
                        for f in fields.iter() {
                            let key_id = self.ctx.string(f.key().rc_name());
                            if let Some(v) = map.get(f.key().name()) {
                                match f.pattern() {
                                    None => {
                                        frame.set_var(key_id, v.clone());
                                    }
                                    Some(pattern) => {
                                        self.bind_pattern(pattern, v, frame)?;
                                    }
                                }
                            } else {
                                return Err(RuntimeError::Other("pattern match failed".into()));
                            }
                        }
                        Ok(())
                    }
                    _ => Err(RuntimeError::Other("pattern match failed".into())),
                },
                _ => Err(RuntimeError::Other("pattern match failed".into())),
            },
        }
    }
}

/// Evaluate a number literal in *expression* position to a runtime number,
/// honoring its suffix (`1i` is an integer, `1f`/`1.0`/`1e3` are floats).
/// Pat position interprets literals through [`vm::NumberPat`], where
/// the suffix additionally selects matching strictness.
fn number_value(n: &LitNum) -> Result<Number, RuntimeError> {
    match n.suffix() {
        None | Some("i" | "f") => {}
        Some(suffix) => {
            return Err(RuntimeError::TypeError(
                format!("unsupported number suffix: {}", suffix).into(),
            ));
        }
    }

    if n.has_dot() || n.has_exponent() || n.suffix() == Some("f") {
        let f = n
            .value()
            .parse::<f64>()
            .map_err(|_| RuntimeError::TypeError("invalid float literal".into()))?;
        Ok(Number::Float(f))
    } else {
        let i = n
            .value()
            .parse::<i64>()
            .map_err(|_| RuntimeError::TypeError("invalid integer literal".into()))?;
        Ok(Number::Integer(i))
    }
}

/// Evaluate a literal AST node to a runtime value. Used by the AST walker at
/// evaluation time and by the bytecode compiler at compile time (literals
/// become constants).
fn literal_value(lit: &Lit) -> Result<Val, RuntimeError> {
    match lit {
        Lit::Num(n) => Ok(Val::Number(number_value(n)?)),
        Lit::Str(s) => Ok(Val::String(Rc::from(s.unescape()))),
        Lit::Char(c) => Ok(Val::Char(c.unescape())),
    }
}

/// `value.field` — read a record field.
fn field_of(v: &Val, field: &str) -> Result<Val, RuntimeError> {
    match v {
        Val::Object(h) => {
            let borrowed = h.borrow();
            match &borrowed.kind {
                ObjectKind::Record(map) | ObjectKind::Module(map) => map
                    .get(field)
                    .cloned()
                    .ok_or(RuntimeError::FieldError(field.into())),
                _ => Err(RuntimeError::FieldError(field.into())),
            }
        }
        _ => Err(RuntimeError::FieldError(field.into())),
    }
}

/// `value[index]` — read a list element.
fn index_of(objv: &Val, idxv: &Val) -> Result<Val, RuntimeError> {
    match (objv, idxv) {
        (Val::Object(h), Val::Number(Number::Integer(i))) => {
            let borrowed = h.borrow();
            match &borrowed.kind {
                ObjectKind::List(vec) => match usize::try_from(*i) {
                    Ok(i) => vec.get(i).cloned().ok_or(RuntimeError::IndexError(
                        format!("out of bounds: {}", i).into(),
                    )),
                    Err(_) => Err(RuntimeError::IndexError(
                        format!("negative index: {}", i).into(),
                    )),
                },
                ObjectKind::Record(_) | ObjectKind::Module(_) => Err(RuntimeError::IndexError(
                    "indexing record with integer unsupported".into(),
                )),
            }
        }
        _ => Err(RuntimeError::TypeError("indexing non-heap value".into())),
    }
}

/// `target.field = value` — write a record field.
fn assign_field(objv: Val, field: &str, val: Val) -> Result<(), RuntimeError> {
    match objv {
        Val::Object(o) => {
            let mut borrowed = o.borrow_mut();
            if borrowed.frozen {
                return Err(RuntimeError::Other(
                    "attempt to mutate frozen object".into(),
                ));
            }
            match &mut borrowed.kind {
                ObjectKind::Record(map) => {
                    map.insert(field.into(), val);
                    Ok(())
                }
                _ => Err(RuntimeError::FieldError(field.into())),
            }
        }
        _ => Err(RuntimeError::FieldError(field.into())),
    }
}

/// `target[index] = value` — write a list element.
fn assign_index(objv: Val, idxv: Val, val: Val) -> Result<(), RuntimeError> {
    match (objv, idxv) {
        (Val::Object(o), Val::Number(Number::Integer(i))) => {
            let mut borrowed = o.borrow_mut();
            if borrowed.frozen {
                return Err(RuntimeError::Other(
                    "attempt to mutate frozen object".into(),
                ));
            }
            match &mut borrowed.kind {
                ObjectKind::List(vec) => {
                    if i < 0 {
                        return Err(RuntimeError::IndexError(
                            format!("negative index: {}", i).into(),
                        ));
                    }
                    let ui = usize::try_from(i).map_err(|_| {
                        RuntimeError::IndexError(format!("invalid index: {}", i).into())
                    })?;
                    if ui >= vec.len() {
                        return Err(RuntimeError::IndexError(
                            format!("out of bounds: {}", ui).into(),
                        ));
                    }
                    vec[ui] = val;
                    Ok(())
                }
                _ => Err(RuntimeError::IndexError("indexing non-list object".into())),
            }
        }
        _ => Err(RuntimeError::TypeError(
            "index must be integer and target must be object".into(),
        )),
    }
}

fn eval_unary(op: UnOpKind, a: &Val) -> Result<Val, RuntimeError> {
    use raft_ast::UnOpKind::*;
    match op {
        Not => Ok(a.not()),
        BitNot => a.bit_not(),
        Pos => a.pos(),
        Neg => a.neg(),
    }
}

fn eval_binary(op: BinOpKind, a: &Val, b: &Val) -> Result<Val, RuntimeError> {
    use raft_ast::BinOpKind::*;
    match op {
        BitAnd => a.bit_and(b),
        BitOr => a.bit_or(b),
        BitXor => a.bit_xor(b),
        Shl => a.shl(b),
        Shr => a.shr(b),
        Pow => a.pow(b),
        Mul => a.mul(b),
        Div => a.div(b),
        Add => a.add(b),
        Sub => a.sub(b),
        Eq => Ok(Val::bool_(a.cmp(b) == Some(Ordering::Equal))),
        Ne => Ok(Val::bool_(a.cmp(b) != Some(Ordering::Equal))),
        Lt => Ok(Val::bool_(a.cmp(b) == Some(Ordering::Less))),
        Le => Ok(Val::bool_(matches!(
            a.cmp(b),
            Some(Ordering::Less | Ordering::Equal)
        ))),
        Gt => Ok(Val::bool_(a.cmp(b) == Some(Ordering::Greater))),
        Ge => Ok(Val::bool_(matches!(
            a.cmp(b),
            Some(Ordering::Greater | Ordering::Equal)
        ))),
    }
}

fn callee(val: Val) -> Result<FnVal, Val> {
    match val {
        Val::Fn(f) => Ok(f),
        Val::Object(ref h) => {
            let callee = {
                // a record is callable if it holds a function under the special key "__call"
                let borrowed = h.borrow();
                match &borrowed.kind {
                    ObjectKind::Record(map) => match map.get("__call") {
                        Some(Val::Fn(f)) => Some(f.clone()),
                        _ => None,
                    },
                    _ => None,
                }
            };
            match callee {
                Some(f) => Ok(f),
                None => Err(val),
            }
        }
        _ => Err(val),
    }
}

fn callee_ref(val: &Val) -> Option<FnVal> {
    match val {
        Val::Fn(f) => Some(f.clone()),
        Val::Object(h) => {
            // a record is callable if it holds a function under the special key "__call"
            let borrowed = h.borrow();
            match &borrowed.kind {
                ObjectKind::Record(map) => match map.get("__call") {
                    Some(Val::Fn(f)) => Some(f.clone()),
                    _ => None,
                },
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test sources are plain statement blocks, not modules — no export.
    struct TestBlock {
        stmts: Vec<Stmt>,
    }

    impl TestBlock {
        fn stmts(&self) -> &[Stmt] {
            &self.stmts
        }
    }

    fn ast_from_str(s: &str) -> TestBlock {
        let tokens = raft_ast::lexer::parse_str(s, &raft_ast::lexer::Options::wss()).unwrap();
        let mut stream = raft_ast::parser::TokenStream::new(tokens);
        TestBlock {
            stmts: Stmt::parse_many(&mut stream).unwrap(),
        }
    }

    #[test]
    fn assign_pattern_ident_binds_var() {
        let src = "x = 1";
        let module = ast_from_str(src);
        let mut rt = Runtime::new();
        let frame = Rc::new(Frame::new());
        assert_eq!(
            rt.exec_block(module.stmts(), frame.clone()).unwrap(),
            Exec::Value(Val::nil())
        );
        let v = frame.get_var("x", &mut rt);
        match v {
            Val::Number(Number::Integer(i)) => assert_eq!(i, 1),
            _ => panic!("expected integer"),
        }
    }

    #[test]
    fn assign_pattern_list_binds_vars() {
        let src = "[a, b] = [1, 2]";
        let block = ast_from_str(src);
        let mut rt = Runtime::new();
        let frame = Rc::new(Frame::new());
        assert_eq!(
            rt.exec_block(block.stmts(), frame.clone()).unwrap(),
            Exec::Value(Val::nil())
        );
        let va = frame.get_var("a", &mut rt);
        let vb = frame.get_var("b", &mut rt);
        match va {
            Val::Number(Number::Integer(i)) => assert_eq!(i, 1),
            _ => panic!("expected integer for a"),
        }
        match vb {
            Val::Number(Number::Integer(i)) => assert_eq!(i, 2),
            _ => panic!("expected integer for b"),
        }
    }

    #[test]
    fn literal_pattern_match_success_and_failure() {
        let ok_src = "'a' = 'a'";
        let ok_module = ast_from_str(ok_src);
        let mut rt = Runtime::new();
        let frame = Rc::new(Frame::new());
        assert_eq!(
            rt.exec_block(ok_module.stmts(), frame).unwrap(),
            Exec::Value(Val::nil())
        );

        let bad_src = "'a' = 'b'";
        let bad_module = ast_from_str(bad_src);
        let mut rt2 = Runtime::new();
        let frame = Rc::new(Frame::new());
        assert!(rt2.exec_block(bad_module.stmts(), frame).is_err());
    }

    #[test]
    fn field_and_index_assignment() {
        let src = "obj = { x: 1 }\nobj.x = 5\narr = [0, 1]\narr[0] = 7";
        let module = ast_from_str(src);
        let mut rt = Runtime::new();
        let frame = Rc::new(Frame::new());
        assert_eq!(
            rt.exec_block(module.stmts(), frame.clone()).unwrap(),
            Exec::Value(Val::nil())
        );

        // check obj.x
        let objv = frame.get_var("obj", &mut rt);
        match objv {
            Val::Object(o) => {
                let b = o.borrow();
                match &b.kind {
                    ObjectKind::Record(map) => {
                        let vx = map.get("x").expect("field x present");
                        match vx {
                            Val::Number(Number::Integer(i)) => assert_eq!(*i, 5),
                            _ => panic!("expected integer in obj.x"),
                        }
                    }
                    _ => panic!("obj not record"),
                }
            }
            _ => panic!("obj not object"),
        }

        // check arr[0]
        let arrv = frame.get_var("arr", &mut rt);
        match arrv {
            Val::Object(o) => {
                let b = o.borrow();
                match &b.kind {
                    ObjectKind::List(vec) => match &vec[0] {
                        Val::Number(Number::Integer(i)) => assert_eq!(*i, 7),
                        _ => panic!("expected integer in arr[0]"),
                    },
                    _ => panic!("arr not list"),
                }
            }
            _ => panic!("arr not object"),
        }
    }

    #[test]
    fn return_short_circuits_block() {
        let src = "return 5\nx = 1";
        let module = ast_from_str(src);
        let mut rt = Runtime::new();
        let frame = Rc::new(Frame::new());
        let res = rt.exec_block(module.stmts(), frame.clone()).unwrap();
        match res {
            Exec::Return(Val::Number(Number::Integer(i))) => assert_eq!(i, 5),
            _ => panic!("expected return value"),
        }

        assert!(!frame.get_var("x", &mut rt).is_init());
    }

    #[test]
    fn if_else_execution() {
        let src = "if True:\n    x = 1";
        let module = ast_from_str(src);
        let mut rt = Runtime::new();
        let frame = Rc::new(Frame::new());
        assert_eq!(
            rt.exec_block(module.stmts(), frame.clone()).unwrap(),
            Exec::Value(Val::nil())
        );
        match frame.get_var("x", &mut rt) {
            Val::Number(Number::Integer(i)) => assert_eq!(i, 1),
            _ => panic!("expected integer"),
        }

        let src2 = "if False:\n    x = 1\nelse:\n    x = 2";
        let module2 = ast_from_str(src2);
        let mut rt2 = Runtime::new();
        let frame2 = Rc::new(Frame::new());
        assert_eq!(
            rt2.exec_block(module2.stmts(), frame2.clone()).unwrap(),
            Exec::Value(Val::nil())
        );
        match frame2.get_var("x", &mut rt2) {
            Val::Number(Number::Integer(i)) => assert_eq!(i, 2),
            _ => panic!("expected integer"),
        }
    }

    #[test]
    fn frozen_object_mutation_errors() {
        let src = "r = { x: 1 }";
        let module = ast_from_str(src);
        let mut rt = Runtime::new();
        let frame = Rc::new(Frame::new());
        assert_eq!(
            rt.exec_block(module.stmts(), frame.clone()).unwrap(),
            Exec::Value(Val::nil())
        );
        // freeze object
        let rv = frame.get_var("r", &mut rt);
        match rv {
            Val::Object(o) => {
                o.borrow_mut().freeze();
            }
            _ => panic!("r not object"),
        }
        // attempt mutation
        let mut rt2 = rt; // move ownership
        let bad_src = "r.x = 2";
        let bad_module = ast_from_str(bad_src);
        assert!(rt2.exec_block(bad_module.stmts(), frame.clone()).is_err());
    }

    // Loop/else semantics tests (runtime implementation pending). Marked #[ignore]
    #[test]
    fn while_else_execution() {
        let src = "i = 0\nwhile i < 3:\n    i = i + 1\nelse:\n    flag = 1";
        let module = ast_from_str(src);
        let mut rt = Runtime::new();
        let frame = Rc::new(Frame::new());
        let res = rt.exec_block(module.stmts(), frame.clone()).unwrap();
        assert_eq!(res, Exec::Value(Val::nil()));
        match frame.get_var("i", &mut rt) {
            Val::Number(Number::Integer(i)) => assert_eq!(i, 3),
            _ => panic!("expected integer"),
        }
        assert!(frame.get_var("flag", &mut rt).is_init());
    }

    #[test]
    fn while_else_not_on_break_or_return() {
        let src = "i = 0\nwhile i < 3:\n    if i == 1:\n        break\n    i = i + 1\nelse:\n    flag = 1";
        let module = ast_from_str(src);
        let mut rt = Runtime::new();
        let frame = Rc::new(Frame::new());
        let _ = rt.exec_block(module.stmts(), frame.clone()).unwrap();
        // loop should exit via break and else should NOT execute
        assert!(!frame.get_var("flag", &mut rt).is_init());
    }

    #[test]
    fn for_else_execution() {
        let src = "sum = 0\narr = [1, 2]\nfor a in arr:\n    sum = sum + a\nelse:\n    done = 1";
        let module = ast_from_str(src);
        let mut rt = Runtime::new();
        let frame = Rc::new(Frame::new());
        let _ = rt.exec_block(module.stmts(), frame.clone()).unwrap();
        match frame.get_var("sum", &mut rt) {
            Val::Number(Number::Integer(i)) => assert_eq!(i, 3),
            Val::Number(Number::Float(_)) => panic!("unexpected float"),
            _ => panic!("expected numeric sum"),
        }
        assert!(frame.get_var("done", &mut rt).is_init());
    }

    #[test]
    fn fn_values_carry_arity_hints() {
        let src = "fn add3 a b c:\n    return a + b + c\nadd1 = add3 1 2\n";
        let module = ast_from_str(src);
        let mut rt = Runtime::new();
        let frame = Rc::new(Frame::new());
        rt.exec_block(module.stmts(), frame.clone()).unwrap();

        let Val::Fn(_full) = frame.get_var("add3", &mut rt) else {
            panic!("add3 not a function");
        };

        // two arguments preapplied: one left to go
        let Val::Fn(_partial) = frame.get_var("add1", &mut rt) else {
            panic!("add1 not a function");
        };

        // host registrations: default hint is "takes anything"
        rt.register_function("anything", 0, None, |_rt, _args| Val::nil());
        let Val::Fn(_host) = rt.get_var("anything") else {
            panic!("anything not a function");
        };
    }

    #[test]
    fn call_once_dispatch_for_last_reference() {
        struct Probe {
            shared_calls: Rc<core::cell::Cell<u32>>,
            once_calls: Rc<core::cell::Cell<u32>>,
        }

        impl Function for Probe {
            fn min_args(&self) -> usize {
                1
            }

            fn max_args(&self) -> Option<usize> {
                Some(1)
            }

            fn call(&self, rt: &mut dyn Host, args: usize) {
                debug_assert_eq!(args, 1);
                self.shared_calls.set(self.shared_calls.get() + 1);
                rt.stack_push(Val::nil());
            }

            // `self` by value: the hidden bridge already proved uniqueness
            fn call_once(self, rt: &mut dyn Host, args: usize) {
                debug_assert_eq!(args, 1);
                self.once_calls.set(self.once_calls.get() + 1);
                rt.stack_push(Val::nil());
            }
        }

        let shared_calls = Rc::new(core::cell::Cell::new(0));
        let once_calls = Rc::new(core::cell::Cell::new(0));
        let probe = || {
            Val::Fn(FnVal::new(Probe {
                shared_calls: shared_calls.clone(),
                once_calls: once_calls.clone(),
            }))
        };

        let mut rt = Runtime::new();

        // stored in a variable: the global scope keeps a reference alive
        // through the call, so the shared flavor runs
        rt.set_var("probe", probe());
        let block = ast_from_str("probe 1\n");
        let frame = Rc::new(Frame::new());
        for statement in block.stmts() {
            rt.exec_stmt(statement, frame.clone()).unwrap();
        }
        assert_eq!((shared_calls.get(), once_calls.get()), (1, 0));

        // a temporary function value: the argument list holds the last
        // reference, so the consuming flavor runs
        rt.stack.push(Val::nil());
        rt.stack.push(probe());
        rt.apply_value(1).unwrap();
        assert_eq!((shared_calls.get(), once_calls.get()), (1, 1));
    }

    #[test]
    fn bare_reference_to_positive_arity_fn_yields_the_fn() {
        // statement-position reference to a fn needing arguments must not
        // invoke it; `(f)` evaluates to the function value itself
        let src = "fn inc x:\n    return x + 1\ng = (inc)\nr = g 41\n";
        let block = ast_from_str(src);
        let mut rt = Runtime::new();
        let frame = Rc::new(Frame::new());
        rt.exec_block(block.stmts(), frame.clone()).unwrap();
        match frame.get_var("r", &mut rt) {
            Val::Number(Number::Integer(i)) => assert_eq!(i, 42),
            other => panic!("expected 42, got {other:?}"),
        }
    }

    #[test]
    fn for_else_not_on_break_or_return() {
        let src = "arr = [1, 2, 3]\nfor a in arr:\n    if a == 2:\n        break\nelse:\n    finished = 1";
        let module = ast_from_str(src);
        let mut rt = Runtime::new();
        let frame = Rc::new(Frame::new());
        let _ = rt.exec_block(module.stmts(), frame.clone()).unwrap();
        // break inside loop should prevent else from running
        assert!(!frame.get_var("finished", &mut rt).is_init());
    }
}
