//! The Raft → Rust transpiler.
//!
//! [`BundleGenerator`] turns parsed [`raft_ast::Module`]s into a complete,
//! standalone cdylib crate: one Rust module file per Raft module, a
//! `lib.rs` carrying the bundle's exactly-sized name/atom tables, the
//! generated `mod support` runtime (emitted verbatim from a template -
//! the crate depends only on `raft-core`), and a `raft_bundle!` init
//! function (see `raft-ffi`) that interns every name into the host and
//! exposes the loaded modules as a record `RawVal`.
//!
//! Scoping is fully static - no runtime frame chains. Every name a
//! function binds becomes either an ordinary Rust local (`let mut v_x =
//! Val::new_uninit();`, hoisted to the function top) or, when some nested
//! `fn` reads it, a field of a per-call `CaptureN` structure
//! (`Rc<CaptureN>`, fields `RefCell<Val>`). Nested Raft `fn`s are Rust
//! `move` closures capturing the `Rc<CaptureN>` handles of their enclosing
//! functions - transpilation granularity is a whole module, so a
//! transpiled function can never capture AST-walker or bytecode state.
//! Reads follow the walker's shadow-with-fallback rule with a compile-time
//! resolved cascade: own binding if initialized, else each enclosing
//! binder's capture field, else the host's global scope.
//!
//! Codegen is three-address style: every expression is lowered to a
//! sequence of `let _tN = ...;` statements, which keeps pattern binds,
//! argument pushes and control flow trivially composable.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::{self, Write};
use std::path::Path;

use raft_ast::{
    BinOpKind, Expr, ExprKind, Lit, LitNum, Module, Pat, PatKind, Stmt, StmtKind, UnOpKind,
};

/// The fixed body of the generated crate's `mod support` (preceded by the
/// generated `NAME_COUNT`/`ATOM_COUNT` consts it references).
const SUPPORT_TEMPLATE: &str = include_str!("bundle_support.rs");

#[derive(Debug)]
pub struct GenError {
    msg: String,
}

impl GenError {
    fn new(msg: impl Into<String>) -> Self {
        GenError { msg: msg.into() }
    }
}

impl fmt::Display for GenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "transpile error: {}", self.msg)
    }
}

impl std::error::Error for GenError {}

/// Append-only string interner: assigns each distinct name a stable `u32`
/// index - the bundle-local index generated code embeds, mapped to a host
/// `StringId`/`AtomId` at bundle init. Only names that can actually reach
/// the host (global-read fallbacks, atoms) get interned, so the generated
/// tables - and the static id arrays sized after them - are exact.
#[derive(Default)]
struct Interner {
    list: Vec<String>,
    map: HashMap<String, u32>,
}

impl Interner {
    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&i) = self.map.get(s) {
            return i;
        }
        let i = self.list.len() as u32;
        self.list.push(s.to_owned());
        self.map.insert(s.to_owned(), i);
        i
    }
}

// ---------------------------------------------------------------------
// Name analysis: which names a body binds, which it reads, and which of
// its bindings some nested fn reads (the captured set). Mirrors the
// walker's `SlotTable`/`fn_outer_names`/`mark_captured` trio.
// ---------------------------------------------------------------------

/// Names bound by a pattern (`_` excluded; record shorthand `{ key }`
/// binds `key`).
fn pat_binds(pat: &Pat, out: &mut BTreeSet<String>) {
    match pat.kind() {
        PatKind::Ident(id) if id.name() == "_" => {}
        PatKind::Ident(id) => {
            out.insert(id.name().to_owned());
        }
        PatKind::List(items) => {
            for p in items.iter() {
                pat_binds(p, out);
            }
        }
        PatKind::Record(fields) => {
            for f in fields.iter() {
                match f.pattern() {
                    Some(p) => pat_binds(p, out),
                    None => {
                        out.insert(f.key().name().to_owned());
                    }
                }
            }
        }
        PatKind::Atom(_) | PatKind::Literal(_) => {}
    }
}

/// Names assigned anywhere in `stmts` (branch bodies included, nested `fn`
/// bodies excluded) - together with parameter binds, a scope's locals.
fn assigned_names(stmts: &[Stmt], out: &mut BTreeSet<String>) {
    for stmt in stmts {
        match stmt.kind() {
            StmtKind::AssignPat { target, .. } => pat_binds(target, out),
            StmtKind::If {
                then_branch,
                else_branch,
                ..
            } => {
                assigned_names(then_branch, out);
                if let Some(eb) = else_branch {
                    assigned_names(eb, out);
                }
            }
            StmtKind::While {
                body, else_branch, ..
            } => {
                assigned_names(body, out);
                if let Some(eb) = else_branch {
                    assigned_names(eb, out);
                }
            }
            StmtKind::For {
                target,
                body,
                else_branch,
                ..
            } => {
                pat_binds(target, out);
                assigned_names(body, out);
                if let Some(eb) = else_branch {
                    assigned_names(eb, out);
                }
            }
            StmtKind::Fn { name, .. } => {
                out.insert(name.name().to_owned());
            }
            StmtKind::Import { binding, .. } => pat_binds(binding, out),
            StmtKind::Expr(_)
            | StmtKind::AssignField { .. }
            | StmtKind::AssignIndex { .. }
            | StmtKind::Return(_)
            | StmtKind::Yield(_)
            | StmtKind::YieldFrom { .. }
            | StmtKind::Break
            | StmtKind::Continue => {}
        }
    }
}

/// Every identifier read reachable from an expression (record shorthand
/// `{ key }` counts as a read of `key`).
fn expr_reads(expr: &Expr, out: &mut BTreeSet<String>) {
    match expr.kind() {
        ExprKind::Ident(id) => {
            out.insert(id.name().to_owned());
        }
        ExprKind::Atom(_) | ExprKind::Literal(_) => {}
        ExprKind::List(items) => {
            for e in items.iter() {
                expr_reads(e, out);
            }
        }
        ExprKind::Record(fields) => {
            for f in fields.iter() {
                match f.value() {
                    Some(v) => expr_reads(v, out),
                    None => {
                        out.insert(f.key().name().to_owned());
                    }
                }
            }
        }
        ExprKind::Unary(_, e) => expr_reads(e, out),
        ExprKind::Binary(a, _, b) => {
            expr_reads(a, out);
            expr_reads(b, out);
        }
        ExprKind::Apply(callee, args) => {
            expr_reads(callee, out);
            for a in args.iter() {
                expr_reads(a, out);
            }
        }
        ExprKind::Field(obj, _) => expr_reads(obj, out),
        ExprKind::Index(obj, idx) => {
            expr_reads(obj, out);
            expr_reads(idx, out);
        }
        ExprKind::Parenthesized(e) => expr_reads(e, out),
        ExprKind::Await(e) => expr_reads(e, out),
    }
}

/// Reads in a statement; a nested `fn` contributes its own free names
/// (recursively), so a name only read by a grandchild still propagates up.
fn stmt_reads(stmt: &Stmt, out: &mut BTreeSet<String>) {
    match stmt.kind() {
        StmtKind::Expr(e) => expr_reads(e, out),
        StmtKind::AssignPat { value, .. } => expr_reads(value, out),
        StmtKind::AssignField { target, value, .. } => {
            expr_reads(target, out);
            expr_reads(value, out);
        }
        StmtKind::AssignIndex {
            target,
            index,
            value,
        } => {
            expr_reads(target, out);
            expr_reads(index, out);
            expr_reads(value, out);
        }
        StmtKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            expr_reads(cond, out);
            for s in then_branch.iter() {
                stmt_reads(s, out);
            }
            if let Some(eb) = else_branch {
                for s in eb.iter() {
                    stmt_reads(s, out);
                }
            }
        }
        StmtKind::While {
            cond,
            body,
            else_branch,
        } => {
            expr_reads(cond, out);
            for s in body.iter() {
                stmt_reads(s, out);
            }
            if let Some(eb) = else_branch {
                for s in eb.iter() {
                    stmt_reads(s, out);
                }
            }
        }
        StmtKind::For {
            iterable,
            body,
            else_branch,
            ..
        } => {
            expr_reads(iterable, out);
            for s in body.iter() {
                stmt_reads(s, out);
            }
            if let Some(eb) = else_branch {
                for s in eb.iter() {
                    stmt_reads(s, out);
                }
            }
        }
        StmtKind::Return(Some(e))
        | StmtKind::Yield(Some(e))
        | StmtKind::YieldFrom { expr: e, .. } => expr_reads(e, out),
        StmtKind::Return(None)
        | StmtKind::Yield(None)
        | StmtKind::Break
        | StmtKind::Continue => {}
        StmtKind::Fn { params, body, .. } => {
            out.extend(fn_free_names(params, body));
        }
        StmtKind::Import { .. } => {}
    }
}

/// Names read inside a `fn` that its parameters don't bind - i.e. must
/// resolve to an enclosing scope. Only params count as bound (walker
/// parity): a body-assigned name may still read through to an enclosing
/// scope on its first, pre-assignment access under the language's
/// shadow-with-fallback rule.
fn fn_free_names(params: &[Pat], body: &[Stmt]) -> BTreeSet<String> {
    let mut bound = BTreeSet::new();
    for p in params {
        pat_binds(p, &mut bound);
    }
    let mut reads = BTreeSet::new();
    for s in body {
        stmt_reads(s, &mut reads);
    }
    reads.retain(|n| !bound.contains(n));
    reads
}

/// The union of free names of every `fn` nested directly in `stmts`
/// (recursing through control flow, not into `fn` bodies -
/// [`fn_free_names`] itself propagates deeper levels). Intersected with a
/// scope's locals, this is its captured set.
fn nested_fn_free_names(stmts: &[Stmt], out: &mut BTreeSet<String>) {
    for stmt in stmts {
        match stmt.kind() {
            StmtKind::Fn { params, body, .. } => {
                out.extend(fn_free_names(params, body));
            }
            StmtKind::If {
                then_branch,
                else_branch,
                ..
            } => {
                nested_fn_free_names(then_branch, out);
                if let Some(eb) = else_branch {
                    nested_fn_free_names(eb, out);
                }
            }
            StmtKind::While {
                body, else_branch, ..
            } => {
                nested_fn_free_names(body, out);
                if let Some(eb) = else_branch {
                    nested_fn_free_names(eb, out);
                }
            }
            StmtKind::For {
                body, else_branch, ..
            } => {
                nested_fn_free_names(body, out);
                if let Some(eb) = else_branch {
                    nested_fn_free_names(eb, out);
                }
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------
// Scopes: the compile-time resolution structure. One per function (index
// 0 is the module body); every local maps to its storage.
// ---------------------------------------------------------------------

/// Where a bound name lives at runtime.
#[derive(Clone)]
enum Storage {
    /// An ordinary Rust local, hoisted to the function top.
    Plain(String),
    /// A field of the scope's capture structure: `(capture var, field)`.
    Field(String, String),
}

struct Scope {
    locals: HashMap<String, Storage>,
    /// Names that are definitely initialized for the scope's whole body -
    /// parameter binds (a call binds every param before the body runs, or
    /// fails). Reads of these skip the uninit-fallback cascade.
    definite: BTreeSet<String>,
    /// Ancestor capture vars this scope's closure must clone into itself
    /// (its own reads plus everything its descendants reach through it),
    /// in first-use order.
    used_caps: Vec<String>,
}

#[derive(Clone, Copy, PartialEq)]
enum BodyKind {
    /// A module's `load` body - `return`/`break`/`continue` at its top
    /// level are runtime errors, matching the walker.
    Module,
    /// A `fn` body - `return` maps to a Rust `return Ok(..)`.
    Fn,
}

/// How generated code inside a generator body names the live host: a
/// fresh, statement-scoped view derived from the co (never a binding held
/// across an await - see `bundle_support`'s `co_host`).
const GEN_HOST: &str = "(&mut support::co_host(&__co))";

/// The spelling of one iteration step over `it`: awaiting sites (`async
/// for` / `async yield from`) go through the awaitable `iter_next_async`
/// (so an async-gen iterable's pending suspends the enclosing body);
/// everywhere else the synchronous `iter_next`, where pending is an
/// error. Callers verify an awaiting site sits in a body that can await.
fn iter_next_call(b: &Body, it: &str, awaits: bool) -> String {
    if awaits {
        format!("support::iter_next_async(&__co, &mut {it}).await?")
    } else {
        format!("support::iter_next({}, &mut {it})?", b.host)
    }
}

/// Emission state for one Rust function body being generated.
struct Body {
    buf: String,
    indent: usize,
    tmp: u32,
    kind: BodyKind,
    /// One entry per enclosing Raft loop; `Some(flag)` when the loop has
    /// an `else` branch and Raft `break` must set its skip flag.
    loops: Vec<Option<String>>,
    /// The spelling of "a `&mut Host` for the current statement" - the
    /// `host` parameter for ordinary bodies, [`GEN_HOST`] inside a
    /// generator's/async fn's async-block body.
    host: &'static str,
    /// What kind of body this is - `yield`/`yield from` only generate in
    /// a `Gen` body, `await` only in an `Async` one; anywhere else
    /// they're transpile errors.
    mode: BodyMode,
}

#[derive(Clone, Copy, PartialEq)]
enum BodyMode {
    Plain,
    Gen,
    Async,
    AsyncGen,
}

impl BodyMode {
    /// `yield`/`yield from` allowed in this body?
    fn yields(self) -> bool {
        matches!(self, BodyMode::Gen | BodyMode::AsyncGen)
    }

    /// `await` allowed in this body (which also means iteration must go
    /// through the awaitable `iter_next_async`, so an async-gen iterable's
    /// pending can suspend the enclosing body)?
    fn awaits(self) -> bool {
        matches!(self, BodyMode::Async | BodyMode::AsyncGen)
    }
}

impl Body {
    fn new(kind: BodyKind, indent: usize) -> Self {
        Body {
            buf: String::new(),
            indent,
            tmp: 0,
            kind,
            loops: Vec::new(),
            host: "host",
            mode: BodyMode::Plain,
        }
    }

    fn line(&mut self, s: impl AsRef<str>) {
        for _ in 0..self.indent {
            self.buf.push_str("    ");
        }
        self.buf.push_str(s.as_ref());
        self.buf.push('\n');
    }

    /// Emit `let _tN = <expr>;`, returning the temp's name.
    fn temp(&mut self, expr: impl AsRef<str>) -> String {
        let name = format!("_t{}", self.tmp);
        self.tmp += 1;
        self.line(format!("let {} = {};", name, expr.as_ref()));
        name
    }

    fn fresh(&mut self, prefix: &str) -> String {
        let name = format!("{}{}", prefix, self.tmp);
        self.tmp += 1;
        name
    }
}

/// Generates a transpiled-bundle crate: add modules one by one, then
/// render `lib.rs`/`Cargo.toml` (or [`write_crate`](BundleGenerator::write_crate)
/// the whole thing).
pub struct BundleGenerator {
    names: Interner,
    atoms: Interner,
    /// `(raft module name, generated module-file source)`.
    modules: Vec<(String, String)>,
}

impl BundleGenerator {
    pub fn new() -> Self {
        BundleGenerator {
            names: Interner::default(),
            atoms: Interner::default(),
            modules: Vec::new(),
        }
    }

    /// Transpile one Raft module into the source of `src/<name>.rs` and
    /// remember it for the bundle. Returns the generated source.
    pub fn add_module(&mut self, name: &str, module: &Module) -> Result<&str, GenError> {
        validate_module_name(name)?;
        if self.modules.iter().any(|(n, _)| n == name) {
            return Err(GenError::new(format!("duplicate module name `{name}`")));
        }

        let mut cx = ModuleCx {
            names: &mut self.names,
            atoms: &mut self.atoms,
            structs: Vec::new(),
            cap_counter: 0,
            scopes: Vec::new(),
        };

        let mut b = Body::new(BodyKind::Module, 1);

        // module scope: its locals live in `load`'s own Rust frame (or the
        // module capture structure, when module fns read them)
        let mut locals = BTreeSet::new();
        assigned_names(module.stmts(), &mut locals);
        cx.open_scope(&mut b, &locals, BTreeSet::new(), module.stmts());
        b.line("let mut last = Val::nil();");

        for stmt in module.stmts() {
            cx.gen_stmt(&mut b, stmt)?;
        }

        // export { .. } - read each exported binding out of the module
        // scope (falling back to host globals, like the walker).
        b.line("let _ = last;");
        b.line("let mut _exports: Vec<(RcStr, Val)> = Vec::new();");
        for field in module.export().fields() {
            let key = field.key().name();
            let source = match field.value() {
                None => key,
                Some(v) => match v.kind() {
                    ExprKind::Ident(id) => id.name(),
                    _ => {
                        return Err(GenError::new(format!(
                            "export value for `{key}` is not a bare identifier"
                        )));
                    }
                },
            };
            let t = cx.gen_read(&mut b, source);
            b.line(format!("_exports.push((RcStr::new({key:?}), {t}));"));
        }
        b.line("Ok(Val::record(_exports))");
        cx.scopes.pop();

        let mut src = String::new();
        let _ = writeln!(src, "//! Transpiled from Raft module `{name}`.");
        let _ = writeln!(src);

        let _ = writeln!(
            src,
            "use alloc::vec::Vec;"
        );
        let _ = writeln!(
            src,
            "use core::{{cell::{{Cell, RefCell, UnsafeCell}}, cmp::Ordering, pin::Pin, future::Future}};"
        );
        let _ = writeln!(
            src,
            "use raft_core::{{\n    AtomId, CoroKind, CoroStatus, Coroutine, Function, Host, Number, Rc, RcCoro, RcFn, RcStr,\n    RuntimeError, Val, ValEnum, ValsIter, ValsIterStep, ffi,\n}};"
        );

        let _ = writeln!(
            src,
            "use crate::support;"
        );

        for def in &cx.structs {
            let _ = writeln!(src);
            src.push_str(def);
        }
        let _ = writeln!(src);
        let _ = writeln!(
            src,
            "pub fn load(host: &mut Host<'_>) -> Result<Val, RuntimeError> {{"
        );
        src.push_str(&b.buf);
        let _ = writeln!(src, "}}");

        self.modules.push((name.to_owned(), src));
        Ok(&self.modules.last().unwrap().1)
    }

    /// Render the generated crate's `lib.rs`: imports, the exactly-sized
    /// bundle name/atom tables, the `mod support` runtime (generated from
    /// the template, with `NAME_COUNT`/`ATOM_COUNT` computed for this
    /// bundle), one `pub mod <name>;` per module, and the `raft_bundle!`
    /// init block that interns all names into the host, loads each module,
    /// and exposes them as a record in `bundle.modules`.
    pub fn generate_lib_rs(&self) -> String {
        let mut src = String::new();
        let _ = writeln!(
            src,
            "//! Generated by raft-rust from {} Raft module(s). Do not edit.",
            self.modules.len()
        );
        // let _ = writeln!(
        //     src,
        //     "#![no_std]"
        // );
        let _ = writeln!(
            src,
            "#![allow(unused_variables, unused_mut, unused_assignments, unused_imports, unused_parens, unreachable_code, dead_code)]"
        );
        let _ = writeln!(
            src,
            "extern crate alloc;"
        );
        let _ = writeln!(src);
        let _ = writeln!(src);
        let _ = writeln!(
            src,
            "/// Every name the bundle can reach the host with (global-read"
        );
        let _ = writeln!(
            src,
            "/// fallbacks). Interned at init; `support`'s exactly-sized static id"
        );
        let _ = writeln!(src, "/// table maps these indices to host `StringId`s.");
        let _ = writeln!(
            src,
            "static NAMES: [&str; {}] = [",
            self.names.list.len()
        );
        for name in &self.names.list {
            let _ = writeln!(src, "    {name:?},");
        }
        let _ = writeln!(src, "];");
        let _ = writeln!(src);
        let _ = writeln!(
            src,
            "/// Every custom atom name, interned to host `AtomId`s at init."
        );
        let _ = writeln!(
            src,
            "static ATOMS: [&str; {}] = [",
            self.atoms.list.len()
        );
        for name in &self.atoms.list {
            let _ = writeln!(src, "    {name:?},");
        }
        let _ = writeln!(src, "];");
        let _ = writeln!(src);
        let _ = writeln!(src, "mod support {{");
        let _ = writeln!(src);
        let _ = writeln!(
            src,
            "    /// Exact interned-name capacity, calculated for this bundle."
        );
        let _ = writeln!(
            src,
            "    pub const NAME_COUNT: usize = {};",
            self.names.list.len()
        );
        let _ = writeln!(
            src,
            "    /// Exact interned-atom capacity, calculated for this bundle."
        );
        let _ = writeln!(
            src,
            "    pub const ATOM_COUNT: usize = {};",
            self.atoms.list.len()
        );
        let _ = writeln!(src);
        for line in SUPPORT_TEMPLATE.lines() {
            if line.is_empty() {
                src.push('\n');
            } else {
                let _ = writeln!(src, "    {line}");
            }
        }
        let _ = writeln!(src, "}}");
        let _ = writeln!(src);
        for (name, _) in &self.modules {
            let _ = writeln!(src, "pub mod {name};");
        }

        struct ModuleNames<'a> {
            modules: &'a [(String, String)],
        }

        impl core::fmt::Display for ModuleNames<'_> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, "[")?;
                for (name, _) in self.modules {
                    write!(f, "\"{}\",", name.escape_default())?;
                }
                write!(f, "]")
            }
        }

        let names = ModuleNames { modules: &self.modules };

        let _ = writeln!(src);
        let _ = writeln!(src, "raft_core::ffi::raft_bundle!({names} (bundle, ffi_host) => {{", );
        let _ = writeln!(src, "    support::init(ffi_host);");
        let _ = writeln!(
            src,
            "    // SAFETY: `ffi_host.raw` is the valid, exclusively-borrowed host the"
        );
        let _ = writeln!(
            src,
            "    // init contract hands us for the duration of this call."
        );
        let _ = writeln!(
            src,
            "    let mut host = unsafe {{ raft_core::Host::from_raw(ffi_host.raw) }};"
        );
        let _ = writeln!(src, "    let mut modules: alloc::vec::Vec<(raft_core::RcStr, raft_core::Val)> = alloc::vec::Vec::new();");
        for (name, _) in &self.modules {
            let _ = writeln!(
                src,
                "    modules.push((raft_core::RcStr::new({name:?}), {name}::load(&mut host)?));"
            );
        }
        let _ = writeln!(src, "    bundle.modules = raft_core::Val::record(modules).into_raw();");
        let _ = writeln!(src, "    Ok::<(), raft_core::RuntimeError>(())");
        let _ = writeln!(src, "}});");
        src
    }

    /// A `Cargo.toml` for the generated cdylib crate. `raft_repo` is the
    /// path (as written into the manifest, so relative to the generated
    /// crate or absolute) of the Raft repository checkout providing
    /// `raft-core`.
    pub fn generate_cargo_toml(&self, crate_name: &str, raft_repo: &str) -> String {
        format!(
            r#"[package]
name = "{crate_name}"
version = "0.1.0"
edition = "2024"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
raft-core = {{ path = {core:?} }}
"#,
            core = format!("{}/core", raft_repo.trim_end_matches('/')),
        )
    }

    /// Write the complete generated crate - `Cargo.toml`, `src/lib.rs`,
    /// and one `src/<name>.rs` per module - under `dir`.
    pub fn write_crate(
        &self,
        dir: &Path,
        crate_name: &str,
        raft_repo: &str,
    ) -> std::io::Result<()> {
        let src_dir = dir.join("src");
        std::fs::create_dir_all(&src_dir)?;
        std::fs::write(
            dir.join("Cargo.toml"),
            self.generate_cargo_toml(crate_name, raft_repo),
        )?;
        std::fs::write(src_dir.join("lib.rs"), self.generate_lib_rs())?;
        for (name, source) in &self.modules {
            std::fs::write(src_dir.join(format!("{name}.rs")), source)?;
        }
        Ok(())
    }

    /// The bundle-wide identifier-name table accumulated so far.
    pub fn names(&self) -> &[String] {
        &self.names.list
    }

    /// The bundle-wide custom-atom-name table accumulated so far.
    pub fn atoms(&self) -> &[String] {
        &self.atoms.list
    }

    /// The transpiled modules accumulated so far, as `(name, source)`.
    pub fn modules(&self) -> impl Iterator<Item = (&str, &str)> {
        self.modules.iter().map(|(n, s)| (n.as_str(), s.as_str()))
    }
}

impl Default for BundleGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-module codegen state: the bundle-wide interners, this module's
/// capture-struct definitions, and the lexical scope stack.
struct ModuleCx<'a> {
    names: &'a mut Interner,
    atoms: &'a mut Interner,
    /// Capture struct definitions, emitted at the module file's top level.
    structs: Vec<String>,
    cap_counter: u32,
    scopes: Vec<Scope>,
}

impl ModuleCx<'_> {
    /// The bundle name index of `name` (interning it), rendered with a
    /// spelling comment.
    fn name_idx(&mut self, name: &str) -> String {
        let idx = self.names.intern(name);
        format!("{idx}u32 /* {name} */")
    }

    /// Open a new scope binding `locals` (params plus body-assigned names,
    /// with `definite` the param-bound subset) for a body of `stmts`:
    /// decides each local's storage - names some nested `fn` reads become
    /// fields of one fresh `CaptureN` structure shared by the whole scope,
    /// everything else an ordinary Rust local - and emits the hoisted
    /// declarations (capture allocation first, then plain locals).
    fn open_scope(
        &mut self,
        b: &mut Body,
        locals: &BTreeSet<String>,
        definite: BTreeSet<String>,
        stmts: &[Stmt],
    ) {
        let mut nested_free = BTreeSet::new();
        nested_fn_free_names(stmts, &mut nested_free);

        let captured: BTreeSet<&String> = locals.intersection(&nested_free).collect();

        let mut map = HashMap::new();
        let mut taken = HashSet::new();
        let cap_var = if captured.is_empty() {
            None
        } else {
            let cap_var = format!("_cap{}", self.cap_counter);
            let struct_name = format!("Capture{}", self.cap_counter);
            self.cap_counter += 1;

            let mut def = String::new();
            let _ = writeln!(
                def,
                "/// Bindings of one scope that nested functions capture."
            );
            let _ = writeln!(def, "pub(crate) struct {struct_name} {{");
            let mut init = format!("let {cap_var} = Rc::new({struct_name} {{ ");
            for name in &captured {
                let field = unique_var(name, &mut taken);
                let _ = writeln!(def, "    {field}: RefCell<Val>,");
                let _ = write!(init, "{field}: RefCell::new(Val::new_uninit()), ");
                map.insert(
                    (*name).clone(),
                    Storage::Field(cap_var.clone(), field.clone()),
                );
            }
            let _ = writeln!(def, "}}");
            init.push_str("});");
            self.structs.push(def);
            b.line(init);
            Some(cap_var)
        };
        let _ = cap_var;

        for name in locals {
            if map.contains_key(name) {
                continue;
            }
            let var = unique_var(name, &mut taken);
            b.line(format!("let mut {var} = Val::new_uninit();"));
            map.insert(name.clone(), Storage::Plain(var));
        }

        self.scopes.push(Scope {
            locals: map,
            definite,
            used_caps: Vec::new(),
        });
    }

    /// Emit the initialized-fallback cascade read of `name` - own binding,
    /// then each enclosing binder's capture field, then the host global -
    /// erroring like the walker when unbound everywhere. A definitely-
    /// initialized binding (a parameter) terminates the cascade early: a
    /// read that ends on one needs neither the global fallback nor the
    /// unbound check. Returns the temp holding the value.
    fn gen_read(&mut self, b: &mut Body, name: &str) -> String {
        let innermost = self.scopes.len() - 1;
        let mut chain: Vec<String> = Vec::new();
        let mut marks: Vec<(usize, String)> = Vec::new();
        let mut definite_end = false;
        for (k, scope) in self.scopes.iter().enumerate().rev() {
            if let Some(storage) = scope.locals.get(name) {
                match storage {
                    Storage::Plain(var) => {
                        debug_assert_eq!(
                            k, innermost,
                            "plain local `{name}` referenced from a nested fn"
                        );
                        chain.push(format!("{var}.clone()"));
                    }
                    Storage::Field(cap_var, field) => {
                        chain.push(format!("{cap_var}.{field}.borrow().clone()"));
                        if k < innermost {
                            marks.push((k, cap_var.clone()));
                        }
                    }
                }
                if scope.definite.contains(name) {
                    definite_end = true;
                    break;
                }
            }
        }
        // every closure level between the owning scope and here must clone
        // the owner's capture handle into itself
        for (k, cap_var) in marks {
            for j in k + 1..=innermost {
                let used = &mut self.scopes[j].used_caps;
                if !used.contains(&cap_var) {
                    used.push(cap_var.clone());
                }
            }
        }

        if definite_end && chain.len() == 1 {
            return b.temp(chain.pop().unwrap());
        }

        let t = if chain.is_empty() {
            let idx = self.name_idx(name);
            let host = b.host;
            b.temp(format!("support::global_get({host}, {idx})"))
        } else {
            let t = b.fresh("_t");
            let mut sources = chain.into_iter();
            b.line(format!("let mut {t} = {};", sources.next().unwrap()));
            for src in sources {
                b.line(format!("if !{t}.is_init() {{ {t} = {src}; }}"));
            }
            if !definite_end {
                let idx = self.name_idx(name);
                let host = b.host;
                b.line(format!(
                    "if !{t}.is_init() {{ ::core::hint::cold_path(); {t} = support::global_get({host}, {idx}); }}"
                ));
            }
            t
        };
        if definite_end {
            return t;
        }
        b.temp(format!(
            "{t}.init_or_else(|| {{ ::core::hint::cold_path(); RuntimeError::UnboundIdentifier(RcStr::new({name:?})) }})?"
        ))
    }

    /// The storage of `name` in the current scope (bind target). Bind
    /// targets are always among the scope's locals by construction.
    fn own_storage(&self, name: &str) -> &Storage {
        self.scopes
            .last()
            .and_then(|s| s.locals.get(name))
            .expect("bind target not in scope locals")
    }

    /// Emit `<storage of name> = <value>;` in the current scope.
    fn gen_write(&mut self, b: &mut Body, name: &str, value: &str) {
        let line = match self.own_storage(name) {
            Storage::Plain(var) => format!("{var} = {value};"),
            Storage::Field(cap_var, field) => {
                format!("*{cap_var}.{field}.borrow_mut() = {value};")
            }
        };
        b.line(line);
    }

    fn gen_stmt(&mut self, b: &mut Body, stmt: &Stmt) -> Result<(), GenError> {
        match stmt.kind() {
            // `await e` / `x = await e` - the two grammatical positions an
            // await can occupy; anywhere else the generic paths reach
            // gen_expr's Await arm, which rejects it.
            StmtKind::Expr(e)
                if b.mode.awaits() && matches!(e.kind(), ExprKind::Await(_)) =>
            {
                let ExprKind::Await(inner) = e.kind() else {
                    unreachable!()
                };
                let t = self.gen_expr(b, inner, false)?;
                b.line(format!("last = support::await_val(&__co, {t})?.await?;"));
            }
            StmtKind::AssignPat { target, value }
                if b.mode.awaits() && matches!(value.kind(), ExprKind::Await(_)) =>
            {
                let ExprKind::Await(inner) = value.kind() else {
                    unreachable!()
                };
                let t = self.gen_expr(b, inner, false)?;
                let v = b.temp(format!("support::await_val(&__co, {t})?.await?"));
                self.gen_bind(b, target, &v)?;
                b.line("last = Val::nil();");
            }
            StmtKind::Expr(e) => {
                let t = self.gen_expr(b, e, true)?;
                b.line(format!("last = {t};"));
            }
            StmtKind::AssignPat { target, value } => {
                let t = self.gen_expr(b, value, false)?;
                self.gen_bind(b, target, &t)?;
                b.line("last = Val::nil();");
            }
            StmtKind::AssignField {
                target,
                field,
                value,
            } => {
                let t_obj = self.gen_expr(b, target, false)?;
                let t_val = self.gen_expr(b, value, false)?;
                b.line(format!(
                    "support::assign_field({t_obj}, {:?}, {t_val})?;",
                    field.name()
                ));
                b.line("last = Val::nil();");
            }
            StmtKind::AssignIndex {
                target,
                index,
                value,
            } => {
                let t_obj = self.gen_expr(b, target, false)?;
                let t_idx = self.gen_expr(b, index, false)?;
                let t_val = self.gen_expr(b, value, false)?;
                b.line(format!(
                    "support::assign_index({t_obj}, {t_idx}, {t_val})?;"
                ));
                b.line("last = Val::nil();");
            }
            StmtKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let t = self.gen_expr(b, cond, false)?;
                b.line(format!("if !{t}.is_falsey() {{"));
                b.indent += 1;
                for s in then_branch.iter() {
                    self.gen_stmt(b, s)?;
                }
                b.indent -= 1;
                b.line("} else {");
                b.indent += 1;
                match else_branch {
                    Some(eb) => {
                        for s in eb.iter() {
                            self.gen_stmt(b, s)?;
                        }
                    }
                    None => b.line("last = Val::nil();"),
                }
                b.indent -= 1;
                b.line("}");
            }
            StmtKind::While {
                cond,
                body,
                else_branch,
            } => {
                let flag = else_branch.as_ref().map(|_| {
                    let flag = b.fresh("_broke");
                    b.line(format!("let mut {flag} = false;"));
                    flag
                });
                b.line("loop {");
                b.indent += 1;
                let t = self.gen_expr(b, cond, false)?;
                b.line(format!("if {t}.is_falsey() {{ break; }}"));
                b.loops.push(flag.clone());
                for s in body.iter() {
                    self.gen_stmt(b, s)?;
                }
                b.loops.pop();
                b.indent -= 1;
                b.line("}");
                // the loop itself yields Nil; an executed `else` block's
                // value takes over (its statements assign `last`)
                b.line("last = Val::nil();");
                if let (Some(eb), Some(flag)) = (else_branch, flag) {
                    b.line(format!("if !{flag} {{"));
                    b.indent += 1;
                    for s in eb.iter() {
                        self.gen_stmt(b, s)?;
                    }
                    b.indent -= 1;
                    b.line("}");
                }
            }
            StmtKind::For {
                target,
                iterable,
                body,
                else_branch,
                awaits,
            } => {
                if *awaits && !b.mode.awaits() {
                    return Err(GenError::new("async for outside of async function"));
                }
                let flag = else_branch.as_ref().map(|_| {
                    let flag = b.fresh("_broke");
                    b.line(format!("let mut {flag} = false;"));
                    flag
                });
                let t_iter = self.gen_expr(b, iterable, false)?;
                let item = b.fresh("_item");
                let it = b.fresh("_it");
                b.line(format!("let mut {it} = support::iter_new(&{t_iter})?;"));
                b.line(format!(
                    "while let Some({item}) = {} {{",
                    iter_next_call(b, &it, *awaits)
                ));
                b.indent += 1;
                b.loops.push(flag.clone());
                self.gen_bind(b, target, &item)?;
                for s in body.iter() {
                    self.gen_stmt(b, s)?;
                }
                b.loops.pop();
                b.indent -= 1;
                b.line("}");
                b.line("last = Val::nil();");
                if let (Some(eb), Some(flag)) = (else_branch, flag) {
                    b.line(format!("if !{flag} {{"));
                    b.indent += 1;
                    for s in eb.iter() {
                        self.gen_stmt(b, s)?;
                    }
                    b.indent -= 1;
                    b.line("}");
                }
            }
            StmtKind::Return(expr) => {
                let t = match expr {
                    Some(e) => self.gen_expr(b, e, false)?,
                    None => b.temp("Val::nil()"),
                };
                match b.kind {
                    BodyKind::Fn => b.line(format!("return Ok({t});")),
                    // side effects of the expression still happen first,
                    // matching the walker (which errors only once the
                    // Return propagates to the module's top level)
                    BodyKind::Module => b.line(
                        "return Err(RuntimeError::Other(\"break/continue/return at module top level\".into()));",
                    ),
                }
            }
            StmtKind::Yield(expr) => {
                if !b.mode.yields() {
                    return Err(GenError::new("yield statement outside of generator"));
                }
                let t = match expr {
                    Some(e) => self.gen_expr(b, e, false)?,
                    None => b.temp("Val::nil()"),
                };
                b.line(format!("support::gen_yield(&__co, {t}).await;"));
                b.line("last = Val::nil();");
            }
            StmtKind::YieldFrom { expr, awaits } => {
                if !b.mode.yields() {
                    return Err(GenError::new("yield statement outside of generator"));
                }
                if *awaits && b.mode != BodyMode::AsyncGen {
                    return Err(GenError::new(
                        "async yield from outside of async generator",
                    ));
                }
                let t = self.gen_expr(b, expr, false)?;
                let it = b.fresh("_it");
                b.line(format!("let mut {it} = support::iter_new(&{t})?;"));
                let yv = b.fresh("_yv");
                b.line(format!(
                    "while let Some({yv}) = {} {{",
                    iter_next_call(b, &it, *awaits)
                ));
                b.indent += 1;
                b.line(format!("support::gen_yield(&__co, {yv}).await;"));
                b.indent -= 1;
                b.line("}");
                b.line("last = Val::nil();");
            }
            StmtKind::Break => match b.loops.last() {
                Some(Some(flag)) => {
                    let flag = flag.clone();
                    b.line(format!("{{ {flag} = true; break; }}"));
                }
                Some(None) => b.line("break;"),
                None => b.line(match b.kind {
                    BodyKind::Fn => "return Err(RuntimeError::Other(\"break statement outside of loop\".into()));",
                    BodyKind::Module => "return Err(RuntimeError::Other(\"break/continue/return at module top level\".into()));",
                }),
            },
            StmtKind::Continue => match b.loops.last() {
                Some(_) => b.line("continue;"),
                None => b.line(match b.kind {
                    BodyKind::Fn => "return Err(RuntimeError::Other(\"continue statement outside of loop\".into()));",
                    BodyKind::Module => "return Err(RuntimeError::Other(\"break/continue/return at module top level\".into()));",
                }),
            },
            StmtKind::Fn { name, cat: raft_ast::FnCat::Normal, params, body } => {
                let t = self.gen_fn_closure(b, params, body)?;
                self.gen_write(b, name.name(), &t);
                b.line("last = Val::nil();");
            }
            StmtKind::Fn { name, cat: raft_ast::FnCat::Generator, params, body } => {
                let t = self.gen_coro_closure(b, params, body, BodyMode::Gen)?;
                self.gen_write(b, name.name(), &t);
                b.line("last = Val::nil();");
            }
            StmtKind::Fn { name, cat: raft_ast::FnCat::Async, params, body } => {
                let t = self.gen_coro_closure(b, params, body, BodyMode::Async)?;
                self.gen_write(b, name.name(), &t);
                b.line("last = Val::nil();");
            }
            StmtKind::Fn { name, cat: raft_ast::FnCat::AsyncGenerator, params, body } => {
                let t = self.gen_coro_closure(b, params, body, BodyMode::AsyncGen)?;
                self.gen_write(b, name.name(), &t);
                b.line("last = Val::nil();");
            }
            // `import` resolves modules via the host runtime's lookup dirs
            // and cdylib linking, neither of which a transpiled bundle -
            // itself a cdylib, with no host runtime of its own - can do at
            // its own init time. Not supported inside transpiled code yet.
            StmtKind::Import { .. } => {
                return Err(GenError::new("import is not supported inside a transpiled bundle"));
            }
        }
        Ok(())
    }

    /// Generate a nested Raft `fn` as a Rust `move` closure wrapped by
    /// `support::fn_val`, cloning into it the capture handles of every
    /// enclosing function it (or anything nested deeper) reaches. Returns
    /// the temp holding the function value.
    fn gen_fn_closure(
        &mut self,
        b: &mut Body,
        params: &[Pat],
        stmts: &[Stmt],
    ) -> Result<String, GenError> {
        let mut param_binds = BTreeSet::new();
        for p in params {
            pat_binds(p, &mut param_binds);
        }
        let mut locals = param_binds.clone();
        assigned_names(stmts, &mut locals);

        let mut fb = Body::new(BodyKind::Fn, b.indent + 2);
        self.open_scope(&mut fb, &locals, param_binds, stmts);

        // pop arguments - first argument on top of the stack
        for param in params.iter() {
            let arg = fb.temp("host.stack().pop()");
            self.gen_bind(&mut fb, param, &arg)?;
        }
        fb.line("let mut last = Val::nil();");
        for s in stmts.iter() {
            self.gen_stmt(&mut fb, s)?;
        }
        fb.line("Ok(last)");

        let scope = self.scopes.pop().expect("fn scope");

        let t = b.fresh("_t");
        b.line(format!("let {t} = {{"));
        b.indent += 1;
        for cap_var in &scope.used_caps {
            b.line(format!("let {cap_var} = Rc::clone(&{cap_var});"));
        }
        b.line(format!(
            "support::fn_val({}usize, move |host: &mut Host<'_>| -> Result<Val, RuntimeError> {{",
            params.len()
        ));
        b.buf.push_str(&fb.buf);
        b.line("})");
        b.indent -= 1;
        b.line("};");
        Ok(t)
    }

    /// Generate a nested Raft `gen fn`/`async fn`/`async gen fn` as a
    /// creation closure: a normal `support::fn_val` function value whose
    /// full application pops its arguments and builds a suspended,
    /// primed coroutine of the matching kind over an `async move` state
    /// machine (see `bundle_support`'s coroutine section). The body is
    /// generated by the same `gen_stmt` machinery as any function - only
    /// the host spelling ([`GEN_HOST`]) and the `yield`/`await`
    /// statements differ - so nested closures, captures and control flow
    /// all work unchanged inside it.
    fn gen_coro_closure(
        &mut self,
        b: &mut Body,
        params: &[Pat],
        stmts: &[Stmt],
        mode: BodyMode,
    ) -> Result<String, GenError> {
        let kind = match mode {
            BodyMode::Gen => "CoroKind::Gen",
            BodyMode::Async => "CoroKind::Async",
            BodyMode::AsyncGen => "CoroKind::AsyncGen",
            BodyMode::Plain => unreachable!("plain fns take gen_fn_closure"),
        };

        let mut param_binds = BTreeSet::new();
        for p in params {
            pat_binds(p, &mut param_binds);
        }
        let mut locals = param_binds.clone();
        assigned_names(stmts, &mut locals);

        // the async body: scope locals (and the capture struct, if any)
        // live inside the future, persisting across suspensions
        let mut fb = Body::new(BodyKind::Fn, b.indent + 3);
        fb.mode = mode;
        fb.host = GEN_HOST;
        self.open_scope(&mut fb, &locals, param_binds, stmts);

        // bind the argument values moved into the future; runs during
        // creation (coro_create polls to the gen_start suspension), so a
        // pattern-match failure is a creation-time error
        for (i, param) in params.iter().enumerate() {
            let arg = format!("_arg{i}");
            self.gen_bind(&mut fb, param, &arg)?;
        }
        fb.line("support::gen_start().await;");
        fb.line("let mut last = Val::nil();");
        for s in stmts.iter() {
            self.gen_stmt(&mut fb, s)?;
        }
        fb.line("Ok(last)");

        let scope = self.scopes.pop().expect("coro fn scope");

        let t = b.fresh("_t");
        b.line(format!("let {t} = {{"));
        b.indent += 1;
        for cap_var in &scope.used_caps {
            b.line(format!("let {cap_var} = Rc::clone(&{cap_var});"));
        }
        b.line(format!(
            "support::fn_val({}usize, move |host: &mut Host<'_>| -> Result<Val, RuntimeError> {{",
            params.len()
        ));
        b.indent += 1;
        // creation: pop the arguments (first argument on top) into temps
        // the async block takes ownership of
        for i in 0..params.len() {
            b.line(format!("let _arg{i} = host.stack().pop();"));
        }
        b.line("let __co = support::gen_co();");
        b.line("let __co_outer = Rc::clone(&__co);");
        // each creation builds a fresh future: re-clone the capture
        // handles this closure holds into the async block
        for cap_var in &scope.used_caps {
            b.line(format!("let {cap_var} = Rc::clone(&{cap_var});"));
        }
        b.line("let __fut = Box::pin(async move {");
        b.buf.push_str(&fb.buf);
        b.line("});");
        b.line(format!(
            "support::coro_create(host, {kind}, __co_outer, __fut)"
        ));
        b.indent -= 1;
        b.line("})");
        b.indent -= 1;
        b.line("};");
        Ok(t)
    }

    /// Lower an expression to statements, returning the temp holding its
    /// value. `call_pos` marks statement/parenthesized position, where a
    /// bare reference to a zero-argument callable invokes it.
    fn gen_expr(&mut self, b: &mut Body, expr: &Expr, call_pos: bool) -> Result<String, GenError> {
        let t = match expr.kind() {
            ExprKind::Literal(lit) => b.temp(literal_expr(lit)?),
            ExprKind::Atom(a) => b.temp(self.atom_expr(a.name())),
            ExprKind::Ident(id) => {
                let t = self.gen_read(b, id.name());
                if call_pos {
                    let host = b.host;
                    b.temp(format!("support::call_bare({host}, {t})?"))
                } else {
                    t
                }
            }
            ExprKind::List(elements) => {
                let temps: Vec<String> = elements
                    .iter()
                    .map(|e| self.gen_expr(b, e, false))
                    .collect::<Result<_, _>>()?;
                b.temp(format!("Val::list(vec![{}])", temps.join(", ")))
            }
            ExprKind::Record(fields) => {
                let mut entries = Vec::with_capacity(fields.len());
                for f in fields.iter() {
                    let key = f.key().name();
                    let t = match f.value() {
                        Some(v) => self.gen_expr(b, v, false)?,
                        // shorthand `{ key }` reads the variable `key`
                        None => self.gen_read(b, key),
                    };
                    entries.push(format!("(RcStr::new({key:?}), {t})"));
                }
                b.temp(format!("Val::record(vec![{}])", entries.join(", ")))
            }
            ExprKind::Unary(op, operand) => {
                let t = self.gen_expr(b, operand, false)?;
                b.temp(match op.kind() {
                    UnOpKind::Not => format!("{t}.not()"),
                    UnOpKind::BitNot => format!("{t}.bit_not()?"),
                    UnOpKind::Pos => format!("{t}.pos()?"),
                    UnOpKind::Neg => format!("{t}.neg()?"),
                })
            }
            ExprKind::Binary(lhs, op, rhs) => {
                let a = self.gen_expr(b, lhs, false)?;
                let c = self.gen_expr(b, rhs, false)?;
                b.temp(binary_expr(op.kind(), &a, &c))
            }
            ExprKind::Apply(func, args) => {
                let t_fn = self.gen_expr(b, func, false)?;
                let temps: Vec<String> = args
                    .iter()
                    .map(|a| self.gen_expr(b, a, false))
                    .collect::<Result<_, _>>()?;
                // calling convention: first argument on top of the stack
                for t in temps.iter().rev() {
                    let host = b.host;
                    b.line(format!("{host}.stack().push({t});"));
                }
                let host = b.host;
                b.temp(format!(
                    "support::apply({host}, {t_fn}, {}usize)?",
                    temps.len()
                ))
            }
            ExprKind::Field(obj, field) => {
                let t = self.gen_expr(b, obj, false)?;
                b.temp(format!("support::field_of(&{t}, {:?})?", field.name()))
            }
            ExprKind::Index(obj, index) => {
                let t = self.gen_expr(b, obj, false)?;
                let ti = self.gen_expr(b, index, false)?;
                b.temp(format!("support::index_of(&{t}, &{ti})?"))
            }
            ExprKind::Parenthesized(inner) => self.gen_expr(b, inner, true)?,
            // awaits generate only at their two statement-level positions
            // (see gen_stmt) - reaching here means misuse
            ExprKind::Await(_) => {
                return Err(GenError::new("await outside of async function"));
            }
        };
        Ok(t)
    }

    fn atom_expr(&mut self, name: &str) -> String {
        match name {
            "Nil" => "Val::nil()".to_owned(),
            "True" => "Val::true_()".to_owned(),
            "False" => "Val::false_()".to_owned(),
            _ => {
                let idx = self.atoms.intern(name);
                format!("support::atom({idx}u32 /* {name} */)")
            }
        }
    }

    /// Lower binding `val` (a temp holding an owned `Val`) against a
    /// pattern - assignments, function parameters and `for` targets all
    /// come through here. Non-binding sub-patterns compare and `return
    /// Err` on mismatch, exactly like the walker's `bind_pattern`.
    fn gen_bind(&mut self, b: &mut Body, pat: &Pat, val: &str) -> Result<(), GenError> {
        match pat.kind() {
            PatKind::Ident(id) if id.name() == "_" => {}
            PatKind::Ident(id) => self.gen_write(b, id.name(), val),
            PatKind::Atom(a) => {
                let atom = self.atom_expr(a.name());
                b.line(format!(
                    "if !support::same(&{val}, &{atom}) {{ return Err(support::pat_fail()); }}"
                ));
            }
            PatKind::Literal(Lit::Num(n)) => {
                b.line(format!(
                    "if !support::match_number(&{}, &{val}) {{ return Err(support::pat_fail()); }}",
                    number_pat_expr(n)
                ));
            }
            PatKind::Literal(Lit::Str(s)) => {
                let inner = b.fresh("_s");
                b.line(format!("match {val}.unpack() {{"));
                b.indent += 1;
                b.line(format!(
                    "ValEnum::String({inner}) if {inner}.as_str() == {:?} => {{}}",
                    s.unescape()
                ));
                b.line("_ => return Err(support::pat_fail()),");
                b.indent -= 1;
                b.line("}");
            }
            PatKind::Literal(Lit::Char(c)) => {
                let inner = b.fresh("_c");
                b.line(format!("match {val}.unpack() {{"));
                b.indent += 1;
                b.line(format!(
                    "ValEnum::Char({inner}) if {inner} == {:?} => {{}}",
                    c.unescape()
                ));
                b.line("_ => return Err(support::pat_fail()),");
                b.indent -= 1;
                b.line("}");
            }
            PatKind::List(items) => {
                let list = b.fresh("_l");
                b.line(format!("match {val}.unpack() {{"));
                b.indent += 1;
                b.line(format!(
                    "ValEnum::List({list}) if {list}.len() == {}usize => {{",
                    items.len()
                ));
                b.indent += 1;
                for (i, p) in items.iter().enumerate() {
                    let elem = b.temp(format!("{list}.get({i}usize).unwrap()"));
                    self.gen_bind(b, p, &elem)?;
                }
                b.indent -= 1;
                b.line("}");
                b.line("_ => return Err(support::pat_fail()),");
                b.indent -= 1;
                b.line("}");
            }
            PatKind::Record(fields) => {
                let rec = b.fresh("_r");
                b.line(format!("match {val}.unpack() {{"));
                b.indent += 1;
                b.line(format!("ValEnum::Record({rec}) => {{"));
                b.indent += 1;
                for f in fields.iter() {
                    let key = f.key().name();
                    let field_val = b.fresh("_f");
                    b.line(format!("match {rec}.get_field({key:?}) {{"));
                    b.indent += 1;
                    b.line(format!("Some({field_val}) => {{"));
                    b.indent += 1;
                    match f.pattern() {
                        // shorthand `{ key }` binds the variable `key`
                        None => self.gen_write(b, key, &field_val),
                        Some(p) => self.gen_bind(b, p, &field_val)?,
                    }
                    b.indent -= 1;
                    b.line("}");
                    b.line("None => return Err(support::pat_fail()),");
                    b.indent -= 1;
                    b.line("}");
                }
                b.indent -= 1;
                b.line("}");
                b.line("_ => return Err(support::pat_fail()),");
                b.indent -= 1;
                b.line("}");
            }
        }
        Ok(())
    }
}

/// A collision-free Rust identifier for a Raft name within one scope.
fn unique_var(name: &str, taken: &mut HashSet<String>) -> String {
    let mut var = String::from("v_");
    for c in name.chars() {
        var.push(if c.is_ascii_alphanumeric() || c == '_' {
            c
        } else {
            '_'
        });
    }
    let mut candidate = var.clone();
    let mut n = 1;
    while !taken.insert(candidate.clone()) {
        candidate = format!("{var}_{n}");
        n += 1;
    }
    candidate
}

/// A number literal in expression position, honoring its suffix - the
/// walker's `number_value`, evaluated at transpile time.
fn number_expr(n: &LitNum) -> Result<String, GenError> {
    match n.suffix() {
        None | Some("i" | "f") => {}
        Some(suffix) => {
            return Err(GenError::new(format!(
                "unsupported number suffix: {suffix}"
            )));
        }
    }

    if n.has_dot() || n.has_exponent() || n.suffix() == Some("f") {
        let f = n
            .value()
            .parse::<f64>()
            .map_err(|_| GenError::new(format!("invalid float literal: {}", n.value())))?;
        Ok(float_expr(f))
    } else {
        let i = n
            .value()
            .parse::<i64>()
            .map_err(|_| GenError::new(format!("invalid integer literal: {}", n.value())))?;
        Ok(format!("Val::new_int({i}i64)"))
    }
}

/// Emit an exact f64: bit-pattern round trip, valid Rust for every value
/// (`{:?}` would render NaN/infinities unparseably).
fn float_expr(f: f64) -> String {
    format!(
        "Val::new_float(f64::from_bits(0x{:016x}u64) /* {f} */)",
        f.to_bits()
    )
}

fn literal_expr(lit: &Lit) -> Result<String, GenError> {
    match lit {
        Lit::Num(n) => number_expr(n),
        Lit::Str(s) => Ok(format!("Val::string({:?})", s.unescape())),
        Lit::Char(c) => Ok(format!("Val::new_char({:?})", c.unescape())),
    }
}

/// A number literal in pattern position - the walker/VM's
/// `NumberPat::from_literal`, evaluated at transpile time.
fn number_pat_expr(n: &LitNum) -> String {
    if n.has_dot() || n.has_exponent() || n.suffix() == Some("f") {
        match n.value().parse::<f64>() {
            Ok(f) => format!(
                "support::NumberPat::Float(f64::from_bits(0x{:016x}u64) /* {f} */)",
                f.to_bits()
            ),
            Err(_) => "support::NumberPat::Never".to_owned(),
        }
    } else {
        match n.value().parse::<i64>() {
            Ok(i) if n.suffix() == Some("i") => format!("support::NumberPat::Integer({i}i64)"),
            Ok(i) => format!("support::NumberPat::Numeric({i}i64)"),
            Err(_) => "support::NumberPat::Never".to_owned(),
        }
    }
}

fn binary_expr(op: BinOpKind, a: &str, b: &str) -> String {
    use BinOpKind::*;
    match op {
        BitAnd => format!("{a}.bit_and(&{b})?"),
        BitOr => format!("{a}.bit_or(&{b})?"),
        BitXor => format!("{a}.bit_xor(&{b})?"),
        Shl => format!("{a}.shl(&{b})?"),
        Shr => format!("{a}.shr(&{b})?"),
        Pow => format!("{a}.pow(&{b})?"),
        Mul => format!("{a}.mul(&{b})?"),
        Div => format!("{a}.div(&{b})?"),
        Add => format!("{a}.add(&{b})?"),
        Sub => format!("{a}.sub(&{b})?"),
        Eq => format!("support::eq(&{a}, &{b})"),
        Ne => format!("support::ne(&{a}, &{b})"),
        Lt => format!("support::lt(&{a}, &{b})"),
        Le => format!("support::le(&{a}, &{b})"),
        Gt => format!("support::gt(&{a}, &{b})"),
        Ge => format!("support::ge(&{a}, &{b})"),
    }
}

fn validate_module_name(name: &str) -> Result<(), GenError> {
    let mut chars = name.chars();
    let valid_start = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
    let valid_rest = chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    const RESERVED: &[&str] = &[
        "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum",
        "extern", "false", "fn", "for", "gen", "if", "impl", "in", "let", "loop", "match", "mod",
        "move", "mut", "pub", "ref", "return", "self", "static", "struct", "super", "support",
        "trait", "true", "type", "unsafe", "use", "where", "while",
    ];
    if !valid_start || !valid_rest || RESERVED.contains(&name) {
        return Err(GenError::new(format!(
            "module name `{name}` is not a valid Rust module identifier"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_module(source: &str) -> Module {
        let tokens = raft_ast::lexer::parse_str(source, &raft_ast::lexer::Options::wss()).unwrap();
        let mut stream = raft_ast::parser::TokenStream::new(tokens);
        stream.parse_module().unwrap()
    }

    #[test]
    fn generates_module_and_bundle() {
        let module = parse_module(
            "fn add a b:\n    return a + b\nfn inc x:\n    return add x 1\nbase = inc 41\nexport { add, inc, base }\n",
        );
        let mut generator = BundleGenerator::new();
        let src = generator.add_module("math", &module).unwrap().to_owned();

        assert!(src.contains("pub fn load(host: &mut Host<'_>) -> Result<Val, RuntimeError>"));
        // `add` is read by a nested fn (`inc`) → module capture field
        assert!(src.contains("pub(crate) struct Capture0 {"));
        assert!(src.contains("v_add: RefCell<Val>,"));
        // `inc`/`base` are read by no nested fn → ordinary Rust locals
        assert!(src.contains("let mut v_inc = Val::new_uninit();"));
        assert!(src.contains("let mut v_base = Val::new_uninit();"));
        // fns become move closures wrapped by support::fn_val
        assert!(src.contains("let _cap0 = Rc::clone(&_cap0);"));
        assert!(
            src.contains(
                "support::fn_val(2usize, move |host: &mut Host<'_>| -> Result<Val, RuntimeError> {"
            )
        );
        assert!(src.contains("support::apply(host,"));
        assert!(src.contains("Ok(Val::record(_exports))"));

        let lib = generator.generate_lib_rs();
        // exact table sizes and generated support consts
        assert!(lib.contains(&format!(
            "static NAMES: [&str; {}] = [",
            generator.names().len()
        )));
        assert!(lib.contains(&format!(
            "pub const NAME_COUNT: usize = {};",
            generator.names().len()
        )));
        assert!(lib.contains("pub const ATOM_COUNT: usize = 0;"));
        // support runtime generated into lib.rs
        assert!(lib.contains("mod support {"));
        assert!(lib.contains("pub struct TranspiledFn<F>"));
        assert!(lib.contains("pub mod math;"));
        assert!(lib.contains("raft_core::ffi::raft_bundle!([\"math\",] (bundle, ffi_host) => {"));
        assert!(lib.contains("support::init(ffi_host);"));
        assert!(lib.contains("modules.push((RcStr::new(\"math\"), math::load(&mut host)?));"));
        assert!(lib.contains("bundle.modules = Val::record(modules).into_ffi();"));
    }

    #[test]
    fn uncaptured_fn_locals_are_plain_rust_bindings() {
        let module = parse_module(
            "fn area w h:\n    size = w * h\n    return size\nexport { area }\n",
        );
        let mut generator = BundleGenerator::new();
        let src = generator.add_module("plain", &module).unwrap().to_owned();

        // `area`'s params and local are ordinary Rust locals, and nothing
        // in this module is read by a nested fn - no capture struct at all
        assert!(src.contains("let mut v_w = Val::new_uninit();"));
        assert!(src.contains("let mut v_h = Val::new_uninit();"));
        assert!(src.contains("let mut v_size = Val::new_uninit();"));
        assert_eq!(src.matches("struct Capture").count(), 0);
    }

    #[test]
    fn nested_fns_capture_through_parent_capture_struct() {
        let module = parse_module(
            "fn make_counter start:\n    count = start\n    fn step by:\n        count = count + by\n        return count\n    return step\nexport { make_counter }\n",
        );
        let mut generator = BundleGenerator::new();
        let src = generator
            .add_module("counter", &module)
            .unwrap()
            .to_owned();

        // make_counter's scope captures `count` (read by `step`) - all its
        // captured bindings live in the one Capture0 struct
        assert!(src.contains("pub(crate) struct Capture0 {"));
        assert!(src.contains("v_count: RefCell<Val>,"));
        assert_eq!(src.matches("struct Capture").count(), 1);
        // step's own `count` is a plain local, cascading to the capture field
        assert!(src.contains("if !_t"));
        assert!(src.contains(".borrow().clone(); }"));
        // step clones its parent's capture handle into its closure
        assert!(src.contains("let _cap0 = Rc::clone(&_cap0);"));
    }

    #[test]
    fn atoms_are_collected_and_builtins_are_not() {
        let module = parse_module(
            "state = Running\nok = True\nnothing = Nil\nexport { state, ok, nothing }\n",
        );
        let mut generator = BundleGenerator::new();
        let src = generator.add_module("atoms", &module).unwrap().to_owned();

        assert!(src.contains("support::atom(0u32 /* Running */)"));
        assert!(src.contains("Val::true_()"));
        assert!(src.contains("Val::nil()"));
        assert_eq!(generator.atoms(), &["Running".to_owned()]);
    }

    #[test]
    fn control_flow_and_patterns_generate() {
        let module = parse_module(
            "fn classify n:\n    if n < 0:\n        return Negative\n    total = 0\n    while n > 0:\n        total = total + n\n        n = n - 1\n    else:\n        total = total\n    return total\nfn first pair:\n    [head, _] = pair\n    return head\nfn keyed rec:\n    { value } = rec\n    return value\nexport { classify, first, keyed }\n",
        );
        let mut generator = BundleGenerator::new();
        let src = generator.add_module("flow", &module).unwrap().to_owned();

        assert!(src.contains("loop {"));
        assert!(src.contains("is_falsey() { break; }"));
        assert!(src.contains("ValEnum::List("));
        assert!(src.contains("ValEnum::Record("));
        assert!(src.contains("get_field(\"value\")"));
    }

    #[test]
    fn generators_transpile_to_async_state_machines() {
        let module = parse_module(
            "gen fn count n:\n    i = 0\n    while i < n:\n        yield i\n        i = i + 1\nfn sum n:\n    total = 0\n    for x in (count n):\n        total = total + x\n    return total\nexport { count, sum }\n",
        );
        let mut generator = BundleGenerator::new();
        let src = generator.add_module("gens", &module).unwrap().to_owned();

        // creation closure builds and primes an async state machine
        assert!(src.contains("let __co = support::gen_co();"));
        assert!(src.contains("let __fut = Box::pin(async move {"));
        assert!(src.contains("support::gen_start().await;"));
        assert!(src.contains("support::coro_create(host, CoroKind::Gen, __co_outer, __fut)"));
        // yield awaits the poll-once future
        assert!(src.contains("support::gen_yield(&__co, "));
        // generator bodies re-derive the host per statement, never holding
        // one across an await
        assert!(src.contains("(&mut support::co_host(&__co))"));
        // for loops go through the host-aware iteration protocol
        assert!(src.contains("support::iter_new(&"));
        assert!(src.contains("support::iter_next(host, &mut "));
    }

    #[test]
    fn yield_from_transpiles_to_a_delegation_loop() {
        let module = parse_module(
            "gen fn inner n:\n    yield n\ngen fn outer n:\n    yield from (inner n)\nexport { outer }\n",
        );
        let mut generator = BundleGenerator::new();
        let src = generator.add_module("delegate", &module).unwrap().to_owned();

        assert!(src.contains("support::iter_new(&"));
        assert!(src.contains("while let Some(_yv"));
        assert!(src.contains(").await;"));

        // yield from outside a generator is a transpile error
        let bad = parse_module("fn f xs:\n    yield from xs\nexport { f }\n");
        let mut g2 = BundleGenerator::new();
        assert!(g2.add_module("bad", &bad).is_err());
    }

    #[test]
    fn async_fns_transpile_to_awaitable_state_machines() {
        let module = parse_module(
            "async fn add_async a b:\n    return a + b\nasync fn compose x:\n    y = await (add_async x 1)\n    return y\nexport { compose }\n",
        );
        let mut generator = BundleGenerator::new();
        let src = generator.add_module("asyncs", &module).unwrap().to_owned();

        // creation closure builds and primes an async state machine
        assert!(src.contains("support::coro_create(host, CoroKind::Async, __co_outer, __fut)"));
        assert!(src.contains("support::gen_start().await;"));
        // await is a real await on the polled Val::Async
        assert!(src.contains("support::await_val(&__co, "));
        assert!(src.contains(")?.await?"));
    }

    #[test]
    fn async_gen_fns_transpile_with_awaitable_iteration() {
        let module = parse_module(
            "async gen fn agen n:\n    i = 0\n    while i < n:\n        v = await (leaf i)\n        yield v\n        i = i + 1\nasync fn consume n:\n    s = 0\n    async for x in (agen n):\n        s = s + x\n    return s\nexport { agen, consume }\n",
        );
        let mut generator = BundleGenerator::new();
        let src = generator.add_module("agens", &module).unwrap().to_owned();

        // the async gen body both yields and awaits inside one state machine
        assert!(src.contains("support::coro_create(host, CoroKind::AsyncGen, __co_outer, __fut)"));
        assert!(src.contains("support::gen_yield(&__co, "));
        assert!(src.contains("support::await_val(&__co, "));
        // the consuming async fn iterates through the awaitable step, so
        // the async gen's pending suspends it
        assert!(src.contains("support::iter_next_async(&__co, &mut "));
        assert!(src.contains(").await? {"));

        // `async for` outside an async body is a transpile error
        let bad = parse_module("fn f xs:\n    async for x in xs:\n        x\nexport { f }\n");
        let mut g2 = BundleGenerator::new();
        assert!(g2.add_module("bad", &bad).is_err());

        // `async yield from` outside an async gen body is a transpile error
        let bad = parse_module("gen fn g xs:\n    async yield from xs\nexport { g }\n");
        let mut g3 = BundleGenerator::new();
        assert!(g3.add_module("bad2", &bad).is_err());
    }

    #[test]
    fn await_outside_async_is_a_transpile_error() {
        let module = parse_module("fn f x:\n    await x\nexport { f }\n");
        let mut generator = BundleGenerator::new();
        assert!(generator.add_module("bad", &module).is_err());

        // await inside a generator body is also rejected
        let module = parse_module("gen fn g x:\n    await x\nexport { g }\n");
        let mut generator = BundleGenerator::new();
        assert!(generator.add_module("bad2", &module).is_err());
    }

    #[test]
    fn yield_outside_generator_is_a_transpile_error() {
        let module = parse_module("fn f x:\n    yield x\nexport { f }\n");
        let mut generator = BundleGenerator::new();
        assert!(generator.add_module("bad", &module).is_err());

        let module = parse_module("yield 1\nexport { }\n");
        let mut generator = BundleGenerator::new();
        assert!(generator.add_module("bad2", &module).is_err());
    }

    #[test]
    fn duplicate_and_invalid_module_names_error() {
        let module = parse_module("x = 1\nexport { x }\n");
        let mut generator = BundleGenerator::new();
        generator.add_module("a", &module).unwrap();
        assert!(generator.add_module("a", &module).is_err());
        assert!(generator.add_module("mod", &module).is_err());
        assert!(generator.add_module("support", &module).is_err());
        assert!(generator.add_module("9lives", &module).is_err());
    }
}
