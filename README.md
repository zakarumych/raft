# Raft — Minimal Indentation-Based Interpreter

Raft is a small experimental interpreted language and runtime implemented in Rust. It focuses on a compact, indentation-significant syntax (inspired by Python) and a simple, explicit object model that separates cheap clonable values from heap-backed mutable objects.

This repository contains:
- Parser for expressions, patterns, statements and indentation-based blocks
- AST types and span-tracked nodes
- Runtime with values (`Any`) and heap `Object`s (lists, records)
- Scopes: global and per-call local scope
- External host functions injection
- Opaque values for host-managed data
- Unit tests for parser and runtime

Goals
- Provide a clear, minimal interpreter core suitable for experimentation and teaching
- Keep interpreter implementation small and easy to reason about
- Support pattern assignment, immutable literals, mutable heap objects, and host interop

Quick start

Prerequisites
- Rust toolchain (stable) — https://www.rust-lang.org/tools/install

Build and run tests

```bash
# build
cargo build

# run full test suite
cargo test
```

Project layout

- src/
  - ast.rs       — AST node definitions (Expr, Pattern, Stmt, Span)
  - parse.rs     — Indentation-aware parser and unit tests
  - runtime.rs   — Runtime, `Any`, object heap, evaluator and tests
  - literal.rs   — Literal lexing and helpers
  - main.rs      — optional entrypoint wiring

Language quick reference

Expressions
- Literals: numbers (ints, floats), strings, chars
- Identifiers: names bound in scopes
- Atoms: capitalized symbols (useful for booleans)
- Lists: `[1, 2, 3]`
- Records: `{ x: 1, y: 2 }`
- Unary ops: `!a`, `-n`, `~bits`
- Binary ops: arithmetic, bitwise, comparison, power (`**`), precedence handled in parser
- Field / index access: `obj.field`, `arr[0]`
- Function application: `f a b` (application is space-separated)

Statements
- Expression statement: `foo` on its own line
- Assignment to pattern: `<pattern> = <expr>`
  - Patterns: identifiers, atoms, lists, records, and constant literals (numbers/strings/chars)
  - Pattern assignment performs structural match; identifier components bind to current scope
- Field assignment: `<expr>.<ident> = <expr>` — mutate record fields
- Index assignment: `<expr>[<expr>] = <expr>` — mutate list elements (integer index)
- Return: `return <expr>` — immediate return from current block/call
- If: `if <expr>: <stmt>` or `if <expr>:
    <block>` with optional `else:` following rules (same-line inline else or next non-empty line)

Blocks and indentation
- Blocks are indentation-significant.
- Block's indentation level is determined by the first non-empty line whose indent is greater than the outer block's indent (outer may be None for root).
- A block ends when encountering a line with smaller or equal indent (or EOF).
- Blank lines are ignored when locating block boundaries.
- Tabs counted with tab-stop semantics (configurable in parser code).

Runtime model

Values (enum `Any`):
- Number (integer/float) — cheaply clonable
- Char, String, Atom — cheap values
- Object (Rc<RefCell<Object>>) — lists and records live on heap, mutable by default
  - `Object` has `frozen` flag: when true, mutation attempts fail
  - `mutable` flag currently reserved for future use
- External functions: host-provided callables
- Opaque: host-managed opaque value not inspectable by Raft code

Scopes
- Two-level scoping: global scope (Runtime.global) and optional local scope (Runtime.local)
- Setting variable inside a call (host call or function execution) sets it in the local scope; otherwise, sets global
- Getting variable checks local first then global

Host interop
- Hosts can register external functions via `register_external(name, fn)`; those appear as callables in global scope
- External functions receive `&mut Runtime` and evaluated `&[Any]` arguments and return `Any`
- Opaque values can be stored in `Any::Opaque` and carried through Raft code without inspection

Pattern matching and assignment
- Assignment to a pattern evaluates RHS then attempts to match structure
- Identifier pattern binds value into current scope
- Literal patterns compare values for equality (numbers, strings, chars)
- List/record patterns require structural match and bind subpatterns
- Pattern match failures are runtime errors

Design decisions and caveats
- Parser intentionally strict about structural tokens: `.` must be followed by identifier, `[` must close with `]`, required `:` tokens enforced.
- Expression-to-pattern conversion accepts identifiers, atoms, lists, records, and now literal constants.
- Runtime treats booleans as atoms: `True`/`False` atoms are canonicalized internally.
- Function call semantics run inside a fresh local scope via `call_with_local` (future: implement user functions)

Testing
- Parser and runtime unit tests shipped under each module. Run with `cargo test`.
- Add tests for new features as code evolves.

Contributing
- Open issues for bugs or design discussions.
- Keep changes small and focused; add unit tests for parser and runtime behavior.
- Follow repository style: prefer precise, surgical edits.

License
- (Add license file or statement here. Project currently unlicensed — add LICENSE if you intend to publish.)

Contact / Notes
- Experimental project meant for learning and prototyping interpreter ideas.
- Parser and runtime live in a single binary crate for simplicity; splitting into library + binary is a future step.

Enjoy exploring Raft. Contributions and feedback welcome.
