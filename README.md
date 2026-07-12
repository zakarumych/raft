# Raft — a simplicity-first programming language

Raft is a small, dynamically-typed scripting language implemented in Rust.
It favors a minimal syntax and a small set of orthogonal features over a large
surface area — the goal is a language that is easy to read, easy to embed,
and easy to reason about.

> **Status: early / work in progress.** The lexer, parser, tree-walking
> runtime and REPL are functional, including user-defined functions with
> currying, partial application and lexical closures. The language is
> still missing pieces you'd expect from a "complete" language (a standard
> library — see [Roadmap](#roadmap)). APIs and syntax may change without
> notice.

## Project layout

This is a Cargo workspace made up of four crates:

| Crate               | Path        | Description                                                                 |
| ------------------- | ----------- | ---------------------------------------------------------------------------- |
| `raft-lexer`        | `lexer/`    | `no_std`-friendly tokenizer: idents, atoms, numbers, chars, strings, comments, punctuation and delimiter groups (including indentation-based blocks). |
| `raft-ast`          | `ast/`      | AST types plus a recursive-descent parser built on top of the token stream. |
| `raft-runtime`      | `runtime/`  | A tree-walking interpreter that evaluates the AST, plus a stack-based bytecode VM (`vm` module) functions can be compiled to. |
| `raft-repl`         | `repl/`     | Line-by-line REPL wiring the lexer, parser and runtime together.           |

`raft-lexer`, `raft-ast` and `raft-runtime` support an optional `std`
feature (enabled by default) so the front-end can, in principle, run in
`no_std` environments.

## Building & testing

```sh
# Build everything
cargo build

# Run the test suite for lexer and parser
cargo test

# Run the REPL binary
cargo run -p raft-repl
```

## Language tour

The following describes the syntax and semantics currently supported by the
parser and runtime.

### Comments

```raft
// a line comment
/* a block comment */
```

### Literals

```raft
42          // integer
0x2A        // hexadecimal
0o52        // octal
0b101010    // binary
1_000_000   // underscores allowed as digit separators
3.14        // float
6.02e23     // float with exponent
1i          // explicit integer suffix
1.0f        // explicit float suffix

'a'         // char
'\n'        // char with escape

"hello"     // string
"line\n"    // string with escapes
```

### Identifiers vs. atoms

Identifiers starting with a lowercase letter or `_` are ordinary variable
names. Identifiers starting with an uppercase letter are **atoms** — inert,
self-contained symbols equal only to themselves.
The atoms `True`, `False` and `Nil` are used as the language's booleans and
"no value" marker.

```raft
x = 1
Status      // an atom
ok = True
```

### Collections

Lists and records are the two built-in heap-allocated collection types:

```raft
xs = [1, 2, 3]

point = { x: 1, y: 2 }

// shorthand field: takes the value from a variable of the same name
name = "Ada"
person = { name, age: 36 }
```

Access fields and indices with `.` and `[]`:

```raft
point.x
xs[0]
point.x = 10
xs[0] = 10
```

### Operators

From tightest to loosest binding:

| Precedence   | Operators                     | Associativity  |
| ------------ | ------------------------------ | -------------- |
| 5 (tightest) | `&` `\|` `^` `<<` `>>`         | left-to-right  |
| 4            | `**`                           | right-to-left  |
| 3            | `*` `/`                        | left-to-right  |
| 2            | `+` `-`                        | left-to-right  |
| 1 (loosest)  | `==` `!=` `<` `>` `<=` `>=`     | left-to-right  |

Unary operators: `!` (logical not), `~` (bitwise not), `+` (positive identity),
`-` (negate).

### Function application

Function calls use juxtaposition rather than parentheses (parentheses are
only needed for grouping):

```raft
print "hello"
add 1 2
```

Host-registered functions and user-defined functions (see
[Functions](#functions)) are both first-class `Fn` values and are called the
same way.

A function called with fewer arguments than it takes is not an error —
it returns a new function that captures the arguments seen so far and waits for the rest;
supplying more than one function's arity consumes what it needs and re-applies the
leftover arguments to whatever it returned:

```raft
fn add a b:
    return a + b

add1 = add 1   // partial application: a function of one argument
add1 2         // 3
add 1 2        // 3, same as above
```

A function that takes zero arguments can't appear in a juxtaposition (there
are no arguments to juxtapose), so referencing it bare — as a whole
statement, or written in parentheses — calls it instead of yielding the
function value itself:

```raft
quit           // calls quit with no arguments
(quit)         // same
f = quit       // does NOT call quit — plain juxtaposition/assignment RHS
               //   position doesn't trigger the call
```

### Control flow

Blocks are introduced with `:` and are either a single statement on the same
line, or an indented block on the following lines (indentation is
significant, similar to Python):

```raft
if x > 0:
    print "positive"
else if x < 0:
    print "negative"
else:
    print "zero"

while x > 0:
    x = x - 1
else:
    print "done"

for item in xs:
    print item
else:
    print "no more items"
```

`break`, `continue` and `return` are supported inside the relevant blocks.
`while`/`for` loops also support a trailing `else` clause, which runs when
the loop completes without hitting a `break`.

### Functions

Functions are declared with `fn`, using the same inline-or-indented block
syntax as `if`/`while`/`for`:

```raft
fn add a b:
    return a + b

fn greet name:
    print "hello" name
```

`return` is optional — a function with no explicit `return` yields the
value of its last executed statement (nil if the body has none). Parameters
are patterns, so they can destructure their argument directly:

```raft
fn dist { x, y }:
    x * x + y * y
```

Calling a function evaluates its body in a fresh local scope. A nested `fn`
can additionally close over the locals of the `fn` (or module — see
[Modules](#modules)) it's defined in, at any depth:

```raft
fn make_adder n:
    fn add x:
        return x + n
    return add

add5 = make_adder 5
add10 = make_adder 10
add5 3      // 8
add10 3     // 13
```

Each call to `make_adder` produces an independent capture, so `add5` and
`add10` don't share state. Assignment always writes to the current
function's own scope, even for a name also bound in an enclosing scope —
`n = n + 1` reads the outer `n` on the right-hand side but binds the result
as a new local, it never mutates the outer variable. This means a captured
name works as a read-only snapshot from the closure's perspective; there is
no way for a closure to mutate a variable back in its defining scope. See
[Function application](#function-application) for how currying and partial
application work when calling.

### Pat matching & destructuring

Assignment targets can be simple identifiers, or patterns that destructure
lists and records:

```raft
[a, b, c] = xs
{ x, y } = point
```

Patterns can also match literal values and atoms, which is used to
accept/reject values during pattern-matching assignment.

The identifier `_` is a wildcard: it matches anything and binds nothing —
in parameters, destructuring positions, `for` targets, and as a plain
assignment target to discard a value. (It is not a valid record key, in
either expressions or patterns.)

```raft
fn snd _ b:
    return b

[_, m, _] = xs      // keep only the middle element
_ = do_something    // evaluate and discard
```

Number literals match **exactly** (no tolerance; NaN matches NaN and only
NaN, `-0.0` matches `0.0`, infinities match by sign), and their suffix
selects how strictly the numeric *type* is checked, so suffixed literals
double as type discriminators:

```raft
1 = n       // matches integer 1 or float 1.0
1i = n      // matches integer 1 only
1f = n      // matches float 1.0 only (same for 1.0 and 1e0)
```

### No user-defined types

Raft has no `class`/`struct`/`enum` declarations, by design — records
tagged with an atom field stand in for both product and sum types:

```raft
circle = { kind: Circle, radius: 2 }
square = { kind: Square, side: 3 }
```

Dispatching on the tag today means inspecting it yourself (e.g. an `if`
chain, or a pattern-matching assignment against `kind`). Declaring multiple
variants of the same function with different parameter patterns/bodies
(pattern-matched overloads, as in Elixir/Erlang or ML-family languages) is
not yet implemented, but is a planned mechanism for this kind of
tag-directed control flow — see [Roadmap](#roadmap).

### Mutability

Lists and records are mutable heap objects by default and can be frozen
(from host code) to prevent further mutation.

### Modules

A module is a file of Raft code whose **tail statement must be
`export { .. }`** (using record syntax, shorthand included). Importing a
module executes its code once in a fresh environment and turns the export
into an immutable, record-shaped module object; repeated imports return
the cached object. Because module bindings can never change after the
load, functions defined in a module capture its environment — they keep
seeing the module's values and helper functions wherever they are called:

```raft
// geometry.raft
pi = 3
fn sq x:
    return x * x
fn area r:
    return pi * (sq r)
export { pi, sq, area }
```

```raft
geo = import "geometry"
geo.area 5          // 75
{ sq } = geo        // record patterns destructure modules
```

The export must be in tail position, which permits conditional exports:
an `if`/`else` whose branches all end in an `export` is a valid module
tail. Module code reads the importer's globals (so host functions like
`print` are available), but its own bindings never leak out; `_` aside,
whatever is not exported is private. Circular imports are an error.

`import` itself is a host function (the bundled REPL maps
`import "name"` to loading `name.raft`); embedders wire their own source
lookup and call `Runtime::load_module(name, source)`.

### Embedding host functions

Every callable — `fn`-defined (AST-walked or bytecode-compiled), partially
applied, or host-provided — is an `Any::Fn(FnValue)`. An `FnValue` pairs a
`Function` implementor with an argument-count hint: a full application
takes at least `min_args` and at most `max_args` (`None` = unbounded)
arguments; how many a call actually consumes is somewhere in between and
is reported back by the call itself. The runtime uses the lower bound to
build partial applications *before* calling, so implementations never see
an underfull argument list.

The `Function` trait has two call flavors:

- `call(&self, rt, args) -> (Any, usize)` — mandatory; returns the result
  and how many arguments were consumed (consuming fewer than given makes
  the runtime re-apply the leftovers to the returned value).
- `call_once(self, rt, args)` — optional, defaults to `call`. Takes `self`
  by value: the runtime dispatches here when the value being called holds
  the last reference to the function, so implementations can move captured
  state instead of cloning it — e.g. the built-in partial-application
  wrapper moves its captured arguments.

`call_once`'s by-value `self` makes `Function` deliberately
non-dyn-compatible; `FnValue` stores implementors through a hidden
dyn-compatible bridge trait that recovers the by-value call when its
`Rc` unwraps as unique, and falls back to `call` when shared.

The easiest way to expose a Rust closure is `register_external` (hint
`(0, None)`: "takes anything, decides itself how much to consume") or
`register_function` / `Any::function` with a precise hint:

```rust
use raft_runtime::{Any, Runtime};

let mut rt = Runtime::new();

// consumes everything it is given
rt.register_external("print", |_rt, args| {
    for a in args {
        println!("{}", a);
    }
    (Any::nil(), args.len())
});

// takes exactly two arguments — calls with fewer partially apply
rt.register_function("hypot2", 2, Some(2), |_rt, args| {
    // args.len() >= 2 is guaranteed here
    (Any::nil(), 2)
});
```

Stateful or allocation-sensitive callables can implement `Function`
directly and be wrapped with `FnValue::new`/`FnValue::exact`.

The bundled REPL (`repl/src/main.rs`) registers `print` (hint `(0, None)`)
and `quit` (hint `(0, Some(0))`) this way.

### Execution modes: AST walking and bytecode

The runtime has two interchangeable execution modes. By default everything
is interpreted by walking the AST. Alternatively, `fn` definitions can be
compiled to a stack-based instruction set (`raft_runtime::vm::Instr`) that
is encoded into a flat byte array: operand widths — and the most common
operand values, like small slot indices, small integers, `True`/`False`
and operator kinds — are packed into the opcode byte itself; larger
integers follow as compact immediates instead of const-pool entries; jump
targets are fixed 4-byte byte offsets. A small virtual machine executes
the bytes directly (`vm::Code::disassemble` decodes them back for
inspection):

```rust
let mut rt = Runtime::new();
rt.set_compile_fns(true); // `fn` statements compile to bytecode from here on
```

The modes mix freely inside one runtime: a compiled function is an ordinary
`Fn` value with the same calling convention (including currying and partial
application), so AST-interpreted code can call bytecode functions and vice
versa, and host functions work from both. Anything the compiler rejects
falls back to the tree walker transparently.

Compiled functions share a per-runtime `VmContext` (`Runtime::vm`):
constants, variable names and patterns are interned once across all
functions, and every compiled frame executes on the context's single
operand stack instead of allocating its own. That stack is public —
`rt.vm.stack` — so a host function called from compiled code can inspect
the caller's live temporaries (mutate them at your own risk).

The REPL compiles functions by default; pass `--no-vm` to stay on the tree
walker. `cargo bench -p raft-runtime` (or `cargo criterion`, if installed)
runs Criterion benchmarks comparing the two modes, plus compiler and
pattern-binding micro-benchmarks (`runtime/benches/vm.rs`).

## Roadmap

Raft is under active development. Notable gaps today:

- No pattern-matched function overloads yet — declaring multiple variants of
  a function, dispatched by matching each call's arguments against a
  different parameter pattern, is planned (see
  [No user-defined types](#no-user-defined-types)).
- No standard library.

Contributions and ideas are welcome while the language design is still
taking shape.


