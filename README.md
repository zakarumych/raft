# Raft — a simplicity-first programming language

Raft is a small, dynamically-typed scripting language implemented in Rust.
It favors a minimal syntax and a small set of orthogonal features over a large
surface area — the goal is a language that is easy to read, easy to embed,
and easy to reason about.

> **Status: early / work in progress.** The lexer, parser, tree-walking
> runtime and REPL are functional, including user-defined functions with
> currying. The language is still missing pieces you'd expect from a
> "complete" language (modules, a standard library, real closures — see
> [Roadmap](#roadmap)). APIs and syntax may change without notice.

## Project layout

This is a Cargo workspace made up of four crates:

| Crate               | Path        | Description                                                                 |
| ------------------- | ----------- | ---------------------------------------------------------------------------- |
| `raft-lexer`        | `lexer/`    | `no_std`-friendly tokenizer: idents, atoms, numbers, chars, strings, comments, punctuation and delimiter groups (including indentation-based blocks). |
| `raft-ast`          | `ast/`      | AST types plus a recursive-descent parser built on top of the token stream. |
| `raft-runtime`      | `runtime/`  | A tree-walking interpreter that evaluates the AST.                          |
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

Calling a function evaluates its body in a fresh local scope; it does not
close over the caller's local variables (see [Roadmap](#roadmap)) — only
globals are visible inside. See [Function application](#function-application)
for how currying and partial application work when calling.

### Pattern matching & destructuring

Assignment targets can be simple identifiers, or patterns that destructure
lists and records:

```raft
[a, b, c] = xs
{ x, y } = point
```

Patterns can also match literal values and atoms, which is used to
accept/reject values during pattern-matching assignment.

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

### Embedding host functions

Beyond `fn`-defined functions, the host can expose Rust closures to scripts
by registering them on the `Runtime`. A registered closure receives the
runtime and the evaluated arguments, and returns the result together with
how many of the given arguments it actually consumed — the same convention
`fn`-defined functions use internally to support currying (see
[Functions](#functions)):

```rust
use raft_runtime::{Any, Runtime};

let mut rt = Runtime::new();
rt.register_external("print", |_rt, args| {
    for a in args {
        println!("{}", a);
    }
    (Any::nil(), args.len())
});
```

Returning a consumed count smaller than `args.len()` lets a host function
opt into the same partial-application behavior as `fn`-defined functions;
returning `args.len()` (as above) is the common case of "consume everything
given".

The bundled REPL (`repl/src/main.rs`) registers `print` and `quit` this way.

## Roadmap

Raft is under active development. Notable gaps today:

- Functions don't close over the caller's local scope — only globals are
  visible inside a function body, so there are no true lexical closures yet.
- No pattern-matched function overloads yet — declaring multiple variants of
  a function, dispatched by matching each call's arguments against a
  different parameter pattern, is planned (see
  [No user-defined types](#no-user-defined-types)).
- No module or import system.
- No standard library.

Contributions and ideas are welcome while the language design is still
taking shape.


