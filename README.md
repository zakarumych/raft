# Raft — a simplicity-first programming language

Raft is a small, dynamically-typed scripting language implemented in Rust.
It favors a minimal syntax and a small set of orthogonal features over a large
surface area — the goal is a language that is easy to read, easy to embed,
and easy to reason about.

> **Status: early / work in progress.** The lexer, parser and tree-walking
> runtime are functional, but the language is still missing pieces you'd
> expect from a "complete" language (user-defined functions, modules, a
> standard library, a real REPL — see [Roadmap](#roadmap)). APIs and syntax
> may change without notice.

## Project layout

This is a Cargo workspace made up of four crates:

| Crate               | Path        | Description                                                                 |
| ------------------- | ----------- | ---------------------------------------------------------------------------- |
| `raft-lexer`        | `lexer/`    | `no_std`-friendly tokenizer: idents, atoms, numbers, chars, strings, comments, punctuation and delimiter groups (including indentation-based blocks). |
| `raft-ast`          | `ast/`      | AST types plus a recursive-descent parser built on top of the token stream. |
| `raft-runtime`      | `runtime/`  | A tree-walking interpreter that evaluates the AST.                          |
| `raft-repl`         | `repl/`     | Command-line front end (currently a placeholder — see [Roadmap](#roadmap)). |

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
self-evaluating symbols (similar to symbols/keywords in other languages).
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

Unary operators: `!` (logical not), `~` (bitwise not), `+` (identity),
`-` (negate).

### Function application

Function calls use juxtaposition rather than parentheses (parentheses are
only needed for grouping):

```raft
print "hello"
add 1 2
```

Only host-registered functions can currently be called this way — see
[Embedding host functions](#embedding-host-functions).

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

### Pattern matching & destructuring

Assignment targets can be simple identifiers, or patterns that destructure
lists and records:

```raft
[a, b, c] = xs
{ x, y } = point
```

Patterns can also match literal values and atoms, which is used to
accept/reject values during pattern-matching assignment.

### Mutability

Lists and records are mutable heap objects by default and can be frozen
(from host code) to prevent further mutation.

### Embedding host functions

Since Raft doesn't yet have user-defined functions, functionality is exposed
to scripts by registering Rust closures on the `Runtime`:

```rust
use raft_runtime::Runtime;

let mut rt = Runtime::new();
rt.register_external("print", |_rt, args| {
    for a in args {
        println!("{:?}", a);
    }
    // ... return some Any value
});
```

## Roadmap

Raft is under active development. Notable gaps today:

- No user-defined functions/closures — only host-registered external
  functions can be called.
- No module or import system.
- No standard library.
- The `raft-repl` binary is a placeholder ("Hello, world!") and doesn't yet
  wire the lexer/parser/runtime together into an interactive REPL.

Contributions and ideas are welcome while the language design is still
taking shape.


