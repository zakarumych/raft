//! Full transpiled-bundle round trip, driven entirely by the runtime's
//! own bundle support (`raft-runtime`'s `bundle` feature):
//!
//! 1. `Runtime::build_bundle` transpiles a Raft module to a cdylib bundle
//!    crate, builds it by invoking `cargo` (in the same profile this
//!    example runs in), and
//! 2. links the produced cdylib - version check, init against the live
//!    runtime, module registration, library held by the runtime - then
//! 3. the host executes Raft AST that `import`s the bundle's module and
//!    calls its functions.
//!
//! Usage: cargo run -p raft-rust --example bundle_e2e

use std::rc::Rc;

use raft_core::{Number, Val, ValEnum};
use raft_runtime::{BundleBuilder, Exec, Frame, Runtime};

/// The Raft module that gets transpiled into the bundle.
const MATH_RAFT: &str = "\
fn add a b:
    return a + b

fn fact n:
    if n < 2:
        return 1
    return n * (fact (n - 1))

export { add, fact }
";

/// The Raft program the host runtime executes against the linked bundle.
const SCRIPT: &str = "\
import math
answer = math.add (math.fact 3) 36
print answer
";

fn main() {
    let mut rt = Runtime::new();
    rt.register_function("print", 0, None, |rt, args| {
        for arg in rt.stack().drain_top(args).rev() {
            println!("{}", arg);
        }
        Val::nil()
    });

    // transpile + cargo build + link, all through the runtime
    let names = rt
        .build_bundle(&BundleBuilder::new("raft_bundle_e2e").module("math", MATH_RAFT))
        .expect("build and link bundle");
    println!("linked bundle modules: {names:?}");

    let root = Rc::new(Frame::new());
    for stmt in parse_stmts(SCRIPT) {
        match rt.exec_stmt(&stmt, root.clone()) {
            Ok(Exec::Value(_)) => {}
            Ok(other) => panic!("unexpected control flow at top level: {other:?}"),
            Err(e) => panic!("runtime error: {e}"),
        }
    }

    let answer = root.get_var("answer", &mut rt);
    match answer.unpack() {
        ValEnum::Number(Number::Integer(42)) => println!("bundle e2e passed: answer == 42"),
        other => panic!("expected 42, got {:?}", Val::from(other)),
    }
}

fn parse_stmts(source: &str) -> Vec<raft_ast::Stmt> {
    let tokens =
        raft_ast::lexer::parse_str(source, &raft_ast::lexer::Options::wss()).expect("lex script");
    let mut stream = raft_ast::parser::TokenStream::new(tokens);
    raft_ast::Stmt::parse_many(&mut stream).expect("parse script")
}
