//! Transpiled-generator round trip, driven by the runtime's own bundle
//! support - the generator analogue of `bundle_e2e`:
//!
//! 1. `Runtime::build_bundle` transpiles a Raft module containing `gen fn`
//!    definitions (async-state-machine codegen) and builds/links it;
//! 2. a transpiled function iterates a transpiled generator *inside* the
//!    bundle (`sum_upto`);
//! 3. the host's AST walker iterates a bundle-created generator object
//!    directly (`for v in g:`), resuming it across the FFI boundary.
//!
//! Usage: cargo run -p raft-rust --example bundle_gen_e2e

use std::rc::Rc;

use raft_runtime::{BundleBuilder, Exec, Frame, Runtime, RuntimeError, ValEnum};

/// The Raft module that gets transpiled into the bundle.
const GEN_RAFT: &str = "\
gen fn count n:
    i = 0
    while i < n:
        yield i
        i = i + 1

fn sum_upto n:
    total = 0
    for x in (count n):
        total = total + x
    return total

gen fn evens list:
    for x in list:
        if (x & 1) == 0:
            yield x

export { count, sum_upto, evens }
";

/// The Raft program the host runtime executes against the linked bundle.
const SCRIPT: &str = "\
m = import \"gens\"
s = m.sum_upto 5
g = m.count 4
walked = 0
for v in g:
    walked = walked + v
rerun = 0
for v in g:
    rerun = rerun + 100
nums = [1, 2, 3, 4, 5, 6]
esum = 0
for e in (m.evens nums):
    esum = esum + e
";

fn main() {
    let mut rt = Runtime::new();
    register_import(&mut rt);

    let names = rt
        .build_bundle(&BundleBuilder::new("raft_bundle_gen_e2e").module("gens", GEN_RAFT))
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

    // sum_upto 5 = 0+1+2+3+4, iterated entirely inside the bundle
    expect_int(&mut rt, &root, "s", 10);
    // count 4 = 0+1+2+3, resumed by the host's walker across the FFI
    expect_int(&mut rt, &root, "walked", 6);
    // an exhausted generator stays exhausted
    expect_int(&mut rt, &root, "rerun", 0);
    // yield under for/if inside a transpiled generator
    expect_int(&mut rt, &root, "esum", 12);
    println!("bundle generator e2e passed");
}

fn expect_int(rt: &mut Runtime, root: &Frame, name: &str, expected: i64) {
    let v = root.get_var(name, rt);
    match v.unpack() {
        ValEnum::Number(raft_runtime::Number::Integer(i)) if i == expected => {}
        other => panic!("expected {name} == {expected}, got {other:?}", other = raft_runtime::Val::from(other)),
    }
}

fn register_import(rt: &mut Runtime) {
    // import "name" resolves against the runtime's registered modules -
    // which is where linked bundles' modules land
    rt.register_function("import", 1, Some(1), |rt, _args| {
        let name_val = rt.stack().pop();
        let ValEnum::String(name) = name_val.unpack() else {
            rt.set_error(RuntimeError::TypeError(
                "import expects a module name string".into(),
            ));
            return raft_runtime::Val::nil();
        };
        match rt.module(name.as_str()) {
            Some(module) => module,
            None => {
                rt.set_error(RuntimeError::Other(
                    format!("no module named '{name}' registered").into(),
                ));
                raft_runtime::Val::nil()
            }
        }
    });
}

fn parse_stmts(source: &str) -> Vec<raft_ast::Stmt> {
    let tokens =
        raft_ast::lexer::parse_str(source, &raft_ast::lexer::Options::wss()).expect("lex script");
    let mut stream = raft_ast::parser::TokenStream::new(tokens);
    raft_ast::Stmt::parse_many(&mut stream).expect("parse script")
}
