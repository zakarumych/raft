//! Transpiled-async round trip - the async analogue of `bundle_gen_e2e`:
//!
//! 1. `Runtime::build_bundle` transpiles a Raft module containing
//!    `async fn` definitions (async-block state machines with real awaits)
//!    and builds/links it;
//! 2. the host executor (`Runtime::block_on`) drives bundle-created
//!    futures, its task waker ambient on the host across the FFI boundary;
//! 3. a bundle async fn awaits a *host*-provided async leaf (`nap`) that
//!    comes back pending and wakes the task through the thin-pointer FFI
//!    waker - executor waker -> ambient host waker -> std `Waker` adapter
//!    -> wake -> ready queue, all crossing the cdylib boundary.
//!
//! Usage: cargo run -p raft-rust --example bundle_async_e2e

use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

use raft_runtime::{BundleBuilder, Exec, Frame, Runtime, RuntimeError, Val, ValEnum, async_val};

/// The Raft module that gets transpiled into the bundle.
const ASYNC_RAFT: &str = "\
async fn add_async a b:
    return a + b

async fn compose x:
    y = await (add_async x 1)
    z = await (add_async y 2)
    return z

async fn slow_sum n:
    total = 0
    i = 0
    while i < n:
        v = await (nap i)
        total = total + v
        i = i + 1
    return total

async gen fn napping_squares n:
    i = 0
    while i < n:
        v = await (nap i)
        yield v * v
        i = i + 1

async fn sum_squares n:
    s = 0
    async for x in (napping_squares n):
        s = s + x
    return s

export { add_async, compose, slow_sum, sum_squares }
";

/// The Raft program the host runtime executes against the linked bundle.
const SCRIPT: &str = "\
m = import \"asyncs\"
fut1 = m.compose 5
fut2 = m.slow_sum 4
fut3 = m.sum_squares 4
";

/// Pending once, waking its own task before suspending - so resolving it
/// takes a full trip through the executor's ready queue.
struct Nap {
    value: Val,
    polled: bool,
}

impl Future for Nap {
    type Output = Result<Val, RuntimeError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.polled {
            Poll::Ready(Ok(self.value.clone()))
        } else {
            self.polled = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

fn main() {
    let mut rt = Runtime::new();
    register_import(&mut rt);
    rt.register_function("nap", 1, Some(1), |rt, _args| {
        let v = rt.stack().pop();
        async_val(Nap {
            value: v,
            polled: false,
        })
    });

    let names = rt
        .build_bundle(&BundleBuilder::new("raft_bundle_async_e2e").module("asyncs", ASYNC_RAFT))
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

    // chained awaits inside the bundle: 5 + 1 + 2
    let fut1 = root.get_var("fut1", &mut rt);
    let v1 = smol::block_on(rt.eval_async(fut1)).expect("compose resolves");
    assert_eq!(format!("{v1}"), "8", "compose 5");

    // a bundle async fn awaiting a pending host leaf in a loop: 0+1+2+3,
    // with one executor round trip per element
    let fut2 = root.get_var("fut2", &mut rt);
    let v2 = smol::block_on(rt.eval_async(fut2)).expect("slow_sum resolves");
    assert_eq!(format!("{v2}"), "6", "slow_sum 4");

    // a bundle async gen (yields between pending awaits) iterated by a
    // bundle async fn: 0+1+4+9, one executor round trip per element
    let fut3 = root.get_var("fut3", &mut rt);
    let v3 = smol::block_on(rt.eval_async(fut3)).expect("sum_squares resolves");
    assert_eq!(format!("{v3}"), "14", "sum_squares 4");

    println!("bundle async e2e passed");
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
            return Val::nil();
        };
        match rt.module(name.as_str()) {
            Some(module) => module,
            None => {
                rt.set_error(RuntimeError::Other(
                    format!("no module named '{name}' registered").into(),
                ));
                Val::nil()
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
