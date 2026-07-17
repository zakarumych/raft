//! Criterion benchmarks comparing the AST walker against the bytecode VM
//! and against "oxidized" (Raft-transpiled-to-Rust cdylib bundle)
//! execution, plus compiler and binding micro-benchmarks.
//!
//! Run with `cargo bench -p raft-runtime` or `cargo criterion`. The
//! oxidized mode transpiles, cargo-builds and links a bundle per group on
//! first run (cached under the temp dir afterwards).

use std::{hint::black_box, rc::Rc};

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use raft_ast::{Stmt, StmtKind};
use raft_runtime::{BundleBuilder, Frame, Runtime, vm};

fn parse(src: &str) -> Vec<Stmt> {
    let tokens = raft_ast::lexer::parse_str(src, &raft_ast::lexer::Options::wss()).unwrap();
    let mut stream = raft_ast::parser::TokenStream::new(tokens);
    Stmt::parse_many(&mut stream).unwrap()
}

fn runtime_with(defs: &str, compiled: bool) -> (Runtime, Rc<Frame>) {
    let mut rt = Runtime::new();
    rt.set_compile_fns(compiled);
    let frame = Rc::new(Frame::new());

    for stmt in parse(defs) {
        rt.exec_stmt(&stmt, frame.clone()).unwrap();
    }

    (rt, frame)
}

/// A runtime whose `defs` functions were transpiled to Rust, built as a
/// cdylib bundle (same profile as the bench itself - release), linked,
/// and re-exposed as globals so `call` scripts resolve them by name.
fn runtime_oxidized(group: &str, defs: &str) -> (Runtime, Rc<Frame>) {
    let mut rt = Runtime::new();
    let frame = Rc::new(Frame::new());

    // export every top-level fn of `defs` from the bundle module
    let exports: Vec<String> = parse(defs)
        .iter()
        .filter_map(|stmt| match stmt.kind() {
            StmtKind::Fn { name, .. } => Some(name.name().to_string()),
            _ => None,
        })
        .collect();
    let module_src = format!("{defs}export {{ {} }}\n", exports.join(", "));

    let crate_name = format!("raft_bench_{}", group.replace('-', "_"));
    rt.build_bundle(
        &BundleBuilder::new(&crate_name)
            .module("bench", module_src)
            .release(true),
    )
    .unwrap_or_else(|e| panic!("building oxidized bundle for {group}: {e}"));

    let module = rt.module("bench").expect("bundle module registered");
    for name in &exports {
        let f = module.get_field(name).expect("exported fn");
        rt.set_var(name.as_str(), f);
    }
    (rt, frame)
}

/// Benchmark executing `call` (with functions from `defs` predefined) in
/// all three execution modes.
fn bench_modes(c: &mut Criterion, group: &str, defs: &str, call: &str) {
    let mut g = c.benchmark_group(group);
    let call_stmts = parse(call);

    let mut runtimes = vec![
        ("ast-walk", runtime_with(defs, false)),
        ("bytecode", runtime_with(defs, true)),
        ("oxidized", runtime_oxidized(group, defs)),
    ];
    for (mode, (rt, frame)) in &mut runtimes {
        let _ = mode;
        g.bench_function(*mode, |b| {
            b.iter(|| {
                for stmt in &call_stmts {
                    black_box(rt.exec_stmt(stmt, frame.clone()).unwrap());
                }
            })
        });
    }
    g.finish();
}

/// Benchmark executing `call` (with functions from `defs` predefined) in
/// both execution modes.
fn bench_rust<A, R>(c: &mut Criterion, group: &str, f: &dyn Fn(A) -> R, call: A)
where
    A: Clone,
{
    let mut g = c.benchmark_group(group);

    g.bench_function("rust", |b| {
        b.iter(|| {
            black_box(f(call.clone()));
        })
    });

    g.finish();
}

fn bench_python(c: &mut Criterion, group: &str, defs: &str, call: &str) {
    use pyo3::{prelude::*, types::*};
    use std::ffi::CString;

    let mut g = c.benchmark_group(group);

    Python::attach(|py| {
        let defs = CString::new(defs).unwrap();

        let globals = PyDict::new(py);

        py.run(defs.as_c_str(), Some(&globals), None).unwrap();

        let call = CString::new(call).unwrap();

        // let call = PyCode::compile(
        //     py,
        //     call.as_c_str(),
        //     CStr::from_bytes_with_nul(b"bench.py\0").unwrap(),
        //     PyCodeInput::Eval,
        // )
        // .unwrap();

        g.bench_function("python", |b| {
            b.iter(|| {
                Python::attach(|_| {
                    black_box(py.eval(call.as_c_str(), Some(&globals), None).unwrap());
                    // black_box(call.run(Some(&globals), None)).unwrap();
                })
            })
        });
    });

    g.finish();
}

const FIB: &str = "fn fib n:\n  if n < 2: return 1\n  fib (n - 1) + fib (n - 2)\n";
const FIB_PY: &str = "def fib(n):\n  if n < 2:\n    return 1\n  return (fib (n - 1)) + (fib (n - 2))\n";

const LOOP_1000: &str = "fn count n:\n  i = 0\n  total = 0\n  while i < n:\n    total = total + i * 2 - i\n    i = i + 1\n  total\n";
const LOOP_1000_PY: &str = "def count(n):\n  i = 0\n  total = 0\n  while i < n:\n    total = total + i * 2 - i\n    i = i + 1\n  return total\n";

const COLLATZ: &str = "fn collatz n:\n  steps = 0\n  while n != 1:\n    if n & 1 == 0:\n      n = n / 2\n    else:\n      n = 3 * n + 1\n    steps = steps + 1\n  steps\n";
const COLLATZ_PY: &str = "def collatz(n):\n  steps = 0\n  while n != 1:\n    if n & 1 == 0:\n      n = int(n / 2)\n    else:\n      n = 3 * n + 1\n    steps = steps + 1\n  return steps\n";

/// Recursion / call-heavy workload.
fn fib_module(c: &mut Criterion) {
    // fib defined in a module: recursive calls resolve statically to the
    // module slot (final-fn analysis) instead of a global hash lookup
    let mut g = c.benchmark_group("fib-15-module");
    let mut rt = Runtime::new();
    rt.set_compile_fns(true);
    let module = rt
        .load_module("fibmod", &(FIB.to_string() + "export { fib }\n"))
        .unwrap();
    rt.set_var("m", module);
    let frame = Rc::new(Frame::new());
    for stmt in parse("fib = m.fib\n") {
        rt.exec_stmt(&stmt, frame.clone()).unwrap();
    }
    let call_stmts = parse("fib 15\n");
    g.bench_function("bytecode", |b| {
        b.iter(|| {
            for stmt in &call_stmts {
                black_box(rt.exec_stmt(stmt, frame.clone()).unwrap());
            }
        })
    });
    g.finish();
}

fn fib(c: &mut Criterion) {
    bench_modes(c, "fib-15", FIB, "fib 15\n");
    bench_python(c, "fib-15", FIB_PY, "fib(15)\n");

    fn fib(n: usize) -> usize {
        if n < 2 {
            return n;
        }
        fib(n - 1) + fib(n - 2)
    }

    bench_rust(c, "fib-15", black_box(&fib), 15);
}

/// Arithmetic and variable access in a tight loop.
fn tight_loop(c: &mut Criterion) {
    bench_modes(c, "loop-1000", LOOP_1000, "count 1000\n");
    bench_python(c, "loop-1000", LOOP_1000_PY, "count(1000)\n");

    fn count(n: usize) -> usize {
        let mut i = 0;
        let mut total = 0;
        while i < n {
            total = total + i * 2 - i;
            i += 1;
        }
        total
    }

    bench_rust(c, "loop-1000", black_box(&count), 1000);
}

/// Branch-heavy control flow.
fn collatz(c: &mut Criterion) {
    bench_modes(c, "collatz-27", COLLATZ, "collatz 27\n");
    bench_python(c, "collatz-27", COLLATZ_PY, "collatz(27)\n");

    fn collatz(n: usize) -> usize {
        let mut n = n;
        let mut steps = 0;
        while n != 1 {
            if n & 1 == 0 {
                n /= 2;
            } else {
                n = 3 * n + 1;
            }
            steps += 1;
        }
        steps
    }

    bench_rust(c, "collatz-27", black_box(&collatz), 27);
}

/// ~100 lines of Raft: a driver looping over a data set and calling a
/// dozen small helpers - records, lists, atoms, destructuring, bit ops.
/// Call-heavy but without recursion, resembling ordinary scripting code.
const PIPELINE: &str = r#"fn imod a n:
    a - (a / n) * n

fn iabs x:
    if x < 0:
        return 0 - x
    x

fn imin a b:
    if a < b: return a
    b

fn imax a b:
    if a > b: return a
    b

fn clamp lo hi x:
    imax lo (imin hi x)

fn tri n:
    n * (n + 1) / 2

fn parity n:
    if (n & 1) == 0: return Even
    Odd

fn digits n:
    total = 0
    while n > 0:
        total = total + (imod n 10)
        n = n / 10
    total

fn classify n:
    if (imod n 15) == 0:
        return FizzBuzz
    if (imod n 3) == 0:
        return Fizz
    if (imod n 5) == 0:
        return Buzz
    Plain

fn score tag:
    if tag == FizzBuzz:
        return 15
    if tag == Fizz:
        return 3
    if tag == Buzz:
        return 5
    1

fn weight { lo, hi } x:
    clamp lo hi (x * 2 - 7)

fn stats_new:
    { total: 0, evens: 0, odds: 0, peak: 0, tags: 0 }

fn process items bounds:
    st = (stats_new)
    for x in items:
        if x == 12:
            continue
        v = weight bounds x
        tag = classify v
        pts = score tag
        d = digits (iabs v)
        combo = pts * d + (v & 7) - (v >> 2)
        if (parity combo) == Even:
            st.evens = st.evens + 1
        else:
            st.odds = st.odds + 1
        if combo > st.peak:
            st.peak = combo
        st.total = st.total + combo
        st.tags = st.tags + pts
    st

fn crunch rounds:
    items = [3, 7, 12, 19, 4, 25, 8, 30, 11, 6, 21, 14, 9, 16, 5]
    bounds = { lo: 0, hi: 40 }
    acc = 0
    r = 0
    while r < rounds:
        st = process items bounds
        { total, peak } = st
        acc = acc + total + peak - (st.evens * st.odds) + (tri r)
        r = r + 1
    acc
"#;

const PIPELINE_PY: &str = r#"def imod(a, n):
    return a - (a // n) * n

def iabs(x):
    if x < 0:
        return 0 - x
    return x

def imin(a, b):
    if a < b:
        return a
    return b

def imax(a, b):
    if a > b:
        return a
    return b

def clamp(lo, hi, x):
    return imax(lo, imin(hi, x))

def tri(n):
    return n * (n + 1) // 2

def parity(n):
    if (n & 1) == 0:
        return "Even"
    return "Odd"

def digits(n):
    total = 0
    while n > 0:
        total = total + imod(n, 10)
        n = n // 10
    return total

def classify(n):
    if imod(n, 15) == 0:
        return "FizzBuzz"
    if imod(n, 3) == 0:
        return "Fizz"
    if imod(n, 5) == 0:
        return "Buzz"
    return "Plain"

def score(tag):
    if tag == "FizzBuzz":
        return 15
    if tag == "Fizz":
        return 3
    if tag == "Buzz":
        return 5
    return 1

def weight(bounds, x):
    return clamp(bounds["lo"], bounds["hi"], x * 2 - 7)

def stats_new():
    return {"total": 0, "evens": 0, "odds": 0, "peak": 0, "tags": 0}

def process(items, bounds):
    st = stats_new()
    for x in items:
        if x == 12:
            continue
        v = weight(bounds, x)
        tag = classify(v)
        pts = score(tag)
        d = digits(iabs(v))
        combo = pts * d + (v & 7) - (v >> 2)
        if parity(combo) == "Even":
            st["evens"] = st["evens"] + 1
        else:
            st["odds"] = st["odds"] + 1
        if combo > st["peak"]:
            st["peak"] = combo
        st["total"] = st["total"] + combo
        st["tags"] = st["tags"] + pts
    return st

def crunch(rounds):
    items = [3, 7, 12, 19, 4, 25, 8, 30, 11, 6, 21, 14, 9, 16, 5]
    bounds = {"lo": 0, "hi": 40}
    acc = 0
    r = 0
    while r < rounds:
        st = process(items, bounds)
        acc = acc + st["total"] + st["peak"] - st["evens"] * st["odds"] + tri(r)
        r = r + 1
    return acc
"#;

/// A big non-recursive driver calling many small helpers (~100 lines).
fn pipeline(c: &mut Criterion) {
    // sanity: all execution modes must agree on the result being timed
    let results: Vec<String> = [
        runtime_with(PIPELINE, false),
        runtime_with(PIPELINE, true),
        runtime_oxidized("pipeline-check", PIPELINE),
    ]
    .into_iter()
    .map(|(mut rt, frame)| {
        for stmt in &parse("r = crunch 10\n") {
            rt.exec_stmt(stmt, frame.clone()).unwrap();
        }
        format!("{}", frame.get_var("r", &mut rt))
    })
    .collect();
    assert_eq!(results[0], results[1], "walker/VM disagree on pipeline");
    assert_eq!(
        results[0], results[2],
        "walker/oxidized disagree on pipeline"
    );

    bench_modes(c, "pipeline-10", PIPELINE, "crunch 10\n");
    bench_python(c, "pipeline-10", PIPELINE_PY, "crunch(10)\n");
}

/// Generator create/resume overhead with real per-step work: a Fibonacci
/// `gen fn` carrying state across yields, consumed by a `for` - one
/// resume per item, plus one generator creation per call.
const FIB_GENERATOR: &str = "gen fn fib_gen n:
    a = 0
    b = 1
    i = 0
    while i < n:
        yield a
        t = a + b
        a = b
        b = t
        i = i + 1

fn consume n:
    total = 0
    for x in (fib_gen n):
        total = total + x
    total
";

const FIB_GENERATOR_PY: &str = "def fib_gen(n):
    a = 0
    b = 1
    i = 0
    while i < n:
        yield a
        t = a + b
        a = b
        b = t
        i = i + 1

def consume(n):
    total = 0
    for x in fib_gen(n):
        total = total + x
    return total
";

fn fib_generator(c: &mut Criterion) {
    // sanity: all execution modes must agree on the result being timed
    let results: Vec<String> = [
        runtime_with(FIB_GENERATOR, false),
        runtime_with(FIB_GENERATOR, true),
        runtime_oxidized("gen-check", FIB_GENERATOR),
    ]
    .into_iter()
    .map(|(mut rt, frame)| {
        for stmt in &parse("r = consume 50\n") {
            rt.exec_stmt(stmt, frame.clone()).unwrap();
        }
        format!("{}", frame.get_var("r", &mut rt))
    })
    .collect();
    assert_eq!(results[0], results[1], "walker/VM disagree on generator");
    assert_eq!(
        results[0], results[2],
        "walker/oxidized disagree on generator"
    );

    bench_modes(c, "gen-fib-50", FIB_GENERATOR, "consume 50\n");
    bench_python(c, "gen-fib-50", FIB_GENERATOR_PY, "consume(50)\n");

    fn consume(n: usize) -> i64 {
        // Rust's own lazy-iterator counterpart of the Fibonacci generator
        struct Fib {
            a: i64,
            b: i64,
        }
        impl Iterator for Fib {
            type Item = i64;

            fn next(&mut self) -> Option<i64> {
                let out = self.a;
                let t = self.a + self.b;
                self.a = self.b;
                self.b = t;
                Some(out)
            }
        }
        Fib { a: 0, b: 1 }.take(n).sum()
    }

    bench_rust(c, "gen-fib-50", black_box(&consume), 50);
}

/// Generator create/resume overhead with real per-step work: a Fibonacci
/// `gen fn` carrying state across yields, consumed by a `for` - one
/// resume per item, plus one generator creation per call.
const COLLATZ_GENERATOR: &str = "gen fn collatz_gen n:
    while n != 1:
        yield n
        if (n & 1) == 0:
            n = n / 2
        else:
            n = 3 * n + 1
    yield 1

fn consume n:
    total = 0
    for x in (collatz_gen n):
        total = total + x
    total
";

const COLLATZ_GENERATOR_PY: &str = "def collatz_gen(n):
    while n != 1:
        yield n
        if (n & 1) == 0:
            n = n // 2
        else:
            n = 3 * n + 1
    yield 1

def consume(n):
    total = 0
    for x in collatz_gen(n):
        total = total + x
    return total
";

fn collatz_generator(c: &mut Criterion) {
    // sanity: all execution modes must agree on the result being timed
    let results: Vec<String> = [
        runtime_with(COLLATZ_GENERATOR, false),
        runtime_with(COLLATZ_GENERATOR, true),
        runtime_oxidized("gen-check", COLLATZ_GENERATOR),
    ]
    .into_iter()
    .map(|(mut rt, frame)| {
        for stmt in &parse("r = consume 27\n") {
            rt.exec_stmt(stmt, frame.clone()).unwrap();
        }
        format!("{}", frame.get_var("r", &mut rt))
    })
    .collect();
    assert_eq!(results[0], results[1], "walker/VM disagree on generator");
    assert_eq!(
        results[0], results[2],
        "walker/oxidized disagree on generator"
    );

    bench_modes(c, "gen-collatz-27", COLLATZ_GENERATOR, "consume 27\n");
    bench_python(c, "gen-collatz-27", COLLATZ_GENERATOR_PY, "consume(27)\n");

    fn consume(n: i64) -> i64 {
        // Rust's own lazy-iterator counterpart of the Fibonacci generator
        struct Collatz {
            n: Option<i64>,
        }
        impl Iterator for Collatz {
            type Item = i64;

            fn next(&mut self) -> Option<i64> {
                if let Some(n) = self.n {
                    if n == 1 {
                        self.n = None;
                        return Some(1);
                    }
                    let out = n;
                    if (n & 1) == 0 {
                        self.n = Some(n / 2);
                    } else {
                        self.n = Some(3 * n + 1);
                    }
                    Some(out)
                } else {
                    None
                }
            }
        }
        Collatz { n: Some(n) }.sum()
    }

    bench_rust(c, "gen-collatz-27", black_box(&consume), 27);
}

/// Full application vs a curried chain of partial applications.
fn calls(c: &mut Criterion) {
    let defs = "fn add3 a b c:\n    return a + b + c\n";
    bench_modes(c, "call-direct", defs, "add3 1 2 3\n");
    bench_modes(c, "call-curried", defs, "((add3 1) 2) 3\n");
}

/// Pattern binding strategies on the VM: one `Bind` instruction vs the
/// equivalent spelled-out access-and-store sequences, plus pure literal
/// matching (which binds nothing).
fn binding(c: &mut Criterion) {
    let mut g = c.benchmark_group("binding-x100");

    for (name, body) in [
        ("record-destructure", "        { x, y } = p\n"),
        ("record-fields", "        x = p.x\n        y = p.y\n"),
        ("list-destructure", "        [x, y] = q\n"),
        ("list-index", "        x = q[0]\n        y = q[1]\n"),
        ("match-string", "        \"abc\" = t\n"),
        ("match-int", "        3 = x\n"),
        ("assign-baseline", "        x = 3\n        y = 4\n"),
    ] {
        let defs = format!(
            "fn f n:\n    p = {{ x: 3, y: 4 }}\n    q = [3, 4]\n    t = \"abc\"\n    x = 3\n    y = 4\n    i = 0\n    s = 0\n    while i < n:\n{body}        s = s + x - y\n        i = i + 1\n    return s\n"
        );
        let call_stmts = parse("f 100\n");
        let (mut rt, frame) = runtime_with(&defs, true);
        g.bench_function(name, |b| {
            b.iter(|| {
                for stmt in &call_stmts {
                    black_box(rt.exec_stmt(stmt, frame.clone()).unwrap());
                }
            })
        });
    }
    g.finish();
}

/// Bytecode compilation itself (AST → instructions, fresh context each
/// time so pool interning is included).
fn compile(c: &mut Criterion) {
    let mut g = c.benchmark_group("compile");

    for (name, defs) in [
        ("fib", FIB),
        ("loop", LOOP_1000),
        ("collatz", COLLATZ),
        (
            "destructuring",
            "fn dist2 { x, y }:\n    fn sq v:\n        return v * v\n    [a, b] = [x, y]\n    return (sq a) + (sq b)\n",
        ),
    ] {
        let stmts = parse(defs);
        let StmtKind::Fn { params, body, .. } = stmts[0].kind() else {
            panic!("expected fn stmt");
        };

        struct Cx {
            rt: Runtime,
            frame: Rc<Frame>,
        }

        g.bench_function(name, |b| {
            b.iter_batched(
                || Cx {
                    rt: Runtime::new(),
                    frame: Rc::new(Frame::new()),
                },
                |mut cx| {
                    vm::compile_fn(
                        &mut cx.rt,
                        params.clone(),
                        body,
                        vm::CompileParent::Walked(cx.frame.clone()),
                        &[],
                    )
                    .unwrap()
                },
                BatchSize::SmallInput,
            )
        });
    }
    g.finish();
}

criterion_group!(
    benches, fib, fib_module, tight_loop, collatz, pipeline, fib_generator, collatz_generator, calls, binding, compile
);
criterion_main!(benches);
