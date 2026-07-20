use std::{io::Write, pin::Pin, rc::Rc, task::{Context, Poll}};

use raft_ast::{lexer::LexErrorKind, parser::ParseErrorKind};
use raft_runtime::{Exec, Number, RcStr, Val, ValEnum, ValsIter, async_val};

fn main() {
    std::io::stdout().write_all(b"Raft REPL\n").unwrap();
    std::io::stdout().flush().unwrap();

    let mut rt = raft_runtime::Runtime::new();
    let root = Rc::new(raft_runtime::Frame::new());

    // `fn` statements compile to bytecode unless --no-vm is given;
    // top-level statements are always interpreted from the AST.
    rt.set_compile_fns(!std::env::args().any(|arg| arg == "--no-vm"));

    let quit_flag = Rc::new(std::cell::Cell::new(false));

    // print takes any number of arguments, quit takes exactly none
    rt.register_function("print", 0, None, |rt, args| {
        for arg in rt.stack().drain_top(args).rev() {
            println!("{}", arg);
        }
        Val::nil()
    });

    // print takes any number of arguments, quit takes exactly none
    rt.register_function("debugfmt", 0, None, |rt, args| {
        for arg in rt.stack().drain_top(args).rev() {
            println!("{:#?}", arg);
        }
        Val::nil()
    });

    let quit_flag_clone = quit_flag.clone();
    rt.register_function(
        "quit",
        0, Some(0), move |_rt, _args| {
            quit_flag_clone.set(true);
            Val::nil()
        },
    );

    // print takes any number of arguments, quit takes exactly none
    rt.register_function("timeout", 1, Some(1), |rt, _args| {
        struct Timeout {
            deadline: std::time::Instant,
        }

        impl Future for Timeout {
            type Output = Result<Val, raft_runtime::RuntimeError>;

            fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
                if std::time::Instant::now() >= self.deadline {
                    Poll::Ready(Ok(Val::nil()))
                } else {
                    let dur = self.deadline - std::time::Instant::now();

                    let waker = cx.waker().clone();

                    std::thread::spawn(move || {
                        std::thread::sleep(dur);
                        waker.wake();
                    });

                    Poll::Pending
                }
            }
        }

        let duration= match rt.stack().pop().unpack() {
            ValEnum::Number(Number::Integer(secs)) => std::time::Duration::from_secs(secs as u64),
            ValEnum::Number(Number::Float(f)) => std::time::Duration::from_secs_f64(f),
            _ => {
                rt.set_error(raft_runtime::RuntimeError::Other(RcStr::new("Invalid timeout argument type")));
                return Val::nil();
            }
        };

        let deadline = std::time::Instant::now() + duration;

        async_val(Timeout { deadline })
    });

    // `import modulename` is a language statement now (see
    // `raft_ast::StmtKind::Import`): `Runtime::import` (backing
    // `exec_stmt`'s handling of it) searches `module_dirs` for
    // `modulename.raft` and `cdylib_dirs` for a bundle exposing it - both
    // default to the current directory, which is all the REPL needs.

    rt.register_function(
        "list",
        1, Some(1), move |rt, args| {
            assert!(args == 1);
            let mut stack = rt.stack();
            let top = stack.pop();
            let Some(mut vals_iter) = rt.try_(|| ValsIter::new(&top)) else { return Val::nil(); };
            let mut items = Vec::new();
            loop {
                match vals_iter.next(&mut rt.as_host()) {
                    Ok(Some(v)) => items.push(v),
                    Ok(None) => break,
                    Err(e) => {
                        rt.set_error(e);
                        return Val::nil();
                    }
                }
            }

            Val::list(items)
        },
    );

    let mut lines = String::new();

    while quit_flag.get() == false {
        if lines.is_empty() {
            std::io::stdout().write_all(b"> ").unwrap();
        } else {
            std::io::stdout().write_all(b". ").unwrap();
        }

        std::io::stdout().flush().unwrap();

        if 0 == std::io::stdin().read_line(&mut lines).unwrap() {
            break;
        }

        // Remove one last occurrence of '\n', '\r' or '\r\n' from the end of the string
        let stripped = lines
            .strip_suffix("\r\n")
            .or_else(|| lines.strip_suffix('\n'))
            .or_else(|| lines.strip_suffix('\r'))
            .unwrap_or(&lines[..]);

        match raft_ast::lexer::parse_str(stripped, &raft_ast::lexer::Options::wss_repl()) {
            Ok(tokens) => {
                let mut stream = raft_ast::parser::TokenStream::new(tokens);
                match raft_ast::Stmt::parse_many(&mut stream) {
                    Ok(stmts) => {
                        rt.clear_error();
                        lines.clear();

                        for statement in &stmts {
                            match smol::block_on(rt.exec_async(&statement, root.clone())) {
                                Ok(Exec::Value(value)) => {
                                    if value != Val::nil() {
                                        println!("{}", value);
                                    }
                                }
                                Ok(Exec::Break) => {
                                    eprintln!("Unexpected break");
                                    break;
                                }
                                Ok(Exec::Continue) => {
                                    eprintln!("Unexpected continue");
                                    break;
                                }
                                Ok(Exec::Return(_)) => {
                                    eprintln!("Unexpected return");
                                    break;
                                }
                                Err(err) => {
                                    eprintln!("Runtime error: {:?}", err);
                                    break;
                                }
                            }
                        }
                    }
                    Err(err) if err.kind() == ParseErrorKind::UnexpectedEndOfInput => {
                        // Incomplete input, continue reading lines
                    }
                    Err(err) => {
                        lines.clear();
                        eprintln!("Parse error: {:?}", err);
                    }
                }
            }
            Err(err) if err.kind() == LexErrorKind::UnexpectedEndOfInput => {
                // Incomplete input, continue reading lines
            }
            Err(err) => {
                lines.clear();
                eprintln!("Parse error: {:?}", err);
            }
        }
    }
}
