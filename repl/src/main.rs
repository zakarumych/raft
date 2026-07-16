use std::{io::Write, rc::Rc};

use raft_ast::{lexer::LexErrorKind, parser::ParseErrorKind};
use raft_runtime::{Exec, Val, ValsIter};

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

    // import "name" loads ./name.raft (or ./name) as a module: the file is
    // executed once, its `export { .. }` becomes the module object, and
    // repeated imports return the cached module
    rt.register_function(
        "import",
        1, Some(1), |rt, args| {
            let mut stack = rt.stack();
            let mut popped = stack.drain_top(args);
            let name = popped.next();
            drop(popped);
            drop(stack);
            let Some(name) = name.and_then(|v| match v.unpack() {
                raft_runtime::ValEnum::String(name) => Some(name),
                _ => None,
            }) else {
                rt.set_error(raft_runtime::RuntimeError::TypeError(
                    "import expects a module name string".into(),
                ));
                return Val::nil();
            };

            let path = format!("{name}.raft");
            let source =
                std::fs::read_to_string(&path).or_else(|_| std::fs::read_to_string(&name[..]));
            let source = match source {
                Ok(source) => source,
                Err(e) => {
                    rt.set_error(raft_runtime::RuntimeError::Other(
                        format!("cannot read module '{name}': {e}").into(),
                    ));
                    return Val::nil();
                }
            };

            match rt.load_module(&name, &source) {
                Ok(module) => module,
                Err(e) => {
                    rt.set_error(e);
                    Val::nil()
                }
            }
        }
    );

    rt.register_function(
        "list",
        1, Some(1), move |rt, args| {
            assert!(args == 1);
            let mut stack = rt.stack();
            let top = stack.pop();
            let Some(mut vals_iter) = rt.try_(|| ValsIter::new(&top)) else { return Val::nil(); };
            let mut host = rt.as_host();
            let iter = vals_iter.iter(&mut host);

            Val::list(iter)
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
                        lines.clear();

                        for statement in &stmts {
                            match rt.exec_stmt(&statement, root.clone()) {
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
