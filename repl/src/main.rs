use std::{io::Write, rc::Rc};

use raft_ast::{lexer::LexErrorKind, parser::ParseErrorKind};
use raft_runtime::{Any, Exec};

fn main() {
    std::io::stdout().write_all(b"Raft REPL\n").unwrap();
    std::io::stdout().flush().unwrap();

    let mut rt = raft_runtime::Runtime::new();

    // `fn` statements compile to bytecode unless --no-vm is given;
    // top-level statements are always interpreted from the AST.
    rt.set_compile_fns(!std::env::args().any(|arg| arg == "--no-vm"));

    let quit_flag = Rc::new(std::cell::Cell::new(false));

    // print takes any number of arguments, quit takes exactly none
    rt.set_var("print", Any::host_function(0, None, |rt, args| {
        for arg in rt.vm.drain_off_stack(args).rev() {
            println!("{}", arg);
        }
        Any::nil()
    }));

    let quit_flag_clone = quit_flag.clone();
    rt.set_var("quit", Any::host_function(0, Some(0), move |_rt, _args| {
        quit_flag_clone.set(true);
        Any::nil()
    }));

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
        let stripped = lines.strip_suffix("\r\n")
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
                            match rt.exec_stmt(&statement) {
                                Ok(Exec::Value(value)) => {
                                    if value != Any::nil() {
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
