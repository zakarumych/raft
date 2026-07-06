use std::{io::Write, rc::Rc};

use raft_ast::{lexer::LexErrorKind, parser::ParseErrorKind};
use raft_runtime::{Any, ExecControl};

fn main() {
    std::io::stdout().write_all(b"Raft REPL\n").unwrap();
    std::io::stdout().flush().unwrap();

    let mut rt = raft_runtime::Runtime::new();

    rt.set_var("print", Any::External(Rc::new(|_rt, args| {
        for arg in args {
            println!("{}", arg);
        }

        Any::nil()
    })));

    let mut lines = String::new();

    loop {
        std::io::stdout().write_all(b"> ").unwrap();
        std::io::stdout().flush().unwrap();

        if 0 == std::io::stdin().read_line(&mut lines).unwrap() {
            break;
        }

        match raft_ast::lexer::parse_str(&lines, &raft_ast::lexer::Options::wss_repl()) {
            Ok(tokens) => {
                let mut stream = raft_ast::parser::TokenStream::new(tokens);
                match raft_ast::Stmt::parse_many(&mut stream) {
                    Ok(stmts) => {
                        lines.clear();

                        for stmt in &stmts {
                            match rt.exec_stmt(&stmt) {
                                Ok(None) => {}
                                Ok(Some(ExecControl::Break)) => {
                                    eprintln!("Unexpected break");
                                    break;
                                }
                                Ok(Some(ExecControl::Continue)) => {
                                    eprintln!("Unexpected continue");
                                    break;
                                }
                                Ok(Some(ExecControl::Return(_))) => {
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
