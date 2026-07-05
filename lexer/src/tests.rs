use crate::{Stream, Token, lex::Delimiter, parse_stream};

#[test]
fn ident_token() {
    let mut stream = Stream::from_str("foo");
    let tokens = parse_stream(&mut stream).expect("expected ident");
    assert_eq!(tokens.len(), 1);
    match &tokens[0] {
        Token::Ident(i) => assert_eq!(i.repr(), "foo"),
        other => panic!("unexpected token: {:?}", other),
    }
    assert!(stream.is_empty());
}

#[test]
fn punct_token() {
    let mut stream = Stream::from_str("+");
    let tokens = parse_stream(&mut stream).expect("expected punct");
    assert_eq!(tokens.len(), 1);
    match &tokens[0] {
        Token::Punct(p) => assert_eq!(p.repr(), '+'),
        other => panic!("unexpected token: {:?}", other),
    }
    assert!(stream.is_empty());
}

#[test]
fn number_literal_token() {
    let mut stream = Stream::from_str("123");
    let tokens = parse_stream(&mut stream).expect("expected number");
    assert_eq!(tokens.len(), 1);
    match &tokens[0] {
        Token::Literal(l) => {
            assert!(l.is_number());
            assert_eq!(l.as_number().unwrap().value(), "123");
        }
        other => panic!("unexpected token: {:?}", other),
    }
    assert!(stream.is_empty());
}

#[test]
fn string_literal_token() {
    let mut stream = Stream::from_str("\"hi\"");
    let tokens = parse_stream(&mut stream).expect("expected string");
    assert_eq!(tokens.len(), 1);
    match &tokens[0] {
        Token::Literal(l) => {
            assert!(l.is_string());
            assert_eq!(l.as_string().unwrap().repr(), "\"hi\"");
        }
        other => panic!("unexpected token: {:?}", other),
    }
    assert!(stream.is_empty());
}

#[test]
fn char_literal_token() {
    let mut stream = Stream::from_str("'a'");
    let tokens = parse_stream(&mut stream).expect("expected char");
    assert_eq!(tokens.len(), 1);
    match &tokens[0] {
        Token::Literal(l) => {
            assert!(l.is_char());
            assert_eq!(l.as_char().unwrap().repr(), "'a'");
        }
        other => panic!("unexpected token: {:?}", other),
    }
    assert!(stream.is_empty());
}

#[test]
fn block_comment_token() {
    let mut stream = Stream::from_str("/*c*/");
    let tokens = parse_stream(&mut stream).expect("expected comment");
    assert_eq!(tokens.len(), 1);
    match &tokens[0] {
        Token::Comment(c) => assert!(c.repr().starts_with("/*c*/")),
        other => panic!("unexpected token: {:?}", other),
    }
    assert!(stream.is_empty());
}

#[test]
fn group_token() {
    let mut stream = Stream::from_str("(a)");
    let tokens = parse_stream(&mut stream).expect("expected group");
    assert_eq!(tokens.len(), 1);
    match &tokens[0] {
        Token::Group(g) => {
            assert_eq!(g.delimiter(), Delimiter::Parenthesis);
            let inner = g.tokens();
            assert_eq!(inner.len(), 1);
            match &inner[0] {
                Token::Ident(i) => assert_eq!(i.repr(), "a"),
                other => panic!("unexpected inner token: {:?}", other),
            }
        }
        other => panic!("unexpected token: {:?}", other),
    }
    assert!(stream.is_empty());
}

#[test]
fn group_multiline_token() {
    let mut stream = Stream::from_str("(\n\na\n)");
    let tokens = parse_stream(&mut stream).expect("expected group");
    assert_eq!(tokens.len(), 1);
    match &tokens[0] {
        Token::Group(g) => {
            assert_eq!(g.delimiter(), Delimiter::Parenthesis);
            let inner = g.tokens();
            assert_eq!(inner.len(), 3);
            match &inner[0] {
                Token::Newline(_) => {}
                other => panic!("unexpected inner token: {:?}", other),
            }
            match &inner[1] {
                Token::Ident(i) => assert_eq!(i.repr(), "a"),
                other => panic!("unexpected inner token: {:?}", other),
            }
            match &inner[2] {
                Token::Newline(_) => {}
                other => panic!("unexpected inner token: {:?}", other),
            }
        }
        other => panic!("unexpected token: {:?}", other),
    }
    assert!(stream.is_empty());
}

#[test]
fn multiline_parse() {
    let mut stream = Stream::from_str("a\nb\n+");
    let tokens = parse_stream(&mut stream).expect("expected block group");
    assert_eq!(tokens.len(), 5);
    match &tokens[0] {
        Token::Ident(i) => assert_eq!(i.repr(), "a"),
        other => panic!("unexpected token: {:?}", other),
    }
    match &tokens[1] {
        Token::Newline(_) => {}
        other => panic!("unexpected token: {:?}", other),
    }
    match &tokens[2] {
        Token::Ident(i) => assert_eq!(i.repr(), "b"),
        other => panic!("unexpected token: {:?}", other),
    }
    match &tokens[3] {
        Token::Newline(_) => {}
        other => panic!("unexpected token: {:?}", other),
    }
    match &tokens[4] {
        Token::Punct(i) => assert_eq!(i.repr(), '+'),
        other => panic!("unexpected token: {:?}", other),
    }
}

#[test]
fn block_group_parse() {
    let mut stream = Stream::from_str("a\n  b\n    c\n  d\n+");
    let tokens = parse_stream(&mut stream).expect("expected block group");

    assert_eq!(tokens.len(), 4);
    match &tokens[0] {
        Token::Ident(i) => assert_eq!(i.repr(), "a"),
        other => panic!("unexpected token: {:?}", other),
    }
    match &tokens[1] {
        Token::Newline(_) => {}
        other => panic!("unexpected token: {:?}", other),
    }
    match &tokens[2] {
        Token::Group(group) => {
            assert_eq!(group.delimiter(), Delimiter::Block);
            let group_tokens = group.tokens();
            assert_eq!(group_tokens.len(), 5);
            match &group_tokens[0] {
                Token::Ident(i) => assert_eq!(i.repr(), "b"),
                other => panic!("unexpected token: {:?}", other),
            }
            match &group_tokens[1] {
                Token::Newline(_) => {}
                other => panic!("unexpected token: {:?}", other),
            }
            match &group_tokens[2] {
                Token::Group(nested) => {
                    assert_eq!(nested.delimiter(), Delimiter::Block);
                    let nested_tokens = nested.tokens();
                    assert_eq!(nested_tokens.len(), 2);
                    match &nested_tokens[0] {
                        Token::Ident(i) => assert_eq!(i.repr(), "c"),
                        other => panic!("unexpected token: {:?}", other),
                    }
                    match &nested_tokens[1] {
                        Token::Newline(_) => {}
                        other => panic!("unexpected token: {:?}", other),
                    }
                }
                other => panic!("unexpected token: {:?}", other),
            }
            match &group_tokens[3] {
                Token::Ident(i) => assert_eq!(i.repr(), "d"),
                other => panic!("unexpected token: {:?}", other),
            }
            match &group_tokens[4] {
                Token::Newline(_) => {}
                other => panic!("unexpected token: {:?}", other),
            }
        }
        other => panic!("unexpected token: {:?}", other),
    }
    match &tokens[3] {
        Token::Punct(i) => assert_eq!(i.repr(), '+'),
        other => panic!("unexpected token: {:?}", other),
    }
}

#[test]
fn group_close_in_outer_block_error() {
    let mut stream = Stream::from_str("  {\n}");
    match parse_stream(&mut stream) {
        Err(err) => assert_eq!(err.kind(), crate::lex::LexErrorKind::UnclosedDelimiter),
        Ok(_) => panic!("expected error, got Ok"),
    }
}

#[test]
fn group_close_in_inner_block_error() {
    let mut stream = Stream::from_str("\n  {\n    }\n");
    match parse_stream(&mut stream) {
        Err(err) => assert_eq!(err.kind(), crate::lex::LexErrorKind::UnclosedDelimiter),
        Ok(_) => panic!("expected error, got Ok"),
    }
}

#[test]
fn groups_mixed_nested_parse() {
    let mut stream = Stream::from_str("(a)\n[b\n  {c\n  }\n]\n{d}\n");
    let tokens = parse_stream(&mut stream).expect("expected block group");

    assert_eq!(tokens.len(), 6);
    match &tokens[0] {
        Token::Group(g) => {
            assert_eq!(g.delimiter(), Delimiter::Parenthesis);
            let ti = g.tokens();
            assert_eq!(ti.len(), 1);
            match &ti[0] {
                Token::Ident(i) => assert_eq!(i.repr(), "a"),
                other => panic!("unexpected token: {:?}", other),
            }
        }
        other => panic!("unexpected token: {:?}", other),
    }
    match &tokens[1] {
        Token::Newline(_) => {}
        other => panic!("unexpected token: {:?}", other),
    }
    match &tokens[2] {
        Token::Group(g) => {
            assert_eq!(g.delimiter(), Delimiter::Bracket);
            let tokens = g.tokens();
            assert_eq!(tokens.len(), 3);
            match &tokens[0] {
                Token::Ident(i) => assert_eq!(i.repr(), "b"),
                other => panic!("unexpected token: {:?}", other),
            }
            match &tokens[1] {
                Token::Newline(_) => {}
                other => panic!("unexpected token: {:?}", other),
            }
            match &tokens[2] {
                Token::Group(g) => {
                    assert_eq!(g.delimiter(), Delimiter::Block);
                    let tokens = g.tokens();
                    assert_eq!(tokens.len(), 2);
                    match &tokens[0] {
                        Token::Group(g) => {
                            assert_eq!(g.delimiter(), Delimiter::Brace);
                            let tokens = g.tokens();
                            assert_eq!(tokens.len(), 2);
                            match &tokens[0] {
                                Token::Ident(i) => assert_eq!(i.repr(), "c"),
                                other => panic!("unexpected token: {:?}", other),
                            }
                            match &tokens[1] {
                                Token::Newline(_) => {}
                                other => panic!("unexpected token: {:?}", other),
                            }
                        }
                        other => panic!("unexpected token: {:?}", other),
                    }
                    match &tokens[1] {
                        Token::Newline(_) => {}
                        other => panic!("unexpected token: {:?}", other),
                    }
                }
                other => panic!("unexpected token: {:?}", other),
            }
        }
        other => panic!("unexpected token: {:?}", other),
    }
    match &tokens[3] {
        Token::Newline(_) => {}
        other => panic!("unexpected token: {:?}", other),
    }
    match &tokens[4] {
        Token::Group(g) => {
            assert_eq!(g.delimiter(), Delimiter::Brace);
            let ti = g.tokens();
            assert_eq!(ti.len(), 1);
            match &ti[0] {
                Token::Ident(i) => assert_eq!(i.repr(), "d"),
                other => panic!("unexpected token: {:?}", other),
            }
        }
        other => panic!("unexpected token: {:?}", other),
    }
    match &tokens[5] {
        Token::Newline(_) => {}
        other => panic!("unexpected token: {:?}", other),
    }
}
