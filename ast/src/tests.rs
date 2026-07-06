use raft_lexer::Span;

use crate::{BinaryOpKind, ExprKind, PatKind, StmtKind, UnaryOpKind, parse::TokenStream};

fn tokens_from_str(s: &str) -> TokenStream {
    let mut stream = raft_lexer::Stream::from_str(s);
    TokenStream::new(raft_lexer::parse_stream(&mut stream).unwrap())
}

#[test]
fn idents() {
    let i = tokens_from_str("foo").parse_ident().unwrap();
    assert_eq!(*i.name(), *"foo");
    assert_eq!(i.span(), Span { start: 0, end: 3 });

    assert_eq!(
        tokens_from_str("_bar").parse_ident().unwrap().name(),
        "_bar"
    );
    assert_eq!(
        tokens_from_str("foo_bar").parse_ident().unwrap().name(),
        "foo_bar"
    );
    assert_eq!(tokens_from_str("x1").parse_ident().unwrap().name(), "x1");
}

#[test]
fn atoms() {
    let a = tokens_from_str("Foo").parse_atom().unwrap();
    assert_eq!(a.name(), "Foo");
    assert_eq!(a.span(), Span { start: 0, end: 3 });

    assert_eq!(tokens_from_str("True").parse_atom().unwrap().name(), "True");
    assert_eq!(
        tokens_from_str("MyAtom").parse_atom().unwrap().name(),
        "MyAtom"
    );
}

#[test]
fn ident_not_atom() {
    assert!(tokens_from_str("Foo").parse_ident().is_err());
    assert!(tokens_from_str("foo").parse_atom().is_err());
    assert!(tokens_from_str("1x").parse_ident().is_err());
    assert!(tokens_from_str("1x").parse_atom().is_err());
}

#[test]
fn literal_int() {
    let lit = tokens_from_str("42").parse_literal().unwrap();
    let n = lit.as_number().unwrap();
    assert_eq!(n.repr(), "42");
    assert!(!n.has_dot() && !n.has_exponent());
    assert_eq!(n.integer(), "42");
    assert_eq!(n.span(), Span::new(0, 2));
}

#[test]
fn literal_float_dot() {
    let n = tokens_from_str("4.5").parse_literal().unwrap();
    let n = n.as_number().unwrap();
    assert_eq!(n.repr(), "4.5");
    assert!(n.has_dot());
    assert_eq!(n.integer(), "4");
    assert_eq!(n.fractional(), Some("5"));
}

#[test]
fn literal_float_exp() {
    let n = tokens_from_str("5e-2").parse_literal().unwrap();
    let n = n.as_number().unwrap();
    assert_eq!(n.repr(), "5e-2");
    assert!(n.has_exponent());
    assert_eq!(n.integer(), "5");
    assert_eq!(n.exponent(), Some("-2"));
}

#[test]
fn literal_float_full() {
    let n = tokens_from_str("1.0e10").parse_literal().unwrap();
    let n = n.as_number().unwrap();
    assert_eq!(n.repr(), "1.0e10");
    assert!(n.has_dot() && n.has_exponent());
    assert_eq!(n.fractional(), Some("0"));
    assert_eq!(n.exponent(), Some("10"));
}

#[test]
fn literal_char() {
    let lit = tokens_from_str("'a'").parse_literal().unwrap();
    let c = lit.as_char().unwrap();
    assert_eq!(c.repr(), "'a'");
    assert_eq!(c.unescape(), 'a');
}

#[test]
fn literal_char_escape() {
    let lit = tokens_from_str("\'\\n\'").parse_literal().unwrap();
    assert_eq!(lit.as_char().unwrap().unescape(), '\n');

    let lit = tokens_from_str("'\\t'").parse_literal().unwrap();
    assert_eq!(lit.as_char().unwrap().unescape(), '\t');

    let lit = tokens_from_str("'\\\\'").parse_literal().unwrap();
    assert_eq!(lit.as_char().unwrap().unescape(), '\\');
}

#[test]
fn literal_string() {
    let s = tokens_from_str(r#""hello""#).parse_literal().unwrap();
    let s = s.as_string().unwrap();
    assert_eq!(s.repr(), r#""hello""#);
    assert_eq!(s.unescape(), "hello");
}

#[test]
fn literal_string_escape() {
    let s = tokens_from_str(r#""foo\nbar\n""#).parse_literal().unwrap();
    assert_eq!(s.as_string().unwrap().unescape(), "foo\nbar\n");

    let s = tokens_from_str(r#""""#).parse_literal().unwrap();
    assert_eq!(s.as_string().unwrap().unescape(), "");
}

#[test]
fn literal_dot_not_accessor() {
    let mut s = tokens_from_str("1.foo");
    let lit = s.parse_literal().unwrap();
    assert_eq!(lit.as_number().unwrap().repr(), "1");
    assert_eq!(s.pos(), 1);
}

#[test]
fn unary_ops() {
    assert_eq!(
        tokens_from_str("!").parse_unary_op().unwrap().kind(),
        UnaryOpKind::Not
    );
    assert_eq!(
        tokens_from_str("~").parse_unary_op().unwrap().kind(),
        UnaryOpKind::BitNot
    );
    assert_eq!(
        tokens_from_str("-").parse_unary_op().unwrap().kind(),
        UnaryOpKind::Neg
    );
    assert_eq!(
        tokens_from_str("+").parse_unary_op().unwrap().kind(),
        UnaryOpKind::Pos
    );
    assert!(tokens_from_str("&").parse_unary_op().is_err());
}

#[test]
fn binary_ops() {
    let cases: &[(&str, BinaryOpKind, usize)] = &[
        ("&", BinaryOpKind::BitAnd, 1),
        ("|", BinaryOpKind::BitOr, 1),
        ("^", BinaryOpKind::BitXor, 1),
        ("<<", BinaryOpKind::Shl, 2),
        (">>", BinaryOpKind::Shr, 2),
        ("**", BinaryOpKind::Pow, 2),
        ("*", BinaryOpKind::Mul, 1),
        ("/", BinaryOpKind::Div, 1),
        ("+", BinaryOpKind::Add, 1),
        ("-", BinaryOpKind::Sub, 1),
        ("==", BinaryOpKind::Eq, 2),
        ("!=", BinaryOpKind::Ne, 2),
        ("<=", BinaryOpKind::Le, 2),
        (">=", BinaryOpKind::Ge, 2),
        ("<", BinaryOpKind::Lt, 1),
        (">", BinaryOpKind::Gt, 1),
    ];

    for &(src, expected_op, expected_len) in cases {
        let mut s = tokens_from_str(src);
        let sp = s.parse_binary_op().unwrap();
        assert_eq!(sp.kind(), expected_op);
        assert_eq!(s.pos(), expected_len);
    }
}

#[test]
fn precedence_ordering() {
    assert!(BinaryOpKind::BitAnd.precedence() > BinaryOpKind::Pow.precedence());
    assert!(BinaryOpKind::Pow.precedence() > BinaryOpKind::Mul.precedence());
    assert!(BinaryOpKind::Mul.precedence() > BinaryOpKind::Add.precedence());
    assert!(BinaryOpKind::Add.precedence() > BinaryOpKind::Eq.precedence());
    assert!(BinaryOpKind::Pow.is_right_assoc());
    assert!(!BinaryOpKind::Mul.is_right_assoc());
}

#[test]
fn pattern_ident() {
    let p = tokens_from_str("foo").parse_pat().unwrap();
    assert_eq!(p.span(), Span { start: 0, end: 3 });
    assert!(matches!(p.kind(), PatKind::Ident(i) if i.name() == "foo"));
}

#[test]
fn pattern_atom() {
    let p = tokens_from_str("True").parse_pat().unwrap();
    assert!(matches!(p.kind(), PatKind::Atom(a) if a.name() == "True"));
}

#[test]
fn pattern_list() {
    let p = tokens_from_str("[]").parse_pat().unwrap();
    assert!(matches!(p.kind(), PatKind::List(els) if els.is_empty()));

    let p = tokens_from_str("[a, b, c]").parse_pat().unwrap();
    let PatKind::List(els) = p.kind() else {
        panic!()
    };
    assert_eq!(els.len(), 3);
    assert!(matches!(els[0].kind(), PatKind::Ident(i) if i.name() == "a"));
    assert!(matches!(els[2].kind(), PatKind::Ident(i) if i.name() == "c"));
}

#[test]
fn pattern_record_empty() {
    let p = tokens_from_str("{}").parse_pat().unwrap();
    assert!(matches!(p.kind(), PatKind::Record(f) if f.is_empty()));
}

#[test]
fn pattern_record_shorthand() {
    let p = tokens_from_str("{ foo, bar }").parse_pat().unwrap();
    let PatKind::Record(fields) = p.kind() else {
        panic!()
    };
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].key.name(), "foo");
    assert!(fields[0].pat().is_none());
    assert_eq!(fields[1].key.name(), "bar");
}

#[test]
fn pattern_record_explicit() {
    let p = tokens_from_str("{ x: foo, y: bar }").parse_pat().unwrap();
    let PatKind::Record(fields) = p.kind() else {
        panic!()
    };
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].key.name(), "x");
    assert!(
        matches!(fields[0].pat().unwrap().kind(), PatKind::Ident(i) if i.name() == "foo")
    );
}

#[test]
fn pattern_record_nested() {
    let p = tokens_from_str("{ x: [a, b] }").parse_pat().unwrap();
    let PatKind::Record(fields) = p.kind() else {
        panic!()
    };
    assert!(matches!(
        fields[0].pat().unwrap().kind(),
        PatKind::List(_)
    ));
}

#[test]
fn expr_literal() {
    let e = tokens_from_str("42").parse_expr().unwrap();
    assert_eq!(e.span, Span::new(0, 2));
    assert!(matches!(e.kind(), ExprKind::Literal(_)));
}

#[test]
fn expr_ident() {
    let e = tokens_from_str("foo").parse_expr().unwrap();
    assert!(matches!(e.kind(), ExprKind::Ident(i) if i.name() == "foo"));
}

#[test]
fn expr_atom() {
    let e = tokens_from_str("True").parse_expr().unwrap();
    assert!(matches!(e.kind(), ExprKind::Atom(a) if a.name() == "True"));
}

#[test]
fn expr_unary() {
    let e = tokens_from_str("!a").parse_expr().unwrap();
    let ExprKind::Unary(op, inner) = e.kind() else {
        panic!()
    };
    assert_eq!(op.kind(), UnaryOpKind::Not);
    assert!(matches!(inner.kind(), ExprKind::Ident(i) if i.name() == "a"));
    assert_eq!(e.span(), Span::new(0, 2));
}

#[test]
fn expr_unary_chain() {
    // !!a = !(!a)
    let e = tokens_from_str("!!a").parse_expr().unwrap();
    let ExprKind::Unary(op, inner) = e.kind() else {
        panic!()
    };
    assert_eq!(op.kind(), UnaryOpKind::Not);
    let ExprKind::Unary(op, inner) = inner.kind() else {
        panic!()
    };
    assert_eq!(op.kind(), UnaryOpKind::Not);
    assert!(matches!(inner.kind(), ExprKind::Ident(i) if i.name() == "a"));
}

#[test]
fn expr_binary_simple() {
    let e = tokens_from_str("1 + 2").parse_expr().unwrap();
    let ExprKind::Binary(lhs, op, rhs) = e.kind() else {
        panic!()
    };
    assert_eq!(op.kind(), BinaryOpKind::Add);
    assert!(matches!(lhs.kind(), ExprKind::Literal(_)));
    assert!(matches!(rhs.kind(), ExprKind::Literal(_)));
    assert_eq!(e.span(), Span::new(0, 5));
}

#[test]
fn expr_precedence() {
    // 1 + 2 * 3 = 1 + (2 * 3)
    let e = tokens_from_str("1 + 2 * 3").parse_expr().unwrap();
    let ExprKind::Binary(lhs, op, rhs) = e.kind() else {
        panic!()
    };
    assert_eq!(op.kind(), BinaryOpKind::Add);
    assert!(matches!(lhs.kind(), ExprKind::Literal(_)));
    let ExprKind::Binary(_, inner_op, _) = &rhs.kind() else {
        panic!()
    };
    assert_eq!(inner_op.kind(), BinaryOpKind::Mul);
}

#[test]
fn expr_left_assoc() {
    // a - b - c = (a - b) - c
    let e = tokens_from_str("a - b - c").parse_expr().unwrap();
    let ExprKind::Binary(lhs, op, rhs) = e.kind() else {
        panic!()
    };
    assert_eq!(op.kind(), BinaryOpKind::Sub);
    assert!(matches!(lhs.kind(), ExprKind::Binary(_, _, _)));
    assert!(matches!(rhs.kind(), ExprKind::Ident(i) if i.name() == "c"));
}

#[test]
fn expr_right_assoc() {
    // 2 ** 3 ** 4 = 2 ** (3 ** 4)
    let e = tokens_from_str("2 ** 3 ** 4").parse_expr().unwrap();
    let ExprKind::Binary(lhs, op, rhs) = e.kind() else {
        panic!()
    };
    assert_eq!(op.kind(), BinaryOpKind::Pow);
    assert!(matches!(lhs.kind(), ExprKind::Literal(_)));
    assert!(matches!(rhs.kind(), ExprKind::Binary(_, _, _)));
}

#[test]
fn expr_apply() {
    let e = tokens_from_str("f a b").parse_expr().unwrap();
    let ExprKind::Apply(func, args) = e.kind() else {
        panic!()
    };
    assert!(matches!(func.kind(), ExprKind::Ident(i) if i.name() == "f"));
    assert_eq!(args.len(), 2);
    assert!(matches!(args[0].kind(), ExprKind::Ident(i) if i.name() == "a"));
    assert!(matches!(args[1].kind(), ExprKind::Ident(i) if i.name() == "b"));
}

#[test]
fn expr_apply_unary_arg() {
    // f !a — ! is unambiguously unary, so it's an argument
    let e = tokens_from_str("f !a").parse_expr().unwrap();
    let ExprKind::Apply(_, args) = e.kind() else {
        panic!()
    };
    assert_eq!(args.len(), 1);
    assert!(matches!(args[0].kind(), ExprKind::Unary(op, _) if op.kind() == UnaryOpKind::Not));
}

#[test]
fn expr_apply_then_binary() {
    // f a + b = (f a) + b
    let e = tokens_from_str("f a + b").parse_expr().unwrap();
    let ExprKind::Binary(lhs, op, _) = e.kind() else {
        panic!()
    };
    assert_eq!(op.kind(), BinaryOpKind::Add);
    assert!(matches!(lhs.kind(), ExprKind::Apply(_, _)));
}

#[test]
fn expr_minus_is_binary_not_arg() {
    // f - a = f minus a (NOT application)
    let e = tokens_from_str("f - a").parse_expr().unwrap();
    assert!(matches!(e.kind(), ExprKind::Binary(_, op, _) if op.kind() == BinaryOpKind::Sub));
}

#[test]
fn expr_field() {
    let e = tokens_from_str("foo.bar").parse_expr().unwrap();
    let ExprKind::Field(obj, field) = e.kind() else {
        panic!()
    };
    assert!(matches!(&obj.kind(), ExprKind::Ident(i) if i.name() == "foo"));
    assert_eq!(field.name(), "bar");
    assert_eq!(e.span, Span::new(0, 7));
}

#[test]
fn expr_index() {
    let e = tokens_from_str("arr[0]").parse_expr().unwrap();
    let ExprKind::Index(obj, _) = e.kind() else {
        panic!()
    };
    assert!(matches!(&obj.kind(), ExprKind::Ident(i) if i.name() == "arr"));
    assert_eq!(e.span, Span::new(0, 6));
}

#[test]
fn expr_chained_accessor() {
    // foo.bar[0].baz = Field(Index(Field(foo, bar), 0), baz)
    let e = tokens_from_str("foo.bar[0].baz").parse_expr().unwrap();
    let ExprKind::Field(indexed, baz) = e.kind() else {
        panic!()
    };
    assert_eq!(baz.name(), "baz");
    let ExprKind::Index(field_expr, _) = &indexed.kind() else {
        panic!()
    };
    let ExprKind::Field(root, bar) = &field_expr.kind() else {
        panic!()
    };
    assert!(matches!(&root.kind(), ExprKind::Ident(i) if i.name() == "foo"));
    assert_eq!(bar.name(), "bar");
}

#[test]
fn expr_apply_with_field_arg() {
    // f a.b = f (a.b)
    let e = tokens_from_str("f a.b").parse_expr().unwrap();
    let ExprKind::Apply(_, args) = e.kind() else {
        panic!()
    };
    assert_eq!(args.len(), 1);
    assert!(matches!(&args[0].kind(), ExprKind::Field(_, _)));
}

#[test]
fn expr_list() {
    let e = tokens_from_str("[1, 2, 3]").parse_expr().unwrap();
    let ExprKind::List(els) = e.kind() else {
        panic!()
    };
    assert_eq!(els.len(), 3);
    assert_eq!(e.span, Span::new(0, 9));
}

#[test]
fn expr_list_empty() {
    let e = tokens_from_str("[]").parse_expr().unwrap();
    assert!(matches!(e.kind(), ExprKind::List(els) if els.is_empty()));
}

#[test]
fn expr_record() {
    let e = tokens_from_str("{x: 1, y: 2}").parse_expr().unwrap();
    let ExprKind::Record(fields) = e.kind() else {
        panic!()
    };
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].key.name(), "x");
    assert!(matches!(
        fields[0].value().unwrap().kind(),
        ExprKind::Literal(_)
    ));
    assert_eq!(fields[1].key.name(), "y");
    assert!(matches!(
        fields[1].value().unwrap().kind(),
        ExprKind::Literal(_)
    ));
}

#[test]
fn expr_paren_grouping() {
    // (1 + 2) * 3 — parens override precedence
    let e = tokens_from_str("(1 + 2) * 3").parse_expr().unwrap();
    let ExprKind::Binary(lhs, op, _) = e.kind() else {
        panic!()
    };
    assert_eq!(op.kind(), BinaryOpKind::Mul);
    // lhs is the parenthesized addition; span includes parens
    assert_eq!(lhs.span, Span::new(0, 7));
    assert!(matches!(lhs.kind(), ExprKind::Binary(_, op, _) if op.kind() == BinaryOpKind::Add));
}

#[test]
fn expr_complex() {
    // a + b * c ** d / e > f * g - h ** i / j
    // = (a + ((b * (c ** d)) / e)) > ((f * g) - ((h ** i) / j))
    let e = tokens_from_str("a + b * c ** d / e > f * g - h ** i / j")
        .parse_expr()
        .unwrap();
    let ExprKind::Binary(lhs, op, rhs) = e.kind() else {
        panic!()
    };
    assert_eq!(op.kind(), BinaryOpKind::Gt);
    // lhs = a + ((b * (c**d)) / e)
    let ExprKind::Binary(_, add_op, _) = lhs.kind() else {
        panic!()
    };
    assert_eq!(add_op.kind(), BinaryOpKind::Add);
    // rhs = (f*g) - ((h**i)/j)
    let ExprKind::Binary(_, sub_op, _) = &rhs.kind() else {
        panic!()
    };
    assert_eq!(sub_op.kind(), BinaryOpKind::Sub);
}

#[test]
fn stmt_expr_statement() {
    let stmt = tokens_from_str("foo").parse_stmt().unwrap();
    match stmt.kind() {
        StmtKind::Expr(e) => {
            assert!(matches!(e.kind(), ExprKind::Ident(i) if i.name() == "foo"));
        }
        _ => panic!("expected expr stmt"),
    }
}

#[test]
fn ident_is_keyword() {
    // parse_ident should fail for keywords
    assert!(tokens_from_str("return").parse_ident().is_err());
}

#[test]
fn stmt_return() {
    let stmt = tokens_from_str("return").parse_stmt().unwrap();
    match &stmt.kind() {
        StmtKind::Return(e) => {
            assert!(matches!(e, None));
        }
        _ => panic!("expected return stmt"),
    }

    let stmt = tokens_from_str("return 5").parse_stmt().unwrap();
    match &stmt.kind() {
        StmtKind::Return(e) => {
            assert!(matches!(e.as_ref().unwrap().kind(), ExprKind::Literal(l) if l.is_number()));
        }
        _ => panic!("expected return stmt"),
    }
}

#[test]
fn stmt_if_inline() {
    let stmt = tokens_from_str("if x: y").parse_stmt().unwrap();
    match &stmt.kind() {
        StmtKind::If {
            cond,
            then_branch,
            else_branch: None,
        } => {
            assert!(matches!(&cond.kind(), ExprKind::Ident(i) if i.name() == "x"));
            assert_eq!(then_branch.len(), 1);
            match &then_branch[0].kind() {
                StmtKind::Expr(e) => {
                    assert!(matches!(e.kind(), ExprKind::Ident(i) if i.name() == "y"))
                }
                _ => panic!("expected inline expr as then-branch"),
            }
        }
        _ => panic!("expected if stmt"),
    }
}

#[test]
fn stmt_if_block() {
    let src = "if x:\n    y";
    let stmt = tokens_from_str(src).parse_stmt().unwrap();
    match &stmt.kind() {
        StmtKind::If {
            cond,
            then_branch,
            else_branch: None,
        } => {
            assert!(matches!(&cond.kind(), ExprKind::Ident(i) if i.name() == "x"));
            assert_eq!(then_branch.len(), 1);
            match &then_branch[0].kind() {
                StmtKind::Expr(e) => {
                    assert!(matches!(e.kind(), ExprKind::Ident(i) if i.name() == "y"))
                }
                _ => panic!("expected expr in block"),
            }
        }
        _ => panic!("expected if stmt"),
    }
}

#[test]
fn stmt_if_inline_else_same_line() {
    let stmt = tokens_from_str("if x: y else: z").parse_stmt().unwrap();
    match &stmt.kind() {
        StmtKind::If {
            cond,
            then_branch,
            else_branch: Some(else_branch),
        } => {
            assert!(matches!(&cond.kind(), ExprKind::Ident(i) if i.name() == "x"));
            assert_eq!(then_branch.len(), 1);
            match &then_branch[0].kind() {
                StmtKind::Expr(e) => {
                    assert!(matches!(e.kind(), ExprKind::Ident(i) if i.name() == "y"))
                }
                _ => panic!("expected inline expr as then-branch"),
            }
            assert_eq!(else_branch.len(), 1);
            match &else_branch[0].kind() {
                StmtKind::Expr(e) => {
                    assert!(matches!(e.kind(), ExprKind::Ident(i) if i.name() == "z"))
                }
                _ => panic!("expected inline expr as else-branch"),
            }
        }
        _ => panic!("expected if-else stmt"),
    }
}

#[test]
fn stmt_if_inline_else_next_line() {
    let src = "if x: y\nelse: z";
    let stmt = tokens_from_str(src).parse_stmt().unwrap();
    match &stmt.kind() {
        StmtKind::If {
            cond,
            then_branch,
            else_branch: Some(else_branch),
        } => {
            assert!(matches!(&cond.kind(), ExprKind::Ident(i) if i.name() == "x"));
            match &then_branch[0].kind() {
                StmtKind::Expr(e) => {
                    assert!(matches!(e.kind(), ExprKind::Ident(i) if i.name() == "y"))
                }
                _ => panic!("expected inline expr as then-branch"),
            }
            assert_eq!(else_branch.len(), 1);
            match &else_branch[0].kind() {
                StmtKind::Expr(e) => {
                    assert!(matches!(e.kind(), ExprKind::Ident(i) if i.name() == "z"))
                }
                _ => panic!("expected inline expr as else-branch"),
            }
        }
        _ => panic!("expected if-else stmt"),
    }
}

#[test]
fn stmt_if_block_else_block() {
    let src = "if x:\n    y\nelse:\n    z";
    let stmt = tokens_from_str(src).parse_stmt().unwrap();
    match &stmt.kind() {
        StmtKind::If {
            cond,
            then_branch,
            else_branch: Some(else_branch),
        } => {
            assert!(matches!(&cond.kind(), ExprKind::Ident(i) if i.name() == "x"));
            assert_eq!(then_branch.len(), 1);
            match &then_branch[0].kind() {
                StmtKind::Expr(e) => {
                    assert!(matches!(e.kind(), ExprKind::Ident(i) if i.name() == "y"))
                }
                _ => panic!("expected expr in block"),
            }
            assert_eq!(else_branch.len(), 1);
            match &else_branch[0].kind() {
                StmtKind::Expr(e) => {
                    assert!(matches!(e.kind(), ExprKind::Ident(i) if i.name() == "z"))
                }
                _ => panic!("expected expr in else block"),
            }
        }
        _ => panic!("expected if-else stmt"),
    }
}

#[test]
fn stmt_assign_pattern_ident() {
    let stmt = tokens_from_str("x = 1").parse_stmt().unwrap();
    match &stmt.kind() {
        StmtKind::AssignPattern { target, value } => {
            assert!(matches!(target.kind(), PatKind::Ident(id) if id.name() == "x"));
            assert!(matches!(value.kind(), ExprKind::Literal(l) if l.is_number()));
        }
        _ => panic!("expected assign pattern"),
    }
}

#[test]
fn stmt_assign_pattern_list() {
    let stmt = tokens_from_str("[a, b] = [1, 2]").parse_stmt().unwrap();
    match &stmt.kind() {
        StmtKind::AssignPattern { target, value } => {
            if let PatKind::List(items) = &target.kind() {
                assert_eq!(items.len(), 2);
                assert!(matches!(&items[0].kind(), PatKind::Ident(i) if i.name() == "a"));
                assert!(matches!(&items[1].kind(), PatKind::Ident(i) if i.name() == "b"));
            } else {
                panic!("expected list pattern")
            }
            if let ExprKind::List(vals) = &value.kind() {
                assert_eq!(vals.len(), 2);
            } else {
                panic!("expected list rhs")
            }
        }
        _ => panic!("expected assign pattern list"),
    }
}

#[test]
fn stmt_assign_field() {
    let stmt = tokens_from_str("obj.x = 5").parse_stmt().unwrap();
    match &stmt.kind() {
        StmtKind::AssignField {
            target,
            field,
            value,
        } => {
            assert_eq!(field.name(), "x");
            assert!(matches!(target.kind(), ExprKind::Ident(i) if i.name() == "obj"));
            assert!(matches!(value.kind(), ExprKind::Literal(l) if l.is_number()));
        }
        _ => panic!("expected assign field"),
    }
}

#[test]
fn stmt_assign_index() {
    let stmt = tokens_from_str("arr[0] = 7").parse_stmt().unwrap();
    match &stmt.kind() {
        StmtKind::AssignIndex {
            target,
            index,
            value,
        } => {
            assert!(matches!(target.kind(), ExprKind::Ident(i) if i.name() == "arr"));
            assert!(matches!(index.kind(), ExprKind::Literal(l) if l.is_number()));
            assert!(matches!(value.kind(), ExprKind::Literal(l) if l.is_number()));
        }
        _ => panic!("expected assign index"),
    }
}

#[test]
fn stmt_while_block_else_block() {
    let src = "while x:\n    y\nelse:\n    z";
    let stmt = tokens_from_str(src).parse_stmt().unwrap();
    match &stmt.kind() {
        StmtKind::While {
            cond,
            body: loop_branch,
            else_branch: Some(else_branch),
        } => {
            assert!(matches!(&cond.kind(), ExprKind::Ident(i) if i.name() == "x"));
            assert_eq!(loop_branch.len(), 1);
            match loop_branch[0].kind() {
                StmtKind::Expr(e) => {
                    assert!(matches!(e.kind(), ExprKind::Ident(i) if i.name() == "y"))
                }
                _ => panic!("expected expr in loop branch"),
            }
            assert_eq!(else_branch.len(), 1);
            match else_branch[0].kind() {
                StmtKind::Expr(e) => {
                    assert!(matches!(e.kind(), ExprKind::Ident(i) if i.name() == "z"))
                }
                _ => panic!("expected expr in else branch"),
            }
        }
        _ => panic!("expected while stmt with else"),
    }
}

#[test]
fn stmt_while_inline_else_next_line() {
    let src = "while x: y\nelse: z";
    let stmt = tokens_from_str(src).parse_stmt().unwrap();
    match &stmt.kind() {
        StmtKind::While {
            cond,
            body: loop_branch,
            else_branch: Some(else_branch),
        } => {
            assert!(matches!(&cond.kind(), ExprKind::Ident(i) if i.name() == "x"));
            assert_eq!(loop_branch.len(), 1);
            match &loop_branch[0].kind() {
                StmtKind::Expr(e) => {
                    assert!(matches!(e.kind(), ExprKind::Ident(i) if i.name() == "y"))
                }
                _ => panic!("expected inline expr as loop body"),
            }
            assert_eq!(else_branch.len(), 1);
            match else_branch[0].kind() {
                StmtKind::Expr(e) => {
                    assert!(matches!(e.kind(), ExprKind::Ident(i) if i.name() == "z"))
                }
                _ => panic!("expected inline expr as else branch"),
            }
        }
        _ => panic!("expected while stmt with else"),
    }
}

#[test]
fn stmt_for_block_else_block() {
    let src = "for a in arr:\n    b\nelse:\n    c";
    let stmt = tokens_from_str(src).parse_stmt().unwrap();
    match stmt.kind() {
        StmtKind::For {
            target,
            iterable,
            body: loop_branch,
            else_branch: Some(else_branch),
        } => {
            // target should be identifier pattern 'a'
            match &target.kind() {
                PatKind::Ident(id) => assert_eq!(id.name(), "a"),
                _ => panic!("expected ident pattern as for target"),
            }
            assert!(matches!(iterable.kind(), ExprKind::Ident(i) if i.name() == "arr"));
            assert_eq!(loop_branch.len(), 1);
            match &loop_branch[0].kind() {
                StmtKind::Expr(e) => {
                    assert!(matches!(e.kind(), ExprKind::Ident(i) if i.name() == "b"))
                }
                _ => panic!("expected expr in loop branch"),
            }
            assert_eq!(else_branch.len(), 1);
            match &else_branch[0].kind() {
                StmtKind::Expr(e) => {
                    assert!(matches!(e.kind(), ExprKind::Ident(i) if i.name() == "c"))
                }
                _ => panic!("expected expr in else branch"),
            }
        }
        _ => panic!("expected for stmt with else"),
    }
}

#[test]
fn stmt_for_inline_else_next_line() {
    let src = "for a in arr: b\nelse: c";
    let stmt = tokens_from_str(src).parse_stmt().unwrap();
    match stmt.kind() {
        StmtKind::For {
            target,
            iterable,
            body: loop_branch,
            else_branch: Some(else_branch),
        } => {
            match &target.kind() {
                PatKind::Ident(id) => assert_eq!(id.name(), "a"),
                _ => panic!("expected ident pattern as for target"),
            }
            assert!(matches!(iterable.kind(), ExprKind::Ident(i) if i.name() == "arr"));
            assert_eq!(loop_branch.len(), 1);
            match &loop_branch[0].kind() {
                StmtKind::Expr(e) => {
                    assert!(matches!(e.kind(), ExprKind::Ident(i) if i.name() == "b"))
                }
                _ => panic!("expected inline expr as loop body"),
            }
            assert_eq!(else_branch.len(), 1);
            match &else_branch[0].kind() {
                StmtKind::Expr(e) => {
                    assert!(matches!(e.kind(), ExprKind::Ident(i) if i.name() == "c"))
                }
                _ => panic!("expected inline expr as else branch"),
            }
        }
        _ => panic!("expected for stmt with else"),
    }
}
