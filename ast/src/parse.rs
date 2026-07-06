use std::rc::Rc;

use crate::{
    Atom, BinaryOp, BinaryOpKind, Expr, ExprKind, ExprRecordField, Ident, Literal, Pat, PatKind,
    PatRecordField, Span, Stmt, StmtKind, UnaryOp, UnaryOpKind,
};

pub use raft_lexer::{Token, Stream, parse_stream};

#[derive(Debug)]
pub enum ParseError {
    UnexpectedToken(Span),
    UnexpectedEnd(Span),
    InvalidAssignmentTarget(Span),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Keyword {
    Return,
    Break,
    Continue,
    If,
    Else,
    While,
    For,
    In,
}

impl Keyword {
    pub fn from_ident(ident: &str) -> Option<Self> {
        match ident {
            "return" => Some(Keyword::Return),
            "break" => Some(Keyword::Break),
            "continue" => Some(Keyword::Continue),
            "if" => Some(Keyword::If),
            "else" => Some(Keyword::Else),
            "while" => Some(Keyword::While),
            "for" => Some(Keyword::For),
            "in" => Some(Keyword::In),
            _ => None,
        }
    }
}

pub type ParseResult<T> = Result<T, ParseError>;

// TokenStream: wrapper over token list produced by lexer
pub struct TokenStream {
    tokens: Rc<[Token]>,
    pos: usize,
}

impl TokenStream {
    pub fn new(tokens: Rc<[Token]>) -> Self {
        TokenStream { tokens, pos: 0 }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn parse_ident(&mut self) -> ParseResult<Ident> {
        Ident::parse(self)
    }

    pub fn parse_atom(&mut self) -> ParseResult<Atom> {
        Atom::parse(self)
    }

    pub fn parse_literal(&mut self) -> ParseResult<Literal> {
        Literal::parse(self)
    }

    pub fn parse_unary_op(&mut self) -> ParseResult<UnaryOp> {
        UnaryOp::parse(self)
    }

    pub fn parse_binary_op(&mut self) -> ParseResult<BinaryOp> {
        BinaryOp::parse(self)
    }

    pub fn parse_pat(&mut self) -> ParseResult<Pat> {
        Pat::parse(self)
    }

    pub fn parse_expr(&mut self) -> ParseResult<Expr> {
        Expr::parse(self)
    }

    pub fn parse_stmt(&mut self) -> ParseResult<Stmt> {
        Stmt::parse(self)
    }

    fn start_span(&self) -> Span {
        if self.pos == self.tokens.len() {
            self.end_span()
        } else {
            self.tokens[self.pos].span().start()
        }
    }

    fn end_span(&self) -> Span {
        match self.tokens.last() {
            None => Span { start: 0, end: 0 },
            Some(tok) => tok.span().end(),
        }
    }

    fn span(&self) -> Span {
        if self.pos == self.tokens.len() {
            self.end_span()
        } else {
            let start = self.tokens[self.pos].span().start;
            let end = self.tokens.last().unwrap().span().end;
            Span { start, end }
        }
    }

    fn is_empty(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    fn peek(&self) -> Option<Token> {
        self.tokens.get(self.pos).cloned()
    }

    fn peek1(&self) -> Option<Token> {
        self.tokens.get(self.pos + 1).cloned()
    }

    fn advance(&mut self) {
        self.pos += 1;
    }

    fn skip_newlines(&mut self) -> bool {
        if let Some(Token::Newline(_)) = self.peek() {
            self.advance();
            true
        } else {
            false
        }
    }

    fn skip_comments(&mut self) -> bool {
        if let Some(Token::Comment(_)) = self.peek() {
            self.advance();
            true
        } else {
            false
        }
    }

    fn skip_comments_and_newlines(&mut self) -> bool {
        let mut skipped = false;
        while let Some(tok) = self.peek() {
            match tok {
                Token::Comment(_) | Token::Newline(_) => {
                    self.advance();
                    skipped = true;
                }
                _ => break,
            }
        }
        skipped
    }

    fn expect_end(&self) -> ParseResult<()> {
        if let Some(tok) = self.peek() {
            Err(ParseError::UnexpectedToken(tok.span()))
        } else {
            Ok(())
        }
    }
}

impl Ident {
    // fn peek(stream: &TokenStream) -> Option<Ident> {
    //     match stream.peek() {
    //         Some(Token::Ident(i)) => {
    //             if Keyword::from_ident(i.repr()).is_some() {
    //                 return None;
    //             }

    //             let s = i.repr();
    //             if s.chars().next().map(|c| !c.is_uppercase()).unwrap_or(false) {
    //                 let name = i.rc_repr();
    //                 let span = i.span();
    //                 Some(Ident { name, span })
    //             } else {
    //                 None
    //             }
    //         }
    //         _ => None,
    //     }
    // }

    fn parse(stream: &mut TokenStream) -> ParseResult<Ident> {
        match stream.peek() {
            Some(Token::Ident(i)) => {
                if Keyword::from_ident(i.repr()).is_some() {
                    return Err(ParseError::UnexpectedToken(i.span()));
                }


                let s = i.repr();
                if s.chars().next().map(|c| !c.is_uppercase()).unwrap_or(false) {
                    stream.advance();
                    let name = i.rc_repr();
                    let span = i.span();
                    Ok(Ident { name, span })
                } else {
                    Err(ParseError::UnexpectedToken(i.span()))
                }
            }
            Some(tok) => Err(ParseError::UnexpectedToken(tok.span())),
            None => Err(ParseError::UnexpectedEnd(stream.start_span())),
        }
    }
}

impl Atom {
    // fn peek(stream: &TokenStream) -> Option<Atom> {
    //     match stream.peek() {
    //         Some(Token::Ident(i)) => {
    //             if Keyword::from_ident(i.repr()).is_some() {
    //                 return None;
    //             }

    //             let s = i.repr();
    //             if s.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
    //                 let name = i.rc_repr();
    //                 let span = i.span();
    //                 Some(Atom { name, span })
    //             } else {
    //                 None
    //             }
    //         }
    //         _ => None,
    //     }
    // }

    fn parse(stream: &mut TokenStream) -> ParseResult<Atom> {
        match stream.peek() {
            Some(Token::Ident(i)) => {
                if Keyword::from_ident(i.repr()).is_some() {
                    return Err(ParseError::UnexpectedToken(i.span()));
                }

                let s = i.repr();
                if s.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                    stream.advance();
                    let name = i.rc_repr();
                    let span = i.span();
                    Ok(Atom { name, span })
                } else {
                    Err(ParseError::UnexpectedToken(i.span()))
                }
            }
            Some(tok) => Err(ParseError::UnexpectedToken(tok.span())),
            None => Err(ParseError::UnexpectedEnd(stream.start_span())),
        }
    }
}

impl Literal {
    // fn peek(stream: &TokenStream) -> Option<Literal> {
    //     match stream.peek() {
    //         Some(Token::Literal(l)) => {
    //             let lit = l.clone();
    //             match lit {
    //                 raft_lexer::Literal::Number(n) => Some(Literal::Number(n)),
    //                 raft_lexer::Literal::Char(c) => Some(Literal::Char(c)),
    //                 raft_lexer::Literal::String(s) => Some(Literal::String(s)),
    //             }
    //         }
    //         _ => None,
    //     }
    // }

    fn parse(stream: &mut TokenStream) -> ParseResult<Literal> {
        match stream.peek() {
            Some(Token::Literal(l)) => {
                stream.advance();
                let lit = l.clone();
                match lit {
                    raft_lexer::Literal::Number(n) => Ok(Literal::Number(n)),
                    raft_lexer::Literal::Char(c) => Ok(Literal::Char(c)),
                    raft_lexer::Literal::String(s) => Ok(Literal::String(s)),
                }
            }
            Some(tok) => Err(ParseError::UnexpectedToken(tok.span())),
            None => Err(ParseError::UnexpectedEnd(stream.start_span())),
        }
    }
}

impl UnaryOp {
    fn peek(stream: &TokenStream) -> Option<UnaryOp> {
        match stream.peek() {
            Some(Token::Punct(p)) => {
                let ch = p.repr();
                let kind = match ch {
                    '!' => Some(UnaryOpKind::Not),
                    '~' => Some(UnaryOpKind::BitNot),
                    '+' => Some(UnaryOpKind::Pos),
                    '-' => Some(UnaryOpKind::Neg),
                    _ => None,
                };
                if let Some(k) = kind {
                    let span = p.span();
                    Some(UnaryOp { kind: k, span })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn parse(stream: &mut TokenStream) -> ParseResult<UnaryOp> {
        match stream.peek() {
            Some(Token::Punct(p)) => {
                let ch = p.repr();
                let kind = match ch {
                    '!' => Some(UnaryOpKind::Not),
                    '~' => Some(UnaryOpKind::BitNot),
                    '+' => Some(UnaryOpKind::Pos),
                    '-' => Some(UnaryOpKind::Neg),
                    _ => None,
                };
                if let Some(k) = kind {
                    let span = p.span();
                    stream.advance();
                    Ok(UnaryOp { kind: k, span })
                } else {
                    Err(ParseError::UnexpectedToken(p.span()))
                }
            }
            Some(tok) => Err(ParseError::UnexpectedToken(tok.span())),
            None => Err(ParseError::UnexpectedEnd(stream.start_span())),
        }
    }
}

impl BinaryOp {
    fn peek(stream: &TokenStream) -> Option<BinaryOp> {
        match stream.peek() {
            Some(Token::Punct(p1)) => match (p1.repr(), stream.peek1()) {
                ('<', Some(Token::Punct(p2))) if p2.repr() == '<' => {
                    return Some(BinaryOp {
                        kind: BinaryOpKind::Shl,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('>', Some(Token::Punct(p2))) if p2.repr() == '>' => {
                    return Some(BinaryOp {
                        kind: BinaryOpKind::Shr,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('*', Some(Token::Punct(p2))) if p2.repr() == '*' => {
                    return Some(BinaryOp {
                        kind: BinaryOpKind::Pow,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('=', Some(Token::Punct(p2))) if p2.repr() == '=' => {
                    return Some(BinaryOp {
                        kind: BinaryOpKind::Eq,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('!', Some(Token::Punct(p2))) if p2.repr() == '=' => {
                    return Some(BinaryOp {
                        kind: BinaryOpKind::Ne,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('<', Some(Token::Punct(p2))) if p2.repr() == '=' => {
                    return Some(BinaryOp {
                        kind: BinaryOpKind::Le,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('>', Some(Token::Punct(p2))) if p2.repr() == '=' => {
                    return Some(BinaryOp {
                        kind: BinaryOpKind::Ge,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('&', _) => {
                    return Some(BinaryOp {
                        kind: BinaryOpKind::BitAnd,
                        span: p1.span(),
                    });
                }
                ('|', _) => {
                    return Some(BinaryOp {
                        kind: BinaryOpKind::BitOr,
                        span: p1.span(),
                    });
                }
                ('^', _) => {
                    return Some(BinaryOp {
                        kind: BinaryOpKind::BitXor,
                        span: p1.span(),
                    });
                }
                ('*', _) => {
                    return Some(BinaryOp {
                        kind: BinaryOpKind::Mul,
                        span: p1.span(),
                    });
                }
                ('/', _) => {
                    return Some(BinaryOp {
                        kind: BinaryOpKind::Div,
                        span: p1.span(),
                    });
                }
                ('+', _) => {
                    return Some(BinaryOp {
                        kind: BinaryOpKind::Add,
                        span: p1.span(),
                    });
                }
                ('-', _) => {
                    return Some(BinaryOp {
                        kind: BinaryOpKind::Sub,
                        span: p1.span(),
                    });
                }
                ('<', _) => {
                    return Some(BinaryOp {
                        kind: BinaryOpKind::Lt,
                        span: p1.span(),
                    });
                }
                ('>', _) => {
                    return Some(BinaryOp {
                        kind: BinaryOpKind::Gt,
                        span: p1.span(),
                    });
                }
                _ => None,
            },
            _ => None,
        }
    }

    fn parse(stream: &mut TokenStream) -> ParseResult<BinaryOp> {
        match stream.peek() {
            Some(Token::Punct(p1)) => match (p1.repr(), stream.peek1()) {
                ('<', Some(Token::Punct(p2))) if p2.repr() == '<' => {
                    stream.advance();
                    stream.advance();
                    return Ok(BinaryOp {
                        kind: BinaryOpKind::Shl,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('>', Some(Token::Punct(p2))) if p2.repr() == '>' => {
                    stream.advance();
                    stream.advance();
                    return Ok(BinaryOp {
                        kind: BinaryOpKind::Shr,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('*', Some(Token::Punct(p2))) if p2.repr() == '*' => {
                    stream.advance();
                    stream.advance();
                    return Ok(BinaryOp {
                        kind: BinaryOpKind::Pow,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('=', Some(Token::Punct(p2))) if p2.repr() == '=' => {
                    stream.advance();
                    stream.advance();
                    return Ok(BinaryOp {
                        kind: BinaryOpKind::Eq,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('!', Some(Token::Punct(p2))) if p2.repr() == '=' => {
                    stream.advance();
                    stream.advance();
                    return Ok(BinaryOp {
                        kind: BinaryOpKind::Ne,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('<', Some(Token::Punct(p2))) if p2.repr() == '=' => {
                    stream.advance();
                    stream.advance();
                    return Ok(BinaryOp {
                        kind: BinaryOpKind::Le,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('>', Some(Token::Punct(p2))) if p2.repr() == '=' => {
                    stream.advance();
                    stream.advance();
                    return Ok(BinaryOp {
                        kind: BinaryOpKind::Ge,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('&', _) => {
                    stream.advance();
                    return Ok(BinaryOp {
                        kind: BinaryOpKind::BitAnd,
                        span: p1.span(),
                    });
                }
                ('|', _) => {
                    stream.advance();
                    return Ok(BinaryOp {
                        kind: BinaryOpKind::BitOr,
                        span: p1.span(),
                    });
                }
                ('^', _) => {
                    stream.advance();
                    return Ok(BinaryOp {
                        kind: BinaryOpKind::BitXor,
                        span: p1.span(),
                    });
                }
                ('*', _) => {
                    stream.advance();
                    return Ok(BinaryOp {
                        kind: BinaryOpKind::Mul,
                        span: p1.span(),
                    });
                }
                ('/', _) => {
                    stream.advance();
                    return Ok(BinaryOp {
                        kind: BinaryOpKind::Div,
                        span: p1.span(),
                    });
                }
                ('+', _) => {
                    stream.advance();
                    return Ok(BinaryOp {
                        kind: BinaryOpKind::Add,
                        span: p1.span(),
                    });
                }
                ('-', _) => {
                    stream.advance();
                    return Ok(BinaryOp {
                        kind: BinaryOpKind::Sub,
                        span: p1.span(),
                    });
                }
                ('<', _) => {
                    stream.advance();
                    return Ok(BinaryOp {
                        kind: BinaryOpKind::Lt,
                        span: p1.span(),
                    });
                }
                ('>', _) => {
                    stream.advance();
                    return Ok(BinaryOp {
                        kind: BinaryOpKind::Gt,
                        span: p1.span(),
                    });
                }
                (_, _) => {
                    return Err(ParseError::UnexpectedToken(p1.span()));
                }
            },
            Some(tok) => Err(ParseError::UnexpectedToken(tok.span())),
            None => Err(ParseError::UnexpectedEnd(stream.start_span())),
        }
    }
}

impl Expr {
    pub fn parse(stream: &mut TokenStream) -> ParseResult<Self> {
        Self::parse_binary(stream, 0)
    }

    fn parse_binary(stream: &mut TokenStream, min_prec: u8) -> ParseResult<Self> {
        let mut lhs = Self::parse_application(stream)?;
        loop {
            match BinaryOp::peek(stream) {
                Some(op) => {
                    if op.kind.precedence() > min_prec {
                        for _ in 0..op.kind.token_size() {
                            stream.advance();
                        }
                        let rhs_min = if op.kind.is_right_assoc() {
                            op.kind.precedence() - 1
                        } else {
                            op.kind.precedence()
                        };

                        let rhs = Self::parse_binary(stream, rhs_min)?;
                        let span = Span {
                            start: lhs.span.start,
                            end: rhs.span.end,
                        };
                        lhs = Expr {
                            kind: ExprKind::Binary(Rc::new(lhs), op, Rc::new(rhs)),
                            span,
                        };
                    } else {
                        break;
                    }
                }
                // Some(_) => break,
                None => break,
            }
        }
        Ok(lhs)
    }

    fn parse_application(stream: &mut TokenStream) -> ParseResult<Self> {
        let expr = Self::parse_unary(stream)?;
        let mut args = Vec::new();
        loop {
            match stream.peek() {
                Some(Token::Punct(_)) => {
                    if BinaryOp::peek(stream).is_some() {
                        break;
                    }
                    if UnaryOp::peek(stream).is_none() {
                        break;
                    }
                }
                Some(Token::Ident(ident)) => {
                    match Keyword::from_ident(ident.repr()) {
                        Some(_) => break,
                        None => {}
                    }
                }
                Some(Token::Literal(_)) => {}
                Some(Token::Group(g)) if g.delimiter() == raft_lexer::Delimiter::Block => break,
                Some(Token::Group(_)) => {}
                Some(Token::Comment(_)) => {
                    stream.advance();
                    continue;
                }
                Some(Token::Newline(_)) => break,
                None => break,
            };

            match Self::parse_unary(stream) {
                Ok(arg) => {
                    args.push(arg);
                }
                Err(_) => break,
            }
        }

        if args.is_empty() {
            Ok(expr)
        } else {
            let span = Span {
                start: expr.span.start,
                end: args.last().unwrap().span.end,
            };
            Ok(Expr {
                kind: ExprKind::Apply(Rc::new(expr), Rc::from(args)),
                span,
            })
        }
    }

    fn parse_unary(stream: &mut TokenStream) -> ParseResult<Self> {
        if let Ok(op) = UnaryOp::parse(stream) {
            let operand = Self::parse_unary(stream)?;
            let span = op.span().join(operand.span());
            Ok(Expr {
                kind: ExprKind::Unary(op, Rc::new(operand)),
                span,
            })
        } else {
            Self::parse_accessor(stream)
        }
    }

    fn parse_accessor(stream: &mut TokenStream) -> ParseResult<Self> {
        let mut expr = Self::parse_primary(stream)?;
        loop {
            match stream.peek() {
                Some(Token::Punct(p)) if p.repr() == '.' => {
                    stream.advance();

                    let field = Ident::parse(stream)?;
                    let span = expr.span().join(field.span());
                    expr = Expr {
                        kind: ExprKind::Field(Rc::new(expr), field),
                        span,
                    };
                }
                Some(Token::Group(g))
                    if matches!(g.delimiter(), raft_lexer::Delimiter::Bracket) =>
                {
                    stream.advance();

                    let mut group_stream = TokenStream::new(g.rc_tokens());

                    let span = expr.span().join(g.span());
                    let index = Expr::parse(&mut group_stream)?;
                    group_stream.skip_comments_and_newlines();
                    group_stream.expect_end()?;

                    expr = Expr {
                        kind: ExprKind::Index(Rc::new(expr), Rc::new(index)),
                        span,
                    };
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_primary(stream: &mut TokenStream) -> ParseResult<Self> {
        match stream.peek() {
            Some(Token::Group(g)) if g.delimiter() == raft_lexer::Delimiter::Parenthesis => {
                let mut group_stream = TokenStream::new(g.rc_tokens());
                group_stream.skip_comments_and_newlines();

                let mut expr = Expr::parse(&mut group_stream)?;
                group_stream.skip_comments_and_newlines();
                group_stream.expect_end()?;

                expr.span = g.span();
                stream.advance();
                Ok(expr)
            }
            Some(Token::Group(g)) if g.delimiter() == raft_lexer::Delimiter::Bracket => {
                let mut group_stream = TokenStream::new(g.rc_tokens());
                group_stream.skip_comments_and_newlines();

                let mut items = Vec::new();
                while group_stream.peek().is_some() {
                    let e = Expr::parse(&mut group_stream)?;
                    group_stream.skip_comments_and_newlines();

                    items.push(e);
                    match group_stream.peek() {
                        Some(Token::Punct(p)) if p.repr() == ',' => {
                            group_stream.advance();
                            group_stream.skip_comments_and_newlines();
                            continue;
                        }
                        Some(tok) => return Err(ParseError::UnexpectedToken(tok.span())),
                        None => break,
                    }
                }

                stream.advance();
                Ok(Expr {
                    kind: ExprKind::List(Rc::from(items)),
                    span: g.span(),
                })
            }
            Some(Token::Group(g)) if g.delimiter() == raft_lexer::Delimiter::Brace => {
                let mut group_stream = TokenStream::new(g.rc_tokens());
                group_stream.skip_comments_and_newlines();

                let mut fields = Vec::new();
                while group_stream.peek().is_some() {
                    let key = Ident::parse(&mut group_stream)?;
                    group_stream.skip_comments_and_newlines();

                    match group_stream.peek() {
                        Some(Token::Punct(p)) if p.repr() == ':' => {
                            group_stream.advance();
                            group_stream.skip_comments_and_newlines();

                            let value = Expr::parse(&mut group_stream)?;
                            group_stream.skip_comments_and_newlines();

                            let field_span = key.span.join(value.span);
                            fields.push(ExprRecordField {
                                key,
                                value: Some(value),
                                span: field_span,
                            });
                            match group_stream.peek() {
                                Some(Token::Punct(p)) if p.repr() == ',' => {
                                    group_stream.advance();
                                    group_stream.skip_comments_and_newlines();
                                    continue;
                                }
                                Some(tok) => {
                                    return Err(ParseError::UnexpectedToken(tok.span()));
                                }
                                None => break,
                            }
                        }
                        Some(Token::Punct(p)) if p.repr() == ',' => {
                            group_stream.advance();
                            group_stream.skip_comments_and_newlines();

                            let key_span = key.span;
                            fields.push(ExprRecordField {
                                key: key.clone(),
                                value: None,
                                span: key_span,
                            });

                            continue;
                        }
                        Some(tok) => {
                            return Err(ParseError::UnexpectedToken(tok.span()));
                        }
                        None => {
                            let key_span = key.span;
                            fields.push(ExprRecordField {
                                key: key.clone(),
                                value: None,
                                span: key_span,
                            });
                            break;
                        }
                    }
                }

                stream.advance();
                Ok(Expr {
                    kind: ExprKind::Record(Rc::from(fields)),
                    span: g.span(),
                })
            }
            Some(Token::Literal(l)) => {
                stream.advance();

                let lit = l.clone();
                let span = lit.span();
                Ok(Expr {
                    kind: ExprKind::Literal(match lit {
                        raft_lexer::Literal::Number(n) => Literal::Number(n),
                        raft_lexer::Literal::Char(c) => Literal::Char(c),
                        raft_lexer::Literal::String(s) => Literal::String(s),
                    }),
                    span,
                })
            }
            Some(Token::Ident(i)) => {
                stream.advance();

                let span = i.span();
                match i.repr().chars().next().unwrap() {
                    ch if ch.is_uppercase() => {
                        let a = Atom {
                            name: i.rc_repr(),
                            span,
                        };
                        Ok(Expr {
                            kind: ExprKind::Atom(a),
                            span,
                        })
                    }
                    _ch => {
                        let id = Ident {
                            name: i.rc_repr(),
                            span,
                        };
                        Ok(Expr {
                            kind: ExprKind::Ident(id),
                            span,
                        })
                    }
                }
            }
            Some(tok) => Err(ParseError::UnexpectedToken(tok.span())),
            None => Err(ParseError::UnexpectedEnd(stream.span())),
        }
    }
}

fn peek_keyword(stream: &mut TokenStream) -> Option<(Keyword, Span)> {
    match stream.peek() {
        Some(Token::Ident(i)) => {
            let s = i.repr();
            let kw = match s {
                "return" => Some(Keyword::Return),
                "if" => Some(Keyword::If),
                "else" => Some(Keyword::Else),
                "while" => Some(Keyword::While),
                "for" => Some(Keyword::For),
                "break" => Some(Keyword::Break),
                "continue" => Some(Keyword::Continue),
                "in" => Some(Keyword::In),
                _ => None,
            };
            if let Some(k) = kw {
                let span = i.span();
                return Some((k, span));
            }
            None
        }
        _ => None,
    }
}

fn expr_to_pattern(expr: &Expr) -> Option<Pat> {
    match &expr.kind {
        ExprKind::Ident(i) => Some(Pat {
            span: i.span,
            kind: PatKind::Ident(i.clone()),
        }),
        ExprKind::Atom(a) => Some(Pat {
            span: a.span,
            kind: PatKind::Atom(a.clone()),
        }),
        ExprKind::Literal(l) => Some(Pat {
            span: l.span(),
            kind: PatKind::Literal(l.clone()),
        }),
        ExprKind::List(items) => {
            let mut pats = Vec::new();
            for it in items.iter() {
                if let Some(p) = expr_to_pattern(it) {
                    pats.push(p);
                } else {
                    return None;
                }
            }
            Some(Pat {
                span: Span {
                    start: items.first().map(|e| e.span.start).unwrap_or(0),
                    end: items.last().map(|e| e.span.end).unwrap_or(0),
                },
                kind: PatKind::List(Rc::from(pats)),
            })
        }
        ExprKind::Record(fields) => {
            let mut pats = Vec::new();
            for f in fields.iter() {
                if let Some(value) = &f.value {
                    if let Some(pat) = expr_to_pattern(value) {
                        pats.push(PatRecordField {
                            key: f.key.clone(),
                            pat: Some(pat),
                            span: f.span,
                        });
                    } else {
                        return None;
                    }
                } else {
                    pats.push(PatRecordField {
                        key: f.key.clone(),
                        pat: None,
                        span: f.span,
                    });
                }
            }
            Some(Pat {
                span: Span {
                    start: fields.first().map(|f| f.span.start).unwrap_or(0),
                    end: fields.last().map(|f| f.span.end).unwrap_or(0),
                },
                kind: PatKind::Record(Rc::from(pats)),
            })
        }
        _ => None,
    }
}

impl Pat {
    pub fn parse(stream: &mut TokenStream) -> ParseResult<Self> {
        match stream.peek() {
            Some(Token::Group(g)) if g.delimiter() == raft_lexer::Delimiter::Parenthesis => {
                let mut group_stream = TokenStream::new(g.rc_tokens());
                group_stream.skip_comments_and_newlines();

                let mut pat = Pat::parse(&mut group_stream)?;
                group_stream.skip_comments_and_newlines();
                group_stream.expect_end()?;

                pat.span = g.span();
                stream.advance();
                Ok(pat)
            }
            Some(Token::Group(g)) if g.delimiter() == raft_lexer::Delimiter::Bracket => {
                let mut group_stream = TokenStream::new(g.rc_tokens());
                group_stream.skip_comments_and_newlines();

                let mut items = Vec::new();
                while group_stream.peek().is_some() {
                    let e = Pat::parse(&mut group_stream)?;
                    group_stream.skip_comments_and_newlines();

                    items.push(e);
                    match group_stream.peek() {
                        Some(Token::Punct(p)) if p.repr() == ',' => {
                            group_stream.advance();
                            group_stream.skip_comments_and_newlines();
                            continue;
                        }
                        Some(tok) => return Err(ParseError::UnexpectedToken(tok.span())),
                        None => break,
                    }
                }
                stream.advance();
                Ok(Pat {
                    kind: PatKind::List(Rc::from(items)),
                    span: g.span(),
                })
            }
            Some(Token::Group(g)) if g.delimiter() == raft_lexer::Delimiter::Brace => {
                let mut group_stream = TokenStream::new(g.rc_tokens());
                group_stream.skip_comments_and_newlines();

                let mut fields = Vec::new();
                while group_stream.peek().is_some() {
                    let key = Ident::parse(&mut group_stream)?;
                    group_stream.skip_comments_and_newlines();

                    match group_stream.peek() {
                        Some(Token::Punct(p)) if p.repr() == ':' => {
                            group_stream.advance();
                            group_stream.skip_comments_and_newlines();
                            let pat = Pat::parse(&mut group_stream)?;
                            group_stream.skip_comments_and_newlines();
                            let field_span = key.span.join(pat.span);
                            fields.push(PatRecordField {
                                key,
                                pat: Some(pat),
                                span: field_span,
                            });
                            match group_stream.peek() {
                                Some(Token::Punct(p)) if p.repr() == ',' => {
                                    group_stream.advance();
                                    group_stream.skip_comments_and_newlines();
                                    continue;
                                }
                                Some(tok) => {
                                    return Err(ParseError::UnexpectedToken(tok.span()));
                                }
                                None => break,
                            }
                        }
                        Some(Token::Punct(p)) if p.repr() == ',' => {
                            group_stream.advance();
                            group_stream.skip_comments_and_newlines();

                            let key_span = key.span;
                            fields.push(PatRecordField {
                                key: key.clone(),
                                pat: None,
                                span: key_span,
                            });
                            continue;
                        }
                        Some(tok) => {
                            return Err(ParseError::UnexpectedToken(tok.span()));
                        }
                        None => {
                            let key_span = key.span;
                            fields.push(PatRecordField {
                                key: key.clone(),
                                pat: None,
                                span: key_span,
                            });
                            break;
                        }
                    }
                }
                stream.advance();
                Ok(Pat {
                    kind: PatKind::Record(Rc::from(fields)),
                    span: g.span(),
                })
            }
            Some(Token::Literal(l)) => {
                let lit = l.clone();
                stream.advance();
                Ok(Pat {
                    span: lit.span(),
                    kind: PatKind::Literal(match lit {
                        raft_lexer::Literal::Number(n) => Literal::Number(n),
                        raft_lexer::Literal::Char(c) => Literal::Char(c),
                        raft_lexer::Literal::String(s) => Literal::String(s),
                    }),
                })
            }
            Some(Token::Ident(i)) => {
                stream.advance();

                let span = i.span();
                match i.repr().chars().next().unwrap() {
                    ch if ch.is_uppercase() => {
                        let a = Atom {
                            name: i.rc_repr(),
                            span,
                        };
                        Ok(Pat {
                            kind: PatKind::Atom(a),
                            span,
                        })
                    }
                    _ch => {
                        let id = Ident {
                            name: i.rc_repr(),
                            span,
                        };
                        Ok(Pat {
                            kind: PatKind::Ident(id),
                            span,
                        })
                    }
                }
            }
            Some(tok) => Err(ParseError::UnexpectedToken(tok.span())),
            None => Err(ParseError::UnexpectedEnd(stream.span())),
        }
    }
}

impl Stmt {
    pub fn parse_simple(stream: &mut TokenStream) -> ParseResult<Self> {
        let lhs = Expr::parse(stream)?;

        match stream.peek() {
            Some(Token::Punct(p)) if p.repr() == '=' => {
                stream.advance();
                let rhs = Expr::parse(stream)?;

                match &lhs.kind {
                    ExprKind::Field(obj, field_ident) => {
                        return Ok(Stmt {
                            span: lhs.span.join(rhs.span),
                            kind: StmtKind::AssignField {
                                target: obj.clone(),
                                field: field_ident.clone(),
                                value: rhs,
                            },
                        });
                    }
                    ExprKind::Index(obj, idx) => {
                        return Ok(Stmt {
                            span: lhs.span.join(rhs.span),
                            kind: StmtKind::AssignIndex {
                                target: (*obj).clone(),
                                index: (*idx).clone(),
                                value: rhs,
                            },
                        });
                    }
                    _ => {
                        if let Some(pat) = expr_to_pattern(&lhs) {
                            return Ok(Stmt {
                                span: pat.span.join(rhs.span),
                                kind: StmtKind::AssignPattern {
                                    target: pat,
                                    value: rhs,
                                },
                            });
                        } else {
                            return Err(ParseError::InvalidAssignmentTarget(lhs.span));
                        }
                    }
                }
            }
            _ => {
                return Ok(Stmt {
                    span: lhs.span,
                    kind: StmtKind::Expr(lhs),
                });
            }
        }
    }

    pub fn parse_line(stream: &mut TokenStream) -> ParseResult<Self> {
        if let Some((kw, kw_span)) = peek_keyword(stream) {
            match kw {
                Keyword::Return => {
                    stream.advance();
                    stream.skip_comments();

                    if let Some(Token::Newline(_)) | None = stream.peek() {
                        return Ok(Stmt {
                            span: kw_span,
                            kind: StmtKind::Return(None),
                        });
                    }

                    let expr = Expr::parse(stream)?;
                    return Ok(Stmt {
                        span: kw_span.join(expr.span),
                        kind: StmtKind::Return(Some(expr)),
                    });
                }
                Keyword::Break => {
                    stream.advance();
                    return Ok(Stmt {
                        span: kw_span,
                        kind: StmtKind::Break,
                    });
                }
                Keyword::Continue => {
                    stream.advance();
                    return Ok(Stmt {
                        span: kw_span,
                        kind: StmtKind::Continue,
                    });
                }
                _ => return Err(ParseError::UnexpectedToken(kw_span)),
            }
        }

        Self::parse_simple(stream)
    }

    pub fn parse(stream: &mut TokenStream) -> ParseResult<Self> {
        if let Some((kw, kw_span)) = peek_keyword(stream) {
            match kw {
                Keyword::Return => {
                    stream.advance();
                    stream.skip_comments();
                    if let Some(Token::Newline(_)) | None = stream.peek() {
                        return Ok(Stmt {
                            span: kw_span,
                            kind: StmtKind::Return(None),
                        });
                    }

                    let expr = Expr::parse(stream)?;
                    return Ok(Stmt {
                        span: kw_span.join(expr.span),
                        kind: StmtKind::Return(Some(expr)),
                    });
                }
                Keyword::Break => {
                    stream.advance();
                    return Ok(Stmt {
                        span: kw_span,
                        kind: StmtKind::Break,
                    });
                }
                Keyword::Continue => {
                    stream.advance();
                    return Ok(Stmt {
                        span: kw_span,
                        kind: StmtKind::Continue,
                    });
                }
                Keyword::If => {
                    return Self::parse_if(stream, kw_span);
                }
                Keyword::While => {
                    return Self::parse_while(stream, kw_span);
                }
                Keyword::For => {
                    return Self::parse_for(stream, kw_span);
                }
                _ => return Err(ParseError::UnexpectedToken(kw_span)),
            }
        }

        Self::parse_simple(stream)
    }

    pub fn parse_block(stream: &mut TokenStream) -> ParseResult<Vec<Self>> {
        let mut stmts = Vec::new();
        stream.skip_comments_and_newlines();
        while !stream.is_empty() {
            let stmt = Self::parse(stream)?;
            stmts.push(stmt);
            stream.skip_comments_and_newlines();
        }
        Ok(stmts)
    }

    fn parse_branch(stream: &mut TokenStream) -> ParseResult<Vec<Self>> {
        match stream.peek() {
            Some(Token::Punct(p)) if p.repr() == ':' => {
                stream.advance();
                stream.skip_comments();
                if stream.skip_newlines() {
                    stream.skip_comments_and_newlines();
                    match stream.peek() {
                        Some(Token::Group(g)) if g.delimiter() == raft_lexer::Delimiter::Block => {
                            stream.advance();

                            let mut group_stream = TokenStream::new(g.rc_tokens());
                            return Stmt::parse_block(&mut group_stream);
                        }
                        _ => return Ok(Vec::new()),
                    }
                } else {
                    let stmt = Stmt::parse_line(stream)?;
                    return Ok(vec![stmt]);
                }
            }
            Some(tok) => return Err(ParseError::UnexpectedToken(tok.span())),
            None => return Err(ParseError::UnexpectedEnd(stream.span())),
        }
    }

    fn parse_if(stream: &mut TokenStream, if_span: Span) -> ParseResult<Self> {
        stream.advance();
        let cond = Expr::parse(stream)?;
        let then_branch = Self::parse_branch(stream)?;
        stream.skip_comments_and_newlines();

        let else_branch = Self::parse_else(stream)?;

        let last_stmt = else_branch
            .as_ref()
            .and_then(|b| b.last())
            .or(then_branch.last());
        let span = if_span.join(last_stmt.map(|s| s.span).unwrap_or(cond.span));

        Ok(Stmt {
            span,
            kind: StmtKind::If {
                cond,
                then_branch,
                else_branch,
            },
        })
    }

    fn parse_else(stream: &mut TokenStream) -> ParseResult<Option<Vec<Self>>> {
        if let Some((Keyword::Else, _else_span)) = peek_keyword(stream) {
            stream.advance();

            let stmts = if let Some((Keyword::If, if_span)) = peek_keyword(stream) {
                vec![Self::parse_if(stream, if_span)?]
            } else {
                Self::parse_branch(stream)?
            };

            Ok(Some(stmts))
        } else {
            Ok(None)
        }
    }

    fn parse_while(stream: &mut TokenStream, while_span: Span) -> ParseResult<Self> {
        stream.advance();
        let cond = Expr::parse(stream)?;
        let body = Self::parse_branch(stream)?;
        stream.skip_comments_and_newlines();

        let else_branch = Self::parse_else(stream)?;

        let last_stmt = else_branch.as_ref().and_then(|b| b.last()).or(body.last());
        let span = while_span.join(last_stmt.map(|s| s.span).unwrap_or(cond.span));

        Ok(Stmt {
            span,
            kind: StmtKind::While {
                cond,
                body,
                else_branch,
            },
        })
    }

    fn parse_for(stream: &mut TokenStream, for_span: Span) -> ParseResult<Self> {
        stream.advance();
        let target = Pat::parse(stream)?;

        match peek_keyword(stream) {
            Some((Keyword::In, _)) => {
                stream.advance();
            }
            Some((_, kw_span)) => {
                return Err(ParseError::UnexpectedToken(kw_span));
            }
            None => {
                return Err(ParseError::UnexpectedToken(stream.start_span()));
            }
        }

        let iterable = Expr::parse(stream)?;

        let body = Self::parse_branch(stream)?;
        stream.skip_comments_and_newlines();

        let else_branch = Self::parse_else(stream)?;

        let last_stmt = else_branch.as_ref().and_then(|b| b.last()).or(body.last());
        let span = for_span.join(last_stmt.map(|s| s.span).unwrap_or(iterable.span));

        Ok(Stmt {
            span,
            kind: StmtKind::For {
                target,
                iterable,
                body,
                else_branch,
            },
        })
    }
}
