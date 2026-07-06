use std::rc::Rc;

use crate::{
    Atom, BinaryOp, BinaryOpKind, Expr, ExprKind, ExprRecordField, Ident, Literal, Pat, PatKind,
    PatRecordField, Span, Stmt, StmtKind, UnaryOp, UnaryOpKind,
};

use raft_lexer::Token;

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
}

impl Ident {
    fn peek(stream: &TokenStream) -> Option<Ident> {
        match stream.peek() {
            Some(Token::Ident(i)) => {
                if Keyword::from_ident(i.repr()).is_some() {
                    return None;
                }

                let s = i.repr();
                if s.chars().next().map(|c| !c.is_uppercase()).unwrap_or(false) {
                    let name = i.rc_repr();
                    let span = i.span();
                    Some(Ident { name, span })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

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
    fn peek(stream: &TokenStream) -> Option<Atom> {
        match stream.peek() {
            Some(Token::Ident(i)) => {
                if Keyword::from_ident(i.repr()).is_some() {
                    return None;
                }

                let s = i.repr();
                if s.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                    let name = i.rc_repr();
                    let span = i.span();
                    Some(Atom { name, span })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

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
    fn peek(stream: &TokenStream) -> Option<Literal> {
        match stream.peek() {
            Some(Token::Literal(l)) => {
                let lit = l.clone();
                match lit {
                    raft_lexer::Literal::Number(n) => Some(Literal::Number(n)),
                    raft_lexer::Literal::Char(c) => Some(Literal::Char(c)),
                    raft_lexer::Literal::String(s) => Some(Literal::String(s)),
                }
            }
            _ => None,
        }
    }

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
                            pattern: Some(pat),
                            span: f.span,
                        });
                    } else {
                        return None;
                    }
                } else {
                    pats.push(PatRecordField {
                        key: f.key.clone(),
                        pattern: None,
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

                let mut pattern = Pat::parse(&mut group_stream)?;
                group_stream.skip_comments_and_newlines();
                group_stream.expect_end()?;

                pattern.span = g.span();
                stream.advance();
                Ok(pattern)
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
                            let pattern = Pat::parse(&mut group_stream)?;
                            group_stream.skip_comments_and_newlines();
                            let field_span = key.span.join(pattern.span);
                            fields.push(PatRecordField {
                                key,
                                pattern: Some(pattern),
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
                                pattern: None,
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
                                pattern: None,
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

    fn parse_block(stream: &mut TokenStream) -> ParseResult<Vec<Self>> {
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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(s.pos, 1);
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
            assert_eq!(s.pos, expected_len);
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
        assert!(fields[0].pattern().is_none());
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
            matches!(fields[0].pattern().unwrap().kind(), PatKind::Ident(i) if i.name() == "foo")
        );
    }

    #[test]
    fn pattern_record_nested() {
        let p = tokens_from_str("{ x: [a, b] }").parse_pat().unwrap();
        let PatKind::Record(fields) = p.kind() else {
            panic!()
        };
        assert!(matches!(
            fields[0].pattern().unwrap().kind(),
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
}
