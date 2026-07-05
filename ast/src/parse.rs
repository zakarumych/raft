use std::rc::Rc;

use crate::{
    Atom, BinaryOp, Expr, ExprKind, ExprRecordField, Ident, Literal, Pattern, PatternKind,
    PatternRecordField, Span, Stmt, StmtKind, UnaryOp,
};

use raft_lexer::Token;

#[derive(Debug)]
pub enum ParseError {
    UnexpectedKeyword(Keyword, Span),
    UnexpectedToken(Span),
    UnexpectedEof(Span),
    ExpectedPunct(char, Span),
    ExpectedAtom(Span),
    ExpectedIdent(Span),
    ExpectedKeyword(Keyword, Span),
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
}

// Leaf-node parsers working on TokenStream
impl Ident {
    fn try_parse(stream: &TokenStream) -> Option<Ident> {
        match stream.peek() {
            Some(Token::Ident(i)) => {
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
}

impl Atom {
    fn try_parse(stream: &TokenStream) -> Option<Atom> {
        match stream.peek() {
            Some(Token::Ident(i)) => {
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
}

impl Literal {
    fn try_parse(stream: &TokenStream) -> Option<Literal> {
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
}

impl Expr {
    pub fn parse(stream: &mut TokenStream) -> ParseResult<Self> {
        Self::parse_binary(stream, 0)
    }

    fn parse_binary(stream: &mut TokenStream, min_prec: u8) -> ParseResult<Self> {
        let mut lhs = Self::parse_application(stream)?;
        loop {
            match try_parse_binary_op(stream) {
                Some(op) if op.precedence() > min_prec => {
                    let rhs_min = if op.is_right_assoc() {
                        op.precedence() - 1
                    } else {
                        op.precedence()
                    };

                    let rhs = Self::parse_binary(stream, rhs_min)?;
                    let span = Span {
                        start: lhs.span.start,
                        end: rhs.span.end,
                    };
                    lhs = Expr {
                        kind: ExprKind::Binary(Box::new(lhs), op, Box::new(rhs)),
                        span,
                    };
                }
                Some(_) => {
                    break;
                }
                None => break,
            }
        }
        Ok(lhs)
    }

    fn parse_application(stream: &mut TokenStream) -> ParseResult<Self> {
        let mut expr = Self::parse_unary(stream)?;
        loop {
            match stream.peek() {
                Some(Token::Group(g))
                    if matches!(g.delimiter(), raft_lexer::Delimiter::Parenthesis) =>
                {
                    stream.advance();

                    // build inner token stream from group tokens
                    let mut group_stream = TokenStream::new(g.rc_tokens());
                    let span = expr.span.join(g.span());

                    let arg = Expr::parse(&mut group_stream)?;
                    expr = Expr {
                        kind: ExprKind::Apply(Box::new(expr), vec![arg]),
                        span,
                    };
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_unary(stream: &mut TokenStream) -> ParseResult<Self> {
        if let Some(op) = try_parse_unary_op(stream)? {
            let operand = Self::parse_unary(stream)?;
            let span = Span {
                start: op.span.start,
                end: operand.span.end,
            };
            Ok(Expr {
                kind: ExprKind::Unary(op, Box::new(operand)),
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
                    stream.advance(); // consume '.'
                    // Ident is required after '.'
                    let field = Ident::try_parse(stream)
                        .ok_or(ParseError::ExpectedIdent(stream.start_span()))?;
                    let span = Span {
                        start: expr.span.start,
                        end: field.span.end,
                    };
                    expr = Expr {
                        kind: ExprKind::Field(Box::new(expr), field),
                        span,
                    };
                }
                Some(Token::Group(g))
                    if matches!(g.delimiter(), raft_lexer::Delimiter::Bracket) =>
                {
                    stream.advance();

                    // build inner token stream from group tokens
                    let mut group_stream = TokenStream::new(g.rc_tokens());

                    let span = expr.span.join(g.span());
                    let index = Expr::parse(&mut group_stream)?;
                    if let Some(next) = group_stream.peek() {
                        return Err(ParseError::UnexpectedToken(next.span()));
                    }

                    expr = Expr {
                        kind: ExprKind::Index(Box::new(expr), Box::new(index)),
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
            Some(Token::Group(g)) => {
                stream.advance();

                match g.delimiter() {
                    raft_lexer::Delimiter::Parenthesis => {
                        let mut group_stream = TokenStream::new(g.rc_tokens());
                        group_stream.skip_newlines();

                        let mut expr = Expr::parse(&mut group_stream)?;
                        group_stream.skip_newlines();

                        if let Some(tok) = group_stream.peek() {
                            return Err(ParseError::UnexpectedToken(tok.span()));
                        }

                        expr.span = g.span();
                        Ok(expr)
                    }
                    raft_lexer::Delimiter::Bracket => {
                        // list literal
                        let mut group_stream = TokenStream::new(g.rc_tokens());
                        group_stream.skip_newlines();

                        let mut items = Vec::new();
                        while group_stream.peek().is_some() {
                            let e = Expr::parse(&mut group_stream)?;
                            group_stream.skip_newlines();

                            items.push(e);
                            match group_stream.peek() {
                                Some(Token::Punct(p)) if p.repr() == ',' => {
                                    group_stream.advance();
                                    group_stream.skip_newlines();
                                    continue;
                                }
                                Some(tok) => return Err(ParseError::ExpectedPunct(',', tok.span())),
                                None => break,
                            }
                        }
                        Ok(Expr {
                            kind: ExprKind::List(items),
                            span: g.span(),
                        })
                    }
                    raft_lexer::Delimiter::Brace => {
                        // record literal
                        let mut group_stream = TokenStream::new(g.rc_tokens());
                        group_stream.skip_newlines();

                        let mut fields = Vec::new();
                        while group_stream.peek().is_some() {
                            // key must be ident
                            let key = Ident::try_parse(&mut group_stream)
                                .ok_or(ParseError::ExpectedIdent(group_stream.start_span()))?;
                            group_stream.skip_newlines();

                            // expect ':'
                            match group_stream.peek() {
                                Some(Token::Punct(p)) if p.repr() == ':' => {
                                    group_stream.advance(); // consume ':'
                                    group_stream.skip_newlines();

                                    let value = Expr::parse(&mut group_stream)?;
                                    group_stream.skip_newlines();

                                    let field_span = key.span.join(value.span);
                                    fields.push(ExprRecordField {
                                        key,
                                        value: Some(value),
                                        span: field_span,
                                    });
                                    match group_stream.peek() {
                                        Some(Token::Punct(p)) if p.repr() == ',' => {
                                            group_stream.advance();
                                            group_stream.skip_newlines();
                                            continue;
                                        }
                                        Some(tok) => {
                                            return Err(ParseError::ExpectedPunct(',', tok.span()));
                                        }
                                        None => break,
                                    }
                                }
                                Some(Token::Punct(p)) if p.repr() == ',' => {
                                    group_stream.advance();
                                    group_stream.skip_newlines();

                                    let key_span = key.span;
                                    fields.push(ExprRecordField {
                                        key: key.clone(),
                                        value: None,
                                        span: key_span,
                                    });

                                    continue;
                                }
                                Some(tok) => {
                                    return Err(ParseError::ExpectedPunct(',', tok.span()));
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
                        Ok(Expr {
                            kind: ExprKind::Record(fields),
                            span: g.span(),
                        })
                    }
                    raft_lexer::Delimiter::Block => {
                        return Err(ParseError::UnexpectedToken(g.span()));
                    }
                }
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
            None => Err(ParseError::UnexpectedEof(stream.span())),
        }
    }
}

fn try_parse_unary_op(stream: &mut TokenStream) -> ParseResult<Option<UnaryOp>> {
    match stream.peek() {
        Some(Token::Punct(p)) => {
            let ch = p.repr();
            let kind = match ch {
                '!' => Some(crate::UnaryOpKind::Not),
                '-' => Some(crate::UnaryOpKind::Neg),
                '~' => Some(crate::UnaryOpKind::BitNot),
                _ => None,
            };
            if let Some(k) = kind {
                let span = p.span();
                stream.advance();
                Ok(Some(UnaryOp { kind: k, span }))
            } else {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}

// Binary op parser on TokenStream
fn try_parse_binary_op(stream: &mut TokenStream) -> Option<BinaryOp> {
    match stream.peek() {
        Some(Token::Punct(p1)) => {
            let start = p1.span().start;
            // lookahead for second punct
            match (p1.repr(), stream.peek1()) {
                ('<', Some(Token::Punct(p2))) if p2.repr() == '<' => {
                    // consume both
                    stream.advance();
                    stream.advance();
                    let span = Span {
                        start,
                        end: p2.span().end,
                    };
                    return Some(BinaryOp {
                        kind: crate::BinaryOpKind::Shl,
                        span,
                    });
                }
                ('>', Some(Token::Punct(p2))) if p2.repr() == '>' => {
                    stream.advance();
                    stream.advance();
                    let span = Span {
                        start,
                        end: p2.span().end,
                    };
                    return Some(BinaryOp {
                        kind: crate::BinaryOpKind::Shr,
                        span,
                    });
                }
                ('*', Some(Token::Punct(p2))) if p2.repr() == '*' => {
                    stream.advance();
                    stream.advance();
                    let span = Span {
                        start,
                        end: p2.span().end,
                    };
                    return Some(BinaryOp {
                        kind: crate::BinaryOpKind::Pow,
                        span,
                    });
                }
                ('=', Some(Token::Punct(p2))) if p2.repr() == '=' => {
                    stream.advance();
                    stream.advance();
                    let span = Span {
                        start,
                        end: p2.span().end,
                    };
                    return Some(BinaryOp {
                        kind: crate::BinaryOpKind::Eq,
                        span,
                    });
                }
                ('!', Some(Token::Punct(p2))) if p2.repr() == '=' => {
                    stream.advance();
                    stream.advance();
                    let span = Span {
                        start,
                        end: p2.span().end,
                    };
                    return Some(BinaryOp {
                        kind: crate::BinaryOpKind::Ne,
                        span,
                    });
                }
                ('<', Some(Token::Punct(p2))) if p2.repr() == '=' => {
                    stream.advance();
                    stream.advance();
                    let span = Span {
                        start,
                        end: p2.span().end,
                    };
                    return Some(BinaryOp {
                        kind: crate::BinaryOpKind::Le,
                        span,
                    });
                }
                ('>', Some(Token::Punct(p2))) if p2.repr() == '=' => {
                    stream.advance();
                    stream.advance();
                    let span = Span {
                        start,
                        end: p2.span().end,
                    };
                    return Some(BinaryOp {
                        kind: crate::BinaryOpKind::Ge,
                        span,
                    });
                }
                _ => {
                    // single-char ops
                    let kind = match p1.repr() {
                        '&' => Some(crate::BinaryOpKind::BitAnd),
                        '|' => Some(crate::BinaryOpKind::BitOr),
                        '^' => Some(crate::BinaryOpKind::BitXor),
                        '*' => Some(crate::BinaryOpKind::Mul),
                        '/' => Some(crate::BinaryOpKind::Div),
                        '+' => Some(crate::BinaryOpKind::Add),
                        '-' => Some(crate::BinaryOpKind::Sub),
                        '<' => Some(crate::BinaryOpKind::Lt),
                        '>' => Some(crate::BinaryOpKind::Gt),
                        _ => None,
                    };
                    if let Some(k) = kind {
                        let span = p1.span();
                        stream.advance();
                        return Some(BinaryOp { kind: k, span });
                    }
                }
            }
            None
        }
        _ => None,
    }
}

// Keyword helper for TokenStream: consumes ident if it matches keyword
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

// Convert Expr AST to Pattern if possible
fn expr_to_pattern(expr: &Expr) -> Option<Pattern> {
    match &expr.kind {
        ExprKind::Ident(i) => Some(Pattern {
            span: i.span,
            kind: PatternKind::Ident(i.clone()),
        }),
        ExprKind::Atom(a) => Some(Pattern {
            span: a.span,
            kind: PatternKind::Atom(a.clone()),
        }),
        ExprKind::Literal(l) => Some(Pattern {
            span: l.span(),
            kind: PatternKind::Literal(l.clone()),
        }),
        ExprKind::List(items) => {
            let mut pats = Vec::new();
            for it in items {
                if let Some(p) = expr_to_pattern(it) {
                    pats.push(p);
                } else {
                    return None;
                }
            }
            Some(Pattern {
                span: Span {
                    start: items.first().map(|e| e.span.start).unwrap_or(0),
                    end: items.last().map(|e| e.span.end).unwrap_or(0),
                },
                kind: PatternKind::List(pats),
            })
        }
        ExprKind::Record(fields) => {
            let mut pats = Vec::new();
            for f in fields {
                if let Some(value) = &f.value {
                    if let Some(pat) = expr_to_pattern(value) {
                        pats.push(PatternRecordField {
                            key: f.key.clone(),
                            pattern: Some(pat),
                            span: f.span,
                        });
                    } else {
                        return None;
                    }
                } else {
                    pats.push(PatternRecordField {
                        key: f.key.clone(),
                        pattern: None,
                        span: f.span,
                    });
                }
            }
            Some(Pattern {
                span: Span {
                    start: fields.first().map(|f| f.span.start).unwrap_or(0),
                    end: fields.last().map(|f| f.span.end).unwrap_or(0),
                },
                kind: PatternKind::Record(pats),
            })
        }
        _ => None,
    }
}

// Pattern parsers on TokenStream
impl Pattern {
    pub fn parse(stream: &mut TokenStream) -> ParseResult<Self> {
        match stream.peek() {
            Some(Token::Group(g)) if matches!(g.delimiter(), raft_lexer::Delimiter::Bracket) => {
                // list pattern
                let mut group_stream = TokenStream::new(g.rc_tokens());
                group_stream.skip_newlines();

                let mut items = Vec::new();
                while group_stream.peek().is_some() {
                    let e = Pattern::parse(&mut group_stream)?;
                    group_stream.skip_newlines();

                    items.push(e);
                    match group_stream.peek() {
                        Some(Token::Punct(p)) if p.repr() == ',' => {
                            group_stream.advance();
                            group_stream.skip_newlines();
                            continue;
                        }
                        Some(tok) => return Err(ParseError::ExpectedPunct(',', tok.span())),
                        None => break,
                    }
                }
                Ok(Pattern {
                    kind: PatternKind::List(items),
                    span: g.span(),
                })
            }
            Some(Token::Group(g)) if matches!(g.delimiter(), raft_lexer::Delimiter::Brace) => {
                // record pattern
                let mut group_stream = TokenStream::new(g.rc_tokens());
                group_stream.skip_newlines();

                let mut fields = Vec::new();
                while group_stream.peek().is_some() {
                    // key must be ident
                    let key = Ident::try_parse(&mut group_stream)
                        .ok_or(ParseError::ExpectedIdent(group_stream.start_span()))?;
                    group_stream.skip_newlines();
                    // expect ':'
                    match group_stream.peek() {
                        Some(Token::Punct(p)) if p.repr() == ':' => {
                            group_stream.advance(); // consume ':'
                            group_stream.skip_newlines();
                            let pattern = Pattern::parse(&mut group_stream)?;
                            group_stream.skip_newlines();
                            let field_span = key.span.join(pattern.span);
                            fields.push(PatternRecordField {
                                key,
                                pattern: Some(pattern),
                                span: field_span,
                            });
                            match group_stream.peek() {
                                Some(Token::Punct(p)) if p.repr() == ',' => {
                                    group_stream.advance();
                                    group_stream.skip_newlines();
                                    continue;
                                }
                                Some(tok) => {
                                    return Err(ParseError::ExpectedPunct(',', tok.span()));
                                }
                                None => break,
                            }
                        }
                        Some(Token::Punct(p)) if p.repr() == ',' => {
                            group_stream.advance();
                            group_stream.skip_newlines();

                            let key_span = key.span;
                            fields.push(PatternRecordField {
                                key: key.clone(),
                                pattern: None,
                                span: key_span,
                            });
                            continue;
                        }
                        Some(tok) => {
                            return Err(ParseError::ExpectedPunct(',',tok.span()));
                        }
                        None => {
                            let key_span = key.span;
                            fields.push(PatternRecordField {
                                key: key.clone(),
                                pattern: None,
                                span: key_span,
                            });
                            break;
                        }
                    }
                }
                Ok(Pattern {
                    kind: PatternKind::Record(fields),
                    span: g.span(),
                })
            }
            Some(Token::Literal(l)) => {
                let lit = l.clone();
                stream.advance();
                Ok(Pattern {
                    span: lit.span(),
                    kind: PatternKind::Literal(match lit {
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
                        Ok(Pattern {
                            kind: PatternKind::Atom(a),
                            span,
                        })
                    }
                    _ch => {
                        let id = Ident {
                            name: i.rc_repr(),
                            span,
                        };
                        Ok(Pattern {
                            kind: PatternKind::Ident(id),
                            span,
                        })
                    }
                }
            }
            Some(tok) => Err(ParseError::UnexpectedToken(tok.span())),
            None => Err(ParseError::UnexpectedEof(stream.span())),
        }
    }
}

// Statement parsing on TokenStream
impl Stmt {
    pub fn parse_simple(stream: &mut TokenStream) -> ParseResult<Self> {
        // parse lhs expr
        let lhs = Expr::parse(stream)?;

        // assignment?
        match stream.peek() {
            Some(Token::Punct(p)) if p.repr() == '=' => {
                stream.advance(); // consume '='
                let rhs = Expr::parse(stream)?;

                // build assignment stmt
                match &lhs.kind {
                    ExprKind::Field(obj, field_ident) => {
                        return Ok(Stmt {
                            span: lhs.span.join(rhs.span),
                            kind: StmtKind::AssignField {
                                target: (*obj).clone(),
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
                // expression stmt
                return Ok(Stmt {
                    span: lhs.span,
                    kind: StmtKind::Expr(lhs),
                });
            }
        }
    }

    pub fn parse_line(stream: &mut TokenStream) -> ParseResult<Self> {
        // check for return/break/continue
        if let Some((kw, kw_span)) = peek_keyword(stream) {
            match kw {
                Keyword::Return => {
                    stream.advance();
                    let expr = Expr::parse(stream)?;
                    return Ok(Stmt {
                        span: kw_span.join(expr.span),
                        kind: StmtKind::Return(expr),
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
                k => return Err(ParseError::UnexpectedKeyword(k, kw_span)),
            }
        }
        // otherwise simple stmt
        Self::parse_simple(stream)
    }

    pub fn parse(stream: &mut TokenStream) -> ParseResult<Self> {
        // keyword-first: if/while/for handled here
        if let Some((kw, kw_span)) = peek_keyword(stream) {
            match kw {
                Keyword::Return => {
                    stream.advance();
                    let expr = Expr::parse(stream)?;
                    return Ok(Stmt {
                        span: kw_span.join(expr.span),
                        kind: StmtKind::Return(expr),
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
                k => return Err(ParseError::UnexpectedKeyword(k, kw_span)),
            }
        }
        // not keyword-first: simple stmt
        Self::parse_simple(stream)
    }

    // parse sequence of statements from a TokenStream representing a block
    fn parse_block(stream: &mut TokenStream) -> ParseResult<Vec<Self>> {
        let mut stmts = Vec::new();
        while !stream.is_empty() {
            stream.skip_newlines();

            let stmt = Self::parse(stream)?;
            stmts.push(stmt);
        }
        Ok(stmts)
    }

    fn parse_branch(stream: &mut TokenStream) -> ParseResult<Vec<Self>> {
        // expect ':'
        match stream.peek() {
            Some(Token::Punct(p)) if p.repr() == ':' => {
                stream.advance(); // consume ':'
                if stream.skip_newlines() {
                    // Branch in block
                    match stream.peek() {
                        Some(Token::Group(g)) if g.delimiter() == raft_lexer::Delimiter::Block => {
                            stream.advance();
                            
                            let mut group_stream = TokenStream::new(g.rc_tokens());
                            return Stmt::parse_block(&mut group_stream);
                        }
                        _ => return Ok(Vec::new()),
                    }
                } else {
                    // Branch is inline statement
                    let stmt = Stmt::parse_line(stream)?;
                    return Ok(vec![stmt]);
                }
            }
            Some(tok) => return Err(ParseError::ExpectedPunct(':', tok.span())),
            None => return Err(ParseError::UnexpectedEof(stream.span())),
        }
    }

    fn parse_if(stream: &mut TokenStream, if_span: Span) -> ParseResult<Self> {
        stream.advance();
        let cond = Expr::parse(stream)?;
        let then_branch = Self::parse_branch(stream)?;
        stream.skip_newlines();

        let else_branch = Self::parse_else(stream)?;

        let last_stmt = else_branch.as_ref().and_then(|b| b.last()).or(then_branch.last());
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

            let stmts =if let Some((Keyword::If, if_span)) = peek_keyword(stream) {
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
        stream.skip_newlines();

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
        let target = Pattern::parse(stream)?;

        if let Some((Keyword::In, _)) = peek_keyword(stream) {
            stream.advance();
        } else {
            return Err(ParseError::ExpectedKeyword(Keyword::In, stream.start_span()));
        }

        let iterable = Expr::parse(stream)?;

        let body = Self::parse_branch(stream)?;
        stream.skip_newlines();

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

    #[test]
    fn idents() {
        let i = Stream::new("foo").parse_ident().unwrap();
        assert_eq!(i.name, "foo");
        assert_eq!(i.span, Span { start: 0, end: 3 });

        assert_eq!(Stream::new("_bar").parse_ident().unwrap().name, "_bar");
        assert_eq!(
            Stream::new("foo_bar").parse_ident().unwrap().name,
            "foo_bar"
        );
        assert_eq!(Stream::new("x1").parse_ident().unwrap().name, "x1");
    }

    #[test]
    fn atoms() {
        let a = Stream::new("Foo").parse_atom().unwrap();
        assert_eq!(a.name, "Foo");
        assert_eq!(a.span, Span { start: 0, end: 3 });

        assert_eq!(Stream::new("True").parse_atom().unwrap().name, "True");
        assert_eq!(Stream::new("MyAtom").parse_atom().unwrap().name, "MyAtom");
    }

    #[test]
    fn ident_not_atom() {
        assert!(Stream::new("Foo").parse_ident().is_err());
        assert!(Stream::new("foo").parse_atom().is_err());
    }

    #[test]
    fn literal_int() {
        let lit = Stream::new("42").parse_literal().unwrap();
        let n = lit.as_number().unwrap();
        assert_eq!(n.repr(), "42");
        assert!(!n.has_dot() && !n.has_exponent());
        assert_eq!(n.integer(), "42");
        assert_eq!(n.span(), Span::new(0, 2));
    }

    #[test]
    fn literal_float_dot() {
        let n = Stream::new("4.5").parse_literal().unwrap();
        let n = n.as_number().unwrap();
        assert_eq!(n.repr(), "4.5");
        assert!(n.has_dot());
        assert_eq!(n.integer(), "4");
        assert_eq!(n.fractional(), Some("5"));
    }

    #[test]
    fn literal_float_exp() {
        let n = Stream::new("5e-2").parse_literal().unwrap();
        let n = n.as_number().unwrap();
        assert_eq!(n.repr(), "5e-2");
        assert!(n.has_exponent());
        assert_eq!(n.integer(), "5");
        assert_eq!(n.exponent(), Some("-2"));
    }

    #[test]
    fn literal_float_full() {
        let n = Stream::new("1.0e10").parse_literal().unwrap();
        let n = n.as_number().unwrap();
        assert_eq!(n.repr(), "1.0e10");
        assert!(n.has_dot() && n.has_exponent());
        assert_eq!(n.fractional(), Some("0"));
        assert_eq!(n.exponent(), Some("10"));
    }

    #[test]
    fn literal_char() {
        let lit = Stream::new("'a'").parse_literal().unwrap();
        let c = lit.as_char().unwrap();
        assert_eq!(c.repr(), "'a'");
        assert_eq!(c.unescape(), 'a');
    }

    #[test]
    fn literal_char_escape() {
        assert_eq!(
            Stream::new("'\\n'")
                .parse_literal()
                .unwrap()
                .as_char()
                .unwrap()
                .unescape(),
            '\n'
        );
        assert_eq!(
            Stream::new("'\\t'")
                .parse_literal()
                .unwrap()
                .as_char()
                .unwrap()
                .unescape(),
            '\t'
        );
        assert_eq!(
            Stream::new("'\\\\'")
                .parse_literal()
                .unwrap()
                .as_char()
                .unwrap()
                .unescape(),
            '\\'
        );
    }

    #[test]
    fn literal_string() {
        let s = Stream::new(r#""hello""#).parse_literal().unwrap();
        let s = s.as_string().unwrap();
        assert_eq!(s.repr(), r#""hello""#);
        assert_eq!(s.unescape(), "hello");
    }

    #[test]
    fn literal_string_escape() {
        let s = Stream::new(r#""foo\nbar\n""#).parse_literal().unwrap();
        assert_eq!(s.as_string().unwrap().unescape(), "foo\nbar\n");

        let s = Stream::new(r#""""#).parse_literal().unwrap();
        assert_eq!(s.as_string().unwrap().unescape(), "");
    }

    #[test]
    fn literal_dot_not_accessor() {
        let mut s = Stream::new("1.foo");
        let lit = s.parse_literal().unwrap();
        assert_eq!(lit.as_number().unwrap().repr(), "1");
        assert_eq!(s.pos(), 1);
    }

    #[test]
    fn unary_ops() {
        assert_eq!(Stream::new("!").try_unary_op().unwrap().node, UnaryOp::Not);
        assert_eq!(Stream::new("-").try_unary_op().unwrap().node, UnaryOp::Neg);
        assert_eq!(
            Stream::new("~").try_unary_op().unwrap().node,
            UnaryOp::BitNot
        );
        assert!(Stream::new("+").try_unary_op().is_none());
    }

    #[test]
    fn binary_ops() {
        let cases: &[(&str, BinaryOp, usize)] = &[
            ("&", BinaryOp::BitAnd, 1),
            ("|", BinaryOp::BitOr, 1),
            ("^", BinaryOp::BitXor, 1),
            ("<<", BinaryOp::Shl, 2),
            (">>", BinaryOp::Shr, 2),
            ("**", BinaryOp::Pow, 2),
            ("*", BinaryOp::Mul, 1),
            ("/", BinaryOp::Div, 1),
            ("+", BinaryOp::Add, 1),
            ("-", BinaryOp::Sub, 1),
            ("==", BinaryOp::Eq, 2),
            ("!=", BinaryOp::Ne, 2),
            ("<=", BinaryOp::Le, 2),
            (">=", BinaryOp::Ge, 2),
            ("<", BinaryOp::Lt, 1),
            (">", BinaryOp::Gt, 1),
        ];
        for &(src, expected_op, expected_len) in cases {
            let mut s = Stream::new(src);
            let sp = s.try_binary_op().unwrap();
            assert_eq!(sp.node, expected_op, "op mismatch for {src:?}");
            assert_eq!(s.pos(), expected_len, "len mismatch for {src:?}");
        }
    }

    #[test]
    fn precedence_ordering() {
        assert!(BinaryOp::BitAnd.precedence() > BinaryOp::Pow.precedence());
        assert!(BinaryOp::Pow.precedence() > BinaryOp::Mul.precedence());
        assert!(BinaryOp::Mul.precedence() > BinaryOp::Add.precedence());
        assert!(BinaryOp::Add.precedence() > BinaryOp::Eq.precedence());
        assert!(BinaryOp::Pow.is_right_assoc());
        assert!(!BinaryOp::Mul.is_right_assoc());
    }

    #[test]
    fn pattern_ident() {
        let p = Stream::new("foo").parse_pattern().unwrap();
        assert_eq!(p.span, Span { start: 0, end: 3 });
        assert!(matches!(p.kind, PatternKind::Ident(i) if i.name == "foo"));
    }

    #[test]
    fn pattern_atom() {
        let p = Stream::new("True").parse_pattern().unwrap();
        assert!(matches!(p.kind, PatternKind::Atom(a) if a.name == "True"));
    }

    #[test]
    fn pattern_list() {
        let p = Stream::new("[]").parse_pattern().unwrap();
        assert!(matches!(&p.kind, PatternKind::List(els) if els.is_empty()));

        let p = Stream::new("[a, b, c]").parse_pattern().unwrap();
        let PatternKind::List(els) = &p.kind else {
            panic!()
        };
        assert_eq!(els.len(), 3);
        assert!(matches!(&els[0].kind, PatternKind::Ident(i) if i.name == "a"));
        assert!(matches!(&els[2].kind, PatternKind::Ident(i) if i.name == "c"));
    }

    #[test]
    fn pattern_record_empty() {
        let p = Stream::new("{}").parse_pattern().unwrap();
        assert!(matches!(&p.kind, PatternKind::Record(f) if f.is_empty()));
    }

    #[test]
    fn pattern_record_shorthand() {
        let p = Stream::new("{ foo, bar }").parse_pattern().unwrap();
        let PatternKind::Record(fields) = &p.kind else {
            panic!()
        };
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].key.name, "foo");
        assert!(matches!(&fields[0].pattern.kind, PatternKind::Ident(i) if i.name == "foo"));
        assert_eq!(fields[1].key.name, "bar");
    }

    #[test]
    fn pattern_record_explicit() {
        let p = Stream::new("{ x: foo, y: bar }").parse_pattern().unwrap();
        let PatternKind::Record(fields) = &p.kind else {
            panic!()
        };
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].key.name, "x");
        assert!(matches!(&fields[0].pattern.kind, PatternKind::Ident(i) if i.name == "foo"));
    }

    #[test]
    fn pattern_record_nested() {
        let p = Stream::new("{ x: [a, b] }").parse_pattern().unwrap();
        let PatternKind::Record(fields) = &p.kind else {
            panic!()
        };
        assert!(matches!(&fields[0].pattern.kind, PatternKind::List(_)));
    }

    #[test]
    fn expr_literal() {
        let e = Stream::new("42").parse_expr().unwrap();
        assert_eq!(e.span, Span::new(0, 2));
        assert!(matches!(e.kind, ExprKind::Literal(_)));
    }

    #[test]
    fn expr_ident() {
        let e = Stream::new("foo").parse_expr().unwrap();
        assert!(matches!(&e.kind, ExprKind::Ident(i) if i.name == "foo"));
    }

    #[test]
    fn expr_atom() {
        let e = Stream::new("True").parse_expr().unwrap();
        assert!(matches!(&e.kind, ExprKind::Atom(a) if a.name == "True"));
    }

    #[test]
    fn expr_unary() {
        let e = Stream::new("!a").parse_expr().unwrap();
        let ExprKind::Unary(op, inner) = &e.kind else {
            panic!()
        };
        assert_eq!(op.node, UnaryOp::Not);
        assert!(matches!(&inner.kind, ExprKind::Ident(i) if i.name == "a"));
        assert_eq!(e.span, Span::new(0, 2));
    }

    #[test]
    fn expr_unary_chain() {
        // !!a = !(!a)
        let e = Stream::new("!!a").parse_expr().unwrap();
        let ExprKind::Unary(op, inner) = &e.kind else {
            panic!()
        };
        assert_eq!(op.node, UnaryOp::Not);
        assert!(matches!(&inner.kind, ExprKind::Unary(_, _)));
    }

    #[test]
    fn expr_binary_simple() {
        let e = Stream::new("1 + 2").parse_expr().unwrap();
        let ExprKind::Binary(lhs, op, rhs) = &e.kind else {
            panic!()
        };
        assert_eq!(op.node, BinaryOp::Add);
        assert!(matches!(&lhs.kind, ExprKind::Literal(_)));
        assert!(matches!(&rhs.kind, ExprKind::Literal(_)));
        assert_eq!(e.span, Span::new(0, 5));
    }

    #[test]
    fn expr_precedence() {
        // 1 + 2 * 3 = 1 + (2 * 3)
        let e = Stream::new("1 + 2 * 3").parse_expr().unwrap();
        let ExprKind::Binary(lhs, op, rhs) = &e.kind else {
            panic!()
        };
        assert_eq!(op.node, BinaryOp::Add);
        assert!(matches!(&lhs.kind, ExprKind::Literal(_)));
        let ExprKind::Binary(_, inner_op, _) = &rhs.kind else {
            panic!()
        };
        assert_eq!(inner_op.node, BinaryOp::Mul);
    }

    #[test]
    fn expr_left_assoc() {
        // a - b - c = (a - b) - c
        let e = Stream::new("a - b - c").parse_expr().unwrap();
        let ExprKind::Binary(lhs, op, rhs) = &e.kind else {
            panic!()
        };
        assert_eq!(op.node, BinaryOp::Sub);
        assert!(matches!(&lhs.kind, ExprKind::Binary(_, _, _)));
        assert!(matches!(&rhs.kind, ExprKind::Ident(i) if i.name == "c"));
    }

    #[test]
    fn expr_right_assoc() {
        // 2 ** 3 ** 4 = 2 ** (3 ** 4)
        let e = Stream::new("2 ** 3 ** 4").parse_expr().unwrap();
        let ExprKind::Binary(lhs, op, rhs) = &e.kind else {
            panic!()
        };
        assert_eq!(op.node, BinaryOp::Pow);
        assert!(matches!(&lhs.kind, ExprKind::Literal(_)));
        assert!(matches!(&rhs.kind, ExprKind::Binary(_, _, _)));
    }

    #[test]
    fn expr_apply() {
        let e = Stream::new("f a b").parse_expr().unwrap();
        let ExprKind::Apply(func, args) = &e.kind else {
            panic!()
        };
        assert!(matches!(&func.kind, ExprKind::Ident(i) if i.name == "f"));
        assert_eq!(args.len(), 2);
        assert!(matches!(&args[0].kind, ExprKind::Ident(i) if i.name == "a"));
        assert!(matches!(&args[1].kind, ExprKind::Ident(i) if i.name == "b"));
    }

    #[test]
    fn expr_apply_unary_arg() {
        // f !a — ! is unambiguously unary, so it's an argument
        let e = Stream::new("f !a").parse_expr().unwrap();
        let ExprKind::Apply(_, args) = &e.kind else {
            panic!()
        };
        assert_eq!(args.len(), 1);
        assert!(matches!(&args[0].kind, ExprKind::Unary(op, _) if op.node == UnaryOp::Not));
    }

    #[test]
    fn expr_apply_then_binary() {
        // f a + b = (f a) + b
        let e = Stream::new("f a + b").parse_expr().unwrap();
        let ExprKind::Binary(lhs, op, _) = &e.kind else {
            panic!()
        };
        assert_eq!(op.node, BinaryOp::Add);
        assert!(matches!(&lhs.kind, ExprKind::Apply(_, _)));
    }

    #[test]
    fn expr_minus_is_binary_not_arg() {
        // f - a = f minus a (NOT application)
        let e = Stream::new("f - a").parse_expr().unwrap();
        assert!(matches!(&e.kind, ExprKind::Binary(_, op, _) if op.node == BinaryOp::Sub));
    }

    #[test]
    fn expr_field() {
        let e = Stream::new("foo.bar").parse_expr().unwrap();
        let ExprKind::Field(obj, field) = &e.kind else {
            panic!()
        };
        assert!(matches!(&obj.kind, ExprKind::Ident(i) if i.name == "foo"));
        assert_eq!(field.name, "bar");
        assert_eq!(e.span, Span::new(0, 7));
    }

    #[test]
    fn expr_index() {
        let e = Stream::new("arr[0]").parse_expr().unwrap();
        let ExprKind::Index(obj, _) = &e.kind else {
            panic!()
        };
        assert!(matches!(&obj.kind, ExprKind::Ident(i) if i.name == "arr"));
        assert_eq!(e.span, Span::new(0, 6));
    }

    #[test]
    fn expr_chained_accessor() {
        // foo.bar[0].baz = Field(Index(Field(foo, bar), 0), baz)
        let e = Stream::new("foo.bar[0].baz").parse_expr().unwrap();
        let ExprKind::Field(indexed, baz) = &e.kind else {
            panic!()
        };
        assert_eq!(baz.name, "baz");
        let ExprKind::Index(field_expr, _) = &indexed.kind else {
            panic!()
        };
        let ExprKind::Field(root, bar) = &field_expr.kind else {
            panic!()
        };
        assert!(matches!(&root.kind, ExprKind::Ident(i) if i.name == "foo"));
        assert_eq!(bar.name, "bar");
    }

    #[test]
    fn expr_apply_with_field_arg() {
        // f a.b = f (a.b)
        let e = Stream::new("f a.b").parse_expr().unwrap();
        let ExprKind::Apply(_, args) = &e.kind else {
            panic!()
        };
        assert_eq!(args.len(), 1);
        assert!(matches!(&args[0].kind, ExprKind::Field(_, _)));
    }

    #[test]
    fn expr_list() {
        let e = Stream::new("[1, 2, 3]").parse_expr().unwrap();
        let ExprKind::List(els) = &e.kind else {
            panic!()
        };
        assert_eq!(els.len(), 3);
        assert_eq!(e.span, Span::new(0, 9));
    }

    #[test]
    fn expr_list_empty() {
        let e = Stream::new("[]").parse_expr().unwrap();
        assert!(matches!(&e.kind, ExprKind::List(els) if els.is_empty()));
    }

    #[test]
    fn expr_record() {
        let e = Stream::new("{x: 1, y: 2}").parse_expr().unwrap();
        let ExprKind::Record(fields) = &e.kind else {
            panic!()
        };
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].key.name, "x");
        assert!(matches!(&fields[0].value.kind, ExprKind::Literal(_)));
    }

    #[test]
    fn expr_paren_grouping() {
        // (1 + 2) * 3 — parens override precedence
        let e = Stream::new("(1 + 2) * 3").parse_expr().unwrap();
        let ExprKind::Binary(lhs, op, _) = &e.kind else {
            panic!()
        };
        assert_eq!(op.node, BinaryOp::Mul);
        // lhs is the parenthesized addition; span includes parens
        assert_eq!(lhs.span, Span::new(0, 7));
        assert!(matches!(&lhs.kind, ExprKind::Binary(_, op, _) if op.node == BinaryOp::Add));
    }

    #[test]
    fn expr_complex() {
        // a + b * c ** d / e > f * g - h ** i / j
        // = (a + ((b * (c ** d)) / e)) > ((f * g) - ((h ** i) / j))
        let e = Stream::new("a + b * c ** d / e > f * g - h ** i / j")
            .parse_expr()
            .unwrap();
        let ExprKind::Binary(lhs, op, rhs) = &e.kind else {
            panic!()
        };
        assert_eq!(op.node, BinaryOp::Gt);
        // lhs = a + ((b * (c**d)) / e)
        let ExprKind::Binary(_, add_op, _) = &lhs.kind else {
            panic!()
        };
        assert_eq!(add_op.node, BinaryOp::Add);
        // rhs = (f*g) - ((h**i)/j)
        let ExprKind::Binary(_, sub_op, _) = &rhs.kind else {
            panic!()
        };
        assert_eq!(sub_op.node, BinaryOp::Sub);
    }

    #[test]
    fn stmt_expr_statement() {
        let stmt = Stream::new("foo").parse_stmt(0).unwrap();
        match &stmt.kind {
            StmtKind::Expr(e) => {
                assert!(matches!(&e.kind, ExprKind::Ident(i) if i.name == "foo"));
            }
            _ => panic!("expected expr stmt"),
        }
    }

    #[test]
    fn ident_is_keyword() {
        // parse_ident should fail for keywords
        assert!(Stream::new("return").parse_ident().is_err());
    }

    #[test]
    fn stmt_return() {
        let stmt = Stream::new("return 5").parse_stmt(0).unwrap();
        match &stmt.kind {
            StmtKind::Return(e) => {
                assert!(matches!(&e.kind, ExprKind::Literal(l) if l.is_number()));
            }
            _ => panic!("expected return stmt"),
        }
    }

    #[test]
    fn stmt_if_inline() {
        let stmt = Stream::new("if x: y").parse_stmt(0).unwrap();
        match &stmt.kind {
            StmtKind::If {
                cond,
                then_branch,
                else_branch: None,
            } => {
                assert!(matches!(&cond.kind, ExprKind::Ident(i) if i.name == "x"));
                assert_eq!(then_branch.len(), 1);
                match &then_branch[0].kind {
                    StmtKind::Expr(e) => {
                        assert!(matches!(&e.kind, ExprKind::Ident(i) if i.name == "y"))
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
        let stmt = Stream::new(src).parse_stmt(0).unwrap();
        match &stmt.kind {
            StmtKind::If {
                cond,
                then_branch,
                else_branch: None,
            } => {
                assert!(matches!(&cond.kind, ExprKind::Ident(i) if i.name == "x"));
                assert_eq!(then_branch.len(), 1);
                match &then_branch[0].kind {
                    StmtKind::Expr(e) => {
                        assert!(matches!(&e.kind, ExprKind::Ident(i) if i.name == "y"))
                    }
                    _ => panic!("expected expr in block"),
                }
            }
            _ => panic!("expected if stmt"),
        }
    }

    #[test]
    fn stmt_if_inline_else_same_line() {
        let stmt = Stream::new("if x: y else: z").parse_stmt(0).unwrap();
        match &stmt.kind {
            StmtKind::If {
                cond,
                then_branch,
                else_branch: Some(else_branch),
            } => {
                assert!(matches!(&cond.kind, ExprKind::Ident(i) if i.name == "x"));
                assert_eq!(then_branch.len(), 1);
                match &then_branch[0].kind {
                    StmtKind::Expr(e) => {
                        assert!(matches!(&e.kind, ExprKind::Ident(i) if i.name == "y"))
                    }
                    _ => panic!("expected inline expr as then-branch"),
                }
                assert_eq!(else_branch.len(), 1);
                match &else_branch[0].kind {
                    StmtKind::Expr(e) => {
                        assert!(matches!(&e.kind, ExprKind::Ident(i) if i.name == "z"))
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
        let stmt = Stream::new(src).parse_stmt(0).unwrap();
        match &stmt.kind {
            StmtKind::If {
                cond,
                then_branch,
                else_branch: Some(else_branch),
            } => {
                assert!(matches!(&cond.kind, ExprKind::Ident(i) if i.name == "x"));
                match &then_branch[0].kind {
                    StmtKind::Expr(e) => {
                        assert!(matches!(&e.kind, ExprKind::Ident(i) if i.name == "y"))
                    }
                    _ => panic!("expected inline expr as then-branch"),
                }
                assert_eq!(else_branch.len(), 1);
                match &else_branch[0].kind {
                    StmtKind::Expr(e) => {
                        assert!(matches!(&e.kind, ExprKind::Ident(i) if i.name == "z"))
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
        let stmt = Stream::new(src).parse_stmt(0).unwrap();
        match &stmt.kind {
            StmtKind::If {
                cond,
                then_branch,
                else_branch: Some(else_branch),
            } => {
                assert!(matches!(&cond.kind, ExprKind::Ident(i) if i.name == "x"));
                assert_eq!(then_branch.len(), 1);
                match &then_branch[0].kind {
                    StmtKind::Expr(e) => {
                        assert!(matches!(&e.kind, ExprKind::Ident(i) if i.name == "y"))
                    }
                    _ => panic!("expected expr in block"),
                }
                assert_eq!(else_branch.len(), 1);
                match &else_branch[0].kind {
                    StmtKind::Expr(e) => {
                        assert!(matches!(&e.kind, ExprKind::Ident(i) if i.name == "z"))
                    }
                    _ => panic!("expected expr in else block"),
                }
            }
            _ => panic!("expected if-else stmt"),
        }
    }

    #[test]
    fn stmt_assign_pattern_ident() {
        let stmt = Stream::new("x = 1\n").parse_stmt(0).unwrap();
        match &stmt.kind {
            StmtKind::AssignPattern { target, value } => {
                assert!(matches!(target.kind, PatternKind::Ident(ref id) if id.name == "x"));
                assert!(matches!(&value.kind, ExprKind::Literal(l) if l.is_number()));
            }
            _ => panic!("expected assign pattern"),
        }
    }

    #[test]
    fn stmt_assign_pattern_list() {
        let stmt = Stream::new("[a, b] = [1, 2]\n").parse_stmt(0).unwrap();
        match &stmt.kind {
            StmtKind::AssignPattern { target, value } => {
                if let PatternKind::List(items) = &target.kind {
                    assert_eq!(items.len(), 2);
                    assert!(matches!(&items[0].kind, PatternKind::Ident(i) if i.name == "a"));
                    assert!(matches!(&items[1].kind, PatternKind::Ident(i) if i.name == "b"));
                } else {
                    panic!("expected list pattern")
                }
                if let ExprKind::List(vals) = &value.kind {
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
        let stmt = Stream::new("obj.x = 5\n").parse_stmt(0).unwrap();
        match &stmt.kind {
            StmtKind::AssignField {
                target,
                field,
                value,
            } => {
                assert_eq!(field.name, "x");
                assert!(matches!(&target.kind, ExprKind::Ident(i) if i.name == "obj"));
                assert!(matches!(&value.kind, ExprKind::Literal(l) if l.is_number()));
            }
            _ => panic!("expected assign field"),
        }
    }

    #[test]
    fn stmt_assign_index() {
        let stmt = Stream::new("arr[0] = 7\n").parse_stmt(0).unwrap();
        match &stmt.kind {
            StmtKind::AssignIndex {
                target,
                index,
                value,
            } => {
                assert!(matches!(&target.kind, ExprKind::Ident(i) if i.name == "arr"));
                assert!(matches!(&index.kind, ExprKind::Literal(l) if l.is_number()));
                assert!(matches!(&value.kind, ExprKind::Literal(l) if l.is_number()));
            }
            _ => panic!("expected assign index"),
        }
    }

    #[test]
    fn stmt_while_block_else_block() {
        let src = "while x:\n    y\nelse:\n    z";
        let stmt = Stream::new(src).parse_stmt(0).unwrap();
        match &stmt.kind {
            StmtKind::While {
                cond,
                body: loop_branch,
                else_branch: Some(else_branch),
            } => {
                assert!(matches!(&cond.kind, ExprKind::Ident(i) if i.name == "x"));
                assert_eq!(loop_branch.len(), 1);
                match &loop_branch[0].kind {
                    StmtKind::Expr(e) => {
                        assert!(matches!(&e.kind, ExprKind::Ident(i) if i.name == "y"))
                    }
                    _ => panic!("expected expr in loop branch"),
                }
                assert_eq!(else_branch.len(), 1);
                match &else_branch[0].kind {
                    StmtKind::Expr(e) => {
                        assert!(matches!(&e.kind, ExprKind::Ident(i) if i.name == "z"))
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
        let stmt = Stream::new(src).parse_stmt(0).unwrap();
        match &stmt.kind {
            StmtKind::While {
                cond,
                body: loop_branch,
                else_branch: Some(else_branch),
            } => {
                assert!(matches!(&cond.kind, ExprKind::Ident(i) if i.name == "x"));
                assert_eq!(loop_branch.len(), 1);
                match &loop_branch[0].kind {
                    StmtKind::Expr(e) => {
                        assert!(matches!(&e.kind, ExprKind::Ident(i) if i.name == "y"))
                    }
                    _ => panic!("expected inline expr as loop body"),
                }
                assert_eq!(else_branch.len(), 1);
                match &else_branch[0].kind {
                    StmtKind::Expr(e) => {
                        assert!(matches!(&e.kind, ExprKind::Ident(i) if i.name == "z"))
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
        let stmt = Stream::new(src).parse_stmt(0).unwrap();
        match &stmt.kind {
            StmtKind::For {
                target,
                iter,
                body: loop_branch,
                else_branch: Some(else_branch),
            } => {
                // target should be identifier pattern 'a'
                match &target.kind {
                    PatternKind::Ident(id) => assert_eq!(id.name, "a"),
                    _ => panic!("expected ident pattern as for target"),
                }
                assert!(matches!(&iter.kind, ExprKind::Ident(i) if i.name == "arr"));
                assert_eq!(loop_branch.len(), 1);
                match &loop_branch[0].kind {
                    StmtKind::Expr(e) => {
                        assert!(matches!(&e.kind, ExprKind::Ident(i) if i.name == "b"))
                    }
                    _ => panic!("expected expr in loop branch"),
                }
                assert_eq!(else_branch.len(), 1);
                match &else_branch[0].kind {
                    StmtKind::Expr(e) => {
                        assert!(matches!(&e.kind, ExprKind::Ident(i) if i.name == "c"))
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
        let stmt = Stream::new(src).parse_stmt(0).unwrap();
        match &stmt.kind {
            StmtKind::For {
                target,
                iter,
                body: loop_branch,
                else_branch: Some(else_branch),
            } => {
                match &target.kind {
                    PatternKind::Ident(id) => assert_eq!(id.name, "a"),
                    _ => panic!("expected ident pattern as for target"),
                }
                assert!(matches!(&iter.kind, ExprKind::Ident(i) if i.name == "arr"));
                assert_eq!(loop_branch.len(), 1);
                match &loop_branch[0].kind {
                    StmtKind::Expr(e) => {
                        assert!(matches!(&e.kind, ExprKind::Ident(i) if i.name == "b"))
                    }
                    _ => panic!("expected inline expr as loop body"),
                }
                assert_eq!(else_branch.len(), 1);
                match &else_branch[0].kind {
                    StmtKind::Expr(e) => {
                        assert!(matches!(&e.kind, ExprKind::Ident(i) if i.name == "c"))
                    }
                    _ => panic!("expected inline expr as else branch"),
                }
            }
            _ => panic!("expected for stmt with else"),
        }
    }
}
