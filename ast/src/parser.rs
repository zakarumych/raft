use alloc::rc::Rc;
use core::fmt;

use crate::{
    Atom, BinOp, BinOpKind, Export, Expr, ExprKind, ExprRecordField, FnCat, Ident, Lit, Module,
    Pat, PatKind, PatRecordField, Span, Stmt, StmtKind, UnOp, UnOpKind,
};

use raft_lexer::{Spacing, SpannedSource, Token};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ParseErrorKind {
    UnexpectedToken,
    UnexpectedEndOfInput,
    InvalidAssignmentTarget,
    /// A literal the lexer accepted but the language rejects, e.g. a number
    /// with an unknown suffix (`1z`; only `i` and `f` are valid).
    InvalidLiteral,
}

impl fmt::Display for ParseErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseErrorKind::UnexpectedToken => write!(f, "Unexpected token"),
            ParseErrorKind::UnexpectedEndOfInput => write!(f, "Unexpected end of input"),
            ParseErrorKind::InvalidAssignmentTarget => write!(f, "Invalid assignment target"),
            ParseErrorKind::InvalidLiteral => write!(f, "Invalid literal"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ParseError {
    span: Span,
    kind: ParseErrorKind,
}

impl ParseError {
    pub fn new(kind: ParseErrorKind, span: Span) -> Self {
        Self { span, kind }
    }

    pub fn span(&self) -> Span {
        self.span
    }

    pub fn kind(&self) -> ParseErrorKind {
        self.kind
    }

    pub const fn print<'a>(&'a self, source: &'a str) -> PrintParseError<'a> {
        PrintParseError {
            kind: self.kind,
            spanned_source: SpannedSource::new(source, self.span),
        }
    }
}

pub struct PrintParseError<'a> {
    kind: ParseErrorKind,
    spanned_source: SpannedSource<'a>,
}

impl fmt::Display for PrintParseError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.kind)?;
        write!(f, "{}", self.spanned_source)
    }
}

pub type ParseResult<T> = Result<T, ParseError>;

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
    Fn,
    Export,
    Yield,
    Gen,
    Async,
    Await,
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
            "fn" => Some(Keyword::Fn),
            "export" => Some(Keyword::Export),
            "yield" => Some(Keyword::Yield),
            "gen" => Some(Keyword::Gen),
            "async" => Some(Keyword::Async),
            "await" => Some(Keyword::Await),
            _ => None,
        }
    }

    pub fn peek(stream: &mut TokenStream) -> Option<(Self, Span)> {
        match stream.peek() {
            Some(Token::Ident(i)) => Some((Self::from_ident(i.repr())?, i.span())),
            _ => None,
        }
    }
}

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

    pub fn parse_literal(&mut self) -> ParseResult<Lit> {
        Lit::parse(self)
    }

    pub fn parse_unary_op(&mut self) -> ParseResult<UnOp> {
        UnOp::parse(self)
    }

    pub fn parse_binary_op(&mut self) -> ParseResult<BinOp> {
        BinOp::parse(self)
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

    pub fn parse_module(&mut self) -> ParseResult<Module> {
        Module::parse(self)
    }

    pub fn start_span(&self) -> Span {
        if self.pos == self.tokens.len() {
            self.end_span()
        } else {
            self.tokens[self.pos].span().start()
        }
    }

    pub fn end_span(&self) -> Span {
        match self.tokens.last() {
            None => Span { start: 0, end: 0 },
            Some(tok) => tok.span().end(),
        }
    }

    pub fn span(&self) -> Span {
        if self.pos == self.tokens.len() {
            self.end_span()
        } else {
            let start = self.tokens[self.pos].span().start;
            let end = self.tokens.last().unwrap().span().end;
            Span { start, end }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    pub fn peek(&self) -> Option<Token> {
        self.tokens.get(self.pos).cloned()
    }

    pub fn peek1(&self) -> Option<Token> {
        self.tokens.get(self.pos + 1).cloned()
    }

    pub fn advance(&mut self) {
        self.pos += 1;
    }

    pub fn skip_newline(&mut self) -> bool {
        if let Some(Token::Newline(_)) = self.peek() {
            self.advance();
            true
        } else {
            false
        }
    }

    pub fn skip_comments(&mut self) -> bool {
        if let Some(Token::Comment(_)) = self.peek() {
            self.advance();
            true
        } else {
            false
        }
    }

    pub fn skip_comments_and_newlines(&mut self) -> bool {
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

    pub fn expect_end(&self) -> ParseResult<()> {
        if let Some(tok) = self.peek() {
            Err(ParseError::new(ParseErrorKind::UnexpectedToken, tok.span()))
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
    //             if s.chars().next().map_or(false, |c| !c.is_uppercase()) {
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
                    return Err(ParseError::new(ParseErrorKind::UnexpectedToken, i.span()));
                }

                let s = i.repr();
                if s.chars().next().map_or(false, |c| !c.is_uppercase()) {
                    stream.advance();
                    let name = i.rc_repr();
                    let span = i.span();
                    Ok(Ident { name, span })
                } else {
                    Err(ParseError::new(ParseErrorKind::UnexpectedToken, i.span()))
                }
            }
            Some(tok) => Err(ParseError::new(ParseErrorKind::UnexpectedToken, tok.span())),
            None => Err(ParseError::new(
                ParseErrorKind::UnexpectedEndOfInput,
                stream.end_span(),
            )),
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
    //             if s.chars().next().map_or(false, |c| c.is_uppercase()) {
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
                    return Err(ParseError::new(ParseErrorKind::UnexpectedToken, i.span()));
                }

                let s = i.repr();
                if s.chars().next().map_or(false, |c| c.is_uppercase()) {
                    stream.advance();
                    let name = i.rc_repr();
                    let span = i.span();
                    Ok(Atom { name, span })
                } else {
                    Err(ParseError::new(ParseErrorKind::UnexpectedToken, i.span()))
                }
            }
            Some(tok) => Err(ParseError::new(ParseErrorKind::UnexpectedToken, tok.span())),
            None => Err(ParseError::new(
                ParseErrorKind::UnexpectedEndOfInput,
                stream.end_span(),
            )),
        }
    }
}

enum AtomIdent {
    Atom(Atom),
    Ident(Ident),
}

impl AtomIdent {
    fn from_ident(ident: raft_lexer::Ident) -> Result<Self, ParseError> {
        let s = ident.repr();
        let starts_with_uppercase = s.chars().next().map_or(false, |c| c.is_uppercase());

        if starts_with_uppercase {
            Ok(AtomIdent::Atom(Atom {
                name: ident.rc_repr(),
                span: ident.span(),
            }))
        } else {
            Ok(AtomIdent::Ident(Ident {
                name: ident.rc_repr(),
                span: ident.span(),
            }))
        }
    }
}

impl Lit {
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

    fn parse(stream: &mut TokenStream) -> ParseResult<Lit> {
        match stream.peek() {
            Some(Token::Literal(l)) => {
                stream.advance();
                literal_from_lexer(l.clone())
            }
            Some(tok) => Err(ParseError::new(ParseErrorKind::UnexpectedToken, tok.span())),
            None => Err(ParseError::new(
                ParseErrorKind::UnexpectedEndOfInput,
                stream.end_span(),
            )),
        }
    }
}

/// Convert a lexer literal into an AST literal, validating what the lexer
/// is agnostic about: number suffixes (only `i` and `f` are part of the
/// language). All literal parsing - expressions, patterns, `Literal::parse`
/// - funnels through here.
fn literal_from_lexer(lit: raft_lexer::Lit) -> ParseResult<Lit> {
    match lit {
        raft_lexer::Lit::Num(n) => match n.suffix() {
            None | Some("i" | "f") => Ok(Lit::Num(n)),
            Some(_) => Err(ParseError::new(ParseErrorKind::InvalidLiteral, n.span())),
        },
        raft_lexer::Lit::Char(c) => Ok(Lit::Char(c)),
        raft_lexer::Lit::Str(s) => Ok(Lit::Str(s)),
    }
}

impl UnOp {
    fn peek(stream: &TokenStream) -> Option<UnOp> {
        match stream.peek() {
            Some(Token::Punct(p)) => {
                let ch = p.repr();
                let kind = match ch {
                    '!' => Some(UnOpKind::Not),
                    '~' => Some(UnOpKind::BitNot),
                    '+' => Some(UnOpKind::Pos),
                    '-' => Some(UnOpKind::Neg),
                    _ => None,
                };
                if let Some(k) = kind {
                    let span = p.span();
                    Some(UnOp { kind: k, span })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn parse(stream: &mut TokenStream) -> ParseResult<UnOp> {
        match stream.peek() {
            Some(Token::Punct(p)) => {
                let ch = p.repr();
                let kind = match ch {
                    '!' => Some(UnOpKind::Not),
                    '~' => Some(UnOpKind::BitNot),
                    '+' => Some(UnOpKind::Pos),
                    '-' => Some(UnOpKind::Neg),
                    _ => None,
                };
                if let Some(k) = kind {
                    let span = p.span();
                    stream.advance();
                    Ok(UnOp { kind: k, span })
                } else {
                    Err(ParseError::new(ParseErrorKind::UnexpectedToken, p.span()))
                }
            }
            Some(tok) => Err(ParseError::new(ParseErrorKind::UnexpectedToken, tok.span())),
            None => Err(ParseError::new(
                ParseErrorKind::UnexpectedEndOfInput,
                stream.end_span(),
            )),
        }
    }
}

impl BinOp {
    fn peek(stream: &TokenStream) -> Option<BinOp> {
        match stream.peek() {
            Some(Token::Punct(p1)) => match (p1.repr(), stream.peek1()) {
                ('<', Some(Token::Punct(p2)))
                    if p1.spacing() == Spacing::Joint && p2.repr() == '<' =>
                {
                    return Some(BinOp {
                        kind: BinOpKind::Shl,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('>', Some(Token::Punct(p2)))
                    if p1.spacing() == Spacing::Joint && p2.repr() == '>' =>
                {
                    return Some(BinOp {
                        kind: BinOpKind::Shr,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('*', Some(Token::Punct(p2)))
                    if p1.spacing() == Spacing::Joint && p2.repr() == '*' =>
                {
                    return Some(BinOp {
                        kind: BinOpKind::Pow,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('=', Some(Token::Punct(p2)))
                    if p1.spacing() == Spacing::Joint && p2.repr() == '=' =>
                {
                    return Some(BinOp {
                        kind: BinOpKind::Eq,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('!', Some(Token::Punct(p2)))
                    if p1.spacing() == Spacing::Joint && p2.repr() == '=' =>
                {
                    return Some(BinOp {
                        kind: BinOpKind::Ne,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('<', Some(Token::Punct(p2)))
                    if p1.spacing() == Spacing::Joint && p2.repr() == '=' =>
                {
                    return Some(BinOp {
                        kind: BinOpKind::Le,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('>', Some(Token::Punct(p2)))
                    if p1.spacing() == Spacing::Joint && p2.repr() == '=' =>
                {
                    return Some(BinOp {
                        kind: BinOpKind::Ge,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('&', _) => {
                    return Some(BinOp {
                        kind: BinOpKind::BitAnd,
                        span: p1.span(),
                    });
                }
                ('|', _) => {
                    return Some(BinOp {
                        kind: BinOpKind::BitOr,
                        span: p1.span(),
                    });
                }
                ('^', _) => {
                    return Some(BinOp {
                        kind: BinOpKind::BitXor,
                        span: p1.span(),
                    });
                }
                ('*', _) => {
                    return Some(BinOp {
                        kind: BinOpKind::Mul,
                        span: p1.span(),
                    });
                }
                ('/', _) => {
                    return Some(BinOp {
                        kind: BinOpKind::Div,
                        span: p1.span(),
                    });
                }
                ('+', _) => {
                    return Some(BinOp {
                        kind: BinOpKind::Add,
                        span: p1.span(),
                    });
                }
                ('-', _) => {
                    return Some(BinOp {
                        kind: BinOpKind::Sub,
                        span: p1.span(),
                    });
                }
                ('<', _) => {
                    return Some(BinOp {
                        kind: BinOpKind::Lt,
                        span: p1.span(),
                    });
                }
                ('>', _) => {
                    return Some(BinOp {
                        kind: BinOpKind::Gt,
                        span: p1.span(),
                    });
                }
                _ => None,
            },
            _ => None,
        }
    }

    fn parse(stream: &mut TokenStream) -> ParseResult<BinOp> {
        match stream.peek() {
            Some(Token::Punct(p1)) => match (p1.repr(), stream.peek1()) {
                ('<', Some(Token::Punct(p2))) if p2.repr() == '<' => {
                    stream.advance();
                    stream.advance();
                    return Ok(BinOp {
                        kind: BinOpKind::Shl,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('>', Some(Token::Punct(p2))) if p2.repr() == '>' => {
                    stream.advance();
                    stream.advance();
                    return Ok(BinOp {
                        kind: BinOpKind::Shr,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('*', Some(Token::Punct(p2))) if p2.repr() == '*' => {
                    stream.advance();
                    stream.advance();
                    return Ok(BinOp {
                        kind: BinOpKind::Pow,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('=', Some(Token::Punct(p2))) if p2.repr() == '=' => {
                    stream.advance();
                    stream.advance();
                    return Ok(BinOp {
                        kind: BinOpKind::Eq,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('!', Some(Token::Punct(p2))) if p2.repr() == '=' => {
                    stream.advance();
                    stream.advance();
                    return Ok(BinOp {
                        kind: BinOpKind::Ne,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('<', Some(Token::Punct(p2))) if p2.repr() == '=' => {
                    stream.advance();
                    stream.advance();
                    return Ok(BinOp {
                        kind: BinOpKind::Le,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('>', Some(Token::Punct(p2))) if p2.repr() == '=' => {
                    stream.advance();
                    stream.advance();
                    return Ok(BinOp {
                        kind: BinOpKind::Ge,
                        span: p1.span().join(p2.span()),
                    });
                }
                ('&', _) => {
                    stream.advance();
                    return Ok(BinOp {
                        kind: BinOpKind::BitAnd,
                        span: p1.span(),
                    });
                }
                ('|', _) => {
                    stream.advance();
                    return Ok(BinOp {
                        kind: BinOpKind::BitOr,
                        span: p1.span(),
                    });
                }
                ('^', _) => {
                    stream.advance();
                    return Ok(BinOp {
                        kind: BinOpKind::BitXor,
                        span: p1.span(),
                    });
                }
                ('*', _) => {
                    stream.advance();
                    return Ok(BinOp {
                        kind: BinOpKind::Mul,
                        span: p1.span(),
                    });
                }
                ('/', _) => {
                    stream.advance();
                    return Ok(BinOp {
                        kind: BinOpKind::Div,
                        span: p1.span(),
                    });
                }
                ('+', _) => {
                    stream.advance();
                    return Ok(BinOp {
                        kind: BinOpKind::Add,
                        span: p1.span(),
                    });
                }
                ('-', _) => {
                    stream.advance();
                    return Ok(BinOp {
                        kind: BinOpKind::Sub,
                        span: p1.span(),
                    });
                }
                ('<', _) => {
                    stream.advance();
                    return Ok(BinOp {
                        kind: BinOpKind::Lt,
                        span: p1.span(),
                    });
                }
                ('>', _) => {
                    stream.advance();
                    return Ok(BinOp {
                        kind: BinOpKind::Gt,
                        span: p1.span(),
                    });
                }
                (_, _) => {
                    return Err(ParseError::new(ParseErrorKind::UnexpectedToken, p1.span()));
                }
            },
            Some(tok) => Err(ParseError::new(ParseErrorKind::UnexpectedToken, tok.span())),
            None => Err(ParseError::new(
                ParseErrorKind::UnexpectedEndOfInput,
                stream.end_span(),
            )),
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
            match BinOp::peek(stream) {
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
                    if BinOp::peek(stream).is_some() {
                        break;
                    }
                    if UnOp::peek(stream).is_none() {
                        break;
                    }
                }
                Some(Token::Ident(ident)) => match Keyword::from_ident(ident.repr()) {
                    Some(_) => break,
                    None => {}
                },
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
        if let Ok(op) = UnOp::parse(stream) {
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
                    if field.name() == "_" {
                        return Err(ParseError::new(
                            ParseErrorKind::UnexpectedToken,
                            field.span(),
                        ));
                    }

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

                let expr = Expr::parse(&mut group_stream)?;
                group_stream.skip_comments_and_newlines();
                group_stream.expect_end()?;

                stream.advance();

                Ok(match expr.kind {
                    ExprKind::Parenthesized(kind) => Expr {
                        kind: ExprKind::Parenthesized(kind),
                        span: g.span(),
                    },
                    _ => Expr {
                        span: g.span(),
                        kind: ExprKind::Parenthesized(Rc::new(expr)),
                    },
                })
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
                        Some(tok) => {
                            return Err(ParseError::new(
                                ParseErrorKind::UnexpectedToken,
                                tok.span(),
                            ));
                        }
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
                    if key.name() == "_" {
                        return Err(ParseError::new(ParseErrorKind::UnexpectedToken, key.span()));
                    }

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
                                    return Err(ParseError::new(
                                        ParseErrorKind::UnexpectedToken,
                                        tok.span(),
                                    ));
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
                            return Err(ParseError::new(
                                ParseErrorKind::UnexpectedToken,
                                tok.span(),
                            ));
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
                    kind: ExprKind::Literal(literal_from_lexer(lit)?),
                    span,
                })
            }
            Some(Token::Ident(i)) => {
                stream.advance();

                let span = i.span();
                match AtomIdent::from_ident(i) {
                    Ok(AtomIdent::Atom(a)) => Ok(Expr {
                        kind: ExprKind::Atom(a),
                        span,
                    }),
                    Ok(AtomIdent::Ident(id)) => Ok(Expr {
                        kind: ExprKind::Ident(id),
                        span,
                    }),
                    Err(e) => Err(e),
                }
            }
            Some(tok) => Err(ParseError::new(ParseErrorKind::UnexpectedToken, tok.span())),
            None => Err(ParseError::new(
                ParseErrorKind::UnexpectedEndOfInput,
                stream.end_span(),
            )),
        }
    }
}

fn expr_to_pattern(expr: &Expr) -> Option<Pat> {
    match &expr.kind {
        ExprKind::Ident(i) => Some(Pat {
            span: expr.span(),
            kind: PatKind::Ident(i.clone()),
        }),
        ExprKind::Atom(a) => Some(Pat {
            span: expr.span(),
            kind: PatKind::Atom(a.clone()),
        }),
        ExprKind::Literal(l) => Some(Pat {
            span: expr.span(),
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
                span: expr.span(),
                kind: PatKind::List(Rc::from(pats)),
            })
        }
        ExprKind::Record(fields) => {
            let mut pats = Vec::new();
            for f in fields.iter() {
                if let Some(value) = &f.value {
                    if let Some(pattern) = expr_to_pattern(value) {
                        pats.push(PatRecordField {
                            key: f.key.clone(),
                            pattern: Some(pattern),
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
                span: expr.span(),
                kind: PatKind::Record(Rc::from(pats)),
            })
        }
        ExprKind::Parenthesized(expr) => expr_to_pattern(expr),
        ExprKind::Apply(..) => None,
        ExprKind::Unary(..) => None,
        ExprKind::Binary(..) => None,
        ExprKind::Field(..) => None,
        ExprKind::Index(..) => None,
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
                        Some(tok) => {
                            return Err(ParseError::new(
                                ParseErrorKind::UnexpectedToken,
                                tok.span(),
                            ));
                        }
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
                    if key.name() == "_" {
                        return Err(ParseError::new(ParseErrorKind::UnexpectedToken, key.span()));
                    }

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
                                    return Err(ParseError::new(
                                        ParseErrorKind::UnexpectedToken,
                                        tok.span(),
                                    ));
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
                            return Err(ParseError::new(
                                ParseErrorKind::UnexpectedToken,
                                tok.span(),
                            ));
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
                    kind: PatKind::Literal(literal_from_lexer(lit)?),
                })
            }
            Some(Token::Ident(i)) => {
                stream.advance();

                let span = i.span();
                match AtomIdent::from_ident(i) {
                    Ok(AtomIdent::Atom(a)) => Ok(Pat {
                        kind: PatKind::Atom(a),
                        span,
                    }),
                    Ok(AtomIdent::Ident(id)) => Ok(Pat {
                        kind: PatKind::Ident(id),
                        span,
                    }),
                    Err(e) => Err(e),
                }
            }
            Some(tok) => Err(ParseError::new(ParseErrorKind::UnexpectedToken, tok.span())),
            None => Err(ParseError::new(
                ParseErrorKind::UnexpectedEndOfInput,
                stream.end_span(),
            )),
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
                        if let Some(pattern) = expr_to_pattern(&lhs) {
                            return Ok(Stmt {
                                span: pattern.span.join(rhs.span),
                                kind: StmtKind::AssignPat {
                                    target: pattern,
                                    value: rhs,
                                },
                            });
                        } else {
                            return Err(ParseError::new(
                                ParseErrorKind::InvalidAssignmentTarget,
                                lhs.span,
                            ));
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
        if let Some((kw, kw_span)) = Keyword::peek(stream) {
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
                Keyword::Yield => {
                    stream.advance();
                    stream.skip_comments();

                    if let Some(Token::Newline(_)) | None = stream.peek() {
                        return Ok(Stmt {
                            span: kw_span,
                            kind: StmtKind::Yield(None),
                        });
                    }

                    let expr = Expr::parse(stream)?;
                    return Ok(Stmt {
                        span: kw_span.join(expr.span),
                        kind: StmtKind::Yield(Some(expr)),
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
                _ => return Err(ParseError::new(ParseErrorKind::UnexpectedToken, kw_span)),
            }
        }

        Self::parse_simple(stream)
    }

    pub fn parse(stream: &mut TokenStream) -> ParseResult<Self> {
        if let Some((kw, kw_span)) = Keyword::peek(stream) {
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
                Keyword::Yield => {
                    stream.advance();
                    stream.skip_comments();
                    if let Some(Token::Newline(_)) | None = stream.peek() {
                        return Ok(Stmt {
                            span: kw_span,
                            kind: StmtKind::Yield(None),
                        });
                    }

                    let expr = Expr::parse(stream)?;
                    return Ok(Stmt {
                        span: kw_span.join(expr.span),
                        kind: StmtKind::Yield(Some(expr)),
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
                    stream.advance();
                    return Self::parse_if(stream, kw_span);
                }
                Keyword::While => {
                    stream.advance();
                    return Self::parse_while(stream, kw_span);
                }
                Keyword::For => {
                    stream.advance();
                    return Self::parse_for(stream, kw_span);
                }
                Keyword::Fn => {
                    stream.advance();
                    return Self::parse_fn(stream, FnCat::Normal, kw_span);
                }
                Keyword::Gen => {
                    stream.advance();
                    return Self::parse_gen(stream, false, kw_span);
                }
                Keyword::Async => {
                    stream.advance();
                    return Self::parse_async(stream, kw_span);
                }
                _ => return Err(ParseError::new(ParseErrorKind::UnexpectedToken, kw_span)),
            }
        }

        Self::parse_simple(stream)
    }

    pub fn parse_many(stream: &mut TokenStream) -> ParseResult<Vec<Self>> {
        let mut stmts = Vec::new();
        stream.skip_comments_and_newlines();
        while !stream.is_empty() {
            let statement = Self::parse(stream)?;
            stmts.push(statement);
            stream.skip_comments_and_newlines();
        }
        Ok(stmts)
    }

    pub fn parse_branch(stream: &mut TokenStream) -> ParseResult<Vec<Self>> {
        match stream.peek() {
            Some(Token::Punct(p)) if p.repr() == ':' => {
                stream.advance();
                stream.skip_comments();
                if stream.skip_newline() {
                    stream.skip_comments();
                    match stream.peek() {
                        Some(Token::Group(g)) if g.delimiter() == raft_lexer::Delimiter::Block => {
                            stream.advance();

                            let mut group_stream = TokenStream::new(g.rc_tokens());
                            return Stmt::parse_many(&mut group_stream);
                        }
                        Some(Token::Newline(_)) => {
                            // 2nd newline means empty branch
                            // It is only possible in REPL, as non-repl lexer will skip blank lines.
                            return Ok(Vec::new());
                        }
                        Some(tok) => {
                            return Err(ParseError::new(
                                ParseErrorKind::UnexpectedToken,
                                tok.span(),
                            ));
                        }
                        None => {
                            return Ok(Vec::new());
                        }
                    }
                } else {
                    let statement = Stmt::parse_line(stream)?;
                    return Ok(vec![statement]);
                }
            }
            Some(tok) => return Err(ParseError::new(ParseErrorKind::UnexpectedToken, tok.span())),
            None => {
                return Err(ParseError::new(
                    ParseErrorKind::UnexpectedEndOfInput,
                    stream.end_span(),
                ));
            }
        }
    }

    pub fn parse_if(stream: &mut TokenStream, if_span: Span) -> ParseResult<Self> {
        let cond = Expr::parse(stream)?;
        let then_branch = Self::parse_branch(stream)?;
        stream.skip_comments_and_newlines();

        let else_branch = Self::parse_else(stream)?;

        let last_stmt = else_branch
            .as_ref()
            .and_then(|b| b.last())
            .or(then_branch.last());
        let span = if_span.join(last_stmt.map_or(cond.span, |s| s.span));

        Ok(Stmt {
            span,
            kind: StmtKind::If {
                cond,
                then_branch: then_branch.into(),
                else_branch: else_branch.map(|b| b.into()),
            },
        })
    }

    pub fn parse_else(stream: &mut TokenStream) -> ParseResult<Option<Vec<Self>>> {
        if let Some((Keyword::Else, _else_span)) = Keyword::peek(stream) {
            stream.advance();

            let stmts = if let Some((Keyword::If, if_span)) = Keyword::peek(stream) {
                stream.advance();
                vec![Self::parse_if(stream, if_span)?]
            } else {
                Self::parse_branch(stream)?
            };

            Ok(Some(stmts))
        } else {
            Ok(None)
        }
    }

    pub fn parse_while(stream: &mut TokenStream, while_span: Span) -> ParseResult<Self> {
        let cond = Expr::parse(stream)?;
        let body = Self::parse_branch(stream)?;
        stream.skip_comments_and_newlines();

        let else_branch = Self::parse_else(stream)?;

        let last_stmt = else_branch.as_ref().and_then(|b| b.last()).or(body.last());
        let span = while_span.join(last_stmt.map_or(cond.span, |s| s.span));

        Ok(Stmt {
            span,
            kind: StmtKind::While {
                cond,
                body: body.into(),
                else_branch: else_branch.map(|b| b.into()),
            },
        })
    }

    pub fn parse_for(stream: &mut TokenStream, for_span: Span) -> ParseResult<Self> {
        let target = Pat::parse(stream)?;

        match Keyword::peek(stream) {
            Some((Keyword::In, _)) => {
                stream.advance();
            }
            Some((_, kw_span)) => {
                return Err(ParseError::new(ParseErrorKind::UnexpectedToken, kw_span));
            }
            None => {
                return Err(ParseError::new(
                    ParseErrorKind::UnexpectedEndOfInput,
                    stream.end_span(),
                ));
            }
        }

        let iterable = Expr::parse(stream)?;

        let body = Self::parse_branch(stream)?;
        stream.skip_comments_and_newlines();

        let else_branch = Self::parse_else(stream)?;

        let last_stmt = else_branch.as_ref().and_then(|b| b.last()).or(body.last());
        let span = for_span.join(last_stmt.map_or(iterable.span, |s| s.span));

        Ok(Stmt {
            span,
            kind: StmtKind::For {
                target,
                iterable,
                body: body.into(),
                else_branch: else_branch.map(|b| b.into()),
            },
        })
    }

    pub fn parse_fn(stream: &mut TokenStream, fn_cat: FnCat, fn_span: Span) -> ParseResult<Self> {
        let name = Ident::parse(stream)?;
        let mut params = Vec::new();

        loop {
            match stream.peek() {
                None => {
                    return Err(ParseError::new(
                        ParseErrorKind::UnexpectedEndOfInput,
                        stream.end_span(),
                    ));
                }
                Some(Token::Punct(p)) if p.repr() == ':' => {
                    break;
                }
                _ => {
                    let pattern = Pat::parse(stream)?;
                    params.push(pattern);
                }
            }
        }

        let body = Self::parse_branch(stream)?;

        let span = fn_span.join(body.last().map_or(name.span, |s| s.span));

        Ok(Stmt {
            span,
            kind: StmtKind::Fn {
                cat: fn_cat,
                name,
                params: params.into(),
                body: body.into(),
            },
        })
    }

    pub fn parse_gen(
        stream: &mut TokenStream,
        is_async: bool,
        gen_span: Span,
    ) -> ParseResult<Self> {
        match Keyword::peek(stream) {
            Some((Keyword::Fn, fn_span)) => {
                stream.advance();
                Self::parse_fn(
                    stream,
                    if is_async {
                        FnCat::AsyncGenerator
                    } else {
                        FnCat::Generator
                    },
                    gen_span.join(fn_span),
                )
            }
            Some((_, kw_span)) => {
                return Err(ParseError::new(ParseErrorKind::UnexpectedToken, kw_span));
            }
            None => {
                match stream.peek() {
                    None => {
                        return Err(ParseError::new(
                            ParseErrorKind::UnexpectedEndOfInput,
                            stream.end_span(),
                        ));
                    }
                    Some(tok) => {
                        return Err(ParseError::new(
                            ParseErrorKind::UnexpectedToken,
                            tok.span(),
                        ));
                    }
                }
            }
        }
    }

    pub fn parse_async(stream: &mut TokenStream, async_span: Span) -> ParseResult<Self> {
        match Keyword::peek(stream) {
            Some((Keyword::Fn, fn_span)) => {
                stream.advance();
                Self::parse_fn(stream, FnCat::Async, async_span.join(fn_span))
            }
            Some((Keyword::Gen, gen_span)) => {
                stream.advance();
                Self::parse_gen(stream, true, async_span.join(gen_span))
            }
            Some((_, kw_span)) => {
                return Err(ParseError::new(ParseErrorKind::UnexpectedToken, kw_span));
            }
            None => {
                match stream.peek() {
                    None => {
                        return Err(ParseError::new(
                            ParseErrorKind::UnexpectedEndOfInput,
                            stream.end_span(),
                        ));
                    }
                    Some(tok) => {
                        return Err(ParseError::new(
                            ParseErrorKind::UnexpectedToken,
                            tok.span(),
                        ));
                    }
                }
            }
        }
    }
}

impl Module {
    pub fn parse(stream: &mut TokenStream) -> ParseResult<Self> {
        let mut stmts = Vec::new();
        stream.skip_comments_and_newlines();
        while !stream.is_empty() {
            match Keyword::peek(stream) {
                Some((Keyword::Export, kw_span)) => {
                    stream.advance();
                    // the exported bindings use record syntax
                    let record = Expr::parse(stream)?;
                    let span = kw_span.join(record.span());
                    let ExprKind::Record(fields) = record.kind() else {
                        return Err(ParseError::new(
                            ParseErrorKind::UnexpectedToken,
                            record.span(),
                        ));
                    };
                    // export fields must be bare names (shorthand `{ key }` or
                    // `{ key: name }`), never arbitrary expressions: the
                    // compiled module has no way to compute a value that
                    // isn't already sitting in a known slot
                    for field in fields.iter() {
                        if let Some(value) = field.value() {
                            if !matches!(value.kind(), ExprKind::Ident(_)) {
                                return Err(ParseError::new(
                                    ParseErrorKind::UnexpectedToken,
                                    value.span(),
                                ));
                            }
                        }
                    }
                    // the export is the module's terminator: nothing may follow
                    stream.skip_comments_and_newlines();
                    stream.expect_end()?;
                    return Ok(Module {
                        stmts: Rc::from(stmts),
                        export: Export::new(fields.clone(), span),
                    });
                }
                _ => {}
            }

            let statement = Stmt::parse(stream)?;
            stmts.push(statement);
            stream.skip_comments_and_newlines();
        }

        Err(ParseError::new(
            ParseErrorKind::UnexpectedEndOfInput,
            stream.end_span(),
        ))
    }
}
