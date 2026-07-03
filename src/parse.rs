use std::rc::Rc;
use unicode_ident::{is_xid_continue, is_xid_start};

use crate::ast::{Atom, BinaryOp, Ident, Pattern, PatternKind, RecordPatternField, Span, Spanned, UnaryOp};
use crate::literal::{LexError, LexErrorKind, Literal};

#[derive(Debug)]
pub enum ParseError {
    UnexpectedChar(usize, char),
    UnexpectedEof(usize),
    ExpectedAtom(usize),
    ExpectedIdent(usize),
    Literal(LexError),
}

pub type ParseResult<T> = Result<T, ParseError>;

// ---- Expr -------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    Literal(Literal),
    Ident(Ident),
    Atom(Atom),
    List(Vec<Expr>),
    Record(Vec<ExprRecordField>),
    Unary(Spanned<UnaryOp>, Box<Expr>),
    Binary(Box<Expr>, Spanned<BinaryOp>, Box<Expr>),
    Apply(Box<Expr>, Vec<Expr>),
    Field(Box<Expr>, Ident),
    Index(Box<Expr>, Box<Expr>),
}

#[derive(Debug, Clone)]
pub struct ExprRecordField {
    pub key: Ident,
    pub value: Expr,
    pub span: Span,
}

// ---- Stream -----------------------------------------------------------------

pub struct Stream<'src> {
    src: &'src str,
    pos: usize,
}

pub struct StreamCharIndices<'src> {
    src: &'src str,
    offset: usize,
}

impl<'src> StreamCharIndices<'src> {
    fn new(src: &'src str) -> Self {
        Self { src, offset: 0 }
    }

    pub fn offset(&self) -> usize {
        self.offset
    }
}

impl Iterator for StreamCharIndices<'_> {
    type Item = (usize, char);

    fn next(&mut self) -> Option<Self::Item> {
        let ch = self.src.chars().next()?;
        let pos = self.offset;
        let len = ch.len_utf8();
        self.offset += len;
        self.src = &self.src[len..];
        Some((pos, ch))
    }
}

fn parse_escape(
    start: usize,
    mut chars: impl Iterator<Item = (usize, char)>,
) -> Result<char, LexError> {
    match chars.next() {
        Some((_, 'n')) => Ok('\n'),
        Some((_, 't')) => Ok('\t'),
        Some((_, 'r')) => Ok('\r'),
        Some((_, '\\')) => Ok('\\'),
        Some((_, '\'')) => Ok('\''),
        Some((_, '"')) => Ok('"'),
        Some((_, '0')) => Ok('\0'),
        Some((pos, ch)) => Err(LexError {
            span: Span::new(start + pos, start + pos + ch.len_utf8()),
            kind: LexErrorKind::InvalidEscapeSequence(ch),
        }),
        None => Err(LexError {
            span: Span::new(start, start),
            kind: LexErrorKind::EndOfInput,
        }),
    }
}

// ---- Stream impl ------------------------------------------------------------

impl<'src> Stream<'src> {
    pub fn new(src: &'src str) -> Self {
        Self { src, pos: 0 }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    fn peek(&self) -> Option<char> {
        self.src[self.pos..].chars().next()
    }

    fn peek2(&self) -> Option<char> {
        let mut chars = self.src[self.pos..].chars();
        chars.next();
        chars.next()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.consume(ch.len_utf8());
        Some(ch)
    }

    fn char_indices(&self) -> StreamCharIndices<'src> {
        StreamCharIndices::new(&self.src[self.pos..])
    }

    fn consume(&mut self, len: usize) -> &'src str {
        let start = self.pos;
        self.pos += len;
        &self.src[start..self.pos]
    }

    fn skip_whitespace_inline(&mut self) {
        while matches!(self.peek(), Some(' ') | Some('\t')) {
            self.advance();
        }
    }

    fn expect_char(&mut self, expected: char) -> Result<(), ParseError> {
        match self.peek() {
            Some(ch) if ch == expected => {
                self.advance();
                Ok(())
            }
            Some(ch) => Err(ParseError::UnexpectedChar(self.pos, ch)),
            None => Err(ParseError::UnexpectedEof(self.pos)),
        }
    }

    fn parse_word(&mut self) -> Option<(String, Span)> {
        let start = self.pos();
        let first = self.peek()?;
        if !is_xid_start(first) && first != '_' {
            return None;
        }
        let mut name = String::new();
        name.push(self.advance().unwrap());
        while let Some(ch) = self.peek() {
            if is_xid_continue(ch) {
                name.push(self.advance().unwrap());
            } else {
                break;
            }
        }
        Some((name, Span { start, end: self.pos() }))
    }

    pub fn parse_ident(&mut self) -> ParseResult<Ident> {
        let pos = self.pos();
        match self.peek() {
            Some(ch) if ch.is_uppercase() => return Err(ParseError::ExpectedIdent(pos)),
            Some(ch) if !is_xid_start(ch) && ch != '_' => {
                return Err(ParseError::UnexpectedChar(pos, ch));
            }
            None => return Err(ParseError::UnexpectedEof(pos)),
            _ => {}
        }
        let (name, span) = self.parse_word().unwrap();
        Ok(Ident { name, span })
    }

    pub fn parse_atom(&mut self) -> ParseResult<Atom> {
        let pos = self.pos();
        match self.peek() {
            Some(ch) if !ch.is_uppercase() => return Err(ParseError::ExpectedAtom(pos)),
            None => return Err(ParseError::UnexpectedEof(pos)),
            _ => {}
        }
        let (name, span) = self.parse_word().unwrap();
        Ok(Atom { name, span })
    }

    pub fn at_literal(&self) -> bool {
        match self.peek() {
            Some(c) if c.is_ascii_digit() || c == '"' || c == '\'' => true,
            Some(c) if c.is_ascii_alphabetic() => matches!(self.peek2(), Some('"') | Some('\'')),
            _ => false,
        }
    }

    pub fn parse_literal(&mut self) -> ParseResult<Literal> {
        let start_pos = self.pos;

        match self.peek() {
            Some(ch @ '0'..='9') => {
                let (radix, skip) = match ch {
                    '0' => match self.peek2() {
                        Some('b' | 'B') => (2, 2),
                        Some('o' | 'O') => (8, 2),
                        Some('x' | 'X') => (16, 2),
                        _ => (10, 0),
                    },
                    _ => (10, 0),
                };

                let mut chars = self.char_indices();
                for _ in 0..skip {
                    chars.next().unwrap();
                }
                debug_assert_eq!(skip, chars.offset());

                let len;
                let mut has_point = false;
                let mut has_exponent = false;
                let mut digits = 0;
                let mut exponent_digits = 0;
                let mut follows_exponent = false;
                // Set to the local offset of '.' when we see one; cleared when a
                // fractional digit follows. If still set at termination, we stop
                // before the dot so "1.foo" parses as "1" (dot = field accessor).
                let mut pre_dot_offset: Option<usize> = None;

                loop {
                    match chars.next() {
                        None => {
                            len = pre_dot_offset.unwrap_or(chars.offset());
                            break;
                        }
                        Some((_, ch)) if ch.is_digit(radix) => {
                            digits += 1;
                            if has_exponent {
                                exponent_digits += 1;
                            } else if has_point {
                                pre_dot_offset = None;
                            }
                            follows_exponent = false;
                        }
                        Some((_, '_')) => {
                            follows_exponent = false;
                        }
                        Some((pos, '.')) if !has_exponent && !has_point => {
                            pre_dot_offset = Some(pos);
                            has_point = true;
                            follows_exponent = false;
                        }
                        Some((_, 'e' | 'E')) if !has_exponent => {
                            has_exponent = true;
                            follows_exponent = true;
                        }
                        Some((_, '-' | '+')) if follows_exponent => {
                            follows_exponent = false;
                        }
                        Some((pos, ch)) if is_xid_start(ch) => {
                            if let Some(pre_dot) = pre_dot_offset {
                                len = pre_dot;
                                break;
                            }
                            loop {
                                match chars.next() {
                                    None => { len = chars.offset(); break; }
                                    Some((_, ch)) if is_xid_continue(ch) => {}
                                    Some((pos, _)) => { len = pos; break; }
                                }
                            }
                            break;
                        }
                        Some((pos, _)) => {
                            len = pre_dot_offset.unwrap_or(pos);
                            break;
                        }
                    }
                }

                drop(chars);

                if digits == 0 {
                    return Err(ParseError::Literal(LexError {
                        span: Span::new(start_pos, start_pos + len),
                        kind: LexErrorKind::NoDigitsInNumber,
                    }));
                }
                if has_exponent && exponent_digits == 0 {
                    return Err(ParseError::Literal(LexError {
                        span: Span::new(start_pos, start_pos + len),
                        kind: LexErrorKind::NoDigitsInExponent,
                    }));
                }

                let repr = Rc::from(self.consume(len));
                let end_pos = self.pos;
                Ok(Literal::new_number(repr, Span::new(start_pos, end_pos)))
            }

            Some(ch) if ch == '"' || ch.is_ascii_alphabetic() && self.peek2() == Some('"') => {
                let mut chars = self.char_indices();
                if ch != '"' {
                    chars.next().unwrap();
                }
                chars.next().unwrap(); // opening quote

                while let Some((pos, ch)) = chars.next() {
                    match ch {
                        '"' => {
                            drop(chars);
                            let repr = Rc::from(self.consume(pos + 1));
                            let end_pos = self.pos;
                            return Ok(Literal::new_string(repr, Span::new(start_pos, end_pos)));
                        }
                        '\\' => {
                            parse_escape(pos, chars.by_ref().map(|(p, c)| (p - pos - 1, c)))
                                .map_err(ParseError::Literal)?;
                        }
                        _ => {}
                    }
                }

                Err(ParseError::Literal(LexError {
                    span: Span::new(self.pos, self.pos),
                    kind: LexErrorKind::EndOfInput,
                }))
            }

            Some(ch) if ch == '\'' || ch.is_ascii_alphabetic() && self.peek2() == Some('\'') => {
                let mut chars = self.char_indices();
                if ch != '\'' {
                    chars.next().unwrap();
                }
                chars.next().unwrap(); // opening quote

                match chars.next() {
                    Some((pos, '\'')) => {
                        return Err(ParseError::Literal(LexError {
                            span: Span::new(start_pos, start_pos + pos + 1),
                            kind: LexErrorKind::UnexpectedCharacter('\''),
                        }));
                    }
                    Some((pos, '\\')) => {
                        parse_escape(pos, chars.by_ref().map(|(p, c)| (p - pos - 1, c)))
                            .map_err(ParseError::Literal)?;
                    }
                    Some(_) => {}
                    None => {
                        return Err(ParseError::Literal(LexError {
                            span: Span::new(start_pos, start_pos),
                            kind: LexErrorKind::EndOfInput,
                        }));
                    }
                }

                match chars.next() {
                    Some((pos, '\'')) => {
                        drop(chars);
                        let repr = Rc::from(self.consume(pos + 1));
                        let end_pos = self.pos;
                        Ok(Literal::new_char(repr, Span::new(start_pos, end_pos)))
                    }
                    Some((pos, c)) => Err(ParseError::Literal(LexError {
                        span: Span::new(start_pos, start_pos + pos + c.len_utf8()),
                        kind: LexErrorKind::UnexpectedCharacter(c),
                    })),
                    None => Err(ParseError::Literal(LexError {
                        span: Span::new(start_pos, start_pos),
                        kind: LexErrorKind::EndOfInput,
                    })),
                }
            }

            Some(c) => Err(ParseError::Literal(LexError {
                span: Span::new(start_pos, start_pos + c.len_utf8()),
                kind: LexErrorKind::UnexpectedCharacter(c),
            })),
            None => Err(ParseError::Literal(LexError {
                span: Span::new(start_pos, start_pos),
                kind: LexErrorKind::EndOfInput,
            })),
        }
    }

    pub fn try_unary_op(&mut self) -> Option<Spanned<UnaryOp>> {
        let start = self.pos();
        let op = match self.peek()? {
            '!' => UnaryOp::Not,
            '-' => UnaryOp::Neg,
            '~' => UnaryOp::BitNot,
            _ => return None,
        };
        self.advance();
        Some(Spanned { node: op, span: Span { start, end: self.pos() } })
    }

    // Multi-char ops advance the first char in the match arm, then fall through
    // to the final advance() for the second char. Single-char ops skip the arm advance.
    pub fn try_binary_op(&mut self) -> Option<Spanned<BinaryOp>> {
        let start = self.pos();
        let op = match (self.peek()?, self.peek2()) {
            ('&', _) => BinaryOp::BitAnd,
            ('|', _) => BinaryOp::BitOr,
            ('^', _) => BinaryOp::BitXor,
            ('<', Some('<')) => { self.advance(); BinaryOp::Shl }
            ('>', Some('>')) => { self.advance(); BinaryOp::Shr }
            ('*', Some('*')) => { self.advance(); BinaryOp::Pow }
            ('*', _) => BinaryOp::Mul,
            ('/', _) => BinaryOp::Div,
            ('+', _) => BinaryOp::Add,
            ('-', _) => BinaryOp::Sub,
            ('=', Some('=')) => { self.advance(); BinaryOp::Eq }
            ('!', Some('=')) => { self.advance(); BinaryOp::Ne }
            ('<', Some('=')) => { self.advance(); BinaryOp::Le }
            ('>', Some('=')) => { self.advance(); BinaryOp::Ge }
            ('<', _) => BinaryOp::Lt,
            ('>', _) => BinaryOp::Gt,
            _ => return None,
        };
        self.advance();
        Some(Spanned { node: op, span: Span { start, end: self.pos() } })
    }

    pub fn parse_pattern(&mut self) -> ParseResult<Pattern> {
        match self.peek() {
            Some('[') => self.parse_list_pattern(),
            Some('{') => self.parse_record_pattern(),
            Some(ch) if ch.is_uppercase() && is_xid_start(ch) => {
                let atom = self.parse_atom()?;
                Ok(Pattern { span: atom.span, kind: PatternKind::Atom(atom) })
            }
            Some(ch) if is_xid_start(ch) || ch == '_' => {
                let ident = self.parse_ident()?;
                Ok(Pattern { span: ident.span, kind: PatternKind::Ident(ident) })
            }
            Some(ch) => Err(ParseError::UnexpectedChar(self.pos, ch)),
            None => Err(ParseError::UnexpectedEof(self.pos)),
        }
    }

    fn parse_list_pattern(&mut self) -> ParseResult<Pattern> {
        let start = self.pos;
        self.expect_char('[')?;
        self.skip_whitespace_inline();
        let mut elements = Vec::new();
        if self.peek() != Some(']') {
            elements.push(self.parse_pattern()?);
            self.skip_whitespace_inline();
            while self.peek() == Some(',') {
                self.advance();
                self.skip_whitespace_inline();
                elements.push(self.parse_pattern()?);
                self.skip_whitespace_inline();
            }
        }
        self.expect_char(']')?;
        Ok(Pattern { span: Span { start, end: self.pos }, kind: PatternKind::List(elements) })
    }

    fn parse_record_pattern(&mut self) -> ParseResult<Pattern> {
        let start = self.pos;
        self.expect_char('{')?;
        self.skip_whitespace_inline();
        let mut fields = Vec::new();
        if self.peek() != Some('}') {
            fields.push(self.parse_record_pattern_field()?);
            self.skip_whitespace_inline();
            while self.peek() == Some(',') {
                self.advance();
                self.skip_whitespace_inline();
                fields.push(self.parse_record_pattern_field()?);
                self.skip_whitespace_inline();
            }
        }
        self.expect_char('}')?;
        Ok(Pattern { span: Span { start, end: self.pos }, kind: PatternKind::Record(fields) })
    }

    fn parse_record_pattern_field(&mut self) -> ParseResult<RecordPatternField> {
        let start = self.pos;
        let key = self.parse_ident()?;
        self.skip_whitespace_inline();
        let pattern = if self.peek() == Some(':') {
            self.advance();
            self.skip_whitespace_inline();
            self.parse_pattern()?
        } else {
            // shorthand: { foo } == { foo: foo }
            Pattern { span: key.span, kind: PatternKind::Ident(key.clone()) }
        };
        Ok(RecordPatternField { span: Span { start, end: self.pos }, key, pattern })
    }

    // ---- Expression parsing -------------------------------------------------

    pub fn parse_expr(&mut self) -> ParseResult<Expr> {
        self.parse_binary(0)
    }

    fn parse_binary(&mut self, min_prec: u8) -> ParseResult<Expr> {
        let mut lhs = self.parse_application()?;
        loop {
            self.skip_whitespace_inline();
            let saved = self.pos;
            match self.try_binary_op() {
                Some(op) if op.node.precedence() > min_prec => {
                    self.skip_whitespace_inline();
                    let rhs_min = if op.node.is_right_assoc() {
                        op.node.precedence() - 1
                    } else {
                        op.node.precedence()
                    };
                    let rhs = self.parse_binary(rhs_min)?;
                    let span = Span { start: lhs.span.start, end: rhs.span.end };
                    lhs = Expr {
                        kind: ExprKind::Binary(Box::new(lhs), op, Box::new(rhs)),
                        span,
                    };
                }
                Some(_) => {
                    self.pos = saved;
                    break;
                }
                None => break,
            }
        }
        Ok(lhs)
    }

    fn parse_application(&mut self) -> ParseResult<Expr> {
        let func = self.parse_unary()?;
        let start = func.span.start;
        let mut args = Vec::new();
        loop {
            self.skip_whitespace_inline();
            if !self.can_start_argument() {
                break;
            }
            args.push(self.parse_unary()?);
        }
        if args.is_empty() {
            Ok(func)
        } else {
            let end = args.last().unwrap().span.end;
            Ok(Expr { kind: ExprKind::Apply(Box::new(func), args), span: Span { start, end } })
        }
    }

    fn parse_unary(&mut self) -> ParseResult<Expr> {
        if let Some(op) = self.try_unary_op() {
            let operand = self.parse_unary()?;
            let span = Span { start: op.span.start, end: operand.span.end };
            Ok(Expr { kind: ExprKind::Unary(op, Box::new(operand)), span })
        } else {
            self.parse_accessor()
        }
    }

    fn parse_accessor(&mut self) -> ParseResult<Expr> {
        let mut expr = self.parse_primary()?;
        loop {
            match self.peek() {
                Some('.') => {
                    self.advance();
                    let field = self.parse_ident()?;
                    let span = Span { start: expr.span.start, end: field.span.end };
                    expr = Expr { kind: ExprKind::Field(Box::new(expr), field), span };
                }
                Some('[') => {
                    self.advance();
                    self.skip_whitespace_inline();
                    let index = self.parse_expr()?;
                    self.skip_whitespace_inline();
                    self.expect_char(']')?;
                    let span = Span { start: expr.span.start, end: self.pos };
                    expr = Expr { kind: ExprKind::Index(Box::new(expr), Box::new(index)), span };
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> ParseResult<Expr> {
        let pos = self.pos;
        match self.peek() {
            Some('(') => {
                self.advance();
                self.skip_whitespace_inline();
                let mut expr = self.parse_expr()?;
                self.skip_whitespace_inline();
                self.expect_char(')')?;
                expr.span = Span { start: pos, end: self.pos };
                Ok(expr)
            }
            Some('[') => self.parse_list_expr(),
            Some('{') => self.parse_record_expr(),
            _ if self.at_literal() => {
                let lit = self.parse_literal()?;
                Ok(Expr { span: lit.span(), kind: ExprKind::Literal(lit) })
            }
            Some(ch) if ch.is_uppercase() && is_xid_start(ch) => {
                let atom = self.parse_atom()?;
                Ok(Expr { span: atom.span, kind: ExprKind::Atom(atom) })
            }
            Some(ch) if is_xid_start(ch) || ch == '_' => {
                let ident = self.parse_ident()?;
                Ok(Expr { span: ident.span, kind: ExprKind::Ident(ident) })
            }
            Some(ch) => Err(ParseError::UnexpectedChar(pos, ch)),
            None => Err(ParseError::UnexpectedEof(pos)),
        }
    }

    fn parse_list_expr(&mut self) -> ParseResult<Expr> {
        let start = self.pos;
        self.expect_char('[')?;
        self.skip_whitespace_inline();
        let mut elements = Vec::new();
        if self.peek() != Some(']') {
            elements.push(self.parse_expr()?);
            self.skip_whitespace_inline();
            while self.peek() == Some(',') {
                self.advance();
                self.skip_whitespace_inline();
                elements.push(self.parse_expr()?);
                self.skip_whitespace_inline();
            }
        }
        self.expect_char(']')?;
        Ok(Expr { kind: ExprKind::List(elements), span: Span { start, end: self.pos } })
    }

    fn parse_record_expr(&mut self) -> ParseResult<Expr> {
        let start = self.pos;
        self.expect_char('{')?;
        self.skip_whitespace_inline();
        let mut fields = Vec::new();
        if self.peek() != Some('}') {
            fields.push(self.parse_record_expr_field()?);
            self.skip_whitespace_inline();
            while self.peek() == Some(',') {
                self.advance();
                self.skip_whitespace_inline();
                fields.push(self.parse_record_expr_field()?);
                self.skip_whitespace_inline();
            }
        }
        self.expect_char('}')?;
        Ok(Expr { kind: ExprKind::Record(fields), span: Span { start, end: self.pos } })
    }

    fn parse_record_expr_field(&mut self) -> ParseResult<ExprRecordField> {
        let start = self.pos;
        let key = self.parse_ident()?;
        self.skip_whitespace_inline();
        self.expect_char(':')?;
        self.skip_whitespace_inline();
        let value = self.parse_expr()?;
        Ok(ExprRecordField { span: Span { start, end: self.pos }, key, value })
    }

    // `-` is excluded: `f - a` is binary subtraction, not application.
    // `!` and `~` are included: unambiguously unary.
    fn can_start_argument(&self) -> bool {
        match self.peek() {
            Some(ch) if ch.is_ascii_digit() || ch == '"' || ch == '\'' => true,
            Some(ch) if is_xid_start(ch) || ch == '_' => true,
            Some('(') | Some('[') | Some('{') => true,
            Some('!') | Some('~') => true,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::BinaryOp;

    #[test]
    fn idents() {
        let i = Stream::new("foo").parse_ident().unwrap();
        assert_eq!(i.name, "foo");
        assert_eq!(i.span, Span { start: 0, end: 3 });

        assert_eq!(Stream::new("_bar").parse_ident().unwrap().name, "_bar");
        assert_eq!(Stream::new("foo_bar").parse_ident().unwrap().name, "foo_bar");
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
        assert_eq!(Stream::new("'\\n'").parse_literal().unwrap().as_char().unwrap().unescape(), '\n');
        assert_eq!(Stream::new("'\\t'").parse_literal().unwrap().as_char().unwrap().unescape(), '\t');
        assert_eq!(Stream::new("'\\\\'").parse_literal().unwrap().as_char().unwrap().unescape(), '\\');
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
        assert_eq!(Stream::new("~").try_unary_op().unwrap().node, UnaryOp::BitNot);
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
        let PatternKind::List(els) = &p.kind else { panic!() };
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
        let PatternKind::Record(fields) = &p.kind else { panic!() };
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].key.name, "foo");
        assert!(matches!(&fields[0].pattern.kind, PatternKind::Ident(i) if i.name == "foo"));
        assert_eq!(fields[1].key.name, "bar");
    }

    #[test]
    fn pattern_record_explicit() {
        let p = Stream::new("{ x: foo, y: bar }").parse_pattern().unwrap();
        let PatternKind::Record(fields) = &p.kind else { panic!() };
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].key.name, "x");
        assert!(matches!(&fields[0].pattern.kind, PatternKind::Ident(i) if i.name == "foo"));
    }

    #[test]
    fn pattern_record_nested() {
        let p = Stream::new("{ x: [a, b] }").parse_pattern().unwrap();
        let PatternKind::Record(fields) = &p.kind else { panic!() };
        assert!(matches!(&fields[0].pattern.kind, PatternKind::List(_)));
    }

    // ---- Expression tests ---------------------------------------------------

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
        let ExprKind::Unary(op, inner) = &e.kind else { panic!() };
        assert_eq!(op.node, UnaryOp::Not);
        assert!(matches!(&inner.kind, ExprKind::Ident(i) if i.name == "a"));
        assert_eq!(e.span, Span::new(0, 2));
    }

    #[test]
    fn expr_unary_chain() {
        // !!a = !(!a)
        let e = Stream::new("!!a").parse_expr().unwrap();
        let ExprKind::Unary(op, inner) = &e.kind else { panic!() };
        assert_eq!(op.node, UnaryOp::Not);
        assert!(matches!(&inner.kind, ExprKind::Unary(_, _)));
    }

    #[test]
    fn expr_binary_simple() {
        let e = Stream::new("1 + 2").parse_expr().unwrap();
        let ExprKind::Binary(lhs, op, rhs) = &e.kind else { panic!() };
        assert_eq!(op.node, BinaryOp::Add);
        assert!(matches!(&lhs.kind, ExprKind::Literal(_)));
        assert!(matches!(&rhs.kind, ExprKind::Literal(_)));
        assert_eq!(e.span, Span::new(0, 5));
    }

    #[test]
    fn expr_precedence() {
        // 1 + 2 * 3 = 1 + (2 * 3)
        let e = Stream::new("1 + 2 * 3").parse_expr().unwrap();
        let ExprKind::Binary(lhs, op, rhs) = &e.kind else { panic!() };
        assert_eq!(op.node, BinaryOp::Add);
        assert!(matches!(&lhs.kind, ExprKind::Literal(_)));
        let ExprKind::Binary(_, inner_op, _) = &rhs.kind else { panic!() };
        assert_eq!(inner_op.node, BinaryOp::Mul);
    }

    #[test]
    fn expr_left_assoc() {
        // a - b - c = (a - b) - c
        let e = Stream::new("a - b - c").parse_expr().unwrap();
        let ExprKind::Binary(lhs, op, rhs) = &e.kind else { panic!() };
        assert_eq!(op.node, BinaryOp::Sub);
        assert!(matches!(&lhs.kind, ExprKind::Binary(_, _, _)));
        assert!(matches!(&rhs.kind, ExprKind::Ident(i) if i.name == "c"));
    }

    #[test]
    fn expr_right_assoc() {
        // 2 ** 3 ** 4 = 2 ** (3 ** 4)
        let e = Stream::new("2 ** 3 ** 4").parse_expr().unwrap();
        let ExprKind::Binary(lhs, op, rhs) = &e.kind else { panic!() };
        assert_eq!(op.node, BinaryOp::Pow);
        assert!(matches!(&lhs.kind, ExprKind::Literal(_)));
        assert!(matches!(&rhs.kind, ExprKind::Binary(_, _, _)));
    }

    #[test]
    fn expr_apply() {
        let e = Stream::new("f a b").parse_expr().unwrap();
        let ExprKind::Apply(func, args) = &e.kind else { panic!() };
        assert!(matches!(&func.kind, ExprKind::Ident(i) if i.name == "f"));
        assert_eq!(args.len(), 2);
        assert!(matches!(&args[0].kind, ExprKind::Ident(i) if i.name == "a"));
        assert!(matches!(&args[1].kind, ExprKind::Ident(i) if i.name == "b"));
    }

    #[test]
    fn expr_apply_unary_arg() {
        // f !a — ! is unambiguously unary, so it's an argument
        let e = Stream::new("f !a").parse_expr().unwrap();
        let ExprKind::Apply(_, args) = &e.kind else { panic!() };
        assert_eq!(args.len(), 1);
        assert!(matches!(&args[0].kind, ExprKind::Unary(op, _) if op.node == UnaryOp::Not));
    }

    #[test]
    fn expr_apply_then_binary() {
        // f a + b = (f a) + b
        let e = Stream::new("f a + b").parse_expr().unwrap();
        let ExprKind::Binary(lhs, op, _) = &e.kind else { panic!() };
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
        let ExprKind::Field(obj, field) = &e.kind else { panic!() };
        assert!(matches!(&obj.kind, ExprKind::Ident(i) if i.name == "foo"));
        assert_eq!(field.name, "bar");
        assert_eq!(e.span, Span::new(0, 7));
    }

    #[test]
    fn expr_index() {
        let e = Stream::new("arr[0]").parse_expr().unwrap();
        let ExprKind::Index(obj, _) = &e.kind else { panic!() };
        assert!(matches!(&obj.kind, ExprKind::Ident(i) if i.name == "arr"));
        assert_eq!(e.span, Span::new(0, 6));
    }

    #[test]
    fn expr_chained_accessor() {
        // foo.bar[0].baz = Field(Index(Field(foo, bar), 0), baz)
        let e = Stream::new("foo.bar[0].baz").parse_expr().unwrap();
        let ExprKind::Field(indexed, baz) = &e.kind else { panic!() };
        assert_eq!(baz.name, "baz");
        let ExprKind::Index(field_expr, _) = &indexed.kind else { panic!() };
        let ExprKind::Field(root, bar) = &field_expr.kind else { panic!() };
        assert!(matches!(&root.kind, ExprKind::Ident(i) if i.name == "foo"));
        assert_eq!(bar.name, "bar");
    }

    #[test]
    fn expr_apply_with_field_arg() {
        // f a.b = f (a.b)
        let e = Stream::new("f a.b").parse_expr().unwrap();
        let ExprKind::Apply(_, args) = &e.kind else { panic!() };
        assert_eq!(args.len(), 1);
        assert!(matches!(&args[0].kind, ExprKind::Field(_, _)));
    }

    #[test]
    fn expr_list() {
        let e = Stream::new("[1, 2, 3]").parse_expr().unwrap();
        let ExprKind::List(els) = &e.kind else { panic!() };
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
        let ExprKind::Record(fields) = &e.kind else { panic!() };
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].key.name, "x");
        assert!(matches!(&fields[0].value.kind, ExprKind::Literal(_)));
    }

    #[test]
    fn expr_paren_grouping() {
        // (1 + 2) * 3 — parens override precedence
        let e = Stream::new("(1 + 2) * 3").parse_expr().unwrap();
        let ExprKind::Binary(lhs, op, _) = &e.kind else { panic!() };
        assert_eq!(op.node, BinaryOp::Mul);
        // lhs is the parenthesized addition; span includes parens
        assert_eq!(lhs.span, Span::new(0, 7));
        assert!(matches!(&lhs.kind, ExprKind::Binary(_, op, _) if op.node == BinaryOp::Add));
    }

    #[test]
    fn expr_complex() {
        // a + b * c ** d / e > f * g - h ** i / j
        // = (a + ((b * (c ** d)) / e)) > ((f * g) - ((h ** i) / j))
        let e = Stream::new("a + b * c ** d / e > f * g - h ** i / j").parse_expr().unwrap();
        let ExprKind::Binary(lhs, op, rhs) = &e.kind else { panic!() };
        assert_eq!(op.node, BinaryOp::Gt);
        // lhs = a + ((b * (c**d)) / e)
        let ExprKind::Binary(_, add_op, _) = &lhs.kind else { panic!() };
        assert_eq!(add_op.node, BinaryOp::Add);
        // rhs = (f*g) - ((h**i)/j)
        let ExprKind::Binary(_, sub_op, _) = &rhs.kind else { panic!() };
        assert_eq!(sub_op.node, BinaryOp::Sub);
    }
}
