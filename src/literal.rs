use std::ops::Range;
use std::rc::Rc;

use crate::ast::Span;

// ---- Errors -----------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LexError {
    pub span: Span,
    pub kind: LexErrorKind,
}

#[derive(Debug, Clone)]
pub enum LexErrorKind {
    NoDigitsInNumber,
    NoDigitsInExponent,
    EndOfInput,
    UnexpectedCharacter(char),
    InvalidEscapeSequence(char),
}

// ---- LiteralNumber ----------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LiteralNumber {
    repr: Rc<str>,
    span: Span,
}

impl LiteralNumber {
    pub fn new(repr: Rc<str>, span: Span) -> Self {
        LiteralNumber { repr, span }
    }

    /// Returns the string representation of the literal, e.g. "0xFF", "1.23e-4", "0b1010_u32".
    pub fn repr(&self) -> &str {
        self.repr.as_ref()
    }

    /// Returns the radix of the number literal, e.g. 2 for "0b1010", 8 for "0o755", 16 for "0xFF", and 10 for "123".
    pub fn radix(&self) -> u32 {
        if self.repr.starts_with("0b") || self.repr.starts_with("0B") {
            2
        } else if self.repr.starts_with("0o") || self.repr.starts_with("0O") {
            8
        } else if self.repr.starts_with("0x") || self.repr.starts_with("0X") {
            16
        } else {
            10
        }
    }

    /// Returns if the number literal has a dot, e.g. "1.23" has a dot but "123" does not.
    pub fn has_dot(&self) -> bool {
        self.dot_pos().is_some()
    }

    /// Returns if the number literal has an exponent part, e.g. "1e-4" has an exponent but "1.23" does not.
    pub fn has_exponent(&self) -> bool {
        self.exponent_start().is_some()
    }

    /// Returns if the number literal has a suffix, e.g. "0xFF_u32" has a suffix but "1.23e-4" does not.
    pub fn has_suffix(&self) -> bool {
        self.suffix_start().is_some()
    }

    /// Returns the range of the exponent part in the number literal, e.g. "-4" for "1.23e-4" if it exists.
    pub fn exponent_range(&self) -> Option<Range<usize>> {
        let start = self.exponent_start()?;
        let end = self.suffix_start().unwrap_or(self.repr.len());
        Some(start..end)
    }

    /// Returns the range of the mantissa part in the number literal.
    pub fn mantissa_range(&self) -> Range<usize> {
        self.matissa_start()..self.matissa_end()
    }

    /// Returns the range of the integer part in the number literal.
    pub fn integer_range(&self) -> Range<usize> {
        let start = self.matissa_start();
        let end = self.dot_pos().unwrap_or_else(|| self.matissa_end());
        start..end
    }

    /// Returns the range of the fractional part, e.g. "23" for "1.23e-4".
    pub fn fractional_range(&self) -> Option<Range<usize>> {
        let start = self.dot_pos()? + 1;
        Some(start..self.matissa_end())
    }

    /// Returns the range before any suffix.
    pub fn value_range(&self) -> Range<usize> {
        0..self.suffix_start().unwrap_or(self.repr.len())
    }

    /// Returns the range of the suffix, e.g. "u32" for "0xFF_u32".
    pub fn suffix_range(&self) -> Option<Range<usize>> {
        Some(self.suffix_start()?..self.repr.len())
    }

    pub fn exponent(&self) -> Option<&str> {
        Some(&self.repr[self.exponent_range()?])
    }

    pub fn mantissa(&self) -> &str {
        &self.repr[self.mantissa_range()]
    }

    pub fn integer(&self) -> &str {
        &self.repr[self.integer_range()]
    }

    pub fn fractional(&self) -> Option<&str> {
        Some(&self.repr[self.fractional_range()?])
    }

    pub fn value(&self) -> &str {
        &self.repr[self.value_range()]
    }

    pub fn suffix(&self) -> Option<&str> {
        Some(&self.repr[self.suffix_range()?])
    }

    pub fn span(&self) -> Span {
        self.span
    }

    fn dot_pos(&self) -> Option<usize> {
        self.value().find('.')
    }

    fn exponent_start(&self) -> Option<usize> {
        if self.radix() == 16 {
            return None;
        }
        self.value().find(['e', 'E']).map(|pos| pos + 1)
    }

    fn matissa_start(&self) -> usize {
        if self.repr.starts_with("0b")
            || self.repr.starts_with("0B")
            || self.repr.starts_with("0o")
            || self.repr.starts_with("0O")
            || self.repr.starts_with("0x")
            || self.repr.starts_with("0X")
        {
            2
        } else {
            0
        }
    }

    fn matissa_end(&self) -> usize {
        match self.exponent_start() {
            Some(pos) => pos - 1,
            None => self.suffix_start().unwrap_or(self.repr.len()),
        }
    }

    // Walks past mantissa digits then the optional exponent to find where a
    // suffix (XID_Start identifier) begins. The original find()-based approach
    // incorrectly treats 'e'/'E' in float exponents as suffix starts.
    fn suffix_start(&self) -> Option<usize> {
        let radix = self.radix();
        let mut pos = self.matissa_start();

        while pos < self.repr.len() {
            let ch = self.repr[pos..].chars().next().unwrap();
            if ch.is_digit(radix) || ch == '_' || ch == '.' {
                pos += ch.len_utf8();
            } else {
                break;
            }
        }

        if radix != 16 && pos < self.repr.len() {
            let ch = self.repr[pos..].chars().next().unwrap();
            if ch == 'e' || ch == 'E' {
                pos += 1;
                if pos < self.repr.len() {
                    let sign = self.repr[pos..].chars().next().unwrap();
                    if sign == '+' || sign == '-' {
                        pos += 1;
                    }
                }
                while pos < self.repr.len() {
                    let ch = self.repr[pos..].chars().next().unwrap();
                    if ch.is_ascii_digit() || ch == '_' {
                        pos += ch.len_utf8();
                    } else {
                        break;
                    }
                }
            }
        }

        if pos < self.repr.len()
            && unicode_ident::is_xid_start(self.repr[pos..].chars().next().unwrap())
        {
            Some(pos)
        } else {
            None
        }
    }
}

// ---- LiteralChar ------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LiteralChar {
    repr: Rc<str>,
    span: Span,
}

impl LiteralChar {
    pub fn new(repr: Rc<str>, span: Span) -> Self {
        LiteralChar { repr, span }
    }

    pub fn repr(&self) -> &str {
        self.repr.as_ref()
    }

    pub fn prefix(&self) -> Option<char> {
        match self.repr.chars().next() {
            Some(c) if c.is_ascii_alphabetic() => Some(c),
            _ => None,
        }
    }

    pub fn unescape(&self) -> char {
        let mut chars = self.repr.char_indices();
        match chars.next() {
            Some((_, '\'')) => {}
            Some((_, c)) if c.is_ascii_alphabetic() => match chars.next() {
                Some((_, '\'')) => {}
                _ => self.panic_invalid(),
            },
            _ => self.panic_invalid(),
        }

        let ch = match chars.next() {
            Some((_, '\\')) => match chars.next() {
                Some((_, 'n')) => '\n',
                Some((_, 't')) => '\t',
                Some((_, 'r')) => '\r',
                Some((_, '\\')) => '\\',
                Some((_, '\'')) => '\'',
                Some((_, '"')) => '"',
                Some((_, '0')) => '\0',
                _ => self.panic_invalid(),
            },
            Some((_, ch)) => ch,
            None => self.panic_invalid(),
        };

        match chars.next() {
            Some((_, '\'')) => {}
            _ => self.panic_invalid(),
        }

        ch
    }

    pub fn span(&self) -> Span {
        self.span
    }

    #[track_caller]
    fn panic_invalid(&self) -> ! {
        panic!("Invalid char literal: `{}`", self.repr);
    }
}

// ---- LiteralString ----------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LiteralString {
    repr: Rc<str>,
    span: Span,
}

impl LiteralString {
    pub fn new(repr: Rc<str>, span: Span) -> Self {
        LiteralString { repr, span }
    }

    pub fn repr(&self) -> &str {
        self.repr.as_ref()
    }

    pub fn prefix(&self) -> Option<char> {
        match self.repr.chars().next() {
            Some(c) if c.is_ascii_alphabetic() => Some(c),
            _ => None,
        }
    }

    pub fn unescape(&self) -> String {
        let mut chars = self.repr.char_indices();
        match chars.next() {
            Some((_, '"')) => {}
            Some((_, c)) if c.is_ascii_alphabetic() => match chars.next() {
                Some((_, '"')) => {}
                _ => self.panic_invalid(),
            },
            _ => self.panic_invalid(),
        }

        let mut unescaped = String::new();
        loop {
            let ch = match chars.next() {
                Some((_, '"')) => break,
                Some((_, '\\')) => match chars.next() {
                    Some((_, 'n')) => '\n',
                    Some((_, 't')) => '\t',
                    Some((_, 'r')) => '\r',
                    Some((_, '\\')) => '\\',
                    Some((_, '\'')) => '\'',
                    Some((_, '"')) => '"',
                    Some((_, '0')) => '\0',
                    _ => self.panic_invalid(),
                },
                Some((_, ch)) => ch,
                None => self.panic_invalid(),
            };
            unescaped.push(ch);
        }

        unescaped
    }

    pub fn span(&self) -> Span {
        self.span
    }

    #[track_caller]
    fn panic_invalid(&self) -> ! {
        panic!("Invalid string literal: `{}`", self.repr);
    }
}

// ---- Literal ----------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Literal {
    Number(LiteralNumber),
    Char(LiteralChar),
    String(LiteralString),
}

impl Literal {
    pub fn new_number(repr: Rc<str>, span: Span) -> Self {
        Literal::Number(LiteralNumber { repr, span })
    }

    pub fn new_char(repr: Rc<str>, span: Span) -> Self {
        Literal::Char(LiteralChar { repr, span })
    }

    pub fn new_string(repr: Rc<str>, span: Span) -> Self {
        Literal::String(LiteralString { repr, span })
    }

    pub fn span(&self) -> Span {
        match self {
            Literal::Number(n) => n.span(),
            Literal::Char(c) => c.span(),
            Literal::String(s) => s.span(),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Literal::Number(n) => n.repr(),
            Literal::Char(c) => c.repr(),
            Literal::String(s) => s.repr(),
        }
    }

    pub fn is_number(&self) -> bool {
        matches!(self, Literal::Number(_))
    }

    pub fn as_number(&self) -> Option<&LiteralNumber> {
        match self {
            Literal::Number(n) => Some(n),
            _ => None,
        }
    }

    pub fn is_char(&self) -> bool {
        matches!(self, Literal::Char(_))
    }

    pub fn as_char(&self) -> Option<&LiteralChar> {
        match self {
            Literal::Char(c) => Some(c),
            _ => None,
        }
    }

    pub fn is_string(&self) -> bool {
        matches!(self, Literal::String(_))
    }

    pub fn as_string(&self) -> Option<&LiteralString> {
        match self {
            Literal::String(s) => Some(s),
            _ => None,
        }
    }
}
