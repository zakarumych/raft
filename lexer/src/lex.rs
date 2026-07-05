use core::{char, fmt, ops::Range};

use alloc::{borrow::ToOwned, rc::Rc, string::String, vec::Vec};

use crate::stream::{Span, Stream};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LexErrorKind {
    /// The input ended unexpectedly while parsing a token.
    EndOfInput,

    /// An unexpected character was encountered while parsing a token.
    UnexpectedCharacter(char),

    /// An escaped character was invalid, e.g. non-UTF-8 code point in a Unicode escape sequence.
    InvalidEscapedCharacter(u32),

    /// A delimiter was opened but not closed before outer delimiter was closed.
    UnclosedDelimiter,

    /// A number literal has no digits, e.g. "0x" or "0b_".
    NoDigitsInNumber,

    /// A number literal has exponent part but no digits in it, e.g. "1e" or "1e_".
    NoDigitsInExponent,
}

impl fmt::Display for LexErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LexErrorKind::EndOfInput => write!(f, "Unexpected end of input"),
            LexErrorKind::UnexpectedCharacter(c) => write!(f, "Unexpected character: '{}'", c),
            LexErrorKind::InvalidEscapedCharacter(code) => {
                write!(f, "Invalid escaped character: code point U+{:X}", code)
            }
            LexErrorKind::UnclosedDelimiter => write!(f, "Unclosed delimiter"),
            LexErrorKind::NoDigitsInNumber => write!(f, "Number literal has no digits"),
            LexErrorKind::NoDigitsInExponent => write!(f, "Exponent has no digits"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LexError {
    span: Span,
    kind: LexErrorKind,
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} @ {}..{}", self.kind, self.span.start, self.span.end)
    }
}

impl LexError {
    pub fn span(&self) -> Span {
        self.span
    }

    pub fn kind(&self) -> LexErrorKind {
        self.kind
    }

    pub fn print<'a>(&'a self, source: &'a str) -> PrintLexError<'a> {
        PrintLexError {
            error: self,
            source,
        }
    }
}

pub struct PrintLexError<'a> {
    error: &'a LexError,
    source: &'a str,
}

fn digits(mut n: usize) -> usize {
    if n == 0 {
        return 1;
    }

    let mut count = 0;
    while n > 0 {
        count += 1;
        n /= 10;
    }

    count
}

impl fmt::Display for PrintLexError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.error.kind)?;

        let start_line = self.error.span.start().get_line(self.source);
        let start_column = self.error.span.start().get_column(self.source);
        let end_line = self.error.span.end().get_line(self.source);
        let end_column = self.error.span.end().get_column(self.source);

        let mut source_lines = self.source.lines();

        if start_line > 0 {
            source_lines.nth(start_line - 1);
        }

        let width = digits(end_line);

        for (idx, line) in (start_line..=end_line).zip(source_lines.take(end_line - start_line + 1))
        {
            writeln!(f, "{idx:>width$}: {}", line)?;
            if idx == start_line && idx == end_line {
                let len = end_column - start_column;
                writeln!(f, "{:width$}  {:^^len$}", "", "")?;
            } else if idx == start_line {
                let len = line.len() - start_column;
                let skip = start_column;
                writeln!(f, "{:width$}  {:skip$}{:^^len$}", "", "", "")?;
            } else if idx == end_line {
                let len = end_column;
                writeln!(f, "{:width$}  {:^^len$}", "", "")?;
            } else {
                let len = line.len();
                writeln!(f, "{:width$}  {:^^len$}", "", "")?;
            }
        }

        Ok(())
    }
}

/// Punctuation spacing variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Spacing {
    /// Punctuation is joined with the next token (no space).
    Joint,
    /// Punctuation is separated from the next token by whitespace.
    Alone,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Punct {
    repr: char,
    spacing: Spacing,
    span: Span,
}

impl Punct {
    pub fn new(repr: char, spacing: Spacing, span: Span) -> Self {
        Punct {
            repr,
            spacing,
            span,
        }
    }

    pub fn span(&self) -> Span {
        self.span
    }

    pub fn repr(&self) -> char {
        self.repr
    }

    pub fn eq_str(&self, s: &str) -> bool {
        let mut buf = [0; 4];
        self.repr.encode_utf8(&mut buf) == s
    }

    pub fn spacing(&self) -> Spacing {
        self.spacing
    }

    pub fn is_punct(stream: &Stream) -> bool {
        match stream.peek_char() {
            Some(c) => Self::is_punct_ch(c),
            None => false,
        }
    }

    fn is_punct_ch(ch: char) -> bool {
        matches!(
            ch,
            '!' | '#'
                | '$'
                | '%'
                | '&'
                | '*'
                | '+'
                | ','
                | '-'
                | '.'
                | '/'
                | ':'
                | ';'
                | '<'
                | '='
                | '>'
                | '?'
                | '@'
                | '\\'
                | '^'
                | '_'
                | '`'
                | '|'
                | '~'
        )
    }

    pub fn parse(stream: &mut Stream) -> Result<Self, LexError> {
        let start_pos = stream.pos();
        let ch = stream.peek_char().ok_or(LexError {
            span: Span::new(start_pos, start_pos),
            kind: LexErrorKind::EndOfInput,
        })?;

        if !Self::is_punct_ch(ch) {
            return Err(LexError {
                span: Span::new(start_pos, start_pos + ch.len_utf8()),
                kind: LexErrorKind::UnexpectedCharacter(ch),
            });
        }

        stream.consume(ch.len_utf8());

        let spacing = match stream.peek_char() {
            Some(ch) if Self::is_punct_ch(ch) => Spacing::Joint,
            _ => Spacing::Alone,
        };

        let end_pos = stream.pos();

        Ok(Punct::new(ch, spacing, Span::new(start_pos, end_pos)))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Ident {
    repr: Rc<str>,
    span: Span,
}

impl Ident {
    pub fn new(repr: Rc<str>, span: Span) -> Self {
        Ident { repr, span }
    }

    pub fn span(&self) -> Span {
        self.span
    }

    pub fn repr(&self) -> &str {
        self.repr.as_ref()
    }

    pub fn rc_repr(&self) -> Rc<str> {
        self.repr.clone()
    }

    pub fn is_ident(stream: &Stream) -> bool {
        match stream.peek_char() {
            Some(c) => unicode_ident::is_xid_start(c),
            None => false,
        }
    }

    pub fn parse(stream: &mut Stream) -> Result<Self, LexError> {
        match stream.peek_char() {
            Some(c) => {
                if !unicode_ident::is_xid_start(c) {
                    return Err(LexError {
                        span: Span::new(stream.pos(), stream.pos() + c.len_utf8()),
                        kind: LexErrorKind::UnexpectedCharacter(c),
                    });
                }
            }
            None => {
                return Err(LexError {
                    span: Span::new(stream.pos(), stream.pos()),
                    kind: LexErrorKind::EndOfInput,
                });
            }
        };

        // It is already checked that first character is xid_start
        // xid_continue is superset of xid_start, so it will pass this predicate too.
        let ident_string = stream
            .str_until(|c| !unicode_ident::is_xid_continue(c))
            .to_owned();

        let start = stream.pos();
        stream.consume(ident_string.len());
        let end = stream.pos();

        Ok(Ident::new(Rc::from(ident_string), Span::new(start, end)))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Radix {
    Binary,
    Octal,
    Decimal,
    Hexadecimal,
}

impl Radix {
    pub fn as_u32(&self) -> u32 {
        match self {
            Radix::Binary => 2,
            Radix::Octal => 8,
            Radix::Decimal => 10,
            Radix::Hexadecimal => 16,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LiteralNumber {
    repr: Rc<str>,
    span: Span,
    radix: Radix,
    dot_pos: Option<usize>,
    exp_pos: Option<usize>,
    suffix_start: Option<usize>,
}

impl LiteralNumber {
    pub fn new(
        repr: Rc<str>,
        span: Span,
        radix: Radix,
        dot_pos: Option<usize>,
        exp_pos: Option<usize>,
        suffix_start: Option<usize>,
    ) -> Self {
        LiteralNumber {
            repr,
            span,
            radix,
            dot_pos,
            exp_pos,
            suffix_start,
        }
    }

    /// Returns the string representation of the literal, e.g. "0xFF", "1.23e-4", "0b1010_u32".
    pub fn repr(&self) -> &str {
        self.repr.as_ref()
    }

    pub fn rc_repr(&self) -> Rc<str> {
        self.repr.clone()
    }

    /// Returns the radix of the number literal, e.g. 2 for "0b1010", 8 for "0o755", 16 for "0xFF", and 10 for "123".
    pub fn radix(&self) -> Radix {
        self.radix
    }

    /// Returns if the number literal has a dot, e.g. "1.23" has a dot but "123" does not.
    pub fn has_dot(&self) -> bool {
        self.dot_pos.is_some()
    }

    /// Returns if the number literal has an exponent part, e.g. "1e-4" has an exponent but "1.23" does not.
    pub fn has_exponent(&self) -> bool {
        self.exp_pos.is_some()
    }

    /// Returns if the number literal has a suffix, e.g. "0xFF_u32" has a suffix but "1.23e-4" does not.
    pub fn has_suffix(&self) -> bool {
        self.suffix_start.is_some()
    }

    /// Returns the range of the exponent part in the number literal, e.g. "-4" for "1.23e-4" if it exists.
    pub fn exponent_range(&self) -> Option<Range<usize>> {
        let start = self.exponent_start()?;
        let end = self.suffix_start.unwrap_or(self.repr.len());
        Some(start..end)
    }

    /// Returns the range of the mantissa part in the number literal.
    /// Matissa includes integer part and fractional part, but not exponent part, e.g. "1.23" for "1.23e-4" and "FF" for "0xFF_u32".
    /// All number literals have a mantissa part.
    pub fn mantissa_range(&self) -> Range<usize> {
        let start = self.matissa_start();
        let end = self.matissa_end();
        start..end
    }

    /// Returns the range of the integer part in the number literal, e.g. "1" for "1.23e-4" and "FF" for "0xFF_u32".
    /// Range may be empty if there is no integer part, e.g. ".23" has no integer part.
    /// In this case it is assumed that there is an implicit "0" before the dot, so ".23" is treated as "0.23".
    pub fn integer_range(&self) -> Range<usize> {
        let start = self.matissa_start();
        let end = match self.dot_pos() {
            Some(pos) => pos,
            None => self.matissa_end(),
        };
        start..end
    }

    /// Returns the range of the fractional part in the number literal, e.g. "23" for "1.23e-4".
    /// If there is no fractional part, returns `None`.
    pub fn fractional_range(&self) -> Option<Range<usize>> {
        let start = self.dot_pos? + 1;
        let end = self.matissa_end();
        Some(start..end)
    }

    // Get part of the literal before suffix, e.g. "123" for "123_u32".
    pub fn value_range(&self) -> Range<usize> {
        match self.suffix_start {
            None => 0..self.repr.len(),
            Some(pos) => 0..pos,
        }
    }

    /// Returns the range of the suffix in the number literal, e.g. "u32" for "0xFF_u32" if it exists.
    pub fn suffix_range(&self) -> Option<Range<usize>> {
        let start = self.suffix_start?;
        let end = self.repr.len();
        Some(start..end)
    }

    /// Returns the exponent part of the number literal, e.g. "-4" for "1.23e-4" if it exists.
    pub fn exponent(&self) -> Option<&str> {
        let range = self.exponent_range()?;
        Some(&self.repr[range])
    }

    /// Returns the mantissa part of the number literal, e.g. "1.23" for "1.23e-4" and "FF" for "0xFF_u32".
    pub fn mantissa(&self) -> &str {
        let range = self.mantissa_range();
        &self.repr[range]
    }

    /// Returns the integer part of the number literal, e.g. "1" for "1.23e-4" and "FF" for "0xFF_u32".
    /// If there is no integer part, returns an empty string, e.g. "" for ".23".
    pub fn integer(&self) -> &str {
        let range = self.integer_range();
        &self.repr[range]
    }

    /// Returns the fractional part of the number literal, e.g. "23" for "1.23e-4".
    pub fn fractional(&self) -> Option<&str> {
        let range = self.fractional_range()?;
        Some(&self.repr[range])
    }

    /// Returns the value part of the number literal before suffix, e.g. "123" for "123_u32".
    pub fn value(&self) -> &str {
        let range = self.value_range();
        &self.repr[range]
    }

    /// Returns the suffix of the number literal, e.g. "u32" for "0xFF_u32" if it exists.
    pub fn suffix(&self) -> Option<&str> {
        let range = self.suffix_range()?;
        Some(&self.repr[range])
    }

    fn dot_pos(&self) -> Option<usize> {
        self.dot_pos
    }

    /// Returns starting position of the exponent part in the number literal, if any.
    fn exponent_start(&self) -> Option<usize> {
        match self.exp_pos {
            Some(pos) => Some(pos + 1),
            None => None,
        }
    }

    fn matissa_start(&self) -> usize {
        match self.radix {
            Radix::Binary | Radix::Octal | Radix::Hexadecimal => 2, // Skip "0b", "0o", "0x" prefix
            Radix::Decimal => 0,
        }
    }

    fn matissa_end(&self) -> usize {
        self.exp_pos
            .unwrap_or(self.suffix_start.unwrap_or(self.repr.len()))
    }

    pub fn span(&self) -> Span {
        self.span
    }
}

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

    pub fn rc_repr(&self) -> Rc<str> {
        self.repr.clone()
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
            Some((_, '\\')) => {
                // Shift pos yielded by chars
                match parse_escape(0, chars.by_ref()) {
                    Ok(ch) => ch,
                    Err(_) => self.panic_invalid(),
                }
            }
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

    pub fn rc_repr(&self) -> Rc<str> {
        self.repr.clone()
    }

    pub fn prefix(&self) -> Option<char> {
        match self.repr.chars().next() {
            Some(c) if c.is_ascii_alphabetic() => Some(c),
            _ => None,
        }
    }

    /// Returns the unescaped string literal.
    /// Use only if default unescaping is desired for the given string kind.
    pub fn unescape(&self) -> String {
        let mut chars = self.repr.char_indices();
        match chars.next() {
            Some((_, '\"')) => {}
            Some((_, c)) if c.is_ascii_alphabetic() => match chars.next() {
                Some((_, '\"')) => {}
                _ => self.panic_invalid(),
            },
            _ => self.panic_invalid(),
        }

        let mut unescaped = String::new();

        loop {
            let ch = match chars.next() {
                Some((_, '\"')) => break, // Closing quote
                Some((_, '\\')) => {
                    // Shift pos yielded by chars
                    match parse_escape(0, chars.by_ref()) {
                        Ok(ch) => ch,
                        Err(_) => self.panic_invalid(),
                    }
                }
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Literal {
    Number(LiteralNumber),
    Char(LiteralChar),
    String(LiteralString),
}

impl Literal {
    pub fn new_number(
        repr: Rc<str>,
        span: Span,
        radix: Radix,
        dot_pos: Option<usize>,
        exp_pos: Option<usize>,
        suffix_start: Option<usize>,
    ) -> Self {
        Literal::Number(LiteralNumber {
            repr,
            span,
            radix,
            dot_pos,
            exp_pos,
            suffix_start,
        })
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

    pub fn as_number(&self) -> Option<LiteralNumber> {
        match self {
            Literal::Number(n) => Some(n.clone()),
            _ => None,
        }
    }

    pub fn is_char(&self) -> bool {
        matches!(self, Literal::Char(_))
    }

    pub fn as_char(&self) -> Option<LiteralChar> {
        match self {
            Literal::Char(c) => Some(c.clone()),
            _ => None,
        }
    }

    pub fn is_string(&self) -> bool {
        matches!(self, Literal::String(_))
    }

    pub fn as_string(&self) -> Option<LiteralString> {
        match self {
            Literal::String(s) => Some(s.clone()),
            _ => None,
        }
    }

    fn is_literal(stream: &Stream) -> bool {
        match stream.peek_char() {
            Some(c) if c.is_ascii_digit() || c == '"' || c == '\'' => true,
            Some(c) if c.is_ascii_alphabetic() => match stream.peek_char2() {
                Some(c) => c == '"' || c == '\'', // prefixed char or string literal.
                None => false,
            },
            _ => false,
        }
    }

    fn parse(stream: &mut Stream) -> Result<Self, LexError> {
        let start_pos = stream.pos();

        match stream.peek_char() {
            Some(ch @ '0'..='9') => {
                // If numeric literal starts with '0', it may be followed by radix specifier.
                // If it is not followed by radix specifier, it is still parsed as decimal literal, not octal.
                let (radix, skip) = match ch {
                    '0' => match stream.peek_char2() {
                        Some('b' | 'B') => (Radix::Binary, 2),
                        Some('o' | 'O') => (Radix::Octal, 2),
                        Some('x' | 'X') => (Radix::Hexadecimal, 2),
                        _ => (Radix::Decimal, 0),
                    },
                    _ => (Radix::Decimal, 0),
                };

                let mut chars = stream.char_indices();

                for _ in 0..skip {
                    chars.next().unwrap(); // Skip radix specifier
                }

                debug_assert_eq!(skip, chars.offset());

                let len;
                let mut dot_pos = None;
                let mut exp_pos = None;
                let mut suffix_start = None;
                let mut digits = 0usize;

                // Parse number literal

                'a: loop {
                    match chars.next() {
                        // If chars exhausted, it is end of the literal, but len is still None, so set it to the end of the stream after the loop.
                        None => {
                            // End of input, treat it as end of the literal.
                            len = chars.offset();
                            break;
                        }

                        // Digits are allowed in the number literal, obviously.
                        Some((_, ch)) if ch.is_digit(radix.as_u32()) => {
                            digits += 1;
                        }

                        // Underscore is allowed anywhere in the literal
                        Some((_, '_')) => {}

                        // Dot is allowed in the literal once, only before exponent part
                        // and must be followed by at least one digit.
                        Some((pos, '.'))
                            if exp_pos.is_none()
                                && dot_pos.is_none()
                                && radix == Radix::Decimal =>
                        {
                            // Dot must be followed by at least one digit or it's not part of the literal.
                            'b: loop {
                                match chars.next() {
                                    Some((_, '_')) => {}
                                    Some((_, ch)) if ch.is_digit(radix.as_u32()) => {
                                        digits += 1;
                                        break 'b;
                                    }
                                    None | Some(_) => {
                                        len = pos;
                                        break 'a;
                                    }
                                }
                            }

                            dot_pos = Some(pos);
                        }

                        // Exponent is allowed once.
                        // Note that in HEX literals this pattern will match digit pattern above,
                        // so this one won't be chosen.
                        Some((pos, 'e' | 'E')) if exp_pos.is_none() && radix == Radix::Decimal => {
                            let mut sign_pos = None;
                            'b: loop {
                                match chars.next() {
                                    Some((_, '_')) => {}
                                    Some((pos, '-' | '+')) if sign_pos.is_none() => {
                                        sign_pos = Some(pos);
                                    }
                                    Some((_, ch)) if ch.is_digit(radix.as_u32()) => {
                                        digits += 1;
                                        break 'b;
                                    }
                                    Some((pos, _)) if unicode_ident::is_xid_continue(ch) => {
                                        suffix_start = Some(pos);

                                        loop {
                                            match chars.next() {
                                                None => {
                                                    // End of input, treat it as end of the literal.
                                                    len = chars.offset();
                                                    break 'a;
                                                }
                                                Some((_, ch))
                                                    if unicode_ident::is_xid_continue(ch) => {}
                                                Some((pos, _)) => {
                                                    // End of suffix, so end of the literal.
                                                    len = pos;
                                                    break 'a;
                                                }
                                            }
                                        }
                                    }
                                    Some((pos, _)) => {
                                        suffix_start = Some(pos);
                                        len = sign_pos.unwrap_or(pos);
                                        break 'a;
                                    }
                                    None => {
                                        suffix_start = Some(pos);
                                        len = sign_pos.unwrap_or(chars.offset());
                                        break 'a;
                                    }
                                }
                            }
                            exp_pos = Some(pos);
                        }

                        // If nothing of the above, it may be start of the suffix or invalid character.
                        // Suffix is valid ident, so it starts with XID_Start character.
                        Some((pos, ch)) if unicode_ident::is_xid_start(ch) => {
                            suffix_start = Some(pos);

                            loop {
                                match chars.next() {
                                    None => {
                                        // End of input, treat it as end of the literal.
                                        len = chars.offset();
                                        break 'a;
                                    }
                                    Some((_, ch)) if unicode_ident::is_xid_continue(ch) => {}
                                    Some((pos, _)) => {
                                        // End of suffix, so end of the literal.
                                        len = pos;
                                        break 'a;
                                    }
                                }
                            }
                        }

                        // Otherwise it is invalid character, so literal ends there.
                        Some((pos, _)) => {
                            len = pos;
                            break;
                        }
                    }
                }

                drop(chars);

                if digits == 0 {
                    return Err(LexError {
                        span: Span::new(start_pos, start_pos + len),
                        kind: LexErrorKind::NoDigitsInNumber,
                    });
                }

                let literal_str = stream.consume(len);
                let literal = Rc::from(literal_str);
                let end_pos = stream.pos();

                Ok(Literal::new_number(
                    literal,
                    Span::new(start_pos, end_pos),
                    radix,
                    dot_pos,
                    exp_pos,
                    suffix_start,
                ))
            }
            Some(ch)
                if ch == '"' || (ch.is_ascii_alphabetic() && stream.peek_char2() == Some('"')) =>
            {
                // Parse string literal

                let mut chars = stream.char_indices();
                if ch != '"' {
                    chars.next().unwrap(); // Skip prefix character
                }
                let _ = chars.next().unwrap(); // Skip opening quote

                while let Some((pos, ch)) = chars.next() {
                    match ch {
                        '"' => {
                            drop(chars);

                            let len = pos + 1;

                            let literal_str = stream.consume(len);
                            let literal = Rc::from(literal_str);

                            let end_pos = stream.pos();

                            return Ok(Literal::new_string(literal, Span::new(start_pos, end_pos)));
                        }
                        '\\' => {
                            let start = pos;

                            // Shift pos yielded by chars
                            parse_escape(
                                start,
                                chars.by_ref().map(|(pos, ch)| (pos - start - 1, ch)),
                            )?;
                        }
                        _ => {}
                    }
                }

                Err(LexError {
                    span: stream.span(),
                    kind: LexErrorKind::EndOfInput,
                })
            }
            Some(ch)
                if ch == '\''
                    || (ch.is_ascii_alphabetic() && stream.peek_char2() == Some('\'')) =>
            {
                // Parse char literal

                let mut chars = stream.char_indices();
                if ch != '\'' {
                    chars.next().unwrap(); // Skip prefix character
                }
                let _ = chars.next().unwrap(); // Skip opening quote

                match chars.next() {
                    Some((pos, '\'')) => {
                        return Err(LexError {
                            span: Span::new(start_pos, start_pos + pos + 1),
                            kind: LexErrorKind::UnexpectedCharacter(ch),
                        });
                    }
                    Some((pos, '\\')) => {
                        let start = pos;

                        // Shift pos yielded by chars
                        parse_escape(start, chars.by_ref().map(|(pos, ch)| (pos - start - 1, ch)))?;
                    }
                    Some((_pos, _ch)) => {}
                    None => {
                        return Err(LexError {
                            span: Span::new(start_pos, start_pos),
                            kind: LexErrorKind::EndOfInput,
                        });
                    }
                }

                match chars.next() {
                    Some((pos, '\'')) => {
                        drop(chars);

                        let len = pos + 1;

                        let literal_str = stream.consume(len);
                        let literal = Rc::from(literal_str);

                        let end_pos = stream.pos();

                        Ok(Literal::new_char(literal, Span::new(start_pos, end_pos)))
                    }
                    Some((pos, ch)) => Err(LexError {
                        span: Span::new(start_pos, start_pos + pos + ch.len_utf8()),
                        kind: LexErrorKind::UnexpectedCharacter(ch),
                    }),
                    None => Err(LexError {
                        span: Span::new(start_pos, start_pos),
                        kind: LexErrorKind::EndOfInput,
                    }),
                }
            }
            Some(c) => Err(LexError {
                span: Span::new(start_pos, start_pos + c.len_utf8()),
                kind: LexErrorKind::UnexpectedCharacter(c),
            }),
            None => Err(LexError {
                span: Span::new(start_pos, start_pos),
                kind: LexErrorKind::EndOfInput,
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Delimiter {
    Parenthesis, // ()
    Brace,       // {}
    Bracket,     // []
    Block,       // delimited by increased indentation
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Group {
    delimiter: Delimiter,
    tokens: Rc<[Token]>,
    span: Span,
}

impl Group {
    pub fn new(delimiter: Delimiter, tokens: Vec<Token>, span: Span) -> Self {
        Group {
            delimiter,
            tokens: Rc::from(tokens),
            span,
        }
    }

    pub fn delimiter(&self) -> Delimiter {
        self.delimiter
    }

    pub fn span(&self) -> Span {
        self.span
    }

    fn parse_open(stream: &mut Stream) -> Option<Delimiter> {
        match stream.peek_char() {
            Some('(') => {
                stream.consume(1);
                stream.skip_whitespace();
                Some(Delimiter::Parenthesis)
            }
            Some('{') => {
                stream.consume(1);
                stream.skip_whitespace();
                Some(Delimiter::Brace)
            }
            Some('[') => {
                stream.consume(1);
                stream.skip_whitespace();
                Some(Delimiter::Bracket)
            }
            _ => None,
        }
    }

    /// Parse next closing delimiter from the input stream.
    fn parse_close(stream: &mut Stream) -> Option<Delimiter> {
        match stream.peek_char() {
            Some(')') => {
                stream.consume(1);
                stream.skip_whitespace();
                Some(Delimiter::Parenthesis)
            }
            Some('}') => {
                stream.consume(1);
                stream.skip_whitespace();
                Some(Delimiter::Brace)
            }
            Some(']') => {
                stream.consume(1);
                stream.skip_whitespace();
                Some(Delimiter::Bracket)
            }
            _ => None,
        }
    }

    fn parse_body(
        open_delim: Delimiter,
        start_pos: usize,
        stream: &mut Stream,
        indent: usize,
    ) -> Result<Self, LexError> {
        let mut tokens = Vec::new();

        loop {
            if let Some(token) = Newline::try_parse(stream) {
                tokens.push(Token::Newline(token));
                debug_assert!(Newline::try_parse(stream).is_none(), "Newline::try_parse should consume all consecutive newlines");
            }

            if stream.is_empty() {
                break;
            }

            if stream.is_linestart() {
                // Blank lines would be skipped and consumed by Newline::try_parse, so if we are at the start of a line, it must be a non-blank line.
                let line_indent = stream.line_indent();

                if line_indent > indent {
                    let block_group =
                        Group::parse_body(Delimiter::Block, stream.pos(), stream, line_indent)?;

                    tokens.push(Token::Group(block_group));
                    continue;
                }

                if line_indent < indent {
                    // Indentation decreased, so this is the end of the block.
                    if open_delim == Delimiter::Block {
                        break;
                    } else {
                        return Err(LexError {
                            span: Span::new(start_pos, stream.pos()),
                            kind: LexErrorKind::UnclosedDelimiter,
                        });
                    }
                }
            }

            // Can skip whitespace after indentation was handled.
            stream.skip_whitespace();

            // Check for group start symbols.
            if let Some(open_delim) = Self::parse_open(stream) {
                stream.skip_whitespace();

                let group =
                    Group::parse_body(open_delim, stream.pos(), stream, indent)?;

                tokens.push(Token::Group(group));
                continue;
            }
            
            if let Some(close_delim) = Self::parse_close(stream) {
                stream.skip_whitespace();

                if close_delim == open_delim {
                    break;
                } else {
                    return Err(LexError {
                        span: Span::new(start_pos, stream.pos()),
                        kind: LexErrorKind::UnclosedDelimiter,
                    });
                }
            }

            let token = next_token(stream)?;
            tokens.push(token);
        }

        Ok(Group::new(
            open_delim,
            tokens,
            Span::new(start_pos, stream.pos()),
        ))
    }

    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }

    pub fn rc_tokens(&self) -> Rc<[Token]> {
        self.tokens.clone()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Comment {
    repr: Rc<str>,
    span: Span,
}

impl Comment {
    pub fn new(repr: Rc<str>, span: Span) -> Self {
        Comment { repr, span }
    }

    pub fn repr(&self) -> &str {
        self.repr.as_ref()
    }

    pub fn rc_repr(&self) -> Rc<str> {
        self.repr.clone()
    }

    pub fn span(&self) -> Span {
        self.span
    }

    fn is_comment(stream: &Stream) -> bool {
        match stream.peek_char() {
            Some('/') => match stream.peek_char2() {
                Some('/' | '*') => true,
                _ => false,
            },
            _ => false,
        }
    }

    fn parse(stream: &mut Stream) -> Result<Self, LexError> {
        match stream.peek_char() {
            None => Err(LexError {
                span: stream.span(),
                kind: LexErrorKind::EndOfInput,
            }),
            Some('/') => match stream.peek_char2() {
                None => Err(LexError {
                    span: stream.span(),
                    kind: LexErrorKind::EndOfInput,
                }),
                Some('/') => {
                    // Line comment, consume until end of line or end of input
                    let start_pos = stream.pos();

                    let comment_str = stream.str_until(|ch| matches!(ch, '\n' | '\r'));
                    let comment = Rc::from(comment_str);
                    let end_pos = stream.pos();

                    Ok(Comment::new(comment, Span::new(start_pos, end_pos)))
                }
                Some('*') => {
                    // Block comment, consume until "*/" or end of input
                    let start_pos = stream.pos();
                    let mut chars = stream.char_indices();
                    let _ = chars.next().unwrap(); // Skip first '/'
                    let _ = chars.next().unwrap(); // Skip '*'

                    let len;
                    let mut follows_asterisk = false;

                    loop {
                        match chars.next() {
                            None => {
                                return Err(LexError {
                                    span: Span::new(start_pos, start_pos + chars.offset()),
                                    kind: LexErrorKind::EndOfInput,
                                });
                            }
                            Some((_, '*')) => {
                                follows_asterisk = true;
                            }
                            Some((pos, '/')) if follows_asterisk => {
                                len = pos + 1;
                                break;
                            }
                            Some((_, _)) => {
                                follows_asterisk = false;
                            }
                        }
                    }

                    let comment_str = stream.consume(len);
                    let comment = Rc::from(comment_str);
                    let end_pos = stream.pos();
                    Ok(Comment::new(comment, Span::new(start_pos, end_pos)))
                }
                Some(c) => Err(LexError {
                    span: Span::new(stream.pos(), stream.pos() + c.len_utf8()),
                    kind: LexErrorKind::UnexpectedCharacter(c),
                }),
            },
            Some(c) => Err(LexError {
                span: Span::new(stream.pos(), stream.pos() + c.len_utf8()),
                kind: LexErrorKind::UnexpectedCharacter(c),
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Newline {
    span: Span,
}

impl Newline {
    pub fn new(span: Span) -> Self {
        Newline { span }
    }

    pub fn span(&self) -> Span {
        self.span
    }

    fn try_parse(stream: &mut Stream) -> Option<Self> {
        let start_pos = stream.pos();

        if stream.consume_blank_lines() {
            Some(Newline::new(Span::new(start_pos, stream.pos())))
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Token {
    Punct(Punct),
    Ident(Ident),
    Literal(Literal),
    Group(Group),
    Comment(Comment),
    Newline(Newline),
}

impl Token {
    pub fn span(&self) -> Span {
        match self {
            Token::Punct(p) => p.span(),
            Token::Ident(i) => i.span(),
            Token::Literal(l) => l.span(),
            Token::Group(g) => g.span(),
            Token::Comment(c) => c.span(),
            Token::Newline(n) => n.span(),
        }
    }
}

/// Parses next token from the input stream except Group and Newline tokens.
///
/// # Error
///
/// Returns error if the next token is malformed
/// or if the end of input is reached.
fn next_token(stream: &mut Stream) -> Result<Token, LexError> {
    if stream.is_empty() {
        return Err(LexError {
            span: Span::new(stream.pos(), stream.pos()),
            kind: LexErrorKind::EndOfInput,
        });
    }

    // Must precede punct, as "//" and "/*" would be parsed as two puncts otherwise.
    if Comment::is_comment(stream) {
        let comment = Comment::parse(stream)?;
        stream.skip_whitespace();
        return Ok(Token::Comment(comment));
    }

    if Punct::is_punct(stream) {
        let punct = Punct::parse(stream)?;
        stream.skip_whitespace();
        return Ok(Token::Punct(punct));
    }

    if Ident::is_ident(stream) {
        let ident = Ident::parse(stream)?;
        stream.skip_whitespace();
        return Ok(Token::Ident(ident));
    }

    if Literal::is_literal(stream) {
        let literal = Literal::parse(stream)?;
        stream.skip_whitespace();
        return Ok(Token::Literal(literal));
    }

    let pos = stream.pos();
    match stream.peek_char() {
        None => Err(LexError {
            span: Span::new(pos, pos),
            kind: LexErrorKind::EndOfInput,
        }),
        Some(ch) => Err(LexError {
            span: Span::new(pos, pos + ch.len_utf8()),
            kind: LexErrorKind::UnexpectedCharacter(ch),
        }),
    }
}

pub fn parse_stream(stream: &mut Stream) -> Result<Rc<[Token]>, LexError> {
    let start_pos = stream.pos();
    let group = Group::parse_body(Delimiter::Block, start_pos, stream, 0)?;
    Ok(group.tokens)
}

fn parse_escape(
    span_offset: usize,
    mut chars: impl Iterator<Item = (usize, char)>,
) -> Result<char, LexError> {
    // Skip escaped character
    match chars.next() {
        None => Err(LexError {
            span: Span::point(span_offset),
            kind: LexErrorKind::EndOfInput,
        }),
        Some((_, '\\')) => Ok('\\'),
        Some((_, '\'')) => Ok('\''),
        Some((_, '\"')) => Ok('\"'),
        Some((_, '\n')) => Ok('\n'),
        Some((_, '\r')) => Ok('\r'),
        Some((_, '\t')) => Ok('\t'),
        Some((_, '\0')) => Ok('\0'),
        Some((_, 'x')) => {
            // Hex escape sequence, skip \xNN
            let mut value = 0u8;
            for i in 0..2 {
                match chars.next() {
                    None => {
                        return Err(LexError {
                            span: Span::new(span_offset, span_offset + i + 1),
                            kind: LexErrorKind::EndOfInput,
                        });
                    }
                    Some((_, ch)) if ch.is_ascii_hexdigit() => {
                        value <<= 4;
                        value |= ch.to_digit(16).unwrap() as u8;
                    }
                    Some((pos, ch)) => {
                        debug_assert_eq!(pos, i + 1);

                        // Expected hex digit, got something else.
                        return Err(LexError {
                            span: Span::new(span_offset, span_offset + pos + ch.len_utf8()),
                            kind: LexErrorKind::UnexpectedCharacter(ch),
                        });
                    }
                }
            }

            Ok(char::from(value))
        }
        Some((_, 'u')) => match chars.next() {
            None => {
                return Err(LexError {
                    span: Span::new(span_offset, span_offset + 1),
                    kind: LexErrorKind::EndOfInput,
                });
            }
            Some((_, ch)) if ch.is_ascii_hexdigit() => {
                // Unicode escape sequence, skip \uNNNN - 4 hex digits.

                let mut value = ch.to_digit(16).unwrap();

                for i in 0..3 {
                    match chars.next() {
                        None => {
                            return Err(LexError {
                                span: Span::new(span_offset, span_offset + i + 2),
                                kind: LexErrorKind::EndOfInput,
                            });
                        }
                        Some((_, ch)) if ch.is_ascii_hexdigit() => {
                            value <<= 4;
                            value |= ch.to_digit(16).unwrap();
                        }
                        Some((pos, ch)) => {
                            debug_assert_eq!(pos, i + 2);

                            return Err(LexError {
                                span: Span::new(span_offset, span_offset + pos + ch.len_utf8()),
                                kind: LexErrorKind::UnexpectedCharacter(ch),
                            });
                        }
                    }
                }

                match char::from_u32(value) {
                    Some(ch) => Ok(ch),
                    None => Err(LexError {
                        span: Span::new(span_offset, span_offset + 5),
                        kind: LexErrorKind::InvalidEscapedCharacter(value),
                    }),
                }
            }

            Some((_, '{')) => {
                // Unicode escape sequence, skip \u{NNNNNN} - up to 6 hex digits inside braces.
                let mut closing_brace_pos = None;
                let mut value = 0u32;

                for i in 0..6 {
                    match chars.next() {
                        None => {
                            return Err(LexError {
                                span: Span::new(span_offset, span_offset + i + 2),
                                kind: LexErrorKind::EndOfInput,
                            });
                        }
                        Some((pos, '}')) => {
                            closing_brace_pos = Some(pos);
                            break;
                        }
                        Some((_, ch)) if ch.is_ascii_hexdigit() => {
                            value <<= 4;
                            value |= ch.to_digit(16).unwrap();
                        }
                        Some((pos, ch)) => {
                            debug_assert_eq!(pos, i + 2);

                            return Err(LexError {
                                span: Span::new(span_offset, span_offset + pos + ch.len_utf8()),
                                kind: LexErrorKind::UnexpectedCharacter(ch),
                            });
                        }
                    }
                }

                let closing_brace_pos = match closing_brace_pos {
                    None => match chars.next() {
                        Some((pos, '}')) => {
                            debug_assert_eq!(pos, 8);
                            8
                        }
                        None => {
                            return Err(LexError {
                                span: Span::new(span_offset, span_offset + 8),
                                kind: LexErrorKind::EndOfInput,
                            });
                        }
                        Some((pos, ch)) => {
                            debug_assert_eq!(pos, 8);

                            return Err(LexError {
                                span: Span::new(span_offset, span_offset + 8 + ch.len_utf8()),
                                kind: LexErrorKind::UnexpectedCharacter(ch),
                            });
                        }
                    },
                    Some(pos) => {
                        debug_assert!(pos < 8);
                        pos
                    }
                };

                match char::from_u32(value) {
                    Some(ch) => Ok(ch),
                    None => Err(LexError {
                        span: Span::new(span_offset, closing_brace_pos + 1),
                        kind: LexErrorKind::InvalidEscapedCharacter(value),
                    }),
                }
            }
            Some((pos, ch)) => {
                debug_assert_eq!(pos, 1);

                return Err(LexError {
                    span: Span::new(span_offset, span_offset + 1),
                    kind: LexErrorKind::UnexpectedCharacter(ch),
                });
            }
        },
        Some((_, 'U')) => {
            // Unicode escape sequence, skip \UXXXXXXXX - 8 hex digits.
            let mut value = 0u32;

            for i in 0..8 {
                match chars.next() {
                    None => {
                        return Err(LexError {
                            span: Span::new(span_offset, span_offset + i + 1),
                            kind: LexErrorKind::EndOfInput,
                        });
                    }
                    Some((_, ch)) if ch.is_ascii_hexdigit() => {
                        value <<= 4;
                        value |= ch.to_digit(16).unwrap();
                    }
                    Some((pos, ch)) => {
                        debug_assert_eq!(pos, i + 1);

                        return Err(LexError {
                            span: Span::new(span_offset, span_offset + pos),
                            kind: LexErrorKind::UnexpectedCharacter(ch),
                        });
                    }
                }
            }

            match char::from_u32(value) {
                Some(ch) => Ok(ch),
                None => Err(LexError {
                    span: Span::new(span_offset, span_offset + 9),
                    kind: LexErrorKind::InvalidEscapedCharacter(value),
                }),
            }
        }
        Some((pos, ch)) => {
            debug_assert_eq!(pos, 0);

            return Err(LexError {
                span: Span::new(span_offset, span_offset + ch.len_utf8()),
                kind: LexErrorKind::UnexpectedCharacter(ch),
            });
        }
    }
}
