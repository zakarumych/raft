#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod lex;
mod stream;
mod span;

pub use self::{
    lex::{
        Comment, Delimiter, Group, Ident, LexError, LexErrorKind, Literal, LiteralChar,
        LiteralNumber, LiteralString, Punct, Token, parse_str, parse_stream, Options
    },
    stream::{Stream},
    span::{Span, SpannedSource},
};

#[cfg(test)]
mod tests;
