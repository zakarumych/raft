#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod buffer;
mod lex;
mod stream;

pub use self::{
    lex::{
        Comment, Delimiter, Group, Ident, LexError, Literal, LexErrorKind, LiteralNumber, LiteralChar, LiteralString, Punct, Token,
        parse_stream,
    },
    stream::{Span, Stream},
};

#[cfg(test)]
mod tests;
