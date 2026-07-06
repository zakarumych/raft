use core::{str::CharIndices};
use alloc::rc::Rc;

use crate::span::Span;

/// A low-level UTF-8 stream for building a lexer.
pub struct Stream {
    input: Rc<str>,
    pos: usize,
}

impl Stream {
    pub fn from_str(input: &str) -> Self {
        Stream {
            input: Rc::from(input),
            pos: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.pos >= self.input.len()
    }

    // /// Peeks at the next character in the stream without consuming it.
    // pub fn fork(&self) -> Self {
    //     Stream {
    //         input: self.input.clone(),
    //         pos: self.pos,
    //     }
    // }

    /// Returns the current position in the stream.
    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn span(&self) -> Span {
        Span::new(self.pos, self.input.len())
    }

    /// Consumes `n` bytes from the stream and returns them as a string slice.
    pub fn consume(&mut self, n: usize) -> &str {
        let len = n.min(self.input.len() - self.pos);
        let start = self.pos;
        self.pos += len;
        &self.input[start..self.pos]
    }

    pub fn peek_char(&self) -> Option<char> {
        self.as_str().chars().next()
    }

    pub fn peek_char2(&self) -> Option<char> {
        self.as_str().chars().nth(1)
    }

    pub fn match_indices<P: Fn(char) -> bool>(
        &self,
        p: P,
    ) -> impl Iterator<Item = (usize, char)> + use<'_, P> {
        self.as_str().char_indices().filter(move |&(_, c)| p(c))
    }

    pub fn str_until<P: Fn(char) -> bool>(&self, p: P) -> &str {
        match self.as_str().find(p) {
            None => self.as_str(),
            Some(pos) => &self.as_str()[..pos],
        }
    }

    pub fn char_indices(&self) -> CharIndices<'_> {
        self.as_str().char_indices()
    }

    fn as_str(&self) -> &str {
        &self.input[self.pos..]
    }

    /// Advances the stream past any whitespace characters.
    pub fn skip_whitespace(&mut self) {
        let s = self.as_str();
        let len = s.len();
        match s.find(|c: char| !matches!(c, ' ' | '\t')) {
            None => {
                self.consume(len);
            }
            Some(pos) => {
                self.consume(pos);
            }
        }
    }

    pub fn is_linestart(&self) -> bool {
        self.pos == 0
            || self.input[..self.pos].ends_with('\n')
            || self.input[..self.pos].ends_with('\r')
    }

    fn consume_newline(&mut self) {
        if self.as_str().starts_with("\r\n") {
            self.consume(2);
        } else if self.as_str().starts_with('\n') || self.as_str().starts_with('\r') {
            self.consume(1);
        }
    }

    /// Consumes blank lines.
    pub fn skip_blank_lines(&mut self) {
        if !self.is_linestart() {
            return;
        }

        'a: loop {
            let s = self.as_str();

            let mut chars = s.char_indices();
            while let Some((i, c)) = chars.next() {
                match c {
                    ' ' | '\t' => {}
                    '\n' | '\r' => {
                        self.consume(i);
                        self.consume_newline();
                        continue 'a;
                    }
                    _ => return,
                }
            }

            // Whitespace until the end of the stream, consume it all.
            self.pos = self.input.len();
            return;
        }

    }

    /// Assuming the current position is at a newline, returns the indentation level.
    pub fn line_indent(&self) -> usize {
        debug_assert!(self.is_linestart());
        let s = self.as_str();
        let mut count = 0;
        for c in s.chars() {
            match c {
                ' ' => count += 1,
                '\t' => count = (count + 4) & !3, // Assuming a tab stop every 4 spaces
                _ => break,
            }
        }
        count
    }

    // fn skip_comment(&mut self) -> bool {
    //     if self.as_str().starts_with("//") {
    //         // Single-line comment
    //         if let Some(pos) = self.as_str().find('\n') {
    //             self.consume(pos);
    //         } else {
    //             // Consume until the end of the stream
    //             self.consume(self.end - self.pos);
    //         }
    //         true
    //     } else if self.as_str().starts_with("/*") {
    //         // Multi-line comment
    //         if let Some(end_pos) = self.as_str().find("*/") {
    //             self.consume(end_pos + 2); // +2 to consume the closing */
    //         } else {
    //             // Unterminated comment, consume until the end of the stream
    //             self.consume(self.end - self.pos);
    //         }
    //         true
    //     } else {
    //         false
    //     }
    // }

    // /// Advances the stream past any whitespace characters and comments.
    // pub fn skip_comments_and_whitespace(&mut self) {
    //     self.skip_whitespace();

    //     loop {
    //         if !self.skip_comment() {
    //             break;
    //         }

    //         self.skip_whitespace();
    //     }
    // }
}

