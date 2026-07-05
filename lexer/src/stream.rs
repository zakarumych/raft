use core::str::CharIndices;

use crate::buffer::Buffer;

/// A value specifying a span of text in the input stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    #[inline]
    pub fn new(start: usize, end: usize) -> Self {
        Span { start, end }
    }

    #[inline]
    pub fn point(point: usize) -> Self {
        Span {
            start: point,
            end: point,
        }
    }

    #[inline]
    pub fn join(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    /// Returns a zero-length span at the start of this span.
    pub fn start(&self) -> Span {
        Span::new(self.start, self.start)
    }

    /// Returns a zero-length span at the end of this span.
    pub fn end(&self) -> Span {
        Span::new(self.end, self.end)
    }

    /// Returns the source code corresponding to this span from the given buffer.
    /// The buffer should be the same one used to create the stream that produced this span.
    pub fn get_source<'a>(&self, source: &'a str) -> &'a str {
        &source[self.start..self.end]
    }

    /// Returns the line number corresponding to start of this span in the given buffer.
    ///
    /// The buffer should be the same one used to create the stream that produced this span.
    ///
    /// To get line numbers for the end of the span, use `span.end().get_line(buffer)`.
    pub fn get_line(&self, source: &str) -> usize {
        source[..self.start].chars().filter(|&c| c == '\n').count()
    }

    /// Returns the column number corresponding to the start of this span in the given buffer.
    ///
    /// The buffer should be the same one used to create the stream that produced this span.
    ///
    /// To get column numbers for the end of the span, use `span.end().get_column(buffer)`.
    pub fn get_column(&self, source: &str) -> usize {
        let line_start = source[..self.start].rfind('\n').map_or(0, |pos| pos + 1);
        source[line_start..self.start].chars().count()
    }
}

/// A low-level UTF-8 stream for building a lexer.
pub struct Stream {
    input: Buffer,
    pos: usize,
    end: usize,
}

impl Stream {
    pub fn from_str(input: &str) -> Self {
        Stream {
            input: Buffer::from_str(input),
            pos: 0,
            end: input.len(),
        }
    }

    /// Cuts the stream at the specified end position.
    pub fn cut_at(&mut self, end: usize) {
        self.end = self.end.min(self.pos + end);
    }

    pub fn is_empty(&self) -> bool {
        self.pos >= self.end
    }

    /// Peeks at the next character in the stream without consuming it.
    pub fn fork(&self) -> Self {
        Stream {
            input: self.input.clone(),
            pos: self.pos,
            end: self.end,
        }
    }

    /// Returns the current position in the stream.
    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn span(&self) -> Span {
        Span::new(self.pos, self.end)
    }

    /// Consumes `n` bytes from the stream and returns them as a string slice.
    pub fn consume(&mut self, n: usize) -> &str {
        let len = n.min(self.end - self.pos);
        let start = self.pos;
        self.pos += len;
        &self.input.as_str()[start..self.pos]
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
        &self.input.as_str()[self.pos..self.end]
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
            || self.input.as_str()[..self.pos].ends_with('\n')
            || self.input.as_str()[..self.pos].ends_with('\r')
    }

    /// Consumes blank lines
    /// From the current position if only whitespace characters present until a newline
    /// consumes the line and repeats until a non-blank line is found or the stream is empty.
    /// 
    /// Returns true if at least one blank line was consumed, false otherwise.
    pub fn consume_blank_lines(&mut self) -> bool {
        let mut newline_consumed = false;

        'a: loop {
            let s = self.as_str();
            let mut chars = s.char_indices();
            while let Some((i, c)) = chars.next() {
                match c {
                    ' ' | '\t' => {}
                    '\n' | '\r' => {
                        self.consume(i);
                        if self.as_str().starts_with("\r\n") {
                            self.consume(2);
                        } else {
                            self.consume(1);
                        }
                        newline_consumed = true;
                        continue 'a;
                    }
                    _ => break,
                }
            }

            return newline_consumed;
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
