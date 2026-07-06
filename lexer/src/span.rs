use core::fmt;


/// A value specifying a span of text in the input stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    #[inline]
    pub const fn new(start: usize, end: usize) -> Self {
        Span { start, end }
    }

    #[inline]
    pub const fn point(point: usize) -> Self {
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
    pub const fn start(&self) -> Span {
        Span::new(self.start, self.start)
    }

    /// Returns a zero-length span at the end of this span.
    pub const fn end(&self) -> Span {
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

pub struct SpannedSource<'a> {
    source: &'a str,
    span: Span,
}

impl<'a> SpannedSource<'a> {
    pub const fn new(source: &'a str, span: Span) -> Self {
        SpannedSource { source, span }
    }
}

impl fmt::Display for SpannedSource<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {

        let start_line = self.span.start().get_line(self.source);
        let start_column = self.span.start().get_column(self.source);
        let end_line = self.span.end().get_line(self.source);
        let end_column = self.span.end().get_column(self.source);

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
