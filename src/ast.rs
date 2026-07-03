#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

#[derive(Debug, Clone)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Atom {
    pub name: String,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Not,    // !
    Neg,    // -
    BitNot, // ~
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    // Bit ops — tightest, left-to-right
    BitAnd, // &
    BitOr,  // |
    BitXor, // ^
    Shl,    // <<
    Shr,    // >>
    // Power — right-to-left
    Pow,    // **
    // Multiplicative — left-to-right
    Mul,    // *
    Div,    // /
    // Additive — left-to-right
    Add, // +
    Sub, // -
    // Comparison — left-to-right, loosest
    Eq, // ==
    Ne, // !=
    Lt, // <
    Gt, // >
    Le, // <=
    Ge, // >=
}

#[derive(Debug, Clone, PartialEq)]
pub struct Pattern {
    pub kind: PatternKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PatternKind {
    Ident(Ident),
    Atom(Atom),
    List(Vec<Pattern>),
    Record(Vec<RecordPatternField>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecordPatternField {
    pub key: Ident,
    pub pattern: Pattern,
    pub span: Span,
}

impl BinaryOp {
    pub fn precedence(self) -> u8 {
        match self {
            Self::BitAnd | Self::BitOr | Self::BitXor | Self::Shl | Self::Shr => 5,
            Self::Pow => 4,
            Self::Mul | Self::Div => 3,
            Self::Add | Self::Sub => 2,
            Self::Eq | Self::Ne | Self::Lt | Self::Gt | Self::Le | Self::Ge => 1,
        }
    }

    pub fn is_right_assoc(self) -> bool {
        matches!(self, Self::Pow)
    }
}
