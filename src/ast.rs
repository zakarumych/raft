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

#[derive(Debug, Clone, PartialEq)]
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

// ---- Expr -------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
pub struct ExprRecordField {
    pub key: Ident,
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Pattern {
    pub kind: PatternKind,
    pub span: Span,
}

use crate::literal::Literal;

#[derive(Debug, Clone, PartialEq)]
pub enum PatternKind {
    Ident(Ident),
    Atom(Atom),
    Literal(Literal),
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

// Statements and blocks
#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind {
    Expr(Expr),
    AssignPattern(Pattern, Expr),
    AssignField { target: Box<Expr>, field: Ident, value: Expr },
    AssignIndex { target: Box<Expr>, index: Box<Expr>, value: Expr },
    Return(Expr),
    If { cond: Expr, then_branch: Vec<Stmt>, else_branch: Option<Vec<Stmt>> },
}
