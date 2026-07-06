#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::rc::Rc;

pub use raft_lexer::{Span, LiteralNumber, LiteralChar, LiteralString};

pub mod parser;

pub use raft_lexer as lexer;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Ident {
    span: Span,
    name: Rc<str>,
}

impl Ident {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn rc_name(&self) -> Rc<str> {
        self.name.clone()
    }

    pub fn span(&self) -> Span {
        self.span
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Atom {
    span: Span,
    name: Rc<str>,
}

impl Atom {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn rc_name(&self) -> Rc<str> {
        self.name.clone()
    }

    pub fn span(&self) -> Span {
        self.span
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Literal {
    Number(LiteralNumber),
    Char(LiteralChar),
    String(LiteralString),
}

impl Literal {
    pub fn span(&self) -> Span {
        match self {
            Literal::Number(n) => n.span(),
            Literal::Char(c) => c.span(),
            Literal::String(s) => s.span(),
        }
    }

    pub fn is_number(&self) -> bool {
        matches!(self, Literal::Number(_))
    }

    pub fn is_char(&self) -> bool {
        matches!(self, Literal::Char(_))
    }

    pub fn is_string(&self) -> bool {
        matches!(self, Literal::String(_))
    }

    pub fn as_number(&self) -> Option<&LiteralNumber> {
        match self {
            Literal::Number(n) => Some(n),
            _ => None,
        }
    }

    pub fn as_char(&self) -> Option<&LiteralChar> {
        match self {
            Literal::Char(c) => Some(c),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&LiteralString> {
        match self {
            Literal::String(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOpKind {
    Not,    // !
    BitNot, // ~
    Pos,    // +
    Neg,    // -
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct UnaryOp {
    span: Span,
    kind: UnaryOpKind,
}

impl UnaryOp {
    pub fn new(kind: UnaryOpKind, span: Span) -> Self {
        Self { kind, span }
    }

    pub fn kind(&self) -> UnaryOpKind {
        self.kind
    }

    pub fn is_(&self, kind: UnaryOpKind) -> bool {
        self.kind == kind
    }

    pub fn span(&self) -> Span {
        self.span
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOpKind {
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

impl BinaryOpKind {
    pub fn precedence(&self) -> u8 {
        use BinaryOpKind::*;

        match self {
            BitAnd | BitOr | BitXor | Shl | Shr => 5,
            Pow => 4,
            Mul | Div => 3,
            Add | Sub => 2,
            Eq | Ne | Lt | Gt | Le | Ge => 1,
        }
    }

    pub fn is_right_assoc(&self) -> bool {
        use BinaryOpKind::*;

        match self {
            Pow => true,
            _ => false,
        }
    }

    pub fn token_size(&self) -> usize {
        use BinaryOpKind::*;

        match self {
            BitAnd | BitOr | BitXor | Mul | Div | Add | Sub | Lt | Gt => 1,
            Shl | Shr | Pow | Eq | Ne | Le | Ge => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BinaryOp {
    kind: BinaryOpKind,
    span: Span,
}

impl BinaryOp {
    pub fn new(kind: BinaryOpKind, span: Span) -> Self {
        Self { kind, span }
    }

    pub fn kind(&self) -> BinaryOpKind {
        self.kind
    }

    pub fn is_(&self, kind: BinaryOpKind) -> bool {
        self.kind == kind
    }

    pub fn span(&self) -> Span {
        self.span
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExprRecordField {
    span: Span,
    key: Ident,
    value: Option<Expr>,
}

impl ExprRecordField {
    pub fn new(key: Ident, value: Option<Expr>, span: Span) -> Self {
        Self { key, value, span }
    }

    pub fn key(&self) -> &Ident {
        &self.key
    }

    pub fn value(&self) -> Option<&Expr> {
        self.value.as_ref()
    }

    pub fn span(&self) -> Span {
        self.span
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExprKind {
    Literal(Literal),
    Ident(Ident),
    Atom(Atom),
    List(Rc<[Expr]>),
    Record(Rc<[ExprRecordField]>),
    Unary(UnaryOp, Rc<Expr>),
    Binary(Rc<Expr>, BinaryOp, Rc<Expr>),
    Apply(Rc<Expr>, Rc<[Expr]>),
    Field(Rc<Expr>, Ident),
    Index(Rc<Expr>, Rc<Expr>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Expr {
    span: Span,
    kind: ExprKind,
}

impl Expr {
    pub fn new(kind: ExprKind, span: Span) -> Self {
        Self { kind, span }
    }

    pub fn kind(&self) -> &ExprKind {
        &self.kind
    }

    pub fn span(&self) -> Span {
        self.span
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PatRecordField {
    span: Span,
    key: Ident,
    pat: Option<Pat>,
}

impl PatRecordField {
    pub fn new(key: Ident, pat: Option<Pat>, span: Span) -> Self {
        Self { key, pat, span }
    }

    pub fn key(&self) -> &Ident {
        &self.key
    }

    pub fn pat(&self) -> Option<&Pat> {
        self.pat.as_ref()
    }

    pub fn span(&self) -> Span {
        self.span
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PatKind {
    Ident(Ident),
    Atom(Atom),
    Literal(Literal),
    List(Rc<[Pat]>),
    Record(Rc<[PatRecordField]>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Pat {
    span: Span,
    kind: PatKind,
}

impl Pat {
    pub fn new(kind: PatKind, span: Span) -> Self {
        Self { kind, span }
    }

    pub fn kind(&self) -> &PatKind {
        &self.kind
    }

    pub fn span(&self) -> Span {
        self.span
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum StmtKind {
    Expr(Expr),
    AssignPattern { target: Pat, value: Expr },
    AssignField { target: Rc<Expr>, field: Ident, value: Expr },
    AssignIndex { target: Rc<Expr>, index: Rc<Expr>, value: Expr },
    If { cond: Expr, then_branch: Vec<Stmt>, else_branch: Option<Vec<Stmt>> },
    While { cond: Expr, body: Vec<Stmt>, else_branch: Option<Vec<Stmt>> },
    For { target: Pat, iterable: Expr, body: Vec<Stmt>, else_branch: Option<Vec<Stmt>> },
    Return(Option<Expr>),
    Break,
    Continue,
}


// Statements and blocks
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Stmt {
    span: Span,
    kind: StmtKind,
}

impl Stmt {
    pub fn new(kind: StmtKind, span: Span) -> Self {
        Self { kind, span }
    }

    pub fn kind(&self) -> &StmtKind {
        &self.kind
    }

    pub fn span(&self) -> Span {
        self.span
    }
}