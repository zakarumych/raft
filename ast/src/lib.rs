#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::rc::Rc;

pub use raft_lexer::{Span, LiteralNumber, LiteralChar, LiteralString, Token};

pub mod parse;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Ident {
    pub span: Span,
    pub name: Rc<str>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Atom {
    pub span: Span,
    pub name: Rc<str>,
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOpKind {
    Not,    // !
    Neg,    // -
    BitNot, // ~
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct UnaryOp {
    pub span: Span,
    pub kind: UnaryOpKind,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BinaryOp {
    pub kind: BinaryOpKind,
    pub span: Span,
}

impl BinaryOp {
    pub fn precedence(&self) -> u8 {
        use BinaryOpKind::*;

        match self.kind {
            BitAnd | BitOr | BitXor | Shl | Shr => 5,
            Pow => 4,
            Mul | Div => 3,
            Add | Sub => 2,
            Eq | Ne | Lt | Gt | Le | Ge => 1,
        }
    }

    pub fn is_right_assoc(&self) -> bool {
        use BinaryOpKind::*;

        match self.kind {
            Pow => true,
            _ => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Expr {
    pub span: Span,
    pub kind: ExprKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExprKind {
    Literal(Literal),
    Ident(Ident),
    Atom(Atom),
    List(Vec<Expr>),
    Record(Vec<ExprRecordField>),
    Unary(UnaryOp, Box<Expr>),
    Binary(Box<Expr>, BinaryOp, Box<Expr>),
    Apply(Box<Expr>, Vec<Expr>),
    Field(Box<Expr>, Ident),
    Index(Box<Expr>, Box<Expr>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExprRecordField {
    pub span: Span,
    pub key: Ident,
    pub value: Option<Expr>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Pattern {
    pub span: Span,
    pub kind: PatternKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PatternKind {
    Ident(Ident),
    Atom(Atom),
    Literal(Literal),
    List(Vec<Pattern>),
    Record(Vec<PatternRecordField>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PatternRecordField {
    pub span: Span,
    pub key: Ident,
    pub pattern: Option<Pattern>,
}

// Statements and blocks
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Stmt {
    pub span: Span,
    pub kind: StmtKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum StmtKind {
    Expr(Expr),
    AssignPattern { target: Pattern, value: Expr },
    AssignField { target: Box<Expr>, field: Ident, value: Expr },
    AssignIndex { target: Box<Expr>, index: Box<Expr>, value: Expr },
    If { cond: Expr, then_branch: Vec<Stmt>, else_branch: Option<Vec<Stmt>> },
    While { cond: Expr, body: Vec<Stmt>, else_branch: Option<Vec<Stmt>> },
    For { target: Pattern, iterable: Expr, body: Vec<Stmt>, else_branch: Option<Vec<Stmt>> },
    Return(Expr),
    Break,
    Continue,
}
