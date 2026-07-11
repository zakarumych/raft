#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::rc::Rc;

pub use raft_lexer::{LitChar, LitNum, LitStr, Span};

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
pub enum Lit {
    Num(LitNum),
    Char(LitChar),
    Str(LitStr),
}

impl Lit {
    pub fn span(&self) -> Span {
        match self {
            Lit::Num(n) => n.span(),
            Lit::Char(c) => c.span(),
            Lit::Str(s) => s.span(),
        }
    }

    pub fn is_number(&self) -> bool {
        matches!(self, Lit::Num(_))
    }

    pub fn is_char(&self) -> bool {
        matches!(self, Lit::Char(_))
    }

    pub fn is_string(&self) -> bool {
        matches!(self, Lit::Str(_))
    }

    pub fn as_number(&self) -> Option<&LitNum> {
        match self {
            Lit::Num(n) => Some(n),
            _ => None,
        }
    }

    pub fn as_char(&self) -> Option<&LitChar> {
        match self {
            Lit::Char(c) => Some(c),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&LitStr> {
        match self {
            Lit::Str(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnOpKind {
    Not,    // !
    BitNot, // ~
    Pos,    // +
    Neg,    // -
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct UnOp {
    span: Span,
    kind: UnOpKind,
}

impl UnOp {
    pub fn new(kind: UnOpKind, span: Span) -> Self {
        Self { kind, span }
    }

    pub fn kind(&self) -> UnOpKind {
        self.kind
    }

    pub fn is_(&self, kind: UnOpKind) -> bool {
        self.kind == kind
    }

    pub fn span(&self) -> Span {
        self.span
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinOpKind {
    // Bit ops — tightest, left-to-right
    BitAnd, // &
    BitOr,  // |
    BitXor, // ^
    Shl,    // <<
    Shr,    // >>
    // Power — right-to-left
    Pow, // **
    // Multiplicative — left-to-right
    Mul, // *
    Div, // /
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

impl BinOpKind {
    pub fn precedence(&self) -> u8 {
        use BinOpKind::*;

        match self {
            BitAnd | BitOr | BitXor | Shl | Shr => 5,
            Pow => 4,
            Mul | Div => 3,
            Add | Sub => 2,
            Eq | Ne | Lt | Gt | Le | Ge => 1,
        }
    }

    pub fn is_right_assoc(&self) -> bool {
        use BinOpKind::*;

        match self {
            Pow => true,
            _ => false,
        }
    }

    pub fn token_size(&self) -> usize {
        use BinOpKind::*;

        match self {
            BitAnd | BitOr | BitXor | Mul | Div | Add | Sub | Lt | Gt => 1,
            Shl | Shr | Pow | Eq | Ne | Le | Ge => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BinOp {
    kind: BinOpKind,
    span: Span,
}

impl BinOp {
    pub fn new(kind: BinOpKind, span: Span) -> Self {
        Self { kind, span }
    }

    pub fn kind(&self) -> BinOpKind {
        self.kind
    }

    pub fn is_(&self, kind: BinOpKind) -> bool {
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
    Ident(Ident),
    Atom(Atom),
    Literal(Lit),
    List(Rc<[Expr]>),
    Record(Rc<[ExprRecordField]>),
    Unary(UnOp, Rc<Expr>),
    Binary(Rc<Expr>, BinOp, Rc<Expr>),
    Apply(Rc<Expr>, Rc<[Expr]>),
    Field(Rc<Expr>, Ident),
    Index(Rc<Expr>, Rc<Expr>),
    Parenthesized(Rc<Expr>),
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
    pattern: Option<Pat>,
}

impl PatRecordField {
    pub fn new(key: Ident, pattern: Option<Pat>, span: Span) -> Self {
        Self { key, pattern, span }
    }

    pub fn key(&self) -> &Ident {
        &self.key
    }

    pub fn pattern(&self) -> Option<&Pat> {
        self.pattern.as_ref()
    }

    pub fn span(&self) -> Span {
        self.span
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PatKind {
    Ident(Ident),
    Atom(Atom),
    Literal(Lit),
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
    AssignPat {
        target: Pat,
        value: Expr,
    },
    AssignField {
        target: Rc<Expr>,
        field: Ident,
        value: Expr,
    },
    AssignIndex {
        target: Rc<Expr>,
        index: Rc<Expr>,
        value: Expr,
    },
    If {
        cond: Expr,
        then_branch: Rc<[Stmt]>,
        else_branch: Option<Rc<[Stmt]>>,
    },
    While {
        cond: Expr,
        body: Rc<[Stmt]>,
        else_branch: Option<Rc<[Stmt]>>,
    },
    For {
        target: Pat,
        iterable: Expr,
        body: Rc<[Stmt]>,
        else_branch: Option<Rc<[Stmt]>>,
    },
    Return(Option<Expr>),
    Break,
    Continue,
    Fn {
        name: Ident,
        params: Rc<[Pat]>,
        body: Rc<[Stmt]>,
    },
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

pub struct Export {
    span: Span,
    fields: Rc<[ExprRecordField]>,
}

impl Export {
    pub fn new(fields: Rc<[ExprRecordField]>, span: Span) -> Self {
        Export { fields, span }
    }

    pub fn fields(&self) -> &[ExprRecordField] {
        &self.fields
    }

    pub fn rc_fields(&self) -> Rc<[ExprRecordField]> {
        self.fields.clone()
    }

    pub fn span(&self) -> Span {
        self.span
    }
}

pub struct Module {
    stmts: Rc<[Stmt]>,

    /// `export { .. }` — the mandatory tail statement of a module,
    /// declaring its public bindings with record syntax.
    export: Export,
}

impl Module {
    pub fn new(stmts: Rc<[Stmt]>, export: Export) -> Self {
        Module { stmts, export }
    }

    pub fn stmts(&self) -> &[Stmt] {
        &self.stmts
    }

    pub fn rc_stmts(&self) -> Rc<[Stmt]> {
        self.stmts.clone()
    }

    pub fn export(&self) -> &Export {
        &self.export
    }
}
