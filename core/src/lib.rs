//! Core Raft object model: `Val` and everything it is built from.
//!
//! Strictly `no_std` (`alloc` only) so it can be shared, unmodified, by
//! every execution mode — the tree-walking/bytecode `raft-runtime`, and
//! eventually transpiled-to-Rust modules (compiled into a `cdylib` or
//! bundled straight into the host binary via proc-macro).
//!
//! This crate cannot depend on a concrete runtime (that would recreate the
//! cycle `Val -> Runtime -> Val`), so anywhere `Val`-adjacent code needs to
//! call back into whatever is executing it — pushing/popping the operand
//! stack, reporting an error, dispatching a call — it goes through the
//! [`Host`] trait instead. `raft-runtime`'s `Runtime` implements `Host`;
//! other hosts (e.g. a transpiled module's own driver) can too.
#![no_std]

extern crate alloc;

use alloc::{rc::Rc, string::String, vec::Vec};

use core::{
    cell::RefCell,
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
};

use smallvec::SmallVec;

// ZST for fixed-state hash maps.
// This allows codegen to see
// that constant hashing see it used
// unlike storing a `foldhash::fast::FixedState` directly, which may have
// different internal state.
//
// This should optimize away hashing of constant keys at compile time.
// See assembly output of https://play.rust-lang.org/?version=stable&mode=release&edition=2024&gist=96867b416d6d26191223f2a7af37e320
#[derive(Clone, Copy, Debug, Default)]
pub struct FixedHashState;

impl core::hash::BuildHasher for FixedHashState {
    type Hasher = foldhash::fast::FoldHasher<'static>;

    #[inline(always)]
    fn build_hasher(&self) -> Self::Hasher {
        foldhash::fast::FixedState::default().build_hasher()
    }
}

pub type FixedHashMap<K, V> = hashbrown::HashMap<K, V, FixedHashState>;

#[derive(Copy, Clone)]
pub enum Number {
    Integer(i64),
    Float(f64),
}

impl Number {
    pub fn neg(self) -> Result<Number, RuntimeError> {
        match self {
            Number::Integer(i) => Ok(Number::Integer(i.wrapping_neg())),
            Number::Float(f) => Ok(Number::Float(-f)),
        }
    }

    pub fn add(self, rhs: Self) -> Result<Number, RuntimeError> {
        match (self, rhs) {
            (Number::Integer(i1), Number::Integer(i2)) => Ok(Number::Integer(i1.wrapping_add(i2))),
            (Number::Float(f1), Number::Float(f2)) => Ok(Number::Float(f1 + f2)),
            (Number::Integer(i), Number::Float(f)) => Ok(Number::Float((i as f64) + f)),
            (Number::Float(f), Number::Integer(i)) => Ok(Number::Float(f + (i as f64))),
        }
    }

    pub fn sub(self, rhs: Self) -> Result<Number, RuntimeError> {
        match (self, rhs) {
            (Number::Integer(i1), Number::Integer(i2)) => Ok(Number::Integer(i1.wrapping_sub(i2))),
            (Number::Float(f1), Number::Float(f2)) => Ok(Number::Float(f1 - f2)),
            (Number::Integer(i), Number::Float(f)) => Ok(Number::Float((i as f64) - f)),
            (Number::Float(f), Number::Integer(i)) => Ok(Number::Float(f - (i as f64))),
        }
    }

    pub fn mul(self, rhs: Self) -> Result<Number, RuntimeError> {
        match (self, rhs) {
            (Number::Integer(i1), Number::Integer(i2)) => Ok(Number::Integer(i1.wrapping_mul(i2))),
            (Number::Float(f1), Number::Float(f2)) => Ok(Number::Float(f1 * f2)),
            (Number::Integer(i), Number::Float(f)) => Ok(Number::Float((i as f64) * f)),
            (Number::Float(f), Number::Integer(i)) => Ok(Number::Float(f * (i as f64))),
        }
    }

    pub fn div(self, rhs: Self) -> Result<Number, RuntimeError> {
        match (self, rhs) {
            (Number::Integer(i1), Number::Integer(i2)) => {
                if i2 == 0 {
                    return Err(RuntimeError::Other("division by zero".into()));
                }
                Ok(Number::Integer(i1 / i2))
            }
            (Number::Float(f1), Number::Float(f2)) => Ok(Number::Float(f1 / f2)),
            (Number::Integer(i), Number::Float(f)) => Ok(Number::Float((i as f64) / f)),
            (Number::Float(f), Number::Integer(i)) => Ok(Number::Float(f / (i as f64))),
        }
    }

    pub fn pow(self, rhs: Self) -> Result<Number, RuntimeError> {
        match (self, rhs) {
            (Number::Integer(i1), Number::Integer(i2)) if i2 >= 0 => {
                Ok(Number::Integer(i1.wrapping_pow(i2 as u32)))
            }
            (Number::Integer(i1), Number::Integer(i2)) => {
                Ok(Number::Float(libm::pow(i1 as f64, i2 as f64)))
            }
            (Number::Float(f1), Number::Float(f2)) => Ok(Number::Float(libm::pow(f1, f2))),
            (Number::Integer(i), Number::Float(f)) => Ok(Number::Float(libm::pow(i as f64, f))),
            (Number::Float(f), Number::Integer(i)) => Ok(Number::Float(libm::pow(f, i as f64))),
        }
    }

    pub fn cmp(self, rhs: Self) -> Ordering {
        match (self, rhs) {
            (Number::Integer(i1), Number::Integer(i2)) => i1.cmp(&i2),
            (Number::Float(f1), Number::Float(f2)) => {
                f1.partial_cmp(&f2).unwrap_or(Ordering::Equal)
            }
            (Number::Integer(i), Number::Float(f)) => {
                (i as f64).partial_cmp(&f).unwrap_or(Ordering::Equal)
            }
            (Number::Float(f), Number::Integer(i)) => {
                f.partial_cmp(&(i as f64)).unwrap_or(Ordering::Equal)
            }
        }
    }
}

impl fmt::Debug for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Number::Integer(i) => write!(f, "{i}i"),
            Number::Float(fl) => write!(f, "{fl}f"),
        }
    }
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Number::Integer(i) => write!(f, "{i}"),
            Number::Float(fl) => write!(f, "{fl}"),
        }
    }
}

// Object kinds
#[derive(Debug)]
pub enum ObjectKind {
    List(Vec<Val>),
    Record(FixedHashMap<Rc<str>, Val>),
    /// An imported module's exported bindings. Record-shaped (field access
    /// and record patterns work on it) but immutable by construction.
    Module(FixedHashMap<Rc<str>, Val>),
}

#[derive(Debug)]
pub struct Object {
    pub kind: ObjectKind,
    /// cost flag prevents mutation when true
    pub frozen: bool,
    /// mutable by default
    pub mutable: bool,
}

impl fmt::Display for Object {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ObjectKind::List(elements) => {
                write!(f, "[")?;
                for (i, elem) in elements.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", elem)?;
                }
                write!(f, "]")
            }
            ObjectKind::Record(fields) => {
                write!(f, "{{")?;
                for (i, (key, value)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", key, value)?;
                }
                write!(f, "}}")
            }
            ObjectKind::Module(fields) => {
                write!(f, "module {{")?;
                for (i, (key, value)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", key, value)?;
                }
                write!(f, "}}")
            }
        }
    }
}

impl Object {
    pub fn new_list(elements: Vec<Val>) -> Self {
        Object {
            kind: ObjectKind::List(elements),
            frozen: false,
            mutable: true,
        }
    }

    pub fn new_record(fields: FixedHashMap<Rc<str>, Val>) -> Self {
        Object {
            kind: ObjectKind::Record(fields),
            frozen: false,
            mutable: true,
        }
    }

    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    pub fn cmp(&self, other: &Self) -> Option<Ordering> {
        match (&self.kind, &other.kind) {
            (ObjectKind::List(v1), ObjectKind::List(v2)) => {
                let mut v1iter = v1.iter();
                let mut v2iter = v2.iter();
                for (a, b) in v1iter.by_ref().zip(v2iter.by_ref()) {
                    let ord = a.cmp(b)?;
                    if ord != Ordering::Equal {
                        return Some(ord);
                    }
                }
                Some(v1.len().cmp(&v2.len()))
            }
            _ => None, // different kinds are considered incomparable
        }
    }

    pub fn get_field(&self, key: &str) -> Option<Val> {
        match &self.kind {
            ObjectKind::Record(fields) | ObjectKind::Module(fields) => fields.get(key).cloned(),
            _ => None,
        }
    }

    pub fn get_index(&self, index: usize) -> Option<Val> {
        match &self.kind {
            ObjectKind::List(elements) => elements.get(index).cloned(),
            _ => None,
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub enum Atom {
    Nil,
    True,
    False,
    Custom(Rc<str>),
}

impl PartialEq<str> for Atom {
    fn eq(&self, other: &str) -> bool {
        match self {
            Atom::Nil => other == "Nil",
            Atom::True => other == "True",
            Atom::False => other == "False",
            Atom::Custom(s) => &s[..] == other,
        }
    }
}

impl Hash for Atom {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Atom::Nil => "Nil".hash(state),
            Atom::True => "True".hash(state),
            Atom::False => "False".hash(state),
            Atom::Custom(s) => s[..].hash(state),
        }
    }
}

impl Atom {
    pub fn new(s: Rc<str>) -> Self {
        match &s[..] {
            "Nil" => Atom::Nil,
            "True" => Atom::True,
            "False" => Atom::False,
            _ => Atom::Custom(s),
        }
    }

    pub fn is_falsey(&self) -> bool {
        matches!(self, Atom::False | Atom::Nil)
    }

    pub fn is_false(&self) -> bool {
        matches!(self, Atom::False)
    }

    pub fn is_true(&self) -> bool {
        matches!(self, Atom::True)
    }

    pub fn is_nil(&self) -> bool {
        matches!(self, Atom::Nil)
    }
}

impl fmt::Debug for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Atom::Nil => write!(f, "Nil"),
            Atom::True => write!(f, "True"),
            Atom::False => write!(f, "False"),
            Atom::Custom(s) => write!(f, "{:?}", s),
        }
    }
}

impl fmt::Display for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Atom::Nil => write!(f, "Nil"),
            Atom::True => write!(f, "True"),
            Atom::False => write!(f, "False"),
            Atom::Custom(s) => write!(f, "{}", s),
        }
    }
}

// Value reference used by interpreter. Clone cheap for literals and Rc for heap objects.
#[derive(Clone)]
pub enum Val {
    Number(Number),
    Char(char),
    String(Rc<str>),
    Atom(Atom),                    // atoms like True/False or symbols
    Object(Rc<RefCell<Object>>),   // lists and records live here
    Fn(FnVal),                     // function value: fn-defined (AST or bytecode), partial, or host
    Opaque(Rc<dyn core::any::Any>), // opaque value, uninterpretable by raft code
    /// Internal sentinel: a local slot that has not been assigned yet
    /// (reads of it fall back to the global scope). Never observable from
    /// Raft code or host functions.
    #[doc(hidden)]
    Uninit,
}

impl core::cmp::PartialEq for Val {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Some(Ordering::Equal)
    }
}

impl fmt::Debug for Val {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            match self {
                Val::Number(n) => write!(f, "Number({:#?})", n),
                Val::Char(c) => write!(f, "Char({:#?})", c),
                Val::String(s) => write!(f, "String({:#?})", s),
                Val::Atom(a) => write!(f, "Atom({:#?})", a),
                Val::Object(o) => write!(f, "Object({:#?})", o.borrow()),
                Val::Fn(fun) => write!(f, "{:#?}", fun),
                Val::Opaque(val) => write!(f, "Opaque({:p})", &**val),
                Val::Uninit => write!(f, "<uninit>"),
            }
        } else {
            match self {
                Val::Number(n) => write!(f, "Number({:?})", n),
                Val::Char(c) => write!(f, "Char({:?})", c),
                Val::String(s) => write!(f, "String({:?})", s),
                Val::Atom(a) => write!(f, "Atom({:?})", a),
                Val::Object(o) => write!(f, "Object({:?})", o.borrow()),
                Val::Fn(fun) => write!(f, "{:?}", fun),
                Val::Opaque(val) => write!(f, "Opaque({:p})", &**val),
                Val::Uninit => write!(f, "<uninit>"),
            }
        }
    }
}

impl fmt::Display for Val {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Val::Number(n) => write!(f, "{}", n),
            Val::Char(c) => write!(f, "{}", c),
            Val::String(s) => write!(f, "{}", s),
            Val::Atom(a) => write!(f, "{}", a),
            Val::Object(o) => write!(f, "{}", o.borrow()),
            Val::Fn(_) => write!(f, "<fn>"),
            Val::Opaque(val) => write!(f, "{:p}", &**val),
            Val::Uninit => write!(f, "<uninit>"),
        }
    }
}

impl Val {
    #[inline]
    pub fn bool_(b: bool) -> Self {
        match b {
            true => Val::Atom(Atom::True),
            false => Val::Atom(Atom::False),
        }
    }

    #[inline]
    pub fn true_() -> Val {
        Val::Atom(Atom::True)
    }

    #[inline]
    pub fn false_() -> Val {
        Val::Atom(Atom::False)
    }

    #[inline]
    pub fn nil() -> Val {
        Val::Atom(Atom::Nil)
    }

    #[inline]
    #[doc(hidden)]
    pub fn is_init(&self) -> bool {
        !matches!(self, Val::Uninit)
    }

    #[inline]
    #[doc(hidden)]
    pub fn init_or<E>(self, err: E) -> Result<Val, E> {
        match self {
            Val::Uninit => Err(err),
            _ => Ok(self),
        }
    }

    #[inline]
    #[doc(hidden)]
    pub fn init_or_else<F, E>(self, f: F) -> Result<Val, E>
    where
        F: FnOnce() -> E,
    {
        match self {
            Val::Uninit => Err(f()),
            _ => Ok(self),
        }
    }

    #[inline]
    pub fn new_atom(s: Rc<str>) -> Val {
        Val::Atom(Atom::new(s))
    }

    #[inline]
    /// Wrap exported bindings into an immutable module object.
    pub fn new_module(fields: FixedHashMap<Rc<str>, Val>) -> Val {
        Val::Object(Rc::new(RefCell::new(Object {
            kind: ObjectKind::Module(fields),
            frozen: true,
            mutable: false,
        })))
    }

    #[inline]
    pub fn new_record(fields: FixedHashMap<Rc<str>, Val>) -> Val {
        Val::Object(Rc::new(RefCell::new(Object::new_record(fields))))
    }

    #[inline]
    pub fn new_list(elements: Vec<Val>) -> Val {
        Val::Object(Rc::new(RefCell::new(Object::new_list(elements))))
    }

    #[inline]
    pub fn pos(&self) -> Result<Val, RuntimeError> {
        match self {
            Val::Number(n) => Ok(Val::Number(*n)),
            _ => Err(RuntimeError::TypeError("pos on non-numeric value".into())),
        }
    }

    #[inline]
    pub fn neg(&self) -> Result<Val, RuntimeError> {
        match self {
            Val::Number(n) => Ok(Val::Number(n.neg()?)),
            _ => Err(RuntimeError::TypeError(
                "negation on non-numeric value".into(),
            )),
        }
    }

    #[inline]
    pub fn not(&self) -> Val {
        Val::bool_(is_falsey(self))
    }

    #[inline]
    pub fn bit_not(&self) -> Result<Val, RuntimeError> {
        match self {
            Val::Number(Number::Integer(i)) => Ok(Val::Number(Number::Integer(!i))),
            _ => Err(RuntimeError::TypeError(
                "bitwise not on non-integer value".into(),
            )),
        }
    }

    #[inline]
    pub fn add(&self, rhs: &Val) -> Result<Val, RuntimeError> {
        match (self, rhs) {
            (Val::Number(n1), Val::Number(n2)) => Ok(Val::Number(n1.add(*n2)?)),
            (Val::String(s1), Val::String(s2)) => {
                let mut s = String::new();
                s.push_str(&*s1);
                s.push_str(&*s2);
                Ok(Val::String(Rc::from(s)))
            }
            (Val::String(s1), Val::Char(c2)) => {
                let mut s = String::new();
                s.push_str(&*s1);
                s.push(*c2);
                Ok(Val::String(Rc::from(s)))
            }
            (Val::Char(c1), Val::String(s2)) => {
                let mut s = String::new();
                s.push(*c1);
                s.push_str(&*s2);
                Ok(Val::String(Rc::from(s)))
            }
            (Val::Char(c1), Val::Char(c2)) => {
                let mut s = String::new();
                s.push(*c1);
                s.push(*c2);
                Ok(Val::String(Rc::from(s)))
            }
            _ => Err(RuntimeError::TypeError(
                "addition on not numeric or string value".into(),
            )),
        }
    }

    #[inline]
    pub fn sub(&self, rhs: &Val) -> Result<Val, RuntimeError> {
        match (self, rhs) {
            (Val::Number(n1), Val::Number(n2)) => Ok(Val::Number(n1.sub(*n2)?)),
            _ => Err(RuntimeError::TypeError(
                "subtraction on non-numeric value".into(),
            )),
        }
    }

    #[inline]
    pub fn mul(&self, rhs: &Val) -> Result<Val, RuntimeError> {
        match (self, rhs) {
            (Val::Number(n1), Val::Number(n2)) => Ok(Val::Number(n1.mul(*n2)?)),
            _ => Err(RuntimeError::TypeError(
                "multiplication on non-numeric value".into(),
            )),
        }
    }

    #[inline]
    pub fn div(&self, rhs: &Val) -> Result<Val, RuntimeError> {
        match (self, rhs) {
            (Val::Number(n1), Val::Number(n2)) => Ok(Val::Number(n1.div(*n2)?)),
            _ => Err(RuntimeError::TypeError(
                "division on non-numeric value".into(),
            )),
        }
    }

    #[inline]
    pub fn pow(&self, rhs: &Val) -> Result<Val, RuntimeError> {
        match (self, rhs) {
            (Val::Number(n1), Val::Number(n2)) => Ok(Val::Number(n1.pow(*n2)?)),
            _ => Err(RuntimeError::TypeError(
                "exponentiation on non-numeric value".into(),
            )),
        }
    }

    #[inline]
    pub fn cmp(&self, rhs: &Val) -> Option<Ordering> {
        match (self, rhs) {
            (Val::Number(n1), Val::Number(n2)) => Some(n1.cmp(*n2)),
            (Val::Atom(a1), Val::Atom(a2)) => {
                if a1 == a2 {
                    Some(Ordering::Equal)
                } else {
                    None
                }
            }
            (Val::String(s1), Val::String(s2)) => Some(s1.cmp(s2)),
            (Val::Char(c1), Val::Char(c2)) => Some(c1.cmp(c2)),
            (Val::Object(o1), Val::Object(o2)) => o1.borrow().cmp(&o2.borrow()),
            (Val::Fn(e1), Val::Fn(e2)) => Rc::ptr_eq(&e1.0, &e2.0).then(|| Ordering::Equal),
            (Val::Opaque(o1), Val::Opaque(o2)) => {
                core::ptr::eq(Rc::as_ptr(o1), Rc::as_ptr(o2)).then(|| Ordering::Equal)
            }
            _ => None, // different kinds are considered incomparable
        }
    }

    #[inline]
    pub fn bit_and(&self, rhs: &Val) -> Result<Val, RuntimeError> {
        match (self, rhs) {
            (Val::Number(Number::Integer(i1)), Val::Number(Number::Integer(i2))) => {
                Ok(Val::Number(Number::Integer(i1 & i2)))
            }
            _ => Err(RuntimeError::TypeError(
                "bitwise and on non-integer value".into(),
            )),
        }
    }

    #[inline]
    pub fn bit_or(&self, rhs: &Val) -> Result<Val, RuntimeError> {
        match (self, rhs) {
            (Val::Number(Number::Integer(i1)), Val::Number(Number::Integer(i2))) => {
                Ok(Val::Number(Number::Integer(i1 | i2)))
            }
            _ => Err(RuntimeError::TypeError(
                "bitwise or on non-integer value".into(),
            )),
        }
    }

    #[inline]
    pub fn bit_xor(&self, rhs: &Val) -> Result<Val, RuntimeError> {
        match (self, rhs) {
            (Val::Number(Number::Integer(i1)), Val::Number(Number::Integer(i2))) => {
                Ok(Val::Number(Number::Integer(i1 ^ i2)))
            }
            _ => Err(RuntimeError::TypeError(
                "bitwise xor on non-integer value".into(),
            )),
        }
    }

    #[inline]
    pub fn shl(&self, rhs: &Val) -> Result<Val, RuntimeError> {
        match (self, rhs) {
            (Val::Number(Number::Integer(i1)), Val::Number(Number::Integer(i2))) => {
                if *i2 < 0 {
                    return Err(RuntimeError::TypeError(
                        "shift left by negative value".into(),
                    ));
                }
                Ok(Val::Number(Number::Integer(i1.wrapping_shl(*i2 as u32))))
            }
            _ => Err(RuntimeError::TypeError(
                "shift left on non-integer value".into(),
            )),
        }
    }

    #[inline]
    pub fn shr(&self, rhs: &Val) -> Result<Val, RuntimeError> {
        match (self, rhs) {
            (Val::Number(Number::Integer(i1)), Val::Number(Number::Integer(i2))) => {
                if *i2 < 0 {
                    return Err(RuntimeError::TypeError(
                        "shift right by negative value".into(),
                    ));
                }
                Ok(Val::Number(Number::Integer(i1.wrapping_shr(*i2 as u32))))
            }
            _ => Err(RuntimeError::TypeError(
                "shift right on non-integer value".into(),
            )),
        }
    }

    pub fn get_field(&self, key: &str) -> Option<Val> {
        match &self {
            Val::Object(o) => o.borrow().get_field(key),
            _ => None,
        }
    }

    pub fn get_index(&self, index: usize) -> Option<Val> {
        match &self {
            Val::Object(o) => o.borrow().get_index(index),
            _ => None,
        }
    }

    #[inline]
    pub fn iter(&self) -> Result<impl IntoIterator<Item = Val> + use<>, RuntimeError> {
        struct ObjectIter {
            object: Rc<RefCell<Object>>,
            pos: usize,
        }

        impl Iterator for ObjectIter {
            type Item = Val;

            #[inline]
            fn next(&mut self) -> Option<Val> {
                let object = self.object.borrow();

                match &object.kind {
                    ObjectKind::List(list) => {
                        if self.pos < list.len() {
                            let item = list[self.pos].clone();
                            self.pos += 1;
                            Some(item)
                        } else {
                            None
                        }
                    }
                    ObjectKind::Record(record) | ObjectKind::Module(record) => {
                        if self.pos < record.len() {
                            let key = record.keys().nth(self.pos).unwrap().clone();
                            let value = record.get(&key).unwrap().clone();
                            self.pos += 1;
                            Some(Val::new_record(core::iter::once((key, value)).collect()))
                        } else {
                            None
                        }
                    }
                }
            }
        }

        match self {
            Val::Object(o) => Ok(ObjectIter {
                object: o.clone(),
                pos: 0,
            }),
            _ => Err(RuntimeError::TypeError(
                "iteration on non-heap value".into(),
            )),
        }
    }

    /// Wrap a host closure into a function value with the given
    /// argument-count hint. `(0, None)` means "takes anything" — the
    /// closure then decides how many arguments to consume.
    #[inline]
    pub fn host_function<F>(min_args: usize, max_args: Option<usize>, f: F) -> Val
    where
        F: Fn(&mut dyn Host, usize) -> Val + 'static,
    {
        Val::Fn(FnVal::new(HostFn {
            min_args,
            max_args,
            fun: f,
        }))
    }
}

pub fn is_falsey(v: &Val) -> bool {
    match v {
        Val::Number(Number::Integer(0)) => true,
        Val::Number(Number::Float(f)) if *f == 0.0 => true,
        Val::Atom(a) => a.is_false(),
        Val::Object(o) => match &o.borrow().kind {
            ObjectKind::List(v) => v.is_empty(),
            ObjectKind::Record(m) | ObjectKind::Module(m) => m.is_empty(),
        },
        _ => false,
    }
}

#[derive(Clone, Debug)]
pub enum RuntimeError {
    UnboundIdentifier(Rc<str>),
    NotAFunction(Rc<str>),
    TypeError(Rc<str>),
    IndexError(Rc<str>),
    FieldError(Rc<str>),
    Other(Rc<str>),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeError::UnboundIdentifier(s) => write!(f, "Unbound identifier: {}", s),
            RuntimeError::NotAFunction(s) => write!(f, "Not a function: {}", s),
            RuntimeError::TypeError(s) => write!(f, "Type error: {}", s),
            RuntimeError::IndexError(s) => write!(f, "Index error: {}", s),
            RuntimeError::FieldError(s) => write!(f, "Field error: {}", s),
            RuntimeError::Other(s) => write!(f, "{}", s),
        }
    }
}

/// Object-safe bridge from `Val`-adjacent code back to whatever is actually
/// executing it. `raft-runtime`'s `Runtime` implements this; a hypothetical
/// transpiled-Rust module driver would too. `Function`/`DynFn`/`HostFn`/
/// `PartialFn` only ever see a `&mut dyn Host` — never a concrete runtime
/// type — which is what lets this crate stay independent of `raft-runtime`.
///
/// Implementations that need machinery this trait doesn't generalize (AST
/// walking, bytecode dispatch, module loading, ...) reach for
/// [`Host::as_any_mut`] and downcast back to their concrete type.
pub trait Host {
    fn stack_push(&mut self, v: Val);

    fn stack_pop(&mut self) -> Val;

    fn stack_len(&self) -> usize;

    /// Remove the top `out.len()` values and copy them into `out`, in the
    /// order they sit in the stack (oldest-of-the-drained-range first,
    /// most-recently-pushed last). No allocation — callers supply the
    /// buffer (typically an `Uninit`-filled `SmallVec`).
    fn stack_drain_top_into(&mut self, out: &mut [Val]);

    /// Push a batch of values in order (the first element ends up deepest),
    /// cloning each one out of `values`.
    fn stack_extend(&mut self, values: &[Val]);

    fn set_error(&mut self, err: RuntimeError);

    fn as_any_mut(&mut self) -> &mut dyn core::any::Any;
}

/// A callable value: `fn`-defined functions (AST-walked or compiled to
/// bytecode), partially-applied functions, and host-provided closures all
/// implement this.
///
/// The `Val` result is the return value; the runtime handles partial
/// application and argument consumption internally.
///
/// Callers must supply at least [`Function::min_args`] arguments — the
/// runtime handles partial application *before* calling (see
/// [`FnVal::partial`]), so implementations never see an underfull call.
///
/// This trait is deliberately **not** dyn-compatible: `call_once` takes
/// `self` by value, giving implementors natural move semantics with no
/// `Rc` plumbing. [`FnVal`] stores implementors through a hidden
/// dyn-compatible bridge trait instead (see `DynFn`).
pub trait Function: Sized + 'static {
    /// Minimum number of arguments this function consumes in a call.
    /// If less than that many are supplied, the runtime will return a partially-applied
    /// function value instead of calling.
    fn min_args(&self) -> usize;

    fn max_args(&self) -> Option<usize> {
        None
    }

    /// Call through a shared reference.
    fn call(&self, rt: &mut dyn Host, args: usize);

    /// Consuming flavor of [`call`](Function::call). The runtime dispatches
    /// here when the value being called holds the last reference to this
    /// function, so implementations can exploit unique ownership (move
    /// captured state instead of cloning it — see `PartialFn`). Defaults to
    /// delegating to `call`.
    #[inline]
    fn call_once(self, rt: &mut dyn Host, args: usize) {
        self.call(rt, args)
    }

    #[inline]
    fn call_rc(self: Rc<Self>, rt: &mut dyn Host, args: usize) {
        match Rc::try_unwrap(self) {
            Ok(f) => f.call_once(rt, args),
            Err(f) => f.call(rt, args),
        }
    }

    fn debug_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<fn>")
    }
}

/// Dyn-compatible bridge over [`Function`]. Blanket-implemented for every
/// `Function` implementor. Not sealed — `raft-runtime`'s AST-walked and
/// compiled function values implement it directly (they need machinery
/// this crate can't generalize), reaching `Host::as_any_mut` to get back
/// their concrete runtime. The by-value `call_once` can't live on a trait
/// object, so this recovers it behind `Rc` — unique ownership unwraps and
/// truly consumes, shared ownership falls back to the borrowing `call`.
/// That unwrap attempt doubles as the "is this the last reference?"
/// dispatch.
#[doc(hidden)]
pub trait DynFn: 'static {
    fn min_args(&self) -> usize;

    fn max_args(&self) -> Option<usize>;

    fn dyn_call(self: Rc<Self>, rt: &mut dyn Host, args: usize) -> usize;

    fn debug_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

fn dyn_call_impl<F: Function>(f: Rc<F>, rt: &mut dyn Host, args: usize) -> usize {
    let min_args = f.min_args();
    if args < min_args {
        core::hint::cold_path();

        if args == 0 {
            rt.stack_push(Val::Fn(FnVal(f)));
            return 0;
        }

        let partial = FnVal::partial(f, rt, args);
        rt.stack_push(Val::Fn(partial));
        return args;
    }

    let max_args = f.max_args();

    match max_args {
        Some(max_args) if max_args < args => {
            f.call_rc(rt, max_args);
            max_args
        }
        _ => {
            f.call_rc(rt, args);
            args
        }
    }
}

impl<F: Function> DynFn for F {
    #[inline]
    fn min_args(&self) -> usize {
        Function::min_args(self)
    }

    #[inline]
    fn max_args(&self) -> Option<usize> {
        Function::max_args(self)
    }

    #[inline]
    fn dyn_call(self: Rc<Self>, rt: &mut dyn Host, args: usize) -> usize {
        dyn_call_impl(self, rt, args)
    }

    #[inline]
    fn debug_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Function::debug_fmt(self, f)
    }
}

/// Adapter implementing [`Function`] for plain host closures. (A blanket
/// impl on `F: Fn(..)` would conflict with the concrete `Function` impls
/// under coherence rules, hence the newtype.)
struct HostFn<F> {
    min_args: usize,
    max_args: Option<usize>,
    fun: F,
}

impl<F> Function for HostFn<F>
where
    F: Fn(&mut dyn Host, usize) -> Val + 'static,
{
    #[inline]
    fn min_args(&self) -> usize {
        self.min_args
    }

    #[inline]
    fn max_args(&self) -> Option<usize> {
        self.max_args
    }

    #[inline]
    fn call(&self, rt: &mut dyn Host, args: usize) {
        debug_assert!(args >= self.min_args);
        let ret = match self.max_args {
            Some(max_args) if args > max_args => (self.fun)(rt, max_args),
            _ => (self.fun)(rt, args),
        };
        rt.stack_push(ret);
    }

    #[inline]
    fn call_rc(self: Rc<Self>, rt: &mut dyn Host, args: usize) {
        self.call(rt, args)
    }

    #[inline]
    fn debug_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.max_args {
            Some(max_args) => write!(f, "<fn {}..={}>", self.min_args, max_args),
            None => write!(f, "<fn {}..>", self.min_args),
        }
    }
}

/// A function value: a shared [`Function`] implementor together with a
/// hint of how many arguments a full application takes — at least
/// `min_args`, at most `max_args` (`None` = unbounded). The number a call
/// actually consumes is somewhere in between and is reported back by
/// [`Function::call`].
#[derive(Clone)]
pub struct FnVal(Rc<dyn DynFn>);

impl FnVal {
    #[inline]
    pub fn new(func: impl Function + 'static) -> Self {
        FnVal(Rc::new(func))
    }

    /// Raw data-pointer identity of the wrapped function, for callers that
    /// need to compare "is this the same function" without going through
    /// `dyn DynFn` fat-pointer comparison (unreliable — see call sites).
    #[inline]
    pub fn as_ptr(&self) -> *const () {
        Rc::as_ptr(&self.0) as *const ()
    }

    #[inline]
    #[doc(hidden)]
    pub fn new_dyn(func: impl DynFn + 'static) -> Self {
        FnVal(Rc::new(func))
    }

    /// Wrap an already-shared `Rc<T>` without allocating a new one (unlike
    /// [`new_dyn`](FnVal::new_dyn), which takes ownership of a fresh value).
    #[inline]
    #[doc(hidden)]
    pub fn from_rc<T: DynFn + 'static>(rc: Rc<T>) -> Self {
        FnVal(rc)
    }

    /// Dispatch a call: [`Function::call_once`] when this `FnVal` holds
    /// the last reference to the function, [`Function::call`] otherwise
    /// (the bridge's unwrap attempt makes that decision — cloning the
    /// `FnVal` first therefore naturally selects the shared flavor).
    #[inline]
    pub fn invoke(self, rt: &mut dyn Host, args: &mut usize) {
        let consumed = self.0.dyn_call(rt, *args);
        *args -= consumed;
    }

    /// Capture `args` (fewer than `min_args` of them) and return a function
    /// value awaiting the rest.
    #[inline]
    pub fn partial<F: Function>(fun: Rc<F>, rt: &mut dyn Host, args: usize) -> Self {
        FnVal::partial_dyn(fun, rt, args)
    }

    /// Capture `args` (fewer than `min_args` of them) and return a function
    /// value awaiting the rest.
    #[inline]
    #[doc(hidden)]
    pub fn partial_dyn<F: DynFn>(fun: Rc<F>, rt: &mut dyn Host, args: usize) -> Self {
        let mut preapplied: SmallVec<[Val; 4]> = smallvec::smallvec![Val::Uninit; args];
        rt.stack_drain_top_into(&mut preapplied);
        FnVal(Rc::new(PartialFn { fun, preapplied }))
    }
}

impl fmt::Debug for FnVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.debug_fmt(f)
    }
}

/// A function with some arguments already applied, waiting for the rest.
struct PartialFn<F> {
    fun: Rc<F>,
    preapplied: SmallVec<[Val; 4]>,
}

impl<F: DynFn> DynFn for PartialFn<F> {
    #[inline]
    fn min_args(&self) -> usize {
        self.fun.min_args().saturating_sub(self.preapplied.len())
    }

    #[inline]
    fn max_args(&self) -> Option<usize> {
        self.fun
            .max_args()
            .map(|max| max.saturating_sub(self.preapplied.len()))
    }

    fn dyn_call(mut self: Rc<Self>, rt: &mut dyn Host, args: usize) -> usize {
        if self.preapplied.len() + args < self.fun.min_args() {
            // still not enough: capture the new arguments too. `preapplied`
            // is kept in stack order — first argument LAST — so the newly
            // supplied (positionally later) arguments go to the front.
            let f = match Rc::get_mut(&mut self) {
                Some(f) => {
                    let mut newly: SmallVec<[Val; 4]> = smallvec::smallvec![Val::Uninit; args];
                    rt.stack_drain_top_into(&mut newly);
                    f.preapplied.insert_many(0, newly);
                    Val::Fn(FnVal(self))
                }
                None => Val::Fn(FnVal(Rc::new(PartialFn {
                    fun: self.fun.clone(),
                    preapplied: {
                        let mut new_preapplied: SmallVec<[Val; 4]> =
                            smallvec::smallvec![Val::Uninit; args];
                        rt.stack_drain_top_into(&mut new_preapplied);
                        new_preapplied.extend(self.preapplied.iter().cloned());
                        new_preapplied
                    },
                }))),
            };

            rt.stack_push(f);
            args
        } else {
            core::hint::cold_path();

            let pre_args = self.preapplied.len();
            let total_args = pre_args + args;
            match self.fun.max_args() {
                Some(max_args) if max_args < total_args => match Rc::try_unwrap(self) {
                    Ok(me) => {
                        rt.stack_extend(&me.preapplied);
                        let consumed = me.fun.dyn_call(rt, max_args);
                        consumed - pre_args
                    }
                    Err(me) => {
                        rt.stack_extend(&me.preapplied);
                        let consumed = me.fun.clone().dyn_call(rt, max_args);
                        consumed - pre_args
                    }
                },
                _ => match Rc::try_unwrap(self) {
                    Ok(me) => {
                        rt.stack_extend(&me.preapplied);
                        let consumed = me.fun.dyn_call(rt, total_args);
                        consumed - pre_args
                    }
                    Err(me) => {
                        rt.stack_extend(&me.preapplied);
                        let consumed = me.fun.clone().dyn_call(rt, total_args);
                        consumed - pre_args
                    }
                },
            }
        }
    }

    fn debug_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<partial ")?;
        for arg in self.preapplied.iter() {
            write!(f, "{:?} ", arg)?;
        }
        self.fun.debug_fmt(f)?;
        write!(f, ">")
    }
}
