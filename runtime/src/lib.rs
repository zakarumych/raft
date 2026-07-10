#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::{collections::BTreeMap, rc::Rc, vec::Vec};
use core::{cell::RefCell, cmp::Ordering, fmt};
use smallvec::SmallVec;
use std::{collections::HashMap, hash::{Hash, Hasher}};

use raft_ast::{BinOpKind, Expr, ExprKind, Lit, LitNum, Pat, PatKind, Stmt, StmtKind, UnOpKind};

pub mod vm;

#[derive(Copy, Clone)]
pub enum Number {
    Integer(i64),
    Float(f64),
}

impl Number {
    fn neg(self) -> Result<Number, RuntimeError> {
        match self {
            Number::Integer(i) => Ok(Number::Integer(i.wrapping_neg())),
            Number::Float(f) => Ok(Number::Float(-f)),
        }
    }

    fn add(self, rhs: Self) -> Result<Number, RuntimeError> {
        match (self, rhs) {
            (Number::Integer(i1), Number::Integer(i2)) => Ok(Number::Integer(i1.wrapping_add(i2))),
            (Number::Float(f1), Number::Float(f2)) => Ok(Number::Float(f1 + f2)),
            (Number::Integer(i), Number::Float(f)) => Ok(Number::Float((i as f64) + f)),
            (Number::Float(f), Number::Integer(i)) => Ok(Number::Float(f + (i as f64))),
        }
    }

    fn sub(self, rhs: Self) -> Result<Number, RuntimeError> {
        match (self, rhs) {
            (Number::Integer(i1), Number::Integer(i2)) => Ok(Number::Integer(i1.wrapping_sub(i2))),
            (Number::Float(f1), Number::Float(f2)) => Ok(Number::Float(f1 - f2)),
            (Number::Integer(i), Number::Float(f)) => Ok(Number::Float((i as f64) - f)),
            (Number::Float(f), Number::Integer(i)) => Ok(Number::Float(f - (i as f64))),
        }
    }

    fn mul(self, rhs: Self) -> Result<Number, RuntimeError> {
        match (self, rhs) {
            (Number::Integer(i1), Number::Integer(i2)) => Ok(Number::Integer(i1.wrapping_mul(i2))),
            (Number::Float(f1), Number::Float(f2)) => Ok(Number::Float(f1 * f2)),
            (Number::Integer(i), Number::Float(f)) => Ok(Number::Float((i as f64) * f)),
            (Number::Float(f), Number::Integer(i)) => Ok(Number::Float(f * (i as f64))),
        }
    }

    fn div(self, rhs: Self) -> Result<Number, RuntimeError> {
        match (self, rhs) {
            (Number::Integer(i1), Number::Integer(i2)) => {
                if i2 == 0 {
                    return Err(RuntimeError::Other("division by zero".to_owned()));
                }
                Ok(Number::Integer(i1 / i2))
            }
            (Number::Float(f1), Number::Float(f2)) => Ok(Number::Float(f1 / f2)),
            (Number::Integer(i), Number::Float(f)) => Ok(Number::Float((i as f64) / f)),
            (Number::Float(f), Number::Integer(i)) => Ok(Number::Float(f / (i as f64))),
        }
    }

    fn pow(self, rhs: Self) -> Result<Number, RuntimeError> {
        match (self, rhs) {
            (Number::Integer(i1), Number::Integer(i2)) if i2 >= 0 => {
                Ok(Number::Integer(i1.wrapping_pow(i2 as u32)))
            }
            (Number::Integer(i1), Number::Integer(i2)) => {
                Ok(Number::Float((i1 as f64).powf(i2 as f64)))
            }
            (Number::Float(f1), Number::Float(f2)) => Ok(Number::Float(f1.powf(f2))),
            (Number::Integer(i), Number::Float(f)) => Ok(Number::Float((i as f64).powf(f))),
            (Number::Float(f), Number::Integer(i)) => Ok(Number::Float(f.powi(i as i32))),
        }
    }

    fn cmp(self, rhs: Self) -> Ordering {
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
    List(Vec<Any>),
    Record(BTreeMap<Rc<str>, Any>),
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
        }
    }
}

impl Object {
    pub fn new_list(elements: Vec<Any>) -> Self {
        Object {
            kind: ObjectKind::List(elements),
            frozen: false,
            mutable: true,
        }
    }

    pub fn new_record(fields: BTreeMap<Rc<str>, Any>) -> Self {
        Object {
            kind: ObjectKind::Record(fields),
            frozen: false,
            mutable: true,
        }
    }

    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    fn cmp(&self, other: &Self) -> Option<Ordering> {
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
pub enum Any {
    Number(Number),
    Char(char),
    String(Rc<str>),
    Atom(Atom),                 // atoms like True/False or symbols
    Object(Rc<RefCell<Object>>),   // lists and records live here
    Fn(FnValue),                   // function value: fn-defined (AST or bytecode), partial, or host
    Opaque(Rc<dyn std::any::Any>), // opaque value, uninterpretable by raft code
    /// Internal sentinel: a local slot that has not been assigned yet
    /// (reads of it fall back to the global scope). Never observable from
    /// Raft code or host functions.
    Uninit,
}

impl core::cmp::PartialEq for Any {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Some(Ordering::Equal)
    }
}

impl fmt::Debug for Any {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Any::Number(n) => write!(f, "Number({:?})", n),
            Any::Char(c) => write!(f, "Char({:?})", c),
            Any::String(s) => write!(f, "String({:?})", s),
            Any::Atom(a) => write!(f, "Atom({:?})", a),
            Any::Object(o) => write!(f, "Object({:?})", o.borrow()),
            Any::Fn(_) => write!(f, "<fn>"),
            Any::Opaque(val) => write!(f, "Opaque({:p})", &**val),
            Any::Uninit => write!(f, "<uninit>"),
        }
    }
}

impl fmt::Display for Any {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Any::Number(n) => write!(f, "{}", n),
            Any::Char(c) => write!(f, "{}", c),
            Any::String(s) => write!(f, "{}", s),
            Any::Atom(a) => write!(f, "{}", a),
            Any::Object(o) => write!(f, "{}", o.borrow()),
            Any::Fn(_) => write!(f, "<fn>"),
            Any::Opaque(val) => write!(f, "{:p}", &**val),
            Any::Uninit => write!(f, "<uninit>"),
        }
    }
}

impl Any {
    #[inline]
    pub fn bool_(b: bool) -> Self {
        match b {
            true => Any::Atom(Atom::True),
            false => Any::Atom(Atom::False),
        }
    }

    #[inline]
    pub fn true_() -> Any {
        Any::Atom(Atom::True)
    }

    #[inline]
    pub fn false_() -> Any {
        Any::Atom(Atom::False)
    }

    #[inline]
    pub fn nil() -> Any {
        Any::Atom(Atom::Nil)
    }

    #[cold]
    #[inline]
    fn cold_nil() -> Any {
        Any::nil()
    }

    pub fn new_atom(s: Rc<str>) -> Any {
        Any::Atom(Atom::new(s))
    }

    #[inline]
    pub fn new_record(fields: BTreeMap<Rc<str>, Any>) -> Any {
        Any::Object(Rc::new(RefCell::new(Object::new_record(fields))))
    }

    #[inline]
    pub fn new_list(elements: Vec<Any>) -> Any {
        Any::Object(Rc::new(RefCell::new(Object::new_list(elements))))
    }

    #[inline]
    fn pos(&self) -> Result<Any, RuntimeError> {
        match self {
            Any::Number(n) => Ok(Any::Number(*n)),
            _ => Err(RuntimeError::TypeError("pos on non-numeric value".into())),
        }
    }

    #[inline]
    fn neg(&self) -> Result<Any, RuntimeError> {
        match self {
            Any::Number(n) => Ok(Any::Number(n.neg()?)),
            _ => Err(RuntimeError::TypeError(
                "negation on non-numeric value".into(),
            )),
        }
    }

    #[inline]
    fn not(&self) -> Any {
        Any::bool_(is_falsey(self))
    }

    #[inline]
    fn bit_not(&self) -> Result<Any, RuntimeError> {
        match self {
            Any::Number(Number::Integer(i)) => Ok(Any::Number(Number::Integer(!i))),
            _ => Err(RuntimeError::TypeError(
                "bitwise not on non-integer value".into(),
            )),
        }
    }

    #[inline]
    fn add(&self, rhs: &Any) -> Result<Any, RuntimeError> {
        match (self, rhs) {
            (Any::Number(n1), Any::Number(n2)) => Ok(Any::Number(n1.add(*n2)?)),
            (Any::String(s1), Any::String(s2)) => {
                let mut s = String::new();
                s.push_str(&*s1);
                s.push_str(&*s2);
                Ok(Any::String(Rc::from(s)))
            }
            (Any::String(s1), Any::Char(c2)) => {
                let mut s = String::new();
                s.push_str(&*s1);
                s.push(*c2);
                Ok(Any::String(Rc::from(s)))
            }
            (Any::Char(c1), Any::String(s2)) => {
                let mut s = String::new();
                s.push(*c1);
                s.push_str(&*s2);
                Ok(Any::String(Rc::from(s)))
            }
            (Any::Char(c1), Any::Char(c2)) => {
                let mut s = String::new();
                s.push(*c1);
                s.push(*c2);
                Ok(Any::String(Rc::from(s)))
            }
            _ => Err(RuntimeError::TypeError(
                "addition on not numeric or string value".into(),
            )),
        }
    }

    #[inline]
    fn sub(&self, rhs: &Any) -> Result<Any, RuntimeError> {
        match (self, rhs) {
            (Any::Number(n1), Any::Number(n2)) => Ok(Any::Number(n1.sub(*n2)?)),
            _ => Err(RuntimeError::TypeError(
                "subtraction on non-numeric value".into(),
            )),
        }
    }

    #[inline]
    fn mul(&self, rhs: &Any) -> Result<Any, RuntimeError> {
        match (self, rhs) {
            (Any::Number(n1), Any::Number(n2)) => Ok(Any::Number(n1.mul(*n2)?)),
            _ => Err(RuntimeError::TypeError(
                "multiplication on non-numeric value".into(),
            )),
        }
    }

    #[inline]
    fn div(&self, rhs: &Any) -> Result<Any, RuntimeError> {
        match (self, rhs) {
            (Any::Number(n1), Any::Number(n2)) => Ok(Any::Number(n1.div(*n2)?)),
            _ => Err(RuntimeError::TypeError(
                "division on non-numeric value".into(),
            )),
        }
    }

    #[inline]
    fn pow(&self, rhs: &Any) -> Result<Any, RuntimeError> {
        match (self, rhs) {
            (Any::Number(n1), Any::Number(n2)) => Ok(Any::Number(n1.pow(*n2)?)),
            _ => Err(RuntimeError::TypeError(
                "exponentiation on non-numeric value".into(),
            )),
        }
    }

    #[inline]
    fn cmp(&self, rhs: &Any) -> Option<Ordering> {
        match (self, rhs) {
            (Any::Number(n1), Any::Number(n2)) => Some(n1.cmp(*n2)),
            (Any::Atom(a1), Any::Atom(a2)) => {
                if a1 == a2 {
                    Some(Ordering::Equal)
                } else {
                    None
                }
            }
            (Any::String(s1), Any::String(s2)) => Some(s1.cmp(s2)),
            (Any::Char(c1), Any::Char(c2)) => Some(c1.cmp(c2)),
            (Any::Object(o1), Any::Object(o2)) => o1.borrow().cmp(&o2.borrow()),
            (Any::Fn(e1), Any::Fn(e2)) => Rc::ptr_eq(&e1.0, &e2.0).then(|| Ordering::Equal),
            (Any::Opaque(o1), Any::Opaque(o2)) => {
                std::ptr::eq(Rc::as_ptr(o1), Rc::as_ptr(o2)).then(|| Ordering::Equal)
            }
            _ => None, // different kinds are considered incomparable
        }
    }

    #[inline]
    fn bit_and(&self, rhs: &Any) -> Result<Any, RuntimeError> {
        match (self, rhs) {
            (Any::Number(Number::Integer(i1)), Any::Number(Number::Integer(i2))) => {
                Ok(Any::Number(Number::Integer(i1 & i2)))
            }
            _ => Err(RuntimeError::TypeError(
                "bitwise and on non-integer value".into(),
            )),
        }
    }

    #[inline]
    fn bit_or(&self, rhs: &Any) -> Result<Any, RuntimeError> {
        match (self, rhs) {
            (Any::Number(Number::Integer(i1)), Any::Number(Number::Integer(i2))) => {
                Ok(Any::Number(Number::Integer(i1 | i2)))
            }
            _ => Err(RuntimeError::TypeError(
                "bitwise or on non-integer value".into(),
            )),
        }
    }

    #[inline]
    fn bit_xor(&self, rhs: &Any) -> Result<Any, RuntimeError> {
        match (self, rhs) {
            (Any::Number(Number::Integer(i1)), Any::Number(Number::Integer(i2))) => {
                Ok(Any::Number(Number::Integer(i1 ^ i2)))
            }
            _ => Err(RuntimeError::TypeError(
                "bitwise xor on non-integer value".into(),
            )),
        }
    }

    #[inline]
    fn shl(&self, rhs: &Any) -> Result<Any, RuntimeError> {
        match (self, rhs) {
            (Any::Number(Number::Integer(i1)), Any::Number(Number::Integer(i2))) => {
                if *i2 < 0 {
                    return Err(RuntimeError::TypeError(
                        "shift left by negative value".into(),
                    ));
                }
                Ok(Any::Number(Number::Integer(i1.wrapping_shl(*i2 as u32))))
            }
            _ => Err(RuntimeError::TypeError(
                "shift left on non-integer value".into(),
            )),
        }
    }

    #[inline]
    fn shr(&self, rhs: &Any) -> Result<Any, RuntimeError> {
        match (self, rhs) {
            (Any::Number(Number::Integer(i1)), Any::Number(Number::Integer(i2))) => {
                if *i2 < 0 {
                    return Err(RuntimeError::TypeError(
                        "shift right by negative value".into(),
                    ));
                }
                Ok(Any::Number(Number::Integer(i1.wrapping_shr(*i2 as u32))))
            }
            _ => Err(RuntimeError::TypeError(
                "shift right on non-integer value".into(),
            )),
        }
    }

    // fn le(&self, rhs: &Any) -> Result<Any, RuntimeError> {
    //     match (self, rhs) {
    //         (Any::Number(n1), Any::Number(n2)) => Ok(Any::bool_(n1.cmp(*n2) != Ordering::Greater)),
    //         _ => Err(RuntimeError::TypeError(
    //             "comparison on non-numeric value".into(),
    //         )),
    //     }
    // }

    // fn ge(&self, rhs: &Any) -> Result<Any, RuntimeError> {
    //     match (self, rhs) {
    //         (Any::Number(n1), Any::Number(n2)) => Ok(Any::bool_(n1.cmp(*n2) != Ordering::Less)),
    //         _ => Err(RuntimeError::TypeError(
    //             "comparison on non-numeric value".into(),
    //         )),
    //     }
    // }

    // fn lt(&self, rhs: &Any) -> Result<Any, RuntimeError> {
    //     match (self, rhs) {
    //         (Any::Number(n1), Any::Number(n2)) => Ok(Any::bool_(n1.cmp(*n2) == Ordering::Less)),
    //         _ => Err(RuntimeError::TypeError(
    //             "comparison on non-numeric value".into(),
    //         )),
    //     }
    // }

    // fn gt(&self, rhs: &Any) -> Result<Any, RuntimeError> {
    //     match (self, rhs) {
    //         (Any::Number(n1), Any::Number(n2)) => Ok(Any::bool_(n1.cmp(*n2) == Ordering::Greater)),
    //         _ => Err(RuntimeError::TypeError(
    //             "comparison on non-numeric value".into(),
    //         )),
    //     }
    // }

    #[inline]
    fn iter(&self) -> Result<impl IntoIterator<Item = Any> + use<>, RuntimeError> {
        struct ObjectIter {
            object: Rc<RefCell<Object>>,
            pos: usize,
        }

        impl Iterator for ObjectIter {
            type Item = Any;

            #[inline]
            fn next(&mut self) -> Option<Any> {
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
                    ObjectKind::Record(record) => {
                        if self.pos < record.len() {
                            let key = record.keys().nth(self.pos).unwrap().clone();
                            let value = record.get(&key).unwrap().clone();
                            self.pos += 1;
                            Some(Any::new_record(BTreeMap::from([(key, value)])))
                        } else {
                            None
                        }
                    }
                }
            }
        }

        match self {
            Any::Object(o) => Ok(ObjectIter {
                object: o.clone(),
                pos: 0,
            }),
            _ => Err(RuntimeError::TypeError(
                "iteration on non-heap value".into(),
            )),
        }
    }

    #[inline]
    fn fn_from_ast(params: Rc<[Pat]>, body: Rc<[Stmt]>) -> Any {
        Any::Fn(FnValue(Rc::new(AstFn { params, body })))
    }

    /// Wrap a host closure into a function value with the given
    /// argument-count hint. `(0, None)` means "takes anything" — the closure
    /// then decides how many arguments to consume, as
    /// [`Runtime::register_external`] assumes.
    #[inline]
    pub fn host_function<F>(min_args: usize, max_args: Option<usize>, f: F) -> Any
    where
        F: Fn(&mut Runtime, usize) -> Any + 'static,
    {
        Any::Fn(FnValue::new(HostFn {
            min_args,
            max_args,
            fun: f,
        }))
    }
}

/// A callable value: `fn`-defined functions (AST-walked or compiled to
/// bytecode), partially-applied functions, and host-provided closures all
/// implement this.
///
/// The `Any` result is the return value; the runtime handles partial application
/// and argument consumption internally.
/// runtime re-apply the leftovers to the returned value.
///
/// Callers must supply at least [`FnValue::min_args`] arguments — the
/// runtime handles partial application *before* calling (see
/// [`FnValue::partial`]), so implementations never see an underfull call.
///
/// This trait is deliberately **not** dyn-compatible: `call_once` takes
/// `self` by value, giving implementors natural move semantics with no
/// `Rc` plumbing. [`FnValue`] stores implementors through a hidden
/// dyn-compatible bridge trait instead (see `DynFunction`).
pub trait Function: Sized + 'static {
    /// Minimum number of arguments this function consumes in a call.
    /// If less than that many are supplied, the runtime will return a partially-applied
    /// function value instead of calling.
    fn min_args(&self) -> usize;

    fn max_args(&self) -> Option<usize> {
        None
    }

    /// Call through a shared reference.
    fn call(&self, rt: &mut Runtime, args: usize) -> Any;

    /// Consuming flavor of [`call`](Function::call). The runtime dispatches
    /// here when the value being called holds the last reference to this
    /// function, so implementations can exploit unique ownership (move
    /// captured state instead of cloning it — see `PartialFn`). Defaults to
    /// delegating to `call`.
    #[inline]
    fn call_once(self, rt: &mut Runtime, args: usize) -> Any {
        self.call(rt, args)
    }

    /// Whether the runtime must push a fresh name-keyed local scope around
    /// calls. Functions that manage their own locals (compiled functions
    /// use stack-frame slots) return `false` and skip that work; they must
    /// then never touch `Runtime::local`.
    #[inline]
    fn wants_local_scope(&self) -> bool {
        true
    }
}

/// Dyn-compatible bridge over [`Function`]. Blanket-implemented for every
/// `Function` implementor and never exposed publicly: the by-value
/// `call_once` can't live on a trait object, so this recovers it behind
/// `Rc` — unique ownership unwraps and truly consumes, shared ownership
/// falls back to the borrowing `call`. That unwrap attempt doubles as the
/// "is this the last reference?" dispatch.
trait DynFunction: 'static {
    fn dyn_call_once(self: Rc<Self>, rt: &mut Runtime, args: usize) -> (Any, usize);
}

impl<F: Function> DynFunction for F {
    fn dyn_call_once(self: Rc<Self>, rt: &mut Runtime, args: usize) -> (Any, usize) {
        let min_args = self.min_args();
        if args < min_args {
            if args == 0 {
                return (Any::Fn(FnValue(self)), 0);
            }
            return (Any::Fn(FnValue::partial(self, rt, args)), args);
        }

        let scoped = self.wants_local_scope();
        let invoke = move |rt: &mut Runtime| {
            let max_args = self.max_args();

            match max_args {
                Some(max_args) if max_args < args => match Rc::try_unwrap(self) {
                    Ok(f) => (f.call_once(rt, max_args), max_args),
                    Err(shared) => (shared.call(rt, max_args), max_args),
                },
                _ => match Rc::try_unwrap(self) {
                    Ok(f) => (f.call_once(rt, args), args),
                    Err(shared) => (shared.call(rt, args), args),
                },
            }
        };

        if scoped {
            rt.call_with_local(invoke)
        } else {
            invoke(rt)
        }
    }
}

// impl<F: Function> Function for Rc<F> {
//     fn min_args(&self) -> usize {
//         (**self).min_args()
//     }

//     fn call(&self, rt: &mut Runtime, args: usize) -> Any {
//         (**self).call(rt, args)
//     }

//     fn call_once(self, rt: &mut Runtime, args: usize) -> Any {
//         match Rc::try_unwrap(self) {
//             Ok(f) => f.call_once(rt, args),
//             Err(shared) => shared.call(rt, args),
//         }
//     }
// }

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
    F: Fn(&mut Runtime, usize) -> Any + 'static,
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
    fn call(&self, rt: &mut Runtime, args: usize) -> Any {
        debug_assert!(args >= self.min_args);
        match self.max_args {
            Some(max_args) if args > max_args => (self.fun)(rt, max_args),
            _ => (self.fun)(rt, args),
        }
    }
}

/// A function value: a shared [`Function`] implementor together with a
/// hint of how many arguments a full application takes — at least
/// `min_args`, at most `max_args` (`None` = unbounded). The number a call
/// actually consumes is somewhere in between and is reported back by
/// [`Function::call`].
#[derive(Clone)]
pub struct FnValue(Rc<dyn DynFunction>);

impl FnValue {
    pub fn new(func: impl Function + 'static) -> Self {
        FnValue(Rc::new(func))
    }

    /// Dispatch a call: [`Function::call_once`] when this `FnValue` holds
    /// the last reference to the function, [`Function::call`] otherwise
    /// (the bridge's unwrap attempt makes that decision — cloning the
    /// `FnValue` first therefore naturally selects the shared flavor).
    fn invoke(self, rt: &mut Runtime, args: &mut usize) -> Any {
        let (val, consumed) = self.0.dyn_call_once(rt, *args);
        *args -= consumed;
        val
    }

    /// Capture `args` (fewer than `min_args` of them) and return a function
    /// value awaiting the rest.
    pub fn partial<F: Function>(fun: Rc<F>, runtime: &mut Runtime, args: usize) -> Self {
        FnValue(Rc::new(PartialFn {
            fun,
            preapplied: runtime.vm.drain_off_stack(args).collect(),
        }))
    }
}

/// A function with some arguments already applied, waiting for the rest.
struct PartialFn<F> {
    fun: Rc<F>,
    preapplied: SmallVec<[Any; 4]>,
}

impl<F: Function> PartialFn<F> {}

impl<F: Function> DynFunction for PartialFn<F> {
    fn dyn_call_once(mut self: Rc<Self>, rt: &mut Runtime, args: usize) -> (Any, usize) {
        if self.preapplied.len() + args < self.fun.min_args() {
            // still not enough: capture the new arguments too. `preapplied`
            // is kept in stack order — first argument LAST — so the newly
            // supplied (positionally later) arguments go to the front.
            let f = match Rc::get_mut(&mut self) {
                Some(f) => {
                    let newly: SmallVec<[Any; 4]> = rt.vm.drain_off_stack(args).collect();
                    f.preapplied.insert_many(0, newly);
                    Any::Fn(FnValue(self))
                }
                None => Any::Fn(FnValue(Rc::new(PartialFn {
                    fun: self.fun.clone(),
                    preapplied: {
                        let mut new_preapplied: SmallVec<[Any; 4]> =
                            rt.vm.drain_off_stack(args).collect();
                        new_preapplied.extend(self.preapplied.iter().cloned());
                        new_preapplied
                    },
                }))),
            };
            (f, args)
        } else {
            let scoped = self.fun.wants_local_scope();
            let invoke = move |rt: &mut Runtime| {
                let pre_args = self.preapplied.len();
                let total_args = pre_args + args;
                match self.fun.max_args() {
                    Some(max_args) if max_args < total_args => match Rc::try_unwrap(self) {
                        Ok(me) => {
                            rt.vm.extend_stack(me.preapplied);
                            let val = match Rc::try_unwrap(me.fun) {
                                Ok(f) => f.call_once(rt, max_args),
                                Err(shared) => shared.call(rt, max_args),
                            };
                            (val, max_args - pre_args)
                        }
                        Err(me) => {
                            rt.vm.extend_stack(me.preapplied.iter().cloned());
                            (me.fun.call(rt, max_args), max_args - pre_args)
                        }
                    },
                    _ => match Rc::try_unwrap(self) {
                        Ok(me) => {
                            rt.vm.extend_stack(me.preapplied);
                            let val = match Rc::try_unwrap(me.fun) {
                                Ok(f) => f.call_once(rt, total_args),
                                Err(shared) => shared.call(rt, total_args),
                            };
                            (val, args)
                        }
                        Err(me) => {
                            rt.vm.extend_stack(me.preapplied.iter().cloned());
                            (me.fun.call(rt, total_args), args)
                        }
                    },
                }
            };

            if scoped {
                rt.call_with_local(invoke)
            } else {
                invoke(rt)
            }
        }
    }
}

/// An `fn`-defined function executed by walking its AST body.
struct AstFn {
    params: Rc<[Pat]>,
    body: Rc<[Stmt]>,
}

impl Function for AstFn {
    fn min_args(&self) -> usize {
        self.params.len()
    }

    fn max_args(&self) -> Option<usize> {
        // consumes exactly its parameter count; the runtime clamps
        // over-application and re-applies the leftovers to the result
        Some(self.params.len())
    }

    fn call(&self, rt: &mut Runtime, args: usize) -> Any {
        debug_assert!(rt.vm.stack_len() >= args);
        debug_assert_eq!(args, self.params.len());

        // first argument is on top of the stack
        for param in self.params.iter() {
            let arg = rt.vm.pop_stack();
            if let Err(e) = rt.bind_pattern(param, &arg) {
                rt.set_error(e);
                return Any::nil();
            }
        }

        match rt.exec_block(&self.body) {
            Ok(Exec::Value(v)) => v,
            Ok(Exec::Return(v)) => v,
            Ok(Exec::Break) => {
                rt.set_error(RuntimeError::Other(
                    "break statement outside of loop".to_owned(),
                ));
                Any::nil()
            }
            Ok(Exec::Continue) => {
                rt.set_error(RuntimeError::Other(
                    "continue statement outside of loop".to_owned(),
                ));
                Any::nil()
            }
            Err(e) => {
                rt.set_error(e);
                Any::nil()
            }
        }
    }
}

// Runtime with two scopes: global and optional local
pub struct Runtime {
    pub global: HashMap<Rc<str>, Any>,
    pub local: Option<SmallVec<[(Rc<str>, Any); 16]>>,
    pub status: Result<(), RuntimeError>,
    /// Shared bytecode context: interned constant/name/pattern pools that
    /// all compiled functions index into, plus the operand stack their
    /// frames execute on (`vm.stack` is public — peek at it from host
    /// functions for the fun of it).
    pub vm: vm::VmContext,
    /// When true, `fn` statements are compiled to stack-based bytecode
    /// (see [`vm`]) instead of being closed over as AST. Both kinds of
    /// function are plain `Any::Fn` values and can call each other freely.
    compile_fns: bool,
}

#[derive(Clone, Debug)]
pub enum RuntimeError {
    UnboundIdentifier(String),
    NotAFunction(String),
    TypeError(String),
    IndexError(String),
    FieldError(String),
    Other(String),
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

#[derive(Clone, Debug, PartialEq)]
pub enum Exec {
    /// Stmt executed successfully, no control flow change.
    Value(Any),
    /// Return statement encountered.
    Return(Any),
    /// Continue statement encountered.
    Continue,
    /// Break statement encountered.
    Break,
}

impl Runtime {
    pub fn new() -> Self {
        Runtime {
            global: HashMap::new(),
            local: None,
            status: Ok(()),
            vm: vm::VmContext::new(),
            compile_fns: false,
        }
    }

    /// Choose how `fn` statements executed from here on are realized:
    /// `true` compiles them to bytecode run by [`vm::run`], `false` (the
    /// default) keeps the tree-walking closure. The modes mix freely within
    /// one runtime — functions defined either way call each other through
    /// the same `Any::Fn` interface.
    pub fn set_compile_fns(&mut self, enabled: bool) {
        self.compile_fns = enabled;
    }

    pub fn compile_fns(&self) -> bool {
        self.compile_fns
    }

    pub fn set_error(&mut self, err: RuntimeError) {
        self.status = Err(err);
    }

    /// Register a host function in the global scope.
    /// The closure is called with the runtime and the number of arguments it was given;
    /// it can then pop them off the stack and return a value.
    pub fn register_function<F>(
        &mut self,
        name: &str,
        min_args: usize,
        max_args: Option<usize>,
        f: F,
    ) where
        F: Fn(&mut Runtime, usize) -> Any + 'static,
    {
        self.global
            .insert(name.into(), Any::host_function(min_args, max_args, f));
    }

    /// Set variable according to scope rules. If local scope exists, set there; otherwise global.
    pub fn set_var(&mut self, name: impl Into<Rc<str>>, val: Any) {
        let name = name.into();
        if let Some(local) = &mut self.local {
            for (n, v) in local.iter_mut() {
                if *n == name {
                    *v = val;
                    return;
                }
            }
            local.push((name, val));
        } else {
            self.global.insert(name, val);
        }
    }

    /// Get variable: check local first, then global.
    pub fn get_var(&self, name: &str) -> Option<Any> {
        if let Some(local) = &self.local {
            for (n, v) in local.iter().rev() {
                if &**n == name {
                    return Some(v.clone());
                }
            }
        }
        self.global.get(name).cloned()
    }

    /// Enter local scope for function execution. Returns guard which restores previous local on drop.
    /// Run closure inside a newly-created local scope. Previous local is restored after closure returns.
    fn call_with_local<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        let prev = self.local.take();
        self.local = Some(SmallVec::new());
        let res = f(self);
        self.local = prev;
        res
    }

    pub fn eval(&mut self, expr: &Expr) -> Result<Any, RuntimeError> {
        self.eval_impl(expr, false)
    }

    fn eval_impl(&mut self, expr: &Expr, call_fn: bool) -> Result<Any, RuntimeError> {
        match expr.kind() {
            ExprKind::Literal(lit) => literal_value(lit),
            ExprKind::Atom(a) => Ok(Any::new_atom(a.rc_name())),
            ExprKind::Ident(i) => {
                let name = i.name();
                let val = self
                    .get_var(name)
                    .ok_or(RuntimeError::UnboundIdentifier(name.to_owned()))?;

                if call_fn {
                    self.call_bare(val)
                } else {
                    Ok(val)
                }
            }
            ExprKind::List(elements) => {
                let mut vec = Vec::with_capacity(elements.len());
                for e in elements.iter() {
                    vec.push(self.eval(e)?);
                }
                Ok(Any::new_list(vec))
            }
            ExprKind::Record(fields) => {
                let mut map = BTreeMap::new();
                for f in fields.iter() {
                    let key = f.key().rc_name();

                    let val = match f.value() {
                        None => self
                            .get_var(&key[..])
                            .ok_or(RuntimeError::UnboundIdentifier(key[..].to_owned()))?,
                        Some(value) => self.eval(value)?,
                    };

                    map.insert(key, val);
                }
                Ok(Any::new_record(map))
            }
            ExprKind::Unary(op, operand) => {
                let v = self.eval(operand)?;
                eval_unary(op.kind(), &v)
            }
            ExprKind::Binary(lhs, op, rhs) => {
                let a = self.eval(lhs)?;
                let b = self.eval(rhs)?;
                eval_binary(op.kind(), &a, &b)
            }
            ExprKind::Apply(func, args) => {
                let fval = self.eval(func)?;

                let base = self.vm.stack_len();
                for a in args.iter() {
                    match self.eval(a) {
                        Ok(arg) => self.vm.push_stack(arg),
                        Err(e) => {
                            // don't strand already-evaluated arguments
                            self.vm.truncate_stack(base);
                            return Err(e);
                        }
                    }
                }
                // calling convention: first argument on top of the stack,
                // same as the reversal Instr::Call performs
                self.vm.reverse_stack(args.len());

                self.apply_value(fval, args.len())
            }
            ExprKind::Field(obj, field_ident) => {
                let v = self.eval(obj)?;
                field_of(&v, field_ident.name())
            }
            ExprKind::Index(obj, index_expr) => {
                let objv = self.eval(obj)?;
                let idxv = self.eval(index_expr)?;
                index_of(&objv, &idxv)
            }
            ExprKind::Parenthesized(expr) => self.eval_impl(expr, true),
        }
    }

    /// Call `fval` with already-evaluated arguments, following the language's
    /// application rules: each callee consumes as many arguments as it wants
    /// (possibly returning a partially-applied function), and leftover
    /// arguments are re-applied to whatever it returned. Shared by the AST
    /// walker (`ExprKind::Apply`) and the bytecode VM (`vm::Instr::Call`).
    pub(crate) fn apply_value(
        &mut self,
        mut fval: Any,
        mut args: usize,
    ) -> Result<Any, RuntimeError> {
        while args > 0 {
            let callee = match fval {
                Any::Fn(f) => Ok(f),
                Any::Object(h) => {
                    // a record is callable if it holds a function under the special key "__call"
                    let borrowed = h.borrow();
                    match &borrowed.kind {
                        ObjectKind::Record(map) => match map.get("__call") {
                            Some(Any::Fn(f)) => Ok(f.clone()),
                            _ => Err(RuntimeError::NotAFunction("object not callable".into())),
                        },
                        _ => Err(RuntimeError::NotAFunction("list not callable".into())),
                    }
                }
                _ => Err(RuntimeError::NotAFunction("value is not callable".into())),
            };
            let callee = match callee {
                Ok(callee) => callee,
                Err(e) => {
                    // don't strand the unconsumed arguments
                    drop(self.vm.drain_off_stack(args));
                    return Err(e);
                }
            };

            // the callee establishes its own function-local scope (see
            // DynFunction::dyn_call_once)
            let ret = callee.invoke(self, &mut args);
            if self.status.is_err() {
                drop(self.vm.drain_off_stack(args));
                self.status.clone()?;
            }

            fval = ret;
        }

        Ok(fval)
    }

    /// A value referenced in call position with no arguments: zero-argument
    /// functions are invoked, everything else passes through unchanged.
    /// Shared by the AST walker (bare/parenthesized idents) and the bytecode
    /// VM (`vm::Instr::CallBare`).
    pub(crate) fn call_bare(&mut self, val: Any) -> Result<Any, RuntimeError> {
        match val {
            // only a function that may take zero arguments is invoked; one
            // that needs more would just evaluate to itself anyway, so the
            // arity hint lets us skip the call entirely
            Any::Fn(f) => {
                // call inside function-local scope
                let ret = f.invoke(self, &mut 0);
                self.status.clone()?;
                Ok(ret)
            }
            val => Ok(val),
        }
    }

    pub fn exec_stmt(&mut self, statement: &Stmt) -> Result<Exec, RuntimeError> {
        match statement.kind() {
            StmtKind::Expr(e) => {
                let val = self.eval_impl(e, true)?;
                Ok(Exec::Value(val))
            }
            StmtKind::AssignPat { target, value } => {
                let val = self.eval(value)?;
                self.bind_pattern(target, &val)?;
                Ok(Exec::Value(Any::nil()))
            }
            StmtKind::AssignField {
                target,
                field,
                value,
            } => {
                let objv = self.eval(target)?;
                let val = self.eval(value)?;
                assign_field(objv, field.rc_name(), val)?;
                Ok(Exec::Value(Any::nil()))
            }
            StmtKind::AssignIndex {
                target,
                index,
                value,
            } => {
                let objv = self.eval(target)?;
                let idxv = self.eval(index)?;
                let val = self.eval(value)?;
                assign_index(objv, idxv, val)?;
                Ok(Exec::Value(Any::nil()))
            }
            StmtKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let cv = self.eval(cond)?;
                if !is_falsey(&cv) {
                    self.exec_block(then_branch)
                } else {
                    if let Some(eb) = else_branch {
                        self.exec_block(eb)
                    } else {
                        Ok(Exec::Value(Any::nil()))
                    }
                }
            }
            StmtKind::While {
                cond,
                body,
                else_branch,
            } => loop {
                let cv = self.eval(cond)?;
                if is_falsey(&cv) {
                    if let Some(eb) = else_branch {
                        break self.exec_block(eb);
                    }
                    break Ok(Exec::Value(Any::nil()));
                }
                match self.exec_block(body)? {
                    Exec::Value(_) => continue,
                    Exec::Return(v) => break Ok(Exec::Return(v)),
                    Exec::Continue => continue,
                    Exec::Break => break Ok(Exec::Value(Any::nil())),
                }
            },
            StmtKind::For {
                target,
                iterable,
                body,
                else_branch,
            } => {
                let iter_val = self.eval(iterable)?;
                let values = iter_val.iter()?;

                for value in values {
                    self.bind_pattern(target, &value)?;

                    match self.exec_block(body)? {
                        Exec::Return(v) => return Ok(Exec::Return(v)),
                        Exec::Break => return Ok(Exec::Value(Any::nil())),
                        Exec::Continue => continue,
                        Exec::Value(_) => continue,
                    }
                }

                if let Some(else_branch) = else_branch {
                    self.exec_block(else_branch)
                } else {
                    Ok(Exec::Value(Any::nil()))
                }
            }
            StmtKind::Return(None) => Ok(Exec::Return(Any::nil())),
            StmtKind::Return(Some(expr)) => {
                let v = self.eval(expr)?;
                Ok(Exec::Return(v))
            }
            StmtKind::Break => Ok(Exec::Break),
            StmtKind::Continue => Ok(Exec::Continue),
            StmtKind::Fn { name, params, body } => {
                let fval = if self.compile_fns {
                    match vm::compile_fn(&mut self.vm, params.clone(), body) {
                        Ok(compiled) => compiled.into_function(),
                        // constructs the compiler rejects still run on the AST walker
                        Err(_) => Any::fn_from_ast(params.clone(), body.clone()),
                    }
                } else {
                    Any::fn_from_ast(params.clone(), body.clone())
                };
                self.set_var(name.rc_name(), fval);
                Ok(Exec::Value(Any::nil()))
            }
        }
    }

    /// Execute block of statements. Stops and returns Some(value) if a return happens.
    fn exec_block(&mut self, stmts: &[Stmt]) -> Result<Exec, RuntimeError> {
        let mut last_val = Any::nil();
        for s in stmts {
            match self.exec_stmt(s)? {
                Exec::Value(val) => last_val = val,
                Exec::Return(val) => return Ok(Exec::Return(val)),
                Exec::Continue => return Ok(Exec::Continue),
                Exec::Break => return Ok(Exec::Break),
            }
        }
        Ok(Exec::Value(last_val))
    }

    fn bind_pattern(&mut self, pattern: &Pat, val: &Any) -> Result<(), RuntimeError> {
        match pattern.kind() {
            PatKind::Ident(id) => {
                if id.name() != "_" {
                    self.set_var(id.rc_name(), val.clone());
                }
                Ok(())
            }
            PatKind::Atom(a) => match val {
                Any::Atom(av) if av == a.name() => Ok(()),
                _ => Err(RuntimeError::Other("pattern match failed".into())),
            },
            PatKind::Literal(lit) => {
                // compare literal with value
                match (lit, val) {
                    (Lit::Num(nlit), Any::Number(actual)) => {
                        // suffix-aware, exact matching — same rules as the
                        // compiled representation (see vm::NumberPat)
                        if vm::NumberPat::from_literal(nlit).matches(*actual) {
                            Ok(())
                        } else {
                            Err(RuntimeError::Other("pattern match failed".into()))
                        }
                    }
                    (Lit::Str(slit), Any::String(s)) => {
                        if slit.unescape() == &**s {
                            Ok(())
                        } else {
                            Err(RuntimeError::Other("pattern match failed".into()))
                        }
                    }
                    (Lit::Char(clit), Any::Char(c)) => {
                        if clit.unescape() == *c {
                            Ok(())
                        } else {
                            Err(RuntimeError::Other("pattern match failed".into()))
                        }
                    }
                    _ => Err(RuntimeError::Other("pattern match failed".into())),
                }
            }
            PatKind::List(items) => match val {
                Any::Object(o) => match &o.borrow().kind {
                    ObjectKind::List(vec) => {
                        if vec.len() != items.len() {
                            return Err(RuntimeError::Other("pattern match failed".into()));
                        }
                        for (p, v) in items.iter().zip(vec.iter()) {
                            self.bind_pattern(p, v)?;
                        }
                        Ok(())
                    }
                    _ => Err(RuntimeError::Other("pattern match failed".into())),
                },
                _ => Err(RuntimeError::Other("pattern match failed".into())),
            },
            PatKind::Record(fields) => match val {
                Any::Object(o) => match &o.borrow().kind {
                    ObjectKind::Record(map) => {
                        for f in fields.iter() {
                            if let Some(v) = map.get(f.key().name()) {
                                match f.pattern() {
                                    None => {
                                        self.set_var(f.key().rc_name(), v.clone());
                                    }
                                    Some(pattern) => {
                                        self.bind_pattern(pattern, v)?;
                                    }
                                }
                            } else {
                                return Err(RuntimeError::Other("pattern match failed".into()));
                            }
                        }
                        Ok(())
                    }
                    _ => Err(RuntimeError::Other("pattern match failed".into())),
                },
                _ => Err(RuntimeError::Other("pattern match failed".into())),
            },
        }
    }
}

fn is_falsey(v: &Any) -> bool {
    match v {
        Any::Number(Number::Integer(0)) => true,
        Any::Number(Number::Float(f)) if *f == 0.0 => true,
        Any::Atom(a) => a.is_false(),
        Any::Object(o) => match &o.borrow().kind {
            ObjectKind::List(v) => v.is_empty(),
            ObjectKind::Record(m) => m.is_empty(),
        },
        _ => false,
    }
}

/// Evaluate a number literal in *expression* position to a runtime number,
/// honoring its suffix (`1i` is an integer, `1f`/`1.0`/`1e3` are floats).
/// Pat position interprets literals through [`vm::NumberPat`], where
/// the suffix additionally selects matching strictness.
pub(crate) fn number_value(n: &LitNum) -> Result<Number, RuntimeError> {
    match n.suffix() {
        None | Some("i" | "f") => {}
        Some(suffix) => {
            return Err(RuntimeError::TypeError(format!(
                "unsupported number suffix: {}",
                suffix
            )));
        }
    }

    if n.has_dot() || n.has_exponent() || n.suffix() == Some("f") {
        let f = n
            .value()
            .parse::<f64>()
            .map_err(|_| RuntimeError::TypeError("invalid float literal".into()))?;
        Ok(Number::Float(f))
    } else {
        let i = n
            .value()
            .parse::<i64>()
            .map_err(|_| RuntimeError::TypeError("invalid integer literal".into()))?;
        Ok(Number::Integer(i))
    }
}

/// Evaluate a literal AST node to a runtime value. Used by the AST walker at
/// evaluation time and by the bytecode compiler at compile time (literals
/// become constants).
pub(crate) fn literal_value(lit: &Lit) -> Result<Any, RuntimeError> {
    match lit {
        Lit::Num(n) => Ok(Any::Number(number_value(n)?)),
        Lit::Str(s) => Ok(Any::String(Rc::from(s.unescape()))),
        Lit::Char(c) => Ok(Any::Char(c.unescape())),
    }
}

/// `value.field` — read a record field.
pub(crate) fn field_of(v: &Any, name: &str) -> Result<Any, RuntimeError> {
    match v {
        Any::Object(h) => {
            let borrowed = h.borrow();
            match &borrowed.kind {
                ObjectKind::Record(map) => map
                    .get(name)
                    .cloned()
                    .ok_or(RuntimeError::FieldError(name.to_owned())),
                _ => Err(RuntimeError::FieldError(name.to_owned())),
            }
        }
        _ => Err(RuntimeError::FieldError(name.to_owned())),
    }
}

/// `value[index]` — read a list element.
pub(crate) fn index_of(objv: &Any, idxv: &Any) -> Result<Any, RuntimeError> {
    match (objv, idxv) {
        (Any::Object(h), Any::Number(Number::Integer(i))) => {
            let borrowed = h.borrow();
            match &borrowed.kind {
                ObjectKind::List(vec) => match usize::try_from(*i) {
                    Ok(i) => vec
                        .get(i)
                        .cloned()
                        .ok_or(RuntimeError::IndexError(format!("out of bounds: {}", i))),
                    Err(_) => Err(RuntimeError::IndexError(format!("negative index: {}", i))),
                },
                ObjectKind::Record(_) => Err(RuntimeError::IndexError(
                    "indexing record with integer unsupported".into(),
                )),
            }
        }
        _ => Err(RuntimeError::TypeError("indexing non-heap value".into())),
    }
}

/// `target.field = value` — write a record field.
pub(crate) fn assign_field(objv: Any, field: Rc<str>, val: Any) -> Result<(), RuntimeError> {
    match objv {
        Any::Object(o) => {
            let mut borrowed = o.borrow_mut();
            if borrowed.frozen {
                return Err(RuntimeError::Other(
                    "attempt to mutate frozen object".into(),
                ));
            }
            match &mut borrowed.kind {
                ObjectKind::Record(map) => {
                    map.insert(field, val);
                    Ok(())
                }
                _ => Err(RuntimeError::FieldError(field[..].to_owned())),
            }
        }
        _ => Err(RuntimeError::FieldError(field[..].to_owned())),
    }
}

/// `target[index] = value` — write a list element.
pub(crate) fn assign_index(objv: Any, idxv: Any, val: Any) -> Result<(), RuntimeError> {
    match (objv, idxv) {
        (Any::Object(o), Any::Number(Number::Integer(i))) => {
            let mut borrowed = o.borrow_mut();
            if borrowed.frozen {
                return Err(RuntimeError::Other(
                    "attempt to mutate frozen object".into(),
                ));
            }
            match &mut borrowed.kind {
                ObjectKind::List(vec) => {
                    if i < 0 {
                        return Err(RuntimeError::IndexError(format!("negative index: {}", i)));
                    }
                    let ui = usize::try_from(i)
                        .map_err(|_| RuntimeError::IndexError(format!("invalid index: {}", i)))?;
                    if ui >= vec.len() {
                        return Err(RuntimeError::IndexError(format!("out of bounds: {}", ui)));
                    }
                    vec[ui] = val;
                    Ok(())
                }
                _ => Err(RuntimeError::IndexError("indexing non-list object".into())),
            }
        }
        _ => Err(RuntimeError::TypeError(
            "index must be integer and target must be object".into(),
        )),
    }
}

pub(crate) fn eval_unary(op: UnOpKind, a: &Any) -> Result<Any, RuntimeError> {
    use raft_ast::UnOpKind::*;
    match op {
        Not => Ok(a.not()),
        BitNot => a.bit_not(),
        Pos => a.pos(),
        Neg => a.neg(),
    }
}

pub(crate) fn eval_binary(op: BinOpKind, a: &Any, b: &Any) -> Result<Any, RuntimeError> {
    use raft_ast::BinOpKind::*;
    match op {
        BitAnd => a.bit_and(b),
        BitOr => a.bit_or(b),
        BitXor => a.bit_xor(b),
        Shl => a.shl(b),
        Shr => a.shr(b),
        Pow => a.pow(b),
        Mul => a.mul(b),
        Div => a.div(b),
        Add => a.add(b),
        Sub => a.sub(b),
        Eq => Ok(Any::bool_(a.cmp(b) == Some(Ordering::Equal))),
        Ne => Ok(Any::bool_(a.cmp(b) != Some(Ordering::Equal))),
        Lt => Ok(Any::bool_(a.cmp(b) == Some(Ordering::Less))),
        Le => Ok(Any::bool_(matches!(
            a.cmp(b),
            Some(Ordering::Less | Ordering::Equal)
        ))),
        Gt => Ok(Any::bool_(a.cmp(b) == Some(Ordering::Greater))),
        Ge => Ok(Any::bool_(matches!(
            a.cmp(b),
            Some(Ordering::Greater | Ordering::Equal)
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ast_from_str(s: &str) -> raft_ast::Module {
        let tokens = raft_ast::lexer::parse_str(s, &raft_ast::lexer::Options::wss()).unwrap();
        let mut stream = raft_ast::parser::TokenStream::new(tokens);
        stream.parse_module().unwrap()
    }

    #[test]
    fn assign_pattern_ident_binds_var() {
        let src = "x = 1";
        let module = ast_from_str(src);
        let mut rt = Runtime::new();
        assert_eq!(
            rt.exec_block(module.stmts()).unwrap(),
            Exec::Value(Any::nil())
        );
        let v = rt.get_var("x").expect("x bound");
        match v {
            Any::Number(Number::Integer(i)) => assert_eq!(i, 1),
            _ => panic!("expected integer"),
        }
    }

    #[test]
    fn assign_pattern_list_binds_vars() {
        let src = "[a, b] = [1, 2]";
        let module = ast_from_str(src);
        let mut rt = Runtime::new();
        assert_eq!(
            rt.exec_block(module.stmts()).unwrap(),
            Exec::Value(Any::nil())
        );
        let va = rt.get_var("a").expect("a bound");
        let vb = rt.get_var("b").expect("b bound");
        match va {
            Any::Number(Number::Integer(i)) => assert_eq!(i, 1),
            _ => panic!("expected integer for a"),
        }
        match vb {
            Any::Number(Number::Integer(i)) => assert_eq!(i, 2),
            _ => panic!("expected integer for b"),
        }
    }

    #[test]
    fn literal_pattern_match_success_and_failure() {
        let ok_src = "'a' = 'a'";
        let ok_module = ast_from_str(ok_src);
        let mut rt = Runtime::new();
        assert_eq!(
            rt.exec_block(ok_module.stmts()).unwrap(),
            Exec::Value(Any::nil())
        );

        let bad_src = "'a' = 'b'";
        let bad_module = ast_from_str(bad_src);
        let mut rt2 = Runtime::new();
        assert!(rt2.exec_block(bad_module.stmts()).is_err());
    }

    #[test]
    fn field_and_index_assignment() {
        let src = "obj = { x: 1 }\nobj.x = 5\narr = [0, 1]\narr[0] = 7";
        let module = ast_from_str(src);
        let mut rt = Runtime::new();
        assert_eq!(
            rt.exec_block(module.stmts()).unwrap(),
            Exec::Value(Any::nil())
        );

        // check obj.x
        let objv = rt.get_var("obj").expect("obj bound");
        match objv {
            Any::Object(o) => {
                let b = o.borrow();
                match &b.kind {
                    ObjectKind::Record(map) => {
                        let vx = map.get("x").expect("field x present");
                        match vx {
                            Any::Number(Number::Integer(i)) => assert_eq!(*i, 5),
                            _ => panic!("expected integer in obj.x"),
                        }
                    }
                    _ => panic!("obj not record"),
                }
            }
            _ => panic!("obj not object"),
        }

        // check arr[0]
        let arrv = rt.get_var("arr").expect("arr bound");
        match arrv {
            Any::Object(o) => {
                let b = o.borrow();
                match &b.kind {
                    ObjectKind::List(vec) => match &vec[0] {
                        Any::Number(Number::Integer(i)) => assert_eq!(*i, 7),
                        _ => panic!("expected integer in arr[0]"),
                    },
                    _ => panic!("arr not list"),
                }
            }
            _ => panic!("arr not object"),
        }
    }

    #[test]
    fn return_short_circuits_block() {
        let src = "return 5\nx = 1";
        let module = ast_from_str(src);
        let mut rt = Runtime::new();
        let res = rt.exec_block(module.stmts()).unwrap();
        match res {
            Exec::Return(Any::Number(Number::Integer(i))) => assert_eq!(i, 5),
            _ => panic!("expected return value"),
        }
        assert!(rt.get_var("x").is_none());
    }

    #[test]
    fn if_else_execution() {
        let src = "if True:\n    x = 1";
        let module = ast_from_str(src);
        let mut rt = Runtime::new();
        assert_eq!(
            rt.exec_block(module.stmts()).unwrap(),
            Exec::Value(Any::nil())
        );
        match rt.get_var("x").expect("x bound") {
            Any::Number(Number::Integer(i)) => assert_eq!(i, 1),
            _ => panic!("expected integer"),
        }

        let src2 = "if False:\n    x = 1\nelse:\n    x = 2";
        let module2 = ast_from_str(src2);
        let mut rt2 = Runtime::new();
        assert_eq!(
            rt2.exec_block(module2.stmts()).unwrap(),
            Exec::Value(Any::nil())
        );
        match rt2.get_var("x").expect("x bound") {
            Any::Number(Number::Integer(i)) => assert_eq!(i, 2),
            _ => panic!("expected integer"),
        }
    }

    #[test]
    fn frozen_object_mutation_errors() {
        let src = "r = { x: 1 }";
        let module = ast_from_str(src);
        let mut rt = Runtime::new();
        assert_eq!(
            rt.exec_block(module.stmts()).unwrap(),
            Exec::Value(Any::nil())
        );
        // freeze object
        let rv = rt.get_var("r").expect("r bound");
        match rv {
            Any::Object(o) => {
                o.borrow_mut().freeze();
            }
            _ => panic!("r not object"),
        }
        // attempt mutation
        let mut rt2 = rt; // move ownership
        let bad_src = "r.x = 2";
        let bad_module = ast_from_str(bad_src);
        assert!(rt2.exec_block(bad_module.stmts()).is_err());
    }

    // Loop/else semantics tests (runtime implementation pending). Marked #[ignore]
    #[test]
    fn while_else_execution() {
        let src = "i = 0\nwhile i < 3:\n    i = i + 1\nelse:\n    flag = 1";
        let module = ast_from_str(src);
        let mut rt = Runtime::new();
        let res = rt.exec_block(module.stmts()).unwrap();
        assert_eq!(res, Exec::Value(Any::nil()));
        match rt.get_var("i").expect("i bound") {
            Any::Number(Number::Integer(i)) => assert_eq!(i, 3),
            _ => panic!("expected integer"),
        }
        assert!(rt.get_var("flag").is_some());
    }

    #[test]
    fn while_else_not_on_break_or_return() {
        let src = "i = 0\nwhile i < 3:\n    if i == 1:\n        break\n    i = i + 1\nelse:\n    flag = 1";
        let module = ast_from_str(src);
        let mut rt = Runtime::new();
        let _ = rt.exec_block(module.stmts()).unwrap();
        // loop should exit via break and else should NOT execute
        assert!(rt.get_var("flag").is_none());
    }

    #[test]
    fn for_else_execution() {
        let src = "sum = 0\narr = [1, 2]\nfor a in arr:\n    sum = sum + a\nelse:\n    done = 1";
        let module = ast_from_str(src);
        let mut rt = Runtime::new();
        let _ = rt.exec_block(module.stmts()).unwrap();
        match rt.get_var("sum").expect("sum bound") {
            Any::Number(Number::Integer(i)) => assert_eq!(i, 3),
            Any::Number(Number::Float(_)) => panic!("unexpected float"),
            _ => panic!("expected numeric sum"),
        }
        assert!(rt.get_var("done").is_some());
    }

    #[test]
    fn fn_values_carry_arity_hints() {
        let src = "fn add3 a b c:\n    return a + b + c\nadd1 = add3 1 2\n";
        let module = ast_from_str(src);
        let mut rt = Runtime::new();
        rt.exec_block(module.stmts()).unwrap();

        let Some(Any::Fn(_full)) = rt.get_var("add3") else {
            panic!("add3 not a function");
        };

        // two arguments preapplied: one left to go
        let Some(Any::Fn(_partial)) = rt.get_var("add1") else {
            panic!("add1 not a function");
        };

        // host registrations: default hint is "takes anything"
        rt.register_function("anything", 0, None,|_rt, _args| Any::nil());
        let Some(Any::Fn(_host)) = rt.get_var("anything") else {
            panic!("anything not a function");
        };
    }

    #[test]
    fn call_once_dispatch_for_last_reference() {
        struct Probe {
            shared_calls: Rc<core::cell::Cell<u32>>,
            once_calls: Rc<core::cell::Cell<u32>>,
        }

        impl Function for Probe {
            fn min_args(&self) -> usize {
                1
            }

            fn max_args(&self) -> Option<usize> {
                Some(1)
            }

            fn call(&self, _rt: &mut Runtime, args: usize) -> Any {
                debug_assert_eq!(args, 1);
                self.shared_calls.set(self.shared_calls.get() + 1);
                Any::nil()
            }

            // `self` by value: the hidden bridge already proved uniqueness
            fn call_once(self, _rt: &mut Runtime, args: usize) -> Any {
                debug_assert_eq!(args, 1);
                self.once_calls.set(self.once_calls.get() + 1);
                Any::nil()
            }
        }

        let shared_calls = Rc::new(core::cell::Cell::new(0));
        let once_calls = Rc::new(core::cell::Cell::new(0));
        let probe = || {
            Any::Fn(FnValue::new(
                Probe {
                    shared_calls: shared_calls.clone(),
                    once_calls: once_calls.clone(),
                },
            ))
        };

        let mut rt = Runtime::new();

        // stored in a variable: the global scope keeps a reference alive
        // through the call, so the shared flavor runs
        rt.set_var("probe", probe());
        let module = ast_from_str("probe 1\n");
        for statement in module.stmts() {
            rt.exec_stmt(statement).unwrap();
        }
        assert_eq!((shared_calls.get(), once_calls.get()), (1, 0));

        // a temporary function value: the argument list holds the last
        // reference, so the consuming flavor runs
        rt.vm.push_stack(Any::nil());
        rt.apply_value(probe(), 1).unwrap();
        assert_eq!((shared_calls.get(), once_calls.get()), (1, 1));
    }

    #[test]
    fn bare_reference_to_positive_arity_fn_yields_the_fn() {
        // statement-position reference to a fn needing arguments must not
        // invoke it; `(f)` evaluates to the function value itself
        let src = "fn inc x:\n    return x + 1\ng = (inc)\nr = g 41\n";
        let module = ast_from_str(src);
        let mut rt = Runtime::new();
        rt.exec_block(module.stmts()).unwrap();
        match rt.get_var("r") {
            Some(Any::Number(Number::Integer(i))) => assert_eq!(i, 42),
            other => panic!("expected 42, got {other:?}"),
        }
    }

    #[test]
    fn for_else_not_on_break_or_return() {
        let src = "arr = [1, 2, 3]\nfor a in arr:\n    if a == 2:\n        break\nelse:\n    finished = 1";
        let module = ast_from_str(src);
        let mut rt = Runtime::new();
        let _ = rt.exec_block(module.stmts()).unwrap();
        // break inside loop should prevent else from running
        assert!(rt.get_var("finished").is_none());
    }
}
