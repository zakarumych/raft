use std::{cell::RefCell, cmp::Ordering, collections::HashMap, fmt, rc::Rc};

use crate::{literal::Literal, ast::{Expr, ExprKind}};

struct KnownAtoms {
    false_: Rc<str>,
    true_: Rc<str>,
}

impl KnownAtoms {
    fn with<R, F: FnOnce(&Self) -> R>(f: F) -> R {
        thread_local! { static ATOMS: KnownAtoms = KnownAtoms {
            false_: Rc::from("False"),
            true_: Rc::from("True"),
        }; }
        ATOMS.with(|atoms| f(atoms))
    }

    fn false_() -> Rc<str> {
        Self::with(|atoms| atoms.false_.clone())
    }

    fn true_() -> Rc<str> {
        Self::with(|atoms| atoms.true_.clone())
    }
}

fn bool_atom(b: bool) -> Rc<str> {
    if b {
        KnownAtoms::true_()
    } else {
        KnownAtoms::false_()
    }
}

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
                    return Err(RuntimeError::Other("division by zero".to_string()));
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
            (Number::Integer(i1), Number::Integer(i2))  => {
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
            (Number::Float(f1), Number::Float(f2)) => f1.partial_cmp(&f2).unwrap_or(Ordering::Equal),
            (Number::Integer(i), Number::Float(f)) => (i as f64).partial_cmp(&f).unwrap_or(Ordering::Equal),
            (Number::Float(f), Number::Integer(i)) => f.partial_cmp(&(i as f64)).unwrap_or(Ordering::Equal),
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

// Object kinds
#[derive(Debug)]
pub enum ObjectKind {
    List(Vec<Any>),
    Record(HashMap<Rc<str>, Any>),
}

#[derive(Debug)]
pub struct Object {
    pub kind: ObjectKind,
    /// cost flag prevents mutation when true
    pub frozen: bool,
    /// mutable by default
    pub mutable: bool,
}

impl Object {
    pub fn new_list(elements: Vec<Any>) -> Self {
        Object {
            kind: ObjectKind::List(elements),
            frozen: false,
            mutable: true,
        }
    }

    pub fn new_record(fields: HashMap<Rc<str>, Any>) -> Self {
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
            },
            _ => None, // different kinds are considered incomparable
        }
    }
}

// Value reference used by interpreter. Clone cheap for literals and Rc for heap objects.
#[derive(Clone)]
pub enum Any {
    Number(Number),
    Char(char),
    String(Rc<str>),
    Atom(Rc<str>),                 // atoms like True/False or symbols
    Object(Rc<RefCell<Object>>),   // lists and records live here
    External(Rc<ExternalFn>),      // host-provided function
    Opaque(Rc<dyn std::any::Any>), // opaque value, uninterpretable by raft code
}

impl fmt::Debug for Any {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Any::Number(n) => write!(f, "Number({:?})", n),
            Any::Char(c) => write!(f, "Char({})", c),
            Any::String(s) => write!(f, "String({})", s),
            Any::Atom(a) => write!(f, "Atom({})", a),
            Any::Object(o) => write!(f, "Object({:?})", o.borrow()),
            Any::External(_) => write!(f, "External(<fn>)"),
            Any::Opaque(val) => write!(f, "Opaque({:p})", &**val),
        }
    }
}

impl Any {
    fn neg(&self) -> Result<Any, RuntimeError> {
        match self {
            Any::Number(n) => Ok(Any::Number(n.neg()?)),
            _ => Err(RuntimeError::TypeError(
                "negation on non-numeric value".into(),
            )),
        }
    }

    fn not(&self) -> Any {
        Any::Atom(bool_atom(is_falsey(self)))
    }

    fn bit_not(&self) -> Result<Any, RuntimeError> {
        match self {
            Any::Number(Number::Integer(i)) => Ok(Any::Number(Number::Integer(!i))),
            _ => Err(RuntimeError::TypeError(
                "bitwise not on non-integer value".into(),
            )),
        }
    }

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
                "addition on not numeric or string value".into()
            )),
        }
    }
    
    fn sub(&self, rhs: &Any) -> Result<Any, RuntimeError> {
        match (self, rhs) {
            (Any::Number(n1), Any::Number(n2)) => Ok(Any::Number(n1.sub(*n2)?)),
            _ => Err(RuntimeError::TypeError(
                "subtraction on non-numeric value".into(),
            )),
        }
    }

    fn mul(&self, rhs: &Any) -> Result<Any, RuntimeError> {
        match (self, rhs) {
            (Any::Number(n1), Any::Number(n2)) => Ok(Any::Number(n1.mul(*n2)?)),
            _ => Err(RuntimeError::TypeError(
                "multiplication on non-numeric value".into(),
            )),
        }
    }

    fn div(&self, rhs: &Any) -> Result<Any, RuntimeError> {
        match (self, rhs) {
            (Any::Number(n1), Any::Number(n2)) => Ok(Any::Number(n1.div(*n2)?)),
            _ => Err(RuntimeError::TypeError(
                "division on non-numeric value".into(),
            )),
        }
    }

    fn pow(&self, rhs: &Any) -> Result<Any, RuntimeError> {
        match (self, rhs) {
            (Any::Number(n1), Any::Number(n2)) => Ok(Any::Number(n1.pow(*n2)?)),
            _ => Err(RuntimeError::TypeError(
                "exponentiation on non-numeric value".into(),
            )),
        }
    }

    fn cmp(&self, rhs: &Any) -> Option<Ordering> {
        match (self, rhs) {
            (Any::Number(n1), Any::Number(n2)) => Some(n1.cmp(*n2)),
            (Any::Atom(a1), Any::Atom(a2)) => if a1 == a2 { Some(Ordering::Equal) } else { None },
            (Any::String(s1), Any::String(s2)) => Some(s1.cmp(s2)),
            (Any::Char(c1), Any::Char(c2)) => Some(c1.cmp(c2)),
            (Any::Object(o1), Any::Object(o2)) => {
                o1.borrow().cmp(&o2.borrow())
            }
            (Any::External(e1), Any::External(e2)) => {
                std::ptr::eq(Rc::as_ptr(e1), Rc::as_ptr(e2)).then(|| Ordering::Equal)
            }
            (Any::Opaque(o1), Any::Opaque(o2)) => {
                std::ptr::eq(Rc::as_ptr(o1), Rc::as_ptr(o2)).then(|| Ordering::Equal)
            }
            _ => None, // different kinds are considered incomparable
        }
    }

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

    fn le(&self, rhs: &Any) -> Result<Any, RuntimeError> {
        match (self, rhs) {
            (Any::Number(n1), Any::Number(n2)) => Ok(Any::Atom(bool_atom(n1.cmp(*n2) != Ordering::Greater))),
            _ => Err(RuntimeError::TypeError(
                "comparison on non-numeric value".into(),
            )),
        }
    }

    fn ge(&self, rhs: &Any) -> Result<Any, RuntimeError> {
        match (self, rhs) {
            (Any::Number(n1), Any::Number(n2)) => Ok(Any::Atom(bool_atom(n1.cmp(*n2) != Ordering::Less))),
            _ => Err(RuntimeError::TypeError(
                "comparison on non-numeric value".into(),
            )),
        }
    }

    fn lt(&self, rhs: &Any) -> Result<Any, RuntimeError> {
        match (self, rhs) {
            (Any::Number(n1), Any::Number(n2)) => Ok(Any::Atom(bool_atom(n1.cmp(*n2) == Ordering::Less))),
            _ => Err(RuntimeError::TypeError(
                "comparison on non-numeric value".into(),
            )),
        }
    }

    fn gt(&self, rhs: &Any) -> Result<Any, RuntimeError> {
        match (self, rhs) {
            (Any::Number(n1), Any::Number(n2)) => Ok(Any::Atom(bool_atom(n1.cmp(*n2) == Ordering::Greater))),
            _ => Err(RuntimeError::TypeError(
                "comparison on non-numeric value".into(),
            )),
        }
    }
}

// External function type. Receives runtime and evaluated args, returns Any or error string.
pub type ExternalFn = dyn Fn(&mut Runtime, &[Any]) -> Any;

// Runtime with two scopes: global and optional local
pub struct Runtime {
    pub global: HashMap<String, Any>,
    pub local: Option<HashMap<String, Any>>,
    pub status: Result<(), RuntimeError>,
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

impl Runtime {
    pub fn new() -> Self {
        Runtime {
            global: HashMap::new(),
            local: None,
            status: Ok(()),
        }
    }

    pub fn set_error(&mut self, err: RuntimeError) {
        self.status = Err(err);
    }

    pub fn register_external<F>(&mut self, name: &str, f: F)
    where
        F: Fn(&mut Runtime, &[Any]) -> Any + 'static,
    {
        self.global
            .insert(name.to_string(), Any::External(Rc::new(f)));
    }

    /// Set variable according to scope rules. If local scope exists, set there; otherwise global.
    pub fn set_var(&mut self, name: impl Into<String>, val: Any) {
        let name = name.into();
        if let Some(local) = &mut self.local {
            local.insert(name, val);
        } else {
            self.global.insert(name, val);
        }
    }

    /// Get variable: check local first, then global.
    pub fn get_var(&self, name: &str) -> Option<Any> {
        if let Some(local) = &self.local {
            if let Some(v) = local.get(name) {
                return Some(v.clone());
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
        self.local = Some(HashMap::new());
        let res = f(self);
        self.local = prev;
        res
    }

    pub fn eval(&mut self, expr: &Expr) -> Result<Any, RuntimeError> {
        match &expr.kind {
            ExprKind::Literal(lit) => match lit {
                Literal::Number(n) => {
                    match n.suffix() {
                        None => {}
                        Some("i" | "f") => {
                            // valid suffixes, but we don't care about them for now
                        }
                        _ => {
                            return Err(RuntimeError::TypeError(format!(
                                "unsupported number suffix: {}",
                                n.suffix().unwrap()
                            )));
                        }
                    }

                    if n.has_dot() || n.has_exponent() || n.suffix() == Some("f") {
                        let f = n
                            .value()
                            .parse::<f64>()
                            .map_err(|_| RuntimeError::TypeError("invalid float literal".into()))?;
                        Ok(Any::Number(Number::Float(f)))
                    } else {
                        let i = n.value().parse::<i64>().map_err(|_| {
                            RuntimeError::TypeError("invalid integer literal".into())
                        })?;
                        Ok(Any::Number(Number::Integer(i)))
                    }
                }
                Literal::String(s) => Ok(Any::String(Rc::from(s.unescape()))),
                Literal::Char(c) => Ok(Any::Char(c.unescape())),
            },
            ExprKind::Atom(a) => Ok(Any::Atom(a.name[..].into())),
            ExprKind::Ident(i) => {
                let name = &i.name;
                self.get_var(name)
                    .ok_or(RuntimeError::UnboundIdentifier(name.clone()))
            }
            ExprKind::List(elements) => {
                let mut vec = Vec::with_capacity(elements.len());
                for e in elements {
                    vec.push(self.eval(e)?);
                }
                Ok(Any::Object(Rc::new(RefCell::new(Object::new_list(vec)))))
            }
            ExprKind::Record(fields) => {
                let mut map = HashMap::new();
                for f in fields {
                    let key = f.key.name[..].into();
                    let val = self.eval(&f.value)?;
                    map.insert(key, val);
                }
                Ok(Any::Object(Rc::new(RefCell::new(Object::new_record(map)))))
            }
            ExprKind::Unary(op, operand) => {
                let v = self.eval(operand)?;
                eval_unary(op.node, &v)
            }
            ExprKind::Binary(lhs, op, rhs) => {
                let a = self.eval(lhs)?;
                let b = self.eval(rhs)?;
                eval_binary(op.node, &a, &b)
            }
            ExprKind::Apply(func, args) => {
                let fval = self.eval(func)?;
                let mut evaled_args = Vec::with_capacity(args.len());
                for a in args {
                    evaled_args.push(self.eval(a)?);
                }
                match fval {
                    Any::External(f) => {
                        // call inside function-local scope
                        let ret = self.call_with_local(|rt| f(rt, &evaled_args));
                        self.status.clone()?;
                        Ok(ret)
                    }
                    Any::Object(h) => {
                        // allow calling a heap object only if it holds an External under a special key "__call" (record)
                        let borrowed = h.borrow();
                        match &borrowed.kind {
                            ObjectKind::Record(map) => {
                                if let Some(Any::External(ext)) = map.get("__call") {
                                    let ret = self.call_with_local(|rt| ext(rt, &evaled_args));
                                    self.status.clone()?;
                                    Ok(ret)
                                } else {
                                    Err(RuntimeError::NotAFunction(
                                        "heap object not callable".into(),
                                    ))
                                }
                            }
                            _ => Err(RuntimeError::NotAFunction("list not callable".into())),
                        }
                    }
                    _ => Err(RuntimeError::NotAFunction("value is not callable".into())),
                }
            }
            ExprKind::Field(obj, field_ident) => {
                let v = self.eval(obj)?;
                match &v {
                    Any::Object(h) => {
                        let borrowed = h.borrow();
                        match &borrowed.kind {
                            ObjectKind::Record(map) => map
                                .get(&*field_ident.name)
                                .cloned()
                                .ok_or(RuntimeError::FieldError(field_ident.name.clone())),
                            _ => Err(RuntimeError::FieldError(field_ident.name.clone())),
                        }
                    }
                    _ => Err(RuntimeError::FieldError(field_ident.name.clone())),
                }
            }
            ExprKind::Index(obj, index_expr) => {
                let objv = self.eval(obj)?;
                let idxv = self.eval(index_expr)?;
                match (&objv, &idxv) {
                    (Any::Object(h), Any::Number(Number::Integer(i))) => {
                        let borrowed = h.borrow();
                        match &borrowed.kind {
                            ObjectKind::List(vec) => match usize::try_from(*i) {
                                Ok(i) => vec.get(i).cloned().ok_or(RuntimeError::IndexError(
                                    format!("out of bounds: {}", i),
                                )),
                                Err(_) => {
                                    Err(RuntimeError::IndexError(format!("negative index: {}", i)))
                                }
                            },
                            ObjectKind::Record(_) => Err(RuntimeError::IndexError(
                                "indexing record with integer unsupported".into(),
                            )),
                        }
                    }
                    _ => Err(RuntimeError::TypeError("indexing non-heap value".into())),
                }
            }
        }
    }

    
    /// Execute single statement. Returns Ok(Some(value)) if statement caused a return with value,
    /// Ok(None) if execution should continue normally, or Err on runtime error.
    pub fn exec_stmt(&mut self, stmt: &crate::ast::Stmt) -> Result<Option<Any>, RuntimeError> {
        use crate::ast::StmtKind;
        match &stmt.kind {
            StmtKind::Expr(e) => { let _ = self.eval(e)?; Ok(None) }
            StmtKind::AssignPattern(pat, expr) => {
                let val = self.eval(expr)?;
                self.bind_pattern(pat, &val)?;
                Ok(None)
            }
            StmtKind::AssignField { target, field, value } => {
                let objv = self.eval(target)?;
                let val = self.eval(value)?;
                match objv {
                    Any::Object(o) => {
                        let mut borrowed = o.borrow_mut();
                        if borrowed.frozen {
                            return Err(RuntimeError::Other("attempt to mutate frozen object".into()));
                        }
                        match &mut borrowed.kind {
                            ObjectKind::Record(map) => {
                                map.insert(field.name[..].into(), val);
                                Ok(None)
                            }
                            _ => Err(RuntimeError::FieldError(field.name.clone())),
                        }
                    }
                    _ => Err(RuntimeError::FieldError(field.name.clone())),
                }
            }
            StmtKind::AssignIndex { target, index, value } => {
                let objv = self.eval(target)?;
                let idxv = self.eval(index)?;
                let val = self.eval(value)?;
                match (objv, idxv) {
                    (Any::Object(o), Any::Number(Number::Integer(i))) => {
                        let mut borrowed = o.borrow_mut();
                        if borrowed.frozen {
                            return Err(RuntimeError::Other("attempt to mutate frozen object".into()));
                        }
                        match &mut borrowed.kind {
                            ObjectKind::List(vec) => {
                                if i < 0 {
                                    return Err(RuntimeError::IndexError(format!("negative index: {}", i)));
                                }
                                let ui = usize::try_from(i).map_err(|_| RuntimeError::IndexError(format!("invalid index: {}", i)))?;
                                if ui >= vec.len() {
                                    return Err(RuntimeError::IndexError(format!("out of bounds: {}", ui)));
                                }
                                vec[ui] = val;
                                Ok(None)
                            }
                            _ => Err(RuntimeError::IndexError("indexing non-list object".into())),
                        }
                    }
                    _ => Err(RuntimeError::TypeError("index must be integer and target must be heap object".into())),
                }
            }
            StmtKind::Return(expr) => {
                let v = self.eval(expr)?;
                Ok(Some(v))
            }
            StmtKind::If { cond, then_branch, else_branch } => {
                let cv = self.eval(cond)?;
                if !is_falsey(&cv) {
                    if let Some(r) = self.exec_block(then_branch)? { return Ok(Some(r)); }
                } else if let Some(eb) = else_branch {
                    if let Some(r) = self.exec_block(eb)? { return Ok(Some(r)); }
                }
                Ok(None)
            }
        }
    }

    /// Execute block of statements. Stops and returns Some(value) if a return happens.
    pub fn exec_block(&mut self, stmts: &[crate::ast::Stmt]) -> Result<Option<Any>, RuntimeError> {
        for s in stmts {
            if let Some(rv) = self.exec_stmt(s)? {
                return Ok(Some(rv));
            }
        }
        Ok(None)
    }

    fn bind_pattern(&mut self, pat: &crate::ast::Pattern, val: &Any) -> Result<(), RuntimeError> {
        use crate::ast::PatternKind;
        match &pat.kind {
            PatternKind::Ident(id) => {
                self.set_var(id.name.clone(), val.clone());
                Ok(())
            }
            PatternKind::Atom(a) => {
                match val {
                    Any::Atom(av) if &**av == &*a.name => Ok(()),
                    _ => Err(RuntimeError::Other("pattern match failed".into())),
                }
            }
            PatternKind::Literal(lit) => {
                // compare literal with value
                match (lit, val) {
                    (crate::literal::Literal::Number(nlit), Any::Number(Number::Integer(i))) => {
                        // compare textual repr? Use value ranges
                        let repr = nlit.value();
                        if let Ok(parsed) = repr.parse::<i64>() {
                            if parsed == *i { return Ok(()); }
                        }
                        Err(RuntimeError::Other("pattern match failed".into()))
                    }
                    (crate::literal::Literal::Number(nlit), Any::Number(Number::Float(f))) => {
                        let repr = nlit.value();
                        if let Ok(parsed) = repr.parse::<f64>() {
                            if (parsed - *f).abs() < std::f64::EPSILON { return Ok(()); }
                        }
                        Err(RuntimeError::Other("pattern match failed".into()))
                    }
                    (crate::literal::Literal::String(slit), Any::String(s)) => {
                        if slit.unescape() == &**s { Ok(()) } else { Err(RuntimeError::Other("pattern match failed".into())) }
                    }
                    (crate::literal::Literal::Char(clit), Any::Char(c)) => {
                        if clit.unescape() == *c { Ok(()) } else { Err(RuntimeError::Other("pattern match failed".into())) }
                    }
                    _ => Err(RuntimeError::Other("pattern match failed".into())),
                }
            }
            PatternKind::List(items) => {
                match val {
                    Any::Object(o) => match &o.borrow().kind {
                        ObjectKind::List(vec) => {
                            if vec.len() != items.len() { return Err(RuntimeError::Other("pattern match failed".into())); }
                            for (p, v) in items.iter().zip(vec.iter()) {
                                self.bind_pattern(p, v)?;
                            }
                            Ok(())
                        }
                        _ => Err(RuntimeError::Other("pattern match failed".into())),
                    }
                    _ => Err(RuntimeError::Other("pattern match failed".into())),
                }
            }
            PatternKind::Record(fields) => {
                match val {
                    Any::Object(o) => match &o.borrow().kind {
                        ObjectKind::Record(map) => {
                            for f in fields {
                                if let Some(v) = map.get(&*f.key.name) {
                                    self.bind_pattern(&f.pattern, v)?;
                                } else {
                                    return Err(RuntimeError::Other("pattern match failed".into()));
                                }
                            }
                            Ok(())
                        }
                        _ => Err(RuntimeError::Other("pattern match failed".into())),
                    }
                    _ => Err(RuntimeError::Other("pattern match failed".into())),
                }
            }
        }
    }
}

fn is_falsey(v: &Any) -> bool {
    match v {
        Any::Number(Number::Integer(0)) => true,
        Any::Number(Number::Float(f)) if *f == 0.0 => true,
        Any::Atom(a) => **a == *"False",
        Any::Object(o) => match &o.borrow().kind {
            ObjectKind::List(v) => v.is_empty(),
            ObjectKind::Record(m) => m.is_empty(),
        },
        _ => false,
    }
}

fn eval_unary(op: crate::ast::UnaryOp, a: &Any) -> Result<Any, RuntimeError> {
    use crate::ast::UnaryOp::*;
    match op {
        Not => Ok(a.not()),
        Neg => a.neg(),
        BitNot => a.bit_not(),
    }
}

fn eval_binary(op: crate::ast::BinaryOp, a: &Any, b: &Any) -> Result<Any, RuntimeError> {
    use crate::ast::BinaryOp::*;
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
        Eq => Ok(Any::Atom(bool_atom(a.cmp(b) == Some(Ordering::Equal)))),
        Ne => Ok(Any::Atom(bool_atom(a.cmp(b) != Some(Ordering::Equal)))),
        Lt => Ok(Any::Atom(bool_atom(a.cmp(b) == Some(Ordering::Less)))),
        Le => Ok(Any::Atom(bool_atom(matches!(a.cmp(b), Some(Ordering::Less | Ordering::Equal))))),
        Gt => Ok(Any::Atom(bool_atom(a.cmp(b) == Some(Ordering::Greater)))),
        Ge => Ok(Any::Atom(bool_atom(matches!(a.cmp(b), Some(Ordering::Greater | Ordering::Equal))))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::Stream;

    #[test]
    fn assign_pattern_ident_binds_var() {
        let src = "x = 1";
        let stmts = Stream::new(src).parse_block(None).unwrap();
        let mut rt = Runtime::new();
        assert!(rt.exec_block(&stmts).unwrap().is_none());
        let v = rt.get_var("x").expect("x bound");
        match v {
            Any::Number(Number::Integer(i)) => assert_eq!(i, 1),
            _ => panic!("expected integer"),
        }
    }

    #[test]
    fn assign_pattern_list_binds_vars() {
        let src = "[a, b] = [1, 2]";
        let stmts = Stream::new(src).parse_block(None).unwrap();
        let mut rt = Runtime::new();
        assert!(rt.exec_block(&stmts).unwrap().is_none());
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
        let ok_stmts = Stream::new(ok_src).parse_block(None).unwrap();
        let mut rt = Runtime::new();
        assert!(rt.exec_block(&ok_stmts).unwrap().is_none());

        let bad_src = "'a' = 'b'";
        let bad_stmts = Stream::new(bad_src).parse_block(None).unwrap();
        let mut rt2 = Runtime::new();
        assert!(rt2.exec_block(&bad_stmts).is_err());
    }

    #[test]
    fn field_and_index_assignment() {
        let src = "obj = { x: 1 }\nobj.x = 5\narr = [0, 1]\narr[0] = 7";
        let stmts = Stream::new(src).parse_block(None).unwrap();
        let mut rt = Runtime::new();
        assert!(rt.exec_block(&stmts).unwrap().is_none());

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
                    ObjectKind::List(vec) => {
                        match &vec[0] {
                            Any::Number(Number::Integer(i)) => assert_eq!(*i, 7),
                            _ => panic!("expected integer in arr[0]"),
                        }
                    }
                    _ => panic!("arr not list"),
                }
            }
            _ => panic!("arr not object"),
        }
    }

    #[test]
    fn return_short_circuits_block() {
        let src = "return 5\nx = 1";
        let stmts = Stream::new(src).parse_block(None).unwrap();
        let mut rt = Runtime::new();
        let res = rt.exec_block(&stmts).unwrap();
        match res {
            Some(Any::Number(Number::Integer(i))) => assert_eq!(i, 5),
            _ => panic!("expected return value"),
        }
        assert!(rt.get_var("x").is_none());
    }

    #[test]
    fn if_else_execution() {
        let src = "if True:\n    x = 1";
        let stmts = Stream::new(src).parse_block(None).unwrap();
        let mut rt = Runtime::new();
        assert!(rt.exec_block(&stmts).unwrap().is_none());
        match rt.get_var("x").expect("x bound") {
            Any::Number(Number::Integer(i)) => assert_eq!(i, 1),
            _ => panic!("expected integer"),
        }

        let src2 = "if False:\n    x = 1\nelse:\n    x = 2";
        let stmts2 = Stream::new(src2).parse_block(None).unwrap();
        let mut rt2 = Runtime::new();
        assert!(rt2.exec_block(&stmts2).unwrap().is_none());
        match rt2.get_var("x").expect("x bound") {
            Any::Number(Number::Integer(i)) => assert_eq!(i, 2),
            _ => panic!("expected integer"),
        }
    }

    #[test]
    fn frozen_object_mutation_errors() {
        let src = "r = { x: 1 }";
        let stmts = Stream::new(src).parse_block(None).unwrap();
        let mut rt = Runtime::new();
        assert!(rt.exec_block(&stmts).unwrap().is_none());
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
        let bad_stmts = Stream::new(bad_src).parse_block(None).unwrap();
        assert!(rt2.exec_block(&bad_stmts).is_err());
    }
}
