// Bundle runtime support, emitted verbatim by raft-rust into the generated
// crate's `mod support` (after the generated `NAME_COUNT`/`ATOM_COUNT`
// consts this text references). Semantics mirror raft-runtime's AST
// walker: `apply`/`call_bare` are `apply_value_ast`/`call_ast`, and the
// `field_of`/`assign_index`/comparison helpers copy the walker's shared
// semantics helpers verbatim. `Val`s cross the FFI boundary as `RawVal`
// only here (`Val::into_ffi`/`Val::from_ffi`); generated module code never
// touches a raw value.

struct BundleState {
    /// The host's callback table. Only the *function pointers* are ever
    /// used after init - `RaftFFIHost::raw` is valid during the init call
    /// only (the host `Runtime` may move afterwards); every later callback
    /// takes the live raw pointer from the `&mut Host` it is handed.
    host: Cell<Option<ffi::RaftFFIHost>>,
    /// Host `StringId` for each `NAMES` index, populated at init.
    name_ids: UnsafeCell<[usize; NAME_COUNT]>,
    /// Host `AtomId` for each `ATOMS` index, populated at init.
    atom_ids: UnsafeCell<[usize; ATOM_COUNT]>,
}

// SAFETY: a bundle is loaded into a single-threaded host (the whole object
// model - `Rc`, `Cell` refcounts - is single-threaded already); this static
// is never actually shared across threads.
unsafe impl Sync for BundleState {}

static STATE: BundleState = BundleState {
    host: Cell::new(None),
    name_ids: UnsafeCell::new([0; NAME_COUNT]),
    atom_ids: UnsafeCell::new([0; ATOM_COUNT]),
};

/// Initialize the bundle's state: remember the host and intern every name
/// and atom the bundle uses, filling the exactly-sized static id tables.
/// Called once, from the `raft_bundle!` init block, before any module
/// loads.
pub fn init(host: &ffi::RaftFFIHost) {
    STATE.host.set(Some(*host));

    // SAFETY: single-threaded (see the `Sync` impl above); no other
    // reference to these arrays exists while init runs.
    let name_ids = unsafe { &mut *STATE.name_ids.get() };
    for (i, name) in super::NAMES.iter().enumerate() {
        // SAFETY: crossing into the host's `InternFn` ABI with a valid
        // UTF-8 view and the host pointer it handed us.
        name_ids[i] = unsafe { (host.intern_string)(host.raw, name.as_ptr(), name.len()) };
    }

    let atom_ids = unsafe { &mut *STATE.atom_ids.get() };
    for (i, name) in super::ATOMS.iter().enumerate() {
        // SAFETY: as above, for the atom table.
        atom_ids[i] = unsafe { (host.intern_atom)(host.raw, name.as_ptr(), name.len()) };
    }
}

fn host() -> ffi::RaftFFIHost {
    STATE
        .host
        .get()
        .expect("raft bundle used before initialization")
}

/// The `Val` for the bundle's custom atom index `idx`.
pub fn atom(idx: u32) -> Val {
    // SAFETY: single-threaded; written only during `init`.
    Val::new_atom(AtomId(unsafe { (*STATE.atom_ids.get())[idx as usize] }))
}

/// Read host global by bundle name index. Uninit when unbound.
pub fn global_get(host_view: &mut Host<'_>, idx: u32) -> Val {
    let h = host();
    // SAFETY: single-threaded id-table read (written only during `init`),
    // then crossing into the host's `GetVarFn` ABI with the live host
    // pointer; ownership of the returned value transfers to us.
    unsafe {
        Val::from_ffi((h.getvar)(
            host_view.as_raw(),
            (*STATE.name_ids.get())[idx as usize],
        ))
    }
}

/// Report a runtime error to the host - the transpiled equivalent of the
/// walker's `Runtime::set_error`. Called at a `Function::call` boundary
/// when a transpiled body fails (the failing call pushes no result; the
/// caller detects the pending error via `check_host_error`).
pub fn report_error(host_view: &mut Host<'_>, e: &RuntimeError) {
    let h = host();
    let msg = Val::string(&format!("{e}"));
    // SAFETY: crossing into the host's `SetErrorFn` ABI with the live host
    // pointer; ownership of the message transfers to the host.
    unsafe { (h.set_error)(host_view.as_raw(), msg.into_ffi()) }
}

/// Did the last dispatched call leave the host in an error state? Takes
/// (clears) the pending error - the propagating `Err` re-reports it at the
/// next `Function::call` boundary, so nothing is lost.
fn check_host_error(host_view: &mut Host<'_>) -> Result<(), RuntimeError> {
    let h = host();
    // SAFETY: crossing into the host's `TakeErrorFn` ABI with the live
    // host pointer; ownership of the returned message (if any) transfers
    // to us.
    let pending = unsafe { Val::from_ffi((h.take_error)(host_view.as_raw())) };
    if pending.is_init() {
        Err(RuntimeError::Other(format!("{pending}").into()))
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------
// TranspiledFn: every Raft `fn` in a transpiled module becomes a Rust
// closure (capturing the `Rc<CaptureN>` structures of its enclosing
// functions) wrapped in one of these. Partial application, currying, and
// arity dispatch all happen in `raft-core`'s generic `call_dispatch`,
// exactly as for walked/compiled functions; the body only ever sees a
// full argument set, and pops it itself (first argument on top).
// ---------------------------------------------------------------------

pub struct TranspiledFn<F> {
    arity: usize,
    body: F,
}

impl<F> Function for TranspiledFn<F>
where
    F: Fn(&mut Host<'_>) -> Result<Val, RuntimeError> + 'static,
{
    fn min_args(&self) -> usize {
        self.arity
    }

    fn max_args(&self) -> Option<usize> {
        // consumes exactly its parameter count; the dispatch clamps
        // over-application and re-applies the leftovers to the result
        Some(self.arity)
    }

    fn call(&self, host: &mut Host, args: usize) {
        debug_assert_eq!(args, self.arity);
        match (self.body)(host) {
            Ok(v) => host.stack().push(v),
            // no result pushed - the caller's post-call error check
            // (`check_host_error`, or the host's own status check) sees this
            Err(e) => report_error(host, &e),
        }
    }
}

/// Wrap a generated closure as a `Val::Fn`.
pub fn fn_val<F>(arity: usize, body: F) -> Val
where
    F: Fn(&mut Host<'_>) -> Result<Val, RuntimeError> + 'static,
{
    Val::from(ValEnum::Fn(RcFn::new(TranspiledFn { arity, body })))
}

// ---------------------------------------------------------------------
// Application - the walker's `apply_value_ast`/`call_ast`, with the
// host-status check replaced by the FFI `take_error` round trip.
// ---------------------------------------------------------------------

fn callee(val: &Val) -> Option<RcFn> {
    match val.unpack() {
        ValEnum::Fn(f) => Some(f),
        // a record is callable if it holds a function under the special key "__call"
        ValEnum::Record(record) => match record.get_field("__call")?.unpack() {
            ValEnum::Fn(f) => Some(f),
            _ => None,
        },
        _ => None,
    }
}

fn truncate_args(host: &mut Host, args: usize) {
    let mut stack = host.stack();
    let len = stack.len();
    stack.truncate(len - args);
}

/// Call `fval` with `args` already-evaluated arguments on the stack (first
/// argument on top), following the language's application rules: each
/// callee consumes as many arguments as it wants (possibly returning a
/// partially-applied function), and leftover arguments are re-applied to
/// whatever it returned.
pub fn apply(host: &mut Host, mut fval: Val, mut args: usize) -> Result<Val, RuntimeError> {
    while args > 0 {
        let consumed = match fval.call_as_fn(host, args) {
            Some(consumed) => consumed,
            None => match callee(&fval) {
                Some(f) => f.call(host, args),
                None => {
                    // don't strand the unconsumed arguments
                    truncate_args(host, args);
                    return Err(RuntimeError::NotAFunction(
                        format!("{fval:?} is not callable").into(),
                    ));
                }
            },
        };

        args -= consumed;
        // checked before popping the callee's result: a failed callee
        // pushed nothing
        if let Err(e) = check_host_error(host) {
            truncate_args(host, args);
            return Err(e);
        }
        fval = host.stack().pop();
    }

    Ok(fval)
}

/// Statement-position bare reference: invoke a zero-argument callable, or
/// yield the value itself (a positive-arity function referenced bare is
/// not called - the dispatch returns it unchanged).
pub fn call_bare(host: &mut Host, fval: Val) -> Result<Val, RuntimeError> {
    if fval.call_as_fn(host, 0).is_none() {
        match callee(&fval) {
            Some(f) => {
                f.call(host, 0);
            }
            None => return Ok(fval),
        }
    }
    check_host_error(host)?;
    Ok(host.stack().pop())
}

// ---------------------------------------------------------------------
// Shared semantics helpers - verbatim ports of the walker's private
// `field_of`/`index_of`/`assign_field`/`assign_index` and its comparison
// operators.
// ---------------------------------------------------------------------

pub fn pat_fail() -> RuntimeError {
    RuntimeError::Other("pattern match failed".into())
}

/// Raft `==` as a plain bool (for pattern matching).
pub fn same(a: &Val, b: &Val) -> bool {
    a.cmp(b) == Some(Ordering::Equal)
}

pub fn eq(a: &Val, b: &Val) -> Val {
    Val::bool_(a.cmp(b) == Some(Ordering::Equal))
}

pub fn ne(a: &Val, b: &Val) -> Val {
    Val::bool_(a.cmp(b) != Some(Ordering::Equal))
}

pub fn lt(a: &Val, b: &Val) -> Val {
    Val::bool_(a.cmp(b) == Some(Ordering::Less))
}

pub fn le(a: &Val, b: &Val) -> Val {
    Val::bool_(matches!(a.cmp(b), Some(Ordering::Less | Ordering::Equal)))
}

pub fn gt(a: &Val, b: &Val) -> Val {
    Val::bool_(a.cmp(b) == Some(Ordering::Greater))
}

pub fn ge(a: &Val, b: &Val) -> Val {
    Val::bool_(matches!(
        a.cmp(b),
        Some(Ordering::Greater | Ordering::Equal)
    ))
}

/// `value.field` - read a record field.
pub fn field_of(v: &Val, field: &str) -> Result<Val, RuntimeError> {
    match v.unpack() {
        ValEnum::Record(record) => record
            .get_field(field)
            .ok_or_else(|| RuntimeError::FieldError(field.into())),
        _ => Err(RuntimeError::FieldError(field.into())),
    }
}

/// `value[index]` - read a list element.
pub fn index_of(objv: &Val, idxv: &Val) -> Result<Val, RuntimeError> {
    match (objv.unpack(), idxv.unpack()) {
        (ValEnum::List(list), ValEnum::Number(Number::Integer(i))) => match usize::try_from(i) {
            Ok(i) => list
                .get(i)
                .ok_or_else(|| RuntimeError::IndexError(format!("out of bounds: {}", i).into())),
            Err(_) => Err(RuntimeError::IndexError(
                format!("negative index: {}", i).into(),
            )),
        },
        (ValEnum::Record(_), ValEnum::Number(Number::Integer(_))) => Err(RuntimeError::IndexError(
            "indexing record with integer unsupported".into(),
        )),
        _ => Err(RuntimeError::TypeError("indexing non-heap value".into())),
    }
}

/// `target.field = value` - write a record field.
pub fn assign_field(objv: Val, field: &str, val: Val) -> Result<(), RuntimeError> {
    match objv.unpack() {
        ValEnum::Record(record) => {
            record.set_field(field, val);
            Ok(())
        }
        _ => Err(RuntimeError::FieldError(field.into())),
    }
}

/// `target[index] = value` - write a list element.
pub fn assign_index(objv: Val, idxv: Val, val: Val) -> Result<(), RuntimeError> {
    match (objv.unpack(), idxv.unpack()) {
        (ValEnum::List(list), ValEnum::Number(Number::Integer(i))) => {
            if i < 0 {
                return Err(RuntimeError::IndexError(
                    format!("negative index: {}", i).into(),
                ));
            }
            let ui = usize::try_from(i)
                .map_err(|_| RuntimeError::IndexError(format!("invalid index: {}", i).into()))?;
            if ui >= list.len() {
                return Err(RuntimeError::IndexError(
                    format!("out of bounds: {}", ui).into(),
                ));
            }
            list.set(ui, val);
            Ok(())
        }
        (ValEnum::Record(_), _) => Err(RuntimeError::IndexError("indexing non-list object".into())),
        _ => Err(RuntimeError::TypeError(
            "index must be integer and target must be object".into(),
        )),
    }
}

// ---------------------------------------------------------------------
// Number-literal patterns - the walker/VM's `NumberPat`, generated at
// transpile time from the literal's spelling.
// ---------------------------------------------------------------------

pub enum NumberPat {
    /// `1i` - matches the integer `1` only, never a float.
    Integer(i64),
    /// `1f`, `1.0`, `1e3` - matches exactly this float. Never an integer.
    Float(f64),
    /// `1` - matches the integer `1`, or a float that *is* that integer.
    Numeric(i64),
    /// An out-of-range literal: matches nothing at all.
    Never,
}

pub fn match_number(pat: &NumberPat, v: &Val) -> bool {
    let ValEnum::Number(actual) = v.unpack() else {
        return false;
    };
    match (pat, actual) {
        (NumberPat::Integer(p), Number::Integer(i)) => *p == i,
        (NumberPat::Integer(_), Number::Float(_)) => false,
        (NumberPat::Float(p), Number::Float(f)) => *p == f || (p.is_nan() && f.is_nan()),
        (NumberPat::Float(_), Number::Integer(_)) => false,
        (NumberPat::Numeric(p), Number::Integer(i)) => *p == i,
        (NumberPat::Numeric(p), Number::Float(f)) => {
            // integral f64 values in [-2^63, 2^63) convert to i64 exactly;
            // the range guard keeps `as`-cast saturation from faking
            // equality (2^63 is not i64::MAX)
            const LO: f64 = i64::MIN as f64; // -2^63, exactly representable
            const HI: f64 = -(i64::MIN as f64); // 2^63
            f.is_finite() && f.fract() == 0.0 && f >= LO && f < HI && (f as i64) == *p
        }
        (NumberPat::Never, _) => false,
    }
}
