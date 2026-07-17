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

#[inline]
fn host() -> ffi::RaftFFIHost {
    STATE
        .host
        .get()
        .expect("raft bundle used before initialization")
}

/// The `Val` for the bundle's custom atom index `idx`.
#[inline]
pub fn atom(idx: u32) -> Val {
    // SAFETY: single-threaded; written only during `init`.
    Val::new_atom(AtomId(unsafe { (*STATE.atom_ids.get())[idx as usize] }))
}

/// Read host global by bundle name index. Uninit when unbound.
#[inline]
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
#[cold]
#[inline(never)]
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
#[inline]
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
    #[inline]
    fn min_args(&self) -> usize {
        self.arity
    }

    #[inline]
    fn max_args(&self) -> Option<usize> {
        // consumes exactly its parameter count; the dispatch clamps
        // over-application and re-applies the leftovers to the result
        Some(self.arity)
    }

    #[inline]
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
#[inline]
pub fn fn_val<F>(arity: usize, body: F) -> Val
where
    F: Fn(&mut Host<'_>) -> Result<Val, RuntimeError> + 'static,
{
    Val::from(ValEnum::Fn(RcFn::new(TranspiledFn { arity, body })))
}

// ---------------------------------------------------------------------
// Generators: a transpiled `gen fn` body is an `async move` block - the
// Rust compiler builds the suspendable state machine - and `yield` is an
// await on a poll-once future. One resume = one poll with a noop waker,
// running the body to its next yield. The body never holds a `Host`
// across an await: it re-derives a fresh, statement-scoped view from the
// co's live pointer (`co_host`, set at the top of every poll) at each
// use, so no `&mut RawHost` outlives the resume that produced it.
// ---------------------------------------------------------------------

/// The channel between a suspended generator body and its `resume`: the
/// value the pending yield produced, and the live host pointer of the
/// resume currently polling (null between resumes).
pub struct GenCo {
    yielded: Cell<Option<Val>>,
    host: Cell<*mut ffi::RawHost>,
}

pub fn gen_co() -> Rc<GenCo> {
    Rc::new(GenCo {
        yielded: Cell::new(None),
        host: Cell::new(core::ptr::null_mut()),
    })
}

/// A fresh view of the live host, valid for the current statement.
/// Generator bodies name the host through this - never a binding held
/// across an await.
#[inline]
pub fn co_host(co: &GenCo) -> Host<'_> {
    let raw = co.host.get();
    debug_assert!(!raw.is_null(), "generator body ran outside a resume");
    // SAFETY: only reachable while a `poll_gen` is on the stack (it set
    // the pointer just before polling and clears it after), where the
    // host is valid and exclusively this generator's for the duration.
    unsafe { Host::from_raw(raw) }
}

/// The future behind a `yield` statement: first poll parks the value in
/// the co and suspends; the next poll (the next resume) completes.
pub struct GenYield<'a> {
    co: &'a GenCo,
    value: Option<Val>,
}

impl core::future::Future for GenYield<'_> {
    type Output = ();

    fn poll(
        self: core::pin::Pin<&mut Self>,
        _cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<()> {
        // no self-references - `GenYield` is trivially `Unpin`
        let this = self.get_mut();
        match this.value.take() {
            Some(v) => {
                this.co.yielded.set(Some(v));
                core::task::Poll::Pending
            }
            None => core::task::Poll::Ready(()),
        }
    }
}

pub fn gen_yield<'a>(co: &'a Rc<GenCo>, value: Val) -> GenYield<'a> {
    GenYield {
        co,
        value: Some(value),
    }
}

/// The implicit suspension right after parameter binding: `coro_create`
/// polls up to here, so a parameter pattern-match failure surfaces at
/// creation time (matching the walker/VM), before the first real resume.
pub struct GenStart {
    polled: bool,
}

impl core::future::Future for GenStart {
    type Output = ();

    fn poll(
        self: core::pin::Pin<&mut Self>,
        _cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<()> {
        let this = self.get_mut();
        if this.polled {
            core::task::Poll::Ready(())
        } else {
            this.polled = true;
            core::task::Poll::Pending
        }
    }
}

pub fn gen_start() -> GenStart {
    GenStart { polled: false }
}

type GenFut = core::pin::Pin<Box<dyn core::future::Future<Output = Result<Val, RuntimeError>>>>;

/// One poll of a generator/async-fn body with the co's host pointer live
/// for its duration. Polls with the host's ambient waker (the executor's
/// task waker during an async poll) so ordinary Rust futures awaited
/// inside the body wake the right task; generator resumes have no ambient
/// waker and get the noop.
fn poll_gen(
    co: &GenCo,
    host: &mut Host,
    fut: &mut GenFut,
) -> core::task::Poll<Result<Val, RuntimeError>> {
    co.host.set(host.as_raw());
    let waker = host.rust_waker();
    let mut cx = core::task::Context::from_waker(&waker);
    let r = fut.as_mut().poll(&mut cx);
    co.host.set(core::ptr::null_mut());
    r
}

/// A live transpiled coroutine (a `gen fn`'s generator or an `async fn`'s
/// future - the kind decides how polls map onto the [`CoroStatus`]
/// protocol). `fut` is `None` once finished - and also while a resume is
/// polling, so a re-entrant resume through the body's own calls reports
/// exhaustion instead of polling recursively.
pub struct TranspiledCoro {
    kind: CoroKind,
    co: Rc<GenCo>,
    fut: RefCell<Option<GenFut>>,
    /// An async resolution produced while priming (a body that never
    /// suspends) - delivered by the first real resume's Yielded step.
    resolved: RefCell<Option<Val>>,
}

impl Coroutine for TranspiledCoro {
    fn resume(&self, host: &mut Host, args: usize) -> CoroStatus {
        debug_assert_eq!(args, 0, "gen/async coroutines take no resume arguments");
        if let Some(v) = self.resolved.borrow_mut().take() {
            // `fut` is already None, so the mandatory follow-up resume
            // reports the final Done
            host.stack().push(v);
            return CoroStatus::Yielded;
        }
        let taken = self.fut.borrow_mut().take();
        let Some(mut fut) = taken else {
            // finished (or re-entrantly resumed): keeps reporting Done
            return CoroStatus::Done;
        };
        match poll_gen(&self.co, host, &mut fut) {
            core::task::Poll::Pending => match self.kind {
                // a generator body suspends only at yields (the creation
                // prime consumed `gen_start`), so the value is in the co
                CoroKind::Gen => {
                    let v = self.co.yielded.take().unwrap_or_else(Val::nil);
                    *self.fut.borrow_mut() = Some(fut);
                    host.stack().push(v);
                    CoroStatus::Yielded
                }
                // an async body suspends only at awaits (the transpiler
                // rejects `yield` in async bodies)
                CoroKind::Async => {
                    *self.fut.borrow_mut() = Some(fut);
                    CoroStatus::Pending
                }
                // an async-gen body suspends at both - the co says which
                // this one was: a parked value is a yield, nothing means
                // an await came back pending
                CoroKind::AsyncGen => {
                    *self.fut.borrow_mut() = Some(fut);
                    match self.co.yielded.take() {
                        Some(v) => {
                            host.stack().push(v);
                            CoroStatus::Yielded
                        }
                        None => CoroStatus::Pending,
                    }
                }
            },
            core::task::Poll::Ready(Ok(v)) => match self.kind {
                // a generator's return value is discarded - the end
                // itself is the signal
                CoroKind::Gen | CoroKind::AsyncGen => CoroStatus::Done,
                // an async body's result is delivered as the one Yielded
                // step; `fut` is consumed, so the mandatory follow-up
                // resume reports the final Done
                CoroKind::Async => {
                    host.stack().push(v);
                    CoroStatus::Yielded
                }
            },
            core::task::Poll::Ready(Err(e)) => {
                report_error(host, &e);
                CoroStatus::Done
            }
        }
    }
}

/// Prime a freshly-built coroutine body (run parameter binding, up to the
/// implicit [`gen_start`] suspension) and wrap it as a `Val::Coro` of
/// `kind`. A bind failure comes back as this call's error, matching
/// walker/VM creation.
pub fn coro_create(
    host: &mut Host,
    kind: CoroKind,
    co: Rc<GenCo>,
    fut: GenFut,
) -> Result<Val, RuntimeError> {
    let mut fut = fut;
    let (fut, resolved) = match poll_gen(&co, host, &mut fut) {
        core::task::Poll::Pending => (Some(fut), None),
        core::task::Poll::Ready(Err(e)) => return Err(e),
        // ran to completion while priming (a body that never suspends):
        // a generator is simply exhausted; an async resolution is handed
        // to the first real resume
        core::task::Poll::Ready(Ok(v)) => match kind {
            CoroKind::Gen | CoroKind::AsyncGen => (None, None),
            CoroKind::Async => (None, Some(v)),
        },
    };
    Ok(Val::from(ValEnum::Coro(RcCoro::new(
        kind,
        TranspiledCoro {
            kind,
            co,
            fut: RefCell::new(fut),
            resolved: RefCell::new(resolved),
        },
    ))))
}

/// The future behind an `await` statement: resumes the awaited async
/// coroutine once per poll of the enclosing body, completing when it
/// delivers its resolution.
pub struct AwaitVal<'a> {
    co: &'a GenCo,
    target: RcCoro,
}

impl core::future::Future for AwaitVal<'_> {
    type Output = Result<Val, RuntimeError>;

    fn poll(
        self: core::pin::Pin<&mut Self>,
        _cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Self::Output> {
        // no self-references - `AwaitVal` is trivially `Unpin`
        let this = self.get_mut();
        let mut host = co_host(this.co);
        match this.target.resume(&mut host, 0) {
            // the leaf that reported pending grabbed the ambient waker
            // itself; nothing to register here
            CoroStatus::Pending => core::task::Poll::Pending,
            CoroStatus::Yielded => {
                let v = host.stack().pop();
                match this.target.resume(&mut host, 0) {
                    CoroStatus::Done => match check_host_error(&mut host) {
                        Ok(()) => core::task::Poll::Ready(Ok(v)),
                        Err(e) => core::task::Poll::Ready(Err(e)),
                    },
                    _ => core::task::Poll::Ready(Err(RuntimeError::Other(
                        "async coroutine yielded more than once".into(),
                    ))),
                }
            }
            CoroStatus::Done => match check_host_error(&mut host) {
                Ok(()) => core::task::Poll::Ready(Err(RuntimeError::Other(
                    "async coroutine finished without a result".into(),
                ))),
                Err(e) => core::task::Poll::Ready(Err(e)),
            },
        }
    }
}

pub fn await_val<'a>(co: &'a Rc<GenCo>, v: Val) -> Result<AwaitVal<'a>, RuntimeError> {
    match v.unpack() {
        ValEnum::Coro(target) if target.kind() == Some(CoroKind::Async) => {
            Ok(AwaitVal { co, target })
        }
        _ => Err(RuntimeError::TypeError("await on non-async value".into())),
    }
}

// ---------------------------------------------------------------------
// `for` iteration - `raft_core::ValsIter` (lists, records, generators),
// with the host-error check a generator step needs folded in.
// ---------------------------------------------------------------------

#[inline]
pub fn iter_new(v: &Val) -> Result<ValsIter, RuntimeError> {
    ValsIter::new(v)
}

/// The next item, or `Ok(None)` at a clean end. A generator step that
/// failed reports exhaustion with the host error pending - taken and
/// propagated here. Synchronous: an async-gen iterable's pending is an
/// error here (see [`iter_next_async`]).
#[inline]
pub fn iter_next(host: &mut Host, iter: &mut ValsIter) -> Result<Option<Val>, RuntimeError> {
    match iter.next(host)? {
        Some(v) => Ok(Some(v)),
        None => {
            check_host_error(host)?;
            Ok(None)
        }
    }
}

/// The awaitable [`iter_next`]: one iteration step per poll of the
/// enclosing body, suspending it when an async-gen iterable reports
/// pending (the leaf that reported it grabbed the ambient waker itself).
pub struct IterNextFut<'a> {
    co: &'a GenCo,
    iter: &'a mut ValsIter,
}

impl core::future::Future for IterNextFut<'_> {
    type Output = Result<Option<Val>, RuntimeError>;

    fn poll(
        self: core::pin::Pin<&mut Self>,
        _cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Self::Output> {
        // no self-references - `IterNextFut` is trivially `Unpin`
        let this = self.get_mut();
        let mut host = co_host(this.co);
        match this.iter.step(&mut host) {
            Err(e) => core::task::Poll::Ready(Err(e)),
            Ok(ValsIterStep::Item(v)) => core::task::Poll::Ready(Ok(Some(v))),
            Ok(ValsIterStep::End) => match check_host_error(&mut host) {
                Ok(()) => core::task::Poll::Ready(Ok(None)),
                Err(e) => core::task::Poll::Ready(Err(e)),
            },
            Ok(ValsIterStep::Pending) => core::task::Poll::Pending,
        }
    }
}

pub fn iter_next_async<'a>(co: &'a Rc<GenCo>, iter: &'a mut ValsIter) -> IterNextFut<'a> {
    IterNextFut { co, iter }
}

// ---------------------------------------------------------------------
// Application - the walker's `apply_value_ast`/`call_ast`, with the
// host-status check replaced by the FFI `take_error` round trip.
// ---------------------------------------------------------------------

#[inline]
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

#[inline]
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

#[inline(always)]
pub fn pat_fail() -> RuntimeError {
    RuntimeError::Other("pattern match failed".into())
}

/// Raft `==` as a plain bool (for pattern matching).
#[inline(always)]
pub fn same(a: &Val, b: &Val) -> bool {
    a.cmp(b) == Some(Ordering::Equal)
}

#[inline(always)]
pub fn eq(a: &Val, b: &Val) -> Val {
    Val::bool_(a.cmp(b) == Some(Ordering::Equal))
}

#[inline(always)]
pub fn ne(a: &Val, b: &Val) -> Val {
    Val::bool_(a.cmp(b) != Some(Ordering::Equal))
}

#[inline(always)]
pub fn lt(a: &Val, b: &Val) -> Val {
    Val::bool_(a.cmp(b) == Some(Ordering::Less))
}

#[inline(always)]
pub fn le(a: &Val, b: &Val) -> Val {
    Val::bool_(matches!(a.cmp(b), Some(Ordering::Less | Ordering::Equal)))
}

#[inline(always)]
pub fn gt(a: &Val, b: &Val) -> Val {
    Val::bool_(a.cmp(b) == Some(Ordering::Greater))
}

#[inline(always)]
pub fn ge(a: &Val, b: &Val) -> Val {
    Val::bool_(matches!(
        a.cmp(b),
        Some(Ordering::Greater | Ordering::Equal)
    ))
}

/// `value.field` - read a record field.
#[inline]
pub fn field_of(v: &Val, field: &str) -> Result<Val, RuntimeError> {
    match v.unpack() {
        ValEnum::Record(record) => record
            .get_field(field)
            .ok_or_else(|| RuntimeError::FieldError(field.into())),
        _ => Err(RuntimeError::FieldError(field.into())),
    }
}

/// `value[index]` - read a list element.
#[inline]
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
#[inline]
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
#[inline]
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

#[inline]
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
