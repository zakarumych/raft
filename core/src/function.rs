// ---------------------------------------------------------------------
// Function/RcFn: Val::Fn's backing. `raft-core`'s own `Function` trait
// (partial application, min/max args) bridges into `rc::Callable`
// (the minimal bound `rc::fn_vtable`/`DynRc::erase_fn` need), same
// relationship the old `Function`/`DynFn` split had - just retargeted
// at `*mut ffi::RawHost` instead of `&mut dyn Host`.
//
// Note: unlike the old design, there is no `call_once`/`Rc::try_unwrap`
// fast path here - safely replicating "move captured state instead of
// cloning it" through a fully type-erased `DynRc<FnVTable, Void>` would
// need a dedicated "take" vtable slot (deallocation must stay owned by
// `destroy`, so a `call` shim can't `try_unwrap`-and-deallocate without
// risking a double-free at the outer `Drop`). `PartialFn` always clones
// its captured args on the shared path instead.
// ---------------------------------------------------------------------

use core::{fmt, marker::PhantomData, mem::ManuallyDrop, ptr::NonNull};

use raft_ffi::{RcPtr, Void};
use smallvec::SmallVec;

use crate::{
    cast_rc_inner, erase_rc_inner,
    host::Host,
    rc::{DynRc, Rc},
    val::{Val, ValEnum},
    vtable::any_vtable,
};

/// A callable value: `fn`-defined functions (AST-walked or compiled to
/// bytecode), partially-applied functions, and host-provided closures all
/// implement this.
///
/// Callers must supply at least [`Function::min_args`] arguments - the
/// generic dispatch in [`call_dispatch`]-via-[`rc::Callable`] handles
/// partial application *before* calling [`Function::call`], so
/// implementations never see an underfull call. `host` is a safe view -
/// nothing in this trait, or anything implementing it, ever touches a raw
/// pointer; that's confined to `rc::call_shim` and [`RcFn::call`].
pub trait Function: Sized + 'static {
    /// Minimum number of arguments this function consumes in a call. If
    /// fewer are supplied, the dispatch returns a partially-applied
    /// function value instead of calling.
    fn min_args(&self) -> usize;

    fn max_args(&self) -> Option<usize> {
        None
    }

    /// Push exactly one result. Caller (the generic dispatch below)
    /// guarantees `args` is between `min_args` and (if any) `max_args`.
    fn call(&self, host: &mut Host, args: usize);

    fn debug_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<fn>")
    }
}

/// Minimal callable bound for building a per-concrete-type
/// [`raft_ffi::FnVTable`] - deliberately smaller than `raft-core`'s own
/// `Function` trait (no min/max-args or partial-application semantics
/// here, just "invoke, given a host and an argument count, returning how
/// many were actually consumed"). `Function` bridges into this.
///
/// Receives the whole `RcInner<Self>` *box* pointer, not `&self`: the
/// dispatch needs to touch the box's `strong` count (cloning the callee
/// into a partial-application value), and a pointer derived from a `&Self`
/// reference has provenance for the value field only - walking back to
/// the header through it is undefined behavior that optimizers really do
/// exploit (dropping the refcount bump entirely).
pub trait Callable: 'static + Sized {
    /// Dispatch a call on the value inside `this`.
    ///
    /// # Safety
    /// `this` must be a live `RcInner<Self>` box carrying provenance for
    /// the whole allocation, whose strong count is kept alive by the
    /// caller for the duration of the call.
    unsafe fn call_raw(this: RcPtr<Self>, args: usize, host: &mut Host) -> usize;
}

impl<F: Function> Callable for F {
    #[inline]
    unsafe fn call_raw(this: RcPtr<F>, args: usize, host: &mut Host) -> usize {
        // SAFETY: `this` is the live, whole-provenance `RcInner<F>` box
        // (caller's contract) - viewing it as a borrowed `Rc<F>` (in
        // `ManuallyDrop`, so no double-decrement) lets `call_dispatch`
        // clone it into a partial-application value when needed.
        let rc = ManuallyDrop::new(unsafe { Rc::<F>::from_raw_box(this.cast()) });
        call_dispatch(&rc, host, args)
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
    F: Fn(&mut Host, usize) -> Val + 'static,
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
    fn call(&self, host: &mut Host, args: usize) {
        debug_assert!(args >= self.min_args);
        let ret = (self.fun)(host, args);
        host.stack().push(ret);
    }

    #[inline]
    fn debug_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.max_args {
            Some(max_args) => write!(f, "<fn {}..={}>", self.min_args, max_args),
            None => write!(f, "<fn {}..>", self.min_args),
        }
    }
}

/// A function with some arguments already applied, waiting for the rest.
/// Wraps the callee as an already-type-erased [`RcFn`] rather than a
/// generic `Rc<F>` - re-partial-applying a `PartialFn` would otherwise
/// nest `PartialFn<PartialFn<PartialFn<..>>>` without bound, and since
/// `call_dispatch`'s own body (reachable, regardless of whether it's ever
/// taken at runtime) constructs one more nesting level, that blows the
/// compiler's monomorphization recursion limit. Going through `RcFn::call`
/// costs one extra `extern "C"` dispatch per incremental partial
/// application instead.
struct PartialFn {
    fun: RcFn,
    min_args: usize,
    max_args: Option<usize>,
    preapplied: SmallVec<[Val; 4]>,
}

impl Function for PartialFn {
    #[inline]
    fn min_args(&self) -> usize {
        self.min_args
    }

    #[inline]
    fn max_args(&self) -> Option<usize> {
        self.max_args
    }

    fn call(&self, host: &mut Host, args: usize) {
        let len = self.preapplied.len();
        host.stack().extend_from_slice(&self.preapplied);
        self.fun.call(host, len + args);
    }

    fn debug_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<partial ")?;
        for arg in self.preapplied.iter() {
            write!(f, "{:?} ", arg)?;
        }
        write!(f, "<fn>>")
    }
}

#[repr(transparent)]
#[derive(Clone)]
pub struct RcFn {
    ptr: DynRc<raft_ffi::FnVTable, Void>,
}

impl RcFn {
    #[inline]
    pub fn new<F: Function>(f: F) -> Self {
        Self::from_rc(Rc::new(f))
    }

    #[inline]
    pub unsafe fn from_raw(ptr: DynRc<raft_ffi::FnVTable, Void>) -> Self {
        RcFn { ptr }
    }

    pub fn into_raw(self) -> DynRc<raft_ffi::FnVTable, Void> {
        self.ptr
    }

    #[inline]
    pub fn from_rc<F: Function>(rc: Rc<F>) -> Self {
        RcFn { ptr: erase_fn(rc) }
    }

    /// Raw identity of the underlying allocation, for callers that need
    /// to compare "is this the same function value" (a self-recursion
    /// fast path, say) without any dispatch overhead. Not meaningful
    /// across two different `Val::Fn`-producing calls unless both sides
    /// agree on how they derived their pointer (see call sites).
    #[inline]
    pub fn as_ptr(&self) -> NonNull<()> {
        DynRc::as_ptr(&self.ptr).cast()
    }

    /// Dispatch a call, pushing exactly one result and returning how many
    /// of `args` were actually consumed (fewer than `args` if this value
    /// only partially applies, in which case a new partial-application
    /// `Val::Fn` was pushed instead of a result).
    #[inline]
    pub fn call(&self, host: &mut Host, args: usize) -> usize {
        let vtable = DynRc::vtable(&self.ptr);
        // SAFETY: crossing into the `extern "C"` `CallFn` ABI - the one
        // place a raw pointer is unavoidable; `host.as_raw()` is still
        // the exact valid, exclusively-borrowed `RawHost` `host` wraps.
        unsafe {
            (vtable.call)(
                erase_rc_inner(DynRc::rc_ptr(&self.ptr)),
                args,
                host.as_raw(),
            )
        }
    }
}

impl fmt::Debug for RcFn {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(_f, "<fn>")
    }
}

fn call_dispatch<F: Function>(rc: &Rc<F>, host: &mut Host, args: usize) -> usize {
    let min_args = rc.min_args();
    if args < min_args {
        if args == 0 {
            host.stack()
                .push(Val::from(ValEnum::Fn(RcFn::from_rc(rc.clone()))));
            return 0;
        }
        let mut preapplied: SmallVec<[Val; 4]> =
            smallvec::smallvec![Val::from(ValEnum::Uninit); args];
        host.stack().drain_top_into(&mut preapplied);
        let partial = Rc::new(PartialFn {
            fun: RcFn::from_rc(rc.clone()),
            min_args: min_args - args,
            max_args: rc.max_args().map(|max| max - args),
            preapplied,
        });
        host.stack()
            .push(Val::from(ValEnum::Fn(RcFn::from_rc(partial))));
        return args;
    }

    match rc.max_args() {
        Some(max_args) if max_args < args => {
            rc.call(host, max_args);
            max_args
        }
        _ => {
            rc.call(host, args);
            args
        }
    }
}

/// One shared, monomorphized `&'static FnVTable` per concrete `F` - same
/// idea as [`any_vtable`], plus the `call` shim `Fn` values need.
pub fn fn_vtable<F: Callable>() -> &'static raft_ffi::FnVTable {
    struct Holder<F>(PhantomData<F>);
    impl<F: Callable> Holder<F> {
        const VTABLE: raft_ffi::FnVTable = raft_ffi::FnVTable {
            any: any_vtable::<F>(),
            call: call_shim::<F>,
        };
    }
    &Holder::<F>::VTABLE
}

/// Same idea as [`erase`], but for the `Fn` tag: `F` gets its own
/// monomorphized [`raft_ffi::FnVTable`] (not the bare `AnyVTable`
/// `Opaque` uses), since each concrete `Function` implementor needs its
/// own `call` dispatch, not just `destroy`. Takes an already-built
/// `Rc<F>` (not a fresh value) so callers that need to reuse an
/// existing, possibly-shared reference (cloning it into a
/// partial-application wrapper, say) don't pay for a second allocation.
pub fn erase_fn<F: Callable>(rc: Rc<F>) -> DynRc<raft_ffi::FnVTable, Void> {
    unsafe { DynRc::new(Rc::into_raw_box(rc).cast(), fn_vtable::<F>()) }
}

unsafe extern "C" fn call_shim<F: Callable>(
    data: raft_ffi::RcPtr<Void>,
    args: usize,
    host: *mut raft_ffi::RawHost,
) -> usize {
    // SAFETY: `host`, per `raft_ffi::CallFn`'s contract, is a valid,
    // exclusively-held `RawHost` for the duration of this call - this is
    // the one place that raw pointer gets turned into a reference.
    let mut host = unsafe { Host::from_raw(host) };
    // SAFETY: derived from `data` per the contract above; non-null since
    // `data` was.
    unsafe { F::call_raw(cast_rc_inner(data), args, &mut host) }
}

impl Val {
    /// Wrap a host closure into a function value with the given
    /// argument-count hint. `(0, None)` means "takes anything" - the
    /// closure then decides how many arguments to consume.
    #[inline]
    pub fn host_function<F>(min_args: usize, max_args: Option<usize>, f: F) -> Val
    where
        F: Fn(&mut Host, usize) -> Val + 'static,
    {
        Val::from(ValEnum::Fn(RcFn::new(HostFn {
            min_args,
            max_args,
            fun: f,
        })))
    }
}
