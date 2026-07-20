use core::{fmt, marker::PhantomData, mem::ManuallyDrop, ptr::NonNull};

use raft_ffi::{
    CORO_DONE, CORO_KIND_ASYNC, CORO_KIND_ASYNC_GEN, CORO_KIND_GEN, CORO_PENDING, CORO_YIELD,
    CoroHeader, RcPtr, Void,
};

use crate::{
    cast_rc_inner, erase_rc_inner,
    host::Host,
    rc::{DynRc, Rc},
    vtable::any_vtable,
};

/// What flavor of coroutine a [`RcCoro`] is - read straight off the
/// value's FFI-safe [`CoroHeader`] prefix. Decides the protocol its
/// consumer holds it to (see [`CoroStatus`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoroKind {
    /// A `gen fn`'s coroutine: iteration expects `Yielded`s ended by one
    /// `Done`; `Pending` is a protocol error.
    Gen,
    /// An `async fn`'s coroutine: awaiting expects `Pending`s (propagated
    /// to the task) until one `Yielded` delivers the resolution, then one
    /// final `Done`; `Done` before any yield is a protocol error (unless
    /// the host error state says why).
    Async,
    /// An `async gen fn`'s coroutine - the combination: `Yielded` per
    /// produced value, `Pending` when it must wait (propagated to the
    /// driving task), `Done` at the end. Only iterable from an async
    /// context, where pending can suspend the consumer.
    AsyncGen,
}

impl CoroKind {
    #[inline]
    pub fn to_u8(self) -> u8 {
        match self {
            CoroKind::Gen => CORO_KIND_GEN,
            CoroKind::Async => CORO_KIND_ASYNC,
            CoroKind::AsyncGen => CORO_KIND_ASYNC_GEN,
        }
    }

    #[inline]
    pub fn from_u8(byte: u8) -> Option<CoroKind> {
        match byte {
            CORO_KIND_GEN => Some(CoroKind::Gen),
            CORO_KIND_ASYNC => Some(CoroKind::Async),
            CORO_KIND_ASYNC_GEN => Some(CoroKind::AsyncGen),
            _ => None,
        }
    }
}

/// What one [`Coroutine::resume`] step did - the typed view of the FFI
/// status byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoroStatus {
    /// Finished: nothing was pushed, resuming again keeps returning this.
    /// A failed coroutine also finishes this way, with the host's
    /// pending-error state set.
    Done,
    /// Pushed exactly one value onto the host stack; may be resumed again.
    Yielded,
    /// Must wait before it can continue - resume after the waker (cloned
    /// from the host's ambient one) signals. An immediate re-resume is
    /// allowed but may keep coming back `Pending`. Nothing was pushed.
    Pending,
}

impl CoroStatus {
    #[inline]
    pub fn to_u8(self) -> u8 {
        match self {
            CoroStatus::Done => CORO_DONE,
            CoroStatus::Yielded => CORO_YIELD,
            CoroStatus::Pending => CORO_PENDING,
        }
    }

    #[inline]
    fn from_u8(byte: u8) -> CoroStatus {
        match byte {
            CORO_YIELD => CoroStatus::Yielded,
            CORO_PENDING => CoroStatus::Pending,
            _ => CoroStatus::Done,
        }
    }
}

/// One coroutine object - what calling a `gen fn`/`async fn` with a full
/// argument set returns, and what host-provided suspendable leaves
/// implement. The kind ([`CoroKind`], stamped into the value's header at
/// wrap time) decides the consumer-side protocol; `resume` is the single
/// execution entry point for every kind. The waker of the poll currently
/// driving the host is ambient ([`rc::Host::waker`]) - a leaf that comes
/// back `Pending` clones it out and wakes it when it can continue.
pub trait Coroutine: Sized + 'static {
    /// Advance the coroutine, with `args` arguments on the stack (0 for
    /// today's generator and async kinds). See [`CoroStatus`] for what
    /// each result means and leaves on the stack.
    fn resume(&self, host: &mut Host, args: usize) -> CoroStatus;

    fn debug_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<coro>")
    }
}

// ---------------------------------------------------------------------
// CoroBox/RcCoro: Val::Coro's backing. Mirrors the Function/RcFn pair:
// `Coroutine` is the public trait coroutine objects implement, boxed
// together with the FFI-visible `CoroHeader` (the kind byte) in a
// `CoroBox` that bridges into `rc::Resumable` (the minimal bound
// `rc::coro_vtable`/`erase_coro` need). A coroutine object is what
// calling a `gen fn`/`async fn` with a full argument set returns; each
// `resume` runs the body to its next suspension point and reports how
// it stopped as a `CoroStatus` byte.
// ---------------------------------------------------------------------

/// Minimal resumable bound for building a per-concrete-type
/// [`raft_ffi::CoroVTable`] - the [`Callable`] of coroutine objects.
/// `raft-core`'s own `Coroutine` trait bridges into this (through its
/// header-carrying `CoroBox` wrapper).
pub trait Resumable: 'static + Sized {
    /// Resume the coroutine inside `this` with `args` arguments on the
    /// stack (0 for today's generator and async kinds), returning the raw
    /// status byte ([`raft_ffi::CORO_DONE`]/[`raft_ffi::CORO_YIELD`]/
    /// [`raft_ffi::CORO_PENDING`]).
    ///
    /// # Safety
    /// As [`Callable::call_raw`]'s.
    unsafe fn resume_raw(this: RcPtr<Self>, args: usize, host: &mut Host) -> u8;
}

/// The heap payload behind every `Val::Coro`: the FFI-safe [`CoroHeader`]
/// first (so foreign code can read the kind byte without knowing the
/// concrete `C`), then the actual coroutine state.
#[repr(C)]
struct CoroBox<C> {
    header: CoroHeader,
    inner: C,
}

impl<C: Coroutine> Resumable for CoroBox<C> {
    #[inline]
    unsafe fn resume_raw(this: raft_ffi::RcPtr<Self>, args: usize, host: &mut Host) -> u8 {
        // SAFETY: `this` is the live, whole-provenance `RcInner<CoroBox<C>>`
        // box (caller's contract) - viewing it as a borrowed `Rc` (in
        // `ManuallyDrop`, so no double-decrement) for the call's duration.
        let rc = ManuallyDrop::new(unsafe { Rc::<CoroBox<C>>::from_raw_box(this.cast()) });
        rc.inner.resume(host, args).to_u8()
    }
}

#[repr(transparent)]
#[derive(Clone)]
pub struct RcCoro {
    ptr: DynRc<raft_ffi::CoroVTable, CoroHeader>,
}

impl RcCoro {
    #[inline]
    pub fn new<C: Coroutine>(kind: CoroKind, c: C) -> Self {
        RcCoro {
            ptr: erase_coro(Rc::new(CoroBox {
                header: CoroHeader { kind: kind.to_u8() },
                inner: c,
            })),
        }
    }

    #[inline]
    pub unsafe fn from_raw(ptr: DynRc<raft_ffi::CoroVTable, CoroHeader>) -> Self {
        RcCoro { ptr }
    }

    pub fn into_raw(self) -> DynRc<raft_ffi::CoroVTable, CoroHeader> {
        self.ptr
    }

    pub fn as_ptr(&self) -> NonNull<()> {
        DynRc::as_ptr(&self.ptr).cast()
    }

    /// The kind stamped into the value's header at creation. Decides which
    /// resume protocol callers hold this coroutine to (see [`CoroStatus`]).
    #[inline]
    pub fn kind(&self) -> Option<CoroKind> {
        // SAFETY: the data pointer is the live `CoroBox<C>` payload, whose
        // `repr(C)` layout starts with the `CoroHeader` regardless of `C`.
        let kind = self.ptr.kind;
        CoroKind::from_u8(kind)
    }

    /// Resume the coroutine. `args` is how many stack-top values it
    /// consumes as resume arguments (0 for both current kinds). The
    /// returned [`CoroStatus`] says how it stopped: [`CoroStatus::Yielded`]
    /// pushed exactly one value onto the host stack; [`CoroStatus::Done`]
    /// and [`CoroStatus::Pending`] pushed nothing. A failure sets the
    /// host's pending-error state and reports `Done` - callers check that
    /// state before treating `Done` as a clean end.
    #[inline]
    pub fn resume(&self, host: &mut Host, args: usize) -> CoroStatus {
        let vtable = DynRc::vtable(&self.ptr);
        // SAFETY: crossing into the `extern "C"` `CoroResumeFn` ABI;
        // `host.as_raw()` is the exact valid, exclusively-borrowed
        // `RawHost` `host` wraps.
        let status = unsafe {
            (vtable.resume)(
                erase_rc_inner(DynRc::rc_ptr(&self.ptr)),
                args,
                host.as_raw(),
            )
        };
        CoroStatus::from_u8(status)
    }
}

impl fmt::Debug for RcCoro {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind() {
            Some(CoroKind::Gen) => write!(f, "<gen>"),
            Some(CoroKind::Async) => write!(f, "<async>"),
            Some(CoroKind::AsyncGen) => write!(f, "<async gen>"),
            None => write!(f, "<coro>"),
        }
    }
}

/// One shared, monomorphized `&'static CoroVTable` per concrete `C` -
/// same idea as [`fn_vtable`], with `resume` in place of `call`.
pub fn coro_vtable<C: Resumable>() -> &'static raft_ffi::CoroVTable {
    struct Holder<C>(PhantomData<C>);
    impl<C: Resumable> Holder<C> {
        const VTABLE: raft_ffi::CoroVTable = raft_ffi::CoroVTable {
            any: any_vtable::<C>(),
            resume: resume_shim::<C>,
        };
    }
    &Holder::<C>::VTABLE
}

/// As [`erase_fn`], for the `Coro` tag: `C` gets its own monomorphized
/// [`raft_ffi::CoroVTable`] whose `resume` shim dispatches to this exact
/// coroutine type. `C`'s payload must start with a
/// [`raft_ffi::CoroHeader`] (see `raft-core`'s `CoroBox`) - consumers
/// read the kind straight off the value.
pub fn erase_coro<C: Resumable>(rc: Rc<C>) -> DynRc<raft_ffi::CoroVTable, CoroHeader> {
    unsafe { DynRc::new(Rc::into_raw_box(rc).cast(), coro_vtable::<C>()) }
}

unsafe extern "C" fn resume_shim<C: Resumable>(
    data: raft_ffi::RcPtr<Void>,
    args: usize,
    host: *mut raft_ffi::RawHost,
) -> u8 {
    // SAFETY: `host`, per `raft_ffi::CoroResumeFn`'s contract, is a valid,
    // exclusively-held `RawHost` for the duration of this call.
    let mut host = unsafe { Host::from_raw(host) };
    // SAFETY: derived from `data` per the contract above; non-null since
    // `data` was.
    unsafe { C::resume_raw(cast_rc_inner(data), args, &mut host) }
}
