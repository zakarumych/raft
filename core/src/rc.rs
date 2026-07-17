//! Refcounted box used throughout `raft-core`, in place of `alloc::rc::Rc`.
//! Strong-count-only (no `Weak` - unused anywhere in this codebase), and
//! built directly on [`ffi::RcBox`] (not a locally duplicated header type)
//! so [`Rc::into_raw_box`]/[`Rc::from_raw_box`] are plain `NonNull::cast`s
//! of the existing allocation into/out of an [`ffi::FFIRc`]'s `data` field
//! - no new allocation, no offset math, no `c_void`.

use alloc::{
    alloc::{Layout, alloc, dealloc, handle_alloc_error, realloc},
    boxed::Box,
};
use core::{cell::Cell, marker::PhantomData, ops::Deref, ptr::NonNull, sync::atomic::Ordering};

use raft_ffi::{RawVal, RawVec, RcInner, RcPtr, Void};

use crate::{AnyVTable, Val};

pub struct Rc<T> {
    ptr: RcPtr<T>,
}

impl<T> Rc<T> {
    pub fn new(value: T) -> Self {
        let ptr = Box::into_raw(Box::new(RcInner {
            strong: Cell::new(1),
            value,
        }));

        Rc {
            ptr: unsafe { RcPtr::from(&mut *ptr) },
        }
    }

    pub fn rc_box(&self) -> &RcInner<T> {
        // SAFETY: `self.ptr` is always a live allocation.
        unsafe { self.ptr.as_ref() }
    }

    #[inline]
    pub fn as_ptr(this: &Self) -> *const T {
        &this.rc_box().value
    }

    #[inline]
    pub fn strong_count(this: &Self) -> usize {
        this.rc_box().strong.get()
    }

    #[inline]
    pub fn ptr_eq(a: &Self, b: &Self) -> bool {
        core::ptr::eq(a.ptr.as_ptr(), b.ptr.as_ptr())
    }

    #[inline]
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        if Rc::strong_count(this) == 1 {
            // SAFETY: unique owner, exclusive borrow via `this` is sound.
            Some(unsafe { &mut this.ptr.as_mut().value })
        } else {
            None
        }
    }

    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        if Rc::strong_count(&this) == 1 {
            // SAFETY: unique owner - read the value out, then free the
            // allocation without running `T`'s destructor a second time.
            let value = unsafe { core::ptr::read(&this.ptr.as_ref().value) };

            unsafe {
                dealloc(this.ptr.as_ptr() as *mut u8, Layout::new::<RcInner<T>>());
            }

            Ok(value)
        } else {
            Err(this)
        }
    }

    /// # Safety
    /// `ptr` must have been produced by `Rc::<T>::into_raw_box` on a value
    /// of this exact `T`, and must not be used to construct more live
    /// `Rc<T>`s than the allocation's strong count actually backs.
    #[inline]
    pub unsafe fn from_raw_box(ptr: NonNull<RcInner<()>>) -> Self {
        Rc {
            ptr: ptr.cast::<RcInner<T>>(),
        }
    }

    /// Consume `this` without dropping, returning the whole allocation
    /// (header included) narrowed to a type-erased `NonNull<RcInner<()>>`
    /// - the shape [`DynRc`]/opaque vtable-backed types store. No new
    /// allocation. Pair with [`from_raw_box`](Rc::from_raw_box).
    #[inline]
    pub fn into_raw_box(this: Self) -> NonNull<RcInner<()>> {
        let ptr = this.ptr.cast::<RcInner<()>>();
        core::mem::forget(this);
        ptr
    }

    /// Consume `this` without dropping, returning a pointer to just the
    /// inner value (skipping the header). Pair with
    /// [`from_raw`](Rc::from_raw).
    #[inline]
    pub fn into_raw(this: Self) -> *const T {
        let ptr = Rc::as_ptr(&this);
        core::mem::forget(this);
        ptr
    }

    /// # Safety
    /// `ptr` must have been produced by `Rc::<T>::into_raw`/`Rc::<T>::as_ptr`
    /// on a value of this exact `T`, and must not be used to construct
    /// more live `Rc<T>`s than the allocation's strong count actually
    /// backs.
    #[inline]
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        let offset = core::mem::offset_of!(RcInner<T>, value);
        // SAFETY: `ptr` points at the `value` field of a live `RcInner<T>`.
        let box_ptr = unsafe { (ptr as *const u8).sub(offset) } as *mut RcInner<T>;
        Rc {
            // SAFETY: derived from a non-null `ptr`.
            ptr: unsafe { NonNull::new_unchecked(box_ptr) },
        }
    }
}

impl<T> Clone for Rc<T> {
    #[inline]
    fn clone(&self) -> Self {
        // SAFETY: `self.ptr` is always a live allocation.
        let strong = unsafe { &self.ptr.as_ref().strong };
        strong.set(strong.get() + 1);
        Rc { ptr: self.ptr }
    }
}

impl<T> Deref for Rc<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: `self.ptr` is always a live allocation.
        unsafe { &self.ptr.as_ref().value }
    }
}

impl<T> Drop for Rc<T> {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` is always a live allocation until freed below.
        let strong = unsafe { &self.ptr.as_ref().strong };
        let count = strong.get() - 1;
        strong.set(count);
        if count == 0 {
            unsafe {
                core::ptr::drop_in_place(&mut self.ptr.as_mut().value);
                dealloc(self.ptr.as_ptr() as *mut u8, Layout::new::<RcInner<T>>());
            }
        }
    }
}

pub struct DynRc<V: AnyVTable, T> {
    ptr: RcPtr<T>,
    vtable: &'static V,
}

impl<V, T> DynRc<V, T>
where
    V: AnyVTable,
{
    #[inline(always)]
    pub unsafe fn new(ptr: RcPtr<T>, vtable: &'static V) -> Self {
        DynRc { ptr, vtable }
    }

    #[inline(always)]
    pub fn rc_box(&self) -> &RcInner<T> {
        // SAFETY: `self.ptr` is always a live allocation.
        unsafe { self.ptr.as_ref() }
    }

    #[inline(always)]
    pub fn as_ptr(this: &Self) -> *const T {
        &this.rc_box().value
    }

    #[inline(always)]
    pub fn strong_count(this: &Self) -> usize {
        this.rc_box().strong.get()
    }

    #[inline(always)]
    pub fn ptr_eq(a: &Self, b: &Self) -> bool {
        core::ptr::eq(a.ptr.as_ptr(), b.ptr.as_ptr())
    }

    #[inline(always)]
    pub fn data_ptr(this: &Self) -> RcPtr<T> {
        this.ptr
    }

    #[inline(always)]
    pub fn vtable(this: &Self) -> &'static V {
        this.vtable
    }
}

impl<V, T> Deref for DynRc<V, T>
where
    V: AnyVTable,
{
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: `self.ptr` is always a live allocation.
        &self.rc_box().value
    }
}

impl<V, T> Clone for DynRc<V, T>
where
    V: AnyVTable,
{
    #[inline]
    fn clone(&self) -> Self {
        // SAFETY: `self.ptr` is always a live allocation.
        let strong = unsafe { &self.ptr.as_ref().strong };
        strong.set(strong.get() + 1);
        DynRc {
            ptr: self.ptr,
            vtable: self.vtable,
        }
    }
}

impl<V, T> Drop for DynRc<V, T>
where
    V: AnyVTable,
{
    fn drop(&mut self) {
        // SAFETY: `self.ptr` is always a live allocation until freed below.
        let strong = unsafe { &self.ptr.as_ref().strong };
        let count = strong.get() - 1;
        strong.set(count);
        if count == 0 {
            let vtable = self.vtable.any();
            unsafe {
                (vtable.destroy)(self.ptr.cast());
            }
        }
    }
}

/// Build a `DynRc` around a freshly-allocated `T`, type-erasing it behind
/// a monomorphized (one static instance per concrete `T`)
/// [`raft_ffi::AnyVTable`] whose `destroy` knows how to tear this exact
/// `T` down. For `Opaque` values: `raft-core` never learns the concrete
/// type back, only reaches it through `vtable`.
///
/// (A free function, not a `DynRc` method: `DynRc<V, T>`'s own `V`/`T`
/// are unrelated to the `T` being erased here, so attaching it to that
/// `impl` block just makes every call site need a turbofish to pin down
/// otherwise-unconstrained parameters.)
pub fn erase<T: 'static>(value: T) -> DynRc<raft_ffi::AnyVTable, Void> {
    let rc = Rc::new(value);
    DynRc {
        ptr: Rc::into_raw_box(rc).cast::<RcInner<Void>>(),
        vtable: any_vtable::<T>(),
    }
}

/// Same idea as [`erase`], but for the `Fn` tag: `F` gets its own
/// monomorphized [`raft_ffi::FnVTable`] (not the bare `AnyVTable`
/// `Opaque` uses), since each concrete `Function` implementor needs its
/// own `call` dispatch, not just `destroy`. Takes an already-built
/// `Rc<F>` (not a fresh value) so callers that need to reuse an
/// existing, possibly-shared reference (cloning it into a
/// partial-application wrapper, say) don't pay for a second allocation.
pub fn erase_fn<F: Callable>(rc: Rc<F>) -> DynRc<raft_ffi::FnVTable, Void> {
    DynRc {
        ptr: Rc::into_raw_box(rc).cast::<RcInner<Void>>(),
        vtable: fn_vtable::<F>(),
    }
}

/// As [`erase_fn`], for the `Coro` tag: `C` gets its own monomorphized
/// [`raft_ffi::CoroVTable`] whose `resume` shim dispatches to this exact
/// coroutine type. `C`'s payload must start with a
/// [`raft_ffi::CoroHeader`] (see `raft-core`'s `CoroBox`) - consumers
/// read the kind straight off the value.
pub fn erase_coro<C: Resumable>(rc: Rc<C>) -> DynRc<raft_ffi::CoroVTable, Void> {
    DynRc {
        ptr: Rc::into_raw_box(rc).cast::<RcInner<Void>>(),
        vtable: coro_vtable::<C>(),
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

unsafe extern "C" fn resume_shim<C: Resumable>(
    data: raft_ffi::VoidPtr,
    args: usize,
    host: *mut raft_ffi::RawHost,
) -> u8 {
    // SAFETY: as `call_shim`'s - `data` points at the `value` field of a
    // live `RcInner<C>` with whole-box provenance.
    let offset = core::mem::offset_of!(RcInner<C>, value);
    let box_ptr = unsafe { data.as_ptr().cast::<u8>().sub(offset) } as *mut RcInner<C>;
    // SAFETY: `host`, per `raft_ffi::CoroResumeFn`'s contract, is a valid,
    // exclusively-held `RawHost` for the duration of this call.
    let mut host = unsafe { Host::from_raw(host) };
    // SAFETY: derived from `data` per the contract above; non-null since
    // `data` was.
    unsafe { C::resume_raw(NonNull::new_unchecked(box_ptr), args, &mut host) }
}

unsafe extern "C" fn call_shim<F: Callable>(
    data: raft_ffi::VoidPtr,
    args: usize,
    host: *mut raft_ffi::RawHost,
) -> usize {
    // SAFETY: `data` points at the `value` field of a live `RcInner<F>`
    // and carries whole-box provenance (see `RcFn::call`/`Val::call_as_fn`,
    // which derive it via raw place projection, never through a
    // reference) - stepping back to the box start stays in bounds.
    let offset = core::mem::offset_of!(RcInner<F>, value);
    let box_ptr = unsafe { data.as_ptr().cast::<u8>().sub(offset) } as *mut RcInner<F>;
    // SAFETY: `host`, per `raft_ffi::CallFn`'s contract, is a valid,
    // exclusively-held `RawHost` for the duration of this call - this is
    // the one place that raw pointer gets turned into a reference.
    let mut host = unsafe { Host::from_raw(host) };
    // SAFETY: derived from `data` per the contract above; non-null since
    // `data` was.
    unsafe { F::call_raw(NonNull::new_unchecked(box_ptr), args, &mut host) }
}

/// Safe view over a host's operand stack - the only thing
/// `Function`/`Callable` implementors ever need. Never exposes the raw
/// `*mut ffi::RawHost` it was built from as part of its public API; every
/// method here derives a reference from it internally and is safe to
/// call. (`as_raw` is `pub(crate)`, not public - only [`crate::RcFn::call`]
/// needs it, to cross back into the `extern "C"` `CallFn` ABI, which is
/// the one place a raw pointer is unavoidable.)
pub struct Host<'a> {
    raw: &'a mut raft_ffi::RawHost,
}

impl<'a> Host<'a> {
    /// # Safety
    /// `raw` must point at a valid `RawHost`, exclusively borrowed for `'a`.
    #[inline(always)]
    pub unsafe fn from_raw(raw: *mut raft_ffi::RawHost) -> Self {
        Host {
            raw: unsafe { &mut *raw },
        }
    }

    /// Recover the raw pointer this `Host` wraps. An escape hatch: a
    /// concrete host implementation (`raft-runtime`'s `Runtime`, say,
    /// arranging its own layout so it *is* a `RawHost` at heart, and
    /// casting `&mut self` to build a `Host` in the first place) is the
    /// only sound caller - it can cast this back to its own type. Ordinary
    /// `Function` implementations never need this; `Host`'s other methods
    /// cover everything they should touch.
    #[inline(always)]
    pub fn as_raw(&mut self) -> *mut raft_ffi::RawHost {
        self.raw
    }

    #[inline(always)]
    pub fn stack(&mut self) -> Stack<'_> {
        // SAFETY: struct invariant (`from_raw`'s contract).
        Stack {
            raw: &mut self.raw.stack,
        }
    }

    /// The waker of the poll currently driving this host, cloned into an
    /// owned handle - `None` outside any poll. A leaf async value stores
    /// this and wakes it when its result becomes available.
    pub fn waker(&self) -> Option<FfiWaker> {
        NonNull::new(self.raw.waker).map(|ptr| {
            // SAFETY: a non-null `RawHost::waker` is a live waker for the
            // duration of the poll currently on the stack (the executor's
            // contract); cloning through its vtable yields an owned
            // reference independent of that window.
            unsafe {
                let w = ptr.as_ref();
                w.strong.fetch_add(1, Ordering::Relaxed);
            }
            FfiWaker { ptr }
        })
    }

    /// A real `core::task::Waker` for the current poll (a noop waker
    /// outside any poll) - what adapting an ordinary Rust future needs.
    pub fn rust_waker(&self) -> core::task::Waker {
        match self.waker() {
            Some(w) => w.into_waker(),
            None => core::task::Waker::noop().clone(),
        }
    }
}

// ---------------------------------------------------------------------
// FfiWaker: an owned reference to a host waker crossing the FFI boundary
// as a thin pointer (vtable in the allocation's prefix - see raft-ffi's
// async section). Adapting to `core::task::Waker` is a thin round trip:
// the RawWaker data pointer IS the header pointer.
//
// Single-threaded contract: despite `Waker`'s nominal `Send + Sync`, a
// host waker must only be cloned/woken/dropped on the host's own thread.
// ---------------------------------------------------------------------

pub struct FfiWaker {
    ptr: raft_ffi::WakerPtr,
}

impl FfiWaker {
    /// # Safety
    /// `ptr` must be an owned reference to a live FFI waker (its vtable's
    /// `drop` must be safe to call exactly once for it).
    #[inline]
    pub unsafe fn from_raw(ptr: raft_ffi::WakerPtr) -> Self {
        FfiWaker { ptr }
    }

    /// Hand the owned reference across a raw boundary - pair with
    /// [`FfiWaker::from_raw`], or the vtable's own `wake`/`drop`.
    #[inline]
    pub fn into_raw(self) -> raft_ffi::WakerPtr {
        let ptr = self.ptr;
        core::mem::forget(self);
        ptr
    }

    /// The raw pointer behind this reference, without giving it up - for
    /// installing it as a host's ambient waker while still owning it.
    #[inline]
    pub fn as_raw(this: &Self) -> raft_ffi::WakerPtr {
        this.ptr
    }

    #[inline]
    fn vtable(&self) -> &raft_ffi::WakerVTable {
        // SAFETY: struct invariant - `ptr` is a live waker whose header
        // holds a valid, effectively-'static vtable.
        unsafe { &*(*self.ptr.as_ptr()).vtable }
    }

    /// Wake the task this waker belongs to.
    #[inline]
    pub fn wake(&self) {
        // SAFETY: struct invariant; `wake_by_ref` borrows.
        unsafe { (self.vtable().wake)(self.ptr) }
    }

    /// Adapt into a real `core::task::Waker` (consuming this reference):
    /// the `RawWaker` data pointer is the header pointer itself, and one
    /// static vtable forwards every operation - no allocation per hop.
    pub fn into_raw_waker(self) -> core::task::RawWaker {
        let ptr = self.into_raw();

        // SAFETY: `FFI_WAKER_VTABLE`'s fns uphold RawWaker's contract by
        // forwarding to the header's own vtable, which owns the reference
        // semantics.

        core::task::RawWaker::new(ptr.cast().as_ptr(), &FFI_WAKER_VTABLE)
    }

    /// Adapt into a real `core::task::Waker` (consuming this reference):
    /// the `RawWaker` data pointer is the header pointer itself, and one
    /// static vtable forwards every operation - no allocation per hop.
    pub fn into_waker(self) -> core::task::Waker {
        // SAFETY: `FFI_WAKER_VTABLE`'s fns uphold RawWaker's contract by
        // forwarding to the header's own vtable, which owns the reference
        // semantics.
        unsafe { core::task::Waker::from_raw(self.into_raw_waker()) }
    }
}

impl Clone for FfiWaker {
    fn clone(&self) -> Self {
        unsafe {
            let w = self.ptr.as_ref();
            w.strong.fetch_add(1, Ordering::Relaxed);
        }

        FfiWaker { ptr: self.ptr }
    }
}

impl Drop for FfiWaker {
    fn drop(&mut self) {
        let destroy = unsafe {
            let w = self.ptr.as_ref();
            w.strong.fetch_sub(1, Ordering::Release) == 1
        };

        if destroy {
            let destroy_fn = self.vtable().destroy;
            // SAFETY: struct invariant - this releases the one reference
            // `self` owned.
            unsafe { destroy_fn(self.ptr) }
        }
    }
}

static FFI_WAKER_VTABLE: core::task::RawWakerVTable = core::task::RawWakerVTable::new(
    ffi_waker_clone,
    ffi_waker_wake,
    ffi_waker_wake_by_ref,
    ffi_waker_drop,
);

/// # Safety (all four)
/// `data` is a `WakerPtr` previously released by [`FfiWaker::into_waker`]
/// or by `ffi_waker_clone` - a live, owned FFI waker reference.
unsafe fn ffi_waker_clone(data: *const ()) -> core::task::RawWaker {
    match NonNull::new(data as *mut ()) {
        None => core::task::RawWaker::new(core::ptr::null_mut(), &FFI_WAKER_VTABLE),
        Some(ptr) => unsafe {
            let ptr = ptr.cast::<raft_ffi::WakerHeader>();
            let w = ptr.as_ref();
            w.strong.fetch_add(1, Ordering::Relaxed);
            core::task::RawWaker::new(ptr.as_ptr() as *mut (), &FFI_WAKER_VTABLE)
        },
    }
}

unsafe fn ffi_waker_wake(data: *const ()) {
    unsafe {
        ffi_waker_wake_by_ref(data);
        ffi_waker_drop(data);
    }
}

unsafe fn ffi_waker_wake_by_ref(data: *const ()) {
    if let Some(ptr) = NonNull::new(data as *mut ()) {
        unsafe {
            let ptr = ptr.cast::<raft_ffi::WakerHeader>();
            let w = ptr.as_ref();
            let wake = (&*w.vtable).wake;
            wake(ptr);
        }
    }
}

unsafe fn ffi_waker_drop(data: *const ()) {
    if let Some(ptr) = NonNull::new(data as *mut ()) {
        unsafe {
            let ptr = ptr.cast::<raft_ffi::WakerHeader>();
            let (destroy, fun) = {
                let w = ptr.as_ref();
                (w.strong.fetch_sub(1, Ordering::Release) == 1, (&*w.vtable).destroy)
            };
            if destroy {
                fun(ptr);
            }
        }
    }
}

pub struct DrainIter<'a> {
    raw: &'a mut RawVec<RawVal>,
    start: usize,
    end: usize,
}

impl<'a> Drop for DrainIter<'a> {
    fn drop(&mut self) {
        for i in self.start..self.end {
            let raw = unsafe { self.raw.ptr.add(i) };
            unsafe { drop(Val::from_raw(core::ptr::read(raw))) };
        }
    }
}

impl<'a> Iterator for DrainIter<'a> {
    type Item = Val;

    fn next(&mut self) -> Option<Val> {
        if self.start < self.end {
            let raw = unsafe { self.raw.ptr.add(self.start) };
            self.start += 1;
            Some(Val::from_raw(unsafe { core::ptr::read(raw) }))
        } else {
            None
        }
    }
}

impl<'a> DoubleEndedIterator for DrainIter<'a> {
    fn next_back(&mut self) -> Option<Val> {
        if self.start < self.end {
            self.end -= 1;
            let raw = unsafe { self.raw.ptr.add(self.end) };
            Some(Val::from_raw(unsafe { core::ptr::read(raw) }))
        } else {
            None
        }
    }
}

impl<'a> ExactSizeIterator for DrainIter<'a> {
    fn len(&self) -> usize {
        self.end - self.start
    }
}

/// A borrowing view over `Host`'s own operand stack - `Host` embeds
/// a `ffi::RawStack` directly (see `Host`'s doc comment) rather than
/// owning a `Vec<Val>` field, so this reconstructs a real `Vec<Val>` (via
/// [`StackGuard`]) on each access rather than holding one permanently.
/// `Host::stack()` is the only way to get one.
pub struct Stack<'a> {
    raw: &'a mut raft_ffi::RawVec<RawVal>,
}

#[cold]
#[inline(never)]
fn stack_out_of_bounds() -> ! {
    panic!("Attempted to access a stack element beyond the current stack size");
}

impl<'a> Stack<'a> {
    #[doc(hidden)]
    pub unsafe fn new(raw: &'a mut raft_ffi::RawVec<RawVal>) -> Self {
        Stack { raw }
    }

    /// Reserve `n` not-yet-assigned locals on top of the stack.
    #[inline(always)]
    pub fn extend_uninit(&mut self, n: usize) {
        unsafe {
            raw_vec_reserve(self.raw, n);
            for _ in 0..n {
                raw_vec_push(self.raw, raw_val_uninit());
            }
        }
    }

    #[inline(always)]
    pub fn push(&mut self, v: Val) {
        unsafe {
            raw_vec_push(self.raw, v.into_raw());
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.raw.size
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.raw.size == 0
    }

    #[inline(always)]
    pub fn pop(&mut self) -> Val {
        match unsafe { raw_vec_pop(self.raw) } {
            None => stack_out_of_bounds(),
            Some(raw) => Val::from_raw(raw),
        }
    }

    #[inline(always)]
    pub fn peek(&self) -> &Val {
        match unsafe { raw_vec_last(self.raw) } {
            None => stack_out_of_bounds(),
            Some(raw) => Val::from_raw_ref(raw),
        }
    }

    #[inline]
    pub fn extend(&mut self, values: impl IntoIterator<Item = Val>) {
        let iter = values.into_iter();

        let reserve = match iter.size_hint() {
            (lower, Some(upper)) if (upper / 2) < lower => upper,
            (lower, Some(_upper)) => lower.next_power_of_two(),
            (lower, _) => lower,
        };

        unsafe { raw_vec_reserve(self.raw, reserve) };
        for v in iter {
            unsafe { raw_vec_push(self.raw, v.into_raw()) };
        }
    }

    /// Push a batch of values in order (the first ends up deepest),
    /// cloning each one out of `values`.
    #[inline]
    pub fn extend_from_slice(&mut self, values: &[Val]) {
        unsafe { raw_vec_reserve(self.raw, values.len()) };
        for v in values {
            unsafe { raw_vec_push(self.raw, v.clone().into_raw()) };
        }
    }

    #[inline(always)]
    pub fn reverse(&mut self, count: usize) {
        if count < 2 {
            return;
        }

        if self.raw.size < count {
            stack_out_of_bounds();
        }

        let start = self.raw.size - count;

        for i in 0..(count / 2) {
            let j = count - 1 - i;
            // SAFETY: `i`/`j` are both < `count`, which is <= `self.raw.size`
            // (caller's contract).
            unsafe {
                let pi = self.raw.ptr.add(start + i);
                let pj = self.raw.ptr.add(start + j);
                core::ptr::swap(pi, pj);
            }
        }
    }

    #[inline(always)]
    pub fn drain_top(&mut self, count: usize) -> DrainIter<'_> {
        if self.raw.size < count {
            stack_out_of_bounds();
        }

        self.raw.size -= count;

        DrainIter {
            start: self.raw.size,
            end: self.raw.size + count,
            raw: self.raw,
        }
    }

    #[inline]
    pub fn drain_top_into(&mut self, out: &mut [Val]) {
        let n = out.len();
        debug_assert!(self.raw.size >= n);
        let start = self.raw.size - n;
        // SAFETY: `[start, start+n)` are `n` live, initialized elements
        // (just asserted `self.raw.size >= n`).
        for (i, slot) in out.iter_mut().enumerate() {
            // SAFETY: reading the bits without touching the source's own
            // refcount - the source slots are truncated away right after,
            // so ownership transfers to `out` exactly once.
            *slot = Val::from_raw(unsafe { core::ptr::read(self.raw.ptr.add(start + i)) });
        }
        // SAFETY: elements `[start, size)` were just moved out above via
        // `ptr::read`, not dropped - shrinking past them without running
        // their (nonexistent, since they're moved-from) destructors again
        // is exactly `raw_vec_truncate_no_drop`'s contract.
        unsafe { raw_vec_truncate_no_drop(&mut self.raw, start) };
    }

    #[inline(always)]
    pub fn truncate(&mut self, len: usize) {
        unsafe {
            for i in len..self.raw.size {
                let raw = self.raw.ptr.add(i);
                drop(Val::from_raw(core::ptr::read(raw)));
            }

            raw_vec_truncate_no_drop(&mut self.raw, len);
        }
    }

    /// Read frame slot `slot` of the frame based at `base`.
    #[inline(always)]
    pub fn get(&self, idx: usize) -> &Val {
        if self.raw.size <= idx {
            stack_out_of_bounds();
        }

        // SAFETY: `idx < self.raw.size` (just asserted) - live element.
        unsafe { Val::from_raw_ref(&*self.raw.ptr.add(idx)) }
    }

    /// Write frame slot `slot` of the frame based at `base`.
    #[inline(always)]
    pub fn set(&mut self, idx: usize, v: Val) {
        if self.raw.size <= idx {
            stack_out_of_bounds();
        }

        // SAFETY: `idx < self.raw.size` (just asserted) - live element.
        unsafe {
            let raw = self.raw.ptr.add(idx);
            drop(Val::from_raw(core::ptr::read(raw)));
            raw.write(v.into_raw());
        }
    }
}

#[inline(always)]
fn raw_val_uninit() -> raft_ffi::RawVal {
    // SAFETY: `MASK_TAG_UNINIT` is a nonzero constant, never dereferenced
    // as a real pointer.
    raft_ffi::RawVal {
        tag: raft_ffi::RawTag {
            tag_ptr: unsafe { NonNull::new_unchecked(raft_ffi::MASK_TAG_UNINIT as *mut Void) },
        },
        data: raft_ffi::RawData { nothing: () },
    }
}

// ---------------------------------------------------------------------
// Direct `RawVec<T>` growth/mutation primitives - no `Vec<T>`
// reconstruction. Each op touches `header.ptr`/`size`/`capacity`
// directly and leaves them self-consistent when it returns (no separate
// "guard" object whose `Drop` writes them back later); growth mirrors
// `Vec`'s doubling policy so the allocations stay `realloc`-compatible
// with each other call over call.
// ---------------------------------------------------------------------

/// Ensure `header` has room for `additional` more elements beyond its
/// current `size`, growing (`Vec`-style doubling) and reallocating in
/// place if not.
///
/// # Safety
/// `header.ptr`/`size`/`capacity` must be valid `alloc`/`realloc`
/// raw parts for `T` (or all-zero, for a not-yet-allocated header) -
/// i.e. whatever `RawVec<T>::default()`, or a prior call here, leaves
/// behind. `T` must not be a zero-sized type.
#[inline(always)]
pub unsafe fn raw_vec_reserve<T>(header: &mut RawVec<T>, additional: usize) {
    let required = header.size + additional;
    if required <= header.capacity {
        return;
    }

    raw_vec_reserve_slow(header, required);

    #[cold]
    #[inline(never)]
    fn raw_vec_reserve_slow<T>(header: &mut RawVec<T>, required: usize) {
        let new_cap = required.max(header.capacity.saturating_mul(2)).max(16);
        let new_layout = Layout::array::<T>(new_cap).expect("capacity overflow");
        let new_ptr = if header.capacity == 0 {
            // SAFETY: `new_layout` is non-zero-sized (`new_cap >= 16`, `T` is
            // never a ZST per this function's contract).
            unsafe { alloc(new_layout) }
        } else {
            let old_layout = Layout::array::<T>(header.capacity).expect("capacity overflow");
            // SAFETY: `header.ptr` was allocated with `old_layout` by a prior
            // call here (`Vec`'s own growth policy - global allocator,
            // `Layout::array::<T>` sizing - matches exactly, so a `header`
            // seeded from a `Vec`'s raw parts is just as valid a starting
            // point).
            unsafe { realloc(header.ptr as *mut u8, old_layout, new_layout.size()) }
        };
        if new_ptr.is_null() {
            handle_alloc_error(new_layout);
        }
        header.ptr = new_ptr as *mut T;
        header.capacity = new_cap;
    }
}

/// # Safety
/// As [`raw_vec_reserve`]'s.
#[inline(always)]
pub unsafe fn raw_vec_push<T>(header: &mut RawVec<T>, value: T) {
    // SAFETY: caller's contract.
    unsafe { raw_vec_reserve(header, 1) };
    // SAFETY: the reserve above guarantees `size < capacity`.
    unsafe { header.ptr.add(header.size).write(value) };
    header.size += 1;
}

/// # Safety
/// `header.ptr`/`size` must describe live, initialized elements (as
/// [`raw_vec_reserve`]'s contract, minus the allocator part - this
/// never grows).
#[inline(always)]
pub unsafe fn raw_vec_pop<T>(header: &mut RawVec<T>) -> Option<T> {
    if header.size == 0 {
        return None;
    }
    header.size -= 1;
    // SAFETY: index `size` (post-decrement) was live and initialized the
    // instant before this call; ownership moves to the caller, and this
    // slot is no longer in `[0, size)` so it won't be read again.
    Some(unsafe { header.ptr.add(header.size).read() })
}

/// # Safety
/// `header.ptr`/`size` must describe live, initialized elements (as
/// [`raw_vec_reserve`]'s contract, minus the allocator part - this
/// never grows).
#[inline(always)]
pub unsafe fn raw_vec_last<T>(header: &RawVec<T>) -> Option<&T> {
    if header.size == 0 {
        return None;
    }
    // SAFETY: index `size-1` (post-decrement) was live and initialized.
    Some(unsafe { &*header.ptr.add(header.size - 1) })
}

/// Remove the element at `index`, shifting everything after it left by
/// one (like `Vec::remove`).
///
/// # Safety
/// As [`raw_vec_pop`]'s; `index < header.size`.
#[inline(always)]
pub unsafe fn raw_vec_remove<T>(header: &mut RawVec<T>, index: usize) -> T {
    debug_assert!(index < header.size);
    // SAFETY: `index < header.size` (caller's contract) - live element.
    let removed = unsafe { header.ptr.add(index).read() };
    let tail = header.size - index - 1;
    if tail > 0 {
        // SAFETY: `[index+1, size)` and `[index, size-1)` are both
        // within the live allocation; `read()` above left `index`'s
        // slot logically empty (not yet overwritten), which this fills.
        unsafe { core::ptr::copy(header.ptr.add(index + 1), header.ptr.add(index), tail) };
    }
    header.size -= 1;
    removed
}

/// Shrink `header.size` to `new_len` without dropping the elements past
/// it - for callers that have already moved them out (e.g. read their
/// bits directly to transfer ownership elsewhere).
///
/// # Safety
/// `new_len <= header.size`.
#[inline(always)]
pub unsafe fn raw_vec_truncate_no_drop<T>(header: &mut RawVec<T>, new_len: usize) {
    debug_assert!(new_len <= header.size);
    header.size = new_len;
}

/// One shared, monomorphized `&'static FnVTable` per concrete `F` - same
/// idea as [`any_vtable`], plus the `call` shim `Fn` values need.
pub fn fn_vtable<F: Callable>() -> &'static raft_ffi::FnVTable {
    struct Holder<F>(PhantomData<F>);
    impl<F: Callable> Holder<F> {
        const VTABLE: raft_ffi::FnVTable = raft_ffi::FnVTable {
            any: raft_ffi::AnyVTable {
                destroy: destroy_shim::<F>,
            },
            call: call_shim::<F>,
        };
    }
    &Holder::<F>::VTABLE
}

/// One shared, monomorphized `&'static CoroVTable` per concrete `C` -
/// same idea as [`fn_vtable`], with `resume` in place of `call`.
pub fn coro_vtable<C: Resumable>() -> &'static raft_ffi::CoroVTable {
    struct Holder<C>(PhantomData<C>);
    impl<C: Resumable> Holder<C> {
        const VTABLE: raft_ffi::CoroVTable = raft_ffi::CoroVTable {
            any: raft_ffi::AnyVTable {
                destroy: destroy_shim::<C>,
            },
            resume: resume_shim::<C>,
        };
    }
    &Holder::<C>::VTABLE
}

/// A `T`-typed [`raft_ffi::DestroyFn`]: reclaims a box previously handed
/// off via [`Rc::into_raw_box`]/[`DynRc::erase`] (destructor +
/// deallocation) for this exact `T`.
///
/// # Safety
/// `ptr` must be a live `Rc::<T>::into_raw_box` allocation for this exact
/// `T`, with no other live `Rc<T>`/`DynRc` still referencing it.
pub unsafe extern "C" fn destroy_shim<T>(ptr: RcPtr<Void>) {
    unsafe {
        let ptr = ptr.cast::<RcInner<T>>();
        core::ptr::drop_in_place(&mut (*ptr.as_ptr()).value);
        dealloc(ptr.as_ptr() as *mut u8, Layout::new::<RcInner<T>>());
    }
}

/// One shared, monomorphized `&'static AnyVTable` per concrete `T` -
/// built once (first use) per instantiation, reused by every `DynRc::erase`
/// call for that `T`.
pub fn any_vtable<T: 'static>() -> &'static raft_ffi::AnyVTable {
    struct Holder<T>(PhantomData<T>);
    impl<T: 'static> Holder<T> {
        const VTABLE: raft_ffi::AnyVTable = raft_ffi::AnyVTable {
            destroy: destroy_shim::<T>,
        };
    }
    &Holder::<T>::VTABLE
}
