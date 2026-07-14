//! Refcounted box used throughout `raft-core`, in place of `alloc::rc::Rc`.
//! Strong-count-only (no `Weak` — unused anywhere in this codebase), and
//! built directly on [`ffi::RcBox`] (not a locally duplicated header type)
//! so [`Rc::into_raw_box`]/[`Rc::from_raw_box`] are plain `NonNull::cast`s
//! of the existing allocation into/out of an [`ffi::FFIRc`]'s `data` field
//! — no new allocation, no offset math, no `c_void`.

use alloc::{
    alloc::{Layout, dealloc},
    boxed::Box,
    vec::Vec,
};
use core::{
    cell::Cell,
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

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
            // SAFETY: unique owner — read the value out, then free the
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
    /// — the shape [`DynRc`]/opaque vtable-backed types store. No new
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

/// Minimal callable bound for building a per-concrete-type
/// [`raft_ffi::FnVTable`] — deliberately smaller than `raft-core`'s own
/// `Function` trait (no min/max-args or partial-application semantics
/// here, just "invoke, given a host and an argument count, returning how
/// many were actually consumed"). `Function` bridges into this.
pub trait Callable: 'static {
    fn call(&self, args: usize, host: &mut Host) -> usize;
}

unsafe extern "C" fn call_shim<F: Callable>(
    data: raft_ffi::VoidPtr,
    args: usize,
    host: *mut raft_ffi::RawHost,
) -> usize {
    // SAFETY: `data` points at a live `F` (see `fn_vtable`'s contract).
    let f = unsafe { &*(data.as_ptr() as *const F) };
    // SAFETY: `host`, per `raft_ffi::CallFn`'s contract, is a valid,
    // exclusively-held `RawHost` for the duration of this call — this is
    // the one place that raw pointer gets turned into a reference.
    let mut host = unsafe { Host::from_raw(host) };
    f.call(args, &mut host)
}

/// Safe view over a host's operand stack — the only thing
/// `Function`/`Callable` implementors ever need. Never exposes the raw
/// `*mut ffi::RawHost` it was built from as part of its public API; every
/// method here derives a reference from it internally and is safe to
/// call. (`as_raw` is `pub(crate)`, not public — only [`crate::RcFn::call`]
/// needs it, to cross back into the `extern "C"` `CallFn` ABI, which is
/// the one place a raw pointer is unavoidable.)
pub struct Host<'a> {
    raw: *mut raft_ffi::RawHost,
    _marker: PhantomData<&'a mut raft_ffi::RawHost>,
}

impl<'a> Host<'a> {
    /// # Safety
    /// `raw` must point at a valid `RawHost`, exclusively borrowed for `'a`.
    #[inline]
    pub unsafe fn from_raw(raw: *mut raft_ffi::RawHost) -> Self {
        Host {
            raw,
            _marker: PhantomData,
        }
    }

    /// Recover the raw pointer this `Host` wraps. An escape hatch: a
    /// concrete host implementation (`raft-runtime`'s `Runtime`, say,
    /// arranging its own layout so it *is* a `RawHost` at heart, and
    /// casting `&mut self` to build a `Host` in the first place) is the
    /// only sound caller — it can cast this back to its own type. Ordinary
    /// `Function` implementations never need this; `Host`'s other methods
    /// cover everything they should touch.
    #[inline]
    pub fn as_raw(&mut self) -> *mut raft_ffi::RawHost {
        self.raw
    }

    #[inline]
    fn stack(&mut self) -> &mut raft_ffi::RawStack {
        // SAFETY: struct invariant (`from_raw`'s contract).
        unsafe { &mut (*self.raw).stack }
    }

    pub fn push(&mut self, v: Val) {
        // SAFETY: a host's stack is always a valid `Vec<RawVal>`-shaped
        // allocation.
        let mut guard = unsafe { RawVecGuard::<RawVal>::new(self.stack()) };
        guard.push(Val::into_raw(v));
    }

    pub fn pop(&mut self) -> Val {
        // SAFETY: as above.
        let mut guard = unsafe { RawVecGuard::<RawVal>::new(self.stack()) };
        match guard.pop() {
            Some(raw) => Val::from_raw(raw),
            None => {
                debug_assert!(false, "host stack underflow");
                Val::from_raw(raw_val_nothing())
            }
        }
    }

    /// Remove the top `out.len()` values into `out`, oldest-of-the-drained-
    /// range first.
    pub fn drain_top_into(&mut self, out: &mut [Val]) {
        // SAFETY: as above.
        let mut guard = unsafe { RawVecGuard::<RawVal>::new(self.stack()) };
        let n = out.len();
        debug_assert!(guard.len() >= n);
        let start = guard.len() - n;
        for (slot, raw) in out.iter_mut().zip(guard[start..].iter()) {
            // SAFETY: reading the bits without touching the source's own
            // refcount — the source slots are truncated away right after,
            // so ownership transfers to `out` exactly once.
            *slot = Val::from_raw(unsafe { core::ptr::read(raw) });
        }
        guard.truncate(start);
    }

    /// Push a batch of values in order (the first ends up deepest),
    /// cloning each one out of `values`.
    pub fn extend(&mut self, values: &[Val]) {
        // SAFETY: as above.
        let mut guard = unsafe { RawVecGuard::<RawVal>::new(self.stack()) };
        guard.reserve(values.len());
        for v in values {
            guard.push(Val::into_raw(v.clone()));
        }
    }
}

#[inline]
fn raw_val_nothing() -> raft_ffi::RawVal {
    // SAFETY: `MASK_TAG_UNINIT` is a nonzero constant, never dereferenced
    // as a real pointer.
    raft_ffi::RawVal {
        tag: raft_ffi::RawTag {
            tag_ptr: unsafe { NonNull::new_unchecked(raft_ffi::MASK_TAG_UNINIT as *mut Void) },
        },
        data: raft_ffi::RawData { nothing: () },
    }
}

/// RAII guard: reconstructs a `Vec<T>` from a [`raft_ffi::RawVec<T>`]-
/// shaped header's raw parts for the duration of some (possibly growing,
/// possibly panicking) operation, and writes the resulting ptr/size/
/// capacity back to the header on drop — including mid-panic-unwind — so
/// a growth failure can never leave the header describing memory a
/// temporarily-reconstructed `Vec` already freed out from under it. The
/// `Vec` itself is never actually dropped (its buffer is handed back to
/// `header`, not deallocated here).
pub struct RawVecGuard<'a, T> {
    header: &'a mut RawVec<T>,
    vec: ManuallyDrop<Vec<T>>,
}

impl<'a, T> RawVecGuard<'a, T> {
    /// # Safety
    /// `header.ptr`/`size`/`capacity` must be valid `Vec::<T>::from_raw_parts`
    /// arguments.
    #[inline]
    pub unsafe fn new(header: &'a mut RawVec<T>) -> Self {
        // SAFETY: caller's contract above.
        let vec = unsafe { Vec::from_raw_parts(header.ptr, header.size, header.capacity) };
        RawVecGuard {
            header,
            vec: ManuallyDrop::new(vec),
        }
    }
}

impl<'a, T> Deref for RawVecGuard<'a, T> {
    type Target = Vec<T>;

    #[inline]
    fn deref(&self) -> &Vec<T> {
        &self.vec
    }
}

impl<'a, T> DerefMut for RawVecGuard<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Vec<T> {
        &mut self.vec
    }
}

impl<'a, T> Drop for RawVecGuard<'a, T> {
    fn drop(&mut self) {
        self.header.ptr = self.vec.as_mut_ptr();
        self.header.size = self.vec.len();
        self.header.capacity = self.vec.capacity();
    }
}

/// One shared, monomorphized `&'static FnVTable` per concrete `F` — same
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

/// One shared, monomorphized `&'static AnyVTable` per concrete `T` —
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
