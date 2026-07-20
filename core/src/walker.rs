
// ---------------------------------------------------------------------
// FfiWaker: an owned reference to a host waker crossing the FFI boundary
// as a thin pointer (vtable in the allocation's prefix - see raft-ffi's
// async section). Adapting to `core::task::Waker` is a thin round trip:
// the RawWaker data pointer IS the header pointer.
//
// Single-threaded contract: despite `Waker`'s nominal `Send + Sync`, a
// host waker must only be cloned/woken/dropped on the host's own thread.
// ---------------------------------------------------------------------

use core::{ptr::NonNull, sync::atomic::Ordering};

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
