// ---------------------------------------------------------------------
// RcOpaque: Val::Opaque's backing - a fully host-erased `T`, monomorphized
// per concrete `T` (see `rc::any_vtable`/`rc::DynRc::erase`).
// ---------------------------------------------------------------------


use core::ptr::NonNull;

use raft_ffi::Void;

use crate::{
    erase_rc_inner, rc::{DynRc, Rc}, vtable::any_vtable_ref,
};

#[repr(transparent)]
#[derive(Clone)]
pub struct RcOpaque {
    ptr: DynRc<raft_ffi::AnyVTable, Void>,
}

impl RcOpaque {
    pub fn new<T: 'static>(value: T) -> Self {
        RcOpaque { ptr: erase(value) }
    }

    pub unsafe fn from_raw(ptr: DynRc<raft_ffi::AnyVTable, Void>) -> Self {
        RcOpaque { ptr }
    }

    pub fn into_raw(self) -> DynRc<raft_ffi::AnyVTable, Void> {
        self.ptr
    }

    pub fn as_ptr(&self) -> NonNull<()> {
        DynRc::as_ptr(&self.ptr).cast()
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
    unsafe {
        DynRc::new(erase_rc_inner(Rc::into_raw_box(rc)), any_vtable_ref::<T>())
    }
}
