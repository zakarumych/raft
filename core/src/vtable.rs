use core::{alloc::Layout, marker::PhantomData, ptr::NonNull};

use alloc::alloc::dealloc;
use raft_ffi::{RcInner, RcPtr, Void};

use crate::cast_rc_inner;

/// Projects any concrete vtable down to the common `clone`/`drop`-shaped
/// [`ffi::AnyVTable`] prefix every vtable embeds - what [`DynRc`] needs to
/// tear a value down without knowing its richer, kind-specific shape.
pub trait AnyVTable: 'static {
    fn any(&self) -> &raft_ffi::AnyVTable;
}

impl AnyVTable for raft_ffi::AnyVTable {
    #[inline(always)]
    fn any(&self) -> &raft_ffi::AnyVTable {
        self
    }
}

impl AnyVTable for raft_ffi::StringVTable {
    #[inline(always)]
    fn any(&self) -> &raft_ffi::AnyVTable {
        &self.any
    }
}

impl AnyVTable for raft_ffi::RecordVTable {
    #[inline(always)]
    fn any(&self) -> &raft_ffi::AnyVTable {
        &self.any
    }
}

impl AnyVTable for raft_ffi::ListVTable {
    #[inline(always)]
    fn any(&self) -> &raft_ffi::AnyVTable {
        &self.any
    }
}

impl AnyVTable for raft_ffi::FnVTable {
    #[inline(always)]
    fn any(&self) -> &raft_ffi::AnyVTable {
        &self.any
    }
}

impl AnyVTable for raft_ffi::CoroVTable {
    #[inline(always)]
    fn any(&self) -> &raft_ffi::AnyVTable {
        &self.any
    }
}

pub const fn any_vtable<T: 'static>() -> raft_ffi::AnyVTable {
    raft_ffi::AnyVTable {
        destroy: destroy_shim::<T>,
    }
}

/// One shared, monomorphized `&'static AnyVTable` per concrete `T` -
/// built once (first use) per instantiation, reused by every `DynRc::erase`
/// call for that `T`.
pub fn any_vtable_ref<T: 'static>() -> &'static raft_ffi::AnyVTable {
    struct Holder<T>(PhantomData<T>);
    impl<T: 'static> Holder<T> {
        const VTABLE: raft_ffi::AnyVTable = any_vtable::<T>();
    }
    &Holder::<T>::VTABLE
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
        let ptr: NonNull<RcInner<T>> = cast_rc_inner(ptr);
        core::ptr::drop_in_place(&mut (*ptr.as_ptr()).value);
        dealloc(ptr.as_ptr() as *mut u8, Layout::new::<RcInner<T>>());
    }
}
