//! Refcounted box used throughout `raft-core`, in place of `alloc::rc::Rc`.
//! Strong-count-only (no `Weak` - unused anywhere in this codebase), and
//! built directly on [`ffi::RcBox`] (not a locally duplicated header type)
//! so [`Rc::into_raw_box`]/[`Rc::from_raw_box`] are plain `NonNull::cast`s
//! of the existing allocation into/out of an [`ffi::FFIRc`]'s `data` field
//! - no new allocation, no offset math, no `c_void`.

use alloc::{
    alloc::{Layout, dealloc},
    boxed::Box,
};
use core::{cell::Cell, mem::ManuallyDrop, ops::Deref, ptr::NonNull};

use raft_ffi::{RcInner, RcPtr,};

use crate::{vtable::AnyVTable};

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
    pub fn as_ptr(this: &Self) -> NonNull<T> {
        NonNull::from_ref(&this.rc_box().value)
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
    pub fn into_raw_box(this: Self) -> NonNull<RcInner<T>> {
        let me = ManuallyDrop::new(this);
        me.ptr
    }

    /// Consume `this` without dropping, returning a pointer to just the
    /// inner value (skipping the header). Pair with
    /// [`from_raw`](Rc::from_raw).
    #[inline]
    pub fn into_raw(this: Self) -> NonNull<T> {
        let me = ManuallyDrop::new(this);
        NonNull::from_ref(&me.rc_box().value)
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
    pub fn as_ptr(this: &Self) -> NonNull<T> {
        NonNull::from_ref(&this.rc_box().value)
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
    pub fn rc_ptr(this: &Self) -> RcPtr<T> {
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
