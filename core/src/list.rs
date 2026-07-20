// ---------------------------------------------------------------------
// RcList: Val::List's backing. `ListVal` is a `RawVec<RawVal>` -
// growable, via `Vec<Val>`'s own realloc, in place, behind a stable
// outer `RcInner` (only the *inner* buffer moves on push).
// ---------------------------------------------------------------------

use core::{alloc::Layout, cmp::Ordering, fmt, mem::ManuallyDrop, ptr::NonNull};

use alloc::vec::Vec;
use raft_ffi::{RawListVal, RawVal, RcInner, Void};

use crate::{
    cast_rc_inner,
    rc::{DynRc, Rc},
    val::{Val, ValEnum},
    vec::{raw_vec_pop, raw_vec_push},
};

unsafe extern "C" fn list_destroy(ptr: raft_ffi::RcPtr<Void>) {
    let mut ptr = unsafe { cast_rc_inner(ptr) };
    // SAFETY: strong count just hit zero (`DynRc::drop`, the only
    // caller) - this is the sole reference. Turning it into a reference
    // immediately confines everything after to safe slice/Vec ops.
    let inner: &mut RcInner<RawListVal> = unsafe { ptr.as_mut() };
    let header = &inner.value;
    // SAFETY: `header`'s ptr/size/capacity are valid `Vec<Val>` raw parts.
    let elements =
        unsafe { Vec::from_raw_parts(header.ptr as *mut Val, header.size, header.capacity) };
    drop(elements); // drops each `Val` (real teardown), then frees the buffer
    // SAFETY: deallocating the exact allocation `RcList::new` made.
    unsafe {
        alloc::alloc::dealloc(
            ptr.as_ptr() as *mut u8,
            Layout::new::<RcInner<RawListVal>>(),
        );
    }
}

unsafe extern "C" fn list_get_shim(data: raft_ffi::VoidPtr, index: usize) -> RawVal {
    let mut ptr = data.cast();
    let header: &mut RawListVal = unsafe { ptr.as_mut() };

    // SAFETY: `inner.ptr`/`size` describe a valid `[Val]`.
    let slice: &[Val] =
        unsafe { core::slice::from_raw_parts(header.ptr as *const Val, header.size) };
    match slice.get(index) {
        Some(v) => v.clone().into_raw(),
        None => Val::from(ValEnum::Uninit).into_raw(),
    }
}

unsafe extern "C" fn list_set_shim(data: raft_ffi::VoidPtr, index: usize, val: RawVal) {
    let mut ptr = data.cast();
    let header: &mut RawListVal = unsafe { ptr.as_mut() };

    let slice: &mut [Val] =
        unsafe { core::slice::from_raw_parts_mut(header.ptr as *mut Val, header.size) };
    if let Some(slot) = slice.get_mut(index) {
        *slot = unsafe { Val::from_raw(val) }; // assignment drops the old value
    }
}

unsafe extern "C" fn list_swap_shim(data: raft_ffi::VoidPtr, index: usize, val: RawVal) -> RawVal {
    let mut ptr = data.cast();
    let header: &mut RawListVal = unsafe { ptr.as_mut() };
    let slice: &mut [Val] =
        unsafe { core::slice::from_raw_parts_mut(header.ptr as *mut Val, header.size) };
    match slice.get_mut(index) {
        Some(slot) => core::mem::replace(slot, unsafe { Val::from_raw(val) }).into_raw(),
        None => val,
    }
}

unsafe extern "C" fn list_push_shim(data: raft_ffi::VoidPtr, val: RawVal) {
    let mut ptr = data.cast();
    let header: &mut RawListVal = unsafe { ptr.as_mut() };
    // SAFETY: `header` describes valid `RawVec<RawVal>` raw parts.
    unsafe { raw_vec_push(header, val) };
}

unsafe extern "C" fn list_pop_shim(data: raft_ffi::VoidPtr) -> RawVal {
    let mut ptr = data.cast();
    let header: &mut RawListVal = unsafe { ptr.as_mut() };
    // SAFETY: as `list_push_shim`.
    unsafe { raw_vec_pop(header) }.unwrap_or_else(|| Val::from(ValEnum::Uninit).into_raw())
}

unsafe extern "C" fn list_elements_shim(data: raft_ffi::VoidPtr) -> *const RawVal {
    let mut ptr = data.cast();
    let header: &mut RawListVal = unsafe { ptr.as_mut() };
    header.ptr
}

static LIST_VTABLE: raft_ffi::ListVTable = raft_ffi::ListVTable {
    any: raft_ffi::AnyVTable {
        destroy: list_destroy,
    },
    elements: list_elements_shim,
    get: list_get_shim,
    set: list_set_shim,
    swap: list_swap_shim,
    push: list_push_shim,
    pop: list_pop_shim,
};

#[repr(transparent)]
#[derive(Clone)]
pub struct RcList {
    ptr: DynRc<raft_ffi::ListVTable, RawListVal>,
}

impl RcList {
    pub fn new(elements: impl IntoIterator<Item = Val>) -> Self {
        let elements: Vec<Val> = elements.into_iter().collect();
        let len = elements.len();
        let cap = elements.capacity();
        let mut elements = ManuallyDrop::new(elements);
        let header = RawListVal {
            ptr: elements.as_mut_ptr() as *mut RawVal,
            size: len,
            capacity: cap,
        };
        let rc = Rc::new(header);
        RcList {
            ptr: unsafe { DynRc::new(Rc::into_raw_box(rc).cast(), &LIST_VTABLE) },
        }
    }

    pub unsafe fn from_raw(ptr: DynRc<raft_ffi::ListVTable, RawListVal>) -> Self {
        RcList { ptr }
    }

    pub fn into_raw(self) -> DynRc<raft_ffi::ListVTable, RawListVal> {
        self.ptr
    }

    #[inline]
    fn header(&self) -> &raft_ffi::RawListVal {
        // SAFETY: single-threaded; no live `&mut` alias held across this.
        &self.ptr.rc_box().value
    }

    pub fn len(&self) -> usize {
        self.header().size
    }

    pub fn as_slice(&self) -> &[Val] {
        let header = self.header();

        assert!(!header.ptr.is_null());

        unsafe { core::slice::from_raw_parts(header.ptr as *const Val, header.size) }
    }

    pub fn get(&self, index: usize) -> Option<Val> {
        self.as_slice().get(index).map(|v| v.clone())
    }

    #[inline]
    fn value_ptr(&self) -> NonNull<RawListVal> {
        // SAFETY: `self.ptr.rc_box().value.get()` is always a live,
        // correctly-aligned `*mut ListVal`.
        NonNull::from_ref(&self.ptr.rc_box().value)
    }

    /// `target[index] = value` - in place, no length change.
    pub fn set(&self, index: usize, val: Val) {
        let vtable = DynRc::vtable(&self.ptr);
        // SAFETY: `value_ptr()` is a live `ListVal` for `vtable`.
        unsafe { (vtable.set)(self.value_ptr().cast(), index, val.into_raw()) };
    }

    /// Append to the end, growing in place (the outer handle stays valid
    /// - see `RcList`'s doc comment).
    pub fn push(&self, val: Val) {
        let vtable = DynRc::vtable(&self.ptr);
        // SAFETY: as `set`'s.
        unsafe { (vtable.push)(self.value_ptr().cast(), val.into_raw()) };
    }

    /// Remove and return the last element, if any.
    pub fn pop(&self) -> Option<Val> {
        if self.len() == 0 {
            return None;
        }
        let vtable = DynRc::vtable(&self.ptr);
        // SAFETY: as `set`'s.
        let popped = unsafe { (vtable.pop)(self.value_ptr().cast()) };
        Some(unsafe { Val::from_raw(popped) })
    }

    pub fn eq(&self, other: &RcList) -> bool {
        let (a, b) = (self.as_slice(), other.as_slice());
        a == b
    }

    pub fn cmp(&self, other: &RcList) -> Option<Ordering> {
        let (a, b) = (self.as_slice(), other.as_slice());
        Iterator::partial_cmp(a.iter(), b.iter())
    }
}

impl PartialEq for RcList {
    fn eq(&self, other: &Self) -> bool {
        self.eq(other)
    }
}

impl PartialOrd for RcList {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.cmp(other)
    }
}

impl fmt::Debug for RcList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

impl fmt::Display for RcList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, elem) in self.as_slice().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", elem)?;
        }
        write!(f, "]")
    }
}
