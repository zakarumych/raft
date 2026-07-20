// ---------------------------------------------------------------------
// RcRecord: Val::Record's backing. `RecordVal` is a
// `RawVec<RawFieldVal>`; each entry's `name` is a borrowed `Str` view
// into its own small, singly-owned (non-refcounted - never aliased)
// allocation, freed explicitly on overwrite/removal/destroy.
// ---------------------------------------------------------------------

use core::{alloc::Layout, cmp::Ordering, fmt, mem::ManuallyDrop, ptr::NonNull};

use alloc::vec::Vec;
use raft_ffi::{RawFieldVal, RawRecordVal, RawVal, RcInner, RcPtr, Str, Void};

use crate::{
    rc::{DynRc, Rc},
    string::RcStr,
    val::{Val, ValEnum},
    vec::{raw_vec_push, raw_vec_remove},
};

fn alloc_key_bytes(s: &str) -> Str {
    if s.is_empty() {
        return Str {
            ptr: NonNull::<u8>::dangling().as_ptr(),
            len: 0,
        };
    }
    let layout = Layout::array::<u8>(s.len()).unwrap();
    // SAFETY: `layout` is non-zero-sized (checked above).
    let ptr = unsafe { alloc::alloc::alloc(layout) };
    if ptr.is_null() {
        alloc::alloc::handle_alloc_error(layout);
    }
    unsafe { core::ptr::copy_nonoverlapping(s.as_ptr(), ptr, s.len()) };
    Str { ptr, len: s.len() }
}

unsafe fn free_key_bytes(s: Str) {
    if s.len == 0 {
        return;
    }
    let layout = Layout::array::<u8>(s.len).unwrap();
    unsafe { alloc::alloc::dealloc(s.ptr as *mut u8, layout) };
}

fn key_str(s: &Str) -> &str {
    // SAFETY: every `Str` here was produced by `alloc_key_bytes` from a
    // valid `&str`.
    unsafe { core::str::from_utf8_unchecked(core::slice::from_raw_parts(s.ptr, s.len)) }
}

unsafe extern "C" fn record_destroy(ptr: RcPtr<Void>) {
    let mut ptr = ptr.cast::<RcInner<RawRecordVal>>();
    {
        // SAFETY: strong count just hit zero (`DynRc::drop`, the only
        // caller) - this is the sole reference. Turning it into a reference
        // immediately confines everything after to safe operations.
        let inner: &mut RcInner<RawRecordVal> = unsafe { ptr.as_mut() };
        let header = &mut inner.value;
        // SAFETY: `header`'s ptr/size/capacity are valid `Vec<RawFieldVal>`
        // raw parts. `RawFieldVal` has no `Drop` impl of its own, so this
        // `Vec`'s own drop just frees the buffer - field cleanup is manual,
        // below.
        let fields = unsafe { Vec::from_raw_parts(header.ptr, header.size, header.capacity) };
        for field in &fields {
            // SAFETY: copying the `Copy`-able `(ptr, len)` fields out (not
            // moving `field.name`, which sits behind a shared reference),
            // then freeing that exact allocation and dropping the `Val` this
            // entry owned.
            unsafe {
                free_key_bytes(Str {
                    ptr: field.name.ptr,
                    len: field.name.len,
                });
                drop(Val::from_raw(core::ptr::read(&field.val)));
            }
        }
        drop(fields);
    }
    // SAFETY: deallocating the exact allocation `RcRecord::new` made.
    unsafe {
        alloc::alloc::dealloc(
            ptr.as_ptr() as *mut u8,
            Layout::new::<RcInner<RawRecordVal>>(),
        );
    }
}

fn record_fields_slice(header: &RawRecordVal) -> &[RawFieldVal] {
    // SAFETY: `header.ptr`/`size` describe a valid `[RawFieldVal]`.
    unsafe { core::slice::from_raw_parts(header.ptr, header.size) }
}

unsafe extern "C" fn record_fields_shim(data: raft_ffi::VoidPtr) -> *const RawFieldVal {
    let header: &RawRecordVal = unsafe { &*(data.as_ptr() as *const RawRecordVal) };
    header.ptr
}

/// # Safety
/// `name_ptr`/`name_len` must describe a valid UTF-8 `&str`.
unsafe fn key_view<'a>(name_ptr: *const u8, name_len: usize) -> &'a str {
    unsafe { core::str::from_utf8_unchecked(core::slice::from_raw_parts(name_ptr, name_len)) }
}

unsafe extern "C" fn record_get_by_name_shim(
    data: raft_ffi::VoidPtr,
    name_ptr: *const u8,
    name_len: usize,
) -> RawVal {
    let header: &RawRecordVal = unsafe { &*(data.as_ptr() as *const RawRecordVal) };
    let key: &str = unsafe { key_view(name_ptr, name_len) };
    match record_fields_slice(header)
        .iter()
        .find(|f| key_str(&f.name) == key)
    {
        Some(f) => unsafe { Val::clone_raw(&f.val).into_raw() },
        None => Val::from(ValEnum::Uninit).into_raw(),
    }
}

unsafe extern "C" fn record_get_shim(data: raft_ffi::VoidPtr, index: usize) -> RawVal {
    let header: &RawRecordVal = unsafe { &*(data.as_ptr() as *const RawRecordVal) };
    match record_fields_slice(header).get(index) {
        Some(f) => unsafe { Val::clone_raw(&f.val).into_raw() },
        None => Val::from(ValEnum::Uninit).into_raw(),
    }
}

unsafe extern "C" fn record_set_by_name_shim(
    data: raft_ffi::VoidPtr,
    name_ptr: *const u8,
    name_len: usize,
    val: RawVal,
) {
    let mut ptr: NonNull<RawRecordVal> = data.cast();
    let header: &mut RawRecordVal = unsafe { ptr.as_mut() };
    let key: &str = unsafe { key_view(name_ptr, name_len) };
    {
        if !header.ptr.is_aligned() {
            panic!("Unaligned record array pointer `*{:p} = {:p}`", header as *const _, header.ptr);
        }

        // Scoped so this slice's borrow of `header` ends before the
        // (mutually-exclusive) `raw_vec_push` borrow below, in the
        // not-found case.
        let slice: &mut [RawFieldVal] =
            unsafe { core::slice::from_raw_parts_mut(header.ptr, header.size) };
        if let Some(entry) = slice.iter_mut().find(|f| key_str(&f.name) == key) {
            drop(unsafe { Val::from_raw(core::mem::replace(&mut entry.val, val)) });
            return;
        }
    }
    // SAFETY: `header` describes valid `RawVec<RawFieldVal>` raw parts.
    unsafe {
        raw_vec_push(
            header,
            RawFieldVal {
                name: alloc_key_bytes(key),
                val,
            },
        )
    };
}

unsafe extern "C" fn record_set_shim(data: raft_ffi::VoidPtr, index: usize, val: RawVal) {
    let header: &RawRecordVal = unsafe { &*(data.as_ptr() as *const RawRecordVal) };
    let slice: &mut [RawFieldVal] =
        unsafe { core::slice::from_raw_parts_mut(header.ptr, header.size) };
    if let Some(entry) = slice.get_mut(index) {
        drop(unsafe { Val::from_raw(core::mem::replace(&mut entry.val, val)) });
    }
}

unsafe extern "C" fn record_swap_shim(
    data: raft_ffi::VoidPtr,
    index: usize,
    val: RawVal,
) -> RawVal {
    let header: &RawRecordVal = unsafe { &*(data.as_ptr() as *const RawRecordVal) };
    let slice: &mut [RawFieldVal] =
        unsafe { core::slice::from_raw_parts_mut(header.ptr, header.size) };
    match slice.get_mut(index) {
        Some(entry) => core::mem::replace(&mut entry.val, val),
        None => val,
    }
}

unsafe extern "C" fn record_swap_by_name_shim(
    data: raft_ffi::VoidPtr,
    name_ptr: *const u8,
    name_len: usize,
    val: RawVal,
) -> RawVal {
    let header: &RawRecordVal = unsafe { &*(data.as_ptr() as *const RawRecordVal) };
    let key: &str = unsafe { key_view(name_ptr, name_len) };
    let slice: &mut [RawFieldVal] =
        unsafe { core::slice::from_raw_parts_mut(header.ptr, header.size) };
    match slice.iter_mut().find(|f| key_str(&f.name) == key) {
        Some(entry) => core::mem::replace(&mut entry.val, val),
        None => val,
    }
}

fn record_rem_at(header: &mut RawRecordVal, index: usize) {
    if index >= header.size {
        return;
    }
    // SAFETY: `header` describes valid `RawVec<RawFieldVal>` raw parts;
    // `index < header.size` (checked above).
    let removed = unsafe { raw_vec_remove(header, index) };
    unsafe { free_key_bytes(removed.name) };
    drop(unsafe { Val::from_raw(removed.val) });
}

unsafe extern "C" fn record_rem_shim(data: raft_ffi::VoidPtr, index: usize) {
    let header: &mut RawRecordVal = unsafe { &mut *(data.as_ptr() as *mut RawRecordVal) };
    record_rem_at(header, index);
}

unsafe extern "C" fn record_rem_by_name_shim(
    data: raft_ffi::VoidPtr,
    name_ptr: *const u8,
    name_len: usize,
) {
    let header: &mut RawRecordVal = unsafe { &mut *(data.as_ptr() as *mut RawRecordVal) };
    let key: &str = unsafe { key_view(name_ptr, name_len) };
    if let Some(index) = record_fields_slice(header)
        .iter()
        .position(|f| key_str(&f.name) == key)
    {
        record_rem_at(header, index);
    }
}

static RECORD_VTABLE: raft_ffi::RecordVTable = raft_ffi::RecordVTable {
    any: raft_ffi::AnyVTable {
        destroy: record_destroy,
    },
    fields: record_fields_shim,
    get: record_get_shim,
    get_by_name: record_get_by_name_shim,
    set: record_set_shim,
    set_by_name: record_set_by_name_shim,
    rem: record_rem_shim,
    rem_by_name: record_rem_by_name_shim,
    swap: record_swap_shim,
    swap_by_name: record_swap_by_name_shim,
};

#[repr(transparent)]
#[derive(Clone)]
pub struct RcRecord {
    ptr: DynRc<raft_ffi::RecordVTable, RawRecordVal>,
}

impl RcRecord {
    pub fn new(fields: impl IntoIterator<Item = (RcStr, Val)>) -> Self {
        let fields = fields.into_iter();

        let vec = fields
            .map(|(key, val)| RawFieldVal {
                name: alloc_key_bytes(key.as_str()),
                val: val.into_raw(),
            })
            .collect::<Vec<_>>();

        let mut vec = ManuallyDrop::new(vec);
        let header = RawRecordVal {
            ptr: vec.as_mut_ptr(),
            size: vec.len(),
            capacity: vec.capacity(),
        };
        let rc = Rc::new(header);

        RcRecord {
            ptr: unsafe { DynRc::new(Rc::into_raw_box(rc), &RECORD_VTABLE) },
        }
    }

    pub unsafe fn from_raw(ptr: DynRc<raft_ffi::RecordVTable, RawRecordVal>) -> Self {
        RcRecord { ptr }
    }

    pub fn into_raw(self) -> DynRc<raft_ffi::RecordVTable, RawRecordVal> {
        self.ptr
    }

    #[inline]
    fn header(&self) -> &RawRecordVal {
        // SAFETY: single-threaded; no live `&mut` alias held across this.
        &self.ptr.rc_box().value
    }

    pub fn len(&self) -> usize {
        self.header().size
    }

    pub fn get_field(&self, key: &str) -> Option<Val> {
        record_fields_slice(self.header())
            .iter()
            .find(|f| key_str(&f.name) == key)
            .map(|f| unsafe { Val::clone_raw(&f.val) })
    }

    pub fn entry_at(&self, index: usize) -> Option<(&str, Val)> {
        record_fields_slice(self.header())
            .get(index)
            .map(|f| (key_str(&f.name), unsafe { Val::clone_raw(&f.val) }))
    }

    #[inline]
    fn value_ptr(&self) -> NonNull<RawRecordVal> {
        DynRc::as_ptr(&self.ptr)
    }

    /// Insert (or overwrite) a field by name.
    pub fn set_field(&self, key: &str, val: Val) {
        let vtable = DynRc::vtable(&self.ptr);
        // SAFETY: `value_ptr()` is a live `RecordVal` for `vtable`;
        // `key.as_ptr()`/`.len()` describe a valid UTF-8 view for the
        // duration of this call.
        unsafe {
            (vtable.set_by_name)(
                self.value_ptr().cast(),
                key.as_ptr(),
                key.len(),
                val.into_raw(),
            )
        };
    }

    /// Remove a field by name, if present.
    pub fn remove_field(&self, key: &str) {
        let vtable = DynRc::vtable(&self.ptr);
        // SAFETY: as `set_field`'s.
        unsafe { (vtable.rem_by_name)(self.value_ptr().cast(), key.as_ptr(), key.len()) };
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, Val)> {
        record_fields_slice(self.header())
            .iter()
            .map(|f| (key_str(&f.name), unsafe { Val::clone_raw(&f.val) }))
    }

    pub fn eq(&self, other: &RcRecord) -> bool {
        let (a, b) = (
            record_fields_slice(self.header()),
            record_fields_slice(other.header()),
        );

        a.len() == b.len() && Iterator::eq(a.iter().map(raw_field_cmp), b.iter().map(raw_field_cmp))
    }

    pub fn cmp(&self, other: &RcRecord) -> Option<Ordering> {
        let (a, b) = (
            record_fields_slice(self.header()),
            record_fields_slice(other.header()),
        );

        Iterator::partial_cmp(a.iter().map(raw_field_cmp), b.iter().map(raw_field_cmp))
    }
}

fn raw_field_cmp(r: &RawFieldVal) -> (&str, &Val) {
    (key_str(&r.name), unsafe { Val::from_raw_ref(&r.val) })
}

impl PartialEq for RcRecord {
    fn eq(&self, other: &Self) -> bool {
        self.eq(other)
    }
}

impl PartialOrd for RcRecord {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.cmp(other)
    }
}

impl fmt::Debug for RcRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(
                record_fields_slice(self.header())
                    .iter()
                    .map(|field| (key_str(&field.name), unsafe { Val::clone_raw(&field.val) })),
            )
            .finish()
    }
}

impl fmt::Display for RcRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        for (i, field) in record_fields_slice(self.header()).iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            let value = unsafe { Val::clone_raw(&field.val) };
            write!(f, "{}: {}", key_str(&field.name), value)?;
        }
        write!(f, "}}")
    }
}
