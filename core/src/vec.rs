
// ---------------------------------------------------------------------
// Direct `RawVec<T>` growth/mutation primitives - no `Vec<T>`
// reconstruction. Each op touches `header.ptr`/`size`/`capacity`
// directly and leaves them self-consistent when it returns (no separate
// "guard" object whose `Drop` writes them back later); growth mirrors
// `Vec`'s doubling policy so the allocations stay `realloc`-compatible
// with each other call over call.
// ---------------------------------------------------------------------

use core::alloc::Layout;

use alloc::alloc::{alloc, handle_alloc_error, realloc};
use raft_ffi::RawVec;

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
pub unsafe fn raw_vec_set_len<T>(header: &mut RawVec<T>, new_len: usize) {
    debug_assert!(new_len <= header.size);
    header.size = new_len;
}
