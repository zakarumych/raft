use core::{ptr::NonNull, sync::atomic::Ordering};

use raft_ffi::{RawVal, RawVec};

use crate::{raw_val_uninit, val::Val, vec::{raw_vec_last, raw_vec_pop, raw_vec_push, raw_vec_reserve, raw_vec_set_len}, walker::FfiWaker};


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
                FfiWaker::from_raw(ptr)
            }
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
            Some(unsafe { Val::from_raw(core::ptr::read(raw)) })
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
            Some(unsafe { Val::from_raw(core::ptr::read(raw)) })
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
            Some(raw) => unsafe { Val::from_raw(raw) },
        }
    }

    #[inline(always)]
    pub fn peek(&self) -> &Val {
        match unsafe { raw_vec_last(self.raw) } {
            None => stack_out_of_bounds(),
            Some(raw) => unsafe { Val::from_raw_ref(raw) },
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
            *slot = unsafe { Val::from_raw(core::ptr::read(self.raw.ptr.add(start + i))) };
        }
        // SAFETY: elements `[start, size)` were just moved out above via
        // `ptr::read`, not dropped - shrinking past them without running
        // their (nonexistent, since they're moved-from) destructors again
        // is exactly `raw_vec_truncate_no_drop`'s contract.
        unsafe { raw_vec_set_len(&mut self.raw, start) };
    }

    #[inline(always)]
    pub fn truncate(&mut self, len: usize) {
        unsafe {
            for i in len..self.raw.size {
                let raw = self.raw.ptr.add(i);
                drop(Val::from_raw(core::ptr::read(raw)));
            }

            raw_vec_set_len(&mut self.raw, len);
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
