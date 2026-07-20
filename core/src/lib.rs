//! Core Raft object model: `Val` and everything it is built from.
//!
//! Strictly `no_std` (`alloc` only) so it can be shared, unmodified, by
//! every execution mode - the tree-walking/bytecode `raft-runtime`, and
//! eventually transpiled-to-Rust modules (compiled into a `cdylib` or
//! bundled straight into the host binary via proc-macro).
//!
//! `Val` is `#[repr(transparent)]` over `raft_ffi::RawVal` - a real,
//! compact (2-word) tagged pointer, not a Rust enum. This crate's job is
//! to make that representation *safe* to use (under the assumption any
//! vtable a handle carries honors ffi's contract) and ergonomic, not just
//! zero-cost. [`ValEnum`] is the ergonomic, `match`-able view: `unpack`
//! clones into it (heap kinds bump a refcount; scalars are free), `pack`
//! (`From<ValEnum> for Val`) builds a `Val` back out of it.
#![no_std]

#[cfg(feature = "std")]
extern crate std;

use core::ptr::NonNull;

extern crate alloc;

mod atom;
mod coro;
mod error;
mod function;
mod host;
mod iter;
mod list;
mod num;
mod opaque;
mod rc;
mod record;
mod string;
mod val;
mod vec;
mod vtable;
mod walker;

pub use self::{
    atom::{Atom, AtomId},
    coro::{CoroKind, CoroStatus, Coroutine, RcCoro, Resumable, erase_coro},
    error::RuntimeError,
    function::{RcFn, Function, Callable, erase_fn},
    host::{Host, Stack},
    iter::{IterVals, ValsIter, ValsIterStep},
    list::RcList,
    num::{Number, float_int_cmp, float_int_eq},
    opaque::{RcOpaque, erase},
    rc::{DynRc, Rc},
    record::RcRecord,
    string::RcStr,
    val::{Val, ValEnum, ValKind},
    vtable::AnyVTable,
    walker::FfiWaker,
};

pub use raft_ffi as ffi;

#[inline(always)]
unsafe fn cast_rc_inner<T>(ptr: raft_ffi::RcPtr<raft_ffi::Void>) -> raft_ffi::RcPtr<T> {
    ptr.cast::<raft_ffi::RcInner<T>>()
}

#[inline(always)]
fn erase_rc_inner<T>(ptr: raft_ffi::RcPtr<T>) -> raft_ffi::RcPtr<raft_ffi::Void> {
    ptr.cast::<raft_ffi::RcInner<raft_ffi::Void>>()
}

#[inline(always)]
fn raw_val_uninit() -> raft_ffi::RawVal {
    // SAFETY: `MASK_TAG_UNINIT` is a nonzero constant, never dereferenced
    // as a real pointer.
    raft_ffi::RawVal {
        tag: raft_ffi::RawTag {
            tag_ptr: unsafe { NonNull::new_unchecked(raft_ffi::MASK_TAG_UNINIT as *mut _) },
        },
        data: raft_ffi::RawData { nothing: () },
    }
}
