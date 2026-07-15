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

extern crate alloc;

pub mod rc;

pub use raft_ffi as ffi;
use raft_ffi::{LenVal, RcInner, Str, Void};

use alloc::{alloc::Layout, string::String, vec::Vec};

use core::{
    cell::UnsafeCell, cmp::Ordering, fmt, hash::{Hash, Hasher}, mem::ManuallyDrop, ops::Deref, ptr::NonNull,
};

use smallvec::SmallVec;

use crate::rc::{Callable, DynRc, Rc, erase, erase_fn};

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

#[derive(Copy, Clone)]
pub enum Number {
    Integer(i64),
    Float(f64),
}

impl Number {
    pub fn neg(self) -> Result<Number, RuntimeError> {
        match self {
            Number::Integer(i) => Ok(Number::Integer(i.wrapping_neg())),
            Number::Float(f) => Ok(Number::Float(-f)),
        }
    }

    pub fn add(self, rhs: Self) -> Result<Number, RuntimeError> {
        match (self, rhs) {
            (Number::Integer(i1), Number::Integer(i2)) => Ok(Number::Integer(i1.wrapping_add(i2))),
            (Number::Float(f1), Number::Float(f2)) => Ok(Number::Float(f1 + f2)),
            (Number::Integer(i), Number::Float(f)) => Ok(Number::Float((i as f64) + f)),
            (Number::Float(f), Number::Integer(i)) => Ok(Number::Float(f + (i as f64))),
        }
    }

    pub fn sub(self, rhs: Self) -> Result<Number, RuntimeError> {
        match (self, rhs) {
            (Number::Integer(i1), Number::Integer(i2)) => Ok(Number::Integer(i1.wrapping_sub(i2))),
            (Number::Float(f1), Number::Float(f2)) => Ok(Number::Float(f1 - f2)),
            (Number::Integer(i), Number::Float(f)) => Ok(Number::Float((i as f64) - f)),
            (Number::Float(f), Number::Integer(i)) => Ok(Number::Float(f - (i as f64))),
        }
    }

    pub fn mul(self, rhs: Self) -> Result<Number, RuntimeError> {
        match (self, rhs) {
            (Number::Integer(i1), Number::Integer(i2)) => Ok(Number::Integer(i1.wrapping_mul(i2))),
            (Number::Float(f1), Number::Float(f2)) => Ok(Number::Float(f1 * f2)),
            (Number::Integer(i), Number::Float(f)) => Ok(Number::Float((i as f64) * f)),
            (Number::Float(f), Number::Integer(i)) => Ok(Number::Float(f * (i as f64))),
        }
    }

    pub fn div(self, rhs: Self) -> Result<Number, RuntimeError> {
        match (self, rhs) {
            (Number::Integer(i1), Number::Integer(i2)) => {
                if i2 == 0 {
                    return Err(RuntimeError::Other("division by zero".into()));
                }
                Ok(Number::Integer(i1 / i2))
            }
            (Number::Float(f1), Number::Float(f2)) => Ok(Number::Float(f1 / f2)),
            (Number::Integer(i), Number::Float(f)) => Ok(Number::Float((i as f64) / f)),
            (Number::Float(f), Number::Integer(i)) => Ok(Number::Float(f / (i as f64))),
        }
    }

    pub fn pow(self, rhs: Self) -> Result<Number, RuntimeError> {
        match (self, rhs) {
            (Number::Integer(i1), Number::Integer(i2)) if i2 >= 0 => {
                Ok(Number::Integer(i1.wrapping_pow(i2 as u32)))
            }
            (Number::Integer(i1), Number::Integer(i2)) => {
                Ok(Number::Float(libm::pow(i1 as f64, i2 as f64)))
            }
            (Number::Float(f1), Number::Float(f2)) => Ok(Number::Float(libm::pow(f1, f2))),
            (Number::Integer(i), Number::Float(f)) => Ok(Number::Float(libm::pow(i as f64, f))),
            (Number::Float(f), Number::Integer(i)) => Ok(Number::Float(libm::pow(f, i as f64))),
        }
    }

    pub fn cmp(self, rhs: Self) -> Ordering {
        match (self, rhs) {
            (Number::Integer(i1), Number::Integer(i2)) => i1.cmp(&i2),
            (Number::Float(f1), Number::Float(f2)) => {
                f1.partial_cmp(&f2).unwrap_or(Ordering::Equal)
            }
            (Number::Integer(i), Number::Float(f)) => {
                (i as f64).partial_cmp(&f).unwrap_or(Ordering::Equal)
            }
            (Number::Float(f), Number::Integer(i)) => {
                f.partial_cmp(&(i as f64)).unwrap_or(Ordering::Equal)
            }
        }
    }
}

impl fmt::Debug for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Number::Integer(i) => write!(f, "{i}i"),
            Number::Float(fl) => write!(f, "{fl}f"),
        }
    }
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Number::Integer(i) => write!(f, "{i}"),
            Number::Float(fl) => write!(f, "{fl}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct AtomId(pub usize);

impl fmt::Display for AtomId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "atom#{:x}", self.0)
    }
}

#[derive(Clone, PartialEq, Eq)]
pub enum Atom {
    Nil,
    True,
    False,
    Custom(AtomId),
}

impl Hash for Atom {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Atom::Nil => "Nil".hash(state),
            Atom::True => "True".hash(state),
            Atom::False => "False".hash(state),
            Atom::Custom(id) => id.hash(state),
        }
    }
}

impl Atom {
    pub fn is_falsey(&self) -> bool {
        matches!(self, Atom::False | Atom::Nil)
    }

    pub fn is_false(&self) -> bool {
        matches!(self, Atom::False)
    }

    pub fn is_true(&self) -> bool {
        matches!(self, Atom::True)
    }

    pub fn is_nil(&self) -> bool {
        matches!(self, Atom::Nil)
    }
}

impl fmt::Debug for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Atom::Nil => write!(f, "Nil"),
            Atom::True => write!(f, "True"),
            Atom::False => write!(f, "False"),
            Atom::Custom(id) => write!(f, "{:?}", id),
        }
    }
}

impl fmt::Display for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Atom::Nil => write!(f, "Nil"),
            Atom::True => write!(f, "True"),
            Atom::False => write!(f, "False"),
            Atom::Custom(s) => write!(f, "{}", s),
        }
    }
}

#[derive(Clone, Debug)]
pub enum RuntimeError {
    UnboundIdentifier(RcStr),
    NotAFunction(RcStr),
    TypeError(RcStr),
    IndexError(RcStr),
    FieldError(RcStr),
    Other(RcStr),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeError::UnboundIdentifier(s) => write!(f, "Unbound identifier: {}", s),
            RuntimeError::NotAFunction(s) => write!(f, "Not a function: {}", s),
            RuntimeError::TypeError(s) => write!(f, "Type error: {}", s),
            RuntimeError::IndexError(s) => write!(f, "Index error: {}", s),
            RuntimeError::FieldError(s) => write!(f, "Field error: {}", s),
            RuntimeError::Other(s) => write!(f, "{}", s),
        }
    }
}

// ---------------------------------------------------------------------
// Val: repr(transparent) over ffi::RawVal, plus the tag/vtable plumbing.
// ---------------------------------------------------------------------

#[repr(transparent)]
pub struct Val {
    raw: ffi::RawVal,
}

impl Val {
    #[inline]
    fn tag_ptr(&self) -> *mut ffi::Void {
        self.raw.tag.tag_ptr.as_ptr()
    }

    /// A pure integer read for the `match` in `unpack`/`Drop` - never
    /// round-tripped back into a pointer, so no provenance concern here.
    #[inline]
    fn tag_bits(&self) -> usize {
        self.tag_ptr().addr() & ffi::MASK_BITS
    }

    #[inline]
    fn vtable_ptr<V>(&self) -> *const V {
        // Masking through `map_addr` (not a `usize` round-trip) keeps
        // this pointer's original provenance - the `&'static V` it was
        // built from in `pack_heap`.
        self.tag_ptr().map_addr(|a| a & !ffi::MASK_BITS).cast::<V>()
    }

    #[inline]
    fn vtable_ref<V>(&self) -> &'static V {
        // SAFETY: for heap tags, `tag_ptr`'s high bits are a `&'static V`
        // address stamped in by whichever `pack_heap` call built this
        // handle for this exact tag (see `From<ValEnum> for Val`).
        unsafe { &*self.vtable_ptr::<V>() }
    }

    #[inline]
    fn heap_ptr<T>(&self) -> raft_ffi::RcPtr<T> {
        // SAFETY: for heap tags, `data.ptr` is a live `RcInner<T>` box for
        // this exact `T`, set by the matching `pack_heap` call. `.cast()`
        // (not a `usize` round-trip) keeps its provenance.
        unsafe { self.raw.data.ptr.cast::<RcInner<T>>() }
    }

    /// Clone this handle's heap payload into an owned, independently-
    /// refcounted `DynRc`. The momentary, non-owning peek needed to reach
    /// `.clone()` is entirely local to this function and can never
    /// outlive `self`'s borrow - unlike a hypothetical `&self ->
    /// ManuallyDrop<DynRc<..>>` helper returning the peek itself would
    /// allow (nothing would then stop `self` from being dropped - and
    /// the reference it (uniquely, if this was the last one) owned freed
    /// - while that returned peek was still around, unrelated to
    /// anything `.clone()` itself does).
    #[inline]
    fn clone_heap<V: AnyVTable, T>(&self) -> DynRc<V, T> {
        // SAFETY: `heap_ptr`/`vtable_ref`'s contracts (heap tag, correct
        // `T`/`V`). Constructing/using a `ManuallyDrop` is always safe -
        // only dropping the inner value early or twice isn't - and nothing
        // here drops it: `.clone()` is a real, independent refcount bump,
        // and the `ManuallyDrop` peek itself is simply discarded after.
        let peek =
            ManuallyDrop::new(unsafe { DynRc::new(self.heap_ptr::<T>(), self.vtable_ref::<V>()) });
        (*peek).clone()
    }

    /// Tear down this handle's heap payload - reconstructs the exact same
    /// owning reference `self` already represented (not an extra one) and
    /// actually drops it, running real teardown exactly once. Only called
    /// from `Drop for Val`, once, on `self`'s own way out.
    #[inline]
    fn drop_heap<V: AnyVTable, T>(&mut self) {
        // SAFETY: as `clone_heap`'s, but this one genuinely owns the
        // reference (it's `self`'s own, being torn down), so actually
        // dropping it here - instead of peeking - is correct.
        drop(unsafe { DynRc::<V, T>::new(self.heap_ptr::<T>(), self.vtable_ref::<V>()) });
    }

    #[inline]
    fn from_tag_only(tag: usize) -> Val {
        Val {
            raw: ffi::RawVal {
                tag: ffi::RawTag {
                    // SAFETY: every `MASK_TAG_*` constant is nonzero.
                    tag_ptr: unsafe { NonNull::new_unchecked(tag as *mut ffi::Void) },
                },
                data: ffi::RawData { nothing: () },
            },
        }
    }

    #[inline]
    fn from_data_only(tag: usize, data: ffi::RawData) -> Val {
        Val {
            raw: ffi::RawVal {
                tag: ffi::RawTag {
                    // SAFETY: every `MASK_TAG_*` constant is nonzero.
                    tag_ptr: unsafe { NonNull::new_unchecked(tag as *mut ffi::Void) },
                },
                data,
            },
        }
    }

    fn pack_heap<V: AnyVTable, T>(tag: usize, dyn_rc: DynRc<V, T>) -> Val {
        let vtable_ptr = DynRc::vtable(&dyn_rc) as *const V;
        debug_assert_eq!(
            vtable_ptr.addr() & ffi::MASK_BITS,
            0,
            "vtable must be 16-byte aligned"
        );
        let data_ptr = DynRc::data_ptr(&dyn_rc);
        core::mem::forget(dyn_rc);
        // OR-ing the tag into the (nonzero, 16-byte-aligned) vtable
        // pointer's low bits through `map_addr` - not a `usize` round-trip
        // - keeps its provenance; `vtable_ref` recovers it the same way.
        let tagged = vtable_ptr.cast::<ffi::Void>().map_addr(|a| a | tag);
        Val {
            raw: ffi::RawVal {
                tag: ffi::RawTag {
                    // SAFETY: `tagged` is derived from a live `&'static V`,
                    // nonzero; OR-ing in the (also nonzero) tag bits keeps
                    // it nonzero.
                    tag_ptr: unsafe { NonNull::new_unchecked(tagged as *mut ffi::Void) },
                },
                data: ffi::RawData {
                    ptr: data_ptr.cast(),
                },
            },
        }
    }

    /// Move `self`'s raw bits out without running `Drop` - for handing
    /// ownership across a `RawVal`-returning boundary (the receiver
    /// becomes the new owner of whatever heap reference this held).
    #[inline(always)]
    fn into_raw(self) -> ffi::RawVal {
        // SAFETY: copying the bits out is fine - `mem::forget` below means
        // `self`'s own `Drop` never runs, so this isn't a double-owned copy.
        let raw = unsafe { core::ptr::read(&self.raw) };
        core::mem::forget(self);
        raw
    }

    #[inline(always)]
    fn from_raw(raw: ffi::RawVal) -> Val {
        Val { raw }
    }

    #[inline(always)]
    fn from_raw_ref(raw: &ffi::RawVal) -> &Val {
        // SAFETY: `Val` is `#[repr(transparent)]` over `RawVal`.
        unsafe { &*(raw as *const ffi::RawVal as *const Val) }
    }

    /// Clone a `Val` out of a borrowed `RawVal` (e.g. one embedded in a
    /// list/record entry) without taking ownership of the original.
    ///
    /// # Safety
    /// `raw` must be a valid `RawVal`.
    unsafe fn clone_raw(raw: &ffi::RawVal) -> Val {
        // SAFETY: bitwise-copying the tag/data union doesn't touch any
        // refcount; wrapping in `ManuallyDrop` means only the `.clone()`
        // below (a real, correct refcount bump for heap kinds) has any
        // effect - dropping the `ManuallyDrop` peek is a no-op.
        let peek = ManuallyDrop::new(Val { raw: unsafe { core::ptr::read(raw) } });
        (*peek).clone()
    }

    /// The ergonomic, `match`-able view of this value. Heap kinds bump a
    /// refcount (same cost as `.clone()`ing the underlying handle);
    /// scalars are free.
    #[inline(always)]
    pub fn unpack(&self) -> ValEnum {
        match self.tag_bits() {
            ffi::MASK_TAG_UNINIT => ValEnum::Uninit,
            ffi::MASK_TAG_INT => ValEnum::Number(Number::Integer(unsafe { self.raw.data.int })),
            ffi::MASK_TAG_FLOAT => ValEnum::Number(Number::Float(unsafe { self.raw.data.flt })),
            ffi::MASK_TAG_CHAR => {
                let bits = unsafe { self.raw.data.int } as u32;
                ValEnum::Char(char::from_u32(bits).unwrap_or('\u{FFFD}'))
            }
            ffi::MASK_TAG_ATOM_NIL => ValEnum::Atom(Atom::Nil),
            ffi::MASK_TAG_ATOM_TRUE => ValEnum::Atom(Atom::True),
            ffi::MASK_TAG_ATOM_FALSE => ValEnum::Atom(Atom::False),
            ffi::MASK_TAG_ATOM_ID => {
                let id = unsafe { self.raw.data.int } as usize;
                ValEnum::Atom(Atom::Custom(AtomId(id)))
            }
            ffi::MASK_TAG_STRING => ValEnum::String(RcStr {
                ptr: self.clone_heap::<raft_ffi::StringVTable, LenVal>(),
            }),
            ffi::MASK_TAG_RECORD => ValEnum::Record(RcRecord {
                ptr: self.clone_heap::<raft_ffi::RecordVTable, UnsafeCell<ffi::RecordVal>>(),
            }),
            ffi::MASK_TAG_LIST => ValEnum::List(RcList {
                ptr: self.clone_heap::<raft_ffi::ListVTable, UnsafeCell<ffi::ListVal>>(),
            }),
            ffi::MASK_TAG_FN => ValEnum::Fn(RcFn {
                ptr: self.clone_heap::<raft_ffi::FnVTable, Void>(),
            }),
            ffi::MASK_TAG_OPAQUE => ValEnum::Opaque(RcOpaque {
                ptr: self.clone_heap::<raft_ffi::AnyVTable, Void>(),
            }),
            _ => unsafe {
                core::hint::unreachable_unchecked()
            },
        }
    }

    /// Which variant this is, without touching a heap kind's refcount -
    /// unlike [`unpack`](Val::unpack), which always builds an owned
    /// [`ValEnum`] (a real `clone()` for heap kinds). Callers that only
    /// need to branch on shape - not read or hold the payload - should
    /// use this instead: `if val.kind() == ValKind::Fn { ... }` costs one
    /// masked pointer read, no `Cell` bump, no matching drop.
    #[inline(always)]
    pub fn kind(&self) -> ValKind {
        match self.tag_bits() {
            ffi::MASK_TAG_UNINIT => ValKind::Uninit,
            ffi::MASK_TAG_INT | ffi::MASK_TAG_FLOAT => ValKind::Number,
            ffi::MASK_TAG_CHAR => ValKind::Char,
            ffi::MASK_TAG_ATOM_NIL
            | ffi::MASK_TAG_ATOM_TRUE
            | ffi::MASK_TAG_ATOM_FALSE
            | ffi::MASK_TAG_ATOM_ID => ValKind::Atom,
            ffi::MASK_TAG_STRING => ValKind::String,
            ffi::MASK_TAG_RECORD => ValKind::Record,
            ffi::MASK_TAG_LIST => ValKind::List,
            ffi::MASK_TAG_FN => ValKind::Fn,
            ffi::MASK_TAG_OPAQUE => ValKind::Opaque,
            _ => unsafe { core::hint::unreachable_unchecked() },
        }
    }

    /// Raft truthiness, without cloning a heap kind just to check it.
    /// `0`/`0.0`/`False`/an empty list or record are falsey; everything
    /// else (including `Nil` - matching this crate's existing `is_falsey`
    /// free function) is truthy. List/Record only need their length,
    /// read directly off the heap header - no [`clone_heap`](Val::clone_heap).
    #[inline]
    pub fn is_falsey(&self) -> bool {
        match self.tag_bits() {
            ffi::MASK_TAG_INT => unsafe { self.raw.data.int == 0 },
            ffi::MASK_TAG_FLOAT => unsafe { self.raw.data.flt == 0.0 },
            ffi::MASK_TAG_ATOM_FALSE => true,
            ffi::MASK_TAG_RECORD => {
                // SAFETY: tag confirmed `Record`; `heap_ptr` is a live
                // `RcInner<UnsafeCell<RecordVal>>` for this exact tag.
                let inner = unsafe { self.heap_ptr::<UnsafeCell<ffi::RecordVal>>().as_ref() };
                unsafe { (*inner.value.get()).size == 0 }
            }
            ffi::MASK_TAG_LIST => {
                // SAFETY: as the `Record` arm's, for `ListVal`.
                let inner = unsafe { self.heap_ptr::<UnsafeCell<ffi::ListVal>>().as_ref() };
                unsafe { (*inner.value.get()).size == 0 }
            }
            _ => false,
        }
    }

    /// If this is a `Fn`, dispatch a call directly off `self`'s own
    /// borrow - no [`clone_heap`](Val::clone_heap)/[`RcFn`] built and torn
    /// down around it. Sound because the call is synchronous and `self`
    /// (which already owns a live reference) outlives it; equivalent to,
    /// but skips the refcount bump+drop pair that `unpack()` matching
    /// `ValEnum::Fn` and calling through the resulting `RcFn` would pay.
    #[inline]
    pub fn call_as_fn(&self, host: &mut rc::Host, args: usize) -> Option<usize> {
        if self.tag_bits() != ffi::MASK_TAG_FN {
            return None;
        }
        let vtable = self.vtable_ref::<raft_ffi::FnVTable>();
        // SAFETY: tag confirmed `Fn`; `heap_ptr` is a live `RcInner<Void>`
        // box for this exact tag. Raw place projection (never a reference)
        // keeps whole-box provenance on `data` - `rc::call_shim` walks it
        // back to the box header to reach the strong count.
        let data = unsafe {
            raft_ffi::VoidPtr::new_unchecked(&raw mut (*self.heap_ptr::<Void>().as_ptr()).value)
        };
        // SAFETY: crossing into the `extern "C"` `CallFn` ABI, same as
        // `RcFn::call` - `host.as_raw()` is the exact valid, exclusively
        // borrowed `RawHost` `host` wraps.
        Some(unsafe { (vtable.call)(data, args, host.as_raw()) })
    }

    /// Hand ownership of this value across an FFI boundary as its raw
    /// 2-word representation. The receiver becomes the owner of whatever
    /// heap reference this held - pair with [`Val::from_ffi`] on the other
    /// side, or leak it.
    #[inline(always)]
    pub fn into_ffi(self) -> ffi::RawVal {
        self.into_raw()
    }

    /// Take ownership of a `RawVal` received across an FFI boundary.
    ///
    /// # Safety
    /// `raw` must be a valid `RawVal` whose ownership is being transferred
    /// to this handle (produced by [`Val::into_ffi`] or an equivalent
    /// contract-honoring producer), not still owned by the other side.
    #[inline(always)]
    pub unsafe fn from_ffi(raw: ffi::RawVal) -> Val {
        Val::from_raw(raw)
    }

    pub fn new_uninit() -> Val {
        Val::from_tag_only(ffi::MASK_TAG_UNINIT)
    }

    pub fn new_int(i: i64) -> Val {
        Val::from_data_only(ffi::MASK_TAG_INT, ffi::RawData { int: i })
    }

    pub fn new_float(f: f64) -> Val {
        Val::from_data_only(ffi::MASK_TAG_FLOAT, ffi::RawData { flt: f })
    }

    pub fn new_char(c: char) -> Val {
        Val::from_data_only(ffi::MASK_TAG_CHAR, ffi::RawData { int: c as i64 })
    }

    pub fn new_nil() -> Val {
        Val::from_tag_only(ffi::MASK_TAG_ATOM_NIL)
    }

    pub fn new_false() -> Val {
        Val::from_tag_only(ffi::MASK_TAG_ATOM_FALSE)
    }

    pub fn new_true() -> Val {
        Val::from_tag_only(ffi::MASK_TAG_ATOM_TRUE)
    }
}

/// Cheap, clone-free discriminant for [`Val::kind`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ValKind {
    Uninit,
    Number,
    Char,
    Atom,
    String,
    List,
    Record,
    Fn,
    Opaque,
}

impl From<ValEnum> for Val {
    #[inline(always)]
    fn from(v: ValEnum) -> Val {
        match v {
            ValEnum::Uninit => Val::from_tag_only(ffi::MASK_TAG_UNINIT),
            ValEnum::Number(Number::Integer(i)) => {
                Val::from_data_only(ffi::MASK_TAG_INT, ffi::RawData { int: i })
            }
            ValEnum::Number(Number::Float(f)) => {
                Val::from_data_only(ffi::MASK_TAG_FLOAT, ffi::RawData { flt: f })
            }
            ValEnum::Char(c) => {
                Val::from_data_only(ffi::MASK_TAG_CHAR, ffi::RawData { int: c as i64 })
            }
            ValEnum::Atom(Atom::Nil) => Val::from_tag_only(ffi::MASK_TAG_ATOM_NIL),
            ValEnum::Atom(Atom::True) => Val::from_tag_only(ffi::MASK_TAG_ATOM_TRUE),
            ValEnum::Atom(Atom::False) => Val::from_tag_only(ffi::MASK_TAG_ATOM_FALSE),
            ValEnum::Atom(Atom::Custom(id)) => {
                Val::from_data_only(ffi::MASK_TAG_ATOM_ID, ffi::RawData { int: id.0 as i64 })
            }
            ValEnum::String(s) => Val::pack_heap(ffi::MASK_TAG_STRING, s.ptr),
            ValEnum::Record(r) => Val::pack_heap(ffi::MASK_TAG_RECORD, r.ptr),
            ValEnum::List(l) => Val::pack_heap(ffi::MASK_TAG_LIST, l.ptr),
            ValEnum::Fn(f) => Val::pack_heap(ffi::MASK_TAG_FN, f.ptr),
            ValEnum::Opaque(o) => Val::pack_heap(ffi::MASK_TAG_OPAQUE, o.ptr),
        }
    }
}

impl Clone for Val {
    /// Bumps a heap kind's refcount directly and bit-copies the raw
    /// tag/data - *not* `Val::from(self.unpack())`. That round trip would
    /// build a full `ValEnum` (a `RcStr`/`RcList`/.../`RcFn` wrapper) and
    /// then immediately re-pack it (`pack_heap`: re-fetch the vtable,
    /// re-derive the tagged pointer via `map_addr`, `NonNull` construction)
    /// - all to end up with the exact same bits this already has, since a
    /// clone never changes which allocation a `Val` points at. `unpack()`
    /// is for callers that actually want the ergonomic, kind-matched view;
    /// a clone doesn't.
    #[inline(always)]
    fn clone(&self) -> Val {
        match self.tag_bits() {
            ffi::MASK_TAG_STRING
            | ffi::MASK_TAG_RECORD
            | ffi::MASK_TAG_LIST
            | ffi::MASK_TAG_FN
            | ffi::MASK_TAG_OPAQUE => {
                // SAFETY: heap tag confirmed; `RcInner<T>::strong` is
                // `#[repr(C)]`'s first field regardless of `T` - the same
                // invariant `rc::erase`/`rc::erase_fn` already rely on when
                // they cast `RcPtr<T>` down to `RcPtr<Void>` and read
                // `.strong` through it. Bumping it here in place is exactly
                // what `DynRc::clone` does, minus building the `DynRc` at all.
                let inner = unsafe { self.heap_ptr::<Void>().as_ref() };
                inner.strong.set(inner.strong.get() + 1);
            }
            _ => {}
        }
        // SAFETY: bitwise-copying the tag/data union never touches a
        // refcount by itself - for heap kinds, the bump above already made
        // this an independent, correctly-counted owning reference; for
        // scalars there was never a count to touch.
        Val { raw: unsafe { core::ptr::read(&self.raw) } }
    }
}

impl Drop for Val {
    fn drop(&mut self) {
        match self.tag_bits() {
            ffi::MASK_TAG_STRING => self.drop_heap::<raft_ffi::StringVTable, LenVal>(),
            ffi::MASK_TAG_RECORD => {
                self.drop_heap::<raft_ffi::RecordVTable, UnsafeCell<ffi::RecordVal>>()
            }
            ffi::MASK_TAG_LIST => {
                self.drop_heap::<raft_ffi::ListVTable, UnsafeCell<ffi::ListVal>>()
            }
            ffi::MASK_TAG_FN => self.drop_heap::<raft_ffi::FnVTable, Void>(),
            ffi::MASK_TAG_OPAQUE => self.drop_heap::<raft_ffi::AnyVTable, Void>(),
            _ => {}
        }
    }
}

/// The ergonomic, ownable, `match`-able shape of a [`Val`]. See `Val`'s
/// own doc comment for how the two relate.
pub enum ValEnum {
    Number(Number),
    Char(char),
    String(RcStr),
    Atom(Atom),
    List(RcList),
    Record(RcRecord),
    Fn(RcFn),
    Opaque(RcOpaque),
    /// Internal sentinel: a local slot that has not been assigned yet
    /// (reads of it fall back to the global scope), or "not found" from
    /// a list/record lookup. Never observable from Raft code or host
    /// functions.
    #[doc(hidden)]
    Uninit,
}

impl core::cmp::PartialEq for Val {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Some(Ordering::Equal)
    }
}

impl fmt::Debug for Val {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.unpack() {
            ValEnum::Number(n) => write!(f, "Number({:?})", n),
            ValEnum::Char(c) => write!(f, "Char({:?})", c),
            ValEnum::String(s) => write!(f, "String({:?})", s.as_str()),
            ValEnum::Atom(a) => write!(f, "Atom({:?})", a),
            ValEnum::List(l) => write!(f, "List({:?})", l),
            ValEnum::Record(r) => write!(f, "Record({:?})", r),
            ValEnum::Fn(_) => write!(f, "<fn>"),
            ValEnum::Opaque(o) => write!(f, "Opaque({:p})", DynRc::data_ptr(&o.ptr).as_ptr()),
            ValEnum::Uninit => write!(f, "<uninit>"),
        }
    }
}

impl fmt::Display for Val {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.unpack() {
            ValEnum::Number(n) => write!(f, "{}", n),
            ValEnum::Char(c) => write!(f, "{}", c),
            ValEnum::String(s) => write!(f, "{}", s.as_str()),
            ValEnum::Atom(a) => write!(f, "{}", a),
            ValEnum::List(l) => write!(f, "{}", l),
            ValEnum::Record(r) => write!(f, "{}", r),
            ValEnum::Fn(_) => write!(f, "<fn>"),
            ValEnum::Opaque(o) => write!(f, "{:p}", DynRc::data_ptr(&o.ptr).as_ptr()),
            ValEnum::Uninit => write!(f, "<uninit>"),
        }
    }
}

impl Val {
    #[inline]
    pub fn bool_(b: bool) -> Self {
        Val::from(ValEnum::Atom(if b { Atom::True } else { Atom::False }))
    }

    #[inline]
    pub fn true_() -> Val {
        Val::from(ValEnum::Atom(Atom::True))
    }

    #[inline]
    pub fn false_() -> Val {
        Val::from(ValEnum::Atom(Atom::False))
    }

    #[inline]
    pub fn nil() -> Val {
        Val::from(ValEnum::Atom(Atom::Nil))
    }

    #[inline]
    #[doc(hidden)]
    pub fn is_init(&self) -> bool {
        self.tag_bits() != ffi::MASK_TAG_UNINIT
    }

    #[inline]
    #[doc(hidden)]
    pub fn init_or<E>(self, err: E) -> Result<Val, E> {
        if self.is_init() { Ok(self) } else { Err(err) }
    }

    #[inline]
    #[doc(hidden)]
    pub fn init_or_else<F, E>(self, f: F) -> Result<Val, E>
    where
        F: FnOnce() -> E,
    {
        if self.is_init() { Ok(self) } else { Err(f()) }
    }

    #[inline]
    pub fn new_atom(id: AtomId) -> Val {
        Val::from(ValEnum::Atom(Atom::Custom(id)))
    }

    #[inline]
    pub fn string(s: &str) -> Val {
        Val::from(ValEnum::String(RcStr::new(s)))
    }

    #[inline]
    pub fn list(elements: Vec<Val>) -> Val {
        Val::from(ValEnum::List(RcList::new(elements)))
    }

    #[inline]
    pub fn record(fields: Vec<(RcStr, Val)>) -> Val {
        Val::from(ValEnum::Record(RcRecord::new(fields)))
    }

    #[inline]
    pub fn opaque<T: 'static>(value: T) -> Val {
        Val::from(ValEnum::Opaque(RcOpaque::new(value)))
    }

    #[inline]
    pub fn pos(&self) -> Result<Val, RuntimeError> {
        match self.unpack() {
            ValEnum::Number(n) => Ok(Val::from(ValEnum::Number(n))),
            _ => Err(RuntimeError::TypeError("pos on non-numeric value".into())),
        }
    }

    #[inline]
    pub fn neg(&self) -> Result<Val, RuntimeError> {
        match self.unpack() {
            ValEnum::Number(n) => Ok(Val::from(ValEnum::Number(n.neg()?))),
            _ => Err(RuntimeError::TypeError(
                "negation on non-numeric value".into(),
            )),
        }
    }

    #[inline]
    pub fn not(&self) -> Val {
        Val::bool_(is_falsey(self))
    }

    #[inline]
    pub fn bit_not(&self) -> Result<Val, RuntimeError> {
        match self.unpack() {
            ValEnum::Number(Number::Integer(i)) => {
                Ok(Val::from(ValEnum::Number(Number::Integer(!i))))
            }
            _ => Err(RuntimeError::TypeError(
                "bitwise not on non-integer value".into(),
            )),
        }
    }

    #[inline]
    pub fn add(&self, rhs: &Val) -> Result<Val, RuntimeError> {
        match (self.unpack(), rhs.unpack()) {
            (ValEnum::Number(n1), ValEnum::Number(n2)) => {
                Ok(Val::from(ValEnum::Number(n1.add(n2)?)))
            }
            (ValEnum::String(s1), ValEnum::String(s2)) => {
                let mut s = String::new();
                s.push_str(s1.as_str());
                s.push_str(s2.as_str());
                Ok(Val::string(&s))
            }
            (ValEnum::String(s1), ValEnum::Char(c2)) => {
                let mut s = String::new();
                s.push_str(s1.as_str());
                s.push(c2);
                Ok(Val::string(&s))
            }
            (ValEnum::Char(c1), ValEnum::String(s2)) => {
                let mut s = String::new();
                s.push(c1);
                s.push_str(s2.as_str());
                Ok(Val::string(&s))
            }
            (ValEnum::Char(c1), ValEnum::Char(c2)) => {
                let mut s = String::new();
                s.push(c1);
                s.push(c2);
                Ok(Val::string(&s))
            }
            _ => Err(RuntimeError::TypeError(
                "addition on not numeric or string value".into(),
            )),
        }
    }

    #[inline]
    pub fn sub(&self, rhs: &Val) -> Result<Val, RuntimeError> {
        match (self.unpack(), rhs.unpack()) {
            (ValEnum::Number(n1), ValEnum::Number(n2)) => {
                Ok(Val::from(ValEnum::Number(n1.sub(n2)?)))
            }
            _ => Err(RuntimeError::TypeError(
                "subtraction on non-numeric value".into(),
            )),
        }
    }

    #[inline]
    pub fn mul(&self, rhs: &Val) -> Result<Val, RuntimeError> {
        match (self.unpack(), rhs.unpack()) {
            (ValEnum::Number(n1), ValEnum::Number(n2)) => {
                Ok(Val::from(ValEnum::Number(n1.mul(n2)?)))
            }
            _ => Err(RuntimeError::TypeError(
                "multiplication on non-numeric value".into(),
            )),
        }
    }

    #[inline]
    pub fn div(&self, rhs: &Val) -> Result<Val, RuntimeError> {
        match (self.unpack(), rhs.unpack()) {
            (ValEnum::Number(n1), ValEnum::Number(n2)) => {
                Ok(Val::from(ValEnum::Number(n1.div(n2)?)))
            }
            _ => Err(RuntimeError::TypeError(
                "division on non-numeric value".into(),
            )),
        }
    }

    #[inline]
    pub fn pow(&self, rhs: &Val) -> Result<Val, RuntimeError> {
        match (self.unpack(), rhs.unpack()) {
            (ValEnum::Number(n1), ValEnum::Number(n2)) => {
                Ok(Val::from(ValEnum::Number(n1.pow(n2)?)))
            }
            _ => Err(RuntimeError::TypeError(
                "exponentiation on non-numeric value".into(),
            )),
        }
    }

    #[inline]
    pub fn cmp(&self, rhs: &Val) -> Option<Ordering> {
        match (self.unpack(), rhs.unpack()) {
            (ValEnum::Number(n1), ValEnum::Number(n2)) => Some(n1.cmp(n2)),
            (ValEnum::Atom(a1), ValEnum::Atom(a2)) => {
                if a1 == a2 {
                    Some(Ordering::Equal)
                } else {
                    None
                }
            }
            (ValEnum::String(s1), ValEnum::String(s2)) => Some(s1.as_str().cmp(s2.as_str())),
            (ValEnum::Char(c1), ValEnum::Char(c2)) => Some(c1.cmp(&c2)),
            (ValEnum::List(l1), ValEnum::List(l2)) => l1.cmp(&l2),
            (ValEnum::Record(r1), ValEnum::Record(r2)) => {
                DynRc::ptr_eq(&r1.ptr, &r2.ptr).then_some(Ordering::Equal)
            }
            (ValEnum::Fn(f1), ValEnum::Fn(f2)) => {
                DynRc::ptr_eq(&f1.ptr, &f2.ptr).then_some(Ordering::Equal)
            }
            (ValEnum::Opaque(o1), ValEnum::Opaque(o2)) => {
                DynRc::ptr_eq(&o1.ptr, &o2.ptr).then_some(Ordering::Equal)
            }
            _ => None, // different kinds are considered incomparable
        }
    }

    #[inline]
    pub fn bit_and(&self, rhs: &Val) -> Result<Val, RuntimeError> {
        match (self.unpack(), rhs.unpack()) {
            (ValEnum::Number(Number::Integer(i1)), ValEnum::Number(Number::Integer(i2))) => {
                Ok(Val::from(ValEnum::Number(Number::Integer(i1 & i2))))
            }
            _ => Err(RuntimeError::TypeError(
                "bitwise and on non-integer value".into(),
            )),
        }
    }

    #[inline]
    pub fn bit_or(&self, rhs: &Val) -> Result<Val, RuntimeError> {
        match (self.unpack(), rhs.unpack()) {
            (ValEnum::Number(Number::Integer(i1)), ValEnum::Number(Number::Integer(i2))) => {
                Ok(Val::from(ValEnum::Number(Number::Integer(i1 | i2))))
            }
            _ => Err(RuntimeError::TypeError(
                "bitwise or on non-integer value".into(),
            )),
        }
    }

    #[inline]
    pub fn bit_xor(&self, rhs: &Val) -> Result<Val, RuntimeError> {
        match (self.unpack(), rhs.unpack()) {
            (ValEnum::Number(Number::Integer(i1)), ValEnum::Number(Number::Integer(i2))) => {
                Ok(Val::from(ValEnum::Number(Number::Integer(i1 ^ i2))))
            }
            _ => Err(RuntimeError::TypeError(
                "bitwise xor on non-integer value".into(),
            )),
        }
    }

    #[inline]
    pub fn shl(&self, rhs: &Val) -> Result<Val, RuntimeError> {
        match (self.unpack(), rhs.unpack()) {
            (ValEnum::Number(Number::Integer(i1)), ValEnum::Number(Number::Integer(i2))) => {
                if i2 < 0 {
                    return Err(RuntimeError::TypeError(
                        "shift left by negative value".into(),
                    ));
                }
                Ok(Val::from(ValEnum::Number(Number::Integer(
                    i1.wrapping_shl(i2 as u32),
                ))))
            }
            _ => Err(RuntimeError::TypeError(
                "shift left on non-integer value".into(),
            )),
        }
    }

    #[inline]
    pub fn shr(&self, rhs: &Val) -> Result<Val, RuntimeError> {
        match (self.unpack(), rhs.unpack()) {
            (ValEnum::Number(Number::Integer(i1)), ValEnum::Number(Number::Integer(i2))) => {
                if i2 < 0 {
                    return Err(RuntimeError::TypeError(
                        "shift right by negative value".into(),
                    ));
                }
                Ok(Val::from(ValEnum::Number(Number::Integer(
                    i1.wrapping_shr(i2 as u32),
                ))))
            }
            _ => Err(RuntimeError::TypeError(
                "shift right on non-integer value".into(),
            )),
        }
    }

    pub fn get_field(&self, key: &str) -> Option<Val> {
        match self.unpack() {
            ValEnum::Record(r) => r.get_field(key),
            _ => None,
        }
    }

    pub fn get_index(&self, index: usize) -> Option<Val> {
        match self.unpack() {
            ValEnum::List(l) => l.get(index),
            _ => None,
        }
    }

    #[inline]
    pub fn iter(&self) -> Result<impl IntoIterator<Item = Val> + use<>, RuntimeError> {
        enum ValIter {
            List(RcList, usize),
            Record(RcRecord, usize),
        }

        impl Iterator for ValIter {
            type Item = Val;

            #[inline]
            fn next(&mut self) -> Option<Val> {
                match self {
                    ValIter::List(l, pos) => {
                        let item = l.get(*pos)?;
                        *pos += 1;
                        Some(item)
                    }
                    ValIter::Record(r, pos) => {
                        let (key, value) = r.entry_at(*pos)?;
                        *pos += 1;
                        Some(Val::record(alloc::vec![(RcStr::new(key), value)]))
                    }
                }
            }
        }

        match self.unpack() {
            ValEnum::List(l) => Ok(ValIter::List(l, 0)),
            ValEnum::Record(r) => Ok(ValIter::Record(r, 0)),
            _ => Err(RuntimeError::TypeError(
                "iteration on non-heap value".into(),
            )),
        }
    }

    /// Wrap a host closure into a function value with the given
    /// argument-count hint. `(0, None)` means "takes anything" - the
    /// closure then decides how many arguments to consume.
    #[inline]
    pub fn host_function<F>(min_args: usize, max_args: Option<usize>, f: F) -> Val
    where
        F: Fn(&mut rc::Host, usize) -> Val + 'static,
    {
        Val::from(ValEnum::Fn(RcFn::new(HostFn {
            min_args,
            max_args,
            fun: f,
        })))
    }
}

pub fn is_falsey(v: &Val) -> bool {
    v.is_falsey()
}

// ---------------------------------------------------------------------
// RcStr: Val::String's backing. Immutable once built (`LenVal`'s inline
// flexible-array shape suits this fine - no growth needed).
// ---------------------------------------------------------------------

unsafe extern "C" fn string_destroy(ptr: raft_ffi::RcPtr<Void>) {
    unsafe {
        let ptr = ptr.cast::<RcInner<LenVal>>();
        let len = (*ptr.as_ptr()).value.len;
        let (layout, _) = Layout::new::<RcInner<LenVal>>()
            .extend(Layout::array::<u8>(len).unwrap())
            .unwrap();
        alloc::alloc::dealloc(ptr.as_ptr() as *mut u8, layout.pad_to_align());
    }
}

static STRING_VTABLE: raft_ffi::StringVTable = raft_ffi::StringVTable {
    any: raft_ffi::AnyVTable {
        destroy: string_destroy,
    },
};

#[repr(transparent)]
#[derive(Clone)]
pub struct RcStr {
    ptr: DynRc<raft_ffi::StringVTable, LenVal>,
}

impl RcStr {
    pub fn new(s: &str) -> Self {
        let header_layout = Layout::new::<RcInner<LenVal>>();
        let (layout, offset) = header_layout
            .extend(Layout::for_value(s.as_bytes()))
            .expect("RcStr layout overflow");
        let layout = layout.pad_to_align();

        // SAFETY: `layout` is non-zero-sized (header alone is nonzero).
        let raw = unsafe { alloc::alloc::alloc(layout) };
        let Some(base) = NonNull::new(raw) else {
            alloc::alloc::handle_alloc_error(layout);
        };

        let inner = base.cast::<RcInner<LenVal>>();
        unsafe {
            inner.as_ptr().write(RcInner {
                strong: core::cell::Cell::new(1),
                value: LenVal {
                    len: s.len(),
                    val: [],
                },
            });
            let data_ptr = base.as_ptr().add(offset);
            core::ptr::copy_nonoverlapping(s.as_ptr(), data_ptr, s.len());
        }

        RcStr {
            ptr: unsafe { DynRc::new(inner, &STRING_VTABLE) },
        }
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        let lenval = &self.ptr.rc_box().value;
        // SAFETY: bytes were copied in by `new`, from a valid `&str`.
        unsafe {
            core::str::from_utf8_unchecked(core::slice::from_raw_parts(
                lenval.val.as_ptr(),
                lenval.len,
            ))
        }
    }
}

impl Deref for RcStr {
    type Target = str;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl From<&str> for RcStr {
    fn from(s: &str) -> Self {
        RcStr::new(s)
    }
}

impl From<String> for RcStr {
    fn from(s: String) -> Self {
        RcStr::new(&s)
    }
}

impl From<alloc::rc::Rc<str>> for RcStr {
    fn from(s: alloc::rc::Rc<str>) -> Self {
        RcStr::new(&s)
    }
}

impl AsRef<str> for RcStr {
    #[inline]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Hash for RcStr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl core::borrow::Borrow<str> for RcStr {
    #[inline]
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl PartialEq for RcStr {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl PartialEq<str> for RcStr {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<RcStr> for str {
    fn eq(&self, other: &RcStr) -> bool {
        self == other.as_str()
    }
}

impl Eq for RcStr {}

impl PartialOrd for RcStr {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RcStr {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_str().cmp(other.as_str())
    }
}

impl fmt::Debug for RcStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.as_str())
    }
}

impl fmt::Display for RcStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ---------------------------------------------------------------------
// RcList: Val::List's backing. `ffi::ListVal` is a `RawVec<RawVal>` -
// growable, via `Vec<Val>`'s own realloc, in place, behind a stable
// outer `RcInner` (only the *inner* buffer moves on push).
// ---------------------------------------------------------------------

unsafe extern "C" fn list_destroy(ptr: raft_ffi::RcPtr<Void>) {
    let ptr = ptr.cast::<RcInner<UnsafeCell<ffi::ListVal>>>();
    // SAFETY: strong count just hit zero (`DynRc::drop`, the only
    // caller) - this is the sole reference. Turning it into a reference
    // immediately confines everything after to safe slice/Vec ops.
    let inner: &mut RcInner<UnsafeCell<ffi::ListVal>> = unsafe { &mut *ptr.as_ptr() };
    let header = inner.value.get_mut();
    // SAFETY: `header`'s ptr/size/capacity are valid `Vec<Val>` raw parts.
    let elements = unsafe { Vec::from_raw_parts(header.ptr as *mut Val, header.size, header.capacity) };
    drop(elements); // drops each `Val` (real teardown), then frees the buffer
    // SAFETY: deallocating the exact allocation `RcList::new` made.
    unsafe {
        alloc::alloc::dealloc(
            ptr.as_ptr() as *mut u8,
            Layout::new::<RcInner<UnsafeCell<ffi::ListVal>>>(),
        );
    }
}

unsafe extern "C" fn list_get_shim(data: raft_ffi::VoidPtr, index: usize) -> ffi::RawVal {
    // SAFETY: `data` is a live `ffi::ListVal` for the duration of this call.
    let header: &ffi::ListVal = unsafe { &*(data.as_ptr() as *const ffi::ListVal) };
    // SAFETY: `header.ptr`/`size` describe a valid `[Val]`.
    let slice: &[Val] = unsafe { core::slice::from_raw_parts(header.ptr as *const Val, header.size) };
    match slice.get(index) {
        Some(v) => v.clone().into_raw(),
        None => Val::from(ValEnum::Uninit).into_raw(),
    }
}

unsafe extern "C" fn list_set_shim(data: raft_ffi::VoidPtr, index: usize, val: ffi::RawVal) {
    let header: &ffi::ListVal = unsafe { &*(data.as_ptr() as *const ffi::ListVal) };
    let slice: &mut [Val] =
        unsafe { core::slice::from_raw_parts_mut(header.ptr as *mut Val, header.size) };
    if let Some(slot) = slice.get_mut(index) {
        *slot = Val::from_raw(val); // assignment drops the old value
    }
}

unsafe extern "C" fn list_swap_shim(
    data: raft_ffi::VoidPtr,
    index: usize,
    val: ffi::RawVal,
) -> ffi::RawVal {
    let header: &ffi::ListVal = unsafe { &*(data.as_ptr() as *const ffi::ListVal) };
    let slice: &mut [Val] =
        unsafe { core::slice::from_raw_parts_mut(header.ptr as *mut Val, header.size) };
    match slice.get_mut(index) {
        Some(slot) => core::mem::replace(slot, Val::from_raw(val)).into_raw(),
        None => val,
    }
}

unsafe extern "C" fn list_push_shim(data: raft_ffi::VoidPtr, val: ffi::RawVal) {
    let header: &mut ffi::ListVal = unsafe { &mut *(data.as_ptr() as *mut ffi::ListVal) };
    // SAFETY: `header` describes valid `RawVec<RawVal>` raw parts.
    unsafe { rc::raw_vec_push(header, val) };
}

unsafe extern "C" fn list_pop_shim(data: raft_ffi::VoidPtr) -> ffi::RawVal {
    let header: &mut ffi::ListVal = unsafe { &mut *(data.as_ptr() as *mut ffi::ListVal) };
    // SAFETY: as `list_push_shim`.
    unsafe { rc::raw_vec_pop(header) }.unwrap_or_else(|| Val::from(ValEnum::Uninit).into_raw())
}

unsafe extern "C" fn list_elements_shim(data: raft_ffi::VoidPtr) -> *const ffi::RawVal {
    let header: &ffi::ListVal = unsafe { &*(data.as_ptr() as *const ffi::ListVal) };
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
    ptr: DynRc<raft_ffi::ListVTable, UnsafeCell<ffi::ListVal>>,
}

impl RcList {
    pub fn new(elements: Vec<Val>) -> Self {
        let len = elements.len();
        let cap = elements.capacity();
        let mut elements = ManuallyDrop::new(elements);
        let header = ffi::ListVal {
            ptr: elements.as_mut_ptr() as *mut ffi::RawVal,
            size: len,
            capacity: cap,
        };
        let rc = Rc::new(UnsafeCell::new(header));
        RcList {
            ptr: unsafe { DynRc::new(Rc::into_raw_box(rc).cast(), &LIST_VTABLE) },
        }
    }

    #[inline]
    fn header(&self) -> &ffi::ListVal {
        // SAFETY: single-threaded; no live `&mut` alias held across this.
        unsafe { &*self.ptr.rc_box().value.get() }
    }

    pub fn len(&self) -> usize {
        self.header().size
    }

    pub fn as_slice(&self) -> &[Val] {
        let header = self.header();
        unsafe { core::slice::from_raw_parts(header.ptr as *const Val, header.size) }
    }

    pub fn get(&self, index: usize) -> Option<Val> {
        self.as_slice().get(index).map(|v| v.clone())
    }

    #[inline]
    fn value_ptr(&self) -> raft_ffi::VoidPtr {
        // SAFETY: `self.ptr.rc_box().value.get()` is always a live,
        // correctly-aligned `*mut ffi::ListVal`.
        unsafe { raft_ffi::VoidPtr::new_unchecked(self.ptr.rc_box().value.get() as *mut Void) }
    }

    /// `target[index] = value` - in place, no length change.
    pub fn set(&self, index: usize, val: Val) {
        let vtable = DynRc::vtable(&self.ptr);
        // SAFETY: `value_ptr()` is a live `ffi::ListVal` for `vtable`.
        unsafe { (vtable.set)(self.value_ptr(), index, val.into_raw()) };
    }

    /// Append to the end, growing in place (the outer handle stays valid
    /// - see `RcList`'s doc comment).
    pub fn push(&self, val: Val) {
        let vtable = DynRc::vtable(&self.ptr);
        // SAFETY: as `set`'s.
        unsafe { (vtable.push)(self.value_ptr(), val.into_raw()) };
    }

    /// Remove and return the last element, if any.
    pub fn pop(&self) -> Option<Val> {
        if self.len() == 0 {
            return None;
        }
        let vtable = DynRc::vtable(&self.ptr);
        // SAFETY: as `set`'s.
        let popped = unsafe { (vtable.pop)(self.value_ptr()) };
        Some(Val::from_raw(popped))
    }

    pub fn cmp(&self, other: &RcList) -> Option<Ordering> {
        let (a, b) = (self.as_slice(), other.as_slice());
        for (x, y) in a.iter().zip(b.iter()) {
            match x.cmp(y) {
                Some(Ordering::Equal) => continue,
                other => return other,
            }
        }
        Some(a.len().cmp(&b.len()))
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

// ---------------------------------------------------------------------
// RcRecord: Val::Record's backing. `ffi::RecordVal` is a
// `RawVec<RawFieldVal>`; each entry's `name` is a borrowed `Str` view
// into its own small, singly-owned (non-refcounted - never aliased)
// allocation, freed explicitly on overwrite/removal/destroy.
// ---------------------------------------------------------------------

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

unsafe extern "C" fn record_destroy(ptr: raft_ffi::RcPtr<Void>) {
    let ptr = ptr.cast::<RcInner<UnsafeCell<ffi::RecordVal>>>();
    // SAFETY: strong count just hit zero (`DynRc::drop`, the only
    // caller) - this is the sole reference. Turning it into a reference
    // immediately confines everything after to safe operations.
    let inner: &mut RcInner<UnsafeCell<ffi::RecordVal>> = unsafe { &mut *ptr.as_ptr() };
    let header = inner.value.get_mut();
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
        }
        drop(Val::from_raw(unsafe { core::ptr::read(&field.val) }));
    }
    drop(fields);
    // SAFETY: deallocating the exact allocation `RcRecord::new` made.
    unsafe {
        alloc::alloc::dealloc(
            ptr.as_ptr() as *mut u8,
            Layout::new::<RcInner<UnsafeCell<ffi::RecordVal>>>(),
        );
    }
}

fn record_fields_slice(header: &ffi::RecordVal) -> &[ffi::RawFieldVal] {
    // SAFETY: `header.ptr`/`size` describe a valid `[RawFieldVal]`.
    unsafe { core::slice::from_raw_parts(header.ptr, header.size) }
}

unsafe extern "C" fn record_fields_shim(data: raft_ffi::VoidPtr) -> *const ffi::RawFieldVal {
    let header: &ffi::RecordVal = unsafe { &*(data.as_ptr() as *const ffi::RecordVal) };
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
) -> ffi::RawVal {
    let header: &ffi::RecordVal = unsafe { &*(data.as_ptr() as *const ffi::RecordVal) };
    let key: &str = unsafe { key_view(name_ptr, name_len) };
    match record_fields_slice(header).iter().find(|f| key_str(&f.name) == key) {
        Some(f) => unsafe { Val::clone_raw(&f.val).into_raw() },
        None => Val::from(ValEnum::Uninit).into_raw(),
    }
}

unsafe extern "C" fn record_get_shim(data: raft_ffi::VoidPtr, index: usize) -> ffi::RawVal {
    let header: &ffi::RecordVal = unsafe { &*(data.as_ptr() as *const ffi::RecordVal) };
    match record_fields_slice(header).get(index) {
        Some(f) => unsafe { Val::clone_raw(&f.val).into_raw() },
        None => Val::from(ValEnum::Uninit).into_raw(),
    }
}

unsafe extern "C" fn record_set_by_name_shim(
    data: raft_ffi::VoidPtr,
    name_ptr: *const u8,
    name_len: usize,
    val: ffi::RawVal,
) {
    let header: &mut ffi::RecordVal = unsafe { &mut *(data.as_ptr() as *mut ffi::RecordVal) };
    let key: &str = unsafe { key_view(name_ptr, name_len) };
    {
        // Scoped so this slice's borrow of `header` ends before the
        // (mutually-exclusive) `raw_vec_push` borrow below, in the
        // not-found case.
        let slice: &mut [ffi::RawFieldVal] =
            unsafe { core::slice::from_raw_parts_mut(header.ptr, header.size) };
        if let Some(entry) = slice.iter_mut().find(|f| key_str(&f.name) == key) {
            drop(Val::from_raw(core::mem::replace(&mut entry.val, val)));
            return;
        }
    }
    // SAFETY: `header` describes valid `RawVec<RawFieldVal>` raw parts.
    unsafe {
        rc::raw_vec_push(
            header,
            ffi::RawFieldVal {
                name: alloc_key_bytes(key),
                val,
            },
        )
    };
}

unsafe extern "C" fn record_set_shim(data: raft_ffi::VoidPtr, index: usize, val: ffi::RawVal) {
    let header: &ffi::RecordVal = unsafe { &*(data.as_ptr() as *const ffi::RecordVal) };
    let slice: &mut [ffi::RawFieldVal] =
        unsafe { core::slice::from_raw_parts_mut(header.ptr, header.size) };
    if let Some(entry) = slice.get_mut(index) {
        drop(Val::from_raw(core::mem::replace(&mut entry.val, val)));
    }
}

unsafe extern "C" fn record_swap_shim(
    data: raft_ffi::VoidPtr,
    index: usize,
    val: ffi::RawVal,
) -> ffi::RawVal {
    let header: &ffi::RecordVal = unsafe { &*(data.as_ptr() as *const ffi::RecordVal) };
    let slice: &mut [ffi::RawFieldVal] =
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
    val: ffi::RawVal,
) -> ffi::RawVal {
    let header: &ffi::RecordVal = unsafe { &*(data.as_ptr() as *const ffi::RecordVal) };
    let key: &str = unsafe { key_view(name_ptr, name_len) };
    let slice: &mut [ffi::RawFieldVal] =
        unsafe { core::slice::from_raw_parts_mut(header.ptr, header.size) };
    match slice.iter_mut().find(|f| key_str(&f.name) == key) {
        Some(entry) => core::mem::replace(&mut entry.val, val),
        None => val,
    }
}

fn record_rem_at(header: &mut ffi::RecordVal, index: usize) {
    if index >= header.size {
        return;
    }
    // SAFETY: `header` describes valid `RawVec<RawFieldVal>` raw parts;
    // `index < header.size` (checked above).
    let removed = unsafe { rc::raw_vec_remove(header, index) };
    unsafe { free_key_bytes(removed.name) };
    drop(Val::from_raw(removed.val));
}

unsafe extern "C" fn record_rem_shim(data: raft_ffi::VoidPtr, index: usize) {
    let header: &mut ffi::RecordVal = unsafe { &mut *(data.as_ptr() as *mut ffi::RecordVal) };
    record_rem_at(header, index);
}

unsafe extern "C" fn record_rem_by_name_shim(
    data: raft_ffi::VoidPtr,
    name_ptr: *const u8,
    name_len: usize,
) {
    let header: &mut ffi::RecordVal = unsafe { &mut *(data.as_ptr() as *mut ffi::RecordVal) };
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
    ptr: DynRc<raft_ffi::RecordVTable, UnsafeCell<ffi::RecordVal>>,
}

impl RcRecord {
    pub fn new(fields: Vec<(RcStr, Val)>) -> Self {
        let mut vec: Vec<ffi::RawFieldVal> = Vec::with_capacity(fields.len());
        for (key, val) in fields {
            vec.push(ffi::RawFieldVal {
                name: alloc_key_bytes(key.as_str()),
                val: val.into_raw(),
            });
        }
        let mut vec = ManuallyDrop::new(vec);
        let header = ffi::RecordVal {
            ptr: vec.as_mut_ptr(),
            size: vec.len(),
            capacity: vec.capacity(),
        };
        let rc = Rc::new(UnsafeCell::new(header));
        RcRecord {
            ptr: unsafe { DynRc::new(Rc::into_raw_box(rc).cast(), &RECORD_VTABLE) },
        }
    }

    #[inline]
    fn header(&self) -> &ffi::RecordVal {
        // SAFETY: single-threaded; no live `&mut` alias held across this.
        unsafe { &*self.ptr.rc_box().value.get() }
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
    fn value_ptr(&self) -> raft_ffi::VoidPtr {
        // SAFETY: `self.ptr.rc_box().value.get()` is always a live,
        // correctly-aligned `*mut ffi::RecordVal`.
        unsafe { raft_ffi::VoidPtr::new_unchecked(self.ptr.rc_box().value.get() as *mut Void) }
    }

    /// Insert (or overwrite) a field by name.
    pub fn set_field(&self, key: &str, val: Val) {
        let vtable = DynRc::vtable(&self.ptr);
        // SAFETY: `value_ptr()` is a live `ffi::RecordVal` for `vtable`;
        // `key.as_ptr()`/`.len()` describe a valid UTF-8 view for the
        // duration of this call.
        unsafe { (vtable.set_by_name)(self.value_ptr(), key.as_ptr(), key.len(), val.into_raw()) };
    }

    /// Remove a field by name, if present.
    pub fn remove_field(&self, key: &str) {
        let vtable = DynRc::vtable(&self.ptr);
        // SAFETY: as `set_field`'s.
        unsafe { (vtable.rem_by_name)(self.value_ptr(), key.as_ptr(), key.len()) };
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

// ---------------------------------------------------------------------
// RcOpaque: Val::Opaque's backing - a fully host-erased `T`, monomorphized
// per concrete `T` (see `rc::any_vtable`/`rc::DynRc::erase`).
// ---------------------------------------------------------------------

#[repr(transparent)]
#[derive(Clone)]
pub struct RcOpaque {
    ptr: DynRc<raft_ffi::AnyVTable, Void>,
}

impl RcOpaque {
    pub fn new<T: 'static>(value: T) -> Self {
        RcOpaque {
            ptr: erase(value),
        }
    }
}

// ---------------------------------------------------------------------
// Function/RcFn: Val::Fn's backing. `raft-core`'s own `Function` trait
// (partial application, min/max args) bridges into `rc::Callable`
// (the minimal bound `rc::fn_vtable`/`DynRc::erase_fn` need), same
// relationship the old `Function`/`DynFn` split had - just retargeted
// at `*mut ffi::RawHost` instead of `&mut dyn Host`.
//
// Note: unlike the old design, there is no `call_once`/`Rc::try_unwrap`
// fast path here - safely replicating "move captured state instead of
// cloning it" through a fully type-erased `DynRc<FnVTable, Void>` would
// need a dedicated "take" vtable slot (deallocation must stay owned by
// `destroy`, so a `call` shim can't `try_unwrap`-and-deallocate without
// risking a double-free at the outer `Drop`). `PartialFn` always clones
// its captured args on the shared path instead.
// ---------------------------------------------------------------------

/// A callable value: `fn`-defined functions (AST-walked or compiled to
/// bytecode), partially-applied functions, and host-provided closures all
/// implement this.
///
/// Callers must supply at least [`Function::min_args`] arguments - the
/// generic dispatch in [`call_dispatch`]-via-[`rc::Callable`] handles
/// partial application *before* calling [`Function::call`], so
/// implementations never see an underfull call. `host` is a safe view -
/// nothing in this trait, or anything implementing it, ever touches a raw
/// pointer; that's confined to `rc::call_shim` and [`RcFn::call`].
pub trait Function: Sized + 'static {
    /// Minimum number of arguments this function consumes in a call. If
    /// fewer are supplied, the dispatch returns a partially-applied
    /// function value instead of calling.
    fn min_args(&self) -> usize;

    fn max_args(&self) -> Option<usize> {
        None
    }

    /// Push exactly one result. Caller (the generic dispatch below)
    /// guarantees `args` is between `min_args` and (if any) `max_args`.
    fn call(&self, host: &mut rc::Host, args: usize);

    fn debug_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<fn>")
    }
}

fn call_dispatch<F: Function>(rc: &Rc<F>, host: &mut rc::Host, args: usize) -> usize {
    let min_args = rc.min_args();
    if args < min_args {
        if args == 0 {
            host.stack().push(Val::from(ValEnum::Fn(RcFn::from_rc(rc.clone()))));
            return 0;
        }
        let mut preapplied: SmallVec<[Val; 4]> =
            smallvec::smallvec![Val::from(ValEnum::Uninit); args];
        host.stack().drain_top_into(&mut preapplied);
        let partial = Rc::new(PartialFn {
            fun: RcFn::from_rc(rc.clone()),
            min_args: min_args - args,
            max_args: rc.max_args().map(|max| max - args),
            preapplied,
        });
        host.stack().push(Val::from(ValEnum::Fn(RcFn::from_rc(partial))));
        return args;
    }

    match rc.max_args() {
        Some(max_args) if max_args < args => {
            rc.call(host, max_args);
            max_args
        }
        _ => {
            rc.call(host, args);
            args
        }
    }
}

impl<F: Function> Callable for F {
    #[inline]
    unsafe fn call_raw(this: raft_ffi::RcPtr<F>, args: usize, host: &mut rc::Host) -> usize {
        // SAFETY: `this` is the live, whole-provenance `RcInner<F>` box
        // (caller's contract) - viewing it as a borrowed `Rc<F>` (in
        // `ManuallyDrop`, so no double-decrement) lets `call_dispatch`
        // clone it into a partial-application value when needed.
        let rc = ManuallyDrop::new(unsafe { Rc::<F>::from_raw_box(this.cast()) });
        call_dispatch(&rc, host, args)
    }
}

/// Adapter implementing [`Function`] for plain host closures. (A blanket
/// impl on `F: Fn(..)` would conflict with the concrete `Function` impls
/// under coherence rules, hence the newtype.)
struct HostFn<F> {
    min_args: usize,
    max_args: Option<usize>,
    fun: F,
}

impl<F> Function for HostFn<F>
where
    F: Fn(&mut rc::Host, usize) -> Val + 'static,
{
    #[inline]
    fn min_args(&self) -> usize {
        self.min_args
    }

    #[inline]
    fn max_args(&self) -> Option<usize> {
        self.max_args
    }

    #[inline]
    fn call(&self, host: &mut rc::Host, args: usize) {
        debug_assert!(args >= self.min_args);
        let ret = (self.fun)(host, args);
        host.stack().push(ret);
    }

    #[inline]
    fn debug_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.max_args {
            Some(max_args) => write!(f, "<fn {}..={}>", self.min_args, max_args),
            None => write!(f, "<fn {}..>", self.min_args),
        }
    }
}

/// A function with some arguments already applied, waiting for the rest.
/// Wraps the callee as an already-type-erased [`RcFn`] rather than a
/// generic `Rc<F>` - re-partial-applying a `PartialFn` would otherwise
/// nest `PartialFn<PartialFn<PartialFn<..>>>` without bound, and since
/// `call_dispatch`'s own body (reachable, regardless of whether it's ever
/// taken at runtime) constructs one more nesting level, that blows the
/// compiler's monomorphization recursion limit. Going through `RcFn::call`
/// costs one extra `extern "C"` dispatch per incremental partial
/// application instead.
struct PartialFn {
    fun: RcFn,
    min_args: usize,
    max_args: Option<usize>,
    preapplied: SmallVec<[Val; 4]>,
}

impl Function for PartialFn {
    #[inline]
    fn min_args(&self) -> usize {
        self.min_args
    }

    #[inline]
    fn max_args(&self) -> Option<usize> {
        self.max_args
    }

    fn call(&self, host: &mut rc::Host, args: usize) {
        let len = self.preapplied.len();
        host.stack().extend_from_slice(&self.preapplied);
        self.fun.call(host, len + args);
    }

    fn debug_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<partial ")?;
        for arg in self.preapplied.iter() {
            write!(f, "{:?} ", arg)?;
        }
        write!(f, "<fn>>")
    }
}

#[repr(transparent)]
#[derive(Clone)]
pub struct RcFn {
    ptr: DynRc<raft_ffi::FnVTable, Void>,
}

impl RcFn {
    #[inline]
    pub fn new<F: Function>(f: F) -> Self {
        Self::from_rc(Rc::new(f))
    }

    #[inline]
    pub fn from_rc<F: Function>(rc: Rc<F>) -> Self {
        RcFn { ptr: erase_fn(rc) }
    }

    /// Raw identity of the underlying allocation, for callers that need
    /// to compare "is this the same function value" (a self-recursion
    /// fast path, say) without any dispatch overhead. Not meaningful
    /// across two different `Val::Fn`-producing calls unless both sides
    /// agree on how they derived their pointer (see call sites).
    #[inline]
    pub fn as_ptr(&self) -> *const () {
        DynRc::data_ptr(&self.ptr).as_ptr() as *const ()
    }

    /// Dispatch a call, pushing exactly one result and returning how many
    /// of `args` were actually consumed (fewer than `args` if this value
    /// only partially applies, in which case a new partial-application
    /// `Val::Fn` was pushed instead of a result).
    #[inline]
    pub fn call(&self, host: &mut rc::Host, args: usize) -> usize {
        let vtable = DynRc::vtable(&self.ptr);
        // SAFETY: raw place projection from the live box pointer (never a
        // reference) keeps whole-box provenance on `data` - `rc::call_shim`
        // walks it back to the box header to reach the strong count.
        let data = unsafe {
            raft_ffi::VoidPtr::new_unchecked(&raw mut (*DynRc::data_ptr(&self.ptr).as_ptr()).value)
        };
        // SAFETY: crossing into the `extern "C"` `CallFn` ABI - the one
        // place a raw pointer is unavoidable; `host.as_raw()` is still
        // the exact valid, exclusively-borrowed `RawHost` `host` wraps.
        unsafe { (vtable.call)(data, args, host.as_raw()) }
    }
}

impl fmt::Debug for RcFn {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(_f, "<fn>")
    }
}
