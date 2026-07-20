// ---------------------------------------------------------------------
// Val: repr(transparent) over RawVal, plus the tag/vtable plumbing.
// ---------------------------------------------------------------------

use core::{cmp::Ordering, fmt, mem::ManuallyDrop, ptr::NonNull};

use alloc::string::String;
use raft_ffi::{
    CoroHeader, MASK_BITS, MASK_TAG_ATOM_FALSE, MASK_TAG_ATOM_ID, MASK_TAG_ATOM_NIL, MASK_TAG_ATOM_TRUE, MASK_TAG_CHAR, MASK_TAG_CORO, MASK_TAG_FLOAT, MASK_TAG_FN, MASK_TAG_INT, MASK_TAG_LIST, MASK_TAG_OPAQUE, MASK_TAG_RECORD, MASK_TAG_STRING, MASK_TAG_UNINIT, RawData, RawListVal, RawRecordVal, RawStringVal, RawTag, RawVal, RcInner, Void,
};

use crate::{
    atom::{Atom, AtomId},
    coro::RcCoro,
    error::RuntimeError,
    function::RcFn,
    host::Host,
    list::RcList,
    num::Number,
    opaque::RcOpaque,
    rc::DynRc,
    record::RcRecord,
    string::RcStr,
    vtable::AnyVTable,
};

#[repr(transparent)]
pub struct Val {
    raw: RawVal,
}

impl Val {
    #[inline]
    fn tag_ptr(&self) -> *mut Void {
        self.raw.tag.tag_ptr.as_ptr()
    }

    /// A pure integer read for the `match` in `unpack`/`Drop` - never
    /// round-tripped back into a pointer, so no provenance concern here.
    #[inline]
    fn tag_bits(&self) -> usize {
        self.tag_ptr().addr() & MASK_BITS
    }

    #[inline]
    fn vtable_ptr<V>(&self) -> *const V {
        // Masking through `map_addr` (not a `usize` round-trip) keeps
        // this pointer's original provenance - the `&'static V` it was
        // built from in `pack_heap`.
        self.tag_ptr().map_addr(|a| a & !MASK_BITS).cast::<V>()
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
            raw: RawVal {
                tag: RawTag {
                    // SAFETY: every `MASK_TAG_*` constant is nonzero.
                    tag_ptr: unsafe { NonNull::new_unchecked(tag as *mut Void) },
                },
                data: RawData { nothing: () },
            },
        }
    }

    #[inline]
    fn from_data_only(tag: usize, data: RawData) -> Val {
        Val {
            raw: RawVal {
                tag: RawTag {
                    // SAFETY: every `MASK_TAG_*` constant is nonzero.
                    tag_ptr: unsafe { NonNull::new_unchecked(tag as *mut Void) },
                },
                data,
            },
        }
    }

    fn pack_heap<V: AnyVTable, T>(tag: usize, dyn_rc: DynRc<V, T>) -> Val {
        let vtable_ptr = DynRc::vtable(&dyn_rc) as *const V;
        debug_assert_eq!(
            vtable_ptr.addr() & MASK_BITS,
            0,
            "vtable must be 16-byte aligned"
        );
        let data_ptr: NonNull<RcInner<T>> = NonNull::from_ref(DynRc::rc_box(&dyn_rc));
        core::mem::forget(dyn_rc);
        // OR-ing the tag into the (nonzero, 16-byte-aligned) vtable
        // pointer's low bits through `map_addr` - not a `usize` round-trip
        // - keeps its provenance; `vtable_ref` recovers it the same way.
        let tagged = vtable_ptr.cast::<Void>().map_addr(|a| a | tag);
        Val {
            raw: RawVal {
                tag: RawTag {
                    // SAFETY: `tagged` is derived from a live `&'static V`,
                    // nonzero; OR-ing in the (also nonzero) tag bits keeps
                    // it nonzero.
                    tag_ptr: unsafe { NonNull::new_unchecked(tagged as *mut Void) },
                },
                data: RawData {
                    ptr: data_ptr.cast(),
                },
            },
        }
    }

    /// Move `self`'s raw bits out without running `Drop` - for handing
    /// ownership across a `RawVal`-returning boundary (the receiver
    /// becomes the new owner of whatever heap reference this held).
    #[inline(always)]
    pub fn into_raw(self) -> RawVal {
        // SAFETY: copying the bits out is fine - `mem::forget` below means
        // `self`'s own `Drop` never runs, so this isn't a double-owned copy.

        let me = ManuallyDrop::new(self);
        unsafe { core::ptr::read(&me.raw) }
    }

    #[inline(always)]
    pub unsafe fn from_raw(raw: RawVal) -> Val {
        Val { raw }
    }

    #[inline(always)]
    pub unsafe fn from_raw_ref(raw: &RawVal) -> &Val {
        // SAFETY: `Val` is `#[repr(transparent)]` over `RawVal`.
        unsafe { &*(raw as *const RawVal as *const Val) }
    }

    /// Clone a `Val` out of a borrowed `RawVal` (e.g. one embedded in a
    /// list/record entry) without taking ownership of the original.
    ///
    /// # Safety
    /// `raw` must be a valid `RawVal`.
    pub unsafe fn clone_raw(raw: &RawVal) -> Val {
        unsafe { Val::from_raw_ref(raw) }.clone()
    }

    /// The ergonomic, `match`-able view of this value. Heap kinds bump a
    /// refcount (same cost as `.clone()`ing the underlying handle);
    /// scalars are free.
    #[inline(always)]
    pub fn unpack(&self) -> ValEnum {
        match self.tag_bits() {
            MASK_TAG_UNINIT => ValEnum::Uninit,
            MASK_TAG_INT => ValEnum::Number(Number::Integer(unsafe { self.raw.data.int })),
            MASK_TAG_FLOAT => ValEnum::Number(Number::Float(unsafe { self.raw.data.flt })),
            MASK_TAG_CHAR => {
                let bits = unsafe { self.raw.data.int } as u32;
                ValEnum::Char(char::from_u32(bits).unwrap_or('\u{FFFD}'))
            }
            MASK_TAG_ATOM_NIL => ValEnum::Atom(Atom::Nil),
            MASK_TAG_ATOM_TRUE => ValEnum::Atom(Atom::True),
            MASK_TAG_ATOM_FALSE => ValEnum::Atom(Atom::False),
            MASK_TAG_ATOM_ID => {
                let id = unsafe { self.raw.data.int } as usize;
                ValEnum::Atom(Atom::Custom(AtomId(id)))
            }
            MASK_TAG_STRING => ValEnum::String(unsafe {
                RcStr::from_raw(self.clone_heap::<raft_ffi::StringVTable, RawStringVal>())
            }),
            MASK_TAG_RECORD => ValEnum::Record(unsafe {
                RcRecord::from_raw(self.clone_heap::<raft_ffi::RecordVTable, RawRecordVal>())
            }),
            MASK_TAG_LIST => ValEnum::List(unsafe {
                RcList::from_raw(self.clone_heap::<raft_ffi::ListVTable, RawListVal>())
            }),
            MASK_TAG_FN => ValEnum::Fn(unsafe {
                RcFn::from_raw(self.clone_heap::<raft_ffi::FnVTable, Void>())
            }),
            MASK_TAG_CORO => ValEnum::Coro(unsafe {
                RcCoro::from_raw(self.clone_heap::<raft_ffi::CoroVTable, CoroHeader>())
            }),
            MASK_TAG_OPAQUE => ValEnum::Opaque(unsafe {
                RcOpaque::from_raw(self.clone_heap::<raft_ffi::AnyVTable, Void>())
            }),
            _ => unsafe { core::hint::unreachable_unchecked() },
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
            MASK_TAG_UNINIT => ValKind::Uninit,
            MASK_TAG_INT | MASK_TAG_FLOAT => ValKind::Number,
            MASK_TAG_CHAR => ValKind::Char,
            MASK_TAG_ATOM_NIL | MASK_TAG_ATOM_TRUE | MASK_TAG_ATOM_FALSE | MASK_TAG_ATOM_ID => {
                ValKind::Atom
            }
            MASK_TAG_STRING => ValKind::String,
            MASK_TAG_RECORD => ValKind::Record,
            MASK_TAG_LIST => ValKind::List,
            MASK_TAG_FN => ValKind::Fn,
            MASK_TAG_CORO => ValKind::Coro,
            MASK_TAG_OPAQUE => ValKind::Opaque,
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
            MASK_TAG_INT => unsafe { self.raw.data.int == 0 },
            MASK_TAG_FLOAT => unsafe { self.raw.data.flt == 0.0 },
            MASK_TAG_ATOM_FALSE => true,
            MASK_TAG_RECORD => {
                // SAFETY: tag confirmed `Record`; `heap_ptr` is a live
                // `RcInner<UnsafeCell<RecordVal>>` for this exact tag.
                let inner = unsafe { self.heap_ptr::<RawRecordVal>().as_ref() };
                inner.value.size == 0
            }
            MASK_TAG_LIST => {
                // SAFETY: as the `Record` arm's, for `ListVal`.
                let inner = unsafe { self.heap_ptr::<RawListVal>().as_ref() };
                inner.value.size == 0
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
    pub fn call_as_fn(&self, host: &mut Host, args: usize) -> Option<usize> {
        if self.tag_bits() != MASK_TAG_FN {
            return None;
        }
        let vtable = self.vtable_ref::<raft_ffi::FnVTable>();

        // SAFETY: crossing into the `extern "C"` `CallFn` ABI, same as
        // `RcFn::call` - `host.as_raw()` is the exact valid, exclusively
        // borrowed `RawHost` `host` wraps.
        Some(unsafe { (vtable.call)(self.heap_ptr::<Void>(), args, host.as_raw()) })
    }

    pub fn new_uninit() -> Val {
        Val::from_tag_only(MASK_TAG_UNINIT)
    }

    pub fn new_int(i: i64) -> Val {
        Val::from_data_only(MASK_TAG_INT, RawData { int: i })
    }

    pub fn new_float(f: f64) -> Val {
        Val::from_data_only(MASK_TAG_FLOAT, RawData { flt: f })
    }

    pub fn new_char(c: char) -> Val {
        Val::from_data_only(MASK_TAG_CHAR, RawData { int: c as i64 })
    }

    pub fn new_nil() -> Val {
        Val::from_tag_only(MASK_TAG_ATOM_NIL)
    }

    pub fn new_false() -> Val {
        Val::from_tag_only(MASK_TAG_ATOM_FALSE)
    }

    pub fn new_true() -> Val {
        Val::from_tag_only(MASK_TAG_ATOM_TRUE)
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
    Coro,
    Opaque,
}

impl From<ValEnum> for Val {
    #[inline(always)]
    fn from(v: ValEnum) -> Val {
        match v {
            ValEnum::Uninit => Val::from_tag_only(MASK_TAG_UNINIT),
            ValEnum::Number(Number::Integer(i)) => {
                Val::from_data_only(MASK_TAG_INT, RawData { int: i })
            }
            ValEnum::Number(Number::Float(f)) => {
                Val::from_data_only(MASK_TAG_FLOAT, RawData { flt: f })
            }
            ValEnum::Char(c) => Val::from_data_only(MASK_TAG_CHAR, RawData { int: c as i64 }),
            ValEnum::Atom(Atom::Nil) => Val::from_tag_only(MASK_TAG_ATOM_NIL),
            ValEnum::Atom(Atom::True) => Val::from_tag_only(MASK_TAG_ATOM_TRUE),
            ValEnum::Atom(Atom::False) => Val::from_tag_only(MASK_TAG_ATOM_FALSE),
            ValEnum::Atom(Atom::Custom(id)) => {
                Val::from_data_only(MASK_TAG_ATOM_ID, RawData { int: id.0 as i64 })
            }
            ValEnum::String(s) => Val::pack_heap(MASK_TAG_STRING, s.into_raw()),
            ValEnum::Record(r) => Val::pack_heap(MASK_TAG_RECORD, r.into_raw()),
            ValEnum::List(l) => Val::pack_heap(MASK_TAG_LIST, l.into_raw()),
            ValEnum::Fn(f) => Val::pack_heap(MASK_TAG_FN, f.into_raw()),
            ValEnum::Coro(c) => Val::pack_heap(MASK_TAG_CORO, c.into_raw()),
            ValEnum::Opaque(o) => Val::pack_heap(MASK_TAG_OPAQUE, o.into_raw()),
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
            MASK_TAG_STRING | MASK_TAG_RECORD | MASK_TAG_LIST | MASK_TAG_FN | MASK_TAG_CORO
            | MASK_TAG_OPAQUE => {
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
        Val {
            raw: unsafe { core::ptr::read(&self.raw) },
        }
    }
}

impl Drop for Val {
    fn drop(&mut self) {
        match self.tag_bits() {
            MASK_TAG_STRING => self.drop_heap::<raft_ffi::StringVTable, RawStringVal>(),
            MASK_TAG_RECORD => self.drop_heap::<raft_ffi::RecordVTable, RawRecordVal>(),
            MASK_TAG_LIST => self.drop_heap::<raft_ffi::ListVTable, RawListVal>(),
            MASK_TAG_FN => self.drop_heap::<raft_ffi::FnVTable, Void>(),
            MASK_TAG_CORO => self.drop_heap::<raft_ffi::CoroVTable, Void>(),
            MASK_TAG_OPAQUE => self.drop_heap::<raft_ffi::AnyVTable, Void>(),
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
    Coro(RcCoro),
    Opaque(RcOpaque),
    /// Internal sentinel: a local slot that has not been assigned yet
    /// (reads of it fall back to the global scope), or "not found" from
    /// a list/record lookup. Never observable from Raft code or host
    /// functions.
    #[doc(hidden)]
    Uninit,
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
            ValEnum::Coro(c) => write!(f, "{c:?}"),
            ValEnum::Opaque(o) => write!(f, "Opaque({:p})", o.as_ptr()),
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
            ValEnum::Coro(c) => write!(f, "{c:?}"),
            ValEnum::Opaque(o) => write!(f, "{:p}", o.as_ptr()),
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
        self.tag_bits() != MASK_TAG_UNINIT
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
    pub fn list(elements: impl IntoIterator<Item = Val>) -> Val {
        Val::from(ValEnum::List(RcList::new(elements)))
    }

    #[inline]
    pub fn record(fields: impl IntoIterator<Item = (RcStr, Val)>) -> Val {
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
        Val::bool_(self.is_falsey())
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
                let mut s = String::with_capacity(s1.len() + s2.len());
                s.push_str(s1.as_str());
                s.push_str(s2.as_str());
                Ok(Val::string(&s))
            }
            (ValEnum::String(s1), ValEnum::Char(c2)) => {
                let mut s = String::with_capacity(s1.len() + c2.len_utf8());
                s.push_str(s1.as_str());
                s.push(c2);
                Ok(Val::string(&s))
            }
            (ValEnum::Char(c1), ValEnum::String(s2)) => {
                let mut s = String::with_capacity(c1.len_utf8() + s2.len());
                s.push(c1);
                s.push_str(s2.as_str());
                Ok(Val::string(&s))
            }
            (ValEnum::Char(c1), ValEnum::Char(c2)) => {
                let mut s = String::with_capacity(c1.len_utf8() + c2.len_utf8());
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
    pub fn eq(&self, rhs: &Val) -> bool {
        match (self.unpack(), rhs.unpack()) {
            (ValEnum::Number(n1), ValEnum::Number(n2)) => n1 == n2,
            (ValEnum::Atom(a1), ValEnum::Atom(a2)) => {
                a1 == a2
            }
            (ValEnum::String(s1), ValEnum::String(s2)) => s1.as_str() == s2.as_str(),
            (ValEnum::Char(c1), ValEnum::Char(c2)) => c1 == c2,
            (ValEnum::List(l1), ValEnum::List(l2)) => l1 == l2,
            (ValEnum::Record(r1), ValEnum::Record(r2)) => {
                Iterator::eq(r1.iter(), r2.iter())
            }
            (ValEnum::Fn(f1), ValEnum::Fn(f2)) => {
                f1.as_ptr() == f2.as_ptr()
            }
            (ValEnum::Coro(c1), ValEnum::Coro(c2)) => {
                c1.as_ptr() == c2.as_ptr()
            }
            (ValEnum::Opaque(o1), ValEnum::Opaque(o2)) => {
                o1.as_ptr() == o2.as_ptr()
            }
            _ => false, // different kinds are considered inequal
        }
    }

    #[inline]
    pub fn cmp(&self, rhs: &Val) -> Option<Ordering> {
        match (self.unpack(), rhs.unpack()) {
            (ValEnum::Number(n1), ValEnum::Number(n2)) => n1.cmp(n2),
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
            (ValEnum::Record(r1), ValEnum::Record(r2)) => r1.cmp(&r2),
            (ValEnum::Fn(f1), ValEnum::Fn(f2)) => {
                (f1.as_ptr() == f2.as_ptr()).then_some(Ordering::Equal)
            }
            (ValEnum::Coro(c1), ValEnum::Coro(c2)) => {
                (c1.as_ptr() == c2.as_ptr()).then_some(Ordering::Equal)
            }
            (ValEnum::Opaque(o1), ValEnum::Opaque(o2)) => {
                (o1.as_ptr() == o2.as_ptr()).then_some(Ordering::Equal)
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
                        Some(Val::record([(RcStr::new(key), value)].into_iter()))
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
}

impl PartialEq for Val {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.eq(other)
    }
}

impl PartialOrd for Val {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.cmp(other)
    }
}