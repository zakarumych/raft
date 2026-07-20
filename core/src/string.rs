
// ---------------------------------------------------------------------
// RcStr: Val::String's backing. Immutable once built (`RawStringVal`'s inline
// flexible-array shape suits this fine - no growth needed).
// ---------------------------------------------------------------------

use core::{alloc::Layout, cmp::Ordering, fmt, hash::{Hash, Hasher}, ops::Deref, ptr::NonNull};

use alloc::string::String;
use raft_ffi::{RawStringVal, RcInner, Void};

use crate::rc::DynRc;

unsafe extern "C" fn string_destroy(ptr: raft_ffi::RcPtr<Void>) {
    unsafe {
        let ptr = ptr.cast::<RcInner<RawStringVal>>();
        let len = ptr.as_ref().value.len;
        let (layout, _) = Layout::new::<RcInner<RawStringVal>>()
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
    ptr: DynRc<raft_ffi::StringVTable, RawStringVal>,
}

impl RcStr {
    pub fn new(s: &str) -> Self {
        let header_layout = Layout::new::<RcInner<RawStringVal>>();
        let (layout, offset) = header_layout
            .extend(Layout::for_value(s.as_bytes()))
            .expect("RcStr layout overflow");
        let layout = layout.pad_to_align();

        // SAFETY: `layout` is non-zero-sized (header alone is nonzero).
        let raw = unsafe { alloc::alloc::alloc(layout) };
        let Some(base) = NonNull::new(raw) else {
            alloc::alloc::handle_alloc_error(layout);
        };

        let inner = base.cast::<RcInner<RawStringVal>>();
        unsafe {
            inner.as_ptr().write(RcInner {
                strong: core::cell::Cell::new(1),
                value: RawStringVal {
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

    pub fn into_raw(self) -> DynRc<raft_ffi::StringVTable, RawStringVal> {
        self.ptr
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

    pub unsafe fn from_raw(ptr: DynRc<raft_ffi::StringVTable, RawStringVal>) -> Self {
        RcStr { ptr }
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
