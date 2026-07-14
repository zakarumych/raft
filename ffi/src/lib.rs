//! FFI layer is strictly no_std, and has zero dependencies.
//! It declares ABI for dynamically loaded Raft modules.
#![no_std]
use core::{cell::Cell, ffi::c_void, ptr::NonNull};

pub type Void = c_void;

#[repr(C)]
pub struct LenVal {
    pub len: usize,
    pub val: [u8; 0],
}

#[repr(C)]
pub struct Str {
    pub ptr: *const u8,
    pub len: usize,
}

#[repr(C)]
pub struct RawFieldVal {
    pub name: Str,
    pub val: RawVal,
}

#[repr(C)]
pub struct RcInner<T> {
    pub strong: Cell<usize>,
    pub value: T,
}

pub type RcPtr<T> = NonNull<RcInner<T>>;
pub type VoidPtr = NonNull<Void>;
pub type VoidRcPtr = NonNull<RcInner<Void>>;

/// A growable, reallocatable buffer: `len` live elements out of `capacity`
/// allocated, behind a movable inner `ptr`. Whoever holds a `RawVec<T>` by
/// value (embedded in a stable, non-moving header — [`RawHost`]'s stack,
/// or a list/record's own outer allocation) keeps a fixed address even as
/// `ptr` gets reallocated underneath it: growing (`push`, stack growth,
/// ...) may move the *elements*, never the header holding this struct.
/// Callers must not hold element pointers/references across a push.
#[repr(C)]
pub struct RawVec<T> {
    pub ptr: *mut T,
    pub size: usize,
    pub capacity: usize,
}

pub type RawStack = RawVec<RawVal>;

/// A list's backing store: elements, growable in place (see [`RawVec`]).
pub type ListVal = RawVec<RawVal>;

/// A record's backing store: name/value fields, growable in place.
pub type RecordVal = RawVec<RawFieldVal>;

#[repr(C)]
pub struct RawHost {
    pub stack: RawStack,
}

pub type DestroyFn = unsafe extern "C" fn(RcPtr<Void>);
pub type FieldsFn = unsafe extern "C" fn(VoidPtr) -> *const RawFieldVal;
pub type ElementsFn = unsafe extern "C" fn(VoidPtr) -> *const RawVal;
pub type GetFn = unsafe extern "C" fn(VoidPtr, usize) -> RawVal;
pub type GetByNameFn = unsafe extern "C" fn(VoidPtr, *const u8, usize) -> RawVal;
pub type SetFn = unsafe extern "C" fn(VoidPtr, usize, RawVal);
pub type SetByNameFn = unsafe extern "C" fn(VoidPtr, *const u8, usize, RawVal);
pub type RemFn = unsafe extern "C" fn(VoidPtr, usize);
pub type RemByNameFn = unsafe extern "C" fn(VoidPtr, *const u8, usize);
pub type SwapFn = unsafe extern "C" fn(VoidPtr, usize, RawVal) -> RawVal;
pub type SwapByNameFn = unsafe extern "C" fn(VoidPtr, *const u8, usize, RawVal) -> RawVal;
pub type PushFn = unsafe extern "C" fn(VoidPtr, RawVal);
pub type PopFn = unsafe extern "C" fn(VoidPtr) -> RawVal;
pub type StackGrowFn = unsafe extern "C" fn(*mut RawStack, usize);
pub type CallFn = unsafe extern "C" fn(VoidPtr, usize, *mut RawHost) -> usize;

#[repr(C, align(16))]
pub struct AnyVTable {
    pub destroy: DestroyFn,
}

#[repr(C, align(16))]
pub struct StringVTable {
    pub any: AnyVTable,
}

#[repr(C, align(16))]
pub struct RecordVTable {
    pub any: AnyVTable,
    pub fields: FieldsFn,
    pub get: GetFn,
    pub get_by_name: GetByNameFn,
    pub set: SetFn,
    pub set_by_name: SetByNameFn,
    pub rem: RemFn,
    pub rem_by_name: RemByNameFn,
    pub swap: SwapFn,
    pub swap_by_name: SwapByNameFn,
}

#[repr(C, align(16))]
pub struct ListVTable {
    pub any: AnyVTable,
    pub elements: ElementsFn,
    pub get: GetFn,
    pub set: SetFn,
    pub swap: SwapFn,
    pub push: PushFn,
    pub pop: PopFn,
}

#[repr(C, align(16))]
pub struct HostVTable {
    pub any: AnyVTable,
    pub getvar: GetFn,
    pub setvar: SetFn,
    pub grow: StackGrowFn,
}

#[repr(C, align(16))]
pub struct FnVTable {
    pub any: AnyVTable,
    pub call: CallFn,
}

/// Bits occupied by the MASK_TAG_* values in a ValTag.
pub const MASK_BITS: usize = 0b1111;

/// Value is uninitialized.
pub const MASK_TAG_UNINIT: usize = 0b0001;

pub const MASK_TAG_INT: usize = 0b0010;
pub const MASK_TAG_FLOAT: usize = 0b0011;
pub const MASK_TAG_CHAR: usize = 0b0100;
pub const MASK_TAG_ATOM_NIL: usize = 0b0101;
pub const MASK_TAG_ATOM_FALSE: usize = 0b0110;
pub const MASK_TAG_ATOM_TRUE: usize = 0b0111;

/// Data of the Val is ID of the atom in the atom table.
pub const MASK_TAG_ATOM_ID: usize = 0b1000;

/// ValTag contains vtable for string object.
pub const MASK_TAG_STRING: usize = 0b1001;

/// ValTag contains vtable for record object.
pub const MASK_TAG_RECORD: usize = 0b1010;

/// ValTag contains vtable for list object.
pub const MASK_TAG_LIST: usize = 0b1011;

/// ValTag contains vtable for function object.
pub const MASK_TAG_FN: usize = 0b1100;

/// ValTag contains vtable for opaque object.
pub const MASK_TAG_OPAQUE: usize = 0b1101;

#[repr(C)]
pub struct RawTag {
    pub tag_ptr: NonNull<Void>,
}

#[repr(C)]
pub union RawData {
    pub nothing: (),
    pub int: i64,
    pub flt: f64,
    pub ptr: NonNull<Void>,
}

#[repr(C)]
pub struct RawVal {
    pub tag: RawTag,
    pub data: RawData,
}

/// The version of the FFI crate.
/// Bundle may only be used with the same version of the FFI crate.
///
/// On initialization, the bundle must check that the version of the FFI crate it was compiled against
/// matches the version of the FFI crate in the host process.
pub const FFI_VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");

/// A bundle of Raft modules exposed by compiled cdylib.
#[repr(C)]
pub struct RaftFFIBundle {
    modules: *mut RaftFFIModule,
    modules_count: i32,
}

#[repr(C)]
pub struct RaftFFIModule {}

pub type RaftFFIMakeAtomFn = unsafe extern "C" fn(*const u8, i32);

pub type RaftFFIVersionFn = unsafe extern "C" fn() -> *const u8;

/// Type signature for the function that initializes the Raft bundle in the cdylib.
pub type RaftFFIInitBundleFn = unsafe extern "C" fn(&mut RaftFFIBundle, *mut u8, i32) -> i32;

/// Name of the function that initializes the Raft bundle in the cdylib.
pub const INIT_RAFT_BUNDLE_FN_NAME: &str = "raft_ffi_init_bundle";

pub const FFI_VERSION_STATIC_NAME: &str = "raft_ffi_version";

#[doc(hidden)]
pub mod for_macro {
    use core::fmt::Write;

    struct ErrorBuffer {
        buf: *mut u8,
        len: i32,
        used: i32,
    }

    impl Write for ErrorBuffer {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            let bytes = s.as_bytes();
            let len = bytes.len();

            let fitting_len = core::cmp::min(len, (self.len - self.used) as usize);
            unsafe {
                core::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    self.buf.add(self.used as usize) as *mut u8,
                    fitting_len,
                );
            }
            self.used += fitting_len as i32;
            Ok(())
        }
    }

    pub trait InitResult {
        fn into_code(self, error_buf: *mut u8, error_buf_len: i32) -> i32;
    }

    impl InitResult for () {
        fn into_code(self, _error_buf: *mut u8, _error_buf_len: i32) -> i32 {
            0
        }
    }

    impl<T> InitResult for Result<(), T>
    where
        T: core::fmt::Display,
    {
        fn into_code(self, error_buf: *mut u8, error_buf_len: i32) -> i32 {
            match self {
                Ok(_) => 0,
                Err(code) => {
                    let mut buf = ErrorBuffer {
                        buf: error_buf,
                        len: error_buf_len,
                        used: 0,
                    };
                    let _ = core::write!(&mut buf, "{}", code);
                    -buf.used
                }
            }
        }
    }
}

#[macro_export]
macro_rules! raft_bundle {
    ($bundle:pat => $code:block) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn raft_ffi_version() -> *const u8 {
            const _: $crate::RaftFFIVersionFn = raft_ffi_version;

            $crate::FFI_VERSION.as_ptr()
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn raft_ffi_init_bundle(
            bundle: &mut $crate::RaftFFIBundle,
            error_buf: *mut u8,
            error_buf_len: i32,
        ) -> i32 {
            const _: $crate::RaftFFIInitBundleFn = raft_ffi_init_bundle;

            use $crate::for_macro::InitResult;

            fn inner($bundle: &mut $crate::RaftFFIBundle) -> impl InitResult {
                $code
            }

            inner(bundle).into_code(error_buf, error_buf_len)
        }
    };
}
