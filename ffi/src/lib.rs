//! FFI layer is strictly no_std, and has zero dependencies.
//! It declares ABI for dynamically loaded Raft modules.
#![no_std]
use core::{cell::Cell, ffi::c_void, ptr::NonNull, sync::atomic::AtomicUsize};

pub type Void = c_void;

#[repr(C)]
pub struct RawStringVal {
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
/// value (embedded in a stable, non-moving header - [`RawHost`]'s stack,
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
pub type RawListVal = RawVec<RawVal>;

/// A record's backing store: name/value fields, growable in place.
pub type RawRecordVal = RawVec<RawFieldVal>;

#[repr(C)]
pub struct RawHost {
    pub stack: RawStack,
    /// The waker of the poll currently driving this host, or null outside
    /// any poll. Borrowed - set by the executor for a poll's duration; a
    /// consumer that wants to keep it must `clone` through its vtable.
    pub waker: *mut WakerHeader,
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
pub type CallFn = unsafe extern "C" fn(RcPtr<Void>, usize, *mut RawHost) -> usize;

/// Resume a coroutine with `args` arguments on the stack (0 for today's
/// generator and async kinds), returning a [`CORO_DONE`]/[`CORO_YIELD`]/
/// [`CORO_PENDING`] status byte.
pub type CoroResumeFn = unsafe extern "C" fn(RcPtr<Void>, usize, *mut RawHost) -> u8;

/// The coroutine finished. Nothing was pushed; resuming again keeps
/// returning this and does nothing. A failed coroutine also finishes this
/// way, with the host's pending-error state set.
pub const CORO_DONE: u8 = 0;

/// The coroutine pushed exactly one value onto the host stack and may be
/// resumed again.
pub const CORO_YIELD: u8 = 1;

/// The coroutine must wait before it can continue - resume it after its
/// waker (cloned from the host's ambient one) signals. Resuming
/// immediately is allowed but may keep returning this, wasting work.
/// Nothing was pushed.
pub const CORO_PENDING: u8 = 2;

/// What flavor of coroutine a `MASK_TAG_CORO` value is - the first field
/// of every coroutine payload (see [`CoroHeader`]). Decides the protocol
/// its consumer holds it to:
///
/// - [`CORO_KIND_GEN`] (`gen fn`): iteration expects `CORO_YIELD`s ended
///   by one `CORO_DONE`; `CORO_PENDING` is a protocol error.
/// - [`CORO_KIND_ASYNC`] (`async fn`): awaiting expects `CORO_PENDING`s
///   (propagated to the task) until one `CORO_YIELD` delivers the
///   resolution, then one final `CORO_DONE`; `CORO_DONE` before any yield
///   is a protocol error (unless the host error state says why).
/// - [`CORO_KIND_ASYNC_GEN`] (`async gen fn`): the combination -
///   `CORO_YIELD` per produced value, `CORO_PENDING` when it must wait
///   (propagated to the driving task), `CORO_DONE` at the end. Only
///   iterable from an async context, where pending can suspend the
///   consumer.
pub const CORO_KIND_GEN: u8 = 0;
pub const CORO_KIND_ASYNC: u8 = 1;
pub const CORO_KIND_ASYNC_GEN: u8 = 2;

/// The FFI-safe prefix every coroutine payload starts with - consumers on
/// either side of the boundary read the kind straight off the value,
/// without a vtable call.
#[repr(C)]
pub struct CoroHeader {
    pub kind: u8,
}

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

#[repr(C, align(16))]
pub struct CoroVTable {
    pub any: AnyVTable,
    pub resume: CoroResumeFn,
}

// ---------------------------------------------------------------------
// Async: wakers and pollable values across the FFI boundary.
//
// `core::task::Waker`'s `RawWakerVTable` is plain-Rust-ABI fn pointers -
// not stable across a cdylib boundary - so a waker crosses as a *thin*
// pointer to a [`WakerHeader`], whose first field is its own repr(C)
// vtable (this crate's usual pattern, prefix-style instead of a fat
// handle). The host's executor owns the concrete allocation behind it
// (refcounted task waker: task id + weak queue handle); everyone else
// only ever touches it through the vtable. The waker of the poll
// currently driving a host is ambient - [`RawHost::waker`] - rather than
// threaded through every call: a leaf async value that needs to wake its
// task later `clone`s it out of the host and stores the owned pointer.
//
// Adapting to a real `core::task::Waker` (for polling ordinary Rust
// futures) is a thin round trip: the `RawWaker` data pointer IS the
// `WakerHeader` pointer, and one static `RawWakerVTable` forwards each
// operation through the header's own vtable. No boxing per hop.
//
// Single-threaded contract: the host executor is not thread-safe; a
// waker must only be cloned, woken and dropped on the host's own thread,
// despite `core::task::Waker`'s nominal `Send + Sync`.
// ---------------------------------------------------------------------

/// The prefix every FFI waker allocation starts with: its vtable. The
/// rest of the allocation is implementation-private to whoever built it.
#[repr(C)]
pub struct WakerHeader {
    pub vtable: *const WakerVTable,
    pub strong: AtomicUsize,
}

pub type WakerPtr = NonNull<WakerHeader>;

pub type WakerWakeFn = unsafe extern "C" fn(WakerPtr);
pub type WakerDestroyFn = unsafe extern "C" fn(WakerPtr);

#[repr(C, align(16))]
pub struct WakerVTable {
    pub wake: WakerWakeFn,
    pub destroy: WakerDestroyFn,
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

/// ValTag contains vtable for a coroutine object (generator or async -
/// see [`CoroHeader`], whose kind byte distinguishes them).
pub const MASK_TAG_CORO: usize = 0b1110;

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

impl RawVal {
    /// The uninitialized value - what a host puts in a `RawVal`-typed slot
    /// before the other side fills it in.
    pub const fn uninit() -> RawVal {
        RawVal {
            tag: RawTag {
                // SAFETY: `MASK_TAG_UNINIT` is a nonzero constant, never
                // dereferenced as a real pointer.
                tag_ptr: unsafe { NonNull::new_unchecked(MASK_TAG_UNINIT as *mut Void) },
            },
            data: RawData { nothing: () },
        }
    }
}

/// The version of the FFI crate.
/// Bundle may only be used with the same version of the FFI crate.
///
/// On initialization, the bundle must check that the version of the FFI crate it was compiled against
/// matches the version of the FFI crate in the host process.
pub const FFI_VERSION: &core::ffi::CStr = unsafe { core::ffi::CStr::from_bytes_with_nul_unchecked(concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes()) };

/// Intern a name (`ptr`/`len` UTF-8 bytes) in the host, returning its id -
/// a `StringId` for identifier names, an `AtomId` for atom names.
pub type InternFn = unsafe extern "C" fn(*mut RawHost, *const u8, usize) -> usize;

/// Read/write a host global variable by interned `StringId`. Reading an
/// unbound name returns the uninit value.
pub type GetVarFn = unsafe extern "C" fn(*mut RawHost, usize) -> RawVal;
pub type SetVarFn = unsafe extern "C" fn(*mut RawHost, usize, RawVal);

/// Report a runtime error to the host (message as a string `RawVal`,
/// ownership transferred). Mirrors what the host's own functions do when
/// they fail mid-call: the failing callee pushes no result and the caller
/// checks the error status after the call returns.
pub type SetErrorFn = unsafe extern "C" fn(*mut RawHost, RawVal);

/// Take (and clear) the host's pending error, if any: returns the message
/// as a string `RawVal` (ownership transferred), or the uninit value when
/// no error is pending.
pub type TakeErrorFn = unsafe extern "C" fn(*mut RawHost) -> RawVal;

/// Everything a loaded bundle needs from its host: the raw host itself
/// (stack + whatever concrete state sits behind it) plus the callbacks the
/// bundle can't perform on its own - name interning, global-variable
/// access, and error signaling.
///
/// # Pointer validity
/// `raw` is only guaranteed valid for the duration of the init call it is
/// passed to - the host is free to move afterwards. A bundle must not
/// retain it; every post-init callback invocation must use the live host
/// pointer of the call it is servicing (the `*mut RawHost` its `CallFn`
/// received). The *function pointers* are process-stable and may be kept.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RaftFFIHost {
    pub raw: *mut RawHost,
    pub intern_string: InternFn,
    pub intern_atom: InternFn,
    pub getvar: GetVarFn,
    pub setvar: SetVarFn,
    pub set_error: SetErrorFn,
    pub take_error: TakeErrorFn,
}

/// A bundle of Raft modules exposed by a compiled cdylib. The host
/// constructs it (with `modules` uninit) and passes it to the bundle's
/// init function, which fills `modules` in.
#[repr(C)]
pub struct RaftFFIBundle {
    /// A record value `{ module_name: exports_record, .. }` - one field
    /// per compiled module, each holding that module's export record.
    /// Ownership transfers to the host once init returns success.
    pub modules: RawVal,
}

pub type RaftFFIVersionFn = unsafe extern "C" fn() -> *const u8;

/// Type signature for the function that initializes the Raft bundle in the cdylib.
pub type RaftFFIInitBundleFn =
    unsafe extern "C" fn(&mut RaftFFIBundle, &RaftFFIHost, *mut u8, usize) -> i32;

/// Type signature for the function that returns the name of a module in the Raft bundle, given its index.
/// If index is out of bounds, the function writes a null pointer.
pub type RaftFFIModuleNameFn = unsafe extern "C" fn(u32, *mut *const u8) -> usize;

/// Name of the function that initializes the Raft bundle in the cdylib.
pub const INIT_RAFT_BUNDLE_FN_NAME: &str = "raft_ffi_init_bundle";

pub const FFI_VERSION_STATIC_NAME: &str = "raft_ffi_version";

pub const MODULE_NAME_FN_NAME: &str = "raft_ffi_module_name";

#[doc(hidden)]
pub mod for_macro {
    use core::fmt::Write;

    struct ErrorBuffer {
        buf: *mut u8,
        len: usize,
        used: usize,
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
            self.used += fitting_len;
            Ok(())
        }
    }

    pub trait InitResult {
        fn into_code(self, error_buf: *mut u8, error_buf_len: usize) -> i32;
    }

    impl InitResult for () {
        fn into_code(self, _error_buf: *mut u8, _error_buf_len: usize) -> i32 {
            0
        }
    }

    impl<T> InitResult for Result<(), T>
    where
        T: core::fmt::Display,
    {
        fn into_code(self, error_buf: *mut u8, error_buf_len: usize) -> i32 {
            match self {
                Ok(_) => 0,
                Err(code) => {
                    let mut buf = ErrorBuffer {
                        buf: error_buf,
                        len: error_buf_len,
                        used: 0,
                    };
                    let _ = core::write!(&mut buf, "{}", code);
                    -(buf.used as i32)
                }
            }
        }
    }
}

#[macro_export]
macro_rules! raft_bundle {
    ([ $($module:literal),+ $(,)? ] ($bundle:pat, $host:pat) => $code:block) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn raft_ffi_version() -> *const u8 {
            const _: $crate::RaftFFIVersionFn = raft_ffi_version;

            $crate::FFI_VERSION.as_ptr()
        }

        pub unsafe extern "C" fn raft_ffi_module_name(idx: u32, ptr: *mut *const u8) -> usize {
            const _: $crate::RaftFFIModuleNameFn = raft_ffi_module_name;
            const MODULE_NAMES: &'static [&'static str] = &[$($module),+];
            match usize::try_from(idx) {
                Some(idx) if idx < MODULE_NAMES.len() => {
                    let name = MODULE_NAMES[idx];
                    *ptr = name.as_ptr();
                    name.len()
                }
                _ => {
                    *ptr = core::ptr::null();
                    0
                }
            }
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn raft_ffi_init_bundle(
            bundle: &mut $crate::RaftFFIBundle,
            host: &$crate::RaftFFIHost,
            error_buf: *mut u8,
            error_buf_len: usize,
        ) -> i32 {
            const _: $crate::RaftFFIInitBundleFn = raft_ffi_init_bundle;

            use $crate::for_macro::InitResult;

            fn inner(
                $bundle: &mut $crate::RaftFFIBundle,
                $host: &$crate::RaftFFIHost,
            ) -> impl InitResult {
                $code
            }

            inner(bundle, host).into_code(error_buf, error_buf_len)
        }
    };
}
