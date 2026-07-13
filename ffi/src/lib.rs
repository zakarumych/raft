// FFI layer is strictly no_std.
#![no_std]

/// The version of the FFI crate.
/// Bundle may only be used with the same version of the FFI crate.
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

    impl<T> InitResult for Result<(), T> where T: core::fmt::Display {
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
        pub unsafe extern "C" fn raft_ffi_init_bundle(bundle: &mut $crate::RaftFFIBundle, error_buf: *mut u8, error_buf_len: i32) -> i32 {
            const _: $crate::RaftFFIInitBundleFn = raft_ffi_init_bundle;

            use $crate::for_macro::InitResult;

            fn inner($bundle: &mut $crate::RaftFFIBundle) -> impl InitResult {
                $code
            }

            inner(bundle).into_code(error_buf, error_buf_len)
        }
    };
}
