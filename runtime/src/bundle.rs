//! Transpiled-bundle support (feature `bundle`): [`Runtime`] can generate
//! a bundle crate from Raft sources (via `raft-rust`), build it by
//! invoking `cargo`, and link the produced cdylib - registering its
//! modules and holding the loaded library for the runtime's whole
//! lifetime (values produced by a bundle carry vtable and code pointers
//! into it, so it must never be unloaded while the runtime lives).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::string::{String, ToString};
use std::vec::Vec;

use raft_core::ffi;
use raft_rust::BundleGenerator;

use crate::{Runtime, RuntimeError, Val, ValEnum};

/// A recipe for [`Runtime::build_bundle`]: which Raft modules to
/// transpile, and where/how to build the resulting crate. Every knob has
/// a default - `BundleBuilder::new("my_bundle").module("math", SRC)` is a
/// complete spec.
pub struct BundleBuilder {
    name: String,
    /// `(module name, Raft source)` pairs, transpiled in order.
    modules: Vec<(String, String)>,
    /// Where to write the generated crate. Default: a per-bundle directory
    /// under the system temp dir.
    dir: Option<PathBuf>,
    /// The Raft repository checkout providing `raft-core`, written into
    /// the generated `Cargo.toml`. Default: the checkout this runtime was
    /// compiled from.
    raft_repo: Option<PathBuf>,
    /// Build profile. Default: the profile this runtime itself was built
    /// in (release when compiled without debug assertions).
    release: Option<bool>,
}

impl BundleBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        BundleBuilder {
            name: name.into(),
            modules: Vec::new(),
            dir: None,
            raft_repo: None,
            release: None,
        }
    }

    /// Add a Raft module (by name and source) to the bundle.
    pub fn module(mut self, name: impl Into<String>, source: impl Into<String>) -> Self {
        self.modules.push((name.into(), source.into()));
        self
    }

    /// Override the directory the generated crate is written to.
    pub fn dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.dir = Some(dir.into());
        self
    }

    /// Override the Raft repository path written into the generated
    /// `Cargo.toml`.
    pub fn raft_repo(mut self, path: impl Into<PathBuf>) -> Self {
        self.raft_repo = Some(path.into());
        self
    }

    /// Override the build profile (default: same profile as this runtime).
    pub fn release(mut self, release: bool) -> Self {
        self.release = Some(release);
        self
    }
}

fn other(msg: impl core::fmt::Display) -> RuntimeError {
    RuntimeError::Other(msg.to_string().into())
}

struct Bundle {
    lib: libloading::Library,
    module_names: Vec<String>,
}

impl Runtime {
    /// Generate, build and link a transpiled bundle: transpiles the
    /// builder's Raft modules into a cdylib crate, builds it by invoking
    /// `cargo` (in this runtime's own build profile unless overridden),
    /// and [links](Runtime::link_bundle) the produced library. Returns the
    /// registered module names.
    pub fn build_bundle(&mut self, bundle: &BundleBuilder) -> Result<Vec<String>, RuntimeError> {
        validate_crate_name(&bundle.name)?;
        if bundle.modules.is_empty() {
            return Err(other("bundle has no modules"));
        }

        let mut generator = BundleGenerator::new();
        for (name, source) in &bundle.modules {
            let module = parse_bundle_module(name, source)?;
            generator
                .add_module(name, &module)
                .map_err(|e| other(std::format_args!("module '{name}': {e}")))?;
        }

        let dir = bundle
            .dir
            .clone()
            .unwrap_or_else(|| std::env::temp_dir().join("raft-bundles").join(&bundle.name));

        // the checkout this runtime was compiled from - a sensible default
        // for local development; embedders can override via `raft_repo`
        let raft_repo = bundle.raft_repo.clone().unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("raft repo root")
                .to_path_buf()
        });
        let raft_repo = raft_repo
            .to_str()
            .ok_or_else(|| other("non-UTF-8 raft repo path"))?
            .replace('\\', "/");

        generator
            .write_crate(&dir, &bundle.name, &raft_repo)
            .map_err(|e| other(std::format_args!("writing bundle crate: {e}")))?;

        // default to the profile this runtime itself was built in
        let release = bundle.release.unwrap_or(!cfg!(debug_assertions));

        let mut cmd = Command::new("cargo");
        cmd.arg("build")
            .current_dir(&dir)
            // pin the artifact location even when the ambient environment
            // redirects cargo's target dir
            .env("CARGO_TARGET_DIR", dir.join("target"));
        if release {
            cmd.arg("--release");
        }
        let output = cmd
            .output()
            .map_err(|e| other(std::format_args!("running cargo: {e}")))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(other(std::format_args!(
                "cargo build of bundle '{}' failed:\n{stderr}",
                bundle.name
            )));
        }

        let profile = if release { "release" } else { "debug" };
        let artifact = dir.join("target").join(profile).join(std::format!(
            "{}{}{}",
            std::env::consts::DLL_PREFIX,
            bundle.name.replace('-', "_"),
            std::env::consts::DLL_SUFFIX
        ));
        self.link_bundle(artifact)
    }

    /// Link a bundle cdylib: loads it, checks its raft-ffi
    /// version against this runtime's.
    /// Returns the registered module names.
    fn pre_link_bundle(&mut self, path: impl AsRef<Path>) -> Result<Bundle, RuntimeError> {
        let path = path.as_ref();

        // SAFETY: loading a library is inherently trusting its init code.
        let lib = unsafe { libloading::Library::new(path) }
            .map_err(|e| other(std::format_args!("loading {}: {e}", path.display())))?;

        // SAFETY: symbol name/type pairs are fixed by the raft-ffi contract.
        // If symbol is found it is assumed to be a valid function pointer of the expected type.
        let version: libloading::Symbol<ffi::RaftFFIVersionFn> =
            unsafe { lib.get(ffi::FFI_VERSION_STATIC_NAME.as_bytes()) }
                .map_err(|e| other(std::format_args!("missing version symbol: {e}")))?;

        // SAFETY: the version fn returns a static NUL-terminated string.
        let bundle_version =
            unsafe { core::ffi::CStr::from_ptr(version() as *const core::ffi::c_char) };

        if bundle_version != ffi::FFI_VERSION {
            return Err(other(std::format_args!(
                "bundle raft-ffi version '{bundle_version:?}' does not match host '{:?}'",
                ffi::FFI_VERSION
            )));
        }

        let module_name: libloading::Symbol<ffi::RaftFFIModuleNameFn> =
            unsafe { lib.get(ffi::MODULE_NAME_FN_NAME.as_bytes()) }
                .map_err(|e| other(std::format_args!("missing module name symbol: {e}")))?;

        let mut names = Vec::new();

        for i in 0.. {
            let mut ptr = core::ptr::null();
            let len = unsafe { module_name(i, &mut ptr) };

            if ptr.is_null() {
                break;
            }

            let slice = unsafe { core::slice::from_raw_parts(ptr, len) };

            let name = core::str::from_utf8(slice)
                .map_err(|e| other(core::format_args!("module name is not UTF-8: {e}")))?;

            names.push(name.to_owned());
        }

        Ok(Bundle {
            lib,
            module_names: names,
        })
    }

    /// Link an already-built bundle cdylib: loads it, checks its raft-ffi
    /// version against this runtime's, initializes it (interning its names
    /// through this runtime's [`ffi_host`](Runtime::ffi_host)), registers
    /// every module it exposes, and holds the library for the runtime's
    /// lifetime. Returns the registered module names.
    fn init_bundle(&mut self, bundle: Bundle) -> Result<Vec<String>, RuntimeError> {
        let modules = {
            // SAFETY: as the version symbol's.
            let init: libloading::Symbol<ffi::RaftFFIInitBundleFn> =
                unsafe { bundle.lib.get(ffi::INIT_RAFT_BUNDLE_FN_NAME.as_bytes()) }
                    .map_err(|e| other(std::format_args!("missing init symbol: {e}")))?;

            let host = self.ffi_host();
            let mut bundle = ffi::RaftFFIBundle {
                modules: ffi::RawVal::uninit(),
            };
            let mut error_buf = [0u8; 1024];
            // SAFETY: `host` wraps this exact runtime; buffer/len are a
            // valid writable region, per `RaftFFIInitBundleFn`'s contract.
            let code = unsafe { init(&mut bundle, &host, error_buf.as_mut_ptr(), error_buf.len()) };
            if code < 0 {
                let error_len = match usize::try_from(-code) {
                    Ok(len) if len <= error_buf.len() => len,
                    _ => return Err(other("bundle init failed with incorrect error length")),
                };

                let msg =
                    core::str::from_utf8(&error_buf[..error_len]).unwrap_or("<non-UTF-8 error>");
                return Err(other(std::format_args!("bundle init failed: {msg}")));
            }
            // SAFETY: init succeeded - ownership of the modules record
            // transfers to the host.
            unsafe { Val::from_raw(bundle.modules) }
        };

        let ValEnum::Record(record) = modules.unpack() else {
            return Err(other("bundle modules value is not a record"));
        };
        let mut names = Vec::with_capacity(record.len());
        for i in 0..record.len() {
            let (name, module) = record
                .entry_at(i)
                .expect("record length changed during iteration");
            let name = name.to_string();
            self.register_module(&name, module);
            names.push(name);
        }

        // hold the library for the runtime's lifetime: the values just
        // registered carry vtable/code pointers into it
        self.libraries.push(bundle.lib);
        Ok(names)
    }

    /// Link an already-built bundle cdylib: loads it, checks its raft-ffi
    /// version against this runtime's, initializes it (interning its names
    /// through this runtime's [`ffi_host`](Runtime::ffi_host)), registers
    /// every module it exposes, and holds the library for the runtime's
    /// lifetime. Returns the registered module names.
    pub fn link_bundle(&mut self, path: impl AsRef<Path>) -> Result<Vec<String>, RuntimeError> {
        let bundle = self.pre_link_bundle(path)?;
        self.init_bundle(bundle)
    }

    /// Scan `cdylib_dirs` in order for a bundle exposing `name`: every
    /// dylib-suffixed file in each directory is [linked](Runtime::link_bundle)
    /// and checked - the first one that exposes `name` stays linked (with
    /// everything else it exposes also registered); every other candidate
    /// tried along the way is unlinked again. A candidate that fails to
    /// link at all (not a raft bundle, version mismatch, ...) is silently
    /// skipped rather than aborting the whole search.
    pub(crate) fn find_cdylib_module(&mut self, name: &str) -> Result<Option<Val>, RuntimeError> {
        let dirs = self.cdylib_dirs.clone();
        for dir in &dirs {
            let Ok(entries) = std::fs::read_dir(dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str())
                    != Some(std::env::consts::DLL_EXTENSION)
                {
                    continue;
                }
                if let Some(module) = self.try_link_cdylib_for(&path, name)? {
                    return Ok(Some(module));
                }
            }
        }
        Ok(None)
    }

    /// Link `path` and check whether it exposes `name`: keeps it linked
    /// (and everything it exposes registered) and returns the module if
    /// so; otherwise unregisters everything it just registered, unlinks
    /// the library (dropping the just-registered `Val`s first - they carry
    /// pointers into it), and returns `None`. A link failure also returns
    /// `None` (see [`Runtime::find_cdylib_module`]).
    fn try_link_cdylib_for(
        &mut self,
        path: &Path,
        name: &str,
    ) -> Result<Option<Val>, RuntimeError> {
        let Ok(bundle) = self.pre_link_bundle(path) else {
            return Ok(None);
        };

        if bundle.module_names.iter().any(|mn| mn == name) {
            self.init_bundle(bundle)?;
            let id = self.ctx.string(name);
            return Ok(self.modules.get(&id).cloned());
        } else {
            Ok(None)
        }
    }
}

fn parse_bundle_module(name: &str, source: &str) -> Result<raft_ast::Module, RuntimeError> {
    let tokens = raft_ast::lexer::parse_str(source, &raft_ast::lexer::Options::wss())
        .map_err(|e| other(std::format_args!("module '{name}': lex error: {e:?}")))?;
    let mut stream = raft_ast::parser::TokenStream::new(tokens);
    stream
        .parse_module()
        .map_err(|e| other(std::format_args!("module '{name}': parse error: {e:?}")))
}

fn validate_crate_name(name: &str) -> Result<(), RuntimeError> {
    let mut chars = name.chars();
    let valid_start = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
    let valid_rest = chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if !valid_start || !valid_rest {
        return Err(other(std::format_args!(
            "'{name}' is not a valid bundle crate name"
        )));
    }
    Ok(())
}
