//! Raft → Rust transpilation.
//!
//! [`BundleGenerator`] consumes parsed [`raft_ast::Module`]s and produces
//! a complete, standalone cdylib crate (depending only on `raft-core`):
//! one Rust module file per Raft module, and a `lib.rs` carrying the
//! bundle's exactly-sized name/atom tables, the generated `mod support`
//! runtime (emitted from `bundle_support.rs`, this crate's template), and
//! the `raft_bundle!` init function (see `raft-ffi`) that interns every
//! name into the host and exposes the loaded modules as a record `RawVal`.
//!
//! Transpiled code uses ordinary Rust frames: each Raft binding is either
//! a plain Rust local or — when some nested `fn` reads it — a field of
//! that scope's single per-call `Rc<CaptureN>` structure; nested `fn`s are
//! Rust `move` closures capturing the enclosing capture handles. Values
//! are `raft_core::Val`s throughout; conversion to and from `RawVal`
//! happens only at the FFI boundary, inside the generated `support`
//! module.

mod generate;

pub use generate::{BundleGenerator, GenError};
