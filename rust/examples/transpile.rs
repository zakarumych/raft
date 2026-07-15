//! Transpile Raft modules into a complete cdylib bundle crate.
//!
//! Usage:
//!   cargo run -p raft-rust --example transpile -- <out-dir> <raft-repo-path> <module.raft>...
//!
//! Writes `<out-dir>/Cargo.toml`, `<out-dir>/src/lib.rs` and one
//! `<out-dir>/src/<name>.rs` per input file (named after its file stem).
//! `<raft-repo-path>` is written into the generated Cargo.toml as the path
//! the `raft-core` dependency lives under.

use std::path::Path;

use raft_rust::BundleGenerator;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [out_dir, raft_repo, sources @ ..] = args.as_slice() else {
        eprintln!("usage: transpile <out-dir> <raft-repo-path> <module.raft>...");
        std::process::exit(2);
    };
    if sources.is_empty() {
        eprintln!("no input modules given");
        std::process::exit(2);
    }

    let mut generator = BundleGenerator::new();
    for source_path in sources {
        let path = Path::new(source_path);
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_else(|| die(&format!("bad module path: {source_path}")));
        let source = std::fs::read_to_string(path)
            .unwrap_or_else(|e| die(&format!("reading {source_path}: {e}")));

        let tokens = raft_ast::lexer::parse_str(&source, &raft_ast::lexer::Options::wss())
            .unwrap_or_else(|e| die(&format!("{source_path}: lex error: {e:?}")));
        let mut stream = raft_ast::parser::TokenStream::new(tokens);
        let module = stream
            .parse_module()
            .unwrap_or_else(|e| die(&format!("{source_path}: parse error: {e:?}")));

        generator
            .add_module(name, &module)
            .unwrap_or_else(|e| die(&format!("{source_path}: {e}")));
        eprintln!("transpiled module `{name}`");
    }

    let out = Path::new(out_dir);
    let crate_name = out
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("raft_bundle");
    generator
        .write_crate(out, crate_name, raft_repo)
        .unwrap_or_else(|e| die(&format!("writing crate: {e}")));
    eprintln!("bundle crate written to {out_dir}");
}

fn die(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(1)
}
