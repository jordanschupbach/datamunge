fn main() {
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo:rerun-if-env-changed=DATAMUNGE_SYS_NO_PKG_CONFIG");
    println!("cargo:rerun-if-changed=src/bspline.cpp");

    if std::env::var_os("DATAMUNGE_SYS_NO_PKG_CONFIG").is_some() {
        return;
    }
    let library = pkg_config::Config::new()
        .cargo_metadata(false)
        .probe("datamunge")
        .expect("pkg-config could not find `datamunge`");
    cc::Build::new()
        .cpp(true)
        .std("c++23")
        .includes(&library.include_paths)
        .file("src/bspline.cpp")
        .compile("datamunge_rust_bspline");
    pkg_config::probe_library("datamunge").expect("pkg-config could not find `datamunge`");
}
