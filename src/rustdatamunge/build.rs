fn main() {
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo:rerun-if-env-changed=DATAMUNGE_SYS_NO_PKG_CONFIG");

    if std::env::var_os("DATAMUNGE_SYS_NO_PKG_CONFIG").is_some() {
        return;
    }

    // Prefer system-installed `datamunge` (e.g. via `nix develop .#rust`).
    // pkg-config will emit the correct link flags.
    pkg_config::probe_library("datamunge").expect("pkg-config could not find `datamunge`");
}
