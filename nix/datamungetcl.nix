{
  pkgs ? import <nixpkgs> { },
}:

let
  datamunge = import ./datamunge.nix { inherit pkgs; };
in
pkgs.stdenv.mkDerivation rec {
  pname = "datamungetcl";
  version = "0.0.1";

  src = pkgs.lib.cleanSource ../.;

  nativeBuildInputs = [
    pkgs.cmake
    pkgs.pkg-config
    pkgs.swig
  ];

  buildInputs = [
    datamunge
    pkgs.tcl
    pkgs.tk
  ];

  configurePhase = ''
    cmake -S src/datamungetcl -B build/datamungetcl -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${datamunge}"
  '';

  buildPhase = ''
    cmake --build build/datamungetcl -j $NIX_BUILD_CORES
  '';

  installPhase = ''
        tclVersionDir="$(${pkgs.tcl}/bin/tclsh <<< 'puts [info library]' | sed -E 's|.*/(tcl[0-9]+\\.[0-9]+).*|\\1|')"
        if [ -z "$tclVersionDir" ]; then
          echo "Could not determine Tcl version directory from [info library]" >&2
          exit 1
        fi

        pkgDir="$out/lib/$tclVersionDir"
        mkdir -p "$pkgDir"

        cp -v build/datamungetcl/Datamunge.so "$pkgDir/"

        if [ ! -f src/datamungetcl/pkgIndex.tcl ]; then
          mkdir -p src/datamungetcl
          cat > src/datamungetcl/pkgIndex.tcl <<'EOF'
    # Tcl package index for the SWIG-generated Datamunge extension.
    #
    # Notes:
    # - The SWIG-generated init function is `Datamunge_Init` (loaded via `load ... Datamunge`).
    # - SWIG's Tcl backend provides `package provide datamunge 0.0` and installs commands
    #   in the global namespace (e.g. `hello`), which doesn't match this repo's
    #   intended contract (`package require Datamunge 0.0.1` and `datamunge::...`).
    # - This shim normalizes the package/version and exports a minimal `datamunge::`
    #   namespace API expected by `examples/` and `tests/` (formerly `bindings_tests/`).

    package ifneeded Datamunge 0.0.1 [list apply {{dir} {
      load [file join $dir "Datamunge[info sharedlibextension]"] Datamunge

      namespace eval datamunge {}
      foreach {src dst} {
        hello datamunge::hello
        new_DVector datamunge::new_DVector
        DVector_size datamunge::DVector_size
        DVector_get datamunge::DVector_get
        DVector_set datamunge::DVector_set
        DVector_push datamunge::DVector_push
        DVector_pop datamunge::DVector_pop
        DVector_push datamunge::DVector_push_back
        delete_DVector datamunge::delete_DVector
        new_DPair datamunge::new_DPair
        DPair_first_get datamunge::DPair_first_get
        DPair_second_get datamunge::DPair_second_get
        DPair_first_set datamunge::DPair_first_set
        DPair_second_set datamunge::DPair_second_set
        delete_DPair datamunge::delete_DPair
      } {
        if {[llength [info commands $src]] && ![llength [info commands $dst]]} {
          interp alias {} $dst {} $src
        }
      }

      package provide Datamunge 0.0.1
    }} $dir]
    EOF
        fi

        cp -v src/datamungetcl/pkgIndex.tcl "$pkgDir/"
  '';

  meta = with pkgs.lib; {
    description = "Tcl (SWIG) bindings for the datamunge library.";
    license = licenses.unlicense;
    platforms = platforms.linux;
  };
}
