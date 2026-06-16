{
  pkgs ? import <nixpkgs> { },
}:

let
  octra = import ./octra.nix { inherit pkgs; };
in
pkgs.stdenv.mkDerivation rec {
  pname = "octratcl";
  version = "0.0.1";

  src = pkgs.lib.cleanSource ../.;

  nativeBuildInputs = [
    pkgs.cmake
    pkgs.pkg-config
    pkgs.swig
  ];

  buildInputs = [
    octra
    pkgs.tcl
    pkgs.tk
  ];

  configurePhase = ''
    cmake -S src/octratcl -B build/octratcl -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${octra}"
  '';

  buildPhase = ''
    cmake --build build/octratcl -j $NIX_BUILD_CORES
  '';

  installPhase = ''
        tclVersionDir="$(${pkgs.tcl}/bin/tclsh <<< 'puts [info library]' | sed -E 's|.*/(tcl[0-9]+\\.[0-9]+).*|\\1|')"
        if [ -z "$tclVersionDir" ]; then
          echo "Could not determine Tcl version directory from [info library]" >&2
          exit 1
        fi

        pkgDir="$out/lib/$tclVersionDir"
        mkdir -p "$pkgDir"

        cp -v build/octratcl/Octra.so "$pkgDir/"

        if [ ! -f src/octratcl/pkgIndex.tcl ]; then
          mkdir -p src/octratcl
          cat > src/octratcl/pkgIndex.tcl <<'EOF'
    # Tcl package index for the SWIG-generated Octra extension.
    #
    # Notes:
    # - The SWIG-generated init function is `Octra_Init` (loaded via `load ... Octra`).
    # - SWIG's Tcl backend provides `package provide octra 0.0` and installs commands
    #   in the global namespace (e.g. `hello`), which doesn't match this repo's
    #   intended contract (`package require Octra 0.0.1` and `octra::...`).
    # - This shim normalizes the package/version and exports a minimal `octra::`
    #   namespace API expected by `examples/` and `tests/` (formerly `bindings_tests/`).

    package ifneeded Octra 0.0.1 [list apply {{dir} {
      load [file join $dir "Octra[info sharedlibextension]"] Octra

      namespace eval octra {}
      foreach {src dst} {
        hello octra::hello
        new_DVector octra::new_DVector
        DVector_size octra::DVector_size
        DVector_get octra::DVector_get
        DVector_set octra::DVector_set
        DVector_push octra::DVector_push
        DVector_pop octra::DVector_pop
        DVector_push octra::DVector_push_back
        delete_DVector octra::delete_DVector
        new_DPair octra::new_DPair
        DPair_first_get octra::DPair_first_get
        DPair_second_get octra::DPair_second_get
        DPair_first_set octra::DPair_first_set
        DPair_second_set octra::DPair_second_set
        delete_DPair octra::delete_DPair
      } {
        if {[llength [info commands $src]] && ![llength [info commands $dst]]} {
          interp alias {} $dst {} $src
        }
      }

      package provide Octra 0.0.1
    }} $dir]
    EOF
        fi

        cp -v src/octratcl/pkgIndex.tcl "$pkgDir/"
  '';

  meta = with pkgs.lib; {
    description = "Tcl (SWIG) bindings for the octra library.";
    license = licenses.unlicense;
    platforms = platforms.linux;
  };
}
