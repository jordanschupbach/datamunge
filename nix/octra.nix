{
  pkgs ? import <nixpkgs> { config.allowUnfree = true; },
}:

let
  octra = pkgs.stdenv.mkDerivation rec {
    pname = "octra";
    version = "0.0.1";

    src = pkgs.lib.cleanSourceWith {
      src = ../.;
      filter =
        path: type:
        let
          base = builtins.baseNameOf path;
        in
        !(
          base == ".git" || base == "build" || base == "dist" || base == "node_modules" || base == "result"
        );
    };

    nativeBuildInputs = [
      pkgs.cmake
      pkgs.pkg-config
    ];

    buildInputs = [
      pkgs.clang
    ];

    cmakeFlags = [
      "-DCMAKE_CXX_COMPILER=clang++"
      "-DCMAKE_INSTALL_LIBDIR=lib"
    ];

    configurePhase = ''
      cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$out ${pkgs.lib.escapeShellArgs cmakeFlags}
    '';

    buildPhase = ''
      cmake --build build -j $NIX_BUILD_CORES
    '';

    installPhase = ''
      cmake --install build --prefix $out
    '';
  };
in
octra
