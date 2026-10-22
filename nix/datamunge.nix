{
  pkgs ? import <nixpkgs> { config.allowUnfree = true; },
}:

let
  datamunge = pkgs.stdenv.mkDerivation rec {
    pname = "datamunge";
    version = "0.0.1";

    src = import ./source.nix { lib = pkgs.lib; };

    nativeBuildInputs = [
      pkgs.cmake
      pkgs.pkg-config
    ];

    buildInputs = [
      pkgs.clang
      pkgs.arrow-cpp
      pkgs.xorg.libX11.dev
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
datamunge
