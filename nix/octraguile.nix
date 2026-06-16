{
  pkgs ? import <nixpkgs> { },
}:

let
  octra = import ./octra.nix { inherit pkgs; };
in
pkgs.stdenv.mkDerivation rec {
  pname = "octraguile";
  version = "0.0.1";

  src = pkgs.lib.cleanSource ../.;

  nativeBuildInputs = [
    pkgs.cmake
    pkgs.pkg-config
    pkgs.swig
    pkgs.patchelf
  ];

  buildInputs = [
    octra
    pkgs.guile
  ];

  configurePhase = ''
    cmake -S src/octraguile -B build/octraguile \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="${octra}"
  '';

  buildPhase = ''
    cmake --build build/octraguile -j $NIX_BUILD_CORES
  '';

  installPhase = ''
    cmake --install build/octraguile --prefix "$out"

    # Ensure the extension can find liboctra.so at runtime inside the Nix store.
    octraLib="$(find "${octra}" -name 'liboctra.so' -print -quit)"
    if [ -z "$octraLib" ]; then
      echo "Could not find liboctra.so in ${octra}" >&2
      find "${octra}" -maxdepth 4 -type f -name 'liboctra*' -print >&2 || true
      exit 1
    fi
    octraLibDir="$(dirname "$octraLib")"

    soPath="$(find "$out" -path '*/guile/*/extensions/octra.so' -print -quit)"
    if [ -z "$soPath" ]; then
      echo "Could not find installed octra.so under $out" >&2
      find "$out" -maxdepth 6 -type f -print >&2
      exit 1
    fi

    existingRpath="$(${pkgs.patchelf}/bin/patchelf --print-rpath "$soPath" || true)"
    if [ -n "$existingRpath" ]; then
      ${pkgs.patchelf}/bin/patchelf --set-rpath "$octraLibDir:$existingRpath" "$soPath"
    else
      ${pkgs.patchelf}/bin/patchelf --set-rpath "$octraLibDir" "$soPath"
    fi
  '';

  meta = with pkgs.lib; {
    description = "Guile (SWIG) bindings for the octra library.";
    license = licenses.unlicense;
    platforms = platforms.linux;
  };
}
