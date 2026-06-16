{
  pkgs ? import <nixpkgs> { },
}:

let
  octra = import ./octra.nix { inherit pkgs; };
in
pkgs.stdenv.mkDerivation rec {
  pname = "octraoctave";
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
    pkgs.octave
  ];

  configurePhase = ''
    cmake -S src/octraoctave -B build/octraoctave \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="${octra}"
  '';

  buildPhase = ''
    cmake --build build/octraoctave -j $NIX_BUILD_CORES
  '';

  installPhase = ''
    cmake --install build/octraoctave --prefix "$out"

    # Ensure the Octave module can find liboctra.so at runtime inside the Nix store.
    octraLib="$(find "${octra}" -name 'liboctra.so' -print -quit)"
    if [ -z "$octraLib" ]; then
      echo "Could not find liboctra.so in ${octra}" >&2
      find "${octra}" -maxdepth 4 -type f -name 'liboctra*' -print >&2 || true
      exit 1
    fi
    octraLibDir="$(dirname "$octraLib")"

    octFilePath="$(find "$out" -type f -name 'octra.oct' -print -quit)"
    if [ -z "$octFilePath" ]; then
      echo "Could not find installed octra.oct under $out" >&2
      find "$out" -maxdepth 6 -type f -print >&2
      exit 1
    fi

    existingRpath="$(${pkgs.patchelf}/bin/patchelf --print-rpath "$octFilePath" || true)"
    if [ -n "$existingRpath" ]; then
      ${pkgs.patchelf}/bin/patchelf --set-rpath "$octraLibDir:$existingRpath" "$octFilePath"
    else
      ${pkgs.patchelf}/bin/patchelf --set-rpath "$octraLibDir" "$octFilePath"
    fi
  '';

  meta = with pkgs.lib; {
    description = "Octave (SWIG) bindings for the octra library.";
    license = licenses.unlicense;
    platforms = platforms.linux;
  };
}
