{
  pkgs ? import <nixpkgs> { },
}:

let
  datamunge = import ./datamunge.nix { inherit pkgs; };
in
pkgs.stdenv.mkDerivation rec {
  pname = "datamungeguile";
  version = "0.0.1";

  src = pkgs.lib.cleanSource ../.;

  nativeBuildInputs = [
    pkgs.cmake
    pkgs.pkg-config
    pkgs.swig
    pkgs.patchelf
  ];

  buildInputs = [
    datamunge
    pkgs.guile
  ];

  configurePhase = ''
    cmake -S src/datamungeguile -B build/datamungeguile \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="${datamunge}"
  '';

  buildPhase = ''
    cmake --build build/datamungeguile -j $NIX_BUILD_CORES
  '';

  installPhase = ''
    cmake --install build/datamungeguile --prefix "$out"

    # Ensure the extension can find libdatamunge.so at runtime inside the Nix store.
    datamungeLib="$(find "${datamunge}" -name 'libdatamunge.so' -print -quit)"
    if [ -z "$datamungeLib" ]; then
      echo "Could not find libdatamunge.so in ${datamunge}" >&2
      find "${datamunge}" -maxdepth 4 -type f -name 'libdatamunge*' -print >&2 || true
      exit 1
    fi
    datamungeLibDir="$(dirname "$datamungeLib")"

    soPath="$(find "$out" -path '*/guile/*/extensions/datamunge.so' -print -quit)"
    if [ -z "$soPath" ]; then
      echo "Could not find installed datamunge.so under $out" >&2
      find "$out" -maxdepth 6 -type f -print >&2
      exit 1
    fi

    existingRpath="$(${pkgs.patchelf}/bin/patchelf --print-rpath "$soPath" || true)"
    if [ -n "$existingRpath" ]; then
      ${pkgs.patchelf}/bin/patchelf --set-rpath "$datamungeLibDir:$existingRpath" "$soPath"
    else
      ${pkgs.patchelf}/bin/patchelf --set-rpath "$datamungeLibDir" "$soPath"
    fi
  '';

  meta = with pkgs.lib; {
    description = "Guile (SWIG) bindings for the datamunge library.";
    license = licenses.unlicense;
    platforms = platforms.linux;
  };
}
