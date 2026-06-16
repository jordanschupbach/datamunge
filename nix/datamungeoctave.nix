{
  pkgs ? import <nixpkgs> { },
}:

let
  datamunge = import ./datamunge.nix { inherit pkgs; };
in
pkgs.stdenv.mkDerivation rec {
  pname = "datamungeoctave";
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
    pkgs.octave
  ];

  configurePhase = ''
    cmake -S src/datamungeoctave -B build/datamungeoctave \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="${datamunge}"
  '';

  buildPhase = ''
    cmake --build build/datamungeoctave -j $NIX_BUILD_CORES
  '';

  installPhase = ''
    cmake --install build/datamungeoctave --prefix "$out"

    # Ensure the Octave module can find libdatamunge.so at runtime inside the Nix store.
    datamungeLib="$(find "${datamunge}" -name 'libdatamunge.so' -print -quit)"
    if [ -z "$datamungeLib" ]; then
      echo "Could not find libdatamunge.so in ${datamunge}" >&2
      find "${datamunge}" -maxdepth 4 -type f -name 'libdatamunge*' -print >&2 || true
      exit 1
    fi
    datamungeLibDir="$(dirname "$datamungeLib")"

    octFilePath="$(find "$out" -type f -name 'datamunge.oct' -print -quit)"
    if [ -z "$octFilePath" ]; then
      echo "Could not find installed datamunge.oct under $out" >&2
      find "$out" -maxdepth 6 -type f -print >&2
      exit 1
    fi

    existingRpath="$(${pkgs.patchelf}/bin/patchelf --print-rpath "$octFilePath" || true)"
    if [ -n "$existingRpath" ]; then
      ${pkgs.patchelf}/bin/patchelf --set-rpath "$datamungeLibDir:$existingRpath" "$octFilePath"
    else
      ${pkgs.patchelf}/bin/patchelf --set-rpath "$datamungeLibDir" "$octFilePath"
    fi
  '';

  meta = with pkgs.lib; {
    description = "Octave (SWIG) bindings for the datamunge library.";
    license = licenses.unlicense;
    platforms = platforms.linux;
  };
}
