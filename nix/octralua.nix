{
  pkgs ? import <nixpkgs> { },
}:

let
  octra = import ./octra.nix { inherit pkgs; };
  lua = pkgs.lua5_4 or (pkgs.lua54 or pkgs.lua);
in
pkgs.stdenv.mkDerivation rec {
  pname = "octralua";
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
    lua
  ];

  configurePhase = ''
    cmake -S src/octralua -B build/octralua \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="${octra}"
  '';

  buildPhase = ''
    cmake --build build/octralua -j $NIX_BUILD_CORES
  '';

  installPhase = ''
    cmake --install build/octralua --prefix "$out"

    # Ensure the Lua module can find liboctra.so at runtime inside the Nix store.
    octraLib="$(find "${octra}" -name 'liboctra.so' -print -quit)"
    if [ -z "$octraLib" ]; then
      echo "Could not find liboctra.so in ${octra}" >&2
      find "${octra}" -maxdepth 4 -type f -name 'liboctra*' -print >&2 || true
      exit 1
    fi
    octraLibDir="$(dirname "$octraLib")"
    soPath="$(find "$out" -path '*/lib*/lua/octra.so' -print -quit)"
    if [ -z "$soPath" ]; then
      echo "Could not find installed octra.so under $out" >&2
      find "$out" -maxdepth 4 -type f -print >&2
      exit 1
    fi
    existingRpath="$(${pkgs.patchelf}/bin/patchelf --print-rpath "$soPath" || true)"
    if [ -n "$existingRpath" ]; then
      ${pkgs.patchelf}/bin/patchelf --set-rpath "$octraLibDir:$existingRpath" "$soPath"
    else
      ${pkgs.patchelf}/bin/patchelf --set-rpath "$octraLibDir" "$soPath"
    fi

    luaVersion="$(${lua}/bin/lua -e 'io.write((_VERSION or ""):match("%d+%.%d+") or "")')"
    if [ -z "$luaVersion" ]; then
      echo "Could not determine Lua version from _VERSION" >&2
      ${lua}/bin/lua -e 'print("_VERSION=" .. tostring(_VERSION))' >&2
      exit 1
    fi

    # Standard Lua search paths are versioned; provide versioned aliases.
    soDir="$(dirname "$soPath")"
    mkdir -p "$soDir/$luaVersion" "$out/share/lua/$luaVersion"
    ln -sf "$soPath" "$soDir/$luaVersion/octra.so"

    if [ -f "$out/share/lua/octra.lua" ]; then
      ln -sf "$out/share/lua/octra.lua" "$out/share/lua/$luaVersion/octra.lua"
    else
      echo "Expected Lua loader at $out/share/lua/octra.lua" >&2
      find "$out/share" -maxdepth 4 -type f -print >&2 || true
      exit 1
    fi
  '';

  meta = with pkgs.lib; {
    description = "Lua (SWIG) bindings for the octra library.";
    license = licenses.unlicense;
    platforms = platforms.linux;
  };
}
