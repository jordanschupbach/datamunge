{
  pkgs ? import <nixpkgs> { },
}:

let
  datamunge = import ./datamunge.nix { inherit pkgs; };
  lua = pkgs.lua5_4 or (pkgs.lua54 or pkgs.lua);
in
pkgs.stdenv.mkDerivation rec {
  pname = "datamungelua";
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
    pkgs.swig
    pkgs.patchelf
  ];

  buildInputs = [
    datamunge
    pkgs.arrow-cpp
    lua
  ];

  configurePhase = ''
    export CMAKE_PREFIX_PATH="${datamunge}:${pkgs.arrow-cpp}''${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
    cmake -S src/datamungelua -B build/datamungelua \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH"
  '';

  buildPhase = ''
    cmake --build build/datamungelua -j $NIX_BUILD_CORES
  '';

  installPhase = ''
    cmake --install build/datamungelua --prefix "$out"

    # Ensure the Lua module can find libdatamunge at runtime inside the Nix store
    # (.so on Linux, .dylib on macOS).
    datamungeLib="$(find "${datamunge}" \( -name 'libdatamunge.so' -o -name 'libdatamunge.dylib' \) -print -quit)"
    if [ -z "$datamungeLib" ]; then
      echo "Could not find libdatamunge shared library in ${datamunge}" >&2
      find "${datamunge}" -maxdepth 4 -type f -name 'libdatamunge*' -print >&2 || true
      exit 1
    fi
    datamungeLibDir="$(dirname "$datamungeLib")"
    soPath="$(find "$out" -path '*/lib*/lua/datamunge.so' -print -quit)"
    if [ -z "$soPath" ]; then
      echo "Could not find installed datamunge.so under $out" >&2
      find "$out" -maxdepth 4 -type f -print >&2
      exit 1
    fi
    if [ "$(uname)" = "Darwin" ]; then
      # On macOS the Nix-built dylib carries an absolute install_name, so the module
      # resolves it without rpath patching; just add an rpath for robustness.
      ${pkgs.darwin.cctools}/bin/install_name_tool -add_rpath "$datamungeLibDir" "$soPath" || true
    else
      existingRpath="$(${pkgs.patchelf}/bin/patchelf --print-rpath "$soPath" || true)"
      if [ -n "$existingRpath" ]; then
        ${pkgs.patchelf}/bin/patchelf --set-rpath "$datamungeLibDir:$existingRpath" "$soPath"
      else
        ${pkgs.patchelf}/bin/patchelf --set-rpath "$datamungeLibDir" "$soPath"
      fi
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
    ln -sf "$soPath" "$soDir/$luaVersion/datamunge.so"

    if [ -f "$out/share/lua/datamunge.lua" ]; then
      ln -sf "$out/share/lua/datamunge.lua" "$out/share/lua/$luaVersion/datamunge.lua"
    else
      echo "Expected Lua loader at $out/share/lua/datamunge.lua" >&2
      find "$out/share" -maxdepth 4 -type f -print >&2 || true
      exit 1
    fi
  '';

  meta = with pkgs.lib; {
    description = "Lua (SWIG) bindings for the datamunge library.";
    license = licenses.unlicense;
    platforms = platforms.unix;
  };
}
