{
  pkgs ? import <nixpkgs> { },
}:

let
  datamunge = import ./datamunge.nix { inherit pkgs; };
in
pkgs.stdenv.mkDerivation rec {
  pname = "datamunged";
  version = "0.0.1";

  src = pkgs.lib.cleanSource ../.;

  nativeBuildInputs = [
    pkgs.pkg-config
    pkgs.swig
    pkgs.dub
    pkgs.ldc
    pkgs.stdenv.cc
  ];

  buildInputs = [
    datamunge
  ];

  buildPhase = ''
        runHook preBuild

        mkdir -p src/datamunged/source
        swig -c++ -d -Iinclude \
          -o src/datamunged/source/datamunged_wrap.cpp \
          -outdir src/datamunged/source \
          src/datamunged/swig/datamunged.i

        export HOME="$TMPDIR"
        export DUB_HOME="$TMPDIR/dub"
        mkdir -p "$DUB_HOME"

        export PKG_CONFIG_PATH="${datamunge}/lib/pkgconfig''${PKG_CONFIG_PATH:+:}$PKG_CONFIG_PATH"
        export DATAMUNGE_PREFIX="$(${pkgs.pkg-config}/bin/pkg-config --variable=prefix datamunge)"
        export DATAMUNGE_LIBDIR="$(${pkgs.pkg-config}/bin/pkg-config --variable=libdir datamunge)"
        export DATAMUNGE_CFLAGS="$(${pkgs.pkg-config}/bin/pkg-config --cflags datamunge)"
        export DATAMUNGE_LDFLAGS="$(${pkgs.pkg-config}/bin/pkg-config --libs datamunge)"

        export CFLAGS="$DATAMUNGE_CFLAGS ''${CFLAGS:-}"
        export CXXFLAGS="$DATAMUNGE_CFLAGS ''${CXXFLAGS:-}"
        export LDFLAGS="$DATAMUNGE_LDFLAGS ''${LDFLAGS:-}"

        export LIBRARY_PATH="$DATAMUNGE_LIBDIR''${LIBRARY_PATH:+:}$LIBRARY_PATH"
        export LD_LIBRARY_PATH="$DATAMUNGE_LIBDIR''${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"

        c++ -shared -fPIC \
          $DATAMUNGE_CFLAGS \
          -o src/datamunged/source/libdatamunge_wrap.so \
          src/datamunged/source/datamunged_wrap.cpp \
          $DATAMUNGE_LDFLAGS \
          -Wl,-rpath,"$DATAMUNGE_LIBDIR"

        if [ ! -f src/datamunged/dub.json ]; then
          cat > src/datamunged/dub.json <<'EOF'
    {
      "name": "datamunged",
      "description": "D (SWIG) bindings for the datamunge library.",
      "license": "Unlicense",
      "version": "0.0.1",
      "targetType": "library",
      "sourcePaths": ["source"],
      "importPaths": ["source"]
    }
    EOF
        fi

        (cd src/datamunged && dub build --compiler=ldc2 --build=release)

        runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    pkgDir="$out/share/dub/packages/datamunged-${version}"
    mkdir -p "$pkgDir"
    cp -R src/datamunged/dub.json src/datamunged/source "$pkgDir/"

    # Also ship the built artifact for convenience (name differs by compiler/platform).
    mkdir -p "$out/lib"
    cp -v src/datamunged/source/libdatamunge_wrap.so "$out/lib/"
    for ext in a so dylib lib; do
      if ! find src/datamunged -maxdepth 2 -type f -name "*.$ext" -exec cp -v {} "$out/lib/" \; 2>/dev/null; then
        :
      fi
    done

    runHook postInstall
  '';

  meta = with pkgs.lib; {
    description = "D (SWIG) bindings for the datamunge library.";
    license = licenses.unlicense;
    platforms = platforms.linux;
  };
}
