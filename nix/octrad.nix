{
  pkgs ? import <nixpkgs> { },
}:

let
  octra = import ./octra.nix { inherit pkgs; };
in
pkgs.stdenv.mkDerivation rec {
  pname = "octrad";
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
    octra
  ];

  buildPhase = ''
        runHook preBuild

        mkdir -p src/octrad/source
        swig -c++ -d -Iinclude \
          -o src/octrad/source/octrad_wrap.cpp \
          -outdir src/octrad/source \
          src/octrad/swig/octrad.i

        export HOME="$TMPDIR"
        export DUB_HOME="$TMPDIR/dub"
        mkdir -p "$DUB_HOME"

        export PKG_CONFIG_PATH="${octra}/lib/pkgconfig''${PKG_CONFIG_PATH:+:}$PKG_CONFIG_PATH"
        export OCTRA_PREFIX="$(${pkgs.pkg-config}/bin/pkg-config --variable=prefix octra)"
        export OCTRA_LIBDIR="$(${pkgs.pkg-config}/bin/pkg-config --variable=libdir octra)"
        export OCTRA_CFLAGS="$(${pkgs.pkg-config}/bin/pkg-config --cflags octra)"
        export OCTRA_LDFLAGS="$(${pkgs.pkg-config}/bin/pkg-config --libs octra)"

        export CFLAGS="$OCTRA_CFLAGS ''${CFLAGS:-}"
        export CXXFLAGS="$OCTRA_CFLAGS ''${CXXFLAGS:-}"
        export LDFLAGS="$OCTRA_LDFLAGS ''${LDFLAGS:-}"

        export LIBRARY_PATH="$OCTRA_LIBDIR''${LIBRARY_PATH:+:}$LIBRARY_PATH"
        export LD_LIBRARY_PATH="$OCTRA_LIBDIR''${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"

        c++ -shared -fPIC \
          $OCTRA_CFLAGS \
          -o src/octrad/source/liboctra_wrap.so \
          src/octrad/source/octrad_wrap.cpp \
          $OCTRA_LDFLAGS \
          -Wl,-rpath,"$OCTRA_LIBDIR"

        if [ ! -f src/octrad/dub.json ]; then
          cat > src/octrad/dub.json <<'EOF'
    {
      "name": "octrad",
      "description": "D (SWIG) bindings for the octra library.",
      "license": "Unlicense",
      "version": "0.0.1",
      "targetType": "library",
      "sourcePaths": ["source"],
      "importPaths": ["source"]
    }
    EOF
        fi

        (cd src/octrad && dub build --compiler=ldc2 --build=release)

        runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    pkgDir="$out/share/dub/packages/octrad-${version}"
    mkdir -p "$pkgDir"
    cp -R src/octrad/dub.json src/octrad/source "$pkgDir/"

    # Also ship the built artifact for convenience (name differs by compiler/platform).
    mkdir -p "$out/lib"
    cp -v src/octrad/source/liboctra_wrap.so "$out/lib/"
    for ext in a so dylib lib; do
      if ! find src/octrad -maxdepth 2 -type f -name "*.$ext" -exec cp -v {} "$out/lib/" \; 2>/dev/null; then
        :
      fi
    done

    runHook postInstall
  '';

  meta = with pkgs.lib; {
    description = "D (SWIG) bindings for the octra library.";
    license = licenses.unlicense;
    platforms = platforms.linux;
  };
}
