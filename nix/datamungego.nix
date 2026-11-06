# Go (SWIG) binding for datamunge, as a Go module directory that a go.work file can `use`.
#
# The SWIG C++ wrapper is compiled here into a static archive, and the generated datamunge.go is
# given a `#cgo LDFLAGS` line linking it (plus the core library, with an rpath). The installed
# module therefore holds no C++ source, so cgo never recompiles the ~1.6 MB wrapper on a cold
# GOCACHE, and a program needs no CGO_* environment to build against it.
#
# Compiling the 1.5 MB generated datamunge.go itself still takes go a couple of minutes, so the
# package is also built here once into a Go build cache, $out/share/go-cache, which
# nix/report/go.nix copies into the user's (writable) GOCACHE. Cache keys depend on the go and cc
# toolchains and the package's store path, so the shell must use the same pkgs.go and stdenv.
#
# Output: $out/share/go/datamunge (go.mod + datamunge.go), $out/lib/libdatamunge_go.a,
#         $out/share/go-cache
{
  pkgs,
  datamunge,
}:
pkgs.stdenv.mkDerivation {
  pname = "datamungego";
  version = datamunge.version;

  # Only the headers and the interface file feed SWIG / the wrapper compile.
  src = import ./source.nix {
    lib = pkgs.lib;
    only = [
      "include"
      "src/godatamunge/swig"
    ];
  };

  nativeBuildInputs = [
    pkgs.swig
    pkgs.pkg-config
    pkgs.go
  ];
  buildInputs = [ datamunge ];

  buildPhase = ''
    runHook preBuild

    mkdir -p gen
    swig -go -c++ -intgosize 64 -Iinclude \
      -o gen/godatamunge_wrap.cxx -oh gen/godatamunge_wrap.h -outdir gen \
      src/godatamunge/swig/godatamunge.i

    c++ -std=c++23 -O2 -fPIC $(pkg-config --cflags datamunge) -Igen \
      -c gen/godatamunge_wrap.cxx -o gen/godatamunge_wrap.o
    ar rcs gen/libdatamunge_go.a gen/godatamunge_wrap.o

    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    libdir="$(pkg-config --variable=libdir datamunge)"
    mod="$out/share/go/datamunge"
    mkdir -p "$mod" "$out/lib"
    cp gen/libdatamunge_go.a "$out/lib/"
    printf 'module datamunge\n\ngo 1.21\n' > "$mod/go.mod"
    # Link flags go first in the cgo preamble (the first "/*" line after the package clause).
    awk -v flags="#cgo LDFLAGS: $out/lib/libdatamunge_go.a -L$libdir -ldatamunge -Wl,-rpath,$libdir -lstdc++ -lm" '
      !done && /^\/\*$/ && seen { print; print flags; done = 1; next }
      /^package datamunge/ { seen = 1 }
      { print }
      END { if (!done) exit 1 }
    ' gen/datamunge.go > "$mod/datamunge.go"

    # Warm a build cache with the package (and the std packages a report typically uses).
    export HOME="$TMPDIR" GOTOOLCHAIN=local GOPROXY=off GOCACHE="$out/share/go-cache"
    mkdir -p warm
    printf 'go 1.21\n\nuse %s\n' "$mod" > warm/go.work
    printf 'package main\n\nimport (\n\t"datamunge"\n\t"fmt"\n\t"math"\n\t"os"\n)\n\nfunc main() {\n\tfmt.Println(datamunge.Runif(0, 1, 1).Size(), math.Pi)\n\tos.Exit(0)\n}\n' > warm/main.go
    (cd warm && GOWORK="$PWD/go.work" go build -o warm main.go && ./warm)
    rm -f "$GOCACHE/testexpire.txt"

    runHook postInstall
  '';

  meta = {
    description = "Go (SWIG) bindings for the datamunge library";
    license = pkgs.lib.licenses.unlicense;
    platforms = pkgs.lib.platforms.unix;
  };
}
