{
  pkgs,
  bindingPkgs,
  ...
}:
let
  # Built here rather than passed in from flake.nix: only this shell uses it.
  datamungego = import ../datamungego.nix { pkgs = bindingPkgs; };
in
pkgs.mkShell {
  # The same go and stdenv cc as the package, so its prebuilt build cache entries are hits.
  packages = [
    bindingPkgs.go
    bindingPkgs.stdenv.cc
  ];
  # `import "datamunge"` resolves through a workspace using the (read-only) store module, linked
  # against the store's libdatamunge_go. GOCACHE is a writable copy seeded from the package's cache,
  # so the first `go run` doesn't spend minutes compiling the generated datamunge.go.
  shellHook = ''
    # The same CGO_LDFLAGS as the package's warm cache (part of the cache keys); CGO_* inherited
    # from an outer shell (the dev shells point them at build/go/prefix) would miss it.
    unset CGO_CPPFLAGS CGO_CFLAGS CGO_CXXFLAGS GOFLAGS
    export CGO_ENABLED=1 CGO_LDFLAGS="${datamungego.cgoLdflags}"
    reportGoDir="''${XDG_CACHE_HOME:-$HOME/.cache}/datamunge-report-go"
    export GOCACHE="$reportGoDir/gocache"
    mkdir -p "$GOCACHE"
    seeded="$GOCACHE/.seeded-${builtins.baseNameOf datamungego}"
    if [ ! -e "$seeded" ]; then
      cp -r --no-preserve=mode "${datamungego}/share/go-cache/." "$GOCACHE/" && touch "$seeded"
    fi
    printf 'go 1.24\n\nuse %s\n' "${datamungego.module}" > "$reportGoDir/go.work"
    export GOWORK="$reportGoDir/go.work"
    export GOTOOLCHAIN=local GOPROXY=off
    export DATAMUNGE_GO_MODULE="${datamungego.module}"
  '';
}
