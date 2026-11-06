{
  pkgs,
  bindingPkgs,
  datamunge,
  ...
}:
let
  # Built here rather than passed in from flake.nix: only this shell uses it.
  datamungego = import ../datamungego.nix {
    pkgs = bindingPkgs;
    inherit datamunge;
  };
in
pkgs.mkShell {
  # The same go and stdenv cc as the package, so its prebuilt build cache entries are hits.
  packages = [
    bindingPkgs.go
    pkgs.pkg-config
  ];
  # `import "datamunge"` resolves through a workspace using the (read-only) store module, which
  # already carries its link flags. GOCACHE is a writable copy seeded from the package's cache,
  # so the first `go run` doesn't spend minutes compiling the generated datamunge.go.
  shellHook = ''
    reportGoDir="''${XDG_CACHE_HOME:-$HOME/.cache}/datamunge-report-go"
    export GOCACHE="$reportGoDir/gocache"
    mkdir -p "$GOCACHE"
    seeded="$GOCACHE/.seeded-${builtins.baseNameOf datamungego}"
    if [ ! -e "$seeded" ]; then
      cp -r --no-preserve=mode "${datamungego}/share/go-cache/." "$GOCACHE/" && touch "$seeded"
    fi
    printf 'go 1.21\n\nuse %s\n' "${datamungego}/share/go/datamunge" > "$reportGoDir/go.work"
    export GOWORK="$reportGoDir/go.work"
    export GOTOOLCHAIN=local
    export DATAMUNGE_GO_MODULE="${datamungego}/share/go/datamunge"
  '';
}
