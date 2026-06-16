{
  pkgs ? import <nixpkgs> { },
}:

let
  octra = import ./octra.nix { inherit pkgs; };
  ocamlPkgs = pkgs.ocamlPackages;
in
pkgs.stdenv.mkDerivation rec {
  pname = "octraocaml";
  version = "0.0.1";

  src = pkgs.lib.cleanSource ../.;

  nativeBuildInputs = [
    pkgs.swig
    pkgs.pkg-config
    ocamlPkgs.ocaml
    ocamlPkgs.dune_3
    ocamlPkgs.findlib
  ];

  buildInputs = [
    octra
  ];

  buildPhase = ''
    runHook preBuild

    export OCTRA_PREFIX="${octra}"
    export HOME="$TMPDIR"
    export XDG_CACHE_HOME="$TMPDIR/xdg-cache"

    mkdir -p src/octraocaml/src

    swig -ocaml -c++ -Iinclude \
      -o src/octraocaml/src/octra_ocaml_wrap.cxx \
      -outdir src/octraocaml/src \
      src/octraocaml/swig/octraocaml.i

    (cd src/octraocaml && dune build)

    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    export OCTRA_PREFIX="${octra}"
    export HOME="$TMPDIR"
    export XDG_CACHE_HOME="$TMPDIR/xdg-cache"
    (cd src/octraocaml && dune install --prefix "$out")

    runHook postInstall
  '';

  meta = with pkgs.lib; {
    description = "OCaml (SWIG) bindings for the octra library.";
    license = licenses.unlicense;
    platforms = platforms.linux;
  };
}
