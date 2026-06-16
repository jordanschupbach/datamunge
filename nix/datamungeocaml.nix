{
  pkgs ? import <nixpkgs> { },
}:

let
  datamunge = import ./datamunge.nix { inherit pkgs; };
  ocamlPkgs = pkgs.ocamlPackages;
in
pkgs.stdenv.mkDerivation rec {
  pname = "datamungeocaml";
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
    datamunge
  ];

  buildPhase = ''
    runHook preBuild

    export DATAMUNGE_PREFIX="${datamunge}"
    export HOME="$TMPDIR"
    export XDG_CACHE_HOME="$TMPDIR/xdg-cache"

    mkdir -p src/datamungeocaml/src

    swig -ocaml -c++ -Iinclude \
      -o src/datamungeocaml/src/datamunge_ocaml_wrap.cxx \
      -outdir src/datamungeocaml/src \
      src/datamungeocaml/swig/datamungeocaml.i

    (cd src/datamungeocaml && dune build)

    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    export DATAMUNGE_PREFIX="${datamunge}"
    export HOME="$TMPDIR"
    export XDG_CACHE_HOME="$TMPDIR/xdg-cache"
    (cd src/datamungeocaml && dune install --prefix "$out")

    runHook postInstall
  '';

  meta = with pkgs.lib; {
    description = "OCaml (SWIG) bindings for the datamunge library.";
    license = licenses.unlicense;
    platforms = platforms.linux;
  };
}
