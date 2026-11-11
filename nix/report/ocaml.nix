{
  pkgs,
  datamunge,
  datamungeocaml,
  ...
}:
pkgs.mkShell {
  packages = [
    pkgs.ocamlPackages.ocaml
    pkgs.ocamlPackages.findlib
  ];
  # `#use "topfind";; #require "datamungeocaml";;` finds the dune-installed library here.
  shellHook = ''
    export OCAMLPATH="${datamungeocaml}/lib"
    export OCAMLTOP_INCLUDE_PATH="$(dirname "$(ocamlfind query findlib)")"
    export CAML_LD_LIBRARY_PATH="${datamungeocaml}/lib/stublibs:${datamunge}/lib/datamunge-${datamunge.version}"
    export LD_LIBRARY_PATH="${datamunge}/lib/datamunge-${datamunge.version}"
  '';
}
