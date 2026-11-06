{ pkgs, datamungeguile, ... }:
let
  v = pkgs.guile.effectiveVersion or "3.0";
in
pkgs.mkShell {
  packages = [ pkgs.guile ];
  shellHook = ''
    export GUILE_LOAD_PATH="${datamungeguile}/share/guile/site/${v}"
    export GUILE_EXTENSION_PATH="${datamungeguile}/lib/guile/${v}/extensions:${datamungeguile}/lib64/guile/${v}/extensions"
  '';
}
