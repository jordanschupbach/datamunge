args@{ pkgs, phpPackage, ... }:
let
  datamungephp = import ../datamungephp.nix args;
in
pkgs.mkShell {
  packages = [ phpPackage ];
  # Leading ':' keeps PHP's compiled-in scan dir alongside the extension's conf.d.
  shellHook = ''export PHP_INI_SCAN_DIR=":${datamungephp}/etc/php/conf.d"'';
}
