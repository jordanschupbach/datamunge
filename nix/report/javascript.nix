{ pkgs, datamungejs, ... }:
pkgs.mkShell {
  packages = [ pkgs.nodejs ];
  # buildNpmPackage installs the package as lib/node_modules/@datamunge/datamungejs.
  shellHook = ''export NODE_PATH="${datamungejs}/lib/node_modules"'';
}
