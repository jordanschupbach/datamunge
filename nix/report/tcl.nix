{ pkgs, datamungetcl, ... }:
pkgs.mkShell {
  packages = [ pkgs.tcl ];
  # The package installs under lib/tcl<major.minor>.
  shellHook = ''export TCLLIBPATH="${datamungetcl}/lib/tcl${pkgs.tcl.release}"'';
}
