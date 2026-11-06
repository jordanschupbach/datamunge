{ pkgs, datamungeoctave, ... }:
pkgs.mkShell {
  packages = [ pkgs.octave ];
  shellHook = ''export OCTAVE_PATH="${datamungeoctave}/share/octave/site/m"'';
}
