{ pkgs, octruby, ... }:
pkgs.mkShell {
  packages = [ pkgs.ruby ];
  shellHook = ''export RUBYLIB="${octruby}/lib"'';
}
