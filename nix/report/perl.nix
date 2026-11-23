{
  pkgs,
  datamunge,
  bindingPkgs,
  ...
}:
let
  datamungeperl = import ../datamungeperl.nix {
    pkgs = bindingPkgs;
    inherit datamunge;
  };
in
pkgs.mkShell {
  packages = [ pkgs.perl ];
  shellHook = ''export PERL5LIB="${datamungeperl}/lib/perl5''${PERL5LIB:+:}$PERL5LIB"'';
}
