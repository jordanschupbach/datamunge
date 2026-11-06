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
    swig = bindingPkgs.swig;
  };
in
pkgs.mkShell {
  packages = [ pkgs.perl ];
  shellHook = ''export PERL5LIB="${datamungeperl}/lib/perl5/site_perl''${PERL5LIB:+:}$PERL5LIB"'';
}
