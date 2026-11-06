{
  pkgs,
  datamunge,
  datamunged,
  ...
}:
let
  libdir = "${datamunge}/lib/datamunge-${datamunge.version}";
in
pkgs.mkShell {
  packages = [
    pkgs.ldc
    pkgs.dub
  ];
  # The generated datamunge module (source/) and libdatamunge_wrap, which it dlopens at startup.
  shellHook = ''
    export DATAMUNGED_SOURCE="${datamunged}/share/dub/packages/datamunged-${datamunged.version}/source"
    export LD_LIBRARY_PATH="${datamunged}/lib:${libdir}"
  '';
}
