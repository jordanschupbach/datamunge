{ pkgs, datamunge, ... }:
pkgs.mkShell {
  packages = [
    datamunge
    pkgs.pkg-config
    pkgs.stdenv.cc
  ];
  shellHook = ''
    export PKG_CONFIG_PATH="${datamunge}/lib/pkgconfig''${PKG_CONFIG_PATH:+:}$PKG_CONFIG_PATH"
  '';
}
