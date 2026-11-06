{ pkgs, pydatamunge, ... }:
pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (ps: [
      pydatamunge
      ps.numpy
      # mep sets the matplotlib backend for blocks with file=, even when they don't plot with it.
      ps.matplotlib
    ]))
  ];
  # Ignore any PYTHONPATH from the caller (a working-tree build would shadow the nix one).
  shellHook = "unset PYTHONPATH";
}
