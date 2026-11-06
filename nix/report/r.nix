{ pkgs, datamunger, ... }:
pkgs.mkShell {
  packages = [ (pkgs.rWrapper.override { packages = [ datamunger ]; }) ];
  # Only the nix-built package: drop library paths set by other shells (build/r/library).
  shellHook = "unset R_LIBS R_LIBS_USER";
}
