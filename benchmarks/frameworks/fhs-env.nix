# FHS-style environment for running pip-installed PyTorch / TensorFlow wheels
# on NixOS (no nix-ld on this machine). Build with:
#
#   nix build --impure --expr 'import ./benchmarks/frameworks/fhs-env.nix {}' \
#       -o build/fw-env
#
# then run commands inside it with
#
#   build/fw-env/bin/fw-env -c 'python3 -m venv build/fw-venv && ...'
#
# The host NVIDIA driver (libcuda.so.1) lives under /run/opengl-driver/lib on
# NixOS; buildFHSEnv bind-mounts it, and `profile` puts it on LD_LIBRARY_PATH so
# the bundled CUDA runtime in the torch/tensorflow wheels can dlopen it.
{ pkgs ? import (builtins.getFlake "nixpkgs") {
    system = "x86_64-linux";
    config.allowUnfree = true;
  }
}:
pkgs.buildFHSEnv {
  name = "fw-env";
  targetPkgs = p: with p; [
    python313
    zlib
    stdenv.cc.cc.lib # libstdc++ / libgcc_s for the manylinux wheels
    libGL
    glib
    ncurses
    expat
    libxcrypt
    openssl
    util-linux
    which
  ];
  profile = ''
    export LD_LIBRARY_PATH=/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    # Keep pip's downloads and temp files off the small root filesystem.
    export PIP_CACHE_DIR=''${PIP_CACHE_DIR:-/mnt/projects/datamunge/build/pip-cache}
    export TMPDIR=''${TMPDIR:-/mnt/projects/datamunge/build/tmp}
    export PIP_DISABLE_PIP_VERSION_CHECK=1
  '';
  runScript = "bash";
}
