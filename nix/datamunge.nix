{
  pkgs ? import <nixpkgs> { config.allowUnfree = true; },
  enableCuda ? false,
  enableMujoco ? false,
  enableCublas ? false,
  cudaToolkit ? null,
}:

let
  datamunge = pkgs.stdenv.mkDerivation rec {
    pname = "datamunge";
    version = "0.0.1";

    # Only what the core's CMake build reads; binding and tooling edits never rebuild the core.
    src = import ./source.nix {
      lib = pkgs.lib;
      only = [
        "CMakeLists.txt"
        "cmake"
        "include"
        "src/datamunge"
        "datamunge.pc.in"
        "third_party_licenses"
        "tools/prepare_example_data.cpp"
      ];
    };

    nativeBuildInputs = [
      pkgs.cmake
      pkgs.pkg-config
    ];

    buildInputs = [
      pkgs.clang
      pkgs.arrow-cpp
      pkgs.libarchive
      pkgs.openssl
      pkgs.nlohmann_json
      pkgs.hdf5-cpp
      pkgs.xorg.libX11.dev
    ]
    # No BLAS/LAPACK: datamunge::nn ships its own GEMM/GEMV/level-1 kernels
    # (src/datamunge/nn/blas_kernels.hpp, AVX-512/AVX2/baseline chosen at run
    # time) and built-in decompositions. DATAMUNGE_USE_BLAS / DATAMUNGE_USE_LAPACK
    # exist only to compare against an external library.
    ++ pkgs.lib.optionals enableCublas [ cudaToolkit ]
    ++ pkgs.lib.optional enableMujoco pkgs.mujoco;

    cmakeFlags = [
      "-DCMAKE_CXX_COMPILER=clang++"
      "-DDATAMUNGE_ENABLE_MUJOCO=${if enableMujoco then "ON" else "OFF"}"
      "-DCMAKE_INSTALL_LIBDIR=lib"
      "-DDATAMUNGE_ENABLE_CUDA=${if enableCuda then "ON" else "OFF"}"
      "-DDATAMUNGE_ENABLE_CUBLAS=${if enableCublas then "ON" else "OFF"}"
    ]
    ++ pkgs.lib.optional enableCublas "-DCUDAToolkit_ROOT=${cudaToolkit}";

    configurePhase = ''
      cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$out ${pkgs.lib.escapeShellArgs cmakeFlags}
    '';

    buildPhase = ''
      cmake --build build -j $NIX_BUILD_CORES -- --jobserver-style=pipe
    '';

    installPhase = ''
      cmake --install build --prefix $out
    '';
  };
in
datamunge
