{
  pkgs ? import <nixpkgs> { config.allowUnfree = true; },
  enableCuda ? false,
  enableCublas ? false,
  cudaToolkit ? null,
}:

let
  datamunge = pkgs.stdenv.mkDerivation rec {
    pname = "datamunge";
    version = "0.0.1";

    src = import ./source.nix { lib = pkgs.lib; };

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
    # CBLAS for datamunge::nn GEMM (macOS uses the Accelerate framework instead).
    ++ pkgs.lib.optionals pkgs.stdenv.isLinux [ pkgs.openblas ]
    ++ pkgs.lib.optionals enableCublas [ cudaToolkit ];

    cmakeFlags = [
      "-DCMAKE_CXX_COMPILER=clang++"
      "-DCMAKE_INSTALL_LIBDIR=lib"
      "-DDATAMUNGE_ENABLE_CUDA=${if enableCuda then "ON" else "OFF"}"
      "-DDATAMUNGE_ENABLE_CUBLAS=${if enableCublas then "ON" else "OFF"}"
    ]
    ++ pkgs.lib.optional enableCublas "-DCUDAToolkit_ROOT=${cudaToolkit}";

    configurePhase = ''
      cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$out ${pkgs.lib.escapeShellArgs cmakeFlags}
    '';

    buildPhase = ''
      cmake --build build -j $NIX_BUILD_CORES
    '';

    installPhase = ''
      cmake --install build --prefix $out
    '';
  };
in
datamunge
