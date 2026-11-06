# PHP (SWIG -php7) extension for datamunge, linked against the nix-built core package.
#
# Installs the extension plus a conf.d ini that loads it, so a shell only needs
#   PHP_INI_SCAN_DIR=":${datamungephp}/etc/php/conf.d"
# (the leading ':' keeps PHP's own scan dir). The local workflow (`just build-php`) builds the
# same wrapper through src/datamungePHP/CMakeLists.txt, which compiles the core from source via
# CPM; here we link the already-built `datamunge` package instead, so this rebuilds only when the
# core, the SWIG interface or PHP itself changes (sources come from ./source.nix).
{
  pkgs,
  datamunge,
  swigPkg,
  phpPackage,
  phpDev ? [ ],
  ...
}:
pkgs.stdenv.mkDerivation {
  pname = "datamungephp";
  version = "0.0.1";

  src = import ./source.nix { lib = pkgs.lib; };

  nativeBuildInputs = [
    pkgs.cmake
    pkgs.pkg-config
    swigPkg
    phpPackage
  ]
  ++ phpDev;

  buildInputs = [
    datamunge
    pkgs.arrow-cpp
  ];

  dontUseCmakeConfigure = true;

  configurePhase = ''
    runHook preConfigure

    mkdir -p build/gen
    # Same invocation as `just prebuild-php` (run from include/ so datamunge/... resolves).
    (cd include && swig -c++ -php7 -outdir ../build/gen \
      -o ../build/gen/datamunge_php_wrap.cpp -oh ../build/gen/datamunge_php_wrap.h \
      ../src/datamungePHP/swig/datamungePHP.i)

    phpIncludes="$(php-config --includes | sed 's/-I//g')"
    cat > build/gen/CMakeLists.txt <<EOF
    cmake_minimum_required(VERSION 3.16)
    project(datamungephp LANGUAGES CXX)
    find_package(datamunge CONFIG REQUIRED)
    add_library(datamunge MODULE datamunge_php_wrap.cpp)
    set_target_properties(datamunge PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON
      POSITION_INDEPENDENT_CODE ON PREFIX "" OUTPUT_NAME datamunge)
    target_include_directories(datamunge PRIVATE \''${CMAKE_CURRENT_LIST_DIR} $phpIncludes)
    target_link_libraries(datamunge PRIVATE datamunge::datamunge)
    if(APPLE)
      target_link_options(datamunge PRIVATE -undefined dynamic_lookup)
    endif()
    EOF

    cmake -S build/gen -B build/ext -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="${datamunge};${pkgs.arrow-cpp}"

    runHook postConfigure
  '';

  buildPhase = ''
    runHook preBuild
    cmake --build build/ext -j $NIX_BUILD_CORES
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    extDir="$out/lib/php/extensions"
    mkdir -p "$extDir" "$out/etc/php/conf.d"
    cp -v build/ext/datamunge.so "$extDir/datamunge.so"

    # Find libdatamunge at run time without LD_LIBRARY_PATH.
    datamungeLibDir="$(dirname "$(find "${datamunge}" \( -name 'libdatamunge.so' -o -name 'libdatamunge.dylib' \) -print -quit)")"
    ${pkgs.lib.optionalString pkgs.stdenv.isLinux ''
      oldRpath="$(${pkgs.patchelf}/bin/patchelf --print-rpath "$extDir/datamunge.so" || true)"
      ${pkgs.patchelf}/bin/patchelf --set-rpath "$datamungeLibDir''${oldRpath:+:$oldRpath}" "$extDir/datamunge.so"
    ''}

    printf 'extension=%s\n' "$extDir/datamunge.so" > "$out/etc/php/conf.d/datamunge.ini"

    runHook postInstall
  '';

  meta = with pkgs.lib; {
    description = "PHP (SWIG) extension for the datamunge library.";
    license = licenses.unlicense;
    platforms = platforms.unix;
  };
}
