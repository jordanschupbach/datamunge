# Java (SWIG/JNI) binding built entirely by nix: SWIG generates the Java sources and the JNI
# wrapper, c++ builds libdatamunge_jni against the nix-built core, javac packs the sources into
# a jar. No Gradle, so no dependency cache is needed. Outputs:
#   $out/share/java/jdatamunge.jar   (package js.datamunge.jdatamunge)
#   $out/lib/libdatamunge_jni.so     (loaded by the jar's `datamunge` class)
{
  pkgs,
  lib,
  datamunge,
  swigPkg,
  jdk ? pkgs.jdk,
  ...
}:
pkgs.stdenv.mkDerivation {
  pname = "jdatamunge";
  version = "0.0.1";

  # Only the headers and the SWIG interface feed this build.
  src = import ./source.nix {
    inherit lib;
    only = [
      "include"
      "src/jdatamunge-datamunge/swig"
    ];
  };

  nativeBuildInputs = [
    swigPkg
    jdk
    pkgs.pkg-config
    pkgs.perl
  ];
  buildInputs = [ datamunge ];

  dontConfigure = true;

  buildPhase = ''
    runHook preBuild

    pkg="js.datamunge.jdatamunge"
    gen="$PWD/build/java/src/js/datamunge/jdatamunge"
    mkdir -p "$gen" build/java/classes

    # Same SWIG step as `just prebuild-java`, written to a scratch dir instead of the tree.
    (cd include && swig -c++ -java -package "$pkg" -outdir "$gen" \
      -o ../build/java/datamunge_java_wrap.cpp ../src/jdatamunge-datamunge/swig/jdatamunge.i)

    # The JNI library is libdatamunge_jni (libdatamunge is the core); load it from the module class.
    perl -0777 -pi -e 's/public class datamunge \{/public class datamunge {\n  static { System.loadLibrary("datamunge_jni"); }/s' \
      "$gen/datamunge.java"

    libdir="$(pkg-config --variable=libdir datamunge)"
    c++ -std=c++23 -O2 -shared -fPIC -w \
      -Iinclude $(pkg-config --cflags datamunge) \
      -I${jdk.home}/include -I${jdk.home}/include/linux \
      build/java/datamunge_java_wrap.cpp -o build/java/libdatamunge_jni.so \
      $(pkg-config --libs datamunge) -Wl,-rpath,"$libdir"

    find build/java/src -name '*.java' > build/java/sources.txt
    javac -nowarn -encoding UTF-8 -d build/java/classes @build/java/sources.txt
    jar cf build/java/jdatamunge.jar -C build/java/classes .

    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    install -Dm644 build/java/jdatamunge.jar "$out/share/java/jdatamunge.jar"
    install -Dm755 build/java/libdatamunge_jni.so "$out/lib/libdatamunge_jni.so"
    runHook postInstall
  '';

  meta = {
    description = "Java (SWIG/JNI) bindings for the datamunge library.";
    license = lib.licenses.unlicense;
    platforms = lib.platforms.linux;
  };
}
