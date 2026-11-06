{
  pkgs,
  lib,
  datamunge,
  swigPkg,
  ...
}:
let
  jdk = pkgs.jdk;
  jdatamunge = import ../datamungejava.nix {
    inherit
      pkgs
      lib
      datamunge
      swigPkg
      jdk
      ;
  };
in
pkgs.mkShell {
  packages = [ jdk ];
  # javac and java both read CLASSPATH. The JNI library is found through LD_LIBRARY_PATH (part of
  # java.library.path on Linux); JDK_JAVA_OPTIONS is unset because a value inherited from the
  # working-tree .#java shell would point java.library.path at an unbuilt cmake dir instead.
  shellHook = ''
    unset JDK_JAVA_OPTIONS
    export CLASSPATH="${jdatamunge}/share/java/jdatamunge.jar''${CLASSPATH:+:$CLASSPATH}"
    export LD_LIBRARY_PATH="${jdatamunge}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  '';
  passthru = { inherit jdatamunge; };
}
