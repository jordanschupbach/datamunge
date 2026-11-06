# Perl (SWIG) binding for datamunge, installed as a standard Perl module:
#   $out/lib/perl5/site_perl/Datamunge.pm
#   $out/lib/perl5/site_perl/auto/Datamunge/Datamunge.so
# Put "$out/lib/perl5/site_perl" on PERL5LIB (nix/report/perl.nix does).
{
  pkgs ? import <nixpkgs> { },
  datamunge ? import ./datamunge.nix { inherit pkgs; },
  swig ? pkgs.swig,
  perl ? pkgs.perl,
}:

pkgs.stdenv.mkDerivation {
  pname = "datamungeperl";
  version = "0.0.1";

  # Only the headers SWIG reads and the Perl binding dir: other bindings' edits never rebuild this.
  src = import ./source.nix {
    lib = pkgs.lib;
    only = [
      "include"
      "src/perldatamunge"
    ];
  };

  nativeBuildInputs = [
    swig
    perl
    pkgs.pkg-config
    pkgs.gnumake
  ];

  buildInputs = [ datamunge ];

  buildPhase = ''
    runHook preBuild

    cd src/perldatamunge
    # Regenerate the wrapper from the .i file (the checked-in copy may be stale).
    rm -f Datamunge_wrap.cxx Datamunge_wrap.h lib/Datamunge.pm Makefile MYMETA.* pm_to_blib Datamunge.bs
    mkdir -p lib
    swig -perl5 -c++ -I../../include \
      -o Datamunge_wrap.cxx -oh Datamunge_wrap.h -outdir lib \
      swig/perldatamunge.i

    export DATAMUNGE_INCLUDE_DIR="$(pkg-config --variable=includedir datamunge)"
    export DATAMUNGE_LIB_DIR="$(pkg-config --variable=libdir datamunge)"
    export CXX=c++
    perl Makefile.PL INSTALL_BASE="$out" INSTALLSITEARCH="$out/lib/perl5/site_perl" \
      INSTALLSITELIB="$out/lib/perl5/site_perl"
    make -j $NIX_BUILD_CORES

    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    make install
    rm -rf "$out/man" "$out/lib/perl5/site_perl/perllocal.pod" "$out/lib/perl5/x86_64-linux"* 2>/dev/null || true
    runHook postInstall
  '';

  meta = with pkgs.lib; {
    description = "Perl (SWIG) bindings for the datamunge library.";
    license = licenses.unlicense;
    platforms = platforms.unix;
  };
}
