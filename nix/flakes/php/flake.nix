{
  description = "PHP from source";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      systems = [ "x86_64-linux" ];
      packages = builtins.listToAttrs (
        map (system: {
          name = system;
          value = nixpkgs.legacyPackages.${system}.stdenv.mkDerivation {
            pname = "php";
            version = "8.5.0";
            src = nixpkgs.legacyPackages.${system}.fetchurl {
              url = "https://github.com/php/php-src/archive/refs/tags/php-8.5.0.tar.gz";
              sha256 = "sha256-xeRGSW80xC01OtAG6TVy6/zKTcSFNftCgnD+Cv/OOLk=";
            };
            unpackPhase = ''
              mkdir -p $out
              tar -xzf $src -C $out --strip-components=1
            '';
            phases = [
              "unpackPhase"
              "preConfigure"
              "configurePhase"
              "buildPhase"
              "installPhase"
            ];
            preConfigure = ''
              # ls ${nixpkgs.legacyPackages.${system}.libxml2}
              ls ${nixpkgs.legacyPackages.${system}.readline}/lib/
              export LIBXML_CFLAGS=$(pkg-config --cflags libxml-2.0)
              export LIBXML_LIBS=$(pkg-config --libs libxml-2.0)
              export SQLITE_CFLAGS=$(pkg-config --cflags sqlite3)
              export SQLITE_LIBS=$(pkg-config --libs sqlite3)
              export PKG_CONFIG_PATH=${nixpkgs.legacyPackages.${system}.readline}/lib/pkgconfig
              # export PKG_CONFIG_PATH=${nixpkgs.legacyPackages.${system}.readline}/lib/pkgconfig
              # export PKG_CONFIG_PATH=${nixpkgs.legacyPackages.${system}.readline}/lib/pkgconfig:${
                nixpkgs.legacyPackages.${system}.libxml2
              }/lib/pkgconfig
              cd $out
              ./buildconf --force
            '';

            configurePhase = ''
              ./configure \
                --prefix=$out \
                --with-readline=${nixpkgs.legacyPackages.${system}.readline.dev}
                # --with-config-file-path=$out/etc \
                # --with-config-file-scan-dir=$out/etc/php.d \
                # --with-readline \ 
                # --enable-cli \
                # --disable-cgi \
                # --disable-fpm \
                # --without-pear
            '';

            buildPhase = ''
              make -j30 # TODO: detect number of cores
            '';

            installPhase = ''
              make install
            '';

            nativeBuildInputs = [
              nixpkgs.legacyPackages.${system}.pkg-config
              nixpkgs.legacyPackages.${system}.autoconf
              nixpkgs.legacyPackages.${system}.automake
              nixpkgs.legacyPackages.${system}.libxml2
              nixpkgs.legacyPackages.${system}.libtool
              nixpkgs.legacyPackages.${system}.bison
              nixpkgs.legacyPackages.${system}.re2c
              nixpkgs.legacyPackages.${system}.sqlite
              nixpkgs.legacyPackages.${system}.readline.dev
              # nixpkgs.legacyPackages.${system}.readline.dev
            ];

            buildInputs = [
              # nixpkgs.legacyPackages.${system}.pkg-config
              nixpkgs.legacyPackages.${system}.autoconf
              nixpkgs.legacyPackages.${system}.libxml2
              nixpkgs.legacyPackages.${system}.bzip2
              nixpkgs.legacyPackages.${system}.bison
              nixpkgs.legacyPackages.${system}.re2c
              nixpkgs.legacyPackages.${system}.sqlite
              nixpkgs.legacyPackages.${system}.readline.dev
              # nixpkgs.legacyPackages.${system}.readline.dev
            ];

          };
        }) systems
      );
    in
    {
      packages = packages;
      defaultPackage.x86_64-linux = packages.x86_64-linux;
    };
}
