{
  description = "DATAMUNGE";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    systems.url = "github:nix-systems/default";
    flake-utils = {
      url = "github:numtide/flake-utils";
      inputs.systems.follows = "systems";
    };
    php-from-source.url = "path:./nix/flakes/php";
  };
  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      php-from-source,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        inherit (pkgs) lib;
        has = builtins.hasAttr;
        nodePackages = if has "nodePackages" pkgs then pkgs.nodePackages else { };
        opt = cond: xs: lib.optionals cond xs;
        datamunge = import ./nix/datamunge.nix { inherit pkgs; };
        phpPackage = php-from-source.packages.${system}; # Get the custom PHP package
        lua = pkgs.lua5_4 or (pkgs.lua54 or pkgs.lua);

        # {{{ Bindings

        # Swig javascript next evolution
        swig-jse = pkgs.stdenv.mkDerivation {

          name = "swig-jse";

          src = pkgs.fetchFromGitHub {
            owner = "mmomtchev";
            repo = "swig";
            rev = "aa2e126a14c6456ab0e4b3b7bfd56c11c5a8dc02";
            sha256 = "sha256-E/sfMQQb8DFT8kxQwlqy8/hFI/JXvJDbGp7MvwseJhs=";
          };

          buildInputs = [
            pkgs.autoconf
            pkgs.automake
            pkgs.bison
            pkgs.libtool
            pkgs.pcre2
          ];

          buildPhase = ''
            ./autogen.sh
            ./configure --prefix=$out
            make
          '';

          installPhase = ''
            make install
          '';

        };

        pythonPkgs = pkgs.python3.pkgs;
        pydatamunge = import ./nix/pydatamunge.nix {
          inherit (pkgs)
            lib
            stdenv
            fetchPypi
            python
            libxml2
            pkg-config
            ;
          inherit (pythonPkgs) buildPythonPackage setuptools;
        };

        datamungejs = import ./nix/datamungejs.nix {
          inherit (pkgs)
            lib
            buildNpmPackage
            libxml2
            pkg-config
            ;
        };

        datamunger = pkgs.rPackages.buildRPackage {
          name = "datamunger";
          src = pkgs.lib.cleanSource ./.;
          buildInputs = [
            pkgs.libxml2
            pkgs.pkg-config
            pkgs.R
          ];
        };

        datamungetcl = import ./nix/datamungetcl.nix { pkgs = pkgs; };
        octruby = import ./nix/octruby.nix { pkgs = pkgs; };
        datamungelua = import ./nix/datamungelua.nix { pkgs = pkgs; };
        datamungeocaml = import ./nix/datamungeocaml.nix { pkgs = pkgs; };
        datamungeguile = import ./nix/datamungeguile.nix { pkgs = pkgs; };
        datamungeoctave = import ./nix/datamungeoctave.nix { pkgs = pkgs; };
        datamunged = import ./nix/datamunged.nix { pkgs = pkgs; };

        # }}} Bindings

        gradleWrapped = pkgs.gradle-packages.gradle.wrapped;
        jdatamungeGradleDeps = gradleWrapped.passthru.fetchDeps {
          pkg = pkgs.stdenvNoCC.mkDerivation {
            pname = "jdatamunge";
            version = "0.0.1";
            src = pkgs.emptyDirectory;
            installPhase = "mkdir -p $out";
          };
          data = ./nix/jdatamunge-gradle-deps.json;
        };

        formatCheckTools = [
          pkgs.bash
          (pkgs.python3.withPackages (ps: [ ps.ruff ]))
          (if has "clang-tools" pkgs then pkgs.clang-tools else pkgs.clang)
          pkgs.cmake-format
          pkgs.nixfmt
          pkgs.shfmt
          pkgs.cargo
          pkgs.rustfmt
          pkgs.go
          pkgs.ocamlPackages.ocamlformat
          pkgs.R
          pkgs.rPackages.styler
          pkgs.guile
        ]
        ++ opt (has "prettier" nodePackages) [ nodePackages.prettier ]
        ++ opt (has "stylua" pkgs) [ pkgs.stylua ]
        ++ opt (has "google-java-format" pkgs) [ pkgs.google-java-format ]
        ++ opt (has "ktlint" pkgs) [ pkgs.ktlint ]
        ++ opt (has "dotnet-sdk_10" pkgs) [ pkgs.dotnet-sdk_10 ]
        ++ opt (has "php-cs-fixer" pkgs) [ pkgs.php-cs-fixer ]
        ++ opt (has "tclfmt" pkgs) [ pkgs.tclfmt ]
        ++ opt (has "dfmt" pkgs) [ pkgs.dfmt ]
        ++ opt (has "rufo" pkgs) [ pkgs.rufo ];

        lintTools = [
          pkgs.bash
          pkgs.cmake
          pkgs.gnumake
          pkgs.stdenv.cc
          pkgs.pkg-config
          pkgs.libxml2
          pkgs.shellcheck
          pkgs.statix
          pkgs.deadnix
          pkgs.cppcheck
          (if has "clang-tools" pkgs then pkgs.clang-tools else pkgs.clang)
          (pkgs.python3.withPackages (ps: [ ps.ruff ]))
          pkgs.go
          pkgs.golangci-lint
          pkgs.cargo
          pkgs.clippy
          pkgs.yamllint
          pkgs.actionlint
          phpPackage
          pkgs.R
          pkgs.rPackages.lintr
          pkgs.ruby
          pkgs.rubocop
          pkgs.perl
          pkgs.perlPackages.PerlCritic
        ]
        ++ opt (has "eslint" nodePackages) [ nodePackages.eslint ]
        ++ opt (has "typescript" nodePackages) [ nodePackages.typescript ]
        ++ opt (has "markdownlint-cli2" nodePackages) [ nodePackages.markdownlint-cli2 ]
        ++ opt (has "luacheck" pkgs) [ pkgs.luacheck ]
        ++ opt (has "dscanner" pkgs) [ pkgs.dscanner ];

      in
      {
        checks = {
          format = pkgs.stdenvNoCC.mkDerivation {
            name = "datamunge-format-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = formatCheckTools;
            phases = [
              "unpackPhase"
              "checkPhase"
              "installPhase"
            ];
            doCheck = true;
            checkPhase = ''
              bash ./scripts/format_check.sh
            '';
            installPhase = "mkdir -p $out";
          };

          lint = pkgs.stdenvNoCC.mkDerivation {
            name = "datamunge-lint";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = lintTools;
            phases = [
              "unpackPhase"
              "checkPhase"
              "installPhase"
            ];
            doCheck = true;
            checkPhase = ''
              bash ./scripts/lint.sh
            '';
            installPhase = "mkdir -p $out";
          };

          cpp = pkgs.stdenv.mkDerivation {
            name = "datamunge-cpp-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [
              pkgs.cmake
              pkgs.pkg-config
              pkgs.clang
              pkgs.gtest
              pkgs.arrow-cpp
            ];
            phases = [
              "unpackPhase"
              "buildPhase"
              "checkPhase"
              "installPhase"
            ];
            buildPhase = ''
              cmake -S tests -B build/tests -DCMAKE_BUILD_TYPE=Release
              cmake --build build/tests -j $NIX_BUILD_CORES
            '';
            doCheck = true;
            checkPhase = ''
              ctest --test-dir build/tests --output-on-failure
            '';
            installPhase = "mkdir -p $out";
          };

          python = pkgs.stdenv.mkDerivation {
            name = "datamunge-python-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [
              (pkgs.python3.withPackages (ps: [
                self.packages.${system}.pydatamunge
                ps.pytest
              ]))
            ];
            phases = [
              "unpackPhase"
              "checkPhase"
              "installPhase"
            ];
            doCheck = true;
            checkPhase = ''
              pytest -q tests/python
            '';
            installPhase = "mkdir -p $out";
          };

          r = pkgs.stdenv.mkDerivation {
            name = "datamunge-r-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [
              pkgs.R
              pkgs.rPackages.testthat
            ];
            phases = [
              "unpackPhase"
              "checkPhase"
              "installPhase"
            ];
            doCheck = true;
            checkPhase = ''
              if [ ! -d tests/testthat ]; then
                echo "Expected tests/testthat to exist in source tree." >&2
                echo "PWD: $PWD" >&2
                find . -maxdepth 3 -type d -print >&2
                exit 1
              fi
              R -q -e "testthat::test_local(\".\")"
            '';
            installPhase = "mkdir -p $out";
          };

          javascript = pkgs.buildNpmPackage {
            pname = "datamunge-javascript-check";
            version = "0.0.1";
            src = pkgs.lib.cleanSource ./.;
            npmDepsHash = "sha256-3AVJuVdQXXQ9oYoT0Zh9s0hwQMDTFsXyd90sCBTO4aw=";
            nativeBuildInputs = [
              pkgs.python3
              pkgs.pkg-config
            ];
            buildInputs = [
              pkgs.libxml2
            ];
            env.npm_config_nodedir = "${pkgs.nodejs}";
            buildPhase = ''
              runHook preBuild
              npm run build
              runHook postBuild
            '';
            doCheck = true;
            checkPhase = ''
              runHook preCheck
              npm test
              runHook postCheck
            '';
          };

          java = pkgs.stdenv.mkDerivation {
            name = "datamunge-java-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [
              gradleWrapped
              pkgs.jdk
              pkgs.cmake
              pkgs.pkg-config
            ];
            buildInputs = [
              pkgs.libxml2
            ];

            # Provide a deterministic HTTP replay cache for Gradle via mitm-cache.
            mitmCache = jdatamungeGradleDeps;

            phases = [
              "unpackPhase"
              "configurePhase"
              "buildPhase"
              "installPhase"
            ];
            configurePhase = "runHook preConfigure";
            buildPhase = ''
              cmake -S src/jdatamunge-datamunge -B src/jdatamunge-datamunge/build/cmake -DCMAKE_BUILD_TYPE=Release
              cmake --build src/jdatamunge-datamunge/build/cmake -j $NIX_BUILD_CORES
              export LD_LIBRARY_PATH="$PWD/src/jdatamunge-datamunge/build/cmake:''${LD_LIBRARY_PATH:-}"
              gradle test --no-configuration-cache
            '';
            installPhase = "mkdir -p $out";
          };

          tcl = pkgs.stdenv.mkDerivation {
            name = "datamunge-tcl-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [
              pkgs.tcl
              pkgs.tk
            ];
            buildInputs = [
              datamungetcl
            ];
            phases = [
              "unpackPhase"
              "checkPhase"
              "installPhase"
            ];
            doCheck = true;
            checkPhase = ''
              tclVersionDir="$(${pkgs.tcl}/bin/tclsh <<< 'puts [info library]' | sed -E 's|.*/(tcl[0-9]+\\.[0-9]+).*|\\1|')"
              export TCLLIBPATH="${datamungetcl}/lib/$tclVersionDir"
              tclsh tests/tcl/test_datamunge.tcl
            '';
            installPhase = "mkdir -p $out";
          };

          lua = pkgs.stdenv.mkDerivation {
            name = "datamunge-lua-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [
              lua
            ];
            buildInputs = [
              datamungelua
            ];
            phases = [
              "unpackPhase"
              "checkPhase"
              "installPhase"
            ];
            doCheck = true;
            checkPhase = ''
              luaVersion="$(${lua}/bin/lua -e 'io.write((_VERSION or ""):match("%d+%.%d+") or "")')"
              if [ -z "$luaVersion" ]; then
                echo "Could not determine Lua version from _VERSION" >&2
                ${lua}/bin/lua -e 'print("_VERSION=" .. tostring(_VERSION))' >&2
                exit 1
              fi
              export LUA_PATH="${datamungelua}/share/lua/$luaVersion/?.lua;${datamungelua}/share/lua/?.lua;./?.lua;;"
              export LUA_CPATH="${datamungelua}/lib/lua/$luaVersion/?.so;${datamungelua}/lib/lua/?.so;${datamungelua}/lib64/lua/$luaVersion/?.so;${datamungelua}/lib64/lua/?.so;;"
              ${lua}/bin/lua tests/lua/test_datamunge.lua
            '';
            installPhase = "mkdir -p $out";
          };

          ruby = pkgs.stdenv.mkDerivation {
            name = "datamunge-ruby-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [
              pkgs.ruby
            ];
            buildInputs = [
              octruby
            ];
            phases = [
              "unpackPhase"
              "checkPhase"
              "installPhase"
            ];
            doCheck = true;
            checkPhase = ''
              export RUBYLIB="${octruby}/lib''${RUBYLIB:+:}$RUBYLIB"
              ruby -I tests/ruby -e 'require "test_datamunge"'
            '';
            installPhase = "mkdir -p $out";
          };

          guile = pkgs.stdenv.mkDerivation {
            name = "datamunge-guile-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [
              pkgs.guile
            ];
            buildInputs = [
              datamungeguile
            ];
            phases = [
              "unpackPhase"
              "checkPhase"
              "installPhase"
            ];
            doCheck = true;
            checkPhase = ''
              effectiveVersion="$(${pkgs.pkg-config}/bin/pkg-config --variable=effective-version guile-3.0 2>/dev/null || true)"
              if [ -z "$effectiveVersion" ]; then
                effectiveVersion="3.0"
              fi

              export GUILE_LOAD_PATH="${datamungeguile}/share/guile/site/$effectiveVersion''${GUILE_LOAD_PATH:+:}$GUILE_LOAD_PATH"
              export GUILE_EXTENSION_PATH="${datamungeguile}/lib/guile/$effectiveVersion/extensions:${datamungeguile}/lib64/guile/$effectiveVersion/extensions''${GUILE_EXTENSION_PATH:+:}$GUILE_EXTENSION_PATH"
              ${pkgs.guile}/bin/guile -s tests/guile/test_datamunge.scm
            '';
            installPhase = "mkdir -p $out";
          };

          octave = pkgs.stdenv.mkDerivation {
            name = "datamunge-octave-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [
              pkgs.octave
            ];
            buildInputs = [
              datamungeoctave
            ];
            phases = [
              "unpackPhase"
              "checkPhase"
              "installPhase"
            ];
            doCheck = true;
            checkPhase = ''
              export OCTAVE_PATH="${datamungeoctave}/share/octave/site/m''${OCTAVE_PATH:+:}$OCTAVE_PATH"
              ${pkgs.octave}/bin/octave -qf --eval 'test("tests/octave/test_datamunge.m")'
            '';
            installPhase = "mkdir -p $out";
          };

          d = pkgs.stdenv.mkDerivation {
            name = "datamunge-d-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [
              pkgs.dub
              pkgs.ldc
              pkgs.pkg-config
              pkgs.stdenv.cc
            ];
            buildInputs = [
              datamunge
              datamunged
            ];
            phases = [
              "unpackPhase"
              "checkPhase"
              "installPhase"
            ];
            doCheck = true;
            checkPhase = ''
              export HOME="$TMPDIR"
              export DUB_HOME="$TMPDIR/dub"
              mkdir -p "$DUB_HOME"

              export PKG_CONFIG_PATH="${datamunge}/lib/pkgconfig''${PKG_CONFIG_PATH:+:}$PKG_CONFIG_PATH"
              export DATAMUNGE_PREFIX="$(${pkgs.pkg-config}/bin/pkg-config --variable=prefix datamunge)"
              export DATAMUNGE_LIBDIR="$(${pkgs.pkg-config}/bin/pkg-config --variable=libdir datamunge)"
              export DATAMUNGE_CFLAGS="$(${pkgs.pkg-config}/bin/pkg-config --cflags datamunge)"
              export DATAMUNGE_LDFLAGS="$(${pkgs.pkg-config}/bin/pkg-config --libs datamunge)"
              export CFLAGS="$DATAMUNGE_CFLAGS ''${CFLAGS:-}"
              export CXXFLAGS="$DATAMUNGE_CFLAGS ''${CXXFLAGS:-}"
              export LDFLAGS="$DATAMUNGE_LDFLAGS ''${LDFLAGS:-}"
              export LIBRARY_PATH="$DATAMUNGE_LIBDIR''${LIBRARY_PATH:+:}$LIBRARY_PATH"
              export LD_LIBRARY_PATH="${datamunged}/lib:$DATAMUNGE_LIBDIR''${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"

              datamungedDubPackages="$TMPDIR/dub-packages"
              mkdir -p "$datamungedDubPackages"
              cp -R "${datamunged}/share/dub/packages/datamunged-0.0.1" "$datamungedDubPackages/"
              chmod -R u+w "$datamungedDubPackages/datamunged-0.0.1" || true
              dub add-path "$datamungedDubPackages" >/dev/null
              dub test --root tests/d --compiler=ldc2 --build=release
            '';
            installPhase = "mkdir -p $out";
          };

          rust = pkgs.rustPlatform.buildRustPackage {
            pname = "datamunge-rust-check";
            version = "0.0.1";
            src = pkgs.lib.cleanSource ./.;
            cargoLock = {
              lockFile = ./tests/rust/Cargo.lock;
            };
            nativeBuildInputs = [
              pkgs.pkg-config
            ];
            buildInputs = [
              datamunge
            ];
            doCheck = true;
            buildPhase = ''
              runHook preBuild
              export HOME="$TMPDIR"
              export CARGO_NET_OFFLINE=true
              export PKG_CONFIG_PATH="${datamunge}/lib/pkgconfig''${PKG_CONFIG_PATH:+:}$PKG_CONFIG_PATH"
              export LD_LIBRARY_PATH="$(${pkgs.pkg-config}/bin/pkg-config --variable=libdir datamunge)''${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"
              cargo build --manifest-path tests/rust/Cargo.toml --offline --locked --release
              runHook postBuild
            '';
            checkPhase = ''
              runHook preCheck
              export HOME="$TMPDIR"
              export CARGO_NET_OFFLINE=true
              export PKG_CONFIG_PATH="${datamunge}/lib/pkgconfig''${PKG_CONFIG_PATH:+:}$PKG_CONFIG_PATH"
              export LD_LIBRARY_PATH="$(${pkgs.pkg-config}/bin/pkg-config --variable=libdir datamunge)''${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"
              cargo test --manifest-path tests/rust/Cargo.toml --offline --locked --release
              runHook postCheck
            '';
            installPhase = "mkdir -p $out";
          };

          csharp =
            let
              nativeDatamunge = pkgs.stdenv.mkDerivation {
                pname = "datamunge-csharp-native";
                version = "0.0.1";
                src = pkgs.lib.cleanSource ./.;
                nativeBuildInputs = [
                  pkgs.cmake
                  pkgs.pkg-config
                ];
                buildInputs = [
                  pkgs.libxml2
                ];
                phases = [
                  "unpackPhase"
                  "buildPhase"
                  "installPhase"
                ];
                buildPhase = ''
                  cmake -S src/datamungedotnet -B build -DCMAKE_BUILD_TYPE=Release
                  cmake --build build -j $NIX_BUILD_CORES
                '';
                installPhase = ''
                  mkdir -p $out/lib
                  cp -v build/libdatamunge_csharp.so $out/lib/
                  cp -v build/_deps/datamunge-build/libdatamunge.so $out/lib/
                '';
              };
            in
            pkgs.buildDotnetModule {
              name = "datamunge-csharp-check";
              src = pkgs.lib.cleanSourceWith {
                src = ./.;
                filter = path: type: builtins.baseNameOf path != "dotnet-tools.json";
              };
              dotnet-sdk = pkgs.dotnet-sdk_10;
              nugetDeps = ./nix/nuget-deps.json;
              testProjectFile = "src/datamungedotnet.tests/datamungedotnet.tests.csproj";
              runtimeDeps = [ nativeDatamunge ];
              doCheck = true;
              dontDotnetInstall = true;
              installPhase = "mkdir -p $out";
            };
        };

        packages = {
          inherit
            datamunge
            pydatamunge
            datamungejs
            datamunger
            datamungetcl
            octruby
            datamungelua
            datamungeocaml
            datamungeguile
            datamungeoctave
            datamunged
            ;

          rename-datamunge = pkgs.writeShellApplication {
            name = "rename-datamunge";
            text = ''exec "${./rename_datamunge}" "$@"'';
          };
        };

        apps.rename-datamunge = {
          type = "app";
          program = "${self.packages.${system}.rename-datamunge}/bin/rename-datamunge";
          meta = {
            description = "Rename the template project (datamunge -> <newname>) across files and paths.";
          };
        };

        devShells.default = pkgs.mkShell {
          packages = [

            pkgs.libxml2

            datamunge
            pkgs.lcov
            pkgs.clang
            pkgs.doctest
            pkgs.pkg-config

            (pkgs.python3.withPackages (
              python-pkgs: with python-pkgs; [
                python-lsp-server
              ]
            ))

            # Doc export tooling
            pkgs.emacs
            pkgs.direnv
            pkgs.just
            pkgs.jq

            #
            swig-jse

            # Core library + pkg-config visibility
            datamunge
            pkgs.pkg-config
            pkgs.cmake
            pkgs.gnumake
            pkgs.stdenv.cc
            pkgs.swig

            # Language runtimes + bindings for runnable examples
            pydatamunge

            datamungejs
            pkgs.nodejs

            datamunger
            pkgs.R

            octruby
            pkgs.ruby

            pkgs.perl

            phpPackage

            datamungelua
            lua

            datamungetcl
            pkgs.tcl

            datamungeoctave
            pkgs.octave

            datamungeguile
            pkgs.guile

            # For building/running OCaml + Go examples during export
            pkgs.ocamlPackages.ocaml
            pkgs.ocamlPackages.dune_3
            pkgs.ocamlPackages.findlib
            pkgs.ocamlPackages.alcotest
            pkgs.go
          ];

          shellHook = ''
            export PKG_CONFIG_PATH="${datamunge}/lib/pkgconfig''${PKG_CONFIG_PATH:+:}$PKG_CONFIG_PATH"

            # Octave: make the installed .m files discoverable.
            export DATAMUNGE_PREFIX="${datamunge}"
            export OCTAVE_PATH="${datamungeoctave}/share/octave/site/m''${OCTAVE_PATH:+:}$OCTAVE_PATH"

            # Guile: make the installed module + extension discoverable.
            effectiveVersion="$(pkg-config --variable=effective-version guile-3.0 2>/dev/null || true)"
            if [ -z "$effectiveVersion" ]; then
              effectiveVersion="3.0"
            fi
            export GUILE_LOAD_PATH="${datamungeguile}/share/guile/site/$effectiveVersion''${GUILE_LOAD_PATH:+:}$GUILE_LOAD_PATH"
            export GUILE_EXTENSION_PATH="${datamungeguile}/lib/guile/$effectiveVersion/extensions:${datamungeguile}/lib64/guile/$effectiveVersion/extensions''${GUILE_EXTENSION_PATH:+:}$GUILE_EXTENSION_PATH"
          '';

        };

        devShells.format =
          let
            maybePkg = name: if has name pkgs then [ pkgs.${name} ] else [ ];
            maybeNodePkg = name: if has name nodePackages then [ nodePackages.${name} ] else [ ];
            pythonFormatPkgs = pkgs.python3.withPackages (ps: [
              ps.ruff
            ]);
          in
          pkgs.mkShell {
            packages = [
              pkgs.just
              pkgs.git
              pkgs.python3

              # C/C++
              (if builtins.hasAttr "clang-tools" pkgs then pkgs.clang-tools else pkgs.clang)

              # CMake
              pythonFormatPkgs
              pkgs.cmake-format

              # Nix + shell
              pkgs.nixfmt
              pkgs.shfmt

              # JS/TS/JSON/MD/YAML
              pkgs.nodejs

              # Rust
              pkgs.cargo
              pkgs.rustfmt

              # Go
              pkgs.go

              # OCaml
              pkgs.ocamlPackages.ocamlformat

              # R
              pkgs.R
              pkgs.rPackages.styler

              # Guile (Scheme)
              pkgs.guile

              # Lua
              # PHP
            ]
            ++ maybeNodePkg "prettier"
            ++ maybePkg "stylua"
            ++ maybePkg "perltidy"
            ++ maybePkg "google-java-format"
            ++ maybePkg "ktlint"
            ++ maybePkg "dotnet-sdk_10"
            ++ maybePkg "dotnet-format"
            ++ maybePkg "rufo"
            ++ maybePkg "php-cs-fixer"
            ++ maybePkg "tclfmt"
            ++ maybePkg "dfmt";
          };

        devShells.quality = pkgs.mkShell {
          packages = [
            pkgs.just
            pkgs.git
          ]
          ++ formatCheckTools
          ++ lintTools;
        };

        # NOTE: :( this ... seems to fail a lot
        devShells.cpp = pkgs.mkShell {

          packages = [

            datamunge
            pkgs.just
            pkgs.emacs
            pkgs.direnv
            pkgs.jq
            pkgs.lcov
            pkgs.gtest
            (pkgs.python3.withPackages (
              python-pkgs: with python-pkgs; [
                jinja2
                pygments
              ]
            ))
            pkgs.clang
            pkgs.libxml2
            pkgs.pkg-config
            pkgs.arrow-cpp
            pkgs.cling
            pkgs.doxygen
            pkgs.graphviz
            pkgs.doctest
            pkgs.cmake
            pkgs.xorg.libX11

            # pkgs.nodejs
            # pkgs.prefetch-npm-deps
            # pkgs.nodePackages.npm

          ];
        };

        devShells.java = pkgs.mkShell {
          packages = [
            gradleWrapped
            pkgs.jdk
            pkgs.cmake
            pkgs.just
          ];

          shellHook = ''
            export CMAKE_PATH=${pkgs.cmake}/bin/cmake
          '';

        };

        devShells.python = pkgs.mkShell {
          packages = [
            # pkgs.cmake
            pkgs.pkg-config
            pkgs.libxml2
            pkgs.just
            (pkgs.python3.withPackages (
              python-pkgs: with python-pkgs; [
                pydatamunge
                ipython
                pip
                pytest
                numpy
                matplotlib
                python-lsp-server
              ]
            ))
          ];
        };

        devShells.jsbuild = pkgs.mkShell {
          packages = [
            pkgs.libxml2
            pkgs.pkg-config
            pkgs.python3
            pkgs.nodejs
            pkgs.just
            pkgs.prefetch-npm-deps
            pkgs.nodePackages.npm
          ];
        };

        devShells.javascript = pkgs.mkShell {
          packages = [
            datamungejs
            pkgs.libxml2
            pkgs.pkg-config
            pkgs.python3
            pkgs.nodejs
            pkgs.just
            pkgs.prefetch-npm-deps
            pkgs.nodePackages.npm
          ];
        };

        devShells.r = pkgs.mkShell {
          packages = [
            datamunger
            pkgs.R
            pkgs.rPackages.testthat
            pkgs.pkg-config
            pkgs.libxml2
            pkgs.just
          ];
        };

        devShells.csharp = pkgs.mkShell {
          packages = [
            pkgs.cmake
            pkgs.dotnet-sdk_10
            pkgs.mono
            pkgs.dotnet-repl
            pkgs.just
          ];
        };

        devShells.php = pkgs.mkShell {
          packages = [
            phpPackage
            pkgs.libxml2
            pkgs.pkg-config
            pkgs.cmake
            pkgs.stdenv.cc
            pkgs.just
          ];
        };

        devShells.go = pkgs.mkShell {
          packages = [
            pkgs.go
            pkgs.stdenv.cc
            pkgs.pkg-config
            pkgs.cmake
            pkgs.just
          ];
        };

        devShells.rust = pkgs.mkShell {
          packages = [
            datamunge
            pkgs.rustc
            pkgs.cargo
            pkgs.rustfmt
            pkgs.clippy
            pkgs.rust-bindgen
            pkgs.clang
            pkgs.llvmPackages.libclang
            pkgs.pkg-config
            pkgs.evcxr
            pkgs.just
          ];

          shellHook = ''
            export PKG_CONFIG_PATH="${datamunge}/lib/pkgconfig''${PKG_CONFIG_PATH:+:}$PKG_CONFIG_PATH"
            export LD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)''${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"
            export LIBCLANG_PATH="${pkgs.llvmPackages.libclang.lib}/lib"
          '';
        };

        devShells.d = pkgs.mkShell {
          packages = [
            datamunge
            datamunged
            pkgs.dub
            pkgs.ldc
            pkgs.swig
            pkgs.pkg-config
            pkgs.stdenv.cc
            pkgs.just
          ];

          shellHook = ''
            export PKG_CONFIG_PATH="${datamunge}/lib/pkgconfig''${PKG_CONFIG_PATH:+:}$PKG_CONFIG_PATH"
            export DATAMUNGE_PREFIX="$(pkg-config --variable=prefix datamunge)"
            export DATAMUNGE_LIBDIR="$(pkg-config --variable=libdir datamunge)"
            export DATAMUNGE_CFLAGS="$(pkg-config --cflags datamunge)"
            export DATAMUNGE_LDFLAGS="$(pkg-config --libs datamunge)"
            export CFLAGS="$DATAMUNGE_CFLAGS ''${CFLAGS:-}"
            export CXXFLAGS="$DATAMUNGE_CFLAGS ''${CXXFLAGS:-}"
            export LDFLAGS="$DATAMUNGE_LDFLAGS ''${LDFLAGS:-}"
            export LIBRARY_PATH="$DATAMUNGE_LIBDIR''${LIBRARY_PATH:+:}$LIBRARY_PATH"
            export LD_LIBRARY_PATH="${datamunged}/lib:$DATAMUNGE_LIBDIR''${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"

            export DUB_HOME="$(pwd)/build/dub"
            mkdir -p "$DUB_HOME"

            datamungedDubPackages="$(pwd)/build/dub-packages"
            mkdir -p "$datamungedDubPackages"
            if [ ! -d "$datamungedDubPackages/datamunged-0.0.1" ]; then
              cp -R "${datamunged}/share/dub/packages/datamunged-0.0.1" "$datamungedDubPackages/"
              chmod -R u+w "$datamungedDubPackages/datamunged-0.0.1" || true
            fi
            dub add-path "$datamungedDubPackages" >/dev/null 2>&1 || true
          '';
        };

        devShells.perl = pkgs.mkShell {
          packages = [
            pkgs.perl
            pkgs.perlPackages.ExtUtilsMakeMaker
            pkgs.perlPackages.TestMore
            pkgs.gnumake
            pkgs.stdenv.cc
            pkgs.just
          ];
        };

        devShells.tcl = pkgs.mkShell {
          packages = [
            datamunge
            datamungetcl
            pkgs.tcl
            pkgs.tk
            pkgs.arrow-cpp
            pkgs.swig
            pkgs.cmake
            pkgs.pkg-config
            pkgs.just
          ];

          shellHook = ''
            tclVersionDir="$(${pkgs.tcl}/bin/tclsh <<< 'puts [info library]' | sed -E 's|.*/(tcl[0-9]+\\.[0-9]+).*|\\1|')"
            export TCLLIBPATH="${datamungetcl}/lib/$tclVersionDir''${TCLLIBPATH:+ $TCLLIBPATH}"
          '';
        };

        devShells.ruby = pkgs.mkShell {
          packages = [
            datamunge
            octruby
            pkgs.ruby
            pkgs.swig
            pkgs.pkg-config
            pkgs.libxml2
            pkgs.just
          ];

          shellHook = ''
            export DATAMUNGE_PREFIX="${datamunge}"
            export RUBYLIB="${octruby}/lib''${RUBYLIB:+:}$RUBYLIB"
          '';
        };

        devShells.octave = pkgs.mkShell {
          packages = [
            datamunge
            datamungeoctave
            pkgs.octave
            pkgs.arrow-cpp
            pkgs.swig
            pkgs.cmake
            pkgs.pkg-config
            pkgs.just
          ];

          shellHook = ''
            export DATAMUNGE_PREFIX="${datamunge}"
            export OCTAVE_PATH="${datamungeoctave}/share/octave/site/m''${OCTAVE_PATH:+:}$OCTAVE_PATH"
          '';
        };

        devShells.lua = pkgs.mkShell {
          packages = [
            datamunge
            datamungelua
            lua
            pkgs.arrow-cpp
            pkgs.swig
            pkgs.cmake
            pkgs.pkg-config
            pkgs.just
          ];

          shellHook = ''
            luaVersion="$(${lua}/bin/lua -e 'io.write((_VERSION or ""):match("%d+%.%d+") or "")')"
            if [ -z "$luaVersion" ]; then
              echo "Could not determine Lua version from _VERSION" >&2
              ${lua}/bin/lua -e 'print("_VERSION=" .. tostring(_VERSION))' >&2
              exit 1
            fi
            export LUA_PATH="${datamungelua}/share/lua/$luaVersion/?.lua;${datamungelua}/share/lua/?.lua;./?.lua;;"
            export LUA_CPATH="${datamungelua}/lib/lua/$luaVersion/?.so;${datamungelua}/lib/lua/?.so;${datamungelua}/lib64/lua/$luaVersion/?.so;${datamungelua}/lib64/lua/?.so;;"
          '';
        };

        devShells.ocaml = pkgs.mkShell {
          packages = [
            datamunge
            pkgs.swig
            pkgs.pkg-config
            pkgs.stdenv.cc
            pkgs.gnumake
            pkgs.ocamlPackages.ocaml
            pkgs.ocamlPackages.dune_3
            pkgs.ocamlPackages.findlib
            pkgs.ocamlPackages.utop
            pkgs.ocamlPackages.alcotest
            pkgs.just
          ];

          shellHook = ''
            export DATAMUNGE_PREFIX="${datamunge}"
            export PKG_CONFIG_PATH="${datamunge}/lib/pkgconfig''${PKG_CONFIG_PATH:+:}$PKG_CONFIG_PATH"
            export LD_LIBRARY_PATH="${datamunge}/lib/datamunge-0.0.1''${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"
            export CAML_LD_LIBRARY_PATH="${datamunge}/lib/datamunge-0.0.1''${CAML_LD_LIBRARY_PATH:+:}$CAML_LD_LIBRARY_PATH"
            export XDG_CACHE_HOME="$(pwd)/build/xdg-cache"
          '';
        };

        devShells.guile = pkgs.mkShell {
          packages = [
            datamunge
            datamungeguile
            pkgs.guile
            pkgs.arrow-cpp
            pkgs.swig
            pkgs.cmake
            pkgs.pkg-config
            pkgs.just
          ];

          shellHook = ''
            effectiveVersion="$(pkg-config --variable=effective-version guile-3.0 2>/dev/null || true)"
            if [ -z "$effectiveVersion" ]; then
              effectiveVersion="3.0"
            fi

            export GUILE_LOAD_PATH="${datamungeguile}/share/guile/site/$effectiveVersion''${GUILE_LOAD_PATH:+:}$GUILE_LOAD_PATH"
            export GUILE_EXTENSION_PATH="${datamungeguile}/lib/guile/$effectiveVersion/extensions:${datamungeguile}/lib64/guile/$effectiveVersion/extensions''${GUILE_EXTENSION_PATH:+:}$GUILE_EXTENSION_PATH"
          '';
        };

        devShells.docs-pages = pkgs.mkShell {
          packages = [
            # Doc export tooling
            pkgs.emacs
            pkgs.direnv
            pkgs.just
            pkgs.jq
            pkgs.texliveSmall

            # Core library + pkg-config visibility
            datamunge
            pkgs.pkg-config
            pkgs.cmake
            pkgs.gnumake
            pkgs.stdenv.cc
            pkgs.swig

            # Language runtimes + bindings for runnable examples
            pydatamunge
            pkgs.python3

            datamungejs
            pkgs.nodejs

            datamunger
            pkgs.R

            octruby
            pkgs.ruby

            pkgs.perl

            phpPackage

            datamungelua
            lua

            datamungetcl
            pkgs.tcl

            datamungeoctave
            pkgs.octave

            datamungeguile
            pkgs.guile

            # For building/running OCaml + Go examples during export
            pkgs.ocamlPackages.ocaml
            pkgs.ocamlPackages.dune_3
            pkgs.ocamlPackages.findlib
            pkgs.ocamlPackages.alcotest
            pkgs.go
          ];

          shellHook = ''
            export PKG_CONFIG_PATH="${datamunge}/lib/pkgconfig''${PKG_CONFIG_PATH:+:}$PKG_CONFIG_PATH"

            # Octave: make the installed .m files discoverable.
            export DATAMUNGE_PREFIX="${datamunge}"
            export OCTAVE_PATH="${datamungeoctave}/share/octave/site/m''${OCTAVE_PATH:+:}$OCTAVE_PATH"

            # Guile: make the installed module + extension discoverable.
            effectiveVersion="$(pkg-config --variable=effective-version guile-3.0 2>/dev/null || true)"
            if [ -z "$effectiveVersion" ]; then
              effectiveVersion="3.0"
            fi
            export GUILE_LOAD_PATH="${datamungeguile}/share/guile/site/$effectiveVersion''${GUILE_LOAD_PATH:+:}$GUILE_LOAD_PATH"
            export GUILE_EXTENSION_PATH="${datamungeguile}/lib/guile/$effectiveVersion/extensions:${datamungeguile}/lib64/guile/$effectiveVersion/extensions''${GUILE_EXTENSION_PATH:+:}$GUILE_EXTENSION_PATH"
          '';
        };
      }
    );
}
