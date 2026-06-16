{
  description = "OCTRA";
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
        octra = import ./nix/octra.nix { inherit pkgs; };
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
        pyoctra = import ./nix/pyoctra.nix {
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

        octrajs = import ./nix/octrajs.nix {
          inherit (pkgs)
            lib
            buildNpmPackage
            libxml2
            pkg-config
            ;
        };

        octrar = pkgs.rPackages.buildRPackage {
          name = "octrar";
          src = pkgs.lib.cleanSource ./.;
          buildInputs = [
            pkgs.libxml2
            pkgs.pkg-config
            pkgs.R
          ];
        };

        octratcl = import ./nix/octratcl.nix { pkgs = pkgs; };
        octruby = import ./nix/octruby.nix { pkgs = pkgs; };
        octralua = import ./nix/octralua.nix { pkgs = pkgs; };
        octraocaml = import ./nix/octraocaml.nix { pkgs = pkgs; };
        octraguile = import ./nix/octraguile.nix { pkgs = pkgs; };
        octraoctave = import ./nix/octraoctave.nix { pkgs = pkgs; };
        octrad = import ./nix/octrad.nix { pkgs = pkgs; };

        # }}} Bindings

        gradleWrapped = pkgs.gradle-packages.gradle.wrapped;
        joctraGradleDeps = gradleWrapped.passthru.fetchDeps {
          pkg = pkgs.stdenvNoCC.mkDerivation {
            pname = "joctra";
            version = "0.0.1";
            src = pkgs.emptyDirectory;
            installPhase = "mkdir -p $out";
          };
          data = ./nix/joctra-gradle-deps.json;
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
            name = "octra-format-check";
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
            name = "octra-lint";
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
            name = "octra-cpp-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [
              pkgs.cmake
              pkgs.pkg-config
              pkgs.clang
              pkgs.gtest
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
            name = "octra-python-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [
              (pkgs.python3.withPackages (ps: [
                self.packages.${system}.pyoctra
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
            name = "octra-r-check";
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
            pname = "octra-javascript-check";
            version = "0.0.1";
            src = pkgs.lib.cleanSource ./.;
            npmDepsHash = "sha256-hPHfLevEm7v3hC/NhK1uF+7+UTlT7trPOuD3+f7avHY=";
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
            name = "octra-java-check";
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
            mitmCache = joctraGradleDeps;

            phases = [
              "unpackPhase"
              "configurePhase"
              "buildPhase"
              "installPhase"
            ];
            configurePhase = "runHook preConfigure";
            buildPhase = ''
              cmake -S src/joctra-octra -B src/joctra-octra/build/cmake -DCMAKE_BUILD_TYPE=Release
              cmake --build src/joctra-octra/build/cmake -j $NIX_BUILD_CORES
              export LD_LIBRARY_PATH="$PWD/src/joctra-octra/build/cmake:''${LD_LIBRARY_PATH:-}"
              gradle test --no-configuration-cache
            '';
            installPhase = "mkdir -p $out";
          };

          tcl = pkgs.stdenv.mkDerivation {
            name = "octra-tcl-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [
              pkgs.tcl
              pkgs.tk
            ];
            buildInputs = [
              octratcl
            ];
            phases = [
              "unpackPhase"
              "checkPhase"
              "installPhase"
            ];
            doCheck = true;
            checkPhase = ''
              tclVersionDir="$(${pkgs.tcl}/bin/tclsh <<< 'puts [info library]' | sed -E 's|.*/(tcl[0-9]+\\.[0-9]+).*|\\1|')"
              export TCLLIBPATH="${octratcl}/lib/$tclVersionDir"
              tclsh tests/tcl/test_octra.tcl
            '';
            installPhase = "mkdir -p $out";
          };

          lua = pkgs.stdenv.mkDerivation {
            name = "octra-lua-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [
              lua
            ];
            buildInputs = [
              octralua
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
              export LUA_PATH="${octralua}/share/lua/$luaVersion/?.lua;${octralua}/share/lua/?.lua;./?.lua;;"
              export LUA_CPATH="${octralua}/lib/lua/$luaVersion/?.so;${octralua}/lib/lua/?.so;${octralua}/lib64/lua/$luaVersion/?.so;${octralua}/lib64/lua/?.so;;"
              ${lua}/bin/lua tests/lua/test_octra.lua
            '';
            installPhase = "mkdir -p $out";
          };

          ruby = pkgs.stdenv.mkDerivation {
            name = "octra-ruby-check";
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
              ruby -I tests/ruby -e 'require "test_octra"'
            '';
            installPhase = "mkdir -p $out";
          };

          guile = pkgs.stdenv.mkDerivation {
            name = "octra-guile-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [
              pkgs.guile
            ];
            buildInputs = [
              octraguile
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

              export GUILE_LOAD_PATH="${octraguile}/share/guile/site/$effectiveVersion''${GUILE_LOAD_PATH:+:}$GUILE_LOAD_PATH"
              export GUILE_EXTENSION_PATH="${octraguile}/lib/guile/$effectiveVersion/extensions:${octraguile}/lib64/guile/$effectiveVersion/extensions''${GUILE_EXTENSION_PATH:+:}$GUILE_EXTENSION_PATH"
              ${pkgs.guile}/bin/guile -s tests/guile/test_octra.scm
            '';
            installPhase = "mkdir -p $out";
          };

          octave = pkgs.stdenv.mkDerivation {
            name = "octra-octave-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [
              pkgs.octave
            ];
            buildInputs = [
              octraoctave
            ];
            phases = [
              "unpackPhase"
              "checkPhase"
              "installPhase"
            ];
            doCheck = true;
            checkPhase = ''
              export OCTAVE_PATH="${octraoctave}/share/octave/site/m''${OCTAVE_PATH:+:}$OCTAVE_PATH"
              ${pkgs.octave}/bin/octave -qf --eval 'test("tests/octave/test_octra.m")'
            '';
            installPhase = "mkdir -p $out";
          };

          d = pkgs.stdenv.mkDerivation {
            name = "octra-d-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [
              pkgs.dub
              pkgs.ldc
              pkgs.pkg-config
              pkgs.stdenv.cc
            ];
            buildInputs = [
              octra
              octrad
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

              export PKG_CONFIG_PATH="${octra}/lib/pkgconfig''${PKG_CONFIG_PATH:+:}$PKG_CONFIG_PATH"
              export OCTRA_PREFIX="$(${pkgs.pkg-config}/bin/pkg-config --variable=prefix octra)"
              export OCTRA_LIBDIR="$(${pkgs.pkg-config}/bin/pkg-config --variable=libdir octra)"
              export OCTRA_CFLAGS="$(${pkgs.pkg-config}/bin/pkg-config --cflags octra)"
              export OCTRA_LDFLAGS="$(${pkgs.pkg-config}/bin/pkg-config --libs octra)"
              export CFLAGS="$OCTRA_CFLAGS ''${CFLAGS:-}"
              export CXXFLAGS="$OCTRA_CFLAGS ''${CXXFLAGS:-}"
              export LDFLAGS="$OCTRA_LDFLAGS ''${LDFLAGS:-}"
              export LIBRARY_PATH="$OCTRA_LIBDIR''${LIBRARY_PATH:+:}$LIBRARY_PATH"
              export LD_LIBRARY_PATH="${octrad}/lib:$OCTRA_LIBDIR''${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"

              octradDubPackages="$TMPDIR/dub-packages"
              mkdir -p "$octradDubPackages"
              cp -R "${octrad}/share/dub/packages/octrad-0.0.1" "$octradDubPackages/"
              chmod -R u+w "$octradDubPackages/octrad-0.0.1" || true
              dub add-path "$octradDubPackages" >/dev/null
              dub test --root tests/d --compiler=ldc2 --build=release
            '';
            installPhase = "mkdir -p $out";
          };

          rust = pkgs.rustPlatform.buildRustPackage {
            pname = "octra-rust-check";
            version = "0.0.1";
            src = pkgs.lib.cleanSource ./.;
            cargoLock = {
              lockFile = ./tests/rust/Cargo.lock;
            };
            nativeBuildInputs = [
              pkgs.pkg-config
            ];
            buildInputs = [
              octra
            ];
            doCheck = true;
            buildPhase = ''
              runHook preBuild
              export HOME="$TMPDIR"
              export CARGO_NET_OFFLINE=true
              export PKG_CONFIG_PATH="${octra}/lib/pkgconfig''${PKG_CONFIG_PATH:+:}$PKG_CONFIG_PATH"
              export LD_LIBRARY_PATH="$(${pkgs.pkg-config}/bin/pkg-config --variable=libdir octra)''${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"
              cargo build --manifest-path tests/rust/Cargo.toml --offline --locked --release
              runHook postBuild
            '';
            checkPhase = ''
              runHook preCheck
              export HOME="$TMPDIR"
              export CARGO_NET_OFFLINE=true
              export PKG_CONFIG_PATH="${octra}/lib/pkgconfig''${PKG_CONFIG_PATH:+:}$PKG_CONFIG_PATH"
              export LD_LIBRARY_PATH="$(${pkgs.pkg-config}/bin/pkg-config --variable=libdir octra)''${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"
              cargo test --manifest-path tests/rust/Cargo.toml --offline --locked --release
              runHook postCheck
            '';
            installPhase = "mkdir -p $out";
          };

          csharp =
            let
              nativeOctra = pkgs.stdenv.mkDerivation {
                pname = "octra-csharp-native";
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
                  cmake -S src/octradotnet -B build -DCMAKE_BUILD_TYPE=Release
                  cmake --build build -j $NIX_BUILD_CORES
                '';
                installPhase = ''
                  mkdir -p $out/lib
                  cp -v build/liboctra_csharp.so $out/lib/
                  cp -v build/_deps/octra-build/liboctra.so $out/lib/
                '';
              };
            in
            pkgs.buildDotnetModule {
              name = "octra-csharp-check";
              src = pkgs.lib.cleanSourceWith {
                src = ./.;
                filter = path: type: builtins.baseNameOf path != "dotnet-tools.json";
              };
              dotnet-sdk = pkgs.dotnet-sdk_10;
              nugetDeps = ./nix/nuget-deps.json;
              testProjectFile = "src/octradotnet.tests/octradotnet.tests.csproj";
              runtimeDeps = [ nativeOctra ];
              doCheck = true;
              dontDotnetInstall = true;
              installPhase = "mkdir -p $out";
            };
        };

        packages = {
          inherit
            octra
            pyoctra
            octrajs
            octrar
            octratcl
            octruby
            octralua
            octraocaml
            octraguile
            octraoctave
            octrad
            ;

          rename-octra = pkgs.writeShellApplication {
            name = "rename-octra";
            text = ''exec "${./rename_octra}" "$@"'';
          };
        };

        apps.rename-octra = {
          type = "app";
          program = "${self.packages.${system}.rename-octra}/bin/rename-octra";
          meta = {
            description = "Rename the template project (octra -> <newname>) across files and paths.";
          };
        };

        devShells.default = pkgs.mkShell {
          packages = [

            pkgs.libxml2

            octra
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
            octra
            pkgs.pkg-config
            pkgs.cmake
            pkgs.gnumake
            pkgs.stdenv.cc
            pkgs.swig

            # Language runtimes + bindings for runnable examples
            pyoctra

            octrajs
            pkgs.nodejs

            octrar
            pkgs.R

            octruby
            pkgs.ruby

            pkgs.perl

            phpPackage

            octralua
            lua

            octratcl
            pkgs.tcl

            octraoctave
            pkgs.octave

            octraguile
            pkgs.guile

            # For building/running OCaml + Go examples during export
            pkgs.ocamlPackages.ocaml
            pkgs.ocamlPackages.dune_3
            pkgs.ocamlPackages.findlib
            pkgs.ocamlPackages.alcotest
            pkgs.go
          ];

          shellHook = ''
            export PKG_CONFIG_PATH="${octra}/lib/pkgconfig''${PKG_CONFIG_PATH:+:}$PKG_CONFIG_PATH"

            # Octave: make the installed .m files discoverable.
            export OCTRA_PREFIX="${octra}"
            export OCTAVE_PATH="${octraoctave}/share/octave/site/m''${OCTAVE_PATH:+:}$OCTAVE_PATH"

            # Guile: make the installed module + extension discoverable.
            effectiveVersion="$(pkg-config --variable=effective-version guile-3.0 2>/dev/null || true)"
            if [ -z "$effectiveVersion" ]; then
              effectiveVersion="3.0"
            fi
            export GUILE_LOAD_PATH="${octraguile}/share/guile/site/$effectiveVersion''${GUILE_LOAD_PATH:+:}$GUILE_LOAD_PATH"
            export GUILE_EXTENSION_PATH="${octraguile}/lib/guile/$effectiveVersion/extensions:${octraguile}/lib64/guile/$effectiveVersion/extensions''${GUILE_EXTENSION_PATH:+:}$GUILE_EXTENSION_PATH"
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

            octra
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
            pkgs.cling
            pkgs.doxygen
            pkgs.graphviz
            pkgs.doctest
            pkgs.cmake

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
                pyoctra
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
            octrajs
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
            octrar
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
            octra
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
            export PKG_CONFIG_PATH="${octra}/lib/pkgconfig''${PKG_CONFIG_PATH:+:}$PKG_CONFIG_PATH"
            export LD_LIBRARY_PATH="$(pkg-config --variable=libdir octra)''${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"
            export LIBCLANG_PATH="${pkgs.llvmPackages.libclang.lib}/lib"
          '';
        };

        devShells.d = pkgs.mkShell {
          packages = [
            octra
            octrad
            pkgs.dub
            pkgs.ldc
            pkgs.swig
            pkgs.pkg-config
            pkgs.stdenv.cc
            pkgs.just
          ];

          shellHook = ''
            export PKG_CONFIG_PATH="${octra}/lib/pkgconfig''${PKG_CONFIG_PATH:+:}$PKG_CONFIG_PATH"
            export OCTRA_PREFIX="$(pkg-config --variable=prefix octra)"
            export OCTRA_LIBDIR="$(pkg-config --variable=libdir octra)"
            export OCTRA_CFLAGS="$(pkg-config --cflags octra)"
            export OCTRA_LDFLAGS="$(pkg-config --libs octra)"
            export CFLAGS="$OCTRA_CFLAGS ''${CFLAGS:-}"
            export CXXFLAGS="$OCTRA_CFLAGS ''${CXXFLAGS:-}"
            export LDFLAGS="$OCTRA_LDFLAGS ''${LDFLAGS:-}"
            export LIBRARY_PATH="$OCTRA_LIBDIR''${LIBRARY_PATH:+:}$LIBRARY_PATH"
            export LD_LIBRARY_PATH="${octrad}/lib:$OCTRA_LIBDIR''${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"

            export DUB_HOME="$(pwd)/build/dub"
            mkdir -p "$DUB_HOME"

            octradDubPackages="$(pwd)/build/dub-packages"
            mkdir -p "$octradDubPackages"
            if [ ! -d "$octradDubPackages/octrad-0.0.1" ]; then
              cp -R "${octrad}/share/dub/packages/octrad-0.0.1" "$octradDubPackages/"
              chmod -R u+w "$octradDubPackages/octrad-0.0.1" || true
            fi
            dub add-path "$octradDubPackages" >/dev/null 2>&1 || true
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
            octra
            octratcl
            pkgs.tcl
            pkgs.tk
            pkgs.swig
            pkgs.cmake
            pkgs.pkg-config
            pkgs.just
          ];

          shellHook = ''
            tclVersionDir="$(${pkgs.tcl}/bin/tclsh <<< 'puts [info library]' | sed -E 's|.*/(tcl[0-9]+\\.[0-9]+).*|\\1|')"
            export TCLLIBPATH="${octratcl}/lib/$tclVersionDir''${TCLLIBPATH:+ $TCLLIBPATH}"
          '';
        };

        devShells.ruby = pkgs.mkShell {
          packages = [
            octra
            octruby
            pkgs.ruby
            pkgs.swig
            pkgs.pkg-config
            pkgs.libxml2
            pkgs.just
          ];

          shellHook = ''
            export OCTRA_PREFIX="${octra}"
            export RUBYLIB="${octruby}/lib''${RUBYLIB:+:}$RUBYLIB"
          '';
        };

        devShells.octave = pkgs.mkShell {
          packages = [
            octra
            octraoctave
            pkgs.octave
            pkgs.swig
            pkgs.cmake
            pkgs.pkg-config
            pkgs.just
          ];

          shellHook = ''
            export OCTRA_PREFIX="${octra}"
            export OCTAVE_PATH="${octraoctave}/share/octave/site/m''${OCTAVE_PATH:+:}$OCTAVE_PATH"
          '';
        };

        devShells.lua = pkgs.mkShell {
          packages = [
            octra
            octralua
            lua
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
            export LUA_PATH="${octralua}/share/lua/$luaVersion/?.lua;${octralua}/share/lua/?.lua;./?.lua;;"
            export LUA_CPATH="${octralua}/lib/lua/$luaVersion/?.so;${octralua}/lib/lua/?.so;${octralua}/lib64/lua/$luaVersion/?.so;${octralua}/lib64/lua/?.so;;"
          '';
        };

        devShells.ocaml = pkgs.mkShell {
          packages = [
            octra
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
            export OCTRA_PREFIX="${octra}"
            export PKG_CONFIG_PATH="${octra}/lib/pkgconfig''${PKG_CONFIG_PATH:+:}$PKG_CONFIG_PATH"
            export LD_LIBRARY_PATH="${octra}/lib/octra-0.0.1''${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"
            export CAML_LD_LIBRARY_PATH="${octra}/lib/octra-0.0.1''${CAML_LD_LIBRARY_PATH:+:}$CAML_LD_LIBRARY_PATH"
            export XDG_CACHE_HOME="$(pwd)/build/xdg-cache"
          '';
        };

        devShells.guile = pkgs.mkShell {
          packages = [
            octra
            octraguile
            pkgs.guile
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

            export GUILE_LOAD_PATH="${octraguile}/share/guile/site/$effectiveVersion''${GUILE_LOAD_PATH:+:}$GUILE_LOAD_PATH"
            export GUILE_EXTENSION_PATH="${octraguile}/lib/guile/$effectiveVersion/extensions:${octraguile}/lib64/guile/$effectiveVersion/extensions''${GUILE_EXTENSION_PATH:+:}$GUILE_EXTENSION_PATH"
          '';
        };

        devShells.docs-pages = pkgs.mkShell {
          packages = [
            # Doc export tooling
            pkgs.emacs
            pkgs.direnv
            pkgs.just
            pkgs.jq

            # Core library + pkg-config visibility
            octra
            pkgs.pkg-config
            pkgs.cmake
            pkgs.gnumake
            pkgs.stdenv.cc
            pkgs.swig

            # Language runtimes + bindings for runnable examples
            pyoctra
            pkgs.python3

            octrajs
            pkgs.nodejs

            octrar
            pkgs.R

            octruby
            pkgs.ruby

            pkgs.perl

            phpPackage

            octralua
            lua

            octratcl
            pkgs.tcl

            octraoctave
            pkgs.octave

            octraguile
            pkgs.guile

            # For building/running OCaml + Go examples during export
            pkgs.ocamlPackages.ocaml
            pkgs.ocamlPackages.dune_3
            pkgs.ocamlPackages.findlib
            pkgs.ocamlPackages.alcotest
            pkgs.go
          ];

          shellHook = ''
            export PKG_CONFIG_PATH="${octra}/lib/pkgconfig''${PKG_CONFIG_PATH:+:}$PKG_CONFIG_PATH"

            # Octave: make the installed .m files discoverable.
            export OCTRA_PREFIX="${octra}"
            export OCTAVE_PATH="${octraoctave}/share/octave/site/m''${OCTAVE_PATH:+:}$OCTAVE_PATH"

            # Guile: make the installed module + extension discoverable.
            effectiveVersion="$(pkg-config --variable=effective-version guile-3.0 2>/dev/null || true)"
            if [ -z "$effectiveVersion" ]; then
              effectiveVersion="3.0"
            fi
            export GUILE_LOAD_PATH="${octraguile}/share/guile/site/$effectiveVersion''${GUILE_LOAD_PATH:+:}$GUILE_LOAD_PATH"
            export GUILE_EXTENSION_PATH="${octraguile}/lib/guile/$effectiveVersion/extensions:${octraguile}/lib64/guile/$effectiveVersion/extensions''${GUILE_EXTENSION_PATH:+:}$GUILE_EXTENSION_PATH"
          '';
        };
      }
    );
}
