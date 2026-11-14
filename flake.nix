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
    # SWIG fork used for every binding's wrapper generation (in place of nixpkgs' swig).
    # To build against a local checkout instead, see DATAMUNGE_SWIG_SRC in .envrc / justfile.
    swig = {
      url = "github:jordanschupbach/swig";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    # crepl tracking its main branch on GitHub. Bump with `nix flake update crepl`.
    # To test an uncommitted local checkout, pass
    # `--override-input crepl path:/mnt/projects/crepl`.
    crepl = {
      url = "github:jordanschupbach/crepl";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    # Rosetta generates the Python extension (pydatamunge._datamunge) from C++ headers.
    # Pinned from GitHub so the flake works anywhere (including as an input of another flake).
    # When a checkout sits next to this one (../rosetta), direnv and the justfile build against
    # it instead (--override-input rosetta git+file://..., uncommitted edits included); point
    # DATAMUNGE_ROSETTA_SRC at another checkout, or set it to "none" to use the pin. Bump the
    # pin with `nix flake update rosetta` after pushing rosetta.
    rosetta = {
      url = "github:jordanschupbach/rosetta";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    # mep (editor and `mep-mepml`), from its main branch: the docs build runs and
    # exports the mepml reports with it. `just update-mep` bumps the lock to the
    # latest main. Only its package is used; its own nixpkgs pin is kept (that is
    # what mep is built and tested against). mep's flake has a datamunge input of
    # its own (for its dev shell, which nothing here evaluates): it follows this
    # flake ("" is the root) rather than locking a second, older datamunge.
    mep = {
      url = "github:jordanschupbach/mep";
      inputs.datamunge.follows = "";
    };
  };
  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      php-from-source,
      swig,
      crepl,
      rosetta,
      mep,
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
        # `pkgs` with nixpkgs' swig replaced by the fork, for the binding packages in nix/ (which
        # take swig from `pkgs`). Only used there, so nothing else in nixpkgs gets rebuilt.
        swigPkg = swig.packages.${system}.swig;
        creplPkg = crepl.packages.${system}.default;
        rosettaPkg = rosetta.packages.${system}.default;
        mepPkg = mep.packages.${system}.default;
        bindingPkgs = pkgs // {
          swig = swigPkg;
        };
        datamunge = import ./nix/datamunge.nix { inherit pkgs; };
        # CUDA is opt-in and Linux-only; ordinary packages/shells never evaluate
        # unfree SDKs. The running host supplies the NVIDIA driver and devices.
        cudaSupported = builtins.elem system [
          "x86_64-linux"
          "aarch64-linux"
        ];
        cudaPkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        # Keep a coherent release and exclude driver compatibility shims: the
        # real host driver supplies libcuda. No cuDNN/NCCL or graphics SDK is needed.
        cuda = cudaPkgs.cudaPackages_12_8.overrideScope (_final: _prev: { cuda_compat = null; });
        cudaToolkit = cudaPkgs.symlinkJoin {
          name = "datamunge-cuda-toolkit-12.8";
          paths = [
            cuda.cuda_nvcc
            cuda.cuda_cudart
            cuda.cuda_cccl
            (lib.getOutput "include" cuda.libcublas)
            (lib.getLib cuda.libcublas)
            (lib.getOutput "include" cuda.cuda_nvrtc)
            (lib.getLib cuda.cuda_nvrtc)
          ];
        };
        cudaHook = ''
          export CUDAToolkit_ROOT="${cudaToolkit}"
          export CUDA_PATH="${cudaToolkit}"
          export LD_LIBRARY_PATH="/run/opengl-driver/lib:${cudaToolkit}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        '';
        # The custom php-from-source flake only targets x86_64-linux; fall back to the
        # nixpkgs PHP (which supports darwin) on any host it doesn't provide.
        phpPackage =
          if has system php-from-source.packages then php-from-source.packages.${system} else pkgs.php;
        lua = pkgs.lua5_4 or (pkgs.lua54 or pkgs.lua);
        # SBCL plus the ASDF systems the generated Common Lisp system loads (cffi, trivial-garbage).
        # sbcl.withPackages' wrapper fails nixpkgs' makeWrapper check, so shells export the source
        # registry it would set; the store systems keep their prebuilt fasls in place.
        lispSystems =
          let
            closure = libs: lib.unique (lib.concatMap (p: [ p ] ++ closure (p.lispLibs or [ ])) libs);
          in
          closure [
            pkgs.sbclPackages.cffi
            pkgs.sbclPackages.trivial-garbage
          ];
        lispHook = ''
          export CL_SOURCE_REGISTRY="''${CL_SOURCE_REGISTRY:+$CL_SOURCE_REGISTRY:}${
            lib.concatMapStringsSep ":" (p: "${p}//") lispSystems
          }"
          export ASDF_OUTPUT_TRANSLATIONS="${lib.concatMapStringsSep ":" (p: "${p}:${p}") lispSystems}"
        '';
        # php-config/phpize for building the extension; nixpkgs' wrapped PHP keeps them in the
        # unwrapped package's dev output.
        phpDev = lib.optional (phpPackage ? unwrapped) phpPackage.unwrapped.dev;
        # The prebuilt SDK: nixpkgs' source-built dotnet-sdk_10 doesn't build on aarch64-darwin.
        dotnetSdk = pkgs.dotnetCorePackages.sdk_10_0-bin;

        # Dev shells deliberately do NOT include the nix-built binding packages (pydatamunge,
        # datamunger, datamungejs, ...). Every binding depends on the core `datamunge` package, so
        # listing them here made each C++ edit rebuild every binding on the next direnv reload.
        # Instead, shells only build the core, and each language finds the binding built in the
        # working tree by `just build-<lang>`; rebuilding a binding is an explicit, manual step.
        # (The nix binding packages are still exposed under `packages` and used by `checks` and
        # the hermetic `docs-pages` shell.)
        # OpenMP OpenBLAS ignores OPENBLAS_NUM_THREADS. Large default BLAS teams
        # make the small NN examples much slower; retain explicit user budgets.
        nnThreadHook = ''
          export OMP_NUM_THREADS="''${OMP_NUM_THREADS:-1}"
          export OPENBLAS_NUM_THREADS="''${OPENBLAS_NUM_THREADS:-1}"
        '';
        localBindingsHook = nnThreadHook + ''
          datamungeRoot="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
          export DATAMUNGE_PREFIX="${datamunge}"
          export PKG_CONFIG_PATH="${datamunge}/lib/pkgconfig''${PKG_CONFIG_PATH:+:}$PKG_CONFIG_PATH"

          # Python commands use the installed wheel in build/venv/pydatamunge-run.
          # mep's Python hover (K) and completion introspect pydatamunge with that interpreter.
          export MEP_PYTHON="''${MEP_PYTHON:-$datamungeRoot/build/venv/pydatamunge-run/bin/python}"
          # just build-r
          export R_LIBS="$datamungeRoot/build/r/library''${R_LIBS:+:}$R_LIBS"
          # just build-javascript: build/Release/datamungejs.node behind ./index.js
          mkdir -p "$datamungeRoot/build/node_modules/@datamunge"
          ln -sfn "$datamungeRoot" "$datamungeRoot/build/node_modules/@datamunge/datamungejs"
          export NODE_PATH="$datamungeRoot/build/node_modules''${NODE_PATH:+:}$NODE_PATH"
          # just build-ruby (Rosetta): `require 'datamunge'` loads the committed
          # src/datamungeruby/lib/datamunge.rb, which require_relative's datamunge/datamunge.so; the
          # build installs both into build/ruby/prefix/lib, which comes first. mep's Ruby LSP follows
          # the require through RUBYLIB to the YARD docs in src/datamungeruby/lib/datamunge.rb.
          export RUBYLIB="$datamungeRoot/build/ruby/prefix/lib:$datamungeRoot/src/datamungeruby/lib''${RUBYLIB:+:}$RUBYLIB"
          # just build-perl
          export PERL5LIB="$datamungeRoot/build/perl/lib/perl5''${PERL5LIB:+:}$PERL5LIB"
          # just build-tcl (Rosetta; installs the package directory datamunge/ into build/tcl/prefix/lib):
          # `package require datamunge` from tclsh. TCLLIBPATH is a Tcl list (space-separated). mep's
          # Tcl LSP reads the documented proc/oo::class stubs of the committed
          # src/datamungetcl/pkg/datamunge/datamunge.tcl through MEP_TCL_PATH (it scans each directory
          # and its package subdirectories).
          export TCLLIBPATH="$datamungeRoot/build/tcl/prefix/lib''${TCLLIBPATH:+ $TCLLIBPATH}"
          export MEP_TCL_PATH="$datamungeRoot/src/datamungetcl/pkg''${MEP_TCL_PATH:+:$MEP_TCL_PATH}"
          # just build-lua (installs into build/lua/prefix)
          export LUA_PATH="$datamungeRoot/build/lua/prefix/share/lua/?.lua;''${LUA_PATH:-;}"
          export LUA_CPATH="$datamungeRoot/build/lua/prefix/lib/lua/?.so;$datamungeRoot/build/lua/prefix/lib64/lua/?.so;''${LUA_CPATH:-;}"
          # just build-lisp (installs datamunge.asd, .lisp and .so into build/lisp/prefix):
          # (asdf:load-system "datamunge") from SBCL in the default or lisp shell.
          export CL_SOURCE_REGISTRY="$datamungeRoot/build/lisp/prefix/share/common-lisp/source//''${CL_SOURCE_REGISTRY:+:$CL_SOURCE_REGISTRY}"
          # just build-julia (installs the Datamunge package into build/julia/prefix):
          # `using Datamunge` from Julia in the default or julia shell (and mep's Julia LSP).
          export JULIA_LOAD_PATH="$datamungeRoot/build/julia/prefix/share/julia:''${JULIA_LOAD_PATH:-@:@v#.#:@stdlib}"
          # just build-csharp (installs the native library into build/csharp/prefix): examples and
          # tests `#:project` src/datamungecsharp/Datamunge.csproj, whose loader reads this variable.
          csharpLibrary="$datamungeRoot/build/csharp/prefix/share/dotnet/Datamunge/lib/datamunge_csharp.so"
          if [ "$(uname)" = Darwin ]; then csharpLibrary="''${csharpLibrary%.so}.dylib"; fi
          export DATAMUNGE_CSHARP_LIBRARY="''${DATAMUNGE_CSHARP_LIBRARY:-$csharpLibrary}"
          # just build-fortran (installs libdatamunge_fortran, libdatamunge_fortran_mod.a and the .mod
          # files into build/fortran/prefix): compile with $DATAMUNGE_FORTRAN_FLAGS after the source
          # (just run-fortran spells it out). mep's Fortran LSP resolves `use datamunge` through
          # MEP_FORTRAN_PATH to the committed src/datamungefortran/datamunge.f90 and its FORD docs.
          fortranPrefix="$datamungeRoot/build/fortran/prefix"
          export DATAMUNGE_FORTRAN_FLAGS="''${DATAMUNGE_FORTRAN_FLAGS:--I$fortranPrefix/include/datamunge/fortran -L$fortranPrefix/lib -ldatamunge_fortran_mod -ldatamunge_fortran -Wl,-rpath,$fortranPrefix/lib}"
          export MEP_FORTRAN_PATH="$datamungeRoot/src/datamungefortran''${MEP_FORTRAN_PATH:+:$MEP_FORTRAN_PATH}"
          # just build-cobol (installs libdatamungecobol and the copybooks into build/cobol/prefix):
          # COPY "datamunge.cpy" finds them through COBCPY (cobc and mep's COBOL LSP).
          export COBCPY="$datamungeRoot/build/cobol/prefix/share/cobol/copy''${COBCPY:+:$COBCPY}"
          # just build-octave (Rosetta; installs __datamunge__.oct and +datamunge into
          # build/octave/prefix/share/octave/site/m): datamunge.DataFrame (...) from octave/octave-cli
          # in the default or octave shell. A site init file adds that directory to the load path
          # (rather than OCTAVE_PATH) after silencing Octave:shadowed-function, which Octave raises on
          # every path refresh for datamunge.resize and datamunge.Data.load (they shadow nothing).
          # mep's Octave LSP resolves datamunge.* through MEP_OCTAVE_PATH to the committed package
          # src/datamungeoctave/inst/+datamunge and its help text; mep's Octave console (which reads
          # the site init file) adds MEP_OCTAVE_PATH to its load path too.
          octaveSiteInit="$datamungeRoot/build/octave/octaverc"
          mkdir -p "$datamungeRoot/build/octave"
          {
            if [ -n "''${OCTAVE_SITE_INITFILE:-}" ] && [ "$OCTAVE_SITE_INITFILE" != "$octaveSiteInit" ]; then
              printf 'source ("%s");\n' "$OCTAVE_SITE_INITFILE"
            fi
            printf 'warning ("off", "Octave:shadowed-function");\nif (isfolder ("%s"))\n  addpath ("%s");\nend\n' \
              "$datamungeRoot/build/octave/prefix/share/octave/site/m" "$datamungeRoot/build/octave/prefix/share/octave/site/m"
          } > "$octaveSiteInit"
          export OCTAVE_SITE_INITFILE="$octaveSiteInit"
          export MEP_OCTAVE_PATH="$datamungeRoot/src/datamungeoctave/inst''${MEP_OCTAVE_PATH:+:$MEP_OCTAVE_PATH}"
          # just build-guile (Rosetta; installs datamunge.scm, datamunge/ and libdatamunge_guile.so
          # together into build/guile/prefix/share/guile/site/<version>): (use-modules (datamunge))
          # from guile in the default or guile shell; the module loads the extension beside it.
          # mep's Scheme LSP reads the docstrings of the committed src/datamungeguile/scheme/datamunge.scm
          # through MEP_GUILE_PATH.
          guileVersion="$(pkg-config --variable=effective-version guile-3.0 2>/dev/null || echo 3.0)"
          # (empty, not a failure, when guile-3.0.pc isn't on PKG_CONFIG_PATH -- as in the default shell)
          guileVersion="''${guileVersion:-3.0}"
          export GUILE_LOAD_PATH="$datamungeRoot/build/guile/prefix/share/guile/site/$guileVersion''${GUILE_LOAD_PATH:+:}$GUILE_LOAD_PATH"
          # the precompiled datamunge.go (built by build-guile), so loading needn't compile the module
          export GUILE_LOAD_COMPILED_PATH="$datamungeRoot/build/guile/prefix/lib/guile/$guileVersion/site-ccache''${GUILE_LOAD_COMPILED_PATH:+:}$GUILE_LOAD_COMPILED_PATH"
          export MEP_GUILE_PATH="$datamungeRoot/src/datamungeguile/scheme''${MEP_GUILE_PATH:+:$MEP_GUILE_PATH}"
          # just build-ocaml (Rosetta; installs the findlib package datamunge/ into
          # build/ocaml/prefix/lib): `ocamlfind ocamlopt -package datamunge`, dune
          # `(libraries datamunge)` or `#require "datamunge"` in the toplevel. Native programs link
          # libdatamunge_stubs.a and libdatamunge (rpath); bytecode and the toplevel load
          # dlldatamunge_stubs.so from the package directory (CAML_LD_LIBRARY_PATH), and the toplevel
          # finds `#use "topfind"` through OCAMLTOP_INCLUDE_PATH. mep's OCaml server reads the odoc
          # comments of the committed src/datamungeocaml/ocaml/datamunge/datamunge.mli through
          # MEP_OCAML_PATH.
          export OCAMLPATH="$datamungeRoot/build/ocaml/prefix/lib''${OCAMLPATH:+:}$OCAMLPATH"
          export CAML_LD_LIBRARY_PATH="$datamungeRoot/build/ocaml/prefix/lib/datamunge''${CAML_LD_LIBRARY_PATH:+:}$CAML_LD_LIBRARY_PATH"
          export MEP_OCAML_PATH="$datamungeRoot/src/datamungeocaml/ocaml''${MEP_OCAML_PATH:+:$MEP_OCAML_PATH}"
          if command -v ocamlfind >/dev/null 2>&1; then
            export OCAMLTOP_INCLUDE_PATH="$(dirname "$(ocamlfind query findlib)")''${OCAMLTOP_INCLUDE_PATH:+:}$OCAMLTOP_INCLUDE_PATH"
          fi
          # just build-go (Rosetta; installs libdatamunge_go into build/go/prefix/lib): the committed
          # module src/datamungego/go/datamunge links -ldatamunge_go, found through CGO_LDFLAGS (with
          # an rpath, so programs need no LD_LIBRARY_PATH). The workspace build/go/go.work holds that
          # module plus examples/go and tests/go, so `import "datamunge"` resolves from the root
          # (`go run examples/go/x.go`) and from a Go program anywhere (a mepml block's temp file).
          # mep's Go server reads the godoc of the committed module through the replace directives
          # in examples/go/go.mod and tests/go/go.mod.
          export DATAMUNGE_GO_LIB="''${DATAMUNGE_GO_LIB:-$datamungeRoot/build/go/prefix/lib}"
          # go compares the current directory with the workspace's paths without resolving
          # symlinks, so they are spelled as the shell reaches the repository (git's root is the
          # resolved path; $PWD may run through a symlink).
          goRoot="$datamungeRoot"
          goPrefix="$(git rev-parse --show-prefix 2>/dev/null || true)"
          goLogical="''${PWD%/}"
          if [ -n "$goPrefix" ]; then goLogical="''${goLogical%/''${goPrefix%/}}"; fi
          if [ "$goLogical" != "$datamungeRoot" ] && [ "$goLogical" -ef "$datamungeRoot" ]; then goRoot="$goLogical"; fi
          mkdir -p "$datamungeRoot/build/go"
          printf 'go 1.24\n\nuse (\n\t%s\n\t%s\n\t%s\n)\n' "$goRoot/src/datamungego/go/datamunge" \
            "$goRoot/examples/go" "$goRoot/tests/go" > "$datamungeRoot/build/go/go.work"
          export GOWORK="$datamungeRoot/build/go/go.work"
          export CGO_LDFLAGS="-L$DATAMUNGE_GO_LIB -Wl,-rpath,$DATAMUNGE_GO_LIB''${CGO_LDFLAGS:+ $CGO_LDFLAGS}"
          # just build-java (Rosetta; installs datamunge.jar and libdatamunge_java into
          # build/java/prefix): `java -cp $MEP_JAVA_CLASSPATH App.java`; the package datamunge loads
          # the JNI library named by DATAMUNGE_JAVA_LIBRARY. mep's Java mode puts MEP_JAVA_CLASSPATH
          # on its class path, and its Java server reads the Javadoc of the committed sources
          # src/datamungejava/java through MEP_JAVA_SOURCEPATH.
          javaLibrary="$datamungeRoot/build/java/prefix/lib/libdatamunge_java.so"
          if [ "$(uname)" = Darwin ]; then javaLibrary="''${javaLibrary%.so}.dylib"; fi
          export DATAMUNGE_JAVA_LIBRARY="''${DATAMUNGE_JAVA_LIBRARY:-$javaLibrary}"
          export MEP_JAVA_CLASSPATH="$datamungeRoot/build/java/prefix/share/java/datamunge.jar''${MEP_JAVA_CLASSPATH:+:$MEP_JAVA_CLASSPATH}"
          export MEP_JAVA_SOURCEPATH="$datamungeRoot/src/datamungejava/java''${MEP_JAVA_SOURCEPATH:+:$MEP_JAVA_SOURCEPATH}"
          # just build-apl (Rosetta; installs libdatamunge_gnu_apl.so into build/apl/prefix/lib and the
          # script datamunge.apl into build/apl/prefix/share/apl): `apl --script -f
          # $DATAMUNGE_APL_SCRIPT -f x.apl`; the script ⎕FX's the library DATAMUNGE_APL_LIBRARY names.
          # mep's APL server reads the ⍝ documentation of the committed
          # src/datamungeapl/apl/datamunge.apl through MEP_APL_PATH.
          export DATAMUNGE_APL_LIBRARY="''${DATAMUNGE_APL_LIBRARY:-$datamungeRoot/build/apl/prefix/lib/libdatamunge_gnu_apl.so}"
          export DATAMUNGE_APL_SCRIPT="''${DATAMUNGE_APL_SCRIPT:-$datamungeRoot/src/datamungeapl/apl/datamunge.apl}"
          export MEP_APL_PATH="$datamungeRoot/src/datamungeapl/apl''${MEP_APL_PATH:+:$MEP_APL_PATH}"
          # just build-d: libdatamunge_wrap, which the generated D module dlopens by leaf name at
          # startup. (macOS ignores LD_LIBRARY_PATH there; its fallback path can't shadow anything.)
          case "$(uname -s)" in
            Darwin) export DYLD_FALLBACK_LIBRARY_PATH="$datamungeRoot/build/d:''${DYLD_FALLBACK_LIBRARY_PATH:-$HOME/lib:/usr/local/lib:/usr/lib}" ;;
            *) export LD_LIBRARY_PATH="$datamungeRoot/build/d''${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" ;;
          esac
          # just build-php writes an ini loading the extension here (the leading ':' keeps PHP's
          # own scan directory too)
          export PHP_INI_SCAN_DIR=":$datamungeRoot/build/php/conf.d"
        '';

        # {{{ Bindings

        pythonPkgs = pkgs.python3.pkgs;
        pydatamunge = import ./nix/pydatamunge.nix {
          inherit datamunge swigPkg rosettaPkg;
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
          inherit datamunge;
          inherit (pkgs)
            lib
            buildNpmPackage
            libxml2
            pkg-config
            ;
        };

        datamunger = pkgs.rPackages.buildRPackage {
          name = "datamunger";
          src = import ./nix/source.nix { inherit lib; };
          buildInputs = [
            pkgs.libxml2
            pkgs.pkg-config
            pkgs.R
            pkgs.xorg.libX11.dev
            datamunge
          ];
          propagatedBuildInputs = [ pkgs.rPackages.R6 ];
          # Link the nix-built core instead of recompiling it here (see src/Makevars).
          env.DATAMUNGE_USE_SYSTEM_LIB = "1";
          env.ROSETTA = "${rosettaPkg}/bin/rosetta";
          nativeBuildInputs = [
            rosettaPkg
            pkgs.python3
            pkgs.pkg-config
            (pkgs.rWrapper.override { packages = [ pkgs.rPackages.R6 pkgs.rPackages.roxygen2 ]; })
          ];
          preConfigure = "python3 scripts/prebuild_bindings.py r";
        };

        datamungetcl = import ./nix/datamungetcl.nix { pkgs = bindingPkgs; };
        datamungejava = import ./nix/datamungejava.nix { pkgs = bindingPkgs; };
        datamungego = import ./nix/datamungego.nix { pkgs = bindingPkgs; };
        datamungeruby = import ./nix/datamungeruby.nix { pkgs = bindingPkgs; };
        datamungelua = import ./nix/datamungelua.nix { pkgs = bindingPkgs; };
        datamungelisp = import ./nix/datamungelisp.nix { pkgs = bindingPkgs; };
        datamungejulia = import ./nix/datamungejulia.nix { pkgs = bindingPkgs; };
        datamungecsharp = import ./nix/datamungecsharp.nix { pkgs = bindingPkgs; inherit dotnetSdk; };
        datamungecobol = import ./nix/datamungecobol.nix { pkgs = bindingPkgs; };
        datamungefortran = import ./nix/datamungefortran.nix { pkgs = bindingPkgs; };
        # GnuCOBOL: cobc lives in the bin output, which a shell's packages leave out.
        gnuCobol = [ pkgs.gnucobol pkgs.gnucobol.bin ];
        datamungeocaml = import ./nix/datamungeocaml.nix { pkgs = bindingPkgs; };
        # GNU APL (Linux only: nixpkgs marks gnuapl broken on macOS). Its native functions compile
        # against GNU APL's configured headers, datamungeapl.gnuAplHeaders.
        datamungeapl = import ./nix/datamungeapl.nix { pkgs = bindingPkgs; };
        datamungeguile = import ./nix/datamungeguile.nix { pkgs = bindingPkgs; };
        datamungeoctave = import ./nix/datamungeoctave.nix { pkgs = bindingPkgs; };
        datamunged = import ./nix/datamunged.nix { pkgs = bindingPkgs; };

        # }}} Bindings

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

        # Report shells (nix/report/<lang>.nix): a language runtime plus its nix-built binding, for
        # mepml report blocks (`nix-shell="git+file:../..#report-<lang>"`). Unlike the dev shells
        # these do carry the binding packages; they are never activated by direnv, so a C++ edit
        # only rebuilds a binding when a report in that language is next run.
        reportShellArgs = {
          inherit
            pkgs
            lib
            datamunge
            swigPkg
            bindingPkgs
            lua
            phpPackage
            phpDev
            dotnetSdk
            datamungecsharp
            datamungefortran
            datamungejava
            datamungego
            pydatamunge
            datamunger
            datamungejs
            datamungeruby
            datamungelua
            datamungetcl
            datamungeoctave
            datamungeguile
            datamungeocaml
            datamungeapl
            datamunged
            ;
        };
        reportShell = language: import (./nix/report + "/${language}.nix") reportShellArgs;

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
              pytest --import-mode=importlib -q tests/python bindings_tests/python
            '';
            installPhase = "mkdir -p $out";
          };

          r = pkgs.stdenv.mkDerivation {
            name = "datamunge-r-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [
              pkgs.R
              pkgs.rPackages.testthat
              datamunger
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
              Rscript -e 'library(datamunger); testthat::test_dir("tests/testthat", stop_on_failure=TRUE); testthat::test_dir("bindings_tests/r", stop_on_failure=TRUE)'
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
            nativeBuildInputs = [ pkgs.jdk ];
            phases = [
              "unpackPhase"
              "checkPhase"
              "installPhase"
            ];
            doCheck = true;
            # tests/java/TestDatamunge.java (a single-file launch) against the nix-built jar and JNI
            # library; it reads the committed sources src/datamungejava/java for the documentation checks.
            checkPhase = ''
              export DATAMUNGE_JAVA_LIBRARY="${datamungejava.library}"
              java -cp "${datamungejava.jar}" tests/java/TestDatamunge.java src/datamungejava/java
            '';
            installPhase = "mkdir -p $out";
          };

          go = pkgs.stdenv.mkDerivation {
            name = "datamunge-go-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [
              bindingPkgs.go
              bindingPkgs.stdenv.cc
            ];
            phases = [
              "unpackPhase"
              "checkPhase"
              "installPhase"
            ];
            doCheck = true;
            # tests/go (go test, offline) against the nix-built library and module (a workspace
            # using it, so tests/go/go.mod's replace of the committed copy is overridden); it reads
            # the committed src/datamungego/go/datamunge for the documentation checks. The build
            # cache starts from the package's warm one.
            checkPhase = ''
              export HOME="$TMPDIR" GOTOOLCHAIN=local GOPROXY=off GOFLAGS= GOCACHE="$TMPDIR/go-cache"
              export CGO_ENABLED=1 CGO_LDFLAGS="${datamungego.cgoLdflags}"
              cp -r --no-preserve=mode "${datamungego}/share/go-cache" "$GOCACHE"
              printf 'go 1.24\n\nuse (\n\t%s\n\t%s\n)\n' "${datamungego.module}" "$PWD/tests/go" > "$TMPDIR/go.work"
              export GOWORK="$TMPDIR/go.work"
              (cd tests/go && go test -count=1 -v .)
            '';
            installPhase = "mkdir -p $out";
          };

          ocaml = pkgs.stdenv.mkDerivation {
            name = "datamunge-ocaml-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [
              pkgs.ocamlPackages.ocaml
              pkgs.ocamlPackages.findlib
            ];
            phases = [
              "unpackPhase"
              "checkPhase"
              "installPhase"
            ];
            doCheck = true;
            # tests/ocaml/test_datamunge.ml, compiled natively against the nix-built findlib package;
            # it reads the committed src/datamungeocaml/ocaml/datamunge/datamunge.mli for the
            # documentation checks.
            checkPhase = ''
              export OCAMLPATH="${datamungeocaml}/lib"
              mkdir -p build && cp tests/ocaml/test_datamunge.ml build/
              ocamlfind ocamlopt -package datamunge -linkpkg -o build/test_datamunge build/test_datamunge.ml
              ./build/test_datamunge src/datamungeocaml/ocaml/datamunge/datamunge.mli
            '';
            installPhase = "mkdir -p $out";
          };

          apl = pkgs.stdenv.mkDerivation {
            name = "datamunge-apl-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [ pkgs.gnuapl ];
            phases = [
              "unpackPhase"
              "checkPhase"
              "installPhase"
            ];
            doCheck = true;
            # tests/apl/test_datamunge.apl against the nix-built library and script; it reads the
            # committed src/datamungeapl/apl/datamunge.apl (DATAMUNGE_APL_DOCS) for the documentation
            # checks.
            checkPhase = ''
              export HOME="$TMPDIR"
              export DATAMUNGE_APL_LIBRARY="${datamungeapl.library}"
              export DATAMUNGE_APL_DOCS="$PWD/src/datamungeapl/apl/datamunge.apl"
              apl --script --noSV --noColor --silent -f "${datamungeapl.script}" -f tests/apl/test_datamunge.apl \
                | tee apl-test.log
              grep -q "^DATAMUNGE APL PASS" apl-test.log
            '';
            installPhase = "mkdir -p $out";
          };

          tcl = pkgs.stdenv.mkDerivation {
            name = "datamunge-tcl-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [ pkgs.tcl ];
            phases = [
              "unpackPhase"
              "checkPhase"
              "installPhase"
            ];
            doCheck = true;
            # tests/tcl/test_datamunge.tcl against the nix-built package; it reads the committed
            # src/datamungetcl/pkg/datamunge/datamunge.tcl for the documentation checks.
            checkPhase = ''
              export TCLLIBPATH="${datamungetcl}/lib"
              tclsh tests/tcl/test_datamunge.tcl src/datamungetcl/pkg/datamunge/datamunge.tcl
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

          lisp = pkgs.stdenv.mkDerivation {
            name = "datamunge-lisp-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [ pkgs.sbcl ];
            phases = [
              "unpackPhase"
              "checkPhase"
              "installPhase"
            ];
            doCheck = true;
            checkPhase = lispHook + ''
              export CL_SOURCE_REGISTRY="${datamungelisp}/share/common-lisp/source//:$CL_SOURCE_REGISTRY"
              export HOME="$TMPDIR"
              sbcl --noinform --non-interactive --load tests/lisp/test_datamunge.lisp
            '';
            installPhase = "mkdir -p $out";
          };

          julia = pkgs.stdenv.mkDerivation {
            name = "datamunge-julia-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [ pkgs.julia-bin ];
            phases = [
              "unpackPhase"
              "checkPhase"
              "installPhase"
            ];
            doCheck = true;
            checkPhase = ''
              export JULIA_LOAD_PATH="${datamungejulia}/share/julia:@stdlib"
              export JULIA_DEPOT_PATH="$TMPDIR/julia-depot"
              export HOME="$TMPDIR"
              julia --startup-file=no --threads=2 tests/julia/test_datamunge.jl
            '';
            installPhase = "mkdir -p $out";
          };

          cobol = pkgs.stdenv.mkDerivation {
            name = "datamunge-cobol-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = gnuCobol;
            phases = [
              "unpackPhase"
              "checkPhase"
              "installPhase"
            ];
            doCheck = true;
            checkPhase = ''
              export COBCPY="${datamungecobol}/share/cobol/copy"
              cobc -x -free -fstatic-call -o test_datamunge tests/cobol/test_datamunge.cob \
                -L${datamungecobol}/lib -ldatamungecobol -Q -Wl,-rpath,${datamungecobol}/lib
              ./test_datamunge
            '';
            installPhase = "mkdir -p $out";
          };

          ruby = pkgs.stdenv.mkDerivation {
            name = "datamunge-ruby-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [ pkgs.ruby ];
            phases = [
              "unpackPhase"
              "checkPhase"
              "installPhase"
            ];
            doCheck = true;
            # tests/ruby/test_datamunge.rb against the nix-built extension; it reads the committed
            # src/datamungeruby/lib/datamunge.rb for the YARD documentation checks.
            checkPhase = ''
              export HOME="$TMPDIR"
              ruby -I "${datamungeruby}/lib" tests/ruby/test_datamunge.rb src/datamungeruby/lib/datamunge.rb
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
              # tests/guile/test_datamunge.scm against the nix-built extension and module; it reads
              # the committed src/datamungeguile/scheme/datamunge.scm for the docstring checks.
              export HOME="$TMPDIR"
              export GUILE_LOAD_PATH="${datamungeguile}/share/guile/site/${pkgs.guile.effectiveVersion or "3.0"}''${GUILE_LOAD_PATH:+:}$GUILE_LOAD_PATH"
              export GUILE_LOAD_COMPILED_PATH="${datamungeguile}/lib/guile/${pkgs.guile.effectiveVersion or "3.0"}/site-ccache''${GUILE_LOAD_COMPILED_PATH:+:}$GUILE_LOAD_COMPILED_PATH"
              ${pkgs.guile}/bin/guile --no-auto-compile -s tests/guile/test_datamunge.scm src/datamungeguile/scheme/datamunge.scm
            '';
            installPhase = "mkdir -p $out";
          };

          octave = pkgs.stdenv.mkDerivation {
            name = "datamunge-octave-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [ pkgs.octave ];
            phases = [
              "unpackPhase"
              "checkPhase"
              "installPhase"
            ];
            doCheck = true;
            # tests/octave/test_datamunge.m against the nix-built oct-file and package (as just
            # test-octave does; see OCTAVE in the justfile for the warning).
            checkPhase = ''
              export HOME="$TMPDIR"
              octave-cli --norc --no-history --quiet --eval \
                "warning off Octave:shadowed-function; addpath (\"${datamungeoctave}/share/octave/site/m\"); source tests/octave/test_datamunge.m"
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

          csharp = pkgs.stdenv.mkDerivation {
            name = "datamunge-csharp-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [ dotnetSdk ];
            phases = [
              "unpackPhase"
              "checkPhase"
              "installPhase"
            ];
            doCheck = true;
            # tests/csharp/test_datamunge.cs is a file-based app (`#:project` the generated project);
            # it needs no NuGet packages (tests/csharp/nuget.config clears the sources).
            checkPhase = ''
              export HOME="$TMPDIR" DOTNET_CLI_HOME="$TMPDIR" DOTNET_NOLOGO=1 DOTNET_CLI_TELEMETRY_OPTOUT=1
              export DATAMUNGE_CSHARP_LIBRARY="${datamungecsharp}/share/dotnet/Datamunge/lib/datamunge_csharp${pkgs.stdenv.hostPlatform.extensions.sharedLibrary}"
              dotnet run tests/csharp/test_datamunge.cs
            '';
            installPhase = "mkdir -p $out";
          };

          fortran = pkgs.stdenv.mkDerivation {
            name = "datamunge-fortran-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [ pkgs.gfortran ];
            phases = [
              "unpackPhase"
              "checkPhase"
              "installPhase"
            ];
            doCheck = true;
            # tests/fortran/test_datamunge.f90 against the nix-built modules and native library; it reads
            # src/datamungefortran/datamunge.f90 for the FORD documentation checks.
            checkPhase = ''
              prefix="${datamungefortran}"
              mkdir -p "$TMPDIR/mod"
              gfortran -std=f2018 -Wall -Wno-compare-reals -I "$prefix/include/datamunge/fortran" -J "$TMPDIR/mod" \
                tests/fortran/test_datamunge.f90 -L "$prefix/lib" -ldatamunge_fortran_mod -ldatamunge_fortran \
                -Wl,-rpath,"$prefix/lib" -o "$TMPDIR/test_datamunge"
              "$TMPDIR/test_datamunge" src/datamungefortran/datamunge.f90
            '';
            installPhase = "mkdir -p $out";
          };
        };

        packages = {
          # Optional reference engine; standard CPU/CUDA packages use native vsim alone.
          datamunge-mujoco = import ./nix/datamunge.nix { inherit pkgs; enableMujoco = true; };
          inherit
            datamunge
            pydatamunge
            datamungejs
            datamunger
            datamungetcl
            datamungejava
            datamungego
            datamungeruby
            datamungelua
            datamungelisp
            datamungejulia
            datamungecsharp
            datamungecobol
            datamungefortran
            datamungeocaml
            datamungeapl
            datamungeguile
            datamungeoctave
            datamunged
            ;

          mep = mepPkg;

          rename-datamunge = pkgs.writeShellApplication {
            name = "rename-datamunge";
            text = ''exec "${./rename_datamunge}" "$@"'';
          };
        }
        // lib.optionalAttrs cudaSupported {
          datamunge-cuda = import ./nix/datamunge.nix {
            pkgs = cudaPkgs;
            enableCuda = true;
            # Context::matmul uses the in-house GEMM kernels; cuBLAS is a comparison path only.
            enableCublas = false;
            inherit cudaToolkit;
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

            creplPkg
            mepPkg

            pkgs.libxml2

            datamunge
            pkgs.lcov
            pkgs.clang
            pkgs.doctest
            pkgs.pkg-config

            (pkgs.python3.withPackages (
              python-pkgs: with python-pkgs; [
                python-lsp-server
                numpy
              ]
            ))

            # Doc export tooling
            pkgs.emacs
            pkgs.direnv
            pkgs.just
            pkgs.jq

            # Core library + pkg-config visibility
            datamunge
            pkgs.pkg-config
            pkgs.cmake
            pkgs.gnumake
            pkgs.stdenv.cc
            swigPkg

            # Language runtimes for runnable examples. The bindings themselves come from the
            # working tree (`just build-<lang>`), not nix; see localBindingsHook.
            pkgs.nodejs
            pkgs.R
            # datamunger's runtime dep; mep's R UI mode runs bare `R` from this shell.
            pkgs.rPackages.R6
            pkgs.rPackages.testthat
            pkgs.rPackages.roxygen2
            pkgs.ruby
            pkgs.perl
            phpPackage
            lua
            # Common Lisp: mep's Lisp UI mode runs `sbcl` from this shell (see lispHook).
            pkgs.sbcl
            # Julia: mep's Julia UI mode runs `julia` from this shell (JULIA_LOAD_PATH above).
            pkgs.julia-bin
          ]
          # COBOL: mep's COBOL UI mode runs `cobc` from this shell.
          ++ gnuCobol
          ++ [
            pkgs.tcl
            pkgs.octave
            pkgs.guile
            pkgs.jdk
            pkgs.ldc
            dotnetSdk
          ]
          ++ phpDev
          ++ [

            # For building/running OCaml + Go examples during export
            pkgs.ocamlPackages.ocaml
            pkgs.ocamlPackages.dune_3
            pkgs.ocamlPackages.findlib
            pkgs.ocamlPackages.alcotest
            pkgs.go
          ];

          shellHook = localBindingsHook + lispHook;
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
        devShells.cpp-tools = pkgs.mkShell {

          packages = [
            creplPkg
            # mep-mepml runs and exports the docs' mepml reports (just docs).
            mepPkg
            pkgs.curl
            pkgs.libarchive
            pkgs.openssl
            pkgs.nlohmann_json
            pkgs.hdf5-cpp
            pkgs.ffmpeg

            # Build the working tree with CMake; entering this shell must not
            # first rebuild an installed copy of the library after every edit.
            pkgs.just
            pkgs.emacs
            pkgs.direnv
            pkgs.jq
            pkgs.lcov
            pkgs.gtest
            (pkgs.python3.withPackages (
              python-pkgs: with python-pkgs; [
                numpy
                jinja2
                pygments
                pytest
                # The documentation website (scripts/website).
                markdown-it-py
              ]
            ))
            pkgs.clang
            # clangd provides definition hover checks for Doxygen source comments.
            (if has "clang-tools" pkgs then pkgs.clang-tools else pkgs.clang)
            pkgs.libxml2
            pkgs.pkg-config
            pkgs.arrow-cpp
            swigPkg
            pkgs.cling
            pkgs.doxygen
            pkgs.graphviz
            pkgs.doctest
            pkgs.cmake
            pkgs.xorg.libX11.dev
            # Method diagrams are rendered by the native C++ graphics library.
            # benchmarks/ (nn GEMM, conv and training benchmarks)
            pkgs.gbenchmark

            # pkgs.nodejs
            # pkgs.prefetch-npm-deps
            # pkgs.nodePackages.npm

          ]
          # OpenBLAS is not a dependency (datamunge::nn has in-house BLAS kernels);
          # it stays in this shell only so -DDATAMUNGE_USE_BLAS=ON comparison
          # builds work. `just bench-blas` uses the blas-bench shell below.
          ++ pkgs.lib.optionals pkgs.stdenv.isLinux [ pkgs.openblas ];
          shellHook = nnThreadHook;
        };

        # Comparators for `just bench-blas` (benchmarks/blas/blas_compare.cpp loads
        # them with dlopen): nixpkgs OpenBLAS (Haswell kernels on x86-64), OpenBLAS
        # with its AVX-512 kernels, BLIS, and the netlib reference BLAS.
        devShells.blas-bench = pkgs.mkShell {
          inputsFrom = [ self.devShells.${system}.cpp-tools ];
          shellHook =
            nnThreadHook
            + lib.optionalString pkgs.stdenv.isLinux ''
              export DATAMUNGE_BLAS_LIBS="openblas=${pkgs.openblas}/lib/libopenblas.so;openblas-avx512=${
                pkgs.openblas.override { enableAVX512 = pkgs.stdenv.hostPlatform.isx86_64; }
              }/lib/libopenblas.so;blis=${pkgs.blis}/lib/libblis.so;reference=${pkgs.lapack-reference}/lib/libblas.so:fixed-threads:slow"
            '';
        };

        # Optional simulation/compute SDKs; ordinary CPU builds keep cpp-tools sufficient.
        devShells.simulation = pkgs.mkShell {
          inputsFrom = [ self.devShells.${system}.cpp-tools ];
          packages = [
            pkgs.vulkan-headers
            pkgs.vulkan-loader
            pkgs.vulkan-tools
            pkgs.shaderc
          ]
          ++ lib.optionals cudaSupported [ cudaToolkit ];
          shellHook = lib.optionalString cudaSupported cudaHook;
        };

        devShells.${if cudaSupported then "cuda" else null} = pkgs.mkShell {
          inputsFrom = [ self.devShells.${system}.cpp-tools ];
          packages = [ cudaToolkit ];
          shellHook = cudaHook;
        };

        # Keep installed-library workflows (e.g. repl-cpp) available without
        # making working-tree NN builds first rebuild that installed package.
        devShells.cpp = pkgs.mkShell {
          inputsFrom = [ self.devShells.${system}.cpp-tools ];
          packages = [
            datamunge
            pkgs.curl
          ];
        };

        devShells.java = pkgs.mkShell {
          packages = [
            datamunge
            # javac, jar, java and jshell, plus the JNI headers (build-java builds libdatamunge_java).
            pkgs.jdk
            # just prebuild-java: scripts/prebuild_bindings.py java runs rosetta.
            rosettaPkg
            pkgs.python3
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
            pkgs.just
            # datamungeConfig.cmake find_dependency(Arrow) for build-java.
            pkgs.arrow-cpp
          ];

          shellHook = localBindingsHook;
        };

        devShells.python = pkgs.mkShell {
          packages = [
            pkgs.curl
            rosettaPkg
            swigPkg
            # pkgs.cmake
            pkgs.pkg-config
            pkgs.libxml2
            pkgs.just
            (pkgs.python3.withPackages (
              python-pkgs: with python-pkgs; [
                ipython
                pip
                setuptools
                wheel
                pytest
                numpy
                matplotlib
                python-lsp-server
              ]
            ))
          ];
          # `just build-python` installs pydatamunge into build/venv/pydatamunge-run (a
          # --system-site-packages venv over this shell's python), so put that venv first on
          # PATH: bare `python` then imports the working-tree binding plus numpy/ipython/etc.
          shellHook = localBindingsHook + ''
            pydatamungeVenv="$datamungeRoot/build/venv/pydatamunge-run"
            # (run it, not just -x: a venv whose base python was garbage-collected must not
            # shadow the shell's python, which `just build-python` uses to recreate it)
            if "$pydatamungeVenv/bin/python" -c "" 2>/dev/null; then
              export VIRTUAL_ENV="$pydatamungeVenv"
              export PATH="$pydatamungeVenv/bin:$PATH"
            else
              echo "pydatamunge is not built yet; run \`just build-python\` to create $pydatamungeVenv" >&2
            fi
          '';
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
            # just prebuild-javascript: scripts/prebuild_bindings.py javascript runs rosetta.
            rosettaPkg
            pkgs.libxml2
            pkgs.pkg-config
            pkgs.python3
            pkgs.nodejs
            pkgs.just
            pkgs.prefetch-npm-deps
            pkgs.nodePackages.npm
          ];
          shellHook = localBindingsHook;
        };

        devShells.r = pkgs.mkShell {
          packages = [
            pkgs.curl
            swigPkg
            pkgs.python3
            pkgs.R
            pkgs.rPackages.R6
            pkgs.rPackages.testthat
            # Turns rosetta's generated roxygen comments into man/ pages (scripts/prebuild_bindings.py r).
            pkgs.rPackages.roxygen2
            rosettaPkg
            pkgs.pkg-config
            pkgs.libxml2
            pkgs.xorg.libX11.dev
            pkgs.just
          ];
          shellHook = localBindingsHook;
        };

        devShells.csharp = pkgs.mkShell {
          packages = [
            datamunge
            dotnetSdk
            # just prebuild-csharp: scripts/prebuild_bindings.py csharp runs rosetta.
            rosettaPkg
            pkgs.python3
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
            pkgs.just
            # datamungeConfig.cmake find_dependency(Arrow) for build-csharp.
            pkgs.arrow-cpp
          ]
          # just repl-csharp (it doesn't build on aarch64-darwin)
          ++ opt pkgs.stdenv.isLinux [ pkgs.dotnet-repl ];

          shellHook = localBindingsHook;
        };

        devShells.php = pkgs.mkShell {
          packages = [
            phpPackage
          ]
          ++ phpDev
          ++ [
            pkgs.libxml2
            pkgs.pkg-config
            pkgs.cmake
            pkgs.stdenv.cc
            pkgs.just
            # just build-php builds the core from source via CPM, which
            # requires the data-asset dependencies (DATAMUNGE_ENABLE_DATA_ASSETS).
            pkgs.libarchive
            pkgs.openssl
            pkgs.nlohmann_json
            pkgs.hdf5-cpp
          ];
        };

        devShells.go = pkgs.mkShell {
          packages = [
            datamunge
            # go (cgo builds the examples and tests against build/go/prefix/lib/libdatamunge_go).
            pkgs.go
            pkgs.stdenv.cc
            # just prebuild-go: scripts/prebuild_bindings.py go runs rosetta.
            rosettaPkg
            pkgs.python3
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
            pkgs.just
            # datamungeConfig.cmake find_dependency(Arrow) for build-go.
            pkgs.arrow-cpp
          ];

          shellHook = localBindingsHook;
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
            swigPkg
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
            # macOS resolves runtime libraries (incl. the SWIG wrapper loaded by dlopen) via DYLD_LIBRARY_PATH.
            export DYLD_LIBRARY_PATH="${datamunged}/lib:$DATAMUNGE_LIBDIR''${DYLD_LIBRARY_PATH:+:}$DYLD_LIBRARY_PATH"

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
            # tclsh, tcl.h and the stub library (build-tcl builds a stubs-enabled extension).
            pkgs.tcl
            pkgs.tk
            # just prebuild-tcl: scripts/prebuild_bindings.py tcl runs rosetta.
            rosettaPkg
            pkgs.python3
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
            pkgs.just
            # datamungeConfig.cmake find_dependency(Arrow) for build-tcl.
            pkgs.arrow-cpp
          ];

          shellHook = localBindingsHook;
        };

        devShells.ruby = pkgs.mkShell {
          packages = [
            datamunge
            pkgs.ruby
            # just prebuild-ruby: scripts/prebuild_bindings.py ruby runs rosetta.
            rosettaPkg
            pkgs.python3
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
            pkgs.just
            # datamungeConfig.cmake find_dependency(Arrow) for build-ruby.
            pkgs.arrow-cpp
          ];

          shellHook = localBindingsHook;
        };

        devShells.octave = pkgs.mkShell {
          packages = [
            datamunge
            # octave-cli and mkoctfile (build-octave reads its include and library directories).
            pkgs.octave
            # just prebuild-octave: scripts/prebuild_bindings.py octave runs rosetta.
            rosettaPkg
            pkgs.python3
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
            pkgs.just
            # datamungeConfig.cmake find_dependency(Arrow) for build-octave.
            pkgs.arrow-cpp
          ];

          shellHook = localBindingsHook;
        };

        devShells.lua = pkgs.mkShell {
          packages = [
            datamunge
            lua
            pkgs.arrow-cpp
            # just prebuild-lua: scripts/prebuild_bindings.py lua runs rosetta.
            rosettaPkg
            pkgs.python3
            pkgs.cmake
            pkgs.pkg-config
            pkgs.just
          ];

          shellHook = localBindingsHook;
        };

        devShells.lisp = pkgs.mkShell {
          packages = [
            datamunge
            pkgs.sbcl
            # just prebuild-lisp: scripts/prebuild_bindings.py lisp runs rosetta.
            rosettaPkg
            pkgs.python3
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
            pkgs.just
            # datamungeConfig.cmake find_dependency(Arrow) for build-lisp.
            pkgs.arrow-cpp
          ];

          shellHook = localBindingsHook + lispHook;
        };

        devShells.julia = pkgs.mkShell {
          packages = [
            datamunge
            pkgs.julia-bin
            # just prebuild-julia: scripts/prebuild_bindings.py julia runs rosetta.
            rosettaPkg
            pkgs.python3
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
            pkgs.just
            # datamungeConfig.cmake find_dependency(Arrow) for build-julia.
            pkgs.arrow-cpp
          ];

          shellHook = localBindingsHook;
        };

        devShells.fortran = pkgs.mkShell {
          packages = [
            datamunge
            pkgs.gfortran
            # just prebuild-fortran: scripts/prebuild_bindings.py fortran runs rosetta.
            rosettaPkg
            pkgs.python3
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
            pkgs.just
            # datamungeConfig.cmake find_dependency(Arrow) for build-fortran.
            pkgs.arrow-cpp
          ];

          shellHook = localBindingsHook;
        };

        devShells.cobol = pkgs.mkShell {
          packages = [
            datamunge
            pkgs.arrow-cpp
            # just prebuild-cobol: scripts/prebuild_bindings.py cobol runs rosetta.
            rosettaPkg
            pkgs.python3
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
            pkgs.just
          ]
          ++ gnuCobol;

          shellHook = localBindingsHook;
        };

        devShells.ocaml = pkgs.mkShell {
          packages = [
            datamunge
            # ocamlc/ocamlopt and caml/*.h (build-ocaml builds the stubs and the library), ocamlfind
            # (`-package datamunge`, the toplevel's `#require`), odoc for the .mli documentation,
            # dune for `(libraries datamunge)` consumers and utop for repl-ocaml.
            pkgs.ocamlPackages.ocaml
            pkgs.ocamlPackages.findlib
            pkgs.ocamlPackages.odoc
            pkgs.ocamlPackages.dune_3
            pkgs.ocamlPackages.utop
            # just prebuild-ocaml: scripts/prebuild_bindings.py ocaml runs rosetta.
            rosettaPkg
            pkgs.python3
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
            pkgs.just
            # datamungeConfig.cmake find_dependency(Arrow) for build-ocaml.
            pkgs.arrow-cpp
          ];

          shellHook = localBindingsHook;
        };

        devShells.apl = pkgs.mkShell {
          packages = [
            datamunge
            # The interpreter (apl --script, repl-apl). build-apl compiles the native functions against
            # its configured headers, ROSETTA_GNU_APL_SOURCE_DIR below.
            pkgs.gnuapl
            # just prebuild-apl: scripts/prebuild_bindings.py apl runs rosetta.
            rosettaPkg
            pkgs.python3
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
            pkgs.just
            # datamungeConfig.cmake find_dependency(Arrow) for build-apl.
            pkgs.arrow-cpp
          ];

          shellHook = ''
            export ROSETTA_GNU_APL_SOURCE_DIR=${datamungeapl.gnuAplHeaders}
          ''
          + localBindingsHook;
        };

        devShells.guile = pkgs.mkShell {
          packages = [
            datamunge
            pkgs.guile
            # just prebuild-guile: scripts/prebuild_bindings.py guile runs rosetta.
            rosettaPkg
            pkgs.python3
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
            pkgs.just
            # datamungeConfig.cmake find_dependency(Arrow) for build-guile.
            pkgs.arrow-cpp
          ];

          shellHook = localBindingsHook;
        };

        # The example reports contain only C++ and shell Babel blocks. Keep
        # their exporter independent of the unrelated language bindings.
        devShells.org-examples = pkgs.mkShell {
          packages = [
            ((pkgs.emacsPackagesFor pkgs.emacs).emacsWithPackages (epkgs: [
              epkgs.htmlize
            ]))
            (pkgs.texlive.combine {
              inherit (pkgs.texlive)
                scheme-small
                wrapfig
                ulem
                capt-of
                ;
            })
            pkgs.librsvg
            datamunge
            pkgs.pkg-config
            pkgs.stdenv.cc
          ];
          shellHook = ''
            export PKG_CONFIG_PATH="${datamunge}/lib/pkgconfig''${PKG_CONFIG_PATH:+:}$PKG_CONFIG_PATH"
            datamungeLibDir="${datamunge}/lib/datamunge-${datamunge.version}"
            export DYLD_LIBRARY_PATH="$datamungeLibDir''${DYLD_LIBRARY_PATH:+:}$DYLD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="$datamungeLibDir''${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"
            export NIX_LDFLAGS="-rpath $datamungeLibDir"
          '';
        };

        devShells.report-apl = reportShell "apl";
        devShells.report-cpp = reportShell "cpp";
        devShells.report-python = reportShell "python";
        devShells.report-r = reportShell "r";
        devShells.report-csharp = reportShell "csharp";
        devShells.report-d = reportShell "d";
        devShells.report-go = reportShell "go";
        devShells.report-guile = reportShell "guile";
        devShells.report-java = reportShell "java";
        devShells.report-javascript = reportShell "javascript";
        devShells.report-lua = reportShell "lua";
        devShells.report-ocaml = reportShell "ocaml";
        devShells.report-octave = reportShell "octave";
        devShells.report-perl = reportShell "perl";
        devShells.report-php = reportShell "php";
        devShells.report-ruby = reportShell "ruby";
        devShells.report-tcl = reportShell "tcl";

        devShells.docs-pages = pkgs.mkShell {
          packages = [
            # Doc export tooling
            pkgs.emacs
            pkgs.direnv
            pkgs.just
            pkgs.jq
            # scheme-small plus the packages Org's default LaTeX article preamble pulls in
            # that scheme-small itself doesn't carry (wrapfig/ulem/capt-of).
            (pkgs.texlive.combine {
              inherit (pkgs.texlive)
                scheme-small
                wrapfig
                ulem
                capt-of
                ;
            })
            # rsvg-convert: init.el rewrites every \includegraphics{...svg} to a
            # sibling PDF at export time (pdflatex cannot rasterize SVG itself).
            pkgs.librsvg
            # pdfinfo, used by the `view-pdf`/`org-pdf` justfile recipes to sanity-check output.
            pkgs.poppler-utils

            # Core library + pkg-config visibility
            datamunge
            pkgs.pkg-config
            pkgs.cmake
            pkgs.gnumake
            pkgs.stdenv.cc
            swigPkg

            # Language runtimes + bindings for runnable examples
            pydatamunge
            pkgs.python3
            datamungejs
            pkgs.nodejs
            datamunger
            pkgs.R
            datamungeruby
            pkgs.ruby
            pkgs.perl
            phpPackage
            datamungelua
            lua
            datamungetcl
            pkgs.tcl
            datamungejava
            pkgs.jdk
            datamungeoctave
            pkgs.octave
            datamungeguile
            pkgs.guile
            pkgs.ocamlPackages.ocaml
            pkgs.ocamlPackages.dune_3
            pkgs.ocamlPackages.findlib
            pkgs.ocamlPackages.alcotest
            pkgs.go
          ];

          shellHook = ''
            export PKG_CONFIG_PATH="${datamunge}/lib/pkgconfig''${PKG_CONFIG_PATH:+:}$PKG_CONFIG_PATH"

            export DATAMUNGE_PREFIX="${datamunge}"
            export OCTAVE_PATH="${datamungeoctave}/share/octave/site/m''${OCTAVE_PATH:+:}$OCTAVE_PATH"
            export RUBYLIB="${datamungeruby}/lib''${RUBYLIB:+:}$RUBYLIB"
            export TCLLIBPATH="${datamungetcl}/lib''${TCLLIBPATH:+ $TCLLIBPATH}"
            export DATAMUNGE_JAVA_LIBRARY="${datamungejava.library}"
            export CLASSPATH="${datamungejava.jar}''${CLASSPATH:+:}$CLASSPATH"

            export GUILE_LOAD_PATH="${datamungeguile}/share/guile/site/${pkgs.guile.effectiveVersion or "3.0"}''${GUILE_LOAD_PATH:+:}$GUILE_LOAD_PATH"
            export GUILE_LOAD_COMPILED_PATH="${datamungeguile}/lib/guile/${pkgs.guile.effectiveVersion or "3.0"}/site-ccache''${GUILE_LOAD_COMPILED_PATH:+:}$GUILE_LOAD_COMPILED_PATH"
          '';
        };
      }
    );
}
