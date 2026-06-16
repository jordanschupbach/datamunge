TARGET := "octra_ex"
BENCH_TARGET := ""
JOBS := "20"

NIX_DEVELOP := "nix develop --accept-flake-config --option eval-cache false"
BINDINGS_DIR := "src"

# {{{ run commands



run: run-cpp

test: test-cpp

format:
  {{ NIX_DEVELOP }} .#format --command bash -lc './scripts/format.sh'

format-check:
  {{ NIX_DEVELOP }} .#quality --command bash -lc './scripts/format_check.sh'

lint:
  {{ NIX_DEVELOP }} .#quality --command bash -lc './scripts/lint.sh'

quality: format-check lint

fmt: format

run-all: run-cpp run-csharp run-java run-go run-rust run-d run-python run-php run-perl run-tcl run-lua run-ruby run-r run-guile run-javascript run-ocaml run-octave

run-csharp: build-csharp
  {{ NIX_DEVELOP }} .#csharp --command bash -lc 'LD_LIBRARY_PATH="$(pwd)/build/dotnet/release/_deps/octra-build:$(pwd)/build/dotnet/release${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" dotnet run --project ./{{ BINDINGS_DIR }}/octradotnet'

run-java: build-java
  {{ NIX_DEVELOP }} .#java --command bash -lc "gradle run --no-configuration-cache --args='{{ TARGET }}'"

run-go: build-go
  {{ NIX_DEVELOP }} .#go --command bash -lc 'cd {{ BINDINGS_DIR }}/gooctra && LD_LIBRARY_PATH="$(pwd)/../../build:$LD_LIBRARY_PATH" CGO_CPPFLAGS="-I$(pwd)/../../include" CGO_LDFLAGS="-L$(pwd)/../../build -loctra" go run ../../examples/go/{{ TARGET }}.go'

run-rust: build-rust
  {{ NIX_DEVELOP }} .#rust --command bash -lc 'export LD_LIBRARY_PATH="$(pkg-config --variable=libdir octra)${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" && cargo run --manifest-path {{ BINDINGS_DIR }}/rustoctra/Cargo.toml --example octra_ex'

run-d: build-d
  bash -lc 'set -euo pipefail; \
    compiler=""; \
    if command -v ldc2 >/dev/null 2>&1; then compiler="--compiler=ldc2"; elif command -v dmd >/dev/null 2>&1; then compiler="--compiler=dmd"; fi; \
    if command -v nix >/dev/null 2>&1; then \
      {{ NIX_DEVELOP }} .#d --command bash -lc "rm -rf build/dub-packages/octrad-0.0.1 && cd examples/d && dub run $compiler --build=release"; \
    else \
      rm -rf build/dub-packages/octrad-0.0.1 && cd examples/d && dub run $compiler --build=release; \
    fi'

run-python: build-python
  {{ NIX_DEVELOP }} .#python --command bash -lc 'build/venv/pyoctra-run/bin/python examples/python/{{ TARGET }}.py'

run-php: build-php
  {{ NIX_DEVELOP }} .#php --command bash -lc 'php --php-ini .user.ini examples/php/octra_ex.php'

run-perl: build-perl
  {{ NIX_DEVELOP }} .#perl --command bash -lc 'export PERL5LIB="$(pwd)/build/perl/lib/perl5:$PERL5LIB" && export LD_LIBRARY_PATH="$(pwd)/build:$LD_LIBRARY_PATH" && perl examples/perl/octra_ex.pl'

run-tcl: build-tcl
  {{ NIX_DEVELOP }} .#tcl --command bash -lc 'export TCLLIBPATH="$(pwd)/build/octratcl${TCLLIBPATH:+ $TCLLIBPATH}" && tclsh examples/tcl/octra_ex.tcl'

run-lua: build-lua
  {{ NIX_DEVELOP }} .#lua --command bash -lc 'cmake --install build/octralua --prefix build/lua/prefix >/dev/null && export LUA_CPATH="$(pwd)/build/lua/prefix/lib/lua/?.so;$(pwd)/build/lua/prefix/lib64/lua/?.so;;" && export LUA_PATH="$(pwd)/build/lua/prefix/share/lua/?.lua;;" && export LD_LIBRARY_PATH="$(pkg-config --variable=libdir octra)${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" && lua examples/lua/octra_ex.lua'

run-ruby: build-ruby
  {{ NIX_DEVELOP }} .#ruby --command bash -lc 'export LD_LIBRARY_PATH="$(pkg-config --variable=libdir octra)${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" && ruby -I {{ BINDINGS_DIR }}/octruby/lib examples/ruby/octra_ex.rb'

run-r: build-r
   {{ NIX_DEVELOP }} .#r --command bash -lc 'R_LIBS_USER="$(pwd)/build/r/library${R_LIBS_USER:+:}$R_LIBS_USER" Rscript examples/r/octra_ex.r'

run-guile: build-guile
  {{ NIX_DEVELOP }} .#guile --command bash -lc 'cmake --install build/octraguile --prefix build/guile/prefix >/dev/null && guile_effective="$(pkg-config --variable=effective-version guile-3.0 2>/dev/null || echo 3.0)" && export GUILE_LOAD_PATH="$(pwd)/build/guile/prefix/share/guile/site/$guile_effective${GUILE_LOAD_PATH:+:}$GUILE_LOAD_PATH" && export LD_LIBRARY_PATH="$(pwd)/build/guile/prefix/lib/guile/$guile_effective/extensions:$(pwd)/build${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" && guile --no-auto-compile -s examples/guile/octra_ex.scm'

run-javascript: build-javascript
   {{ NIX_DEVELOP }} .#javascript --command bash -lc 'node ./examples/javascript/octra_ex.js'

run-ocaml:
  {{ NIX_DEVELOP }} .#ocaml --command bash -lc 'just install-ocaml && mkdir -p build/ocaml && export OCAMLPATH="$(pwd)/build/ocaml/prefix/lib${OCAMLPATH:+:}$OCAMLPATH" && ocamlfind ocamlopt -package octraocaml -linkpkg examples/ocaml/octra_ex.ml -o build/ocaml/octra_ex && ./build/ocaml/octra_ex'

run-octave: build-octave
  {{ NIX_DEVELOP }} .#octave --command bash -lc 'octave -qf --path "$(pwd)/build/octraoctave" examples/octave/octra_ex.m'

run-cpp: examples
    @echo "Running target {{ TARGET }}"
    ./build/debug/examples/{{ TARGET }}

run-benchmark:
    @echo "Running Benchmarks"
    ./build/benchmarks/${BENCH_TARGET}


# }}} run commands

# {{{ prebuild commands

prebuild-swig: prebuild-python prebuild-javascript prebuild-csharp prebuild-r prebuild-perl prebuild-ruby prebuild-tcl prebuild-lua prebuild-d prebuild-guile prebuild-octave prebuild-go prebuild-php prebuild-java prebuild-ocaml
  @echo "SWIG wrappers regenerated (with Doxygen comments enabled)"

prebuild-python:
  {{ NIX_DEVELOP }} .#cpp --command bash -lc "cd ./include && swig -doxygen -c++ -python -o ../src/octra_python_wrap.cpp -oh ../src/octra_python_wrap.h ../src/pyoctra/swig/pyoctra.i && mv ../src/octra.py ../src/pyoctra/octra.py"

prebuild-javascript:
  {{ NIX_DEVELOP }} .#cpp --command bash -lc "cd ./include && swig -javascript -typescript -napi -c++ -o ../src/octra_js_wrap.cpp -oh ../src/octra_js_wrap.h ../src/octrajs/src/octrajs.i"
  # Inject deterministic JS->C callback bridge helpers (SWIG Node backend doesn't support directors here).
  perl -0777 -pi -e 's/#include <napi.h>\n/#include <napi.h>\n#include \"octra_js_callbacks.inl\"\n/s' src/octra_js_wrap.cpp
  perl -0777 -pi -e 's/SWIG_InitializeModule\(env\);\n/SWIG_InitializeModule(env);\n  OctraJS_RegisterCallbackBridge(env, exports);\n/s' src/octra_js_wrap.cpp

prebuild-csharp:
  {{ NIX_DEVELOP }} .#cpp --command bash -lc "find ./{{ BINDINGS_DIR }}/octradotnet -type f -name '*.cs' ! -name 'Program.cs' -exec rm {} +"
  {{ NIX_DEVELOP }} .#cpp --command bash -lc "cd ./include && swig -doxygen -c++ -csharp -dllimport octra_csharp -o ../{{ BINDINGS_DIR }}/octradotnet/octra_csharp_wrap.cpp -oh ../{{ BINDINGS_DIR }}/octradotnet/octra_csharp_wrap.h ../src/octradotnet/swig/octradotnet.i"
  {{ NIX_DEVELOP }} .#cpp --command bash -lc "sed -i 's/DllImport(\"octra\"/DllImport(\"octra_csharp\"/g' ./{{ BINDINGS_DIR }}/octradotnet/octraPINVOKE.cs"

prebuild-r:
  {{ NIX_DEVELOP }} .#cpp --command bash -lc "cd ./include && swig -c++ -r -o ../src/octra_r_wrap.cpp -oh ../src/octra_r_wrap.h ../src/octrar/swig/octrar.i && mv ../src/octrar.R ../R"

prebuild-perl:
  {{ NIX_DEVELOP }} .#cpp --command bash -lc "mkdir -p {{ BINDINGS_DIR }}/perloctra/lib && swig -perl5 -c++ -Iinclude -o {{ BINDINGS_DIR }}/perloctra/Octra_wrap.cxx -oh {{ BINDINGS_DIR }}/perloctra/Octra_wrap.h -outdir {{ BINDINGS_DIR }}/perloctra/lib src/perloctra/swig/perloctra.i"

# Ruby (SWIG)
prebuild-ruby:
  {{ NIX_DEVELOP }} .#ruby --command bash -lc "mkdir -p {{ BINDINGS_DIR }}/octruby/ext/octruby {{ BINDINGS_DIR }}/octruby/lib/octruby && swig -ruby -c++ -Iinclude -o {{ BINDINGS_DIR }}/octruby/ext/octruby/octruby_wrap.cxx -oh {{ BINDINGS_DIR }}/octruby/ext/octruby/octruby_wrap.h -outdir {{ BINDINGS_DIR }}/octruby/lib/octruby src/octruby/swig/octruby.i"

# Tcl (SWIG)
prebuild-tcl:
  {{ NIX_DEVELOP }} .#tcl --command bash -lc "mkdir -p build/octratcl/swig && swig -tcl8 -c++ -Iinclude -o build/octratcl/swig/octra_tcl_wrap.cxx -oh build/octratcl/swig/octra_tcl_wrap.h src/octratcl/swig/octratcl.i"

# Lua (SWIG)
prebuild-lua:
  {{ NIX_DEVELOP }} .#lua --command bash -lc "mkdir -p build/octralua-swig && swig -lua -c++ -Iinclude -outdir build/octralua-swig -o build/octralua-swig/octra_lua_wrap.cxx -oh build/octralua-swig/octra_lua_wrap.h src/octralua/swig/octralua.i"

# D (SWIG)
prebuild-d:
  bash -lc 'set -euo pipefail; \
    cmd="mkdir -p {{ BINDINGS_DIR }}/octrad/source && swig -c++ -d -Iinclude -o {{ BINDINGS_DIR }}/octrad/source/octrad_wrap.cpp -oh {{ BINDINGS_DIR }}/octrad/source/octrad_wrap.h -outdir {{ BINDINGS_DIR }}/octrad/source src/octrad/swig/octrad.i"; \
    if command -v nix >/dev/null 2>&1 && {{ NIX_DEVELOP }} .#d --command true >/dev/null 2>&1; then \
      {{ NIX_DEVELOP }} .#d --command bash -lc "$cmd"; \
    else \
      bash -lc "$cmd"; \
    fi'

# Guile (SWIG)
prebuild-guile:
  {{ NIX_DEVELOP }} .#guile --command bash -lc "mkdir -p build/octraguile-swig && swig -guile -c++ -Iinclude -o build/octraguile-swig/octra_guile_wrap.cxx -oh build/octraguile-swig/octra_guile_wrap.h src/octraguile/swig/octraguile.i"

prebuild-octave:
  {{ NIX_DEVELOP }} .#cpp --command bash -lc "mkdir -p build/octraoctave-swig && swig -octave -c++ -Iinclude -o build/octraoctave-swig/octra_octave_wrap.cxx -oh build/octraoctave-swig/octra_octave_wrap.h src/octraoctave/swig/octraoctave.i"

prebuild-go:
  {{ NIX_DEVELOP }} .#cpp --command bash -lc "swig -go -c++ -intgosize 64 -Iinclude -o {{ BINDINGS_DIR }}/gooctra/gooctra_wrap.cxx -oh {{ BINDINGS_DIR }}/gooctra/gooctra_wrap.h -outdir {{ BINDINGS_DIR }}/gooctra src/gooctra/swig/gooctra.i"

prebuild-php:
  {{ NIX_DEVELOP }} .#cpp --command bash -lc "cd ./include && swig -c++ -php7 -o ../src/octra_php_wrap.cpp -oh ../src/octra_php_wrap.h ../src/octraPHP/swig/octraPHP.i"

prebuild-java:
    {{ NIX_DEVELOP }} .#java --command bash -lc "find {{ BINDINGS_DIR }}/joctra/src/main/java/js/octra/joctra -type f -name '*.java' ! -name 'App.java' ! -path '{{ BINDINGS_DIR }}/joctra/src/main/java/js/octra/joctra/examples/*' -exec rm {} +"
    {{ NIX_DEVELOP }} .#java --command bash -lc "rm -rf {{ BINDINGS_DIR }}/joctra-octra/build/cmake"
    {{ NIX_DEVELOP }} .#java --command bash -lc "cd ./include && swig -doxygen -c++ -java -o ../{{ BINDINGS_DIR }}/joctra-octra/octra_java_wrap.cpp -oh ../{{ BINDINGS_DIR }}/joctra-octra/octra_java_wrap.h -package js.octra.joctra -outdir ../{{ BINDINGS_DIR }}/joctra/src/main/java/js/octra/joctra ../src/joctra-octra/swig/joctra.i"
    {{ NIX_DEVELOP }} .#java --command bash -lc "sed -i 's/System.loadLibrary(\"octra\")/System.loadLibrary(\"octra_jni\")/g' {{ BINDINGS_DIR }}/joctra/src/main/java/js/octra/joctra/App.java {{ BINDINGS_DIR }}/joctra/src/main/java/js/octra/joctra/examples/StlEx.java"
    {{ NIX_DEVELOP }} .#java --command bash -lc "perl -0777 -pi -e 's/public class octra \\{/public class octra {\\n  static { System.loadLibrary(\"octra_jni\"); }/s' {{ BINDINGS_DIR }}/joctra/src/main/java/js/octra/joctra/octra.java"

prebuild-ocaml:
  {{ NIX_DEVELOP }} .#ocaml --command bash -lc "test -n \"${OCTRA_PREFIX:-}\" || (echo 'OCTRA_PREFIX is not set' >&2; exit 1) && mkdir -p {{ BINDINGS_DIR }}/octraocaml/src && swig -ocaml -c++ -Iinclude -o {{ BINDINGS_DIR }}/octraocaml/src/octra_ocaml_wrap.cxx -oh {{ BINDINGS_DIR }}/octraocaml/src/octra_ocaml_wrap.h -outdir {{ BINDINGS_DIR }}/octraocaml/src src/octraocaml/swig/octraocaml.i"

# }}} prebuild commands

# {{{ rust (bindgen) commands

prebuild-rust:
  {{ NIX_DEVELOP }} .#rust --command bash -lc 'inc="$(pkg-config --variable=includedir octra)" && bindgen "$inc/octra/octra_c.h" --allowlist-function "octra_.*" --allowlist-type "octra_.*" --no-layout-tests --rustfmt-bindings -o {{ BINDINGS_DIR }}/rustoctra/src/bindings.rs'

# }}} rust (bindgen) commands

# {{{ build commands

build-php: prebuild-php
  rm -rf build/octraPHP
  {{ NIX_DEVELOP }} .#php --command bash -lc 'cmake -S src/octraPHP -B build/octraPHP'
  {{ NIX_DEVELOP }} .#php --command bash -lc 'cmake --build build/octraPHP -j{{ JOBS }} --verbose'


build: build-debug

build-example-installed:
  {{ NIX_DEVELOP }} .#cpp --command bash -lc 'cmake -S examples/cpp -B build/debug/examples-installed -DCMAKE_BUILD_TYPE=Debug -DCMAKE_MAKE_PROGRAM=$(command -v make) -DBUILD_W_INSTALLED=ON'


build-release:
  {{ NIX_DEVELOP }} .#cpp --command bash -lc "cmake -S . -B build/release --preset=release-clang-linux-x86 -DCMAKE_MAKE_PROGRAM=$(command -v make)"
  ln -sf build/release/compile_commands.json compile_commands.json

build-debug:
  {{ NIX_DEVELOP }} .#cpp --command bash -lc "cmake -S . -B build/debug --preset=debug-clang-linux-x86 -DCMAKE_MAKE_PROGRAM=$(command -v make)"
  ln -sf build/debug/compile_commands.json compile_commands.json

build-cpp:
    @echo "Building octra"
    {{ NIX_DEVELOP }} .#cpp --command bash -lc "cmake -S . -B build"
    {{ NIX_DEVELOP }} .#cpp --command bash -lc "cmake --build build -j{{ JOBS }} --verbose"
    {{ NIX_DEVELOP }} .#cpp --command bash -lc "find ./build -name 'compile_commands.json' -exec cat {} + | jq -s add > compile_commands.json"


build-csharp: prebuild-csharp
  {{ NIX_DEVELOP }} .#csharp --command bash -lc "rm -rf build/dotnet/release"
  {{ NIX_DEVELOP }} .#csharp --command bash -lc "cmake -S ./{{ BINDINGS_DIR }}/octradotnet -B build/dotnet/release -DCMAKE_MAKE_PROGRAM=$(command -v make)"
  {{ NIX_DEVELOP }} .#csharp --command bash -lc "cmake --build build/dotnet/release -j{{ JOBS }} --verbose"
  {{ NIX_DEVELOP }} .#csharp --command bash -lc "cd ./{{ BINDINGS_DIR }}/octradotnet && dotnet build"


build-javascript: prebuild-javascript
    {{ NIX_DEVELOP }} .#jsbuild --command bash -lc "npm --prefix . run build"

build-python: prebuild-python
  rm -rf build/venv/pyoctra-run
  {{ NIX_DEVELOP }} .#python --command bash -lc 'python -m venv --system-site-packages build/venv/pyoctra-run'
  {{ NIX_DEVELOP }} .#python --command bash -lc 'build/venv/pyoctra-run/bin/python -m pip install -e . --no-build-isolation'


build-java: prebuild-java
  {{ NIX_DEVELOP }} .#java --command bash -lc "gradle cmakeBuild"
  {{ NIX_DEVELOP }} .#java --command bash -lc "gradle build"

build-dotnet:
    {{ NIX_DEVELOP }} . --command bash -lc "cmake -S {{ BINDINGS_DIR }}/octradotnet -B build/octradotnet"
    {{ NIX_DEVELOP }} . --command bash -lc "cmake --build build/octradotnet"
    # nix develop ./octradotnet --command bash -c "just --justfile ./octradotnet/justfile build"

build-go: prebuild-go build-cpp
  # For Go bindings, we need to run go build on the generated files
  # Note: This assumes the SWIG-generated files are already in place from prebuild-go
  {{ NIX_DEVELOP }} .#go --command bash -lc 'cd {{ BINDINGS_DIR }}/gooctra && CGO_CPPFLAGS="-I$(pwd)/../../include" CGO_LDFLAGS="-L$(pwd)/../../build -loctra" go build'

build-d: prebuild-d
  bash -lc 'set -euo pipefail; \
    compiler=""; \
    if command -v ldc2 >/dev/null 2>&1; then compiler="--compiler=ldc2"; elif command -v dmd >/dev/null 2>&1; then compiler="--compiler=dmd"; fi; \
    if command -v nix >/dev/null 2>&1 && {{ NIX_DEVELOP }} .#d --command true >/dev/null 2>&1; then \
      {{ NIX_DEVELOP }} .#d --command bash -lc "cd {{ BINDINGS_DIR }}/octrad && dub build $compiler --build=release --force"; \
    else \
      cd {{ BINDINGS_DIR }}/octrad && dub build $compiler --build=release --force; \
    fi'

build-perl: prebuild-perl build-cpp
  {{ NIX_DEVELOP }} .#perl --command bash -lc 'cd {{ BINDINGS_DIR }}/perloctra && rm -rf blib Makefile Makefile.old pm_to_blib MYMETA.* && perl Makefile.PL INSTALL_BASE="$(pwd)/../../build/perl" && make -j{{ JOBS }} && make install'

build-ruby: prebuild-ruby
  {{ NIX_DEVELOP }} .#ruby --command bash -lc "set -euo pipefail; cd {{ BINDINGS_DIR }}/octruby/ext/octruby && ruby extconf.rb && make -j1 && so=\"\$(find . -type f -name 'octruby*.so' -print -quit)\" && test -n \"\$so\" && mkdir -p ../../lib/octruby && cp -f \"\$so\" ../../lib/octruby/octruby.so"

build-r: prebuild-r
  rm -rf build/r/library
  {{ NIX_DEVELOP }} .#r --command bash -lc 'mkdir -p build/r/library && R CMD INSTALL -l build/r/library .'

build-tcl: prebuild-tcl
  rm -rf build/octratcl
  {{ NIX_DEVELOP }} .#tcl --command bash -lc 'cmake -S src/octratcl -B build/octratcl -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(pkg-config --variable=prefix octra)"'
  {{ NIX_DEVELOP }} .#tcl --command bash -lc 'cmake --build build/octratcl -j{{ JOBS }} --verbose'
  {{ NIX_DEVELOP }} .#tcl --command bash -lc 'cp -v {{ BINDINGS_DIR }}/octratcl/pkgIndex.tcl build/octratcl/'

build-lua: prebuild-lua
  rm -rf build/octralua
  {{ NIX_DEVELOP }} .#lua --command bash -lc 'cmake -S src/octralua -B build/octralua -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(pkg-config --variable=prefix octra)"'
  {{ NIX_DEVELOP }} .#lua --command bash -lc 'cmake --build build/octralua -j{{ JOBS }} --verbose'

build-rust:
  {{ NIX_DEVELOP }} .#rust --command bash -lc 'cargo build --manifest-path {{ BINDINGS_DIR }}/rustoctra/Cargo.toml'

build-guile: prebuild-guile
  rm -rf build/octraguile
  {{ NIX_DEVELOP }} .#guile --command bash -lc 'cmake -S src/octraguile -B build/octraguile -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(pkg-config --variable=prefix octra)"'
  {{ NIX_DEVELOP }} .#guile --command bash -lc 'cmake --build build/octraguile -j{{ JOBS }} --verbose'

build-octave: prebuild-octave
  {{ NIX_DEVELOP }} .#octave --command bash -lc 'cmake -S src/octraoctave -B build/octraoctave -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(pkg-config --variable=prefix octra)"'
  {{ NIX_DEVELOP }} .#octave --command bash -lc 'cmake --build build/octraoctave -j{{ JOBS }} --verbose'

build-ocaml: prebuild-ocaml
  {{ NIX_DEVELOP }} .#ocaml --command bash -lc 'test -n "${OCTRA_PREFIX:-}" || (echo "OCTRA_PREFIX is not set" >&2; exit 1) && cd {{ BINDINGS_DIR }}/octraocaml && dune build'

install-ocaml: build-ocaml
  {{ NIX_DEVELOP }} .#ocaml --command bash -lc 'test -n "${OCTRA_PREFIX:-}" || (echo "OCTRA_PREFIX is not set" >&2; exit 1) && mkdir -p build/ocaml/prefix && cd {{ BINDINGS_DIR }}/octraocaml && dune install --prefix "$(pwd)/../../build/ocaml/prefix"'




# }}} build commands

# {{{ repl commands

repl-javascript: prebuild-javascript build-javascript
  {{ NIX_DEVELOP }} .#javascript --command bash -lc 'node'

repl-python: prebuild-python
  {{ NIX_DEVELOP }} .#python --command bash -lc 'ipython'

repl-r: prebuild-r
  {{ NIX_DEVELOP }} .#r --command bash -lc 'R'

repl-php: build-php
  {{ NIX_DEVELOP }} .#php --command bash -lc 'php -a --php-ini .user.ini'

repl-perl: build-perl
  {{ NIX_DEVELOP }} .#perl --command bash -lc 'export PERL5LIB="$(pwd)/build/perl/lib/perl5:$PERL5LIB" && export LD_LIBRARY_PATH="$(pwd)/build:$LD_LIBRARY_PATH" && perl -de 1'

repl-csharp: build-csharp
  {{ NIX_DEVELOP }} .#csharp --command bash -lc 'LD_LIBRARY_PATH="$(pwd)/build/dotnet/release/_deps/octra-build:$(pwd)/build/dotnet/release${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" if command -v dotnet-repl >/dev/null 2>&1; then dotnet-repl; elif command -v csi >/dev/null 2>&1; then csi; else echo "No C# REPL found (expected dotnet-repl or csi)" >&2; exit 1; fi'

repl-java:
  {{ NIX_DEVELOP }} .#java --command bash -lc 'export LD_LIBRARY_PATH={{ BINDINGS_DIR }}/joctra-octra/build/cmake:$LD_LIBRARY_PATH && jshell --class-path ./{{ BINDINGS_DIR }}/joctra/build/libs/joctra.jar'

repl-cpp:
  {{ NIX_DEVELOP }} .#cpp --command bash -lc 'cling $(pkg-config --cflags octra) $(pkg-config --libs-only-L octra) -loctra -std=c++17'

repl-tcl: build-tcl
  {{ NIX_DEVELOP }} .#tcl --command bash -lc 'export TCLLIBPATH="$(pwd)/build/octratcl${TCLLIBPATH:+ $TCLLIBPATH}" && tclsh'

repl-lua:
  {{ NIX_DEVELOP }} .#lua --command bash -lc 'lua'

repl-ruby:
  {{ NIX_DEVELOP }} .#ruby --command bash -lc 'irb -r octruby'

repl-ocaml:
  {{ NIX_DEVELOP }} .#ocaml --command bash -lc 'utop'

repl-guile:
  {{ NIX_DEVELOP }} .#guile --command bash -lc 'guile'

repl-rust:
  {{ NIX_DEVELOP }} .#rust --command bash -lc 'export LD_LIBRARY_PATH="$(pkg-config --variable=libdir octra)${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" && cd {{ BINDINGS_DIR }}/rustoctra && evcxr'

repl-octave:
  {{ NIX_DEVELOP }} .#octave --command bash -lc 'octraoctave_prefix="$(nix eval --raw .#octraoctave)" && octave -qf --path "$octraoctave_prefix/share/octave/site/m"'

repl-d:
  bash -lc 'set -euo pipefail; \
    compiler=""; \
    if command -v ldc2 >/dev/null 2>&1; then compiler="--compiler=ldc2"; elif command -v dmd >/dev/null 2>&1; then compiler="--compiler=dmd"; fi; \
    if command -v nix >/dev/null 2>&1 && {{ NIX_DEVELOP }} .#d --command true >/dev/null 2>&1; then \
      {{ NIX_DEVELOP }} .#d --command bash -lc "cd examples/d && dub run $compiler --build=release"; \
    else \
      cd examples/d && dub run $compiler --build=release; \
    fi'

# }}} repl commands

# {{{ test commands

test-python-build: prebuild-python
  rm -rf build/venv/pyoctra-build
  {{ NIX_DEVELOP }} .#python --command bash -lc 'python -m venv --system-site-packages build/venv/pyoctra-build'
  {{ NIX_DEVELOP }} .#python --command bash -lc 'build/venv/pyoctra-build/bin/python setup.py sdist bdist_wheel'

test-python: prebuild-python
  rm -rf build/venv/pyoctra
  {{ NIX_DEVELOP }} .#python --command bash -lc 'python -m venv --system-site-packages build/venv/pyoctra'
  {{ NIX_DEVELOP }} .#python --command bash -lc 'build/venv/pyoctra/bin/python -m pip install -e . --no-build-isolation'
  {{ NIX_DEVELOP }} .#python --command bash -lc 'build/venv/pyoctra/bin/python -m pytest -q tests/python'

test-r: prebuild-r
  {{ NIX_DEVELOP }} .#r --command bash -lc 'R -q -e "testthat::test_local(\".\")"'

test-csharp: build-csharp
  {{ NIX_DEVELOP }} .#csharp --command bash -lc 'LD_LIBRARY_PATH="$(pwd)/build/dotnet/release/_deps/octra-build:$(pwd)/build/dotnet/release${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" dotnet test ./{{ BINDINGS_DIR }}/octradotnet.tests'

test-java: build-java
  {{ NIX_DEVELOP }} .#java --command bash -lc 'export LD_LIBRARY_PATH={{ BINDINGS_DIR }}/joctra-octra/build/cmake:$LD_LIBRARY_PATH && gradle test'

test-rust:
  {{ NIX_DEVELOP }} .#rust --command bash -lc 'export LD_LIBRARY_PATH="$(pkg-config --variable=libdir octra)${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" && cargo test --manifest-path tests/rust/Cargo.toml'

test-php: build-php
  {{ NIX_DEVELOP }} .#php --command bash -lc 'php -d assert.exception=1 -d zend.assertions=1 --php-ini .user.ini tests/php/test_octra.php'

test-lua:
  {{ NIX_DEVELOP }} .#lua --command bash -lc 'lua tests/lua/test_octra.lua'

test-perl: build-perl
  {{ NIX_DEVELOP }} .#perl --command bash -lc 'export PERL5LIB="$(pwd)/build/perl/lib/perl5:$PERL5LIB" && export LD_LIBRARY_PATH="$(pwd)/build:$LD_LIBRARY_PATH" && prove -l tests/perl'

test-tcl: build-tcl
  {{ NIX_DEVELOP }} .#tcl --command bash -lc 'export TCLLIBPATH="$(pwd)/build/octratcl${TCLLIBPATH:+ $TCLLIBPATH}" && tclsh tests/tcl/test_octra.tcl'

test-ruby:
  {{ NIX_DEVELOP }} .#ruby --command bash -lc 'export LD_LIBRARY_PATH="$(pkg-config --variable=libdir octra)${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" && ruby -I tests/ruby -e "require \"test_octra\""'

test-guile:
  {{ NIX_DEVELOP }} .#guile --command bash -lc 'guile --no-auto-compile -s tests/guile/test_octra.scm'

test-octave:
  {{ NIX_DEVELOP }} .#octave --command bash -lc 'octraoctave_prefix="$(nix eval --raw .#octraoctave)" && octave -qf --path "$octraoctave_prefix/share/octave/site/m" --eval '"'"'test("tests/octave/test_octra.m")'"'"''

test-d: prebuild-d
  bash -lc 'set -euo pipefail; \
    compiler=""; \
    if command -v ldc2 >/dev/null 2>&1; then compiler="--compiler=ldc2"; elif command -v dmd >/dev/null 2>&1; then compiler="--compiler=dmd"; fi; \
    if command -v nix >/dev/null 2>&1 && {{ NIX_DEVELOP }} .#d --command true >/dev/null 2>&1; then \
      {{ NIX_DEVELOP }} .#d --command bash -lc "dub test --root tests/d $compiler --build=release"; \
    else \
      dub test --root tests/d $compiler --build=release; \
    fi'


test-go: build-go
    {{ NIX_DEVELOP }} .#go --command bash -lc 'cd {{ BINDINGS_DIR }}/gooctra && LD_LIBRARY_PATH="$(pwd)/../../build:$LD_LIBRARY_PATH" CGO_CPPFLAGS="-I$(pwd)/../../include" CGO_LDFLAGS="-L$(pwd)/../../build -loctra" go test ./...'


test-javascript: build-javascript
    @echo "Running Javascript Tests"
    {{ NIX_DEVELOP }} .#javascript --command bash -lc "npm run test"

test-ocaml:
  {{ NIX_DEVELOP }} .#ocaml --command bash -lc 'just install-ocaml && export OCAMLPATH="$(pwd)/build/ocaml/prefix/lib${OCAMLPATH:+:}$OCAMLPATH" && cd tests/ocaml && dune runtest'


test-cpp:
    @echo "Running Tests"
    @bash -lc 'set -euo pipefail; \
      gtest_prefix=""; \
      for p in /nix/store/*-gtest-*-dev; do \
        if [ -f "$p/lib/cmake/GTest/GTestConfig.cmake" ]; then gtest_prefix="$p"; break; fi; \
      done; \
      rm -rf build/debug/tests; \
      cmake -S tests -B build/debug/tests -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_MAKE_PROGRAM="$(command -v make)" ${gtest_prefix:+-DCMAKE_PREFIX_PATH="$gtest_prefix"}; \
      cmake --build build/debug/tests -j{{ JOBS }} --verbose; \
      find ./build/ -name "compile_commands.json" -exec cat {} + | jq -s add > compile_commands.json; \
      ctest --test-dir build/debug/tests --output-on-failure'


# }}} test commands

# {{{ utilities

rename NEW:
  ./rename_octra {{ NEW }}

jq:
    {{ NIX_DEVELOP }} . --command bash -lc "find ./build -name 'compile_commands.json' -exec cat {} + | jq -s add > compile_commands.json"


playground:
    @echo "Building playground"
    {{ NIX_DEVELOP }} .#cpp --command bash -lc "cmake -S playground -B build/debug/playground"
    {{ NIX_DEVELOP }} .#cpp --command bash -lc "cmake --build build/debug/playground -j {{JOBS}} --verbose"
    ./build/debug/playground/playground_cpp

flamechart:
    @echo "Running Performance Tests"
    @perf record -F 99 -g ./build/examples/${TARGET}
    @perf script > out.perf
    @if [ ! -d "Flamegraph" ]; then \
    	git clone https://github.com/brendangregg/Flamegraph.git; \
    fi
    @./Flamegraph/stackcollapse-perf.pl out.perf > out.folded
    @./Flamegraph/flamegraph.pl out.folded > flamegraph.svg


benchmark:
    @echo "Building Examples"
    cmake -S benchmarks -B build/benchmarks
    cmake --build build/benchmarks -j${JOBS} --verbose


memcheck:
  valgrind --leak-check=full --track-origins=yes ./build/debug/examples/{{TARGET}}


coverage:
    rm -rf build/coverage
    @bash -lc 'set -euo pipefail; \
      gtest_prefix=""; \
      for p in /nix/store/*-gtest-*-dev; do \
        if [ -f "$p/lib/cmake/GTest/GTestConfig.cmake" ]; then gtest_prefix="$p"; break; fi; \
      done; \
      cmake -S tests -B build/coverage/tests -DCMAKE_BUILD_TYPE=Debug -DENABLE_TEST_COVERAGE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_MAKE_PROGRAM="$(command -v make)" ${gtest_prefix:+-DCMAKE_PREFIX_PATH="$gtest_prefix"}; \
      cmake --build build/coverage/tests -j{{ JOBS }} --verbose; \
      ctest --test-dir build/coverage/tests --output-on-failure'
    mkdir -p build/coverage
    lcov -c --rc lcov_function_coverage=0 --ignore-errors mismatch,inconsistent --directory build/coverage/tests --output-file build/coverage/coverage.info
    lcov --ignore-errors inconsistent --ignore-errors unused -r build/coverage/coverage.info '/nix/store/*' --output-file build/coverage/coverage.info
    lcov --ignore-errors inconsistent --ignore-errors unused -r build/coverage/coverage.info '*/_deps/*' --output-file build/coverage/coverage.info
    genhtml --rc genhtml_function_coverage=0 --ignore-errors inconsistent --ignore-errors corrupt build/coverage/coverage.info --output-directory build/coverage/html

test-coverage: coverage
    @bash -lc 'set -euo pipefail; \
      info="build/coverage/coverage.info"; \
      test -f "$info"; \
      awk -v want1="$(pwd)/src/octra/octra.cpp" -v want2="$(pwd)/src/octra/octra_c.cpp" '\'' \
        function finish() { \
          if (!in_wanted) return; \
          if (total == 0) { \
            printf("coverage: %s: no line data found\\n", file) > "/dev/stderr"; \
            exit 2; \
          } \
          if (hit != total) { \
            printf("coverage: %s: %d/%d lines (%.2f%%)\\n", file, hit, total, (100.0*hit/total)) > "/dev/stderr"; \
            exit 1; \
          } \
        } \
        /^SF:/ { \
          finish(); \
          file = substr($0, 4); \
          in_wanted = (file == want1 || file == want2); \
          total = 0; hit = 0; \
          next; \
        } \
        /^DA:/ { \
          if (!in_wanted) next; \
          split(substr($0, 4), a, ","); \
          total++; \
          if (a[2] + 0 > 0) hit++; \
          next; \
        } \
        /^end_of_record/ { finish(); in_wanted=0; next } \
        END { finish(); print "coverage: OK (octra.cpp and octra_c.cpp are 100%)" } \
      '\'' "$info"'



debuggable:
  {{ NIX_DEVELOP }} .#cpp --command bash -lc 'clang++ -g -O0 debug.cpp -o debug $(pkg-config --cflags octra) $(pkg-config --libs-only-L octra) $(pkg-config --cflags libxml-2.0) $(pkg-config --libs-only-L libxml-2.0) -std=c++20 -loctra -lxml2'


clean:
    rm -rf build/
    rm -rf {{ BINDINGS_DIR }}/octradotnet/bin
    rm -rf {{ BINDINGS_DIR }}/octradotnet/obj
    rm -rf {{ BINDINGS_DIR }}/joctra-octra/build/
    rm -rf {{ BINDINGS_DIR }}/joctra/build/


# }}} utilities

# {{{ docs commands

prebuild-docs-pages:
    @echo "Exporting Org pages -> Markdown"
    @bash -lc 'set -euo pipefail; \
      export_cmd='\''set -euo pipefail; shopt -s nullglob; for f in docs/org/pages/*.org; do base="$(basename "$f" .org)"; out="docs/pages/${base}.md"; emacs --batch -Q -l docs/org-to-md.el -- "$f" "$out"; done'\''; \
      if command -v nix >/dev/null 2>&1; then \
        if nix develop --accept-flake-config .#docs-pages --command bash -lc "$export_cmd" >/dev/null 2>&1; then \
          nix develop --accept-flake-config .#docs-pages --command bash -lc "$export_cmd"; \
        else \
          bash -lc "$export_cmd"; \
        fi; \
      else \
        bash -lc "$export_cmd"; \
      fi'

docs: build
    @echo "Building docs"
    just prebuild-docs-pages
    {{ NIX_DEVELOP }} .#cpp --command bash -lc "cmake -S docs -B build/debug/docs"
    {{ NIX_DEVELOP }} .#cpp --command bash -lc "cmake --build build/debug/docs --target GenerateDocs"

docs-bindings: prebuild-swig
    @echo "Binding documentation is emitted as SWIG-generated docstrings/comments (per language)."

docs-all: docs docs-bindings
    @echo "Core + binding docs are up to date."

# }}} docs commands

# {{{ example commands

examples:
    @echo "Building Examples"
    {{ NIX_DEVELOP }} .#cpp --command bash -lc "cmake -S examples/cpp -B build/debug/examples --preset=debug -DCMAKE_MAKE_PROGRAM=$(command -v make) -DBUILD_W_INSTALLED=OFF"
    {{ NIX_DEVELOP }} .#cpp --command bash -lc "cmake --build build/debug/examples -j{{ JOBS }} --verbose"
    {{ NIX_DEVELOP }} .#cpp --command bash -lc "find ./build -name 'compile_commands.json' -exec cat {} + | jq -s add > compile_commands.json"
    # nix develop . --command bash -c "make -C ./build/debug/examples -j10 --verbose"

example EXAMPLE:
    @echo "Running Example {{ EXAMPLE }}:"
    ./build/debug/examples/{{ EXAMPLE }}

example-python:
    {{ NIX_DEVELOP }} .#python --command bash -lc "python examples/python/octra_ex.py"

# }}} example commands

# {{{ view commands

view-flamechart:
    $(BROWSER) ./flamegraph.svg

view-docs:
    @echo "Opening docs"
    qutebrowser ./build/debug/docs/doxygen/html/index.html

# }}} view commands

# {{{ windows specifics

windows-run: build
    @echo "Running target ${TARGET}"
    build/debug/examples/${TARGET}

# }}} windows specifics

# {{{ all commands

test-nix:
  nix flake check --accept-flake-config -L

build-nix PACKAGE="octra":
  nix build --accept-flake-config ".#{{ PACKAGE }}"

update-java-deps:
  nix/update-joctra-gradle-deps.sh

update-csharp-deps:
  nix/update-nuget-deps.sh

all-test:
  bash ./scripts/test_all.sh

test-all: all-test

# }}} all commands
