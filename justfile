TARGET := "glm_iris_ex"
BENCH_TARGET := "linalg_bench"
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

# dotnet run always compiles whatever's in datamungedotnet/Program.cs (no per-example TARGET
# mechanism, like D's dub) -- copy the selected example over Program.cs before building, same
# pattern as run-d's examples-src/ -> source/app.d copy.
run-csharp: prebuild-csharp
  {{ NIX_DEVELOP }} .#csharp --command bash -lc "rm -rf build/dotnet/release"
  {{ NIX_DEVELOP }} .#csharp --command bash -lc "cmake -S ./{{ BINDINGS_DIR }}/datamungedotnet -B build/dotnet/release -DCMAKE_MAKE_PROGRAM=$(command -v make)"
  {{ NIX_DEVELOP }} .#csharp --command bash -lc "cmake --build build/dotnet/release -j{{ JOBS }} --verbose"
  cp examples/csharp/{{ TARGET }}.cs {{ BINDINGS_DIR }}/datamungedotnet/Program.cs
  {{ NIX_DEVELOP }} .#csharp --command bash -lc 'LD_LIBRARY_PATH="$(pwd)/build/dotnet/release/_deps/datamunge-build:$(pwd)/build/dotnet/release${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" dotnet run --project ./{{ BINDINGS_DIR }}/datamungedotnet'

run-java: build-java
  {{ NIX_DEVELOP }} .#java --command bash -lc "gradle run --no-configuration-cache --args='{{ TARGET }}'"

run-go: build-go
  {{ NIX_DEVELOP }} .#go --command bash -lc 'cd {{ BINDINGS_DIR }}/godatamunge && LD_LIBRARY_PATH="$(pwd)/../../build:$LD_LIBRARY_PATH" CGO_CPPFLAGS="-I$(pwd)/../../include" CGO_LDFLAGS="-L$(pwd)/../../build -ldatamunge" go run ../../examples/go/{{ TARGET }}.go'

run-rust: build-rust
  {{ NIX_DEVELOP }} .#rust --command bash -lc 'export LD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" && cargo run --manifest-path {{ BINDINGS_DIR }}/rustdatamunge/Cargo.toml --example {{ TARGET }}'

run-d: build-d
  bash -lc 'set -euo pipefail; \
    compiler=""; \
    if command -v ldc2 >/dev/null 2>&1; then compiler="--compiler=ldc2"; elif command -v dmd >/dev/null 2>&1; then compiler="--compiler=dmd"; fi; \
    cp examples/d/examples-src/{{ TARGET }}.d examples/d/source/app.d; \
    if command -v nix >/dev/null 2>&1; then \
      {{ NIX_DEVELOP }} .#d --command bash -lc "rm -rf build/dub-packages/datamunged-0.0.1 && cd examples/d && dub run $compiler --build=release"; \
    else \
      rm -rf build/dub-packages/datamunged-0.0.1 && cd examples/d && dub run $compiler --build=release; \
    fi'

run-python: build-python
  {{ NIX_DEVELOP }} .#python --command bash -lc 'build/venv/pydatamunge-run/bin/python examples/python/{{ TARGET }}.py'

run-php: build-php
  {{ NIX_DEVELOP }} .#php --command bash -lc 'php --php-ini .user.ini examples/php/{{ TARGET }}.php'

run-perl: build-perl
  {{ NIX_DEVELOP }} .#perl --command bash -lc 'export PERL5LIB="$(pwd)/build/perl/lib/perl5:$PERL5LIB" && export LD_LIBRARY_PATH="$(pwd)/build:$LD_LIBRARY_PATH" && perl examples/perl/{{ TARGET }}.pl'

run-tcl: build-tcl
  {{ NIX_DEVELOP }} .#tcl --command bash -lc 'export TCLLIBPATH="$(pwd)/build/datamungetcl${TCLLIBPATH:+ $TCLLIBPATH}" && tclsh examples/tcl/{{ TARGET }}.tcl'

run-lua: build-lua
  {{ NIX_DEVELOP }} .#lua --command bash -lc 'cmake --install build/datamungelua --prefix build/lua/prefix >/dev/null && export LUA_CPATH="$(pwd)/build/lua/prefix/lib/lua/?.so;$(pwd)/build/lua/prefix/lib64/lua/?.so;;" && export LUA_PATH="$(pwd)/build/lua/prefix/share/lua/?.lua;;" && export LD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" && lua examples/lua/{{ TARGET }}.lua'

run-ruby: build-ruby
  {{ NIX_DEVELOP }} .#ruby --command bash -lc 'export LD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" && ruby -I {{ BINDINGS_DIR }}/octruby/lib examples/ruby/{{ TARGET }}.rb'

run-r: build-r
   {{ NIX_DEVELOP }} .#r --command bash -lc 'R_LIBS_USER="$(pwd)/build/r/library${R_LIBS_USER:+:}$R_LIBS_USER" Rscript examples/r/{{ TARGET }}.r'

run-guile: build-guile
  {{ NIX_DEVELOP }} .#guile --command bash -lc 'cmake --install build/datamungeguile --prefix build/guile/prefix >/dev/null && guile_effective="$(pkg-config --variable=effective-version guile-3.0 2>/dev/null || echo 3.0)" && export GUILE_LOAD_PATH="$(pwd)/build/guile/prefix/share/guile/site/$guile_effective${GUILE_LOAD_PATH:+:}$GUILE_LOAD_PATH" && export LD_LIBRARY_PATH="$(pwd)/build/guile/prefix/lib/guile/$guile_effective/extensions:$(pwd)/build${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" && guile --no-auto-compile -s examples/guile/{{ TARGET }}.scm'

run-javascript: build-javascript
   {{ NIX_DEVELOP }} .#javascript --command bash -lc 'node ./examples/javascript/{{ TARGET }}.js'

run-ocaml:
  {{ NIX_DEVELOP }} .#ocaml --command bash -lc 'just install-ocaml && mkdir -p build/ocaml && export OCAMLPATH="$(pwd)/build/ocaml/prefix/lib${OCAMLPATH:+:}$OCAMLPATH" && ocamlfind ocamlopt -package datamungeocaml -linkpkg examples/ocaml/{{ TARGET }}.ml -o build/ocaml/{{ TARGET }} && ./build/ocaml/{{ TARGET }}'

run-octave: build-octave
  {{ NIX_DEVELOP }} .#octave --command bash -lc 'octave -qf --path "$(pwd)/build/datamungeoctave" examples/octave/{{ TARGET }}.m'

run-cpp: examples
    @echo "Running target {{ TARGET }}"
    {{ NIX_DEVELOP }} .#cpp --command bash -lc './build/debug/examples/{{ TARGET }}'

run-plot: examples
    @echo "Running plot example"
    {{ NIX_DEVELOP }} .#cpp --command bash -lc './build/debug/examples/plot_ex'

view-plot: examples
    @echo "Running plot example and opening the scatter plot"
    {{ NIX_DEVELOP }} .#cpp --command bash -lc './build/debug/examples/plot_ex'

run-benchmark:
    @echo "Running Benchmarks"
    ./build/benchmarks/${BENCH_TARGET}


# }}} run commands

# {{{ prebuild commands

prebuild-swig: prebuild-python prebuild-javascript prebuild-csharp prebuild-r prebuild-perl prebuild-ruby prebuild-tcl prebuild-lua prebuild-d prebuild-guile prebuild-octave prebuild-go prebuild-php prebuild-java prebuild-ocaml
  @echo "SWIG wrappers regenerated (with Doxygen comments enabled)"

prebuild-python:
  {{ NIX_DEVELOP }} .#cpp --command bash -lc "cd ./include && swig -doxygen -c++ -python -o ../src/datamunge_python_wrap.cpp -oh ../src/datamunge_python_wrap.h ../src/pydatamunge/swig/pydatamunge.i && mv ../src/datamunge.py ../src/pydatamunge/datamunge.py"

prebuild-javascript:
  {{ NIX_DEVELOP }} .#cpp --command bash -lc "cd ./include && swig -javascript -typescript -napi -c++ -o ../src/datamunge_js_wrap.cpp -oh ../src/datamunge_js_wrap.h ../src/datamungejs/src/datamungejs.i"
  # Inject deterministic JS->C callback bridge helpers (SWIG Node backend doesn't support directors here).
  perl -0777 -pi -e 's/#include <napi.h>\n/#include <napi.h>\n#include \"datamunge_js_callbacks.inl\"\n/s' src/datamunge_js_wrap.cpp
  perl -0777 -pi -e 's/SWIG_InitializeModule\(env\);\n/SWIG_InitializeModule(env);\n  DatamungeJS_RegisterCallbackBridge(env, exports);\n/s' src/datamunge_js_wrap.cpp

prebuild-csharp:
  {{ NIX_DEVELOP }} .#cpp --command bash -lc "find ./{{ BINDINGS_DIR }}/datamungedotnet -type f -name '*.cs' ! -name 'Program.cs' -exec rm {} +"
  {{ NIX_DEVELOP }} .#cpp --command bash -lc "cd ./include && swig -doxygen -c++ -csharp -dllimport datamunge_csharp -o ../{{ BINDINGS_DIR }}/datamungedotnet/datamunge_csharp_wrap.cpp -oh ../{{ BINDINGS_DIR }}/datamungedotnet/datamunge_csharp_wrap.h ../src/datamungedotnet/swig/datamungedotnet.i"
  {{ NIX_DEVELOP }} .#cpp --command bash -lc "sed -i 's/DllImport(\"datamunge\"/DllImport(\"datamunge_csharp\"/g' ./{{ BINDINGS_DIR }}/datamungedotnet/datamungePINVOKE.cs"

prebuild-r:
  {{ NIX_DEVELOP }} .#cpp --command bash -lc "cd ./include && swig -c++ -r -o ../src/datamunge_r_wrap.cpp -oh ../src/datamunge_r_wrap.h ../src/datamunger/swig/datamunger.i && mv ../src/datamunger.R ../R"
  # Work around a swig-jse R-backend codegen bug: for a namespace-level `enum class` (not
  # nested inside a class -- nested enums like plot::DataSeries::Kind are unaffected),
  # defineEnumeration()'s .values=c(...) table calls a differently-named (and never-generated)
  # .Call symbol than the enum's own per-member accessor functions use, so the package fails
  # to load at all. Insert the missing doubled prefix (e.g. R_swig_TrendType_None_get ->
  # R_swig_TrendType_TrendType_None_get) for every such enum.
  perl -0777 -pi -e "s/'R_swig_TrendType_(?!TrendType_)/'R_swig_TrendType_TrendType_/g" R/datamunger.R
  perl -0777 -pi -e "s/'R_swig_SeasonalType_(?!SeasonalType_)/'R_swig_SeasonalType_SeasonalType_/g" R/datamunger.R
  perl -0777 -pi -e "s/'R_swig_Alternative_(?!Alternative_)/'R_swig_Alternative_Alternative_/g" R/datamunger.R
  perl -0777 -pi -e "s/'R_swig_PAdjustMethod_(?!PAdjustMethod_)/'R_swig_PAdjustMethod_PAdjustMethod_/g" R/datamunger.R
  # Work around a second swig-jse R-backend bug: std::vector<std::size_t> (used throughout
  # this codebase, vs. the bare std::vector<size_t> the SizeVector %template/%apply fix is
  # keyed to -- see the size_t %apply notes elsewhere in this file) gets its own, never-
  # registered S4 class name for RETURN values specifically (parameters are fine via the
  # %apply fix; only the return-value class registration differs). Point every reference at
  # the class that's actually registered.
  perl -0777 -pi -e "s/_p_std__vectorT_std__size_t_std__allocatorT_std__size_t_t_t/_p_std__vectorT_size_t_t/g" R/datamunger.R
  # Third swig-jse R-backend bug, layered on top of the above: for a std::vector<size_t>
  # RETURN VALUE (as opposed to a constructor that creates a genuine new SizeVector object),
  # the underlying .Call already returns a plain R integer vector directly (matching how
  # vector<double> returns work -- no pointer involved), so wrapping it in
  # new("_p_std__vectorT_size_t_t", ref=ans) is wrong and throws an "invalid object for slot
  # ref" error. Strip that erroneous wrapping. Genuine SizeVector-constructing calls are
  # unaffected because they're followed by a reg.finalizer(...) line that this pattern doesn't
  # match, so only by-value vector<size_t> returns (e.g. KMeans_labels, DataFrame_shape,
  # Tensor_shape) get unwrapped.
  perl -0777 -pi -e "s/;ans = (\.Call\('R_swig_[^\n]*?PACKAGE='datamunger'\));\n\s*ans <- if \(is\.null\(ans\)\) ans\n\s*else new\(\"_p_std__vectorT_size_t_t\", ref=ans\);\n\s*\n\s*ans\n/;\$1;\n/g" R/datamunger.R

prebuild-perl:
  {{ NIX_DEVELOP }} .#cpp --command bash -lc "mkdir -p {{ BINDINGS_DIR }}/perldatamunge/lib && swig -perl5 -c++ -Iinclude -o {{ BINDINGS_DIR }}/perldatamunge/Datamunge_wrap.cxx -oh {{ BINDINGS_DIR }}/perldatamunge/Datamunge_wrap.h -outdir {{ BINDINGS_DIR }}/perldatamunge/lib src/perldatamunge/swig/perldatamunge.i"

# Ruby (SWIG)
prebuild-ruby:
  {{ NIX_DEVELOP }} .#ruby --command bash -lc "mkdir -p {{ BINDINGS_DIR }}/octruby/ext/octruby {{ BINDINGS_DIR }}/octruby/lib/octruby && swig -ruby -c++ -Iinclude -o {{ BINDINGS_DIR }}/octruby/ext/octruby/octruby_wrap.cxx -oh {{ BINDINGS_DIR }}/octruby/ext/octruby/octruby_wrap.h -outdir {{ BINDINGS_DIR }}/octruby/lib/octruby src/octruby/swig/octruby.i"

# Tcl (SWIG)
prebuild-tcl:
  {{ NIX_DEVELOP }} .#tcl --command bash -lc "mkdir -p build/datamungetcl/swig && swig -tcl8 -c++ -Iinclude -o build/datamungetcl/swig/datamunge_tcl_wrap.cxx -oh build/datamungetcl/swig/datamunge_tcl_wrap.h src/datamungetcl/swig/datamungetcl.i"

# Lua (SWIG)
prebuild-lua:
  {{ NIX_DEVELOP }} .#lua --command bash -lc "mkdir -p build/datamungelua-swig && swig -lua -c++ -Iinclude -outdir build/datamungelua-swig -o build/datamungelua-swig/datamunge_lua_wrap.cxx -oh build/datamungelua-swig/datamunge_lua_wrap.h src/datamungelua/swig/datamungelua.i"

# D (SWIG)
prebuild-d:
  bash -lc 'set -euo pipefail; \
    cmd="mkdir -p {{ BINDINGS_DIR }}/datamunged/source && swig -c++ -d -Iinclude -o {{ BINDINGS_DIR }}/datamunged/source/datamunged_wrap.cpp -oh {{ BINDINGS_DIR }}/datamunged/source/datamunged_wrap.h -outdir {{ BINDINGS_DIR }}/datamunged/source src/datamunged/swig/datamunged.i"; \
    if command -v nix >/dev/null 2>&1 && {{ NIX_DEVELOP }} .#d --command true >/dev/null 2>&1; then \
      {{ NIX_DEVELOP }} .#d --command bash -lc "$cmd"; \
    else \
      bash -lc "$cmd"; \
    fi'

# Guile (SWIG)
prebuild-guile:
  {{ NIX_DEVELOP }} .#guile --command bash -lc "mkdir -p build/datamungeguile-swig && swig -guile -c++ -Iinclude -o build/datamungeguile-swig/datamunge_guile_wrap.cxx -oh build/datamungeguile-swig/datamunge_guile_wrap.h src/datamungeguile/swig/datamungeguile.i"

prebuild-octave:
  {{ NIX_DEVELOP }} .#cpp --command bash -lc "mkdir -p build/datamungeoctave-swig && swig -octave -c++ -Iinclude -o build/datamungeoctave-swig/datamunge_octave_wrap.cxx -oh build/datamungeoctave-swig/datamunge_octave_wrap.h src/datamungeoctave/swig/datamungeoctave.i"

prebuild-go:
  {{ NIX_DEVELOP }} .#cpp --command bash -lc "swig -go -c++ -intgosize 64 -Iinclude -o {{ BINDINGS_DIR }}/godatamunge/godatamunge_wrap.cxx -oh {{ BINDINGS_DIR }}/godatamunge/godatamunge_wrap.h -outdir {{ BINDINGS_DIR }}/godatamunge src/godatamunge/swig/godatamunge.i"

prebuild-php:
  {{ NIX_DEVELOP }} .#cpp --command bash -lc "cd ./include && swig -c++ -php7 -o ../src/datamunge_php_wrap.cpp -oh ../src/datamunge_php_wrap.h ../src/datamungePHP/swig/datamungePHP.i"

prebuild-java:
    {{ NIX_DEVELOP }} .#java --command bash -lc "find {{ BINDINGS_DIR }}/jdatamunge/src/main/java/js/datamunge/jdatamunge -type f -name '*.java' ! -name 'App.java' ! -path '{{ BINDINGS_DIR }}/jdatamunge/src/main/java/js/datamunge/jdatamunge/examples/*' -exec rm {} +"
    {{ NIX_DEVELOP }} .#java --command bash -lc "rm -rf {{ BINDINGS_DIR }}/jdatamunge-datamunge/build/cmake"
    {{ NIX_DEVELOP }} .#java --command bash -lc "cd ./include && swig -doxygen -c++ -java -o ../{{ BINDINGS_DIR }}/jdatamunge-datamunge/datamunge_java_wrap.cpp -oh ../{{ BINDINGS_DIR }}/jdatamunge-datamunge/datamunge_java_wrap.h -package js.datamunge.jdatamunge -outdir ../{{ BINDINGS_DIR }}/jdatamunge/src/main/java/js/datamunge/jdatamunge ../src/jdatamunge-datamunge/swig/jdatamunge.i"
    {{ NIX_DEVELOP }} .#java --command bash -lc "sed -i 's/System.loadLibrary(\"datamunge\")/System.loadLibrary(\"datamunge_jni\")/g' {{ BINDINGS_DIR }}/jdatamunge/src/main/java/js/datamunge/jdatamunge/App.java {{ BINDINGS_DIR }}/jdatamunge/src/main/java/js/datamunge/jdatamunge/examples/StlEx.java"
    {{ NIX_DEVELOP }} .#java --command bash -lc "perl -0777 -pi -e 's/public class datamunge \\{/public class datamunge {\\n  static { System.loadLibrary(\"datamunge_jni\"); }/s' {{ BINDINGS_DIR }}/jdatamunge/src/main/java/js/datamunge/jdatamunge/datamunge.java"

prebuild-ocaml:
  {{ NIX_DEVELOP }} .#ocaml --command bash -lc "test -n \"${DATAMUNGE_PREFIX:-}\" || (echo 'DATAMUNGE_PREFIX is not set' >&2; exit 1) && mkdir -p {{ BINDINGS_DIR }}/datamungeocaml/src && swig -ocaml -c++ -Iinclude -o {{ BINDINGS_DIR }}/datamungeocaml/src/datamunge_ocaml_wrap.cxx -oh {{ BINDINGS_DIR }}/datamungeocaml/src/datamunge_ocaml_wrap.h -outdir {{ BINDINGS_DIR }}/datamungeocaml/src src/datamungeocaml/swig/datamungeocaml.i"
  # Work around a swig-jse OCaml-backend codegen bug: for ANY `enum class` (nested or
  # namespace-level), the generated SWIG_ENUM__... initializers reference the enumerator by
  # its bare name in the ENCLOSING namespace (e.g. `datamunge::stats::Additive`) instead of
  # correctly qualifying it with the enum class name (`datamunge::stats::TrendType::Additive`)
  # -- a hard C++ compile error, not a warning. Insert the missing enum-class qualifier for
  # every affected enumerator (see datamunge_ocaml_bindings.md memory for the full diagnosis).
  perl -0777 -pi -e "s/= datamunge::plot::DataSeries::(Scatter|Line|Bar)\b/= static_cast<int>(datamunge::plot::DataSeries::Kind::\$1)/g" {{ BINDINGS_DIR }}/datamungeocaml/src/datamunge_ocaml_wrap.cxx
  perl -0777 -pi -e "s/= datamunge::stats::(None|Additive|AdditiveDamped)\b/= static_cast<int>(datamunge::stats::TrendType::\$1)/g" {{ BINDINGS_DIR }}/datamungeocaml/src/datamunge_ocaml_wrap.cxx
  perl -0777 -pi -e "s/= datamunge::stats::Multiplicative\b/= static_cast<int>(datamunge::stats::SeasonalType::Multiplicative)/g" {{ BINDINGS_DIR }}/datamungeocaml/src/datamunge_ocaml_wrap.cxx
  perl -0777 -pi -e "s/= datamunge::stats::(TwoSided|Less|Greater)\b/= static_cast<int>(datamunge::stats::Alternative::\$1)/g" {{ BINDINGS_DIR }}/datamungeocaml/src/datamunge_ocaml_wrap.cxx

# }}} prebuild commands

# {{{ rust (bindgen) commands

prebuild-rust:
  {{ NIX_DEVELOP }} .#rust --command bash -lc 'inc="$(pkg-config --variable=includedir datamunge)" && bindgen "$inc/datamunge/datamunge_c.h" --allowlist-function "datamunge_.*" --allowlist-type "datamunge_.*" --no-layout-tests --rustfmt-bindings -o {{ BINDINGS_DIR }}/rustdatamunge/src/bindings.rs'

# }}} rust (bindgen) commands

# {{{ build commands

build-php: prebuild-php
  rm -rf build/datamungePHP
  {{ NIX_DEVELOP }} .#php --command bash -lc 'cmake -S src/datamungePHP -B build/datamungePHP'
  {{ NIX_DEVELOP }} .#php --command bash -lc 'cmake --build build/datamungePHP -j{{ JOBS }} --verbose'


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
    @echo "Building datamunge"
    {{ NIX_DEVELOP }} .#cpp --command bash -lc "cmake -S . -B build"
    {{ NIX_DEVELOP }} .#cpp --command bash -lc "cmake --build build -j{{ JOBS }} --verbose"
    {{ NIX_DEVELOP }} .#cpp --command bash -lc "find ./build -name 'compile_commands.json' -exec cat {} + | jq -s add > compile_commands.json"


build-csharp: prebuild-csharp
  {{ NIX_DEVELOP }} .#csharp --command bash -lc "rm -rf build/dotnet/release"
  {{ NIX_DEVELOP }} .#csharp --command bash -lc "cmake -S ./{{ BINDINGS_DIR }}/datamungedotnet -B build/dotnet/release -DCMAKE_MAKE_PROGRAM=$(command -v make)"
  {{ NIX_DEVELOP }} .#csharp --command bash -lc "cmake --build build/dotnet/release -j{{ JOBS }} --verbose"
  {{ NIX_DEVELOP }} .#csharp --command bash -lc "cd ./{{ BINDINGS_DIR }}/datamungedotnet && dotnet build"


build-javascript: prebuild-javascript
    {{ NIX_DEVELOP }} .#jsbuild --command bash -lc "npm --prefix . run build"

build-python: prebuild-python
  rm -rf build/venv/pydatamunge-run
  {{ NIX_DEVELOP }} .#python --command bash -lc 'python -m venv --system-site-packages build/venv/pydatamunge-run'
  {{ NIX_DEVELOP }} .#python --command bash -lc 'build/venv/pydatamunge-run/bin/python -m pip install -e . --no-build-isolation'


build-java: prebuild-java
  {{ NIX_DEVELOP }} .#java --command bash -lc "gradle cmakeBuild"
  {{ NIX_DEVELOP }} .#java --command bash -lc "gradle build"

build-dotnet:
    {{ NIX_DEVELOP }} . --command bash -lc "cmake -S {{ BINDINGS_DIR }}/datamungedotnet -B build/datamungedotnet"
    {{ NIX_DEVELOP }} . --command bash -lc "cmake --build build/datamungedotnet"
    # nix develop ./datamungedotnet --command bash -c "just --justfile ./datamungedotnet/justfile build"

build-go: prebuild-go build-cpp
  # For Go bindings, we need to run go build on the generated files
  # Note: This assumes the SWIG-generated files are already in place from prebuild-go
  {{ NIX_DEVELOP }} .#go --command bash -lc 'cd {{ BINDINGS_DIR }}/godatamunge && CGO_CPPFLAGS="-I$(pwd)/../../include" CGO_LDFLAGS="-L$(pwd)/../../build -ldatamunge" go build'

build-d: prebuild-d
  bash -lc 'set -euo pipefail; \
    compiler=""; \
    if command -v ldc2 >/dev/null 2>&1; then compiler="--compiler=ldc2"; elif command -v dmd >/dev/null 2>&1; then compiler="--compiler=dmd"; fi; \
    if command -v nix >/dev/null 2>&1 && {{ NIX_DEVELOP }} .#d --command true >/dev/null 2>&1; then \
      {{ NIX_DEVELOP }} .#d --command bash -lc "cd {{ BINDINGS_DIR }}/datamunged && dub build $compiler --build=release --force"; \
    else \
      cd {{ BINDINGS_DIR }}/datamunged && dub build $compiler --build=release --force; \
    fi'

build-perl: prebuild-perl build-cpp
  {{ NIX_DEVELOP }} .#perl --command bash -lc 'cd {{ BINDINGS_DIR }}/perldatamunge && rm -rf blib Makefile Makefile.old pm_to_blib MYMETA.* && perl Makefile.PL INSTALL_BASE="$(pwd)/../../build/perl" && make -j{{ JOBS }} && make install'

build-ruby: prebuild-ruby
  {{ NIX_DEVELOP }} .#ruby --command bash -lc "set -euo pipefail; cd {{ BINDINGS_DIR }}/octruby/ext/octruby && ruby extconf.rb && make -j1 && so=\"\$(find . -type f -name 'octruby*.so' -print -quit)\" && test -n \"\$so\" && mkdir -p ../../lib/octruby && cp -f \"\$so\" ../../lib/octruby/octruby.so"

build-r: prebuild-r
  rm -rf build/r/library
  {{ NIX_DEVELOP }} .#r --command bash -lc 'mkdir -p build/r/library && R CMD INSTALL -l build/r/library .'

build-tcl: prebuild-tcl
  rm -rf build/datamungetcl
  {{ NIX_DEVELOP }} .#tcl --command bash -lc 'cmake -S src/datamungetcl -B build/datamungetcl -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(pkg-config --variable=prefix datamunge);$(pkg-config --variable=prefix arrow)"'
  {{ NIX_DEVELOP }} .#tcl --command bash -lc 'cmake --build build/datamungetcl -j{{ JOBS }} --verbose'
  {{ NIX_DEVELOP }} .#tcl --command bash -lc 'cp -v {{ BINDINGS_DIR }}/datamungetcl/pkgIndex.tcl build/datamungetcl/'

build-lua: prebuild-lua
  rm -rf build/datamungelua
  {{ NIX_DEVELOP }} .#lua --command bash -lc 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungelua -B build/datamungelua -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#lua --command bash -lc 'cmake --build build/datamungelua -j{{ JOBS }} --verbose'

build-rust:
  {{ NIX_DEVELOP }} .#rust --command bash -lc 'cargo build --manifest-path {{ BINDINGS_DIR }}/rustdatamunge/Cargo.toml'

build-guile: prebuild-guile
  rm -rf build/datamungeguile
  {{ NIX_DEVELOP }} .#guile --command bash -lc 'cmake -S src/datamungeguile -B build/datamungeguile -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(pkg-config --variable=prefix datamunge);$(pkg-config --variable=prefix arrow)"'
  {{ NIX_DEVELOP }} .#guile --command bash -lc 'cmake --build build/datamungeguile -j{{ JOBS }} --verbose'

build-octave: prebuild-octave
  {{ NIX_DEVELOP }} .#octave --command bash -lc 'cmake -S src/datamungeoctave -B build/datamungeoctave -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(pkg-config --variable=prefix datamunge);$(pkg-config --variable=prefix arrow)"'
  {{ NIX_DEVELOP }} .#octave --command bash -lc 'cmake --build build/datamungeoctave -j{{ JOBS }} --verbose'

build-ocaml: prebuild-ocaml
  {{ NIX_DEVELOP }} .#ocaml --command bash -lc 'test -n "${DATAMUNGE_PREFIX:-}" || (echo "DATAMUNGE_PREFIX is not set" >&2; exit 1) && cd {{ BINDINGS_DIR }}/datamungeocaml && dune build'

install-ocaml: build-ocaml
  {{ NIX_DEVELOP }} .#ocaml --command bash -lc 'test -n "${DATAMUNGE_PREFIX:-}" || (echo "DATAMUNGE_PREFIX is not set" >&2; exit 1) && mkdir -p build/ocaml/prefix && cd {{ BINDINGS_DIR }}/datamungeocaml && dune install --prefix "$(pwd)/../../build/ocaml/prefix"'




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
  {{ NIX_DEVELOP }} .#csharp --command bash -lc 'LD_LIBRARY_PATH="$(pwd)/build/dotnet/release/_deps/datamunge-build:$(pwd)/build/dotnet/release${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" if command -v dotnet-repl >/dev/null 2>&1; then dotnet-repl; elif command -v csi >/dev/null 2>&1; then csi; else echo "No C# REPL found (expected dotnet-repl or csi)" >&2; exit 1; fi'

repl-java:
  {{ NIX_DEVELOP }} .#java --command bash -lc 'export LD_LIBRARY_PATH={{ BINDINGS_DIR }}/jdatamunge-datamunge/build/cmake:$LD_LIBRARY_PATH && jshell --class-path ./{{ BINDINGS_DIR }}/jdatamunge/build/libs/jdatamunge.jar'

repl-cpp:
  {{ NIX_DEVELOP }} .#cpp --command bash -lc 'cling $(pkg-config --cflags datamunge) $(pkg-config --libs-only-L datamunge) -ldatamunge -std=c++17'

repl-tcl: build-tcl
  {{ NIX_DEVELOP }} .#tcl --command bash -lc 'export TCLLIBPATH="$(pwd)/build/datamungetcl${TCLLIBPATH:+ $TCLLIBPATH}" && tclsh'

repl-lua:
  {{ NIX_DEVELOP }} .#lua --command bash -lc 'lua'

repl-ruby:
  {{ NIX_DEVELOP }} .#ruby --command bash -lc 'irb -r octruby'

repl-ocaml:
  {{ NIX_DEVELOP }} .#ocaml --command bash -lc 'utop'

repl-guile:
  {{ NIX_DEVELOP }} .#guile --command bash -lc 'guile'

repl-rust:
  {{ NIX_DEVELOP }} .#rust --command bash -lc 'export LD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" && cd {{ BINDINGS_DIR }}/rustdatamunge && evcxr'

repl-octave:
  {{ NIX_DEVELOP }} .#octave --command bash -lc 'datamungeoctave_prefix="$(nix eval --raw .#datamungeoctave)" && octave -qf --path "$datamungeoctave_prefix/share/octave/site/m"'

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
  rm -rf build/venv/pydatamunge-build
  {{ NIX_DEVELOP }} .#python --command bash -lc 'python -m venv --system-site-packages build/venv/pydatamunge-build'
  {{ NIX_DEVELOP }} .#python --command bash -lc 'build/venv/pydatamunge-build/bin/python setup.py sdist bdist_wheel'

test-python: prebuild-python
  rm -rf build/venv/pydatamunge
  {{ NIX_DEVELOP }} .#python --command bash -lc 'python -m venv --system-site-packages build/venv/pydatamunge'
  {{ NIX_DEVELOP }} .#python --command bash -lc 'build/venv/pydatamunge/bin/python -m pip install -e . --no-build-isolation'
  {{ NIX_DEVELOP }} .#python --command bash -lc 'build/venv/pydatamunge/bin/python -m pytest -q tests/python'

test-r: prebuild-r
  {{ NIX_DEVELOP }} .#r --command bash -lc 'R -q -e "testthat::test_local(\".\")"'

test-csharp: build-csharp
  {{ NIX_DEVELOP }} .#csharp --command bash -lc 'LD_LIBRARY_PATH="$(pwd)/build/dotnet/release/_deps/datamunge-build:$(pwd)/build/dotnet/release${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" dotnet test ./{{ BINDINGS_DIR }}/datamungedotnet.tests'

test-java: build-java
  {{ NIX_DEVELOP }} .#java --command bash -lc 'export LD_LIBRARY_PATH={{ BINDINGS_DIR }}/jdatamunge-datamunge/build/cmake:$LD_LIBRARY_PATH && gradle test'

test-rust:
  {{ NIX_DEVELOP }} .#rust --command bash -lc 'export LD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" && cargo test --manifest-path tests/rust/Cargo.toml'

test-php: build-php
  {{ NIX_DEVELOP }} .#php --command bash -lc 'php -d assert.exception=1 -d zend.assertions=1 --php-ini .user.ini tests/php/test_datamunge.php'

test-lua:
  {{ NIX_DEVELOP }} .#lua --command bash -lc 'lua tests/lua/test_datamunge.lua'

test-perl: build-perl
  {{ NIX_DEVELOP }} .#perl --command bash -lc 'export PERL5LIB="$(pwd)/build/perl/lib/perl5:$PERL5LIB" && export LD_LIBRARY_PATH="$(pwd)/build:$LD_LIBRARY_PATH" && prove -l tests/perl'

test-tcl: build-tcl
  {{ NIX_DEVELOP }} .#tcl --command bash -lc 'export TCLLIBPATH="$(pwd)/build/datamungetcl${TCLLIBPATH:+ $TCLLIBPATH}" && tclsh tests/tcl/test_datamunge.tcl'

test-ruby:
  {{ NIX_DEVELOP }} .#ruby --command bash -lc 'export LD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" && ruby -I tests/ruby -e "require \"test_datamunge\""'

test-guile:
  {{ NIX_DEVELOP }} .#guile --command bash -lc 'guile --no-auto-compile -s tests/guile/test_datamunge.scm'

test-octave:
  {{ NIX_DEVELOP }} .#octave --command bash -lc 'datamungeoctave_prefix="$(nix eval --raw .#datamungeoctave)" && octave -qf --path "$datamungeoctave_prefix/share/octave/site/m" --eval '"'"'test("tests/octave/test_datamunge.m")'"'"''

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
    {{ NIX_DEVELOP }} .#go --command bash -lc 'cd {{ BINDINGS_DIR }}/godatamunge && LD_LIBRARY_PATH="$(pwd)/../../build:$LD_LIBRARY_PATH" CGO_CPPFLAGS="-I$(pwd)/../../include" CGO_LDFLAGS="-L$(pwd)/../../build -ldatamunge" go test ./...'


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
  ./rename_datamunge {{ NEW }}

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
    @echo "Building benchmarks"
    {{ NIX_DEVELOP }} .#cpp --command bash -lc "cmake -S benchmarks -B build/benchmarks -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=$(command -v make)"
    {{ NIX_DEVELOP }} .#cpp --command bash -lc "cmake --build build/benchmarks -j{{ JOBS }}"
    @echo "Running {{ BENCH_TARGET }}"
    ./build/benchmarks/{{ BENCH_TARGET }} --benchmark_format=json \
        --benchmark_out="benchmarks/results/$(date +%Y%m%d_%H%M%S).json"
    @echo "Results saved to benchmarks/results/"

benchmark-compare BASE NEW:
    @echo "Comparing {{ BASE }} vs {{ NEW }}"
    python3 build/benchmarks/_deps/benchmark-src/tools/compare.py benchmarks \
        benchmarks/results/{{ BASE }} benchmarks/results/{{ NEW }}


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
      awk -v want1="$(pwd)/src/datamunge/datamunge.cpp" -v want2="$(pwd)/src/datamunge/datamunge_c.cpp" '\'' \
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
        END { finish(); print "coverage: OK (datamunge.cpp and datamunge_c.cpp are 100%)" } \
      '\'' "$info"'



debuggable:
  {{ NIX_DEVELOP }} .#cpp --command bash -lc 'clang++ -g -O0 debug.cpp -o debug $(pkg-config --cflags datamunge) $(pkg-config --libs-only-L datamunge) $(pkg-config --cflags libxml-2.0) $(pkg-config --libs-only-L libxml-2.0) -std=c++20 -ldatamunge -lxml2'


clean:
    rm -rf build/
    rm -rf {{ BINDINGS_DIR }}/datamungedotnet/bin
    rm -rf {{ BINDINGS_DIR }}/datamungedotnet/obj
    rm -rf {{ BINDINGS_DIR }}/jdatamunge-datamunge/build/
    rm -rf {{ BINDINGS_DIR }}/jdatamunge/build/


# }}} utilities

# {{{ docs commands

org-export INPUT OUTPUT:
    @echo "Exporting {{ INPUT }} -> {{ OUTPUT }}"
    @bash -lc 'set -euo pipefail; \
      export_cmd='\''emacs --batch -Q -l init.el -- "{{ INPUT }}" "{{ OUTPUT }}"'\''; \
      if command -v nix >/dev/null 2>&1; then \
        nix develop --accept-flake-config .#docs-pages --command bash -lc "$export_cmd"; \
      else \
        bash -lc "$export_cmd"; \
      fi'

org-example FORMAT="html":
    just org-export examples/org/dense_linear_algebra.org build/org/dense_linear_algebra.{{ FORMAT }}

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
    {{ NIX_DEVELOP }} .#python --command bash -lc "python examples/python/datamunge_ex.py"

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

build-nix PACKAGE="datamunge":
  nix build --accept-flake-config ".#{{ PACKAGE }}"

update-java-deps:
  nix/update-jdatamunge-gradle-deps.sh

update-csharp-deps:
  nix/update-nuget-deps.sh

all-test:
  bash ./scripts/test_all.sh

test-all: all-test

# }}} all commands
