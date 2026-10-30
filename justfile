TARGET := "dataframe_ex"
BENCH_TARGET := "linalg_bench"
JOBS := "20"
NN_JOBS := env_var_or_default("DATAMUNGE_EXAMPLE_BUILD_JOBS", "4")

# Set DATAMUNGE_SWIG_SRC to a local checkout of the swig fork (e.g. ~/projects/swig) to build
# against it instead of the github:jordanschupbach/swig revision pinned in flake.lock.
SWIG_SRC := env_var_or_default("DATAMUNGE_SWIG_SRC", "")
SWIG_OVERRIDE := if SWIG_SRC == "" { "" } else { " --override-input swig git+file://" + SWIG_SRC }
NIX_DEVELOP := "nix develop --accept-flake-config --option eval-cache false" + SWIG_OVERRIDE
# PKG_CONFIG_PATH the docs' mepml report blocks build against (empty: the shell's own;
# a directory with a datamunge.pc for the working tree makes them use the current library).
MEPML_PKG_CONFIG_PATH := env_var_or_default("DATAMUNGE_MEPML_PKG_CONFIG_PATH", "")
BINDINGS_DIR := "src"

# {{{ run commands



run: run-cpp

test: test-cpp

format:
  {{ NIX_DEVELOP }} .#format --command bash -c './scripts/format.sh'

format-check:
  {{ NIX_DEVELOP }} .#quality --command bash -c './scripts/format_check.sh'

lint:
  {{ NIX_DEVELOP }} .#quality --command bash -c './scripts/lint.sh'

quality: format-check lint

fmt: format

run-all: run-cpp run-csharp run-java run-go run-rust run-d run-python run-php run-perl run-tcl run-lua run-ruby run-r run-guile run-javascript run-ocaml run-octave

# dotnet run always compiles whatever's in datamungedotnet/Program.cs (no per-example TARGET
# mechanism, like D's dub) -- copy the selected example over Program.cs before building, same
# pattern as run-d's examples-src/ -> source/app.d copy.
run-csharp: prebuild-csharp
  {{ NIX_DEVELOP }} .#csharp --command bash -c "rm -rf build/dotnet/release"
  {{ NIX_DEVELOP }} .#csharp --command bash -c "cmake -S ./{{ BINDINGS_DIR }}/datamungedotnet -B build/dotnet/release -DCMAKE_MAKE_PROGRAM=$(command -v make)"
  {{ NIX_DEVELOP }} .#csharp --command bash -c "cmake --build build/dotnet/release -j{{ JOBS }} --verbose"
  cp examples/csharp/{{ TARGET }}.cs {{ BINDINGS_DIR }}/datamungedotnet/Program.cs
  {{ NIX_DEVELOP }} .#csharp --command bash -c 'LD_LIBRARY_PATH="$(pwd)/build/dotnet/release/_deps/datamunge-build:$(pwd)/build/dotnet/release${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" dotnet run --project ./{{ BINDINGS_DIR }}/datamungedotnet'

run-java: build-java
  {{ NIX_DEVELOP }} .#java --command bash -c "gradle run --no-configuration-cache --args='{{ TARGET }}'"

run-go: build-go
  {{ NIX_DEVELOP }} .#go --command bash -c 'cd {{ BINDINGS_DIR }}/godatamunge && LD_LIBRARY_PATH="$(pwd)/../../build:$LD_LIBRARY_PATH" DYLD_LIBRARY_PATH="$(pwd)/../../build:$DYLD_LIBRARY_PATH" CGO_CPPFLAGS="-I$(pwd)/../../include" CGO_LDFLAGS="-L$(pwd)/../../build -ldatamunge" go run ../../examples/go/{{ TARGET }}.go'

run-rust: build-rust
  {{ NIX_DEVELOP }} .#rust --command bash -c 'export LD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" && cargo run --manifest-path {{ BINDINGS_DIR }}/rustdatamunge/Cargo.toml --example {{ TARGET }}'

run-d: build-d
  bash -c 'set -euo pipefail; \
    compiler=""; \
    if command -v ldc2 >/dev/null 2>&1; then compiler="--compiler=ldc2"; elif command -v dmd >/dev/null 2>&1; then compiler="--compiler=dmd"; fi; \
    cp examples/d/examples-src/{{ TARGET }}.d examples/d/source/app.d; \
    if command -v nix >/dev/null 2>&1; then \
      {{ NIX_DEVELOP }} .#d --command bash -c "rm -rf build/dub-packages/datamunged-0.0.1 && cd examples/d && dub run $compiler --build=release"; \
    else \
      rm -rf build/dub-packages/datamunged-0.0.1 && cd examples/d && dub run $compiler --build=release; \
    fi'

run-python: build-python
  {{ NIX_DEVELOP }} .#python --command bash -c 'env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/{{ TARGET }}.py'

run-php: build-php
  {{ NIX_DEVELOP }} .#php --command bash -c 'php --php-ini .user.ini examples/php/{{ TARGET }}.php'

run-perl: build-perl
  {{ NIX_DEVELOP }} .#perl --command bash -c 'export PERL5LIB="$(pwd)/build/perl/lib/perl5:$PERL5LIB" && export LD_LIBRARY_PATH="$(pwd)/build:$LD_LIBRARY_PATH" && perl examples/perl/{{ TARGET }}.pl'

run-tcl: build-tcl
  {{ NIX_DEVELOP }} .#tcl --command bash -c 'ext="$(tclsh <<< "puts [info sharedlibextension]")" && if [ -f build/datamungetcl/Datamunge.so ] && [ ! -e "build/datamungetcl/Datamunge$ext" ]; then cp -f build/datamungetcl/Datamunge.so "build/datamungetcl/Datamunge$ext"; fi && export TCLLIBPATH="$(pwd)/build/datamungetcl${TCLLIBPATH:+ $TCLLIBPATH}" && export DYLD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)${DYLD_LIBRARY_PATH:+:}$DYLD_LIBRARY_PATH" && tclsh examples/tcl/{{ TARGET }}.tcl'

run-lua: build-lua
  {{ NIX_DEVELOP }} .#lua --command bash -c 'cmake --install build/datamungelua --prefix build/lua/prefix >/dev/null && export LUA_CPATH="$(pwd)/build/lua/prefix/lib/lua/?.so;$(pwd)/build/lua/prefix/lib64/lua/?.so;;" && export LUA_PATH="$(pwd)/build/lua/prefix/share/lua/?.lua;;" && export LD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" && export DYLD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)${DYLD_LIBRARY_PATH:+:}$DYLD_LIBRARY_PATH" && lua examples/lua/{{ TARGET }}.lua'

run-ruby: build-ruby
  {{ NIX_DEVELOP }} .#ruby --command bash -c 'export LD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" && export DYLD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)${DYLD_LIBRARY_PATH:+:}$DYLD_LIBRARY_PATH" && ruby -I {{ BINDINGS_DIR }}/octruby/lib examples/ruby/{{ TARGET }}.rb'

run-r: build-r
   {{ NIX_DEVELOP }} .#r --command bash -c 'R_LIBS_USER="$(pwd)/build/r/library${R_LIBS_USER:+:}$R_LIBS_USER" Rscript examples/r/{{ TARGET }}.r'

run-guile: build-guile
  {{ NIX_DEVELOP }} .#guile --command bash -c 'cmake --install build/datamungeguile --prefix build/guile/prefix >/dev/null && guile_effective="$(pkg-config --variable=effective-version guile-3.0 2>/dev/null || echo 3.0)" && export GUILE_LOAD_PATH="$(pwd)/build/guile/prefix/share/guile/site/$guile_effective${GUILE_LOAD_PATH:+:}$GUILE_LOAD_PATH" && export LD_LIBRARY_PATH="$(pwd)/build/guile/prefix/lib/guile/$guile_effective/extensions:$(pwd)/build${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" && export DYLD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)${DYLD_LIBRARY_PATH:+:}$DYLD_LIBRARY_PATH" && guile --no-auto-compile -s examples/guile/{{ TARGET }}.scm'

run-javascript: build-javascript
   {{ NIX_DEVELOP }} .#javascript --command bash -c 'node ./examples/javascript/{{ TARGET }}.js'

run-ocaml:
  {{ NIX_DEVELOP }} .#ocaml --command bash -c 'just install-ocaml && mkdir -p build/ocaml && export OCAMLPATH="$(pwd)/build/ocaml/prefix/lib${OCAMLPATH:+:}$OCAMLPATH" && ocamlfind ocamlopt -package datamungeocaml -linkpkg examples/ocaml/{{ TARGET }}.ml -o build/ocaml/{{ TARGET }} && ./build/ocaml/{{ TARGET }}'

run-octave: build-octave
  {{ NIX_DEVELOP }} .#octave --command bash -c 'octave -qf --path "$(pwd)/build/datamungeoctave" examples/octave/{{ TARGET }}.m'

# run-all-<lang>: build the binding once, then run EVERY example for that language,
# printing a per-example header and a passed/failed summary (exits non-zero on any failure).

run-all-lua: build-lua
  {{ NIX_DEVELOP }} .#lua --command bash -c 'cmake --install build/datamungelua --prefix build/lua/prefix >/dev/null && \
    export LUA_CPATH="$(pwd)/build/lua/prefix/lib/lua/?.so;$(pwd)/build/lua/prefix/lib64/lua/?.so;;" && \
    export LUA_PATH="$(pwd)/build/lua/prefix/share/lua/?.lua;;" && \
    export LD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" && \
    export DYLD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)${DYLD_LIBRARY_PATH:+:}$DYLD_LIBRARY_PATH" && \
    pass=0; fail=0; failed=""; \
    for f in examples/lua/*.lua; do n="$(basename "$f")"; printf "\n========== %s ==========\n" "$n"; \
      if lua "$f"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## lua: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-guile: build-guile
  {{ NIX_DEVELOP }} .#guile --command bash -c 'cmake --install build/datamungeguile --prefix build/guile/prefix >/dev/null && \
    guile_effective="$(pkg-config --variable=effective-version guile-3.0 2>/dev/null || echo 3.0)" && \
    export GUILE_LOAD_PATH="$(pwd)/build/guile/prefix/share/guile/site/$guile_effective${GUILE_LOAD_PATH:+:}$GUILE_LOAD_PATH" && \
    export LD_LIBRARY_PATH="$(pwd)/build/guile/prefix/lib/guile/$guile_effective/extensions:$(pwd)/build${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" && \
    export DYLD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)${DYLD_LIBRARY_PATH:+:}$DYLD_LIBRARY_PATH" && \
    pass=0; fail=0; failed=""; \
    for f in examples/guile/*.scm; do n="$(basename "$f")"; printf "\n========== %s ==========\n" "$n"; \
      if guile --no-auto-compile -s "$f"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## guile: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-ruby: build-ruby
  {{ NIX_DEVELOP }} .#ruby --command bash -c 'export LD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" && \
    export DYLD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)${DYLD_LIBRARY_PATH:+:}$DYLD_LIBRARY_PATH" && \
    pass=0; fail=0; failed=""; \
    for f in examples/ruby/*.rb; do n="$(basename "$f")"; printf "\n========== %s ==========\n" "$n"; \
      if ruby -I {{ BINDINGS_DIR }}/octruby/lib "$f"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## ruby: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-tcl: build-tcl
  {{ NIX_DEVELOP }} .#tcl --command bash -c 'ext="$(tclsh <<< "puts [info sharedlibextension]")" && \
    if [ -f build/datamungetcl/Datamunge.so ] && [ ! -e "build/datamungetcl/Datamunge$ext" ]; then cp -f build/datamungetcl/Datamunge.so "build/datamungetcl/Datamunge$ext"; fi && \
    export TCLLIBPATH="$(pwd)/build/datamungetcl${TCLLIBPATH:+ $TCLLIBPATH}" && \
    export DYLD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)${DYLD_LIBRARY_PATH:+:}$DYLD_LIBRARY_PATH" && \
    pass=0; fail=0; failed=""; \
    for f in examples/tcl/*.tcl; do n="$(basename "$f")"; printf "\n========== %s ==========\n" "$n"; \
      if tclsh "$f"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## tcl: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-octave: build-octave
  {{ NIX_DEVELOP }} .#octave --command bash -c 'pass=0; fail=0; failed=""; \
    for f in examples/octave/*.m; do n="$(basename "$f")"; printf "\n========== %s ==========\n" "$n"; \
      if octave -qf --path "$(pwd)/build/datamungeoctave" "$f"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## octave: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-perl: build-perl
  {{ NIX_DEVELOP }} .#perl --command bash -c 'export PERL5LIB="$(pwd)/build/perl/lib/perl5:$PERL5LIB" && \
    export LD_LIBRARY_PATH="$(pwd)/build:$LD_LIBRARY_PATH" && \
    export DYLD_LIBRARY_PATH="$(pwd)/build:$DYLD_LIBRARY_PATH" && \
    pass=0; fail=0; failed=""; \
    for f in examples/perl/*.pl; do n="$(basename "$f")"; printf "\n========== %s ==========\n" "$n"; \
      if perl "$f"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## perl: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-go: build-go
  {{ NIX_DEVELOP }} .#go --command bash -c 'root="$(pwd)"; cd {{ BINDINGS_DIR }}/godatamunge && \
    export LD_LIBRARY_PATH="$root/build:$LD_LIBRARY_PATH" && export DYLD_LIBRARY_PATH="$root/build:$DYLD_LIBRARY_PATH" && \
    export CGO_CPPFLAGS="-I$root/include" && export CGO_LDFLAGS="-L$root/build -ldatamunge" && \
    pass=0; fail=0; failed=""; \
    for f in "$root"/examples/go/*.go; do case "$f" in *_test.go) continue;; esac; n="$(basename "$f")"; printf "\n========== %s ==========\n" "$n"; \
      if go run "$f"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## go: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-d: build-d
  {{ NIX_DEVELOP }} .#d --command bash -c 'compiler=""; if command -v ldc2 >/dev/null 2>&1; then compiler="--compiler=ldc2"; elif command -v dmd >/dev/null 2>&1; then compiler="--compiler=dmd"; fi; \
    pass=0; fail=0; failed=""; \
    for f in examples/d/examples-src/*.d; do n="$(basename "$f")"; printf "\n========== %s ==========\n" "$n"; \
      cp "$f" examples/d/source/app.d; \
      if ( rm -rf build/dub-packages/datamunged-0.0.1 && cd examples/d && dub run $compiler --build=release ); then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## d: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-python: build-python
  {{ NIX_DEVELOP }} .#python --command bash -c 'pass=0; fail=0; failed=""; \
    for f in examples/python/*.py; do n="$(basename "$f")"; printf "\n========== %s ==========\n" "$n"; \
      if build/venv/pydatamunge-run/bin/python "$f"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## python: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-r: build-r
  {{ NIX_DEVELOP }} .#r --command bash -c 'export R_LIBS_USER="$(pwd)/build/r/library${R_LIBS_USER:+:}$R_LIBS_USER"; \
    pass=0; fail=0; failed=""; \
    for f in examples/r/*.r; do n="$(basename "$f")"; printf "\n========== %s ==========\n" "$n"; \
      if Rscript "$f"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## r: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-java: build-java
  {{ NIX_DEVELOP }} .#java --command bash -c 'pass=0; fail=0; failed=""; \
    for f in {{ BINDINGS_DIR }}/jdatamunge/src/main/java/js/datamunge/jdatamunge/examples/*.java; do \
      cls="$(basename "$f" .java)"; [ "$cls" = "ExampleRunner" ] && continue; \
      n="$(echo "$cls" | sed -E "s/([a-z0-9])([A-Z])/\1_\2/g" | tr "[:upper:]" "[:lower:]")"; \
      printf "\n========== %s ==========\n" "$n"; \
      if gradle run --no-configuration-cache --args="$n"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## java: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-cpp: examples
  {{ NIX_DEVELOP }} .#cpp --command bash -c 'pass=0; fail=0; failed=""; \
    while IFS= read -r f; do n="$(basename "$f" .cpp)"; b="build/debug/examples/$n"; \
      printf "\n========== %s ==========\n" "$n"; \
      if [ -x "$b" ] && "$b"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done < <(find examples/cpp/source -path examples/cpp/source/nn -prune -o -type f -name "*.cpp" -print); \
    printf "\n########## cpp: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

## Build and run only the selected working-tree example; forward optional program arguments.
[positional-arguments]
run-cpp *ARGS:
    @echo "Running target {{ TARGET }}"
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake -S examples/cpp -B build/debug/examples -DCMAKE_BUILD_TYPE=Debug -DBUILD_W_INSTALLED=OFF
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake --build build/debug/examples --target {{ TARGET }} --parallel {{ JOBS }}
    {{ NIX_DEVELOP }} .#cpp-tools --command ./build/debug/examples/{{ TARGET }} "$@"

run-plot:
    @echo "Running plot example"
    {{ NIX_DEVELOP }} .#cpp-tools --command bash -c 'cmake -S examples/cpp -B build/debug/examples --preset=debug -DCMAKE_MAKE_PROGRAM="$(command -v make)" -DBUILD_W_INSTALLED=OFF'
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake --build build/debug/examples --target plot_ex --parallel {{ JOBS }}
    ./build/debug/examples/plot_ex

view-plot: examples
    @echo "Running plot example and opening the scatter plot"
    {{ NIX_DEVELOP }} .#cpp --command bash -c './build/debug/examples/plot_ex'

run-benchmark:
    @echo "Running Benchmarks"
    python3 scripts/benchmark_backend.py run --no-build --suite {{ BENCH_TARGET }}


# }}} run commands

# {{{ prebuild commands

# Regenerate the embedded Computer Modern plot font (src/datamunge/plot/cmu_serif_font_data.hpp)
# from CMU Serif + Latin Modern Math, using the flake's pinned nixpkgs.
embed-plot-font:
  #!/usr/bin/env bash
  set -euo pipefail
  mapfile -t deps < <(nix build --inputs-from . --no-link --print-out-paths \
    nixpkgs#texlivePackages.cm-unicode.tex nixpkgs#texlivePackages.lm-math.tex \
    nixpkgs#python3 nixpkgs#python3Packages.fonttools)
  PYTHONPATH="$(echo "${deps[3]}"/lib/python3*/site-packages)" "${deps[2]}/bin/python3" scripts/embed_plot_font.py \
    --font "${deps[0]}/fonts/opentype/public/cm-unicode/cmunrm.otf" \
    --fallback "${deps[1]}/fonts/opentype/public/lm-math/latinmodern-math.otf"

# Update the pinned swig fork revision in flake.lock to the latest github:jordanschupbach/swig.
update-swig:
  nix flake update swig

prebuild-swig: prebuild-python prebuild-javascript prebuild-csharp prebuild-r prebuild-perl prebuild-ruby prebuild-tcl prebuild-lua prebuild-d prebuild-guile prebuild-octave prebuild-go prebuild-php prebuild-java prebuild-ocaml
  @echo "SWIG wrappers regenerated (with Doxygen comments enabled)"

prebuild-python:
  {{ NIX_DEVELOP }} .#python --command python scripts/prebuild_bindings.py python

prebuild-javascript:
  {{ NIX_DEVELOP }} .#cpp --command bash -c "cd ./include && swig -javascript -napi -c++ -o ../src/datamunge_js_wrap.cpp -oh ../src/datamunge_js_wrap.h ../src/datamungejs/src/datamungejs.i"
  # Inject deterministic JS->C callback bridge helpers (SWIG Node backend doesn't support directors here).
  perl -0777 -pi -e 's/#include <napi.h>\n/#include <napi.h>\n#include \"datamunge_js_callbacks.inl\"\n/s' src/datamunge_js_wrap.cpp
  perl -0777 -pi -e 's/SWIG_InitializeModule\(env\);\n/SWIG_InitializeModule(env);\n  DatamungeJS_RegisterCallbackBridge(env, exports);\n/s' src/datamunge_js_wrap.cpp

prebuild-csharp:
  {{ NIX_DEVELOP }} .#cpp --command bash -c "find ./{{ BINDINGS_DIR }}/datamungedotnet -type f -name '*.cs' ! -name 'Program.cs' -exec rm {} +"
  {{ NIX_DEVELOP }} .#cpp --command bash -c "cd ./include && swig -doxygen -c++ -csharp -dllimport datamunge_csharp -o ../{{ BINDINGS_DIR }}/datamungedotnet/datamunge_csharp_wrap.cpp -oh ../{{ BINDINGS_DIR }}/datamungedotnet/datamunge_csharp_wrap.h ../src/datamungedotnet/swig/datamungedotnet.i"
  {{ NIX_DEVELOP }} .#cpp --command bash -c "sed -i 's/DllImport(\"datamunge\"/DllImport(\"datamunge_csharp\"/g' ./{{ BINDINGS_DIR }}/datamungedotnet/datamungePINVOKE.cs"

prebuild-r:
  {{ NIX_DEVELOP }} .#r --command python3 scripts/prebuild_bindings.py r

prebuild-perl:
  {{ NIX_DEVELOP }} .#cpp --command bash -c "mkdir -p {{ BINDINGS_DIR }}/perldatamunge/lib && swig -perl5 -c++ -Iinclude -o {{ BINDINGS_DIR }}/perldatamunge/Datamunge_wrap.cxx -oh {{ BINDINGS_DIR }}/perldatamunge/Datamunge_wrap.h -outdir {{ BINDINGS_DIR }}/perldatamunge/lib src/perldatamunge/swig/perldatamunge.i"

# Ruby (SWIG)
prebuild-ruby:
  {{ NIX_DEVELOP }} .#ruby --command bash -c "mkdir -p {{ BINDINGS_DIR }}/octruby/ext/octruby {{ BINDINGS_DIR }}/octruby/lib/octruby && swig -ruby -c++ -Iinclude -o {{ BINDINGS_DIR }}/octruby/ext/octruby/octruby_wrap.cxx -oh {{ BINDINGS_DIR }}/octruby/ext/octruby/octruby_wrap.h -outdir {{ BINDINGS_DIR }}/octruby/lib/octruby src/octruby/swig/octruby.i"

# Tcl (SWIG)
prebuild-tcl:
  {{ NIX_DEVELOP }} .#tcl --command bash -c "mkdir -p build/datamungetcl/swig && swig -tcl8 -c++ -Iinclude -o build/datamungetcl/swig/datamunge_tcl_wrap.cxx -oh build/datamungetcl/swig/datamunge_tcl_wrap.h src/datamungetcl/swig/datamungetcl.i"

# Lua (SWIG)
prebuild-lua:
  {{ NIX_DEVELOP }} .#lua --command bash -c "mkdir -p build/datamungelua-swig && swig -lua -c++ -Iinclude -outdir build/datamungelua-swig -o build/datamungelua-swig/datamunge_lua_wrap.cxx -oh build/datamungelua-swig/datamunge_lua_wrap.h src/datamungelua/swig/datamungelua.i"

# D (SWIG)
prebuild-d:
  bash -c 'set -euo pipefail; \
    cmd="mkdir -p {{ BINDINGS_DIR }}/datamunged/source && swig -c++ -d -Iinclude -o {{ BINDINGS_DIR }}/datamunged/source/datamunged_wrap.cpp -oh {{ BINDINGS_DIR }}/datamunged/source/datamunged_wrap.h -outdir {{ BINDINGS_DIR }}/datamunged/source src/datamunged/swig/datamunged.i"; \
    if command -v nix >/dev/null 2>&1 && {{ NIX_DEVELOP }} .#d --command true >/dev/null 2>&1; then \
      {{ NIX_DEVELOP }} .#d --command bash -c "$cmd"; \
    else \
      bash -c "$cmd"; \
    fi'

# Guile (SWIG)
prebuild-guile:
  {{ NIX_DEVELOP }} .#guile --command bash -c "mkdir -p build/datamungeguile-swig && swig -guile -c++ -Iinclude -o build/datamungeguile-swig/datamunge_guile_wrap.cxx -oh build/datamungeguile-swig/datamunge_guile_wrap.h src/datamungeguile/swig/datamungeguile.i"

prebuild-octave:
  {{ NIX_DEVELOP }} .#cpp --command bash -c "mkdir -p build/datamungeoctave-swig && swig -octave -c++ -Iinclude -o build/datamungeoctave-swig/datamunge_octave_wrap.cxx -oh build/datamungeoctave-swig/datamunge_octave_wrap.h src/datamungeoctave/swig/datamungeoctave.i"

prebuild-go:
  {{ NIX_DEVELOP }} .#cpp --command bash -c "swig -go -c++ -intgosize 64 -Iinclude -o {{ BINDINGS_DIR }}/godatamunge/godatamunge_wrap.cxx -oh {{ BINDINGS_DIR }}/godatamunge/godatamunge_wrap.h -outdir {{ BINDINGS_DIR }}/godatamunge src/godatamunge/swig/godatamunge.i"

prebuild-php:
  {{ NIX_DEVELOP }} .#cpp --command bash -c "cd ./include && swig -c++ -php7 -o ../src/datamunge_php_wrap.cpp -oh ../src/datamunge_php_wrap.h ../src/datamungePHP/swig/datamungePHP.i"

prebuild-java:
    {{ NIX_DEVELOP }} .#java --command bash -c "find {{ BINDINGS_DIR }}/jdatamunge/src/main/java/js/datamunge/jdatamunge -type f -name '*.java' ! -name 'App.java' ! -path '{{ BINDINGS_DIR }}/jdatamunge/src/main/java/js/datamunge/jdatamunge/examples/*' -exec rm {} +"
    {{ NIX_DEVELOP }} .#java --command bash -c "rm -rf {{ BINDINGS_DIR }}/jdatamunge-datamunge/build/cmake"
    {{ NIX_DEVELOP }} .#java --command bash -c "cd ./include && swig -doxygen -c++ -java -o ../{{ BINDINGS_DIR }}/jdatamunge-datamunge/datamunge_java_wrap.cpp -oh ../{{ BINDINGS_DIR }}/jdatamunge-datamunge/datamunge_java_wrap.h -package js.datamunge.jdatamunge -outdir ../{{ BINDINGS_DIR }}/jdatamunge/src/main/java/js/datamunge/jdatamunge ../src/jdatamunge-datamunge/swig/jdatamunge.i"
    {{ NIX_DEVELOP }} .#java --command bash -c "perl -pi -e 's/System\\.loadLibrary\\(\"datamunge\"\\)/System.loadLibrary(\"datamunge_jni\")/g' {{ BINDINGS_DIR }}/jdatamunge/src/main/java/js/datamunge/jdatamunge/App.java {{ BINDINGS_DIR }}/jdatamunge/src/main/java/js/datamunge/jdatamunge/examples/StlEx.java"
    {{ NIX_DEVELOP }} .#java --command bash -c "perl -0777 -pi -e 's/public class datamunge \\{/public class datamunge {\\n  static { System.loadLibrary(\"datamunge_jni\"); }/s' {{ BINDINGS_DIR }}/jdatamunge/src/main/java/js/datamunge/jdatamunge/datamunge.java"

prebuild-ocaml:
  {{ NIX_DEVELOP }} .#ocaml --command bash -c "test -n \"${DATAMUNGE_PREFIX:-}\" || (echo 'DATAMUNGE_PREFIX is not set' >&2; exit 1) && mkdir -p {{ BINDINGS_DIR }}/datamungeocaml/src && swig -ocaml -c++ -Iinclude -o {{ BINDINGS_DIR }}/datamungeocaml/src/datamunge_ocaml_wrap.cxx -oh {{ BINDINGS_DIR }}/datamungeocaml/src/datamunge_ocaml_wrap.h -outdir {{ BINDINGS_DIR }}/datamungeocaml/src src/datamungeocaml/swig/datamungeocaml.i"

# }}} prebuild commands

# {{{ rust (bindgen) commands

prebuild-rust:
  {{ NIX_DEVELOP }} .#rust --command bash -c 'inc="$(pkg-config --variable=includedir datamunge)" && bindgen "$inc/datamunge/datamunge_c.h" --allowlist-function "datamunge_.*" --allowlist-type "datamunge_.*" --no-layout-tests --rustfmt-bindings -o {{ BINDINGS_DIR }}/rustdatamunge/src/bindings.rs'

# }}} rust (bindgen) commands

# {{{ build commands

build-php: prebuild-php
  rm -rf build/datamungePHP
  {{ NIX_DEVELOP }} .#php --command bash -c 'cmake -S src/datamungePHP -B build/datamungePHP'
  {{ NIX_DEVELOP }} .#php --command bash -c 'cmake --build build/datamungePHP -j{{ JOBS }} --verbose'
  # Loaded through PHP_INI_SCAN_DIR (set by the default dev shell), so a plain `php script.php` has it.
  mkdir -p build/php/conf.d
  bash -c 'for f in build/datamungePHP/libdatamungePHP.*; do printf "extension=%s\n" "$(pwd)/$f" > build/php/conf.d/datamunge.ini; done'


build: build-cpp

build-example-installed:
  {{ NIX_DEVELOP }} .#cpp --command bash -c 'cmake -S examples/cpp -B build/debug/examples-installed -DCMAKE_BUILD_TYPE=Debug -DCMAKE_MAKE_PROGRAM=$(command -v make) -DBUILD_W_INSTALLED=ON'


build-release:
  {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake -S . -B build/release --preset=release-clang-linux-x86 -DCMAKE_MAKE_PROGRAM=$(command -v make)"
  ln -sf build/release/compile_commands.json compile_commands.json

build-debug:
  {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake -S . -B build/debug --preset=debug-clang-linux-x86 -DCMAKE_MAKE_PROGRAM=$(command -v make)"
  ln -sf build/debug/compile_commands.json compile_commands.json

build-cpp:
    @echo "Building datamunge"
    {{ NIX_DEVELOP }} .#cpp-tools --command bash -c "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release"
    {{ NIX_DEVELOP }} .#cpp-tools --command bash -c "cmake --build build --config Release -j{{ JOBS }} --verbose"
    {{ NIX_DEVELOP }} .#cpp-tools --command bash -c "find ./build -name 'compile_commands.json' -exec cat {} + | jq -s add > compile_commands.json"


build-csharp: prebuild-csharp
  {{ NIX_DEVELOP }} .#csharp --command bash -c "rm -rf build/dotnet/release"
  {{ NIX_DEVELOP }} .#csharp --command bash -c "cmake -S ./{{ BINDINGS_DIR }}/datamungedotnet -B build/dotnet/release -DCMAKE_MAKE_PROGRAM=$(command -v make)"
  {{ NIX_DEVELOP }} .#csharp --command bash -c "cmake --build build/dotnet/release -j{{ JOBS }} --verbose"
  {{ NIX_DEVELOP }} .#csharp --command bash -c "cd ./{{ BINDINGS_DIR }}/datamungedotnet && dotnet build"


build-javascript: prebuild-javascript
    {{ NIX_DEVELOP }} .#jsbuild --command bash -c "npm --prefix . run build"

build-python: prebuild-python
  {{ NIX_DEVELOP }} .#python --command bash -c 'test -x build/venv/pydatamunge-run/bin/python || python -m venv --system-site-packages build/venv/pydatamunge-run'
  {{ NIX_DEVELOP }} .#python --command bash -c 'if test -f build/nn-python/enabled; then export DATAMUNGE_BUILD_NN_EXAMPLES=${DATAMUNGE_BUILD_NN_EXAMPLES:-1}; fi; CMAKE_BUILD_PARALLEL_LEVEL={{ JOBS }} build/venv/pydatamunge-run/bin/python -m pip install --force-reinstall . --no-deps --no-build-isolation'


build-java: prebuild-java
  {{ NIX_DEVELOP }} .#java --command bash -c "gradle cmakeBuild"
  {{ NIX_DEVELOP }} .#java --command bash -c "gradle build"

build-dotnet:
    {{ NIX_DEVELOP }} . --command bash -c "cmake -S {{ BINDINGS_DIR }}/datamungedotnet -B build/datamungedotnet"
    {{ NIX_DEVELOP }} . --command bash -c "cmake --build build/datamungedotnet"
    # nix develop ./datamungedotnet --command bash -c "just --justfile ./datamungedotnet/justfile build"

build-go: prebuild-go build-cpp
  # For Go bindings, we need to run go build on the generated files
  # Note: This assumes the SWIG-generated files are already in place from prebuild-go
  {{ NIX_DEVELOP }} .#go --command bash -c 'cd {{ BINDINGS_DIR }}/godatamunge && CGO_CPPFLAGS="-I$(pwd)/../../include" CGO_LDFLAGS="-L$(pwd)/../../build -ldatamunge" go build'

build-d: prebuild-d
  # The native wrapper the generated module dlopens at startup (found via the dev shells' library path).
  {{ NIX_DEVELOP }} .#d --command bash -c 'mkdir -p build/d && ext="$(case "$(uname -s)" in Darwin) echo dylib;; *) echo so;; esac)" && c++ -std=c++17 -shared -fPIC -O1 $(pkg-config --cflags datamunge) -o "build/d/libdatamunge_wrap.$ext" {{ BINDINGS_DIR }}/datamunged/source/datamunged_wrap.cpp $(pkg-config --libs datamunge) -Wl,-rpath,$(pkg-config --variable=libdir datamunge)'
  bash -c 'set -euo pipefail; \
    compiler=""; \
    if command -v ldc2 >/dev/null 2>&1; then compiler="--compiler=ldc2"; elif command -v dmd >/dev/null 2>&1; then compiler="--compiler=dmd"; fi; \
    if command -v nix >/dev/null 2>&1 && {{ NIX_DEVELOP }} .#d --command true >/dev/null 2>&1; then \
      {{ NIX_DEVELOP }} .#d --command bash -c "cd {{ BINDINGS_DIR }}/datamunged && dub build $compiler --build=release --force"; \
    else \
      cd {{ BINDINGS_DIR }}/datamunged && dub build $compiler --build=release --force; \
    fi'

build-perl: prebuild-perl build-cpp
  {{ NIX_DEVELOP }} .#perl --command bash -c 'cd {{ BINDINGS_DIR }}/perldatamunge && rm -rf blib Makefile Makefile.old pm_to_blib MYMETA.* && perl Makefile.PL INSTALL_BASE="$(pwd)/../../build/perl" && make -j{{ JOBS }} && make install'

build-ruby: prebuild-ruby
  {{ NIX_DEVELOP }} .#ruby --command bash -c "set -euo pipefail; cd {{ BINDINGS_DIR }}/octruby/ext/octruby && ruby extconf.rb && make -j1 && dlext=\"\$(ruby -e 'print RbConfig::CONFIG[\"DLEXT\"]')\" && so=\"\$(find . -type f \( -name 'octruby*.so' -o -name \"octruby*.\$dlext\" \) -print -quit)\" && test -n \"\$so\" && mkdir -p ../../lib/octruby && cp -f \"\$so\" \"../../lib/octruby/octruby.\$dlext\""

package-r: prebuild-r
  {{ NIX_DEVELOP }} .#r --command python3 scripts/build_r_source.py

build-r: prebuild-r
  {{ NIX_DEVELOP }} .#r --command bash -c 'mkdir -p build/r/library && R CMD INSTALL -l build/r/library .'

build-tcl: prebuild-tcl
  rm -rf build/datamungetcl
  {{ NIX_DEVELOP }} .#tcl --command bash -c 'cmake -S src/datamungetcl -B build/datamungetcl -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(pkg-config --variable=prefix datamunge);$(pkg-config --variable=prefix arrow)"'
  {{ NIX_DEVELOP }} .#tcl --command bash -c 'cmake --build build/datamungetcl -j{{ JOBS }} --verbose'
  {{ NIX_DEVELOP }} .#tcl --command bash -c 'cp -v {{ BINDINGS_DIR }}/datamungetcl/pkgIndex.tcl build/datamungetcl/'
  # pkgIndex.tcl loads Datamunge[info sharedlibextension] (.dylib on macOS); CMake names the module .so.
  {{ NIX_DEVELOP }} .#tcl --command bash -c 'ext="$(tclsh <<< "puts [info sharedlibextension]")" && if [ "$ext" != ".so" ]; then cp -f build/datamungetcl/Datamunge.so "build/datamungetcl/Datamunge$ext"; fi'

build-lua: prebuild-lua
  rm -rf build/datamungelua
  {{ NIX_DEVELOP }} .#lua --command bash -c 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungelua -B build/datamungelua -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#lua --command bash -c 'cmake --build build/datamungelua -j{{ JOBS }} --verbose'
  {{ NIX_DEVELOP }} .#lua --command bash -c 'rm -rf build/lua/prefix && cmake --install build/datamungelua --prefix build/lua/prefix'

build-rust:
  {{ NIX_DEVELOP }} .#rust --command bash -c 'cargo build --manifest-path {{ BINDINGS_DIR }}/rustdatamunge/Cargo.toml'

build-guile: prebuild-guile
  rm -rf build/datamungeguile
  {{ NIX_DEVELOP }} .#guile --command bash -c 'cmake -S src/datamungeguile -B build/datamungeguile -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_PREFIX_PATH="$(pkg-config --variable=prefix datamunge);$(pkg-config --variable=prefix arrow)"'
  {{ NIX_DEVELOP }} .#guile --command bash -c 'cmake --build build/datamungeguile -j{{ JOBS }} --verbose'
  {{ NIX_DEVELOP }} .#guile --command bash -c 'rm -rf build/guile/prefix && cmake --install build/datamungeguile --prefix build/guile/prefix'

build-octave: prebuild-octave
  {{ NIX_DEVELOP }} .#octave --command bash -c 'cmake -S src/datamungeoctave -B build/datamungeoctave -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(pkg-config --variable=prefix datamunge);$(pkg-config --variable=prefix arrow)"'
  {{ NIX_DEVELOP }} .#octave --command bash -c 'cmake --build build/datamungeoctave -j{{ JOBS }} --verbose'

build-ocaml: prebuild-ocaml
  {{ NIX_DEVELOP }} .#ocaml --command bash -c 'test -n "${DATAMUNGE_PREFIX:-}" || (echo "DATAMUNGE_PREFIX is not set" >&2; exit 1) && cd {{ BINDINGS_DIR }}/datamungeocaml && dune build'

install-ocaml: build-ocaml
  {{ NIX_DEVELOP }} .#ocaml --command bash -c 'test -n "${DATAMUNGE_PREFIX:-}" || (echo "DATAMUNGE_PREFIX is not set" >&2; exit 1) && mkdir -p build/ocaml/prefix && cd {{ BINDINGS_DIR }}/datamungeocaml && dune install --prefix "$(pwd)/../../build/ocaml/prefix"'




# }}} build commands

# {{{ repl commands

repl-javascript: prebuild-javascript build-javascript
  {{ NIX_DEVELOP }} .#javascript --command bash -c 'node'

repl-python: build-python
  {{ NIX_DEVELOP }} .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python

repl-r: build-r
  {{ NIX_DEVELOP }} .#r --command bash -c 'R_LIBS="$(pwd)/build/r/library" R'

repl-php: build-php
  {{ NIX_DEVELOP }} .#php --command bash -c 'php -a --php-ini .user.ini'

repl-perl: build-perl
  {{ NIX_DEVELOP }} .#perl --command bash -c 'export PERL5LIB="$(pwd)/build/perl/lib/perl5:$PERL5LIB" && export LD_LIBRARY_PATH="$(pwd)/build:$LD_LIBRARY_PATH" && perl -de 1'

repl-csharp: build-csharp
  {{ NIX_DEVELOP }} .#csharp --command bash -c 'LD_LIBRARY_PATH="$(pwd)/build/dotnet/release/_deps/datamunge-build:$(pwd)/build/dotnet/release${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" if command -v dotnet-repl >/dev/null 2>&1; then dotnet-repl; elif command -v csi >/dev/null 2>&1; then csi; else echo "No C# REPL found (expected dotnet-repl or csi)" >&2; exit 1; fi'

repl-java:
  {{ NIX_DEVELOP }} .#java --command bash -c 'export LD_LIBRARY_PATH={{ BINDINGS_DIR }}/jdatamunge-datamunge/build/cmake:$LD_LIBRARY_PATH && jshell --class-path ./{{ BINDINGS_DIR }}/jdatamunge/build/libs/jdatamunge.jar'

# C++ REPL on clang-repl, with the std headers and <datamunge/datamunge.hpp>
# already included. Optional EXAMPLE preloads examples/cpp/source/<topic>/<EXAMPLE>.cpp
# and calls its main(), e.g. `just repl-cpp dataframe_ex`.
repl-cpp EXAMPLE="":
  {{ NIX_DEVELOP }} .#cpp --command bash -lc 'exec scripts/cpp_repl.sh {{ EXAMPLE }}'

# Same, but on cling instead of clang-repl.
repl-cling EXAMPLE="":
  {{ NIX_DEVELOP }} .#cpp --command bash -lc 'exec scripts/cpp_repl.sh --cling {{ EXAMPLE }}'

repl-tcl: build-tcl
  {{ NIX_DEVELOP }} .#tcl --command bash -c 'export TCLLIBPATH="$(pwd)/build/datamungetcl${TCLLIBPATH:+ $TCLLIBPATH}" && tclsh'

repl-lua: build-lua
  {{ NIX_DEVELOP }} .#lua --command bash -c 'lua'

repl-ruby: build-ruby
  {{ NIX_DEVELOP }} .#ruby --command bash -c 'irb -r octruby'

repl-ocaml:
  {{ NIX_DEVELOP }} .#ocaml --command bash -c 'utop'

repl-guile: build-guile
  {{ NIX_DEVELOP }} .#guile --command bash -c 'guile'

repl-rust:
  {{ NIX_DEVELOP }} .#rust --command bash -c 'export LD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" && cd {{ BINDINGS_DIR }}/rustdatamunge && evcxr'

repl-octave: build-octave
  {{ NIX_DEVELOP }} .#octave --command bash -c 'octave -qf'

repl-d:
  bash -c 'set -euo pipefail; \
    compiler=""; \
    if command -v ldc2 >/dev/null 2>&1; then compiler="--compiler=ldc2"; elif command -v dmd >/dev/null 2>&1; then compiler="--compiler=dmd"; fi; \
    if command -v nix >/dev/null 2>&1 && {{ NIX_DEVELOP }} .#d --command true >/dev/null 2>&1; then \
      {{ NIX_DEVELOP }} .#d --command bash -c "cd examples/d && dub run $compiler --build=release"; \
    else \
      cd examples/d && dub run $compiler --build=release; \
    fi'

# }}} repl commands

# {{{ test commands

test-python-build: prebuild-python
  rm -rf build/venv/pydatamunge-build
  {{ NIX_DEVELOP }} .#python --command bash -c 'python -m venv --system-site-packages build/venv/pydatamunge-build'
  {{ NIX_DEVELOP }} .#python --command bash -c 'build/venv/pydatamunge-build/bin/python setup.py sdist bdist_wheel'

# Test installed packages, with no source-tree override or older installed binding.
test-python: build-python
  {{ NIX_DEVELOP }} .#python --command bash -c 'env -u PYTHONPATH PYTHONNOUSERSITE=1 build/venv/pydatamunge-run/bin/python -m pytest --import-mode=importlib -q tests/python bindings_tests/python'

test-r: build-r
  {{ NIX_DEVELOP }} .#r --command Rscript -e 'library(datamunger, lib.loc="build/r/library"); stopifnot(startsWith(find.package("datamunger"), normalizePath("build/r/library"))); testthat::test_dir("tests/testthat", stop_on_failure=TRUE); testthat::test_dir("bindings_tests/r", stop_on_failure=TRUE)'


test-csharp: build-csharp
  {{ NIX_DEVELOP }} .#csharp --command bash -c 'LD_LIBRARY_PATH="$(pwd)/build/dotnet/release/_deps/datamunge-build:$(pwd)/build/dotnet/release${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" dotnet test ./{{ BINDINGS_DIR }}/datamungedotnet.tests'

test-java: build-java
  {{ NIX_DEVELOP }} .#java --command bash -c 'export LD_LIBRARY_PATH={{ BINDINGS_DIR }}/jdatamunge-datamunge/build/cmake:$LD_LIBRARY_PATH && gradle test'

test-rust:
  {{ NIX_DEVELOP }} .#rust --command bash -c 'export LD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" && cargo test --manifest-path tests/rust/Cargo.toml'

test-php: build-php
  {{ NIX_DEVELOP }} .#php --command bash -c 'php -d assert.exception=1 -d zend.assertions=1 --php-ini .user.ini tests/php/test_datamunge.php'

test-lua: build-lua
  {{ NIX_DEVELOP }} .#lua --command bash -c 'lua tests/lua/test_datamunge.lua'

test-perl: build-perl
  {{ NIX_DEVELOP }} .#perl --command bash -c 'export PERL5LIB="$(pwd)/build/perl/lib/perl5:$PERL5LIB" && export LD_LIBRARY_PATH="$(pwd)/build:$LD_LIBRARY_PATH" && prove -l tests/perl'

test-tcl: build-tcl
  {{ NIX_DEVELOP }} .#tcl --command bash -c 'export TCLLIBPATH="$(pwd)/build/datamungetcl${TCLLIBPATH:+ $TCLLIBPATH}" && tclsh tests/tcl/test_datamunge.tcl'

test-ruby: build-ruby
  {{ NIX_DEVELOP }} .#ruby --command bash -c 'export LD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" && ruby -I tests/ruby -e "require \"test_datamunge\""'

test-guile: build-guile
  {{ NIX_DEVELOP }} .#guile --command bash -c 'guile --no-auto-compile -s tests/guile/test_datamunge.scm'

test-octave: build-octave
  {{ NIX_DEVELOP }} .#octave --command bash -c 'octave -qf --eval '"'"'test("tests/octave/test_datamunge.m")'"'"''

test-d: prebuild-d
  bash -c 'set -euo pipefail; \
    compiler=""; \
    if command -v ldc2 >/dev/null 2>&1; then compiler="--compiler=ldc2"; elif command -v dmd >/dev/null 2>&1; then compiler="--compiler=dmd"; fi; \
    if command -v nix >/dev/null 2>&1 && {{ NIX_DEVELOP }} .#d --command true >/dev/null 2>&1; then \
      {{ NIX_DEVELOP }} .#d --command bash -c "dub test --root tests/d $compiler --build=release"; \
    else \
      dub test --root tests/d $compiler --build=release; \
    fi'


test-go: build-go
    {{ NIX_DEVELOP }} .#go --command bash -c 'cd {{ BINDINGS_DIR }}/godatamunge && LD_LIBRARY_PATH="$(pwd)/../../build:$LD_LIBRARY_PATH" CGO_CPPFLAGS="-I$(pwd)/../../include" CGO_LDFLAGS="-L$(pwd)/../../build -ldatamunge" go test ./...'


test-javascript: build-javascript
    @echo "Running Javascript Tests"
    {{ NIX_DEVELOP }} .#javascript --command bash -c "npm run test"

test-ocaml:
  {{ NIX_DEVELOP }} .#ocaml --command bash -c 'just install-ocaml && export OCAMLPATH="$(pwd)/build/ocaml/prefix/lib${OCAMLPATH:+:}$OCAMLPATH" && cd tests/ocaml && dune runtest'


test-cpp:
    @echo "Running Tests"
    @bash -c 'set -euo pipefail; \
      gtest_prefix=""; \
      for p in /nix/store/*-gtest-*-dev; do \
        if [ -f "$p/lib/cmake/GTest/GTestConfig.cmake" ]; then gtest_prefix="$p"; break; fi; \
      done; \
      rm -rf build/debug/tests; \
      cmake -S tests -B build/debug/tests -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_MAKE_PROGRAM="$(command -v make)" ${gtest_prefix:+-DCMAKE_PREFIX_PATH="$gtest_prefix"}; \
      cmake --build build/debug/tests -j{{ JOBS }} --verbose; \
      find ./build/ -name "compile_commands.json" -exec cat {} + | jq -s add > compile_commands.json; \
      ctest --test-dir build/debug/tests --output-on-failure'

# Incrementally rebuild the C++ tests and run only the nn (tensor + autograd) suites.
test-nn:
  {{ NIX_DEVELOP }} .#cpp-tools --command bash -c "cmake --build build/debug/tests -j{{ NN_JOBS }} && ./build/debug/tests/run_tests --gtest_filter='Nn*'"


# }}} test commands

# {{{ nn examples (datamunge::nn ports of ~/projects/tflow_examples)

configure-nn-examples:
  {{ NIX_DEVELOP }} .#cpp-tools --command bash -c "cmake -S examples/nn -B build/nn-examples -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=$(command -v make)"

build-nn-examples: configure-nn-examples
  {{ NIX_DEVELOP }} .#cpp-tools --command bash -c "cmake --build build/nn-examples -j{{ NN_JOBS }}"

build-nn-sections: configure-nn-examples
  {{ NIX_DEVELOP }} .#cpp-tools --command cmake --build build/nn-examples --parallel {{ NN_JOBS }} --target nn_sections_1_2

run-nn NAME *ARGS: configure-nn-examples
  {{ NIX_DEVELOP }} .#cpp-tools --command python3 scripts/run_nn_examples.py --build-only {{ NAME }}
  {{ NIX_DEVELOP }} .#cpp-tools --command python3 scripts/run_nn_examples.py --force {{ NAME }} --example-args {{ ARGS }}

# Explicit opt-in to long convergence experiments.
run-nn-full NAME *ARGS: configure-nn-examples
  {{ NIX_DEVELOP }} .#cpp-tools --command python3 scripts/run_nn_examples.py --build-only {{ NAME }}
  {{ NIX_DEVELOP }} .#cpp-tools --command python3 scripts/run_nn_examples.py --full --force {{ NAME }} --example-args {{ ARGS }}

# Run the nn examples that changed since they last passed (each exits nonzero when its checks
# fail). Keys hash each example and every repo header it includes; options: --force (run all),
# --with-library (also rerun after library changes), --list, NAME... (only those examples).
run-all-nn *ARGS: build-nn-examples
  {{ NIX_DEVELOP }} .#cpp-tools --command python3 scripts/run_nn_examples.py {{ ARGS }}

# Optional SWIG-backed replicas of the migrated CNN / autoencoder sections.
prebuild-nn-python: prebuild-python
  {{ NIX_DEVELOP }} .#python --command python scripts/prebuild_nn_api.py python
  {{ NIX_DEVELOP }} .#python --command python scripts/prebuild_nn_examples.py

build-nn-python: prebuild-nn-python
  {{ NIX_DEVELOP }} .#python --command bash -c 'test -x build/venv/pydatamunge-run/bin/python || python -m venv --system-site-packages build/venv/pydatamunge-run'
  {{ NIX_DEVELOP }} .#python --command bash -c 'DATAMUNGE_BUILD_NN_EXAMPLES=1 CMAKE_BUILD_PARALLEL_LEVEL={{ NN_JOBS }} build/venv/pydatamunge-run/bin/python -m pip install --force-reinstall . --no-deps --no-build-isolation'
  touch build/nn-python/enabled

test-nn-python: build-nn-python
  {{ NIX_DEVELOP }} .#python --command build/venv/pydatamunge-run/bin/python -m pytest --import-mode=importlib -q tests/python/test_nn_language_examples.py tests/python/test_nn_functional_api.py

run-nn-python NAME *ARGS: build-nn-python
  {{ NIX_DEVELOP }} .#python --command build/venv/pydatamunge-run/bin/python scripts/run_nn_python_examples.py --force {{ NAME }} --example-args {{ ARGS }}

run-nn-python-full NAME *ARGS: build-nn-python
  {{ NIX_DEVELOP }} .#python --command build/venv/pydatamunge-run/bin/python scripts/run_nn_python_examples.py --full --force {{ NAME }} --example-args {{ ARGS }}

run-all-nn-python *ARGS: build-nn-python
  {{ NIX_DEVELOP }} .#python --command build/venv/pydatamunge-run/bin/python scripts/run_nn_python_examples.py {{ ARGS }}

prebuild-nn-r:
  {{ NIX_DEVELOP }} .#r --command Rscript --vanilla scripts/prebuild_nn_r.R

build-nn-r: prebuild-nn-r
  mkdir -p build/r/library
  {{ NIX_DEVELOP }} .#r --command env DATAMUNGE_USE_SYSTEM_LIB=1 MAKEFLAGS=-j{{ NN_JOBS }} R CMD INSTALL -l build/r/library build/nn-r-native/source/datamungerNN

package-nn-r: build-nn-r
  {{ NIX_DEVELOP }} .#r --command bash -c 'mkdir -p dist && cd dist && R CMD build --no-manual ../build/nn-r-native/source/datamungerNN'

test-nn-r: build-nn-r
  {{ NIX_DEVELOP }} .#r --command Rscript -e 'library(datamungerNN, lib.loc="build/r/library"); testthat::test_dir("bindings_tests/r-nn", stop_on_failure=TRUE)'

run-nn-r NAME *ARGS: build-nn-r
  {{ NIX_DEVELOP }} .#r --command env R_LIBS_USER=build/r/library DATAMUNGE_EXAMPLES_FULL=0 Rscript --vanilla -e 'args <- commandArgs(TRUE); quit(status=datamungerNN::nn_example_run(args[[1L]], args[-1L]))' {{ NAME }} {{ ARGS }}

run-nn-r-full NAME *ARGS: build-nn-r
  {{ NIX_DEVELOP }} .#r --command env R_LIBS_USER=build/r/library DATAMUNGE_EXAMPLES_FULL=1 Rscript --vanilla -e 'args <- commandArgs(TRUE); quit(status=datamungerNN::nn_example_run(args[[1L]], args[-1L]))' {{ NAME }} {{ ARGS }}

run-all-nn-r *ARGS: build-nn-r
  {{ NIX_DEVELOP }} .#r --command env R_LIBS_USER=build/r/library Rscript --vanilla -e 'for (name in datamungerNN::nn_example_names()) { cat("\n==", name, "==\n"); status <- datamungerNN::nn_example_run(name, commandArgs(TRUE)); if (status != 0L) quit(status=status) }' {{ ARGS }}

# Genuine language implementations; these commands never build C++ experiment adapters.
build-nn-api-python: prebuild-python
  {{ NIX_DEVELOP }} .#python --command bash -c 'test -x build/venv/pydatamunge-run/bin/python || python -m venv --system-site-packages build/venv/pydatamunge-run'
  {{ NIX_DEVELOP }} .#python --command bash -c 'DATAMUNGE_USE_SYSTEM_LIB=1 DATAMUNGE_BUILD_NN_EXAMPLES=0 CMAKE_BUILD_PARALLEL_LEVEL={{ NN_JOBS }} build/venv/pydatamunge-run/bin/python -m pip install . --no-deps --no-build-isolation'

# Native workflows and reusable SWIG components; excludes the optional C++ experiment bridge.
test-nn-api-python: build-nn-api-python
  {{ NIX_DEVELOP }} .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python -m pytest --import-mode=importlib -q tests/python/test_nn_example_regressions.py tests/python/test_nn_dataset_assets.py tests/python/test_nn_cifar_workflows.py tests/python/test_nn_memory_workflows.py tests/python/test_nn_components.py tests/python/test_nn_example_workflow_ownership.py tests/python/test_nn_functional_api.py tests/python/test_nn_migration_inventory.py

python-example NAME *ARGS: build-nn-api-python
  {{ NIX_DEVELOP }} .#python --command build/venv/pydatamunge-run/bin/python -I scripts/python_example.py {{ NAME }} {{ ARGS }}

r-example NAME *ARGS: build-nn-r
  {{ NIX_DEVELOP }} .#r --command env R_LIBS_USER=build/r/library Rscript --vanilla -e 'args <- commandArgs(TRUE); quit(status=datamungerNN::nn_example_run(args[[1L]], args[-1L]))' {{ NAME }} {{ ARGS }}

nn-language-status:
  {{ NIX_DEVELOP }} .#cpp-tools --command python3 scripts/nn_migration_status.py --refresh

# Original MNIST IDX files plus balanced, disjoint pools used by autoencoder examples.
nn-mnist-assets:
  {{ NIX_DEVELOP }} .#cpp --command datamunge-prepare-data mnist-assets

nn-mnist-assets-r: build-nn-r
  {{ NIX_DEVELOP }} .#r --command env R_LIBS_USER=build/r/library Rscript --vanilla -e 'root <- Sys.getenv("DATAMUNGE_DATA_DIR", "build/datasets"); datamungerNN::nn_components$dataset_assets$EnsureMnist(file.path(root,"mnist")); datamungerNN::nn_components$dataset_assets$EnsureSaliency(file.path(root,"saliency"))'

nn-memory-assets:
  {{ NIX_DEVELOP }} .#cpp --command datamunge-prepare-data memory

nn-memory-assets-r: build-nn-r
  {{ NIX_DEVELOP }} .#r --command env R_LIBS_USER=build/r/library Rscript --vanilla -e 'root <- Sys.getenv("DATAMUNGE_DATA_DIR", "build/datasets"); datamungerNN::nn_components$memory_components$PrepareData(root, FALSE); cat("Prepared local synthetic memory tasks in", root, "\n")'

# Where each example keeps its workflow (main / local function / pasted or header harness),
# which shared headers it uses and whether they are SWIG-bound. Writes examples/nn/example_audit.json
# and docs/nn-example-audit.md; --check fails when a shared header owns a training loop.
nn-audit *ARGS:
  {{ NIX_DEVELOP }} .#cpp-tools --command python3 scripts/nn_example_audit.py --markdown docs/nn-example-audit.md {{ ARGS }}

# Run C++/Python/R for each NAME (epochs=5) and record passing evidence in the ledger.
validate-nn-ports *ARGS:
  {{ NIX_DEVELOP }} .#cpp-tools --command python3 scripts/validate_nn_ports.py {{ ARGS }}

# Exact persisted-array comparison across the three language output directories.
nn-compare *ARGS:
  {{ NIX_DEVELOP }} .#cpp-tools --command python3 scripts/compare_nn_outputs.py {{ ARGS }}

# }}} nn examples

# {{{ utilities

rename NEW:
  ./rename_datamunge {{ NEW }}

jq:
    {{ NIX_DEVELOP }} . --command bash -c "find ./build -name 'compile_commands.json' -exec cat {} + | jq -s add > compile_commands.json"


[positional-arguments]
playground *ARGS:
    @just TARGET=playground run-cpp "$@"

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
    {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake -S benchmarks -B build/benchmarks -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=$(command -v make)"
    {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake --build build/benchmarks --config Release -j{{ JOBS }}"
    mkdir -p benchmarks/results
    @echo "Running {{ BENCH_TARGET }}"
    python3 scripts/benchmark_backend.py run --no-build --suite {{ BENCH_TARGET }}
    @echo "Results saved to benchmarks/results/"

# Build and run the datamunge::nn speed benchmarks (GEMM, conv, pooling, batch norm, a cnn_mnist epoch).
bench-nn *ARGS:
    {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake -S benchmarks -B build/benchmarks -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=$(command -v make)"
    {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake --build build/benchmarks --config Release --target nn_bench -j{{ JOBS }}"
    mkdir -p benchmarks/results
    python3 scripts/benchmark_backend.py run --no-build --suite nn_bench -- {{ ARGS }}

# Persistent observations and raw-run import; history emits TSV for plotting.
benchmark-history *ARGS:
    python3 scripts/benchmark_backend.py history {{ ARGS }}

benchmark-record FILE *ARGS:
    python3 scripts/benchmark_backend.py record '{{ FILE }}' {{ ARGS }}

benchmark-compare BASE NEW:
    python3 scripts/benchmark_backend.py compare '{{ BASE }}' '{{ NEW }}'

# Focused backend suites; forward Google Benchmark options after --.
bench-backend *ARGS:
    {{ NIX_DEVELOP }} .#cpp-tools --command python3 scripts/benchmark_backend.py run --jobs {{ JOBS }} {{ ARGS }}

bench-backend-smoke:
    {{ NIX_DEVELOP }} .#cpp-tools --command python3 scripts/benchmark_backend.py run --jobs {{ JOBS }} --smoke

# Check the agreed operation matrix and all current tensor optimizer registrations.
bench-backend-audit:
    python3 scripts/benchmark_backend.py audit

# Optional isolated TensorFlow CPU comparison (see benchmarks/TENSORFLOW.md).
bench-tensorflow *ARGS:
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake -S benchmarks -B build/benchmarks -DCMAKE_BUILD_TYPE=Release
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake --build build/benchmarks --config Release --target backend_bench -j {{ JOBS }}
    build/tensorflow-venv/bin/python scripts/benchmark_tensorflow.py {{ ARGS }}

# Existing correctness tests linked to the same core as the benchmarks.
bench-backend-check:
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake -S benchmarks -B build/benchmarks -DCMAKE_BUILD_TYPE=Release -DDATAMUNGE_BENCH_CHECKS=ON
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake --build build/benchmarks --config Release --target backend_checks -j {{ JOBS }}
    ./build/benchmarks/backend_checks
    python3 -m unittest discover -s benchmarks -p 'test*runner.py'



memcheck:
  valgrind --leak-check=full --track-origins=yes ./build/debug/examples/{{TARGET}}


coverage:
    rm -rf build/coverage
    @bash -c 'set -euo pipefail; \
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
    @bash -c 'set -euo pipefail; \
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
  {{ NIX_DEVELOP }} .#cpp --command bash -c 'clang++ -g -O0 debug.cpp -o debug $(pkg-config --cflags datamunge) $(pkg-config --libs-only-L datamunge) $(pkg-config --cflags libxml-2.0) $(pkg-config --libs-only-L libxml-2.0) -std=c++20 -ldatamunge -lxml2'


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
    @bash -c 'set -euo pipefail; \
      export_cmd='\''DATAMUNGE_SKIP_DIRENV=1 emacs --batch -Q -l init.el -- "{{ INPUT }}" "{{ OUTPUT }}"'\''; \
      if command -v nix >/dev/null 2>&1; then \
        nix develop --accept-flake-config .#org-examples --command bash -c "$export_cmd"; \
      else \
        bash -c "$export_cmd"; \
      fi'

org-example NAME="dense_linear_algebra" FORMAT="html":
    just org-export examples/org/{{ NAME }}.org build/org/{{ NAME }}.{{ FORMAT }}

# Build every examples/org/*.org file to HTML and PDF under build/org/.
org-examples:
    @echo "Exporting Org examples -> HTML and PDF"
    @bash -c 'set -euo pipefail; \
      export_cmd='\''set -euo pipefail; shopt -s nullglob; files=(examples/org/*.org); if ((${#files[@]} == 0)); then echo "No Org examples found under examples/org/" >&2; exit 1; fi; mkdir -p build/org; for f in "${files[@]}"; do base="$(basename "$f" .org)"; for format in html pdf; do out="build/org/${base}.${format}"; echo "Exporting $f -> $out"; DATAMUNGE_SKIP_DIRENV=1 emacs --batch -Q -l init.el -- "$f" "$out"; done; done'\''; \
      if command -v nix >/dev/null 2>&1; then \
        nix develop --accept-flake-config .#org-examples --command bash -c "$export_cmd"; \
      else \
        bash -c "$export_cmd"; \
      fi'

# Regenerate examples/mepml/*.mepml from examples/org/*.org (needs mep-mepml on PATH),
# compiling and running each report's C++ block to fill in its results.
mepml-from-org:
    python3 scripts/org_to_mepml.py --run examples/org examples/mepml

# Run every code block of the docs' mepml reports (docs/mepml/examples/**/*.mepml, or
# FILES) with mep (the flake's, from the cpp shell), as its C-c C-c does, writing
# each block's results and figures back into its report. The C++ blocks build against
# datamunge found by pkg-config. Prints each block's outcome and fails if one failed
# (a failed block keeps its old results). `just docs` runs only the changed reports.
docs-mepml-run *FILES:
    #!/usr/bin/env bash
    set -euo pipefail
    files=({{ if FILES == "" { "$(find docs/mepml/examples -name '*.mepml' | sort)" } else { FILES } }})
    pc="{{ MEPML_PKG_CONFIG_PATH }}"
    # mep finds its own libraries; an outside LD_LIBRARY_PATH can stop it starting.
    {{ NIX_DEVELOP }} .#cpp --command env -u LD_LIBRARY_PATH ${pc:+PKG_CONFIG_PATH="$pc"} \
      bash -c 'python3 scripts/website/reports.py --check-toolchain "$@" && exec mep-mepml run --timeout 1800 "$@"' _ "${files[@]}"

# The website, as `just docs` builds it (kept for its older name).
docs-mepml *ARGS: (docs ARGS)

# Bump the flake's mep (the editor and `mep-mepml`, which the docs build uses) to the
# latest commit of its main branch.
update-mep:
    nix flake update mep

prebuild-docs-pages:
    @echo "Exporting Org pages -> Markdown"
    @bash -c 'set -euo pipefail; \
      export_cmd='\''set -euo pipefail; shopt -s nullglob; for f in docs/org/pages/*.org; do base="$(basename "$f" .org)"; out="docs/pages/${base}.md"; emacs --batch -Q -l docs/org-to-md.el -- "$f" "$out"; done'\''; \
      if command -v nix >/dev/null 2>&1; then \
        if nix develop --accept-flake-config .#docs-pages --command bash -c "$export_cmd" >/dev/null 2>&1; then \
          nix develop --accept-flake-config .#docs-pages --command bash -c "$export_cmd"; \
        else \
          bash -c "$export_cmd"; \
        fi; \
      else \
        bash -c "$export_cmd"; \
      fi'

# Convenience alias: build every docs/org/pages/*.org page to Markdown (docs/pages/*.md).
docs-md: prebuild-docs-pages
    @echo "Org pages exported to Markdown under docs/pages/"

# Export a single arbitrary Org file to Markdown, executing Babel blocks best-effort
# (failures are captured into an "Export notes" section instead of aborting the export).
docs-md-file INPUT OUTPUT:
    @echo "Exporting {{ INPUT }} -> {{ OUTPUT }} (Markdown, best-effort Babel)"
    @bash -c 'set -euo pipefail; \
      export_cmd='\''emacs --batch -Q -l docs/org-to-md.el -- "{{ INPUT }}" "{{ OUTPUT }}"'\''; \
      if command -v nix >/dev/null 2>&1; then \
        nix develop --accept-flake-config .#docs-pages --command bash -c "$export_cmd"; \
      else \
        bash -c "$export_cmd"; \
      fi'

# Build the documentation website in build/debug/docs/site: Doxygen reads the headers
# into XML, mep runs the mepml reports that changed since their last run (and every one
# that failed) and exports them, and scripts/website/build.py writes the site, with a
# "Report build status" page. Fails if a report's code block failed. ARGS go to CMake:
# -DDOCS_MEPML_FORCE=ON reruns every report, -DDOCS_MEPML_JOBS=N runs N reports at once
# (default: one per core, as memory allows), -DDOCS_MEPML_PCH=OFF compiles the blocks
# with plain g++ (not clang++ with a precompiled dm.hpp), -DDOCS_MEPML_RUN=OFF uses the
# stored results, -DDOCS_MEPML_ALLOW_FAILURES=ON builds the site anyway.
docs *ARGS:
    @echo "Building docs"
    {{ NIX_DEVELOP }} .#cpp --command cmake -U MEP_MEPML_EXECUTABLE -S docs -B build/debug/docs -DDOCS_MEPML_PKG_CONFIG_PATH="{{ MEPML_PKG_CONFIG_PATH }}" {{ ARGS }}
    {{ NIX_DEVELOP }} .#cpp --command cmake --build build/debug/docs --target Website

# Serve the built website at http://localhost:PORT (it also works opened from disk).
docs-serve PORT="8000":
    @test -f build/debug/docs/site/index.html || { echo "No website yet: run 'just docs' first" >&2; exit 1; }
    @echo "Serving the website at http://localhost:{{ PORT }}/"
    python3 -m http.server {{ PORT }} --bind 127.0.0.1 -d build/debug/docs/site

# Embed data-raw/datasets/*.csv into src/datamunge/datasets/embedded/ (the built-in datasets).
generate-datasets:
    python3 -I scripts/generate_datasets.py

# Inventory every maintained core definition, including private/static internals.
# Add --check and repeatable --path arguments to enforce a completed source batch.
# Regenerate include/datamunge/dir_registry.hpp (the member tables behind dm::dir()) from fresh Doxygen XML.
generate-dir:
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake -S docs -B build/debug/docs
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake --build build/debug/docs --target GenerateDocsXml
    {{ NIX_DEVELOP }} .#cpp-tools --command python3 scripts/generate_dir_registry.py

docs-audit *ARGS:
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake -S docs -B build/debug/docs
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake --build build/debug/docs --target GenerateDocsXml
    {{ NIX_DEVELOP }} .#cpp-tools --command python3 scripts/check_documentation.py --json build/debug/docs/coverage.json {{ ARGS }}

# Check descriptions returned by clangd at representative source definitions.
docs-hover *ARGS:
    {{ NIX_DEVELOP }} .#cpp-tools --command python3 scripts/check_documentation_hover.py {{ ARGS }}

# Native library simulation tests; optional GPU tests report unavailable backends explicitly.
# Opt-in SDK build; the host NVIDIA driver is discovered at runtime, including NixOS.
build-cuda *CMAKE_ARGS:
    {{ NIX_DEVELOP }} .#cuda --command cmake -S . -B build/cuda -DCMAKE_BUILD_TYPE=Release -DDATAMUNGE_ENABLE_CUDA=ON -DDATAMUNGE_ENABLE_CUBLAS=ON {{ CMAKE_ARGS }}
    {{ NIX_DEVELOP }} .#cuda --command cmake --build build/cuda --target datamunge -j{{ JOBS }}

test-cuda *CMAKE_ARGS:
    {{ NIX_DEVELOP }} .#cuda --command cmake -S tests -B build/cuda-tests -DCMAKE_BUILD_TYPE=Release -DDATAMUNGE_ENABLE_CUDA=ON -DDATAMUNGE_ENABLE_CUBLAS=ON {{ CMAKE_ARGS }}
    {{ NIX_DEVELOP }} .#cuda --command cmake --build build/cuda-tests --target run_simulation_tests run_nn_device_tests check_simulation_allocations -j{{ JOBS }}
    {{ NIX_DEVELOP }} .#cuda --command env DATAMUNGE_REQUIRE_CUDA=1 ctest --test-dir build/cuda-tests -R '^(simulation|nn-device)\.' --output-on-failure

test-simulation *CMAKE_ARGS:
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake -S tests -B build/simulation -DCMAKE_BUILD_TYPE=Release {{ CMAKE_ARGS }}
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake --build build/simulation --target run_simulation_tests check_simulation_allocations -j{{ JOBS }}
    {{ NIX_DEVELOP }} .#cpp-tools --command ctest --test-dir build/simulation -R '^simulation\.' --output-on-failure

# Regenerate the native Vulkan shader test asset with a standard SDK shader compiler.
prebuild-compute-fixtures *ARGS:
    python3 scripts/compile_compute_fixture.py {{ ARGS }}

prebuild-simulation-shaders *ARGS:
    python3 scripts/prebuild_simulation_shaders.py {{ ARGS }}

# Native tensor storage shaders; generated SPIR-V is consumed without an SDK.
prebuild-nn-storage-shaders *ARGS:
    python3 scripts/prebuild_nn_storage_shaders.py {{ ARGS }}

test-nn-device *CMAKE_ARGS:
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake -S tests -B build/simulation -DCMAKE_BUILD_TYPE=Release {{ CMAKE_ARGS }}
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake --build build/simulation --target run_nn_device_tests -j{{ JOBS }}
    {{ NIX_DEVELOP }} .#cpp-tools --command ctest --test-dir build/simulation -R '^nn-device\.' --output-on-failure

test-simulation-examples: test-simulation
    {{ NIX_DEVELOP }} .#cpp-tools --command python3 scripts/check_simulation_examples.py --library-dir build/simulation/_deps/datamunge-build

# Real Doxygen fixtures verify that incomplete or stale inventories cannot pass.
test-docs:
    {{ NIX_DEVELOP }} .#cpp-tools --command python3 -m unittest discover -s tests/python -p test_documentation_audit.py -v
    {{ NIX_DEVELOP }} .#cpp-tools --command python3 -m pytest -q tests/python/test_website_toolchain.py tests/python/test_website_model_diagrams.py

docs-bindings: prebuild-swig
    @echo "Binding documentation is emitted as SWIG-generated docstrings/comments (per language)."

docs-all: docs docs-bindings
    @echo "Core + binding docs are up to date."

# README.mepml is the source of README.org: export it with mep's mepml tool
# (build/README.org, which the document's //? Post: copies back to the top).
# Edit README.mepml, not README.org.
readme MEP_MEPML="mep-mepml":
    {{ MEP_MEPML }} build README.mepml org

# }}} docs commands

# {{{ pdf commands
#
# PDF export reuses org-export's generic init.el pipeline (backend inferred from the
# .pdf extension). SVG figures can't be rasterized by pdflatex directly, so init.el
# converts every \includegraphics{...svg} to a sibling PDF via rsvg-convert on export;
# both pdflatex and rsvg-convert live in the .#docs-pages devShell.

# Build one examples/org/*.org file to PDF, e.g. `just org-pdf lda_iris_analysis`.
org-pdf NAME="dense_linear_algebra":
    just org-export examples/org/{{ NAME }}.org build/org/{{ NAME }}.pdf

# Build every docs/org/pages/*.org page to PDF under build/org/pages-pdf/.
docs-pdf:
    @echo "Exporting Org pages -> PDF"
    @bash -c 'set -euo pipefail; \
      export_cmd='\''set -euo pipefail; shopt -s nullglob; mkdir -p build/org/pages-pdf; for f in docs/org/pages/*.org; do base="$(basename "$f" .org)"; out="build/org/pages-pdf/${base}.pdf"; emacs --batch -Q -l init.el -- "$f" "$out"; done'\''; \
      if command -v nix >/dev/null 2>&1; then \
        nix develop --accept-flake-config .#docs-pages --command bash -c "$export_cmd"; \
      else \
        bash -c "$export_cmd"; \
      fi'

# Open a PDF with the user's preferred viewer: $BROWSER, then the first common
# viewer found on PATH.
view-pdf PATH:
    @bash -c 'set -euo pipefail; \
      f="{{ PATH }}"; \
      if [ ! -f "$f" ]; then echo "No such file: $f" >&2; exit 1; fi; \
      if [ -n "${BROWSER:-}" ]; then exec "$BROWSER" "$f"; fi; \
      for viewer in xdg-open zathura evince okular mupdf open; do \
        if command -v "$viewer" >/dev/null 2>&1; then exec "$viewer" "$f"; fi; \
      done; \
      echo "No PDF viewer found on PATH; open $f manually." >&2; exit 1'

# Build then open a single example's PDF, e.g. `just view-org-pdf lda_iris_analysis`.
view-org-pdf NAME="dense_linear_algebra": (org-pdf NAME)
    just view-pdf build/org/{{ NAME }}.pdf

# }}} pdf commands

# {{{ example commands

examples:
    @echo "Building Examples"
    {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake -S examples/cpp -B build/debug/examples --preset=debug -DCMAKE_MAKE_PROGRAM=$(command -v make) -DBUILD_W_INSTALLED=OFF"
    {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake --build build/debug/examples -j{{ JOBS }} --verbose"
    {{ NIX_DEVELOP }} .#cpp --command bash -c "find ./build -name 'compile_commands.json' -exec cat {} + | jq -s add > compile_commands.json"
    # nix develop . --command bash -c "make -C ./build/debug/examples -j10 --verbose"

example EXAMPLE:
    @echo "Running Example {{ EXAMPLE }}:"
    ./build/debug/examples/{{ EXAMPLE }}

example-python:
    {{ NIX_DEVELOP }} .#python --command bash -c "python examples/python/datamunge_ex.py"

# }}} example commands

# {{{ view commands

view-flamechart:
    $(BROWSER) ./flamegraph.svg

view-docs:
    @echo "Opening docs"
    qutebrowser ./build/debug/docs/site/index.html

# }}} view commands

# {{{ windows specifics

windows-run: build
    @echo "Running target ${TARGET}"
    build/debug/examples/${TARGET}

# }}} windows specifics

# {{{ all commands

test-nn-model-plot:
  {{ NIX_DEVELOP }} .#cpp-tools --command cmake -S tests -B build/debug/tests -DCMAKE_BUILD_TYPE=Debug
  {{ NIX_DEVELOP }} .#cpp-tools --command cmake --build build/debug/tests --target run_nn_model_plot_tests -j{{ NN_JOBS }}
  {{ NIX_DEVELOP }} .#cpp-tools --command ctest --test-dir build/debug/tests -R '^nn-model-plot\.' --output-on-failure --no-tests=error

test-nn-metrics-plot:
  {{ NIX_DEVELOP }} .#cpp-tools --command cmake -S tests -B build/debug/tests -DCMAKE_BUILD_TYPE=Debug
  {{ NIX_DEVELOP }} .#cpp-tools --command cmake --build build/debug/tests --target run_nn_metrics_plot_tests -j{{ NN_JOBS }}
  {{ NIX_DEVELOP }} .#cpp-tools --command ctest --test-dir build/debug/tests -R '^nn-metrics-plot\.' --output-on-failure --no-tests=error

test-nix:
  nix flake check --accept-flake-config -L{{ SWIG_OVERRIDE }}

build-nix PACKAGE="datamunge":
  nix build --accept-flake-config{{ SWIG_OVERRIDE }} ".#{{ PACKAGE }}"

update-java-deps:
  nix/update-jdatamunge-gradle-deps.sh

update-csharp-deps:
  nix/update-nuget-deps.sh

all-test:
  bash ./scripts/test_all.sh

test-all: all-test

# }}} all commands
