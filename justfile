TARGET := "bspline_ex"
BENCH_TARGET := "linalg_bench"
JOBS := "20"

# Set DATAMUNGE_SWIG_SRC to a local checkout of the swig fork (e.g. ~/projects/swig) to build
# against it instead of the github:jordanschupbach/swig revision pinned in flake.lock.
SWIG_SRC := env_var_or_default("DATAMUNGE_SWIG_SRC", "")
SWIG_OVERRIDE := if SWIG_SRC == "" { "" } else { " --override-input swig git+file://" + SWIG_SRC }
NIX_DEVELOP := "nix develop --accept-flake-config --option eval-cache false" + SWIG_OVERRIDE
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
  {{ NIX_DEVELOP }} .#python --command bash -c 'build/venv/pydatamunge-run/bin/python examples/python/{{ TARGET }}.py'

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
    for f in examples/cpp/source/*.cpp; do n="$(basename "$f" .cpp)"; b="build/debug/examples/$n"; \
      printf "\n========== %s ==========\n" "$n"; \
      if [ -x "$b" ] && "$b"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## cpp: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-cpp: examples
    @echo "Running target {{ TARGET }}"
    {{ NIX_DEVELOP }} .#cpp --command bash -c './build/debug/examples/{{ TARGET }}'

run-plot: examples
    @echo "Running plot example"
    {{ NIX_DEVELOP }} .#cpp --command bash -c './build/debug/examples/plot_ex'

view-plot: examples
    @echo "Running plot example and opening the scatter plot"
    {{ NIX_DEVELOP }} .#cpp --command bash -c './build/debug/examples/plot_ex'

run-benchmark:
    @echo "Running Benchmarks"
    ./build/benchmarks/${BENCH_TARGET}


# }}} run commands

# {{{ prebuild commands

# Update the pinned swig fork revision in flake.lock to the latest github:jordanschupbach/swig.
update-swig:
  nix flake update swig

prebuild-swig: prebuild-python prebuild-javascript prebuild-csharp prebuild-r prebuild-perl prebuild-ruby prebuild-tcl prebuild-lua prebuild-d prebuild-guile prebuild-octave prebuild-go prebuild-php prebuild-java prebuild-ocaml
  @echo "SWIG wrappers regenerated (with Doxygen comments enabled)"

prebuild-python:
  {{ NIX_DEVELOP }} .#cpp --command bash -c "cd ./include && swig -doxygen -c++ -python -o ../src/datamunge_python_wrap.cpp -oh ../src/datamunge_python_wrap.h ../src/pydatamunge/swig/pydatamunge.i && mv ../src/datamunge.py ../src/pydatamunge/datamunge.py"

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
  {{ NIX_DEVELOP }} .#cpp --command bash -c "cd ./include && swig -c++ -r -o ../src/datamunge_r_wrap.cpp -oh ../src/datamunge_r_wrap.h ../src/datamunger/swig/datamunger.i && mv ../src/datamunger.R ../R"

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


build: build-debug

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
    {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake -S . -B build"
    {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake --build build -j{{ JOBS }} --verbose"
    {{ NIX_DEVELOP }} .#cpp --command bash -c "find ./build -name 'compile_commands.json' -exec cat {} + | jq -s add > compile_commands.json"


build-csharp: prebuild-csharp
  {{ NIX_DEVELOP }} .#csharp --command bash -c "rm -rf build/dotnet/release"
  {{ NIX_DEVELOP }} .#csharp --command bash -c "cmake -S ./{{ BINDINGS_DIR }}/datamungedotnet -B build/dotnet/release -DCMAKE_MAKE_PROGRAM=$(command -v make)"
  {{ NIX_DEVELOP }} .#csharp --command bash -c "cmake --build build/dotnet/release -j{{ JOBS }} --verbose"
  {{ NIX_DEVELOP }} .#csharp --command bash -c "cd ./{{ BINDINGS_DIR }}/datamungedotnet && dotnet build"


build-javascript: prebuild-javascript
    {{ NIX_DEVELOP }} .#jsbuild --command bash -c "npm --prefix . run build"

build-python: prebuild-python
  rm -rf build/venv/pydatamunge-run
  {{ NIX_DEVELOP }} .#python --command bash -c 'python -m venv --system-site-packages build/venv/pydatamunge-run'
  {{ NIX_DEVELOP }} .#python --command bash -c 'build/venv/pydatamunge-run/bin/python -m pip install -e . --no-build-isolation'


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

build-r: prebuild-r
  rm -rf build/r/library
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

repl-python: prebuild-python
  {{ NIX_DEVELOP }} .#python --command bash -c 'ipython'

repl-r: prebuild-r
  {{ NIX_DEVELOP }} .#r --command bash -c 'R'

repl-php: build-php
  {{ NIX_DEVELOP }} .#php --command bash -c 'php -a --php-ini .user.ini'

repl-perl: build-perl
  {{ NIX_DEVELOP }} .#perl --command bash -c 'export PERL5LIB="$(pwd)/build/perl/lib/perl5:$PERL5LIB" && export LD_LIBRARY_PATH="$(pwd)/build:$LD_LIBRARY_PATH" && perl -de 1'

repl-csharp: build-csharp
  {{ NIX_DEVELOP }} .#csharp --command bash -c 'LD_LIBRARY_PATH="$(pwd)/build/dotnet/release/_deps/datamunge-build:$(pwd)/build/dotnet/release${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" if command -v dotnet-repl >/dev/null 2>&1; then dotnet-repl; elif command -v csi >/dev/null 2>&1; then csi; else echo "No C# REPL found (expected dotnet-repl or csi)" >&2; exit 1; fi'

repl-java:
  {{ NIX_DEVELOP }} .#java --command bash -c 'export LD_LIBRARY_PATH={{ BINDINGS_DIR }}/jdatamunge-datamunge/build/cmake:$LD_LIBRARY_PATH && jshell --class-path ./{{ BINDINGS_DIR }}/jdatamunge/build/libs/jdatamunge.jar'

repl-cpp:
  {{ NIX_DEVELOP }} .#cpp --command bash -c 'cling $(pkg-config --cflags datamunge) $(pkg-config --libs-only-L datamunge) -ldatamunge -std=c++17'

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

test-python: prebuild-python
  rm -rf build/venv/pydatamunge
  {{ NIX_DEVELOP }} .#python --command bash -c 'python -m venv --system-site-packages build/venv/pydatamunge'
  {{ NIX_DEVELOP }} .#python --command bash -c 'build/venv/pydatamunge/bin/python -m pip install -e . --no-build-isolation'
  # PYTHONPATH=src is required despite the editable install: --system-site-packages pulls in
  # the nix flake's own pinned pydatamunge build, which otherwise shadows this fresh local one
  # on sys.path (the pinned build wins even though the venv's own editable install should take
  # precedence, since it's older code without the changes just built above).
  {{ NIX_DEVELOP }} .#python --command bash -c 'PYTHONPATH=src build/venv/pydatamunge/bin/python -m pytest -q tests/python'

test-r: build-r
  {{ NIX_DEVELOP }} .#r --command bash -c 'R -q -e "testthat::test_local(\".\")"'

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
  {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake --build build/debug/tests -j{{ JOBS }} && ./build/debug/tests/run_tests --gtest_filter='Nn*'"


# }}} test commands

# {{{ nn examples (datamunge::nn ports of ~/projects/tflow_examples)

build-nn-examples:
  {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake -S examples/nn -B build/nn-examples -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=$(command -v make)"
  {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake --build build/nn-examples -j{{ JOBS }}"

run-nn NAME *ARGS: build-nn-examples
  ./build/nn-examples/bin/{{ NAME }} {{ ARGS }}

# Run every nn example; each exits nonzero when its checks fail.
run-all-nn: build-nn-examples
  @bash -c 'pass=0; fail=0; failed=""; \
    for f in $(find examples/nn -path "examples/nn/[0-9]*" -name "*.cpp" | sort); do n="$(basename "$f" .cpp)"; \
      printf "\n========== %s ==========\n" "$n"; \
      if ./build/nn-examples/bin/"$n"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## nn examples: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

# }}} nn examples

# {{{ utilities

rename NEW:
  ./rename_datamunge {{ NEW }}

jq:
    {{ NIX_DEVELOP }} . --command bash -c "find ./build -name 'compile_commands.json' -exec cat {} + | jq -s add > compile_commands.json"


playground:
    @echo "Building playground"
    {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake -S playground -B build/debug/playground"
    {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake --build build/debug/playground -j {{JOBS}} --verbose"
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
    {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake -S benchmarks -B build/benchmarks -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=$(command -v make)"
    {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake --build build/benchmarks -j{{ JOBS }}"
    @echo "Running {{ BENCH_TARGET }}"
    ./build/benchmarks/{{ BENCH_TARGET }} --benchmark_format=json \
        --benchmark_out="benchmarks/results/$(date +%Y%m%d_%H%M%S).json"
    @echo "Results saved to benchmarks/results/"

# Build and run the datamunge::nn speed benchmarks (GEMM, conv, pooling, batch norm, a cnn_mnist epoch).
bench-nn *ARGS:
    {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake -S benchmarks -B build/benchmarks -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=$(command -v make)"
    {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake --build build/benchmarks --target nn_bench -j{{ JOBS }}"
    mkdir -p benchmarks/results
    ./build/benchmarks/nn_bench --benchmark_format=json --benchmark_out="benchmarks/results/nn_$(date +%Y%m%d_%H%M%S).json" \
        --benchmark_out_format=json {{ ARGS }}

benchmark-compare BASE NEW:
    @echo "Comparing {{ BASE }} vs {{ NEW }}"
    python3 build/benchmarks/_deps/benchmark-src/tools/compare.py benchmarks \
        benchmarks/results/{{ BASE }} benchmarks/results/{{ NEW }}


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

docs: build
    @echo "Building docs"
    just prebuild-docs-pages
    {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake -S docs -B build/debug/docs"
    {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake --build build/debug/docs --target GenerateDocs"

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
    qutebrowser ./build/debug/docs/doxygen/html/index.html

# }}} view commands

# {{{ windows specifics

windows-run: build
    @echo "Running target ${TARGET}"
    build/debug/examples/${TARGET}

# }}} windows specifics

# {{{ all commands

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
