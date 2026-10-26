#!/usr/bin/env bash
# Launch a C++ REPL (clang-repl or cling) with datamunge already loaded.
#
# Usage (normally via `just repl-cpp` / `just repl-cling`):
#   scripts/cpp_repl.sh [--cling] [--bare] [example]
#
#   scripts/cpp_repl.sh                  # clang-repl, datamunge headers preloaded
#   scripts/cpp_repl.sh dataframe_ex     # ... plus examples/cpp/source/dataframe_ex.cpp, main() called
#   scripts/cpp_repl.sh --cling          # cling instead of clang-repl
#   scripts/cpp_repl.sh --bare           # skip the prelude (nothing included, nothing using'd)
set -euo pipefail

engine="clang-repl"
prelude=1
example=""
while (($#)); do
  case "$1" in
    --cling) engine="cling" ;;
    --clang-repl) engine="clang-repl" ;;
    --bare) prelude=0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) example="$1" ;;
  esac
  shift
done

example_path=""
if [[ -n "$example" ]]; then
  for candidate in "$example" "examples/cpp/source/${example}.cpp" "examples/cpp/source/${example}"; do
    if [[ -f "$candidate" ]]; then example_path="$candidate"; break; fi
  done
  if [[ -z "$example_path" ]]; then
    echo "No such example: $example (looked in examples/cpp/source/)" >&2
    exit 1
  fi
fi

dm_cflags="$(pkg-config --cflags datamunge)"
dm_libdir="$(pkg-config --variable=libdir datamunge)"

# Match CMakeLists.txt's `CXX_STANDARD 23` for libdatamunge.so.
std="${DATAMUNGE_REPL_STD:-c++23}"

# nixpkgs' cling (clang 18) and clang-repl (clang 21) both reject gcc 15's
# libstdc++ headers unless these two diagnostics are downgraded to warnings.
compat_flags=(-Wno-invalid-constexpr -Wno-unknown-attributes)

lines=()
if [[ "$engine" == "cling" ]]; then
  cmd=(cling "${compat_flags[@]}" $dm_cflags "-L${dm_libdir}" -ldatamunge "-std=${std}")
  quit=".q"
  cat >&2 <<'WARN'
!! WARNING: nixpkgs' cling is clang 18 against gcc 15's libstdc++, and it
!! MISCOMPILES datamunge: calls silently return zeroed results and the heap
!! gets corrupted (e.g. describe_numeric() on a 3-row column reports n=0).
!! It is kept only for comparison -- use `just repl-cpp` (clang-repl) for real work.
WARN
else
  # clang-repl is the unwrapped binary, so it does not inherit the nix clang
  # wrapper's system include paths. Recover them from the wrapped clang++.
  mapfile -t sysinc < <(
    echo | clang++ -x c++ -E -v - 2>&1 |
      sed -n '/#include <\.\.\.> search starts here:/,/End of search list./p' |
      grep '^ /' | sed 's/^ //'
  )
  cmd=(clang-repl)
  for d in "${sysinc[@]}"; do cmd+=(-Xcc -isystem -Xcc "$d"); done
  for f in "${compat_flags[@]}" $dm_cflags "-std=${std}"; do cmd+=(-Xcc "$f"); done
  # clang-repl has no -l/-L; it dlopens shared libraries with %lib.
  lines+=("%lib ${dm_libdir}/libdatamunge.so")
  quit="%quit"
fi

# Neither interpreter preloads the standard library, so a bare `std::cout << ...`
# fails with "no member named 'cout' in namespace 'std'" -- and in cling that first
# failure then cascades into a wall of libstdc++ errors on every later line.
if ((prelude)); then
  lines+=(
    "#include <iostream>"
    "#include <optional>"
    "#include <string>"
    "#include <vector>"
    "#include <datamunge/datamunge.hpp>"
    # NOT `using namespace datamunge;` -- the umbrella header has facade types
    # (datamunge::DataFrame) that collide by name with the real ones
    # (datamunge::dstruct::DataFrame), making every unqualified use ambiguous.
    "namespace dm = datamunge;"
  )
fi

if [[ -n "$example_path" ]]; then
  lines+=("#include \"${example_path}\"" "main();")
fi

hint="# ${engine}: datamunge loaded"
((prelude)) && hint+=" (std headers + <datamunge/datamunge.hpp>; dm = datamunge, e.g. dm::dstruct::DataFrame)"
[[ -n "$example_path" ]] && hint+="; ${example_path} ran"
echo "${hint}; type ${quit} to exit" >&2

if ((${#lines[@]} == 0)); then
  exec "${cmd[@]}"
fi

# Feed the preload lines first, then hand stdin back over to the user.
if [[ -t 0 ]]; then
  printf '%s\n' "${lines[@]}" | cat - /dev/tty | "${cmd[@]}"
else
  { printf '%s\n' "${lines[@]}"; cat; } | "${cmd[@]}"
fi
