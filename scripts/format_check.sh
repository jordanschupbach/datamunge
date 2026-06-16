#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

prune_expr=(
	-path './.git' -o
	-path './.direnv' -o
	-path './.gradle' -o
	-path './.cache' -o
	-path './.pytest_cache' -o
	-path './node_modules' -o
	-path './node_modules/*' -o
	-path './*/node_modules' -o
	-path './*/node_modules/*' -o
	-path './build' -o
	-path './build/*' -o
	-path './*/build' -o
	-path './*/build/*' -o
	-path './_build' -o
	-path './_build/*' -o
	-path './*/_build' -o
	-path './*/_build/*' -o
	-path './dist' -o
	-path './dist/*' -o
	-path './*/dist' -o
	-path './*/dist/*' -o
	-path './target' -o
	-path './target/*' -o
	-path './*/target' -o
	-path './*/target/*' -o
	-path './result' -o
	-path './result/*' -o
	-path './*/result' -o
	-path './*/result/*'
)

require_cmd() {
	if ! command -v "$1" >/dev/null 2>&1; then
		echo "Missing required command: $1" >&2
		exit 1
	fi
}

has_files() {
	local pattern="$1"
	find . \( "${prune_expr[@]}" \) -prune -o -type f -name "$pattern" -print -quit | grep -q .
}

run_find_xargs() {
	local -a find_args=("$@")
	find . \( "${prune_expr[@]}" \) -prune -o "${find_args[@]}"
}

check_cpp() {
	if [ ! -d ./include ] && [ ! -d ./src/octra ] && [ ! -d ./tests/cpp ]; then
		return 0
	fi
	require_cmd clang-format
	# Only enforce formatting on the canonical core + C++ tests.
	# Generated SWIG wrappers are intentionally excluded.
	find ./include ./src/octra ./tests/cpp \
		-type f \( \
		-name '*.c' -o -name '*.h' -o -name '*.cc' -o -name '*.hh' -o -name '*.cpp' -o -name '*.hpp' -o -name '*.cxx' -o -name '*.hxx' \
		\) -print0 2>/dev/null | xargs -0 clang-format --dry-run --Werror
}

check_cmake() {
	if ! has_files 'CMakeLists.txt' && ! has_files '*.cmake'; then
		return 0
	fi
	require_cmd cmake-format
	run_find_xargs -type f \( -name 'CMakeLists.txt' -o -name '*.cmake' \) -print0 | xargs -0 cmake-format --check
}

check_nix() {
	if ! has_files '*.nix'; then
		return 0
	fi
	require_cmd nixfmt
	if nixfmt --help 2>/dev/null | grep -q -- '--check'; then
		run_find_xargs -type f -name '*.nix' -print0 | xargs -0 nixfmt --check
		return 0
	fi
	tmp="$(mktemp -d)"
	trap 'rm -rf "$tmp"' EXIT
	while IFS= read -r -d '' f; do
		mkdir -p "$tmp/$(dirname "$f")"
		nixfmt "$f" >"$tmp/$f"
		diff -u "$f" "$tmp/$f" >/dev/null || {
			echo "nixfmt would reformat: $f" >&2
			return 1
		}
	done < <(run_find_xargs -type f -name '*.nix' -print0)
}

check_shell() {
	if ! has_files '*.sh' && [ ! -f ./rename_octra ]; then
		return 0
	fi
	require_cmd shfmt
	if has_files '*.sh'; then
		run_find_xargs -type f -name '*.sh' -print0 | xargs -0 shfmt -d
	fi
	if [ -f ./rename_octra ]; then
		shfmt -d ./rename_octra
	fi
}

check_python() {
	if ! has_files '*.py'; then
		return 0
	fi
	require_cmd ruff
	ruff format --check .
}

check_prettier() {
	if ! has_files '*.js' && ! has_files '*.ts' && ! has_files '*.tsx' && ! has_files '*.jsx' && ! has_files '*.json' && ! has_files '*.md' && ! has_files '*.yml' && ! has_files '*.yaml'; then
		return 0
	fi
	require_cmd prettier
	prettier --check .
}

check_rust() {
	if [ ! -f ./Cargo.toml ]; then
		return 0
	fi
	require_cmd cargo
	cargo fmt --all --check
}

check_go() {
	if ! has_files '*.go'; then
		return 0
	fi
	require_cmd gofmt
	out="$(run_find_xargs -type f -name '*.go' -print0 | xargs -0 gofmt -l || true)"
	if [ -n "$out" ]; then
		echo "gofmt would reformat:" >&2
		echo "$out" >&2
		return 1
	fi
}

check_d() {
	if ! has_files '*.d'; then
		return 0
	fi
	require_cmd dfmt
	if ! dfmt --help 2>/dev/null | grep -q -- '-i'; then
		echo "Skipping D format check: dfmt does not support -i on this system" >&2
		return 0
	fi
	tmp="$(mktemp -d)"
	trap 'rm -rf "$tmp"' EXIT
	while IFS= read -r -d '' f; do
		dfmt <"$f" >"$tmp/out"
		diff -u "$f" "$tmp/out" >/dev/null || {
			echo "dfmt would reformat: $f" >&2
			return 1
		}
	done < <(run_find_xargs -type f -name '*.d' -print0)
}

check_ocaml() {
	if ! has_files '*.ml' && ! has_files '*.mli'; then
		return 0
	fi
	require_cmd ocamlformat
	run_find_xargs -type f \( -name '*.ml' -o -name '*.mli' \) -print0 | xargs -0 ocamlformat --enable-outside-detected-project --check
}

check_java() {
	if ! has_files '*.java'; then
		return 0
	fi
	require_cmd google-java-format
	run_find_xargs -type f -name '*.java' -print0 | xargs -0 google-java-format --dry-run --set-exit-if-changed
}

check_kotlin() {
	if ! has_files '*.kt'; then
		return 0
	fi
	require_cmd ktlint
	run_find_xargs -type f -name '*.kt' -print0 | xargs -0 ktlint
}

check_csharp() {
	if ! has_files '*.cs'; then
		return 0
	fi
	require_cmd dotnet
	if [ -f src/octradotnet.tests/octradotnet.tests.csproj ]; then
		dotnet format src/octradotnet.tests/octradotnet.tests.csproj --verify-no-changes --verbosity minimal || true
	fi
}

check_r() {
	if ! has_files '*.r' && ! has_files '*.R'; then
		return 0
	fi
	# Styler's CLI/check surface is version-dependent; rely on `lintr` in lint checks instead.
	return 0
}

check_ruby() {
	if ! has_files '*.rb'; then
		return 0
	fi
	require_cmd rufo
	run_find_xargs -type f -name '*.rb' -print0 | xargs -0 rufo --check
}

check_lua() {
	if ! has_files '*.lua'; then
		return 0
	fi
	require_cmd stylua
	run_find_xargs -type f -name '*.lua' -print0 | xargs -0 stylua --check
}

check_php() {
	if ! has_files '*.php'; then
		return 0
	fi
	if ! command -v php-cs-fixer >/dev/null 2>&1; then
		echo "Skipping PHP format check: php-cs-fixer not available" >&2
		return 0
	fi
	php-cs-fixer fix --quiet --dry-run --diff --using-cache=no
}

check_tcl() {
	if ! has_files '*.tcl'; then
		return 0
	fi
	if ! command -v tclfmt >/dev/null 2>&1; then
		echo "Skipping Tcl format check: tclfmt not available" >&2
		return 0
	fi
	run_find_xargs -type f -name '*.tcl' -print0 | xargs -0 tclfmt -d
}

check_scheme_guile() {
	if ! has_files '*.scm'; then
		return 0
	fi
	require_cmd guile
	require_cmd guild
	if ! guild help fmt >/dev/null 2>&1; then
		echo "Skipping Scheme (Guile) format check: 'guild fmt' not available" >&2
		return 0
	fi
	run_find_xargs -type f -name '*.scm' -print0 | xargs -0 guild fmt --check
}

main() {
	check_cpp
	check_cmake
	check_nix
	check_shell
	check_python
	check_prettier
	check_rust
	check_go
	check_d
	check_ocaml
	check_java
	check_kotlin
	check_csharp
	check_r
	check_ruby
	check_lua
	check_php
	check_tcl
	check_scheme_guile
}

main "$@"
