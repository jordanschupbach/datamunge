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

lint_shell() {
	require_cmd shellcheck
	local -a files=()
	while IFS= read -r -d '' f; do
		case "$f" in
		./nix/update-joctra-gradle-deps.sh | ./nix/update-nuget-deps.sh) continue ;;
		esac
		files+=("$f")
	done < <(run_find_xargs -type f -name '*.sh' -print0)
	if [ -f ./rename_octra ]; then
		files+=("./rename_octra")
	fi
	if [ "${#files[@]}" -gt 0 ]; then
		shellcheck -x "${files[@]}"
	fi
}

lint_nix() {
	if ! has_files '*.nix'; then
		return 0
	fi
	require_cmd statix
	require_cmd deadnix
	# Statix is valuable but noisy on older template patterns; treat it as advisory.
	statix check . || true
	deadnix . || true
}

lint_cpp() {
	# Keep this focused on the canonical core (exclude generated SWIG wrappers).
	if [ ! -d include ] && [ ! -d src/octra ]; then
		return 0
	fi
	require_cmd cppcheck
	cppcheck \
		--enable=warning,performance,portability \
		--error-exitcode=1 \
		--inline-suppr \
		--std=c++20 \
		-I include \
		src/octra include tests/cpp

	if command -v clang-tidy >/dev/null 2>&1; then
		require_cmd cmake
		rm -rf build/tidy
		cmake -S . -B build/tidy -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON >/dev/null
		# Run clang-tidy only on core sources.
		clang-tidy -p build/tidy src/octra/octra.cpp src/octra/octra_c.cpp
	fi
}

lint_python() {
	if ! has_files '*.py'; then
		return 0
	fi
	require_cmd ruff
	ruff check .
}

lint_js_ts() {
	if ! has_files '*.js' && ! has_files '*.ts' && ! has_files '*.tsx' && ! has_files '*.jsx'; then
		return 0
	fi
	if command -v eslint >/dev/null 2>&1 && { [ -f eslint.config.cjs ] || [ -f eslint.config.js ]; }; then
		eslint .
	fi
	if command -v tsc >/dev/null 2>&1 && [ -f tsconfig.json ]; then
		tsc -p tsconfig.json --noEmit
	fi
}

lint_markdown_yaml_actions() {
	if command -v markdownlint-cli2 >/dev/null 2>&1; then
		markdownlint-cli2 "**/*.md" "#.direnv" "#.git" "#node_modules" "#build" "#dist" "#target" "#result"
	fi
	if command -v yamllint >/dev/null 2>&1; then
		yamllint -c .yamllint.yml .
	fi
	if command -v actionlint >/dev/null 2>&1; then
		if [ -d .github/workflows ]; then
			actionlint .github/workflows/*.yml
		else
			actionlint
		fi
	fi
}

lint_go() {
	if [ ! -f src/gooctra/go.mod ]; then
		return 0
	fi
	require_cmd go
	# Go bindings rely on CGO + the native lib; only run build-based checks if `octra`
	# is available via pkg-config (e.g. in the Nix `.#go`/`.#default` shells).
	if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists octra 2>/dev/null; then
		(
			cd src/gooctra
			cache="${TMPDIR:-/tmp}/go-build"
			mkdir -p "$cache"
			HOME="${TMPDIR:-/tmp}" GOCACHE="$cache" go test ./...
		)
		if command -v golangci-lint >/dev/null 2>&1; then
			(
				cd src/gooctra
				cache="${TMPDIR:-/tmp}/go-build"
				mkdir -p "$cache"
				HOME="${TMPDIR:-/tmp}" GOCACHE="$cache" golangci-lint run
			)
		fi
	else
		echo "Skipping Go build checks: 'octra' not available via pkg-config" >&2
	fi
}

lint_rust() {
	if [ ! -f Cargo.toml ]; then
		return 0
	fi
	require_cmd cargo
	# Rust bindings need the native library via pkg-config; only run clippy when
	# `octra` is available (e.g. in the Nix `.#default` shell after a build).
	if ! (command -v pkg-config >/dev/null 2>&1 && pkg-config --exists octra 2>/dev/null); then
		echo "Skipping Rust clippy: 'octra' not available via pkg-config" >&2
		return 0
	fi
	if ! (cargo fetch --locked --offline >/dev/null 2>&1); then
		echo "Skipping Rust clippy: dependencies not available offline" >&2
		return 0
	fi
	if command -v cargo-clippy >/dev/null 2>&1 || cargo clippy --version >/dev/null 2>&1; then
		cargo clippy --workspace --all-targets --locked --offline
	fi
}

lint_d() {
	if ! has_files '*.d'; then
		return 0
	fi
	if command -v dscanner >/dev/null 2>&1; then
		# Dscanner can be sensitive to compiler/version mismatches; treat as advisory.
		run_find_xargs -type f -name '*.d' -print0 | xargs -0 dscanner --styleCheck || true
	fi
}

lint_lua() {
	if ! has_files '*.lua'; then
		return 0
	fi
	if command -v luacheck >/dev/null 2>&1; then
		run_find_xargs -type f -name '*.lua' -print0 | xargs -0 luacheck
	fi
}

lint_ruby() {
	if ! has_files '*.rb'; then
		return 0
	fi
	if command -v rubocop >/dev/null 2>&1; then
		HOME="${TMPDIR:-/tmp}" rubocop || true
	fi
}

lint_perl() {
	if ! has_files '*.pl' && ! has_files '*.pm'; then
		return 0
	fi
	if command -v perlcritic >/dev/null 2>&1; then
		run_find_xargs -type f \( -name '*.pl' -o -name '*.pm' \) -print0 | xargs -0 perlcritic || true
	fi
}

lint_php() {
	if ! has_files '*.php'; then
		return 0
	fi
	require_cmd php
	run_find_xargs -type f -name '*.php' -print0 | xargs -0 -I{} php -l {}
}

lint_r() {
	if ! has_files '*.r' && ! has_files '*.R'; then
		return 0
	fi
	if command -v R >/dev/null 2>&1; then
		R -q -e 'if (!requireNamespace("lintr", quietly=TRUE)) quit(status=1); lintr::lint_package(".")'
	fi
}

main() {
	lint_shell
	lint_nix
	lint_cpp
	lint_python
	lint_js_ts
	lint_markdown_yaml_actions
	lint_go
	lint_rust
	lint_d
	lint_lua
	lint_ruby
	lint_perl
	lint_php
	lint_r
}

main "$@"
