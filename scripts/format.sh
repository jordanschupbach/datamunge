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

format_cpp() {
	if ! has_files '*.c' && ! has_files '*.h' && ! has_files '*.cc' && ! has_files '*.hh' && ! has_files '*.cpp' && ! has_files '*.hpp' && ! has_files '*.cxx' && ! has_files '*.hxx'; then
		return 0
	fi
	require_cmd clang-format
	run_find_xargs -type f \( \
		-name '*.c' -o -name '*.h' -o -name '*.cc' -o -name '*.hh' -o -name '*.cpp' -o -name '*.hpp' -o -name '*.cxx' -o -name '*.hxx' \
		\) -print0 | xargs -0 clang-format -i
}

format_cmake() {
	if ! has_files 'CMakeLists.txt' && ! has_files '*.cmake'; then
		return 0
	fi
	require_cmd cmake-format
	run_find_xargs -type f \( -name 'CMakeLists.txt' -o -name '*.cmake' \) -print0 | xargs -0 cmake-format -i
}

format_nix() {
	if ! has_files '*.nix'; then
		return 0
	fi
	require_cmd nixfmt
	run_find_xargs -type f -name '*.nix' -print0 | xargs -0 nixfmt
}

format_shell() {
	if ! has_files '*.sh' && [ ! -f ./rename_octra ]; then
		return 0
	fi
	require_cmd shfmt
	if has_files '*.sh'; then
		run_find_xargs -type f -name '*.sh' -print0 | xargs -0 shfmt -w
	fi
	if [ -f ./rename_octra ]; then
		shfmt -w ./rename_octra
	fi
}

format_python() {
	if ! has_files '*.py'; then
		return 0
	fi
	require_cmd ruff
	ruff format .
}

format_js_ts_json_md_yaml() {
	if ! has_files '*.js' && ! has_files '*.ts' && ! has_files '*.tsx' && ! has_files '*.jsx' && ! has_files '*.json' && ! has_files '*.md' && ! has_files '*.yml' && ! has_files '*.yaml'; then
		return 0
	fi
	require_cmd prettier
	prettier --write .
}

format_rust() {
	if [ ! -f ./Cargo.toml ]; then
		return 0
	fi
	require_cmd cargo
	cargo fmt --all
}

format_go() {
	if ! has_files '*.go'; then
		return 0
	fi
	require_cmd gofmt
	run_find_xargs -type f -name '*.go' -print0 | xargs -0 gofmt -w
}

format_d() {
	if ! has_files '*.d'; then
		return 0
	fi
	require_cmd dfmt
	if ! dfmt --help 2>/dev/null | grep -q -- '-i'; then
		echo "Skipping D formatting: dfmt does not support -i on this system" >&2
		return 0
	fi
	run_find_xargs -type f -name '*.d' -print0 | xargs -0 dfmt -i
}

format_ocaml() {
	if ! has_files '*.ml' && ! has_files '*.mli'; then
		return 0
	fi
	require_cmd ocamlformat
	run_find_xargs -type f \( -name '*.ml' -o -name '*.mli' \) -print0 | xargs -0 ocamlformat --enable-outside-detected-project -i
}

format_java() {
	if ! has_files '*.java'; then
		return 0
	fi
	require_cmd google-java-format
	run_find_xargs -type f -name '*.java' -print0 | xargs -0 google-java-format -i
}

format_kotlin() {
	if ! has_files '*.kt'; then
		return 0
	fi
	require_cmd ktlint
	run_find_xargs -type f -name '*.kt' -print0 | xargs -0 ktlint -F
}

format_csharp() {
	if ! has_files '*.cs'; then
		return 0
	fi
	require_cmd dotnet
	if [ -f src/octradotnet.tests/octradotnet.tests.csproj ]; then
		dotnet format src/octradotnet.tests/octradotnet.tests.csproj --no-restore --verbosity minimal || true
	fi
}

format_r() {
	if ! has_files '*.r' && ! has_files '*.R'; then
		return 0
	fi
	require_cmd Rscript
	Rscript -e 'if (!requireNamespace("styler", quietly=TRUE)) quit(status=1); styler::style_dir(".", recursive=TRUE, filetype=c(\".R\", \".r\"))'
}

format_ruby() {
	if ! has_files '*.rb'; then
		return 0
	fi
	require_cmd rufo
	run_find_xargs -type f -name '*.rb' -print0 | xargs -0 rufo
}

format_lua() {
	if ! has_files '*.lua'; then
		return 0
	fi
	require_cmd stylua
	run_find_xargs -type f -name '*.lua' -print0 | xargs -0 stylua
}

format_perl() {
	if ! has_files '*.pl' && ! has_files '*.pm'; then
		return 0
	fi
	require_cmd perltidy
	run_find_xargs -type f \( -name '*.pl' -o -name '*.pm' \) -print0 | xargs -0 perltidy -b -bext=.bak -q
	run_find_xargs -type f -name '*.bak' -print0 | xargs -0 rm -f
}

format_php() {
	if ! has_files '*.php'; then
		return 0
	fi
	if ! command -v php-cs-fixer >/dev/null 2>&1; then
		echo "Skipping PHP formatting: php-cs-fixer not available" >&2
		return 0
	fi
	php-cs-fixer fix --quiet
}

format_tcl() {
	if ! has_files '*.tcl'; then
		return 0
	fi
	if ! command -v tclfmt >/dev/null 2>&1; then
		echo "Skipping Tcl formatting: tclfmt not available" >&2
		return 0
	fi
	run_find_xargs -type f -name '*.tcl' -print0 | xargs -0 tclfmt -w
}

format_scheme_guile() {
	if ! has_files '*.scm'; then
		return 0
	fi
	require_cmd guile
	require_cmd guild
	if ! guild help fmt >/dev/null 2>&1; then
		echo "Skipping Scheme (Guile) formatting: 'guild fmt' not available" >&2
		return 0
	fi
	run_find_xargs -type f -name '*.scm' -print0 | xargs -0 guild fmt -w
}

format_octave() {
	if ! has_files '*.m'; then
		return 0
	fi
	python3 - <<'PY'
import os, pathlib
PRUNE = {'.git','.direnv','.gradle','.cache','.pytest_cache','node_modules','build','dist','target','result'}
def should_prune(path):
    parts = pathlib.Path(path).parts
    return any(p in PRUNE for p in parts)
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if not should_prune(os.path.join(root,d))]
    for f in files:
        if not f.endswith('.m'):
            continue
        p = pathlib.Path(root, f)
        try:
            text = p.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            continue
        new = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
        if new != text:
            p.write_text(new, encoding='utf-8')
PY
}

main() {
	format_cpp
	format_cmake
	format_nix
	format_shell
	format_python
	format_js_ts_json_md_yaml
	format_rust
	format_go
	format_d
	format_ocaml
	format_java
	format_kotlin
	format_csharp
	format_r
	format_ruby
	format_lua
	format_perl
	format_php
	format_tcl
	format_scheme_guile
	format_octave
}

main "$@"
