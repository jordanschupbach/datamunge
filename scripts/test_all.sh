#!/usr/bin/env bash

set -uo pipefail

languages=(
	"cpp:test-cpp"
	"python:test-python"
	"r:test-r"
	"javascript:test-javascript"
	"csharp:test-csharp"
	"java:test-java"
	"rust:test-rust"
	"php:test-php"
	"lua:test-lua"
	"perl:test-perl"
	"tcl:test-tcl"
	"ruby:test-ruby"
	"guile:test-guile"
	"octave:test-octave"
	"d:test-d"
	"go:test-go"
	"ocaml:test-ocaml"
)

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/octra-test-all.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

failures=0

declare -a summary_lines=()
declare -a failed_languages=()
declare -a failed_targets=()
declare -a failed_logs=()

extract_failure_summary() {
	local logfile="$1"
	local summary

	summary="$(
		rg -i -m 12 \
			-e '(^|[^[:alpha:]])fail(ed|ure|ures)?([^[:alpha:]]|$)' \
			-e '(^|[^[:alpha:]])error([^[:alpha:]]|$)' \
			-e 'exception' \
			-e 'traceback' \
			-e 'assert' \
			-e '^not ok' \
			-e '^\s*not ok' \
			-e 'segmentation fault' \
			-e 'panic' \
			"$logfile" 2>/dev/null || true
	)"

	if [[ -z "$summary" ]]; then
		summary="$(tail -n 20 "$logfile" | sed '/^[[:space:]]*$/d')"
	fi

	printf '%s\n' "$summary"
}

printf 'Running %d test targets\n' "${#languages[@]}"

for entry in "${languages[@]}"; do
	language="${entry%%:*}"
	target="${entry#*:}"
	logfile="$tmpdir/${language}.log"

	printf '\n[%s] just %s\n' "$language" "$target"

	just "$target" >"$logfile" 2>&1
	rc=$?

	if ((rc == 0)); then
		summary_lines+=("PASS  ${language} (${target})")
		printf '[%s] PASS\n' "$language"
		continue
	fi

	failures=$((failures + 1))
	summary_lines+=("FAIL  ${language} (${target}) [exit ${rc}]")
	failed_languages+=("$language")
	failed_targets+=("$target")
	failed_logs+=("$logfile")
	printf '[%s] FAIL (exit %d)\n' "$language" "$rc"
done

printf '\nSummary\n'
for line in "${summary_lines[@]}"; do
	printf '%s\n' "$line"
done

if ((failures == 0)); then
	printf '\nAll language tests passed.\n'
	exit 0
fi

printf '\nFailure details\n'
for i in "${!failed_languages[@]}"; do
	language="${failed_languages[$i]}"
	target="${failed_targets[$i]}"
	logfile="${failed_logs[$i]}"

	printf '\n[%s] just %s\n' "$language" "$target"
	extract_failure_summary "$logfile"
done

printf '\n%d test target(s) failed.\n' "$failures"
exit 1
