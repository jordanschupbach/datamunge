#!/usr/bin/env bash
set -euo pipefail

# Generates nix/joctra-gradle-deps.json by recording all HTTP fetches performed
# by Gradle (via mitm-cache), then compressing the lockfile into the format
# consumed by nixpkgs' gradle fetchDeps.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_JSON="$REPO_ROOT/nix/joctra-gradle-deps.json"
INIT_DEPS="$REPO_ROOT/nix/joctra-init-deps.gradle"

if [[ ! -f "$INIT_DEPS" ]]; then
	echo "Missing $INIT_DEPS" >&2
	exit 1
fi

# Discover nixpkgs source to get compress-deps-json.py
NIXPKGS_SRC="$(nix eval --impure --raw --expr 'toString <nixpkgs>')"
COMPRESS_PY="$NIXPKGS_SRC/pkgs/development/tools/build-managers/gradle/compress-deps-json.py"
if [[ ! -f "$COMPRESS_PY" ]]; then
	echo "Missing $COMPRESS_PY" >&2
	exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

CERT_DIR="$TMP/mitm-cert"
mkdir -p "$CERT_DIR"

# Start mitm-cache recorder
pushd "$CERT_DIR" >/dev/null
openssl genrsa -out ca.key 2048 >/dev/null 2>&1
openssl req -x509 -new -nodes -key ca.key -sha256 -days 1 -out ca.cer -subj "/C=AL/ST=a/L=a/O=a/OU=a/CN=example.org" >/dev/null 2>&1
MITM_CACHE_DIR="$CERT_DIR"

MITM_CACHE_HOST=127.0.0.1

MITM_PID=
for attempt in {1..5}; do
	MITM_CACHE_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
	MITM_CACHE_ADDRESS="$MITM_CACHE_HOST:$MITM_CACHE_PORT"

	mitm-cache -l"$MITM_CACHE_ADDRESS" record \
		--reject '\.(md5|sha(1|256|512:?):?)$' \
		--forget-redirects-from '.*' \
		--record-text '/maven-metadata\.xml$' \
		>/dev/null 2>/dev/null &
	MITM_PID=$!

	sleep 0.1
	if kill -0 "$MITM_PID" 2>/dev/null; then
		break
	fi
	MITM_PID=
done

if [[ -z "${MITM_PID:-}" ]]; then
	echo "mitm-cache failed to start" >&2
	exit 1
fi

# Wait for server (mitm-cache is an HTTP proxy; a plain curl to host:port is enough)
for i in {0..40}; do
	kill -0 "$MITM_PID" 2>/dev/null || {
		echo "mitm-cache exited early" >&2
		exit 1
	}
	if curl -s -o /dev/null "$MITM_CACHE_ADDRESS" >/dev/null 2>&1; then
		break
	fi
	sleep 0.25
	if [[ $i -eq 40 ]]; then
		echo "mitm-cache failed to start" >&2
		exit 1
	fi
done

# These are consumed by nixpkgs' gradle wrapper to configure proxy + JVM trust store flags.
export MITM_CACHE_CERT_DIR="$CERT_DIR"
export MITM_CACHE_CA="$CERT_DIR/ca.cer"
export MITM_CACHE_HOST
export MITM_CACHE_PORT
export MITM_CACHE_ADDRESS

export GRADLE_USER_HOME="$TMP/gradle-home"
mkdir -p "$GRADLE_USER_HOME"

# Run inside the repo so Gradle sees settings.gradle.kts
cd "$REPO_ROOT"

# Ensure the native lib exists so tests that try to load it don't fail while capturing.
# (Dependency capture should not require running tests, but some builds resolve via buildSrc).
nix develop --accept-flake-config .#java --command bash -lc "\
  set -euo pipefail; \
  KS=\"$CERT_DIR/keystore\"; \
  KSPWD=\"\$(python -c 'import secrets; print(secrets.token_hex(16))')\"; \
  echo y | keytool -importcert -file \"$CERT_DIR/ca.cer\" -alias octra -keystore \"\$KS\" -storepass \"\$KSPWD\" >/dev/null; \
  cmake -S src/joctra-octra -B src/joctra-octra/build/cmake -DCMAKE_BUILD_TYPE=Release; \
  cmake --build src/joctra-octra/build/cmake -j \"\${NIX_BUILD_CORES:-4}\"; \
  export LD_LIBRARY_PATH=\"$REPO_ROOT/src/joctra-octra/build/cmake:\${LD_LIBRARY_PATH:-}\"; \
  gradle --no-daemon --console plain --no-configuration-cache \
    -Dhttp.proxyHost=\"$MITM_CACHE_HOST\" -Dhttp.proxyPort=\"$MITM_CACHE_PORT\" \
    -Dhttps.proxyHost=\"$MITM_CACHE_HOST\" -Dhttps.proxyPort=\"$MITM_CACHE_PORT\" \
    -Djavax.net.ssl.trustStore=\"\$KS\" -Djavax.net.ssl.trustStorePassword=\"\$KSPWD\" \
    test\
"

# Stop mitm-cache and wait for out.json
kill -s SIGINT "$MITM_PID" 2>/dev/null || true
wait "$MITM_PID" 2>/dev/null || true

RAW_JSON="$MITM_CACHE_DIR/out.json"
if [[ ! -f "$RAW_JSON" ]]; then
	echo "mitm-cache did not produce $RAW_JSON" >&2
	exit 1
fi

python "$COMPRESS_PY" "$RAW_JSON" "$OUT_JSON"
echo "Wrote $OUT_JSON"
