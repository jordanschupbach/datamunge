#!/usr/bin/env bash
set -euo pipefail

# Regenerates nix/nuget-deps.json for buildDotnetModule (offline NuGet fetches).
# This does a one-time online restore into a temp NUGET_PACKAGES directory, then
# converts that cache into a pinned lockfile via nuget-to-json.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_JSON="$REPO_ROOT/nix/nuget-deps.json"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

export NUGET_PACKAGES="$TMP/nuget"
mkdir -p "$NUGET_PACKAGES"

nix develop --accept-flake-config "$REPO_ROOT"#csharp --command bash -lc "\
  set -euo pipefail; \
  dotnet restore \"$REPO_ROOT/src/octradotnet.tests/octradotnet.tests.csproj\" \
"

nix shell --accept-flake-config nixpkgs#nuget-to-json nixpkgs#dotnet-sdk_10 --command bash -lc "\
  set -euo pipefail; \
  nuget-to-json \"$NUGET_PACKAGES\" > \"$OUT_JSON\" \
"

chmod 0644 "$OUT_JSON"
echo "Wrote $OUT_JSON"
