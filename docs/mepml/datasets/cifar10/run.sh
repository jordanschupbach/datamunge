#!/usr/bin/env bash
# Recompute the dataset report with the repository's matching native C++ package.
# --train also replaces the retained model with a newly tuned, converged run.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p build/tmp
export TMPDIR="$PWD/build/tmp"
case "${1:-}" in
  '') export DATAMUNGE_CIFAR10_RETRAIN=0 ;;
  --train) export DATAMUNGE_CIFAR10_RETRAIN=1 ;;
  *) printf 'Usage: %s [--train]\n' "$0" >&2; exit 2 ;;
esac
pkg-config --exists datamunge
command -v mep-mepml >/dev/null
if [[ "$DATAMUNGE_CIFAR10_RETRAIN" == 0 ]]; then
  sha256sum -c artifacts/checksums.sha256 > build/artifact-check.log
fi
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
mep-mepml run --timeout "${DATAMUNGE_CIFAR10_TIMEOUT:-10800}" --report build/execution.json cifar10.mepml
if [[ "$DATAMUNGE_CIFAR10_RETRAIN" == 1 ]]; then
  ./record-runtime.sh > artifacts/runtime_manifest.txt
fi
sha256sum artifacts/*.npz artifacts/*.tsv artifacts/runtime_manifest.txt > artifacts/checksums.sha256
