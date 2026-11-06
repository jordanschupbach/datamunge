#!/usr/bin/env bash
# Record native toolchain and reproduction-source identities; does not acquire data.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
study_root=$(cd ../../../../.. && pwd)
printf 'Recorded UTC: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
printf 'Execution: CPU; Float32; datamunge threads=8; OPENBLAS_NUM_THREADS=%s\n' "${OPENBLAS_NUM_THREADS:-1}"
printf 'CXX environment: %s\n' "${CXX:-unset}"
printf 'Default c++: '
c++ --version | sed -n '1p'
printf 'Default g++: '
g++ --version | sed -n '1p'
uname -a
printf 'Package version: '
pkg-config --modversion datamunge
study_libdir=$(pkg-config --variable=libdir datamunge)
printf 'Package library directory: %s\n' "$study_libdir"
for study_library in "$study_libdir/libdatamunge.so" "$study_libdir/libdatamunge.dylib" "$study_libdir/datamunge.dll"; do
  if [[ -f "$study_library" ]]; then sha256sum "$study_library"; fi
done
if [[ -n "${1:-}" ]]; then sha256sum "$1"; fi
printf 'Report C++ blocks SHA-256: '
awk '/^```\{cpp/{active=1;next} active && /^```/{active=0;next} active {print}' cifar10.mepml | sha256sum
cd "$study_root"
printf 'Git revision: '
git rev-parse HEAD 2>/dev/null || printf 'unavailable\n'
printf 'Reproduction checkout source fingerprints (including dirty sources):\n'
rg --files -0 include/datamunge/nn src/datamunge/nn include/datamunge/datasets src/datamunge/datasets |
  LC_ALL=C sort -z | xargs -0 sha256sum
sha256sum CMakeLists.txt include/dm/dm.hpp docs/mepml/examples/datasets/cifar10/run.sh
