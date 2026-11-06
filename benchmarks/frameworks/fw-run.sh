#!/usr/bin/env bash
# Run a command inside the FHS env (build/fw-env) with a clean library path.
#
#   benchmarks/frameworks/fw-run.sh torch python squeezenet_torch.py --device cuda
#   benchmarks/frameworks/fw-run.sh tf    python squeezenet_tf.py --threads 8
#
# First argument selects the venv (torch | tf); the rest is the command, with
# `python` resolving to that venv's interpreter.  The outer LD_LIBRARY_PATH /
# PYTHONPATH (from direnv / the datamunge build) are dropped so the wheels'
# bundled libraries are the only ones found; the FHS profile re-adds
# /run/opengl-driver/lib for libcuda.so.1.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$1"; shift
case "$VENV" in
  torch) PY="$ROOT/build/fw-venv-torch/bin/python" ;;
  tf)    PY="$ROOT/build/fw-venv-tf/bin/python" ;;
  *) echo "usage: fw-run.sh torch|tf CMD..." >&2; exit 2 ;;
esac
if [[ "${1:-}" == "python" ]]; then shift; set -- "$PY" "$@"; fi
printf -v CMD '%q ' "$@"
EXTRA=""
if [[ "$VENV" == "tf" ]]; then
  # TensorFlow 2.21 dlopens libcusolver.so.11 etc. by soname and does not
  # preload the pip nvidia-*-cu12 wheels, so put their lib dirs on the path
  # (after /run/opengl-driver/lib, which the FHS profile prepends for libcuda).
  NV=$(ls -d "$ROOT"/build/fw-venv-tf/lib/python3.*/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
  [[ -n "$NV" ]] && EXTRA="export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH}:${NV%:}\"; "
fi
exec env -u LD_LIBRARY_PATH -u PYTHONPATH "$ROOT/build/fw-env/bin/fw-env" -c "cd $(printf %q "$ROOT") && ${EXTRA}${CMD}"
