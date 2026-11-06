#!/usr/bin/env bash
# Run the full PyTorch / TensorFlow SqueezeNet baseline matrix, one process per
# configuration, writing benchmarks/results/frameworks/<framework>_<device>_<threads>[_<variant>].json
# and regenerating summary.md.  Each run: 10 warmup steps, then >=100 timed
# steps AND >=20 s.  Override the thread list with THREADS="1 4", the timing
# with STEPS/BUDGET_SECONDS, and pick a subset with `run_all.sh torch-cpu torch-cuda tf-cpu tf-cuda`.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FW="$ROOT/benchmarks/frameworks"
OUT="$ROOT/benchmarks/results/frameworks"
RUN="$FW/fw-run.sh"
THREADS="${THREADS:-1 4 8 16 32}"
STEPS="${STEPS:-100}"
SECONDS_BUDGET="${BUDGET_SECONDS:-20}"  # not $SECONDS: that is bash's shell-uptime builtin
VERSION="${VERSION:-10}"
PARTS="${*:-torch-cpu torch-cuda tf-cpu tf-cuda extra}"
mkdir -p "$OUT"

torch() { "$RUN" torch python "$FW/squeezenet_torch.py" --version "$VERSION" --steps "$STEPS" --seconds "$SECONDS_BUDGET" "$@"; }
tf()    { "$RUN" tf    python "$FW/squeezenet_tf.py"    --version "$VERSION" --steps "$STEPS" --seconds "$SECONDS_BUDGET" "$@"; }

for part in $PARTS; do
  case "$part" in
    torch-cpu)
      for t in $THREADS; do
        torch --device cpu --threads "$t" --json "$OUT/pytorch_cpu_${t}.json"
        torch --device cpu --threads "$t" --no-augment --json "$OUT/pytorch_cpu_${t}_noaug.json"
        torch --device cpu --threads "$t" --memory-format channels_last --json "$OUT/pytorch_cpu_${t}_channels_last.json"
      done ;;
    torch-cuda)
      torch --device cuda --json "$OUT/pytorch_cuda_0.json"
      torch --device cuda --no-augment --json "$OUT/pytorch_cuda_0_noaug.json"
      torch --device cuda --tf32 off --json "$OUT/pytorch_cuda_0_tf32off.json"
      torch --device cuda --cudnn-benchmark --json "$OUT/pytorch_cuda_0_cudnnbench.json"
      torch --device cuda --cudnn-benchmark --no-augment --json "$OUT/pytorch_cuda_0_cudnnbench_noaug.json"
      torch --device cuda --memory-format channels_last --json "$OUT/pytorch_cuda_0_channels_last.json"
      # training sanity check with the C++ report's warmup-cosine schedule, 10 epochs = 2500 steps
      torch --device cuda --schedule warmup_cosine --steps 2490 --seconds 1 --json "$OUT/pytorch_cuda_0_warmup_cosine_10ep.json"
      torch --device cuda --lr 1e-4 --steps 1500 --seconds 1 --json "$OUT/pytorch_cuda_0_lr1e-4.json" ;;
    tf-cpu)
      for t in $THREADS; do
        tf --device cpu --threads "$t" --json "$OUT/tensorflow_cpu_${t}.json"
        tf --device cpu --threads "$t" --no-augment --json "$OUT/tensorflow_cpu_${t}_noaug.json"
      done
      tf --device cpu --threads 8 --mode eager --json "$OUT/tensorflow_cpu_8_eager.json"
      tf --device cpu --threads 8 --mode xla --json "$OUT/tensorflow_cpu_8_xla.json" ;;
    extra)
      # strict single-pool TF runs (inter-op=1, so "1 thread" really is one core)
      tf --device cpu --threads 1 --inter-threads 1 --json "$OUT/tensorflow_cpu_1_inter1.json"
      tf --device cpu --threads 4 --inter-threads 1 --json "$OUT/tensorflow_cpu_4_inter1.json"
      tf --device cpu --threads 8 --inter-threads 1 --json "$OUT/tensorflow_cpu_8_inter1.json"
      # repeatability (laptop CPU: clocks/thermals drift between back-to-back runs)
      torch --device cpu --threads 8 --json "$OUT/pytorch_cpu_8_rep2.json"
      tf --device cpu --threads 8 --json "$OUT/tensorflow_cpu_8_rep2.json"
      # CPU training sanity with the C++ warmup-cosine schedule (600 steps, untimed budget)
      torch --device cpu --threads 8 --schedule warmup_cosine --steps 600 --seconds 1 --json "$OUT/pytorch_cpu_8_warmup_cosine.json"
      tf --device cpu --threads 8 --schedule warmup_cosine --steps 600 --seconds 1 --json "$OUT/tensorflow_cpu_8_warmup_cosine.json" ;;
    tf-cuda)
      tf --device cuda --warmup 30 --json "$OUT/tensorflow_cuda_0.json"
      tf --device cuda --warmup 30 --no-augment --json "$OUT/tensorflow_cuda_0_noaug.json"
      tf --device cuda --warmup 30 --mode xla --json "$OUT/tensorflow_cuda_0_xla.json"
      tf --device cuda --warmup 30 --schedule warmup_cosine --steps 2490 --seconds 1 --json "$OUT/tensorflow_cuda_0_warmup_cosine_10ep.json" ;;
    *) echo "unknown part $part" >&2 ;;
  esac
done
python3 "$FW/summarize.py" "$OUT" > "$OUT/summary.md"
echo "wrote $OUT/summary.md"
