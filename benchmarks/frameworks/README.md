# Framework baselines: SqueezeNet CIFAR-10 training throughput

External PyTorch and TensorFlow/Keras reference points for the in-house
`datamunge::nn` SqueezeNet training workload in
[`docs/mepml/examples/nn/1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/squeezenet.mepml`](../../docs/mepml/examples/nn/1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/squeezenet.mepml)
(model: [`include/datamunge/nn/squeezenet_models.hpp`](../../include/datamunge/nn/squeezenet_models.hpp)).

Both scripts reproduce the same step: SqueezeNet v1.0 (CIFAR stem, 96-channel
3x3/1 SAME conv1, Caffe ceil-mode 3x3/2 max pools 32->16->8->4, Fire2..Fire9,
dropout keep 0.5, conv10 1x1 + ReLU, global average pool), batch 32, float32,
loss = mean sparse softmax cross-entropy + `5e-5 * sum(kernel^2)` over every
parameter with `ndim >= 2` (kept inside the loss, not as weight decay), Adam
lr 5e-4 (constant), first 8000 rows of `data_batch_1.bin` (250 steps/epoch),
and on-the-fly augmentation (zero-pad 4 on the [0,1] image, random 32x32 crop,
random horizontal flip, then `2x-1`) done with batched tensor ops inside the
step.  `--no-augment` feeds pre-scaled batches straight from the dataset so
the pure model-step cost is isolated.  Parameter counts are asserted
(729034 for v1.0, 727626 for v1.1) to match the C++ model.

## Files

| file | purpose |
|---|---|
| `fhs-env.nix` | `buildFHSEnv` expression that lets pip wheels run on NixOS |
| `fw-run.sh` | run a command inside the FHS env with the torch or tf venv |
| `squeezenet_torch.py` | PyTorch benchmark (`--device cpu\|cuda --threads N --[no-]augment --steps --seconds --version 10\|11 --memory-format --tf32 --cudnn-benchmark --json`) |
| `squeezenet_tf.py` | TensorFlow/Keras benchmark (`--device cpu\|cuda --threads N --inter-threads --mode graph\|eager\|xla --[no-]augment --steps --seconds --version --json`) |
| `run_all.sh` | the full matrix, one process per configuration, then `summarize.py` |
| `summarize.py` | `benchmarks/results/frameworks/*.json` -> `summary.md` |

Results live in [`benchmarks/results/frameworks/`](../results/frameworks/)
(`<framework>_<device>_<threads>[_<variant>].json` plus `summary.md`).

## Recreating the environment (NixOS, no nix-ld)

The stock `python3` here is a Nix store interpreter without pip, and the
manylinux wheels need an FHS layout (`/usr/lib/libstdc++.so.6`, `/lib/libz.so.1`)
plus the host driver's `libcuda.so.1`.  `buildFHSEnv` provides both: it
bind-mounts `/run/opengl-driver` and the `profile` in `fhs-env.nix` puts
`/run/opengl-driver/lib` on `LD_LIBRARY_PATH`.  Everything large goes under
`build/` (gitignored, big filesystem); nothing is written to `/nix` or `/tmp`
beyond the tiny env derivation.

```sh
cd /mnt/projects/datamunge

# 1. FHS env (python 3.13, zlib, libstdc++, libGL, glib, ...) -> build/fw-env
nix build --impure --expr 'import ./benchmarks/frameworks/fhs-env.nix {}' -o build/fw-env

# 2. venvs inside the env (two: torch's and TF's pinned nvidia-* wheels differ)
FW=build/fw-env/bin/fw-env
env -u LD_LIBRARY_PATH -u PYTHONPATH $FW -c '
  python3 -m venv build/fw-venv-torch
  build/fw-venv-torch/bin/python -m pip install --upgrade pip
  build/fw-venv-torch/bin/python -m pip install numpy torch --index-url https://download.pytorch.org/whl/cu128'
env -u LD_LIBRARY_PATH -u PYTHONPATH $FW -c '
  python3 -m venv build/fw-venv-tf
  build/fw-venv-tf/bin/python -m pip install --upgrade pip
  build/fw-venv-tf/bin/python -m pip install numpy "tensorflow[and-cuda]"'
```

`fhs-env.nix` also exports `PIP_CACHE_DIR=build/pip-cache` and
`TMPDIR=build/tmp` inside the env so pip never fills the root filesystem.  The
outer shell has `OMP_NUM_THREADS=1` and an `LD_LIBRARY_PATH` pointing into the
datamunge build tree; `fw-run.sh` unsets the latter, and the scripts set the
thread environment themselves from `--threads`.

## Running

```sh
# single configurations
benchmarks/frameworks/fw-run.sh torch python benchmarks/frameworks/squeezenet_torch.py \
    --device cpu --threads 8 --json benchmarks/results/frameworks/pytorch_cpu_8.json
benchmarks/frameworks/fw-run.sh torch python benchmarks/frameworks/squeezenet_torch.py --device cuda
benchmarks/frameworks/fw-run.sh tf python benchmarks/frameworks/squeezenet_tf.py --device cpu --threads 8
benchmarks/frameworks/fw-run.sh tf python benchmarks/frameworks/squeezenet_tf.py --device cuda

# the whole matrix (THREADS="1 4 8 16 32" by default; parts: torch-cpu torch-cuda tf-cpu tf-cuda)
benchmarks/frameworks/run_all.sh
THREADS="4 32" benchmarks/frameworks/run_all.sh torch-cpu
```

Each run does 10 untimed warmup steps (TF: includes `tf.function` tracing),
then timed steps until both `--steps` (100) and `--seconds` (20) are
satisfied, reads the clock after `torch.cuda.synchronize()` /
`tf.test.experimental.sync_devices()` on GPU, and reports steps/s, images/s,
seconds per 250-step epoch and the final-step NLL (chance = ln 10 = 2.303;
a value pinned exactly there means the ReLU class-map head died).

## Gotchas found while setting this up

* **`libcusolver.so.11` / TF-GPU.** TensorFlow 2.21 dlopens the CUDA libraries
  by soname and does not preload the `nvidia-*-cu12` pip wheels, so with a
  bare `LD_LIBRARY_PATH` it finds libcuda/cudnn/cublas/cufft/cusparse but not
  cusolver, logs only "Cannot dlopen some GPU libraries" and silently falls
  back to CPU.  `fw-run.sh tf` appends every `site-packages/nvidia/*/lib`
  directory to `LD_LIBRARY_PATH`; `TF_CPP_VMODULE=dso_loader=3` shows which
  library is missing.
* **sm_120 (RTX 5080, Blackwell).** The torch `cu128` wheel ships native
  `sm_120` SASS (`torch.cuda.get_arch_list()` lists it).  TF 2.21 is built
  against CUDA 12.5.1 without `sm_120` binaries and JIT-compiles every kernel
  from PTX on first use (warning: "could take 30 minutes or longer"; in
  practice the first warmup took ~20 s and the driver caches the result in
  `~/.nv/ComputeCache`).  Use `--warmup 30` or more for TF-GPU so the JIT
  stays out of the timed window.
* **`$SECONDS` is a bash builtin.**  Do not use it as an environment
  override name; `run_all.sh` uses `BUDGET_SECONDS`.
* **`OMP_NUM_THREADS=1` in the outer shell** makes `nproc` report 1 and torch
  default to one thread.  Both scripts set `OMP_NUM_THREADS`/`MKL_NUM_THREADS`
  from `--threads` before importing the framework; the JSON records
  `cpu_utilization` (process CPU time / wall time over the timed loop) so the
  effective thread count can be checked.
* **Dead ReLU class-map head.**  With a constant lr of 5e-4 from step 0 the
  conv10+ReLU head sometimes collapses (NLL pinned at ln 10 = 2.303, zero
  gradient); it depends on seed/kernel numerics.  The C++ report's 100-step
  linear warmup avoids it: `--schedule warmup_cosine` reproduces that
  schedule and reaches NLL ~1.4 after 2500 steps (10 epochs) in both
  frameworks, matching the C++ v1.0 training curve.  Throughput is identical
  either way.
* **TF CPU runs print `failed call to cuInit: CUDA_ERROR_NO_DEVICE`** because
  the script hides the GPU with `CUDA_VISIBLE_DEVICES=-1`; harmless.
* **TF-GPU grappler message** `layout failed: INVALID_ARGUMENT: Size of values 0
  does not match size of permutation 4 ... dropout/SelectV2` is a non-fatal
  layout-optimizer fallback.

## Results (2026-10-30, Ryzen 9 9955HX 16C/32T, RTX 5080 Laptop, driver 580.142)

Full table with every variant: [`benchmarks/results/frameworks/summary.md`](../results/frameworks/summary.md).
Versions: torch 2.11.0+cu128 (CUDA 12.8 runtime, cuDNN 9.19, native `sm_120`
SASS), TensorFlow 2.21.0 / Keras 3.15.1 (CUDA 12.5.1 build, cuDNN 9.27 wheel,
PTX-JIT on `sm_120`, oneDNN on), numpy 2.5.4, Python 3.13.12.

**Caveat:** all CPU rows were measured (14:38–14:59 local) while another
session was rebuilding the library and running 16-thread benchmarks on the
same machine, so they are contaminated lower bounds (the 32-thread collapse
is at least partly contention); redo them on a quiet machine with the
commands below.  The GPU rows (14:51–14:56) ran with the GPU idle and stand.

| configuration | steps/s | img/s | s/epoch | final NLL | note |
|---|---:|---:|---:|---:|---|
| C++ `datamunge::nn` v1.0, 4 threads (mepml report, 286 s / 10 epochs) | 8.7 | 280 | 28.6 | 1.38 (train, 10 ep) | published reference, not re-measured here |
| PyTorch CPU 1 thread, aug on | 10.8 | 345 | 23.2 | 2.303 | contaminated |
| PyTorch CPU 4 threads, aug on | 26.5 | 849 | 9.4 | 2.303 | contaminated |
| PyTorch CPU 8 threads, aug on | 31.6 | 1011 | 7.9 | 2.303 | contaminated; repeat 29.9 |
| PyTorch CPU 16 threads, aug on | 28.0 | 896 | 8.9 | 2.303 | contaminated |
| PyTorch CPU 32 threads, aug on | 9.7 | 312 | 25.7 | 2.303 | contaminated, SMT oversubscription |
| PyTorch CPU 8 threads, aug off | 30.9 | 990 | 8.1 | 2.199 | contaminated |
| PyTorch CPU 16 threads, channels_last, aug on | 32.1 | 1027 | 7.8 | 2.303 | contaminated |
| TensorFlow CPU 1 thread (inter-op 2), aug on | 25.1 | 802 | 10.0 | 2.309 | contaminated; uses 1.5 cores |
| TensorFlow CPU 1 thread (inter-op 1), aug on | 15.5 | 495 | 16.2 | 2.116 | contaminated; strict single core |
| TensorFlow CPU 4 threads (inter-op 2), aug on | 42.8 | 1368 | 5.8 | 2.346 | contaminated |
| TensorFlow CPU 8 threads (inter-op 2), aug on | 47.6 | 1522 | 5.3 | 2.303 | contaminated; repeat 45.9; 59.0 in an earlier quiet 3 s smoke run |
| TensorFlow CPU 16 threads (inter-op 2), aug on | 43.8 | 1401 | 5.7 | 1.997 | contaminated |
| TensorFlow CPU 32 threads (inter-op 2), aug on | 26.5 | 847 | 9.4 | 2.299 | contaminated |
| TensorFlow CPU 8 threads, eager | 9.6 | 308 | 25.9 | 2.294 | contaminated |
| TensorFlow CPU 8 threads, XLA | 16.5 | 527 | 15.2 | 2.304 | contaminated |
| PyTorch CUDA default (TF32 conv allowed), aug on | 224–252 | 7164–8079 | 1.0–1.1 | 1.85–2.06 | two runs |
| PyTorch CUDA, aug off | 280 | 8948 | 0.9 | 0.61 | |
| PyTorch CUDA, strict fp32 (TF32 off), aug on | 247 | 7899 | 1.0 | 1.29 | |
| PyTorch CUDA, cudnn.benchmark, aug on | 283–295 | 9052–9455 | 0.8–0.9 | 1.47–1.77 | two runs |
| PyTorch CUDA, cudnn.benchmark, aug off | 298–303 | 9551–9694 | 0.8 | 0.40–0.59 | two runs |
| PyTorch CUDA, channels_last, aug on | 265–275 | 8465–8808 | 0.9 | 2.303 | dead head both runs |
| TensorFlow CUDA graph, aug on | 178–183 | 5695–5844 | 1.4 | 1.65–2.30 | PTX JIT, two runs |
| TensorFlow CUDA graph, aug off | 178–184 | 5703–5893 | 1.4 | 2.303 | two runs |
| TensorFlow CUDA XLA, aug on | 366–379 | 11719–12129 | 0.7 | 1.07–2.30 | two runs |

GPU runs are launch-bound at batch 32 (the Python process sits at 0.99–1.17
cores; GPU utilization is low), so they measure framework overhead more than
the RTX 5080.  Augmentation costs ~10 % on torch-CUDA and nothing measurable
on CPU.

### Training-parity check against the C++ driver

With the C++ report's schedule (`--schedule warmup_cosine`: linear warmup over
100 steps to 5e-4, cosine to 0 at 2500 steps) and the same loss, both
frameworks reproduce the C++ v1.0 curve: final-step NLL after 2500 steps
(10 epochs) is 1.46/1.38 in two torch-CUDA runs (`pytorch_cuda_0_warmup_cosine_10ep.json`,
mean of the last 50 steps 1.36), 1.73 after 600 steps on torch-CPU
(`pytorch_cpu_8_warmup_cosine.json`), and 1.22/1.61 on TF-CUDA
(`tensorflow_cuda_0_warmup_cosine_10ep.json`); the C++ report's selected
train NLL is 1.38.  Constant lr 5e-4 from step 0 (the throughput
configuration) trains in most runs but can kill the ReLU class-map head
(NLL pinned at 2.303); lr 1e-4 constant always trains (NLL 1.65 after 1500
steps, `pytorch_cuda_0_lr1e-4.json`).  Throughput is unaffected by the
schedule.

### Re-running the CPU cases on a quiet machine

```sh
cd /mnt/projects/datamunge
# everything CPU (15 torch + 12 tf + 7 extra runs, ~35 s each):
THREADS="1 4 8 16 32" benchmarks/frameworks/run_all.sh torch-cpu tf-cpu extra

# or individual cases (T in 1 4 8 16 32); each writes the JSON named here:
R=benchmarks/results/frameworks; F=benchmarks/frameworks
$F/fw-run.sh torch python $F/squeezenet_torch.py --version 10 --steps 100 --seconds 20 --device cpu --threads $T --json $R/pytorch_cpu_$T.json
$F/fw-run.sh torch python $F/squeezenet_torch.py --version 10 --steps 100 --seconds 20 --device cpu --threads $T --no-augment --json $R/pytorch_cpu_${T}_noaug.json
$F/fw-run.sh torch python $F/squeezenet_torch.py --version 10 --steps 100 --seconds 20 --device cpu --threads $T --memory-format channels_last --json $R/pytorch_cpu_${T}_channels_last.json
$F/fw-run.sh tf python $F/squeezenet_tf.py --version 10 --steps 100 --seconds 20 --device cpu --threads $T --json $R/tensorflow_cpu_$T.json
$F/fw-run.sh tf python $F/squeezenet_tf.py --version 10 --steps 100 --seconds 20 --device cpu --threads $T --no-augment --json $R/tensorflow_cpu_${T}_noaug.json
$F/fw-run.sh tf python $F/squeezenet_tf.py --version 10 --steps 100 --seconds 20 --device cpu --threads $T --inter-threads 1 --json $R/tensorflow_cpu_${T}_inter1.json
$F/fw-run.sh tf python $F/squeezenet_tf.py --version 10 --steps 100 --seconds 20 --device cpu --threads 8 --mode eager --json $R/tensorflow_cpu_8_eager.json
$F/fw-run.sh tf python $F/squeezenet_tf.py --version 10 --steps 100 --seconds 20 --device cpu --threads 8 --mode xla --json $R/tensorflow_cpu_8_xla.json
$F/fw-run.sh torch python $F/squeezenet_torch.py --version 10 --device cpu --threads 8 --schedule warmup_cosine --steps 600 --seconds 1 --json $R/pytorch_cpu_8_warmup_cosine.json
$F/fw-run.sh tf python $F/squeezenet_tf.py --version 10 --device cpu --threads 8 --schedule warmup_cosine --steps 600 --seconds 1 --json $R/tensorflow_cpu_8_warmup_cosine.json
python3 $F/summarize.py $R > $R/summary.md
```
