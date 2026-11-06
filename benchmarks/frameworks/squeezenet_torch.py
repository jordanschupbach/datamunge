#!/usr/bin/env python3
"""SqueezeNet (CIFAR-10, v1.0/v1.1) training-throughput baseline in PyTorch.

Reproduces the training block of
docs/mepml/examples/nn/1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/squeezenet.mepml
(model: include/datamunge/nn/squeezenet_models.hpp) so that the in-house C++
implementation can be compared against an external framework on the SAME
workload:

* input 32x32x3 float32 images scaled to [-1, 1], batch 32;
* stem conv1 3x3/1 SAME (96 ch for v1.0, 64 for v1.1) + bias + ReLU;
* Caffe-style ceil-mode 3x3/2 max pools (32 -> 16 -> 8 -> 4);
* Fire2..Fire9 with (squeeze, expand) = (16,64)x2 (32,128)x2 (48,192)x2 (64,256)x2,
  pools after fire4/fire8 (v1.0) or fire3/fire5 (v1.1);
* dropout (keep 0.5) on the final feature map, conv10 1x1 -> 10 classes + bias
  + ReLU, global average pool -> logits;
* loss = mean sparse softmax cross-entropy + 5e-5 * sum of squares of every
  parameter with ndim >= 2 (kept in the loss, not as weight_decay);
* Adam lr 5e-4 (constant; the C++ report uses a warmup-cosine schedule, which
  does not change the per-step cost);
* data: first 8000 rows of CIFAR-10 data_batch_1.bin, one epoch = 250 steps;
* augmentation (default on): zero-pad 4 px on the [0,1] image, random 32x32
  crop, random horizontal flip, then x -> 2x-1, done with batched tensor ops
  in-process (no DataLoader workers).  --no-augment feeds pre-scaled batches
  straight from the dataset so the pure model-step cost is isolated.

Timing: `--warmup` untimed steps, then timed steps until BOTH `--steps` steps
and `--seconds` seconds have elapsed (wall clock, torch.cuda.synchronize()
before reading the clock on CUDA).  The script prints a one-line summary and
writes a JSON record with `--json`.

Thread control: `--threads N` sets OMP_NUM_THREADS / MKL_NUM_THREADS *before*
torch is imported and calls torch.set_num_threads(N).  Launch each thread
count as its own process.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time

FIRES = [(16, 64), (16, 64), (32, 128), (32, 128), (48, 192), (48, 192), (64, 256), (64, 256)]
EXPECTED_PARAMS = {10: 729034, 11: 727626}
L2_COEFF = 5e-5
BATCH = 32
TRAIN_ROWS = 8000
STEPS_PER_EPOCH = TRAIN_ROWS // BATCH


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = leave torch defaults)")
    ap.add_argument("--augment", dest="augment", action="store_true", default=True)
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--steps", type=int, default=100, help="minimum number of timed steps")
    ap.add_argument("--seconds", type=float, default=20.0, help="minimum timed wall-clock budget")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--version", type=int, default=10, choices=[10, 11])
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--schedule", default="constant", choices=["constant", "warmup_cosine"],
                    help="warmup_cosine = the C++ report's schedule: linear 0->lr over --warmup-steps, "
                         "then cosine to 0 at --schedule-steps")
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--schedule-steps", type=int, default=10 * STEPS_PER_EPOCH)
    ap.add_argument("--memory-format", default="contiguous", choices=["contiguous", "channels_last"])
    ap.add_argument("--tf32", default="default", choices=["default", "off", "on"],
                    help="CUDA only: TF32 for cuDNN conv / matmul (default = torch defaults)")
    ap.add_argument("--cudnn-benchmark", action="store_true")
    ap.add_argument("--compile", action="store_true", help="wrap the model in torch.compile (experimental)")
    ap.add_argument("--data", default=os.path.expanduser("~/.cache/datamunge/datasets/cifar10/data_batch_1.bin"))
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--json", default=None, help="write the JSON record here")
    ap.add_argument("--label", default=None, help="free-form label stored in the JSON record")
    return ap.parse_args()


ARGS = parse_args()
if ARGS.threads > 0:
    os.environ["OMP_NUM_THREADS"] = str(ARGS.threads)
    os.environ["MKL_NUM_THREADS"] = str(ARGS.threads)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

if ARGS.threads > 0:
    torch.set_num_threads(ARGS.threads)


# --------------------------------------------------------------------------- data
def load_cifar_batch(path: str, rows: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return (images uint8 [rows,3,32,32] CHW, labels int64 [rows], is_real)."""
    if os.path.exists(path):
        raw = np.fromfile(path, dtype=np.uint8).reshape(-1, 3073)[:rows]
        labels = raw[:, 0].astype(np.int64)
        images = raw[:, 1:].reshape(-1, 3, 32, 32)
        return images, labels, True
    rng = np.random.default_rng(0)
    print(f"WARNING: {path} missing, using random data of the same shape", file=sys.stderr)
    return (rng.integers(0, 256, (rows, 3, 32, 32), dtype=np.uint8),
            rng.integers(0, 10, rows, dtype=np.int64), False)


# --------------------------------------------------------------------------- model
class Fire(nn.Module):
    def __init__(self, cin: int, squeeze: int, expand: int):
        super().__init__()
        self.squeeze = nn.Conv2d(cin, squeeze, 1)
        self.expand1x1 = nn.Conv2d(squeeze, expand, 1)
        self.expand3x3 = nn.Conv2d(squeeze, expand, 3, padding=1)

    def forward(self, x):
        s = F.relu(self.squeeze(x))
        return torch.cat([F.relu(self.expand1x1(s)), F.relu(self.expand3x3(s))], 1)


class SqueezeNet(nn.Module):
    """SqueezeNet v1.0/v1.1 with the CIFAR stem, as in squeezenet_models.hpp."""

    def __init__(self, version: int = 10, classes: int = 10, keep_probability: float = 0.5):
        super().__init__()
        stem = 96 if version == 10 else 64
        pools = {4, 8} if version == 10 else {3, 5}
        self.conv1 = nn.Conv2d(3, stem, 3, padding=1)
        # Caffe ceil mode == torch ceil_mode=True (pads bottom/right only).
        self.pool = nn.MaxPool2d(3, 2, ceil_mode=True)
        layers: list[nn.Module] = []
        cin = stem
        for i, (s, e) in enumerate(FIRES):
            layers.append(Fire(cin, s, e))
            cin = 2 * e
            if i + 2 in pools:
                layers.append(nn.MaxPool2d(3, 2, ceil_mode=True))
        self.fires = nn.Sequential(*layers)
        self.dropout = nn.Dropout(1.0 - keep_probability)
        self.conv10 = nn.Conv2d(cin, classes, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")  # He-uniform
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.conv10.weight, 0.0, 0.01)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.fires(x)
        x = self.dropout(x)
        x = F.relu(self.conv10(x))  # ReLU class maps, as in the C++ model
        return x.mean((2, 3))


# --------------------------------------------------------------------------- augmentation
def augment_batch(x01: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Zero-pad 4 px, random 32x32 crop, random h-flip, then 2x-1. x01: [B,3,32,32] in [0,1]."""
    b = x01.shape[0]
    dev = x01.device
    xp = F.pad(x01, (4, 4, 4, 4))  # constant 0 on the [0,1] image
    oy = torch.randint(0, 9, (b,), device=dev, generator=gen)
    ox = torch.randint(0, 9, (b,), device=dev, generator=gen)
    ar = torch.arange(32, device=dev)
    rows = (oy[:, None] + ar)[:, None, :, None]  # [B,1,32,1]
    cols = (ox[:, None] + ar)[:, None, None, :]  # [B,1,1,32]
    bi = torch.arange(b, device=dev)[:, None, None, None]
    ci = torch.arange(3, device=dev)[None, :, None, None]
    crop = xp[bi, ci, rows, cols]  # [B,3,32,32]
    flip = (torch.rand(b, device=dev, generator=gen) < 0.5)[:, None, None, None]
    crop = torch.where(flip, crop.flip(3), crop)
    return 2.0 * crop - 1.0


# --------------------------------------------------------------------------- main
def main() -> int:
    a = ARGS
    torch.manual_seed(a.seed)
    use_cuda = a.device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        print("CUDA requested but torch.cuda.is_available() is False", file=sys.stderr)
        return 2
    dev = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        if a.tf32 == "off":
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        elif a.tf32 == "on":
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = bool(a.cudnn_benchmark)
    mem_fmt = torch.channels_last if a.memory_format == "channels_last" else torch.contiguous_format

    images_u8, labels_np, real_data = load_cifar_batch(a.data, TRAIN_ROWS)
    x01 = torch.from_numpy(images_u8).to(dev).float().div_(255.0)  # [8000,3,32,32] in [0,1]
    y = torch.from_numpy(labels_np).to(dev)
    x_scaled = (2.0 * x01 - 1.0).contiguous(memory_format=mem_fmt)  # for --no-augment

    model = SqueezeNet(a.version).to(dev).to(memory_format=mem_fmt)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params != EXPECTED_PARAMS[a.version]:
        print(f"parameter count {n_params} != expected {EXPECTED_PARAMS[a.version]}", file=sys.stderr)
        return 1
    kernels = [p for p in model.parameters() if p.ndim >= 2]
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.999), eps=1e-8)
    step_model = torch.compile(model) if a.compile else model
    model.train()

    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed + 1)
    perm_gen = torch.Generator(device="cpu")
    perm_gen.manual_seed(a.seed + 2)
    order = torch.randperm(TRAIN_ROWS, generator=perm_gen).to(dev)
    cursor = 0

    def next_rows():
        nonlocal order, cursor
        if cursor + BATCH > TRAIN_ROWS:
            order = torch.randperm(TRAIN_ROWS, generator=perm_gen).to(dev)
            cursor = 0
        rows = order[cursor:cursor + BATCH]
        cursor += BATCH
        return rows

    iteration = 0

    def learning_rate(step: int) -> float:
        """Mirror datamunge::nn::schedule::warmup_cosine(peak, 0, warmup, duration, step)."""
        if a.schedule == "constant":
            return a.lr
        if step < a.warmup_steps:
            return a.lr * (step + 1) / a.warmup_steps
        progress = min(1.0, (step - a.warmup_steps) / max(1, a.schedule_steps - a.warmup_steps))
        return 0.5 * a.lr * (1.0 + math.cos(math.pi * progress))

    def train_step() -> torch.Tensor:
        nonlocal iteration
        if a.schedule != "constant":
            lr = learning_rate(iteration)
            for g in opt.param_groups:
                g["lr"] = lr
        iteration += 1
        rows = next_rows()
        if a.augment:
            xb = augment_batch(x01[rows], gen).contiguous(memory_format=mem_fmt)
        else:
            xb = x_scaled[rows]
        yb = y[rows]
        logits = step_model(xb)
        nll = F.cross_entropy(logits, yb)
        l2 = sum(torch.sum(w * w) for w in kernels)
        loss = nll + L2_COEFF * l2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return nll.detach()

    def sync():
        if use_cuda:
            torch.cuda.synchronize()

    # warmup
    for _ in range(a.warmup):
        last = train_step()
    sync()

    # timed section
    timed = 0
    losses: list[float] = []
    cpu0 = time.process_time()
    t0 = time.perf_counter()
    while True:
        last = train_step()
        timed += 1
        if timed % 10 == 0:
            losses.append(float(last))  # periodic host sync is cheap relative to a step
        if timed >= a.steps:
            sync()
            elapsed = time.perf_counter() - t0
            if elapsed >= a.seconds:
                break
    sync()
    elapsed = time.perf_counter() - t0
    cpu_seconds = time.process_time() - cpu0  # user+sys over all threads of this process
    final_loss = float(last)
    steps_per_s = timed / elapsed
    img_per_s = steps_per_s * BATCH
    s_per_epoch = STEPS_PER_EPOCH / steps_per_s
    tail = losses[-5:] if losses else [final_loss]

    info = {
        "framework": "pytorch",
        "label": a.label,
        "model": f"squeezenet_v{a.version / 10:.1f}_cifar",
        "version": a.version,
        "device": a.device,
        "threads": torch.get_num_threads() if not use_cuda else None,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "augment": a.augment,
        "memory_format": a.memory_format,
        "tf32": a.tf32 if use_cuda else None,
        "cudnn_benchmark": bool(a.cudnn_benchmark) if use_cuda else None,
        "torch_compile": a.compile,
        "batch": BATCH,
        "lr": a.lr,
        "schedule": a.schedule,
        "l2_coeff": L2_COEFF,
        "params": n_params,
        "real_data": real_data,
        "warmup_steps": a.warmup,
        "timed_steps": timed,
        "timed_seconds": elapsed,
        "cpu_seconds": cpu_seconds,
        "cpu_utilization": cpu_seconds / elapsed,
        "steps_per_s": steps_per_s,
        "images_per_s": img_per_s,
        "seconds_per_epoch": s_per_epoch,
        "final_nll": final_loss,
        "nll_trace_every_10_steps": losses,
        "mean_nll_last_50_steps": float(np.mean(tail)),
        "ln10": math.log(10.0),
        "timing_method": "time.perf_counter around the timed loop; torch.cuda.synchronize() before "
                         "start/stop on CUDA; warmup excluded; NLL read every 10 steps; "
                         "cpu_utilization = process CPU time / wall time over the timed loop",
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if use_cuda else None,
            "parallel_info": torch.__config__.parallel_info(),
        },
        "host": {"cpu_count": os.cpu_count(), "node": platform.node(), "platform": platform.platform()},
    }
    if use_cuda:
        props = torch.cuda.get_device_properties(0)
        cap = f"sm_{props.major}{props.minor}"
        arch_list = torch.cuda.get_arch_list()
        info["gpu"] = {
            "name": props.name,
            "capability": cap,
            "total_memory_gb": props.total_memory / 2**30,
            "wheel_arch_list": arch_list,
            "native_sass_for_gpu": cap in arch_list,
            "ptx_jit_fallback": cap not in arch_list,
            "driver": None,
        }
        try:
            import subprocess
            info["gpu"]["driver"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            pass
    summary = (f"pytorch {torch.__version__} | {info['model']} | {a.device}"
               f"{'' if use_cuda else f' threads={info['threads']}'} | aug={'on' if a.augment else 'off'}"
               f" | fmt={a.memory_format}"
               f"{'' if not use_cuda else f' tf32={a.tf32} cudnn.benchmark={a.cudnn_benchmark}'}"
               f" | {steps_per_s:.2f} steps/s | {img_per_s:.0f} img/s | {s_per_epoch:.1f} s/epoch"
               f" | {timed} steps in {elapsed:.1f}s | final NLL {final_loss:.3f}")
    print(summary)
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump(info, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
