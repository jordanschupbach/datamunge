#!/usr/bin/env python3
"""Collect benchmarks/results/frameworks/*.json into a Markdown summary table.

Usage: summarize.py RESULTS_DIR > summary.md
Rows are sorted framework, device, threads, variant.  Only the Python standard
library is used so it runs with the plain nix python3.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys


def notes(r: dict) -> str:
    bits = []
    if r["framework"] == "pytorch":
        if r.get("memory_format") == "channels_last":
            bits.append("channels_last")
        if r.get("device") == "cuda":
            bits.append(f"tf32={r.get('tf32')}")
            if r.get("cudnn_benchmark"):
                bits.append("cudnn.benchmark")
            g = r.get("gpu") or {}
            if g:
                bits.append("native sm" if g.get("native_sass_for_gpu") else "PTX JIT")
        if r.get("torch_compile"):
            bits.append("torch.compile")
    else:
        bits.append(f"mode={r.get('mode')}")
        if r.get("device") == "cpu" and r.get("inter_op_threads") is not None:
            bits.append(f"inter-op={r['inter_op_threads']}")
    if r.get("schedule") == "warmup_cosine":
        bits.append("warmup-cosine schedule (training sanity run)")
    if r.get("lr") not in (None, 5e-4):
        bits.append(f"lr={r['lr']:g}")
    if not r.get("real_data", True):
        bits.append("RANDOM DATA")
    nll = r.get("final_nll")
    if nll is not None and abs(nll - math.log(10)) < 2e-3:
        bits.append("NLL stuck at ln10 (dead head?)")
    return ", ".join(bits)


def main() -> int:
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    rows = []
    for path in sorted(glob.glob(os.path.join(d, "*.json"))):
        with open(path) as fh:
            r = json.load(fh)
        rows.append((r, os.path.basename(path)))

    def key(item):
        r, name = item
        return (r["framework"], r["device"], r.get("threads") or 0, name)

    rows.sort(key=key)
    print("# SqueezeNet v1.0 CIFAR-10 training baselines (PyTorch / TensorFlow)")
    print()
    print("Batch 32, float32, Adam lr 5e-4, loss = NLL + 5e-5*sum(kernel^2); one epoch = 250 steps "
          "(8000 rows of CIFAR-10 data_batch_1).  `aug` = on-the-fly pad-4/random-crop/h-flip on the "
          "batch inside the timed step; `off` = pre-scaled batches straight from the dataset.  "
          "Timing: 10 warmup steps excluded, then >=100 steps and >=20 s.  "
          "NLL = mean sparse softmax cross-entropy of the last timed step (chance = ln 10 = 2.303).")
    print()
    print("| framework | device | threads | aug | steps/s | img/s | s/epoch | timed steps | CPU util | final NLL | notes |")
    print("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for r, name in rows:
        v = r["versions"]
        fw = f"{'torch' if r['framework'] == 'pytorch' else 'tf'} {v.get('torch') or v.get('tensorflow')}"
        dev = r["device"]
        if dev == "cuda" and r.get("gpu"):
            dev = "cuda (" + (r["gpu"].get("name") or "?").replace("NVIDIA GeForce ", "") + ")"
        th = r.get("threads") if r["device"] == "cpu" else "-"
        util = r.get("cpu_utilization")
        util_s = f"{util:.1f}x" if util is not None else "-"
        print(f"| {fw} | {dev} | {th} | {'on' if r['augment'] else 'off'} | {r['steps_per_s']:.2f} | "
              f"{r['images_per_s']:.0f} | {r['seconds_per_epoch']:.1f} | {r['timed_steps']} | {util_s} | "
              f"{r['final_nll']:.3f} | {notes(r)} |")
    print()
    print("Source records: " + ", ".join(f"`{n}`" for _, n in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
