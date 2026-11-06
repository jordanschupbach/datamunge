#!/usr/bin/env python3
"""SqueezeNet (CIFAR-10, v1.0/v1.1) training-throughput baseline in TensorFlow/Keras.

Same workload as squeezenet_torch.py / the C++ reference
(docs/mepml/examples/nn/1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/squeezenet.mepml,
include/datamunge/nn/squeezenet_models.hpp): NHWC float32, batch 32, 32x32x3
inputs in [-1,1], CIFAR stem 3x3/1 SAME, Caffe ceil-mode 3x3/2 max pools
(32 -> 16 -> 8 -> 4), Fire2..Fire9, dropout keep 0.5, conv10 1x1 + ReLU,
global average pool, loss = mean sparse softmax CE + 5e-5 * sum of squared
conv kernels, Adam lr 5e-4, first 8000 rows of data_batch_1.bin (250 steps per
epoch), on-the-fly pad-4 / random-crop / random-flip augmentation done with
batched tensor ops inside the compiled train step (--no-augment feeds
pre-scaled batches instead).

Ceil-mode pooling: tf MaxPool2D(3, 2, padding="same") pads 0 on top/left and 1
on bottom/right for every side used here (32, 16, 8) and ignores padded
positions, which is exactly Caffe's ceil mode / the C++ CeilMaxPool.

Modes: --mode graph (tf.function, default), eager, or xla (tf.function with
jit_compile=True).  Threads: --threads N sets OMP_NUM_THREADS /
TF_NUM_INTRAOP_THREADS before import and
tf.config.threading.set_intra_op_parallelism_threads(N) (inter-op pool is
--inter-threads, default 2).  Launch each thread count as its own process.
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
    ap.add_argument("--threads", type=int, default=0, help="intra-op threads (0 = TF default)")
    ap.add_argument("--inter-threads", type=int, default=2, help="inter-op threads (0 = TF default)")
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
    ap.add_argument("--mode", default="graph", choices=["graph", "eager", "xla"])
    ap.add_argument("--data", default=os.path.expanduser("~/.cache/datamunge/datasets/cifar10/data_batch_1.bin"))
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--json", default=None, help="write the JSON record here")
    ap.add_argument("--label", default=None)
    return ap.parse_args()


ARGS = parse_args()
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
if ARGS.threads > 0:
    os.environ["OMP_NUM_THREADS"] = str(ARGS.threads)
    os.environ["MKL_NUM_THREADS"] = str(ARGS.threads)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(ARGS.threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(ARGS.inter_threads)
if ARGS.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # keep TF strictly on the CPU

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

if ARGS.threads > 0:
    tf.config.threading.set_intra_op_parallelism_threads(ARGS.threads)
    tf.config.threading.set_inter_op_parallelism_threads(ARGS.inter_threads)
if ARGS.mode == "eager":
    tf.config.run_functions_eagerly(True)

from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402


# --------------------------------------------------------------------------- data
def load_cifar_batch(path: str, rows: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return (images uint8 [rows,32,32,3] NHWC, labels int32 [rows], is_real)."""
    if os.path.exists(path):
        raw = np.fromfile(path, dtype=np.uint8).reshape(-1, 3073)[:rows]
        labels = raw[:, 0].astype(np.int32)
        images = raw[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return np.ascontiguousarray(images), labels, True
    rng = np.random.default_rng(0)
    print(f"WARNING: {path} missing, using random data of the same shape", file=sys.stderr)
    return (rng.integers(0, 256, (rows, 32, 32, 3), dtype=np.uint8),
            rng.integers(0, 10, rows, dtype=np.int32), False)


# --------------------------------------------------------------------------- model
def conv(x, filters, k, name, init="he_uniform"):
    return layers.Conv2D(filters, k, padding="same", activation="relu", kernel_initializer=init,
                         bias_initializer="zeros", name=name)(x)


def fire(x, squeeze, expand, name):
    s = conv(x, squeeze, 1, f"{name}_squeeze1x1")
    e1 = conv(s, expand, 1, f"{name}_expand1x1")
    e3 = conv(s, expand, 3, f"{name}_expand3x3")
    return layers.Concatenate(axis=-1, name=f"{name}_concat")([e1, e3])


def ceil_max_pool(x, name):
    # SAME padding with k=3,s=2 on sides 32/16/8 pads bottom/right by one only.
    return layers.MaxPool2D(3, 2, padding="same", name=name)(x)


def build_squeezenet(version: int = 10, classes: int = 10, keep_probability: float = 0.5) -> keras.Model:
    stem = 96 if version == 10 else 64
    pools = {4, 8} if version == 10 else {3, 5}
    inp = keras.Input((32, 32, 3), batch_size=BATCH, dtype="float32")
    x = conv(inp, stem, 3, "conv1")
    x = ceil_max_pool(x, "pool1")
    for i, (s, e) in enumerate(FIRES):
        x = fire(x, s, e, f"fire{i + 2}")
        if i + 2 in pools:
            x = ceil_max_pool(x, f"pool{i + 2}")
    x = layers.Dropout(1.0 - keep_probability, name="dropout")(x)
    x = conv(x, classes, 1, "conv10", init=keras.initializers.RandomNormal(stddev=0.01))
    out = layers.GlobalAveragePooling2D(name="logits")(x)
    return keras.Model(inp, out, name=f"squeezenet_v{version}")


# --------------------------------------------------------------------------- augmentation
def augment_batch(x01: tf.Tensor, gen: tf.random.Generator) -> tf.Tensor:
    """Zero-pad 4 px, random 32x32 crop, random h-flip, then 2x-1. x01: [B,32,32,3] in [0,1]."""
    b = tf.shape(x01)[0]
    xp = tf.pad(x01, [[0, 0], [4, 4], [4, 4], [0, 0]])
    oy = gen.uniform([b], 0, 9, dtype=tf.int32)
    ox = gen.uniform([b], 0, 9, dtype=tf.int32)
    ar = tf.range(32)
    rows = oy[:, None] + ar  # [B,32]
    cols = ox[:, None] + ar
    crop = tf.gather(xp, rows, axis=1, batch_dims=1)  # [B,32,40,3]
    crop = tf.gather(crop, cols, axis=2, batch_dims=1)  # [B,32,32,3]
    flip = gen.uniform([b]) < 0.5
    crop = tf.where(flip[:, None, None, None], tf.reverse(crop, [2]), crop)
    return 2.0 * crop - 1.0


def _mkl_enabled():
    try:
        from tensorflow.python.framework import test_util
        return bool(test_util.IsMklEnabled())
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- main
def main() -> int:
    a = ARGS
    gpus = tf.config.list_physical_devices("GPU")
    use_cuda = a.device == "cuda"
    if use_cuda and not gpus:
        print("CUDA requested but TensorFlow sees no GPU", file=sys.stderr)
        return 2
    if use_cuda:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    tf.keras.utils.set_random_seed(a.seed)
    device_name = "/GPU:0" if use_cuda else "/CPU:0"

    images_u8, labels_np, real_data = load_cifar_batch(a.data, TRAIN_ROWS)
    with tf.device(device_name):
        x01 = tf.constant(images_u8, dtype=tf.float32) / 255.0
        x_scaled = 2.0 * x01 - 1.0
        y = tf.constant(labels_np, dtype=tf.int32)
        model = build_squeezenet(a.version)
        gen = tf.random.Generator.from_seed(a.seed + 1)
    n_params = int(sum(int(np.prod(w.shape)) for w in model.trainable_variables))
    if n_params != EXPECTED_PARAMS[a.version]:
        print(f"parameter count {n_params} != expected {EXPECTED_PARAMS[a.version]}", file=sys.stderr)
        return 1
    variables = model.trainable_variables
    kernels = [w for w in variables if len(w.shape) >= 2]
    class WarmupCosine(keras.optimizers.schedules.LearningRateSchedule):
        """Mirror datamunge::nn::schedule::warmup_cosine(peak, 0, warmup, duration, step)."""

        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            warm = a.lr * (step + 1.0) / float(a.warmup_steps)
            progress = tf.minimum(1.0, (step - a.warmup_steps) / float(max(1, a.schedule_steps - a.warmup_steps)))
            cos = 0.5 * a.lr * (1.0 + tf.cos(math.pi * progress))
            return tf.where(step < a.warmup_steps, warm, cos)

        def get_config(self):
            return {}

    lr = WarmupCosine() if a.schedule == "warmup_cosine" else a.lr
    opt = keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    opt.build(variables)

    def _step(rows):
        if a.augment:
            xb = augment_batch(tf.gather(x01, rows), gen)
        else:
            xb = tf.gather(x_scaled, rows)
        yb = tf.gather(y, rows)
        with tf.GradientTape() as tape:
            logits = model(xb, training=True)
            nll = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yb, logits=logits))
            l2 = tf.add_n([tf.reduce_sum(tf.square(w)) for w in kernels])
            loss = nll + L2_COEFF * l2
        grads = tape.gradient(loss, variables)
        opt.apply_gradients(zip(grads, variables))
        return nll

    if a.mode == "eager":
        step_fn = _step
    else:
        step_fn = tf.function(_step, jit_compile=(a.mode == "xla"),
                              input_signature=[tf.TensorSpec([BATCH], tf.int32)])

    rng = np.random.default_rng(a.seed + 2)
    order = rng.permutation(TRAIN_ROWS)
    cursor = 0

    def next_rows():
        nonlocal order, cursor
        if cursor + BATCH > TRAIN_ROWS:
            order = rng.permutation(TRAIN_ROWS)
            cursor = 0
        rows = order[cursor:cursor + BATCH]
        cursor += BATCH
        return tf.constant(rows, dtype=tf.int32)

    def sync():
        if use_cuda:
            try:
                tf.test.experimental.sync_devices()
            except Exception:  # noqa: BLE001
                pass

    t_trace = time.perf_counter()
    for _ in range(a.warmup):
        last = step_fn(next_rows())
    _ = float(last)
    sync()
    trace_seconds = time.perf_counter() - t_trace

    timed = 0
    losses: list[float] = []
    cpu0 = time.process_time()
    t0 = time.perf_counter()
    while True:
        last = step_fn(next_rows())
        timed += 1
        if timed % 10 == 0:
            losses.append(float(last))
        if timed >= a.steps:
            sync()
            elapsed = time.perf_counter() - t0
            if elapsed >= a.seconds:
                break
    final_loss = float(last)
    sync()
    elapsed = time.perf_counter() - t0
    cpu_seconds = time.process_time() - cpu0  # user+sys over all threads of this process
    steps_per_s = timed / elapsed
    img_per_s = steps_per_s * BATCH
    s_per_epoch = STEPS_PER_EPOCH / steps_per_s
    tail = losses[-5:] if losses else [final_loss]

    build = dict(tf.sysconfig.get_build_info())
    info = {
        "framework": "tensorflow",
        "label": a.label,
        "model": f"squeezenet_v{a.version / 10:.1f}_cifar",
        "version": a.version,
        "device": a.device,
        "mode": a.mode,
        "threads": tf.config.threading.get_intra_op_parallelism_threads() if not use_cuda else None,
        "inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads() if not use_cuda else None,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "augment": a.augment,
        "batch": BATCH,
        "lr": a.lr,
        "schedule": a.schedule,
        "l2_coeff": L2_COEFF,
        "params": n_params,
        "real_data": real_data,
        "warmup_steps": a.warmup,
        "warmup_seconds_incl_tracing": trace_seconds,
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
        "timing_method": "time.perf_counter around the timed loop; tf.test.experimental.sync_devices() "
                         "before start/stop on GPU; warmup (incl. tracing) excluded; NLL read every 10 steps; "
                         "cpu_utilization = process CPU time / wall time over the timed loop",
        "versions": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
            "cuda_build": build.get("cuda_version"),
            "cudnn_build": build.get("cudnn_version"),
            "is_cuda_build": build.get("is_cuda_build"),
            "mkl_onednn": _mkl_enabled(),
        },
        "host": {"cpu_count": os.cpu_count(), "node": platform.node(), "platform": platform.platform()},
    }
    if use_cuda:
        det = tf.config.experimental.get_device_details(gpus[0])
        info["gpu"] = {"name": det.get("device_name"), "compute_capability": det.get("compute_capability")}
    summary = (f"tensorflow {tf.__version__} | {info['model']} | {a.device}"
               f"{'' if use_cuda else f' threads={info['threads']}/{info['inter_op_threads']}'}"
               f" | mode={a.mode} | aug={'on' if a.augment else 'off'}"
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
