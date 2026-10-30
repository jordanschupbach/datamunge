#!/usr/bin/env python3
"""Independent TensorFlow/Keras feature/objective/gradient test oracles only.

This script never prepares report numeric data or runs a demonstrated fit.
Use the pinned isolated CPU environment in cnn_style_reference_requirements.txt.
Prepare original sources first with datamunge-prepare-data style DIRECTORY.
This optional development oracle is never invoked by report data preparation.
"""
from __future__ import annotations
import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)


def extractor(root: Path, version: int, average: bool):
    """Build independent Keras convolutions with imported genuine source weights."""
    archive = np.load(root / f"vgg{version}.npz")
    depths = (2, 2, 4, 4, 4) if version == 19 else (2, 2, 3, 3, 3)
    widths = (64, 128, 256, 512, 512)
    layers = {}
    incoming = 3
    for b, depth in enumerate(depths, 1):
        for l in range(1, depth + 1):
            name = f"block{b}_conv{l}"
            layer = tf.keras.layers.Conv2D(widths[b-1], 3, padding="same", activation="relu", name=name, trainable=False)
            layer.build((None, None, None, incoming))
            layer.set_weights([archive[name + "_W"], archive[name + "_b"]])
            layers[name] = layer
            incoming = widths[b-1]

    def extract(rgb, blocks=5):
        x = tf.gather(rgb, [2, 1, 0], axis=3) * tf.constant(255., tf.float32) - tf.constant([103.939, 116.779, 123.68], tf.float32)
        outputs = {}
        for b, depth in enumerate(depths[:blocks], 1):
            for l in range(1, depth + 1):
                name = f"block{b}_conv{l}"
                x = layers[name](x)
                outputs[name] = x
                if version == 19 and b == 5 and l == 1:
                    return outputs
            if b < blocks:
                x = tf.nn.avg_pool2d(x, 2, 2, "VALID") if average else tf.nn.max_pool2d(x, 2, 2, "VALID")
        return outputs
    return extract


def gram(x, channels=False):
    flat = tf.reshape(x, [x.shape[0], -1, x.shape[3]])
    return tf.matmul(flat, flat, transpose_a=True) / tf.constant(flat.shape[1] * (flat.shape[2] if channels else 1), tf.float32)


def fixture(shape, phase):
    coordinates = np.arange(np.prod(shape), dtype=np.float32)
    return tf.constant((.45 + .3 * np.sin(coordinates * np.float32(.031) + np.float32(phase))).reshape(shape))


def save(root, name, values):
    np.savez(root / name, **{key: np.asarray(value, dtype=np.float32) for key, value in values.items()})
    print(f"Independent Keras fixture {name}:{len(values)} arrays.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    root = parser.parse_args().root
    v19 = extractor(root, 19, True)
    names = [f"block{i}_conv1" for i in range(1, 6)]
    x = tf.Variable(fixture((1, 32, 48, 3), .1))
    target = v19(fixture(x.shape, 1.3))
    targets = {name: gram(target[name]) for name in names}
    with tf.GradientTape() as tape:
        net = v19(x)
        content = .5 * tf.reduce_sum(tf.square(net["block4_conv2"] - target["block4_conv2"]))
        style = .2 * tf.add_n([.25 * tf.reduce_mean(tf.square(gram(net[name]) - targets[name])) for name in names])
        objective = tf.constant(1e-7, tf.float32) * content + tf.constant(1e-5, tf.float32) * style
    values = {"input": x, "content_target": target["block4_conv2"], "content_loss": content, "style_loss": style, "objective": objective, "image_gradient": tape.gradient(objective, x), "block4_conv2": net["block4_conv2"]}
    values.update({name: net[name] for name in names})
    values.update({name + "_gram": targets[name] for name in names})
    save(root, "reference.npz", values)
    del v19

    v16 = extractor(root, 16, False)
    names = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3"]
    x = tf.Variable(fixture((2, 32, 40, 3), .3))
    target = v16(fixture(x.shape, 1.7), 4)
    targets = {name: gram(target[name], True) for name in names}
    with tf.GradientTape() as tape:
        net = v16(x, 4)
        content = tf.reduce_mean(tf.square(net["block2_conv2"] - target["block2_conv2"]))
        style = tf.add_n([tf.reduce_mean(tf.reduce_sum(tf.square(gram(net[name], True) - targets[name]), axis=[1, 2])) for name in names])
        center = x[:, :-1, :-1, :]
        tv = .5 * tf.reduce_mean(tf.square(center-x[:, :-1, 1:, :]) + tf.square(center-x[:, 1:, :-1, :]))
        objective = 1e-5 * content + 1e-7 * style + .03 * tv
    values = {"input": x, "content_target": target["block2_conv2"], "content_loss": content, "style_loss": style, "tv": tv, "objective": objective, "image_gradient": tape.gradient(objective, x)}
    values.update({name: net[name] for name in names})
    values.update({name + "_gram": targets[name] for name in names})
    save(root, "johnson_reference.npz", values)

    x = tf.Variable(fixture((2, 32, 32, 3), .7))
    truth = fixture(x.shape, 1.9)
    mask = tf.constant(np.broadcast_to((((np.arange(32)[:, None] - 15) ** 2 + (np.arange(32)[None, :] - 17) ** 2) > 64)[None, :, :, None], x.shape).astype(np.float32))
    def pools(image):
        f = v16(image, 3)
        return [tf.nn.max_pool2d(f[name], 2, 2, "VALID") / 255. for name in names[:3]]
    target_features = pools(truth)
    with tf.GradientTape() as tape:
        composite = mask * truth + (1.-mask) * x
        raw_features, composite_features = pools(x), pools(composite)
        observed = tf.reduce_mean(mask * tf.abs(x-truth))
        hole = tf.reduce_mean((1.-mask) * tf.abs(x-truth))
        perceptual = tf.add_n([tf.reduce_mean(tf.abs(r-t)) + tf.reduce_mean(tf.abs(c-t)) for r,c,t in zip(raw_features, composite_features, target_features)])
        style_raw = tf.add_n([tf.reduce_mean(tf.abs(gram(r, True)-gram(t, True))) for r,t in zip(raw_features,target_features)])
        style_comp = tf.add_n([tf.reduce_mean(tf.abs(gram(c, True)-gram(t, True))) for c,t in zip(composite_features,target_features)])
        region = tf.nn.max_pool2d(1.-mask,3,1,"SAME")
        horizontal = region[:,:,:-1,:]*region[:,:,1:,:]*tf.abs(composite[:,:,:-1,:]-composite[:,:,1:,:])
        vertical = region[:,:-1,:,:]*region[:,1:,:,:]*tf.abs(composite[:,:-1,:,:]-composite[:,1:,:,:])
        tv = (tf.reduce_sum(horizontal)+tf.reduce_sum(vertical))/tf.constant(np.prod(x.shape),tf.float32)
        total = observed + 6.*hole + .05*perceptual + 120.*(style_raw+style_comp) + .1*tv
    values = {"prediction":x,"truth":truth,"mask":mask,"components":tf.stack([total,observed,hole,perceptual,style_raw,style_comp,tv]),"gradient":tape.gradient(total,x)}
    values.update({f"raw_pool{i+1}":f for i,f in enumerate(raw_features)})
    values.update({f"target_pool{i+1}":f for i,f in enumerate(target_features)})
    save(root,"partial_reference.npz",values)
    (root/"reference_manifest.json").write_text(json.dumps({"tensorflow":tf.__version__,"keras":tf.keras.__version__,"purpose":"Independent native feature/objective/every-input-gradient tests only","VGG19":"Genuine frozen full-width Keras convolutions with average pools","VGG16":"Genuine frozen full-width Keras convolutions with max pools","source":"scripts/cnn_style_reference.py","precision":"Float32,oneDNN disabled,two CPU threads"},indent=2)+"\n")


if __name__ == "__main__":
    main()
