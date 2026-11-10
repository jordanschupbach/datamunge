#!/usr/bin/env python3
"""Optional independent TensorFlow audit for the original frozen Inception GraphDef.

Reads exact model/input digests and native stage archives. No dataset preparation,
training, other-checkout calls or substitute feature network. TensorFlow is needed
only for this development oracle; native reports must not import this script.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import tensorflow as tf


def check(root):
    model = root / 'assets/classify_image_graph_def.pb'
    raw = model.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == '009d6814d1bc560d4e7b236e170e9b2d5ca6f4b57bd8037f6db05776204415c6'
    proto = tf.compat.v1.GraphDef()
    proto.ParseFromString(raw)
    constant_count = constant_values = 0
    with np.load(root / 'native-constants.npz', allow_pickle=False) as native:
        expected_names = set()
        for node in proto.node:
            if node.op != 'Const' or node.name == 'DecodeJpeg/contents':
                continue
            expected_names.add(node.name)
            expected = tf.make_ndarray(node.attr['value'].tensor)
            actual = native[node.name]
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            assert actual.tobytes() == expected.tobytes(), node.name
            constant_count += 1
            constant_values += actual.size
        assert set(native.files) == expected_names
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(proto, name='')
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=1)
    comparisons, total, worst = [], 0, 0.
    with np.load(root / 'inception-inputs.npz', allow_pickle=False) as inputs:
        with tf.compat.v1.Session(graph=graph, config=config) as session:
            for name in inputs.files:
                with np.load(root / ('native-' + name + '.npz'), allow_pickle=False) as native:
                    names = list(native.files)
                    values = session.run([graph.get_tensor_by_name(n + ':0') for n in names],
                                         {graph.get_tensor_by_name('ExpandDims:0'): inputs[name]})
                    np.savez_compressed(root / ('tensorflow-' + name + '.npz'), **dict(zip(names, values)))
                    for key, expected in zip(names, values):
                        actual = native[key]
                        assert actual.dtype == expected.dtype and actual.shape == expected.shape
                        error = float(np.max(abs(actual.astype(float) - expected.astype(float))))
                        np.testing.assert_allclose(actual, expected, atol=6e-4, rtol=8e-3,
                                                   err_msg=name + '/' + key)
                        if key in ['ResizeBilinear', 'Mul']:
                            assert actual.tobytes() == expected.tobytes(), key
                        if key == 'pool_3':
                            assert error < 3e-5, (name, key, error)
                        if key == 'softmax':
                            assert error < 4e-6, (name, key, error)
                        comparisons.append(dict(input=name, node=key, values=int(actual.size), maximum_error=error))
                        total += actual.size
                        worst = max(worst, error)
    return dict(passed=True, tensorflow=tf.__version__, graph_sha256=digest, nodes=len(proto.node),
                numeric_constants_bit_exact=constant_count, constant_values_bit_exact=constant_values,
                values_checked=total, maximum_absolute_error=worst, comparisons=comparisons,
                oneDNN_disabled=True, original_graph_and_1008_classes=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('build/nn-report-conversion/gan_diagnostics'))
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    result = json.dumps(check(args.root), indent=2) + '\n'
    print(result, end='')
    if args.json:
        args.json.write_text(result)


if __name__ == '__main__':
    main()
