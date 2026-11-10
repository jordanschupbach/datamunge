#!/usr/bin/env python3
"""Optional NumPy oracle for the visible-only MAE report; never prepares data or fits.

Implements patch indexing, unsigned selection, transformer forward and analytic
reverse rules independently of datamunge. Checks every first-step parameter
gradient, Adam rounding, all pools, mask schedules, attention and code controls.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def arrays(path):
    """Read owned numeric arrays, closing the archive and rejecting pickle."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def unbroadcast(value, shape):
    """Sum broadcast dimensions in an analytic reverse rule."""
    while value.ndim > len(shape):
        value = value.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1:
            value = value.sum(axis=axis, keepdims=True)
    return value.reshape(shape)


class Value:
    """Small local reverse graph using NumPy and explicit elementary derivatives."""
    def __init__(self, value, parents=()):
        self.v = np.asarray(value, dtype=np.float64)
        self.parents = parents
        self.g = np.zeros_like(self.v)

    def __add__(self, other):
        other = value(other)
        return Value(self.v + other.v,
                     ((self, lambda g: unbroadcast(g, self.v.shape)),
                      (other, lambda g: unbroadcast(g, other.v.shape))))

    def __mul__(self, other):
        other = value(other)
        return Value(self.v * other.v,
                     ((self, lambda g: unbroadcast(g * other.v, self.v.shape)),
                      (other, lambda g: unbroadcast(g * self.v, other.v.shape))))

    def __matmul__(self, other):
        other = value(other)
        return Value(self.v @ other.v,
                     ((self, lambda g: unbroadcast(g @ other.v.swapaxes(-1, -2), self.v.shape)),
                      (other, lambda g: unbroadcast(self.v.swapaxes(-1, -2) @ g, other.v.shape))))

    def reshape(self, shape):
        return Value(self.v.reshape(shape), ((self, lambda g: g.reshape(self.v.shape)),))

    def transpose(self, axes):
        inverse = np.argsort(axes)
        return Value(self.v.transpose(axes), ((self, lambda g: g.transpose(inverse)),))

    def backward(self, seed):
        ordered, seen = [], set()

        def visit(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            for parent, _ in node.parents:
                visit(parent)
            ordered.append(node)

        visit(self)
        self.g = seed
        for node in reversed(ordered):
            for parent, rule in node.parents:
                parent.g += rule(node.g)


def value(x):
    """Wrap constants without importing the native library."""
    return x if isinstance(x, Value) else Value(x)


def normalized(x):
    """Population LayerNorm, epsilon 1e-5, and its full continuous reverse rule."""
    centered = x.v - x.v.mean(axis=-1, keepdims=True)
    inverse = 1 / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + 1e-5)
    z = centered * inverse
    return Value(z, ((x, lambda g: inverse * (g - g.mean(axis=-1, keepdims=True)
                                             - z * (g * z).mean(axis=-1, keepdims=True))),))


def softmax(x):
    """Stable last-axis softmax and Jacobian-vector product."""
    e = np.exp(x.v - x.v.max(axis=-1, keepdims=True))
    y = e / e.sum(axis=-1, keepdims=True)
    return Value(y, ((x, lambda g: y * (g - (g * y).sum(axis=-1, keepdims=True))),))


def gelu(x):
    """Tanh approximation and analytic derivative, including the cubic term."""
    c = math.sqrt(2 / math.pi)
    t = np.tanh(c * (x.v + .044715 * x.v ** 3))
    y = .5 * x.v * (1 + t)
    derivative = .5 * (1 + t) + .5 * x.v * (1 - t * t) * c * (1 + 3 * .044715 * x.v * x.v)
    return Value(y, ((x, lambda g: g * derivative),))


def sigmoid(x):
    """Independent sigmoid and elementwise derivative."""
    y = 1 / (1 + np.exp(-x.v))
    return Value(y, ((x, lambda g: g * y * (1 - y)),))


def positions(width):
    """Source's grouped row/column sin/cos, rounded to Float32 then promoted."""
    result = np.zeros((16, width))
    for patch in range(16):
        for axis in range(2):
            coordinate = patch // 4 if axis == 0 else patch % 4
            for k in range(width // 4):
                angle = coordinate * math.pow(10000., -2. * k / (width / 2.))
                result[patch, axis * width // 2 + k] = np.float32(math.sin(angle))
                result[patch, axis * width // 2 + width // 4 + k] = np.float32(math.cos(angle))
    return result


def selection(ids, epoch=0, seed=1907):
    """Independent integer modulo-2^32 mixing and stable ranking, no RNG."""
    mask = (1 << 32) - 1
    result = np.zeros((len(ids), 4, 16), dtype=np.float32)
    for b, record in enumerate(ids):
        ranked = []
        for patch in range(16):
            x = (seed ^ ((int(record) * 0x9e3779b9) & mask)
                 ^ ((epoch * 0x85ebca6b) & mask) ^ ((patch * 0xc2b2ae35) & mask))
            x = ((x ^ (x >> 16)) * 0x7feb352d) & mask
            x = ((x ^ (x >> 15)) * 0x846ca68b) & mask
            x ^= x >> 16
            ranked.append((x, patch))
        for row, (_, patch) in enumerate(sorted(ranked)[:4]):
            result[b, row, patch] = 1
    return result


def patchify(images):
    """Use explicit image indices rather than the native reshape/transpose chain."""
    return np.stack([images[:, (p // 4) * 7:(p // 4 + 1) * 7,
                                   (p % 4) * 7:(p % 4 + 1) * 7, 0].reshape(-1, 49)
                     for p in range(16)], axis=1)


def decode(p, code, selector):
    """Target-free location restoration, mask token, transformer and pixel head."""
    projected = code @ p['decoder_projection/W'] + p['decoder_projection/b']
    restored = value(selector.swapaxes(1, 2)) @ projected
    hidden = 1 - selector.sum(axis=1)
    x = restored + value(hidden[:, :, None]) * p['mask_token'] + positions(16)
    x, attention = block(p, 'decoder', x, 16)
    x = layer_norm(p, 'decoder/norm', x)
    return sigmoid(x @ p['prediction/W'] + p['prediction/b']), attention


def layer_norm(p, name, x):
    """Shared learned affine LayerNorm."""
    return normalized(x) * p[name + '/gamma'] + p[name + '/beta']


def block(p, name, x, width):
    """Independent two-head attention and residual feed-forward block."""
    batch, tokens, _ = x.v.shape
    h = layer_norm(p, name + '/ln1', x)
    q, k, v = [(h @ p[name + '/attention/W' + suffix]).reshape((batch, tokens, 2, width // 2))
               .transpose((0, 2, 1, 3)) for suffix in ('q', 'k', 'v')]
    attention = softmax((q @ k.transpose((0, 1, 3, 2))) * (1 / math.sqrt(width // 2)))
    merged = (attention @ v).transpose((0, 2, 1, 3)).reshape((batch, tokens, width))
    residual = x + merged @ p[name + '/attention/Wo']
    ff = layer_norm(p, name + '/ln2', residual)
    ff = gelu(ff @ p[name + '/ff1/W'] + p[name + '/ff1/b'])
    return residual + ff @ p[name + '/ff2/W'] + p[name + '/ff2/b'], attention


def forward(p, images, selector):
    """Full visible-only forward, exposing code and attention for verification."""
    patches = patchify(images)
    visible = value(selector) @ value(patches)
    x = visible @ p['patch/W'] + p['patch/b'] + selector @ positions(32)
    x, encoder_attention = block(p, 'encoder', x, 32)
    code = layer_norm(p, 'encoder/norm', x)
    prediction, decoder_attention = decode(p, code, selector)
    return code, prediction, encoder_attention, decoder_attention


def parameters(archive):
    """Independent Float64 leaves with stable archive names."""
    return {key: Value(val) for key, val in archive.items()}


def check(directory):
    """Check saved report arrays, raising on mismatch; return measured evidence."""
    data = arrays(directory / 'masked_data.npz')
    initial = arrays(directory / 'masked_initial.npz')
    first = arrays(directory / 'masked_first_batch.npz')
    expected_gradients = arrays(directory / 'masked_first_gradients.npz')
    updated = arrays(directory / 'masked_first_update.npz')
    best = arrays(directory / 'masked_best.npz')
    outputs = arrays(directory / 'masked_outputs.npz')
    p = parameters(initial)
    _, prediction, _, _ = forward(p, first['images'].astype(np.float64), first['selection'])
    target = patchify(first['images']).astype(np.float64)
    hidden = 1 - first['selection'].sum(axis=1)
    prediction.backward(2 * (prediction.v - target) * hidden[:, :, None] / (49 * hidden.sum()))
    gradient_errors = {}
    assert len(p) == 35 and sum(x.v.size for x in p.values()) == 13649
    for key, leaf in p.items():
        actual = expected_gradients[key]
        np.testing.assert_allclose(leaf.g, actual, atol=2e-8, rtol=1e-4, err_msg=key)
        gradient_errors[key] = float(np.max(np.abs(leaf.g - actual)))
        g = actual
        first_moment, second_moment = g * np.float32(.1), (g * g) * np.float32(.001)
        direction = (first_moment / np.float32(.1)) / (np.sqrt(second_moment / np.float32(.001)) + np.float32(1e-8))
        np.testing.assert_array_equal((initial[key] - direction * np.float32(.003)).view(np.uint32), updated[key].view(np.uint32))
    mask_rows = 0
    for epoch in range(1, 4):
        native = arrays(directory / f'masked_epoch{epoch}_mask.npz')['selection']
        np.testing.assert_array_equal(selection(data['fit_source_ids'], epoch), native)
        mask_rows += len(native)
    errors = {}
    predicted_pixels = 0
    for split in ('fit', 'validation', 'test'):
        selector = selection(data[split + '_source_ids'])
        np.testing.assert_array_equal(selector, outputs[split + '_selection'])
        code, prediction, _, _ = forward(parameters(best), data[split + '_images'], selector)
        for name, native in (('code', code.v), ('prediction', prediction.v)):
            expected = outputs[split + '_' + name]
            np.testing.assert_allclose(native, expected, atol=3e-6, rtol=1e-5)
            errors[split + '_' + name] = float(np.max(np.abs(native - expected)))
        mask_rows += len(selector)
        predicted_pixels += prediction.v.size
    test = data['test_images']
    selector = outputs['test_selection']
    second_selector = selection(data['test_source_ids'], 0, 1913)
    np.testing.assert_array_equal(second_selector, outputs['second_selection'])
    mask_rows += len(second_selector)
    controls = {
        'zero': decode(parameters(best), value(np.zeros((len(test), 4, 32))), selector)[0].v,
        'shuffled': forward(parameters(best), np.roll(test, -1, axis=0), selector)[1].v,
        'second': forward(parameters(best), test, second_selector)[1].v,
        'permuted': forward(parameters(best), test, selector[:, ::-1])[1].v,
    }
    for name, prediction in controls.items():
        np.testing.assert_allclose(prediction, outputs[name + '_prediction'], atol=3e-6, rtol=1e-5)
        errors[name] = float(np.max(np.abs(prediction - outputs[name + '_prediction'])))
        predicted_pixels += prediction.size
    hidden = 1 - selector.sum(axis=1)
    altered = test.copy()
    for pidx in range(16):
        y, x = pidx // 4 * 7, pidx % 4 * 7
        rows = hidden[:, pidx].astype(bool)
        altered[rows, y:y + 7, x:x + 7] = 1 - altered[rows, y:y + 7, x:x + 7]
    original_code, original_prediction, _, _ = forward(parameters(best), test, selector)
    altered_code, altered_prediction, _, _ = forward(parameters(best), altered, selector)
    np.testing.assert_array_equal(original_code.v, altered_code.v)
    np.testing.assert_array_equal(original_prediction.v, altered_prediction.v)
    np.testing.assert_allclose(controls['permuted'], original_prediction.v, atol=1e-13, rtol=0)
    probe = arrays(directory / 'masked_probe.npz')
    code, prediction, enc, dec = forward(parameters(best), probe['images'], probe['selection'])
    np.testing.assert_array_equal(patchify(probe['images']), probe['patches'])
    np.testing.assert_array_equal(probe['selection'] @ probe['patches'], probe['visible'])
    for name, attention in (('encoder_attention', enc.v), ('decoder_attention', dec.v)):
        np.testing.assert_allclose(attention, probe[name], atol=2e-6, rtol=2e-5)
        np.testing.assert_allclose(attention.sum(axis=-1), 1, atol=1e-14, rtol=0)
    result = {
        'parameter_gradients_checked': 13649, 'first_adam_updates_bit_exact': True,
        'gradient_max_errors': gradient_errors, 'forward_max_errors': errors,
        'predicted_pixels_checked': predicted_pixels, 'mask_rows_checked': mask_rows,
        'hidden_input_independence': True, 'visible_order_invariance': True,
        'probe_attentions_and_patch_mapping': True,
    }
    return result


def main():
    """Run the optional oracle on a completed report's local build directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    result = check(args.directory)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
