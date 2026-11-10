#!/usr/bin/env python3
"""Optional NumPy oracle for the native convolutional-autoencoder report.

Reads C++-generated archives only; it never prepares data, downloads or fits.
Checks all held-out pixels, all six saved activation endpoints, all 741 first
parameter derivatives, the Float32 Adam step, baselines and original MNIST IDs.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def arrays(path):
    """Load owned arrays without pickle and immediately close their archive."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def convolution(x, weight, bias):
    """Independent NHWC same-padded stride-one 3x3 cross-correlation."""
    batch, height, width, _ = x.shape
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    result = np.zeros((batch, height, width, len(bias)), dtype=np.float64)
    for dy in range(3):
        for dx in range(3):
            result += np.einsum('nhwc,co->nhwo', padded[:, dy:dy+height, dx:dx+width], weight[dy, dx])
    return result + bias


def convolution_adjoint(x, weight, gradient):
    """Sum all shared-kernel/bias derivatives and scatter-add input adjoints."""
    _, height, width, _ = x.shape
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    dx = np.zeros_like(padded)
    dw = np.zeros_like(weight)
    for y in range(3):
        for col in range(3):
            window = padded[:, y:y+height, col:col+width]
            dw[y, col] = np.einsum('nhwc,nhwo->co', window, gradient)
            dx[:, y:y+height, col:col+width] += np.einsum('nhwo,co->nhwc', gradient, weight[y, col])
    return dx[:, 1:-1, 1:-1], dw, gradient.sum(axis=(0, 1, 2))


def pool(x, factor=2):
    """Average nonoverlapping spatial blocks; preserve channels."""
    n, h, w, c = x.shape
    return x.reshape(n, h//factor, factor, w//factor, factor, c).mean(axis=(2, 4))


def repeat(x, factor=2):
    """Repeat each value spatially, rather than tile the complete image."""
    return x.repeat(factor, axis=1).repeat(factor, axis=2)


def decoder(code, parameters):
    """Independent untied decoder with sigmoid output and retained intermediates."""
    enlarged = repeat(code)
    hidden = np.maximum(convolution(enlarged, parameters['decoder1/W'], parameters['decoder1/b']), 0)
    second = repeat(hidden)
    logits = convolution(second, parameters['decoder2/W'], parameters['decoder2/b'])
    reconstruction = 1 / (1 + np.exp(-logits))
    return dict(enlarged=enlarged, decoder1=hidden, second=second, reconstruction=reconstruction)


def forward(image, parameters):
    """Evaluate all six original endpoints with independent Float64 operations."""
    conv1 = np.maximum(convolution(image, parameters['encoder1/W'], parameters['encoder1/b']), 0)
    pool1 = pool(conv1)
    conv2 = np.maximum(convolution(pool1, parameters['encoder2/W'], parameters['encoder2/b']), 0)
    code = pool(conv2)
    return dict(conv1=conv1, pool1=pool1, conv2=conv2, code=code, **decoder(code, parameters))


def check(directory):
    """Validate numeric artifacts and return measured errors; raise on disagreement."""
    result = arrays(directory / 'convolutional_results.npz')
    initial = arrays(directory / 'convolutional_initial.npz')
    gradients = arrays(directory / 'convolutional_first_gradients.npz')
    updated = arrays(directory / 'convolutional_first_update.npz')
    best = arrays(directory / 'convolutional_best.npz')
    p = {key: value.astype(np.float64) for key, value in initial.items()}
    x = gradients['input'].astype(np.float64)
    states = forward(x, p)
    reconstruction = states['reconstruction']
    gradient = 2 * (reconstruction-x) * reconstruction * (1-reconstruction) / x.size
    expected = {}
    gradient, expected['decoder2/W'], expected['decoder2/b'] = convolution_adjoint(states['second'], p['decoder2/W'], gradient)
    gradient = 4 * pool(gradient) * (states['decoder1'] > 0)
    gradient, expected['decoder1/W'], expected['decoder1/b'] = convolution_adjoint(states['enlarged'], p['decoder1/W'], gradient)
    gradient = 4 * pool(gradient)
    gradient = repeat(gradient) / 4 * (states['conv2'] > 0)
    gradient, expected['encoder2/W'], expected['encoder2/b'] = convolution_adjoint(states['pool1'], p['encoder2/W'], gradient)
    gradient = repeat(gradient) / 4 * (states['conv1'] > 0)
    _, expected['encoder1/W'], expected['encoder1/b'] = convolution_adjoint(x, p['encoder1/W'], gradient)
    gradient_errors = {}
    for key, value in expected.items():
        np.testing.assert_allclose(value, gradients[key], atol=2e-8, rtol=3e-4)
        gradient_errors[key] = float(np.max(np.abs(value-gradients[key])))
        # Isolate optimizer rounding from the independent gradient calculation.
        g = gradients[key]
        first = g * np.float32(.1)
        second = (g*g) * np.float32(.001)
        direction = (first/np.float32(.1))/(np.sqrt(second/np.float32(.001))+np.float32(1e-8))
        np.testing.assert_array_equal(initial[key]-np.float32(.003)*direction, updated[key])
    p = {key: value.astype(np.float64) for key, value in best.items()}
    test = result['test_images'].astype(np.float64)
    states = forward(test, p)
    np.testing.assert_allclose(states['code'], result['code'], atol=3e-6, rtol=3e-6)
    np.testing.assert_allclose(states['reconstruction'], result['reconstruction'], atol=3e-6, rtol=3e-6)
    probe = forward(result['probe'].astype(np.float64), p)
    stage_errors = {}
    for key, stored in [('conv1','conv1'),('pool1','pool1'),('conv2','conv2'),('code','probe_code'),('decoder1','decoder1'),('reconstruction','probe_reconstruction')]:
        np.testing.assert_allclose(probe[key], result[stored], atol=3e-6, rtol=3e-6)
        stage_errors[key] = float(np.max(np.abs(probe[key]-result[stored])))
    fit = result['fit_images'].astype(np.float64)
    baselines = dict(mean=np.broadcast_to(fit.mean(axis=0), test.shape),
                     pooled=repeat(pool(test,4),4),
                     zero=decoder(np.zeros_like(states['code']),p)['reconstruction'],
                     shuffled=decoder(np.roll(states['code'],-1,axis=0),p)['reconstruction'],
                     initial=forward(test,{key: value.astype(np.float64) for key,value in initial.items()})['reconstruction'])
    for key, value in baselines.items():
        np.testing.assert_allclose(value,result[key],atol=3e-6,rtol=3e-6)
    for key, count in [('fit_ids',1000),('validation_ids',200),('test_ids',200)]:
        ids = result[key]
        assert ids.shape == (count,) and len(np.unique(ids)) == count
        lower, upper = (60000,70000) if key == 'test_ids' else (0,60000)
        assert ((ids>=lower)&(ids<upper)).all()
    for left,right in [('fit_ids','validation_ids'),('fit_ids','test_ids'),('validation_ids','test_ids')]:
        assert not np.intersect1d(result[left],result[right]).size
    assert np.array_equal(np.bincount(result['test_labels'],minlength=10),np.full(10,20))
    return dict(passed=True, parameter_derivatives=sum(v.size for v in expected.values()),
                first_gradient_errors=gradient_errors, Float32_Adam_step_bit_identical=True,
                held_out_pixels=test.size, maximum_prediction_error=float(np.max(np.abs(states['reconstruction']-result['reconstruction']))),
                six_stage_errors=stage_errors,
                test_mse=float(np.mean((states['reconstruction']-test)**2)),
                baseline_mse={k:float(np.mean((v-test)**2)) for k,v in baselines.items()}, disjoint_official_source_ids=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = check(args.directory)
    rendered = json.dumps(result,indent=2)+'\n'
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(rendered)
    print(rendered,end='')


if __name__ == '__main__':
    main()
