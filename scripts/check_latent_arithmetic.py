#!/usr/bin/env python3
"""Independent NumPy oracle for frozen native latent-arithmetic artifacts.

Never prepares data or fits a model. Checks fitting-label centroids, original
case selection, Float32 path arithmetic, matched random norms, actual path
derivatives, code-only decoding and both pixel-space controls.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact, gaussian
from check_contractive_autoencoder import sigmoid


def centroids(values, labels):
    """Accumulate each class in row-order double, then cast once to Float32."""
    out = []
    for digit in range(10):
        total = np.zeros(values.shape[1], dtype=np.float64);count = 0
        for value, label in zip(values, labels):
            if label == digit:
                total += value.astype(np.float64);count += 1
        assert count > 0
        out.append((total/count).astype(np.float32))
    return np.stack(out)


def check(directory, upstream=None):
    """Validate every retained analysis endpoint and return measured errors."""
    a, p = arrays(directory/'analysis.npz'), arrays(directory/'model_arrays.npz')
    if upstream:
        original = arrays(upstream)
        for key in p:
            exact(p[key], original[key])
    assert sum(v.size for v in p.values()) == 50992
    identities = set();largest = 0.;forward_values = 0
    for split, count in [('fit', 1000), ('test', 200)]:
        ids, labels = a[split+'_source_ids'], a[split+'_labels']
        assert len(set(ids)) == count and not identities.intersection(ids)
        identities.update(ids)
        assert np.array_equal(np.bincount(labels, minlength=10), np.full(10, count//10))
        lower, upper = (60000, 70000) if split == 'test' else (0, 60000)
        assert ((ids >= lower) & (ids < upper)).all()
        x = a[split+'_images'].reshape(count, 784).astype(np.float64)
        code = np.maximum(x@p['encoder/W'].astype(np.float64)+p['encoder/b'], 0)
        np.testing.assert_allclose(code, a[split+'_code'], atol=3e-6, rtol=3e-6)
        largest = max(largest, float(abs(code-a[split+'_code']).max()));forward_values += code.size
    exact(centroids(a['fit_code'], a['fit_labels']), a['fit_class_code_means'])
    exact(centroids(a['fit_images'].reshape(1000, 784), a['fit_labels']), a['fit_class_pixel_means'])
    rows = np.array([np.flatnonzero(a['test_labels'] == digit)[0] for digit in range(10)], dtype=np.int32)
    targets = np.array([np.flatnonzero(a['test_labels'] == (digit+1)%10)[0] for digit in range(10)], dtype=np.int32)
    exact(rows, a['selected_test_rows']);exact(a['test_source_ids'][rows], a['source_ids'])
    exact(a['test_source_ids'][targets], a['target_ids'])
    exact(a['test_code'][rows], a['a']);exact(a['test_code'][targets], a['b'])
    exact(np.array([j/10 for j in range(11)], dtype=np.float32), a['alpha'])
    exact(np.array([-1, 0, .5, 1, 2], dtype=np.float32), a['levels'])
    means = a['fit_class_code_means']
    direction = np.roll(means, -1, axis=0)-means;exact(direction, a['directions'])
    standard = gaussian((1, 10, 32), 4309)
    np.testing.assert_allclose(standard, a['random_standard'], atol=1e-14, rtol=1e-14)
    random = np.empty_like(direction)
    for row in range(10):
        length = sum(float(v)**2 for v in direction[row])
        norm = sum(float(v)**2 for v in a['random_standard'][0, row])
        for j in range(32):
            random[row, j] = float(a['random_standard'][0, row, j])*math.sqrt(length/norm)
    exact(random, a['random_directions'])
    left, right = a['a'][:, None, :], a['b'][:, None, :]
    t, levels = a['alpha'][None, :, None], a['levels'][None, :, None]
    paths = {'interpolated_codes': (np.float32(1)-t)*left+t*right,
             'reverse_codes': (np.float32(1)-t)*right+t*left,
             'constant_codes': (np.float32(1)-t)*left+t*left,
             'arithmetic_codes': left+levels*direction[:, None, :],
             'random_codes': left+levels*random[:, None, :]}
    for key, value in paths.items():
        exact(value, a[key])
    exact(a['interpolated_codes'][:, 0], a['a']);exact(a['interpolated_codes'][:, -1], a['b'])
    exact(a['arithmetic_codes'][:, 1], a['a'])
    td = a['alpha'].astype(np.float64)
    np.testing.assert_allclose(a['gradient_a'], np.full((10, 32), sum(1-td)), atol=1e-13, rtol=0)
    np.testing.assert_allclose(a['gradient_b'], np.full((10, 32), sum(td)), atol=1e-13, rtol=0)
    delta = (a['b'].astype(np.float64)-a['a'].astype(np.float64)).sum()
    np.testing.assert_allclose(a['gradient_alpha'], np.full(11, delta), atol=1e-12, rtol=0)
    for codes, images in [('interpolated_codes', 'interpolated_images'), ('arithmetic_codes', 'arithmetic_images'),
                          ('random_codes', 'random_images'), ('a', 'endpoint_a'), ('b', 'endpoint_b')]:
        code = a[codes].reshape(-1, 32).astype(np.float64)
        expected = sigmoid(code@p['decoder/W'].astype(np.float64)+p['decoder/b'])
        np.testing.assert_allclose(expected, a[images], atol=1e-6, rtol=3e-6)
        largest = max(largest, float(abs(expected-a[images]).max()));forward_values += expected.size
    pixels = a['test_images'].reshape(200, 784)
    blend = (np.float32(1)-t)*pixels[rows, None, :]+t*pixels[targets, None, :]
    exact(blend.reshape(110, 784), a['pixel_blends'])
    decoded_blend = (np.float32(1)-t)*a['endpoint_a'][:, None, :]+t*a['endpoint_b'][:, None, :]
    exact(decoded_blend.reshape(110, 784), a['decoder_endpoint_blends'])
    means = a['fit_class_pixel_means'];raw = pixels[rows, None, :]+levels*(np.roll(means, -1, axis=0)-means)[:, None, :]
    exact(np.clip(raw, 0, 1).reshape(50, 784), a['pixel_arithmetic_clipped'])
    nonlinearity = float(np.abs(a['interpolated_images'].astype(np.float64)-a['decoder_endpoint_blends']).max())
    reversal = float(np.abs(a['interpolated_codes'].astype(np.float64)-a['reverse_codes'][:, ::-1]).max())
    metrics = json.loads((directory/'metrics.json').read_text())
    assert metrics['additional_training_updates'] == 0 and metrics['parameters'] == 50992
    assert metrics['pairs'] == 10 and metrics['interpolation_steps'] == 11 and metrics['arithmetic_steps'] == 5
    assert nonlinearity == metrics['decoder_nonlinearity_max_difference']
    assert reversal == metrics['reverse_code_max_error']
    assert int((a['arithmetic_codes'] < 0).sum()) == metrics['negative_arithmetic_code_entries']
    assert int(((raw < 0) | (raw > 1)).sum()) == metrics['pixel_arithmetic_clipped_entries']
    return dict(passed=True, additional_training_updates=0, forward_values_checked=forward_values,
                maximum_forward_error=largest, Gaussian_values_checked=320, path_values_exact=sum(v.size for v in paths.values()),
                path_derivatives_checked=651, fitting_centroid_values_exact=8160, selected_pairs=10,
                supervised_posthoc_directions=True, random_norms_matched=True, metrics=metrics,
                upstream_parameters_exact=bool(upstream))


def main():
    """Read native artifacts and optionally save an independent verification receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args();result = check(args.directory, args.upstream)
    text = json.dumps(result, indent=2)+'\n'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True);args.report.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
