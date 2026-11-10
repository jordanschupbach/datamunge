#!/usr/bin/env python3
"""Optional NumPy oracle for the four matched activation-sparse report models.

Never prepares/downloads data or fits. Independently checks all first parameter
derivatives, Adam rounding, all pools, first/final minibatch penalty derivatives,
components, unit means and frozen-code controls.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def arrays(path):
    """Load owned numeric arrays with pickle disabled and close the archive."""
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def constants():
    """Source Float32 rho/complement and clipping endpoints, promoted exactly."""
    rho = np.float32(.05)
    epsilon = np.float32(1e-5)
    return float(rho), float(np.float32(1) - rho), float(epsilon), float(np.float32(1) - epsilon)


def decode(p, code):
    """Independent untied sigmoid decoder."""
    return 1 / (1 + np.exp(-(code @ p['decoder/W'] + p['decoder/b'])))


def forward(p, image):
    """Independent sigmoid encoder and decoder, without activity selection."""
    code = 1 / (1 + np.exp(-(image @ p['encoder/W'] + p['encoder/b'])))
    return code, decode(p, code)


def penalty_derivative(code, unit_mean, alpha, beta):
    """Exact analytic clipping/absolute conventions for the supplied unit means."""
    rho, other, low, high = constants()
    clipped = np.clip(unit_mean, low, high)
    inside = (unit_mean >= low) & (unit_mean <= high)
    return (alpha * np.sign(code) + beta * inside * (-rho / clipped + other / (1 - clipped))) / code.size


def parameters(archive):
    """Own independent Float64 parameter arrays."""
    return {key: value.astype(np.float64) for key, value in archive.items()}


def check(directory):
    """Raise on a mismatch; return counts and measured numerical differences."""
    data = arrays(directory / 'sparse_data.npz')
    records, pixels, probes = [], 0, 0
    initial_reference = None
    for variant, alpha, beta in [('plain', 0., 0.), ('l1', float(np.float32(.05)), 0.),
                                  ('kl', 0., float(np.float32(.05))),
                                  ('combined', float(np.float32(.05)), float(np.float32(.05)))]:
        prefix = directory / ('sparse_' + variant)
        load = lambda suffix: arrays(Path(str(prefix) + '_' + suffix + '.npz'))
        initial = load('initial')
        if initial_reference is None:
            initial_reference = initial
        else:
            for key in initial:
                np.testing.assert_array_equal(initial[key].view(np.uint32), initial_reference[key].view(np.uint32))
        batch = load('first_batch')['images'].reshape(-1, 784).astype(np.float64)
        p = parameters(initial)
        code, prediction = forward(p, batch)
        delta = 2 * (prediction - batch) * prediction * (1 - prediction) / batch.size
        activity = penalty_derivative(code, code.mean(axis=0), alpha, beta)
        encoder_delta = (delta @ p['decoder/W'].T + activity) * code * (1 - code)
        expected = {'encoder/W': batch.T @ encoder_delta, 'encoder/b': encoder_delta.sum(axis=0),
                    'decoder/W': code.T @ delta, 'decoder/b': delta.sum(axis=0)}
        actual, updated = load('first_gradients'), load('first_update')
        gradient_errors = {}
        for key, gradient in expected.items():
            np.testing.assert_allclose(gradient, actual[key], atol=2e-8, rtol=3e-4, err_msg=variant + '/' + key)
            gradient_errors[key] = float(np.max(np.abs(gradient - actual[key])))
            g = actual[key]
            first, second = g * np.float32(.1), (g * g) * np.float32(.001)
            direction = (first / np.float32(.1)) / (np.sqrt(second / np.float32(.001)) + np.float32(1e-8))
            np.testing.assert_array_equal((initial[key] - direction * np.float32(.003)).view(np.uint32), updated[key].view(np.uint32))
        probe_error = 0.
        for name in ('first_probe', 'partial_probe'):
            probe = load(name)
            rows = 32 if name == 'first_probe' else 8
            assert probe['code'].shape == (rows, 64)
            means = probe['code'].astype(np.float64).mean(axis=0)
            np.testing.assert_allclose(means, probe['unit_mean'], atol=2e-7, rtol=1e-6)
            rho, other, low, high = constants()
            np.testing.assert_array_equal(np.clip(probe['unit_mean'], np.float32(low), np.float32(high)), probe['clipped_mean'])
            expected_activity = penalty_derivative(probe['code'].astype(np.float64), probe['unit_mean'].astype(np.float64), alpha, beta)
            np.testing.assert_allclose(expected_activity, probe['penalty_gradient'], atol=2e-8, rtol=3e-5)
            probe_error = max(probe_error, float(np.max(np.abs(expected_activity - probe['penalty_gradient']))))
            probes += expected_activity.size
        p = parameters(load('best'))
        outputs, controls = load('outputs'), load('controls')
        errors = {}
        for split in ('fit', 'validation', 'test'):
            image = data[split + '_images'].reshape(-1, 784).astype(np.float64)
            code, prediction = forward(p, image)
            np.testing.assert_allclose(code, outputs[split + '_code'], atol=2e-6, rtol=3e-6)
            np.testing.assert_allclose(prediction, outputs[split + '_reconstruction'], atol=1e-6, rtol=3e-6)
            errors[split + '_prediction'] = float(np.max(np.abs(prediction - outputs[split + '_reconstruction'])))
            means = outputs[split + '_code'].astype(np.float64).mean(axis=0)
            np.testing.assert_allclose(means, outputs[split + '_unit_mean'], atol=2e-6, rtol=3e-6)
            # Components use native rounded batch means, independently validated above.
            rho, other, low, high = constants()
            c = np.clip(outputs[split + '_unit_mean'].astype(np.float64), low, high)
            kl = (rho * np.log(rho / c) + other * np.log(other / (1 - c))).mean()
            l1 = np.abs(outputs[split + '_code'].astype(np.float64)).mean()
            mse = ((outputs[split + '_reconstruction'].astype(np.float64) - image) ** 2).mean()
            components = [mse + alpha * l1 + beta * kl, mse, l1, kl]
            np.testing.assert_allclose(components, outputs[split + '_components'], atol=2e-6, rtol=5e-6)
            pixels += prediction.size
        for name, control in [('zero_reconstruction', np.zeros_like(code)),
                              ('shifted_reconstruction', np.roll(code, -1, axis=0))]:
            prediction = decode(p, control)
            np.testing.assert_allclose(prediction, controls[name], atol=1e-6, rtol=3e-6)
            errors[name] = float(np.max(np.abs(prediction - controls[name])))
            pixels += prediction.size
        records.append({'variant': variant, 'parameter_derivatives': sum(x.size for x in expected.values()),
                        'gradient_max_errors': gradient_errors, 'first_Adam_bit_exact': True,
                        'penalty_probe_max_error': probe_error, 'prediction_max_errors': errors})
    for name, count in [('fit', 1000), ('validation', 200), ('test', 200)]:
        ids = data[name + '_source_ids']
        assert ids.shape == (count,) and len(np.unique(ids)) == count
    assert not np.intersect1d(data['fit_source_ids'], data['validation_source_ids']).size
    assert ((data['test_source_ids'] >= 60000) & (data['test_source_ids'] < 70000)).all()
    return {'passed': True, 'matched_initial_parameters': True, 'parameter_derivatives': 404800,
            'penalty_probe_derivatives': probes, 'prediction_pixels': pixels, 'variants': records}


def main():
    """Run the optional check against a completed report's local build directory."""
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
