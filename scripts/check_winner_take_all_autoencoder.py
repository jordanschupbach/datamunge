#!/usr/bin/env python3
"""Optional independent NumPy oracle for lifetime winner-take-all report artifacts.

Never prepares data or fits. Checks stable column-wise masks, ReLU/MSE first
parameter derivatives, Adam rounding, full-code validation selection, every
pool/control prediction and the original disjoint MNIST record identities.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def arrays(path):
    """Read owning numeric arrays and close their archive, with pickle disabled."""
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def floating(p):
    """Promote independent parameter copies for Float64 arithmetic."""
    return {key: value.astype(np.float64) for key, value in p.items()}


def selector(raw, plain=False):
    """Stable descending rows per column; exactly ceil(B/8) slots per feature."""
    if plain:
        return np.ones_like(raw)
    result = np.zeros_like(raw)
    winners = np.argsort(-raw, axis=0, kind='stable')[:(len(raw) + 7) // 8]
    np.put_along_axis(result, winners, 1, axis=0)
    return result


def decode(p, code):
    """Independent sigmoid decoder of explicit codes, with no image or gate."""
    return 1 / (1 + np.exp(-(code @ p['decoder/W'] + p['decoder/b'])))


def forward(p, x, plain=False):
    """ReLU activities, training selector/code/prediction and full deployment."""
    raw = np.maximum(x.astype(np.float64) @ p['encoder/W'] + p['encoder/b'], 0)
    mask = selector(raw, plain)
    code = raw * mask
    return raw, mask, code, decode(p, code), decode(p, raw)


def exact(a, b):
    """Require identical shape, dtype and all value bytes, preserving signed zero."""
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()


def check(directory):
    """Validate full report artifacts or raise; return counts and numerical errors."""
    data = arrays(directory / 'wta_data.npz')
    x = {s: data[s + '_images'].reshape(-1, 784) for s in ('fit', 'validation', 'test')}
    records, pixels, derivatives, codes = [], 0, 0, 0
    for variant in ('competitive', 'plain'):
        plain = variant == 'plain'
        def load(name):
            return arrays(directory / ('wta_' + variant + '_' + name + '.npz'))
        initial, updated, actual = load('initial'), load('first_update'), load('first_gradients')
        batch = load('first_batch')['images'].reshape(-1, 784).astype(np.float64)
        p = floating(initial)
        raw, mask, code, prediction, _ = forward(p, batch, plain)
        saved = load('first_probe')
        np.testing.assert_allclose(raw, saved['raw'], atol=3e-6, rtol=5e-6)
        np.testing.assert_array_equal(mask, saved['mask'])
        delta = 2 * (prediction - batch) * prediction * (1 - prediction) / batch.size
        back = (delta @ p['decoder/W'].T) * mask * (raw > 0)
        expected = {'encoder/W': batch.T @ back, 'encoder/b': back.sum(axis=0),
                    'decoder/W': code.T @ delta, 'decoder/b': delta.sum(axis=0)}
        errors = {}
        for key, gradient in expected.items():
            np.testing.assert_allclose(gradient, actual[key], atol=5e-9, rtol=5e-4, err_msg=variant + '/' + key)
            errors[key] = float(np.max(np.abs(gradient - actual[key])))
            g = actual[key]
            first, second = g * np.float32(.1), (g * g) * np.float32(.001)
            direction = (first / np.float32(.1)) / (np.sqrt(second / np.float32(.001)) + np.float32(1e-8))
            exact(initial[key] - direction * np.float32(.003), updated[key])
            derivatives += gradient.size
        candidates, snapshots, validation_errors = [], [], {}
        for epoch in range(4):
            snapshot = load('epoch' + str(epoch))
            snapshots.append(snapshot)
            deployment = forward(floating(snapshot), x['validation'], plain)[-1]
            native = load('epoch' + str(epoch) + '_validation')['prediction']
            np.testing.assert_allclose(deployment, native, atol=2e-6, rtol=5e-6)
            validation_errors[str(epoch)] = float(np.max(np.abs(deployment - native)))
            candidates.append(float(np.mean((deployment - x['validation']) ** 2)))
            pixels += deployment.size
        selection = load('selection')
        selected = int(np.argmin(candidates[1:])) + 1  # completed epochs only
        assert selected == int(selection['epoch'])
        np.testing.assert_allclose(candidates[1:], selection['validation_path'], atol=5e-8, rtol=5e-7)
        best, model = load('best'), load('model_arrays')
        for key in best:
            exact(best[key], snapshots[selected][key])
            exact(best[key], model[key])
            exact(initial[key], snapshots[0][key])
        p, outputs = floating(best), load('outputs')
        prediction_errors = {}
        for split in x:
            raw, _, _, _, deployment = forward(p, x[split], plain)
            np.testing.assert_allclose(raw, outputs[split + '_code'], atol=4e-6, rtol=5e-6)
            np.testing.assert_allclose(deployment, outputs[split + '_reconstruction'], atol=2e-6, rtol=5e-6)
            prediction_errors[split] = float(np.max(np.abs(deployment - outputs[split + '_reconstruction'])))
            pixels += deployment.size
            codes += raw.size
        test_code = forward(p, x['test'], plain)[0]
        for name, value in [('zero', np.zeros_like(test_code)), ('shuffled', np.roll(test_code, -1, axis=0))]:
            prediction = decode(p, value)
            np.testing.assert_allclose(prediction, outputs[name + '_reconstruction'], atol=2e-6, rtol=5e-6)
            prediction_errors[name] = float(np.max(np.abs(prediction - outputs[name + '_reconstruction'])))
            pixels += prediction.size
        initial_output = load('initial_outputs')
        initial_raw, _, _, _, initial_prediction = forward(floating(initial), x['test'], plain)
        np.testing.assert_allclose(initial_raw, initial_output['code'], atol=4e-6, rtol=5e-6)
        np.testing.assert_allclose(initial_prediction, initial_output['reconstruction'], atol=2e-6, rtol=5e-6)
        pixels += initial_prediction.size
        probe_errors = {}
        for name, weights in [('first_probe', initial), ('partial_probe', load('partial_arrays')), ('selected_probe', best)]:
            probe = load(name)
            image = batch if name == 'first_probe' else probe['images'].reshape(-1, 784)
            a, m, t, prediction, _ = forward(floating(weights), image, plain)
            np.testing.assert_allclose(a, probe['raw'], atol=4e-6, rtol=5e-6)
            # Saved Float32 ranks must match exactly, including zero ties.
            np.testing.assert_array_equal(selector(probe['raw'], plain), probe['mask'])
            np.testing.assert_array_equal(m, probe['mask'])
            exact(probe['raw'] * probe['mask'], probe['training_code'])
            np.testing.assert_allclose(t, probe['training_code'], atol=4e-6, rtol=5e-6)
            np.testing.assert_allclose(prediction, probe['reconstruction'], atol=2e-6, rtol=5e-6)
            keep = len(a) if plain else (len(a) + 7) // 8
            assert np.array_equal(probe['mask'].sum(axis=0), np.full(64, keep))
            if name == 'partial_probe':
                assert a.shape == (8, 64)
            probe_errors[name] = float(np.max(np.abs(prediction - probe['reconstruction'])))
            pixels += prediction.size
        records.append({'variant': variant, 'first_gradient_max_errors': errors,
                        'first_Adam_bit_exact': True, 'selected_epoch': selected,
                        'validation_MSEs': candidates, 'validation_prediction_max_errors': validation_errors,
                        'prediction_max_errors': prediction_errors, 'training_probe_max_errors': probe_errors,
                        'deployed_test_nonzero_fraction': float(np.mean(outputs['test_code'] != 0))})
    left, right = arrays(directory / 'wta_competitive_initial.npz'), arrays(directory / 'wta_plain_initial.npz')
    for key in left:
        exact(left[key], right[key])
    a, b = arrays(directory / 'wta_competitive_initial_outputs.npz'), arrays(directory / 'wta_plain_initial_outputs.npz')
    for key in a:
        exact(a[key], b[key])
    for split in ('fit', 'validation', 'test'):
        ids = data[split + '_source_ids']
        lower, upper = (60000, 70000) if split == 'test' else (0, 60000)
        assert len(np.unique(ids)) == len(x[split]) and np.all((ids >= lower) & (ids < upper))
    for a, b in [('fit', 'validation'), ('fit', 'test'), ('validation', 'test')]:
        assert not np.intersect1d(data[a + '_source_ids'], data[b + '_source_ids']).size
    assert np.array_equal(np.bincount(data['test_labels'], minlength=10), np.full(10, 20))
    mean = x['fit'].astype(np.float64).mean(axis=0).astype(np.float32)
    exact(np.broadcast_to(mean, x['test'].shape), arrays(directory / 'wta_mean.npz')['mean'])
    return {'passed': True, 'parameter_derivatives_checked': derivatives,
            'validation_checkpoints_checked': 8, 'prediction_pixels_checked': pixels,
            'deployed_code_values_checked': codes, 'matched_initial_parameters_and_deployment': True,
            'epoch_zero_ineligible_checked': True, 'disjoint_official_source_ids': True, 'variants': records}


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
