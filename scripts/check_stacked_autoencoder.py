#!/usr/bin/env python3
"""Optional independent NumPy oracle for the staged sigmoid autoencoder report.

Never prepares/downloads data or fits. Verifies all first parameter derivatives,
Adam rounding, sixteen validation checkpoints, exact transfer, frozen cached
representations, held-out layer compositions, code controls and the RMSE bound.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def arrays(path):
    """Read owned numeric arrays with pickle disabled and close the archive."""
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def floating(archive):
    """Promote independent parameter copies to Float64."""
    return {key: value.astype(np.float64) for key, value in archive.items()}


def forward(p, image, start=0):
    """Independent dense sigmoid layers, optionally beginning at a decoder layer."""
    count = len(p) // 2
    value = image.astype(np.float64)
    outputs = []
    for layer in range(start, count):
        value = 1 / (1 + np.exp(-(value @ p[f'layer{layer}/W'] + p[f'layer{layer}/b'])))
        outputs.append(value)
    return outputs


def derivatives(p, image):
    """Full analytic chain rule for the actual mean squared reconstruction loss."""
    image = image.astype(np.float64)
    activations = [image] + forward(p, image)
    result = {}
    prediction = activations[-1]
    delta = 2 * (prediction - image) * prediction * (1 - prediction) / image.size
    for layer in range(len(activations) - 2, -1, -1):
        result[f'layer{layer}/W'] = activations[layer].T @ delta
        result[f'layer{layer}/b'] = delta.sum(axis=0)
        previous = activations[layer]
        delta = (delta @ p[f'layer{layer}/W'].T) * previous * (1 - previous)
    return result


def exact(source, target):
    """Compare dtype, shape and every value byte, including signed zeros."""
    assert source.shape == target.shape and source.dtype == target.dtype
    assert source.tobytes() == target.tobytes()


def check(directory):
    """Validate complete artifacts or raise; return counts and measured errors."""
    data = arrays(directory / 'stacked_data.npz')
    cached = arrays(directory / 'stacked_cache.npz')
    saved = arrays(directory / 'stacked_outputs.npz')
    transferred = arrays(directory / 'stacked_transferred.npz')
    phases, derivative_count, checked_pixels = [], 0, 0
    chosen = {}
    for phase in ('stage1', 'stage2', 'finetune', 'scratch'):
        def load(name):
            return arrays(directory / ('stacked_' + phase + '_' + name + '.npz'))
        initial = load('initial')
        batch = load('first_batch')['images']
        expected = derivatives(floating(initial), batch)
        actual = load('first_gradients')
        updated = load('first_update')
        errors = {}
        for key, gradient in expected.items():
            np.testing.assert_allclose(gradient, actual[key], atol=5e-9, rtol=5e-4, err_msg=phase + '/' + key)
            errors[key] = float(np.max(np.abs(gradient - actual[key])))
            g = actual[key]
            first, second = g * np.float32(.1), (g * g) * np.float32(.001)
            direction = (first / np.float32(.1)) / (np.sqrt(second / np.float32(.001)) + np.float32(1e-8))
            np.testing.assert_array_equal((initial[key] - direction * np.float32(.003)).view(np.uint32), updated[key].view(np.uint32))
            derivative_count += gradient.size
        validation = cached['validation_code'] if phase == 'stage2' else data['validation_images']
        metrics = []
        snapshots = []
        validation_errors = {}
        for epoch in range(4):
            snapshot = load('epoch' + str(epoch))
            snapshots.append(snapshot)
            prediction = forward(floating(snapshot), validation)[-1]
            native = load('epoch' + str(epoch) + '_validation')['prediction']
            np.testing.assert_allclose(prediction, native, atol=1e-6, rtol=5e-6)
            validation_errors[str(epoch)] = float(np.max(np.abs(prediction - native)))
            metrics.append(float(((prediction - validation) ** 2).mean()))
            checked_pixels += prediction.size
        selection = load('selection')
        np.testing.assert_allclose(metrics, selection['validation_path'], atol=2e-8, rtol=5e-7)
        first_candidate = 0 if phase in ('finetune', 'scratch') else 1
        selected = int(np.argmin(metrics[first_candidate:])) + first_candidate
        assert selected == int(selection['epoch'])
        best, model = load('best'), load('model_arrays')
        for key in best:
            exact(best[key], snapshots[selected][key])
            exact(model[key], best[key])
            exact(initial[key], snapshots[0][key])
        chosen[phase] = best
        phases.append({'phase': phase, 'first_gradient_max_errors': errors,
                       'first_Adam_bit_exact': True, 'selected_epoch': selected,
                       'validation_MSEs': metrics, 'validation_prediction_max_errors': validation_errors})
    p1 = floating(chosen['stage1'])
    cache_errors = {}
    for split in ('fit', 'validation', 'test'):
        expected = forward(p1, data[split + '_images'])[0]
        np.testing.assert_allclose(expected, cached[split + '_code'], atol=2e-6, rtol=5e-6)
        cache_errors[split] = float(np.max(np.abs(expected - cached[split + '_code'])))
        exact(cached[split + '_code'], saved['stage1_' + split + '_code'])
    for target, source, phase in [('layer0', 'layer0', 'stage1'), ('layer3', 'layer1', 'stage1'),
                                   ('layer1', 'layer0', 'stage2'), ('layer2', 'layer1', 'stage2')]:
        for suffix in ('/W', '/b'):
            exact(transferred[target + suffix], chosen[phase][source + suffix])
    fine_initial = arrays(directory / 'stacked_finetune_initial.npz')
    for key in transferred:
        exact(transferred[key], fine_initial[key])
    test = data['test_images']
    predictions = {
        'stage1': forward(p1, test),
        'pretrained': forward(floating(transferred), test),
        'finetuned': forward(floating(chosen['finetune']), test),
        'scratch': forward(floating(chosen['scratch']), test),
    }
    prediction_errors = {}
    for name, layers in predictions.items():
        if name != 'stage1':
            np.testing.assert_allclose(layers[1], saved[name + '_test_code'], atol=2e-6, rtol=5e-6)
        prediction = layers[-1]
        np.testing.assert_allclose(prediction, saved[name + '_test_reconstruction'], atol=1e-6, rtol=5e-6)
        prediction_errors[name] = float(np.max(np.abs(prediction - saved[name + '_test_reconstruction'])))
        checked_pixels += prediction.size
    p = floating(chosen['finetune'])
    code = predictions['finetuned'][1]
    for name, control in [('zero', np.zeros_like(code)), ('shuffled', np.roll(code, -1, axis=0))]:
        prediction = forward(p, control, start=2)[-1]
        np.testing.assert_allclose(prediction, saved[name + '_reconstruction'], atol=1e-6, rtol=5e-6)
        prediction_errors[name] = float(np.max(np.abs(prediction - saved[name + '_reconstruction'])))
        checked_pixels += prediction.size
    second = forward(floating(chosen['stage2']), cached['test_code'])[-1]
    independent_bound = arrays(directory / 'stacked_bound.npz')
    np.testing.assert_allclose(second, independent_bound['stage2_test_reconstruction'], atol=1e-6, rtol=5e-6)
    rmse1 = np.sqrt(((predictions['stage1'][-1] - test) ** 2).mean())
    rmse2 = np.sqrt(((second - cached['test_code']) ** 2).mean())
    norm = np.linalg.norm(p1['layer1/W'])
    bound = rmse1 + .25 * norm * np.sqrt(64 / 784) * rmse2
    actual_rmse = np.sqrt(((predictions['pretrained'][-1] - test) ** 2).mean())
    assert actual_rmse <= bound
    np.testing.assert_allclose(norm, independent_bound['decoder_frobenius'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(bound, independent_bound['bound_rmse'], atol=2e-7, rtol=0)
    np.testing.assert_allclose(actual_rmse, independent_bound['actual_rmse'], atol=2e-7, rtol=0)
    mean = data['fit_images'].astype(np.float64).mean(axis=0).astype(np.float32)
    exact(np.broadcast_to(mean, test.shape), saved['mean_reconstruction'])
    return {'passed': True, 'parameter_derivatives_checked': derivative_count,
            'validation_checkpoints_checked': 16, 'prediction_pixels_checked': checked_pixels,
            'all_eight_transfers_byte_exact': True, 'epoch_zero_policy_checked': True,
            'frozen_cache_max_errors': cache_errors, 'heldout_prediction_max_errors': prediction_errors,
            'phases': phases, 'RMSE_bound': {'actual': float(actual_rmse), 'bound': float(bound),
                                          'first_decoder_frobenius': float(norm)}}


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
