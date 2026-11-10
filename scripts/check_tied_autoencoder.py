#!/usr/bin/env python3
"""Optional independent NumPy oracle for the live-transpose autoencoder report.

Reads completed native artifacts, never prepares data or fits. Checks both full
first-batch derivatives, Adam rounding, every validation candidate, all pool/code
controls, exact initial matching, weight tying and the fitting-only pixel mean.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def arrays(path):
    """Read owned numeric arrays with pickle disabled and close their archive."""
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def floating(p):
    """Promote parameter copies for independent Float64 arithmetic."""
    return {key: value.astype(np.float64) for key, value in p.items()}


def decode(p, code, tied):
    """Explicit sigmoid decoder; no image, target or encoder bias enters."""
    weight = p['encoder/W'].T if tied else p['decoder/W']
    return 1 / (1 + np.exp(-(code @ weight + p['decoder/b'])))


def forward(p, x, tied):
    """Independent ReLU encoder and tied or independent sigmoid decoder."""
    code = np.maximum(x.astype(np.float64) @ p['encoder/W'] + p['encoder/b'], 0)
    return code, decode(p, code, tied)


def derivatives(p, x, tied):
    """Complete MSE derivatives, summing both shared-matrix paths exactly once."""
    x = x.astype(np.float64)
    code, prediction = forward(p, x, tied)
    delta = 2 * (prediction - x) * prediction * (1 - prediction) / x.size
    weight = p['encoder/W'].T if tied else p['decoder/W']
    encoder_delta = (delta @ weight.T) * (code > 0)
    encoder = x.T @ encoder_delta
    decoder = code.T @ delta
    result = {'encoder/W': encoder + decoder.T if tied else encoder,
              'encoder/b': encoder_delta.sum(axis=0), 'decoder/b': delta.sum(axis=0)}
    if not tied:
        result['decoder/W'] = decoder
    return result, float(np.linalg.norm(encoder)), float(np.linalg.norm(decoder))


def exact(a, b):
    """Require dtype, shape and every value byte, preserving signed zero."""
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()


def check(directory):
    """Raise on any incomplete or inconsistent report artifact; return evidence."""
    data = arrays(directory / 'tied_data.npz')
    x = {split: data[split + '_images'].reshape(-1, 784) for split in ('fit', 'validation', 'test')}
    census, pixels, codes, records = 0, 0, 0, []
    initial_t = arrays(directory / 'tied_tied_initial.npz')
    initial_u = arrays(directory / 'tied_untied_initial.npz')
    for key in initial_t:
        exact(initial_t[key], initial_u[key])
    exact(initial_t['encoder/W'].T, initial_u['decoder/W'])
    probe = arrays(directory / 'tied_initial_forward.npz')
    exact(probe['tied_code'], probe['untied_code'])
    exact(probe['tied_reconstruction'], probe['untied_reconstruction'])
    for tied in (True, False):
        name = 'tied' if tied else 'untied'
        def load(suffix):
            return arrays(directory / ('tied_' + name + '_' + suffix + '.npz'))
        initial = load('initial')
        assert sum(v.size for v in initial.values()) == (25904 if tied else 50992)
        assert ('decoder/W' in initial) != tied
        batch = load('first_batch')['images'].reshape(-1, 784)
        expected, enc_norm, dec_norm = derivatives(floating(initial), batch, tied)
        assert enc_norm > 0 and dec_norm > 0
        actual, updated = load('first_gradients'), load('first_update')
        errors = {}
        for key, gradient in expected.items():
            np.testing.assert_allclose(gradient, actual[key], atol=5e-9, rtol=5e-4, err_msg=name + '/' + key)
            errors[key] = float(np.max(np.abs(gradient - actual[key])))
            g = actual[key]
            first, second = g * np.float32(.1), (g * g) * np.float32(.001)
            direction = (first / np.float32(.1)) / (np.sqrt(second / np.float32(.001)) + np.float32(1e-8))
            exact(initial[key] - direction * np.float32(.003), updated[key])
            census += gradient.size
        candidates, validation_errors, snapshots = [], {}, []
        for epoch in range(4):
            snapshot = load('epoch' + str(epoch))
            snapshots.append(snapshot)
            _, prediction = forward(floating(snapshot), x['validation'], tied)
            native = load('epoch' + str(epoch) + '_validation')['prediction']
            np.testing.assert_allclose(prediction, native, atol=2e-6, rtol=5e-6)
            validation_errors[str(epoch)] = float(np.max(np.abs(prediction - native)))
            candidates.append(float(np.mean((prediction - x['validation']) ** 2)))
            pixels += prediction.size
        selection = load('selection')
        selected = int(np.argmin(candidates[1:])) + 1  # epoch zero is ineligible
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
            code, prediction = forward(p, x[split], tied)
            np.testing.assert_allclose(code, outputs[split + '_code'], atol=5e-6, rtol=5e-6)
            np.testing.assert_allclose(prediction, outputs[split + '_reconstruction'], atol=2e-6, rtol=5e-6)
            prediction_errors[split] = float(np.max(np.abs(prediction - outputs[split + '_reconstruction'])))
            pixels += prediction.size
            codes += code.size
        code = forward(p, x['test'], tied)[0]
        for control, value in [('zero', np.zeros_like(code)), ('shuffled', np.roll(code, -1, axis=0))]:
            prediction = decode(p, value, tied)
            np.testing.assert_allclose(prediction, outputs[control + '_reconstruction'], atol=2e-6, rtol=5e-6)
            prediction_errors[control] = float(np.max(np.abs(prediction - outputs[control + '_reconstruction'])))
            pixels += prediction.size
        decoder = best['encoder/W'].T if tied else best['decoder/W']
        exact(decoder, load('decoder_weight')['decoder_weight'])
        if not tied:
            assert not np.array_equal(best['decoder/W'], best['encoder/W'].T)
        records.append({'variant': name, 'derivative_max_errors': errors,
                        'encoder_gradient_norm': enc_norm, 'decoder_gradient_norm': dec_norm,
                        'first_Adam_bit_exact': True, 'selected_epoch': selected,
                        'validation_MSEs': candidates, 'validation_prediction_max_errors': validation_errors,
                        'pool_control_prediction_max_errors': prediction_errors})
    gallery = arrays(directory / 'tied_gallery.npz')
    mean = x['fit'].astype(np.float64).mean(axis=0).astype(np.float32)
    exact(np.broadcast_to(mean, x['test'].shape), gallery['mean_reconstruction'])
    return {'passed': True, 'parameter_derivatives_checked': census,
            'validation_checkpoints_checked': 8, 'prediction_pixels_checked': pixels,
            'code_values_checked': codes, 'exact_initial_matching': True,
            'epoch_zero_ineligible_checked': True, 'variants': records}


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
