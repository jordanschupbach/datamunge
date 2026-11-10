#!/usr/bin/env python3
"""Optional independent NumPy oracle for the recurrent autoencoder report.

Never prepares/downloads data or fits. Implements full tanh BPTT for all first
parameter derivatives and held-fixed-target input derivatives, checks Adam
rounding, complete state histories, every pool and frozen-code controls.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def arrays(path):
    """Load owned numeric arrays with pickle disabled, closing the archive."""
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def decode(p, code, steps=28):
    """Repeat a code into an independent zero-state tanh cell and row head."""
    state = np.zeros_like(code)
    history, outputs = [], []
    for _ in range(steps):
        state = np.tanh(code @ p['decoder/Wx'] + state @ p['decoder/Wh'] + p['decoder/b'])
        history.append(state)
        outputs.append(1 / (1 + np.exp(-(state @ p['output/W'] + p['output/b']))))
    return np.stack(outputs, axis=1), np.stack(history, axis=1)


def forward(p, images):
    """Ordered row encoder; only its last state enters decode."""
    rows = images.reshape(len(images), 28, 28).astype(np.float64)
    state = np.zeros((len(rows), 32))
    history = []
    for t in range(28):
        state = np.tanh(rows[:, t] @ p['encoder/Wx'] + state @ p['encoder/Wh'] + p['encoder/b'])
        history.append(state)
    prediction, decoder_states = decode(p, state)
    return state, prediction, np.stack(history, axis=1), decoder_states


def derivatives(p, images):
    """Analytic full BPTT, sharing gradients over all encoder and decoder steps."""
    target = images.reshape(len(images), 28, 28).astype(np.float64)
    code, prediction, encoder, decoder = forward(p, images)
    gradients = {key: np.zeros_like(value) for key, value in p.items()}
    output_delta = 2 * (prediction - target) * prediction * (1 - prediction) / target.size
    code_gradient = np.zeros_like(code)
    future = np.zeros_like(code)
    for t in range(27, -1, -1):
        state = decoder[:, t]
        previous = decoder[:, t - 1] if t else np.zeros_like(code)
        local = output_delta[:, t]
        gradients['output/W'] += state.T @ local
        gradients['output/b'] += local.sum(axis=0)
        delta = (local @ p['output/W'].T + future) * (1 - state * state)
        gradients['decoder/Wx'] += code.T @ delta
        gradients['decoder/Wh'] += previous.T @ delta
        gradients['decoder/b'] += delta.sum(axis=0)
        code_gradient += delta @ p['decoder/Wx'].T
        future = delta @ p['decoder/Wh'].T
    input_gradient = np.zeros_like(target)
    future = code_gradient
    for t in range(27, -1, -1):
        state = encoder[:, t]
        previous = encoder[:, t - 1] if t else np.zeros_like(code)
        delta = future * (1 - state * state)
        gradients['encoder/Wx'] += target[:, t].T @ delta
        gradients['encoder/Wh'] += previous.T @ delta
        gradients['encoder/b'] += delta.sum(axis=0)
        input_gradient[:, t] = delta @ p['encoder/Wx'].T
        future = delta @ p['encoder/Wh'].T
    return gradients, input_gradient


def floating(archive):
    """Promote stored Float32 parameters for independent Float64 calculations."""
    return {key: value.astype(np.float64) for key, value in archive.items()}


def check(directory):
    """Validate all complete report artifacts or raise on mismatch."""
    data = arrays(directory / 'recurrent_data.npz')
    initial = arrays(directory / 'recurrent_initial.npz')
    batch = arrays(directory / 'recurrent_first_batch.npz')['images']
    saved_gradients = arrays(directory / 'recurrent_first_gradients.npz')
    first_update = arrays(directory / 'recurrent_first_update.npz')
    expected, _ = derivatives(floating(initial), batch)
    gradient_errors = {}
    assert len(initial) == 8 and sum(x.size for x in initial.values()) == 4956
    for key, gradient in expected.items():
        np.testing.assert_allclose(gradient, saved_gradients[key], atol=2e-8, rtol=5e-4, err_msg=key)
        gradient_errors[key] = float(np.max(np.abs(gradient - saved_gradients[key])))
        g = saved_gradients[key]
        first, second = g * np.float32(.1), (g * g) * np.float32(.001)
        direction = (first / np.float32(.1)) / (np.sqrt(second / np.float32(.001)) + np.float32(1e-8))
        np.testing.assert_array_equal((initial[key] - direction * np.float32(.003)).view(np.uint32), first_update[key].view(np.uint32))
    p = floating(arrays(directory / 'recurrent_best.npz'))
    outputs = arrays(directory / 'recurrent_outputs.npz')
    errors, pixels = {}, 0
    for split in ('fit', 'validation', 'test'):
        code, prediction, _, _ = forward(p, data[split + '_images'])
        for name, value in (('code', code), ('reconstruction', prediction)):
            np.testing.assert_allclose(value, outputs[split + '_' + name], atol=2e-6, rtol=5e-6)
            errors[split + '_' + name] = float(np.max(np.abs(value - outputs[split + '_' + name])))
        pixels += prediction.size
    initial_outputs = arrays(directory / 'recurrent_initial_outputs.npz')
    code, prediction, _, _ = forward(floating(initial), data['test_images'])
    np.testing.assert_allclose(code, initial_outputs['code'], atol=2e-6, rtol=5e-6)
    np.testing.assert_allclose(prediction, initial_outputs['reconstruction'], atol=8e-7, rtol=5e-6)
    errors['initial_code'] = float(np.max(np.abs(code - initial_outputs['code'])))
    errors['initial_reconstruction'] = float(np.max(np.abs(prediction - initial_outputs['reconstruction'])))
    pixels += prediction.size
    code, _, _, _ = forward(p, data['test_images'])
    controls = {
        'zero': decode(p, np.zeros_like(code))[0],
        'shuffled': decode(p, np.roll(code, -1, axis=0))[0],
        'reversed': forward(p, data['test_images'][:, ::-1])[1],
    }
    for name, prediction in controls.items():
        np.testing.assert_allclose(prediction, outputs[name + '_reconstruction'], atol=2e-6, rtol=5e-6)
        errors[name] = float(np.max(np.abs(prediction - outputs[name + '_reconstruction'])))
        pixels += prediction.size
    states = arrays(directory / 'recurrent_states.npz')
    _, _, encoder, decoder = forward(p, states['images'])
    np.testing.assert_allclose(encoder, states['encoder_states'], atol=2e-6, rtol=5e-6)
    np.testing.assert_allclose(decoder, states['decoder_states'], atol=2e-6, rtol=5e-6)
    input_probe = arrays(directory / 'recurrent_input_gradients.npz')
    _, input_gradient = derivatives(p, input_probe['images'])
    np.testing.assert_allclose(input_gradient, input_probe['gradient'], atol=1e-13, rtol=1e-3)
    row_rms = np.sqrt((input_gradient * input_gradient).mean(axis=(0, 2)))
    np.testing.assert_allclose(row_rms, input_probe['row_rms'], atol=1e-13, rtol=1e-4)
    initial_metrics = {}
    for name, prediction in [('model', outputs['test_reconstruction']),
                              ('fitting_mean', arrays(directory / 'recurrent_mean.npz')['mean'].reshape(200, 28, 28))]:
        initial_metrics[name] = float(((prediction.astype(np.float64) - data['test_images'].reshape(200, 28, 28)) ** 2).mean())
    return {'parameter_derivatives_checked': 4956, 'gradient_max_errors': gradient_errors,
            'first_adam_updates_bit_exact': True, 'prediction_pixels_checked': pixels,
            'forward_max_errors': errors, 'state_history_values_checked': encoder.size + decoder.size,
            'fixed_target_input_gradients_checked': input_gradient.size,
            'input_gradient_max_error': float(np.max(np.abs(input_gradient - input_probe['gradient']))),
            'test_MSE': initial_metrics}


def main():
    """Run the optional independent check for a completed local report directory."""
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
