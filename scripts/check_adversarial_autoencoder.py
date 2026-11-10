#!/usr/bin/env python3
"""Optional independent NumPy oracle for native adversarial-autoencoder artifacts.

Never fits or prepares data. Checks complete initial/actual phase derivatives,
four separate first Adam states, phase isolation, every validation candidate,
all pool/prior predictions, explicit Gaussian banks and the binary factor lattice.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def arrays(path):
    """Read owned numeric arrays with pickle disabled and close their archive."""
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def exact(a, b):
    """Require equal shape, dtype and every value byte including signed zero."""
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()


def forward(p, x, prior):
    """Independent tanh/linear E,G,D forward values and stable summed NLL."""
    x, prior = x.astype(np.float64), prior.astype(np.float64)
    eh = np.tanh(x @ p['0/hidden/W'] + p['0/hidden/b'])
    z = eh @ p['0/code/W'] + p['0/code/b']
    gh = np.tanh(z @ p['1/hidden/W'] + p['1/hidden/b'])
    logits = gh @ p['1/output/W'] + p['1/output/b']
    probability = 1 / (1 + np.exp(-logits))
    rh = np.tanh(prior @ p['2/hidden/W'] + p['2/hidden/b'])
    fh = np.tanh(z @ p['2/hidden/W'] + p['2/hidden/b'])
    real = rh @ p['2/logit/W'] + p['2/logit/b']
    fake = fh @ p['2/logit/W'] + p['2/logit/b']
    nll = (np.logaddexp(0, logits) - x * logits).sum(axis=1)
    result = dict(code=z, logits=logits, probability=probability, prior_logit=real,
                  encoded_logit=fake, nll=nll, reconstruction_loss=nll.mean(),
                  discriminator_loss=np.logaddexp(0, -real).mean() + np.logaddexp(0, fake).mean(),
                  adversarial_loss=np.logaddexp(0, -fake).mean())
    return result, (eh, gh, rh, fh)


def derivatives(p, x, prior):
    """Full analytic chain rules, preserving the two encoder objectives separately."""
    x, prior = x.astype(np.float64), prior.astype(np.float64)
    out, (eh, gh, rh, fh) = forward(p, x, prior)
    z = out['code']
    delta = (out['probability'] - x) / len(x)
    gd = {'1/output/W': gh.T @ delta, '1/output/b': delta.sum(axis=0)}
    back = (delta @ p['1/output/W'].T) * (1 - gh * gh)
    gd.update({'1/hidden/W': z.T @ back, '1/hidden/b': back.sum(axis=0)})
    dz = back @ p['1/hidden/W'].T
    def encoder(delta_z):
        dh = (delta_z @ p['0/code/W'].T) * (1 - eh * eh)
        return {'0/code/W': eh.T @ delta_z, '0/code/b': delta_z.sum(axis=0),
                '0/hidden/W': x.T @ dh, '0/hidden/b': dh.sum(axis=0)}
    ge = encoder(dz)
    dr = (1 / (1 + np.exp(-out['prior_logit'])) - 1) / len(prior)
    df = (1 / (1 + np.exp(-out['encoded_logit']))) / len(x)
    br = (dr @ p['2/logit/W'].T) * (1 - rh * rh)
    bf = (df @ p['2/logit/W'].T) * (1 - fh * fh)
    dd = {'2/logit/W': rh.T @ dr + fh.T @ df, '2/logit/b': dr.sum(axis=0) + df.sum(axis=0),
          '2/hidden/W': prior.T @ br + z.T @ bf, '2/hidden/b': br.sum(axis=0) + bf.sum(axis=0)}
    da = (1 / (1 + np.exp(-out['encoded_logit'])) - 1) / len(x)
    ba = (da @ p['2/logit/W'].T) * (1 - fh * fh)
    return [ge, gd, dd, encoder(ba @ p['2/hidden/W'].T)]


def floating(p):
    """Promote parameter copies for the independent calculations."""
    return {key: value.astype(np.float64) for key, value in p.items()}


def gaussian(shape, seed):
    """Independent MT19937 raw words and Box-Muller transform in Float64."""
    count = int(np.prod(shape))
    words = np.random.RandomState(seed).randint(0, 2**32, size=((count + 1) // 2, 2), dtype=np.uint32)
    u, v = (words.astype(np.float64).T + .5) / 2**32
    radius = np.sqrt(-2 * np.log(u))
    out = np.column_stack([radius * np.cos(2 * np.pi * v), radius * np.sin(2 * np.pi * v)])
    return out.ravel()[:count].reshape(shape)


def check(directory):
    """Raise on incomplete or inconsistent artifacts and return detailed evidence."""
    def load(name):
        return arrays(directory / ('aae_' + name + '.npz'))
    data, noise = load('data'), load('fixed_noise')
    all_ids, pixel_count, gaussian_count = set(), 0, 0
    for split, count in [('fit', 210), ('validation', 42), ('test', 42)]:
        ids, factors, images = data[split + '_source_ids'], data[split + '_factors'], data[split + '_images']
        assert len(ids) == count and not all_ids.intersection(ids)
        all_ids.update(ids)
        for row, identity in enumerate(ids):
            y, x, size, shape = int(identity) % 7, (int(identity) // 7) % 7, (int(identity) // 49) % 3, int(identity) // 147
            np.testing.assert_array_equal(factors[row], [shape, size, x, y])
            residue = (x + 2 * y + size + 3 * shape) % 7
            assert (residue == 0 if split == 'test' else residue == 1 if split == 'validation' else residue >= 2)
            yy, xx = np.mgrid[:28, :28]
            dx, dy = abs(xx - (3 * x + 5)), abs(yy - (3 * y + 5))
            expected = np.maximum(dx, dy) <= size + 2 if shape else np.hypot(dx, dy) <= size + 2
            exact(expected.astype(np.float32)[..., None], images[row])
            pixel_count += 784
    assert len(all_ids) == 294
    for split, count, seed in [('fit', 210, 3541), ('validation', 42, 3547), ('test', 42, 3553), ('generation', 20, 3571)]:
        expected = gaussian((1, count, 4), seed)
        np.testing.assert_allclose(expected, noise[split + '_standard'], atol=1e-14, rtol=1e-14)
        exact(expected.astype(np.float32), noise[split + '_epsilon'])
        gaussian_count += expected.size
    for epoch in range(1, 4):
        bank = load('train_noise_epoch' + str(epoch))
        expected = gaussian((1, 210, 4), 3593 + epoch)
        np.testing.assert_allclose(expected, bank['standard'], atol=1e-14, rtol=1e-14)
        exact(expected.astype(np.float32), bank['epsilon'])
        gaussian_count += expected.size
    initial, batch = load('initial_arrays'), load('first_batch')
    x, prior = batch['images'].reshape(-1, 784), batch['prior']
    initial_gradients = derivatives(floating(initial), x, prior)
    first, _ = forward(floating(initial), x, prior)
    saved_first = load('first_forward')
    for key in first:
        np.testing.assert_allclose(first[key], saved_first[key], atol=1e-4, rtol=5e-6)
    pixel_count += first['probability'].size
    count, errors, phase_errors = 0, {}, []
    for group, expected in enumerate(initial_gradients):
        actual = load('initial_gradients_group' + str(group))
        for key, value in expected.items():
            np.testing.assert_allclose(value, actual[key], atol=2e-6, rtol=5e-4)
            errors['initial/' + str(group) + '/' + key] = float(np.max(np.abs(value - actual[key])))
            count += value.size
    before = initial
    for phase, groups in [(0, [0, 1]), (1, [2]), (2, [3])]:
        expected = derivatives(floating(before), x, prior)
        after = load('first_after_phase' + str(phase))
        changed_keys = set()
        for group in groups:
            actual = load('first_actual_gradients_group' + str(group))
            for key, value in expected[group].items():
                np.testing.assert_allclose(value, actual[key], atol=2e-6, rtol=5e-4)
                errors['actual/' + str(group) + '/' + key] = float(np.max(np.abs(value - actual[key])))
                g = actual[key]
                first, second = g * np.float32(.1), (g * g) * np.float32(.001)
                direction = (first / np.float32(.1)) / (np.sqrt(second / np.float32(.001)) + np.float32(1e-8))
                exact(before[key] - direction * np.float32(.003), after[key])
                changed_keys.add(key)
                count += value.size
        for key in before:
            if key not in changed_keys:
                exact(before[key], after[key])
        phase_errors.append({'phase': phase, 'first_Adam_steps_bit_exact': True, 'unrelated_groups_bit_exact': True})
        before = after
    validation, validation_errors, scores = data['validation_images'].reshape(-1, 784), {}, []
    for epoch in range(4):
        weights = initial if epoch == 0 else load('epoch' + str(epoch) + '_arrays')
        prediction, _ = forward(floating(weights), validation, noise['validation_epsilon'][0])
        saved = load('epoch' + str(epoch) + '_validation')
        np.testing.assert_allclose(prediction['probability'], saved['probability'], atol=2e-6, rtol=5e-6)
        for key in ['reconstruction_loss', 'discriminator_loss', 'adversarial_loss']:
            np.testing.assert_allclose(prediction[key], saved[key], atol=1e-4, rtol=2e-6)
        validation_errors[str(epoch)] = float(np.max(np.abs(prediction['probability'] - saved['probability'])))
        scores.append(float(prediction['reconstruction_loss']))
        pixel_count += prediction['probability'].size
    selection = load('selection')
    selected = int(np.argmin(scores[1:])) + 1
    assert selected == int(selection['epoch'])
    best = load('best')
    for key in best:
        exact(best[key], load('epoch' + str(selected) + '_arrays')[key])
        exact(best[key], load('model_arrays')[key])
    p, outputs, prediction_errors = floating(best), load('outputs'), {}
    for split in ['fit', 'validation', 'test']:
        image = data[split + '_images'].reshape(-1, 784)
        prediction, _ = forward(p, image, noise[split + '_epsilon'][0])
        for key in prediction:
            np.testing.assert_allclose(prediction[key], outputs[split + '_' + key], atol=1e-4, rtol=5e-6)
        prediction_errors[split] = float(np.max(np.abs(prediction['probability'] - outputs[split + '_probability'])))
        pixel_count += prediction['probability'].size
    generation = load('prior')
    exact(generation['z'], noise['generation_epsilon'][0])
    gh = np.tanh(generation['z'].astype(np.float64) @ p['1/hidden/W'] + p['1/hidden/b'])
    prior_prediction = 1 / (1 + np.exp(-(gh @ p['1/output/W'] + p['1/output/b'])))
    np.testing.assert_allclose(prior_prediction, generation['probability'], atol=2e-6, rtol=5e-6)
    pixel_count += prior_prediction.size
    mean = data['fit_images'].reshape(-1, 784).astype(np.float64).mean(axis=0).astype(np.float32)
    exact(np.broadcast_to(mean, (42, 784)), load('mean')['mean'])
    return {'passed': True, 'parameter_derivatives_checked': count, 'gradient_max_errors': errors,
            'four_independent_first_Adam_states_checked': True, 'phases': phase_errors,
            'Gaussian_values_checked': gaussian_count, 'lattice_pixels_checked': 294 * 784,
            'total_lattice_and_prediction_pixels_checked': pixel_count, 'validation_checkpoints_checked': 4,
            'selected_epoch': selected, 'validation_prediction_max_errors': validation_errors,
            'pool_prediction_max_errors': prediction_errors, 'unfiltered_prior_predictions_checked': 20}


def main():
    """Run the optional development oracle on the report's completed build directory."""
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
