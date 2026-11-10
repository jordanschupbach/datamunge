#!/usr/bin/env python3
"""Optional independent NumPy oracle for complete compact VAE-GAN artifacts.

Never fits or prepares data. Checks all nineteen forward fields, every initial
and recomputed D/E/G derivative, three first Adam states and group isolation,
all eligible validation candidates, explicit noise and unfiltered generation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact, gaussian
from check_factor_tc_variational_autoencoder import first_update


def decode(p, z):
    """Independent tanh/sigmoid generator, with its hidden state for adjoints."""
    h = np.tanh(z @ p['1/hidden/W'] + p['1/hidden/b'])
    return 1 / (1 + np.exp(-(h @ p['1/output/W'] + p['1/output/b']))), h


def discriminate(p, x):
    """Learned eight-dimensional tanh feature and scalar real/fake logit."""
    h = np.tanh(x @ p['2/feature/W'] + p['2/feature/b'])
    return h, h @ p['2/logit/W'] + p['2/logit/b']


def forward(p, image, epsilon, prior):
    """All nineteen original endpoints, using independent Float64 arithmetic."""
    x = image.reshape(len(image), -1).astype(np.float64)
    epsilon, prior = epsilon.astype(np.float64), prior.astype(np.float64)
    h = np.tanh(x @ p['0/hidden/W'] + p['0/hidden/b'])
    mu = h @ p['0/mu/W'] + p['0/mu/b']
    lv = h @ p['0/log_variance/W'] + p['0/log_variance/b']
    sigma = np.exp(.5 * lv); z = mu + sigma * epsilon
    probability, gh = decode(p, z); prior_probability, ph = decode(p, prior)
    real, rl = discriminate(p, x)
    reconstruction, fl = discriminate(p, probability)
    generated, pl = discriminate(p, prior_probability)
    feature_nll = .5 * ((real - reconstruction)**2).sum(axis=1)
    kl = .5 * (mu*mu + np.exp(lv) - 1 - lv).sum(axis=1)
    adversarial = .5 * (np.logaddexp(0, -fl).mean() + np.logaddexp(0, -pl).mean())
    clipped = np.clip(probability, float(np.float32(1e-6)), float(np.float32(1) - np.float32(1e-6)))
    out = dict(mu=mu, log_variance=lv, z=z, probability=probability,
               prior_probability=prior_probability, real_feature=real,
               reconstruction_feature=reconstruction, prior_feature=generated,
               real_logit=rl, reconstruction_logit=fl, prior_logit=pl,
               feature_nll=feature_nll, kl=kl,
               pixel_nll=-(x*np.log(clipped)+(1-x)*np.log(1-clipped)).sum(axis=1),
               mse=((x-probability)**2).mean(axis=1),
               encoder_loss=(feature_nll+kl).mean(), generator_loss=feature_nll.mean()+adversarial,
               discriminator_loss=np.logaddexp(0, -rl).mean()+.5*(np.logaddexp(0, fl).mean()+np.logaddexp(0, pl).mean()),
               generator_adversarial=adversarial)
    return out, (x, h, sigma, gh, ph)


def derivatives(p, image, epsilon, prior):
    """Analytic isolated E, G and D leaf adjoints, including both generator uses."""
    out, (x, h, sigma, gh, ph) = forward(p, image, epsilon, prior)
    batch = len(x)
    reconstruction = out['reconstruction_feature']; generated = out['prior_feature']
    feature_back = ((reconstruction-out['real_feature'])/batch)*(1-reconstruction**2)
    probability_back = feature_back @ p['2/feature/W'].T

    def generator_back(z, hidden, probability, delta_probability):
        delta = delta_probability*probability*(1-probability)
        back = (delta @ p['1/output/W'].T)*(1-hidden**2)
        return {'1/output/W': hidden.T @ delta, '1/output/b': delta.sum(axis=0),
                '1/hidden/W': z.T @ back, '1/hidden/b': back.sum(axis=0)}, back @ p['1/hidden/W'].T

    _, dz = generator_back(out['z'], gh, out['probability'], probability_back)
    dm = dz+out['mu']/batch
    dl = .5*dz*sigma*epsilon+.5*(np.exp(out['log_variance'])-1)/batch
    dh = (dm @ p['0/mu/W'].T+dl @ p['0/log_variance/W'].T)*(1-h*h)
    encoder = {'0/mu/W': h.T @ dm, '0/mu/b': dm.sum(axis=0),
               '0/log_variance/W': h.T @ dl, '0/log_variance/b': dl.sum(axis=0),
               '0/hidden/W': x.T @ dh, '0/hidden/b': dh.sum(axis=0)}
    sigmoid = lambda a: 1/(1+np.exp(-a))
    rec_logit = .5*(sigmoid(out['reconstruction_logit'])-1)/batch
    prior_logit = .5*(sigmoid(out['prior_logit'])-1)/batch
    rec_back = (rec_logit @ p['2/logit/W'].T)*(1-reconstruction**2)
    prior_back = (prior_logit @ p['2/logit/W'].T)*(1-generated**2)
    generator, _ = generator_back(out['z'], gh, out['probability'], probability_back+rec_back @ p['2/feature/W'].T)
    other, _ = generator_back(prior, ph, out['prior_probability'], prior_back @ p['2/feature/W'].T)
    generator = {key: value+other[key] for key, value in generator.items()}
    discriminator = {key: np.zeros_like(value) for key, value in p.items() if key.startswith('2/')}
    for values, features, logit, weight, target in [
            (x, out['real_feature'], out['real_logit'], 1., 1.),
            (out['probability'], reconstruction, out['reconstruction_logit'], .5, 0.),
            (out['prior_probability'], generated, out['prior_logit'], .5, 0.)]:
        delta = weight*(sigmoid(logit)-target)/batch
        back = (delta @ p['2/logit/W'].T)*(1-features**2)
        discriminator['2/logit/W'] += features.T @ delta
        discriminator['2/logit/b'] += delta.sum(axis=0)
        discriminator['2/feature/W'] += values.T @ back
        discriminator['2/feature/b'] += back.sum(axis=0)
    return [encoder, generator, discriminator]


def check(directory):
    """Validate the complete source or report run and return measured evidence."""
    data, noise = arrays(directory/'data.npz'), arrays(directory/'fixed_noise.npz')
    pixels = gaussian_values = gradient_values = endpoint_values = endpoint_fields = 0
    gradient_error = endpoint_error = 0.
    seen = set(); yy, xx = np.mgrid[:28, :28]
    for index, (split, count) in enumerate([('fit', 210), ('validation', 42), ('test', 42)]):
        ids, factors, images = [data[split+suffix] for suffix in ['_source_ids', '_factors', '_images']]
        assert len(ids) == count and not seen.intersection(ids); seen.update(ids)
        for row, identity in enumerate(ids):
            shape, size, x, y = int(identity)//147, (int(identity)//49)%3, (int(identity)//7)%7, int(identity)%7
            np.testing.assert_array_equal(factors[row], [shape, size, x, y])
            residue = (x+2*y+size+3*shape)%7
            assert (residue == 0 if split == 'test' else residue == 1 if split == 'validation' else residue >= 2)
            dx, dy = abs(xx-(3*x+5)), abs(yy-(3*y+5))
            expected = np.maximum(dx, dy) <= size+2 if shape else dx*dx+dy*dy <= (size+2)**2
            exact(images[row], expected.astype(np.float32)[..., None]); pixels += 784
        for prefix, seed in [('', 3323+index*6), ('_prior', 3347+index*6)]:
            expected = gaussian((4, count, 4), seed)
            np.testing.assert_allclose(noise[split+prefix+'_standard'], expected, atol=1e-14, rtol=1e-14)
            exact(noise[split+prefix+'_epsilon'], expected.astype(np.float32)); gaussian_values += expected.size
    assert len(seen) == 294
    expected = gaussian((1, 20, 4), 3371)
    np.testing.assert_allclose(noise['generation_standard'], expected, atol=1e-14, rtol=1e-14)
    exact(noise['generation_epsilon'], expected.astype(np.float32)); gaussian_values += expected.size
    for epoch in range(1, 4):
        bank = arrays(directory/f'train_noise_epoch{epoch}.npz')
        for prefix, seed in [('', 3389+epoch), ('prior_', 3407+epoch)]:
            expected = gaussian((1, 210, 4), seed)
            np.testing.assert_allclose(bank[prefix+'standard'], expected, atol=1e-14, rtol=1e-14)
            exact(bank[prefix+'epsilon'], expected.astype(np.float32)); gaussian_values += expected.size
    floating = lambda p: {key: value.astype(np.float64) for key, value in p.items()}
    initial, batch = arrays(directory/'initial_arrays.npz'), arrays(directory/'first_batch.npz')
    x, epsilon, prior = batch['images'], batch['epsilon'].astype(np.float64), batch['prior'].astype(np.float64)
    def compare_gradients(expected, actual):
        nonlocal gradient_error, gradient_values
        assert set(expected) == set(actual)
        for key, value in expected.items():
            np.testing.assert_allclose(actual[key], value, atol=2e-6, rtol=5e-4, err_msg=key)
            gradient_error = max(gradient_error, float(np.max(abs(actual[key]-value)))); gradient_values += value.size
    for group, expected in enumerate(derivatives(floating(initial), x, epsilon, prior)):
        compare_gradients(expected, arrays(directory/f'initial_gradients_group{group}.npz'))
    before = initial
    for group in [2, 0, 1]:
        actual = arrays(directory/f'first_actual_gradients_group{group}.npz')
        compare_gradients(derivatives(floating(before), x, epsilon, prior)[group], actual)
        after = arrays(directory/f'first_after_group{group}.npz'); first_update(before, actual, after)
        for key in before:
            if key not in actual: exact(before[key], after[key])
        before = after
    def compare_forward(out, saved, prefix=''):
        nonlocal endpoint_error, endpoint_values, endpoint_fields
        for key, value in out.items():
            target = saved[prefix+key]
            np.testing.assert_allclose(target, value, atol=4e-4 if key == 'pixel_nll' else 5e-6, rtol=2e-5, err_msg=prefix+key)
            endpoint_error = max(endpoint_error, float(np.max(abs(target-value))))
            endpoint_values += np.size(value); endpoint_fields += 1
    compare_forward(forward(floating(initial), x, epsilon, prior)[0], arrays(directory/'first_forward.npz'))
    selected, best = arrays(directory/'model_arrays.npz'), arrays(directory/'best.npz')
    for key in selected: exact(selected[key], best[key])
    p, saved = floating(selected), arrays(directory/'outputs.npz'); assert len(saved) == 285
    for split, count in [('fit', 210), ('validation', 42), ('test', 42)]:
        for draw in range(-1, 4):
            epsilon = np.zeros((count, 4)) if draw < 0 else noise[split+'_epsilon'][draw]
            prior = noise[split+'_prior_epsilon'][max(draw, 0)]
            compare_forward(forward(p, data[split+'_images'], epsilon, prior)[0], saved,
                            split+('_zero_' if draw < 0 else f'_draw{draw}_'))
    history = np.genfromtxt(directory/'history.tsv', names=True, delimiter='\t'); criteria = []
    for row in history:
        epoch = int(row[0]); q = floating(arrays(directory/f'epoch{epoch}_arrays.npz'))
        draws = [forward(q, data['validation_images'], noise['validation_epsilon'][k], noise['validation_prior_epsilon'][k])[0] for k in range(4)]
        values = [np.mean([out[key].mean() for out in draws]) for key in ['mse', 'pixel_nll', 'feature_nll', 'kl']]
        np.testing.assert_allclose(list(row)[1:], values, atol=4e-4, rtol=2e-5); criteria.append(values[0])
    meta = json.loads((directory/'metrics.json').read_text()); chosen = int(np.argmin(criteria))+1
    assert meta['selected_epoch'] == chosen
    assert meta['parameters'] == 32393 and meta['tensors'] == 14 and meta['epochs'] == 3
    assert meta['steps_per_group'] == 21 and meta['total_optimizer_updates'] == 63
    batches = np.genfromtxt(directory/'batches.tsv', names=True, delimiter='\t', dtype=np.int64)
    assert len(batches) == 630
    for epoch in range(1, 4):
        rows = batches[batches['epoch'] == epoch]
        np.testing.assert_array_equal(np.sort(rows['row']), np.arange(210))
        np.testing.assert_array_equal(rows['source_id'], data['fit_source_ids'][rows['row']])
    for key, value in arrays(directory/f'epoch{chosen}_arrays.npz').items(): exact(value, selected[key])
    prior = arrays(directory/'prior.npz'); exact(prior['z'], noise['generation_epsilon'][0])
    probability, _ = decode(p, prior['z'].astype(np.float64))
    np.testing.assert_allclose(prior['probability'], probability, atol=2e-6, rtol=1e-5)
    draws = [forward(p, data['test_images'], noise['test_epsilon'][k], noise['test_prior_epsilon'][k])[0] for k in range(4)]
    for name, key in [('test_mc_mse', 'mse'), ('test_diagnostic_pixel_nll', 'pixel_nll'), ('test_feature_nll', 'feature_nll'), ('test_kl', 'kl')]:
        np.testing.assert_allclose(meta[name], np.mean([out[key].mean() for out in draws]), atol=4e-4, rtol=2e-5)
    target = data['test_images'].reshape(42, 784).astype(np.float64)
    fitting_mean = data['fit_images'].reshape(210, 784).astype(np.float64).mean(axis=0).astype(np.float32).astype(np.float64)
    np.testing.assert_allclose(meta['fit_mean_mse'], ((target-fitting_mean)**2).mean(), atol=1e-14, rtol=1e-14)
    zero, _ = forward(p, data['test_images'], np.zeros((42, 4)), noise['test_prior_epsilon'][0])
    np.testing.assert_allclose(meta['decode_mu_mse'], ((target-zero['probability'])**2).mean(), atol=2e-6, rtol=1e-5)
    return dict(passed=True, parameter_derivatives_checked=gradient_values, first_gradient_max_error=gradient_error,
                three_first_Adam_steps_bit_exact=True, unrelated_groups_bit_exact=True,
                Gaussian_values_checked=gaussian_values, lattice_pixels_checked=pixels,
                endpoint_fields_checked=endpoint_fields, endpoint_values_checked=endpoint_values,
                maximum_endpoint_error=endpoint_error, eligible_validation_checkpoints_checked=3,
                selected_epoch=chosen, unfiltered_prior_predictions_checked=20)


def main():
    """Run the optional development oracle on completed original/report artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path); parser.add_argument('--json', type=Path)
    args = parser.parse_args(); result = check(args.directory)
    text = json.dumps(result, indent=2)+'\n'
    if args.json: args.json.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
