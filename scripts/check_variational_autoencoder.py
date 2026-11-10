#!/usr/bin/env python3
"""Optional independent NumPy oracle for complete two-coordinate vanilla VAE artifacts.

Never fits or prepares data. Reuses independent Gaussian/ReLU chain rules to
check all first derivatives, exact first Adam, static thresholding, explicit
noise, posterior/prior/lattice decoding and every eligible validation score.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact, gaussian
from check_beta_variational_autoencoder import forward, decode, derivatives
from check_factor_tc_variational_autoencoder import first_update


def check(directory):
    """Validate source/report artifacts and return measured numerical evidence."""
    data, noise = arrays(directory/'data.npz'), arrays(directory/'fixed_noise.npz')
    gaussian_values = pixels = endpoint_values = endpoint_fields = 0
    endpoint_error = gradient_error = 0.
    for split, count, seed in [('fit', 1000, 1991), ('validation', 200, 1987), ('test', 200, 1989)]:
        original, binary, ids = [data[split+suffix] for suffix in ['_original_images', '_images', '_source_ids']]
        assert original.shape == binary.shape == (count, 28, 28, 1)
        assert original.dtype == binary.dtype == np.float32
        assert np.isfinite(original).all() and ((original >= 0) & (original <= 1)).all()
        assert ids.shape == (count,) and len(np.unique(ids)) == count
        exact(binary, (original >= .5).astype(np.float32)); pixels += binary.size
        expected = gaussian((4, count, 2), seed)
        np.testing.assert_allclose(noise[split+'_standard'], expected, atol=1e-14, rtol=1e-14)
        exact(noise[split+'_epsilon'], expected.astype(np.float32)); gaussian_values += expected.size
    assert not set(data['fit_source_ids']).intersection(data['validation_source_ids'])
    # Test IDs refer to the separate MNIST test file; they are not train-file identities.
    assert data['test_labels'].shape == (200,) and np.isin(data['test_labels'], np.arange(10)).all()
    for epoch in range(1, 4):
        bank = arrays(directory/f'train_noise_epoch{epoch}.npz'); expected = gaussian((1, 1000, 2), 1973+epoch)
        np.testing.assert_allclose(bank['standard'], expected, atol=1e-14, rtol=1e-14)
        exact(bank['epsilon'], expected.astype(np.float32)); gaussian_values += expected.size
    initial, batch, actual, updated = [arrays(directory/(name+'.npz')) for name in
                                        ['initial_arrays', 'first_batch', 'first_gradients', 'first_update_arrays']]
    floating = lambda p: {key: value.astype(np.float64) for key, value in p.items()}
    expected = derivatives(floating(initial), batch['images'].reshape(-1, 784), batch['epsilon'], 1.)
    assert set(expected) == set(actual) == set(initial) and len(initial) == 10
    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], value, atol=2e-6, rtol=5e-4, err_msg=key)
        gradient_error = max(gradient_error, float(np.max(abs(actual[key]-value))))
    first_update(initial, actual, updated)
    def compare(a, b):
        nonlocal endpoint_error, endpoint_values, endpoint_fields
        np.testing.assert_allclose(a, b, atol=3e-5, rtol=2e-5)
        endpoint_error = max(endpoint_error, float(np.max(abs(a-b))))
        endpoint_values += np.size(a); endpoint_fields += 1
    selected, best = arrays(directory/'model_arrays.npz'), arrays(directory/'best.npz')
    for key in selected: exact(selected[key], best[key])
    p, saved = floating(selected), arrays(directory/'outputs.npz'); assert len(saved) == 22
    for split, count in [('fit', 1000), ('validation', 200), ('test', 200)]:
        x = data[split+'_images'].reshape(count, 784)
        zero, _ = forward(p, x, np.zeros((count, 2)))
        for field, name in [('mu', 'mu'), ('log_variance', 'log_variance'),
                            ('probability', 'mean_reconstruction'), ('kl', 'kl')]:
            compare(saved[split+'_'+name], zero[field])
        for draw in range(4):
            out, _ = forward(p, x, noise[split+'_epsilon'][draw])
            for field, name in [('z', 'sample_z'), ('probability', 'sample_prediction'), ('nll', 'sample_nll')]:
                compare(saved[split+'_'+name][draw], out[field])
    compare(saved['zero_latent_prediction'], decode(p, np.zeros((200, 2)))[1])
    prior = arrays(directory/'prior.npz'); expected = gaussian((1, 20, 2), 1997)
    np.testing.assert_allclose(prior['standard'], expected, atol=1e-14, rtol=1e-14)
    exact(prior['z'], expected[0].astype(np.float32)); gaussian_values += expected.size
    grid = np.array([[-2+.5*col, 2-.5*row] for row in range(9) for col in range(9)], dtype=np.float32)
    exact(prior['grid_z'], grid)
    for prefix in ['', 'grid_']:
        compare(prior[prefix+'probability'], decode(p, prior[prefix+'z'].astype(np.float64))[1])
    def score(parameters, split):
        draws = [forward(parameters, data[split+'_images'].reshape(-1, 784), e)[0] for e in noise[split+'_epsilon']]
        nll = np.mean([o['nll'].mean() for o in draws]); kl = draws[0]['kl'].mean()
        return nll, kl, nll+kl
    history = np.genfromtxt(directory/'history.tsv', names=True, delimiter='\t'); criteria = []
    for row in history:
        epoch = int(row['epoch']); out = score(floating(arrays(directory/f'epoch{epoch}_arrays.npz')), 'validation')
        for key, value in zip(['validation_reconstruction_nll', 'validation_kl', 'validation_negative_elbo'], out):
            np.testing.assert_allclose(row[key], value, atol=3e-5, rtol=2e-5)
        criteria.append(out[2])
    meta = json.loads((directory/'metrics.json').read_text()); chosen = int(np.argmin(criteria))+1
    assert meta['selected_epoch'] == chosen and meta['epochs'] == 3 and meta['updates'] == 96
    for key, value in arrays(directory/f'epoch{chosen}_arrays.npz').items(): exact(value, selected[key])
    for key, value in zip(['test_reconstruction_nll', 'test_kl', 'test_negative_elbo'], score(p, 'test')):
        np.testing.assert_allclose(meta[key], value, atol=3e-5, rtol=2e-5)
    np.testing.assert_allclose(meta['initial_test_negative_elbo'], score(floating(initial), 'test')[2], atol=3e-5, rtol=2e-5)
    zero, _ = forward(p, data['test_images'].reshape(200, 784), np.zeros((200, 2)))
    np.testing.assert_allclose(meta['mean_decode_test_mse'], ((zero['probability']-data['test_images'].reshape(200, 784))**2).mean(), atol=2e-6, rtol=1e-5)
    return dict(passed=True, first_parameter_derivatives_checked=sum(v.size for v in actual.values()),
                first_gradient_max_error=gradient_error, first_Adam_update_bit_exact=True,
                Gaussian_values_checked=gaussian_values, statically_thresholded_pixels_checked=pixels,
                endpoint_fields_checked=endpoint_fields, endpoint_values_checked=endpoint_values,
                maximum_endpoint_error=endpoint_error, eligible_validation_checkpoints_checked=3,
                selected_epoch=chosen, unfiltered_prior_predictions_checked=20, latent_grid_predictions_checked=81)


def main():
    """Run the optional development oracle against an already completed experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path); parser.add_argument('--json', type=Path)
    args = parser.parse_args(); text = json.dumps(check(args.directory), indent=2)+'\n'
    if args.json: args.json.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
