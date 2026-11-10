#!/usr/bin/env python3
"""Optional independent NumPy oracle for complete Gumbel categorical-VAE artifacts.

No fitting or preparation. Checks explicit MT19937 uniforms/Gumbels, every first
parameter derivative/Adam update, posterior soft/hard predictions, discrete KL,
calibration, uniform priors, source data and all eligible selection criteria.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact
from check_beta_variational_autoencoder import decode
from check_factor_tc_variational_autoencoder import first_update


def bank(shape, seed):
    """Independent raw MT19937 words, midpoint uniforms and Gumbel inverse CDF."""
    words = np.random.RandomState(seed).randint(0, 2**32, size=int(np.prod(shape)), dtype=np.uint32)
    uniform = ((words.astype(np.float64)+.5)/2**32).reshape(shape)
    gumbel = -np.log(-np.log(uniform))
    return uniform, gumbel, gumbel.astype(np.float32)


def sample(logits, noise, temperature):
    """Stable posterior log-softmax, explicit relaxed draw and detached argmax IDs."""
    peak = logits.max(axis=-1, keepdims=True)
    lp = logits-peak-np.log(np.exp(logits-peak).sum(axis=-1, keepdims=True))
    scores = logits+noise
    scaled = scores/temperature;scaled -= scaled.max(axis=-1, keepdims=True)
    exp = np.exp(scaled);relaxed = exp/exp.sum(axis=-1, keepdims=True)
    return lp, np.exp(lp), relaxed, np.argmax(scores, axis=-1).astype(np.int32)


def forward(p, image, noise, temperature=float(np.float32(.7))):
    """Independent eight-leaf categorical encoder and shared soft/hard decoder."""
    image, noise = image.astype(np.float64), noise.astype(np.float64)
    h = np.maximum(image@p['encoder/W']+p['encoder/b'], 0)
    logits = (h@p['categorical/W']+p['categorical/b']).reshape(noise.shape)
    lp, q, y, indices = sample(logits, noise, temperature)
    hard = np.eye(noise.shape[-1])[indices]
    code = y.reshape(len(image), -1);hard_code = hard.reshape(len(image), -1)
    soft_logits, soft_prediction, dh = decode(p, code)
    hard_logits, hard_prediction, _ = decode(p, hard_code)
    nll = (np.logaddexp(0, soft_logits)-image*soft_logits).sum(axis=1)
    hard_nll = (np.logaddexp(0, hard_logits)-image*hard_logits).sum(axis=1)
    kl = (q*(lp+np.log(noise.shape[-1]))).sum(axis=(1, 2))
    return dict(logits=logits, log_probability=lp, probability=q, relaxed=y, indices=indices, hard=hard,
        soft_prediction=soft_prediction, hard_prediction=hard_prediction, nll=nll, hard_nll=hard_nll, kl=kl,
        loss=np.array((nll+kl).mean()), hard_objective=np.array((hard_nll+kl).mean())), (h, code, dh)


def derivatives(p, image, noise, temperature=float(np.float32(.7))):
    """All eight leaf derivatives, including soft path and discrete posterior KL."""
    image, noise = image.astype(np.float64), noise.astype(np.float64)
    o, (h, code, dh) = forward(p, image, noise, temperature);count = len(image)
    delta = (o['soft_prediction']-image)/count
    result = {'output/W': dh.T@delta, 'output/b': delta.sum(axis=0)}
    back = (delta@p['output/W'].T)*(dh > 0)
    result.update({'decoder/W': code.T@back, 'decoder/b': back.sum(axis=0)})
    dy = (back@p['decoder/W'].T).reshape(noise.shape)
    y, q, lp = o['relaxed'], o['probability'], o['log_probability']
    dlogits = y*(dy-(dy*y).sum(axis=-1, keepdims=True))/temperature
    per_group = (q*(lp+np.log(noise.shape[-1]))).sum(axis=-1, keepdims=True)
    dlogits += q*(lp+np.log(noise.shape[-1])-per_group)/count
    flat = dlogits.reshape(count, -1)
    result.update({'categorical/W': h.T@flat, 'categorical/b': flat.sum(axis=0)})
    back = (flat@p['categorical/W'].T)*(h > 0)
    result.update({'encoder/W': image.T@back, 'encoder/b': back.sum(axis=0)})
    return result


def check(directory):
    """Verify all artifacts and return measured first-gradient, prediction and noise errors."""
    data, noise = arrays(directory/'data.npz'), arrays(directory/'fixed_noise.npz')
    pixels, uniforms, identities, unique = 0, 0, set(), set();max_gumbel_error = 0
    for split, count, seed in [('fit', 210, 2927), ('validation', 42, 2939), ('test', 42, 2953)]:
        ids, factors, images = [data[split+s] for s in ['_source_ids', '_factors', '_images']]
        assert len(ids) == count and not identities.intersection(ids)
        identities.update(ids);yy, xx = np.mgrid[:28, :28]
        for row, identity in enumerate(ids):
            shape, size, x, y = int(identity)//147, (int(identity)//49)%3, (int(identity)//7)%7, int(identity)%7
            np.testing.assert_array_equal(factors[row], [shape, size, x, y])
            residue = (x+2*y+size+3*shape)%7
            assert residue == 0 if split == 'test' else residue == 1 if split == 'validation' else residue >= 2
            dx, dy = abs(xx-(3*x+5)), abs(yy-(3*y+5))
            expected = np.maximum(dx, dy) <= size+2 if shape else dx*dx+dy*dy <= (size+2)**2
            exact(expected.astype(np.float32)[..., None], images[row]);unique.add(images[row].tobytes())
        pixels += images.size
        u, g, f = bank((4, count, 4, 8), seed);exact(noise[split+'_uniform'], u)
        np.testing.assert_allclose(noise[split+'_gumbel'], g, atol=2e-15, rtol=1e-14)
        max_gumbel_error = max(max_gumbel_error, float(np.max(np.abs(noise[split+'_gumbel']-g))))
        exact(noise[split+'_noise'], f);uniforms += u.size
    assert len(identities) == len(unique) == 294
    for epoch in range(1, 4):
        actual = arrays(directory/f'train_noise_epoch{epoch}.npz');u, g, f = bank((1, 210, 4, 8), 2969+epoch)
        exact(actual['uniform'], u);np.testing.assert_allclose(actual['gumbel'], g, atol=2e-15, rtol=1e-14)
        exact(actual['noise'], f);uniforms += u.size
        max_gumbel_error = max(max_gumbel_error, float(np.max(np.abs(actual['gumbel']-g))))
    algebra = arrays(directory/'categorical_algebra.npz')
    logits, perturbation = algebra['logits'].astype(np.float64), algebra['noise'].astype(np.float64)
    lp, q, y, ids = sample(logits, perturbation, float(np.float32(.7)))
    kl = (q*(lp+np.log(3))).sum(axis=1)
    rec = y*(y-(y*y).sum(axis=1, keepdims=True))/float(np.float32(.7))
    posterior = .3*q*(lp+np.log(3)-kl[:, None])
    for key, value in [('probability', q), ('relaxed', y), ('kl', kl), ('dlogits', rec+posterior), ('dnoise', rec)]:
        np.testing.assert_allclose(algebra[key], value, atol=3e-7, rtol=1e-5)
    exact(algebra['indices'], ids)
    for key, temperature in [('cold', .25), ('warm', 2)]:
        _, _, relaxed, indices = sample(logits, perturbation, temperature)
        np.testing.assert_allclose(algebra[key], relaxed, atol=2e-7, rtol=1e-5);exact(algebra['indices'], indices)
    np.testing.assert_allclose(algebra['extreme_kl'], np.log(3), atol=2e-7, rtol=1e-6)
    u, g, f = bank((4096, 3), 2999);exact(algebra['calibration_uniform'], u)
    np.testing.assert_allclose(algebra['calibration_gumbel'], g, atol=2e-15, rtol=1e-14)
    exact(algebra['calibration_noise'], f);uniforms += u.size
    max_gumbel_error = max(max_gumbel_error, float(np.max(np.abs(algebra['calibration_gumbel']-g))))
    known_logits = np.log([.2, .3, .5]).astype(np.float32)
    expected_ids = np.argmax(known_logits+f, axis=1).astype(np.int32)
    exact(algebra['calibration_indices'], expected_ids)
    counts = np.bincount(expected_ids, minlength=3).astype(np.int32);exact(algebra['calibration_counts'], counts)
    target = np.array([.2, .3, .5]);assert (np.abs(counts-4096*target) < 6*np.sqrt(4096*target*(1-target))+1).all()
    initial = arrays(directory/'initial_arrays.npz');p = {k: v.astype(np.float64) for k, v in initial.items()}
    batch, actual = arrays(directory/'first_batch.npz'), arrays(directory/'first_gradients.npz')
    x = batch['images'].reshape(-1, 784);expected = derivatives(p, x, batch['noise'])
    assert set(expected) == set(actual);gradient_count, gradient_error = 0, 0
    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], value, atol=3e-6, rtol=8e-4)
        gradient_count += value.size;gradient_error = max(gradient_error, float(np.max(np.abs(actual[key]-value))))
    assert gradient_count == 53104
    first_update(initial, actual, arrays(directory/'first_update_arrays.npz'))
    first, _ = forward(p, x, batch['noise'])
    for key, value in arrays(directory/'first_forward.npz').items():
        if key in ['indices', 'hard']:np.testing.assert_array_equal(value, first[key])
        else:np.testing.assert_allclose(value, first[key], atol=4e-4, rtol=1e-5)
    selected = arrays(directory/'model_arrays.npz');p = {k: v.astype(np.float64) for k, v in selected.items()}
    output = arrays(directory/'outputs.npz');prediction_error, min_margin = 0, float('inf')
    for split in ['fit', 'validation', 'test']:
        x = data[split+'_images'].reshape(-1, 784)
        deterministic, _ = forward(p, x, np.zeros((len(x), 4, 8)))
        exact(output[split+'_map_ids'], deterministic['indices'])
        _, pred, _ = decode(p, deterministic['probability'].reshape(len(x), 32))
        np.testing.assert_allclose(output[split+'_posterior_probability_decode'], pred, atol=1e-5, rtol=1e-5);pixels += pred.size
        np.testing.assert_allclose(output[split+'_map_prediction'], deterministic['hard_prediction'], atol=1e-5, rtol=1e-5);pixels += pred.size
        for k, perturbation in enumerate(noise[split+'_noise']):
            expected, _ = forward(p, x, perturbation)
            for key, value in expected.items():
                saved = output[f'{split}_draw{k}_{key}']
                if key in ['indices', 'hard']:np.testing.assert_array_equal(saved, value)
                else:np.testing.assert_allclose(saved, value, atol=4e-4, rtol=1e-5)
            for key in ['soft_prediction', 'hard_prediction']:
                prediction_error = max(prediction_error, float(np.max(np.abs(output[f'{split}_draw{k}_{key}']-expected[key]))));pixels += expected[key].size
            sorted_scores = np.sort(expected['logits']+perturbation, axis=-1)
            min_margin = min(min_margin, float(np.min(sorted_scores[:, :, -1]-sorted_scores[:, :, -2])))
    prior = arrays(directory/'prior.npz');u, _, _ = bank((20, 4), 2963);exact(prior['uniform'], u);uniforms += u.size
    prior_ids = (8*u).astype(np.int32);exact(prior['indices'], prior_ids)
    _, pred, _ = decode(p, np.eye(8)[prior_ids].reshape(20, 32))
    np.testing.assert_allclose(prior['probability'], pred, atol=1e-5, rtol=1e-5);pixels += pred.size
    history = np.genfromtxt(directory/'history.tsv', delimiter='\t', names=True);scores = []
    for epoch in range(1, 4):
        ep = {k: v.astype(np.float64) for k, v in arrays(directory/f'epoch{epoch}_arrays.npz').items()}
        soft, hard = 0, 0;x = data['validation_images'].reshape(-1, 784)
        for perturbation in noise['validation_noise']:
            o, _ = forward(ep, x, perturbation);soft += o['nll'].mean()/4;hard += o['hard_nll'].mean()/4
        kl = o['kl'].mean();scores.append(soft+kl)
        for key, value in [('validation_soft_nll', soft), ('validation_hard_nll', hard), ('validation_categorical_kl', kl), ('validation_relaxed_objective', soft+kl)]:
            np.testing.assert_allclose(history[key][epoch-1], value, atol=4e-4, rtol=1e-6)
    chosen = int(np.argmin(scores)+1)
    for key, value in arrays(directory/f'epoch{chosen}_arrays.npz').items():exact(value, selected[key])
    metrics = json.loads((directory/'metrics.json').read_text());assert metrics['selected_epoch'] == chosen and metrics['updates'] == 21
    soft = np.mean([output[f'test_draw{k}_nll'].astype(np.float64).mean() for k in range(4)])
    hard = np.mean([output[f'test_draw{k}_hard_nll'].astype(np.float64).mean() for k in range(4)])
    kl = output['test_draw0_kl'].astype(np.float64).mean()
    np.testing.assert_allclose([metrics['test_soft_nll'], metrics['test_hard_nll'], metrics['test_categorical_kl'], metrics['test_relaxed_objective'], metrics['test_hard_mc_negative_elbo']], [soft, hard, kl, soft+kl, hard+kl], atol=1e-12, rtol=0)
    batches = np.genfromtxt(directory/'batches.tsv', delimiter='\t', names=True, dtype=int)
    assert set(batches['step']) == set(range(21))
    for epoch in range(1, 4):np.testing.assert_array_equal(np.sort(batches['row'][batches['epoch'] == epoch]), np.arange(210))
    np.testing.assert_array_equal(batches['source_id'], data['fit_source_ids'][batches['row']])
    return dict(all_first_batch_derivatives_checked=gradient_count, first_gradient_max_error=gradient_error,
        first_Adam_update_bit_exact=True, midpoint_uniform_values_checked=uniforms, Gumbel_max_error=max_gumbel_error,
        lattice_and_prediction_pixels_checked=pixels, all_thirteen_source_endpoints_checked=True,
        posterior_probability_max_error=prediction_error, minimum_hard_sample_score_margin=min_margin,
        calibration_draws_checked=4096, calibration_counts=counts.tolist(), eligible_validation_checkpoints_checked=3, selected_epoch=chosen)


def main():
    """Run the optional development oracle and save a compact validation receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path);parser.add_argument('--json', type=Path)
    args = parser.parse_args();result = check(args.directory)
    if args.json:args.json.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
