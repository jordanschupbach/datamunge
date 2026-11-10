#!/usr/bin/env python3
"""Optional independent NumPy oracle for complete conditional-flow VAE artifacts.

No fitting or preparation. Checks every first derivative/Adam update, all18
posterior/inverse fields at every saved draw, full coupling Jacobians, explicit
Gaussian banks, lattice pixels, standard-normal direct priors and selection.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact, gaussian
from check_beta_variational_autoencoder import decode
from check_factor_tc_variational_autoencoder import first_update


def coupling(p, name, z, context, swap=False, inverse=False):
    """Independent conditional affine forward/inverse and cache for reverse differentiation."""
    half = z.shape[1]//2
    a, b = (z[:, half:], z[:, :half]) if swap else (z[:, :half], z[:, half:])
    inputs = np.concatenate([a, context], axis=1)
    h = np.maximum(inputs@p[name+'/conditioner/W']+p[name+'/conditioner/b'], 0)
    heads = h@p[name+'/heads/W']+p[name+'/heads/b']
    tanh = np.tanh(heads[:, :half]);s = .5*tanh;shift = heads[:, half:]
    changed = (b-shift)*np.exp(-s) if inverse else b*np.exp(s)+shift
    value = np.concatenate([changed, a] if swap else [a, changed], axis=1)
    return (value, (-1 if inverse else 1)*s.sum(axis=1)), (inputs, h, tanh, s, b, swap)


def coupling_adjoint(p, name, cache, dz, dlogdet):
    """All conditioner/head, code and live-context derivatives of a forward coupling."""
    inputs, h, tanh, s, b, swap = cache
    half = b.shape[1]
    da, dbout = (dz[:, half:].copy(), dz[:, :half]) if swap else (dz[:, :half].copy(), dz[:, half:])
    ds = dbout*b*np.exp(s)+dlogdet[:, None]
    dheads = np.concatenate([.5*(1-tanh*tanh)*ds, dbout], axis=1)
    dh = (dheads@p[name+'/heads/W'].T)*(h > 0)
    dinputs = dh@p[name+'/conditioner/W'].T;da += dinputs[:, :half]
    db = dbout*np.exp(s)
    parameters = {name+'/heads/W': h.T@dheads, name+'/heads/b': dheads.sum(axis=0),
                  name+'/conditioner/W': inputs.T@dh, name+'/conditioner/b': dh.sum(axis=0)}
    return np.concatenate([db, da] if swap else [da, db], axis=1), dinputs[:, half:], parameters


def forward(p, x, epsilon):
    """Eighteen Gaussian/flow/density/decoder/inverse endpoints, computed in Float64."""
    x, epsilon = x.astype(np.float64), epsilon.astype(np.float64)
    h = np.maximum(x@p['encoder/W']+p['encoder/b'], 0)
    mu, lv = h@p['mu/W']+p['mu/b'], h@p['log_variance/W']+p['log_variance/b']
    sigma = np.exp(.5*lv);z0 = mu+sigma*epsilon
    (z1, ld0), c0 = coupling(p, 'flow0', z0, h)
    (z, ld1), c1 = coupling(p, 'flow1', z1, h, True)
    logdet = ld0+ld1
    logq0 = -.5*((z0-mu)**2*np.exp(-lv)+lv+np.log(2*np.pi)).sum(axis=1)
    logq = logq0-logdet;logp = -.5*(z*z+np.log(2*np.pi)).sum(axis=1)
    logits, probability, dh = decode(p, z)
    nll = (np.logaddexp(0, logits)-x*logits).sum(axis=1)
    (back1, ild1), _ = coupling(p, 'flow1', z, h, True, True)
    (back0, ild0), _ = coupling(p, 'flow0', back1, h, False, True)
    return dict(context=h, mu=mu, log_variance=lv, z0=z0, z1=z1, z=z, logdet0=ld0, logdet1=ld1,
        logdet=logdet, logq0=logq0, logq=logq, logp=logp, logratio=logq-logp,
        probability=probability, nll=nll, loss=np.array(np.mean(nll+logq-logp)),
        inverse_z0=back0, inverse_logdet=ild1+ild0), (sigma, dh, c0, c1)


def derivatives(p, x, epsilon):
    """All18 leaves, including direct context, base entropy, final prior and negative logdet."""
    x, epsilon = x.astype(np.float64), epsilon.astype(np.float64)
    o, (sigma, dh, c0, c1) = forward(p, x, epsilon);batch = len(x)
    delta = (o['probability']-x)/batch
    result = {'output/W': dh.T@delta, 'output/b': delta.sum(axis=0)}
    back = (delta@p['output/W'].T)*(dh > 0)
    result.update({'decoder/W': o['z'].T@back, 'decoder/b': back.sum(axis=0)})
    dz = back@p['decoder/W'].T+o['z']/batch
    dz1, context1, first = coupling_adjoint(p, 'flow1', c1, dz, np.full(batch, -1/batch));result.update(first)
    dz0, context0, first = coupling_adjoint(p, 'flow0', c0, dz1, np.full(batch, -1/batch));result.update(first)
    dmu = dz0;dlv = .5*dz0*sigma*epsilon-.5/batch
    h = o['context']
    result.update({'mu/W': h.T@dmu, 'mu/b': dmu.sum(axis=0),
                   'log_variance/W': h.T@dlv, 'log_variance/b': dlv.sum(axis=0)})
    back = (dmu@p['mu/W'].T+dlv@p['log_variance/W'].T+context0+context1)*(h > 0)
    result.update({'encoder/W': x.T@back, 'encoder/b': back.sum(axis=0)})
    return result


def check(directory):
    """Check complete source-compatible artifacts and return measured errors/counts."""
    data, noise = arrays(directory/'data.npz'), arrays(directory/'fixed_noise.npz')
    pixels, noise_values, identities, unique = 0, 0, set(), set()
    for split, count, seed in [('fit', 210, 3011), ('validation', 42, 3019), ('test', 42, 3023)]:
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
        expected = gaussian((4, count, 4), seed)
        np.testing.assert_allclose(noise[split+'_standard'], expected, atol=1e-14, rtol=1e-14)
        exact(noise[split+'_epsilon'], expected.astype(np.float32));noise_values += expected.size
    assert len(identities) == len(unique) == 294
    expected = gaussian((1, 20, 4), 3037)
    np.testing.assert_allclose(noise['prior_standard'], expected, atol=1e-14, rtol=1e-14)
    exact(noise['prior_epsilon'], expected.astype(np.float32));noise_values += expected.size
    for epoch in range(1, 4):
        bank = arrays(directory/f'train_noise_epoch{epoch}.npz');expected = gaussian((1, 210, 4), 3041+epoch)
        np.testing.assert_allclose(bank['standard'], expected, atol=1e-14, rtol=1e-14)
        exact(bank['epsilon'], expected.astype(np.float32));noise_values += expected.size
    algebra, unit = arrays(directory/'flow_algebra.npz'), arrays(directory/'unit_flow_arrays.npz')
    unit = {k: v.astype(np.float64) for k, v in unit.items()}
    z, context = algebra['z'].astype(np.float64), algebra['context'].astype(np.float64)
    def mapped(codes):
        (one, ld0), _ = coupling(unit, 'flow0', codes, context)
        (two, ld1), _ = coupling(unit, 'flow1', one, context, True)
        return two, ld0+ld1
    value, logdet = mapped(z)
    np.testing.assert_allclose(algebra['mapped'], value, atol=3e-7, rtol=1e-6)
    np.testing.assert_allclose(algebra['logdet'], logdet, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(algebra['inverse'], z, atol=3e-7, rtol=1e-6)
    np.testing.assert_allclose(algebra['inverse_logdet'], -logdet, atol=1e-7, rtol=1e-6)
    exact(algebra['identity'], algebra['z']);exact(algebra['identity_logdet'], np.zeros(3, dtype=np.float32))
    jac = np.zeros((3, 4, 4))
    for axis in range(4):
        delta = np.zeros_like(z);delta[:, axis] = 1e-6
        jac[:, :, axis] = (mapped(z+delta)[0]-mapped(z-delta)[0])/2e-6
    np.testing.assert_allclose(algebra['jacobian'], jac, atol=2e-7, rtol=1e-5)
    np.testing.assert_allclose(np.log(np.linalg.det(jac)), logdet, atol=1e-9, rtol=1e-8)
    initial = arrays(directory/'initial_arrays.npz');p = {k: v.astype(np.float64) for k, v in initial.items()}
    batch, actual = arrays(directory/'first_batch.npz'), arrays(directory/'first_gradients.npz')
    x = batch['images'].reshape(-1, 784);expected = derivatives(p, x, batch['epsilon'])
    assert set(expected) == set(actual)
    gradient_count, gradient_error = 0, 0
    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], value, atol=4e-6, rtol=8e-4)
        gradient_count += value.size;gradient_error = max(gradient_error, float(np.max(np.abs(actual[key]-value))))
    assert gradient_count == 52672
    first_update(initial, actual, arrays(directory/'first_update_arrays.npz'))
    first, _ = forward(p, x, batch['epsilon'])
    for key, value in arrays(directory/'first_forward.npz').items():
        np.testing.assert_allclose(value, first[key], atol=4e-4, rtol=1e-5)
    selected = arrays(directory/'model_arrays.npz');p = {k: v.astype(np.float64) for k, v in selected.items()}
    output = arrays(directory/'outputs.npz');prediction_error, density_error, inverse_error = 0, 0, 0
    for split in ['fit', 'validation', 'test']:
        x = data[split+'_images'].reshape(-1, 784)
        for label, epsilon in [('zero', np.zeros((len(x), 4)))]+[(f'draw{k}', e) for k, e in enumerate(noise[split+'_epsilon'])]:
            expected, _ = forward(p, x, epsilon)
            for key, value in expected.items():
                np.testing.assert_allclose(output[f'{split}_{label}_{key}'], value, atol=5e-4, rtol=2e-5)
            prediction_error = max(prediction_error, float(np.max(np.abs(output[f'{split}_{label}_probability']-expected['probability']))))
            for key in ['logdet', 'logq0', 'logq', 'logp', 'logratio']:
                density_error = max(density_error, float(np.max(np.abs(output[f'{split}_{label}_{key}']-expected[key]))))
            inverse_error = max(inverse_error, float(np.max(np.abs(output[f'{split}_{label}_inverse_z0']-output[f'{split}_{label}_z0']))));pixels += expected['probability'].size
    prior = arrays(directory/'prior.npz');exact(prior['z'], noise['prior_epsilon'][0])
    _, prediction, _ = decode(p, prior['z'].astype(np.float64))
    np.testing.assert_allclose(prior['probability'], prediction, atol=1e-5, rtol=1e-5);pixels += prediction.size
    history = np.genfromtxt(directory/'history.tsv', delimiter='\t', names=True);scores = []
    for epoch in range(1, 4):
        ep = {k: v.astype(np.float64) for k, v in arrays(directory/f'epoch{epoch}_arrays.npz').items()}
        nll, ratio, det = 0, 0, 0;x = data['validation_images'].reshape(-1, 784)
        for epsilon in noise['validation_epsilon']:
            o, _ = forward(ep, x, epsilon);nll += o['nll'].mean()/4;ratio += o['logratio'].mean()/4;det += o['logdet'].mean()/4
        scores.append(nll+ratio)
        for key, value in [('validation_nll', nll), ('validation_logq_minus_logp', ratio), ('validation_logdet', det), ('validation_negative_elbo', nll+ratio)]:
            np.testing.assert_allclose(history[key][epoch-1], value, atol=4e-4, rtol=1e-6)
    chosen = int(np.argmin(scores)+1)
    for key, value in arrays(directory/f'epoch{chosen}_arrays.npz').items():exact(value, selected[key])
    metrics = json.loads((directory/'metrics.json').read_text());assert metrics['selected_epoch'] == chosen and metrics['updates'] == 21
    nll = np.mean([output[f'test_draw{k}_nll'].astype(np.float64).mean() for k in range(4)])
    ratio = np.mean([output[f'test_draw{k}_logratio'].astype(np.float64).mean() for k in range(4)])
    np.testing.assert_allclose([metrics['test_nll'], metrics['test_mc_logq_minus_logp'], metrics['test_negative_elbo']], [nll, ratio, nll+ratio], atol=1e-12, rtol=0)
    batches = np.genfromtxt(directory/'batches.tsv', delimiter='\t', names=True, dtype=int)
    assert set(batches['step']) == set(range(21))
    for epoch in range(1, 4):np.testing.assert_array_equal(np.sort(batches['row'][batches['epoch'] == epoch]), np.arange(210))
    np.testing.assert_array_equal(batches['source_id'], data['fit_source_ids'][batches['row']])
    return dict(all_first_batch_derivatives_checked=gradient_count, first_gradient_max_error=gradient_error,
        first_Adam_update_bit_exact=True, Gaussian_values_checked=noise_values,
        lattice_and_prediction_pixels_checked=pixels, full_Jacobian_entries_checked=48,
        full_Jacobian_max_error=float(np.max(np.abs(algebra['jacobian']-jac))), all_eighteen_endpoints_checked=True,
        posterior_probability_max_error=prediction_error, density_field_max_error=density_error,
        inverse_roundtrip_max_error=inverse_error, eligible_validation_checkpoints_checked=3, selected_epoch=chosen)


def main():
    """Run the optional development oracle and save a compact evidence receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path);parser.add_argument('--json', type=Path)
    args = parser.parse_args();result = check(args.directory)
    if args.json:args.json.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
