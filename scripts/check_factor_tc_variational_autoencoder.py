#!/usr/bin/env python3
"""Optional independent NumPy oracle for complete FactorVAE/full-reference betaTCVAE artifacts.

Performs no fitting or preparation. Checks all first VAE/discriminator derivatives,
Float32 first Adam updates, live reference-head paths, explicit Gaussian noise,
finite-mixture algebra, lattice pixels, selected predictions and fitting-only probes.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact, gaussian
from check_beta_variational_autoencoder import forward, decode, derivatives, probes


def mixture(z, mu, lv):
    """Log densities and analytic adjoints of the mean signed log ratio, in Float64."""
    delta = z[:, None, :] - mu[None, :, :]
    inv = np.exp(-lv)[None, :, :]
    c = -.5*(delta*delta*inv+lv[None, :, :]+np.log(2*np.pi))
    def lse(x, axis):
        peak = x.max(axis=axis, keepdims=True)
        return np.squeeze(peak+np.log(np.exp(x-peak).sum(axis=axis, keepdims=True)), axis=axis)
    joint = lse(c.sum(axis=2), 1)-np.log(len(mu))
    marginal = lse(c, 1)
    product = (marginal-np.log(len(mu))).sum(axis=1)
    w = np.exp(c.sum(axis=2)-lse(c.sum(axis=2), 1)[:, None])
    v = np.exp(c-marginal[:, None, :])
    dc = (w[:, :, None]-v)/len(z)
    return (joint, product, joint-product), (-(dc*delta*inv).sum(axis=1),
        (dc*delta*inv).sum(axis=0), (.5*dc*(delta*delta*inv-1)).sum(axis=0))


def discriminator(p, z):
    """Tanh hidden layer and signed linear density-ratio logit."""
    h = np.tanh(z@p['hidden/W']+p['hidden/b'])
    return (h@p['ratio/W']+p['ratio/b']).ravel(), h


def disc_adjoint(p, z, adjoint):
    """Parameter and input derivatives for arbitrary scalar logit adjoints."""
    _, h = discriminator(p, z)
    delta = adjoint[:, None]
    g = (delta@p['ratio/W'].T)*(1-h*h)
    return dict(zip(['hidden/W', 'hidden/b', 'ratio/W', 'ratio/b'],
                    [z.T@g, g.sum(axis=0), h.T@delta, delta.sum(axis=0)])), g@p['hidden/W'].T


def encoder_adjoint(p, x, h, dmu, dlv):
    """Both Gaussian-head and shared ReLU encoder derivatives for one image pool."""
    back = (dmu@p['mu/W'].T+dlv@p['log_variance/W'].T)*(h > 0)
    return {'mu/W': h.T@dmu, 'mu/b': dmu.sum(axis=0),
            'log_variance/W': h.T@dlv, 'log_variance/b': dlv.sum(axis=0),
            'encoder/W': x.T@back, 'encoder/b': back.sum(axis=0)}


def vae_derivatives(p, d, x, epsilon, refs, factor):
    """Base ELBO plus all sample and reference TC chain-rule derivatives."""
    result = derivatives(p, x, epsilon, 1)
    out, (h, sigma, _) = forward(p, x, epsilon)
    if factor:
        _, dz = disc_adjoint(d, out['z'], np.full(len(x), 3/len(x)))
    else:
        hr = np.maximum(refs@p['encoder/W']+p['encoder/b'], 0)
        mur, lvr = hr@p['mu/W']+p['mu/b'], hr@p['log_variance/W']+p['log_variance/b']
        _, (dz, dmu, dlv) = mixture(out['z'], mur, lvr)
        dz *= 3
        for key, value in encoder_adjoint(p, refs, hr, 3*dmu, 3*dlv).items():
            result[key] += value
    for key, value in encoder_adjoint(p, x.astype(np.float64), h, dz, .5*dz*sigma*epsilon).items():
        result[key] += value
    return result


def first_update(initial, gradients, updated):
    """Source-compatible Float32 Adam first update, verified by exact value bytes."""
    for key, g in gradients.items():
        first, second = g*np.float32(.1), (g*g)*np.float32(.001)
        direction = (first/np.float32(.1))/(np.sqrt(second/np.float32(.001))+np.float32(1e-8))
        exact(initial[key]-direction*np.float32(.003), updated[key])


def check(directory):
    """Validate complete evidence and return measured numerical errors and counts."""
    data, noise = arrays(directory/'data.npz'), arrays(directory/'fixed_noise.npz')
    pixels, noise_values, gradients_count = 0, 0, 0
    identities, unique_images = set(), set()
    for split, count, seed in [('fit', 210, 2273), ('validation', 42, 2267), ('test', 42, 2269)]:
        ids, factors, images = [data[split+s] for s in ['_source_ids', '_factors', '_images']]
        assert len(ids) == count and not identities.intersection(ids)
        identities.update(ids)
        yy, xx = np.mgrid[:28, :28]
        for row, identity in enumerate(ids):
            shape, size, x, y = int(identity)//147, (int(identity)//49)%3, (int(identity)//7)%7, int(identity)%7
            np.testing.assert_array_equal(factors[row], [shape, size, x, y])
            residue = (x+2*y+size+3*shape)%7
            assert residue == 0 if split == 'test' else residue == 1 if split == 'validation' else residue >= 2
            dx, dy = abs(xx-(3*x+5)), abs(yy-(3*y+5))
            expected = np.maximum(dx, dy) <= size+2 if shape else dx*dx+dy*dy <= (size+2)**2
            exact(expected.astype(np.float32)[..., None], images[row])
            unique_images.add(images[row].tobytes())
        pixels += images.size
        expected = gaussian((4, count, 4), seed)
        np.testing.assert_allclose(noise[split+'_standard'], expected, atol=1e-14, rtol=1e-14)
        exact(noise[split+'_epsilon'], expected.astype(np.float32));noise_values += expected.size
    assert len(identities) == len(unique_images) == 294
    expected = gaussian((1, 20, 4), 2281)
    np.testing.assert_allclose(noise['prior_standard'], expected, atol=1e-14, rtol=1e-14)
    exact(noise['prior_epsilon'], expected.astype(np.float32));noise_values += expected.size
    for epoch in range(1, 4):
        bank = arrays(directory/f'train_noise_epoch{epoch}.npz');expected = gaussian((1, 210, 4), 2251+epoch)
        np.testing.assert_allclose(bank['standard'], expected, atol=1e-14, rtol=1e-14)
        exact(bank['epsilon'], expected.astype(np.float32));noise_values += expected.size
    algebra = arrays(directory/'tc_algebra.npz')
    values, gradients = mixture(*[algebra[k].astype(np.float64) for k in ['z', 'mu', 'log_variance']])
    for key, value in zip(['log_joint', 'log_product', 'tc', 'dz', 'dmu', 'dlv'], values+gradients):
        np.testing.assert_allclose(algebra[key], value, atol=5e-7, rtol=1e-5)
    np.testing.assert_allclose(algebra['single_component_tc'], 0, atol=1e-6)
    assert np.isfinite(algebra['extreme_tc']).all()
    refs = data['fit_images'].reshape(-1, 784).astype(np.float64)
    evidence = []
    for factor, name in [(True, 'factor_vae'), (False, 'beta_tcvae')]:
        root = directory/name
        initial = arrays(root/'initial_arrays.npz');p = {k: v.astype(np.float64) for k, v in initial.items()}
        d = {k: v.astype(np.float64) for k, v in arrays(root/'initial_discriminator.npz').items()} if factor else None
        batch, actual, updated = [arrays(root/(k+'.npz')) for k in ['first_batch', 'first_gradients', 'first_update_arrays']]
        expected = vae_derivatives(p, d, batch['images'].reshape(-1, 784), batch['epsilon'], refs, factor)
        gradient_error = 0
        assert set(expected) == set(actual)
        for key, value in expected.items():
            np.testing.assert_allclose(actual[key], value, atol=3e-6, rtol=8e-4)
            gradient_error = max(gradient_error, float(np.max(np.abs(actual[key]-value))));gradients_count += value.size
        first_update(initial, actual, updated)
        disc_error = None
        if factor:
            db, dg, du = [arrays(root/(k+'.npz')) for k in ['first_discriminator_batch', 'first_discriminator_gradients', 'first_discriminator_update']]
            # The first D samples use the already updated VAE and independent fitting rows/noise.
            o, _ = forward({k: v.astype(np.float64) for k, v in updated.items()}, db['images'].reshape(-1, 784), db['epsilon'])
            np.testing.assert_allclose(db['joint'], o['z'], atol=2e-6, rtol=1e-5)
            for axis in range(4):exact(np.sort(db['joint'][:, axis]), np.sort(db['product'][:, axis]))
            jl, _ = discriminator(d, db['joint']);pl, _ = discriminator(d, db['product'])
            gj, _ = disc_adjoint(d, db['joint'], .5*(1/(1+np.exp(-jl))-1)/len(jl))
            gp, _ = disc_adjoint(d, db['product'], .5/(1+np.exp(-pl))/len(pl))
            disc_error = 0
            for key in gj:
                value = gj[key]+gp[key]
                np.testing.assert_allclose(dg[key], value, atol=1e-7, rtol=1e-4)
                disc_error = max(disc_error, float(np.max(np.abs(dg[key]-value))));gradients_count += value.size
            first_update(arrays(root/'initial_discriminator.npz'), dg, du)
        for epoch in range(1, 4):
            bank = arrays(root/f'discriminator_noise_epoch{epoch}.npz');expected = gaussian((1, 210, 4), 2333+epoch)
            np.testing.assert_allclose(bank['standard'], expected, atol=1e-14, rtol=1e-14)
            exact(bank['epsilon'], expected.astype(np.float32));noise_values += expected.size
        output, selected = arrays(root/'outputs.npz'), arrays(root/'model_arrays.npz')
        p = {k: v.astype(np.float64) for k, v in selected.items()}
        d = {k: v.astype(np.float64) for k, v in arrays(root/'model_discriminator.npz').items()} if factor else None
        hr = np.maximum(refs@p['encoder/W']+p['encoder/b'], 0)
        mur, lvr = hr@p['mu/W']+p['mu/b'], hr@p['log_variance/W']+p['log_variance/b']
        prediction_error, tc_error = 0, 0
        for split in ['fit', 'validation', 'test']:
            x = data[split+'_images'].reshape(-1, 784)
            mean, _ = forward(p, x, np.zeros((len(x), 4)))
            for field, key in [('mu', 'mu'), ('log_variance', 'log_variance'), ('probability', 'mean_prediction'), ('kl', 'kl')]:
                np.testing.assert_allclose(output[split+'_'+key], mean[field], atol=5e-5, rtol=1e-5)
            for k, epsilon in enumerate(noise[split+'_epsilon']):
                o, _ = forward(p, x, epsilon)
                for field, key in [('z', 'sample_z'), ('probability', 'sample_prediction'), ('nll', 'sample_nll')]:
                    np.testing.assert_allclose(output[split+'_'+key][k], o[field], atol=4e-4, rtol=1e-5)
                tc = discriminator(d, o['z'])[0].mean() if factor else mixture(o['z'], mur, lvr)[0][2].mean()
                np.testing.assert_allclose(output[split+'_sample_tc'][k], tc, atol=8e-5, rtol=1e-5)
                tc_error = max(tc_error, abs(float(output[split+'_sample_tc'][k])-tc))
                prediction_error = max(prediction_error, float(np.max(np.abs(o['probability']-output[split+'_sample_prediction'][k]))));pixels += o['probability'].size
        prior = arrays(root/'prior.npz');exact(prior['z'], noise['prior_epsilon'][0])
        center = output['fit_mu'].astype(np.float64).mean(axis=0).astype(np.float32);exact(center, prior['traversal_center'])
        traversal = np.tile(center, (36, 1))
        for axis in range(4):traversal[axis*9:(axis+1)*9, axis] += np.arange(-2, 2.01, .5, dtype=np.float32)
        exact(traversal, prior['traversal_z'])
        for codes, key in [('z', 'probability'), ('traversal_z', 'traversal_probability')]:
            _, prediction, _ = decode(p, prior[codes].astype(np.float64))
            np.testing.assert_allclose(prior[key], prediction, atol=1e-5, rtol=1e-5);pixels += prediction.size
        expected = probes(output['fit_mu'], output['test_mu'], data['fit_factors'], data['test_factors'], output['fit_log_variance'])
        for key, value in arrays(root/'factor_probes.npz').items():
            np.testing.assert_allclose(value, expected[key], atol=2e-11, rtol=2e-11)
        history = np.genfromtxt(root/'history.tsv', delimiter='\t', names=True);scores = []
        for epoch in range(1, 4):
            ep = {k: v.astype(np.float64) for k, v in arrays(root/f'epoch{epoch}_arrays.npz').items()}
            ed = {k: v.astype(np.float64) for k, v in arrays(root/f'epoch{epoch}_discriminator.npz').items()} if factor else None
            eh = np.maximum(refs@ep['encoder/W']+ep['encoder/b'], 0)
            em, el = eh@ep['mu/W']+ep['mu/b'], eh@ep['log_variance/W']+ep['log_variance/b']
            x = data['validation_images'].reshape(-1, 784);nll, tc = 0, 0
            for epsilon in noise['validation_epsilon']:
                o, _ = forward(ep, x, epsilon);nll += o['nll'].mean()/4
                tc += (discriminator(ed, o['z'])[0].mean() if factor else mixture(o['z'], em, el)[0][2].mean())/4
            kl = o['kl'].mean();scores.append(nll+kl)
            for key, value in [('validation_nll', nll), ('validation_kl', kl), ('validation_negative_elbo', nll+kl), ('validation_weighted_objective', nll+kl+3*tc)]:
                np.testing.assert_allclose(history[key][epoch-1], value, atol=4e-4, rtol=1e-6)
        chosen = int(np.argmin(scores)+1)
        for key, value in arrays(root/f'epoch{chosen}_arrays.npz').items():exact(value, selected[key])
        if factor:
            for key, value in arrays(root/f'epoch{chosen}_discriminator.npz').items():exact(value, arrays(root/'model_discriminator.npz')[key])
        metrics = json.loads((root/'metrics.json').read_text());assert metrics['selected_epoch'] == chosen and metrics['updates'] == 21
        nll = output['test_sample_nll'].astype(np.float64).mean();kl = output['test_kl'].astype(np.float64).mean()
        np.testing.assert_allclose([metrics['test_nll'], metrics['test_kl'], metrics['test_negative_elbo'], metrics['test_weighted_objective']],
                                   [nll, kl, nll+kl, nll+kl+3*output['test_sample_tc'].astype(np.float64).mean()], atol=1e-12, rtol=0)
        batches = np.genfromtxt(root/'batches.tsv', delimiter='\t', names=True, dtype=int)
        assert set(batches['step']) == set(range(21))
        for epoch in range(1, 4):np.testing.assert_array_equal(np.sort(batches['row'][batches['epoch'] == epoch]), np.arange(210))
        np.testing.assert_array_equal(batches['source_id'], data['fit_source_ids'][batches['row']])
        evidence.append(dict(model=name, selected_epoch=chosen, first_VAE_gradient_max_error=gradient_error,
            first_discriminator_gradient_max_error=disc_error, prediction_max_error=prediction_error, tc_max_error=tc_error,
            first_Adam_updates_bit_exact=True, thirteen_probe_arrays_checked=True))
    for key, value in arrays(directory/'factor_vae/initial_arrays.npz').items():exact(value, arrays(directory/'beta_tcvae/initial_arrays.npz')[key])
    return dict(all_first_batch_derivatives_checked=gradients_count, Gaussian_values_checked=noise_values,
        lattice_and_prediction_pixels_checked=pixels, eligible_validation_checkpoints_checked=6, models=evidence)


def main():
    """Run the optional development oracle and save its compact validation receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path);parser.add_argument('--json', type=Path)
    args = parser.parse_args();result = check(args.directory)
    if args.json:args.json.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
