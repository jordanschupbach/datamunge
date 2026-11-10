#!/usr/bin/env python3
"""Independent NumPy audit of the pinned original NICE density experiment.

Uses analytic coupling derivatives and matrix Jacobian composition, without
datamunge, autograd, refitting or replacement data. Captured Float32 gradients
are independently checked before replaying every original Adam write exactly.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from check_adversarial_autoencoder import arrays, exact, floating


LOG_2PI = float(np.float32(math.log(2 * math.pi)))


def gaussian(count, seed):
    """Scalar Box-Muller pairs from raw midpoint MT19937 words, in Float64."""
    words = np.random.RandomState(seed).randint(0, 2**32, (count, 2), dtype=np.uint32)
    result = np.empty((count, 2))
    for i, (a, b) in enumerate(words):
        u, v = (int(a) + .5) / 2**32, (int(b) + .5) / 2**32
        radius, angle = math.sqrt(-2 * math.log(u)), 2 * math.pi * v
        result[i] = radius * math.cos(angle), radius * math.sin(angle)
    return result


def coupling(p, x, block, inverse=False):
    name, condition = f'coupling{block}', block % 2
    a = x[:, condition:condition + 1]
    hidden = np.tanh(a @ p[name + '/hidden/W'] + p[name + '/hidden/b'])
    shift = hidden @ p[name + '/shift/W'] + p[name + '/shift/b']
    sign = -1 if inverse else 1
    value = x.copy()
    value[:, 1 - condition:2 - condition] += sign * shift
    slope = ((1 - hidden * hidden) * p[name + '/hidden/W']) @ p[name + '/shift/W']
    jacobian = np.broadcast_to(np.eye(2), (len(x), 2, 2)).copy()
    jacobian[:, 1 - condition, condition] = sign * slope[:, 0]

    def backward(dy, partials):
        dshift = sign * dy[:, 1 - condition:2 - condition]
        partials[name + '/shift/W'] = hidden.T @ dshift
        partials[name + '/shift/b'] = dshift.sum(axis=0)
        dhidden = (dshift @ p[name + '/shift/W'].T) * (1 - hidden * hidden)
        partials[name + '/hidden/W'] = a.T @ dhidden
        partials[name + '/hidden/b'] = dhidden.sum(axis=0)
        dx = dy.copy()
        dx[:, condition:condition + 1] += dhidden @ p[name + '/hidden/W'].T
        return dx

    return value, jacobian, backward


def forward(p, x):
    h = x.astype(float)
    jacobian = np.broadcast_to(np.eye(2), (len(x), 2, 2)).copy()
    fields, backwards = {}, []
    for block in range(4):
        h, local, back = coupling(p, h, block)
        jacobian = local @ jacobian
        fields[f'coupling{block}'] = h
        backwards.append(back)
    scale = np.exp(p['log_scale'])
    z, jacobian = h * scale, jacobian * scale[None, :, None]
    logdet = p['log_scale'].sum()
    base = -(.5 * (z * z).sum(axis=1) + LOG_2PI)
    fields.update(latent=z, unscaled=h, forward_logdet=np.asarray(logdet),
                  base_log_probability=base, log_probability=base + logdet,
                  loss=np.asarray(-(base + logdet).mean()),
                  jacobian_row0=jacobian[:, 0], jacobian_row1=jacobian[:, 1])

    def backward():
        dy = z / len(z)
        partials = {'log_scale': (dy * z).sum(axis=0) - 1}
        dy = dy * scale
        for back in reversed(backwards):
            dy = back(dy, partials)
        return partials, dy

    return fields, backward


def inverse(p, z):
    scale = np.exp(-p['log_scale'])
    h = z.astype(float) * scale
    jacobian = np.broadcast_to(np.diag(scale), (len(z), 2, 2)).copy()
    for block in reversed(range(4)):
        h, local, _ = coupling(p, h, block, True)
        jacobian = local @ jacobian
    return h, -p['log_scale'].sum(), jacobian


def evaluate(p, x):
    result, _ = forward(p, x)
    rec, logdet, _ = inverse(p, result['latent'])
    return dict(result, reconstruction=rec, inverse_logdet=np.asarray(logdet))


def sample(p, z):
    x, logdet, jacobian = inverse(p, z)
    return dict(prior=z, samples=x, prior_roundtrip=forward(p, x)[0]['latent'],
                inverse_logdet=np.asarray(logdet), inverse_jacobian_row0=jacobian[:, 0],
                inverse_jacobian_row1=jacobian[:, 1])


def truth(x):
    u, v = x[:, 0].astype(float), x[:, 1].astype(float)
    return -math.log(2 * math.pi) - .5 * (u * u + (v - .5 * (u * u - 1))**2)


def check(root):
    pins = json.loads((root / 'baseline.json').read_text())
    source = root / 'original-output'
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for file, digest in pins['source_pins'].items():
        assert sha(root / ('original-' + Path(file).name)) == digest
    for group, directory in [('archive_sha256', source), ('other_files_sha256', source),
                             ('phase_probe_sha256', root)]:
        for file, digest in pins[group].items():
            assert sha(directory / file) == digest
    assert not pins['prepared_input_sha256'] and pins['optimizer_updates'] == 48
    counts = dict(data_values=0, forward_values=0, gradient_values=0, optimizer_writes=0,
                  exact_weight_values=0, finite_differences=0, batch_rows=0,
                  forward_Jacobian_entries=0, inverse_Jacobian_entries=0, prior_samples=0)
    errors = dict(forward=0., gradient=0., finite_difference=0., normalization=0.)
    audited = set()

    def load(name):
        audited.add(name + '.npz')
        return arrays(source / (name + '.npz'))

    def compare(actual, expected, label, gradient=False):
        assert set(actual) == set(expected), (label, set(actual) ^ set(expected))
        kind = 'gradient' if gradient else 'forward'
        for key, value in expected.items():
            np.testing.assert_allclose(actual[key], value, atol=1e-6 if gradient else 4e-6,
                                       rtol=1e-4 if gradient else 2e-5, err_msg=label + '/' + key)
            counts[kind + '_values'] += value.size
            errors[kind] = max(errors[kind], float(np.max(np.abs(actual[key] - value))))

    data = load('data')
    for index, (split, n, seed) in enumerate([('fit', 512, 9203), ('validation', 128, 9209), ('test', 128, 9221)]):
        base = gaussian(n, seed)
        exact(base, data[split + '_base'])
        points = base.copy()
        points[:, 1] += .5 * (points[:, 0]**2 - 1)
        exact(points.astype(np.float32), data[split + '_points'])
        exact(np.arange(n, dtype=np.int32) + index * 10000, data[split + '_source_ids'])
        counts['data_values'] += n * 5
    prior = gaussian(1024, 9257).astype(np.float32)
    exact(prior, data['sample_prior'])
    counts['data_values'] += prior.size
    assert len(data) == 10
    protocol = json.loads((source / 'source_protocol.json').read_text())
    assert protocol['seeds'] == [9203, 9209, 9221] and protocol['sample_prior_seed'] == 9257
    p0, pf = load('initial_arrays'), load('final')
    assert len(p0) == 17 and sum(v.size for v in p0.values()) == 198
    for i in range(4):
        assert p0[f'coupling{i}/hidden/W'].shape == (1, 16)
        assert p0[f'coupling{i}/shift/W'].shape == (16, 1)
        exact(p0[f'coupling{i}/hidden/b'], np.full(16, .05, np.float32))
        exact(p0[f'coupling{i}/shift/b'], np.zeros(1, np.float32))
    exact(p0['log_scale'], np.zeros(2, np.float32))
    batch = np.genfromtxt(source / 'batches.tsv', names=True, delimiter='\t', dtype=np.int64)
    assert len(batch) == 1536 and np.array_equal(np.unique(batch['step']), np.arange(48))
    counts['batch_rows'] = len(batch)
    for epoch in [1, 2, 3]:
        exact(np.sort(batch[batch['epoch'] == epoch]['row']), np.arange(512))
    first = load('first_batch')['points']
    exact(first, data['fit_points'][batch[batch['step'] == 0]['row']])
    for name, p, x in [('initial_outputs', p0, data['test_points']), ('first_forward', p0, first),
                       ('outputs', pf, data['test_points'])]:
        compare(load(name), evaluate(floating(p), x), name)
    before = p0
    moment, variance = [{k: np.zeros_like(v) for k, v in p0.items()} for _ in range(2)]
    for step in range(48):
        rows = batch[batch['step'] == step]['row']
        assert len(rows) == 32 and np.all(batch[batch['step'] == step]['epoch'] == step // 16 + 1)
        _, back = forward(floating(before), data['fit_points'][rows])
        partials, _ = back()
        gradients = arrays(root / 'phases' / f'batch{step}-gradient.npz')
        compare(gradients, partials, f'batch{step}', True)
        after = arrays(root / 'phases' / f'batch{step}-after.npz')
        for key, value in before.items():
            g = gradients[key]
            moment[key] = moment[key] * np.float32(.9) + g * np.float32(.1)
            variance[key] = variance[key] * np.float32(.999) + (g * g) * np.float32(.001)
            m = moment[key] / np.float32(1 - .9**(step + 1))
            v = variance[key] / np.float32(1 - .999**(step + 1))
            exact(value - m / (np.sqrt(v) + np.float32(1e-8)) * np.float32(.003), after[key])
            counts['exact_weight_values'] += value.size
        exact(arrays(root / 'phases' / f'batch{step}-iterations.npz')['iterations'], np.array([step + 1], np.int64))
        if step == 0:
            compare(load('initial_gradients'), partials, 'initial-gradients', True)
            for key, value in load('first_after').items():
                exact(value, after[key])
        if step % 16 == 15:
            for key, value in load(f'epoch{step // 16 + 1}_arrays').items():
                exact(value, after[key])
        before = after
        counts['optimizer_writes'] += 1
    for key, value in before.items():
        exact(value, pf[key])
        exact(value, load('model_arrays')[key])
    final = floating(pf)
    _, back = forward(final, first)
    partials, dx = back()
    compare(load('final_gradients'), partials, 'final-gradients', True)
    for key, value in final.items():
        index = np.unravel_index(np.argmax(abs(partials[key])), value.shape)
        hi, lo = [{k: v.copy() for k, v in final.items()} for _ in range(2)]
        hi[key][index] += 1e-5
        lo[key][index] -= 1e-5
        fd = (forward(hi, first)[0]['loss'] - forward(lo, first)[0]['loss']) / 2e-5
        np.testing.assert_allclose(fd, partials[key][index], atol=2e-9, rtol=2e-6, err_msg=key)
        errors['finite_difference'] = max(errors['finite_difference'], float(abs(fd - partials[key][index])))
        counts['finite_differences'] += 1
    history = np.loadtxt(source / 'history.tsv', skiprows=1)
    for epoch in [1, 2, 3]:
        p = floating(load(f'epoch{epoch}_arrays'))
        out = evaluate(p, data['validation_points'])
        compare(load(f'epoch{epoch}_validation'), out, f'epoch{epoch}')
        np.testing.assert_allclose(history[epoch - 1, 1:], [out['loss'], out['forward_logdet']], atol=8e-7, rtol=0)
    sampled = sample(final, prior)
    compare(load('samples'), sampled, 'samples')
    compare(load('sample_density'), evaluate(final, sampled['samples']), 'sample-density')
    counts['prior_samples'] = len(prior)
    test = data['test_points']
    observed = load('outputs')
    jac = np.stack([observed['jacobian_row0'], observed['jacobian_row1']], axis=1).astype(float)
    ijac = np.stack([load('samples')['inverse_jacobian_row0'], load('samples')['inverse_jacobian_row1']], axis=1).astype(float)
    for matrices, logdet in [(jac, observed['forward_logdet']), (ijac, sampled['inverse_logdet'])]:
        assert np.all(np.linalg.det(matrices) > 0)
        np.testing.assert_allclose(np.linalg.slogdet(matrices)[1], logdet, atol=3e-7, rtol=0)
    counts['forward_Jacobian_entries'], counts['inverse_Jacobian_entries'] = jac.size, ijac.size
    for coordinate in range(2):
        hi, lo = test.astype(float), test.astype(float)
        hi[:, coordinate] += 1e-5
        lo[:, coordinate] -= 1e-5
        fd = (forward(final, hi)[0]['latent'] - forward(final, lo)[0]['latent']) / 2e-5
        np.testing.assert_allclose(fd, jac[:, :, coordinate], atol=2e-7, rtol=1e-5)
    roundtrip = forward(final, sampled['samples'])[0]['latent']
    np.testing.assert_allclose(roundtrip, prior, atol=2e-14, rtol=0)
    logp = evaluate(final, sampled['samples'])['log_probability']
    base = -(.5 * (prior.astype(float)**2).sum(axis=1) + LOG_2PI)
    errors['normalization'] = float(abs(logp + sampled['inverse_logdet'] - base).max())
    assert errors['normalization'] < 2e-14
    grid = load('grid')
    coordinates = np.array([[(x + .5) / 16 - 4, 4 - (y + .5) / 16]
                            for y in range(128) for x in range(128)], dtype=np.float32)
    exact(grid['points'], coordinates)
    compare({'model_log_probability': grid['model_log_probability']},
            {'model_log_probability': forward(final, coordinates)[0]['log_probability']}, 'grid')
    np.testing.assert_allclose(grid['true_log_probability'], truth(coordinates), atol=1e-14, rtol=0)
    baseline = load('baseline')
    fit = data['fit_points'].astype(float)
    mean = fit.mean(axis=0)
    covariance = (fit - mean).T @ (fit - mean) / len(fit)
    np.testing.assert_allclose(baseline['fit_mean'], mean, atol=1e-15, rtol=0)
    np.testing.assert_allclose(baseline['fit_covariance'], covariance, atol=4e-15, rtol=0)
    np.testing.assert_allclose(baseline['true_test_log_probability'], truth(test), atol=1e-14, rtol=0)
    metrics = json.loads((source / 'metrics.json').read_text())
    a = test.astype(float) - mean
    full_nll = math.log(2 * math.pi) + .5 * np.linalg.slogdet(covariance)[1] + .5 * np.einsum('ni,ij,nj->n', a, np.linalg.inv(covariance), a).mean()
    diagonal_nll = math.log(2 * math.pi) + .5 * np.log(np.diag(covariance)).sum() + .5 * (a * a / np.diag(covariance)).sum(axis=1).mean()
    identity_nll = math.log(2 * math.pi) + .5 * (test.astype(float)**2).sum(axis=1).mean()
    np.testing.assert_allclose([full_nll, diagonal_nll, identity_nll, -truth(test).mean()],
        [metrics[k] for k in ['fit_full_gaussian_nll', 'fit_diagonal_gaussian_nll', 'identity_gaussian_nll', 'true_test_nll']], atol=4e-15, rtol=0)
    assert metrics['test_nll'] > metrics['fit_full_gaussian_nll'] > metrics['true_test_nll']
    assert audited == set(pins['archive_sha256']), audited ^ set(pins['archive_sha256'])
    return dict(passed=True, counts=counts, maximum_errors=errors, archives_checked=len(audited),
                raw_midpoint_MT_Box_Muller_and_all_banana_rows_exact=True,
                all_48_Adam_writes_and_iterations_bit_exact=True,
                all_17_leaf_finite_differences=True, analytic_coupling_matrix_Jacobians=True,
                positive_forward_and_inverse_determinants_and_global_logdet=True,
                all_1024_prior_inverse_samples_and_normalization_identities=True,
                fitting_only_Gaussian_baselines_and_analytic_truth_checked=True,
                original_rounded_Float32_log_2pi_constant_preserved=LOG_2PI,
                short_fit_worse_than_fitting_only_Gaussian_preserved=True,
                test_nll=metrics['test_nll'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('build/nn-report-conversion/nice_density'))
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    rendered = json.dumps(check(args.root), indent=2) + '\n'
    print(rendered, end='')
    if args.json:
        args.json.write_text(rendered)


if __name__ == '__main__':
    main()
