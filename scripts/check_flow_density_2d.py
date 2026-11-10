#!/usr/bin/env python3
"""Independently audit the original three-family RealNVP density study.

Explicit NumPy couplings, analytic matrix Jacobians and hand-derived pullbacks
come from the independent RealNVP oracle. Rebuild raw MT19937/Box-Muller data,
all three transforms/truths and quality diagnostics without datamunge, external
autodiff, refitting or replacement observations. Check every saved optimizer
write, all 266 parameter coordinates per family and all source archives.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from check_adversarial_autoencoder import arrays, exact, floating
from check_nice_density import gaussian
from check_real_nvp_density import forward, inverse, evaluate, sample, LOG_2PI


def gaussian3(count, seed):
    """Preserve the flat pair stream across row boundaries, including odd counts."""
    size = count * 3
    words = np.random.RandomState(seed).randint(0, 2**32, ((size + 1)//2, 2), dtype=np.uint32)
    result = []
    for a, b in words:
        u, v = (int(a) + .5) / 2**32, (int(b) + .5) / 2**32
        radius, angle = math.sqrt(-2 * math.log(u)), 2 * math.pi * v
        result.extend([radius * math.cos(angle), radius * math.sin(angle)])
    return np.array(result[:size]).reshape(count, 3)


def phi(z):
    return .5 * math.erfc(-float(z) / math.sqrt(2.))


def observations(base, kind):
    components = np.minimum(7, np.array([int(8 * phi(z)) for z in base[:, 2]], np.int32))
    if kind == 0:
        points = np.column_stack([base[:, 0], base[:, 1] + .5 * (base[:, 0]**2 - 1)])
        components.fill(-1)
    elif kind == 1:
        points = np.array([[2 * math.cos(2*math.pi*c/8) + .15*u,
                            2 * math.sin(2*math.pi*c/8) + .15*v]
                           for (u,v,_),c in zip(base, components)])
    else:
        points = np.array([[2*(c%2) + (c//2)%2 - 2 + phi(u), c//2 - 2 + phi(v)]
                           for (u,v,_),c in zip(base, components)])
    return points.astype(np.float32), components


def checker_support(x, y):
    return -2 <= x < 2 and -2 <= y < 2 and (math.floor(x+2) + math.floor(y+2)) % 2 == 0


def true_log_probability(points, kind):
    result = []
    for x,y in points.astype(float):
        if kind == 0:
            v = y - .5*(x*x - 1)
            result.append(-math.log(2*math.pi) - .5*(x*x + v*v))
        elif kind == 2:
            result.append(-math.log(8.) if checker_support(x,y) else -math.inf)
        else:
            terms = [-math.log(8*2*math.pi*.15*.15) - .5*((x-2*math.cos(2*math.pi*j/8))**2
                      +(y-2*math.sin(2*math.pi*j/8))**2)/(.15*.15) for j in range(8)]
            largest = max(terms)
            result.append(largest + math.log(sum(math.exp(v-largest) for v in terms)))
    return np.array(result)


def diagnostics(points):
    support, curvature, counts = 0., 0., np.zeros(8, np.int32)
    for x,y in points.astype(float):
        support += checker_support(x,y)/len(points)
        curvature += y*(x*x-1)/len(points)
        distances = [math.hypot(x-2*math.cos(2*math.pi*j/8), y-2*math.sin(2*math.pi*j/8)) for j in range(8)]
        nearest = int(np.argmin(distances))
        if distances[nearest] <= .45:
            counts[nearest] += 1
    return dict(mode_counts=counts, checkerboard_support_fraction=support,
                eight_gaussian_fraction_within_three_sigma_of_centers=int(counts.sum())/len(points),
                eight_gaussian_modes_covered_at_three_sigma=int(np.count_nonzero(counts)),
                banana_curvature_moment=curvature)


def audit_variant(root, pins, name, kind):
    source = root / 'original-output' / name
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
        base = gaussian3(n, seed)
        exact(base, data[split + '_base'])
        points, components = observations(base, kind)
        exact(points, data[split + '_points'])
        exact(components, data[split + '_components'])
        exact(np.arange(n, dtype=np.int32) + index * 10000 + kind * 100000,
              data[split + '_source_ids'])
        counts['data_values'] += n * 7
    prior = gaussian(1024, 9257).astype(np.float32)
    exact(prior, data['sample_prior'])
    counts['data_values'] += prior.size
    assert len(data) == 13
    protocol = json.loads((source / 'source_protocol.json').read_text())
    assert protocol['seeds'] == [9203, 9209, 9221] and protocol['sample_prior_seed'] == 9257
    assert protocol['base'] == 'standardGaussian3D' and protocol['source_id_family_offset'] == kind * 100000
    assert protocol['epochs_per_dataset'] == 3 and protocol['component_rule'] == 'floor(8*Phi(base2))'
    p0, pf = load('initial_arrays'), load('final')
    assert len(p0) == 17 and sum(v.size for v in p0.values()) == 266
    for i in range(4):
        assert p0[f'coupling{i}/hidden/W'].shape == (1, 16)
        assert p0[f'coupling{i}/heads/W'].shape == (16, 2)
        exact(p0[f'coupling{i}/hidden/b'], np.full(16, .05, np.float32))
        exact(p0[f'coupling{i}/heads/b'], np.zeros(2, np.float32))
    exact(p0['log_scale'], np.zeros(2, np.float32))
    batch = np.genfromtxt(source / 'batches.tsv', names=True, delimiter='\t', dtype=np.int64)
    assert len(batch) == 1536 and np.array_equal(np.unique(batch['step']), np.arange(48))
    counts['batch_rows'] = len(batch)
    for epoch in [1, 2, 3]:
        exact(np.sort(batch[batch['epoch'] == epoch]['row']), np.arange(512))
    first = load('first_batch')['points']
    exact(first, data['fit_points'][batch[batch['step'] == 0]['row']])
    for output_name, p, x in [('initial_outputs', p0, data['test_points']), ('first_forward', p0, first),
                       ('outputs', pf, data['test_points'])]:
        compare(load(output_name), evaluate(floating(p), x), output_name)
    before = p0
    moment, variance = [{k: np.zeros_like(v) for k, v in p0.items()} for _ in range(2)]
    for step in range(48):
        rows = batch[batch['step'] == step]['row']
        assert len(rows) == 32 and np.all(batch[batch['step'] == step]['epoch'] == step // 16 + 1)
        _, back = forward(floating(before), data['fit_points'][rows])
        partials, _ = back()
        gradients = arrays(root / 'phases' / name / f'batch{step}-gradient.npz')
        compare(gradients, partials, f'batch{step}', True)
        after = arrays(root / 'phases' / name / f'batch{step}-after.npz')
        for key, value in before.items():
            g = gradients[key]
            moment[key] = moment[key] * np.float32(.9) + g * np.float32(.1)
            variance[key] = variance[key] * np.float32(.999) + (g * g) * np.float32(.001)
            m = moment[key] / np.float32(1 - .9**(step + 1))
            v = variance[key] / np.float32(1 - .999**(step + 1))
            exact(value - m / (np.sqrt(v) + np.float32(1e-8)) * np.float32(.003), after[key])
            counts['exact_weight_values'] += value.size
        exact(arrays(root / 'phases' / name / f'batch{step}-iterations.npz')['iterations'], np.array([step + 1], np.int64))
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
        for index in np.ndindex(value.shape):
            hi, lo = [{k: v.copy() for k, v in final.items()} for _ in range(2)]
            hi[key][index] += 1e-5
            lo[key][index] -= 1e-5
            fd = (forward(hi, first)[0]['loss'] - forward(lo, first)[0]['loss']) / 2e-5
            np.testing.assert_allclose(fd, partials[key][index], atol=2e-9, rtol=2e-6,
                                       err_msg=name + '/' + key + '/' + str(index))
            errors['finite_difference'] = max(errors['finite_difference'], float(abs(fd - partials[key][index])))
            counts['finite_differences'] += 1
    history = np.loadtxt(source / 'history.tsv', skiprows=1)
    for epoch in [1, 2, 3]:
        p = floating(load(f'epoch{epoch}_arrays'))
        out = evaluate(p, data['validation_points'])
        compare(load(f'epoch{epoch}_validation'), out, f'epoch{epoch}')
        np.testing.assert_allclose(history[epoch - 1, 1:], [out['loss'], out['forward_logdet'].mean()], atol=8e-7, rtol=0)
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
        np.testing.assert_allclose(np.linalg.slogdet(matrices)[1], logdet, atol=1e-6, rtol=0)
    counts['forward_Jacobian_entries'], counts['inverse_Jacobian_entries'] = jac.size, ijac.size
    assert np.ptp(observed['forward_logdet']) > 1e-5
    assert np.ptp(sampled['inverse_logdet']) > 1e-5
    np.testing.assert_allclose(observed['forward_logdet'] + observed['inverse_logdet'], 0, atol=5e-6, rtol=0)
    density_at_samples = forward(final, sampled['samples'])[0]
    np.testing.assert_allclose(density_at_samples['forward_logdet'] + sampled['inverse_logdet'], 0, atol=1e-14, rtol=0)
    for block in range(4):
        assert np.max(abs(observed[f'coupling_scale{block}'])) <= .8
        np.testing.assert_allclose(observed[f'coupling_scale{block}'][:, 0], observed[f'coupling_logdet{block}'], atol=0, rtol=0)

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
    np.testing.assert_allclose(grid['true_log_probability'], true_log_probability(coordinates, kind), atol=1e-14, rtol=0)
    baseline = load('baseline')
    fit = data['fit_points'].astype(float)
    mean = fit.mean(axis=0)
    covariance = (fit - mean).T @ (fit - mean) / len(fit)
    np.testing.assert_allclose(baseline['fit_mean'], mean, atol=1e-15, rtol=0)
    np.testing.assert_allclose(baseline['fit_covariance'], covariance, atol=4e-15, rtol=0)
    np.testing.assert_allclose(baseline['true_test_log_probability'], true_log_probability(test, kind), atol=1e-14, rtol=0)
    metrics = json.loads((source / 'metrics.json').read_text())
    a = test.astype(float) - mean
    full_nll = math.log(2 * math.pi) + .5 * np.linalg.slogdet(covariance)[1] + .5 * np.einsum('ni,ij,nj->n', a, np.linalg.inv(covariance), a).mean()
    diagonal_nll = math.log(2 * math.pi) + .5 * np.log(np.diag(covariance)).sum() + .5 * (a * a / np.diag(covariance)).sum(axis=1).mean()
    identity_nll = math.log(2 * math.pi) + .5 * (test.astype(float)**2).sum(axis=1).mean()
    np.testing.assert_allclose([full_nll, diagonal_nll, identity_nll, -true_log_probability(test, kind).mean()],
        [metrics[k] for k in ['fit_full_gaussian_nll', 'fit_diagonal_gaussian_nll', 'identity_gaussian_nll', 'true_test_nll']], atol=1e-14, rtol=0)
    diagnostic = diagnostics(load('samples')['samples'])
    recorded = json.loads((source / 'sample_diagnostics.json').read_text())
    exact(load('sample_diagnostics')['mode_counts_within_three_sigma'], diagnostic.pop('mode_counts'))
    assert recorded == diagnostic, (name, recorded, diagnostic)
    counts['diagnostic_sample_rows'] = 1024
    counts['truth_grid_rows'] = 16384
    counts['true_zero_density_grid_rows'] = int(np.isneginf(grid['true_log_probability']).sum())
    assert audited == {k[len(name)+1:] for k in pins['archive_sha256'] if k.startswith(name + '/')}
    return dict(passed=True, counts=counts, maximum_errors=errors, archives_checked=len(audited),
                all_266_coordinate_NLL_finite_differences=True,
                all_48_Adam_writes_and_iterations_bit_exact=True,
                actual_forward_and_inverse_matrix_Jacobians=True,
                all_prior_samples_and_normalization_identities=True,
                source_truth_including_legitimate_negative_infinity=True,
                source_sample_diagnostics_exact=recorded,
                source_metrics={k:v for k,v in metrics.items() if k!='train_seconds'})


def check(root):
    pins = json.loads((root/'baseline.json').read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for file, digest in pins['source_pins'].items():
        assert sha(root/('original-'+Path(file).name)) == digest
    for group, folder in [('archive_sha256', root/'original-output'),
                          ('other_files_sha256',root/'original-output'),('phase_probe_sha256',root)]:
        for file, digest in pins[group].items():
            assert sha(folder/file) == digest
    assert pins['optimizer_updates'] == 144 and not pins['prepared_input_sha256']
    variants = {name:audit_variant(root,pins,name,kind) for kind,name in enumerate(pins['variants'])}
    counts = {k:sum(v['counts'][k] for v in variants.values()) for k in variants['banana']['counts']}
    assert counts['optimizer_writes']==144 and counts['finite_differences']==798
    inits = [arrays(root/'original-output'/n/'initial_arrays.npz') for n in variants]
    orders = [(root/'original-output'/n/'batches.tsv').read_bytes() for n in variants]
    for init,order in zip(inits[1:],orders[1:]):
        assert orders[0]==order
        for key in inits[0]:
            exact(inits[0][key],init[key])
    summary = (root/'original-output/summary.tsv').read_text().splitlines()
    assert len(summary)==4
    for line,(name,variant) in zip(summary[1:],variants.items()):
        fields=line.split('\t');assert fields[0]==name
        m,d=variant['source_metrics'],variant['source_sample_diagnostics_exact']
        selected = d['banana_curvature_moment'] if name=='banana' else d['eight_gaussian_fraction_within_three_sigma_of_centers'] if name=='eight_gaussians' else d['checkerboard_support_fraction']
        np.testing.assert_array_equal([float(v) for v in fields[1:]], [m['test_nll'],m['true_test_nll'],m['fit_full_gaussian_nll'],selected])
    # Quality failures are observations, not failed invertibility checks.
    banana,eight,checker = [variants[n] for n in pins['variants']]
    assert banana['source_metrics']['true_test_nll'] < banana['source_metrics']['test_nll'] < banana['source_metrics']['fit_full_gaussian_nll']
    assert eight['source_sample_diagnostics_exact']['eight_gaussian_modes_covered_at_three_sigma']==8
    assert eight['source_sample_diagnostics_exact']['eight_gaussian_fraction_within_three_sigma_of_centers'] < .16
    assert checker['source_metrics']['test_nll'] > checker['source_metrics']['fit_full_gaussian_nll']
    assert checker['source_sample_diagnostics_exact']['checkerboard_support_fraction'] < .43
    return dict(passed=True,counts=counts,variants=variants,archives_checked=sum(v['archives_checked'] for v in variants.values()),
                same_initial_weights_and_batch_orders_but_separate_fits=True,
                three_dimensional_flat_Box_Muller_stream_not_two_dimensional_banana=True,
                erfc_Phi_components_and_jitter_and_all_source_IDs_exact=True,
                eight_mode_coverage_does_not_imply_concentrated_mode_samples=True,
                checkerboard_support_leakage_and_Gaussian_shortfall_preserved=True,
                full_final_selection_and_common_three_epoch_budgets=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('build/nn-report-conversion/flow_density_2d'))
    parser.add_argument('--json',type=Path)
    args=parser.parse_args()
    rendered=json.dumps(check(args.root),indent=2)+'\n'
    print(rendered,end='')
    if args.json:
        args.json.write_text(rendered)


if __name__=='__main__':
    main()
