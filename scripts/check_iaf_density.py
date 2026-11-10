#!/usr/bin/env python3
"""Independent NumPy audit of the pinned original inverse autoregressive flow.

Composes strict masked conditioner derivatives through each sequential density
recovery and its final head re-evaluation. Sampling uses known base coordinates
in parallel. No datamunge/autograd, refitting or substitute data.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact, floating
from check_nice_density import gaussian, truth, LOG_2PI
from check_maf_density import inventory,conditioner


def forward(p,x):
    h=x.astype(float);jacobian=np.broadcast_to(np.eye(2),(len(x),2,2)).copy()
    logdet=np.zeros(len(x));fields={};backwards=[]
    for i in reversed(range(4)):
        observed=h;recovered=np.zeros_like(h);local=np.zeros_like(jacobian);calls=[]
        for c in ([1,0] if i%2 else [0,1]):
            # Capture the conditioner on this immutable recovered prefix.
            mean,scale,cj,back=conditioner(p,recovered.copy(),i)
            factor=np.exp(-scale[:,c]);value=(observed[:,c]-mean[:,c])*factor
            slope=factor[:,None]*cj[:,c]+value[:,None]*cj[:,2+c]
            row=factor[:,None]*np.eye(2)[c]-np.einsum('ni,nij->nj',slope,local)
            local[:,c]=row;recovered[:,c]=value;calls.append((c,value,factor,back))
        # The source explicitly recomputes heads on the fully recovered input
        # for its density determinant, retaining this extra parameter/input graph.
        mean,scale,cj,last=conditioner(p,recovered.copy(),i)
        ld=-scale.sum(axis=1);logdet=logdet+ld;jacobian=local@jacobian;h=recovered
        fields.update({f'conditioner_input{i}':h,f'coupling{i}':h,
            f'coupling_logdet{i}':ld,f'coupling_scale{i}':scale,f'coupling_shift{i}':mean})
        for k in range(4):fields[f'conditioner_jacobian{i}_{k}']=cj[:,k]
        backwards.append((calls,last))
    factor=np.exp(p['log_scale']);z=h*factor;jacobian=factor[None,:,None]*jacobian
    logdet=logdet+p['log_scale'].sum();base=-(.5*(z*z).sum(axis=1)+LOG_2PI)
    fields.update(latent=z,unscaled=h,forward_logdet=logdet,
        base_log_probability=base,log_probability=base+logdet,loss=np.asarray(-(base+logdet).mean()),
        jacobian_row0=jacobian[:,0],jacobian_row1=jacobian[:,1])
    def backward():
        dy=z/len(z);dld=-np.ones(len(z))/len(z)
        partials={'log_scale':(dy*z).sum(axis=0)+dld.sum()};dy=dy*factor
        def back_add(call,dmean,dscale):
            added={};dx=call(dmean,dscale,added)
            for key,value in added.items():partials[key]=partials.get(key,np.zeros_like(value))+value
            return dx
        for calls,last in reversed(backwards):
            dy=dy+back_add(last,np.zeros_like(dy),np.broadcast_to(-dld[:,None],dy.shape))
            direct=np.zeros_like(dy)
            for c,value,local_factor,call in reversed(calls):
                dc=dy[:,c].copy();dm=np.zeros_like(dy);ds=np.zeros_like(dy)
                direct[:,c]=dc*local_factor;dm[:,c]=-dc*local_factor;ds[:,c]=-dc*value
                dy=dy.copy();dy[:,c]=0;dy=dy+back_add(call,dm,ds)
            dy=direct
        return partials,dy
    return fields,backward


def inverse(p,z):
    factor=np.exp(-p['log_scale']);h=z.astype(float)*factor
    jacobian=np.broadcast_to(np.diag(factor),(len(z),2,2)).copy();ld=np.zeros(len(z))-p['log_scale'].sum()
    for i in range(4):
        mean,scale,cj,_=conditioner(p,h,i);factor=np.exp(scale)
        local=factor[:,:,None]*np.eye(2)+(h*factor)[:,:,None]*cj[:,2:]+cj[:,:2]
        jacobian=local@jacobian;h=h*factor+mean;ld=ld+scale.sum(axis=1)
    return h,ld,jacobian


def evaluate(p,x):
    fields,_=forward(p,x);value,ld,_=inverse(p,fields['latent'])
    return dict(fields,reconstruction=value,inverse_logdet=ld)


def sample(p,z):
    value,ld,jacobian=inverse(p,z)
    return dict(prior=z,samples=value,prior_roundtrip=forward(p,value)[0]['latent'],
        inverse_logdet=ld,inverse_jacobian_row0=jacobian[:,0],inverse_jacobian_row1=jacobian[:,1])


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
    assert len(p0)==17 and sum(v.size for v in p0.values())==466
    masks=load('masks');assert len(masks)==20
    masked={}
    for i in range(4):
        inv=inventory(i)
        for key,value in inv.items():exact(masks[f'block{i}_{key}'],value)
        for suffix,shape in [('hidden/W',(2,16)),('heads/W',(16,4))]:
            name=f'autoregressive{i}/{suffix}';assert p0[name].shape==shape
            masked[name]=inv['input_mask' if suffix=='hidden/W' else 'output_mask']==0
        exact(p0[f'autoregressive{i}/hidden/b'],np.full(16,.05,np.float32))
        exact(p0[f'autoregressive{i}/heads/b'],np.zeros(4,np.float32))
    assert sum(v.sum() for v in masked.values())==160
    counts['mask_inventory_values']=sum(v.size for v in masks.values())
    counts['masked_zero_gradients']=counts['masked_unchanged_weights']=0
    counts['strict_conditioner_Jacobian_entries']=0
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
            if key in masked:
                selected=masked[key];assert np.all(g[selected]==0)
                exact(value[selected],after[key][selected])
                counts['masked_zero_gradients']+=int(selected.sum())
                counts['masked_unchanged_weights']+=int(selected.sum())
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
    for name in ['initial_outputs','first_forward','outputs','epoch1_validation','epoch2_validation','epoch3_validation','sample_density']:
        captured=load(name)
        for i in range(4):
            inv=inventory(i)
            for k in range(4):
                disallowed=inv['input_degrees']>=inv['output_degrees'][k]
                matrix=captured[f'conditioner_jacobian{i}_{k}']
                assert np.all(matrix[:,disallowed]==0),(name,i,k)
                counts['strict_conditioner_Jacobian_entries']+=int(matrix[:,disallowed].size)
    masked_finite_differences=0
    for key,selected in masked.items():
        for flat in np.flatnonzero(selected):
            hi={k:v.copy() for k,v in final.items()};lo={k:v.copy() for k,v in final.items()}
            hi[key].flat[flat]+=.1;lo[key].flat[flat]-=.1
            assert forward(hi,first)[0]['loss']==forward(lo,first)[0]['loss']
            masked_finite_differences+=1
    counts['masked_finite_differences']=masked_finite_differences
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
        assert np.max(abs(observed[f'coupling_scale{block}']))<=.8
        np.testing.assert_allclose(-observed[f'coupling_scale{block}'].sum(axis=1),observed[f'coupling_logdet{block}'],atol=0,rtol=0)

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
    assert metrics['fit_full_gaussian_nll'] > metrics['test_nll'] > metrics['true_test_nll']
    assert audited == set(pins['archive_sha256']), audited ^ set(pins['archive_sha256'])
    return dict(passed=True, counts=counts, maximum_errors=errors, archives_checked=len(audited),
                raw_midpoint_MT_Box_Muller_and_all_banana_rows_exact=True,
                all_48_Adam_writes_and_iterations_bit_exact=True,
                all_17_leaf_finite_differences=True,all_160_masked_finite_differences_zero=True, strict_degree_masks_and_actual_conditioner_Jacobians=True,
                all_160_masked_coordinates_have_zero_partials_and_never_change=True,
                sequential_density_with_final_recovered_input_head_reevaluation=True,
                parallel_sampling_and_analytic_matrix_Jacobians=True,
                bounded_point8_tanh_log_scales_and_all_stage_fields=True,
                spatially_varying_forward_and_inverse_path_logdets=True,
                positive_actual_forward_and_inverse_determinants_match_per_point_logdets=True,
                all_1024_prior_inverse_samples_and_normalization_identities=True,
                fitting_only_Gaussian_baselines_and_analytic_truth_checked=True,
                original_rounded_Float32_log_2pi_constant_preserved=LOG_2PI,
                source_improvement_over_fitting_only_Gaussian_preserved=True,
                test_nll=metrics['test_nll'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('build/nn-report-conversion/iaf_density'))
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    rendered = json.dumps(check(args.root), indent=2) + '\n'
    print(rendered, end='')
    if args.json:
        args.json.write_text(rendered)


if __name__ == '__main__':
    main()
