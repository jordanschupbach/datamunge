#!/usr/bin/env python3
"""Independent NumPy audit of the pinned original continuous normalizing flow.

Manual dense/trace pullbacks and a reverse RK4 computation check all source
updates; analytic matrix composition checks the actual discrete map. Retains
Float32 source times/coefficients and numerical inverse/divergence residuals.
No native helper, external autodiff, refit or replacement data.
"""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays,exact,floating
from check_nice_density import gaussian,truth,LOG_2PI


def field(p,x,time,trace_mode=3):
    x=np.asarray(x,dtype=float);time=float(np.float32(time));encoded=np.column_stack([x,np.full(len(x),time)])
    wi=p['field/hidden/W'];wo=p['field/output/W'];hidden=np.tanh(encoded@wi+p['field/hidden/b']);slope=1-hidden*hidden
    velocity=.3*(hidden@wo+p['field/output/b']);a,b=wi[:2];c,d=wo.T
    exact_product=a*c+b*d;plus_product=(a+b)*(c+d);minus_product=(a-b)*(c-d)
    exact_trace=.3*(slope*exact_product).sum(axis=1);plus=.3*(slope*plus_product).sum(axis=1);minus=.3*(slope*minus_product).sum(axis=1)
    divergence=[exact_trace,plus,minus,.5*(plus+minus)][trace_mode]
    jacobian=.3*np.einsum('jo,nj,ij->noi',wo,slope,wi[:2])
    def backward(dvelocity,ddivergence,partials):
        dv=.3*dvelocity;gwo=hidden.T@dv;gbo=dv.sum(axis=0);dh=dv@wo.T
        if trace_mode in (0,3):product=exact_product
        elif trace_mode==1:product=plus_product
        else:product=minus_product
        dh+=(-.6*ddivergence[:,None]*hidden)*product
        gproduct=.3*(ddivergence[:,None]*slope).sum(axis=0)
        gwi_extra=np.zeros_like(wi);gwo_extra=np.zeros_like(wo)
        if trace_mode in (0,3):
            gwi_extra[0]=gproduct*c;gwi_extra[1]=gproduct*d;gwo_extra[:,0]=gproduct*a;gwo_extra[:,1]=gproduct*b
        elif trace_mode==1:
            gwi_extra[0]=gwi_extra[1]=gproduct*(c+d);gwo_extra[:,0]=gwo_extra[:,1]=gproduct*(a+b)
        else:
            gwi_extra[0]=gproduct*(c-d);gwi_extra[1]=-gproduct*(c-d);gwo_extra[:,0]=gproduct*(a-b);gwo_extra[:,1]=-gproduct*(a-b)
        dz=dh*slope;gwi=encoded.T@dz+gwi_extra;gwo+=gwo_extra;gbi=dz.sum(axis=0)
        for key,value in [('field/hidden/W',gwi),('field/hidden/b',gbi),('field/output/W',gwo),('field/output/b',gbo)]:partials[key]=partials.get(key,np.zeros_like(value))+value
        return dz@wi[:2].T
    return dict(velocity=velocity,divergence=divergence,exact_trace=exact_trace,hutchinson_plus=plus,hutchinson_minus=minus),jacobian,backward


def integrate(p,x,inverse=False,steps=8,trace_mode=3):
    h=x.astype(float);ld=np.zeros(len(x));identity=np.broadcast_to(np.eye(2),(len(x),2,2));jac=identity.copy();fields={};backs=[];stage=0
    dt=np.float32((-1 if inverse else 1)/steps);half=np.float32(np.float32(.5)*dt);sixth=np.float32(dt/np.float32(6))
    for i in range(steps):
        time=np.float32(dt*np.float32(i));time=np.float32(1+time) if inverse else time
        a,ja,ba=field(p,h,time,trace_mode);b,jb,bb=field(p,h+float(half)*a['velocity'],np.float32(time+half),trace_mode)
        c,jc,bc=field(p,h+float(half)*b['velocity'],np.float32(time+half),trace_mode);d,jd,bd=field(p,h+float(dt)*c['velocity'],np.float32(time+dt),trace_mode)
        jb=jb@(identity+float(half)*ja);jc=jc@(identity+float(half)*jb);jd=jd@(identity+float(dt)*jc)
        local=identity+float(sixth)*((ja+jd)+2*(jb+jc));jac=local@jac
        h=h+float(sixth)*((a['velocity']+d['velocity'])+2*(b['velocity']+c['velocity']))
        ld=ld+float(sixth)*((a['divergence']+d['divergence'])+2*(b['divergence']+c['divergence']))
        def back(dh,dl,partials,ba=ba,bb=bb,bc=bc,bd=bd):
            direct=dh.copy();ga=float(sixth)*dh;gb=2*float(sixth)*dh;gc=gb.copy();gd=ga.copy()
            dd=bd(gd,float(sixth)*dl,partials);direct+=dd;gc+=float(dt)*dd
            dc=bc(gc,2*float(sixth)*dl,partials);direct+=dc;gb+=float(half)*dc
            db=bb(gb,2*float(sixth)*dl,partials);direct+=db;ga+=float(half)*db
            return direct+ba(ga,float(sixth)*dl,partials)
        backs.append(back)
        if (i+1)%(steps//4)==0:
            endpoint,_,_=field(p,h,np.float32(time+dt),trace_mode)
            fields.update({f'coupling{stage}':h,f'coupling_logdet{stage}':ld,f'coupling_scale{stage}':endpoint['divergence'],f'coupling_shift{stage}':endpoint['velocity']});stage+=1
    def backward(dh,dl,partials):
        for back in reversed(backs):dh=back(dh,dl,partials)
        return dh
    return h,ld,jac,fields,backward


def forward(p,x):
    h,ld,jac,fields,back=integrate(p,x);scale=np.exp(p['log_scale']);z=h*scale;jac=scale[None,:,None]*jac;ld=ld+p['log_scale'].sum();base=-(.5*(z*z).sum(axis=1)+LOG_2PI)
    fields.update(latent=z,unscaled=h,forward_logdet=ld,base_log_probability=base,log_probability=base+ld,loss=np.asarray(-(base+ld).mean()),jacobian_row0=jac[:,0],jacobian_row1=jac[:,1])
    def backward():
        dz=z/len(z);dl=-np.ones(len(z))/len(z);partials={'log_scale':(dz*z).sum(axis=0)+dl.sum()}
        dx=back(dz*scale,dl,partials);return partials,dx
    return fields,backward


def inverse(p,z):
    scale=np.exp(-p['log_scale']);h,ld,jac,_,_=integrate(p,z.astype(float)*scale,True)
    return h,ld-p['log_scale'].sum(),jac@np.diag(scale)


def evaluate(p,x):
    result,_=forward(p,x);value,ld,_=inverse(p,result['latent']);f,j,_=field(p,x,.5)
    result.update(reconstruction=value,inverse_logdet=ld,field_velocity=f['velocity'],field_exact_trace=f['exact_trace'],field_hutchinson_plus=f['hutchinson_plus'],field_hutchinson_minus=f['hutchinson_minus'],field_jacobian_row0=j[:,0],field_jacobian_row1=j[:,1])
    return result


def sample(p,z):
    value,ld,jac=inverse(p,z)
    return dict(prior=z,samples=value,prior_roundtrip=forward(p,value)[0]['latent'],inverse_logdet=ld,inverse_jacobian_row0=jac[:,0],inverse_jacobian_row1=jac[:,1])


def check_field_partials(p,x,counts):
    maximum=0;partials_count=0;input_count=0
    for mode in range(4):
        f,j,back=field(p,x,.5,mode);partials={};dx=back(2*f['velocity']/len(x),np.full(len(x),.3/len(x)),partials)
        np.testing.assert_allclose(f['exact_trace'],np.trace(j,axis1=1,axis2=2),atol=1e-16,rtol=0)
        plus=j.sum(axis=(1,2));minus=j[:,0,0]-j[:,0,1]-j[:,1,0]+j[:,1,1]
        np.testing.assert_allclose(f['hutchinson_plus'],plus,atol=1e-16,rtol=0);np.testing.assert_allclose(f['hutchinson_minus'],minus,atol=1e-16,rtol=0)
        def objective(parameters,points):
            g,_,_=field(parameters,points,.5,mode);return (g['velocity']**2).sum(axis=1).mean()+.3*g['divergence'].mean()
        for key,value in partials.items():
            for ix in np.ndindex(value.shape):
                hi,lo=[{k:v.copy() for k,v in p.items()} for _ in range(2)];hi[key][ix]+=1e-5;lo[key][ix]-=1e-5
                fd=(objective(hi,x)-objective(lo,x))/2e-5;np.testing.assert_allclose(fd,value[ix],atol=3e-10,rtol=3e-6,err_msg=f'field mode{mode} '+key)
                maximum=max(maximum,float(abs(fd-value[ix])));partials_count+=1
        for c in range(2):
            hi=x.astype(float);lo=x.astype(float);hi[:,c]+=1e-5;lo[:,c]-=1e-5
            # The simultaneous coordinate shift checks the sum of every row partial.
            fd=(objective(p,hi)-objective(p,lo))/2e-5;np.testing.assert_allclose(fd,dx[:,c].sum(),atol=3e-10,rtol=3e-6);input_count+=1
    counts['field_parameter_coordinate_finite_differences']=partials_count;counts['field_input_directional_finite_differences']=input_count
    return dict(trace_modes_checked=4,all50_field_parameter_coordinates_each_mode=True,maximum_finite_difference_error=maximum,actual_field_trace_and_both_probe_contractions_checked=True)



def check_inverse_partials(p,z,counts):
    scale=np.exp(-p['log_scale']);scaled=z.astype(float)*scale;value,ld,_,_,back=integrate(p,scaled,True)
    partials={};dx=back(2*value/len(value),np.full(len(value),.3/len(value)),partials);partials['log_scale']=-(dx*scaled).sum(axis=0)-.3
    def objective(parameters):
        v,l,_=inverse(parameters,z);return (v*v).sum(axis=1).mean()+.3*l.mean()
    maximum=0;total=0
    for key,g in partials.items():
        for ix in np.ndindex(g.shape):
            hi,lo=[{k:v.copy() for k,v in p.items()} for _ in range(2)];hi[key][ix]+=1e-5;lo[key][ix]-=1e-5
            fd=(objective(hi)-objective(lo))/2e-5;np.testing.assert_allclose(fd,g[ix],atol=3e-9,rtol=3e-6,err_msg='reverse RK4 '+key)
            maximum=max(maximum,float(abs(fd-g[ix])));total+=1
    counts['inverse_parameter_coordinate_finite_differences']=total
    return dict(all52_reverse_RK4_parameter_coordinates_checked=True,maximum_finite_difference_error=maximum)


def check_numerical_flow(p,x,counts):
    rows=x[:8].astype(float);results=[]
    for steps in [4,8,16,32]:
        value,ld,j,_,_=integrate(p,rows,False,steps);rec,il,_,_,_=integrate(p,value,True,steps)
        err=float(abs(np.linalg.slogdet(j)[1]-ld).max());roundtrip=float(abs(rec-rows).max());cancel=float(abs(ld+il).max())
        assert err<5e-6 and roundtrip<5e-6 and cancel<5e-6
        results.append(dict(steps=steps,actual_discrete_Jacobian_vs_integrated_trace=err,roundtrip=roundtrip,forward_inverse_trace_cancellation=cancel))
    counts['independent_RK4_step_counts_checked']=4
    return dict(numerical_approximation_explicit=True,source_Float32_coefficients_even_in_Double=True,step_counts=results,no_exact_discrete_logdet_or_inverse_claim=True)


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
    assert len(p0)==5 and sum(v.size for v in p0.values())==52
    assert p0['field/hidden/W'].shape==(3,8) and p0['field/output/W'].shape==(8,2)
    exact(p0['field/hidden/b'],np.full(8,.05,np.float32));exact(p0['field/output/b'],np.zeros(2,np.float32))
    exact(p0['log_scale'],np.zeros(2,np.float32))
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
        for index in np.ndindex(value.shape):
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
        np.testing.assert_allclose(history[epoch - 1, 1:], [out['loss'], out['forward_logdet'].mean()], atol=8e-7, rtol=0)
    sampled = sample(final, prior)
    compare(load('samples'), sampled, 'samples')
    compare(load('sample_density'), evaluate(final, sampled['samples']), 'sample-density')
    counts['prior_samples'] = len(prior)
    test = data['test_points']
    observed = load('outputs')
    jac = np.stack([observed['jacobian_row0'], observed['jacobian_row1']], axis=1).astype(float)
    ijac = np.stack([load('samples')['inverse_jacobian_row0'], load('samples')['inverse_jacobian_row1']], axis=1).astype(float)
    for matrices, logdet in [(jac, observed['forward_logdet']), (ijac, load('samples')['inverse_logdet'])]:
        assert np.all(np.linalg.det(matrices) > 0)
        np.testing.assert_allclose(np.linalg.slogdet(matrices)[1], logdet, atol=5e-6, rtol=0)
    counts['forward_Jacobian_entries'], counts['inverse_Jacobian_entries'] = jac.size, ijac.size
    assert np.ptp(observed['forward_logdet']) > 1e-5
    assert np.ptp(sampled['inverse_logdet']) > 1e-5
    np.testing.assert_allclose(observed['forward_logdet'] + observed['inverse_logdet'], 0, atol=5e-6, rtol=0)
    density_at_samples = forward(final, sampled['samples'])[0]
    np.testing.assert_allclose(density_at_samples['forward_logdet'] + sampled['inverse_logdet'], 0, atol=1e-7, rtol=0)
    np.testing.assert_allclose(observed['field_exact_trace'],.5*(observed['field_hutchinson_plus'].astype(float)+observed['field_hutchinson_minus']),atol=1e-6,rtol=0)
    np.testing.assert_allclose(observed['field_exact_trace'],observed['field_jacobian_row0'][:,0].astype(float)+observed['field_jacobian_row1'][:,1],atol=1e-6,rtol=0)
    for stage in range(4):
        assert observed[f'coupling_scale{stage}'].shape==(128,)
        assert observed[f'coupling_shift{stage}'].shape==(128,2)
    counts['source_actual_field_Jacobian_entries']=6784

    for coordinate in range(2):
        hi, lo = test.astype(float), test.astype(float)
        hi[:, coordinate] += 1e-5
        lo[:, coordinate] -= 1e-5
        fd = (forward(final, hi)[0]['latent'] - forward(final, lo)[0]['latent']) / 2e-5
        np.testing.assert_allclose(fd, jac[:, :, coordinate], atol=2e-7, rtol=1e-5)
    roundtrip = forward(final, sampled['samples'])[0]['latent']
    np.testing.assert_allclose(roundtrip, prior, atol=1e-7, rtol=0)
    logp = evaluate(final, sampled['samples'])['log_probability']
    base = -(.5 * (prior.astype(float)**2).sum(axis=1) + LOG_2PI)
    errors['normalization'] = float(abs(logp + sampled['inverse_logdet'] - base).max())
    assert errors['normalization'] < 1e-7
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
    assert metrics['fit_full_gaussian_nll'] > metrics['test_nll'] > metrics['fit_diagonal_gaussian_nll'] > metrics['true_test_nll']
    assert metrics['hidden']==16  # Preserve the original harness metadata; the actual field width is8.
    field_receipt=check_field_partials(final,first,counts)
    numerical_receipt=check_numerical_flow(final,test,counts)
    inverse_receipt=check_inverse_partials(final,prior[:8],counts)
    assert audited == set(pins['archive_sha256']), audited ^ set(pins['archive_sha256'])
    return dict(passed=True,counts=counts,maximum_errors=errors,archives_checked=len(audited),
      raw_midpoint_MT_Box_Muller_and_all_banana_rows_exact=True,all_48_Adam_writes_and_iterations_bit_exact=True,
      all_52_model_parameter_coordinate_finite_differences=True,analytic_RK4_matrix_Jacobians=True,
      manual_field_and_RK4_reverse_accumulation=True,source_Float32_RK4_times_and_coefficients_preserved=True,
      all_four_quarter_states_cumulative_logdets_endpoint_divergences_and_velocities_checked=True,
      deterministic_paired_probes_equal_exact_2D_field_trace=True,field_checks=field_receipt,numerical_checks=numerical_receipt,inverse_checks=inverse_receipt,
      numerical_reverse_flow_and_integrated_divergence_errors_bounded=True,all_1024_prior_inverse_samples_and_normalization_identities=True,
      fitting_only_Gaussian_baselines_and_analytic_truth_checked=True,source_tiny_full_Gaussian_gain_but_diagonal_shortfall_preserved=True,
      actual_field_hidden8_source_generic_metrics_hidden16_preserved_and_identified=True,
      original_rounded_Float32_log_2pi_constant_preserved=LOG_2PI,test_nll=metrics['test_nll'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('build/nn-report-conversion/cnf_density'))
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    rendered = json.dumps(check(args.root), indent=2) + '\n'
    print(rendered, end='')
    if args.json:
        args.json.write_text(rendered)


if __name__ == '__main__':
    main()
