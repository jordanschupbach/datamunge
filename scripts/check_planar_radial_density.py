#!/usr/bin/env python3
"""Independent NumPy audit of both pinned original planar/radial density runs.

Explicit elementary input/parameter differentials compose matrix Jacobians and
manual reverse accumulation. Checks both48-write fits and gradients through
both inverses. No datamunge, external autodiff, refitting or substitute data.
"""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays,exact,floating
from check_nice_density import gaussian,truth,LOG_2PI


def elementary(p,x,block,radial=False,inverse=False):
    """Return values and explicit local differentials, input first then parameters."""
    x=np.asarray(x,dtype=float);n=len(x);name=f'flow{block}/'
    keys=['center','raw_alpha','raw_beta'] if radial else ['angle','log_frequency','raw_projection','perpendicular','bias']
    size=2+sum(p[name+k].size for k in keys);eye=np.eye(size)
    dx=np.broadcast_to(eye[:2],(n,2,size));offset=2;pd={}
    for k in keys:
        v=p[name+k];pd[k]=eye[offset:offset+v.size].reshape(v.shape+(size,));offset+=v.size
    def sigmoid(v):return np.exp(-np.logaddexp(0,-v))
    if radial:
        center=p[name+'center'];alpha=.1+np.logaddexp(0,p[name+'raw_alpha']);beta=np.logaddexp(0,p[name+'raw_beta'])-alpha
        da=sigmoid(p[name+'raw_alpha'])*pd['raw_alpha'];db=sigmoid(p[name+'raw_beta'])*pd['raw_beta']-da
        off=x-center;doff=dx-pd['center'];square=(off*off).sum(axis=1);distance=np.sqrt(square)
        # The source explicitly selects zero distance/zero derivative at center.
        dd=np.einsum('ni,nik->nk',off,doff)/np.where(distance>0,distance,1)[:,None];dd[distance==0]=0
        radius=distance;dr=dd
        if inverse:
            a=alpha+beta-distance;disc=np.sqrt(a*a+4*alpha*distance)
            radius=np.where(a>=0,2*alpha*distance/np.maximum(a+disc,1e-20),.5*(disc-a))
            dr=((alpha+radius)[:,None]*dd-radius[:,None]*db+(distance-radius)[:,None]*da)/(2*radius+a)[:,None]
        denom=alpha+radius;ddenom=da+dr
        factor=1+beta/denom;df=db/denom[:,None]-beta*ddenom/denom[:,None]**2
        radial_factor=1+beta*alpha/denom**2
        drf=(alpha*db+beta*da)/denom[:,None]**2-2*beta*alpha*ddenom/denom[:,None]**3
        ld=np.log(factor)+np.log(radial_factor);dld=df/factor[:,None]+drf/radial_factor[:,None]
        if inverse:
            value=center+off/factor[:,None]
            dv=pd['center']+doff/factor[:,None,None]-off[:,:,None]*df[:,None,:]/factor[:,None,None]**2
            ld=-ld;dld=-dld
        else:
            value=center+off*factor[:,None]
            dv=pd['center']+doff*factor[:,None,None]+off[:,:,None]*df[:,None,:]
        coordinate=radius
    else:
        angle=p[name+'angle'];frequency=np.exp(p[name+'log_frequency']);projection=np.logaddexp(0,p[name+'raw_projection'])-.9
        q=p[name+'perpendicular'];bias=p[name+'bias'];da=pd['angle'];dfreq=frequency*pd['log_frequency'];dp=sigmoid(p[name+'raw_projection'])*pd['raw_projection']
        direction=np.array([np.cos(angle),np.sin(angle)]);tangent=np.array([-np.sin(angle),np.cos(angle)])
        ddirection=tangent[:,None]*da;dtangent=-direction[:,None]*da
        w=frequency*direction;dw=dfreq*direction[:,None]+frequency*ddirection
        u=(projection/frequency)*direction+q*tangent
        du=(dp/frequency-projection*dfreq/frequency**2)*direction[:,None]+(projection/frequency)*ddirection+pd['perpendicular']*tangent[:,None]+q*dtangent
        c=x@w+bias;dc=np.einsum('nik,i->nk',dx,w)+x@dw+pd['bias'];t=c;dt=dc
        if inverse:
            margin=abs(projection)+1;lo=c-margin;hi=c+margin
            for _ in range(32):
                mid=.5*(lo+hi);below=(mid+projection*np.tanh(mid)-c)<0
                lo=np.where(below,mid,lo);hi=np.where(below,hi,mid)
            root=.5*(lo+hi);ht=np.tanh(root);den=1+projection*(1-ht*ht)
            t=root+((c-root)-projection*ht)/den
            # Source root and denominator are stopped; retain this exact rule.
            dt=(dc-ht[:,None]*dp)/den[:,None]
        activation=np.tanh(t);sech=1-activation*activation
        factor=1+projection*sech;df=sech[:,None]*dp-2*(projection*activation*sech)[:,None]*dt
        sign=-1 if inverse else 1
        value=x+sign*activation[:,None]*u
        dv=dx+sign*(activation[:,None,None]*du+u[None,:,None]*sech[:,None,None]*dt[:,None,:])
        ld=sign*np.log(factor);dld=sign*df/factor[:,None];coordinate=t
    def backward(dy,dlogdet,partials):
        d=np.einsum('ni,nik->nk',dy,dv)+dlogdet[:,None]*dld;cursor=2
        for k in keys:
            value=p[name+k];partials[name+k]=d[:,cursor:cursor+value.size].sum(axis=0).reshape(value.shape);cursor+=value.size
        return d[:,:2]
    return value,ld,factor[:,None],coordinate[:,None],dv[:,:,:2],backward


def forward(p,x,radial=False):
    h=x.astype(float);jacobian=np.broadcast_to(np.eye(2),(len(x),2,2)).copy();logdet=np.zeros(len(x));fields={};backs=[]
    for i in range(4):
        h,ld,factor,coordinate,local,back=elementary(p,h,i,radial);jacobian=local@jacobian;logdet=logdet+ld;backs.append(back)
        fields.update({f'coupling{i}':h,f'coupling_logdet{i}':ld,f'coupling_scale{i}':factor,f'coupling_shift{i}':coordinate})
    scale=np.exp(p['log_scale']);z=h*scale;jacobian=scale[None,:,None]*jacobian;logdet=logdet+p['log_scale'].sum();base=-(.5*(z*z).sum(axis=1)+LOG_2PI)
    fields.update(latent=z,unscaled=h,forward_logdet=logdet,base_log_probability=base,log_probability=base+logdet,loss=np.asarray(-(base+logdet).mean()),jacobian_row0=jacobian[:,0],jacobian_row1=jacobian[:,1])
    def backward():
        dy=z/len(z);dld=-np.ones(len(z))/len(z);partials={'log_scale':(dy*z).sum(axis=0)+dld.sum()};dy=dy*scale
        for back in reversed(backs):dy=back(dy,dld,partials)
        return partials,dy
    return fields,backward


def inverse(p,z,radial=False):
    scale=np.exp(-p['log_scale']);h=z.astype(float)*scale;scaled=h.copy();jacobian=np.broadcast_to(np.diag(scale),(len(z),2,2)).copy();logdet=np.zeros(len(z))-p['log_scale'].sum();backs=[]
    for i in reversed(range(4)):
        h,ld,_,_,local,back=elementary(p,h,i,radial,True);jacobian=local@jacobian;logdet=logdet+ld;backs.append(back)
    def backward():
        dy=2*h/len(h);dld=np.full(len(h),.3/len(h));partials={}
        for back in reversed(backs):dy=back(dy,dld,partials)
        partials['log_scale']=-(dy*scaled).sum(axis=0)-dld.sum()
        return partials,dy*scale
    return h,logdet,jacobian,backward


def evaluate(p,x,radial=False):
    fields,_=forward(p,x,radial);value,ld,_,_=inverse(p,fields['latent'],radial)
    return dict(fields,reconstruction=value,inverse_logdet=ld)


def sample(p,z,radial=False):
    value,ld,jac,_=inverse(p,z,radial)
    return dict(prior=z,samples=value,prior_roundtrip=forward(p,value,radial)[0]['latent'],inverse_logdet=ld,inverse_jacobian_row0=jac[:,0],inverse_jacobian_row1=jac[:,1])


def fixtures(load,radial,counts):
    f=load('elementary_fixtures');p=floating({k:v for k,v in f.items() if k.startswith('flow0/')})
    points=np.array([[.3,-.2],[0,0],[1,0],[0,1],[-1,-1],[3,-2],[-3,2],[10,-10],[-10,10]],np.float32)
    if radial:
        exact(f['flow0/center'],np.array([.3,-.2],np.float32))
        exact(f['flow0/raw_alpha'],np.array(.4,np.float32));exact(f['flow0/raw_beta'],np.array(np.log(np.expm1(.12)),np.float32))
    else:
        points[0]=[-.1*np.cos(.6)/np.exp(.3),-.1*np.sin(.6)/np.exp(.3)]
        for k,v in [('angle',.6),('log_frequency',.3),('raw_projection',-6),('perpendicular',.7),('bias',.1)]:exact(f['flow0/'+k],np.array(v,np.float32))
    exact(f['input'],points)
    value,ld,factor,_,jac,_=elementary(p,points,0,radial);iv,il,_,_,ijac,_=elementary(p,points,0,radial,True);rv,*_=elementary(p,value,0,radial,True)
    expected=dict(input=points,forward=value,forward_logdet=ld,roundtrip=rv,inverse=iv,inverse_logdet=il,jacobian_row0=jac[:,0],jacobian_row1=jac[:,1],inverse_jacobian_row0=ijac[:,0],inverse_jacobian_row1=ijac[:,1],**p)
    assert set(f)==set(expected)
    maximum=0
    for k,v in expected.items():
        np.testing.assert_allclose(f[k],v,atol=3e-5,rtol=5e-5,err_msg='strong fixture/'+k)
        maximum=max(maximum,float(abs(f[k]-v).max()))
    for m,l in [(jac,ld),(ijac,il)]:
        assert np.all(np.linalg.det(m)>0);np.testing.assert_allclose(np.linalg.slogdet(m)[1],l,atol=2e-9,rtol=0)
    np.testing.assert_allclose(rv,points,atol=2e-14,rtol=0)
    if radial:
        center_jac=jac[0];center_factor=float(factor[0,0]);np.testing.assert_allclose(center_jac,np.eye(2)*center_factor,atol=1e-15,rtol=0)
        np.testing.assert_allclose(ijac[0],np.eye(2)/center_factor,atol=1e-14,rtol=0)
    counts['strong_fixture_values']=sum(v.size for v in f.values())
    return dict(points=9,all_fields_and_both_actual_Jacobians_checked=True,maximum_Float32_source_difference=maximum,finite_radial_center_derivative=bool(radial),planar_hinge_preserved=not radial)


def inverse_gradients(load,p,radial,compare,counts):
    batch=load('inverse_gradient_batch')['prior'];exact(batch,gaussian(1024,9257).astype(np.float32)[:8])
    value,ld,_,back=inverse(p,batch,radial);partials,_=back();compare(load('inverse_gradients'),partials,'inverse-gradients',True)
    def objective(parameters):
        v,l,_,_=inverse(parameters,batch,radial);return (v*v).sum(axis=1).mean()+.3*l.mean()
    maximum=0;coordinates=0
    for k,v in p.items():
        for ix in np.ndindex(v.shape):
            hi,lo=[{n:a.copy() for n,a in p.items()} for _ in range(2)];hi[k][ix]+=1e-5;lo[k][ix]-=1e-5
            fd=(objective(hi)-objective(lo))/2e-5
            np.testing.assert_allclose(fd,partials[k][ix],atol=2e-8,rtol=2e-6,err_msg='inverse FD/'+k)
            maximum=max(maximum,float(abs(fd-partials[k][ix])));coordinates+=1
    counts['inverse_objective_parameter_partials']=sum(v.size for v in partials.values());counts['inverse_finite_differences']=coordinates
    return dict(batch_rows=8,all_parameter_partials_checked=True,all_parameter_coordinate_finite_differences=coordinates,maximum_finite_difference_error=maximum)


def check_variant(root,variant):
    pins=json.loads((root/'baseline.json').read_text());source=root/'original-output'/variant
    radial=variant=='radial'
    archive_names={k.split('/',1)[1] for k in pins['archive_sha256'] if k.startswith(variant+'/')}
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
    assert len(p0)==(13 if radial else 21) and sum(v.size for v in p0.values())==(18 if radial else 22)
    for i in range(4):
        if radial:
            assert p0[f'flow{i}/center'].shape==(2,)
            exact(p0[f'flow{i}/raw_alpha'],np.array(.4,np.float32));exact(p0[f'flow{i}/raw_beta'],np.array(.56,np.float32))
        else:
            for key,value in [('angle',np.float32(.4)+np.float32(.65)*np.float32(i)),('log_frequency',0),('raw_projection',.4),('perpendicular',np.float32(.02)*(-1 if i%2 else 1)),('bias',.05)]:
                exact(p0[f'flow{i}/{key}'],np.array(value,np.float32))
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
        compare(load(name), evaluate(floating(p),x,radial), name)
    before = p0
    moment, variance = [{k: np.zeros_like(v) for k, v in p0.items()} for _ in range(2)]
    for step in range(48):
        rows = batch[batch['step'] == step]['row']
        assert len(rows) == 32 and np.all(batch[batch['step'] == step]['epoch'] == step // 16 + 1)
        _, back = forward(floating(before),data['fit_points'][rows],radial)
        partials, _ = back()
        gradients = arrays(root / 'phases' / variant / f'batch{step}-gradient.npz')
        compare(gradients, partials, f'batch{step}', True)
        after = arrays(root / 'phases' / variant / f'batch{step}-after.npz')
        for key, value in before.items():
            g = gradients[key]
            moment[key] = moment[key] * np.float32(.9) + g * np.float32(.1)
            variance[key] = variance[key] * np.float32(.999) + (g * g) * np.float32(.001)
            m = moment[key] / np.float32(1 - .9**(step + 1))
            v = variance[key] / np.float32(1 - .999**(step + 1))
            exact(value - m / (np.sqrt(v) + np.float32(1e-8)) * np.float32(.003), after[key])
            counts['exact_weight_values'] += value.size
        exact(arrays(root / 'phases' / variant / f'batch{step}-iterations.npz')['iterations'], np.array([step + 1], np.int64))
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
    _, back = forward(final,first,radial)
    partials, dx = back()
    compare(load('final_gradients'), partials, 'final-gradients', True)
    for key, value in final.items():
        index = np.unravel_index(np.argmax(abs(partials[key])), value.shape)
        hi, lo = [{k: v.copy() for k, v in final.items()} for _ in range(2)]
        hi[key][index] += 1e-5
        lo[key][index] -= 1e-5
        fd = (forward(hi,first,radial)[0]['loss'] - forward(lo,first,radial)[0]['loss']) / 2e-5
        np.testing.assert_allclose(fd, partials[key][index], atol=2e-9, rtol=2e-6, err_msg=key)
        errors['finite_difference'] = max(errors['finite_difference'], float(abs(fd - partials[key][index])))
        counts['finite_differences'] += 1
    history = np.loadtxt(source / 'history.tsv', skiprows=1)
    for epoch in [1, 2, 3]:
        p = floating(load(f'epoch{epoch}_arrays'))
        out = evaluate(p,data['validation_points'],radial)
        compare(load(f'epoch{epoch}_validation'), out, f'epoch{epoch}')
        np.testing.assert_allclose(history[epoch - 1, 1:], [out['loss'], out['forward_logdet'].mean()], atol=8e-7, rtol=0)
    sampled = sample(final,prior,radial)
    compare(load('samples'), sampled, 'samples')
    compare(load('sample_density'), evaluate(final,sampled['samples'],radial), 'sample-density')
    counts['prior_samples'] = len(prior)
    test = data['test_points']
    observed = load('outputs')
    jac = np.stack([observed['jacobian_row0'], observed['jacobian_row1']], axis=1).astype(float)
    ijac = np.stack([load('samples')['inverse_jacobian_row0'], load('samples')['inverse_jacobian_row1']], axis=1).astype(float)
    for matrices, logdet in [(jac, observed['forward_logdet']), (ijac, load('samples')['inverse_logdet'])]:
        assert np.all(np.linalg.det(matrices) > 0)
        np.testing.assert_allclose(np.linalg.slogdet(matrices)[1], logdet, atol=1e-6, rtol=0)
    counts['forward_Jacobian_entries'], counts['inverse_Jacobian_entries'] = jac.size, ijac.size
    assert np.ptp(observed['forward_logdet']) > 1e-5
    assert np.ptp(sampled['inverse_logdet']) > 1e-5
    np.testing.assert_allclose(observed['forward_logdet'] + observed['inverse_logdet'], 0, atol=5e-6, rtol=0)
    density_at_samples = forward(final,sampled['samples'],radial)[0]
    np.testing.assert_allclose(density_at_samples['forward_logdet'] + sampled['inverse_logdet'], 0, atol=1e-14, rtol=0)
    for block in range(4):assert np.all(observed[f'coupling_scale{block}']>0)

    for coordinate in range(2):
        hi, lo = test.astype(float), test.astype(float)
        hi[:, coordinate] += 1e-5
        lo[:, coordinate] -= 1e-5
        fd = (forward(final,hi,radial)[0]['latent'] - forward(final,lo,radial)[0]['latent']) / 2e-5
        np.testing.assert_allclose(fd, jac[:, :, coordinate], atol=2e-7, rtol=1e-5)
    roundtrip = forward(final,sampled['samples'],radial)[0]['latent']
    np.testing.assert_allclose(roundtrip, prior, atol=2e-14, rtol=0)
    logp = evaluate(final,sampled['samples'],radial)['log_probability']
    base = -(.5 * (prior.astype(float)**2).sum(axis=1) + LOG_2PI)
    errors['normalization'] = float(abs(logp + sampled['inverse_logdet'] - base).max())
    assert errors['normalization'] < 2e-14
    grid = load('grid')
    coordinates = np.array([[(x + .5) / 16 - 4, 4 - (y + .5) / 16]
                            for y in range(128) for x in range(128)], dtype=np.float32)
    exact(grid['points'], coordinates)
    compare({'model_log_probability': grid['model_log_probability']},
            {'model_log_probability': forward(final,coordinates,radial)[0]['log_probability']}, 'grid')
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
    assert metrics['test_nll'] > metrics['true_test_nll']
    assert (metrics['test_nll'] < metrics['fit_full_gaussian_nll']) == (not radial)
    fixture=fixtures(load,radial,counts)
    inverse_receipt=inverse_gradients(load,final,radial,compare,counts)
    assert audited==archive_names,(variant,audited^archive_names)
    return dict(passed=True,variant=variant,counts=counts,maximum_errors=errors,archives_checked=len(audited),
      raw_midpoint_MT_Box_Muller_and_all_banana_rows_exact=True,all_48_Adam_writes_and_iterations_bit_exact=True,
      model_leaf_finite_differences=True,analytic_composed_matrix_Jacobians=True,
      positive_actual_forward_and_inverse_determinants_match_per_point_logdets=True,
      all_1024_prior_inverse_samples_and_normalization_identities=True,
      fitting_only_Gaussian_baselines_and_analytic_truth_checked=True,
      original_rounded_Float32_log_2pi_constant_preserved=LOG_2PI,
      source_improves_fitted_Gaussian=not radial,test_nll=metrics['test_nll'],fixture=fixture,inverse_gradients=inverse_receipt)

def check(root):
    pins=json.loads((root/'baseline.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for file,digest in pins['source_pins'].items():assert sha(root/('original-'+Path(file).name))==digest
    for group,directory in [('archive_sha256',root/'original-output'),('other_files_sha256',root/'original-output'),('phase_probe_sha256',root)]:
        for file,digest in pins[group].items():assert sha(directory/file)==digest
    assert not pins['prepared_input_sha256'] and pins['optimizer_updates']==96
    variants=[check_variant(root,v) for v in ['planar','radial']]
    counts={k:sum(v['counts'][k] for v in variants) for k in variants[0]['counts']}
    assert sum(v['archives_checked'] for v in variants)==48
    return dict(passed=True,archives_checked=48,counts=counts,variants=variants,
      no_refit_no_native_helper_no_external_autodiff=True,planar_numerical_bisection32_with_implicit_Newton_differentials=True,
      radial_stable_quadratic_inverse_and_finite_center_derivative=True,radial_underperforms_fitted_Gaussian_preserved=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path('build/nn-report-conversion/planar_radial_density'));parser.add_argument('--json',type=Path)
    args=parser.parse_args();rendered=json.dumps(check(args.root),indent=2)+'\n';print(rendered,end='')
    if args.json:args.json.write_text(rendered)


if __name__=='__main__':main()
