#!/usr/bin/env python3
"""Independent NumPy audit of the pinned original rational-quadratic spline flow.

Explicit twelve-component local differential arithmetic checks input/head
partials, composed matrix Jacobians and hand-written dense backpropagation.
No datamunge or external autodiff, refitting, root search or substitute data.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays,exact,floating
from check_nice_density import gaussian,truth,LOG_2PI

OFFSET=float(np.float32(np.log(np.expm1(.999))))


class Differential:
    """A scalar field and its twelve explicit local input/head partials."""
    def __init__(self,value,partial=None):
        self.v=np.asarray(value,dtype=float)
        self.d=np.zeros(self.v.shape+(12,)) if partial is None else np.asarray(partial,dtype=float)
    @staticmethod
    def wrap(value):return value if isinstance(value,Differential) else Differential(value)
    def __add__(self,other):
        b=self.wrap(other);return Differential(self.v+b.v,self.d+b.d)
    __radd__=__add__
    def __neg__(self):return Differential(-self.v,-self.d)
    def __sub__(self,other):return self+-self.wrap(other)
    def __rsub__(self,other):return self.wrap(other)+-self
    def __mul__(self,other):
        b=self.wrap(other);return Differential(self.v*b.v,self.d*b.v[...,None]+b.d*self.v[...,None])
    __rmul__=__mul__
    def __truediv__(self,other):
        b=self.wrap(other);v=self.v/b.v
        return Differential(v,(self.d-v[...,None]*b.d)/b.v[...,None])
    def __rtruediv__(self,other):return self.wrap(other)/self
    def col(self,start,stop=None):
        stop=start+1 if stop is None else stop
        return Differential(self.v[:,start:stop],self.d[:,start:stop,:])
    def total(self):return Differential(self.v.sum(axis=1,keepdims=True),self.d.sum(axis=1,keepdims=True))
    def log(self):return Differential(np.log(self.v),self.d/self.v[...,None])
    def sqrt(self):
        root=np.sqrt(self.v);return Differential(root,self.d/(2*root[...,None]))
    def softplus(self):return Differential(np.logaddexp(0,self.v),self.d*np.exp(-np.logaddexp(0,-self.v))[...,None])
    def softmax(self):
        exp=np.exp(self.v-self.v.max(axis=1,keepdims=True));p=exp/exp.sum(axis=1,keepdims=True)
        return Differential(p,p[...,None]*(self.d-(p[...,None]*self.d).sum(axis=1,keepdims=True)))


def cat(values):return Differential(np.concatenate([x.v for x in values],axis=1),np.concatenate([x.d for x in values],axis=1))


def spline(input,heads,inverse=False):
    input=np.asarray(input,dtype=float).reshape(-1,1);heads=np.asarray(heads,dtype=float)
    n=len(input);identity=np.eye(12);x=Differential(input,np.broadcast_to(identity[0],(n,1,12)))
    q=Differential(heads,np.broadcast_to(identity[1:],(n,11,12)))
    widths=8*(.01+.96*q.col(0,4).softmax());heights=8*(.01+.96*q.col(4,8).softmax())
    zero=q.col(0)*0;left=zero-4;right=zero+4;one=zero+1
    derivatives=cat([one,.001+(q.col(8,11)+OFFSET).softplus(),one])
    xknots=[left]+[-4+widths.col(0,j).total() for j in range(1,4)]+[right]
    yknots=[left]+[-4+heights.col(0,j).total() for j in range(1,4)]+[right]
    x0all=cat(xknots[:-1]);x1all=cat(xknots[1:]);y0all=cat(yknots[:-1]);y1all=cat(yknots[1:])
    widths=x1all-x0all;heights=y1all-y0all
    inside=(input>=-4)&(input<=4);bounded=Differential(np.clip(input,-4,4),x.d*inside[...,None])
    edges=y1all.v if inverse else x1all.v;index=np.minimum((bounded.v>=edges).sum(axis=1),3);rows=np.arange(n)
    def pick(a):return Differential(a.v[rows,index,None],a.d[rows,index,None,:])
    x0=pick(x0all);y0=pick(y0all);w=pick(widths);h=pick(heights)
    d0=pick(derivatives.col(0,4));d1=pick(derivatives.col(1,5));delta=h/w;curvature=d0+d1-2*delta
    if inverse:
        offset=bounded-y0;a=offset*curvature+h*(delta-d0);b=h*d0-offset*curvature;c=-(delta*offset)
        theta=2*c/(-b-(b*b-4*(a*c)).sqrt());value=x0+theta*w
    else:theta=(bounded-x0)/w
    complement=1-theta;blend=theta*complement;denominator=delta+curvature*blend
    if not inverse:value=y0+h*(delta*theta*theta+d0*blend)/denominator
    numerator=delta*delta*((d1*theta*theta+2*(delta*blend))+d0*complement*complement)
    logdet=numerator.log()-2*denominator.log()
    if inverse:logdet=-logdet
    value=Differential(np.where(inside,value.v,input),np.where(inside[...,None],value.d,x.d))
    ld=Differential(np.where(inside,logdet.v,0),np.where(inside[...,None],logdet.d,0))
    fields=dict(value=value.v,logdet=ld.v[:,0],widths=widths.v,heights=heights.v,derivatives=derivatives.v)
    return fields,value.d[:,0,:],ld.d[:,0,:]


def coupling(p,x,block,inverse=False):
    name=f'coupling{block}';condition=block%2;a=x[:,condition:condition+1];b=x[:,1-condition:2-condition]
    hidden=np.tanh(a@p[name+'/hidden/W']+p[name+'/hidden/b']);heads=hidden@p[name+'/heads/W']+p[name+'/heads/b']
    fields,dvalue,dld=spline(b,heads,inverse);value=x.copy();value[:,1-condition:2-condition]=fields['value']
    dhda=((1-hidden*hidden)*p[name+'/hidden/W'])@p[name+'/heads/W']
    local=np.broadcast_to(np.eye(2),(len(x),2,2)).copy()
    local[:,1-condition,condition]=(dvalue[:,1:]*dhda).sum(axis=1);local[:,1-condition,1-condition]=dvalue[:,0]
    def backward(dy,dlogdet,partials):
        assert not inverse
        dc=dy[:,1-condition:2-condition]
        dheads=dc*dvalue[:,1:]+dlogdet[:,None]*dld[:,1:]
        partials[name+'/heads/W']=hidden.T@dheads;partials[name+'/heads/b']=dheads.sum(axis=0)
        dh=(dheads@p[name+'/heads/W'].T)*(1-hidden*hidden)
        partials[name+'/hidden/W']=a.T@dh;partials[name+'/hidden/b']=dh.sum(axis=0)
        dx=dy.copy();dx[:,condition:condition+1]+=dh@p[name+'/hidden/W'].T
        dx[:,1-condition]=dc[:,0]*dvalue[:,0]+dlogdet*dld[:,0]
        return dx
    return value,fields,local,backward


def forward(p,x):
    h=x.astype(float);jacobian=np.broadcast_to(np.eye(2),(len(x),2,2)).copy();logdet=np.zeros(len(x));fields={};backs=[]
    for i in range(4):
        h,stage,local,back=coupling(p,h,i);jacobian=local@jacobian;logdet=logdet+stage['logdet'];backs.append(back)
        fields.update({f'coupling{i}':h,f'coupling_logdet{i}':stage['logdet'],f'coupling_scale{i}':stage['widths'],f'coupling_shift{i}':stage['heights'],f'coupling_derivatives{i}':stage['derivatives']})
    scale=np.exp(p['log_scale']);z=h*scale;jacobian=scale[None,:,None]*jacobian;logdet=logdet+p['log_scale'].sum()
    base=-(.5*(z*z).sum(axis=1)+LOG_2PI)
    fields.update(latent=z,unscaled=h,forward_logdet=logdet,base_log_probability=base,log_probability=base+logdet,
                  loss=np.asarray(-(base+logdet).mean()),jacobian_row0=jacobian[:,0],jacobian_row1=jacobian[:,1])
    def backward():
        dy=z/len(z);dld=-np.ones(len(z))/len(z);partials={'log_scale':(dy*z).sum(axis=0)+dld.sum()};dy=dy*scale
        for back in reversed(backs):dy=back(dy,dld,partials)
        return partials,dy
    return fields,backward


def inverse(p,z):
    scale=np.exp(-p['log_scale']);h=z.astype(float)*scale;jacobian=np.broadcast_to(np.diag(scale),(len(z),2,2)).copy()
    logdet=np.zeros(len(z))-p['log_scale'].sum()
    for i in reversed(range(4)):
        h,stage,local,_=coupling(p,h,i,True);jacobian=local@jacobian;logdet=logdet+stage['logdet']
    return h,logdet,jacobian


def evaluate(p,x):
    fields,_=forward(p,x);value,ld,_=inverse(p,fields['latent']);return dict(fields,reconstruction=value,inverse_logdet=ld)


def sample(p,z):
    value,ld,jac=inverse(p,z)
    return dict(prior=z,samples=value,prior_roundtrip=forward(p,value)[0]['latent'],inverse_logdet=ld,inverse_jacobian_row0=jac[:,0],inverse_jacobian_row1=jac[:,1])


def fixtures(load,counts):
    f=load('spline_fixtures');x=f['input'].astype(float);heads=f['heads'].astype(float)
    declared=[-10,-4,-3.999,-2.001,-2,-1.999,-.001,0,.001,1.999,2,2.001,3.999,4,10]+[-6+i*12/256 for i in range(257)]
    exact(f['input'],np.array(declared,np.float32).reshape(272,1))
    exact(f['heads'],np.tile(np.array([0,0,0,0,.2,-.3,.1,.4,-.2,.3,.6],np.float32),(272,1)))
    fw,dv,dl=spline(x,heads);iv,di,dil=spline(x,heads,True);rv,_,_=spline(fw['value'],heads,True)
    identity,_,_=spline(x,np.zeros_like(heads));ii,_,_=spline(x,np.zeros_like(heads),True)
    gradient=2*fw['value']*dv-dl+.3*(2*iv['value']*di-dil)
    expected=dict(fw,heads=heads,input=x,roundtrip=rv['value'],inverse=iv['value'],inverse_logdet=iv['logdet'],
        input_jacobian=dv[:,:1],inverse_input_jacobian=di[:,:1],input_gradient=gradient[:,:1],head_gradient=gradient[:,1:],
        identity_value=identity['value'],identity_inverse=ii['value'],identity_logdet=identity['logdet'])
    assert set(expected)==set(f)
    maximum=0
    for key,want in expected.items():
        np.testing.assert_allclose(f[key],want,atol=5e-6,rtol=3e-5,err_msg='fixture/'+key)
        maximum=max(maximum,float(np.max(abs(f[key]-want))))
    assert np.all(dv[:,0]>0) and np.all(di[:,0]>0)
    np.testing.assert_allclose(np.log(dv[:,0]),fw['logdet'],atol=2e-15,rtol=0)
    np.testing.assert_allclose(np.log(di[:,0]),iv['logdet'],atol=2e-15,rtol=0)
    np.testing.assert_allclose(rv['value'],x,atol=4e-15,rtol=0)
    tail=abs(x[:,0])>4;assert np.all(dv[tail,0]==1) and np.all(di[tail,0]==1)
    assert np.all(dv[tail,1:]==0) and np.all(di[tail,1:]==0) and np.all(dl[tail]==0) and np.all(dil[tail]==0)
    # At knots the log-derivative can have different one-sided input/head partials.
    # Finite differences use rows away from both x and y knots; exact knot/tail
    # values and positive first derivatives are checked separately above.
    xk=np.array([-4,-2,0,2,4]);yk=np.r_[-4,-4+np.cumsum(fw['heights'][0,:3]),4]
    selected=np.min(abs(x-np.r_[xk,yk]),axis=1)>1e-4
    def objective(a,b):
        g,_,_=spline(a,b);r,_,_=spline(a,b,True)
        return (g['value'][:,0]**2-g['logdet'])+.3*(r['value'][:,0]**2-r['logdet'])
    fd_error=0
    for c in range(12):
        hi=x.copy();lo=x.copy();hh=heads.copy();lh=heads.copy()
        if c==0:hi+=1e-5;lo-=1e-5
        else:hh[:,c-1]+=1e-5;lh[:,c-1]-=1e-5
        fd=(objective(hi,hh)-objective(lo,lh))/2e-5
        np.testing.assert_allclose(fd[selected],gradient[selected,c],atol=2e-8,rtol=2e-6)
        fd_error=max(fd_error,float(abs(fd[selected]-gradient[selected,c]).max()))
    counts['fixture_values']=sum(v.size for v in f.values());counts['fixture_input_and_head_gradient_values']=gradient.size
    counts['fixture_finite_differences']=int(selected.sum())*12
    return dict(points=272,all_values_and_input_and_head_gradients_checked=True,
        all_knots_neighbours_and_C1_identity_tails_retained=True,closed_quadratic_inverse_no_root_search=True,
        maximum_Float32_source_difference=maximum,maximum_fixture_finite_difference_error=fd_error,
        rounded_offset_zero_heads_are_near_identity_in_Double=True)


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
    assert len(p0) == 17 and sum(v.size for v in p0.values()) == 878
    for i in range(4):
        assert p0[f'coupling{i}/hidden/W'].shape == (1, 16)
        assert p0[f'coupling{i}/heads/W'].shape == (16, 11)
        exact(p0[f'coupling{i}/hidden/b'], np.full(16, .05, np.float32))
        exact(p0[f'coupling{i}/heads/b'], np.zeros(11, np.float32))
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
    # Independently differentiated Double matrices must match their own analytic
    # determinants much more tightly than the original Float32 source tolerance.
    for fields,ld in [(forward(final,test)[0],forward(final,test)[0]['forward_logdet'])]:
        matrix=np.stack([fields['jacobian_row0'],fields['jacobian_row1']],axis=1)
        np.testing.assert_allclose(np.linalg.slogdet(matrix)[1],ld,atol=8e-15,rtol=0)
    _,ld,matrix=inverse(final,prior)
    np.testing.assert_allclose(np.linalg.slogdet(matrix)[1],ld,atol=8e-15,rtol=0)
    counts['forward_Jacobian_entries'], counts['inverse_Jacobian_entries'] = jac.size, ijac.size
    assert np.ptp(observed['forward_logdet']) > 1e-5
    assert np.ptp(sampled['inverse_logdet']) > 1e-5
    np.testing.assert_allclose(observed['forward_logdet'] + observed['inverse_logdet'], 0, atol=5e-6, rtol=0)
    density_at_samples = forward(final, sampled['samples'])[0]
    np.testing.assert_allclose(density_at_samples['forward_logdet'] + sampled['inverse_logdet'], 0, atol=1e-14, rtol=0)
    for block in range(4):
        widths=observed[f'coupling_scale{block}'];heights=observed[f'coupling_shift{block}']
        derivatives=observed[f'coupling_derivatives{block}']
        assert np.min(widths)>0 and np.min(heights)>0 and np.min(derivatives)>0
        np.testing.assert_allclose(widths.sum(axis=1),8,atol=6e-7,rtol=0)
        np.testing.assert_allclose(heights.sum(axis=1),8,atol=6e-7,rtol=0)
        assert np.all(derivatives[:,[0,-1]]==1)

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
    fixture_receipt=fixtures(load,counts)
    assert audited == set(pins['archive_sha256']), audited ^ set(pins['archive_sha256'])
    return dict(passed=True, counts=counts, maximum_errors=errors, archives_checked=len(audited),
                raw_midpoint_MT_Box_Muller_and_all_banana_rows_exact=True,
                all_48_Adam_writes_and_iterations_bit_exact=True,
                all_17_leaf_finite_differences=True, analytic_spline_coupling_matrix_Jacobians=True,
                positive_actual_bin_widths_heights_and_knot_derivatives=True,
                fixture_checks=fixture_receipt,
                source_Float32_softplus_offset_preserved=OFFSET,
                spatially_varying_forward_and_inverse_path_logdets=True,
                positive_actual_forward_and_inverse_determinants_match_per_point_logdets=True,
                all_1024_prior_inverse_samples_and_normalization_identities=True,
                fitting_only_Gaussian_baselines_and_analytic_truth_checked=True,
                original_rounded_Float32_log_2pi_constant_preserved=LOG_2PI,
                source_improvement_over_fitting_only_Gaussian_preserved=True,
                test_nll=metrics['test_nll'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('build/nn-report-conversion/spline_density'))
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    rendered = json.dumps(check(args.root), indent=2) + '\n'
    print(rendered, end='')
    if args.json:
        args.json.write_text(rendered)


if __name__ == '__main__':
    main()
