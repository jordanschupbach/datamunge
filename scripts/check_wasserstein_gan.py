#!/usr/bin/env python3
"""Independent NumPy WGAN and WGAN-GP oracle; reads frozen C++ artifacts only.

No fitting, data preparation or downloads. Includes the critic input Jacobian,
closed-form second-order penalty partials, all first critic/G updates and clips,
fixed final predictions, input/noise streams and batch/optimizer ownership.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact, gaussian


def uniform(rows,seed):
    """Local MT19937 raw midpoint words with one final Float32 cast."""
    words=np.random.RandomState(seed).randint(0,2**32,size=rows,dtype=np.uint32)
    return ((words.astype(float)+.5)/4294967296.).astype(np.float32).reshape(rows,1)


def map_forward(p,group,x):
    """Tanh-hidden linear-output map and its complete analytic adjoint."""
    h=np.tanh(x@p[group+'hidden/W']+p[group+'hidden/b'])
    y=h@p[group+'output/W']+p[group+'output/b']
    def back(delta):
        dh=(delta@p[group+'output/W'].T)*(1-h*h)
        return dh@p[group+'hidden/W'].T,{group+'hidden/W':x.T@dh,group+'hidden/b':dh.sum(axis=0),group+'output/W':h.T@delta,group+'output/b':delta.sum(axis=0)}
    return y,back


def forward(p,x,z,alpha,gp):
    """Independent forward, detached interpolation and all second-order critic partials."""
    fake,back_g=map_forward(p,'0/',z)
    real_score,back_real=map_forward(p,'1/',x);fake_score,back_fake=map_forward(p,'1/',fake)
    interpolation=alpha*x+(1-alpha)*fake
    w,v=p['1/hidden/W'],p['1/output/W'].reshape(16)
    hidden=np.tanh(interpolation@w+p['1/hidden/b']);a=1-hidden*hidden
    jacobian=(a*v)@w.T
    norm=np.sqrt((jacobian*jacobian).sum(axis=1)+1e-12)
    penalty=((norm-1)**2).mean();gap=real_score.mean()-fake_score.mean()
    result=dict(fake=fake,real_score=real_score,fake_score=fake_score,interpolation=interpolation,
                input_gradient=jacobian,symbolic_input_gradient=jacobian,norm=norm,
                penalty=penalty,critic_gap=gap,generator_loss=-fake_score.mean(),critic_loss=-gap+(10*penalty if gp else 0))
    n=len(x);_,real=back_real(np.full_like(real_score,-1/n));_,generated=back_fake(np.full_like(fake_score,1/n));critic={key:real[key]+generated[key] for key in real}
    # The interpolation values are fixed leaves for penalty differentiation:
    # there is no external real/alpha/G path through this chain rule.
    delta=2*(norm-1)[:,None]/norm[:,None]*jacobian/n
    dot=delta@w
    pre=(-2*hidden*a)*v*dot
    penalty_partials={'1/hidden/W':interpolation.T@pre+(delta.T@a)*v,
                      '1/hidden/b':pre.sum(axis=0),'1/output/W':(dot*a).sum(axis=0).reshape(16,1),
                      '1/output/b':np.zeros(1)}
    if gp: critic={key:critic[key]+10*penalty_partials[key] for key in critic}
    dfake,_=back_fake(np.full_like(fake_score,-1/n));_,generator=back_g(dfake)
    return result,generator,critic,penalty_partials


def check(root):
    """Validate every frozen derivative, state transition and final native field."""
    data=arrays(root/'original-output/data.npz');priors=arrays(root/'probe-priors.npz')
    values=0;errors=[];derivatives=0;differences=0;update_values=0
    for split,seed in [('fit',5507),('validation',5519),('test',5521)]:
        n=len(data[split]);noise=gaussian((n,2),seed);centers=np.array([[2*np.cos(2*np.pi*(i%8)/8),2*np.sin(2*np.pi*(i%8)/8)] for i in range(n)])
        expected=(centers+.08*noise).astype(np.float32);exact(data[split],expected)
    for name,count,seed in [('validation_z',128,5527),('test_z',128,5531),('g_z',32,7601)]+[('critic_z'+str(i),32,5601+i) for i in range(3)]:exact(priors[name],gaussian((count,2),seed).astype(np.float32))
    for name,count,seed in [('validation_alpha',128,5537),('test_alpha',128,5549),('g_alpha',32,8601)]+[('critic_alpha'+str(i),32,6601+i) for i in range(3)]:exact(priors[name],uniform(count,seed))
    for gp,variant in [(False,'clipped'),(True,'gradient_penalty')]:
        load=lambda name: arrays(root/variant/(name+'.npz'))
        before=load('probe-initial');original=arrays(root/'original-output'/variant/'initial_arrays.npz')
        for key in before:exact(before[key],original[key])
        first_state={key:np.zeros_like(v) for key,v in before.items()}
        second_state={key:np.zeros_like(v) for key,v in before.items()}
        batches=np.genfromtxt(root/'original-output'/variant/'batches.tsv',names=True,delimiter='\t',dtype=np.int64)
        for epoch in range(1,4): assert sorted(batches[batches['epoch']==epoch]['row'].tolist())==list(range(400))
        rows=batches[batches['step']==0]['row'];exact(priors['first_real'],data['fit'][rows]);x=priors['first_real'].astype(float)
        for step,tag,group in [(1,'c0',1),(2,'c1',1),(3,'c2',1),(1,'g',0)]:
            z=priors['g_z' if group==0 else 'critic_z'+tag[-1]].astype(float)
            alpha=priors['g_alpha' if group==0 else 'critic_alpha'+tag[-1]].astype(float)
            p={k:v.astype(float) for k,v in before.items()}
            expected,gg,gc,_=forward(p,x,z,alpha,gp);native=load('probe-forward-'+tag)
            for key,value in expected.items():
                np.testing.assert_allclose(native[key],value,atol=3e-6,rtol=1e-4,err_msg=variant+'/'+tag+'/'+key)
                values+=np.size(value);errors.append(float(np.max(abs(native[key]-value))))
            actual=load('probe-gradient-'+tag);expected=gc if group else gg
            assert set(actual)==set(expected)
            for key,value in expected.items():
                np.testing.assert_allclose(actual[key],value,atol=6e-6,rtol=2e-3,err_msg=variant+'/'+tag+'/'+key)
                derivatives+=value.size;errors.append(float(np.max(abs(actual[key]-value))))
            after=load('probe-after-step-'+tag) if group else load('probe-after-g')
            for key,value in before.items():
                if key not in actual:exact(value,after[key]);continue
                g=actual[key]
                if gp:
                    first_state[key]=first_state[key]*np.float32(0)+g
                    second_state[key]=second_state[key]*np.float32(.9)+(g*g)*np.float32(.1)
                    direction=first_state[key]/(np.sqrt(second_state[key]/np.float32(1-.9**step))+np.float32(1e-8))
                    delta=direction*np.float32(.0001)
                else:
                    first_state[key]=first_state[key]*np.float32(.9)+(g*g)*np.float32(.1)
                    direction=g/np.sqrt(first_state[key]+np.float32(1e-8))
                    delta=direction*np.float32(.00005)
                exact(value-delta,after[key]);update_values+=value.size
            if group:
                clipped=load('probe-after-'+tag)
                for key,value in after.items(): exact(np.clip(value,np.float32(-.01),np.float32(.01)) if key.startswith('1/') and not gp else value,clipped[key])
                before=clipped
            else:before=after
        final=arrays(root/'original-output'/variant/'final.npz');p={k:v.astype(float) for k,v in final.items()}
        out,gg,gc,_=forward(p,data['test'].astype(float),priors['test_z'].astype(float),priors['test_alpha'].astype(float),gp)
        actual=load('probe-final-output')
        for key,value in out.items():
            np.testing.assert_allclose(actual[key],value,atol=3e-6,rtol=1e-4,err_msg=variant+'/final/'+key);values+=np.size(value);errors.append(float(np.max(abs(actual[key]-value))))
        for group,expected in [(0,gg),(1,gc)]:
            actual=load('probe-final-gradient-'+str(group))
            for key,value in expected.items():
                np.testing.assert_allclose(actual[key],value,atol=6e-6,rtol=2e-3,err_msg=variant+'/final/'+key);derivatives+=value.size;errors.append(float(np.max(abs(actual[key]-value))))
        controls=load('probe-noise-controls')
        for name,z in [('changed',gaussian((128,2),5573).astype(np.float32).astype(float)),('zero',np.zeros((128,2)))]:
            predicted,_=map_forward(p,'0/',z);np.testing.assert_allclose(controls[name],predicted,atol=3e-6,rtol=1e-5);values+=predicted.size
        metrics=json.loads((root/'original-output'/variant/'metrics.json').read_text())
        for key in ['critic_loss','generator_loss','critic_gap','penalty']:np.testing.assert_allclose(metrics['test_'+key],out[key],atol=3e-6,rtol=1e-4)
        assert metrics['generator_steps']==39 and metrics['critic_steps']==117
        if not gp: assert all(np.max(abs(value))<=np.float32(.01) for key,value in final.items() if key.startswith('1/'))
        x=data['test'][:3].astype(float);z=priors['test_z'][:3].astype(float);alpha=priors['test_alpha'][:3].astype(float)
        _,gg,gc,_=forward(p,x,z,alpha,gp)
        for group,expected in [(0,gg),(1,gc)]:
            for key,value in expected.items():
                for index in sorted(set([0,value.size//2,value.size-1])):
                    old=p[key].flat[index];h=1e-6
                    p[key].flat[index]=old+h;plus=forward(p,x,z,alpha,gp)[0]['generator_loss' if group==0 else 'critic_loss']
                    p[key].flat[index]=old-h;minus=forward(p,x,z,alpha,gp)[0]['generator_loss' if group==0 else 'critic_loss']
                    p[key].flat[index]=old
                    # The critic objective stops interpolation through fake, so its
                    # parameter finite difference is valid. G uses only its own loss.
                    np.testing.assert_allclose((plus-minus)/(2*h),value.flat[index],atol=3e-7,rtol=2e-4,err_msg=variant+'/'+key);differences+=1
    return dict(passed=True,forward_values_checked=values,native_parameter_derivatives_checked=derivatives,
                maximum_checked_error=max(errors),independent_final_scalar_finite_differences=differences,
                first_four_updates_per_variant_bit_exact=True,updated_parameter_values_exact=update_values,
               	all_three_clips_include_biases=True,midpoint_uniform_and_Gaussian_banks_exact=True,
                ordered_ring_values_exact=sum(v.size for v in data.values()),batch_rows_checked_per_variant=1200,
                clipping_RMSProp_epsilon_inside_root_and_GP_Adam_separate=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path('build/nn-report-conversion/wasserstein_gan'));parser.add_argument('--json',type=Path)
    args=parser.parse_args();result=check(args.root);rendered=json.dumps(result,indent=2)+'\n';print(rendered,end='')
    if args.json:args.json.write_text(rendered)


if __name__=='__main__':main()
