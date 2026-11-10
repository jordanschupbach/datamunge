#!/usr/bin/env python3
"""Optional independent NumPy oracle for completed beta-VAE report artifacts.

No fitting or preparation: verifies every first-batch derivative, first Adam
update, explicit noise bank, posterior/prior/traversal prediction and fitting-only
probe, plus every eligible common validation criterion and selected checkpoint.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact, gaussian


def decode(p, z):
    """Independent ReLU/linear decoder and Bernoulli probabilities in Float64."""
    h = np.maximum(z @ p['decoder/W'] + p['decoder/b'], 0)
    logits = h @ p['output/W'] + p['output/b']
    return logits, 1/(1+np.exp(-logits)), h


def forward(p, x, epsilon):
    """Independent Gaussian encoder, explicit pathwise sampling and analytic KL."""
    x, epsilon = x.astype(np.float64), epsilon.astype(np.float64)
    h = np.maximum(x @ p['encoder/W'] + p['encoder/b'], 0)
    mu = h @ p['mu/W'] + p['mu/b']
    lv = h @ p['log_variance/W'] + p['log_variance/b']
    sigma = np.exp(.5*lv)
    z = mu + sigma*epsilon
    logits, probability, dh = decode(p, z)
    nll = (np.logaddexp(0, logits)-x*logits).sum(axis=1)
    kl = .5*(mu*mu+np.exp(lv)-1-lv).sum(axis=1)
    return dict(mu=mu, log_variance=lv, z=z, probability=probability, nll=nll, kl=kl), (h, sigma, dh)


def derivatives(p, x, epsilon, beta):
    """All ten leaf derivatives, including both pathwise and analytic-KL branches."""
    x, epsilon = x.astype(np.float64), epsilon.astype(np.float64)
    out, (h, sigma, dh) = forward(p, x, epsilon)
    delta = (out['probability']-x)/len(x)
    result = {'output/W': dh.T @ delta, 'output/b': delta.sum(axis=0)}
    back = (delta @ p['output/W'].T)*(dh > 0)
    result.update({'decoder/W': out['z'].T @ back, 'decoder/b': back.sum(axis=0)})
    dz = back @ p['decoder/W'].T
    dmu = dz + beta*out['mu']/len(x)
    dlv = dz*.5*sigma*epsilon + .5*beta*(np.exp(out['log_variance'])-1)/len(x)
    result.update({'mu/W': h.T @ dmu, 'mu/b': dmu.sum(axis=0),
                   'log_variance/W': h.T @ dlv, 'log_variance/b': dlv.sum(axis=0)})
    back = (dmu @ p['mu/W'].T+dlv @ p['log_variance/W'].T)*(h > 0)
    result.update({'encoder/W': x.T @ back, 'encoder/b': back.sum(axis=0)})
    return result


def probes(fit, test, factors, targets, lv):
    """Independent standardized ridge and best-fitting-axis probes, generic L/K."""
    fit, test, factors, targets, lv = [v.astype(np.float64) for v in (fit,test,factors,targets,lv)]
    center, variance = fit.mean(axis=0), fit.var(axis=0)
    scale = np.maximum(np.sqrt(variance),1e-8)
    factor_mean = factors.mean(axis=0)
    x, xt, y = (fit-center)/scale, (test-center)/scale, factors-factor_mean
    def correlations(a,b):
        a, b = a-a.mean(axis=0), b-b.mean(axis=0)
        aa, bb = (a*a).sum(axis=0), (b*b).sum(axis=0)
        denominator = np.sqrt(aa[:,None]*bb)
        return np.divide(a.T@b,denominator,out=np.zeros_like(denominator),where=(aa[:,None]>1e-20)&(bb>0))
    fc, tc = correlations(fit,factors),correlations(test,targets)
    coef = np.linalg.solve(x.T@x+.001*np.eye(x.shape[1]),x.T@y)
    axis = np.argmax(np.abs(fc),axis=0).astype(np.int32)
    one = (x[:,axis]*y).sum(axis=0)/(.001+(x[:,axis]**2).sum(axis=0))
    prediction, single = factor_mean+xt@coef, factor_mean+xt[:,axis]*one
    ss = ((targets-targets.mean(axis=0))**2).sum(axis=0)
    assert (ss>0).all()
    return dict(latent_mean=center,latent_std=scale,factor_mean=factor_mean,mu_variance=variance,
                mean_kl_per_unit=(.5*(fit*fit+np.exp(lv)-1-lv)).mean(axis=0),
                fit_correlation=fc,test_correlation=tc,ridge_coefficients=coef,best_axis_by_fit=axis,
                ridge_prediction=prediction,single_prediction=single,
                ridge_r2=1-((prediction-targets)**2).sum(axis=0)/ss,
                single_r2=1-((single-targets)**2).sum(axis=0)/ss)


def check(directory):
    """Check complete artifacts and return counts/error bounds without training."""
    data, noise = arrays(directory/'data.npz'),arrays(directory/'fixed_noise.npz')
    pixels, gaussian_values, gradients_checked = 0,0,0
    all_ids = set()
    for split,count in [('fit',210),('validation',42),('test',42)]:
        ids,factors,images = [data[split+suffix] for suffix in ['_source_ids','_factors','_images']]
        assert len(ids)==count and not all_ids.intersection(ids)
        all_ids.update(ids)
        yy,xx = np.mgrid[:28,:28]
        for row,identity in enumerate(ids):
            shape,size,x,y = int(identity)//147,(int(identity)//49)%3,(int(identity)//7)%7,int(identity)%7
            np.testing.assert_array_equal(factors[row],[shape,size,x,y])
            residue=(x+2*y+size+3*shape)%7
            assert residue==0 if split=='test' else residue==1 if split=='validation' else residue>=2
            dx,dy=abs(xx-(3*x+5)),abs(yy-(3*y+5))
            expected=np.maximum(dx,dy)<=size+2 if shape else dx*dx+dy*dy<=(size+2)**2
            exact(expected.astype(np.float32)[...,None],images[row])
        expected=gaussian((4,count,4),dict(fit=2273,validation=2267,test=2269)[split])
        np.testing.assert_allclose(noise[split+'_standard'],expected,atol=1e-14,rtol=1e-14)
        exact(noise[split+'_epsilon'],expected.astype(np.float32));gaussian_values+=expected.size;pixels+=images.size
    assert len(all_ids)==294
    expected=gaussian((1,20,4),2281)
    np.testing.assert_allclose(noise['prior_standard'],expected,atol=1e-14,rtol=1e-14)
    exact(noise['prior_epsilon'],expected.astype(np.float32));gaussian_values+=expected.size
    for epoch in range(1,4):
        bank=arrays(directory/f'train_noise_epoch{epoch}.npz');expected=gaussian((1,210,4),2251+epoch)
        np.testing.assert_allclose(bank['standard'],expected,atol=1e-14,rtol=1e-14)
        exact(bank['epsilon'],expected.astype(np.float32));gaussian_values+=expected.size
    evidence=[]
    for beta in [1,4]:
        root=directory/f'beta{beta}'
        initial=arrays(root/'initial_arrays.npz');p={k:v.astype(np.float64) for k,v in initial.items()}
        batch,actual,updated=[arrays(root/(name+'.npz')) for name in ['first_batch','first_gradients','first_update_arrays']]
        expected=derivatives(p,batch['images'].reshape(-1,784),batch['epsilon'],beta)
        gradient_error=0
        for name,value in expected.items():
            np.testing.assert_allclose(actual[name],value,atol=2e-6,rtol=5e-4)
            gradient_error=max(gradient_error,float(np.max(np.abs(value-actual[name]))));gradients_checked+=value.size
            g=actual[name];first,second=g*np.float32(.1),(g*g)*np.float32(.001)
            direction=(first/np.float32(.1))/(np.sqrt(second/np.float32(.001))+np.float32(1e-8))
            exact(initial[name]-direction*np.float32(.003),updated[name])
        selected=arrays(root/'model_arrays.npz');p={k:v.astype(np.float64) for k,v in selected.items()}
        output=arrays(root/'outputs.npz');prediction_error=0
        for split in ['fit','validation','test']:
            x=data[split+'_images'].reshape(-1,784)
            mean,_=forward(p,x,np.zeros((len(x),4)))
            for field,key in [('mu','mu'),('log_variance','log_variance'),('probability','mean_prediction'),('kl','kl')]:
                np.testing.assert_allclose(mean[field],output[split+'_'+key],atol=4e-5,rtol=1e-5)
            for k,epsilon in enumerate(noise[split+'_epsilon']):
                out,_=forward(p,x,epsilon)
                for field,key in [('z','sample_z'),('probability','sample_prediction'),('nll','sample_nll')]:
                    np.testing.assert_allclose(out[field],output[split+'_'+key][k],atol=3e-4,rtol=1e-5)
                prediction_error=max(prediction_error,float(np.max(np.abs(out['probability']-output[split+'_sample_prediction'][k]))));pixels+=out['probability'].size
        prior=arrays(root/'prior.npz')
        exact(prior['z'],noise['prior_epsilon'][0])
        center=output['fit_mu'].astype(np.float64).mean(axis=0).astype(np.float32)
        exact(center,prior['traversal_center'])
        traversal=np.tile(center,(36,1))
        for d in range(4):traversal[d*9:(d+1)*9,d]+=np.arange(-2,2.01,.5,dtype=np.float32)
        exact(traversal,prior['traversal_z'])
        for codes,prob in [('z','probability'),('traversal_z','traversal_probability')]:
            _,prediction,_=decode(p,prior[codes].astype(np.float64))
            np.testing.assert_allclose(prediction,prior[prob],atol=1e-5,rtol=1e-5);pixels+=prediction.size
        expected=probes(output['fit_mu'],output['test_mu'],data['fit_factors'],data['test_factors'],output['fit_log_variance'])
        saved=arrays(root/'factor_probes.npz')
        for name,value in expected.items():np.testing.assert_allclose(saved[name],value,atol=2e-11,rtol=2e-11)
        history=np.genfromtxt(root/'history.tsv',delimiter='\t',names=True)
        scores=[]
        for epoch in range(1,4):
            ep={k:v.astype(np.float64) for k,v in arrays(root/f'epoch{epoch}_arrays.npz').items()}
            x=data['validation_images'].reshape(-1,784);nll=0
            for epsilon in noise['validation_epsilon']:
                out,_=forward(ep,x,epsilon);nll+=out['nll'].mean()/4
            kl=out['kl'].mean();scores.append(nll+kl)
            for field,value in [('validation_nll',nll),('validation_kl',kl),('validation_negative_elbo',nll+kl),('validation_weighted_objective',nll+beta*kl)]:
                np.testing.assert_allclose(history[field][epoch-1],value,atol=3e-4,rtol=1e-6)
        chosen=int(np.argmin(scores)+1)
        for k,v in arrays(root/f'epoch{chosen}_arrays.npz').items():exact(v,selected[k])
        evidence.append(dict(beta=beta,selected_epoch=chosen,first_gradient_max_error=gradient_error,
                             posterior_probability_max_error=prediction_error,first_Adam_update_bit_exact=True,
                             thirteen_probe_arrays_checked=True))
    exact(arrays(directory/'beta1/initial_arrays.npz')['encoder/W'],arrays(directory/'beta4/initial_arrays.npz')['encoder/W'])
    return dict(all_first_batch_derivatives_checked=gradients_checked,Gaussian_values_checked=gaussian_values,
                lattice_and_prediction_pixels_checked=pixels,eligible_validation_checkpoints_checked=6,models=evidence)


def main():
    """Run the optional development-only oracle and save a compact validation receipt."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('--json',type=Path)
    args=parser.parse_args();result=check(args.directory)
    if args.json:args.json.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
