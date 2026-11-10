#!/usr/bin/env python3
"""Optional independent NumPy oracle for the complete matched K1/K5 IWAE artifacts.

No fitting or preparation. Checks sampled densities, all ten first-batch gradients,
first Adam updates, explicit noise, saved predictions/weights/ESS, all eligible
common K20 validation scores and fitting-only factor probes.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact, gaussian
from check_beta_variational_autoencoder import forward as gaussian_forward, decode, probes
from check_factor_tc_variational_autoencoder import first_update


def forward(p, images, epsilon):
    """Independent sample-major Gaussian forward and stable sampled-density objective."""
    k, b, latent = epsilon.shape
    x = np.tile(images.reshape(b, -1).astype(np.float64), (k, 1))
    eps = epsilon.reshape(k*b, latent).astype(np.float64)
    out, cache = gaussian_forward(p, x, eps)
    # Match the experiment's float-rounded density constant, without using native outputs.
    log2pi = float(np.float32(np.log(2*np.pi)))
    logp = -.5*(out['z']**2+log2pi).sum(axis=1)
    logq = -.5*((out['z']-out['mu'])**2*np.exp(-out['log_variance'])+out['log_variance']+log2pi).sum(axis=1)
    logw = (logp-out['nll']-logq).reshape(k, b)
    peak = logw.max(axis=0, keepdims=True)
    expw = np.exp(logw-peak)
    weights = expw/expw.sum(axis=0, keepdims=True)
    negative = -(peak+np.log(expw.sum(axis=0, keepdims=True))-float(np.float32(np.log(k))))[0]
    out.update(log_weights=logw, normalized_weights=weights, negative_bound=negative,
               ess=1/(weights**2).sum(axis=0))
    out['loss'] = negative.mean()
    return out, (x, eps, cache)


def derivatives(p, images, epsilon):
    """All leaf adjoints of the ordinary pathwise sampled IWAE objective."""
    out, (x, eps, (h, sigma, dh)) = forward(p, images, epsilon)
    k, b, latent = epsilon.shape
    weight = out['normalized_weights'].reshape(k*b, 1)/b
    delta = weight*(out['probability']-x)
    result = {'output/W': dh.T@delta, 'output/b': delta.sum(axis=0)}
    back = (delta@p['output/W'].T)*(dh > 0)
    result.update({'decoder/W': out['z'].T@back, 'decoder/b': back.sum(axis=0)})
    residual = out['z']-out['mu']
    precision = np.exp(-out['log_variance'])
    dz = back@p['decoder/W'].T+weight*(out['z']-residual*precision)
    dmu = dz+weight*residual*precision
    dlv = .5*dz*sigma*eps+.5*weight*(residual**2*precision-1)
    result.update({'mu/W': h.T@dmu, 'mu/b': dmu.sum(axis=0),
                   'log_variance/W': h.T@dlv, 'log_variance/b': dlv.sum(axis=0)})
    back = (dmu@p['mu/W'].T+dlv@p['log_variance/W'].T)*(h > 0)
    result.update({'encoder/W': x.T@back, 'encoder/b': back.sum(axis=0)})
    return result


def check(directory):
    """Validate full source artifacts and return independent counts and error bounds."""
    data, noise = arrays(directory/'data.npz'), arrays(directory/'fixed_noise.npz')
    pixels = gaussian_values = gradients_checked = checkpoints = 0
    ids_seen = set()
    for split, count, seed in [('fit',210,2411),('validation',42,2417),('test',42,2423)]:
        ids, factors, images = [data[split+suffix] for suffix in ['_source_ids','_factors','_images']]
        assert len(ids)==count and not ids_seen.intersection(ids)
        ids_seen.update(ids)
        yy, xx = np.mgrid[:28,:28]
        for row, identity in enumerate(ids):
            shape, size, x, y = int(identity)//147, (int(identity)//49)%3, (int(identity)//7)%7, int(identity)%7
            np.testing.assert_array_equal(factors[row], [shape,size,x,y])
            residue=(x+2*y+size+3*shape)%7
            assert residue==0 if split=='test' else residue==1 if split=='validation' else residue>=2
            dx,dy=abs(xx-(3*x+5)),abs(yy-(3*y+5))
            expected=np.maximum(dx,dy)<=size+2 if shape else dx*dx+dy*dy<=(size+2)**2
            exact(images[row],expected.astype(np.float32)[...,None]);pixels+=784
        expected=gaussian((20,count,4),seed)
        np.testing.assert_allclose(noise[split+'_standard'],expected,atol=1e-14,rtol=1e-14)
        exact(noise[split+'_epsilon'],expected.astype(np.float32));gaussian_values+=expected.size
    assert len(ids_seen)==294
    expected=gaussian((1,20,4),2429)
    np.testing.assert_allclose(noise['prior_standard'],expected,atol=1e-14,rtol=1e-14)
    exact(noise['prior_epsilon'],expected.astype(np.float32));gaussian_values+=expected.size
    for epoch in range(1,4):
        bank=arrays(directory/f'train_noise_epoch{epoch}.npz');expected=gaussian((5,210,4),2437+epoch)
        np.testing.assert_allclose(bank['standard'],expected,atol=1e-14,rtol=1e-14)
        exact(bank['epsilon'],expected.astype(np.float32));gaussian_values+=expected.size
    algebra=arrays(directory/'algebra.npz');w=algebra['log_weights'].astype(np.float64)
    peak=w.max(axis=0);e=np.exp(w-peak);weights=e/e.sum(axis=0)
    np.testing.assert_allclose(algebra['negative_bound'],-peak-np.log(e.sum(axis=0)/3),atol=5e-5,rtol=0)
    np.testing.assert_allclose(algebra['gradient'],-weights/2,atol=1e-7,rtol=0)
    exact(algebra['single'],algebra['duplicate'])
    evidence=[]
    for k in [1,5]:
        root=directory/f'k{k}'
        initial=arrays(root/'initial_arrays.npz');p={key:v.astype(np.float64) for key,v in initial.items()}
        batch,actual,updated=[arrays(root/(name+'.npz')) for name in ['first_batch','first_gradients','first_update_arrays']]
        expected=derivatives(p,batch['images'],batch['epsilon']);gradient_error=0
        assert set(actual)==set(initial)==set(expected) and len(actual)==10
        for name,v in expected.items():
            np.testing.assert_allclose(actual[name],v,atol=2e-5,rtol=5e-3,err_msg=f'K{k}:{name}')
            gradient_error=max(gradient_error,float(np.max(abs(v-actual[name]))));gradients_checked+=v.size
        first_update(initial,actual,updated)
        first,_=forward(p,batch['images'],batch['epsilon'])
        saved_first=arrays(root/'first_forward.npz')
        for name in saved_first:
            np.testing.assert_allclose(saved_first[name],first[name],atol=4e-4,rtol=5e-5,err_msg=name)
        selected=arrays(root/'model_arrays.npz');best=arrays(root/'best.npz')
        for name in selected:exact(selected[name],best[name])
        p={name:v.astype(np.float64) for name,v in selected.items()}
        saved=arrays(root/'outputs.npz');max_probability=max_weight=0
        for split,count in [('fit',210),('validation',42),('test',42)]:
            image=data[split+'_images']
            draw,_=forward(p,image,noise[split+'_epsilon'])
            mean,_=forward(p,image,np.zeros((1,count,4)))
            expected={'mu':draw['mu'][:count],'log_variance':draw['log_variance'][:count],
                      'mean_prediction':mean['probability'],'sample_z':draw['z'],
                      'sample_prediction':draw['probability'],'sample_nll':draw['nll'],
                      'kl':draw['kl'][:count],'log_weights':draw['log_weights'],
                      'normalized_weights':draw['normalized_weights'],'negative_bound':draw['negative_bound'],'ess':draw['ess']}
            for name,value in expected.items():
                tol=2e-4 if name in ['normalized_weights','ess'] else 8e-4
                np.testing.assert_allclose(saved[split+'_'+name],value,atol=tol,rtol=2e-4,err_msg=split+'_'+name)
            max_probability=max(max_probability,float(np.max(abs(draw['probability']-saved[split+'_sample_prediction']))))
            max_weight=max(max_weight,float(np.max(abs(draw['normalized_weights']-saved[split+'_normalized_weights']))))
            pixels+=draw['probability'].size+mean['probability'].size
            assert np.all((draw['ess']>=1-1e-12)&(draw['ess']<=20+1e-12))
        prior=arrays(root/'prior.npz');exact(prior['z'],noise['prior_epsilon'][0])
        prediction=decode(p,prior['z'].astype(np.float64))[1]
        np.testing.assert_allclose(prior['probability'],prediction,atol=2e-6,rtol=1e-5);pixels+=prediction.size
        estimated=arrays(root/'test_estimates.npz')
        for samples in [1,5,20]:
            out,_=forward(p,data['test_images'],noise['test_epsilon'][:samples])
            for name in ['negative_bound','ess']:
                np.testing.assert_allclose(estimated[name+f'_k{samples}'],out[name],atol=8e-4,rtol=2e-4)
        probe=arrays(root/'factor_probes.npz')
        expected=probes(saved['fit_mu'],saved['test_mu'],data['fit_factors'],data['test_factors'],saved['fit_log_variance'])
        for name,value in expected.items():np.testing.assert_allclose(probe[name],value,atol=2e-7,rtol=2e-6)
        history=np.genfromtxt(root/'history.tsv',names=True,delimiter='\t');criteria=[]
        for row in history:
            epoch=int(row[0]);ep=arrays(root/f'epoch{epoch}_arrays.npz');ep={name:v.astype(np.float64) for name,v in ep.items()}
            out,_=forward(ep,data['validation_images'],noise['validation_epsilon'])
            np.testing.assert_allclose(out['loss'],row[2],atol=3e-4,rtol=2e-6)
            np.testing.assert_allclose(out['ess'].mean(),row[3],atol=2e-4,rtol=2e-4)
            criteria.append(out['loss']);checkpoints+=1
        meta=json.loads((root/'metrics.json').read_text());chosen=int(np.argmin(criteria))+1
        assert chosen==meta['selected_epoch']==3
        for name,value in arrays(root/f'epoch{chosen}_arrays.npz').items():exact(value,selected[name])
        batches=np.genfromtxt(root/'batches.tsv',names=True,delimiter='\t',dtype=np.int64)
        assert len(batches)==630 and len(set(batches['step']))==21
        for epoch in [1,2,3]:
            rows=batches[batches['epoch']==epoch];assert sorted(rows['row'])==list(range(210))
            np.testing.assert_array_equal(rows['source_id'],data['fit_source_ids'][rows['row']])
        evidence.append(dict(training_samples=k,first_gradient_max_error=gradient_error,
                             first_Adam_update_bit_exact=True,posterior_probability_max_error=max_probability,
                             normalized_weight_max_error=max_weight,selected_epoch=chosen))
    for name,value in arrays(directory/'k1/initial_arrays.npz').items():exact(value,arrays(directory/'k5/initial_arrays.npz')[name])
    return dict(first_parameter_derivatives_checked=gradients_checked,Gaussian_values_checked=gaussian_values,
                lattice_and_prediction_pixels_checked=pixels,common_validation_checkpoints_checked=checkpoints,models=evidence)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('--json',type=Path)
    args=parser.parse_args();result=check(args.directory)
    text=json.dumps(result,indent=2)+'\n'
    if args.json:args.json.write_text(text)
    print(text,end='')


if __name__=='__main__':main()
