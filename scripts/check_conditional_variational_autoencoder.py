#!/usr/bin/env python3
"""Optional NumPy oracle for complete conditional-VAE report artifacts, with no fitting.

Verifies all first-batch parameter derivatives, exact first Adam update, explicit
Gaussian banks, heldout posterior/prior/shifted-class controls, concatenations,
fitting-only class means and every eligible validation negative-ELBO criterion.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact, gaussian


def decode(p, z, labels):
    """Independent class-conditioned ReLU/linear Bernoulli decoder in Float64."""
    condition=np.eye(10)[labels]
    joined=np.concatenate([z.astype(np.float64),condition],axis=1)
    hidden=np.maximum(joined@p['decoder/W']+p['decoder/b'],0)
    logits=hidden@p['output/W']+p['output/b']
    return 1/(1+np.exp(-logits)),logits,hidden,joined


def forward(p, x, labels, epsilon):
    """Independent class-conditioned Gaussian encoder and explicit pathwise sample."""
    x,epsilon=x.astype(np.float64),epsilon.astype(np.float64)
    joined=np.concatenate([x,np.eye(10)[labels]],axis=1)
    hidden=np.maximum(joined@p['encoder/W']+p['encoder/b'],0)
    mu=hidden@p['mu/W']+p['mu/b'];lv=hidden@p['log_variance/W']+p['log_variance/b']
    sigma=np.exp(.5*lv);z=mu+sigma*epsilon
    probability,logits,dh,dj=decode(p,z,labels)
    nll=(np.logaddexp(0,logits)-x*logits).sum(axis=1)
    kl=.5*(mu*mu+np.exp(lv)-1-lv).sum(axis=1)
    return dict(mu=mu,log_variance=lv,z=z,probability=probability,nll=nll,kl=kl), (joined,hidden,sigma,dh,dj)


def derivatives(p, x, labels, epsilon):
    """Full chain rule through both heads, reparameterization, decoder and analytic KL."""
    x,epsilon=x.astype(np.float64),epsilon.astype(np.float64)
    out,(joined,hidden,sigma,dh,dj)=forward(p,x,labels,epsilon)
    delta=(out['probability']-x)/len(x)
    result={'output/W':dh.T@delta,'output/b':delta.sum(axis=0)}
    back=(delta@p['output/W'].T)*(dh>0)
    result.update({'decoder/W':dj.T@back,'decoder/b':back.sum(axis=0)})
    dz=(back@p['decoder/W'].T)[:,:2]
    dmu=dz+out['mu']/len(x)
    dlv=dz*.5*sigma*epsilon+.5*(np.exp(out['log_variance'])-1)/len(x)
    result.update({'mu/W':hidden.T@dmu,'mu/b':dmu.sum(axis=0),
                   'log_variance/W':hidden.T@dlv,'log_variance/b':dlv.sum(axis=0)})
    back=(dmu@p['mu/W'].T+dlv@p['log_variance/W'].T)*(hidden>0)
    result.update({'encoder/W':joined.T@back,'encoder/b':back.sum(axis=0)})
    return result


def check(directory):
    """Assert complete numerical/conditioning/data fidelity and return measured error bounds."""
    data,noise=arrays(directory/'data.npz'),arrays(directory/'fixed_noise.npz')
    count_pixels,noise_values,identities=0,0,set()
    for split,count,seed in [('fit',1000,2141),('validation',200,2131),('test',200,2137)]:
        images,original,labels,ids=[data[split+suffix] for suffix in ['_images','_original_images','_labels','_source_ids']]
        assert images.shape==(count,28,28,1) and original.shape==images.shape
        assert not identities.intersection(ids);identities.update(ids)
        assert ((ids>=60000)&(ids<70000)).all() if split=='test' else ((ids>=0)&(ids<60000)).all()
        assert (np.bincount(labels,minlength=10)==count//10).all()
        exact(images,(original>=.5).astype(np.float32));count_pixels+=images.size
        raw=np.rint(original*255).astype(np.float32)
        exact(original,raw/np.float32(255))
        expected=gaussian((4,count,2),seed)
        np.testing.assert_allclose(noise[split+'_standard'],expected,atol=1e-14,rtol=1e-14)
        exact(noise[split+'_epsilon'],expected.astype(np.float32));noise_values+=expected.size
    assert len(identities)==1400
    for epoch in range(1,4):
        bank=arrays(directory/f'train_noise_epoch{epoch}.npz');expected=gaussian((1,1000,2),2129+epoch)
        np.testing.assert_allclose(bank['standard'],expected,atol=1e-14,rtol=1e-14)
        exact(bank['epsilon'],expected.astype(np.float32));noise_values+=expected.size
    initial=arrays(directory/'initial_arrays.npz');p={k:v.astype(np.float64) for k,v in initial.items()}
    batch,actual,updated=[arrays(directory/(name+'.npz')) for name in ['first_batch','first_gradients','first_update_arrays']]
    expected=derivatives(p,batch['images'].reshape(-1,784),batch['labels'],batch['epsilon'])
    gradient_error,derivative_count=0,0
    for name,value in expected.items():
        np.testing.assert_allclose(actual[name],value,atol=2e-6,rtol=5e-4)
        gradient_error=max(gradient_error,float(np.max(np.abs(value-actual[name]))));derivative_count+=value.size
        g=actual[name];first,second=g*np.float32(.1),(g*g)*np.float32(.001)
        direction=(first/np.float32(.1))/(np.sqrt(second/np.float32(.001))+np.float32(1e-8))
        exact(initial[name]-direction*np.float32(.003),updated[name])
    assert derivative_count==102932
    weights=arrays(directory/'model_arrays.npz');p={k:v.astype(np.float64) for k,v in weights.items()};output=arrays(directory/'outputs.npz')
    probability_error=0
    for split in ['fit','validation','test']:
        x,labels=data[split+'_images'].reshape(-1,784),data[split+'_labels']
        mean,_=forward(p,x,labels,np.zeros((len(x),2)))
        for field,key in [('mu','mu'),('log_variance','log_variance'),('probability','mean_reconstruction'),('kl','kl')]:
            np.testing.assert_allclose(mean[field],output[split+'_'+key],atol=1e-4,rtol=1e-5)
        count_pixels+=mean['probability'].size
        for k,epsilon in enumerate(noise[split+'_epsilon']):
            out,_=forward(p,x,labels,epsilon)
            for field,key in [('z','sample_z'),('probability','sample_prediction'),('nll','sample_nll')]:
                np.testing.assert_allclose(out[field],output[split+'_'+key][k],atol=3e-4,rtol=1e-5)
            probability_error=max(probability_error,float(np.max(np.abs(out['probability']-output[split+'_sample_prediction'][k]))));count_pixels+=out['probability'].size
    shifted=(data['test_labels']+1)%10;exact(shifted,output['wrong_labels'])
    for codes,name in [(output['test_mu'],'wrong_label_mean_prediction'),(output['test_sample_z'][0],'wrong_label_sample_prediction')]:
        probability,*_=decode(p,codes,shifted);np.testing.assert_allclose(probability,output[name],atol=1e-5,rtol=1e-5);count_pixels+=probability.size
    changed,_=forward(p,data['test_images'].reshape(-1,784),shifted,noise['test_epsilon'][0])
    for field in ['mu','log_variance']:np.testing.assert_allclose(changed[field],output['wrong_label_'+field],atol=1e-5,rtol=1e-5)
    zero,*_=decode(p,np.zeros((200,2)),data['test_labels'])
    np.testing.assert_allclose(zero,output['zero_latent_prediction'],atol=1e-5,rtol=1e-5);count_pixels+=zero.size
    probe=arrays(directory/'condition_probe.npz');hot=np.eye(10,dtype=np.float32)[probe['labels']]
    exact(probe['one_hot'],hot)
    exact(probe['encoder_input'],np.concatenate([probe['images'].reshape(-1,784),hot],axis=1))
    exact(probe['decoder_input'],np.concatenate([probe['z'],hot],axis=1))
    prior=arrays(directory/'prior.npz');expected=gaussian((1,5,2),2143)
    np.testing.assert_allclose(prior['standard'],expected,atol=1e-14,rtol=1e-14)
    exact(prior['base_z'],expected[0].astype(np.float32));noise_values+=expected.size
    exact(prior['z'],np.tile(prior['base_z'],(10,1)))
    exact(prior['labels'],np.repeat(np.arange(10,dtype=np.int32),5))
    grid=np.array([[x,y] for y in np.arange(2,-2.01,-.5) for x in np.arange(-2,2.01,.5)],dtype=np.float32)
    exact(grid,prior['grid_z']);exact(prior['grid_labels'],np.full(81,3,dtype=np.int32))
    for codes,labels,name in [('z','labels','probability'),('grid_z','grid_labels','grid_probability')]:
        probability,*_=decode(p,prior[codes],prior[labels]);np.testing.assert_allclose(probability,prior[name],atol=1e-5,rtol=1e-5);count_pixels+=probability.size
    fit=data['fit_images'].reshape(-1,784).astype(np.float64)
    class_means=np.array([fit[data['fit_labels']==c].mean(axis=0) for c in range(10)],dtype=np.float32)
    class_prediction=class_means[data['test_labels']]
    exact(class_prediction,arrays(directory/'class_mean.npz')['test_class_mean'])
    history=np.genfromtxt(directory/'history.tsv',delimiter='\t',names=True);scores=[]
    for epoch in range(1,4):
        ep={k:v.astype(np.float64) for k,v in arrays(directory/f'epoch{epoch}_arrays.npz').items()};nll=0
        for epsilon in noise['validation_epsilon']:
            out,_=forward(ep,data['validation_images'].reshape(-1,784),data['validation_labels'],epsilon);nll+=out['nll'].mean()/4
        kl=out['kl'].mean();scores.append(nll+kl)
        for field,value in [('validation_reconstruction_nll',nll),('validation_kl',kl),('validation_negative_elbo',nll+kl)]:
            np.testing.assert_allclose(history[field][epoch-1],value,atol=3e-4,rtol=1e-6)
    chosen=int(np.argmin(scores)+1)
    for k,v in arrays(directory/f'epoch{chosen}_arrays.npz').items():exact(v,weights[k])
    test=data['test_images'].reshape(-1,784).astype(np.float64)
    return dict(all_first_batch_derivatives_checked=derivative_count,first_gradient_max_error=gradient_error,
                first_Adam_update_bit_exact=True,posterior_probability_max_error=probability_error,
                Gaussian_values_checked=noise_values,data_and_prediction_pixels_checked=count_pixels,
                selected_epoch=chosen,eligible_validation_checkpoints_checked=3,
                condition_concatenations_and_fit_class_means_bit_exact=True,
                fit_global_mean_test_MSE=float(((fit.mean(axis=0).astype(np.float32)-test)**2).mean()),
                fit_class_mean_test_MSE=float(((class_prediction-test)**2).mean()))


def main():
    """Run the optional development oracle and write a compact evidence receipt."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('--json',type=Path)
    args=parser.parse_args();result=check(args.directory)
    if args.json:args.json.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
