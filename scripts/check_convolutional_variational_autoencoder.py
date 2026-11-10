#!/usr/bin/env python3
"""Optional independent NumPy oracle for completed convolutional-VAE artifacts.

No preparation, fitting or downloads. Checks complete initial/selected derivatives,
Float32 first Adam update, every spatial stage and posterior/prior/grid prediction,
explicit Gaussian banks, static MNIST thresholding and validation selection.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact, gaussian
from check_convolutional_autoencoder import convolution, convolution_adjoint, pool, repeat


def decode(p,z):
    """Independent sample-only projection/repeat/convolution decoder, in Float64."""
    z=z.astype(np.float64)
    base=np.maximum(z@p['decoder_projection/W']+p['decoder_projection/b'],0).reshape(-1,7,7,4)
    enlarged=repeat(base)
    hidden=np.maximum(convolution(enlarged,p['decoder1/W'],p['decoder1/b']),0)
    second=repeat(hidden)
    spatial=convolution(second,p['decoder2/W'],p['decoder2/b'])
    logits=spatial.reshape(-1,784);probability=1/(1+np.exp(-logits))
    return dict(decoder_base=base,decoder1=hidden,spatial_logits=spatial,logits=logits,probability=probability), (enlarged,second)


def forward(p,x,epsilon):
    """Independent pooled spatial encoder, diagonal Gaussian and Bernoulli objective."""
    x,epsilon=x.astype(np.float64),epsilon.astype(np.float64)
    c1=np.maximum(convolution(x,p['encoder1/W'],p['encoder1/b']),0);p1=pool(c1)
    c2=np.maximum(convolution(p1,p['encoder2/W'],p['encoder2/b']),0);p2=pool(c2)
    flat=p2.reshape(-1,196)
    mu=flat@p['mu/W']+p['mu/b'];lv=flat@p['log_variance/W']+p['log_variance/b']
    sigma=np.exp(.5*lv);z=mu+sigma*epsilon
    decoded,cache=decode(p,z)
    nll=(np.logaddexp(0,decoded['logits'])-x.reshape(-1,784)*decoded['logits']).sum(axis=1)
    kl=.5*(mu*mu+np.exp(lv)-1-lv).sum(axis=1)
    return dict(conv1=c1,pool1=p1,conv2=c2,pool2=p2,mu=mu,log_variance=lv,z=z,nll=nll,kl=kl,**decoded),(flat,sigma,cache)


def derivatives(p,x,epsilon):
    """All shared-kernel, Gaussian-head and dense-projection chain-rule derivatives."""
    x,epsilon=x.astype(np.float64),epsilon.astype(np.float64)
    out,(flat,sigma,(enlarged,second))=forward(p,x,epsilon)
    delta=((out['probability']-x.reshape(-1,784))/len(x)).reshape(x.shape)
    expected={}
    g,expected['decoder2/W'],expected['decoder2/b']=convolution_adjoint(second,p['decoder2/W'],delta)
    g=4*pool(g)*(out['decoder1']>0)
    g,expected['decoder1/W'],expected['decoder1/b']=convolution_adjoint(enlarged,p['decoder1/W'],g)
    g=(4*pool(g)*(out['decoder_base']>0)).reshape(-1,196)
    expected['decoder_projection/W']=out['z'].T@g;expected['decoder_projection/b']=g.sum(axis=0)
    dz=g@p['decoder_projection/W'].T
    dmu=dz+out['mu']/len(x)
    dlv=dz*.5*sigma*epsilon+.5*(np.exp(out['log_variance'])-1)/len(x)
    expected.update({'mu/W':flat.T@dmu,'mu/b':dmu.sum(axis=0),'log_variance/W':flat.T@dlv,'log_variance/b':dlv.sum(axis=0)})
    g=(dmu@p['mu/W'].T+dlv@p['log_variance/W'].T).reshape(out['pool2'].shape)
    g=repeat(g)/4*(out['conv2']>0)
    g,expected['encoder2/W'],expected['encoder2/b']=convolution_adjoint(out['pool1'],p['encoder2/W'],g)
    g=repeat(g)/4*(out['conv1']>0)
    _,expected['encoder1/W'],expected['encoder1/b']=convolution_adjoint(x,p['encoder1/W'],g)
    return expected


def check(directory):
    """Validate every complete artifact and return measured finite-precision error bounds."""
    data,noise=arrays(directory/'data.npz'),arrays(directory/'fixed_noise.npz')
    pixels,noise_values,identities=0,0,set()
    for split,count,seed in [('fit',1000,2063),('validation',200,2039),('test',200,2053)]:
        images,original,ids=[data[split+suffix] for suffix in ['_images','_original_images','_source_ids']]
        assert images.shape==(count,28,28,1) and original.shape==images.shape
        assert not identities.intersection(ids);identities.update(ids)
        assert ((ids>=60000)&(ids<70000)).all() if split=='test' else ((ids>=0)&(ids<60000)).all()
        exact(images,(original>=.5).astype(np.float32));pixels+=images.size
        raw=np.rint(original*255).astype(np.float32);exact(original,raw/np.float32(255))
        expected=gaussian((4,count,2),seed)
        np.testing.assert_allclose(noise[split+'_standard'],expected,atol=1e-14,rtol=1e-14)
        exact(noise[split+'_epsilon'],expected.astype(np.float32));noise_values+=expected.size
    assert len(identities)==1400 and (np.bincount(data['test_labels'],minlength=10)==20).all()
    for epoch in range(1,4):
        bank=arrays(directory/f'train_noise_epoch{epoch}.npz');expected=gaussian((1,1000,2),2027+epoch)
        np.testing.assert_allclose(bank['standard'],expected,atol=1e-14,rtol=1e-14)
        exact(bank['epsilon'],expected.astype(np.float32));noise_values+=expected.size
    initial=arrays(directory/'initial_arrays.npz');selected=arrays(directory/'model_arrays.npz')
    gradient_count,gradient_errors=0,{}
    for label,weights,batch_name,gradient_name in [('initial',initial,'first_batch','first_gradients'),('selected',selected,'selected_gradient_batch','selected_gradients')]:
        p={k:v.astype(np.float64) for k,v in weights.items()};batch=arrays(directory/(batch_name+'.npz'));actual=arrays(directory/(gradient_name+'.npz'))
        expected=derivatives(p,batch['images'],batch['epsilon']);assert set(expected)==set(actual)
        maximum=0
        for name,value in expected.items():
            np.testing.assert_allclose(actual[name],value,atol=8e-5,rtol=5e-4)
            maximum=max(maximum,float(np.max(np.abs(value-actual[name]))));gradient_count+=value.size
        gradient_errors[label]=maximum
    assert gradient_count==4234
    updated=arrays(directory/'first_update_arrays.npz');actual=arrays(directory/'first_gradients.npz')
    for name,g in actual.items():
        first,second=g*np.float32(.1),(g*g)*np.float32(.001)
        direction=(first/np.float32(.1))/(np.sqrt(second/np.float32(.001))+np.float32(1e-8))
        exact(initial[name]-direction*np.float32(.003),updated[name])
    p={k:v.astype(np.float64) for k,v in selected.items()};output=arrays(directory/'outputs.npz');prediction_error=0
    for split in ['fit','validation','test']:
        x=data[split+'_images'];mean,_=forward(p,x,np.zeros((len(x),2)))
        for field,key in [('mu','mu'),('log_variance','log_variance'),('probability','mean_reconstruction'),('kl','kl')]:
            np.testing.assert_allclose(mean[field],output[split+'_'+key],atol=3e-5,rtol=1e-5)
        pixels+=mean['probability'].size
        for k,epsilon in enumerate(noise[split+'_epsilon']):
            out,_=forward(p,x,epsilon)
            for field,key in [('z','sample_z'),('probability','sample_prediction'),('nll','sample_nll')]:
                np.testing.assert_allclose(out[field],output[split+'_'+key][k],atol=3e-4,rtol=1e-5)
            prediction_error=max(prediction_error,float(np.max(np.abs(out['probability']-output[split+'_sample_prediction'][k]))));pixels+=out['probability'].size
    stage=arrays(directory/'stages.npz');out,_=forward(p,stage['images'],stage['epsilon']);stage_errors={}
    for i,key in enumerate(['conv1','pool1','conv2','pool2','decoder_base','decoder1','spatial_logits']):
        np.testing.assert_allclose(out[key],stage['stage'+str(i)],atol=3e-5,rtol=1e-5)
        stage_errors[key]=float(np.max(np.abs(out[key]-stage['stage'+str(i)])))
    zero,_=decode(p,np.zeros((200,2)));np.testing.assert_allclose(zero['probability'],output['zero_latent_prediction'],atol=3e-5,rtol=1e-5);pixels+=zero['probability'].size
    prior=arrays(directory/'prior.npz');expected=gaussian((1,20,2),2069)
    np.testing.assert_allclose(prior['standard'],expected,atol=1e-14,rtol=1e-14);exact(prior['z'],expected[0].astype(np.float32));noise_values+=expected.size
    grid=np.array([[x,y] for y in np.arange(2,-2.01,-.5) for x in np.arange(-2,2.01,.5)],dtype=np.float32);exact(grid,prior['grid_z'])
    for key,name in [('z','probability'),('grid_z','grid_probability')]:
        out,_=decode(p,prior[key]);np.testing.assert_allclose(out['probability'],prior[name],atol=3e-5,rtol=1e-5);pixels+=out['probability'].size
    history=np.genfromtxt(directory/'history.tsv',delimiter='\t',names=True);scores=[]
    for epoch in range(1,4):
        ep={k:v.astype(np.float64) for k,v in arrays(directory/f'epoch{epoch}_arrays.npz').items()};nll=0
        for epsilon in noise['validation_epsilon']:
            out,_=forward(ep,data['validation_images'],epsilon);nll+=out['nll'].mean()/4
        kl=out['kl'].mean();scores.append(nll+kl)
        for field,value in [('validation_reconstruction_nll',nll),('validation_kl',kl),('validation_negative_elbo',nll+kl)]:
            np.testing.assert_allclose(history[field][epoch-1],value,atol=3e-4,rtol=1e-6)
    chosen=int(np.argmin(scores)+1)
    for k,v in arrays(directory/f'epoch{chosen}_arrays.npz').items():exact(v,selected[k])
    fit=data['fit_images'].reshape(-1,784).astype(np.float64);test=data['test_images'].reshape(-1,784).astype(np.float64)
    return dict(initial_and_selected_derivatives_checked=gradient_count,gradient_max_errors=gradient_errors,
                first_Adam_update_bit_exact=True,posterior_probability_max_error=prediction_error,
                seven_stage_max_errors=stage_errors,Gaussian_values_checked=noise_values,
                data_and_prediction_pixels_checked=pixels,eligible_validation_checkpoints_checked=3,
                selected_epoch=chosen,fit_pixel_mean_test_MSE=float(((fit.mean(axis=0).astype(np.float32)-test)**2).mean()))


def main():
    """Run this optional development-only numerical oracle and save a compact receipt."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('--json',type=Path)
    args=parser.parse_args();result=check(args.directory)
    if args.json:args.json.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
