#!/usr/bin/env python3
"""Optional independent NumPy oracle for saved deterministic WAE-MMD artifacts.

Never trains, prepares data or downloads. Checks full first derivatives and Adam,
all signed IMQ terms, explicit priors, forwards and reconstruction-based selection.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact, gaussian
from check_factor_tc_variational_autoencoder import first_update
from check_info_variational_autoencoder import decode, mmd
from check_vector_quantized_autoencoder import factor_data


def forward(p, image, prior):
    """Deterministic tanh encoder, sigmoid decoder and signed aggregate discrepancy."""
    x=image.reshape(len(image),-1).astype(np.float64)
    hidden=np.tanh(x@p['encoder/W']+p['encoder/b'])
    code=hidden@p['code/W']+p['code/b']
    logits,probability,decoder_hidden=decode(p,code)
    squared_cost=((x-probability)**2).sum(1)
    discrepancy,(gradient,prior_gradient)=mmd(code,prior.astype(np.float64))
    out=dict(code=code,logits=logits,probability=probability,squared_cost=squared_cost,
             reconstruction_loss=squared_cost.mean(),**discrepancy)
    out['loss']=out['reconstruction_loss']+10*out['mmd']
    return out,(x,hidden,decoder_hidden,gradient,prior_gradient)


def derivatives(p,image,prior):
    """All eight leaf chain rules, with the aggregate MMD branch added at the code."""
    out,(x,hidden,decoder_hidden,aggregate,_)=forward(p,image,prior)
    probability=out['probability']
    delta=2*(probability-x)*probability*(1-probability)/len(x)
    result={'output/W':decoder_hidden.T@delta,'output/b':delta.sum(0)}
    back=(delta@p['output/W'].T)*(1-decoder_hidden*decoder_hidden)
    result.update({'decoder/W':out['code'].T@back,'decoder/b':back.sum(0)})
    code_gradient=back@p['decoder/W'].T+10*aggregate
    result.update({'code/W':hidden.T@code_gradient,'code/b':code_gradient.sum(0)})
    back=(code_gradient@p['code/W'].T)*(1-hidden*hidden)
    result.update({'encoder/W':x.T@back,'encoder/b':back.sum(0)})
    return result


def check(directory):
    """Verify original/report archives and return measured independent evidence."""
    data=arrays(directory/'data.npz');pixels=factor_data(data);noise=arrays(directory/'fixed_noise.npz')
    noise_values=0
    for split,count,seed in [('fit',210,3719),('validation',42,3725),('test',42,3731),('generation',20,3749)]:
        expected=gaussian((1,count,4),seed)
        np.testing.assert_allclose(noise[split+'_standard'],expected,atol=1e-14,rtol=1e-14)
        exact(noise[split+'_epsilon'],expected.astype(np.float32));noise_values+=expected.size
    for epoch in range(1,4):
        bank=arrays(directory/f'train_noise_epoch{epoch}.npz');expected=gaussian((1,210,4),3779+epoch)
        np.testing.assert_allclose(bank['standard'],expected,atol=1e-14,rtol=1e-14)
        exact(bank['epsilon'],expected.astype(np.float32));noise_values+=expected.size
    algebra=arrays(directory/'mmd_algebra.npz');x,y=algebra['x'].astype(np.float64),algebra['y'].astype(np.float64)
    terms,(gx,gy)=mmd(x,y)
    for saved,key in [('xx','mmd_xx'),('yy','mmd_yy'),('xy','mmd_xy'),('mmd','mmd')]:
        np.testing.assert_allclose(algebra[saved],terms[key],atol=2e-7,rtol=2e-6)
    for key,value in [('gradient_x',gx),('gradient_y',gy)]:np.testing.assert_allclose(algebra[key],value,atol=2e-7,rtol=2e-6)
    same,_=mmd(x,x);assert same['mmd']<0 and algebra['same_sample_unbiased_mmd'].item()<0
    np.testing.assert_allclose(algebra['same_sample_unbiased_mmd'],same['mmd'],atol=2e-7,rtol=2e-6)
    assert algebra['all_identical_mmd'].item()==0
    np.testing.assert_allclose(gx.sum(0)+gy.sum(0),0,atol=1e-14,rtol=0)
    initial=arrays(directory/'initial_arrays.npz');p={k:v.astype(np.float64) for k,v in initial.items()}
    batch=arrays(directory/'first_batch.npz');actual=arrays(directory/'first_gradients.npz')
    expected=derivatives(p,batch['images'],batch['prior']);assert set(expected)==set(actual)
    gradient_error=0.
    for key,value in expected.items():
        np.testing.assert_allclose(actual[key],value,atol=3e-6,rtol=5e-4)
        gradient_error=max(gradient_error,float(np.max(np.abs(value-actual[key]))))
    first_update(initial,actual,arrays(directory/'first_update_arrays.npz'))
    fields=values=0;maximum=0.
    def endpoints(out,saved):
        nonlocal fields,values,maximum
        assert set(out)==set(saved)
        for key,value in out.items():
            value=np.asarray(value);np.testing.assert_allclose(saved[key],value,atol=3e-5,rtol=2e-5)
            maximum=max(maximum,float(np.max(np.abs(value-saved[key]))));fields+=1;values+=value.size
    out,_=forward(p,batch['images'],batch['prior']);endpoints(out,arrays(directory/'first_forward.npz'))
    selected=arrays(directory/'model_arrays.npz');p={k:v.astype(np.float64) for k,v in selected.items()};saved=arrays(directory/'outputs.npz')
    for split in ['fit','validation','test']:
        out,_=forward(p,data[split+'_images'],noise[split+'_epsilon'][0]);endpoints(out,{key:saved[split+'_'+key] for key in out})
    history=np.genfromtxt(directory/'history.tsv',delimiter='\t',names=True);criteria=[]
    for epoch in range(1,4):
        ep={k:v.astype(np.float64) for k,v in arrays(directory/f'epoch{epoch}_arrays.npz').items()}
        out,_=forward(ep,data['validation_images'],noise['validation_epsilon'][0]);criteria.append(out['reconstruction_loss'])
        for column,key in [('validation_squared_cost','reconstruction_loss'),('validation_mmd','mmd'),('validation_objective','loss')]:
            np.testing.assert_allclose(history[column][epoch-1],out[key],atol=3e-5,rtol=2e-5)
    meta=json.loads((directory/'metrics.json').read_text());chosen=int(np.argmin(criteria))+1
    assert meta['selected_epoch']==chosen and meta['updates']==21 and meta['epochs']==3 and meta['mmd_weight']==10
    for k,v in arrays(directory/f'epoch{chosen}_arrays.npz').items():exact(v,selected[k])
    for name,key in [('test_squared_cost','reconstruction_loss'),('test_mmd','mmd'),('test_objective','loss')]:
        np.testing.assert_allclose(meta[name],saved['test_'+key],atol=0,rtol=0)
    prior=arrays(directory/'prior.npz');exact(prior['z'],noise['generation_epsilon'][0])
    _,probability,_=decode(p,prior['z']);np.testing.assert_allclose(prior['probability'],probability,atol=3e-6,rtol=2e-5)
    test=data['test_images'].reshape(-1,784).astype(np.float64)
    np.testing.assert_allclose(meta['test_mse'],((saved['test_probability']-test)**2).mean(),atol=5e-13,rtol=1e-14)
    fitmean=data['fit_images'].reshape(-1,784).astype(np.float64).mean(0).astype(np.float32)
    np.testing.assert_allclose(meta['fit_mean_mse'],((fitmean-test)**2).mean(),atol=5e-13,rtol=1e-14)
    batches=np.genfromtxt(directory/'batches.tsv',delimiter='\t',names=True,dtype=np.int64);assert len(batches)==630
    for epoch in range(1,4):
        rows=batches[batches['epoch']==epoch];exact(np.sort(rows['row']),np.arange(210))
        exact(rows['source_id'],data['fit_source_ids'][rows['row']].astype(np.int64))
        exact(rows['step'],np.repeat(np.arange((epoch-1)*7,epoch*7),[32]*6+[18]))
    return dict(passed=True,parameter_derivatives_checked=sum(v.size for v in expected.values()),first_gradient_max_error=gradient_error,
                first_Adam_update_bit_exact=True,lattice_pixels_checked=pixels,Gaussian_values_checked=noise_values,
                signed_MMD_and_translation_controls=True,endpoint_fields_checked=fields,endpoint_values_checked=values,
                maximum_endpoint_error=maximum,eligible_validation_checkpoints_checked=3,selected_epoch=chosen,
                batch_history_rows_checked=630,unfiltered_prior_predictions_checked=20)


def main():
    """Check already completed evidence and optionally save its measured receipt."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('--json',type=Path)
    args=parser.parse_args();text=json.dumps(check(args.directory),indent=2)+'\n'
    if args.json:args.json.write_text(text)
    print(text,end='')


if __name__=='__main__':
    main()
