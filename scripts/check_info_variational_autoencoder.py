#!/usr/bin/env python3
"""Optional independent NumPy oracle for complete paired InfoVAE/MMD-VAE artifacts.

No fitting/preparation. Checks every first leaf derivative, exact first Adam,
signed IMQ matrices/terms, all sixteen forward endpoints, explicit Gaussian banks,
heldout decoding and every eligible own-objective validation criterion.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact, gaussian
from check_factor_tc_variational_autoencoder import first_update


def kernel(x,y):
    """Fixed three-scale IMQ, independently formed from pairwise differences."""
    difference=x[:,None,:]-y[None,:,:]
    distance=(difference*difference).sum(axis=2)
    value=sum(c/(c+distance) for c in (1.,4.,16.))/3
    scale=sum(-2*c/(c+distance)**2 for c in (1.,4.,16.))/3
    return value,(difference,scale)


def mmd(x,y):
    """All seven signed unbiased terms and both independent sample adjoints."""
    n,m=len(x),len(y)
    xx,cxx=kernel(x,x);yy,cyy=kernel(y,y);xy,cxy=kernel(x,y)
    within_x=(xx.sum()-n)/(n*(n-1));within_y=(yy.sum()-m)/(m*(m-1));cross=xy.mean()
    def adjoint(cache,weight):
        d,scale=cache;g=weight[:,:,None]*scale[:,:,None]*d
        return g.sum(axis=1),-g.sum(axis=0)
    a,b=adjoint(cxx,(1-np.eye(n))/(n*(n-1)))
    c,d=adjoint(cyy,(1-np.eye(m))/(m*(m-1)))
    e,f=adjoint(cxy,np.full((n,m),-2/(n*m)))
    return dict(mmd_xx=xx,mmd_yy=yy,mmd_xy=xy,mmd_within_x=within_x,
                mmd_within_y=within_y,mmd_cross=cross,mmd=within_x+within_y-2*cross),(a+b+e,c+d+f)


def decode(p,z):
    """Independent tanh/linear decoder and stable Bernoulli probabilities."""
    h=np.tanh(z@p['decoder/W']+p['decoder/b'])
    logits=h@p['output/W']+p['output/b']
    return logits,1/(1+np.exp(-logits)),h


def forward(p,image,epsilon,prior,alpha):
    """All sixteen endpoints in Float64 with explicit independent prior sampling."""
    x=image.reshape(len(image),-1).astype(np.float64);epsilon=epsilon.astype(np.float64);prior=prior.astype(np.float64)
    h=np.tanh(x@p['encoder/W']+p['encoder/b'])
    mu=h@p['mu/W']+p['mu/b'];lv=h@p['log_variance/W']+p['log_variance/b']
    sigma=np.exp(.5*lv);z=mu+sigma*epsilon
    logits,probability,dh=decode(p,z)
    nll=(np.logaddexp(0,logits)-x*logits).sum(axis=1)
    kl=.5*(mu*mu+np.exp(lv)-1-lv).sum(axis=1)
    discrepancy,(dz,dy)=mmd(z,prior)
    out=dict(mu=mu,log_variance=lv,z=z,logits=logits,probability=probability,nll=nll,
             kl=kl,negative_elbo=(nll+kl).mean(),**discrepancy)
    out['loss']=(nll+(1-alpha)*kl).mean()+(alpha+9)*out['mmd']
    return out,(x,h,sigma,dh,dz,dy)


def derivatives(p,image,epsilon,prior,alpha):
    """Every shared leaf adjoint, including sampled aggregate and local KL branches."""
    out,(x,h,sigma,dh,aggregate,_)=forward(p,image,epsilon,prior,alpha)
    delta=(out['probability']-x)/len(x)
    result={'output/W':dh.T@delta,'output/b':delta.sum(axis=0)}
    back=(delta@p['output/W'].T)*(1-dh*dh)
    result.update({'decoder/W':out['z'].T@back,'decoder/b':back.sum(axis=0)})
    dz=back@p['decoder/W'].T+(alpha+9)*aggregate
    dmu=dz+(1-alpha)*out['mu']/len(x)
    dlv=.5*dz*sigma*epsilon+.5*(1-alpha)*(np.exp(out['log_variance'])-1)/len(x)
    result.update({'mu/W':h.T@dmu,'mu/b':dmu.sum(axis=0),
                   'log_variance/W':h.T@dlv,'log_variance/b':dlv.sum(axis=0)})
    back=(dmu@p['mu/W'].T+dlv@p['log_variance/W'].T)*(1-h*h)
    result.update({'encoder/W':x.T@back,'encoder/b':back.sum(axis=0)})
    return result


def check(directory):
    """Validate complete artifacts against independent arithmetic and return evidence."""
    pixels=gaussian_values=gradient_count=kernel_count=checkpoints=0
    algebra=arrays(directory/'coefficient_algebra.npz')
    for key,value in dict(info=8.875,mmd_only=7.5,vae=8,gradient_mmd=9.5).items():
        assert algebra[key].item()==value
    np.testing.assert_array_equal(algebra['gradient_nll'],[.5,.5])
    np.testing.assert_array_equal(algebra['gradient_kl'],[.25,.25])
    evidence=[]
    for variant,alpha in [('info_vae',.5),('mmd_vae',1.)]:
        root=directory/variant;data=arrays(root/'data.npz');noise=arrays(root/'fixed_noise.npz')
        ids_seen=set();yy,xx=np.mgrid[:28,:28]
        for index,(split,count) in enumerate([('fit',210),('validation',42),('test',42)]):
            ids,factors,images=[data[split+suffix] for suffix in ['_source_ids','_factors','_images']]
            assert len(ids)==count and not ids_seen.intersection(ids);ids_seen.update(ids)
            for row,identity in enumerate(ids):
                shape,size,x,y=int(identity)//147,(int(identity)//49)%3,(int(identity)//7)%7,int(identity)%7
                np.testing.assert_array_equal(factors[row],[shape,size,x,y])
                residue=(x+2*y+size+3*shape)%7
                assert residue==0 if split=='test' else residue==1 if split=='validation' else residue>=2
                dx,dy=abs(xx-(3*x+5)),abs(yy-(3*y+5))
                expected=np.maximum(dx,dy)<=size+2 if shape else dx*dx+dy*dy<=(size+2)**2
                exact(images[row],expected.astype(np.float32)[...,None]);pixels+=784
            for prefix,seed in [('',3911+index*6),('_prior',3943+index*6)]:
                expected=gaussian((4,count,4),seed)
                np.testing.assert_allclose(noise[split+prefix+'_standard'],expected,atol=1e-14,rtol=1e-14)
                exact(noise[split+prefix+'_epsilon'],expected.astype(np.float32));gaussian_values+=expected.size
        assert len(ids_seen)==294
        expected=gaussian((1,20,4),3967)
        np.testing.assert_allclose(noise['generation_standard'],expected,atol=1e-14,rtol=1e-14)
        exact(noise['generation_epsilon'],expected.astype(np.float32));gaussian_values+=expected.size
        for epoch in range(1,4):
            bank=arrays(root/f'train_noise_epoch{epoch}.npz')
            for prefix,seed in [('',4001+epoch),('prior_',4013+epoch)]:
                expected=gaussian((1,210,4),seed)
                np.testing.assert_allclose(bank[prefix+'standard'],expected,atol=1e-14,rtol=1e-14)
                exact(bank[prefix+'epsilon'],expected.astype(np.float32));gaussian_values+=expected.size
        initial,batch,actual,updated=[arrays(root/(name+'.npz')) for name in ['initial_arrays','first_batch','first_gradients','first_update_arrays']]
        p={name:v.astype(np.float64) for name,v in initial.items()}
        expected=derivatives(p,batch['images'],batch['epsilon'],batch['prior'],alpha)
        assert set(expected)==set(initial)==set(actual) and len(actual)==10
        gradient_error=0
        for name,v in expected.items():
            np.testing.assert_allclose(actual[name],v,atol=2e-6,rtol=5e-4,err_msg=variant+':'+name)
            gradient_error=max(gradient_error,float(np.max(abs(v-actual[name]))));gradient_count+=v.size
        first_update(initial,actual,updated)
        out,_=forward(p,batch['images'],batch['epsilon'],batch['prior'],alpha)
        saved=arrays(root/'first_forward.npz');assert len(saved)==16
        for name,v in out.items():
            np.testing.assert_allclose(saved[name],v,atol=3e-4 if name in ['nll','loss','negative_elbo'] else 3e-6,rtol=1e-5,err_msg=name)
            if name in ['mmd_xx','mmd_yy','mmd_xy']:kernel_count+=v.size
        pixels+=out['probability'].size
        selected,best=[arrays(root/(name+'.npz')) for name in ['model_arrays','best']]
        for name,v in selected.items():exact(v,best[name])
        p={name:v.astype(np.float64) for name,v in selected.items()};saved=arrays(root/'outputs.npz')
        probability_error=kernel_error=0
        assert len(saved)==3*5*16
        for split,count in [('fit',210),('validation',42),('test',42)]:
            for draw in range(-1,4):
                prefix=split+('_zero_' if draw<0 else f'_draw{draw}_')
                epsilon=np.zeros((count,4)) if draw<0 else noise[split+'_epsilon'][draw]
                prior=noise[split+'_prior_epsilon'][max(draw,0)]
                out,_=forward(p,data[split+'_images'],epsilon,prior,alpha)
                for name,v in out.items():
                    np.testing.assert_allclose(saved[prefix+name],v,atol=4e-4 if name in ['nll','loss','negative_elbo'] else 5e-6,rtol=2e-5,err_msg=variant+':'+prefix+name)
                    if name in ['mmd_xx','mmd_yy','mmd_xy']:
                        kernel_count+=v.size;kernel_error=max(kernel_error,float(np.max(abs(saved[prefix+name]-v))))
                probability_error=max(probability_error,float(np.max(abs(saved[prefix+'probability']-out['probability']))));pixels+=out['probability'].size
        prior=arrays(root/'prior.npz');exact(prior['z'],noise['generation_epsilon'][0])
        probability=decode(p,prior['z'].astype(np.float64))[1]
        np.testing.assert_allclose(prior['probability'],probability,atol=2e-6,rtol=1e-5);pixels+=probability.size
        history=np.genfromtxt(root/'history.tsv',names=True,delimiter='\t');criteria=[]
        for row in history:
            epoch=int(row[0]);parameters={name:v.astype(np.float64) for name,v in arrays(root/f'epoch{epoch}_arrays.npz').items()}
            values=[]
            for draw in range(4):
                out,_=forward(parameters,data['validation_images'],noise['validation_epsilon'][draw],noise['validation_prior_epsilon'][draw],alpha)
                values.append([out['nll'].mean(),out['kl'].mean(),out['mmd'],out['loss'],out['negative_elbo']])
            score=np.mean(values,axis=0)
            np.testing.assert_allclose(score,list(row)[1:],atol=3e-4,rtol=2e-5)
            criteria.append(score[3]);checkpoints+=1
        meta=json.loads((root/'metrics.json').read_text());chosen=int(np.argmin(criteria))+1
        assert chosen==meta['selected_epoch']==3
        for name,v in arrays(root/f'epoch{chosen}_arrays.npz').items():exact(v,selected[name])
        batches=np.genfromtxt(root/'batches.tsv',names=True,delimiter='\t',dtype=np.int64)
        assert len(batches)==630 and len(set(batches['step']))==21
        for epoch in [1,2,3]:
            rows=batches[batches['epoch']==epoch];assert sorted(rows['row'])==list(range(210))
            np.testing.assert_array_equal(rows['source_id'],data['fit_source_ids'][rows['row']])
        rows=batches[batches['step']==0]['row']
        exact(batch['images'],data['fit_images'][rows]);bank=arrays(root/'train_noise_epoch1.npz')
        exact(batch['epsilon'],bank['epsilon'][0,rows]);exact(batch['prior'],bank['prior_epsilon'][0,rows])
        evidence.append(dict(model=variant,first_gradient_max_error=gradient_error,first_Adam_update_bit_exact=True,
                             posterior_probability_max_error=probability_error,kernel_max_error=kernel_error,selected_epoch=chosen))
    a,b=[arrays(directory/name/'initial_arrays.npz') for name in ['info_vae','mmd_vae']]
    for name,v in a.items():exact(v,b[name])
    for path in (directory/'info_vae').glob('*noise*.npz'):
        a,b=arrays(path),arrays(directory/'mmd_vae'/path.name)
        for name,v in a.items():exact(v,b[name])
    return dict(first_parameter_derivatives_checked=gradient_count,Gaussian_values_checked=gaussian_values,
                lattice_and_prediction_pixels_checked=pixels,kernel_entries_checked=kernel_count,
                own_objective_validation_checkpoints_checked=checkpoints,models=evidence)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('--json',type=Path)
    args=parser.parse_args();result=check(args.directory);text=json.dumps(result,indent=2)+'\n'
    if args.json:args.json.write_text(text)
    print(text,end='')


if __name__=='__main__':main()
