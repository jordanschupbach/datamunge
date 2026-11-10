#!/usr/bin/env python3
"""Optional independent NumPy oracle for complete precision-fusion Ladder VAE artifacts.

No fitting or data preparation. Checks all first leaf derivatives/Adam, all four
fusion partials, explicit Gaussian banks, all fifteen original saved endpoints,
noise interventions, ancestral samples and every eligible validation criterion.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact, gaussian
from check_factor_tc_variational_autoencoder import first_update


def fusion(em,el,pm,pl):
    """Independent logaddexp precision fusion and normalized source weights."""
    lv=-np.logaddexp(-el,-pl)
    a,b=np.exp(lv-el),np.exp(lv-pl)
    return a*em+b*pm,lv,a,b


def fusion_back(em,pm,qm,a,b,dm,dl):
    """Every evidence/prior partial of a scalar depending on fused moments."""
    return dm*a,dl*a+dm*a*(qm-em),dm*b,dl*b+dm*b*(qm-pm)


def conditional(mu,lv,pm,pl):
    """Independent conditional diagonal-Gaussian KL summed over lower coordinates."""
    return .5*((pl-lv)+(np.exp(lv-pl)+(mu-pm)**2*np.exp(-pl))-1).sum(axis=1)


def decode(p,z):
    """Independent lower-code-only ReLU/linear Bernoulli decoder."""
    h=np.maximum(z@p['decoder/W']+p['decoder/b'],0)
    logits=h@p['output/W']+p['output/b']
    return logits,1/(1+np.exp(-logits)),h


def prior(p,top,noise):
    """Image-free learned lower prior and explicit ancestral sample."""
    h=np.maximum(top@p['prior_hidden/W']+p['prior_hidden/b'],0)
    mu=h@p['prior_mu/W']+p['prior_mu/b'];lv=h@p['prior_lv/W']+p['prior_lv/b']
    z=mu+np.exp(.5*lv)*noise
    return dict(top_z=top,lower_noise=noise,prior_mu=mu,prior_lv=lv,lower_z=z,probability=decode(p,z)[1])


def forward(p,image,epsilon):
    """All original fifteen endpoints and full independent chain-rule cache."""
    x=image.reshape(len(image),-1).astype(np.float64);epsilon=epsilon.astype(np.float64)
    h1=np.maximum(x@p['encoder1/W']+p['encoder1/b'],0)
    h2=np.maximum(h1@p['encoder2/W']+p['encoder2/b'],0)
    em=h1@p['evidence_mu/W']+p['evidence_mu/b'];el=h1@p['evidence_lv/W']+p['evidence_lv/b']
    tm=h2@p['top_mu/W']+p['top_mu/b'];tl=h2@p['top_lv/W']+p['top_lv/b']
    top=tm+np.exp(.5*tl)*epsilon[:,4:]
    hp=np.maximum(top@p['prior_hidden/W']+p['prior_hidden/b'],0)
    pm=hp@p['prior_mu/W']+p['prior_mu/b'];pl=hp@p['prior_lv/W']+p['prior_lv/b']
    qm,ql,a,b=fusion(em,el,pm,pl)
    z=qm+np.exp(.5*ql)*epsilon[:,:4]
    logits,probability,hd=decode(p,z)
    nll=(np.logaddexp(0,logits)-x*logits).sum(axis=1)
    lower=conditional(qm,ql,pm,pl);upper=.5*(tm*tm+np.exp(tl)-1-tl).sum(axis=1)
    out=dict(evidence_mu=em,evidence_lv=el,top_mu=tm,top_lv=tl,top_z=top,
             prior_mu=pm,prior_lv=pl,lower_mu=qm,lower_lv=ql,lower_z=z,
             probability=probability,nll=nll,kl_lower=lower,kl_top=upper,loss=(nll+(lower+upper)).mean())
    return out,(x,h1,h2,hp,hd,a,b)


def derivatives(p,image,epsilon):
    """All22 leaf adjoints through decoder, fusion, learned prior and both Gaussian samples."""
    out,(x,h1,h2,hp,hd,a,b)=forward(p,image,epsilon);count=len(x)
    delta=(out['probability']-x)/count
    result={'output/W':hd.T@delta,'output/b':delta.sum(axis=0)}
    back=(delta@p['output/W'].T)*(hd>0)
    result.update({'decoder/W':out['lower_z'].T@back,'decoder/b':back.sum(axis=0)})
    dz=back@p['decoder/W'].T
    residual=out['lower_mu']-out['prior_mu'];precision=np.exp(-out['prior_lv']);ratio=np.exp(out['lower_lv']-out['prior_lv'])
    dm=dz+residual*precision/count
    dl=.5*dz*np.exp(.5*out['lower_lv'])*epsilon[:,:4]+.5*(ratio-1)/count
    dem,delv,dpm,dpl=fusion_back(out['evidence_mu'],out['prior_mu'],out['lower_mu'],a,b,dm,dl)
    dpm-=residual*precision/count
    dpl+=.5*(1-ratio-residual**2*precision)/count
    result.update({'prior_mu/W':hp.T@dpm,'prior_mu/b':dpm.sum(axis=0),
                   'prior_lv/W':hp.T@dpl,'prior_lv/b':dpl.sum(axis=0)})
    dhp=(dpm@p['prior_mu/W'].T+dpl@p['prior_lv/W'].T)*(hp>0)
    result.update({'prior_hidden/W':out['top_z'].T@dhp,'prior_hidden/b':dhp.sum(axis=0)})
    dt=dhp@p['prior_hidden/W'].T
    dtm=dt+out['top_mu']/count
    dtl=.5*dt*np.exp(.5*out['top_lv'])*epsilon[:,4:]+.5*(np.exp(out['top_lv'])-1)/count
    result.update({'top_mu/W':h2.T@dtm,'top_mu/b':dtm.sum(axis=0),
                   'top_lv/W':h2.T@dtl,'top_lv/b':dtl.sum(axis=0),
                   'evidence_mu/W':h1.T@dem,'evidence_mu/b':dem.sum(axis=0),
                   'evidence_lv/W':h1.T@delv,'evidence_lv/b':delv.sum(axis=0)})
    dh2=(dtm@p['top_mu/W'].T+dtl@p['top_lv/W'].T)*(h2>0)
    result.update({'encoder2/W':h1.T@dh2,'encoder2/b':dh2.sum(axis=0)})
    dh1=(dh2@p['encoder2/W'].T+dem@p['evidence_mu/W'].T+delv@p['evidence_lv/W'].T)*(h1>0)
    result.update({'encoder1/W':x.T@dh1,'encoder1/b':dh1.sum(axis=0)})
    return result


def check(directory):
    """Validate untouched or converted complete artifacts and return exact coverage counts."""
    data,noise=arrays(directory/'data.npz'),arrays(directory/'fixed_noise.npz')
    pixels=gaussian_values=gradient_count=0;ids_seen=set();yy,xx=np.mgrid[:28,:28]
    for split,count,seed in [('fit',210,2521),('validation',42,2531),('test',42,2539)]:
        ids,factors,images=[data[split+s] for s in ['_source_ids','_factors','_images']]
        assert len(ids)==count and not ids_seen.intersection(ids);ids_seen.update(ids)
        for row,identity in enumerate(ids):
            shape,size,x,y=int(identity)//147,(int(identity)//49)%3,(int(identity)//7)%7,int(identity)%7
            np.testing.assert_array_equal(factors[row],[shape,size,x,y]);r=(x+2*y+size+3*shape)%7
            assert r==0 if split=='test' else r==1 if split=='validation' else r>=2
            dx,dy=abs(xx-(3*x+5)),abs(yy-(3*y+5))
            expected=np.maximum(dx,dy)<=size+2 if shape else dx*dx+dy*dy<=(size+2)**2
            exact(images[row],expected.astype(np.float32)[...,None]);pixels+=784
        expected=gaussian((4,count,6),seed)
        np.testing.assert_allclose(noise[split+'_standard'],expected,atol=1e-14,rtol=1e-14)
        exact(noise[split+'_epsilon'],expected.astype(np.float32));gaussian_values+=expected.size
    assert len(ids_seen)==294
    expected=gaussian((1,20,6),2543)
    np.testing.assert_allclose(noise['prior_standard'],expected,atol=1e-14,rtol=1e-14)
    exact(noise['prior_epsilon'],expected.astype(np.float32));gaussian_values+=expected.size
    for epoch in range(1,4):
        bank=arrays(directory/f'train_noise_epoch{epoch}.npz');expected=gaussian((1,210,6),2551+epoch)
        np.testing.assert_allclose(bank['standard'],expected,atol=1e-14,rtol=1e-14)
        exact(bank['epsilon'],expected.astype(np.float32));gaussian_values+=expected.size
    algebra=arrays(directory/'fusion_algebra.npz');em,el,pm,pl=[algebra[name].astype(np.float64) for name in ['evidence_mu','evidence_lv','prior_mu','prior_lv']]
    qm,ql,a,b=fusion(em,el,pm,pl);kl=conditional(qm,ql,pm,pl)
    for name,value in [('mu',qm),('lv',ql),('kl',kl)]:np.testing.assert_allclose(algebra[name],value,atol=4e-7,rtol=2e-6)
    dm=2*qm+(qm-pm)*np.exp(-pl);dl=2*ql+.5*(np.exp(ql-pl)-1)
    du,dv,dp,dr=fusion_back(em,pm,qm,a,b,dm,dl)
    dp-=(qm-pm)*np.exp(-pl);dr+=.5*(1-np.exp(ql-pl)-(qm-pm)**2*np.exp(-pl))
    for name,value in [('du',du),('dv',dv),('dp',dp),('dr',dr)]:np.testing.assert_allclose(algebra[name],value,atol=2e-6,rtol=2e-6)
    extreme=np.array([[-80.,80.],[80.,-80.]])
    mu,lv,_,_=fusion(em,extreme,pm,-extreme)
    np.testing.assert_allclose(algebra['extreme_mu'],mu,atol=1e-7,rtol=1e-7)
    exact(algebra['extreme_lv'],lv.astype(np.float32))
    initial,batch,actual,updated=[arrays(directory/(name+'.npz')) for name in ['initial_arrays','first_batch','first_gradients','first_update_arrays']]
    p={name:v.astype(np.float64) for name,v in initial.items()};expected=derivatives(p,batch['images'],batch['epsilon'])
    assert set(expected)==set(initial)==set(actual) and len(actual)==22
    gradient_error=0
    for name,value in expected.items():
        np.testing.assert_allclose(actual[name],value,atol=2e-6,rtol=5e-4,err_msg=name)
        gradient_count+=value.size;gradient_error=max(gradient_error,float(np.max(abs(actual[name]-value))))
    first_update(initial,actual,updated)
    first,_=forward(p,batch['images'],batch['epsilon']);saved=arrays(directory/'first_forward.npz');assert len(saved)==15
    for name,value in first.items():np.testing.assert_allclose(saved[name],value,atol=5e-4 if name in ['nll','loss'] else 5e-6,rtol=2e-5,err_msg=name)
    pixels+=first['probability'].size
    selected,best=[arrays(directory/(name+'.npz')) for name in ['model_arrays','best']]
    for name,v in selected.items():exact(v,best[name])
    p={name:v.astype(np.float64) for name,v in selected.items()};saved=arrays(directory/'outputs.npz');assert len(saved)==3*5*15
    probability_error=field_error=0
    for split,count in [('fit',210),('validation',42),('test',42)]:
        for draw in range(-1,4):
            prefix=split+('_greedy_' if draw<0 else f'_draw{draw}_')
            epsilon=np.zeros((count,6)) if draw<0 else noise[split+'_epsilon'][draw]
            out,_=forward(p,data[split+'_images'],epsilon)
            for name,value in out.items():
                np.testing.assert_allclose(saved[prefix+name],value,atol=8e-4 if name in ['nll','loss'] else 8e-6,rtol=3e-5,err_msg=prefix+name)
                field_error=max(field_error,float(np.max(abs(saved[prefix+name]-value))))
            probability_error=max(probability_error,float(np.max(abs(saved[prefix+'probability']-out['probability']))));pixels+=out['probability'].size
    for group in ['lower','top']:
        epsilon=np.zeros((42,6));epsilon[:,slice(0,4) if group=='lower' else slice(4,6)]=.5
        out,_=forward(p,data['test_images'],epsilon);intervention=arrays(directory/(group+'_noise_intervention.npz'))
        for name,value in out.items():np.testing.assert_allclose(intervention[name],value,atol=8e-4 if name in ['nll','loss'] else 8e-6,rtol=3e-5,err_msg=group+':'+name)
        pixels+=out['probability'].size
    generated=arrays(directory/'prior.npz');bank=noise['prior_epsilon'][0]
    exact(generated['top_z'],bank[:,4:]);exact(generated['lower_noise'],bank[:,:4])
    out=prior(p,bank[:,4:].astype(np.float64),bank[:,:4].astype(np.float64))
    for name,value in out.items():np.testing.assert_allclose(generated[name],value,atol=8e-6,rtol=3e-5)
    pixels+=out['probability'].size
    history=np.genfromtxt(directory/'history.tsv',names=True,delimiter='\t');criteria=[]
    for row in history:
        parameters={name:v.astype(np.float64) for name,v in arrays(directory/f'epoch{int(row[0])}_arrays.npz').items()}
        values=[]
        for draw in range(4):
            o,_=forward(parameters,data['validation_images'],noise['validation_epsilon'][draw])
            values.append([o['nll'].mean(),o['kl_lower'].mean(),o['kl_top'].mean()])
        score=np.mean(values,axis=0);criterion=score.sum()
        np.testing.assert_allclose(list(score)+[criterion],list(row)[2:],atol=3e-4,rtol=2e-5);criteria.append(criterion)
    meta=json.loads((directory/'metrics.json').read_text());chosen=int(np.argmin(criteria))+1;assert chosen==meta['selected_epoch']==3
    for name,v in arrays(directory/f'epoch{chosen}_arrays.npz').items():exact(v,selected[name])
    batches=np.genfromtxt(directory/'batches.tsv',names=True,delimiter='\t',dtype=np.int64);assert len(batches)==630 and len(set(batches['step']))==21
    for epoch in [1,2,3]:
        rows=batches[batches['epoch']==epoch];assert sorted(rows['row'])==list(range(210));np.testing.assert_array_equal(rows['source_id'],data['fit_source_ids'][rows['row']])
    rows=batches[batches['step']==0]['row'];exact(batch['images'],data['fit_images'][rows]);exact(batch['epsilon'],arrays(directory/'train_noise_epoch1.npz')['epsilon'][0,rows])
    return dict(first_parameter_derivatives_checked=gradient_count,Gaussian_values_checked=gaussian_values,
                lattice_and_prediction_pixels_checked=pixels,fusion_partials_checked=16,
                eligible_validation_checkpoints_checked=3,first_gradient_max_error=gradient_error,
                first_Adam_update_bit_exact=True,posterior_probability_max_error=probability_error,
                all_fifteen_fields_max_error=field_error,selected_epoch=chosen)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('--json',type=Path)
    args=parser.parse_args();text=json.dumps(check(args.directory),indent=2)+'\n'
    if args.json:args.json.write_text(text)
    print(text,end='')


if __name__=='__main__':main()
