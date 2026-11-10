#!/usr/bin/env python3
"""Optional independent NumPy oracle for sentence VAE and posterior-collapse artifacts.

Never trains or prepares data. Checks all 3243 first BPTT derivatives/Adam, all
18 saved endpoints, explicit noise, prefix/generation controls, every eligible
validation score and signed empirical-mixture diagnostics.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact, gaussian
from check_factor_tc_variational_autoencoder import first_update


def gru(p, name, x, h):
    """Reset-before GRU with separate update/reset and candidate projections."""
    z, r = np.split(1/(1+np.exp(-(x@p[name+'/Wx_zr']+h@p[name+'/Wh_zr']+p[name+'/b_zr']))),2,axis=1)
    candidate=np.tanh(x@p[name+'/Wx_n']+(r*h)@p[name+'/Wh_n']+p[name+'/b_n'])
    return (1-z)*candidate+z*h,(x,h,z,r,candidate)


def gru_back(p, name, cache, delta, result):
    """All six GRU leaf adjoints plus input and previous-state adjoints."""
    x,h,z,r,c=cache
    da=delta*(1-z)*(1-c*c)
    drh=da@p[name+'/Wh_n'].T
    dz=delta*(h-c)*z*(1-z)
    dr=drh*h*r*(1-r)
    dg=np.concatenate([dz,dr],axis=1)
    for key,value in [('Wx_n',x.T@da),('Wh_n',(r*h).T@da),('b_n',da.sum(axis=0)),
                      ('Wx_zr',x.T@dg),('Wh_zr',h.T@dg),('b_zr',dg.sum(axis=0))]:
        result[name+'/'+key]+=value
    return da@p[name+'/Wx_n'].T+dg@p[name+'/Wx_zr'].T,delta*z+drh*r+dg@p[name+'/Wh_zr'].T


def encode(p,tokens):
    """Independent PAD-masked encoder with direct embedding lookup."""
    h=np.zeros((len(tokens),16));history=[];cache=[]
    for t in range(tokens.shape[1]):
        proposed,c=gru(p,'encoder_gru',p['encoder_embedding'][tokens[:,t]],h)
        mask=(tokens[:,t]!=0)[:,None]
        h=np.where(mask,proposed,h);history.append(h);cache.append((c,mask))
    return h,np.stack(history,axis=1),cache


def teacher(p,z,inputs):
    """Target-free causal prefix decoder with stable token probabilities."""
    initial=np.tanh(z@p['decoder_initial/W']+p['decoder_initial/b'])
    h=initial;history=[];logits=[];probability=[];cache=[]
    for t in range(7):
        h,c=gru(p,'decoder_gru',p['decoder_embedding'][inputs[:,t]],h)
        score=h@p['token_output/W']+p['token_output/b']
        exp=np.exp(score-score.max(axis=1,keepdims=True))
        history.append(h);logits.append(score);probability.append(exp/exp.sum(axis=1,keepdims=True));cache.append(c)
    return dict(decoder_initial=initial,decoder_states=np.stack(history,axis=1),
                logits=np.stack(logits,axis=1),probability=np.stack(probability,axis=1)),cache


def forward(p,tokens,epsilon):
    """Complete eighteen-field reference without using native saved intermediate fields."""
    state,states,encoder_cache=encode(p,tokens)
    mu=state@p['mu/W']+p['mu/b'];lv=state@p['log_variance/W']+p['log_variance/b']
    z=mu+np.exp(.5*lv)*epsilon
    inputs=np.concatenate([np.ones((len(tokens),1),dtype=np.int32),tokens[:,:-1]],axis=1)
    out,decoder_cache=teacher(p,z,inputs)
    logits=out['logits'];shift=logits-logits.max(axis=2,keepdims=True)
    lp=shift-np.log(np.exp(shift).sum(axis=2,keepdims=True))
    mask=(tokens!=0).astype(np.float64)
    nll=-np.take_along_axis(lp,tokens[...,None],axis=2)[...,0]*mask
    predicted=np.argmax(logits,axis=2).astype(np.int32)
    kl=.5*(mu*mu+np.exp(lv)-1-lv).sum(axis=1)
    out.update(decoder_inputs=inputs,encoder_state=state,encoder_states=states,mu=mu,
               log_variance=lv,z=z,predicted_ids=predicted,mask=mask,token_nll=nll,nll=nll.sum(axis=1),
               kl=kl,correct=((predicted==tokens)*mask).sum(axis=1),valid_tokens=mask.sum(axis=1),
               loss=(nll.sum(axis=1)+kl).mean())
    return out,(encoder_cache,decoder_cache)


def derivatives(p,tokens,epsilon,beta):
    """Independent reverse-mode chain for every leaf, including full encoder/decoder BPTT."""
    out,(enc,dec)=forward(p,tokens,epsilon);b=len(tokens)
    result={key:np.zeros_like(value) for key,value in p.items()}
    dh=np.zeros((b,16));target=np.eye(19)[tokens]
    for t in reversed(range(7)):
        score_grad=(out['probability'][:,t]-target[:,t])*out['mask'][:,t,None]/b
        result['token_output/W']+=out['decoder_states'][:,t].T@score_grad
        result['token_output/b']+=score_grad.sum(axis=0)
        dh+=score_grad@p['token_output/W'].T
        dx,dh=gru_back(p,'decoder_gru',dec[t],dh,result)
        np.add.at(result['decoder_embedding'],out['decoder_inputs'][:,t],dx)
    initial_grad=dh*(1-out['decoder_initial']**2)
    result['decoder_initial/W']=out['z'].T@initial_grad
    result['decoder_initial/b']=initial_grad.sum(axis=0)
    dz=initial_grad@p['decoder_initial/W'].T
    dm=dz+beta*out['mu']/b
    dl=dz*epsilon*.5*np.exp(.5*out['log_variance'])+beta*.5*(np.exp(out['log_variance'])-1)/b
    for name,delta in [('mu',dm),('log_variance',dl)]:
        result[name+'/W']=out['encoder_state'].T@delta;result[name+'/b']=delta.sum(axis=0)
    dh=dm@p['mu/W'].T+dl@p['log_variance/W'].T
    for t in reversed(range(7)):
        cache,mask=enc[t];dx,previous=gru_back(p,'encoder_gru',cache,dh*mask,result)
        np.add.at(result['encoder_embedding'],tokens[:,t],dx)
        dh=previous+dh*(1-mask)
    return result


def generate(p,z,uniforms=None):
    """Independent explicit-prefix greedy or inverse-CDF generation with PAD after EOS."""
    h=np.tanh(z@p['decoder_initial/W']+p['decoder_initial/b'])
    previous=np.ones(len(z),dtype=int);ended=np.zeros(len(z),dtype=bool)
    ids=[];logits=[];probabilities=[]
    for t in range(7):
        h,_=gru(p,'decoder_gru',p['decoder_embedding'][previous],h)
        score=h@p['token_output/W']+p['token_output/b']
        exp=np.exp(score-score.max(axis=1,keepdims=True));prob=exp/exp.sum(axis=1,keepdims=True)
        chosen=np.argmax(score,axis=1) if uniforms is None else np.minimum(18,(np.cumsum(prob,axis=1)<=uniforms[:,t,None]).sum(axis=1))
        emitted=np.where(ended,0,chosen);ended|=emitted==2;previous=emitted
        ids.append(emitted);logits.append(score);probabilities.append(prob)
    return dict(ids=np.stack(ids,axis=1).astype(np.int32),logits=np.stack(logits,axis=1),
                probability=np.stack(probabilities,axis=1),ended=ended.astype(np.int32))


def diagnostics(mu,lv,samples):
    """Independent vectorized all-case mixture densities and signed sample decomposition."""
    unit=.5*(mu*mu+np.exp(lv)-1-lv).mean(axis=0);variance=mu.var(axis=0)
    conditional=information=aggregate=0.
    for z in samples:
        diff=z[:,None,:]-mu[None,:,:]
        components=-.5*(diff*diff*np.exp(-lv)[None,:,:]+lv[None,:,:]+np.log(2*np.pi)).sum(axis=2)
        peak=components.max(axis=1)
        mixture=peak+np.log(np.exp(components-peak[:,None]).mean(axis=1))
        prior=-.5*(z*z+np.log(2*np.pi)).sum(axis=1);own=np.diag(components)
        conditional+=(own-prior).mean()/len(samples)
        information+=(own-mixture).mean()/len(samples)
        aggregate+=(mixture-prior).mean()/len(samples)
    return dict(analytic_mean_kl=unit.sum(),unit_kl=unit,mu_variance=variance,
                active_units_variance_threshold_0_01=int((variance>.01).sum()),
                sampled_conditional_kl=conditional,empirical_mixture_information=information,empirical_aggregate_kl=aggregate)


def generation_counts(ids):
    """Count EOS, grammar-valid prefixes and distinct texts, checking PAD suffixes."""
    ended=valid=0;distinct=set()
    roles=[range(3,5),range(5,8),range(8,11),range(11,13),range(13,17)]
    for row in ids.tolist():
        if 2 in row:
            stop=row.index(2);ended+=1
            assert all(token==0 for token in row[stop+1:])
            prefix=row[:stop]
            valid+=int(len(prefix) in (5,6) and
                       all(token in role for token,role in zip(prefix,roles)) and
                       (len(prefix)==5 or prefix[5] in (17,18)))
        else:prefix=row
        distinct.add(tuple(prefix))
    return ended,valid,len(distinct)


def check_data(data):
    """Regenerate every sentence from an independently enumerated grammar dictionary."""
    seen=set();grammar={}
    for identity,f in enumerate(itertools.product(range(2),range(3),range(3),range(2),range(4),range(3))):
        sentence=[3+f[0],5+f[1],8+f[2],11+f[3],13+f[4]]+([16+f[5]] if f[5] else [])+[2]
        residue=2*((f[1]+f[2]+f[5])%3)+(f[0]+f[3]+f[4])%2
        grammar[identity]=(f,sentence,'test' if residue==0 else 'validation' if residue==1 else 'fit')
    for split,count in [('fit',288),('validation',72),('test',72)]:
        tokens=data[split+'_tokens'];assert tokens.shape==(count,7) and tokens.dtype==np.int32
        for row,identity in enumerate(data[split+'_source_ids']):
            assert int(identity) not in seen;seen.add(int(identity))
            factors,sentence,expected_split=grammar[int(identity)];assert split==expected_split
            np.testing.assert_array_equal(data[split+'_factors'][row],factors)
            np.testing.assert_array_equal(tokens[row],sentence+[0]*(7-len(sentence)))
            assert data[split+'_lengths'][row]==len(sentence)
            np.testing.assert_array_equal(data[split+'_decoder_inputs'][row],[1]+tokens[row,:-1].tolist())
    assert len(seen)==432


def check(directory):
    """Check one complete standalone or annealed sentence-model run and report coverage."""
    data,noise=arrays(directory/'data.npz'),arrays(directory/'fixed_noise.npz');check_data(data)
    gaussian_values=0
    for split,count,seed in [('fit',288,3137),('validation',72,3163),('test',72,3181)]:
        expected=gaussian((4,count,4),seed)
        np.testing.assert_allclose(noise[split+'_standard'],expected,atol=1e-14,rtol=1e-14)
        exact(noise[split+'_epsilon'],expected.astype(np.float32));gaussian_values+=expected.size
    for epoch in range(1,4):
        bank=arrays(directory/f'train_noise_epoch{epoch}.npz');expected=gaussian((1,288,4),3191+epoch)
        np.testing.assert_allclose(bank['standard'],expected,atol=1e-14,rtol=1e-14)
        exact(bank['epsilon'],expected.astype(np.float32));gaussian_values+=expected.size
    if 'prior_standard' in noise:
        expected=gaussian((1,20,4),3203)
        np.testing.assert_allclose(noise['prior_standard'],expected,atol=1e-14,rtol=1e-14)
        exact(noise['prior_epsilon'],expected.astype(np.float32));gaussian_values+=expected.size
    initial,batch,actual,updated=[arrays(directory/(name+'.npz')) for name in ['initial_arrays','first_batch','first_gradients','first_update_arrays']]
    p={k:v.astype(np.float64) for k,v in initial.items()}
    expected=derivatives(p,batch['tokens'],batch['epsilon'].astype(np.float64),float(batch.get('beta',1)))
    assert set(actual)==set(expected)==set(initial)
    gradient_count=sum(v.size for v in actual.values());assert gradient_count==3243
    gradient_error=0
    for key in actual:
        np.testing.assert_allclose(actual[key],expected[key],atol=2e-7,rtol=1e-3)
        gradient_error=max(gradient_error,float(np.max(np.abs(actual[key]-expected[key]))))
    first_update(initial,actual,updated)
    first_reference=forward(p,batch['tokens'],batch['epsilon'].astype(np.float64))[0]
    first_saved=arrays(directory/'first_forward.npz')
    selected=arrays(directory/'model_arrays.npz');p={k:v.astype(np.float64) for k,v in selected.items()}
    saved=arrays(directory/'outputs.npz');fields_checked=values_checked=0;max_error=0
    def compare(actual,expected):
        nonlocal fields_checked,values_checked,max_error
        a,b=np.asarray(actual),np.asarray(expected);assert a.shape==b.shape
        if a.dtype.kind in 'iu':np.testing.assert_array_equal(a,b)
        else:
            np.testing.assert_allclose(a,b,atol=3e-5,rtol=2e-5)
            max_error=max(max_error,float(np.max(np.abs(a-b))))
        fields_checked+=1;values_checked+=a.size
    for key,value in first_reference.items():compare(first_saved[key],value)
    for split in ['fit','validation','test']:
        x=data[split+'_tokens']
        for draw in range(4):
            out,_=forward(p,x,noise[split+'_epsilon'][draw].astype(np.float64))
            for key,value in out.items():compare(saved[f'{split}_draw{draw}_'+key],value)
        if split+'_mean_mu' in saved:
            out,_=forward(p,x,np.zeros((len(x),4)))
            for key,value in out.items():compare(saved[split+'_mean_'+key],value)
            for key,value in generate(p,out['mu']).items():compare(saved[split+'_mean_greedy_'+key],value)
    history=np.genfromtxt(directory/'history.tsv',names=True,delimiter='\t')
    scores=[]
    for epoch in range(1,4):
        q={k:v.astype(np.float64) for k,v in arrays(directory/f'epoch{epoch}_arrays.npz').items()}
        draws=[forward(q,data['validation_tokens'],e.astype(np.float64))[0] for e in noise['validation_epsilon']]
        score=np.mean([o['nll'].mean()+o['kl'].mean() for o in draws]);scores.append(score)
        nll=np.mean([o['nll'].mean() for o in draws]);kl=draws[0]['kl'].mean()
        valid=sum(o['valid_tokens'].sum() for o in draws)
        ppl=np.exp(sum(o['nll'].sum() for o in draws)/valid)
        accuracy=sum(o['correct'].sum() for o in draws)/valid
        columns=[('validation_negative_elbo',score),('validation_kl',kl)]
        if 'validation_sequence_nll' in history.dtype.names:
            columns += [('validation_sequence_nll',nll),('validation_teacher_reconstruction_token_perplexity',ppl),('validation_teacher_token_accuracy',accuracy)]
        else:columns += [('validation_nll',nll),('validation_teacher_reconstruction_token_ppl',ppl)]
        for key,value in columns:np.testing.assert_allclose(value,history[key][epoch-1],atol=2e-5,rtol=1e-5)
    meta=json.loads((directory/'metrics.json').read_text());assert meta['selected_epoch']==int(np.argmin(scores))+1
    final=arrays(directory/f"epoch{meta['selected_epoch']}_arrays.npz")
    for key in selected:exact(final[key],selected[key])
    if (directory/'ablations.npz').exists():
        ab=arrays(directory/'ablations.npz')
        for name in ['posterior','zero','shuffled','prior']:
            out,_=teacher(p,ab[name+'_z'].astype(np.float64),ab['inputs'])
            compare(ab[name+'_probability'],out['probability'])
        out,_=forward(p,data['fit_tokens'],noise['fit_epsilon'][0].astype(np.float64))
        samples=[saved['fit_draw'+str(k)+'_z'].astype(np.float64) for k in range(4)]
        # Use saved represented Gaussian moments/draws to isolate the independent density diagnostic.
        expected=diagnostics(saved['fit_draw0_mu'].astype(np.float64),saved['fit_draw0_log_variance'].astype(np.float64),samples)
        actual=json.loads((directory/'fit_posterior_diagnostics.json').read_text())
        for key,value in expected.items():np.testing.assert_allclose(actual[key],value,atol=1e-12,rtol=1e-12)
        test,_=forward(p,data['test_tokens'],noise['test_epsilon'][0].astype(np.float64))
        for name,z in [('mean',test['mu']),('posterior',test['z'])]:
            for key,value in generate(p,z).items():compare(saved['test_'+name+'_greedy_'+key],value)
    else:
        prior=arrays(directory/'prior.npz')
        exact(prior['z'],noise['prior_epsilon'][0])
        random=np.random.RandomState(3217)
        uniform=(random.randint(0,2**32,size=140,dtype=np.uint32).astype(np.float64)+.5)/2**32
        exact(prior['token_uniforms'],uniform.reshape(20,7))
        test,_=forward(p,data['test_tokens'],noise['test_epsilon'][0].astype(np.float64))
        test_draws=[forward(p,data['test_tokens'],e.astype(np.float64))[0] for e in noise['test_epsilon']]
        test_nll=np.mean([out['nll'].mean() for out in test_draws]);test_kl=test_draws[0]['kl'].mean()
        valid_tokens=sum(out['valid_tokens'].sum() for out in test_draws)
        for key,value in [('test_sequence_nll',test_nll),('test_kl',test_kl),
                          ('test_negative_elbo',test_nll+test_kl),
                          ('test_teacher_reconstruction_token_perplexity',np.exp(sum(out['nll'].sum() for out in test_draws)/valid_tokens)),
                          ('test_teacher_token_accuracy',sum(out['correct'].sum() for out in test_draws)/valid_tokens)]:
            np.testing.assert_allclose(meta[key],value,atol=2e-5,rtol=1e-5)
        for key,value in generate(p,test['z']).items():compare(saved['test_posterior_greedy_'+key],value)
        baseline=arrays(directory/'unigram_baseline.npz')
        fit=data['fit_tokens'];counts=np.bincount(fit[fit!=0],minlength=19).astype(np.int32)
        probabilities=(counts+.5)/(counts.sum()+9.5)
        exact(baseline['fit_counts'],counts);exact(baseline['probabilities'],probabilities)
        target=data['test_tokens'];target=target[target!=0]
        average=-np.log(probabilities[target]).mean()
        np.testing.assert_allclose(meta['fit_only_unigram_test_sequence_nll'],average*len(target)/72,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(meta['fit_only_unigram_test_token_perplexity'],np.exp(average),atol=1e-12,rtol=1e-12)
        for name,z,u in [('greedy',prior['z'],None),('sampled',prior['z'],prior['token_uniforms']),('zero',np.zeros_like(prior['z']),None)]:
            for key,value in generate(p,z.astype(np.float64),u).items():compare(prior[name+'_'+key],value)
            if name!='zero':
                for field,value in zip(['eos_count','grammar_valid_count','unique_count'],generation_counts(prior[name+'_ids'])):
                    assert meta['prior_'+name+'_'+field]==value
        padded=arrays(directory/'padding_probe.npz');compare(padded['encoder_state8'],encode(p,padded['tokens8'])[0])
        probe=arrays(directory/'causality_probe.npz')
        for prefix,key in [('inputs','logits'),('changed_inputs','changed_logits')]:
            compare(probe[key],teacher(p,probe['z'].astype(np.float64),probe[prefix])[0]['logits'])
    return dict(first_parameter_derivatives_checked=gradient_count,
                first_gradient_max_error=gradient_error,first_Adam_update_bit_exact=True,
                Gaussian_values_checked=gaussian_values,endpoint_fields_checked=fields_checked,endpoint_values_checked=values_checked,
                maximum_endpoint_error=max_error,eligible_validation_checkpoints_checked=3,selected_epoch=meta['selected_epoch'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('--json',type=Path)
    args=parser.parse_args()
    roots=[args.directory/'constant',args.directory/'warmup'] if (args.directory/'constant').is_dir() else [args.directory]
    results={(root.name if len(roots)>1 else 'sequence'):check(root) for root in roots}
    if len(roots)==2:
        control=arrays(args.directory/'diagnostic_controls.npz')
        for name,mu,samples in [('collapsed',control['mu'],control['epsilon']),
                                ('active',control['active_mu'],np.stack([control['active_z'+str(k)] for k in range(4)]))]:
            out=diagnostics(mu.astype(np.float64),control['log_variance'].astype(np.float64),samples.astype(np.float64))
            actual=json.loads((args.directory/(name+'_control.json')).read_text())
            for key,value in out.items():np.testing.assert_allclose(actual[key],value,atol=1e-12,rtol=1e-12)
    text=json.dumps(results,indent=2)+'\n'
    if args.json:args.json.write_text(text)
    print(text,end='')


if __name__=='__main__':
    main()
