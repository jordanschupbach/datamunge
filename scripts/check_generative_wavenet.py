#!/usr/bin/env python3
"""Independent NumPy/PCM audit of the pinned generative WaveNet experiment.

Reads original arrays, source WAV bytes and captured optimizer writes. Recomputes
all sampled next-code heads from their RF33 contexts rather than implementing
the source circular cache. No fitting, replacement preparation or native math.
"""
import argparse
import hashlib
import json
import wave
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays,exact
from check_contractive_autoencoder import sigmoid


def conv(parameters,name,x,dilation=1):
    w=parameters[name+'/W'][0];b=parameters[name+'/b'];n,length,_=x.shape
    pad=(len(w)-1)*dilation;padded=np.pad(x,((0,0),(pad,0),(0,0)))
    result=np.zeros((n,length,len(b)))
    for tap in range(len(w)):result+=np.einsum('ntc,co->nto',padded[:,tap*dilation:tap*dilation+length],w[tap])
    result+=b
    def backward(dy,partials):
        dx=np.zeros_like(padded);dw=np.zeros_like(w)
        for tap in range(len(w)):
            window=padded[:,tap*dilation:tap*dilation+length]
            dw[tap]=np.einsum('ntc,nto->co',window,dy)
            dx[:,tap*dilation:tap*dilation+length]+=np.einsum('nto,co->ntc',dy,w[tap])
        partials[name+'/W']=dw[None];partials[name+'/b']=dy.sum(axis=(0,1))
        return dx[:,pad:]
    return result,backward


def network(parameters,encoded):
    x,bi=conv(parameters,'input',encoded);blocks=[];skips=[]
    for i in range(5):
        name=f'block{i+1}';joined,bg=conv(parameters,name+'/gate',x,1<<i)
        a,b=np.tanh(joined[:,:,:4]),sigmoid(joined[:,:,4:]);gate=a*b
        contribution,bs=conv(parameters,name+'/skip',gate);skips.append(contribution)
        br=None
        if i<4:
            residual,br=conv(parameters,name+'/residual',gate);x=x+residual
        blocks.append((a,b,bg,bs,br))
    skip=skips[0]
    for value in skips[1:]:skip=skip+value
    h,bp=conv(parameters,'post',np.maximum(skip,0));logits,bo=conv(parameters,'output',np.maximum(h,0))
    def backward(dlogits):
        partials={};dskip=bp(bo(dlogits,partials)*(h>0),partials)*(skip>0);dx=np.zeros_like(x)
        for a,b,bg,bs,br in reversed(blocks):
            dg=bs(dskip,partials)
            if br is not None:dg+=br(dx,partials)
            joined=np.concatenate([dg*b*(1-a*a),dg*a*b*(1-b)],axis=2)
            dx=bg(joined,partials)+(dx if br is not None else 0)
        return partials,bi(dx,partials)
    return logits,backward


def forward(parameters,codes,target):
    logits,backward=network(parameters,np.eye(256)[codes]);z=logits[:,32:]
    maximum=z.max(axis=2,keepdims=True);mass=np.exp(z-maximum);prob=mass/mass.sum(axis=2,keepdims=True)
    logp=z-maximum-np.log(mass.sum(axis=2,keepdims=True));n,length=target[:,32:].shape
    loss=-logp[np.arange(n)[:,None],np.arange(length),target[:,32:]].mean()
    dy=np.zeros_like(logits);d=(prob-np.eye(256)[target[:,32:]])/(n*length);dy[:,32:]=d
    return logits,loss,backward(dy)[0]


def encode(samples):
    v=np.clip(samples,-1,1);y=np.sign(v)*np.log1p(255*np.abs(v))/np.log(256.)
    return np.clip(np.floor((y+1)*127.5+.5),0,255).astype(np.int32)


def decode(codes):
    y=codes.astype(float)*(2./255)-1
    return (np.sign(y)*np.expm1(np.abs(y)*np.log(256.))/255).astype(np.float32)


def check(root):
    pins=json.loads((root/'baseline.json').read_text());source=root/'original-output'
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for file,digest in pins['source_pins'].items():assert sha(root/('original-'+Path(file).name))==digest
    for file,digest in pins['prepared_input_sha256'].items():assert sha(file)==digest
    for group,folder in [('archive_sha256',source),('phase_probe_sha256',root)]:
        for file,digest in pins[group].items():assert sha(folder/file)==digest
    data=arrays(source/'data.npz');manifest=json.loads((Path(pins['data_directory'])/'manifest.json').read_text())
    counts=dict(original_PCM_samples=0,input_target_values=0,prompt_values=0,forward_values=0,
                gradient_values=0,optimizer_writes=0,exact_weight_values=0,finite_differences=0,
                causal_jacobian_entries=0,sampled_codes=0,independent_RF_heads=0,WAV_samples=0,batch_rows=0)
    errors=dict(forward=0.,gradient=0.,sampling_head=0.)
    clips={};record_by_id={}
    for record in manifest['records']:
        path=Path(pins['data_directory'])/record['path'];assert sha(path)==record['sha256']
        with wave.open(str(path)) as w:
            assert (w.getframerate(),w.getnchannels(),w.getsampwidth())==(8000,1,2)
            raw=np.frombuffer(w.readframes(1025),dtype='<i2').astype(float)/32768
        assert len(raw)==1025;counts['original_PCM_samples']+=len(raw)
        codes=encode(raw);clips[record['source_id']]=codes;record_by_id[record['source_id']]=record
        split=record['split']
        for row in np.flatnonzero(data[split+'_source_id']==record['source_id']):
            start=int(data[split+'_start'][row]);assert start in (0,256,512,768)
            exact(codes[start:start+256],data[split+'_input'][row]);exact(codes[start+1:start+257],data[split+'_target'][row])
            assert data[split+'_speaker_id'][row]==record['speaker_id'];counts['input_target_values']+=512
    speakers=[];sources=[]
    for split,n in [('fit',72),('validation',12),('test',24)]:
        assert data[split+'_input'].shape==(n,256) and data[split+'_target'].shape==(n,256)
        speakers.append(set(map(int,data[split+'_speaker_id'])));sources.append(set(map(int,data[split+'_source_id'])))
    for a in range(3):
        for b in range(a):assert speakers[a].isdisjoint(speakers[b]) and sources[a].isdisjoint(sources[b])
    for row,source_id in enumerate(data['prompt_source_ids']):
        exact(data['prompts'][row],clips[int(source_id)][:384]);assert int(source_id) in sources[2];counts['prompt_values']+=384
    exact(encode(decode(np.arange(256,dtype=np.int32)).astype(float)),np.arange(256,dtype=np.int32))
    def compare(actual,expected,label,gradient=False):
        assert set(actual)==set(expected),(label,set(actual)^set(expected));kind='gradient' if gradient else 'forward'
        for key,value in expected.items():
            np.testing.assert_allclose(actual[key],value,atol=2e-7 if gradient else 5e-6,rtol=2e-3 if gradient else 5e-4,err_msg=label+'/'+key)
            counts[kind+'_values']+=value.size;errors[kind]=max(errors[kind],float(abs(actual[key]-value).max()))
    p0,pf=arrays(source/'initial_arrays.npz'),arrays(source/'final.npz')
    assert len(p0)==34 and sum(v.size for v in p0.values())==5068
    batch=np.genfromtxt(source/'batches.tsv',names=True,delimiter='\t',dtype=np.int64);assert len(batch)==216;counts['batch_rows']=len(batch)
    for epoch in [1,2,3]:exact(np.sort(batch[batch['epoch']==epoch]['row']),np.arange(72))
    exact(batch['source_id'],data['fit_source_id'][batch['row']].astype(np.int64))
    first=arrays(source/'first_batch.npz');rows=batch[batch['step']==0]['row']
    exact(first['input'],data['fit_input'][rows]);exact(first['target'],data['fit_target'][rows])
    for name,p,x,y in [('first_forward',p0,first['input'],first['target']),('initial_outputs',p0,data['test_input'],data['test_target']),('outputs',pf,data['test_input'],data['test_target'])]:
        logits,_,_=forward({k:v.astype(float) for k,v in p.items()},x,y);compare(arrays(source/(name+'.npz')),{'logits':logits},name)
    before=p0;moment={k:np.zeros_like(v) for k,v in p0.items()};variance={k:np.zeros_like(v) for k,v in p0.items()}
    for step in range(27):
        rows=batch[batch['step']==step]['row'];assert len(rows)==8
        _,_,partials=forward({k:v.astype(float) for k,v in before.items()},data['fit_input'][rows],data['fit_target'][rows])
        gradients=arrays(root/'phases'/f'batch{step}-gradient.npz');compare(gradients,partials,f'batch{step}',True)
        after=arrays(root/'phases'/f'batch{step}-after.npz')
        for key,value in before.items():
            g=gradients[key];moment[key]=moment[key]*np.float32(.9)+g*np.float32(.1)
            variance[key]=variance[key]*np.float32(.999)+(g*g)*np.float32(.001)
            m=moment[key]/np.float32(1-.9**(step+1));v=variance[key]/np.float32(1-.999**(step+1))
            exact(value-m/(np.sqrt(v)+np.float32(1e-8))*np.float32(.001),after[key]);counts['exact_weight_values']+=value.size
        exact(arrays(root/'phases'/f'batch{step}-iterations.npz')['iterations'],np.array([step+1],np.int64))
        if step==0:
            compare(arrays(source/'initial_gradients.npz'),partials,'initial',True)
            for k,v in after.items():exact(v,arrays(source/'first_after.npz')[k])
        if step%9==8:
            for k,v in after.items():exact(v,arrays(source/f'epoch{step//9+1}_arrays.npz')[k])
        before=after;counts['optimizer_writes']+=1
    for k,v in before.items():exact(v,pf[k]);exact(v,arrays(source/'model_arrays.npz')[k])
    final={k:v.astype(float) for k,v in pf.items()};_,_,partials=forward(final,first['input'],first['target'])
    compare(arrays(source/'final_gradients.npz'),partials,'final',True)
    history=np.loadtxt(source/'history.tsv',skiprows=1)
    for epoch in [1,2,3]:
        p={k:v.astype(float) for k,v in arrays(source/f'epoch{epoch}_arrays.npz').items()}
        logits,nll,_=forward(p,data['validation_input'],data['validation_target'])
        compare(arrays(source/f'epoch{epoch}_validation.npz'),{'logits':logits},f'epoch{epoch}')
        np.testing.assert_allclose(history[epoch-1,1:],[nll,nll/np.log(2)],atol=2e-7,rtol=0)
    causal=arrays(source/'causality.npz');logits,back=network(final,causal['encoded_input'].astype(float))
    compare({'logits':causal['logits']},{'logits':logits},'causal')
    dy=np.zeros_like(logits);dy[0,48,0]=1;dx=back(dy)[1]
    compare({'gradient':causal['gradient']},{'gradient':dx},'causal',True);counts['causal_jacobian_entries']=dx.size
    assert np.all(dx[:,:16]==0) and np.all(dx[:,49:]==0) and abs(dx[:,16:49]).sum()>1e-6
    control=arrays(source/'future_control.npz');exact(control['input'][:,:128],data['test_input'][:,:128])
    assert np.all(control['input'][:,128:]==(data['test_input'][:,128:]+71)%256)
    changed=network(final,np.eye(256)[control['input']])[0];compare({'logits':control['logits']},{'logits':changed},'future-control')
    np.testing.assert_array_equal(control['logits'][:,:128],arrays(source/'outputs.npz')['logits'][:,:128])
    # Central differences use fitted weights: initial zero post biases can place
    # all-zero ReLU inputs exactly on a nondifferentiable boundary. Source
    # initial VJPs are checked above using the explicit derivative-zero rule.
    # Use one active coordinate per parameter leaf at the smooth fitted point.
    initial=final;_,_,partials=forward(initial,first['input'],first['target'])
    for key,value in initial.items():
        index=np.unravel_index(np.argmax(abs(partials[key])),value.shape);hi={k:v.copy() for k,v in initial.items()};lo={k:v.copy() for k,v in initial.items()}
        hi[key][index]+=1e-6;lo[key][index]-=1e-6
        fd=(forward(hi,first['input'],first['target'])[1]-forward(lo,first['input'],first['target'])[1])/2e-6
        np.testing.assert_allclose(fd,partials[key][index],atol=2e-8,rtol=3e-3,err_msg=key);counts['finite_differences']+=1
    minimum_margin=1.
    for name,n,count,warm,seed in [('unconditional',2,2048,1,8729),('continuation',3,512,128,8731)]:
        saved=arrays(source/(name+'.npz'));bank=(np.random.RandomState(seed).randint(0,2**32,(n,count),dtype=np.uint32).astype(float)+.5)/2**32
        exact(bank,saved['uniform']);assert saved['codes'].shape==(n,warm+count)
        if warm==1:assert np.all(saved['codes'][:,:1]==128)
        else:exact(saved['codes'][:,:128],data['prompts'][:,:128]);exact(saved['source_ids'],data['prompt_source_ids'])
        grouped={}
        for row in range(n):
            for t in range(count):
                end=warm+t;length=min(33,end);grouped.setdefault(length,[]).append((row,t,saved['codes'][row,end-length:end]))
        for length,records in grouped.items():
            for start in range(0,len(records),128):
                block=records[start:start+128];contexts=np.stack([r[2] for r in block]);heads=network(final,np.eye(256)[contexts])[0][:,-1]
                actual=np.stack([saved['logits'][r,t] for r,t,_ in block]);np.testing.assert_allclose(actual,heads,atol=2e-6,rtol=1e-3)
                errors['sampling_head']=max(errors['sampling_head'],float(abs(actual-heads).max()))
                weights=np.exp(heads-heads.max(axis=1,keepdims=True));weights/=weights.sum(axis=1,keepdims=True);cdf=np.cumsum(weights,axis=1)
                u=np.array([bank[r,t] for r,t,_ in block]);chosen=np.minimum((cdf<=u[:,None]).sum(axis=1),255).astype(np.int32)
                exact(chosen,np.array([saved['codes'][r,warm+t] for r,t,_ in block],dtype=np.int32))
                minimum_margin=min(minimum_margin,float(abs(cdf-u[:,None]).min()));counts['sampled_codes']+=len(block);counts['independent_RF_heads']+=len(block)
        for row in range(n):
            with wave.open(str(source/f'{name}_{row}.wav')) as w:
                assert (w.getframerate(),w.getnchannels(),w.getsampwidth())==(8000,1,2);actual=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2')
            audio=decode(saved['codes'][row]).astype(float);pcm=np.clip(np.sign(audio)*np.floor(np.abs(audio)*32768+.5),-32768,32767).astype('<i2')
            exact(actual,pcm);counts['WAV_samples']+=len(pcm)
    for row in range(3):
        with wave.open(str(source/f'reference_{row}.wav')) as w:
            assert (w.getframerate(),w.getnchannels(),w.getsampwidth())==(8000,1,2)
            actual=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2')
        audio=decode(data['prompts'][row]).astype(float)
        pcm=np.clip(np.sign(audio)*np.floor(np.abs(audio)*32768+.5),-32768,32767).astype('<i2')
        exact(actual,pcm);counts['WAV_samples']+=len(pcm)
    baseline=arrays(source/'baseline.npz');unigram=np.ones(256);transition=np.ones((256,256))
    x,y=data['fit_input'][:,32:],data['fit_target'][:,32:];unigram+=np.bincount(y.ravel(),minlength=256)
    np.add.at(transition,(x.ravel(),y.ravel()),1);unigram/=unigram.sum();transition/=transition.sum(axis=1,keepdims=True)
    exact(unigram,baseline['unigram_probability']);exact(transition,baseline['transition_probability'])
    metrics=json.loads((source/'metrics.json').read_text());tx,ty=data['test_input'][:,32:],data['test_target'][:,32:]
    np.testing.assert_allclose([-np.log(unigram[ty]).mean(),-np.log(transition[tx,ty]).mean()],[metrics['fit_unigram_nll'],metrics['fit_transition_nll']],atol=1e-12,rtol=0)
    z=arrays(source/'outputs.npz')['logits'][:,32:].astype(float)
    maximum=z.max(axis=2,keepdims=True);logp=z-maximum-np.log(np.exp(z-maximum).sum(axis=2,keepdims=True))
    nll=-logp[np.arange(24)[:,None],np.arange(224),ty].mean()
    np.testing.assert_allclose([nll,nll/np.log(2.)],[metrics['test_nll'],metrics['test_bits_per_mu_law_sample']],atol=1e-12,rtol=0)
    assert metrics['cached_native_probes']==369 and metrics['actual_sampled_codes']==5632
    assert metrics['test_nll']>metrics['fit_unigram_nll']>metrics['fit_transition_nll']
    return dict(passed=True,counts=counts,maximum_errors=errors,all_27_Adam_writes_bit_exact=True,
        original_PCM_mu_law_windows_and_prompt_codes_exact=True,speaker_and_recording_pools_disjoint=True,
        all_5632_samples_from_independent_RF33_heads_exact=True,all_case_major_midpoint_MT_uniforms_exact=True,
        all_future_and_outside_RF_partials_exact_zero=True,all_original_draw_and_reference_WAV_PCM16_samples_exact=True,
        all_34_leaf_finite_differences_at_fitted_smooth_point=True,source_test_host_Double_NLL_exact=True,
        fit_only_start32_Laplace_unigram_and_transition_baselines_exact=True,source_result_worse_than_both_fitted_baselines_preserved=True,
        minimum_sampling_CDF_margin=minimum_margin,test_bits_per_mu_law_sample=metrics['test_bits_per_mu_law_sample'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('build/nn-report-conversion/wavenet'));p.add_argument('--json',type=Path)
    a=p.parse_args();result=check(a.root);rendered=json.dumps(result,indent=2)+'\n';print(rendered,end='')
    if a.json:a.json.write_text(rendered)


if __name__=='__main__':main()
