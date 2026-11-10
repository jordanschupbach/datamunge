#!/usr/bin/env python3
"""Independent NumPy audit of the original compact PixelCNN++ records.

Checks original binary CIFAR rows, exact integer area quantization, analytic
logistic mixture VJPs, all captured Adam writes, whole-pixel Jacobians and
quantized channel-coupled generation. Does not fit or prepare replacement data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays,exact
from check_contractive_autoencoder import sigmoid
from check_gated_pixel_cnn import network,mask


def logistic(image,parameters):
    """Independent stable log-bin masses and analytic parameter gradient of mean channel NLL."""
    raw=parameters.reshape(-1,3,10);rgb=image.reshape(-1,3)
    means=raw[:,:,1:4].copy();scales=np.maximum(raw[:,:,4:7],-7)
    coeff=np.tanh(raw[:,:,7:10]);inv=np.exp(-scales)
    means[:,:,1]+=coeff[:,:,0]*rgb[:,None,0]
    means[:,:,2]+=coeff[:,:,1]*rgb[:,None,0]+coeff[:,:,2]*rgb[:,None,1]
    centered=rgb[:,None,:]-means
    half=float(np.float32(1/255));width=float(np.float32(2/255))
    hi=(centered+half)*inv;lo=(centered-half)*inv;delta=width*inv
    middle=-np.logaddexp(0,-hi)-np.logaddexp(0,lo)+np.log(-np.expm1(-delta))
    low=rgb[:,None,:]<=float(np.float32(-.999));high=rgb[:,None,:]>=float(np.float32(.999))
    mass=np.where(low,-np.logaddexp(0,-hi),np.where(high,-np.logaddexp(0,lo),middle))
    component=mass.sum(axis=2)
    logits=raw[:,:,0];maximum=logits.max(axis=1,keepdims=True)
    logweight=logits-maximum-np.log(np.exp(logits-maximum).sum(axis=1,keepdims=True))
    weighted=component+logweight;maximum=weighted.max(axis=1,keepdims=True)
    pixel=maximum[:,0]+np.log(np.exp(weighted-maximum).sum(axis=1))
    responsibilities=np.exp(weighted-pixel[:,None])
    weight=np.exp(logweight);factor=-responsibilities[:,:,None]/rgb.size
    dm=-(sigmoid(-hi)-sigmoid(lo))*inv
    # d/ds log(1-exp(-width exp(-s))) = -delta/expm1(delta).
    with np.errstate(over='ignore'):
        ds=-hi*sigmoid(-hi)+lo*sigmoid(lo)-delta/np.expm1(delta)
    dm=np.where(low,-inv*sigmoid(-hi),np.where(high,inv*sigmoid(lo),dm))
    ds=np.where(low,-hi*sigmoid(-hi),np.where(high,lo*sigmoid(lo),ds))
    dm*=factor;ds*=factor
    gradient=np.zeros_like(raw);gradient[:,:,0]=(weight-responsibilities)/rgb.size
    gradient[:,:,1:4]=dm;gradient[:,:,4:7]=ds*(raw[:,:,4:7]>=-7)
    gradient[:,:,7]=dm[:,:,1]*rgb[:,None,0]*(1-coeff[:,:,0]**2)
    gradient[:,:,8]=dm[:,:,2]*rgb[:,None,0]*(1-coeff[:,:,1]**2)
    gradient[:,:,9]=dm[:,:,2]*rgb[:,None,1]*(1-coeff[:,:,2]**2)
    out=dict(pixel_log_probability=pixel,component_log_probability=component,
             mixture_log_probability=logweight,means=means,log_scales=scales,
             coefficients=coeff,channel_log_mass=mass,loss=-pixel.mean()/3)
    return out,gradient.reshape(parameters.shape)


def forward(parameters,image):
    context,backward=network(parameters,image,np.zeros(len(image),int),False)
    out,gradient=logistic(image,context['logits'])
    out.update(parameters=context['logits'],vertical=context['vertical'],horizontal=context['horizontal'],
               per_image_nll=-out['pixel_log_probability'].reshape(-1,64).sum(axis=1))
    return out,backward(gradient)[0]


def check(root):
    receipt=json.loads((root/'baseline.json').read_text());source=root/'original-output'
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for file,digest in receipt['source_pins'].items():assert sha(root/('original-'+Path(file).name))==digest
    for file,digest in receipt['prepared_input_sha256'].items():assert sha(file)==digest
    for group,folder in [('archive_sha256',source),('phase_probe_sha256',root)]:
        for file,digest in receipt[group].items():assert sha(folder/file)==digest
    data=arrays(source/'data.npz');counts=dict(forward_values=0,gradient_values=0,optimizer_writes=0,
        exact_weight_values=0,finite_differences=0,ancestral_pixels=0,causal_jacobian_entries=0,
        batch_rows=0,source_RGB_values=0,quantized_RGB_values=0,normalized_channel_distributions=0)
    errors=dict(forward=0.,gradient=0.,sampling_parameters=0.);largest_forward_field=''
    largest_forward_relative_error=0.
    for split,start,n,file in [('fit',0,128,'data_batch_1.bin'),('validation',128,32,'data_batch_1.bin'),('test',64,32,'test_batch.bin')]:
        records=np.fromfile(Path(receipt['data_directory'])/file,np.uint8).reshape(-1,3073)[start:start+n]
        rgb=records[:,1:].reshape(n,3,32,32).transpose(0,2,3,1).astype(np.int32)
        codes=((rgb.reshape(n,8,4,8,4,3).sum(axis=(2,4))+8)//16).astype(np.int32)
        exact(rgb,data[split+'_original']);exact(codes,data[split+'_codes'])
        exact((codes.astype(float)/127.5-1).astype(np.float32),data[split+'_images'])
        exact(records[:,0].astype(np.int32),data[split+'_labels'])
        exact(np.arange(start,start+n,dtype=np.int32)+(50000 if split=='test' else 0),data[split+'_source_ids'])
        counts['source_RGB_values']+=rgb.size;counts['quantized_RGB_values']+=codes.size
    def compare(actual,expected,label,gradient=False):
        nonlocal largest_forward_field,largest_forward_relative_error
        assert set(actual)==set(expected),(label,set(actual)^set(expected))
        kind='gradient' if gradient else 'forward'
        for key,value in expected.items():
            np.testing.assert_allclose(actual[key],value,atol=5e-6 if gradient else 1e-4,
                rtol=2e-3 if gradient else 3e-4,err_msg=label+'/'+key)
            difference=float(np.max(abs(actual[key]-value)))
            if not gradient and difference>errors[kind]:
                largest_forward_field=label+'/'+key
                largest_forward_relative_error=float(np.max(abs(actual[key]-value)/np.maximum(1,abs(value))))
            counts[kind+'_values']+=np.size(value);errors[kind]=max(errors[kind],difference)
    for file in ['normalization','likelihood_fixture']:
        fixture=arrays(source/(file+'.npz'));out,gradient=logistic(fixture['images'].astype(float),fixture['parameters'].astype(float))
        fields=['pixel_log_probability','component_log_probability','mixture_log_probability','channel_log_mass','loss']
        compare({k:fixture[k] for k in fields},{k:out[k] for k in fields},file)
        compare({'gradient':fixture['gradient']},{'gradient':gradient},file,True)
        if file=='normalization':
            np.testing.assert_allclose(np.exp(out['channel_log_mass']).sum(axis=0),np.ones((3,3)),atol=2e-5,rtol=0)
            counts['normalized_channel_distributions']+=9
        else:
            p=fixture['parameters'].astype(float);x=fixture['images'].astype(float)
            for index in [(0,0,0,0),(0,0,0,4),(2,0,0,7),(3,0,0,18),(4,0,0,29),(6,0,0,16)]:
                high,low=p.copy(),p.copy();high[index]+=1e-5;low[index]-=1e-5
                fd=(logistic(x,high)[0]['loss']-logistic(x,low)[0]['loss'])/2e-5
                np.testing.assert_allclose(fd,gradient[index],atol=2e-8,rtol=2e-5);counts['finite_differences']+=1
    p0,pf=arrays(source/'initial_arrays.npz'),arrays(source/'final.npz')
    assert len(p0)==23 and sum(v.size for v in p0.values())==1410
    masks={k:mask(v.shape,'/vertical/' in k,k.startswith('block0/')) for k,v in p0.items()
           if k.endswith('/W') and ('/vertical/' in k or '/horizontal/' in k)}
    assert sum(np.count_nonzero(v==0) for v in masks.values())==448
    batch=np.genfromtxt(source/'batches.tsv',names=True,delimiter='\t',dtype=np.int64)
    assert len(batch)==384;counts['batch_rows']=len(batch)
    for epoch in [1,2,3]:exact(np.sort(batch[batch['epoch']==epoch]['row']),np.arange(128))
    first=arrays(source/'first_batch.npz')['images'];exact(first,data['fit_images'][batch[batch['step']==0]['row']])
    for name,p,x in [('first_forward',p0,first),('initial_outputs',p0,data['test_images']),('outputs',pf,data['test_images'])]:
        out,_=forward({k:v.astype(float) for k,v in p.items()},x.astype(float));compare(arrays(source/(name+'.npz')),out,name)
    before=p0;moment={k:np.zeros_like(v) for k,v in p0.items()};variance={k:np.zeros_like(v) for k,v in p0.items()}
    for step in range(48):
        rows=batch[batch['step']==step]['row'];assert len(rows)==8
        _,partials=forward({k:v.astype(float) for k,v in before.items()},data['fit_images'][rows].astype(float))
        gradient=arrays(root/'phases'/f'batch{step}-gradient.npz');compare(gradient,partials,f'batch{step}',True)
        after=arrays(root/'phases'/f'batch{step}-after.npz')
        for key,value in before.items():
            g=gradient[key];moment[key]=moment[key]*np.float32(.9)+g*np.float32(.1)
            variance[key]=variance[key]*np.float32(.999)+(g*g)*np.float32(.001)
            m=moment[key]/np.float32(1-.9**(step+1));v=variance[key]/np.float32(1-.999**(step+1))
            exact(value-m/(np.sqrt(v)+np.float32(1e-8))*np.float32(.003),after[key]);counts['exact_weight_values']+=value.size
            if key in masks:
                blocked=masks[key]==0;assert np.all(g[blocked]==0) and np.all(partials[key][blocked]==0)
                exact(after[key][blocked],p0[key][blocked])
        exact(arrays(root/'phases'/f'batch{step}-iterations.npz')['iterations'],np.array([step+1],np.int64))
        if step==0:
            compare(arrays(source/'initial_gradients.npz'),partials,'initial',True)
            for k,v in after.items():exact(v,arrays(source/'first_after.npz')[k])
        if step%16==15:
            for k,v in after.items():exact(v,arrays(source/f'epoch{step//16+1}_arrays.npz')[k])
        before=after;counts['optimizer_writes']+=1
    for key,value in before.items():exact(value,pf[key]);exact(value,arrays(source/'model_arrays.npz')[key])
    final={k:v.astype(float) for k,v in pf.items()};_,gradient=forward(final,first.astype(float))
    compare(arrays(source/'final_gradients.npz'),gradient,'final',True)
    for epoch in [1,2,3]:
        p={k:v.astype(float) for k,v in arrays(source/f'epoch{epoch}_arrays.npz').items()}
        loss=forward(p,data['validation_images'].astype(float))[0]['loss']
        recorded=np.loadtxt(source/'history.tsv',skiprows=1)[epoch-1]
        np.testing.assert_allclose(recorded[1:], [loss,loss/np.log(2)],rtol=1e-5,atol=1e-5)
    causal=arrays(source/'causality.npz');x=causal['images'].astype(float);out,back=network(final,x,np.zeros(1,int),False)
    compare({'parameters':causal['parameters']},{'parameters':out['logits']},'causal-head')
    for site in range(64):
        dy=np.zeros_like(out['logits']);dy.reshape(-1,64,30)[:,site]=1
        derivative=back(dy)[1].reshape(64,3)
        np.testing.assert_allclose(causal['jacobian'][site],derivative,atol=2e-7,rtol=2e-3)
        assert np.all(derivative[site:]==0) and np.all(causal['jacobian'][site,site:]==0)
        changed=x.copy();changed.reshape(1,64,3)[:,site:]*=-1
        changed_head=network(final,changed,np.zeros(1,int),False)[0]['logits'].reshape(1,64,30)
        np.testing.assert_array_equal(changed_head[:,:site+1],out['logits'].reshape(1,64,30)[:,:site+1])
        counts['causal_jacobian_entries']+=derivative.size
    # Independent central differences check the network VJP at its largest
    # active gradient coordinate in every parameter leaf, without fitting.
    initial={k:v.astype(float) for k,v in p0.items()}
    _,partials=forward(initial,first.astype(float))
    for key,value in initial.items():
        index=np.unravel_index(np.argmax(abs(partials[key])),value.shape)
        high={k:v.copy() for k,v in initial.items()};low={k:v.copy() for k,v in initial.items()}
        high[key][index]+=1e-4;low[key][index]-=1e-4
        fd=(forward(high,first.astype(float))[0]['loss']-forward(low,first.astype(float))[0]['loss'])/2e-4
        np.testing.assert_allclose(fd,partials[key][index],atol=2e-8,rtol=3e-3);counts['finite_differences']+=1
    component_margin=1.;quantization_margin=1.
    for file,n,seed,p,prefix_size in [('epoch1_samples',8,8639,arrays(source/'epoch1_arrays.npz'),0),
        ('epoch2_samples',8,8639,arrays(source/'epoch2_arrays.npz'),0),('epoch3_samples',8,8639,pf,0),
        ('samples',24,8639,pf,0),('completions',6,8641,pf,32)]:
        saved=arrays(source/(file+'.npz'));bank=(np.random.RandomState(seed).randint(0,2**32,(n,64,4),dtype=np.uint32).astype(float)+.5)/2**32
        exact(bank,saved['uniform']);image=np.zeros((n,8,8,3),np.float32)
        if prefix_size:
            exact(saved['truth'],data['test_images'][:n]);image.reshape(n,64,3)[:,:prefix_size]=saved['truth'].reshape(n,64,3)[:,:prefix_size]
            assert np.all(saved['components'][:,:prefix_size]==-1) and np.all(saved['parameters'][:,:prefix_size]==0)
        for site in range(prefix_size,64):
            head=network({k:v.astype(float) for k,v in p.items()},image.astype(float),np.zeros(n,int),False)[0]['logits'].reshape(n,64,30)[:,site]
            actual=saved['parameters'][:,site];difference=float(abs(head-actual).max());errors['sampling_parameters']=max(errors['sampling_parameters'],difference)
            np.testing.assert_allclose(head,actual,atol=2e-5,rtol=2e-4)
            # Generate directly from the independent NumPy network head.
            raw=head.reshape(n,3,10);w=np.exp(raw[:,:,0]-raw[:,:,0].max(axis=1,keepdims=True));w/=w.sum(axis=1,keepdims=True)
            cumulative=np.cumsum(w,axis=1);chosen=(bank[:,site,0,None]>=cumulative).sum(axis=1);chosen=np.minimum(chosen,2)
            exact(chosen.astype(np.int32),saved['components'][:,site])
            component_margin=min(component_margin,float(abs(cumulative-bank[:,site,0,None]).min()))
            for b in range(n):
                v=raw[b,chosen[b]];pixel=image.reshape(n,64,3)[b,site]
                for c in range(3):
                    mean=v[c+1]
                    if c==1:mean+=np.tanh(v[7])*pixel[0]
                    if c==2:mean+=np.tanh(v[8])*pixel[0]+np.tanh(v[9])*pixel[1]
                    u=bank[b,site,c+1];value=mean+np.exp(max(-7,v[c+4]))*(np.log(u)-np.log1p(-u))
                    code_position=(np.clip(value,-1,1)+1)*127.5
                    fraction=code_position+.5-np.floor(code_position+.5)
                    quantization_margin=min(quantization_margin,fraction,1-fraction)
                    code=int(np.floor(code_position+.5));pixel[c]=np.float32(code/127.5-1)
                    counts['ancestral_pixels']+=1
            exact(image.reshape(n,64,3)[:,site],saved['images'].reshape(n,64,3)[:,site])
        exact(image,saved['images'])
    probability=np.full((3,256),.5)
    for c in range(3):probability[c]+=np.bincount(data['fit_codes'][...,c].ravel(),minlength=256)
    probability/=128*64+128;exact(probability,arrays(source/'baseline.npz')['probability'])
    test=data['test_codes'];base=-np.mean([np.log(probability[c,test[...,c]]) for c in range(3)])
    metrics=json.loads((source/'metrics.json').read_text());np.testing.assert_allclose(base/np.log(2),metrics['fit_channel_histogram_bits'],atol=2e-13,rtol=0)
    assert metrics['test_bits_per_RGB_channel']<metrics['fit_channel_histogram_bits']<8
    return dict(passed=True,counts=counts,maximum_errors=errors,largest_absolute_forward_error_field=largest_forward_field,
        scaled_error_in_largest_absolute_forward_error_field=largest_forward_relative_error,all_48_adam_writes_bit_exact=True,
        all_448_masked_weights_and_partials_exact=True,all_64_whole_pixel_future_flips_exact=True,
        exact_original_CIFAR_rows_and_integer_area_quantization=True,all_256_bin_tail_distributions_normalized=True,
        quantized_sampled_RG_channel_coupling_exact=True,all_sampled_codes_from_independent_numpy_heads_exact=True,all_case_major_MT_uniforms_exact=True,
        minimum_component_selection_margin=component_margin,minimum_rounding_boundary_margin=quantization_margin,
        test_bits_per_RGB_channel=metrics['test_bits_per_RGB_channel'],fit_histogram_bits=metrics['fit_channel_histogram_bits'],
        compact_three_block_variant_not_multiscale_benchmark=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path('build/nn-report-conversion/pixel_cnn_pp'));parser.add_argument('--json',type=Path)
    args=parser.parse_args();result=check(args.root);rendered=json.dumps(result,indent=2)+'\n';print(rendered,end='')
    if args.json:args.json.write_text(rendered)


if __name__=='__main__':main()
