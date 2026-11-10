#!/usr/bin/env python3
"""Independent NumPy StarGAN maps, shared-generator adjoints and exact Adam audit.

Reads frozen native captures only; never prepares data or fits a model. Checks
three disjoint per-class domains, accepted target words, all174 training writes,
source/domain losses, repeated G reconstruction, controls and known-map metrics.
"""
import argparse,json
from pathlib import Path
import numpy as np
from check_pix2pix import arrays,exact,generate as pix_generate,layer
from check_style_gan import dense


def domains(image,domain):
    """0 unchanged,1 intensity inversion,2 source90-degree rotation; all per row."""
    result=image.copy();result[domain==1]=np.ones((),image.dtype)-image[domain==1]
    result[domain==2]=image[domain==2,::-1].swapaxes(1,2)
    return result


def targets(count,seed):
    rng=np.random.RandomState(seed);words=[]
    while len(words)<count:
        word=int(rng.randint(0,2**32,dtype=np.uint32))
        if word<2**32-1:words.append(word)
    raw=np.array(words,dtype=float);return (raw.astype(np.uint64)%3).astype(np.int32),raw


def generate(p,image,target):
    code=np.broadcast_to(np.eye(3)[target][:,None,None,:],(*image.shape[:3],3));out,back=pix_generate(p,np.concatenate([image,code],axis=3))
    def backward(gradient):
        dx,partial=back(gradient);return dx[:,:,:,:1],partial
    return out,backward


def discriminate(p,image):
    first,b1=layer(p,1,'down14',image);h1=np.where(first>0,first,.2*first)
    second,b2=layer(p,1,'down7',h1);h2=np.where(second>0,second,.2*second)
    logits,bd=dense(p,1,'domain',h2.mean(axis=(1,2)));source,bs=layer(p,1,'source',h2)
    shift=logits-logits.max(axis=1,keepdims=True);logprob=shift-np.log(np.exp(shift).sum(axis=1,keepdims=True));probability=np.exp(logprob)
    def backward(ds,dl):
        out={};dy=bs(ds,out)+bd(dl,out)[:,None,None,:]/49;dy*=np.where(second>0,1.,.2);dy=b2(dy,out)*np.where(first>0,1.,.2);return b1(dy,out),out
    return dict(source=source,domain_logits=logits,domain_probability=probability),logprob,backward


def add(out,term):
    for key,value in term.items():out[key]=out.get(key,0)+value


def forward(p,image,original,target):
    fake,bg=generate(p,image,target);reconstruction,bc=generate(p,fake,original)
    real,rl,br=discriminate(p,image);generated,fl,bf=discriminate(p,fake)
    sg=((generated['source']-1)**2).mean();sd=.5*(((real['source']-1)**2).mean()+(generated['source']**2).mean())
    real_ce=-rl[np.arange(len(image)),original].mean();fake_ce=-fl[np.arange(len(image)),target].mean();cycle=np.abs(reconstruction-image).mean()
    out=dict(fake=fake,reconstruction=reconstruction,real_source=real['source'],fake_source=generated['source'],real_domain_logits=real['domain_logits'],fake_domain_logits=generated['domain_logits'],real_domain_probability=real['domain_probability'],fake_domain_probability=generated['domain_probability'],source_loss_g=sg,source_loss_d=sd,real_domain_loss=real_ce,fake_domain_loss=fake_ce,reconstruction_loss=cycle,generator_loss=(sg+fake_ce)+10*cycle,discriminator_loss=sd+real_ce)
    identity=np.eye(3);d={};add(d,br((real['source']-1)/real['source'].size,(real['domain_probability']-identity[original])/len(image))[1]);add(d,bf(generated['source']/generated['source'].size,np.zeros_like(fl))[1])
    df,_=bf(2*(generated['source']-1)/generated['source'].size,(generated['domain_probability']-identity[target])/len(image))
    df_cycle,cycle_part=bc(10*np.sign(reconstruction-image)/reconstruction.size);_,g=bg(df+df_cycle);add(g,cycle_part)
    return out,{**g,**d}


def check(root):
    source=root/'original-output';data=arrays(source/'data.npz');initial=arrays(source/'initial_arrays.npz');final=arrays(source/'final.npz');assert len(initial)==18 and sum(v.size for v in initial.values())==2297
    for split,per in [('fit',30),('validation',6),('test',6)]:
        image,labels,ids=[data[split+'_all_'+k] for k in ['original','digit','source_ids']];seen=np.zeros(10,int);rows=[];unused=[];codes=[]
        for i,c in enumerate(labels):
            if seen[c]<3*per:rows.append(i);codes.append(seen[c]//per)
            else:unused.append(i)
            seen[c]+=1
        codes=np.array(codes,dtype=np.int32);assert np.array_equal(seen,np.full(10,len(image)//10))
        for field,expected in [('domains',codes),('original',image[rows]),('images',domains(image[rows],codes)),('source_ids',ids[rows]),('digits',labels[rows].astype(np.int32)),('unused_source_ids',ids[unused])]:exact(data[split+'_'+field],expected)
        assert not set(ids[rows])&set(ids[unused]);assert np.array_equal(np.bincount(codes),np.full(3,10*per))
    fixed=arrays(source/'fixed_targets.npz')
    for split in ['validation','test']:exact(fixed[split],((data[split+'_domains']+1)%3).astype(np.int32))
    records=np.genfromtxt(source/'batches.tsv',names=True,delimiter='\t',dtype=np.int64);assert len(records)==2700
    before=initial;moment={k:np.zeros_like(v) for k,v in initial.items()};variance={k:np.zeros_like(v) for k,v in initial.items()};iterations=[0,0];gradient_values=forward_values=0;gerror=ferror=0.;batch=0
    def compare(native,expected):
        nonlocal forward_values,ferror
        assert set(native)==set(expected)
        for key,value in expected.items():np.testing.assert_allclose(native[key],value,atol=1e-5,rtol=5e-4,err_msg=key);forward_values+=np.size(value);ferror=max(ferror,float(np.max(abs(native[key]-value))))
    def compare_grad(native,expected,label):
        nonlocal gradient_values,gerror
        assert set(native)==set(expected)
        for key,value in expected.items():np.testing.assert_allclose(native[key],value,atol=3e-5,rtol=4e-3,err_msg=label+'/'+key);gradient_values+=value.size;gerror=max(gerror,float(np.max(abs(native[key]-value))))
    first=arrays(root/'probe-first-input.npz');compare(arrays(root/'probe-first-forward.npz'),forward({k:v.astype(float) for k,v in initial.items()},first['x'].astype(float),first['original'],first['target'])[0])
    stale=arrays(root/'probe-stale-g-gradient.npz');expected=forward({k:v.astype(float) for k,v in initial.items()},first['x'].astype(float),first['original'],first['target'])[1];compare_grad(stale,{k:v for k,v in expected.items() if k[0]=='0'},'stale-g')
    for epoch in range(1,4):
        prior=arrays(root/f'probe-targets{epoch}.npz');expected,raw=targets(900,6337+epoch);exact(prior['target'],expected);exact(prior['raw'],raw)
        visits=records[records['epoch']==epoch];exact(np.sort(visits['row']),np.arange(900));np.testing.assert_array_equal(visits['source_id'],data['fit_source_ids'][visits['row']])
        for step in range(29):
            record=visits[visits['step']==batch];rows=record['row'];assert len(rows)==(4 if step==28 else 32)
            image=data['fit_images'][rows].astype(float);original=data['fit_domains'][rows];target=prior['target'][rows]
            if batch==0:
                for key,value in [('x',data['fit_images'][rows]),('original',original),('target',target)]:exact(first[key],value)
            for group in [1,0]:
                name=f'batch{batch}-'+('d' if group else 'g');native=arrays(root/'phases'/(name+'-gradient.npz'));after=arrays(root/'phases'/(name+'-after.npz'));partials=forward({k:v.astype(float) for k,v in before.items()},image,original,target)[1];compare_grad(native,{k:v for k,v in partials.items() if k[0]==str(group)},name)
                iterations[group]+=1;t=iterations[group]
                for key,value in before.items():
                    if key[0]!=str(group):exact(value,after[key]);continue
                    g=native[key];moment[key]=moment[key]*np.float32(.5)+g*np.float32(.5);variance[key]=variance[key]*np.float32(.999)+(g*g)*np.float32(.001);mean=moment[key]/np.float32(1-.5**t);var=variance[key]/np.float32(1-.999**t);direction=mean/(np.sqrt(var)+np.float32(1e-8));exact(value-direction*np.float32(.0002),after[key])
                before=after
            batch+=1
        checkpoint=arrays(root/f'probe-epoch{epoch}.npz')
        for key,value in checkpoint.items():exact(before[key],value)
        for split in ['validation','test']:compare(arrays(root/f'probe-epoch{epoch}-{split}.npz'),forward({k:v.astype(float) for k,v in checkpoint.items()},data[split+'_images'].astype(float),data[split+'_domains'],fixed[split])[0])
    assert iterations==[87,87] and batch==87
    for key,value in final.items():exact(before[key],value)
    for p,name in [(initial,'initial_outputs.npz'),(final,'outputs.npz')]:compare(arrays(source/name),forward({k:v.astype(float) for k,v in p.items()},data['test_images'].astype(float),data['test_domains'],fixed['test'])[0])
    p={k:v.astype(float) for k,v in final.items()};partials=forward(p,first['x'].astype(float),first['original'],first['target'])[1]
    for group in [0,1]:compare_grad(arrays(source/f'final_gradients_group{group}.npz'),{k:v for k,v in partials.items() if k[0]==str(group)},'final')
    control=arrays(source/'domain_controls.npz')
    for k in range(3):np.testing.assert_allclose(control[f'target{k}_prediction'],generate(p,data['test_images'].astype(float),np.full(180,k))[0],atol=1e-5,rtol=5e-4)
    real=discriminate(p,data['test_images'].astype(float))[0]
    for key,field in [('image_only_source','source'),('image_only_domain_probability','domain_probability')]:np.testing.assert_allclose(control[key],real[field],atol=1e-5,rtol=5e-4)
    known=arrays(source/'diagnostic_known_mapping.npz')['target'];exact(known,domains(data['test_original'],fixed['test']))
    out=arrays(source/'outputs.npz');meta=json.loads((source/'metrics.json').read_text())
    for key in ['discriminator_loss','generator_loss','source_loss_g','source_loss_d','real_domain_loss','fake_domain_loss','reconstruction_loss']:assert meta['test_'+key]==out[key].item()
    np.testing.assert_allclose(meta['diagnostic_known_mapping_mse'],((out['fake'].astype(float)-known.astype(float))**2).mean(),atol=2e-15)
    for field,truth,key in [('real_domain_probability',data['test_domains'],'test_real_domain_accuracy'),('fake_domain_probability',fixed['test'],'test_generated_requested_domain_accuracy')]:assert meta[key]==np.mean(out[field].argmax(axis=1)==truth)
    history=np.genfromtxt(source/'history.tsv',names=True,delimiter='\t')
    for row in history:
        fields=arrays(root/f"probe-epoch{int(row['epoch'])}-validation.npz")
        for key in history.dtype.names[1:]:assert float(row[key])==fields[key].item()
    fresh=arrays(root/'phases/batch0-g-gradient.npz');change=max(float(abs(stale[k]-fresh[k]).max()) for k in stale);assert change>1e-8
    # Smooth finite differences on two cases cover every named leaf.
    x,o,t=first['x'][:2].astype(float),first['original'][:2],first['target'][:2];partials=forward(p,x,o,t)[1];differences=0
    for key,g in partials.items():
        for i in sorted({0,g.size//2,g.size-1}):
            old=p[key].flat[i];h=1e-6;field='generator_loss' if key[0]=='0' else 'discriminator_loss';p[key].flat[i]=old+h;plus=forward(p,x,o,t)[0][field];p[key].flat[i]=old-h;minus=forward(p,x,o,t)[0][field];p[key].flat[i]=old;np.testing.assert_allclose((plus-minus)/(2*h),g.flat[i],atol=6e-7,rtol=4e-3,err_msg=key);differences+=1
    return dict(passed=True,native_parameter_derivatives_checked=gradient_values,maximum_derivative_error=gerror,forward_values_checked=forward_values,maximum_forward_error=ferror,independent_final_scalar_finite_differences=differences,all_174_Adam_updates_bit_exact=True,all_174_phase_ownership_checks_pass=True,optimizer_iterations=iterations,batch_row_visits_checked=2700,accepted_target_words_checked=2700,disjoint_per_class_domains_and_unused_ids_exact=True,all_known_diagnostic_mapping_pixels_exact=True,source_only_and_image_only_controls_checked=True,all_validation_history_and_final_test_scalar_records_exact=True,stale_vs_fresh_first_G_gradient_max_difference=change)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path('build/nn-report-conversion/star_gan'));parser.add_argument('--json',type=Path);a=parser.parse_args();result=check(a.root);text=json.dumps(result,indent=2)+'\n';print(text,end='')
    if a.json:a.json.write_text(text)
if __name__=='__main__':main()
