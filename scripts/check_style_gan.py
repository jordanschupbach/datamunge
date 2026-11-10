#!/usr/bin/env python3
"""Independent NumPy AdaIN/StyleGAN2 forward, analytic adjoints and Adam audit.

Reads original native artifacts only; never prepares data or fits parameters.
Verifies per-pixel mapping RMS, spatial instance normalization, per-example
modulation/demodulation, explicit noise strengths, all384 native Adam writes,
all style/truncation/noise controls and source row/latent/noise banks.
"""
import argparse,json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays,exact,gaussian
from check_big_gan import conv
from check_dcgan import down
from check_convolutional_autoencoder import pool,repeat
from check_contractive_autoencoder import sigmoid
EPS=float(np.float32(1e-8))


def add(out,key,value):out[key]=out.get(key,0)+value


def dense(p,group,name,x,leaky=False):
    prefix=f'{group}/{name}';pre=x@p[prefix+'/W']+p[prefix+'/b'];y=np.where(pre>0,pre,.2*pre) if leaky else pre
    def backward(gradient,out):
        d=gradient*np.where(pre>0,1.,.2) if leaky else gradient;add(out,prefix+'/W',x.T@d);add(out,prefix+'/b',d.sum(axis=0));return d@p[prefix+'/W'].T
    return y,backward


def normalize(x,axes,center):
    """Population divisor plus source-rounded guard, without affine or batch coupling."""
    c=x-x.mean(axis=axes,keepdims=True) if center else x;root=np.sqrt((c*c).mean(axis=axes,keepdims=True)+EPS);y=c/root
    def backward(dy):return (dy-(dy.mean(axis=axes,keepdims=True) if center else 0)-y*(dy*y).mean(axis=axes,keepdims=True))/root
    return y,backward


def mapping(p,z):
    rms,bn=normalize(z,(1,),False);hidden,b1=dense(p,0,'mapping1',rms,True);w,b2=dense(p,0,'mapping2',hidden,True)
    def backward(dw,out):return bn(b1(b2(dw,out),out))
    return w,backward


def modulated(x,weight,style,demodulate):
    """Per-row effective kernels and a closed-form normalization adjoint."""
    n,h,w,c=x.shape;k=weight.shape[0];pad=k//2;mod=weight[None]*style[:,None,None,:,None];den=1/np.sqrt((mod*mod).sum(axis=(1,2,3))+EPS)
    if not demodulate:den=den*0+1
    kernel=mod*den[:,None,None,None,:];padded=np.pad(x,((0,0),(pad,pad),(pad,pad),(0,0)));image=np.zeros((n,h,w,weight.shape[-1]))
    for y in range(k):
        for col in range(k):image+=np.einsum('nhwc,nco->nhwo',padded[:,y:y+h,col:col+w],kernel[:,y,col])
    def backward(dimage):
        dx=np.zeros_like(padded);dk=np.zeros_like(kernel)
        for y in range(k):
            for col in range(k):
                dk[:,y,col]=np.einsum('nhwc,nhwo->nco',padded[:,y:y+h,col:col+w],dimage)
                dx[:,y:y+h,col:col+w]+=np.einsum('nhwo,nco->nhwc',dimage,kernel[:,y,col])
        dm=dk*den[:,None,None,None,:]
        if demodulate:dm-=mod*((dk*mod).sum(axis=(1,2,3))*den**3)[:,None,None,None,:]
        dw=(dm*style[:,None,None,:,None]).sum(axis=0);ds=(dm*weight[None]).sum(axis=(1,2,4))
        return dx[:,pad:pad+h,pad:pad+w],dw,ds
    return image,den,backward


def mod_layer(p,name,x,w,demodulate=True,override=None):
    style,bs=dense(p,0,name+'/style',w);style=style+1 if override is None else override;image,den,bm=modulated(x,p['0/'+name+'/W'],style,demodulate)
    def backward(dy,out):
        dx,dw,ds=bm(dy);add(out,'0/'+name+'/W',dw);return dx,bs(ds,out) if override is None else np.zeros_like(w)
    return image,style,den,backward


def styled(p,name,x,w,noise,version2,override=None):
    if version2:pre,style,den,bc=mod_layer(p,name,x,w,True,override)
    else:
        pre,plain=conv(x,p['0/'+name+'/W']);style,bs=dense(p,0,name+'/style',w);den=np.array(1.)
        def bc(dy,out):
            dx,dw=plain(dy);add(out,'0/'+name+'/W',dw);return dx,None
    strength=p['0/'+name+'/noise_strength'];biased=(pre+noise*strength)+p['0/'+name+'/b'];h=np.where(biased>0,biased,.2*biased)
    if not version2:
        normalized,bn=normalize(h,(1,2),True);scale=style[:,:4]+1;shift=style[:,4:];image=normalized*scale[:,None,None,:]+shift[:,None,None,:]
    else:image=h
    def backward(dy,out):
        if not version2:
            ds=np.concatenate(((dy*normalized).sum(axis=(1,2)),dy.sum(axis=(1,2))),axis=1);dw_style=bs(ds,out);dy=bn(dy*scale[:,None,None,:])
        delta=dy*np.where(biased>0,1.,.2);add(out,'0/'+name+'/noise_strength',(delta*noise).sum(axis=(0,1,2)));add(out,'0/'+name+'/b',delta.sum(axis=(0,1,2)));dn=(delta*strength).sum(axis=3,keepdims=True);dx,dw=bc(delta,out)
        return dx,dw if version2 else dw_style,dn
    return image,style,den,backward


def synthesize(p,w0,w1,n4,n8,version2,override=None):
    constant=np.tile(p['0/constant'],(len(w0),1,1,1));first,s0,d0,b0=styled(p,'layer4',constant,w0,n4,version2,override)
    second,s1,d1,b1=styled(p,'layer8',repeat(first),w1,n8,version2)
    if version2:logit,_,_,br=mod_layer(p,'rgb',second,w1,False)
    else:
        logit,plain=conv(second,p['0/rgb/W'])
        def br(dy,out):
            dx,dw=plain(dy);add(out,'0/rgb/W',dw);return dx,np.zeros_like(w1)
    image=sigmoid(logit+p['0/rgb/b'])
    def backward(dy,out):
        delta=dy*image*(1-image);add(out,'0/rgb/b',delta.sum(axis=(0,1,2)));dx,dw_rgb=br(delta,out);dx,dw1,dn8=b1(dx,out);dx,dw0,dn4=b0(4*pool(dx),out);add(out,'0/constant',dx.sum(axis=0,keepdims=True));return dw0,dw1+dw_rgb,dn4,dn8
    return dict(fake=image,first_style=s0,second_style=s1,first_demodulator=d0,second_demodulator=d1),backward


def discriminate(p,image):
    first,b1=down(image,p['1/down4/W']);first+=p['1/down4/b'];h1=np.where(first>0,first,.2*first);second,b2=conv(h1,p['1/conv/W']);second+=p['1/conv/b'];h2=np.where(second>0,second,.2*second);score,bs=dense(p,1,'score',h2.reshape(len(image),128))
    def backward(gradient,out):
        dy=bs(gradient,out).reshape(h2.shape)*np.where(second>0,1.,.2);add(out,'1/conv/b',dy.sum(axis=(0,1,2)));dy,dw=b2(dy);add(out,'1/conv/W',dw);dy*=np.where(first>0,1.,.2);add(out,'1/down4/b',dy.sum(axis=(0,1,2)));dy,dw=b1(dy);add(out,'1/down4/W',dw);return dy
    return score,backward


def forward(p,x,z,n4,n8,version2):
    w,bw=mapping(p,z);g,bg=synthesize(p,w,w,n4,n8,version2);real,br=discriminate(p,x);fake,bf=discriminate(p,g['fake']);out={**g,'mapped':w,'real_score':real,'fake_score':fake,'d_loss':.5*(((real-1)**2).mean()+(fake*fake).mean()),'g_loss':((fake-1)**2).mean()}
    dg={};dd={};br((real-1)/real.size,dd);bf(fake/fake.size,dd);unused={};dy=bf(2*(fake-1)/fake.size,unused);dw0,dw1,_,_=bg(dy,dg);bw(dw0+dw1,dg)
    return out,{**dg,**dd}


def draw(n,zseed,nseed):
    bank=gaussian((n,80),nseed).astype(np.float32)
    return dict(z=gaussian((n,8),zseed).astype(np.float32),noise4=bank[:,:16].reshape(n,4,4,1),noise8=bank[:,16:].reshape(n,8,8,1))


def check(root):
    """Audit both source trajectories and all source controls without fitting."""
    source=root/'original-output';data=arrays(source/'data.npz');gradient_values=forward_values=0;gerror=ferror=0.;updates=0;differences=0
    for split in ['fit','validation','test']:
        image=data[split+'_original'].reshape(-1,28,28).astype(float);resized=np.zeros((len(image),8,8,1))
        for y in range(8):
            for x in range(8):
                py,px=(y+.5)*3.5-.5,(x+.5)*3.5-.5;iy,ix=int(py),int(px)
                for dy in [0,1]:
                    for dx in [0,1]:resized[:,y,x,0]+=(py-iy if dy else 1-(py-iy))*(px-ix if dx else 1-(px-ix))*image[:,iy+dy,ix+dx]
        exact(resized.astype(np.float32),data[split+'_images'])
    def floating(p):return {k:v.astype(float) for k,v in p.items()}
    def compare(native,expected,label):
        nonlocal forward_values,ferror
        for key,value in expected.items():np.testing.assert_allclose(native[key],value,atol=1e-5,rtol=4e-4,err_msg=label+'/'+key);forward_values+=np.size(value);ferror=max(ferror,float(np.max(abs(native[key]-value))))
    def compare_grad(native,expected,label):
        nonlocal gradient_values,gerror
        assert set(native)==set(expected),(label,set(native)^set(expected))
        for key,value in expected.items():np.testing.assert_allclose(native[key],value,atol=4e-5,rtol=5e-3,err_msg=label+'/'+key);gradient_values+=value.size;gerror=max(gerror,float(np.max(abs(native[key]-value))))
    results={}
    for name,version2,gcount in [('adain',False,661),('modulated',True,625)]:
        folder=source/name;probe=root/name;initial=arrays(folder/'initial_arrays.npz');final=arrays(folder/'final.npz');assert sum(v.size for k,v in initial.items() if k[0]=='0')==gcount and sum(v.size for k,v in initial.items() if k[0]=='1')==493
        inputs={split:arrays(folder/(split+'_inputs.npz')) for split in ['validation','test']}
        for split,zseed,nseed in [('validation',6637,6653),('test',6659,6661)]:
            expected=draw(200,zseed,nseed)
            for key,value in expected.items():exact(value,inputs[split][key])
        first=arrays(probe/'probe-first-input.npz');compare(arrays(probe/'probe-first-forward.npz'),forward(floating(initial),*[first[k].astype(float) for k in ['x','z','noise4','noise8']],version2)[0],name+'/first')
        initial_outputs=arrays(folder/'initial_outputs.npz');control=arrays(probe/'probe-initial-noise-control.npz')
        for key,value in initial_outputs.items():exact(value,control[key])
        batch=np.genfromtxt(folder/'batches.tsv',names=True,delimiter='\t',dtype=np.int64);assert len(batch)==3000
        before=initial;moment={k:np.zeros_like(v) for k,v in initial.items()};variance={k:np.zeros_like(v) for k,v in initial.items()};iteration=[0,0];index=0
        for epoch in range(1,4):
            bank=draw(1000,6689+epoch,6701+epoch);record=batch[batch['epoch']==epoch];exact(np.sort(record['row']),np.arange(1000))
            for step in range(32):
                rows=record[record['step']==index]['row'];assert len(rows)==(8 if step==31 else 32);x=data['fit_images'][rows].astype(float);z,n4,n8=[bank[k][rows].astype(float) for k in ['z','noise4','noise8']]
                if index==0:
                    for key,value in zip(['x','z','noise4','noise8'],[data['fit_images'][rows],bank['z'][rows],bank['noise4'][rows],bank['noise8'][rows]]):exact(first[key],value)
                for group in [1,0]:
                    tag=f'batch{index}'+('-d' if group else '-g');native=arrays(probe/'phases'/(tag+'-gradient.npz'));after=arrays(probe/'phases'/(tag+'-after.npz'));expected=forward(floating(before),x,z,n4,n8,version2)[1];compare_grad(native,{k:v for k,v in expected.items() if k[0]==str(group)},name+'/'+tag)
                    iteration[group]+=1;t=iteration[group];updates+=1
                    for key,value in before.items():
                        if key[0]!=str(group):exact(value,after[key]);continue
                        g=native[key];moment[key]=moment[key]*np.float32(.5)+g*np.float32(.5);variance[key]=variance[key]*np.float32(.999)+(g*g)*np.float32(.001);mean=moment[key]/np.float32(1-.5**t);var=variance[key]/np.float32(1-.999**t);exact(value-(mean/(np.sqrt(var)+np.float32(1e-8)))*np.float32(.0002),after[key])
                    before=after
                index+=1
            checkpoint=arrays(probe/f'probe-epoch{epoch}.npz')
            for key,value in checkpoint.items():exact(before[key],value)
            for split in ['validation','test']:
                expected=forward(floating(checkpoint),data[split+'_images'].astype(float),*[inputs[split][k].astype(float) for k in ['z','noise4','noise8']],version2)[0];compare(arrays(probe/f'probe-epoch{epoch}-{split}.npz'),expected,name+f'/{epoch}/'+split)
        assert iteration==[96,96] and index==96
        for key,value in final.items():exact(before[key],value)
        for p,file in [(initial,'initial_outputs.npz'),(final,'outputs.npz')]:compare(arrays(folder/file),forward(floating(p),data['test_images'].astype(float),*[inputs['test'][k].astype(float) for k in ['z','noise4','noise8']],version2)[0],name+'/'+file)
        p=floating(final);expected=forward(p,*[first[k].astype(float) for k in ['x','z','noise4','noise8']],version2)[1]
        for group in [0,1]:compare_grad(arrays(folder/f'final_gradients_group{group}.npz'),{k:v for k,v in expected.items() if k[0]==str(group)},name+'/final')
        controls=arrays(folder/'style_controls.npz');w,_=mapping(p,inputs['test']['z'].astype(float));other,_=mapping(p,controls['other_z'].astype(float));compare(controls,{'w':w,'other_w':other},name+'/maps');prior,_=mapping(p,controls['center_z'].astype(float));compare(controls,{'center_w':prior},name+'/center');np.testing.assert_allclose(controls['center'],prior.mean(axis=0),atol=3e-6,rtol=3e-4)
        exact(controls['center'],controls['center_w'].astype(float).mean(axis=0).astype(np.float32));other_bank=draw(200,6719,6673);center_bank=draw(2048,6733,6737)
        for key in ['noise4','noise8']:exact(controls['changed_'+key],other_bank[key]);exact(controls['common_'+key],np.repeat(inputs['test'][key][[0]],200,axis=0))
        exact(controls['other_z'],other_bank['z']);exact(controls['center_z'],center_bank['z'])
        variants=[('mixed_second',w,other,inputs['test']['noise4'],inputs['test']['noise8']),('mixed_first',other,w,inputs['test']['noise4'],inputs['test']['noise8']),('other_plain',other,other,inputs['test']['noise4'],inputs['test']['noise8']),('changed_noise_fake',w,w,controls['changed_noise4'],controls['changed_noise8']),('changed_only_noise4',w,w,controls['changed_noise4'],inputs['test']['noise8']),('changed_only_noise8',w,w,inputs['test']['noise4'],controls['changed_noise8'])]
        for key,wa,wb,na,nb in variants:compare(controls,{key:synthesize(p,wa,wb,na.astype(float),nb.astype(float),version2)[0]['fake']},name+'/'+key)
        # Source-rounded truncated W is used for independent synthesis controls.
        for level,psi in enumerate([1,.5,0]):
            target=(controls['center'][None]+np.float32(psi)*(controls['w']-controls['center'][None]));exact(controls['truncated_w'+str(level)],target)
            expected=synthesize(p,target.astype(float),target.astype(float),controls['common_noise4'].astype(float),controls['common_noise8'].astype(float),version2)[0]['fake'];compare(controls,{'truncated_fake'+str(level):expected},name+'/truncated')
        np.testing.assert_allclose(controls['truncated_fake2'],np.repeat(controls['truncated_fake2'][[0]],200,axis=0),atol=3e-6,rtol=0)
        if version2:
            expected=synthesize(p,w,w,inputs['test']['noise4'].astype(float),inputs['test']['noise8'].astype(float),True,controls['doubled_first_style'].astype(float))[0]
            compare(controls,{'doubled_first_style_fake':expected['fake'],'doubled_first_style_demodulator':expected['first_demodulator']},name+'/double-style')
            np.testing.assert_allclose(controls['doubled_first_style_fake'],arrays(folder/'outputs.npz')['fake'],atol=3e-6,rtol=0)
        # Representative final differences cover every named G/D tensor of both variants.
        small=[first[k][:2].astype(float) for k in ['x','z','noise4','noise8']];expected=forward(p,*small,version2)[1];fd=0
        for key,gradient in expected.items():
            for i in sorted({0,gradient.size//2,gradient.size-1}):
                old=p[key].flat[i];h=1e-6;field='g_loss' if key[0]=='0' else 'd_loss';p[key].flat[i]=old+h;plus=forward(p,*small,version2)[0][field];p[key].flat[i]=old-h;minus=forward(p,*small,version2)[0][field];p[key].flat[i]=old
                np.testing.assert_allclose((plus-minus)/(2*h),gradient.flat[i],atol=2e-6,rtol=8e-3,err_msg=name+'/'+key);fd+=1;differences+=1
        history=np.genfromtxt(folder/'history.tsv',names=True,delimiter='\t');meta=json.loads((folder/'metrics.json').read_text());out=arrays(folder/'outputs.npz')
        for row in history:
            saved=arrays(probe/f"probe-epoch{int(row['epoch'])}-validation.npz")
            for key in ['d_loss','g_loss']:assert float(row[key])==saved[key].item()
        for key in ['d_loss','g_loss']:assert meta['test_'+key]==out[key].item()
        results[name]=dict(generator_parameters=gcount,critic_parameters=493,optimizer_iterations=iteration,final_scalar_differences=fd)
    assert updates==384
    return dict(passed=True,variants=results,native_parameter_derivatives_checked=gradient_values,maximum_derivative_error=gerror,forward_values_checked=forward_values,maximum_forward_error=ferror,independent_final_scalar_finite_differences=differences,all_384_Adam_updates_bit_exact=True,all_384_phase_ownership_checks_pass=True,source_resize_values_exact=89600,batch_row_visits_checked=6000,all_latent_and_spatial_noise_banks_exact=True,initial_noise_strengths_have_exactly_zero_effect=True,all_mapping_mixing_truncation_noise_and_demodulation_controls_checked=True,all_validation_history_and_final_test_scalar_records_exact=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path('build/nn-report-conversion/style_gan'));parser.add_argument('--json',type=Path);args=parser.parse_args();result=check(args.root);text=json.dumps(result,indent=2)+'\n';print(text,end='')
    if args.json:args.json.write_text(text)

if __name__=='__main__':main()
