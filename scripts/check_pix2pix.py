#!/usr/bin/env python3
"""Independent NumPy paired Pix2Pix forwards, skip adjoints and Adam transitions.

Reads frozen C++ artifacts; never prepares data or fits parameters. Verifies
integer Sobel conditions, true concatenation skips, patch-mean logistic losses,
paired L1, every phase's group gradients and all192 native Adam writes.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays,exact
from check_dcgan import down,up
from check_big_gan import conv
from check_contractive_autoencoder import sigmoid
from check_auxiliary_classifier_gan import coarse,mmd


def edges(image):
    """Recover 8-bit integers, zero-pad, threshold inclusive raw L1 Sobel magnitude."""
    n,h,w,_=image.shape;integer=np.floor(image.astype(float)*255+.5).astype(np.int32)
    padded=np.pad(integer,((0,0),(1,1),(1,1),(0,0)));gx=np.zeros_like(integer);gy=np.zeros_like(integer)
    sx=[[-1,0,1],[-2,0,2],[-1,0,1]];sy=[[-1,-2,-1],[0,0,0],[1,2,1]]
    for y in range(3):
        for x in range(3):gx+=sx[y][x]*padded[:,y:y+h,x:x+w];gy+=sy[y][x]*padded[:,y:y+h,x:x+w]
    return (np.abs(gx)+np.abs(gy)>=256).astype(np.float32)


def layer(p,group,name,x,activation=None,transpose=False):
    """Biased normal convolution/transpose with explicit independent parameter adjoints."""
    prefix=f'{group}/{name}';weight=p[prefix+'/W'];operation=up if transpose else down if weight.shape[0]==4 else conv
    pre,back=operation(x,weight);pre+=p[prefix+'/b'];y=np.maximum(pre,0) if activation=='relu' else sigmoid(pre) if activation=='sigmoid' else pre
    def backward(gradient,out):
        delta=gradient*(pre>0) if activation=='relu' else gradient*y*(1-y) if activation=='sigmoid' else gradient
        dx,dw=back(delta)
        for key,value in [(prefix+'/W',dw),(prefix+'/b',delta.sum(axis=(0,1,2)))]:out[key]=out.get(key,0)+value
        return dx
    return y,backward


def generate(p,source,details=False):
    """Two encoder maps, two learned upconvolutions, both actual skip concatenations."""
    d14,b1=layer(p,0,'down14',source,'relu');d7,b2=layer(p,0,'down7',d14,'relu');u14,b3=layer(p,0,'up14',d7,'relu',True)
    j14=np.concatenate((u14,d14),axis=3);u28,b4=layer(p,0,'up28',j14,'relu',True);j28=np.concatenate((u28,source),axis=3)
    image,b5=layer(p,0,'output',j28,'sigmoid')
    def backward(gradient):
        out={};dj28=b5(gradient,out);dj14=b4(dj28[:,:,:,:2],out);dd7=b3(dj14[:,:,:,:4],out)
        dd14=b2(dd7,out)+dj14[:,:,:,4:];ds=b1(dd14,out)+dj28[:,:,:,2:]
        return ds,out
    return (dict(down14=d14,down7=d7,up14=u14,joined14=j14,up28=u28,joined28=j28,image=image) if details else image),backward


def discriminate(p,source,image):
    """Conditional concat -> strided leaky maps ->49 raw patch scores per image."""
    first,b1=layer(p,1,'down14',np.concatenate((source,image),axis=3));h1=np.where(first>0,first,.2*first)
    second,b2=layer(p,1,'down7',h1);h2=np.where(second>0,second,.2*second);logit,b3=layer(p,1,'output',h2)
    def backward(gradient,out):
        delta=b3(gradient,out)*np.where(second>0,1.,.2);delta=b2(delta,out)*np.where(first>0,1.,.2);return b1(delta,out)
    return logit,backward


def forward(p,target,source):
    """Stable patch means and source full-scaled paired L1; shared D contributions add."""
    fake,bg=generate(p,source);real,br=discriminate(p,source,target);fl,bf=discriminate(p,source,fake)
    adv=np.logaddexp(0,-fl).mean();l1=np.abs(fake-target).mean()
    result=dict(fake=fake,real_logit=real,fake_logit=fl,g_adversarial=adv,l1=l1,discriminator_loss=np.logaddexp(0,-real).mean()+np.logaddexp(0,fl).mean(),generator_loss=adv+100*l1)
    d={};br((sigmoid(real)-1)/real.size,d);bf(sigmoid(fl)/fl.size,d)
    unused={};df=bf((sigmoid(fl)-1)/fl.size,unused)[:,:,:,1:]+100*np.sign(fake-target)/fake.size;_,g=bg(df)
    return result,{**g,**d}


def check(root):
    """Audit all192 phases, exact Float32 state transitions, held-out fields and controls."""
    output=root/'original-output';data=arrays(output/'data.npz');initial=arrays(output/'initial_arrays.npz');final=arrays(output/'final.npz')
    assert len(initial)==16 and sum(v.size for v in initial.values())==2115
    for split in ['fit','validation','test']:exact(edges(data[split+'_images']),data[split+'_edges'])
    records=np.genfromtxt(output/'batches.tsv',names=True,delimiter='\t',dtype=np.int64);assert len(records)==3000
    before=initial;first={k:np.zeros_like(v) for k,v in initial.items()};second={k:np.zeros_like(v) for k,v in initial.items()};iterations=[0,0]
    gradient_values=forward_values=0;gerror=ferror=0.;batch=0
    def compare(native,expected):
        nonlocal forward_values,ferror
        for key,value in expected.items():
            np.testing.assert_allclose(native[key],value,atol=1e-5,rtol=2e-4,err_msg=key);forward_values+=np.size(value);ferror=max(ferror,float(np.max(abs(native[key]-value))))
        if 'pixel_mmd' in native:np.testing.assert_allclose(native['pixel_mmd'],mmd(coarse(target),coarse(expected['fake'])),atol=3e-6,rtol=3e-4)
    for epoch in range(1,4):
        visits=records[records['epoch']==epoch];exact(np.sort(visits['row']),np.arange(1000));assert len(np.unique(visits['step']))==32
        for step in range(32):
            record=visits[visits['step']==batch];rows=record['row'];np.testing.assert_array_equal(record['source_id'],data['fit_source_ids'][rows]);assert len(rows)==(8 if step==31 else 32)
            target,source=[data['fit_'+k][rows].astype(float) for k in ['images','edges']]
            for group in [1,0]:
                phase=2*batch+(group==0);native=arrays(root/f'phases/phase{phase}-gradient.npz');after=arrays(root/f'phases/phase{phase}-after.npz')
                _,all_gradients=forward({k:v.astype(float) for k,v in before.items()},target,source);expected={k:v for k,v in all_gradients.items() if k[0]==str(group)};assert set(native)==set(expected)
                for key,value in expected.items():
                    np.testing.assert_allclose(native[key],value,atol=2e-5,rtol=3e-3,err_msg=f'phase{phase}/{key}');gradient_values+=value.size;gerror=max(gerror,float(np.max(abs(native[key]-value))))
                iterations[group]+=1;t=iterations[group]
                for key,value in before.items():
                    if key[0]!=str(group):exact(value,after[key]);continue
                    g=native[key];first[key]=first[key]*np.float32(.5)+g*np.float32(.5);second[key]=second[key]*np.float32(.999)+(g*g)*np.float32(.001)
                    mean=first[key]/np.float32(1-.5**t);variance=second[key]/np.float32(1-.999**t);direction=mean/(np.sqrt(variance)+np.float32(1e-8));exact(value-direction*np.float32(.0002),after[key])
                before=after
            batch+=1
        checkpoint=arrays(root/f'probe-epoch{epoch}.npz')
        for key,value in checkpoint.items():exact(before[key],value)
        for split in ['validation','test']:
            target,source=[data[split+'_'+k].astype(float) for k in ['images','edges']];out,_=forward({k:v.astype(float) for k,v in checkpoint.items()},target,source);compare(arrays(root/f'probe-epoch{epoch}-{split}.npz'),out)
    assert iterations==[96,96] and batch==96
    for key,value in final.items():exact(before[key],value)
    target,source=[data['test_'+k].astype(float) for k in ['images','edges']]
    for parameters,file in [(initial,'initial_outputs.npz'),(final,'outputs.npz')]:compare(arrays(output/file),forward({k:v.astype(float) for k,v in parameters.items()},target,source)[0])
    p={k:v.astype(float) for k,v in final.items()};first_batch=arrays(output/'first_batch.npz');out,gradients=forward(p,first_batch['target'].astype(float),first_batch['source'].astype(float))
    for group in [0,1]:
        native=arrays(output/f'final_gradients_group{group}.npz')
        for key,value in native.items():np.testing.assert_allclose(value,gradients[key],atol=2e-5,rtol=3e-3);gradient_values+=value.size;gerror=max(gerror,float(np.max(abs(value-gradients[key]))))
    controls=arrays(output/'condition_controls.npz');blank,_=generate(p,controls['zero_source'].astype(float));wrong,_=discriminate(p,controls['zero_source'].astype(float),target)
    for key,expected in [('zero_source_prediction',blank),('zero_source_real_logits',wrong)]:compare({key:controls[key]},{key:expected})
    baseline=arrays(output/'fit_mean_baseline.npz')['image'];np.testing.assert_allclose(baseline,np.broadcast_to(data['fit_images'].astype(float).mean(axis=0),baseline.shape),atol=2e-6,rtol=2e-5)
    # Final smooth finite differences use two saved paired cases, every named tensor.
    x,c=first_batch['target'][:2].astype(float),first_batch['source'][:2].astype(float);_,partials=forward(p,x,c);differences=0
    for key,gradient in partials.items():
        for index in sorted({0,gradient.size//2,gradient.size-1}):
            old=p[key].flat[index];h=1e-6;field='generator_loss' if key[0]=='0' else 'discriminator_loss';p[key].flat[index]=old+h;plus=forward(p,x,c)[0][field];p[key].flat[index]=old-h;minus=forward(p,x,c)[0][field];p[key].flat[index]=old
            np.testing.assert_allclose((plus-minus)/(2*h),gradient.flat[index],atol=5e-7,rtol=3e-3,err_msg=key);differences+=1
    return dict(passed=True,source_edge_pixels_exact=1097600,native_parameter_derivatives_checked=gradient_values,maximum_derivative_error=gerror,forward_values_checked=forward_values,maximum_forward_error=ferror,independent_final_scalar_finite_differences=differences,all_192_Adam_updates_bit_exact=True,all_192_phase_ownership_checks_pass=True,optimizer_iterations=iterations,batch_rows_checked=3000,conditional_generator_and_discriminator_controls_pass=True,paired_mean_L1_weight=100,patch_logits_per_image=49)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path('build/nn-report-conversion/pix2pix'));parser.add_argument('--json',type=Path);args=parser.parse_args();result=check(args.root);text=json.dumps(result,indent=2)+'\n';print(text,end='')
    if args.json:args.json.write_text(text)

if __name__=='__main__':main()
