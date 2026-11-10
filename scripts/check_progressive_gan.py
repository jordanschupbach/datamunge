#!/usr/bin/env python3
"""Independent NumPy progressive-GAN forward/adjoint and eight-Adam-state oracle.

Reads original C++ artifacts only; no fitting, preparation or downloads. Checks
all active group gradients before writes, every native Adam transition, stage
activity, population pixel RMS normalization, both real fade paths and source
half-pixel resize/noise banks, plus final smooth derivatives and controls.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact, gaussian
from check_big_gan import conv
from check_convolutional_autoencoder import pool, repeat
from check_contractive_autoencoder import sigmoid

EPSILON=float(np.float32(1e-8))


def active(stage,generator):
    """Source activity is independent of evaluating all old/new forward branches."""
    groups=[[0,1],[0,1,2,3],[0,2,3]][stage]
    return groups if generator else [k+4 for k in groups]


def accumulate(out,key,value):
    """Add shared real/fake and old/new branch contributions before any update."""
    out[key]=out.get(key,0)+value


def dense(p,group,name,x,tanh=False):
    """Biased dense layer with optional tanh and a closed-form adjoint."""
    key=f'{group}/{name}';w=p[key+'/W'];linear=x@w+p[key+'/b'];y=np.tanh(linear) if tanh else linear
    def backward(dy,out):
        delta=dy*(1-y*y) if tanh else dy
        accumulate(out,key+'/W',x.T@delta);accumulate(out,key+'/b',delta.sum(axis=0))
        return delta@w.T
    return y,backward


def convolution(p,group,name,x):
    """Biased SAME1x1/3x3 map using an independent shared-kernel adjoint."""
    key=f'{group}/{name}';y,bc=conv(x,p[key+'/W']);y+=p[key+'/b']
    def backward(dy,out):
        dx,dw=bc(dy);accumulate(out,key+'/W',dw);accumulate(out,key+'/b',dy.sum(axis=(0,1,2)));return dx
    return y,backward


def pixel_norm(x):
    """Divide by last-axis RMS with source-rounded epsilon, without centering/affine."""
    root=np.sqrt((x*x).mean(axis=-1,keepdims=True)+EPSILON);y=x/root
    def backward(dy):return (dy-y*(dy*y).mean(axis=-1,keepdims=True))/root
    return y,backward


def feature(p,group,name,x,normalize):
    """Normal-kernel biased convolution, LeakyReLU(.2), optional pixel RMS division."""
    pre,bc=convolution(p,group,name,x);h=np.where(pre>0,pre,.2*pre)
    if normalize:y,bn=pixel_norm(h)
    else:y,bn=h,lambda d:d
    def backward(dy,out):return bc(bn(dy)*np.where(pre>0,1.,.2),out)
    return y,backward


def generate(p,z,alpha):
    """Actual old4/new8 image paths with a shared core and post-sigmoid fade."""
    stem,bs=dense(p,0,'stem',z,True);normalized,bn=pixel_norm(stem.reshape(-1,4,4,4))
    core,bc=feature(p,0,'conv',normalized,True)
    old_logit,bo=convolution(p,1,'rgb',core);old=sigmoid(old_logit)
    first,b1=feature(p,2,'conv1',repeat(core),True)
    second,b2=feature(p,2,'conv2',first,True)
    new_logit,br=convolution(p,3,'rgb',second);new=sigmoid(new_logit)
    image=alpha*new+(1-alpha)*repeat(old)
    def backward(dy,out):
        dc=bo(4*pool(dy*(1-alpha))*old*(1-old),out)
        dn=br(dy*alpha*new*(1-new),out);dn=b2(dn,out);dn=b1(dn,out);dc+=4*pool(dn)
        return bs(bn(bc(dc,out)).reshape(-1,64),out)
    return dict(old4=old,new8=new,fake=image),backward


def discriminate(p,image,alpha,details=False):
    """Old pooled-image RGB versus new RGB/block/pool, mixed before the shared core."""
    low,bo=convolution(p,5,'rgb',pool(image));old=np.where(low>0,low,.2*low)
    high,bh=convolution(p,7,'rgb',image);high_active=np.where(high>0,high,.2*high)
    first,b1=feature(p,6,'conv1',high_active,False);second,b2=feature(p,6,'conv2',first,False)
    mixed=alpha*pool(second)+(1-alpha)*old;core,bc=feature(p,4,'conv',mixed,False)
    score,bs=dense(p,4,'score',core.reshape(-1,64))
    def backward(dy,out):
        dm=bc(bs(dy,out).reshape(-1,4,4,4),out)
        dx=repeat(bo(dm*(1-alpha)*np.where(low>0,1.,.2),out))/4
        dn=repeat(dm*alpha)/4;dn=b2(dn,out);dn=b1(dn,out)
        dx+=bh(dn*np.where(high>0,1.,.2),out)
        return dx
    if details:return dict(old_path=old,new_path=pool(second),mixed=mixed,base=core,score=score)
    return score,backward


def forward(p,x,z,alpha):
    """Seven original fields and source-scaled G/D partials for all eight groups."""
    generated,bg=generate(p,z,alpha);real,br=discriminate(p,x,alpha);fake,bf=discriminate(p,generated['fake'],alpha)
    n=len(x);out={**generated,'real_score':real,'fake_score':fake,
                 'd_loss':.5*(((real-1)**2).mean()+(fake*fake).mean()),'g_loss':((fake-1)**2).mean()}
    critic={};br((real-1)/n,critic);bf(fake/n,critic)
    dummy={};dx=bf(2*(fake-1)/n,dummy);generator={};bg(dx,generator)
    return out,{**generator,**critic}


def check(root):
    """Check all192 captured phases,576 exact Adam states and every held-out path."""
    data=arrays(root/'original-output/data.npz');initial=arrays(root/'original-output/initial_arrays.npz');final=arrays(root/'original-output/final.npz')
    assert len(initial)==24 and sum(v.size for v in initial.values())==2067
    for split in ['fit','validation','test']:
        source=data[split+'_original'].reshape(-1,28,28);resized=np.zeros((len(source),8,8,1))
        for y in range(8):
            for x in range(8):
                py,px=(y+.5)*3.5-.5,(x+.5)*3.5-.5;iy,ix=int(py),int(px)
                for dy in [0,1]:
                    for dx in [0,1]:resized[:,y,x,0]+=(py-iy if dy else 1-(py-iy))*(px-ix if dx else 1-(px-ix))*source[:,iy+dy,ix+dx].astype(float)
        exact(resized.astype(np.float32),data[split+'_images'])
    fixed=arrays(root/'original-output/fixed_noise.npz')
    for name,seed in [('validation_z',6481),('test_z',6491)]:exact(fixed[name],gaussian((200,16),seed).astype(np.float32))
    batches=np.genfromtxt(root/'original-output/batches.tsv',names=True,delimiter='\t',dtype=None,encoding=None)
    before=initial;moments={k:np.zeros_like(v) for k,v in initial.items()};variances={k:np.zeros_like(v) for k,v in initial.items()};iterations=[0]*8
    count=values=forward_values=0;max_error=0.;max_forward=0.
    for epoch in range(1,4):
        stage=epoch-1;noise=gaussian((1000,16),6503+epoch).astype(np.float32)
        subset=batches[batches['epoch']==epoch];assert sorted(subset['row'].tolist())==list(range(1000))
        for step in range(32):
            index=(epoch-1)*32+step;record=subset[subset['step']==index];rows=record['row'];alpha=0. if stage==0 else float(np.float32(step+1)/np.float32(32)) if stage==1 else 1.
            assert np.all(record['stage']==stage) and np.all(record['alpha']==alpha)
            np.testing.assert_array_equal(record['source_id'],data['fit_source_ids'][rows])
            x,z=data['fit_images'][rows].astype(float),noise[rows].astype(float)
            if step==0:
                out,_=forward({k:v.astype(float) for k,v in before.items()},x,z,alpha);native=arrays(root/f'probe-first-stage{stage}.npz')
                for key,value in out.items():
                    np.testing.assert_allclose(native[key],value,atol=8e-6,rtol=5e-4,err_msg=f'stage{stage}/{key}');forward_values+=np.size(value);max_forward=max(max_forward,float(np.max(abs(native[key]-value))))
            for generator in [False,True]:
                groups=active(stage,generator);tag=f'batch{index}'+('-g' if generator else '-d');native=arrays(root/'phases'/(tag+'-gradient.npz'));after=arrays(root/'phases'/(tag+'-after.npz'))
                _,expected=forward({k:v.astype(float) for k,v in before.items()},x,z,alpha)
                selected={k:v for k,v in expected.items() if int(k[0]) in groups};assert set(native)==set(selected)
                for key,value in selected.items():
                    np.testing.assert_allclose(native[key],value,atol=2e-5,rtol=2e-3,err_msg=tag+'/'+key);values+=value.size;max_error=max(max_error,float(np.max(abs(native[key]-value))))
                for group in groups:iterations[group]+=1;count+=1
                for key,value in before.items():
                    group=int(key[0])
                    if group not in groups:exact(value,after[key]);continue
                    g=native[key];t=iterations[group]
                    moments[key]=moments[key]*np.float32(.5)+g*np.float32(.5)
                    variances[key]=variances[key]*np.float32(.999)+(g*g)*np.float32(.001)
                    mean=moments[key]/np.float32(1-.5**t);variance=variances[key]/np.float32(1-.999**t)
                    direction=mean/(np.sqrt(variance)+np.float32(1e-8));exact(value-direction*np.float32(.0002),after[key])
                before=after
        checkpoint=arrays(root/f'probe-epoch{epoch}.npz')
        for key,value in before.items():exact(value,checkpoint[key])
        p={k:v.astype(float) for k,v in checkpoint.items()}
        for split in ['validation','test']:
            out,_=forward(p,data[split+'_images'].astype(float),fixed[split+'_z'].astype(float),0. if stage==0 else 1.)
            native=arrays(root/f'probe-epoch{epoch}-{split}.npz')
            for key,value in out.items():np.testing.assert_allclose(native[key],value,atol=8e-6,rtol=5e-4,err_msg=f'epoch{epoch}/{split}/{key}');forward_values+=np.size(value);max_forward=max(max_forward,float(np.max(abs(native[key]-value))))
    assert iterations==[96,64,64,64,96,64,64,64] and count==576
    for key in final:exact(final[key],before[key])
    p={k:v.astype(float) for k,v in final.items()};smooth=arrays(root/'probe-smooth-input.npz');_,expected=forward(p,smooth['x'].astype(float),smooth['z'].astype(float),.5);native=arrays(root/'original-output/smooth_gradients.npz')
    for key,value in expected.items():np.testing.assert_allclose(native[key],value,atol=2e-5,rtol=2e-3);values+=value.size;max_error=max(max_error,float(np.max(abs(native[key]-value))))
    controls=arrays(root/'original-output/fade_controls.npz')
    for k in range(5):
        out,_=generate(p,fixed['test_z'].astype(float),k*.25);np.testing.assert_allclose(controls['alpha'+str(k)],out['fake'],atol=4e-6,rtol=2e-5);forward_values+=out['fake'].size
    exact(controls['alpha4'],controls['new8']);exact(controls['alpha0'],repeat(controls['old4']))
    out,_=forward(p,data['test_images'].astype(float),fixed['test_z'].astype(float),1.);native=arrays(root/'original-output/outputs.npz')
    for key,value in out.items():np.testing.assert_allclose(native[key],value,atol=8e-6,rtol=5e-4);forward_values+=np.size(value)
    # Independently finite-difference every source-named tensor at representative entries.
    xx,zz=smooth['x'][:2].astype(float),smooth['z'][:2].astype(float);_,expected=forward(p,xx,zz,.5);differences=0
    for key,value in expected.items():
        for index in sorted(set([0,value.size//2,value.size-1])):
            old=p[key].flat[index];h=1e-6;field='g_loss' if int(key[0])<4 else 'd_loss'
            p[key].flat[index]=old+h;plus=forward(p,xx,zz,.5)[0][field]
            p[key].flat[index]=old-h;minus=forward(p,xx,zz,.5)[0][field];p[key].flat[index]=old
            np.testing.assert_allclose((plus-minus)/(2*h),value.flat[index],atol=2e-7,rtol=2e-3,err_msg=key);differences+=1
    return dict(passed=True,native_parameter_derivatives_checked=values,maximum_derivative_error=max_error,
                forward_values_checked=forward_values,maximum_forward_error=max_forward,independent_final_scalar_finite_differences=differences,
                all_576_Adam_updates_bit_exact=True,all_192_phase_ownership_checks_pass=True,optimizer_iterations=iterations,
                resized_pixels_exact=89600,batch_rows_checked=3000,fade_endpoints_bit_exact=True,generator_loss_has_no_half_factor=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path('build/nn-report-conversion/progressive_gan'));parser.add_argument('--json',type=Path)
    args=parser.parse_args();result=check(args.root);rendered=json.dumps(result,indent=2)+'\n';print(rendered,end='')
    if args.json:args.json.write_text(rendered)


if __name__=='__main__':main()
