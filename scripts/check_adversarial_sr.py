#!/usr/bin/env python3
"""Independent NumPy SRGAN/ESRGAN adjoints, frozen features, BN and reset-Adam audit.

Reads captured native arrays and prepared CIFAR/VGG inputs only; never prepares
data or fits a model. Preserves two source architectures, CRD shuffle, source
float-rounded constants, live relativistic means, batch versus saved BN paths,
pre-update moment captures and explicit warm-up-to-GAN optimizer resets.
"""
import argparse,json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays,exact
from check_big_gan import conv
from check_dcgan import down
from check_contractive_autoencoder import sigmoid

BN=['residual0/bn1','residual0/bn2','residual1/bn1','residual1/bn2','body_bn']


def cubic(x):
    x=np.abs(x);return np.where(x<=1,1.5*x*x*x-2.5*x*x+1,np.where(x<2,-.5*x*x*x+2.5*x*x-4*x+2,0))


def weights(length,out):
    scale=out/length;f=min(1,scale);margin=int(np.ceil(2/f))+1;position=np.arange(-margin,length+margin+1);center=(np.arange(out)+.5)/scale-.5;w=f*cubic(f*(center[:,None]-position[None,:]));w/=w.sum(axis=1,keepdims=True);index=position%(2*length);index=np.where(index<length,index,2*length-1-index);result=np.zeros((out,length))
    for i,j in enumerate(index):result[:,j]+=w[:,i]
    return result


def bicubic(x,oh,ow):
    x=x.astype(np.float32).astype(float);h=np.einsum('nhwc,ow->nhoc',x,weights(x.shape[2],ow)).astype(np.float32);return np.einsum('nhwc,oh->nowc',h.astype(float),weights(x.shape[1],oh)).astype(np.float32)


def add(out,key,value):out[key]=out.get(key,0)+value


def layer(p,group,name,x):
    prefix=f'{group}/{name}';w=p[prefix+'/W'];y,back=(down if w.shape[0]==4 else conv)(x,w)
    if prefix+'/b' in p:y+=p[prefix+'/b']
    def backward(dy,out):
        dx,dw=back(dy);add(out,prefix+'/W',dw)
        if prefix+'/b' in p:add(out,prefix+'/b',dy.sum(axis=(0,1,2)))
        return dx
    return y,backward


def prelu(p,name,x):
    a=p['0/'+name+'/alpha'];value=np.where(x>=0,x,x*a)
    def backward(dy,out):add(out,'0/'+name+'/alpha',np.where(x<0,dy*x,0).sum(axis=(0,1,2)));return dy*np.where(x>=0,1,a)
    return value,backward


def bn(p,name,x,training,moments):
    gamma=p['0/'+name+'/gamma'];beta=p['0/'+name+'/beta'];center=x-(x.mean(axis=(0,1,2)) if training else p['2/'+name+'/moving_mean']);var=(center*center).mean(axis=(0,1,2)) if training else p['2/'+name+'/moving_var'];root=np.sqrt(var+float(np.float32(1e-5)));normalized=center/root
    if training:
        count=float(np.float32(x.size)/np.float32(4));correction=float(np.float32(count)/np.float32(count-1));moments['moment/'+name+'/mean']=x.mean(axis=(0,1,2));moments['moment/'+name+'/unbiased_var']=var*correction
    def backward(dy,out):
        add(out,'0/'+name+'/gamma',(dy*normalized).sum(axis=(0,1,2)));add(out,'0/'+name+'/beta',dy.sum(axis=(0,1,2)));delta=dy*gamma
        if training:delta=delta-delta.mean(axis=(0,1,2))-center/(var+float(np.float32(1e-5)))*(delta*center).mean(axis=(0,1,2))
        return delta/root
    return normalized*gamma+beta,backward


def residual(p,name,x,training,moments):
    a,b1=layer(p,0,name+'/conv1',x);a,bb1=bn(p,name+'/bn1',a,training,moments);a,bp=prelu(p,name+'/prelu',a);a,b2=layer(p,0,name+'/conv2',a);a,bb2=bn(p,name+'/bn2',a,training,moments)
    def backward(dy,out):return dy+b1(bb1(bp(b2(bb2(dy,out),out),out),out),out)
    return x+a,backward


def dense_block(p,name,x):
    features=[x];backs=[];pre=[]
    for i in range(4):
        value,back=layer(p,0,name+'/conv'+str(i+1),np.concatenate(features,axis=3));pre.append(value);backs.append(back);features.append(np.where(value>0,value,.2*value))
    tail,b5=layer(p,0,name+'/conv5',np.concatenate(features,axis=3))
    def backward(dy,out):
        joined=b5(.2*dy,out);parts=[joined[:,:,:,:4]]+[joined[:,:,:,4+2*i:6+2*i] for i in range(4)]
        for i in range(3,-1,-1):
            d=backs[i](parts[i+1]*np.where(pre[i]>0,1.,.2),out);parts[0]=parts[0]+d[:,:,:,:4]
            for j in range(i):parts[j+1]=parts[j+1]+d[:,:,:,4+2*j:6+2*j]
        return dy+parts[0]
    return x+.2*tail,backward


def generate(p,lr,bic,enhanced,training):
    moments={};head,bhead=layer(p,0,'head',lr);backs=[]
    if enhanced:
        h=head
        for i in range(3):h,back=dense_block(p,'rrdb/dense'+str(i),h);backs.append(back)
        h,bt=layer(p,0,'body_tail',head+.2*h);h=head+h;bodyshape=h.shape;h=h.repeat(2,axis=1).repeat(2,axis=2);up,bu=layer(p,0,'up',h);up_active=np.where(up>0,up,.2*up);hr,bhr=layer(p,0,'hr',up_active);h=np.where(hr>0,hr,.2*hr);packed=shuffled=h
    else:
        head,bphead=prelu(p,'head_prelu',head);h=head
        for i in range(2):h,back=residual(p,'residual'+str(i),h,training,moments);backs.append(back)
        h,bt=layer(p,0,'body_tail',h);h,bbt=bn(p,'body_bn',h,training,moments);h=head+h;packed,bu=layer(p,0,'up',h);n,height,width,_=packed.shape;shuffled=packed.reshape(n,height,width,4,2,2).transpose(0,1,4,2,5,3).reshape(n,2*height,2*width,4);h,bpu=prelu(p,'up_prelu',shuffled)
    residual_image,bout=layer(p,0,'output',h);image=bic+residual_image
    def backward(dy,out):
        dh=bout(dy,out)
        if enhanced:
            dh=bhr(dh*np.where(hr>0,1.,.2),out);dh=bu(dh*np.where(up>0,1.,.2),out);n,h,w,c=bodyshape;dh=dh.reshape(n,h,2,w,2,c).sum(axis=(2,4));dhead=dh;dbody=bt(dh,out);dhead=dhead+dbody;dh=.2*dbody
            for back in reversed(backs):dh=back(dh,out)
            dhead+=dh
        else:
            n,height,width,_=packed.shape;dh=bpu(dh,out);dh=dh.reshape(n,height,2,width,2,4).transpose(0,1,3,5,2,4).reshape(packed.shape);dh=bu(dh,out);dhead=dh;dh=bt(bbt(dh,out),out)
            for back in reversed(backs):dh=back(dh,out)
            dhead=bphead(dhead+dh,out)
        return bhead(dhead,out)
    return dict(image=image,residual=residual_image,packed=packed,shuffled=shuffled,**moments),backward


def discriminate(p,image,enhanced):
    first,b1=layer(p,1,'head',image);h1=np.where(first>0,first,.2*first);second,b2=layer(p,1,'down16',h1);h2=np.where(second>0,second,.2*second);third,b3=layer(p,1,'down8',h2);h3=np.where(third>0,third,.2*third);pooled=h3.mean(axis=(1,2));score=pooled@p['1/score/W']
    if not enhanced:score+=p['1/score/b']
    def backward(dy,out):
        add(out,'1/score/W',pooled.T@dy)
        if not enhanced:add(out,'1/score/b',dy.sum(axis=0))
        dh=np.broadcast_to((dy@p['1/score/W'].T)[:,None,None,:]/64,h3.shape);dh=b3(dh*np.where(third>0,1.,.2),out);dh=b2(dh*np.where(second>0,1.,.2),out);return b1(dh*np.where(first>0,1.,.2),out)
    return score,backward


def relativistic(real,fake):
    rr=real-fake.mean();rf=fake-real.mean();a,b=sigmoid(rr),sigmoid(rf)
    out=dict(d_loss=.5*(np.logaddexp(0,-rr).mean()+np.logaddexp(0,rf).mean()),adversarial_loss=.5*(np.logaddexp(0,rr).mean()+np.logaddexp(0,-rf).mean()),relative_real=rr,relative_fake=rf)
    return out,(.5*((a-1)-b.mean())/real.size,.5*(b-(a-1).mean())/fake.size),(.5*(a-(b-1).mean())/real.size,.5*((b-1)-a.mean())/fake.size)


def extract(features,image):
    shifted=image[:,:,:,::-1]*255.-np.array([103.939,116.779,123.68],dtype=np.float32).astype(float);value,back=conv(shifted,features['block1_conv1_W']);value+=features['block1_conv1_b']
    def backward(dy):return (back(dy)[0]*255.)[:,:,:,::-1]
    return value,backward


def forward(p,lr,hr,bic,features,enhanced,adversarial,debug=True):
    inference,_=generate(p,lr,bic,enhanced,False);training,bg=generate(p,lr,bic,enhanced,True);real,br=discriminate(p,hr,enhanced);fake,bf=discriminate(p,training['image'],enhanced);fake_inference,_=discriminate(p,inference['image'],enhanced);pred,bv=extract(features,training['image']);truth,_=extract(features,hr);pf=(pred if enhanced else np.maximum(pred,0))/255.;tf=(truth if enhanced else np.maximum(truth,0))/255.;diff=training['image']-hr;fd=pf-tf;pixel=np.abs(diff).mean() if enhanced else (diff*diff).mean();content=np.abs(fd).mean() if enhanced else (fd*fd).mean()
    if enhanced:loss,dd,gg=relativistic(real,fake)
    else:loss=dict(d_loss=np.logaddexp(0,-real).mean()+np.logaddexp(0,fake).mean(),adversarial_loss=np.logaddexp(0,-fake).mean(),relative_real=real,relative_fake=fake);dd=((sigmoid(real)-1)/real.size,sigmoid(fake)/fake.size);gg=(np.zeros_like(real),(sigmoid(fake)-1)/fake.size)
    adv_coefficient=float(np.float32(.005 if enhanced else .001));pixel_coefficient=float(np.float32(.01)) if enhanced else 0.;gan_loss=content+adv_coefficient*loss['adversarial_loss'];gan_loss=gan_loss+pixel_coefficient*pixel if enhanced else gan_loss
    out=dict(prediction=inference['image'],prediction_training=training['image'],residual=inference['residual'],real_score=real,fake_score=fake,fake_inference_score=fake_inference,**loss,g_loss=gan_loss if adversarial else pixel,gan_loss=gan_loss,pixel_loss=pixel,content_loss=content)
    if debug:out.update(prediction_feature=pf,target_feature=tf,packed=training['packed'],shuffled=training['shuffled']);out.update({k:v for k,v in training.items() if k.startswith('moment/')})
    d={};br(dd[0],d);bf(dd[1],d);g={};pixel_dy=np.sign(diff)/diff.size if enhanced else 2*diff/diff.size
    if adversarial:
        dy=bf(adv_coefficient*gg[1],{})+pixel_coefficient*pixel_dy;feature_dy=np.sign(fd)/fd.size if enhanced else 2*fd/fd.size;feature_dy=feature_dy/255.;feature_dy=feature_dy if enhanced else feature_dy*(pred>0);dy+=bv(feature_dy)
    else:dy=pixel_dy
    bg(dy,g);g.update({k:v for k,v in training.items() if k.startswith('moment/')});return out,{**g,**d}


def luma(x):return 16/255.+(65.481*x[:,:,:,:1]+128.553*x[:,:,:,1:2]+24.966*x[:,:,:,2:3])/255.


def ssim(x,y):
    offset=np.arange(-5,6);k=np.exp(-offset*offset/4.5);k/=k.sum();kernel=k[:,None]*k[None,:]
    def avg(z):return np.einsum('nhwcdk,dk->nhwc',np.lib.stride_tricks.sliding_window_view(z,(11,11),axis=(1,2)),kernel)
    mx,my=avg(x),avg(y);vx=np.maximum(avg(x*x)-mx*mx,0);vy=np.maximum(avg(y*y)-my*my,0);cov=avg(x*y)-mx*my
    return (((2*mx*my+.0001)*(2*cov+.0009))/((mx*mx+my*my+.0001)*(vx+vy+.0009))).mean(axis=(1,2,3))


def check(root):
    source=root/'original-output';data=arrays(source/'data.npz');features={k:v.astype(float) for k,v in arrays(source/'frozen_features.npz').items()};pins=json.loads((root/'baseline.json').read_text());raw={name:np.fromfile(Path(pins['data_directory'])/name,np.uint8).reshape(10000,3073) for name in ['data_batch_1.bin','test_batch.bin']};pixel_checks=0
    for split,file,start,count in [('fit','data_batch_1.bin',0,128),('validation','data_batch_1.bin',128,32),('test','test_batch.bin',64,32)]:
        source_rows=raw[file][start:start+count];hr=source_rows[:,1:].reshape(count,3,32,32).transpose(0,2,3,1).astype(np.float32)/np.float32(255);lr=bicubic(hr,16,16);bi=bicubic(lr,32,32)
        for field,value in [('hr',hr),('lr',lr),('bicubic',bi)]:exact(data[split+'_'+field],value);pixel_checks+=value.size
        exact(data[split+'_labels'],source_rows[:,0].astype(np.int32));exact(data[split+'_source_ids'],np.arange(start+(50000 if split=='test' else 0),start+(50000 if split=='test' else 0)+count,dtype=np.int32))
    full_features=arrays(Path(pins['feature_file']));assert sum(v.size for v in features.values())==1792
    for key,value in arrays(source/'frozen_features.npz').items():exact(value,full_features[key])
    values=forwards=0;gerror=ferror=0.;all_updates=0;differences=0;results={}
    def compare(native,expected,label):
        nonlocal forwards,ferror
        assert set(native)==set(expected),(label,set(native)^set(expected))
        for key,value in expected.items():np.testing.assert_allclose(native[key],value,atol=2e-5,rtol=8e-4,err_msg=label+'/'+key);forwards+=np.size(value);ferror=max(ferror,float(np.max(abs(native[key]-value))))
    def compare_grad(native,expected,label):
        nonlocal values,gerror
        assert set(native)==set(expected),(label,set(native)^set(expected))
        for key,value in expected.items():np.testing.assert_allclose(native[key],value,atol=6e-5,rtol=6e-3,err_msg=label+'/'+key);values+=value.size;gerror=max(gerror,float(np.max(abs(native[key]-value))))
    for name,enhanced in [('srgan',False),('esrgan',True)]:
        folder=source/name;probe=root/name;initial=arrays(folder/'initial_arrays.npz');final=arrays(folder/'final.npz');before=initial;records=np.genfromtxt(folder/'batches.tsv',names=True,delimiter='\t',dtype=np.int64);assert len(records)==384
        moment={k:np.zeros_like(v) for k,v in initial.items() if k[0] in '01'};variance={k:np.zeros_like(v) for k,v in moment.items()};iterations=[0,0];updates=[0,0];wrong_reset=0.;wrong_bn=0.
        for epoch in range(1,4):
            phase=epoch>=2;visits=records[records['epoch']==epoch];exact(np.sort(visits['row']),np.arange(128))
            if epoch==2:
                exact(arrays(probe/'probe-reset-before.npz')['iterations'],np.array([16,0],np.int64));exact(arrays(probe/'probe-reset-after.npz')['iterations'],np.array([0,0],np.int64));saved=arrays(probe/'probe-reset-weights.npz')
                for key,value in before.items():exact(saved[key],value)
                old_moment={k:v.copy() for k,v in moment.items()};old_variance={k:v.copy() for k,v in variance.items()};moment={k:np.zeros_like(v) for k,v in moment.items()};variance={k:np.zeros_like(v) for k,v in variance.items()};iterations=[0,0]
            for step in range(16):
                batch=(epoch-1)*16+step;rows=visits[visits['step']==batch]['row'];assert len(rows)==8;lr,hr,bi=[data['fit_'+k][rows].astype(float) for k in ['lr','hr','bicubic']]
                if step==0 and epoch<=2:
                    prefix='warm' if epoch==1 else 'gan';input=arrays(probe/f'probe-{prefix}-input.npz');exact(input['lr'],data['fit_lr'][rows]);exact(input['hr'],data['fit_hr'][rows]);compare(arrays(probe/f'probe-{prefix}-forward.npz'),forward({k:v.astype(float) for k,v in before.items()},lr,hr,bi,features,enhanced,phase)[0],name+'/'+prefix)
                for group in ([1,0] if phase else [0]):
                    label=f'epoch{epoch}-batch{step}-'+('d' if group else 'g');native=arrays(probe/'phases'/(label+'-gradient.npz'));after=arrays(probe/'phases'/(label+'-after.npz'));expected=forward({k:v.astype(float) for k,v in before.items()},lr,hr,bi,features,enhanced,phase)[1];compare_grad(native,{k:v for k,v in expected.items() if k[0]==str(group) or group==0 and k.startswith('moment/')},name+'/'+label)
                    iterations[group]+=1;updates[group]+=1;t=iterations[group]
                    for key,value in before.items():
                        if key[0]==str(group):
                            g=native[key];moment[key]=moment[key]*np.float32(.9)+g*np.float32(.1);variance[key]=variance[key]*np.float32(.999)+(g*g)*np.float32(.001);mean=moment[key]/np.float32(1-.9**t);var=variance[key]/np.float32(1-.999**t);direction=mean/(np.sqrt(var)+np.float32(1e-8));exact(value-direction*np.float32(.0001),after[key])
                            if epoch==2 and step==0 and group==0:
                                oldt=17;om=old_moment[key]*np.float32(.9)+g*np.float32(.1);ov=old_variance[key]*np.float32(.999)+(g*g)*np.float32(.001);candidate=value-(om/np.float32(1-.9**oldt))/(np.sqrt(ov/np.float32(1-.999**oldt))+np.float32(1e-8))*np.float32(.0001);wrong_reset=max(wrong_reset,float(abs(candidate-after[key]).max()))
                        elif group==0 and key[0]=='2':
                            tail=key[2:];base,field=tail.rsplit('/',1);captured=native['moment/'+base+('/mean' if field=='moving_mean' else '/unbiased_var')];exact(value*np.float32(.9)+captured*np.float32(1-np.float32(.9)),after[key])
                        else:exact(value,after[key])
                    before=after;all_updates+=1;exact(arrays(probe/'phases'/(label+'-iterations.npz'))['iterations'],np.array(iterations,np.int64))
                if epoch==1 and step==0 and not enhanced:
                    recomputed=arrays(probe/'probe-poststep-recomputed-moments.npz');new_mom=forward({k:v.astype(float) for k,v in before.items()},lr,hr,bi,features,False,False)[0]
                    for key,value in recomputed.items():np.testing.assert_allclose(value,new_mom[key],atol=2e-5,rtol=8e-4)
                    captured=arrays(probe/'phases/epoch1-batch0-g-gradient.npz');wrong_bn=max(float(abs(recomputed[k]-captured[k]).max()) for k in recomputed);assert wrong_bn>1e-8
            checkpoint=arrays(probe/f'probe-epoch{epoch}.npz')
            for key,value in checkpoint.items():exact(value,before[key])
            for split in ['validation','test']:
                lr,hr,bi=[data[split+'_'+k].astype(float) for k in ['lr','hr','bicubic']];compare(arrays(probe/f'probe-epoch{epoch}-{split}.npz'),forward({k:v.astype(float) for k,v in before.items()},lr,hr,bi,features,enhanced,phase,False)[0],name+'/'+split)
        assert iterations==[32,32] and updates==[48,32]
        for key,value in before.items():exact(value,final[key])
        test=[data['test_'+k].astype(float) for k in ['lr','hr','bicubic']]
        for p,file,phase in [(initial,'initial_outputs.npz',False),(final,'outputs.npz',True)]:compare(arrays(folder/file),forward({k:v.astype(float) for k,v in p.items()},*test,features,enhanced,phase,False)[0],name+'/'+file)
        p={k:v.astype(float) for k,v in final.items()};first=arrays(probe/'probe-gan-input.npz');lr,hr=first['lr'].astype(float),first['hr'].astype(float);bi=bicubic(first['lr'],32,32).astype(float);out,partials=forward(p,lr,hr,bi,features,enhanced,True,False);compare(arrays(folder/'final_first_forward.npz'),out,name+'/final-first')
        for group in [0,1]:compare_grad(arrays(folder/f'final_gradients_group{group}.npz'),{k:v for k,v in partials.items() if k[0]==str(group) or group==0 and k.startswith('moment/')},name+'/final')
        canary=arrays(folder/'target_canary.npz');out,_=forward(p,test[0],np.zeros_like(test[1]),test[2],features,enhanced,True,False);compare(canary,{k:out[k] for k in canary},name+'/canary');testout=arrays(folder/'outputs.npz');exact(canary['prediction'],testout['prediction']);exact(canary['prediction_training'],testout['prediction_training'])
        meta=json.loads((folder/'metrics.json').read_text());prediction=np.clip(luma(testout['prediction'][:,6:26,6:26].astype(float)),0,1);truth=luma(data['test_hr'][:,6:26,6:26].astype(float));baseline=np.clip(luma(data['test_bicubic'][:,6:26,6:26].astype(float)),0,1);mse=((prediction-truth)**2).mean(axis=(1,2,3));base_mse=((baseline-truth)**2).mean(axis=(1,2,3));metrics=dict(test_mean_psnr=(-10*np.log10(mse)).mean(),test_pooled_psnr=-10*np.log10(mse.mean()),bicubic_mean_psnr=(-10*np.log10(base_mse)).mean(),test_mean_ssim=ssim(prediction,truth).mean(),bicubic_mean_ssim=ssim(baseline,truth).mean())
        for key,value in metrics.items():np.testing.assert_allclose(meta[key],value,atol=2e-10,rtol=2e-10,err_msg=key)
        for key in ['g_loss','gan_loss','pixel_loss','content_loss','adversarial_loss','d_loss']:assert meta['test_'+key]==testout[key].item()
        history=np.genfromtxt(folder/'history.tsv',names=True,delimiter='\t')
        for row in history:
            a=arrays(probe/f"probe-epoch{int(row['epoch'])}-validation.npz")
            for key in history.dtype.names[2:]:assert row[key]==a[key].item()
        # Two small final cases cover every named learned leaf, never saved BN state.
        lr,hr,bi=lr[:2],hr[:2],bi[:2];partials=forward(p,lr,hr,bi,features,enhanced,True)[1];local=0
        for key,g in partials.items():
            if key.startswith('moment/'):continue
            for index in sorted({0,g.size//2,g.size-1}):
                old=p[key].flat[index];h=1e-6;field='g_loss' if key[0]=='0' else 'd_loss';p[key].flat[index]=old+h;plus=forward(p,lr,hr,bi,features,enhanced,True,False)[0][field];p[key].flat[index]=old-h;minus=forward(p,lr,hr,bi,features,enhanced,True,False)[0][field];p[key].flat[index]=old;np.testing.assert_allclose((plus-minus)/(2*h),g.flat[index],atol=4e-6,rtol=1e-2,err_msg=key);local+=1
        differences+=local;assert wrong_reset>1e-8;results[name]=dict(optimizer_iterations=iterations,generator_writes=48,critic_writes=32,reset_counters_exact=True,warmup_critic_unchanged=True,wrong_no_reset_weight_max_difference=wrong_reset,poststep_recomputed_BN_moment_max_difference=wrong_bn,final_scalar_finite_differences=local,independent_metrics=metrics)
    assert all_updates==160
    algebra=arrays(source/'relativistic_algebra.npz');out,dd,gg=relativistic(algebra['real'].astype(float),algebra['fake'].astype(float));compare({k:algebra[k] for k in ['d_loss','g_loss','relative_real','relative_fake']},{'d_loss':out['d_loss'],'g_loss':out['adversarial_loss'],'relative_real':out['relative_real'],'relative_fake':out['relative_fake']},'algebra')
    for key,value in zip(['d_real_gradient','d_fake_gradient','g_real_gradient','g_fake_gradient'],[*dd,*gg]):np.testing.assert_allclose(algebra[key],value,atol=1e-7,rtol=1e-6)
    return dict(passed=True,variants=results,native_parameter_derivatives_and_moments_checked=values,maximum_derivative_or_moment_error=gerror,forward_values_checked=forwards,maximum_forward_error=ferror,independent_final_scalar_finite_differences=differences,all_160_Adam_updates_bit_exact=True,all_phase_ownership_and_BN_writes_exact=True,source_HR_LR_bicubic_pixels_exact=int(pixel_checks),all_768_batch_row_visits_checked=True,frozen_1792_values_and_target_stops_checked=True,all_validation_and_test_scalar_records_checked=True,unequal_pool_relativistic_algebra_checked=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path('build/nn-report-conversion/srgan_esrgan'));parser.add_argument('--json',type=Path);a=parser.parse_args();r=check(a.root);s=json.dumps(r,indent=2)+'\n';print(s,end='')
    if a.json:a.json.write_text(s)
if __name__=='__main__':main()
