#!/usr/bin/env python3
"""Independent NumPy DCGAN forward/adjoint checks; reads native artifacts only.

No preparation or fitting. Implements stride-two 4x4 convolutions and transposed
convolutions, affine BN, both phase objectives, all parameter adjoints, source
moving moments, first Adam writes, held-out/noise predictions and row ownership.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact, gaussian


def down(x, w):
    """NHWC 4x4 SAME cross-correlation, stride two and one pixel each side."""
    n, h, width, _ = x.shape
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    y = np.zeros((n, h//2, width//2, w.shape[-1]))
    for a in range(4):
        for b in range(4):
            y += np.einsum('nhwi,io->nhwo', padded[:, a:a+h:2, b:b+width:2], w[a, b])

    def backward(dy):
        dx, dw = np.zeros_like(padded), np.zeros_like(w)
        for a in range(4):
            for b in range(4):
                window = padded[:, a:a+h:2, b:b+width:2]
                dw[a, b] = np.einsum('nhwi,nhwo->io', window, dy)
                dx[:, a:a+h:2, b:b+width:2] += np.einsum('nhwo,io->nhwi', dy, w[a, b])
        return dx[:, 1:-1, 1:-1], dw
    return y, backward


def up(x, w):
    """Scatter transposed convolution with [4,4,out,in] kernel and cropped border."""
    n, h, width, _ = x.shape
    padded = np.zeros((n, 2*h+2, 2*width+2, w.shape[2]))
    for a in range(4):
        for b in range(4):
            padded[:, a:a+2*h:2, b:b+2*width:2] += np.einsum('nhwi,oi->nhwo', x, w[a, b])

    def backward(dy):
        gradient = np.pad(dy, ((0, 0), (1, 1), (1, 1), (0, 0)))
        dx, dw = np.zeros_like(x), np.zeros_like(w)
        for a in range(4):
            for b in range(4):
                window = gradient[:, a:a+2*h:2, b:b+2*width:2]
                dw[a, b] = np.einsum('nhwo,nhwi->oi', window, x)
                dx += np.einsum('nhwo,oi->nhwi', window, w[a, b])
        return dx, dw
    return padded[:, 1:-1, 1:-1], backward


def bn(p, group, state, name, x, training):
    """Population normalization, sample-variance captures; independent analytic adjoint."""
    axes = tuple(range(x.ndim-1))
    gamma, beta = p[group+name+'/gamma'], p[group+name+'/beta']
    if training:
        mu = x.mean(axis=axes)
        variance = ((x-mu)**2).mean(axis=axes)
        count = x.size//x.shape[-1]
        moments = {name+'/batch_mean': mu, name+'/batch_var': variance*count/(count-1) if count > 1 else variance}
    else:
        mu, variance = p[state+name+'/moving_mean'], p[state+name+'/moving_var']
        moments = {}
    inv = 1/np.sqrt(variance+.001)
    normalized = (x-mu)*inv

    def backward(dy):
        dg, db = (dy*normalized).sum(axis=axes), dy.sum(axis=axes)
        if training:
            dx = gamma*inv*(dy-dy.mean(axis=axes)-normalized*(dy*normalized).mean(axis=axes))
        else:
            dx = dy*gamma*inv
        return dx, {group+name+'/gamma': dg, group+name+'/beta': db}
    return normalized*gamma+beta, moments, backward


def generate(p, z, training):
    """Projection, two learned affine BNs, two transposed maps and signed tanh pixels."""
    stem = (z@p['0/projection/W']).reshape(-1, 7, 7, 8)
    normalized, m1, b1 = bn(p, '0/', 'g_state/', 'stem_bn', stem, training)
    relu = np.maximum(normalized, 0)
    first, u1 = up(relu, p['0/up14/W'])
    second, m2, b2 = bn(p, '0/', 'g_state/', 'up14_bn', first, training)
    hidden = np.maximum(second, 0)
    logits, u2 = up(hidden, p['0/output/W'])
    image = np.tanh(logits+p['0/output/b'])

    def backward(gradient):
        delta = gradient*(1-image**2)
        result = {'0/output/b': delta.sum(axis=(0, 1, 2))}
        delta, result['0/output/W'] = u2(delta)
        delta, affine = b2(delta*(second > 0)); result.update(affine)
        delta, result['0/up14/W'] = u1(delta)
        delta, affine = b1(delta*(normalized > 0)); result.update(affine)
        result['0/projection/W'] = z.T@delta.reshape(-1, 392)
        return result
    return image, {**m1, **m2}, backward


def discriminate(p, image, training):
    """Two strided maps, one BN, LeakyReLU(.2), flattened raw score."""
    first, d1 = down(image, p['1/down14/W']); first += p['1/down14/b']
    hidden = np.where(first >= 0, first, .2*first)
    second, d2 = down(hidden, p['1/down7/W'])
    normalized, moments, back = bn(p, '1/', 'd_state/', 'down7_bn', second, training)
    features = np.where(normalized >= 0, normalized, .2*normalized)
    flat = features.reshape(-1, 392)
    logit = flat@p['1/output/W']+p['1/output/b']

    def backward(gradient):
        result = {'1/output/W': flat.T@gradient, '1/output/b': gradient.sum(axis=0)}
        delta = (gradient@p['1/output/W'].T).reshape(features.shape)
        delta, affine = back(delta*np.where(normalized >= 0, 1., .2)); result.update(affine)
        delta, result['1/down7/W'] = d2(delta)
        delta *= np.where(first >= 0, 1., .2)
        result['1/down14/b'] = delta.sum(axis=(0, 1, 2))
        delta, result['1/down14/W'] = d1(delta)
        return delta, result
    return logit, moments, backward


def training(p, x, z):
    """Separate joint-D and inference-D-for-G paths, with no optimizer or moving write."""
    fake, gm, back_g = generate(p, z, True)
    logits, dm, back_d = discriminate(p, np.concatenate((2*x-1, fake)), True)
    for_g, _, back_dg = discriminate(p, fake, False)
    n = len(x)
    out = {'d_train_logits': logits, 'd_for_g_logits': for_g, 'g_train_signed': fake,
           'discriminator_loss': np.logaddexp(0, -logits[:n]).mean()+np.logaddexp(0, logits[n:]).mean(),
           'generator_loss': np.logaddexp(0, -for_g).mean(),
           **{'g/'+k: v for k, v in gm.items()}, **{'d/'+k: v for k, v in dm.items()}}
    delta_d = 1/(1+np.exp(-logits))/n; delta_d[:n] -= 1/n
    _, dd = back_d(delta_d)
    delta_g = -1/(1+np.exp(for_g))/n
    dg, _ = back_dg(delta_g)
    return out, back_g(dg), dd


def inference(p, x, z):
    """Saved-BN generation and raw D scores of real/fake rows, without cross-row coupling."""
    signed, _, _ = generate(p, z, False)
    logits, _, _ = discriminate(p, np.concatenate((2*x-1, signed)), False)
    n = len(x); real, fake = logits[:n], logits[n:]
    return dict(fake=((signed+1)*.5).reshape(n, 784), real_logit=real, fake_logit=fake,
                real_probability=1/(1+np.exp(-real)), fake_probability=1/(1+np.exp(-fake)),
                discriminator_loss=np.logaddexp(0, -real).mean()+np.logaddexp(0, fake).mean(),
                generator_loss=np.logaddexp(0, -fake).mean())


def check(root, data):
    """Inspect frozen C++ archives, analytic derivatives and finite-difference checks."""
    load=lambda name: arrays(root/(name+'.npz'))
    floating=lambda p: {k: v.astype(float) for k, v in p.items()}
    initial, final = load('original-output/initial_arrays'), load('original-output/final')
    raw = arrays(data)
    for split in ['fit','validation','test']:
        raw[split+'_images'] = (raw[split+'_images'].astype(np.float32)/np.float32(255)).astype(float)
    batches = np.genfromtxt(root/'original-output/batches.tsv', names=True, delimiter='\t', dtype=np.int64)
    first_rows = batches[batches['step'] == 0]['row']
    x = raw['fit_images'][first_rows].astype(float)
    z = gaussian((1000, 16), 4862).astype(np.float32).astype(float)[first_rows]
    errors=[]; forward_values=gradient_values=0
    for label, p, xx, zz, archive in [
        ('initial', initial, x, z, 'probe-first'),
        ('after-d', load('probe-after-d'), x, z, 'probe-after-d-forward'),
        ('final', final, raw['test_images'][:32].astype(float), gaussian((200,16),4831).astype(np.float32).astype(float)[:32], 'probe-final-training')]:
        out, gg, gd = training(floating(p), xx, zz)
        saved=load(archive)
        assert set(out)==set(saved),(set(out)^set(saved))
        for key,value in out.items():
            np.testing.assert_allclose(saved[key], value, atol=8e-6, rtol=2e-4, err_msg=label+'/'+key)
            forward_values+=np.size(value); errors.append(float(np.max(abs(saved[key]-value))))
        for group, expected in [(0, gg), (1, gd)]:
            if label=='after-d' and group==1: continue
            actual=load('probe-'+label+'-gradient-'+str(group))
            assert set(actual)==set(expected)
            for key,value in expected.items():
                np.testing.assert_allclose(actual[key],value,atol=8e-6,rtol=2e-3,err_msg=label+'/'+key)
                gradient_values+=value.size; errors.append(float(np.max(abs(actual[key]-value))))
    before=initial
    for group, phase, captures in [(1,'after-d',load('probe-first')),(0,'after-g',load('probe-after-d-forward'))]:
        after=load('probe-'+phase)
        gradients=load('probe-initial-gradient-1' if group else 'probe-after-d-gradient-0')
        changed=set()
        for key,g in gradients.items():
            first=g*np.float32(.5);second=(g*g)*np.float32(.001)
            direction=(first/np.float32(.5))/(np.sqrt(second/np.float32(.001))+np.float32(1e-8))
            exact(before[key]-np.float32(.0002)*direction,after[key]);changed.add(key)
        for layer in (['down7_bn'] if group else ['stem_bn','up14_bn']):
            state='d_state/' if group else 'g_state/';capture='d/' if group else 'g/'
            for suffix,field in [('moving_mean','batch_mean'),('moving_var','batch_var')]:
                key=state+layer+'/'+suffix
                expected=before[key]*np.float32(.9)+captures[capture+layer+'/'+field]*np.float32(1-.9)
                exact(expected,after[key]);changed.add(key)
        for key in before:
            if key not in changed: exact(before[key],after[key])
        before=after
    output=load('original-output/outputs');pred=inference(floating(final),raw['test_images'].astype(float),gaussian((200,16),4831).astype(np.float32).astype(float))
    for key,value in pred.items():
        np.testing.assert_allclose(output[key],value,atol=6e-6,rtol=3e-5,err_msg=key)
        forward_values+=np.size(value);errors.append(float(np.max(abs(output[key]-value))))
    controls=load('probe-noise-controls')
    for name,zz in [('changed',gaussian((200,16),4877).astype(np.float32).astype(float)),('zero',np.zeros((200,16)))]:
        image,_,_=generate(floating(final),zz,False);image=((image+1)*.5).reshape(200,784)
        np.testing.assert_allclose(controls[name],image,atol=3e-6,rtol=2e-5)
        forward_values+=image.size
    for epoch in range(1,4):
        subset=batches[batches['epoch']==epoch];assert len(subset)==1000 and sorted(subset['row'].tolist())==list(range(1000))
        np.testing.assert_array_equal(subset['source_id'],raw['fit_source_ids'][subset['row']].reshape(-1))
    def coarse(image):
        return image.reshape(len(image),4,7,4,7).mean(axis=(2,4)).reshape(len(image),16)/4
    def mmd(a,b):
        def kernel(x,y):
            distance=((x[:,None,:]-y[None,:,:])**2).sum(axis=2)
            return sum(c/(c+distance) for c in [1.,4.,16.])/3
        aa,bb,ab=kernel(a,a),kernel(b,b),kernel(a,b)
        return (aa.sum()-len(a))/(len(a)*(len(a)-1))+(bb.sum()-len(b))/(len(b)*(len(b)-1))-2*ab.mean()
    test=coarse(raw['test_images']);generated=coarse(pred['fake'])
    mean_image=np.broadcast_to(raw['fit_images'].mean(axis=0),(200,28,28,1))
    discrepancy=mmd(test,generated);mean_discrepancy=mmd(test,coarse(mean_image))
    np.testing.assert_allclose(output['pixel_mmd'],discrepancy,atol=8e-7,rtol=1e-5)
    metrics=json.loads((root/'original-output/metrics.json').read_text())
    np.testing.assert_allclose(metrics['fit_mean_coarse_pixel_mmd'],mean_discrepancy,atol=8e-7,rtol=1e-5)
    for field,key in [('test_discriminator_loss','discriminator_loss'),('test_generator_loss','generator_loss')]:
        np.testing.assert_allclose(metrics[field],pred[key],atol=6e-6,rtol=3e-5)
    assert metrics['test_discriminator_real_accuracy']==float((pred['real_logit']>=0).mean())
    assert metrics['test_discriminator_fake_accuracy']==float((pred['fake_logit']<0).mean())
    np.testing.assert_allclose(metrics['generated_mean_pixel'],pred['fake'].mean(),atol=2e-6)
    np.testing.assert_allclose(metrics['generated_mean_per_pixel_variance'],((pred['fake']-pred['fake'].mean(axis=0))**2).mean(),atol=2e-7)
    validation=inference(floating(final),raw['validation_images'],gaussian((200,16),4823).astype(np.float32).astype(float))
    history=np.genfromtxt(root/'original-output/history.tsv',names=True,delimiter='\t')
    assert history['epoch'].tolist()==[1,2,3]
    np.testing.assert_allclose(history['validation_discriminator_loss'][-1],validation['discriminator_loss'],atol=6e-6,rtol=3e-5)
    np.testing.assert_allclose(history['validation_generator_loss'][-1],validation['generator_loss'],atol=6e-6,rtol=3e-5)
    np.testing.assert_allclose(history['validation_coarse_pixel_mmd'][-1],mmd(coarse(raw['validation_images']),coarse(validation['fake'])),atol=8e-7,rtol=1e-5)
    algebra=load('original-output/logistic_algebra');rr,ff=algebra['real'].astype(float),algebra['fake'].astype(float)
    np.testing.assert_array_equal(rr.ravel(),[-80,-2,2,80]);np.testing.assert_array_equal(ff.ravel(),[80,2,-2,-80])
    for key,value in {'discriminator_loss':np.logaddexp(0,-rr).mean()+np.logaddexp(0,ff).mean(),'generator_loss':np.logaddexp(0,-ff).mean(),'d_real_gradient':-1/(1+np.exp(rr))/4,'d_fake_gradient':1/(1+np.exp(-ff))/4,'g_fake_gradient':-1/(1+np.exp(ff))/4}.items():
        np.testing.assert_allclose(algebra[key],value,atol=2e-6,rtol=2e-6)
    ids=[arrays(data)[split+'_source_ids'] for split in ['fit','validation','test']]
    assert len(np.unique(np.concatenate(ids)))==1400 and np.all(ids[0]<60000) and np.all(ids[1]<60000) and np.all(ids[2]>=60000)
    p=floating(final);xx=raw['test_images'][:2].astype(float);zz=gaussian((2,16),4831).astype(np.float32).astype(float)
    _,gg,gd=training(p,xx,zz);differences=0
    for group,expected in [(0,gg),(1,gd)]:
        for key,value in expected.items():
            for index in sorted(set([0,value.size//2,value.size-1])):
                old=p[key].flat[index];step=1e-6
                p[key].flat[index]=old+step;plus=training(p,xx,zz)[0]['generator_loss' if group==0 else 'discriminator_loss']
                p[key].flat[index]=old-step;minus=training(p,xx,zz)[0]['generator_loss' if group==0 else 'discriminator_loss']
                p[key].flat[index]=old
                np.testing.assert_allclose((plus-minus)/(2*step),value.flat[index],atol=2e-6,rtol=2e-4,err_msg=key)
                differences+=1
    return dict(passed=True,forward_values_checked=forward_values,native_parameter_derivatives_checked=gradient_values,
                maximum_checked_error=max(errors),independent_final_scalar_finite_differences=differences,
                first_Adam_writes_bit_exact=True,first_moving_updates_bit_exact=True,phase_ownership=True,batch_rows_checked=len(batches),
                joint_discriminator_BN_and_inference_discriminator_for_G=True,independent_coarse_MMD=discrepancy,independent_fit_mean_coarse_MMD=mean_discrepancy,final_validation_and_test_metrics_checked=True,extreme_logit_loss_and_gradient_values_checked=22)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('build/nn-report-conversion/dcgan'))
    parser.add_argument('--data',type=Path,default=Path('build/nn-report-conversion/datasets/saliency/saliency_data.npz'))
    parser.add_argument('--json',type=Path)
    args=parser.parse_args();result=check(args.root,args.data);rendered=json.dumps(result,indent=2)+'\n';print(rendered,end='')
    if args.json: args.json.write_text(rendered)


if __name__=='__main__': main()
