#!/usr/bin/env python3
"""Independent NumPy dual-stream Gated PixelCNN, all updates and class controls.

Reads pinned unchanged original records and captured partials. No replacement
fit, data preparation, framework runtime or native model math is used.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from check_adversarial_autoencoder import arrays, exact
from check_contractive_autoencoder import sigmoid
from check_pixel_cnn import network as plain_network, uniforms


def mask(shape, vertical, first):
    out = np.zeros(shape)
    if vertical:
        out[0] = 1
        if not first:
            out[1] = 1
    else:
        out[:, 0] = 1
        if not first:
            out[:, 1] = 1
    return out


def conv(x, w):
    """Independent odd rectangular SAME convolution and analytic two-input VJP."""
    n, h, width, _ = x.shape
    kh, kw = w.shape[:2]
    py, px = kh//2, kw//2
    padded = np.pad(x, ((0,0),(py,py),(px,px),(0,0)))
    out = np.zeros((n,h,width,w.shape[-1]))
    for y in range(kh):
        for col in range(kw):
            out += np.einsum('nhwc,co->nhwo', padded[:,y:y+h,col:col+width], w[y,col])

    def backward(dy):
        dx, dw = np.zeros_like(padded), np.zeros_like(w)
        for y in range(kh):
            for col in range(kw):
                window = padded[:,y:y+h,col:col+width]
                dw[y,col] = np.einsum('nhwc,nhwo->co',window,dy)
                dx[:,y:y+h,col:col+width] += np.einsum('nhwo,co->nhwc',dy,w[y,col])
        return dx[:,py:py+h,px:px+width], dw
    return out, backward


def layer(p, name, x, structural=None, bias=True):
    w = p[name+'/W']
    if structural is None:
        structural = np.ones_like(w)
    out, bc = conv(x, w*structural)
    if bias:
        out += p[name+'/b']

    def backward(dy, partials):
        dx,dw = bc(dy)
        partials[name+'/W'] = dw*structural
        if bias:
            partials[name+'/b'] = dy.sum(axis=(0,1,2))
        return dx
    return out, backward


def gate(pre):
    a,b = np.tanh(pre[:,:,:,:4]),sigmoid(pre[:,:,:,4:])
    def backward(dy):
        return np.concatenate([dy*b*(1-a*a),dy*a*b*(1-b)],axis=3)
    return a*b, backward


def network(p, image, labels, conditional):
    v,h = image,image
    blocks = []
    one_hot = np.eye(10)[labels]
    for i in range(3):
        name = f'block{i}'
        pv,bv = layer(p,name+'/vertical',v,mask(p[name+'/vertical/W'].shape,True,i==0))
        ph,bh = layer(p,name+'/horizontal',h,mask(p[name+'/horizontal/W'].shape,False,i==0))
        if conditional:
            pv += (one_hot @ p[name+'/condition_vertical'])[:,None,None,:]
            ph += (one_hot @ p[name+'/condition_horizontal'])[:,None,None,:]
        v,bgv = gate(pv)
        link,bl = layer(p,name+'/vertical_to_horizontal',v,bias=False)
        gh,bgh = gate(ph+link)
        projection,bp = layer(p,name+'/project',gh)
        h = h+projection if i else projection
        blocks.append((name,bv,bh,bgv,bl,bgh,bp))
    logits,bo = layer(p,'output',h)

    def backward(dlogits):
        partials = {}
        dh = bo(dlogits,partials)
        dv = np.zeros_like(v)
        for i,(name,bv,bh,bgv,bl,bgh,bp) in reversed(list(enumerate(blocks))):
            dcombined = bgh(bp(dh,partials))
            dpv = bgv(dv+bl(dcombined,partials))
            if conditional:
                partials[name+'/condition_vertical'] = one_hot.T @ dpv.sum(axis=(1,2))
                partials[name+'/condition_horizontal'] = one_hot.T @ dcombined.sum(axis=(1,2))
            dv = bv(dpv,partials)
            old_h = bh(dcombined,partials)
            dh = old_h+dh if i else old_h
        return partials,dh+dv
    return dict(logits=logits,vertical=v,horizontal=h),backward


def forward(p,image,labels,conditional):
    out,backward = network(p,image,labels,conditional)
    logits = out['logits']
    nll = np.logaddexp(0,logits)-image*logits
    out.update(probability=sigmoid(logits),per_image_nll=nll.sum(axis=(1,2,3)),loss=nll.mean())
    return out,backward((out['probability']-image)/image.size)[0]


def check(root):
    receipt = json.loads((root/'baseline.json').read_text())
    for file,digest in receipt['source_pins'].items():
        assert hashlib.sha256((root/('original-'+Path(file).name)).read_bytes()).hexdigest()==digest
    for file,digest in receipt['prepared_input_sha256'].items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest
    for file,digest in receipt['archive_sha256'].items():
        assert hashlib.sha256((root/'original-output'/file).read_bytes()).hexdigest()==digest
    for file,digest in receipt['phase_probe_sha256'].items():
        assert hashlib.sha256((root/file).read_bytes()).hexdigest()==digest
    # The previous independent PixelCNN audit verifies all original source rows,
    # exact Double resize, Float32 threshold and the shared prepared pool.
    binary_oracle = json.loads((root.parent/'pixel_cnn/independent-oracle.json').read_text())
    assert binary_oracle['passed'] and binary_oracle['counts']['resized_pixels_exact']==89600
    plain_source = root.parent/'pixel_cnn/original-output'
    plain_data = arrays(plain_source/'data.npz')
    counts = dict(forward_values=0,gradient_values=0,optimizer_writes=0,exact_weight_values=0,
                  finite_differences=0,ancestral_pixels=0,causal_jacobian_entries=0,batch_rows=0)
    errors = dict(forward=0.,gradient=0.,conditional=0.)
    def compare(stored,expected,label,gradient=False):
        assert set(stored)==set(expected),(label,set(stored)^set(expected))
        kind = 'gradient' if gradient else 'forward'
        for key,value in expected.items():
            np.testing.assert_allclose(stored[key],value,atol=5e-6 if gradient else 3e-5,
                                       rtol=2e-3 if gradient else 2e-4,err_msg=label+'/'+key)
            counts[kind+'_values'] += np.size(value)
            errors[kind] = max(errors[kind],float(abs(stored[key]-value).max()))
    variants = []
    minimum_margin = 1.
    for conditional,variant,leaves,total in [(False,'unconditional',23,1073),(True,'conditional',29,1553)]:
        source = root/'original-output'/variant
        data = arrays(source/'data.npz')
        for key,value in plain_data.items():
            exact(data[key],value)
        batch = np.genfromtxt(source/'batches.tsv',names=True,delimiter='\t',dtype=np.int64)
        assert len(batch)==3000
        counts['batch_rows'] += len(batch)
        for epoch in [1,2,3]:
            rows = batch[batch['epoch']==epoch]
            exact(np.sort(rows['row']),np.arange(1000))
            exact(rows['source_id'],data['fit_source_ids'][rows['row']].astype(np.int64))
            assert len(np.unique(rows['step']))==32
        p0,pf = arrays(source/'initial_arrays.npz'),arrays(source/'final.npz')
        assert len(p0)==leaves and sum(v.size for v in p0.values())==total
        masks = {key:mask(value.shape,'/vertical/' in key,key.startswith('block0/'))
                 for key,value in p0.items() if key.endswith('/W') and
                 ('/vertical/' in key or '/horizontal/' in key)}
        assert sum(np.count_nonzero(v==0) for v in masks.values())==320
        fb = arrays(source/'first_batch.npz');first = batch[batch['step']==0]['row']
        exact(fb['images'],data['fit_images'][first]);exact(fb['labels'],data['fit_labels'][first])
        x,labels = fb['images'].astype(float),fb['labels'].astype(int)
        for name,parameters,image,classes in [('first_forward',p0,x,labels),
                ('initial_outputs',p0,data['test_images'],data['test_labels']),
                ('outputs',pf,data['test_images'],data['test_labels'])]:
            out,_ = forward({k:v.astype(float) for k,v in parameters.items()},image.astype(float),classes.astype(int),conditional)
            compare(arrays(source/(name+'.npz')),out,variant+'/'+name)
        before = p0
        moment = {k:np.zeros_like(v) for k,v in p0.items()}
        variance = {k:np.zeros_like(v) for k,v in p0.items()}
        for step in range(96):
            rows = batch[batch['step']==step]['row'];assert len(rows)==(8 if step%32==31 else 32)
            _,partials = forward({k:v.astype(float) for k,v in before.items()},data['fit_images'][rows].astype(float),
                                 data['fit_labels'][rows].astype(int),conditional)
            gradient = arrays(root/variant/'phases'/f'batch{step}-gradient.npz')
            compare(gradient,partials,variant+f'/batch{step}',True)
            after = arrays(root/variant/'phases'/f'batch{step}-after.npz')
            assert set(after)==set(before)
            for key,value in before.items():
                g = gradient[key]
                moment[key] = moment[key]*np.float32(.9)+g*np.float32(.1)
                variance[key] = variance[key]*np.float32(.999)+(g*g)*np.float32(.001)
                m = moment[key]/np.float32(1-.9**(step+1))
                v = variance[key]/np.float32(1-.999**(step+1))
                exact(value-m/(np.sqrt(v)+np.float32(1e-8))*np.float32(.003),after[key])
                if key in masks:
                    forbidden = masks[key]==0
                    assert np.all(g[forbidden]==0) and np.all(partials[key][forbidden]==0)
                    exact(after[key][forbidden],p0[key][forbidden])
                counts['exact_weight_values'] += value.size
            exact(arrays(root/variant/'phases'/f'batch{step}-iterations.npz')['iterations'],np.array([step+1],np.int64))
            if step==0:
                compare(arrays(source/'initial_gradients.npz'),partials,variant+'/initial',True)
                for key in after:
                    exact(after[key],arrays(source/'first_after.npz')[key])
            if step%32==31:
                for key,value in arrays(source/f'epoch{step//32+1}_arrays.npz').items():
                    exact(after[key],value)
            before = after;counts['optimizer_writes'] += 1
        for key in pf:
            exact(before[key],pf[key])
        p = {k:v.astype(float) for k,v in pf.items()}
        _,derivatives = forward(p,x,labels,conditional)
        compare(arrays(source/'final_gradients.npz'),derivatives,variant+'/final',True)
        for key,partial in derivatives.items():
            for index in sorted({0,partial.size//2,partial.size-1}):
                plus,minus = dict(p),dict(p);plus[key],minus[key] = p[key].copy(),p[key].copy()
                plus[key].flat[index] += 1e-6;minus[key].flat[index] -= 1e-6
                finite = (forward(plus,x,labels,conditional)[0]['loss']-forward(minus,x,labels,conditional)[0]['loss'])/2e-6
                np.testing.assert_allclose(partial.flat[index],finite,atol=1e-8,rtol=1e-3,err_msg=variant+'/'+key)
                counts['finite_differences'] += 1
        cause = arrays(source/'causality.npz');image = cause['images'].astype(float);classes = cause['labels'].astype(int)
        out,adjoint = network(p,image,classes,conditional);probability = sigmoid(out['logits'])
        np.testing.assert_allclose(cause['probability'],probability,atol=2e-6,rtol=1e-4)
        for site in range(64):
            delta = np.zeros_like(image);delta.flat[site] = 1
            _,dx = adjoint(delta)
            np.testing.assert_allclose(cause['jacobian'][site],dx.reshape(-1),atol=3e-6,rtol=2e-4)
            assert np.all(dx.reshape(-1)[site:]==0)
            changed = image.copy().reshape(-1);changed[site:] = 1-changed[site:];changed = changed.reshape(image.shape)
            flipped = sigmoid(network(p,changed,classes,conditional)[0]['logits'])
            exact(flipped.reshape(-1)[:site+1],probability.reshape(-1)[:site+1])
            if site%8==0:
                exact(cause[f'changed_input{site}'],changed.astype(np.float32))
                np.testing.assert_allclose(cause[f'changed_probability{site}'],flipped,atol=2e-6,rtol=1e-4)
        counts['causal_jacobian_entries'] += 4096
        history = np.genfromtxt(source/'history.tsv',names=True,delimiter='\t')
        for epoch in [1,2,3]:
            pe = {k:v.astype(float) for k,v in arrays(source/f'epoch{epoch}_arrays.npz').items()}
            loss = forward(pe,data['validation_images'].astype(float),data['validation_labels'].astype(int),conditional)[0]['loss']
            np.testing.assert_allclose(history[epoch-1][1],loss,atol=3e-6,rtol=1e-5)
            np.testing.assert_allclose(history[epoch-1][2],loss/np.log(2),atol=5e-6,rtol=1e-5)
        for filename,parameters,n,seed,prefix,common in [
                (f'epoch{e}_samples',arrays(source/f'epoch{e}_arrays.npz'),16,8521,0,False) for e in [1,2,3]]+[
                ('samples',pf,32,8521,0,False),('completions',pf,8,8527,32,False),('class_grid',pf,40,8537,0,True)]:
            stored = arrays(source/(filename+'.npz'))
            u = uniforms(n,seed)
            if common:
                u[4:] = u[np.arange(4,n)%4]
                exact(stored['labels'],np.repeat(np.arange(10),4).astype(np.int32))
            exact(stored['uniform'],u)
            classes = (data['test_labels'][:n] if prefix else np.repeat(np.arange(10),4) if common else np.arange(n)%10).astype(int)
            if 'labels' in stored:
                exact(stored['labels'].astype(int),classes)
            canvas = np.zeros((n,8,8,1))
            if prefix:
                exact(stored['truth'],data['test_images'][:n])
                canvas.reshape(n,64)[:,:prefix] = data['test_images'][:n].reshape(n,64)[:,:prefix]
            assert np.all(stored['conditionals'][:,:prefix]==0)
            parameters = {k:v.astype(float) for k,v in parameters.items()}
            for site in range(prefix,64):
                q = sigmoid(network(parameters,canvas,classes,conditional)[0]['logits']).reshape(n,64)[:,site]
                native = stored['conditionals'][:,site]
                np.testing.assert_allclose(native,q,atol=3e-6,rtol=1e-4,err_msg=variant+'/'+filename+str(site))
                errors['conditional'] = max(errors['conditional'],float(abs(native-q).max()))
                minimum_margin = min(minimum_margin,float(abs(u[:,site]-q).min()))
                exact(u[:,site]<q,u[:,site]<native)
                canvas.reshape(n,64)[:,site] = u[:,site]<q;counts['ancestral_pixels'] += n
            exact(canvas.astype(np.float32),stored['images'])
        control = arrays(source/'class_controls.npz')
        exact(control['labels'],((data['test_labels']+1)%10).astype(np.int32))
        shifted = sigmoid(network(p,data['test_images'].astype(float),control['labels'].astype(int),conditional)[0]['logits'])
        np.testing.assert_allclose(control['probability'],shifted,atol=2e-6,rtol=1e-4)
        native = arrays(source/'outputs.npz')['probability']
        effect = float(abs(control['probability'].astype(float)-native.astype(float)).max())
        assert effect>1e-6 if conditional else effect==0
        if not conditional:
            exact(arrays(source/'class_grid.npz')['images'][:4],arrays(source/'class_grid.npz')['images'][4:8])
        spot = arrays(source/'blindspot.npz');single = data['test_images'][:1].astype(float)
        delta = np.zeros_like(single);delta.flat[34] = 1
        _,dx = network(p,single,data['test_labels'][:1].astype(int),conditional)[1](delta)
        np.testing.assert_allclose(spot['gated_input_gradient'],dx,atol=3e-6,rtol=2e-4)
        assert abs(dx.reshape(-1)[28])>1e-10
        plain = {k:v.astype(float) for k,v in arrays(plain_source/'initial_arrays.npz').items()}
        _,plain_dx = plain_network(plain,single)[1](delta)
        np.testing.assert_allclose(spot['plain_input_gradient'],plain_dx,atol=1e-8,rtol=1e-4)
        assert plain_dx.reshape(-1)[28]==0
        metrics = json.loads((source/'metrics.json').read_text())
        assert metrics['optimizer_updates']==96 and metrics['selected_epoch_fixed_final']==3
        assert metrics['class_probability_max_effect']==effect
        assert metrics['test_bits_per_binary_pixel']>metrics['fit_bernoulli_baseline_bits']
        variants.append(dict(name=variant,parameters=total,leaves=leaves,metrics=metrics,
                             opened_blindspot_partial=float(dx.reshape(-1)[28])))
    assert counts['optimizer_writes']==192 and counts['ancestral_pixels']==15872
    return dict(passed=True,counts=counts,maximum_errors=errors,minimum_sampling_margin=minimum_margin,
                all_192_adam_writes_bit_exact=True,all_320_masked_weights_and_partials_exact_per_variant=True,
                all_128_future_flips_exact=True,shared_four_uniform_streams_across_ten_classes_exact=True,
                conditional_class_effect_and_unconditional_label_invariance=True,both_blindspot_comparisons=True,variants=variants)



def check_rgb(root):
    """Compare both dtypes to the unchanged old RGB3/30 source and NumPy VJPs."""
    reference,native = root/'rgb-reference-output',root/'rgb-native-output'
    values,forward_values,partials,error,gradient_error = 0,0,0,0.,0.
    for dtype in ['float','double']:
        for suffix in ['arrays','forward','gradients']:
            old = arrays(reference/f'{dtype}-{suffix}.npz')
            new = arrays(native/f'{dtype}-{suffix}.npz')
            assert set(old)==set(new)
            for key,value in old.items():
                exact(value,new[key]);values += value.size
        p = {k:v.astype(float) for k,v in arrays(reference/f'{dtype}-arrays.npz').items()}
        stored = arrays(reference/f'{dtype}-forward.npz')
        out,backward = network(p,stored['image'].astype(float),np.zeros(2,int),False)
        for key,value in out.items():
            np.testing.assert_allclose(stored[key],value,atol=2e-8 if dtype=='float' else 1e-14,rtol=1e-4 if dtype=='float' else 1e-11)
            error = max(error,float(abs(stored[key]-value).max()));forward_values += value.size
        weights,image = backward(2*out['logits']/out['logits'].size)
        expected = {**weights,'image':image}
        stored = arrays(reference/f'{dtype}-gradients.npz')
        assert set(stored)==set(expected)
        for key,value in expected.items():
            np.testing.assert_allclose(stored[key],value,atol=1e-9 if dtype=='float' else 1e-14,rtol=1e-4 if dtype=='float' else 1e-11)
            gradient_error = max(gradient_error,float(abs(stored[key]-value).max()));partials += value.size
    return dict(passed=True,source_and_API_values_bit_exact=values,archives=6,independent_forward_values=forward_values,
                independent_partials=partials,maximum_forward_error=error,maximum_gradient_error=gradient_error,
                both_Float32_and_Float64=True,input_channels=3,output_channels=30,parameters=1410,leaves=23)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('build/nn-report-conversion/gated_pixel_cnn'))
    parser.add_argument('--json',type=Path)
    args = parser.parse_args();result = check(args.root)
    result['rgb_context']=check_rgb(args.root.parent/'gated_pixel_cnn_core')
    rendered = json.dumps(result,indent=2)+'\n';print(rendered,end='')
    if args.json:
        args.json.write_text(rendered)


if __name__=='__main__':
    main()
