#!/usr/bin/env python3
"""Independent NumPy MADE degrees, dense VJPs, both orders and density mixture.

Reads frozen original arrays and all captured updates. Never fits another model,
prepares a substitute split or uses native model math for the reference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays,exact
from check_contractive_autoencoder import sigmoid
from check_pixel_cnn import uniforms


def inventory(reverse):
    degree = np.arange(1,65,dtype=np.int32)
    if reverse:degree = degree[::-1].copy()
    h1 = (1+np.arange(16)*62//15).astype(np.int32)
    h2 = h1[(np.arange(16)*5)%16]
    return dict(input_degrees=degree,hidden1_degrees=h1,hidden2_degrees=h2,
                input_mask=(degree[:,None]<=h1[None,:]).astype(np.float32),
                hidden_mask=(h1[:,None]<=h2[None,:]).astype(np.float32),
                output_mask=(h2[:,None]<degree[None,:]).astype(np.float32))


def network(p,image,reverse):
    masks = inventory(reverse)
    x = image.reshape(-1,64)
    w1 = p['input/W']*masks['input_mask']
    w2 = p['hidden/W']*masks['hidden_mask']
    w3 = p['output/W']*masks['output_mask']
    h1 = sigmoid(x@w1+p['input/b'])
    h2 = sigmoid(h1@w2+p['hidden/b'])
    logits = (h2@w3+p['output/b']).reshape(image.shape)
    def backward(dlogits):
        delta = dlogits.reshape(-1,64)
        out = {'output/W':h2.T@delta*masks['output_mask'],'output/b':delta.sum(axis=0)}
        dh2 = (delta@w3.T)*h2*(1-h2)
        out.update({'hidden/W':h1.T@dh2*masks['hidden_mask'],'hidden/b':dh2.sum(axis=0)})
        dh1 = (dh2@w2.T)*h1*(1-h1)
        out.update({'input/W':x.T@dh1*masks['input_mask'],'input/b':dh1.sum(axis=0)})
        return out,(dh1@w1.T).reshape(image.shape)
    return dict(logits=logits,hidden1=h1,hidden2=h2),backward


def forward(p,image,reverse):
    out,backward = network(p,image,reverse)
    logits = out['logits'];nll = np.logaddexp(0,logits)-image*logits
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
    # The pinned common pool/resize were already independently checked, including
    # source IDs, Double interpolation and the rounded Float32 .5 threshold.
    previous = root.parent/'pixel_cnn';binary = json.loads((previous/'independent-oracle.json').read_text())
    assert binary['passed'] and binary['counts']['resized_pixels_exact']==89600
    common_data = arrays(previous/'original-output/data.npz')
    counts = dict(forward_values=0,gradient_values=0,optimizer_writes=0,exact_weight_values=0,
                  finite_differences=0,ancestral_pixels=0,causal_jacobian_entries=0,batch_rows=0,degree_mask_values_exact=0)
    errors = dict(forward=0.,gradient=0.,conditional=0.)
    def compare(stored,expected,label,gradient=False):
        assert set(stored)==set(expected),(label,set(stored)^set(expected))
        kind = 'gradient' if gradient else 'forward'
        for key,value in expected.items():
            np.testing.assert_allclose(stored[key],value,atol=4e-6 if gradient else 3e-5,
                                       rtol=2e-3 if gradient else 2e-4,err_msg=label+'/'+key)
            counts[kind+'_values'] += np.size(value)
            errors[kind] = max(errors[kind],float(abs(stored[key]-value).max()))
    variants = [];minimum_margin = 1.
    for reverse,variant in [(False,'raster'),(True,'reverse')]:
        source = root/'original-output'/variant
        data = arrays(source/'data.npz')
        for key,value in common_data.items():exact(data[key],value)
        inv = inventory(reverse);stored_inventory = arrays(source/'mask_inventory.npz')
        assert set(inv)==set(stored_inventory)
        for key,value in inv.items():exact(stored_inventory[key],value);counts['degree_mask_values_exact'] += value.size
        masks = {name+'/W':inv[key] for name,key in [('input','input_mask'),('hidden','hidden_mask'),('output','output_mask')]}
        assert sum(np.count_nonzero(m==0) for m in masks.values())==1144
        assert set(inv['hidden1_degrees'])==set(inv['hidden2_degrees'])
        batch = np.genfromtxt(source/'batches.tsv',names=True,delimiter='\t',dtype=np.int64)
        assert len(batch)==3000;counts['batch_rows'] += len(batch)
        for epoch in [1,2,3]:
            rows = batch[batch['epoch']==epoch]
            exact(np.sort(rows['row']),np.arange(1000))
            exact(rows['source_id'],data['fit_source_ids'][rows['row']].astype(np.int64))
            assert len(np.unique(rows['step']))==32
        p0,pf = arrays(source/'initial_arrays.npz'),arrays(source/'final.npz')
        assert len(p0)==6 and sum(v.size for v in p0.values())==2400
        first = arrays(source/'first_batch.npz')['images'].astype(float)
        exact(first,data['fit_images'][batch[batch['step']==0]['row']].astype(float))
        for name,parameters,image in [('first_forward',p0,first),('initial_outputs',p0,data['test_images']),('outputs',pf,data['test_images'])]:
            out,_ = forward({k:v.astype(float) for k,v in parameters.items()},image.astype(float),reverse)
            compare(arrays(source/(name+'.npz')),out,variant+'/'+name)
        before = p0;moment = {k:np.zeros_like(v) for k,v in p0.items()};variance = {k:np.zeros_like(v) for k,v in p0.items()}
        for step in range(96):
            rows = batch[batch['step']==step]['row'];assert len(rows)==(8 if step%32==31 else 32)
            _,partials = forward({k:v.astype(float) for k,v in before.items()},data['fit_images'][rows].astype(float),reverse)
            gradient = arrays(root/variant/'phases'/f'batch{step}-gradient.npz')
            compare(gradient,partials,variant+f'/batch{step}',True)
            after = arrays(root/variant/'phases'/f'batch{step}-after.npz');assert set(after)==set(before)
            for key,value in before.items():
                g = gradient[key];moment[key] = moment[key]*np.float32(.9)+g*np.float32(.1)
                variance[key] = variance[key]*np.float32(.999)+(g*g)*np.float32(.001)
                m = moment[key]/np.float32(1-.9**(step+1));v = variance[key]/np.float32(1-.999**(step+1))
                exact(value-m/(np.sqrt(v)+np.float32(1e-8))*np.float32(.003),after[key]);counts['exact_weight_values'] += value.size
                if key in masks:
                    blocked = masks[key]==0;assert np.all(g[blocked]==0) and np.all(partials[key][blocked]==0)
                    exact(after[key][blocked],p0[key][blocked])
            exact(arrays(root/variant/'phases'/f'batch{step}-iterations.npz')['iterations'],np.array([step+1],np.int64))
            if step==0:
                compare(arrays(source/'initial_gradients.npz'),partials,variant+'/initial',True)
                for key in after:exact(after[key],arrays(source/'first_after.npz')[key])
            if step%32==31:
                for key,value in arrays(source/f'epoch{step//32+1}_arrays.npz').items():exact(after[key],value)
            before = after;counts['optimizer_writes'] += 1
        for key in pf:exact(before[key],pf[key])
        p = {k:v.astype(float) for k,v in pf.items()}
        _,derivatives = forward(p,first,reverse);compare(arrays(source/'final_gradients.npz'),derivatives,variant+'/final',True)
        for key,partial in derivatives.items():
            for index in sorted({0,partial.size//2,partial.size-1}):
                plus,minus = dict(p),dict(p);plus[key],minus[key] = p[key].copy(),p[key].copy()
                plus[key].flat[index] += 1e-6;minus[key].flat[index] -= 1e-6
                finite = (forward(plus,first,reverse)[0]['loss']-forward(minus,first,reverse)[0]['loss'])/2e-6
                np.testing.assert_allclose(partial.flat[index],finite,atol=1e-8,rtol=1e-3,err_msg=variant+'/'+key)
                counts['finite_differences'] += 1
        ranks = np.arange(64)[::-1] if reverse else np.arange(64)
        sites = np.argsort(ranks)
        cause = arrays(source/'causality.npz');image = cause['images'].astype(float)
        out,backward = network(p,image,reverse);probability = sigmoid(out['logits'])
        np.testing.assert_allclose(cause['probability'],probability,atol=2e-6,rtol=1e-4)
        for ordinal,site in enumerate(sites):
            delta = np.zeros_like(image);delta.flat[site] = 1
            _,dx = backward(delta)
            np.testing.assert_allclose(cause['jacobian'][site],dx.reshape(-1),atol=3e-6,rtol=2e-4)
            assert np.all(dx.reshape(-1)[ranks>=ordinal]==0)
            changed = image.copy().reshape(-1);changed[ranks>=ordinal] = 1-changed[ranks>=ordinal];changed = changed.reshape(image.shape)
            flipped = sigmoid(network(p,changed,reverse)[0]['logits'])
            exact(flipped.reshape(-1)[ranks<=ordinal],probability.reshape(-1)[ranks<=ordinal])
            if ordinal%8==0:
                exact(cause[f'changed_input{site}'],changed.astype(np.float32))
                np.testing.assert_allclose(cause[f'changed_probability{site}'],flipped,atol=2e-6,rtol=1e-4)
        counts['causal_jacobian_entries'] += 4096
        history = np.genfromtxt(source/'history.tsv',names=True,delimiter='\t')
        for epoch in [1,2,3]:
            pe = {k:v.astype(float) for k,v in arrays(source/f'epoch{epoch}_arrays.npz').items()}
            loss = forward(pe,data['validation_images'].astype(float),reverse)[0]['loss']
            np.testing.assert_allclose(history[epoch-1][1],loss,atol=3e-6,rtol=1e-5)
            np.testing.assert_allclose(history[epoch-1][2],loss/np.log(2),atol=5e-6,rtol=1e-5)
        for filename,parameters,n,seed,prefix in [(f'epoch{e}_samples',arrays(source/f'epoch{e}_arrays.npz'),16,8821,0) for e in [1,2,3]]+[
                ('samples',pf,32,8821,0),('completions',pf,8,8827,32)]:
            stored = arrays(source/(filename+'.npz'));u = uniforms(n,seed);exact(stored['uniform'],u)
            canvas = np.zeros((n,8,8,1))
            if prefix:
                exact(stored['truth'],data['test_images'][:n])
                canvas.reshape(n,64)[:,sites[:prefix]] = data['test_images'][:n].reshape(n,64)[:,sites[:prefix]]
            assert np.all(stored['conditionals'][:,:prefix]==0)
            parameters = {k:v.astype(float) for k,v in parameters.items()}
            for ordinal in range(prefix,64):
                site = sites[ordinal]
                q = sigmoid(network(parameters,canvas,reverse)[0]['logits']).reshape(n,64)[:,site]
                native = stored['conditionals'][:,ordinal]
                np.testing.assert_allclose(native,q,atol=3e-6,rtol=1e-4,err_msg=variant+'/'+filename+str(ordinal))
                errors['conditional'] = max(errors['conditional'],float(abs(native-q).max()))
                minimum_margin = min(minimum_margin,float(abs(u[:,ordinal]-q).min()))
                exact(u[:,ordinal]<q,u[:,ordinal]<native)
                canvas.reshape(n,64)[:,site] = u[:,ordinal]<q;counts['ancestral_pixels'] += n
            exact(canvas.astype(np.float32),stored['images'])
        completed = arrays(source/'completions.npz');truth = completed['truth'];observed = truth.copy().reshape(8,64)
        observed[:,sites[32:]] = .5
        expected = np.stack([truth,observed.reshape(8,8,8,1),completed['images']],axis=1).reshape(24,8,8,1)
        exact(arrays(source/'completion_gallery.npz')['images'],expected)
        metrics = json.loads((source/'metrics.json').read_text())
        assert metrics['optimizer_updates']==96 and metrics['selected_epoch_fixed_final']==3
        assert metrics['test_bits_per_binary_pixel']>metrics['fit_bernoulli_baseline_bits']
        variants.append(dict(name=variant,metrics=metrics))
    ensemble = arrays(root/'original-output/order_ensemble.npz')
    a = arrays(root/'original-output/raster/outputs.npz')['per_image_nll'];b = arrays(root/'original-output/reverse/outputs.npz')['per_image_nll']
    exact(ensemble['raster_per_image_nll'],a);exact(ensemble['reverse_per_image_nll'],b)
    mixture = -(np.logaddexp(-a.astype(float),-b.astype(float))-np.log(2))
    np.testing.assert_allclose(ensemble['per_image_nll'],mixture,atol=1e-13,rtol=0)
    assert np.all(mixture<=(a.astype(float)+b.astype(float))/2+1e-13)
    assert abs(mixture-(a.astype(float)+b.astype(float))/2).max()>.1
    assert counts['optimizer_writes']==192 and counts['ancestral_pixels']==10752
    return dict(passed=True,counts=counts,maximum_errors=errors,minimum_sampling_margin=minimum_margin,
                all_192_adam_writes_bit_exact=True,all_1144_masked_weights_and_partials_exact_per_order=True,
                all_128_order_correct_future_flips_exact=True,reverse_sampling_and_prefix_ordinals_exact=True,
                degrees_and_strict_output_masks_exact=True,equal_probability_mixture_is_not_mean_NLL=True,
                mixture_values_checked=200,mixture_bits_per_binary_pixel=float(mixture.mean()/64/np.log(2)),variants=variants)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('build/nn-report-conversion/made_density'))
    parser.add_argument('--json',type=Path);args=parser.parse_args();result=check(args.root)
    rendered=json.dumps(result,indent=2)+'\n';print(rendered,end='')
    if args.json:args.json.write_text(rendered)


if __name__=='__main__':main()
