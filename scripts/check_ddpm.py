#!/usr/bin/env python3
"""Independent NumPy audit of the pinned continuous-MNIST DDPM experiment.

Reconstruct raw image identities, half-pixel resize, MT noise/time banks and
cosine schedule. Use explicit dense/SiLU pullbacks and finite differences,
check every captured Adam write and every ancestral transition. No native
model calls, external autodiff, replacement pools or refitting.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

from check_adversarial_autoencoder import arrays, exact, floating
from check_nice_density import gaussian


def schedule():
    values = {k: [] for k in ['beta', 'alpha', 'alpha_bar', 'alpha_bar_previous',
              'sqrt_alpha_bar', 'sqrt_one_minus_alpha_bar', 'posterior_variance',
              'posterior_coefficient1', 'posterior_coefficient2']}
    def f(t):
        return math.cos((t + .008) / 1.008 * math.pi / 2)**2
    bar = 1.
    for i in range(32):
        beta = min(.999, 1 - f((i + 1) / 32) / f(i / 32))
        alpha, previous = 1 - beta, bar
        bar *= alpha
        row = [beta, alpha, bar, previous, math.sqrt(bar), math.sqrt(1 - bar),
               beta * (1 - previous) / (1 - bar), beta * math.sqrt(previous) / (1 - bar),
               (1 - previous) * math.sqrt(alpha) / (1 - bar)]
        for key, value in zip(values, row):
            values[key].append(value)
    return {k: np.array(v) for k, v in values.items()}


def noise(n, seed):
    return gaussian(n * 32, seed).reshape(n, 8, 8, 1)


def times(n, seed):
    words = np.random.RandomState(seed).randint(0, 2**32, n, dtype=np.uint32)
    return (32 * (words.astype(float) + .5) / 2**32).astype(np.int32)


def embedding(time):
    # Structural Float32 normalization and phase constants remain rounded even
    # for the independent Double network and finite-difference objective.
    normalized = time.astype(np.float32).reshape(-1, 1) / np.float32(31)
    phase = normalized * np.array([math.pi, 2*math.pi, 4*math.pi, 8*math.pi], np.float32)
    sin = np.array([[math.sin(float(v)) for v in row] for row in phase], np.float32)
    cos = np.array([[math.cos(float(v)) for v in row] for row in phase], np.float32)
    return np.concatenate([sin, cos], axis=1)


def extract(table, key, time):
    return table[key][time].astype(np.float32).reshape(-1, 1, 1, 1)


def noised(table, clean, epsilon, time):
    return extract(table, 'sqrt_alpha_bar', time) * clean + extract(table, 'sqrt_one_minus_alpha_bar', time) * epsilon


def predictor(p, state, time):
    emb = embedding(time).astype(float)
    encoded = np.concatenate([state.reshape(len(state), 64).astype(float), emb], axis=1)
    h = encoded
    backwards = []
    hidden = []
    for name in ['hidden1', 'hidden2', 'output']:
        z = h @ p[name+'/W'] + p[name+'/b']
        sig = 1 / (1 + np.exp(-z)) if name != 'output' else None
        backwards.append((name, h, z, sig))
        h = z * sig if sig is not None else z
        hidden.append(h)
    result = dict(epsilon=h.reshape(state.shape), time_embedding=emb,
                  hidden1=hidden[0], hidden2=hidden[1])

    def adjoint(delta):
        partials = {}
        delta = delta.reshape(len(state), 64)
        for name, input_value, z, sig in reversed(backwards):
            if sig is not None:
                delta = delta * (sig + z * sig * (1 - sig))
            partials[name+'/W'] = input_value.T @ delta
            partials[name+'/b'] = delta.sum(axis=0)
            delta = delta @ p[name+'/W'].T
        return partials, delta[:, :64].reshape(state.shape)
    return result, adjoint


def evaluate(p, clean, epsilon, time, table):
    state = noised(table, clean, epsilon, time)
    out, back = predictor(p, state, time)
    residual = out['epsilon'] - epsilon
    return dict(out, noised=state, loss=np.asarray((residual**2).mean())), back(2 * residual / residual.size)[0]


def reverse(table, state, time, epsilon, innovation):
    """Independent Float32 primitive rounding, separately from network math."""
    signal = extract(table, 'sqrt_alpha_bar', time)
    sigma = extract(table, 'sqrt_one_minus_alpha_bar', time)
    raw = (state - sigma * epsilon) / signal
    clipped = np.clip(raw, np.float32(-1), np.float32(1))
    mean = extract(table, 'posterior_coefficient1', time) * clipped + extract(table, 'posterior_coefficient2', time) * state
    variance = extract(table, 'posterior_variance', time)
    return dict(next=mean + np.sqrt(variance) * innovation, mean=mean, variance=variance,
                x0_raw=raw, x0_clipped=clipped)


def read_idx(path):
    raw = path.read_bytes()
    assert raw[:2] == b'\0\0' and raw[2] == 8
    dims = raw[3]
    shape = struct.unpack('>'+'I'*dims, raw[4:4+4*dims])
    return np.frombuffer(raw, np.uint8, offset=4+4*dims).reshape(shape)


def check(root, raw_mnist):
    receipt = json.loads((root/'baseline.json').read_text())
    source = root/'original-output'
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for file, digest in receipt['source_pins'].items():
        assert sha(root/('original-'+Path(file).name)) == digest
    for file, digest in receipt['prepared_input_sha256'].items():
        assert sha(file) == digest
    for key, folder in [('archive_sha256', source), ('other_files_sha256', source), ('phase_probe_sha256', root)]:
        for file, digest in receipt[key].items():
            assert sha(folder/file) == digest
    audited = set()
    def load(name):
        audited.add(name+'.npz')
        return arrays(source/(name+'.npz'))
    counts = dict(original_pixels=0, source_ID_rows=0, resized_pixels=0, schedule_values=0,
                  noise_values=0, timestep_rows=0, batch_rows=0, forward_values=0,
                  gradient_values=0, exact_weight_values=0, optimizer_writes=0,
                  parameter_coordinate_finite_differences=0, reverse_input_gradient_values=0,
                  reverse_input_direction_finite_differences=0, ancestral_transitions=0,
                  ancestral_sample_rows=0, ancestral_field_values=0, exact_reverse_formula_values=0)
    errors = dict(forward=0., gradient=0., finite_difference=0., input_gradient=0., input_finite_difference=0.)
    def compare(stored, expected, label, gradient=False, tolerance=None):
        assert set(stored) == set(expected), (label, set(stored)^set(expected))
        kind = 'gradient' if gradient else 'forward'
        for key, value in expected.items():
            delta = float(np.max(np.abs(stored[key].astype(float)-value)))
            errors[kind] = max(errors[kind], delta)
            np.testing.assert_allclose(stored[key], value, atol=tolerance or (2e-8 if gradient else 3e-6),
                                       rtol=3e-4 if gradient else 3e-5, err_msg=label+'/'+key)
            counts[kind+'_values'] += value.size
    data = load('data')
    paths = [raw_mnist/name for name in ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
                                       't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']]
    raw = [read_idx(p) for p in paths]
    all_ids = []
    for split, n in [('fit',1000),('validation',200),('test',200)]:
        ids = data[split+'_source_ids'];all_ids.append(ids)
        bank = 2 if split=='test' else 0
        rows = ids.astype(np.int64)-(60000 if bank else 0)
        assert len(ids)==n and len(np.unique(ids))==n
        expected = raw[bank][rows].astype(np.float32).reshape(n,28,28,1)/np.float32(255)
        exact(data[split+'_original'],expected)
        exact(data[split+'_labels'],raw[bank+1][rows].astype(np.int32))
        resized = np.zeros((n,8,8,1))
        for y in range(8):
            for x in range(8):
                py,px=(y+.5)*3.5-.5,(x+.5)*3.5-.5;iy,ix=int(py),int(px)
                for dy in [0,1]:
                    for dx in [0,1]:
                        resized[:,y,x]+=(py-iy if dy else 1-py+iy)*(px-ix if dx else 1-px+ix)*expected[:,iy+dy,ix+dx].astype(float)
        exact(data[split+'_images'],resized.astype(np.float32)*np.float32(2)-np.float32(1))
        assert np.any((data[split+'_images']>-1)&(data[split+'_images']<1))
        counts['original_pixels']+=expected.size;counts['resized_pixels']+=resized.size;counts['source_ID_rows']+=n
    assert len(np.unique(np.concatenate(all_ids)))==1400
    table = schedule();stored = load('schedule');assert set(table)==set(stored)
    for key,value in table.items():
        exact(stored[key],value);counts['schedule_values']+=len(value)
    assert table['posterior_variance'][0]==0 and table['beta'][-1]==.999
    for split,seed in [('validation',11101),('test',11103)]:
        exact(data[split+'_noise'],noise(200,seed).astype(np.float32))
        exact(data[split+'_time'],(np.arange(200)%32).astype(np.int32))
        counts['noise_values']+=12800;counts['timestep_rows']+=200
    p0,pf=load('initial_arrays'),load('final')
    assert len(p0)==6 and sum(v.size for v in p0.values())==5504
    for name,shape in [('hidden1',(72,32)),('hidden2',(32,32)),('output',(32,64))]:
        assert p0[name+'/W'].shape==shape
        exact(p0[name+'/b'],np.zeros(shape[1],np.float32))
    batch=np.genfromtxt(source/'batches.tsv',names=True,delimiter='\t',dtype=np.int64)
    counts['batch_rows']=len(batch);assert len(batch)==3000
    for epoch in [1,2,3]:
        rows=batch[batch['epoch']==epoch];exact(np.sort(rows['row']),np.arange(1000))
        exact(rows['source_id'],data['fit_source_ids'][rows['row']].astype(np.int64))
    first=load('first_batch');exact(first['images'],data['fit_images'][batch[batch['step']==0]['row']])
    before=p0;moment,variance=[{k:np.zeros_like(v) for k,v in p0.items()} for _ in range(2)]
    random=load('training_randomness');assert len(random)==192
    for step in range(96):
        rows=batch[batch['step']==step]['row'];n=len(rows);assert n==(8 if step%32==31 else 32)
        e=noise(n,1100051+step);t=times(n,2100021+step)
        exact(random[f'step{step}_noise'],e);exact(random[f'step{step}_time'],t)
        counts['noise_values']+=e.size;counts['timestep_rows']+=n
        out,partial=evaluate(floating(before),data['fit_images'][rows],e.astype(np.float32),t,table)
        gradients=arrays(root/'phases'/f'batch{step}-gradient.npz')
        compare(gradients,partial,f'gradient{step}',True)
        after=arrays(root/'phases'/f'batch{step}-after.npz')
        for key,value in before.items():
            g=gradients[key]
            moment[key]=moment[key]*np.float32(.9)+g*np.float32(.1)
            variance[key]=variance[key]*np.float32(.999)+(g*g)*np.float32(.001)
            m=moment[key]/np.float32(1-.9**(step+1));v=variance[key]/np.float32(1-.999**(step+1))
            exact(value-m/(np.sqrt(v)+np.float32(1e-8))*np.float32(.003),after[key]);counts['exact_weight_values']+=value.size
        exact(arrays(root/'phases'/f'batch{step}-iterations.npz')['iterations'],np.array([step+1],np.int64))
        if step==0:
            exact(first['noise'],e.astype(np.float32));exact(first['time'],t)
            compare(load('first_forward'),out,'first-forward');compare(load('initial_gradients'),partial,'first-gradient',True)
            for key,value in load('first_after').items():exact(value,after[key])
        if step%32==31:
            for key,value in load(f'epoch{step//32+1}_arrays').items():exact(value,after[key])
        before=after;counts['optimizer_writes']+=1
    for key,value in before.items():exact(value,pf[key]);exact(value,load('model_arrays')[key])
    for name,p in [('initial_outputs',p0),('outputs',pf)]:
        compare(load(name),evaluate(floating(p),data['test_images'],data['test_noise'],data['test_time'],table)[0],name)
    history=np.loadtxt(source/'history.tsv',skiprows=1)
    for epoch in [1,2,3]:
        p=floating(load(f'epoch{epoch}_arrays'));out,_=evaluate(p,data['validation_images'],data['validation_noise'],data['validation_time'],table)
        compare(load(f'epoch{epoch}_validation'),out,f'epoch{epoch}')
        np.testing.assert_allclose(history[epoch-1,1],out['loss'],atol=1e-7,rtol=0)
    final=floating(pf);_,partial=evaluate(final,first['images'],first['noise'],first['time'],table)
    compare(load('final_gradients'),partial,'final-gradients',True)
    for key,value in final.items():
        for index in np.ndindex(value.shape):
            hi,lo=[{k:v.copy() for k,v in final.items()} for _ in range(2)]
            hi[key][index]+=1e-5;lo[key][index]-=1e-5
            fd=(evaluate(hi,first['images'],first['noise'],first['time'],table)[0]['loss']-evaluate(lo,first['images'],first['noise'],first['time'],table)[0]['loss'])/2e-5
            np.testing.assert_allclose(fd,partial[key][index],atol=2e-10,rtol=2e-5,err_msg=key+str(index))
            errors['finite_difference']=max(errors['finite_difference'],float(abs(fd-partial[key][index])));counts['parameter_coordinate_finite_differences']+=1
    # Audit each transition against its own stored Float32 state. This checks
    # every path without accumulating a different BLAS rounding history.
    for name,p,n in [('epoch1_samples',floating(load('epoch1_arrays')),16),('epoch2_samples',floating(load('epoch2_arrays')),16),('epoch3_samples',floating(load('epoch3_arrays')),16),('samples',final,32)]:
        sample=load(name);exact(sample['initial'],noise(n,11201).astype(np.float32));exact(sample['trajectory'][0],sample['initial']);exact(sample['images'],sample['trajectory'][-1])
        for step in range(32):
            state=sample['trajectory'][step];t=np.full(n,31-step,np.int32);innovation=noise(n,11202+step)
            exact(sample['innovations'][step],innovation)
            expected,_=predictor(p,state,t)
            compare({'epsilon':sample['epsilon'][step]},{'epsilon':expected['epsilon']},name+f'/step{step}')
            fields=reverse(table,state,t,sample['epsilon'][step],innovation.astype(np.float32))
            for key in ['mean','x0_raw','x0_clipped']:
                exact(sample[key][step],fields[key]);counts['exact_reverse_formula_values']+=fields[key].size
            exact(sample['trajectory'][step+1],fields['next']);counts['exact_reverse_formula_values']+=fields['next'].size
            if step==31:
                exact(fields['next'],reverse(table,state,t,sample['epsilon'][step],np.full(state.shape,123,np.float32))['next'])
            counts['ancestral_transitions']+=1;counts['ancestral_field_values']+=sum(sample[k][step].size for k in ['epsilon','mean','x0_raw','x0_clipped'])
        counts['ancestral_sample_rows']+=n
    sample=load('samples');expected=np.stack([sample['trajectory'][j*8,b] for b in range(8) for j in range(5)])
    exact(load('trajectory_gallery')['images'],expected)
    denoise=load('denoising');exact(denoise['truth'],data['test_images'][:4]);exact(denoise['noise'],noise(4,11107).astype(np.float32));gallery=[[] for _ in range(4)]
    for b in range(4):gallery[b].append(denoise['truth'][b])
    for level,t_value in enumerate([1,7,15,31]):
        t=np.full(4,t_value,np.int32);exact(denoise[f'time{level}'],t)
        out,_=evaluate(final,denoise['truth'],denoise['noise'],t,table)
        compare({'noised':denoise[f'noised{level}'],'epsilon':denoise[f'epsilon{level}']},{k:out[k] for k in ['noised','epsilon']},f'denoising{level}')
        fields=reverse(table,denoise[f'noised{level}'],t,denoise[f'epsilon{level}'],np.zeros((4,8,8,1),np.float32));exact(denoise[f'x0{level}'],fields['x0_clipped'])
        for b in range(4):gallery[b].extend([denoise[f'noised{level}'][b],denoise[f'x0{level}'][b]])
    exact(denoise['gallery'],np.stack([image for row in gallery for image in row]))
    out=load('outputs');probe=load('reverse_probe');pred,back=predictor(final,out['noised'],data['test_time'])
    compare({'epsilon':probe['epsilon']},{'epsilon':pred['epsilon']},'reverse-predictor')
    fields=reverse(table,out['noised'],data['test_time'],probe['epsilon'],data['test_noise'])
    for key,value in fields.items():exact(probe[key],value)
    delta=np.zeros((200,8,8,1));delta[:,0,0,0]=1;_,input_gradient=back(delta)
    np.testing.assert_allclose(probe['input_gradient'],input_gradient,atol=2e-7,rtol=3e-4)
    errors['input_gradient']=float(abs(probe['input_gradient']-input_gradient).max());counts['reverse_input_gradient_values']=input_gradient.size
    for coordinate in range(64):
        hi,lo=[out['noised'].astype(float).copy() for _ in range(2)]
        hi.reshape(200,64)[:,coordinate]+=1e-5;lo.reshape(200,64)[:,coordinate]-=1e-5
        fd=(predictor(final,hi,data['test_time'])[0]['epsilon'][:,0,0,0]-predictor(final,lo,data['test_time'])[0]['epsilon'][:,0,0,0])/2e-5
        np.testing.assert_allclose(fd,input_gradient.reshape(200,64)[:,coordinate],atol=2e-10,rtol=2e-5)
        errors['input_finite_difference']=max(errors['input_finite_difference'],float(abs(fd-input_gradient.reshape(200,64)[:,coordinate]).max()));counts['reverse_input_direction_finite_differences']+=len(fd)
    control=load('time_control');exact(control['time'],(data['test_time']+13)%32)
    compare({'epsilon':control['epsilon']},{'epsilon':predictor(final,out['noised'],control['time'])[0]['epsilon']},'time-control')
    baseline=load('baseline');fit=data['fit_images'].astype(float).reshape(1000,64);mean=fit.mean(axis=0);var=((fit-mean)**2).mean(axis=0)
    np.testing.assert_allclose(baseline['fit_mean'],mean,atol=3e-14,rtol=0);np.testing.assert_allclose(baseline['fit_variance'],var,atol=3e-14,rtol=0)
    bar=table['alpha_bar'][data['test_time']].reshape(200,1);state=out['noised'].reshape(200,64).astype(float)
    expected=np.sqrt(1-bar)/(bar*var+1-bar)*(state-np.sqrt(bar)*mean)
    np.testing.assert_allclose(baseline['epsilon'].reshape(200,64),expected,atol=2e-13,rtol=0)
    metrics=json.loads((source/'metrics.json').read_text());terminal=table['alpha_bar'][-1]
    checked=dict(zero_epsilon_mse=(data['test_noise'].astype(float)**2).mean(),fit_gaussian_pixel_prior_epsilon_mse=((baseline['epsilon']-data['test_noise'])**2).mean(),test_fixed_noise_x0_clipped_mse=((probe['x0_clipped'].astype(float)-data['test_images'])**2).mean(),terminal_alpha_bar=terminal,fit_average_conditional_terminal_prior_kl_per_image=.5*(terminal*fit**2-terminal-np.log1p(-terminal)).sum()/1000)
    for key,value in checked.items():np.testing.assert_allclose(metrics[key],value,atol=3e-14,rtol=0,err_msg=key)
    exact(np.array(metrics['test_epsilon_mse'],np.float32),out['loss'])
    effect=float(abs(control['epsilon']-probe['epsilon']).max());np.testing.assert_allclose(metrics['time_conditioning_max_effect'],effect,atol=1e-7,rtol=0)
    assert metrics['fit_gaussian_pixel_prior_epsilon_mse']<metrics['test_epsilon_mse']<metrics['zero_epsilon_mse']
    assert metrics['optimizer_updates']==96 and metrics['selected_epoch_fixed_final']==3 and not metrics['replayed_tf_initial_arrays']
    assert audited==set(receipt['archive_sha256']),audited^set(receipt['archive_sha256'])
    return dict(passed=True,archives_checked=len(audited),counts=counts,maximum_errors=errors,raw_MNIST_IDX_sha256={str(p):sha(p) for p in paths},original_image_IDs_and_labels_all_exact=True,continuous_half_pixel_resize_not_thresholded=True,source_Float32_time_embedding_and_coefficients_preserved=True,all96_Adam_writes_bit_exact=True,all5504_parameter_coordinates_finite_differenced=True,every_ancestral_transition_formula_bit_exact_on_source_states=True,all_last_transitions_ignore_innovation_bit_exact=True,no_native_model_or_external_autodiff_or_refit=True,source_Gaussian_baseline_shortfall_preserved=True,source_metrics={k:v for k,v in metrics.items() if k!='training_seconds'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('build/nn-report-conversion/ddpm'))
    parser.add_argument('--raw-mnist',type=Path,default=Path.home()/'.cache/datamunge/datasets/mnist')
    parser.add_argument('--json',type=Path)
    args=parser.parse_args();rendered=json.dumps(check(args.root,args.raw_mnist),indent=2)+'\n';print(rendered,end='')
    if args.json:args.json.write_text(rendered)


if __name__=='__main__':main()
