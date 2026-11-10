#!/usr/bin/env python3
"""Independent NumPy Row-LSTM density, derivatives, causal guards and Adam audit.

Reads the pinned original records and captured native partials; never fits a
substitute model or prepares a different split. All model math runs in Double.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from check_adversarial_autoencoder import arrays, exact
from check_gated_pixel_cnn import conv
from check_contractive_autoencoder import sigmoid


def mask(shape, type_b):
    value = np.zeros(shape)
    value[0] = 1
    value[1, 0] = 1
    if type_b:
        value[1, 1] = 1
    return value


def network(p, image):
    """One vertically recurrent width4 Row LSTM with a1x3 hidden-row kernel."""
    structural = mask(p['input/W'].shape, False)
    input_gates, bi = conv(image, p['input/W']*structural)
    input_gates += p['input/b']
    h = input_gates[:,:1,:,:4]*0.
    c = h
    states, cells, steps = [], [], []
    for y in range(image.shape[1]):
        recurrent, br = conv(h, p['row/W'])
        gates = input_gates[:,y:y+1]+recurrent
        i, f = sigmoid(gates[:,:,:,:4]),sigmoid(gates[:,:,:,4:8]+1)
        o, g = sigmoid(gates[:,:,:,8:12]),np.tanh(gates[:,:,:,12:])
        previous_cell = c
        c = f*c+i*g
        h = o*np.tanh(c)
        steps.append((i,f,o,g,previous_cell,c,br))
        states.append(h);cells.append(c)
    hidden, cell = np.concatenate(states,axis=1),np.concatenate(cells,axis=1)
    logits, bo = conv(hidden,p['output/W'])
    logits += p['output/b']
    out = dict(logits=logits,hidden=hidden,cell=cell,input_gates=input_gates)

    def adjoint(dlogits):
        dh_all, dw = bo(dlogits)
        partials = {'output/W':dw,'output/b':dlogits.sum(axis=(0,1,2)),'row/W':np.zeros_like(p['row/W'])}
        dh, dc = np.zeros_like(h),np.zeros_like(h)
        din = np.zeros_like(input_gates)
        for y in reversed(range(image.shape[1])):
            i,f,o,g,old_c,c,br = steps[y]
            dh = dh+dh_all[:,y:y+1]
            tc = np.tanh(c)
            dcell = dc+dh*o*(1-tc*tc)
            dgates = np.concatenate([dcell*g*i*(1-i),dcell*old_c*f*(1-f),
                                    dh*tc*o*(1-o),dcell*i*(1-g*g)],axis=3)
            din[:,y:y+1] += dgates
            dh, dw = br(dgates)
            partials['row/W'] += dw
            dc = dcell*f
        # The source's shared initial h/c are first input features times zero.
        din[:,:1,:,:4] += (dh+dc)*0.
        dx, dw = bi(din)
        partials['input/W'] = dw*structural
        partials['input/b'] = din.sum(axis=(0,1,2))
        return partials,dx
    return out,adjoint


def forward(p, image):
    out, adjoint = network(p, image)
    logits = out['logits']
    nll = np.logaddexp(0, logits) - image * logits
    out.update(probability=sigmoid(logits), per_image_nll=nll.sum(axis=(1, 2, 3)), loss=nll.mean())
    return out, adjoint((out['probability'] - image) / image.size)[0]


def uniforms(n, seed):
    return (np.random.RandomState(seed).randint(0, 2**32, (n, 64), dtype=np.uint32).astype(float) + .5) / 2**32


def check(root, prepared):
    source = root/'original-output'
    receipt = json.loads((root/'baseline.json').read_text())
    for file, digest in receipt['source_pins'].items():
        frozen = root/('original-'+Path(file).name)
        assert hashlib.sha256(frozen.read_bytes()).hexdigest() == digest, frozen
    for file, digest in receipt['prepared_input_sha256'].items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == digest, file
    for file, digest in receipt['archive_sha256'].items():
        assert hashlib.sha256((source/file).read_bytes()).hexdigest() == digest, file
    raw, data = arrays(prepared), arrays(source/'data.npz')
    resized_values = 0
    for split in ['fit', 'validation', 'test']:
        image = (raw[split+'_images'].astype(np.float32) / np.float32(255)).reshape(-1, 28, 28, 1)
        exact(data[split+'_original'], image)
        for key in ['labels', 'source_ids']:
            exact(data[split+'_'+key], raw[split+'_'+key])
        gray = np.zeros((len(image), 8, 8, 1))
        for y in range(8):
            for x in range(8):
                py, px = (y+.5)*3.5-.5, (x+.5)*3.5-.5
                iy, ix = int(py), int(px)
                for dy in [0, 1]:
                    for dx in [0, 1]:
                        gray[:, y, x] += ((py-iy if dy else 1-py+iy) *
                                         (px-ix if dx else 1-px+ix) * image[:, iy+dy, ix+dx].astype(float))
        exact(data[split+'_gray'], gray.astype(np.float32))
        exact(data[split+'_images'], (gray.astype(np.float32) >= np.float32(.5)).astype(np.float32))
        resized_values += gray.size
    p0 = arrays(source/'initial_arrays.npz')
    pf = arrays(source/'final.npz')
    assert len(p0) == 5 and sum(v.size for v in p0.values()) == 357
    masks = {'input/W':mask(p0['input/W'].shape,False)}
    assert sum(np.count_nonzero(value == 0) for value in masks.values()) == 80
    batch = np.genfromtxt(source/'batches.tsv', names=True, delimiter='\t', dtype=np.int64)
    assert len(batch) == 3000
    for epoch in [1, 2, 3]:
        rows = batch[batch['epoch'] == epoch]
        exact(np.sort(rows['row']), np.arange(1000))
        exact(rows['source_id'], data['fit_source_ids'][rows['row']].astype(np.int64))
        assert len(np.unique(rows['step'])) == 32
    first = arrays(source/'first_batch.npz')['images'].astype(float)
    exact(first, data['fit_images'][batch[batch['step'] == 0]['row']].astype(float))
    counts = dict(forward_values=0, gradient_values=0, optimizer_writes=0,
                  exact_weight_values=0, finite_differences=0, ancestral_pixels=0,
                  causal_jacobian_entries=4096, resized_pixels_exact=resized_values)
    errors = dict(forward=0., gradient=0., conditional=0.)

    def compare(stored, expected, label, gradient=False):
        assert set(stored) == set(expected), (label, set(stored)^set(expected))
        kind = 'gradient' if gradient else 'forward'
        for key, value in expected.items():
            np.testing.assert_allclose(stored[key], value, atol=4e-6 if gradient else 3e-5,
                                       rtol=2e-3 if gradient else 2e-4, err_msg=label+'/'+key)
            counts[kind+'_values'] += np.size(value)
            errors[kind] = max(errors[kind], float(abs(stored[key]-value).max()))

    for name, parameters, image in [('first_forward', p0, first),
                                    ('initial_outputs', p0, data['test_images']),
                                    ('outputs', pf, data['test_images'])]:
        out, _ = forward({key: v.astype(float) for key, v in parameters.items()}, image.astype(float))
        compare(arrays(source/(name+'.npz')), out, name)
    before = p0
    moment = {key: np.zeros_like(value) for key, value in p0.items()}
    variance = {key: np.zeros_like(value) for key, value in p0.items()}
    for step in range(96):
        rows = batch[batch['step'] == step]['row']
        assert len(rows) == (8 if step % 32 == 31 else 32)
        image = data['fit_images'][rows].astype(float)
        _, partials = forward({key: v.astype(float) for key, v in before.items()}, image)
        gradient = arrays(root/'phases'/f'batch{step}-gradient.npz')
        compare(gradient, partials, f'batch{step}', True)
        after = arrays(root/'phases'/f'batch{step}-after.npz')
        assert set(after) == set(before)
        for key, value in before.items():
            g = gradient[key]
            moment[key] = moment[key]*np.float32(.9) + g*np.float32(.1)
            variance[key] = variance[key]*np.float32(.999) + (g*g)*np.float32(.001)
            m = moment[key]/np.float32(1-.9**(step+1))
            v = variance[key]/np.float32(1-.999**(step+1))
            exact(value-m/(np.sqrt(v)+np.float32(1e-8))*np.float32(.003), after[key])
            counts['exact_weight_values'] += value.size
            if key in masks:
                forbidden = masks[key] == 0
                assert np.all(g[forbidden] == 0) and np.all(partials[key][forbidden] == 0)
                exact(after[key][forbidden], p0[key][forbidden])
        exact(arrays(root/'phases'/f'batch{step}-iterations.npz')['iterations'], np.array([step+1], np.int64))
        if step == 0:
            compare(arrays(source/'initial_gradients.npz'), partials, 'initial', True)
            for key in after:
                exact(after[key], arrays(source/'first_after.npz')[key])
        if step % 32 == 31:
            for key, value in arrays(source/f'epoch{1+step//32}_arrays.npz').items():
                exact(after[key], value)
        before = after
        counts['optimizer_writes'] += 1
    for key in pf:
        exact(pf[key], before[key])
    p = {key: value.astype(float) for key, value in pf.items()}
    _, derivatives = forward(p, first)
    compare(arrays(source/'final_gradients.npz'), derivatives, 'final', True)
    # Smooth Row-LSTM scalar finite differences for every parameter leaf.
    for key, partial in derivatives.items():
        for index in sorted({0, partial.size//2, partial.size-1}):
            plus, minus = dict(p), dict(p)
            plus[key], minus[key] = p[key].copy(), p[key].copy()
            plus[key].flat[index] += 1e-6
            minus[key].flat[index] -= 1e-6
            finite = (forward(plus, first)[0]['loss'] - forward(minus, first)[0]['loss'])/2e-6
            np.testing.assert_allclose(partial.flat[index], finite, atol=1e-8, rtol=1e-3, err_msg=key)
            counts['finite_differences'] += 1
    cause = arrays(source/'causality.npz')
    image = data['test_images'][:1].astype(float)
    exact(image, cause['images'].astype(float))
    initial, adjoint = network(p, image)
    probability = sigmoid(initial['logits'])
    np.testing.assert_allclose(cause['probability'], probability, atol=2e-6, rtol=1e-4)
    for site in range(64):
        delta = np.zeros_like(image)
        delta.flat[site] = 1
        _, dx = adjoint(delta)
        np.testing.assert_allclose(cause['jacobian'][site], dx.reshape(-1), atol=3e-6, rtol=2e-4)
        assert np.all(dx.reshape(-1)[site:] == 0)
        changed = image.copy().reshape(-1)
        changed[site:] = 1-changed[site:]
        changed = changed.reshape(image.shape)
        flipped = sigmoid(network(p, changed)[0]['logits'])
        exact(flipped.reshape(-1)[:site+1], probability.reshape(-1)[:site+1])
        if site % 8 == 0:
            exact(cause[f'changed_input{site}'], changed.astype(np.float32))
            np.testing.assert_allclose(cause[f'changed_probability{site}'], flipped, atol=2e-6, rtol=1e-4)
    assert np.all(np.triu(cause['jacobian']) == 0)
    history = np.genfromtxt(source/'history.tsv', names=True, delimiter='\t')
    for epoch in [1, 2, 3]:
        pe = {key: v.astype(float) for key, v in arrays(source/f'epoch{epoch}_arrays.npz').items()}
        loss = forward(pe, data['validation_images'].astype(float))[0]['loss']
        np.testing.assert_allclose(history[epoch-1][1], loss, atol=3e-6, rtol=1e-5)
        np.testing.assert_allclose(history[epoch-1][2], loss/np.log(2), atol=5e-6, rtol=1e-5)
    smallest_margin = 1.
    for filename, parameters, n, seed, prefix in [
            (f'epoch{epoch}_samples', arrays(source/f'epoch{epoch}_arrays.npz'), 16, 8329, 0) for epoch in [1, 2, 3]] + [
            ('samples', pf, 32, 8329, 0), ('completions', pf, 8, 8353, 32)]:
        stored = arrays(source/(filename+'.npz'))
        u = uniforms(n, seed)
        exact(stored['uniform'], u)
        canvas = np.zeros((n, 8, 8, 1))
        if prefix:
            exact(stored['truth'], data['test_images'][:n])
            canvas.reshape(n, 64)[:, :prefix] = data['test_images'][:n].reshape(n, 64)[:, :prefix]
        assert np.all(stored['conditionals'][:, :prefix] == 0)
        params = {key: value.astype(float) for key, value in parameters.items()}
        for site in range(prefix, 64):
            q = sigmoid(network(params, canvas)[0]['logits']).reshape(n, 64)[:, site]
            native = stored['conditionals'][:, site]
            np.testing.assert_allclose(native, q, atol=3e-6, rtol=1e-4, err_msg=filename+str(site))
            errors['conditional'] = max(errors['conditional'], float(abs(native-q).max()))
            smallest_margin = min(smallest_margin, float(abs(u[:, site]-q).min()))
            # The independently rounded probability and captured native probability
            # must select the same branch for every retained midpoint uniform.
            exact((u[:, site] < q), (u[:, site] < native))
            canvas.reshape(n, 64)[:, site] = (u[:, site] < q)
            counts['ancestral_pixels'] += n
        exact(canvas.astype(np.float32), stored['images'])
    gallery = arrays(source/'completion_gallery.npz')['images']
    complete = arrays(source/'completions.npz')
    expected = np.stack([complete['truth'], np.concatenate([complete['truth'].reshape(8,64)[:,:32],
                        np.full((8,32),.5,np.float32)],axis=1).reshape(8,8,8,1), complete['images']],axis=1).reshape(24,8,8,1)
    exact(gallery, expected)
    probability = (data['fit_images'].astype(float).sum(axis=0).reshape(-1)+.5)/1001
    exact(arrays(source/'baseline.npz')['probability'], probability.astype(np.float32))
    test = data['test_images'].astype(float).reshape(-1,64)
    # Match the source's serial Double reduction separately from NumPy's
    # pairwise reduction; both use the unrounded Double posterior.
    nll = 0.
    for row in test:
        for j in range(64):
            nll -= np.log(probability[j]) if row[j] else np.log1p(-probability[j])
    nll /= test.size
    metrics = json.loads((source/'metrics.json').read_text())
    np.testing.assert_allclose(metrics['fit_bernoulli_baseline_nll'], nll, atol=2e-14, rtol=0)
    np.testing.assert_allclose(metrics['fit_bernoulli_baseline_bits'], nll/np.log(2), atol=3e-14, rtol=0)
    assert metrics['optimizer_updates'] == 96 and metrics['selected_epoch_fixed_final'] == 3
    assert metrics['test_nll_per_binary_pixel'] == float(arrays(source/'outputs.npz')['loss'])
    assert metrics['test_bits_per_binary_pixel'] > metrics['fit_bernoulli_baseline_bits']
    return dict(passed=True, counts=counts, maximum_errors=errors, all_96_adam_writes_bit_exact=True,
                masked_80_weights_and_all_partials_bit_exact=True, all_64_future_flips_exact=True,
                minimum_sampling_margin=smallest_margin, batch_rows_checked=len(batch), metrics=metrics,
                baseline_uses_unrounded_double_posterior=True, report_must_preserve_worse_than_baseline_result=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('build/nn-report-conversion/pixel_rnn'))
    parser.add_argument('--data', type=Path, default=Path('build/nn-report-conversion/datasets/saliency/saliency_data.npz'))
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    result = check(args.root, args.data)
    rendered = json.dumps(result, indent=2)+'\n'
    print(rendered, end='')
    if args.json:
        args.json.write_text(rendered)


if __name__ == '__main__':
    main()
