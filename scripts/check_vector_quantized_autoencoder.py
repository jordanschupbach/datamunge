#!/usr/bin/env python3
"""Optional independent NumPy oracle for saved spatial VQ-VAE experiments.

Never trains, prepares data or downloads. Checks nearest assignments, surrogate
straight-through/codebook/commitment derivatives, all initial parameter gradients,
first Adam, complete saved forwards, validation selection and fitting-only prior.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact
from check_convolutional_autoencoder import convolution, convolution_adjoint, pool, repeat
from check_factor_tc_variational_autoencoder import first_update


def quantize(encoded, book):
    """Squared-distance argmin with lowest-index ties; return independent lookup values."""
    distance = (encoded*encoded).sum(1, keepdims=True)+(book*book).sum(1)[None, :]-2*encoded@book.T
    ids = np.argmin(distance, axis=1).astype(np.int32)
    quantized = book[ids]
    error = ((encoded-quantized)**2).mean()
    return dict(indices=ids, quantized=quantized, straight_through=quantized,
                codebook_loss=error, commitment_loss=error)


def decode(p, codes):
    """Code-vector-only repeat/convolution Bernoulli decoder and its adjoint cache."""
    enlarged = repeat(codes.reshape(-1, 7, 7, 4))
    hidden = np.maximum(convolution(enlarged, p['decoder1/W'], p['decoder1/b']), 0)
    second = repeat(hidden)
    logits = convolution(second, p['output/W'], p['output/b']).reshape(-1, 784)
    return 1/(1+np.exp(-logits)), (enlarged, hidden, second, logits)


def forward(p, images):
    """Independent spatial encoder, nearest code and pixel/latent reductions."""
    images = images.astype(np.float64)
    c1 = np.maximum(convolution(images, p['encoder1/W'], p['encoder1/b']), 0)
    p1 = pool(c1)
    c2 = np.maximum(convolution(p1, p['encoder2/W'], p['encoder2/b']), 0)
    encoded = pool(c2).reshape(-1, 4)
    out = dict(encoded=encoded, **quantize(encoded, p['codebook']))
    probability, cache = decode(p, out['quantized'])
    logits = cache[-1]
    nll = (np.logaddexp(0, logits)-images.reshape(-1, 784)*logits).sum(1)
    out.update(probability=probability, nll=nll, reconstruction_loss=nll.mean(),
               loss=nll.mean()+out['codebook_loss']+.25*out['commitment_loss'])
    return out, (c1, p1, c2, cache)


def derivatives(p, images):
    """Explicit surrogate chain rules; reconstruction skips codebook, commitment skips lookup."""
    images = images.astype(np.float64)
    out, (c1, p1, c2, (enlarged, hidden, second, _)) = forward(p, images)
    delta = ((out['probability']-images.reshape(-1, 784))/len(images)).reshape(images.shape)
    expected = {}
    g, expected['output/W'], expected['output/b'] = convolution_adjoint(second, p['output/W'], delta)
    g = 4*pool(g)*(hidden > 0)
    g, expected['decoder1/W'], expected['decoder1/b'] = convolution_adjoint(enlarged, p['decoder1/W'], g)
    # The quantized forward value is exact; reconstruction's surrogate Jacobian to encoded is identity.
    g = (4*pool(g)).reshape(-1, 4)
    residual = out['encoded']-out['quantized']
    g += .5*residual/residual.size
    expected['codebook'] = np.zeros_like(p['codebook'])
    np.add.at(expected['codebook'], out['indices'], -2*residual/residual.size)
    g = repeat(g.reshape(-1, 7, 7, 4))/4*(c2 > 0)
    g, expected['encoder2/W'], expected['encoder2/b'] = convolution_adjoint(p1, p['encoder2/W'], g)
    g = repeat(g)/4*(c1 > 0)
    _, expected['encoder1/W'], expected['encoder1/b'] = convolution_adjoint(images, p['encoder1/W'], g)
    return expected


def factor_data(data):
    """Check every source identity, factor, residue split and generated binary pixel."""
    identities, pictures, pixels = set(), set(), 0
    yy, xx = np.mgrid[:28, :28]
    for split, count in [('fit', 210), ('validation', 42), ('test', 42)]:
        ids, factors, images = [data[split+s] for s in ['_source_ids', '_factors', '_images']]
        assert len(ids) == count and not identities.intersection(ids)
        identities.update(ids)
        for row, identity in enumerate(ids):
            shape, size, x, y = int(identity)//147, (int(identity)//49)%3, (int(identity)//7)%7, int(identity)%7
            exact(factors[row], np.array([shape, size, x, y], dtype=np.int32))
            residue = (x+2*y+size+3*shape)%7
            assert residue == 0 if split == 'test' else residue == 1 if split == 'validation' else residue >= 2
            dx, dy = abs(xx-(3*x+5)), abs(yy-(3*y+5))
            value = np.maximum(dx, dy) <= size+2 if shape else dx*dx+dy*dy <= (size+2)**2
            exact(images[row], value.astype(np.float32)[..., None]);pictures.add(images[row].tobytes())
        pixels += images.size
    assert len(identities) == len(pictures) == 294
    return pixels


def markov(fit):
    """Independent counts with per-image BOS16, then source-order Float64 half-count smoothing."""
    counts = np.zeros((17, 16), dtype=np.int32)
    for row in fit:
        previous = 16
        for code in row:
            counts[previous, code] += 1;previous = int(code)
    return counts, (counts.astype(np.float64)+.5)/(counts.sum(1, keepdims=True)+8.)


def markov_nll(probability, ids):
    """Mean nats per complete grid, with BOS reset at each image."""
    total = 0.
    for row in ids:
        previous = 16
        for code in row:
            total -= np.log(probability[previous, code]);previous = int(code)
    return total/len(ids)


def check(directory):
    """Validate original or report artifacts and return measured numerical evidence."""
    data = arrays(directory/'data.npz');pixels = factor_data(data)
    algebra = arrays(directory/'quantizer_algebra.npz')
    z, book = algebra['encoded'].astype(np.float64), algebra['codebook'].astype(np.float64)
    q = quantize(z, book);exact(q['indices'], np.array([0, 2], dtype=np.int32))
    for field in ['indices', 'quantized', 'straight_through']:exact(algebra[field], q[field].astype(algebra[field].dtype))
    for field in ['reconstruction_dbook', 'dictionary_dencoded', 'commitment_dbook']:exact(algebra[field], np.zeros_like(algebra[field]))
    exact(algebra['reconstruction_dencoded'], algebra['quantized'])
    np.testing.assert_allclose(algebra['commitment_dencoded'], .5*(z-q['quantized'])/z.size, atol=1e-8, rtol=1e-7)
    dictionary = np.zeros_like(book);np.add.at(dictionary, q['indices'], 2*(q['quantized']-z)/z.size)
    np.testing.assert_allclose(algebra['dictionary_dbook'], dictionary, atol=1e-8, rtol=1e-7)
    initial = arrays(directory/'initial_arrays.npz');p = {k:v.astype(np.float64) for k,v in initial.items()}
    batch = arrays(directory/'first_batch.npz')['images'];actual = arrays(directory/'first_gradients.npz')
    expected = derivatives(p, batch);assert set(expected) == set(actual)
    gradient_error = 0
    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], value, atol=8e-6, rtol=3e-4)
        gradient_error = max(gradient_error, float(np.max(np.abs(value-actual[key]))))
    first_update(initial, actual, arrays(directory/'first_update_arrays.npz'))
    maximum, fields, values = 0., 0, 0
    def endpoints(out, saved):
        nonlocal maximum, fields, values
        assert set(out) == set(saved)
        for key, value in out.items():
            value = np.asarray(value)
            if key == 'indices':exact(saved[key], value)
            else:
                np.testing.assert_allclose(saved[key], value, atol=6e-5, rtol=1e-5)
                maximum = max(maximum, float(np.max(np.abs(value-saved[key]))))
            fields += 1;values += value.size
    out, _ = forward(p, batch);endpoints(out, arrays(directory/'first_forward.npz'))
    selected = arrays(directory/'model_arrays.npz');p = {k:v.astype(np.float64) for k,v in selected.items()}
    saved = arrays(directory/'outputs.npz')
    for split in ['fit', 'validation', 'test']:
        out, _ = forward(p, data[split+'_images'])
        endpoints(out, {k:saved[split+'_'+k] for k in out})
        grid = out['indices'].reshape(-1, 49);exact(saved[split+'_index_grid'], grid)
        counts = np.bincount(grid.ravel(), minlength=16).astype(np.int32);exact(saved[split+'_code_counts'], counts)
    history = np.genfromtxt(directory/'history.tsv', delimiter='\t', names=True);criteria = []
    for epoch in range(1, 4):
        ep = {k:v.astype(np.float64) for k,v in arrays(directory/f'epoch{epoch}_arrays.npz').items()}
        out, _ = forward(ep, data['validation_images']);criteria.append(out['reconstruction_loss'])
        for column, key in [('validation_nll', 'reconstruction_loss'), ('validation_codebook_loss', 'codebook_loss'), ('validation_commitment_loss', 'commitment_loss'), ('validation_objective', 'loss')]:
            np.testing.assert_allclose(history[column][epoch-1], out[key], atol=6e-5, rtol=1e-5)
    meta = json.loads((directory/'metrics.json').read_text());chosen = int(np.argmin(criteria))+1
    assert meta['selected_epoch'] == chosen and meta['updates'] == 21 and meta['epochs'] == 3
    for k,v in arrays(directory/f'epoch{chosen}_arrays.npz').items():exact(v, selected[k])
    batches = np.genfromtxt(directory/'batches.tsv', delimiter='\t', names=True, dtype=np.int64)
    assert len(batches) == 630
    for epoch in range(1, 4):
        rows = batches[batches['epoch'] == epoch];exact(np.sort(rows['row']), np.arange(210))
        exact(rows['source_id'], data['fit_source_ids'][rows['row']].astype(np.int64))
        exact(rows['step'], np.repeat(np.arange((epoch-1)*7, epoch*7), [32]*6+[18]))
    prior = arrays(directory/'prior.npz');counts, probability = markov(saved['fit_index_grid'])
    exact(prior['counts'], counts);exact(prior['probabilities'], probability)
    rng = np.random.RandomState(2729);u = (rng.randint(0, 2**32, (20, 49), dtype=np.uint32).astype(np.float64)+.5)/2**32
    exact(prior['uniform'], u)
    ids = np.empty((20, 49), dtype=np.int32)
    for row in range(20):
        previous = 16
        for t in range(49):
            ids[row,t] = min(int(np.searchsorted(np.cumsum(probability[previous]), u[row,t], side='right')), 15)
            previous = ids[row,t]
    exact(prior['sample_ids'], ids)
    prediction, _ = decode(p, p['codebook'][ids.ravel()]);np.testing.assert_allclose(prior['probability'], prediction, atol=5e-6, rtol=1e-5)
    for split in ['fit', 'test']:
        np.testing.assert_allclose(meta['prior_'+split+'_nll_nats_per_grid'], markov_nll(probability, saved[split+'_index_grid']), atol=1e-12, rtol=1e-12)
    for split in ['fit', 'test']:
        c = saved[split+'_code_counts'];v = c[c>0]/c.sum();perplexity = np.exp(-(v*np.log(v)).sum())
        np.testing.assert_allclose(meta[split+'_code_perplexity'], perplexity, atol=1e-14, rtol=1e-14)
    assert meta['fit_used_codes'] == int((saved['fit_code_counts']>0).sum())
    test = data['test_images'].reshape(-1,784).astype(np.float64)
    np.testing.assert_allclose(meta['test_mse'], ((saved['test_probability']-test)**2).mean(), atol=5e-13, rtol=1e-14)
    fitmean = data['fit_images'].reshape(-1,784).astype(np.float64).mean(0).astype(np.float32)
    np.testing.assert_allclose(meta['fit_mean_test_mse'], ((fitmean-test)**2).mean(), atol=5e-13, rtol=1e-14)
    np.testing.assert_allclose(meta['test_nll'], saved['test_reconstruction_loss'], atol=0, rtol=0)
    np.testing.assert_allclose(meta['test_objective'], saved['test_loss'], atol=0, rtol=0)
    return dict(passed=True, parameter_derivatives_checked=sum(v.size for v in expected.values()),
                first_gradient_max_error=gradient_error, first_Adam_update_bit_exact=True,
                lattice_pixels_checked=pixels, endpoint_fields_checked=fields, endpoint_values_checked=values,
                maximum_endpoint_error=maximum, eligible_validation_checkpoints_checked=3, selected_epoch=chosen,
                batch_history_rows_checked=630, fitting_prior_counts_checked=counts.size,
                explicit_uniforms_checked=u.size, unfiltered_prior_images_checked=20)


def main():
    """Check an already completed experiment and optionally save its measured receipt."""
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('directory', type=Path);parser.add_argument('--json', type=Path)
    args = parser.parse_args();text = json.dumps(check(args.directory), indent=2)+'\n'
    if args.json:args.json.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
