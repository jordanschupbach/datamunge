#!/usr/bin/env python3
"""Optional independent NumPy oracle for the spatial hierarchy/miniature-NVAE report.

No fitting or preparation. Checks all first derivatives/Adam updates, the first
six moving-buffer updates, every saved posterior/prior field, Gaussian banks,
noise interventions, isolated spectral gradients and all eligible checkpoints.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact, gaussian
from check_convolutional_autoencoder import pool, repeat
from check_factor_tc_variational_autoencoder import first_update


def split(saved):
    """Separate source trainable leaves and non-optimized state-prefixed arrays."""
    return ({k: v.astype(np.float64) for k, v in saved.items() if not k.startswith('state/')},
            {k[6:]: v.astype(np.float64) for k, v in saved.items() if k.startswith('state/')})


def conv(x, w, b=None, depthwise=False):
    """Independent same-padded NHWC correlation for1x1/3x3 or depthwise kernels."""
    ky, kx = w.shape[:2];hy, hx = ky//2, kx//2
    padded = np.pad(x, ((0, 0), (hy, hy), (hx, hx), (0, 0)))
    output_channels = x.shape[-1] if depthwise else w.shape[-1]
    result = np.zeros((*x.shape[:3], output_channels))
    for y in range(ky):
        for col in range(kx):
            window = padded[:, y:y+x.shape[1], col:col+x.shape[2]]
            result += window*w[y, col, :, 0] if depthwise else np.einsum('nhwc,co->nhwo', window, w[y, col])
    return result if b is None else result+b


def conv_back(x, w, gradient, depthwise=False):
    """All shared-kernel/input derivatives, including channel-wise depthwise scatter."""
    ky, kx = w.shape[:2];hy, hx = ky//2, kx//2
    padded = np.pad(x, ((0, 0), (hy, hy), (hx, hx), (0, 0)))
    dx, dw = np.zeros_like(padded), np.zeros_like(w)
    h, width = x.shape[1:3]
    for y in range(ky):
        for col in range(kx):
            window = padded[:, y:y+h, col:col+width]
            if depthwise:
                dw[y, col, :, 0] = (window*gradient).sum(axis=(0, 1, 2))
                dx[:, y:y+h, col:col+width] += gradient*w[y, col, :, 0]
            else:
                dw[y, col] = np.einsum('nhwc,nhwo->co', window, gradient)
                dx[:, y:y+h, col:col+width] += np.einsum('nhwo,co->nhwc', gradient, w[y, col])
    return dx[:, hy:hy+h, hx:hx+width], dw, gradient.sum(axis=(0, 1, 2))


def swish(x):
    """Unit-slope Swish and its analytic local derivative."""
    s = 1/(1+np.exp(-x))
    return x*s, s+x*s*(1-s)


def bn(p, state, name, x, training):
    """Population normalization and separate unbiased moments for moving-state comparison."""
    count = np.prod(x.shape[:-1]);gamma, beta = p[name+'/gamma'], p[name+'/beta']
    mean = x.mean(axis=(0, 1, 2)) if training else state[name+'/moving_mean']
    variance = ((x-mean)**2).mean(axis=(0, 1, 2)) if training else state[name+'/moving_var']
    inv = 1/np.sqrt(variance+.001);normalized = (x-mean)*inv
    return normalized*gamma+beta, (normalized, inv, gamma, training), (mean, variance*count/(count-1) if count > 1 else variance)


def bn_back(cache, gradient):
    """Batch/inference input and affine-parameter adjoints, with no moving-state gradient."""
    xhat, inv, gamma, training = cache
    db, dg = gradient.sum(axis=(0, 1, 2)), (gradient*xhat).sum(axis=(0, 1, 2))
    if training:
        count = np.prod(gradient.shape[:-1])
        dx = gamma*inv/count*(count*gradient-db-xhat*dg)
    else:
        dx = gradient*gamma*inv
    return dx, dg, db


def top_feature(p, state, top, nvae, training):
    """Top projection plus ordinary Swish convolution or the full miniature BN/SE cell."""
    projected = (top@p['top_projection/W']+p['top_projection/b']).reshape(-1, 7, 7, 4)
    if not nvae:
        a = conv(projected, p['cell/W'], p['cell/b']);feature, derivative = swish(a)
        return feature, (projected, derivative), {}
    expand_input = conv(projected, p['cell/expand/W'])
    normalized, be, me = bn(p, state, 'cell/expand_bn', expand_input, training)
    expand, de = swish(normalized)
    depth_input = conv(expand, p['cell/depthwise/W'], depthwise=True)
    normalized, bd, md = bn(p, state, 'cell/depth_bn', depth_input, training)
    depth, dd = swish(normalized)
    project_input = conv(depth, p['cell/project/W'])
    project, bp, mp = bn(p, state, 'cell/project_bn', project_input, training)
    pooled = project.mean(axis=(1, 2));hidden_pre = pooled@p['cell/se_reduce/W']+p['cell/se_reduce/b']
    hidden, ds = swish(hidden_pre);gate = 1/(1+np.exp(-(hidden@p['cell/se_expand/W']+p['cell/se_expand/b'])))
    feature = projected+.1*(project*gate[:, None, None, :])
    cache = (projected, expand, depth, project, pooled, hidden, gate, de, dd, ds, be, bd, bp)
    moments = dict(zip(['cell/expand_bn', 'cell/depth_bn', 'cell/project_bn'], [me, md, mp]))
    return feature, cache, moments


def top_feature_back(p, cache, gradient, nvae):
    """Complete live top-context adjoint, including BN, squeeze/excitation and residual branch."""
    result = {}
    if not nvae:
        projected, derivative = cache
        dx, result['cell/W'], result['cell/b'] = conv_back(projected, p['cell/W'], gradient*derivative)
        return dx.reshape(len(dx), -1), result
    projected, expand, depth, project, pooled, hidden, gate, de, dd, ds, be, bd, bp = cache
    dproject = .1*gradient*gate[:, None, None, :]
    dgate = .1*(gradient*project).sum(axis=(1, 2))*gate*(1-gate)
    result.update({'cell/se_expand/W': hidden.T@dgate, 'cell/se_expand/b': dgate.sum(axis=0)})
    dh = (dgate@p['cell/se_expand/W'].T)*ds
    result.update({'cell/se_reduce/W': pooled.T@dh, 'cell/se_reduce/b': dh.sum(axis=0)})
    dproject += (dh@p['cell/se_reduce/W'].T)[:, None, None, :]/49
    dproject, result['cell/project_bn/gamma'], result['cell/project_bn/beta'] = bn_back(bp, dproject)
    ddepth, result['cell/project/W'], _ = conv_back(depth, p['cell/project/W'], dproject)
    ddepth, result['cell/depth_bn/gamma'], result['cell/depth_bn/beta'] = bn_back(bd, ddepth*dd)
    dexpand, result['cell/depthwise/W'], _ = conv_back(expand, p['cell/depthwise/W'], ddepth, True)
    dexpand, result['cell/expand_bn/gamma'], result['cell/expand_bn/beta'] = bn_back(be, dexpand*de)
    dx, result['cell/expand/W'], _ = conv_back(projected, p['cell/expand/W'], dexpand)
    return (dx+gradient).reshape(len(dx), -1), result


def decode(p, lower, feature):
    """Image-free decoder with explicit lower samples and sampled-top features."""
    joined = np.concatenate([feature, lower.reshape(-1, 7, 7, 1)], axis=3)
    combine_pre = conv(joined, p['combine/W'], p['combine/b']);combined, dc = swish(combine_pre)
    enlarged = repeat(combined);decoder_pre = conv(enlarged, p['decoder14/W'], p['decoder14/b'])
    decoder, dd = swish(decoder_pre);second = repeat(decoder)
    logits = conv(second, p['output/W'], p['output/b']).reshape(-1, 784)
    return logits, 1/(1+np.exp(-logits)), (joined, enlarged, second, dc, dd)


def forward(p, state, image, epsilon, nvae, training=False):
    """All nineteen hierarchy endpoints with live direct/residual Gaussian conditioning."""
    image, epsilon = image.astype(np.float64).reshape(-1, 784), epsilon.astype(np.float64)
    x = image.reshape(-1, 28, 28, 1)
    c1, d1 = swish(conv(x, p['encoder1/W'], p['encoder1/b']));p1 = pool(c1)
    c2, d2 = swish(conv(p1, p['encoder2/W'], p['encoder2/b']));encoder = pool(c2);flat = encoder.reshape(-1, 196)
    top_mu, top_lv = flat@p['top_mu/W']+p['top_mu/b'], flat@p['top_lv/W']+p['top_lv/b']
    top_sigma = np.exp(.5*top_lv);top_z = top_mu+top_sigma*epsilon[:, 49:]
    feature, cell, moments = top_feature(p, state, top_z, nvae, training)
    prior_mu = conv(feature, p['prior_mu/W'], p['prior_mu/b']).reshape(-1, 49)
    prior_lv = conv(feature, p['prior_lv/W'], p['prior_lv/b']).reshape(-1, 49)
    fusion = np.concatenate([encoder, feature], axis=3)
    posterior, dp = swish(conv(fusion, p['posterior_hidden/W'], p['posterior_hidden/b']))
    heads = conv(posterior, p['posterior_heads/W'], p['posterior_heads/b'])
    hm, hl = heads[:, :, :, 0].reshape(-1, 49), heads[:, :, :, 1].reshape(-1, 49)
    mu, lv = (prior_mu+hm, prior_lv+hl) if nvae else (hm, hl)
    lower_sigma = np.exp(.5*lv);lower_z = mu+lower_sigma*epsilon[:, :49]
    logits, probability, decoder = decode(p, lower_z, feature)
    nll = (np.logaddexp(0, logits)-image*logits).sum(axis=1)
    kl_lower = .5*((prior_lv-lv)+(np.exp(lv-prior_lv)+(mu-prior_mu)**2*np.exp(-prior_lv))-1).sum(axis=1)
    kl_top = .5*(top_mu*top_mu+np.exp(top_lv)-1-top_lv).sum(axis=1)
    regularizer = 1e-4*sum(np.sqrt((p[k]*p[k]).sum()+1e-12) for k in ['prior_mu/W', 'prior_lv/W']) if nvae else 0.
    loss = np.mean(nll+(kl_lower+kl_top))
    return dict(encoder_feature=encoder, top_feature=feature, top_mu=top_mu, top_lv=top_lv, top_z=top_z,
        prior_mu=prior_mu, prior_lv=prior_lv, head_mu=hm, head_lv=hl, lower_mu=mu, lower_lv=lv,
        lower_z=lower_z, probability=probability, nll=nll, kl_lower=kl_lower, kl_top=kl_top,
        loss=np.array(loss), regularizer=np.array(regularizer), train_loss=np.array(loss+regularizer)), \
        (x, p1, d1, d2, flat, top_sigma, lower_sigma, cell, moments, fusion, posterior, dp, decoder)


def derivatives(p, state, image, epsilon, nvae):
    """All26/37 leaves, including residual prior paths, BN/SE and two isolated spectral terms."""
    image = image.astype(np.float64).reshape(-1, 784);epsilon = epsilon.astype(np.float64)
    o, cache = forward(p, state, image, epsilon, nvae, True)
    x, p1, d1, d2, flat, top_sigma, lower_sigma, cell, moments, fusion, posterior, dp, decoder = cache
    joined, enlarged, second, dc, dd = decoder;count = len(image);result = {}
    delta = ((o['probability']-image)/count).reshape(-1, 28, 28, 1)
    g, result['output/W'], result['output/b'] = conv_back(second, p['output/W'], delta)
    g = 4*pool(g)*dd
    g, result['decoder14/W'], result['decoder14/b'] = conv_back(enlarged, p['decoder14/W'], g)
    g = 4*pool(g)*dc
    g, result['combine/W'], result['combine/b'] = conv_back(joined, p['combine/W'], g)
    dfeature, dz = g[:, :, :, :4], g[:, :, :, 4:].reshape(-1, 49)
    difference = o['lower_mu']-o['prior_mu'];inv = np.exp(-o['prior_lv']);ratio = np.exp(o['lower_lv']-o['prior_lv'])
    qmu = difference*inv/count+dz;qlv = .5*(ratio-1)/count+.5*dz*lower_sigma*epsilon[:, :49]
    pmu = -difference*inv/count;plv = .5*(1-ratio-difference*difference*inv)/count
    if nvae:pmu += qmu;plv += qlv
    dheads = np.stack([qmu.reshape(-1, 7, 7), qlv.reshape(-1, 7, 7)], axis=3)
    g, result['posterior_heads/W'], result['posterior_heads/b'] = conv_back(posterior, p['posterior_heads/W'], dheads)
    g, result['posterior_hidden/W'], result['posterior_hidden/b'] = conv_back(fusion, p['posterior_hidden/W'], g*dp)
    dencoder = g[:, :, :, :4];dfeature += g[:, :, :, 4:]
    isolated = {}
    for prefix, gradient in [('prior_mu', pmu), ('prior_lv', plv)]:
        g, dw, db = conv_back(o['top_feature'], p[prefix+'/W'], gradient.reshape(-1, 7, 7, 1));dfeature += g
        isolated['without_regularizer/'+prefix+'/W'] = dw.copy()
        if nvae:
            dr = 1e-4*p[prefix+'/W']/np.sqrt((p[prefix+'/W']**2).sum()+1e-12)
            isolated['regularizer/'+prefix+'/W'] = dr;dw += dr
        result[prefix+'/W'], result[prefix+'/b'] = dw, db
    g, cell_gradients = top_feature_back(p, cell, dfeature, nvae);result.update(cell_gradients)
    result['top_projection/W'], result['top_projection/b'] = o['top_z'].T@g, g.sum(axis=0)
    dz2 = g@p['top_projection/W'].T
    dmu2 = dz2+o['top_mu']/count;dlv2 = .5*dz2*top_sigma*epsilon[:, 49:]+.5*(np.exp(o['top_lv'])-1)/count
    for prefix, gradient in [('top_mu', dmu2), ('top_lv', dlv2)]:
        result[prefix+'/W'], result[prefix+'/b'] = flat.T@gradient, gradient.sum(axis=0)
        dencoder += (gradient@p[prefix+'/W'].T).reshape(-1, 7, 7, 4)
    g = repeat(dencoder)/4*d2
    g, result['encoder2/W'], result['encoder2/b'] = conv_back(p1, p['encoder2/W'], g)
    g = repeat(g)/4*d1
    _, result['encoder1/W'], result['encoder1/b'] = conv_back(x, p['encoder1/W'], g)
    return result, moments, isolated


def check(directory):
    """Validate complete paired artifacts and return measured gradient/field/state errors."""
    results = [];pixels, noises, gradient_count = 0, 0, 0
    for nvae, name in [(False, 'hierarchical'), (True, 'miniature_nvae')]:
        root = directory/name;data, noise = arrays(root/'data.npz'), arrays(root/'fixed_noise.npz')
        ids, unique = set(), set()
        for split_name, count, seed in [('fit', 210, 2621), ('validation', 42, 2633), ('test', 42, 2647)]:
            identities, factors, images = [data[split_name+s] for s in ['_source_ids', '_factors', '_images']]
            assert len(identities) == count and not ids.intersection(identities);ids.update(identities);yy, xx = np.mgrid[:28, :28]
            for row, identity in enumerate(identities):
                shape, size, x, y = int(identity)//147, (int(identity)//49)%3, (int(identity)//7)%7, int(identity)%7
                np.testing.assert_array_equal(factors[row], [shape, size, x, y]);residue = (x+2*y+size+3*shape)%7
                assert residue == 0 if split_name == 'test' else residue == 1 if split_name == 'validation' else residue >= 2
                dx, dy = abs(xx-(3*x+5)), abs(yy-(3*y+5));expected = np.maximum(dx, dy) <= size+2 if shape else dx*dx+dy*dy <= (size+2)**2
                exact(expected.astype(np.float32)[..., None], images[row]);unique.add(images[row].tobytes())
            pixels += images.size;expected = gaussian((4, count, 51), seed)
            np.testing.assert_allclose(noise[split_name+'_standard'], expected, atol=1e-14, rtol=1e-14);exact(noise[split_name+'_epsilon'], expected.astype(np.float32));noises += expected.size
        assert len(ids) == len(unique) == 294
        expected = gaussian((1, 20, 51), 2657)
        np.testing.assert_allclose(noise['prior_standard'], expected, atol=1e-14, rtol=1e-14);exact(noise['prior_epsilon'], expected.astype(np.float32));noises += expected.size
        for epoch in range(1, 4):
            actual = arrays(root/f'train_noise_epoch{epoch}.npz');expected = gaussian((1, 210, 51), 2663+epoch)
            np.testing.assert_allclose(actual['standard'], expected, atol=1e-14, rtol=1e-14);exact(actual['epsilon'], expected.astype(np.float32));noises += expected.size
        algebra = arrays(root/'residual_algebra.npz');exact(algebra['kl'], np.zeros(1, dtype=np.float32))
        np.testing.assert_array_equal(algebra['prior_mu'], np.array([[1, -2, .5]], dtype=np.float32))
        initial = arrays(root/'initial_arrays.npz');p, state = split(initial)
        batch, actual = arrays(root/'first_batch.npz'), arrays(root/'first_gradients.npz')
        expected, moments, isolated = derivatives(p, state, batch['images'], batch['epsilon'], nvae)
        assert set(actual) == set(expected);gradient_error = 0
        for key, value in expected.items():
            np.testing.assert_allclose(actual[key], value, atol=8e-5, rtol=5e-3)
            gradient_count += value.size;gradient_error = max(gradient_error, float(np.max(np.abs(actual[key]-value))))
        updated = arrays(root/'first_update_arrays.npz');first_update({k: v for k, v in initial.items() if not k.startswith('state/')}, actual, updated)
        expected_forward, _ = forward(p, state, batch['images'], batch['epsilon'], nvae, True)
        for key, value in arrays(root/'first_forward.npz').items():np.testing.assert_allclose(value, expected_forward[key], atol=9e-4, rtol=2e-5)
        state_error = 0
        if nvae:
            for layer, (mean, variance) in moments.items():
                for field, value in [('moving_mean', mean), ('moving_var', variance)]:
                    key = layer+'/'+field;expected_state = .9*state[key]+.1*value
                    np.testing.assert_allclose(updated['state/'+key], expected_state, atol=2e-6, rtol=2e-5)
                    state_error = max(state_error, float(np.max(np.abs(updated['state/'+key]-expected_state))))
            actual_isolated = arrays(root/'first_regularizer_gradients.npz')
            for key, value in actual_isolated.items():np.testing.assert_allclose(value, isolated[key], atol=8e-5, rtol=5e-3)
        p, state = split(arrays(root/'model_arrays.npz'));output = arrays(root/'outputs.npz');prediction_error, field_error = 0, 0
        for split_name in ['fit', 'validation', 'test']:
            image = data[split_name+'_images']
            for label, epsilon in [('greedy', np.zeros((len(image), 51)))]+[(f'draw{k}', v) for k, v in enumerate(noise[split_name+'_epsilon'])]:
                expected, _ = forward(p, state, image, epsilon, nvae)
                for key, value in expected.items():
                    saved = output[f'{split_name}_{label}_{key}'];np.testing.assert_allclose(saved, value, atol=9e-4, rtol=2e-5)
                    field_error = max(field_error, float(np.max(np.abs(saved-value))))
                prediction_error = max(prediction_error, float(np.max(np.abs(output[f'{split_name}_{label}_probability']-expected['probability']))));pixels += expected['probability'].size
        for label, region in [('lower', slice(0, 49)), ('top', slice(49, 51))]:
            epsilon = np.zeros((42, 51));epsilon[:, region] = .5;expected, _ = forward(p, state, data['test_images'], epsilon, nvae)
            for key, value in arrays(root/f'{label}_noise_intervention.npz').items():np.testing.assert_allclose(value, expected[key], atol=9e-4, rtol=2e-5)
        prior = arrays(root/'prior.npz');top, lower_noise = noise['prior_epsilon'][0, :, 49:], noise['prior_epsilon'][0, :, :49]
        exact(prior['top_z'], top);exact(prior['lower_noise'], lower_noise)
        feature, _, _ = top_feature(p, state, top.astype(np.float64), nvae, False)
        pm = conv(feature, p['prior_mu/W'], p['prior_mu/b']).reshape(-1, 49);pl = conv(feature, p['prior_lv/W'], p['prior_lv/b']).reshape(-1, 49)
        lower = pm+np.exp(.5*pl)*lower_noise;_, probability, _ = decode(p, lower, feature)
        for key, value in [('prior_mu', pm), ('prior_lv', pl), ('lower_z', lower), ('probability', probability)]:np.testing.assert_allclose(prior[key], value, atol=5e-5, rtol=2e-5)
        pixels += probability.size
        history = np.genfromtxt(root/'history.tsv', delimiter='\t', names=True);scores = []
        for epoch in range(1, 4):
            ep, es = split(arrays(root/f'epoch{epoch}_arrays.npz'));nll, lower, top = 0, 0, 0
            for epsilon in noise['validation_epsilon']:
                o, _ = forward(ep, es, data['validation_images'], epsilon, nvae);nll += o['nll'].mean()/4;lower += o['kl_lower'].mean()/4;top += o['kl_top'].mean()/4
            scores.append(nll+lower+top)
            for key, value in [('validation_nll', nll), ('validation_lower_kl', lower), ('validation_top_kl', top), ('validation_negative_elbo', nll+lower+top)]:np.testing.assert_allclose(history[key][epoch-1], value, atol=9e-4, rtol=2e-6)
        chosen = int(np.argmin(scores)+1)
        for key, value in arrays(root/f'epoch{chosen}_arrays.npz').items():exact(value, arrays(root/'model_arrays.npz')[key])
        metrics = json.loads((root/'metrics.json').read_text());assert metrics['selected_epoch'] == chosen and metrics['updates'] == 21
        batches = np.genfromtxt(root/'batches.tsv', delimiter='\t', names=True, dtype=int);assert set(batches['step']) == set(range(21))
        for epoch in range(1, 4):np.testing.assert_array_equal(np.sort(batches['row'][batches['epoch'] == epoch]), np.arange(210))
        np.testing.assert_array_equal(batches['source_id'], data['fit_source_ids'][batches['row']])
        results.append(dict(model=name, first_gradient_max_error=gradient_error, first_Adam_update_bit_exact=True,
            first_moving_buffer_update_max_error=state_error, posterior_probability_max_error=prediction_error,
            all_nineteen_fields_max_error=field_error, selected_epoch=chosen, isolated_prior_spectral_gradients_checked=nvae))
    return dict(all_first_batch_derivatives_checked=gradient_count, Gaussian_values_checked=noises,
        lattice_and_prediction_pixels_checked=pixels, first_moving_buffers_checked=6,
        eligible_validation_checkpoints_checked=6, models=results)


def main():
    """Run the optional independent development oracle and save its compact receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path);parser.add_argument('--json', type=Path)
    args = parser.parse_args();result = check(args.directory)
    if args.json:args.json.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
