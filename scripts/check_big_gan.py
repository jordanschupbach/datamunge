#!/usr/bin/env python3
"""Independent NumPy compact BigGAN forwards, derivatives, captured moments and controls; no fitting."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from check_adversarial_autoencoder import arrays, exact, gaussian
from check_contractive_autoencoder import sigmoid
from check_convolutional_autoencoder import pool, repeat
from check_spectral_gan import EPSILON, FLOOR


def conv(x, w):
    n, h, width, _ = x.shape
    kernel = w.shape[0]
    pad = kernel // 2
    padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    out = np.zeros((n, h, width, w.shape[-1]))
    for y in range(kernel):
        for col in range(kernel):
            out += np.einsum('nhwc,co->nhwo', padded[:, y:y+h, col:col+width], w[y, col])

    def backward(dy):
        dx, dw = np.zeros_like(padded), np.zeros_like(w)
        for y in range(kernel):
            for col in range(kernel):
                window = padded[:, y:y+h, col:col+width]
                dw[y, col] = np.einsum('nhwc,nhwo->co', window, dy)
                dx[:, y:y+h, col:col+width] += np.einsum('nhwo,co->nhwc', dy, w[y, col])
        return dx[:, pad:pad+h, pad:pad+width], dw

    return out, backward


def kernel(p, group, name):
    key = f'{group}/{name}/W'
    raw = p[key]
    w = raw.reshape(-1, raw.shape[-1])
    u, v = p[f'{group+2}/{name}/u'], p[f'{group+2}/{name}/v']
    sigma = np.sum(v * (w @ u))
    denominator = max(sigma, FLOOR)

    def backward(gradient, out):
        gradient = gradient.reshape(w.shape)
        partial = gradient / denominator
        if sigma > FLOOR:
            partial -= np.sum(gradient * w) / sigma**2 * (v @ u.T)
        out[key] = out.get(key, 0) + partial.reshape(raw.shape)

    return raw / denominator, backward


def dense(p, group, name, x, bias=True):
    w, bw = kernel(p, group, name)
    y = x @ w
    if bias:
        y += p[f'{group}/{name}/b']

    def backward(dy, out):
        bw(x.T @ dy, out)
        if bias:
            key = f'{group}/{name}/b'
            out[key] = out.get(key, 0) + dy.sum(axis=0)
        return dy @ w.T

    return y, backward


def convolution(p, group, name, x, bias=True):
    w, bw = kernel(p, group, name)
    y, bc = conv(x, w)
    if bias:
        y += p[f'{group}/{name}/b']

    def backward(dy, out):
        dx, dw = bc(dy)
        bw(dw, out)
        if bias:
            key = f'{group}/{name}/b'
            out[key] = out.get(key, 0) + dy.sum(axis=(0, 1, 2))
        return dx

    return y, backward


def normalize(p, name, x, training, moments):
    axes = (0, 1, 2)
    if training:
        mean = x.mean(axis=axes)
        variance = ((x - mean)**2).mean(axis=axes)
        count = float(np.float32(x.size) / np.float32(4))
        correction = float(np.float32(count) / np.float32(count - 1))
        moments[f'moment/{name}/mean'] = mean
        moments[f'moment/{name}/unbiased_var'] = variance * correction
    else:
        mean, variance = p[f'4/{name}/moving_mean'], p[f'4/{name}/moving_var']
    inv = 1 / np.sqrt(variance + float(np.float32(1e-5)))
    centered = x - mean
    y = centered * inv

    def backward(dy):
        if not training:
            return dy * inv
        return inv * (dy - dy.mean(axis=axes) - centered * inv**2 * (dy * centered).mean(axis=axes))

    return y, backward


def conditional_norm(p, name, x, condition, training, moments):
    normalized, bn = normalize(p, name, x, training, moments)
    gain, bg = dense(p, 0, name + '/gain', condition)
    gain += 1
    bias, bb = dense(p, 0, name + '/bias', condition)
    y = normalized * gain[:, None, None, :] + bias[:, None, None, :]

    def backward(dy, out):
        dc = bg((dy * normalized).sum(axis=(1, 2)), out) + bb(dy.sum(axis=(1, 2)), out)
        return bn(dy * gain[:, None, None, :]), dc

    return y, backward


def generate(p, z, labels, training=False):
    moments = {}
    one_hot = np.eye(10)[labels]
    shared = one_hot @ p['0/class_embedding']
    xflat, bs = dense(p, 0, 'stem', z[:, :4], False)
    x = xflat.reshape(-1, 4, 4, 4)
    h, bn1 = conditional_norm(p, 'bn1', x, np.concatenate([shared, z[:, 4:8]], axis=1), training, moments)
    c1, bc1 = convolution(p, 0, 'conv1', repeat(np.maximum(h, 0)), False)
    h2, bn2 = conditional_norm(p, 'bn2', c1, np.concatenate([shared, z[:, 8:]], axis=1), training, moments)
    c2, bc2 = convolution(p, 0, 'conv2', np.maximum(h2, 0), False)
    skip, bskip = convolution(p, 0, 'shortcut', x, False)
    normalized, bout = normalize(p, 'output_bn', c2 + repeat(skip), training, moments)
    affine = normalized * p['0/output_bn/gamma'] + p['0/output_bn/beta']
    logit, brgb = convolution(p, 0, 'rgb', np.maximum(affine, 0))
    image = sigmoid(logit)

    def backward(dimage, out):
        daffine = brgb(dimage * image * (1-image), out) * (affine > 0)
        for key, value in [('0/output_bn/gamma', (daffine * normalized).sum(axis=(0, 1, 2))),
                           ('0/output_bn/beta', daffine.sum(axis=(0, 1, 2)))]:
            out[key] = out.get(key, 0) + value
        dsum = bout(daffine * p['0/output_bn/gamma'])
        dh2 = bc2(dsum, out) * (h2 > 0)
        dc1, dcondition2 = bn2(dh2, out)
        dh = 4 * pool(bc1(dc1, out)) * (h > 0)
        dx, dcondition1 = bn1(dh, out)
        dx += bskip(4 * pool(dsum), out)
        dz0 = bs(dx.reshape(-1, 64), out)
        out['0/class_embedding'] = out.get('0/class_embedding', 0) + one_hot.T @ (dcondition1[:, :4] + dcondition2[:, :4])
        return np.concatenate([dz0, dcondition1[:, 4:], dcondition2[:, 4:]], axis=1)

    return image, moments, backward


def discriminate(p, image, labels, details=False):
    h, bc1 = convolution(p, 1, 'conv1', image)
    h2, bc2 = convolution(p, 1, 'conv2', np.maximum(h, 0))
    shortcut, bshortcut = convolution(p, 1, 'shortcut', image)
    summed = pool(h2) + pool(shortcut)
    features = np.maximum(summed, 0).sum(axis=(1, 2))
    source, bs = dense(p, 1, 'score', features)
    table, btable = kernel(p, 1, 'projection')
    one_hot = np.eye(10)[labels]
    embedding = one_hot @ table
    projection = (features * embedding).sum(axis=1, keepdims=True)
    score = source + projection

    def backward(dscore, out):
        df = bs(dscore, out) + dscore * embedding
        btable(one_hot.T @ (dscore * features), out)
        dsum = np.broadcast_to(df[:, None, None, :], summed.shape) * (summed > 0)
        dpool = repeat(dsum) / 4
        return bc1(bc2(dpool, out) * (h > 0), out) + bshortcut(dpool, out)

    return (dict(score=score, source=source, projection=projection, features=features)
            if details else score), backward


def orthogonal(p):
    total, out = 0., {}
    coefficient = float(np.float32(.5e-4))
    for key, value in p.items():
        if not key.startswith('0/'):
            continue
        out[key] = np.zeros_like(value)
        if key.endswith('/W'):
            matrix = value.reshape(-1, value.shape[-1])
            off = matrix.T @ matrix
            np.fill_diagonal(off, 0)
            total += coefficient * np.sum(off**2)
            out[key] = (4 * coefficient * (matrix @ off)).reshape(value.shape)
    return total, out


def forward(p, image, z, labels, batch_statistics=False):
    frozen, _, _ = generate(p, z, labels, False)
    training, moments, _ = generate(p, z, labels, True)
    real, _ = discriminate(p, image, labels)
    fake_training, _ = discriminate(p, training, labels)
    fake_inference, _ = discriminate(p, frozen, labels)
    dl = np.maximum(1-real, 0).mean() + np.maximum(1+fake_training, 0).mean()
    source = -fake_training.mean()
    penalty, _ = orthogonal(p)
    out = dict(fake=training if batch_statistics else frozen, fake_training=training,
               fake_inference=frozen, fake_score_training=fake_training,
               fake_score_inference=fake_inference, real_score=real,
               fake_score=fake_training if batch_statistics else fake_inference,
               d_loss=dl, g_source=source, g_orthogonal=penalty, g_loss=source+penalty)
    if batch_statistics:
        out.update(moments)
    return out


def derivatives(p, image, z, labels):
    fake, moments, bg = generate(p, z, labels, True)
    real, br = discriminate(p, image, labels)
    generated, bf = discriminate(p, fake, labels)
    d, g = {}, {}
    br(-(real < 1).astype(float)/len(real), d)
    bf((generated > -1).astype(float)/len(generated), d)
    bg(bf(-np.ones_like(generated)/len(generated), {}), g)
    _, penalty = orthogonal(p)
    full = {k: v + penalty[k] for k, v in g.items()}
    return g, penalty, dict(full, **moments), d


def rejection_prior(n, seed, limit=np.inf):
    rng = np.random.RandomState(seed)
    proposals, accepted = [], []
    spare = None
    while len(accepted) < n * 12:
        if spare is not None:
            candidate, spare = spare, None
        else:
            u = (int(rng.randint(0, 2**32, dtype=np.uint32))+.5)/2**32
            v = (int(rng.randint(0, 2**32, dtype=np.uint32))+.5)/2**32
            radius, angle = math.sqrt(-2 * math.log(u)), 6.283185307179586476925286766559 * v
            candidate, spare = radius * math.cos(angle), radius * math.sin(angle)
        proposals.append(candidate)
        if abs(candidate) <= limit:
            accepted.append(candidate)
    return np.asarray(accepted, np.float32).reshape(n, 12), np.asarray(proposals, np.float64)


def advance(p, group):
    result = {k: v.copy() for k, v in p.items()}
    for key, raw in p.items():
        if not key.startswith(str(group)+'/') or not key.endswith('/W'):
            continue
        state = str(group+2) + key[1:-1]
        w = raw.reshape(-1, raw.shape[-1])
        v = w @ p[state+'u']
        v /= np.sqrt(np.sum(v*v)+EPSILON)
        u = w.T @ v
        u /= np.sqrt(np.sum(u*u)+EPSILON)
        result[state+'u'], result[state+'v'] = u, v
    return result


def check(root, data):
    directory = root/'original-output'
    raw = arrays(data)
    p0 = {k: v.astype(float) for k, v in arrays(directory/'initial_arrays.npz').items()}
    pf = {k: v.astype(float) for k, v in arrays(directory/'final.npz').items()}
    assert len(p0) == 60 and sum(v.size for v in p0.values()) == 1372
    source_data = arrays(directory/'data.npz')
    # Same half-pixel centers, independent four-term Double interpolation/final Float32 cast.
    for split in ['fit', 'validation', 'test']:
        original = (raw[split+'_images'].astype(np.float32)/np.float32(255)).reshape(-1, 28, 28)
        resized = np.zeros((len(original), 8, 8, 1))
        for y in range(8):
            for x in range(8):
                py, px = (y+.5)*3.5-.5, (x+.5)*3.5-.5
                iy, ix = int(py), int(px)
                for dy in [0, 1]:
                    for dx in [0, 1]:
                        resized[:, y, x, 0] += (py-iy if dy else 1-(py-iy)) * (px-ix if dx else 1-(px-ix)) * original[:, iy+dy, ix+dx].astype(float)
        exact(source_data[split+'_images'], resized.astype(np.float32))
    batch = np.genfromtxt(directory/'batches.tsv', names=True, delimiter='\t', dtype=np.int64)
    for epoch in [1, 2, 3]:
        rows = batch[batch['epoch'] == epoch]
        exact(np.sort(rows['row']), np.arange(1000))
        exact(rows['source_id'], raw['fit_source_ids'][rows['row']].astype(np.int64))
        assert len(np.unique(rows['step'])) == 32
    first = batch[batch['step'] == 0]['row']
    z = gaussian((1000, 12), 6872).astype(np.float32).astype(float)[first]
    x, labels = source_data['fit_images'][first].astype(float), raw['fit_labels'][first].astype(int)
    test_z, _ = rejection_prior(200, 6857)
    test_labels = raw['test_labels'].astype(int)
    values, error = 0, 0.
    for name, p, image, noise, classes, training in [
        ('first_forward', p0, x, z, labels, True),
        ('initial_outputs', p0, source_data['test_images'], test_z, test_labels, False),
        ('outputs', pf, source_data['test_images'], test_z, test_labels, False)]:
        expected = forward(p, image.astype(float), noise.astype(float), classes, training)
        stored = arrays(directory/(name+'.npz'))
        assert set(stored) == set(expected)
        for key, value in expected.items():
            np.testing.assert_allclose(stored[key], value, atol=8e-6, rtol=1e-4, err_msg=name+'/'+key)
            values += np.size(value)
            error = max(error, float(abs(stored[key]-value).max()))
    derivative_values, derivative_error, differences = 0, 0., 0
    stages = [('initial', p0), ('final', pf),
              ('d-advanced', {k: v.astype(float) for k, v in arrays(root/'probe-d-advanced.npz').items()}),
              ('g-advanced', {k: v.astype(float) for k, v in arrays(root/'probe-g-advanced.npz').items()})]
    for stage, p in stages:
        adv, penalty, full, d = derivatives(p, x, z, labels)
        records = []
        if stage == 'initial':
            records = [(adv, directory/'first_adversarial_gradients.npz'), (penalty, directory/'first_orthogonal_gradients.npz'),
                       (full, root/'probe-initial-gradient-0.npz'), (d, root/'probe-initial-gradient-1.npz')]
        elif stage == 'final':
            records = [(full, directory/'final_gradients_group0.npz'), (d, directory/'final_gradients_group1.npz')]
        else:
            records = [(d if stage == 'd-advanced' else full, root/f'probe-{stage}-gradient-{1 if stage == "d-advanced" else 0}.npz')]
        for analytic, path in records:
            stored = arrays(path)
            assert set(stored) == set(analytic)
            for key, partial in analytic.items():
                np.testing.assert_allclose(stored[key], partial, atol=2e-5, rtol=2e-3, err_msg=stage+'/'+key)
                derivative_values += np.size(partial)
                derivative_error = max(derivative_error, float(abs(stored[key]-partial).max()))
        if stage == 'final':
            for key, partial in dict(full, **d).items():
                if key.startswith('moment/'):
                    continue
                objective = 'g_loss' if key.startswith('0/') else 'd_loss'
                for index in sorted({0, partial.size//2, partial.size-1}):
                    plus, minus = dict(p), dict(p)
                    plus[key], minus[key] = p[key].copy(), p[key].copy()
                    plus[key].flat[index] += 1e-6
                    minus[key].flat[index] -= 1e-6
                    finite = (forward(plus, x, z, labels)[objective]-forward(minus, x, z, labels)[objective])/2e-6
                    np.testing.assert_allclose(partial.flat[index], finite, atol=1e-7, rtol=1e-3, err_msg=key)
                    differences += 1
    proposals = 0
    for name, seed, limit in [('validation_prior', 6841, np.inf), ('test_prior', 6857, np.inf)] + [(f'truncated_prior{k}', 6907, limit) for k, limit in enumerate([np.inf, 2, 1, .5])]:
        noise, audit = rejection_prior(200, seed, limit)
        stored = arrays(directory/(name+'.npz'))
        exact(noise, stored['z']);exact(audit, stored['proposals']);proposals += audit.size
    controls = arrays(directory/'class_controls.npz')
    np.testing.assert_allclose(controls['fake'], generate(pf, controls['z'].astype(float), controls['labels'].astype(int))[0], atol=3e-6, rtol=1e-4)
    changed = (test_labels+1)%10
    np.testing.assert_allclose(controls['changed_real_scores'], discriminate(pf, source_data['test_images'].astype(float), changed)[0], atol=8e-6, rtol=1e-4)
    hierarchy = arrays(directory/'hierarchical_controls.npz')
    for chunk in range(3):
        modified = test_z.copy();modified[:, chunk*4:chunk*4+4] = 0
        exact(modified, hierarchy[f'zero_chunk{chunk}_z'])
        expected = generate(pf, modified.astype(float), test_labels)[0]
        np.testing.assert_allclose(hierarchy[f'zero_chunk{chunk}_fake'], expected, atol=4e-6, rtol=1e-4)
    for k in range(4):
        prior, recorded = arrays(directory/f'truncated_prior{k}.npz'), arrays(directory/f'truncated_output{k}.npz')
        expected = generate(pf, prior['z'].astype(float), recorded['labels'].astype(int))[0]
        np.testing.assert_allclose(recorded['fake'], expected, atol=4e-6, rtol=1e-4)
    # Source state ownership: D cache, D weights, G cache, G weights+captured BN.
    for left_name, right_name, owners in [('initial', 'd-advanced', ['3/']), ('d-advanced', 'after-d', ['1/']),
                                         ('after-d', 'g-advanced', ['2/']), ('g-advanced', 'after-g', ['0/', '4/'])]:
        left, right = arrays(root/f'probe-{left_name}.npz'), arrays(root/f'probe-{right_name}.npz')
        for key in left:
            if not any(key.startswith(prefix) for prefix in owners):
                exact(left[key], right[key])
        if owners[0] in ['2/', '3/']:
            expected = advance({k: v.astype(float) for k, v in left.items()}, int(owners[0][0])-2)
            for key in left:
                np.testing.assert_allclose(right[key], expected[key], atol=3e-6, rtol=1e-4)
    captured = arrays(root/'probe-g-advanced-gradient-0.npz')
    before, after = arrays(root/'probe-g-advanced.npz'), arrays(root/'probe-after-g.npz')
    for name in ['bn1', 'bn2', 'output_bn']:
        for suffix, moment in [('moving_mean', 'mean'), ('moving_var', 'unbiased_var')]:
            key = '4/'+name+'/'+suffix
            expected = np.float32(.9)*before[key] + np.float32(1-np.float32(.9))*captured['moment/'+name+'/'+moment]
            exact(expected, after[key])
    metrics = json.loads((directory/'metrics.json').read_text())
    assert metrics['steps_per_group'] == 96 and metrics['optimizer_updates'] == 192 and metrics['selected_epoch_fixed_final'] == 3
    output = arrays(directory/'outputs.npz')
    for field in ['d_loss', 'g_source', 'g_orthogonal', 'g_loss']:
        assert metrics['test_'+field] == float(output[field])
    return dict(passed=True, forward_values_checked=int(values), maximum_forward_error=error,
                native_derivative_values_checked=int(derivative_values), maximum_derivative_error=derivative_error,
                independent_final_scalar_finite_differences=differences, resized_pixels_exact=89600,
                checkpoint_values=1372, rejection_proposals_exact=int(proposals), accepted_prior_values_exact=14400,
                captured_moving_updates_bit_exact=True, explicit_phase_ownership=True, batch_rows_checked=len(batch),
                evaluation_losses_use_batch_statistics_even_when_display_is_inference=True, metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('build/nn-report-conversion/big_gan'))
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
