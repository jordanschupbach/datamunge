#!/usr/bin/env python3
"""Independent NumPy patch SAGAN forwards, spectral/attention gradients and controls; no fitting."""
import argparse
import json
from pathlib import Path

import numpy as np

from check_adversarial_autoencoder import arrays, exact, gaussian
from check_auxiliary_classifier_gan import coarse, mmd
from check_contractive_autoencoder import sigmoid
from check_dense_gans import losses
from check_spectral_gan import EPSILON, FLOOR


def patches(image):
    return image.reshape(-1, 4, 7, 4, 7).transpose(0, 1, 3, 2, 4).reshape(-1, 16, 49)


def pixels(patch):
    return patch.reshape(-1, 4, 4, 7, 7).transpose(0, 1, 3, 2, 4).reshape(-1, 784)


def dense(p, group, name, x, activation=None):
    key = f'{group}/{name}'
    w, b = p[key + '/W'], p[key + '/b']
    u, v = p[f'{group + 2}/{name}/u'], p[f'{group + 2}/{name}/v']
    sigma = np.sum(v * (w @ u))
    denominator = max(sigma, FLOOR)
    effective = w / denominator
    y = x @ effective + b
    if activation == 'tanh':
        y = np.tanh(y)
    elif activation == 'sigmoid':
        y = sigmoid(y)

    def backward(dy, gradients):
        if activation == 'tanh':
            dy = dy * (1 - y * y)
        elif activation == 'sigmoid':
            dy = dy * y * (1 - y)
        deffective = x.T @ dy
        dw = deffective / denominator
        if sigma > FLOOR:
            dw -= np.sum(deffective * w) / sigma**2 * (v @ u.T)
        for suffix, partial in [('/W', dw), ('/b', dy.sum(axis=0))]:
            gradients[key + suffix] = gradients.get(key + suffix, 0) + partial
        return dy @ effective.T

    return y, backward


def attention(p, group, x):
    flat = x.reshape(-1, 8)
    q, bq = dense(p, group, 'attention/query', flat)
    k, bk = dense(p, group, 'attention/key', flat)
    v, bv = dense(p, group, 'attention/value', flat)
    q, k, v = q.reshape(-1, 16, 2), k.reshape(-1, 16, 2), v.reshape(-1, 16, 4)
    scores = q @ k.transpose(0, 2, 1)  # Intentionally no sqrt(2) scale.
    probability = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probability /= probability.sum(axis=-1, keepdims=True)
    mixed = probability @ v
    projected, bp = dense(p, group, 'attention/project', mixed.reshape(-1, 4))
    projected = projected.reshape(x.shape)
    gamma = p[f'{group}/attention/gamma']
    output = x + gamma * projected

    def backward(dy, gradients):
        key = f'{group}/attention/gamma'
        gradients[key] = gradients.get(key, 0) + np.sum(dy * projected)
        dmixed = bp((dy * gamma).reshape(-1, 8), gradients).reshape(-1, 16, 4)
        dp = dmixed @ v.transpose(0, 2, 1)
        dv = probability.transpose(0, 2, 1) @ dmixed
        ds = probability * (dp - np.sum(dp * probability, axis=-1, keepdims=True))
        dq, dk = ds @ k, ds.transpose(0, 2, 1) @ q
        dx = bq(dq.reshape(-1, 2), gradients) + bk(dk.reshape(-1, 2), gradients)
        dx += bv(dv.reshape(-1, 4), gradients)
        return dy + dx.reshape(x.shape)

    return dict(input=x, probability=probability, output=output), backward


def generate(p, noise):
    tokens, bt = dense(p, 0, 'input', noise, 'tanh')
    a, ba = attention(p, 0, tokens.reshape(-1, 16, 8))
    patch, bp = dense(p, 0, 'patch', a['output'].reshape(-1, 8), 'sigmoid')
    image = pixels(patch)

    def backward(dimage, gradients):
        dpatch = patches(dimage).reshape(-1, 49)
        da = bp(dpatch, gradients).reshape(-1, 16, 8)
        return bt(ba(da, gradients).reshape(-1, 128), gradients)

    return image, a, backward


def discriminate(p, image):
    encoded, be = dense(p, 1, 'patch', patches(image).reshape(-1, 49), 'tanh')
    tokens = encoded.reshape(-1, 16, 8) + p['1/position']
    a, ba = attention(p, 1, tokens)
    score, bs = dense(p, 1, 'output', a['output'].mean(axis=1))

    def backward(dscore, gradients):
        dmean = bs(dscore, gradients)
        dtokens = ba(np.repeat(dmean[:, None, :] / 16, 16, axis=1), gradients)
        gradients['1/position'] = gradients.get('1/position', 0) + dtokens.sum(axis=0)
        return pixels(be(dtokens.reshape(-1, 8), gradients))

    return score, a, backward


def forward(p, image, noise, diagnostic=True):
    fake, ga, _ = generate(p, noise)
    real, ra, _ = discriminate(p, image)
    generated, fa, _ = discriminate(p, fake)
    dl, gl = losses(real, generated, 'hinge_gan')
    out = dict(fake=fake, real_score=real, fake_score=generated,
               discriminator_loss=dl, generator_loss=gl,
               g_attention=ga['probability'], g_attention_input=ga['input'],
               g_attention_output=ga['output'], real_attention=ra['probability'],
               real_attention_input=ra['input'], real_attention_output=ra['output'],
               fake_attention=fa['probability'])
    if diagnostic:
        out['pixel_mmd'] = mmd(coarse(image), coarse(fake))
    return out


def gradients(p, image, noise):
    fake, _, bg = generate(p, noise)
    real, _, br = discriminate(p, image)
    generated, _, bf = discriminate(p, fake)
    d, g = {}, {}
    br(-(real < 1).astype(float) / len(image), d)
    bf((generated > -1).astype(float) / len(noise), d)
    pixels_gradient = bf(-np.ones_like(generated) / len(noise), {})
    bg(pixels_gradient, g)
    return g, d


def advance(p, group, iterations=1):
    result = {k: v.copy() for k, v in p.items()}
    for key, w in p.items():
        if not key.startswith(str(group) + '/') or not key.endswith('/W'):
            continue
        state = str(group + 2) + key[len(str(group)):-1]
        u = p[state + 'u']
        for _ in range(iterations):
            v = w @ u
            v /= np.sqrt(np.sum(v * v) + EPSILON)
            u = w.T @ v
            u /= np.sqrt(np.sum(u * u) + EPSILON)
        result[state + 'u'], result[state + 'v'] = u, v
    return result


def check(root, data):
    directory = root / 'original-output'
    raw = arrays(data)
    p0 = {k: v.astype(float) for k, v in arrays(directory / 'initial_arrays.npz').items()}
    pf = {k: v.astype(float) for k, v in arrays(directory / 'final.npz').items()}
    assert len(p0) == 51 and sum(v.size for v in p0.values()) == 3735
    assert sum(v.size for k, v in p0.items() if k[0] in '01') == 3380
    assert p0['0/attention/gamma'] == p0['1/attention/gamma'] == 0
    # Frozen initialization is also the probe's starting state, before the first per-batch advance.
    for k, v in arrays(root / 'probe-initial.npz').items():
        exact(v, arrays(directory / 'initial_arrays.npz')[k])
    cold = {k: v.copy() for k, v in p0.items()}
    for k, v in cold.items():
        if k[0] in '23':
            v.fill(float(np.float32(1) / np.sqrt(np.float32(v.size))))
    warm = advance(advance(cold, 0, 16), 1, 16)
    for k in p0:
        np.testing.assert_allclose(warm[k], p0[k], atol=3e-6, rtol=5e-5, err_msg='warmup/' + k)
    images = {s: (raw[s + '_images'].astype(np.float32) / np.float32(255)).reshape(-1, 784).astype(float)
              for s in ['fit', 'validation', 'test']}
    batch = np.genfromtxt(directory / 'batches.tsv', names=True, delimiter='\t', dtype=np.int64)
    for epoch in [1, 2, 3]:
        rows = batch[batch['epoch'] == epoch]
        exact(np.sort(rows['row']), np.arange(1000))
        exact(rows['source_id'], raw['fit_source_ids'][rows['row']].astype(np.int64))
        assert len(np.unique(rows['step'])) == 32
    assert len(batch) == 3000 and len(np.unique(batch['step'])) == 96
    first = batch[batch['step'] == 0]['row']
    x = images['fit'][first]
    z = gaussian((1000, 16), 4862).astype(np.float32).astype(float)[first]
    tz = gaussian((200, 16), 4831).astype(np.float32).astype(float)
    values, error = 0, 0.
    for name, p, image, noise in [('first_forward', p0, x, z), ('initial_outputs', p0, images['test'], tz), ('outputs', pf, images['test'], tz)]:
        expected, stored = forward(p, image, noise), arrays(directory / (name + '.npz'))
        assert set(expected) == set(stored)
        for k, v in expected.items():
            np.testing.assert_allclose(stored[k], v, atol=5e-6, rtol=3e-5, err_msg=name + '/' + k)
            values += np.size(v)
            error = max(error, float(abs(stored[k] - v).max()))
        for key in ['g_attention', 'real_attention', 'fake_attention']:
            np.testing.assert_allclose(stored[key].sum(axis=-1), 1, atol=3e-7, rtol=0)
        if p is p0:
            exact(stored['g_attention_input'], stored['g_attention_output'])
            exact(stored['real_attention_input'], stored['real_attention_output'])
    derivative_values, derivative_error, differences = 0, 0., 0
    stages = [('initial', p0, [0, 1]),
              ('advanced', {k: v.astype(float) for k, v in arrays(root / 'probe-advanced.npz').items()}, [0, 1]),
              ('after-d', {k: v.astype(float) for k, v in arrays(root / 'probe-after-d.npz').items()}, [0]),
              ('g-advanced', {k: v.astype(float) for k, v in arrays(root / 'probe-g-advanced.npz').items()}, [0]),
              ('final', pf, [0, 1])]
    for stage, p, groups in stages:
        analytic = gradients(p, x, z)
        for group in groups:
            recorded = arrays(directory / f'final_gradients_group{group}.npz' if stage == 'final' else root / f'probe-{stage}-gradient-{group}.npz')
            assert set(recorded) == set(analytic[group])
            for key, partial in analytic[group].items():
                np.testing.assert_allclose(recorded[key], partial, atol=4e-6, rtol=8e-4, err_msg=stage + '/' + key)
                derivative_values += partial.size
                derivative_error = max(derivative_error, float(abs(recorded[key] - partial).max()))
                if stage != 'final':
                    continue
                objective = 'generator_loss' if group == 0 else 'discriminator_loss'
                for index in sorted({0, partial.size // 2, partial.size - 1}):
                    plus, minus = dict(p), dict(p)
                    plus[key], minus[key] = p[key].copy(), p[key].copy()
                    plus[key].flat[index] += 1e-5
                    minus[key].flat[index] -= 1e-5
                    finite = (forward(plus, x, z, False)[objective] - forward(minus, x, z, False)[objective]) / 2e-5
                    np.testing.assert_allclose(partial.flat[index], finite, atol=3e-9, rtol=3e-4, err_msg=key)
                    differences += 1
    # Explicit state/D/state/G phase ownership and independent next-vector calculations.
    for before, after, owner in [('initial', 'advanced', 3), ('advanced', 'after-d', 1), ('after-d', 'g-advanced', 2), ('g-advanced', 'after-g', 0)]:
        left, right = arrays(root / f'probe-{before}.npz'), arrays(root / f'probe-{after}.npz')
        assert any(not np.array_equal(left[k], right[k]) for k in left if k.startswith(str(owner) + '/'))
        for k in left:
            if not k.startswith(str(owner) + '/'):
                exact(left[k], right[k])
        if owner in [2, 3]:
            expected = advance({k: v.astype(float) for k, v in left.items()}, owner - 2)
            for k, v in expected.items():
                np.testing.assert_allclose(right[k], v, atol=3e-6, rtol=5e-5, err_msg=after + '/' + k)
    controls = arrays(directory / 'attention_controls.npz')
    for group in [0, 1]:
        key = str(group) + '/'
        a = controls[key + 'input'].astype(float)
        modified = a.copy()
        modified[:, 0, 0] = (a[:, 0, 0].astype(np.float32) + np.float32(1)).astype(float)
        exact(modified, controls[key + 'modified_input'].astype(float))
        normal, _ = attention(pf, group, a)
        changed, _ = attention(pf, group, modified)
        for field in ['output', 'probability']:
            np.testing.assert_allclose(controls[key + field], normal[field], atol=3e-6, rtol=3e-5)
        np.testing.assert_allclose(controls[key + 'modified_output'], changed['output'], atol=3e-6, rtol=3e-5)
        assert abs(controls[key + 'output'][:, 1:] - controls[key + 'modified_output'][:, 1:]).max() > 1e-9
    noise_control = arrays(directory / 'controls.npz')
    exact(noise_control['changed_z'], gaussian((200, 16), 4877).astype(np.float32))
    exact(noise_control['zero_z'], np.zeros((200, 16), np.float32))
    for prefix in ['changed', 'zero']:
        generated, _, _ = generate(pf, noise_control[prefix + '_z'].astype(float))
        np.testing.assert_allclose(noise_control[prefix + '_images'], generated, atol=3e-6, rtol=3e-5)
    metrics = json.loads((directory / 'metrics.json').read_text())
    assert metrics['steps_per_group'] == 96 and metrics['optimizer_updates'] == 192
    output = arrays(directory / 'outputs.npz')
    for field in ['discriminator_loss', 'generator_loss', 'pixel_mmd']:
        key = 'test_coarse_pixel_mmd' if field == 'pixel_mmd' else 'test_' + field
        assert metrics[key] == float(output[field])
    return dict(passed=True, forward_values_checked=int(values), maximum_forward_error=error,
                native_derivative_values_checked=derivative_values, maximum_derivative_error=derivative_error,
                independent_final_scalar_finite_differences=differences, checkpoint_values_exact=3735,
                state_values_checked=355, warmup_iterations_per_network=16, batch_rows_checked=len(batch),
                explicit_D_state_D_G_state_fresh_G=True, unscaled_attention=True,
                initial_zero_gamma_identity=True, final_nonlocal_effect_in_both_networks=True, metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('build/nn-report-conversion/self_attention_gan'))
    parser.add_argument('--data', type=Path, default=Path('build/nn-report-conversion/datasets/saliency/saliency_data.npz'))
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    result = check(args.root, args.data)
    rendered = json.dumps(result, indent=2) + '\n'
    print(rendered, end='')
    if args.json:
        args.json.write_text(rendered)


if __name__ == '__main__':
    main()
