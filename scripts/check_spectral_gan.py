#!/usr/bin/env python3
"""Independent NumPy checks of the frozen cached-vector spectral GAN; no training."""
import argparse
import json
from pathlib import Path

import numpy as np

from check_adversarial_autoencoder import arrays, exact, gaussian
from check_auxiliary_classifier_gan import coarse, mmd
from check_contractive_autoencoder import sigmoid
from check_dense_gans import losses


FLOOR = float(np.float32(1e-6))
EPSILON = float(np.float32(1e-12))


def effective(p, name):
    w = p['1/' + name + '/W']
    u, v = p['2/' + name + '/u'], p['2/' + name + '/v']
    sigma = np.sum(v * (w @ u))
    return w / max(sigma, FLOOR), sigma


def advance(p, iterations=1):
    result = {k: v.copy() for k, v in p.items()}
    for name in ['hidden', 'output']:
        w, u = p['1/' + name + '/W'], p['2/' + name + '/u']
        for _ in range(iterations):
            v = w @ u
            v = v / np.sqrt(np.sum(v * v) + EPSILON)
            u = w.T @ v
            u = u / np.sqrt(np.sum(u * u) + EPSILON)
        result['2/' + name + '/u'], result['2/' + name + '/v'] = u, v
    return result


def forward(p, image, noise, diagnostic=True):
    wh, sh = effective(p, 'hidden')
    wo, so = effective(p, 'output')
    hidden = np.tanh(noise @ p['0/hidden/W'] + p['0/hidden/b'])
    fake = sigmoid(hidden @ p['0/output/W'] + p['0/output/b'])
    rh = np.tanh(image @ wh + p['1/hidden/b'])
    fh = np.tanh(fake @ wh + p['1/hidden/b'])
    real = rh @ wo + p['1/output/b']
    generated = fh @ wo + p['1/output/b']
    dl, gl = losses(real, generated, 'hinge_gan')
    out = dict(fake=fake, real_score=real, fake_score=generated,
               discriminator_loss=dl, generator_loss=gl)
    if diagnostic:
        out['pixel_mmd'] = mmd(coarse(image), coarse(fake))
    return out, (hidden, rh, fh, wh, wo, sh, so)


def gradients(p, image, noise):
    out, (hidden, rh, fh, wh, wo, sh, so) = forward(p, image, noise, False)
    dr = -(out['real_score'] < 1).astype(float) / len(image)
    df = (out['fake_score'] > -1).astype(float) / len(noise)
    gf = -np.ones_like(out['fake_score']) / len(noise)
    rd = (dr @ wo.T) * (1 - rh * rh)
    fd = (df @ wo.T) * (1 - fh * fh)
    dh, do = image.T @ rd + out['fake'].T @ fd, rh.T @ dr + fh.T @ df

    def raw_gradient(name, derivative, sigma):
        w = p['1/' + name + '/W']
        denominator = max(sigma, FLOOR)
        result = derivative / denominator
        if sigma > FLOOR:
            # u/v are fixed inputs to backprop; sigma remains differentiable in W.
            u, v = p['2/' + name + '/u'], p['2/' + name + '/v']
            result -= np.sum(derivative * w) / sigma**2 * (v @ u.T)
        return result

    d = {'hidden/W': raw_gradient('hidden', dh, sh),
         'hidden/b': rd.sum(axis=0) + fd.sum(axis=0),
         'output/W': raw_gradient('output', do, so),
         'output/b': dr.sum(axis=0) + df.sum(axis=0)}
    pixels = ((gf @ wo.T) * (1 - fh * fh)) @ wh.T
    pixels *= out['fake'] * (1 - out['fake'])
    gh = (pixels @ p['0/output/W'].T) * (1 - hidden * hidden)
    g = {'hidden/W': noise.T @ gh, 'hidden/b': gh.sum(axis=0),
         'output/W': hidden.T @ pixels, 'output/b': pixels.sum(axis=0)}
    return g, d


def check(root, data):
    directory = root / 'original-output'
    raw = arrays(data)
    p0 = {k: v.astype(float) for k, v in arrays(directory / 'initial_arrays.npz').items()}
    pf = {k: v.astype(float) for k, v in arrays(directory / 'final.npz').items()}
    assert len(p0) == 12 and sum(v.size for v in p0.values()) == 52418
    probe_initial = arrays(root / 'probe-initial.npz')
    for k, v in arrays(directory / 'initial_arrays.npz').items():
        exact(probe_initial[k], v)
    before = {k: v.copy() for k, v in p0.items()}
    for name in ['hidden', 'output']:
        for suffix in ['u', 'v']:
            width = p0['2/' + name + '/' + suffix].size
            # Constants are rounded once in Float32 even for Float64 stores.
            initial = np.float32(1) / np.sqrt(np.float32(width))
            before['2/' + name + '/' + suffix] = np.full((width, 1), float(initial))
    warmed = advance(before, 16)
    for k in p0:
        np.testing.assert_allclose(warmed[k], p0[k], atol=2e-6, rtol=3e-5, err_msg='warmup/' + k)
    advanced = {k: v.astype(float) for k, v in arrays(root / 'probe-advanced.npz').items()}
    calculated = advance(p0)
    for k in p0:
        if k.startswith('2/'):
            np.testing.assert_allclose(calculated[k], advanced[k], atol=2e-6, rtol=3e-5, err_msg='advance/' + k)
        else:
            exact(p0[k], advanced[k])
    images = {s: (raw[s + '_images'].astype(np.float32) / np.float32(255)).reshape(-1, 784).astype(float)
              for s in ['fit', 'validation', 'test']}
    batch = np.genfromtxt(directory / 'batches.tsv', names=True, delimiter='\t', dtype=np.int64)
    for epoch in [1, 2, 3]:
        rows = batch[batch['epoch'] == epoch]
        exact(np.sort(rows['row']), np.arange(1000))
        exact(rows['source_id'], raw['fit_source_ids'][rows['row']].astype(np.int64))
        assert len(np.unique(rows['step'])) == 32
    first = batch[batch['step'] == 0]['row']
    x = images['fit'][first]
    z = gaussian((1000, 16), 4862).astype(np.float32).astype(float)[first]
    tz = gaussian((200, 16), 4831).astype(np.float32).astype(float)
    values, error = 0, 0.
    for name, p, image, noise in [('first_forward', p0, x, z),
                                ('initial_outputs', p0, images['test'], tz),
                                ('outputs', pf, images['test'], tz)]:
        expected, _ = forward(p, image, noise)
        stored = arrays(directory / (name + '.npz'))
        assert set(expected) == set(stored)
        for k, v in expected.items():
            np.testing.assert_allclose(stored[k], v, atol=3e-6, rtol=1e-5, err_msg=name + '/' + k)
            values += np.size(v)
            error = max(error, float(abs(stored[k] - v).max()))
    gradient_values, gradient_error, differences = 0, 0., 0
    for stage, p, groups in [('initial', p0, [0, 1]), ('advanced', advanced, [0, 1]),
                             ('after-d', {k: v.astype(float) for k, v in arrays(root / 'probe-after-d.npz').items()}, [0])]:
        analytic = gradients(p, x, z)
        for group in groups:
            stored = {k.removeprefix(str(group) + '/'): v for k, v in
                      arrays(root / f'probe-{stage}-gradient-{group}.npz').items()}
            assert set(stored) == set(analytic[group])
            for k, v in analytic[group].items():
                np.testing.assert_allclose(stored[k], v, atol=3e-6, rtol=4e-4, err_msg=f'{stage}/{group}/{k}')
                gradient_values += v.size
                gradient_error = max(gradient_error, float(abs(stored[k] - v).max()))
                if stage != 'initial':
                    continue
                # Check independent closed derivatives against scalar central differences.
                key = str(group) + '/' + k
                objective = 'generator_loss' if group == 0 else 'discriminator_loss'
                for index in sorted({0, v.size // 2, v.size - 1}):
                    h = 1e-5
                    plus, minus = dict(p), dict(p)
                    plus[key], minus[key] = p[key].copy(), p[key].copy()
                    plus[key].flat[index] += h
                    minus[key].flat[index] -= h
                    finite = (forward(plus, x, z, False)[0][objective] - forward(minus, x, z, False)[0][objective]) / (2 * h)
                    np.testing.assert_allclose(v.flat[index], finite, atol=3e-9, rtol=2e-4)
                    differences += 1
    # Explicitly check ownership for the first ordered D and fresh G writes.
    after_d, after_g = arrays(root / 'probe-after-d.npz'), arrays(root / 'probe-after-g.npz')
    for left, right, prefix in [(arrays(root / 'probe-advanced.npz'), after_d, '1/'), (after_d, after_g, '0/')]:
        assert any(not np.array_equal(left[k], right[k]) for k in left if k.startswith(prefix))
        for k in left:
            if not k.startswith(prefix):
                exact(left[k], right[k])
    metrics = json.loads((directory / 'metrics.json').read_text())
    assert len(batch) == 3000 and len(np.unique(batch['step'])) == 96
    assert metrics['steps_per_group'] == 96 and metrics['optimizer_updates'] == 192
    assert metrics['selected_epoch_fixed_final'] == metrics['epochs'] == 3
    output = arrays(directory / 'outputs.npz')
    for key, field in [('test_discriminator_loss', 'discriminator_loss'),
                       ('test_generator_loss', 'generator_loss'),
                       ('test_coarse_pixel_mmd', 'pixel_mmd')]:
        assert metrics[key] == float(output[field])
    assert metrics['test_discriminator_real_accuracy'] == float((output['real_score'] >= 0).mean())
    assert metrics['test_discriminator_fake_accuracy'] == float((output['fake_score'] < 0).mean())
    generated = output['fake'].astype(float)
    np.testing.assert_allclose(metrics['generated_mean_pixel'], generated.mean(), atol=1e-14, rtol=0)
    np.testing.assert_allclose(metrics['generated_mean_per_pixel_variance'], generated.var(axis=0).mean(), atol=1e-14, rtol=0)
    fit_mean = np.tile(images['fit'].mean(axis=0).astype(np.float32), (200, 1))
    np.testing.assert_allclose(metrics['fit_mean_coarse_pixel_mmd'], mmd(coarse(images['test']), coarse(fit_mean)), atol=3e-7, rtol=0)
    for name in ['hidden', 'output']:
        # The source audit recomputes sigma and Wbar in Float64 before the SVD.
        w = pf['1/' + name + '/W']
        u, v = pf['2/' + name + '/u'], pf['2/' + name + '/v']
        norm = np.linalg.svd(w / max(np.sum(v * (w @ u)), 1e-6), compute_uv=False)[0]
        np.testing.assert_allclose(metrics['effective_spectral_norm_' + name], norm, atol=1e-12, rtol=1e-12)
    controls = arrays(directory / 'controls.npz')
    for key, seed in [('changed', 4877)]:
        exact(controls[key + '_z'], gaussian((200, 16), seed).astype(np.float32))
        produced = forward(pf, images['test'], controls[key + '_z'].astype(float), False)[0]['fake']
        np.testing.assert_allclose(controls[key + '_images'], produced, atol=2e-6, rtol=3e-6)
    exact(controls['zero_z'], np.zeros((200, 16), np.float32))
    np.testing.assert_allclose(controls['zero_images'], forward(pf, images['test'], controls['zero_z'].astype(float), False)[0]['fake'], atol=2e-6, rtol=3e-6)
    assert not np.array_equal(controls['changed_images'], output['fake'])
    assert not np.array_equal(controls['zero_images'], output['fake'])
    algebra = arrays(directory / 'hinge_algebra.npz')
    real, fake = algebra['real'].astype(float), algebra['fake'].astype(float)
    dl, gl = losses(real, fake, 'hinge_gan')
    np.testing.assert_allclose(algebra['discriminator_loss'], dl, atol=2e-6, rtol=1e-6)
    np.testing.assert_allclose(algebra['generator_loss'], gl, atol=2e-6, rtol=1e-6)
    for name, derivative in [('d_real_gradient', -(real < 1).astype(float) / len(real)),
                             ('d_fake_gradient', (fake > -1).astype(float) / len(fake)),
                             ('g_fake_gradient', -np.ones_like(fake) / len(fake))]:
        np.testing.assert_allclose(algebra[name], derivative, atol=0, rtol=2e-6)
    assert algebra['d_real_gradient'][2, 0] == algebra['d_fake_gradient'][2, 0] == 0
    assert algebra['g_fake_gradient'][2, 0] < 0
    return dict(passed=True, forward_values_checked=int(values), maximum_forward_error=error,
                native_derivative_values_checked=gradient_values, maximum_derivative_error=gradient_error,
                independent_scalar_finite_differences=differences, warmup_iterations=16,
                state_values_checked=849, checkpoint_values_exact=52418, batch_rows_checked=len(batch),
                control_pixels_checked=313600, hinge_boundary_and_extreme_gradients=True,
                first_state_D_fresh_G_ownership=True, finite_power_iteration_is_not_unit_norm_certificate=True,
                metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('build/nn-report-conversion/spectral_gan'))
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
