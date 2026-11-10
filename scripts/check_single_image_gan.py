#!/usr/bin/env python3
"""Independent NumPy SinGAN pyramid, adjoints and cold-Adam trajectory audit.

Reads frozen native arrays and one prepared CIFAR row; never prepares data or
fits a model. Checks every fitting batch and frozen lower chain, both warm-start
groups, independent optimizer clocks, all inference sizes and all noise controls.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from check_adversarial_autoencoder import arrays, exact, gaussian
from check_adversarial_sr import bicubic, layer


def noise(n, h, w, scale, seed):
    """One midpoint-MT Gaussian stream, shared across RGB only at scale zero."""
    value = gaussian((n, h, w, 3 if scale else 1), seed).astype(np.float32)
    return value if scale else value.repeat(3, axis=3)


def generate(p, previous, z, amplitude):
    """Three biased SAME convolutions and the live residual skip."""
    a, b1 = layer(p, 0, 'head', previous + z * amplitude)
    b, b2 = layer(p, 0, 'body', np.where(a > 0, a, .2 * a))
    c, b3 = layer(p, 0, 'output', np.where(b > 0, b, .2 * b))
    residual = np.tanh(c)

    def backward(dy, out):
        d = b3(dy * (1 - residual * residual), out)
        d = b2(d * np.where(b > 0, 1., .2), out)
        d = b1(d * np.where(a > 0, 1., .2), out)
        return dy + d, d * amplitude, (d * z).sum()

    return previous + residual, backward


def discriminate(p, image):
    """Raw local scores, without image pooling or a probability transform."""
    a, b1 = layer(p, 1, 'head', image)
    b, b2 = layer(p, 1, 'body', np.where(a > 0, a, .2 * a))
    score, b3 = layer(p, 1, 'output', np.where(b > 0, b, .2 * b))

    def backward(dy, out):
        d = b3(dy, out)
        d = b2(d * np.where(b > 0, 1., .2), out)
        return b1(d * np.where(a > 0, 1., .2), out)

    return score, backward


def forward(p, batch):
    """Full generator LSGAN, half discriminator LSGAN and ten-times fixed MSE."""
    x = {k: v.astype(float) for k, v in batch.items()}
    fake, bg = generate(p, x['previous'], x['noise'], x['amplitude'])
    rec, bc = generate(p, x['rec_previous'], x['rec_noise'], x['amplitude'])
    real_score, br = discriminate(p, x['image'])
    fake_score, bf = discriminate(p, fake)
    adversarial = ((fake_score - 1) ** 2).mean()
    reconstruction = ((rec - x['image']) ** 2).mean()
    out = dict(fake=fake, reconstruction=rec, real_score=real_score,
               fake_score=fake_score, d_loss=.5 * (((real_score - 1) ** 2).mean()
                                                + (fake_score ** 2).mean()),
               g_loss=adversarial + 10 * reconstruction,
               adversarial_loss=adversarial, reconstruction_loss=reconstruction)
    d, g = {}, {}
    br((real_score - 1) / real_score.size, d)
    bf(fake_score / fake_score.size, d)
    bg(bf(2 * (fake_score - 1) / fake_score.size, {}), g)
    bc(20 * (rec - x['image']) / rec.size, g)
    return out, {**g, **d}


def draw(models, amplitudes, fixed, count, n, seed, reconstruction=False,
         height=32, width=32, zero_scale=-1):
    """Independent frozen chain with source integer-truncated rectangular sizes."""
    value = np.zeros((n, height // 2, width // 2, 3), np.float32)
    out = {}
    for k in range(count):
        h = height // 2 if k == 0 else height * 3 // 4 if k == 1 else height
        w = width // 2 if k == 0 else width * 3 // 4 if k == 1 else width
        if k:
            value = bicubic(value, h, w)
        z = fixed[k] if reconstruction else noise(n, h, w, k, seed + 97 * k)
        if k == zero_scale:
            z = np.zeros_like(z)
        previous = value
        # Each native scale computes in Float32, including the inter-scale input.
        value = generate(models[k], previous.astype(float), z.astype(float),
                         float(amplitudes[k]))[0].astype(np.float32)
        out.update({f'scale{k}_noise': z, f'scale{k}_previous': previous,
                    f'scale{k}_fake': value})
    return value, out


def check(root):
    """Check all source checkpoints, 288 writes and explicit caller-owned sampling."""
    source = root / 'original-output'
    data = arrays(source / 'data.npz')
    pins = json.loads((root / 'baseline.json').read_text())
    row = np.fromfile(Path(pins['data_directory']) / 'data_batch_1.bin',
                      np.uint8, count=3073)
    image = row[1:].reshape(1, 3, 32, 32).transpose(0, 2, 3, 1)
    image = np.float32(2) * (image.astype(np.float32) / np.float32(255)) - np.float32(1)
    exact(data['source'], image)
    prepared = image.size
    fixed = []
    for k, size in enumerate([16, 24, 32]):
        target = image if k == 2 else bicubic(image, size, size)
        z = np.zeros((1, size, size, 3), np.float32) if k else noise(1, 16, 16, 0, 7411)
        exact(data[f'target{k}'], target)
        exact(data[f'fixed_noise{k}'], z)
        fixed.append(z)
        prepared += target.size + z.size

    counts = dict(forward_values=0, gradient_values=0, exact_weight_values=0,
                  batch_input_values=0, finite_differences=0, optimizer_writes=0)
    errors = dict(forward=0., gradient=0.)

    def compare(native, expected, label, gradient=False):
        assert set(native) == set(expected), (label, set(native) ^ set(expected))
        kind = 'gradient' if gradient else 'forward'
        for key, value in expected.items():
            np.testing.assert_allclose(native[key], value, atol=8e-6 if gradient else 3e-6,
                                       rtol=3e-3 if gradient else 2e-4,
                                       err_msg=label + '/' + key)
            counts[kind + '_values'] += np.size(value)
            errors[kind] = max(errors[kind], float(np.max(abs(native[key] - value))))

    models, amplitudes, results = [], [], []
    history = np.genfromtxt(source / 'history.tsv', names=True, delimiter='\t')
    steps = np.genfromtxt(source / 'steps.tsv', names=True, delimiter='\t', dtype=np.int64)
    assert len(history) == 9 and len(steps) == 144
    for k, size in enumerate([16, 24, 32]):
        folder, probe = source / f'scale{k}', root / f'scale{k}'
        initial = arrays(folder / 'initial_arrays.npz')
        assert len(initial) == 12
        assert sum(v.size for key, v in initial.items() if key[0] == '0') == 371
        assert sum(v.size for key, v in initial.items() if key[0] == '1') == 297
        if k:
            for key, value in arrays(source / f'scale{k-1}/final.npz').items():
                exact(initial[key], value)
        setup = arrays(probe / 'probe-setup.npz')
        lower_rec = bicubic(draw(models, amplitudes, fixed, k, 1, 0, True)[0], size, size) if k else np.zeros((1, 16, 16, 3), np.float32)
        compare({'previous_reconstruction': setup['previous_reconstruction']},
                {'previous_reconstruction': lower_rec}, f'scale{k}/setup')
        # Use captured native lower output to isolate the source Double RMS rounding.
        delta = setup['previous_reconstruction'].astype(float) - data[f'target{k}'].astype(float)
        amplitude = np.float32(.1 * np.sqrt((delta * delta).sum() / delta.size)) if k else np.float32(1)
        exact(setup['amplitude'], np.array([amplitude], np.float32))
        amplitudes.append(amplitude)
        exact(arrays(probe / 'probe-initial-counters.npz')['iterations'], np.zeros(2, np.int64))
        before = initial
        moment = {key: np.zeros_like(value) for key, value in initial.items()}
        variance = {key: np.zeros_like(value) for key, value in initial.items()}
        iterations = [0, 0]
        stale_difference = 0.
        for step in range(48):
            seed = 7501 + 10000 * k + 307 * step
            record = steps[48 * k + step]
            assert tuple(record) == (k, 1 + step // 16, step, seed)
            batch = arrays(probe / f'probe-batch{step}-input.npz')
            assert set(batch) == {'image', 'previous', 'noise', 'rec_previous', 'rec_noise', 'amplitude'}
            for field, value in [('image', data[f'target{k}']), ('rec_previous', setup['previous_reconstruction']),
                                 ('rec_noise', fixed[k]), ('amplitude', setup['amplitude']),
                                 ('noise', noise(1, size, size, k, seed + 211))]:
                exact(batch[field], value)
                counts['batch_input_values'] += value.size
            previous = bicubic(draw(models, amplitudes, fixed, k, 1, seed)[0], size, size) if k else lower_rec
            compare({'previous': batch['previous']}, {'previous': previous}, f'scale{k}/batch{step}/lower')
            if step == 0:
                diagnostic = batch
                p = {key: value.astype(float) for key, value in before.items()}
                out, partials = forward(p, batch)
                compare(arrays(probe / 'probe-first-forward.npz'), out, f'scale{k}/first')
                compare(arrays(probe / 'probe-stale-g-gradient.npz'),
                        {key: value for key, value in partials.items() if key[0] == '0'}, f'scale{k}/stale', True)
            for group, suffix in [(1, 'd'), (0, 'g')]:
                label = f'batch{step}-{suffix}'
                native = arrays(probe / 'phases' / (label + '-gradient.npz'))
                after_all = arrays(probe / 'phases' / (label + '-after.npz'))
                after = {key.split('/', 1)[1]: value for key, value in after_all.items()
                         if key.startswith(f'scale{k}/')}
                _, partials = forward({key: value.astype(float) for key, value in before.items()}, batch)
                compare(native, {key: value for key, value in partials.items() if key[0] == str(group)},
                        f'scale{k}/' + label, True)
                if step == 0 and group == 0:
                    stale = arrays(probe / 'probe-stale-g-gradient.npz')
                    stale_difference = max(float(abs(native[key] - stale[key]).max()) for key in native)
                    assert stale_difference > 1e-7
                iterations[group] += 1
                t = iterations[group]
                for key, value in before.items():
                    if key[0] == str(group):
                        g = native[key]
                        moment[key] = moment[key] * np.float32(.5) + g * np.float32(.5)
                        variance[key] = variance[key] * np.float32(.999) + (g * g) * np.float32(.001)
                        m = moment[key] / np.float32(1 - .5 ** t)
                        v = variance[key] / np.float32(1 - .999 ** t)
                        exact(value - m / (np.sqrt(v) + np.float32(1e-8)) * np.float32(.0005), after[key])
                    else:
                        exact(value, after[key])
                    counts['exact_weight_values'] += value.size
                assert len(after_all) == 12 * (k + 1)
                for lower in range(k):
                    for key, value in arrays(source / f'scale{lower}/final.npz').items():
                        exact(value, after_all[f'scale{lower}/' + key])
                        counts['exact_weight_values'] += value.size
                exact(arrays(probe / 'phases' / (label + '-iterations.npz'))['iterations'],
                      np.array(iterations, np.int64))
                before = after
                counts['optimizer_writes'] += 1
            if step % 16 == 15:
                epoch = 1 + step // 16
                checkpoint = arrays(probe / f'probe-epoch{epoch}.npz')
                for key, value in checkpoint.items():
                    exact(value, before[key])
                native_out = arrays(probe / f'probe-epoch{epoch}-forward.npz')
                compare(native_out, forward({key: value.astype(float) for key, value in before.items()}, diagnostic)[0],
                        f'scale{k}/epoch{epoch}')
                row = history[3 * k + epoch - 1]
                assert row['scale'] == k and row['epoch'] == epoch
                for field in history.dtype.names[2:]:
                    assert row[field] == native_out[field].item()
        assert iterations == [48, 48]
        final = arrays(folder / 'final.npz')
        for key, value in final.items():
            exact(value, before[key])
        p = {key: value.astype(float) for key, value in final.items()}
        out, partials = forward(p, diagnostic)
        compare(arrays(folder / 'outputs.npz'), out, f'scale{k}/final')
        for group in [0, 1]:
            compare(arrays(folder / f'final_gradients_group{group}.npz'),
                    {key: value for key, value in partials.items() if key[0] == str(group)}, f'scale{k}/final', True)
        for key, gradient in partials.items():
            field = 'g_loss' if key[0] == '0' else 'd_loss'
            for index in sorted({0, gradient.size // 2, gradient.size - 1}):
                old, h = p[key].flat[index], 1e-6
                p[key].flat[index] = old + h
                plus = forward(p, diagnostic)[0][field]
                p[key].flat[index] = old - h
                minus = forward(p, diagnostic)[0][field]
                p[key].flat[index] = old
                np.testing.assert_allclose((plus - minus) / (2 * h), gradient.flat[index], atol=2e-6, rtol=2e-3,
                                           err_msg=f'scale{k}/' + key)
                counts['finite_differences'] += 1
        models.append(p)
        results.append(dict(scale=k, iterations=iterations, amplitude=float(amplitude),
                            stale_generator_gradient_max_difference=stale_difference))

    for file, n, seed, reconstruction, h, w in [('samples', 12, 7901, False, 32, 32),
                                               ('rectangular', 4, 7919, False, 40, 48),
                                               ('reconstruction', 1, 0, True, 32, 32)]:
        _, expected = draw(models, amplitudes, fixed, 3, n, seed, reconstruction, h, w)
        native = arrays(source / (file + '.npz'))
        compare(native, expected, file)
        for k in range(3):
            exact(native[f'scale{k}_noise'], expected[f'scale{k}_noise'])
    compare(arrays(source / 'noise_controls.npz'),
            {f'zero_noise_scale{k}': draw(models, amplitudes, fixed, 3, 12, 7901, zero_scale=k)[0] for k in range(3)}, 'controls')
    samples = arrays(source / 'samples.npz')['scale2_fake'].astype(float)
    reconstruction = arrays(source / 'reconstruction.npz')['scale2_fake'].astype(float)
    metrics = json.loads((source / 'metrics.json').read_text())
    mse = ((reconstruction - image.astype(float)) ** 2).mean()
    diversity = ((samples - samples.mean(axis=0, keepdims=True)) ** 2).mean()
    np.testing.assert_allclose(metrics['training_image_reconstruction_mse'], mse, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(metrics['sample_pixel_variance'], diversity, atol=1e-14, rtol=1e-14)
    exact(arrays(source / 'amplitudes.npz')['amplitudes'], np.array(amplitudes, np.float32))
    assert metrics['total_optimizer_updates'] == counts['optimizer_writes'] == 288
    assert not metrics['heldout_accuracy']
    return dict(passed=True, scales=results, **counts, maximum_errors=errors,
                prepared_image_pyramid_and_fixed_noise_values_exact=prepared,
                all_warm_start_groups_and_cold_clocks_exact=True,
                all_lower_stores_and_unselected_owner_bytes_exact=True,
                all_batch_seed_previous_noise_and_reconstruction_inputs_checked=True,
                all_288_Adam_writes_bit_exact=True, all_nine_history_rows_checked=True,
                all_shared_RGB_and_independent_RGB_noise_exact=True,
                final_reconstruction_mse=mse, final_sample_pixel_variance=diversity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('build/nn-report-conversion/singan'))
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    result = json.dumps(check(args.root), indent=2) + '\n'
    print(result, end='')
    if args.json:
        args.json.write_text(result)


if __name__ == '__main__':
    main()
