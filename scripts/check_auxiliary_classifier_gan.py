#!/usr/bin/env python3
"""Independent NumPy check of frozen ACGAN endpoints and isolated generator derivatives.

Reads native artifacts and already-prepared raw data only. No downloads, model
fitting or calls to datamunge; raw MT19937 Box-Muller is independently replayed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact, gaussian
from check_contractive_autoencoder import sigmoid


def softmax(x):
    e = np.exp(x-x.max(axis=1, keepdims=True))
    return e/e.sum(axis=1, keepdims=True)


def forward(p, image, noise, labels):
    onehot = np.eye(10)[labels]
    joined = np.concatenate([noise, onehot], axis=1)
    hidden = np.tanh(joined@p['0/hidden/W']+p['0/hidden/b'])
    fake = sigmoid(hidden@p['0/output/W']+p['0/output/b'])
    real_hidden = np.tanh(image@p['1/hidden/W']+p['1/hidden/b'])
    fake_hidden = np.tanh(fake@p['1/hidden/W']+p['1/hidden/b'])
    real = real_hidden@p['1/source/W']+p['1/source/b']
    generated = fake_hidden@p['1/source/W']+p['1/source/b']
    real_class = real_hidden@p['1/class/W']+p['1/class/b']
    fake_class = fake_hidden@p['1/class/W']+p['1/class/b']
    rp, fp = softmax(real_class), softmax(fake_class)
    real_ce = -np.log(rp[np.arange(len(labels)), labels]).mean()
    fake_ce = -np.log(fp[np.arange(len(labels)), labels]).mean()
    dloss = np.logaddexp(0, -real).mean()+np.logaddexp(0, generated).mean()
    gloss = np.logaddexp(0, -generated).mean()
    return dict(fake=fake, real_logit=real, fake_logit=generated,
                real_probability=sigmoid(real), fake_probability=sigmoid(generated),
                discriminator_loss=dloss+(real_ce+fake_ce), generator_loss=gloss+fake_ce,
                real_class_logits=real_class, fake_class_logits=fake_class,
                real_class_probability=rp, fake_class_probability=fp,
                real_class_loss=real_ce, fake_class_loss=fake_ce,
                source_discriminator_loss=dloss, source_generator_loss=gloss), (joined, hidden, fake_hidden)


def generator_gradient(p, image, noise, labels, auxiliary):
    out, (joined, hidden, disc) = forward(p, image, noise, labels)
    if auxiliary:
        ds = (out['fake_class_probability']-np.eye(10)[labels])/len(labels)
        df = (ds@p['1/class/W'].T)*(1-disc*disc)@p['1/hidden/W'].T
    else:
        ds = (out['fake_probability']-1)/len(labels)
        df = (ds@p['1/source/W'].T)*(1-disc*disc)@p['1/hidden/W'].T
    delta = df*out['fake']*(1-out['fake'])
    dh = (delta@p['0/output/W'].T)*(1-hidden*hidden)
    return {'0/output/W': hidden.T@delta, '0/output/b': delta.sum(axis=0),
            '0/hidden/W': joined.T@dh, '0/hidden/b': dh.sum(axis=0)}


def coarse(x):
    # Native 7x7 average pooling produces4x4, then explicitly divides by4.
    return x.reshape(-1, 4, 7, 4, 7).mean(axis=(2, 4)).reshape(-1, 16)/4


def mmd(x, y):
    def kernel(a, b):
        d = ((a[:, None, :]-b[None, :, :])**2).sum(axis=2)
        return sum(c/(c+d) for c in (1, 4, 16))/3
    xx, yy, xy = kernel(x, x), kernel(y, y), kernel(x, y)
    return (xx.sum()-len(x))/(len(x)*(len(x)-1))+(yy.sum()-len(y))/(len(y)*(len(y)-1))-2*xy.mean()


def check(directory, data):
    raw = arrays(data)
    initial = {k: v.astype(np.float64) for k, v in arrays(directory/'initial_arrays.npz').items()}
    final = {k: v.astype(np.float64) for k, v in arrays(directory/'final.npz').items()}
    assert sum(v.size for v in initial.values()) == 52219 and len(initial) == 10
    images = {s: (raw[s+'_images'].astype(np.float32)/np.float32(255)).reshape(-1, 784).astype(np.float64)
              for s in ('fit', 'validation', 'test')}
    labels = {s: raw[s+'_labels'].astype(np.int64) for s in images}
    batch = np.genfromtxt(directory/'batches.tsv', names=True, delimiter='\t', dtype=np.int64)
    for epoch in (1, 2, 3):
        rows = batch[batch['epoch'] == epoch]
        exact(np.sort(rows['row']), np.arange(1000))
        exact(rows['source_id'], raw['fit_source_ids'][rows['row']].astype(np.int64))
        assert len(np.unique(rows['step'])) == 32
    assert len(batch) == 3000 and len(np.unique(batch['step'])) == 96
    first = batch[batch['step'] == 0]['row']
    z = gaussian((1000, 16), 4862).astype(np.float32).astype(np.float64)[first]
    x, y = images['fit'][first], labels['fit'][first]
    noise = gaussian((200, 16), 4831).astype(np.float32).astype(np.float64)
    largest = 0.; values = 0
    for name, p, image, latent, target in [
        ('first_forward', initial, x, z, y),
        ('initial_outputs', initial, images['test'], noise, labels['test']),
        ('outputs', final, images['test'], noise, labels['test'])]:
        expected, _ = forward(p, image, latent, target)
        actual = arrays(directory/(name+'.npz'))
        assert set(actual) == set(expected) | {'pixel_mmd'}
        for key, value in expected.items():
            np.testing.assert_allclose(actual[key], value, atol=4e-6, rtol=4e-6, err_msg=name+'/'+key)
            largest = max(largest, float(np.max(np.abs(actual[key]-value)))); values += np.size(value)
        np.testing.assert_allclose(actual['pixel_mmd'], mmd(coarse(image), coarse(expected['fake'])), atol=2e-6, rtol=0)
    derivatives = 0; differences = 0
    for auxiliary in (False, True):
        gradient = generator_gradient(initial, x, z, y, auxiliary)
        reference = arrays(directory/('first_g_aux_gradients.npz' if auxiliary else 'first_g_source_gradients.npz'))
        for key, g in gradient.items():
            np.testing.assert_allclose(reference[key], g, atol=3e-7, rtol=2e-4, err_msg=key)
            derivatives += g.size
            # Verify the independent reverse algebra against scalar finite differences.
            for index in sorted({0, g.size//2, g.size-1}):
                old = initial[key].flat[index]; h = 1e-5
                objective = 'fake_class_loss' if auxiliary else 'source_generator_loss'
                initial[key].flat[index] = old+h; plus = forward(initial, x, z, y)[0][objective]
                initial[key].flat[index] = old-h; minus = forward(initial, x, z, y)[0][objective]
                initial[key].flat[index] = old
                np.testing.assert_allclose((plus-minus)/(2*h), g.flat[index], atol=2e-9, rtol=1e-5)
                differences += 1
    control = arrays(directory/'controls.npz');condition = arrays(directory/'conditioning_controls.npz')
    exact(control['changed_z'], gaussian((200, 16), 4877).astype(np.float32))
    exact(control['zero_z'], np.zeros((200, 16), dtype=np.float32))
    exact(condition['grid_labels'], np.repeat(np.arange(10, dtype=np.int32), 10))
    exact(condition['grid_z'], np.tile(noise[:10].astype(np.float32), (10, 1)))
    for pnoise, py, recorded in [(control['changed_z'], labels['test'], control['changed_images']),
                                (control['zero_z'], labels['test'], control['zero_images']),
                                (condition['grid_z'], condition['grid_labels'], condition['grid_images'])]:
        predicted = forward(final, np.zeros((len(py), 784)), pnoise.astype(np.float64), py)[0]['fake']
        np.testing.assert_allclose(recorded, predicted, atol=4e-6, rtol=4e-6)
        largest = max(largest, float(abs(recorded-predicted).max()));values += recorded.size
    output = arrays(directory/'outputs.npz');metrics = json.loads((directory/'metrics.json').read_text())
    exact(condition['wrong_real_logits'], output['real_logit'])
    exact(condition['wrong_real_class_probability'], output['real_class_probability'])
    exact(condition['image_only_real_class_probability'], output['real_class_probability'])
    fake = output['fake'].astype(np.float64)
    np.testing.assert_allclose(fake.mean(), metrics['generated_mean_pixel'], atol=1e-13, rtol=0)
    np.testing.assert_allclose(fake.var(axis=0).mean(), metrics['generated_mean_per_pixel_variance'], atol=1e-13, rtol=0)
    for key, value in [('test_discriminator_real_accuracy', np.mean(output['real_logit'] >= 0)),
                       ('test_discriminator_fake_accuracy', np.mean(output['fake_logit'] < 0)),
                       ('test_real_aux_class_accuracy', np.mean(output['real_class_probability'].argmax(1) == labels['test'])),
                       ('test_fake_requested_aux_class_accuracy', np.mean(output['fake_class_probability'].argmax(1) == labels['test']))]:
        assert metrics[key] == value
    effect = float(abs(condition['grid_images'][10:].astype(np.float64)-np.tile(condition['grid_images'][:10], (9, 1))).max())
    assert metrics['generator_class_probability_max_effect'] == effect
    assert metrics['discriminator_real_source_condition_max_effect'] == 0 and metrics['invalid_condition_controls_rejected'] == 3
    assert metrics['steps_per_group'] == 96 and metrics['optimizer_updates'] == 192 and metrics['selected_epoch_fixed_final'] == 3
    assert metrics['test_discriminator_loss'] == float(output['discriminator_loss']) and metrics['test_generator_loss'] == float(output['generator_loss'])
    assert metrics['test_coarse_pixel_mmd'] == float(output['pixel_mmd'])
    mean = images['fit'].astype(np.float32).mean(axis=0, dtype=np.float32).astype(np.float64)
    np.testing.assert_allclose(mmd(coarse(images['test']), coarse(np.tile(mean, (200, 1)))), metrics['fit_mean_coarse_pixel_mmd'], atol=2e-6, rtol=0)
    algebra = arrays(directory/'logistic_algebra.npz');real, generated = algebra['real'].astype(float), algebra['fake'].astype(float)
    np.testing.assert_allclose(algebra['d_real_gradient'], (sigmoid(real)-1)/4, atol=1e-8)
    np.testing.assert_allclose(algebra['d_fake_gradient'], sigmoid(generated)/4, atol=1e-8)
    np.testing.assert_allclose(algebra['g_fake_gradient'], (sigmoid(generated)-1)/4, atol=1e-8)
    return dict(passed=True,forward_values_checked=values,maximum_forward_error=largest,
                isolated_generator_derivatives_checked=derivatives,independent_scalar_finite_differences=differences,
                batch_rows_checked=3000,steps_per_group=96,optimizer_updates=192,fixed_final_epoch=3,
                independent_noise_and_condition_controls=True,image_only_discriminator_condition_invariance=True,
                coarse_MMD_is_not_image_quality=True,metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args();result = check(args.directory, args.data)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True);args.report.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
