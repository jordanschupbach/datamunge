#!/usr/bin/env python3
"""Independent NumPy conditional-GAN artifacts and first gradient check; no fitting.

Reads original prepared data and native artifacts only, with an optional native
initial-gradient probe. Both G and D receive the same explicit class condition.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact, gaussian
from check_contractive_autoencoder import sigmoid
from check_auxiliary_classifier_gan import coarse, mmd


def forward(p, image, noise, labels):
    condition = np.eye(10)[labels]
    joined = np.concatenate([noise, condition], axis=1)
    hidden = np.tanh(joined@p['0/hidden/W']+p['0/hidden/b'])
    fake = sigmoid(hidden@p['0/output/W']+p['0/output/b'])
    real_pair = np.concatenate([image, condition], axis=1)
    fake_pair = np.concatenate([fake, condition], axis=1)
    rh = np.tanh(real_pair@p['1/hidden/W']+p['1/hidden/b'])
    fh = np.tanh(fake_pair@p['1/hidden/W']+p['1/hidden/b'])
    real = rh@p['1/output/W']+p['1/output/b']
    generated = fh@p['1/output/W']+p['1/output/b']
    return dict(fake=fake, real_logit=real, fake_logit=generated,
                real_probability=sigmoid(real), fake_probability=sigmoid(generated),
                discriminator_loss=np.logaddexp(0, -real).mean()+np.logaddexp(0, generated).mean(),
                generator_loss=np.logaddexp(0, -generated).mean(),
                pixel_mmd=mmd(coarse(image), coarse(fake))), (joined, hidden, real_pair, fake_pair, rh, fh)


def gradients(p, image, noise, labels):
    o, (joined, hidden, real_pair, fake_pair, rh, fh) = forward(p, image, noise, labels)
    count = len(labels)
    real_delta = -sigmoid(-o['real_logit'])/count
    fake_delta = sigmoid(o['fake_logit'])/count
    real_hidden = (real_delta@p['1/output/W'].T)*(1-rh*rh)
    fake_hidden = (fake_delta@p['1/output/W'].T)*(1-fh*fh)
    d = {'1/hidden/W': real_pair.T@real_hidden+fake_pair.T@fake_hidden,
         '1/hidden/b': real_hidden.sum(axis=0)+fake_hidden.sum(axis=0),
         '1/output/W': rh.T@real_delta+fh.T@fake_delta,
         '1/output/b': real_delta.sum(axis=0)+fake_delta.sum(axis=0)}
    g_delta = -sigmoid(-o['fake_logit'])/count
    pixels = ((g_delta@p['1/output/W'].T)*(1-fh*fh))@p['1/hidden/W'][:784].T
    pixels *= o['fake']*(1-o['fake'])
    g_hidden = (pixels@p['0/output/W'].T)*(1-hidden*hidden)
    g = {'0/hidden/W': joined.T@g_hidden, '0/hidden/b': g_hidden.sum(axis=0),
         '0/output/W': hidden.T@pixels, '0/output/b': pixels.sum(axis=0)}
    return dict(g, **d)


def check(directory, data, native_gradients=None):
    raw = arrays(data)
    p0 = {k: v.astype(np.float64) for k, v in arrays(directory/'initial_arrays.npz').items()}
    pf = {k: v.astype(np.float64) for k, v in arrays(directory/'final.npz').items()}
    assert len(p0) == 8 and sum(v.size for v in p0.values()) == 52209
    images = {s: (raw[s+'_images'].astype(np.float32)/np.float32(255)).reshape(-1, 784).astype(np.float64)
              for s in ['fit', 'validation', 'test']}
    labels = {s: raw[s+'_labels'].astype(np.int64) for s in images}
    batch = np.genfromtxt(directory/'batches.tsv', names=True, delimiter='\t', dtype=np.int64)
    for epoch in [1, 2, 3]:
        rows = batch[batch['epoch'] == epoch]
        exact(np.sort(rows['row']), np.arange(1000))
        exact(rows['source_id'], raw['fit_source_ids'][rows['row']].astype(np.int64))
        assert len(np.unique(rows['step'])) == 32
    assert len(batch) == 3000 and len(np.unique(batch['step'])) == 96
    first = batch[batch['step'] == 0]['row']
    x, z, y = images['fit'][first], gaussian((1000, 16), 4862).astype(np.float32).astype(np.float64)[first], labels['fit'][first]
    test_z = gaussian((200, 16), 4831).astype(np.float32).astype(np.float64)
    values, largest = 0, 0.
    for name, p, image, noise, target in [('first_forward', p0, x, z, y),
            ('initial_outputs', p0, images['test'], test_z, labels['test']),
            ('outputs', pf, images['test'], test_z, labels['test'])]:
        expected, _ = forward(p, image, noise, target);recorded = arrays(directory/(name+'.npz'))
        assert set(expected) == set(recorded)
        for key, value in expected.items():
            np.testing.assert_allclose(recorded[key], value, atol=2e-6, rtol=3e-6, err_msg=name+'/'+key)
            largest = max(largest, float(abs(recorded[key]-value).max()));values += np.size(value)
    analytic = gradients(p0, x, z, y);derivatives = 0;gradient_error = 0.
    if native_gradients is not None:
        stored = arrays(native_gradients);assert set(stored) == set(analytic)
        for key, value in analytic.items():
            np.testing.assert_allclose(stored[key], value, atol=2e-6, rtol=3e-4, err_msg=key)
            derivatives += value.size;gradient_error = max(gradient_error, float(abs(stored[key]-value).max()))
    differences = 0
    for key, value in analytic.items():
        objective = 'generator_loss' if key.startswith('0/') else 'discriminator_loss'
        for index in sorted({0, value.size//2, value.size-1}):
            old = p0[key].flat[index];h = 1e-5
            p0[key].flat[index] = old+h;plus = forward(p0, x, z, y)[0][objective]
            p0[key].flat[index] = old-h;minus = forward(p0, x, z, y)[0][objective];p0[key].flat[index] = old
            np.testing.assert_allclose((plus-minus)/(2*h), value.flat[index], atol=2e-9, rtol=1e-4)
            differences += 1
    output, controls, conditions = [arrays(directory/name) for name in ['outputs.npz', 'controls.npz', 'conditioning_controls.npz']]
    exact(controls['changed_z'], gaussian((200, 16), 4877).astype(np.float32))
    exact(controls['zero_z'], np.zeros((200, 16), np.float32))
    for key, bank in [('changed_images', controls['changed_z']), ('zero_images', controls['zero_z'])]:
        expected = forward(pf, images['test'], bank.astype(np.float64), labels['test'])[0]['fake']
        np.testing.assert_allclose(controls[key], expected, atol=2e-6, rtol=3e-6);values += expected.size
        largest = max(largest, float(abs(controls[key]-expected).max()))
    exact(conditions['grid_z'], np.tile(test_z[:10], (10, 1)).astype(np.float32))
    exact(conditions['grid_labels'], np.repeat(np.arange(10), 10).astype(np.int32))
    expected = forward(pf, images['test'][:100], conditions['grid_z'].astype(np.float64), conditions['grid_labels'])[0]['fake']
    np.testing.assert_allclose(conditions['grid_images'], expected, atol=2e-6, rtol=3e-6);values += expected.size
    wrong = (labels['test']+1) % 10
    expected = forward(pf, images['test'], test_z, wrong)[0]['real_logit']
    np.testing.assert_allclose(conditions['wrong_real_logits'], expected, atol=2e-6, rtol=3e-6);values += expected.size
    assert not np.array_equal(output['real_logit'], conditions['wrong_real_logits'])
    algebra = arrays(directory/'logistic_algebra.npz');real, fake = algebra['real'].astype(np.float64), algebra['fake'].astype(np.float64)
    for key, value in [('d_real_gradient', -sigmoid(-real)/4), ('d_fake_gradient', sigmoid(fake)/4), ('g_fake_gradient', -sigmoid(-fake)/4)]:
        np.testing.assert_allclose(algebra[key], value, atol=0, rtol=2e-6)
    metrics = json.loads((directory/'metrics.json').read_text())
    assert metrics['steps_per_group'] == 96 and metrics['optimizer_updates'] == 192 and metrics['selected_epoch_fixed_final'] == 3
    for key, field in [('test_discriminator_loss', 'discriminator_loss'), ('test_generator_loss', 'generator_loss'), ('test_coarse_pixel_mmd', 'pixel_mmd')]:
        assert metrics[key] == float(output[field])
    assert metrics['invalid_condition_controls_rejected'] == 3
    assert metrics['generator_class_probability_max_effect'] == float(abs(conditions['grid_images'][10:].astype(np.float64)-np.tile(conditions['grid_images'][:10], (9, 1))).max())
    assert metrics['discriminator_real_class_logit_max_effect'] == float(abs(output['real_logit'].astype(np.float64)-conditions['wrong_real_logits']).max())
    np.testing.assert_allclose(metrics['fit_mean_coarse_pixel_mmd'],mmd(coarse(images['test']),coarse(np.tile(images['fit'].mean(axis=0).astype(np.float32),(200,1)))),atol=3e-7,rtol=0)
    return dict(passed=True,forward_values_checked=int(values),maximum_forward_error=largest,
                initial_native_derivatives_checked=derivatives,maximum_initial_derivative_error=gradient_error,
                independent_scalar_finite_differences=differences,batch_rows_checked=3000,steps_per_group=96,
                optimizer_updates=192,fixed_final_epoch=3,both_networks_conditioned=True,
                extreme_logit_derivatives_keep_tiny_paths=True,coarse_MMD_is_not_image_quality=True,metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True);parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--native-gradients', type=Path);parser.add_argument('--report', type=Path)
    args = parser.parse_args();result = check(args.directory, args.data, args.native_gradients)
    if args.report: args.report.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
