#!/usr/bin/env python3
"""Independent NumPy checks of captured InfoGAN priors, forwards and derivatives; no fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact, gaussian
from check_contractive_autoencoder import sigmoid
from check_auxiliary_classifier_gan import softmax, coarse, mmd


def prior(rows, seed):
    """Rejection-sampled categorical word followed by two midpoint uniform words per row."""
    rng = np.random.RandomState(seed)
    category, continuous, raw = [], [], []
    for row in range(rows):
        word = int(rng.randint(0, 2**32, dtype=np.uint32))
        while word >= 4294967290:
            word = int(rng.randint(0, 2**32, dtype=np.uint32))
        category.append(word % 10)
        raw.append(word)
        for axis in range(2):
            word = int(rng.randint(0, 2**32, dtype=np.uint32))
            raw.append(word)
            continuous.append(2*((word+.5)/4294967296.)-1)
    return dict(category=np.array(category, np.int32),
                continuous=np.array(continuous, np.float32).reshape(rows, 2),
                raw=np.array(raw, np.float64).reshape(rows, 3))


def generate(p, noise, category, continuous):
    joined = np.concatenate([noise, np.eye(10)[category], continuous], axis=1)
    hidden = np.tanh(joined@p['0/hidden/W']+p['0/hidden/b'])
    return sigmoid(hidden@p['0/output/W']+p['0/output/b']), (joined, hidden)


def recognize(p, image):
    hidden = np.tanh(image@p['2/hidden/W']+p['2/hidden/b'])
    logits = hidden@p['2/category/W']+p['2/category/b']
    mean = hidden@p['2/continuous/W']+p['2/continuous/b']
    return logits, softmax(logits), mean, hidden


def forward(p, image, noise, category, continuous, diagnostic=True):
    fake, (joined, gh) = generate(p, noise, category, continuous)
    rh = np.tanh(image@p['1/hidden/W']+p['1/hidden/b'])
    fh = np.tanh(fake@p['1/hidden/W']+p['1/hidden/b'])
    real = rh@p['1/output/W']+p['1/output/b']
    generated = fh@p['1/output/W']+p['1/output/b']
    ql, qp, qm, qh = recognize(p, fake)
    cat = -(np.eye(10)[category]*(ql-np.logaddexp.reduce(ql, axis=1, keepdims=True))).sum(axis=1).mean()
    cont = .5*((continuous-qm)**2).sum(axis=1).mean()+np.log(2*np.pi)
    total = cat+cont
    source = np.logaddexp(0, -generated).mean()
    dl = np.logaddexp(0, -real).mean()+np.logaddexp(0, generated).mean()
    out = dict(fake=fake, real_logit=real, fake_logit=generated,
               real_probability=sigmoid(real), fake_probability=sigmoid(generated),
               discriminator_loss=dl, generator_loss=source+total,
               source_generator_loss=source, recognition_loss=total,
               categorical_loss=cat, continuous_loss=cont,
               mi_lower_bound=np.log(10)+2*np.log(2)-total,
               q_logits=ql, q_probability=qp, q_mean=qm)
    if diagnostic:
        out['pixel_mmd'] = mmd(coarse(image), coarse(fake))
    return out, (joined, gh, rh, fh, qh)


def gradients(p, image, noise, category, continuous):
    out, (joined, gh, rh, fh, qh) = forward(p, image, noise, category, continuous, False)
    n, m = len(image), len(noise)
    dr, df = -sigmoid(-out['real_logit'])/n, sigmoid(out['fake_logit'])/m
    rd = (dr@p['1/output/W'].T)*(1-rh*rh)
    fd = (df@p['1/output/W'].T)*(1-fh*fh)
    qc = (out['q_probability']-np.eye(10)[category])/m
    qu = (out['q_mean']-continuous)/m
    qd = (qc@p['2/category/W'].T+qu@p['2/continuous/W'].T)*(1-qh*qh)
    grad = {'1/hidden/W': image.T@rd+out['fake'].T@fd,
            '1/hidden/b': rd.sum(0)+fd.sum(0),
            '1/output/W': rh.T@dr+fh.T@df, '1/output/b': dr.sum(0)+df.sum(0),
            '2/hidden/W': out['fake'].T@qd, '2/hidden/b': qd.sum(0),
            '2/category/W': qh.T@qc, '2/category/b': qc.sum(0),
            '2/continuous/W': qh.T@qu, '2/continuous/b': qu.sum(0)}
    gp = (((-sigmoid(-out['fake_logit'])/m)@p['1/output/W'].T)*(1-fh*fh))@p['1/hidden/W'].T
    gp += qd@p['2/hidden/W'].T
    gp *= out['fake']*(1-out['fake'])
    gd = (gp@p['0/output/W'].T)*(1-gh*gh)
    grad.update({'0/hidden/W': joined.T@gd, '0/hidden/b': gd.sum(0),
                 '0/output/W': gh.T@gp, '0/output/b': gp.sum(0)})
    return grad


def check(directory, data, native_gradients, first_epoch_prior):
    raw = arrays(data)
    images = {s: (raw[s+'_images'].astype(np.float32)/np.float32(255)).reshape(-1, 784).astype(np.float64)
              for s in ['fit', 'validation', 'test']}
    p0 = {k: v.astype(np.float64) for k, v in arrays(directory/'initial_arrays.npz').items()}
    pf = {k: v.astype(np.float64) for k, v in arrays(directory/'final.npz').items()}
    assert len(p0) == 14 and sum(v.size for v in p0.values()) == 77469
    batch = np.genfromtxt(directory/'batches.tsv', names=True, delimiter='\t', dtype=np.int64)
    for epoch in [1, 2, 3]:
        rows = batch[batch['epoch'] == epoch]
        exact(np.sort(rows['row']), np.arange(1000))
        exact(rows['source_id'], raw['fit_source_ids'][rows['row']].astype(np.int64))
        assert len(np.unique(rows['step'])) == 32
    assert len(batch) == 3000 and len(np.unique(batch['step'])) == 96
    fixed = {}
    exact_prior_values = 0
    for split, seed, noise_seed in [('validation', 5347, 5323), ('test', 5351, 5333)]:
        expected = dict(prior(200, seed), z=gaussian((200, 16), noise_seed).astype(np.float32))
        stored = arrays(directory/(split+'_prior.npz'));assert set(stored) == set(expected)
        for key, value in expected.items():
            exact(stored[key], value);exact_prior_values += value.size
        fixed[split] = stored
    first = batch[batch['step'] == 0]['row']
    bank = dict(prior(1000, 5382), z=gaussian((1000, 16), 5364).astype(np.float32))
    if first_epoch_prior:
        stored = arrays(first_epoch_prior)
        for key, value in bank.items():exact(stored[key], value);exact_prior_values += value.size
    x, z = images['fit'][first], bank['z'][first].astype(np.float64)
    category, continuous = bank['category'][first], bank['continuous'][first].astype(np.float64)
    test = fixed['test']
    values, largest = 0, 0.
    for name, p, image, noise, c, u in [
            ('first_forward', p0, x, z, category, continuous),
            ('initial_outputs', p0, images['test'], test['z'], test['category'], test['continuous']),
            ('outputs', pf, images['test'], test['z'], test['category'], test['continuous'])]:
        computed = forward(p, image, noise.astype(np.float64), c, u.astype(np.float64))[0]
        stored = arrays(directory/(name+'.npz'));assert set(computed) == set(stored)
        for key, value in computed.items():
            np.testing.assert_allclose(stored[key], value, atol=3e-6, rtol=4e-6, err_msg=name+'/'+key)
            values += np.size(value);largest = max(largest, float(abs(stored[key]-value).max()))
    analytic, recorded = gradients(p0, x, z, category, continuous), arrays(native_gradients)
    assert set(analytic) == set(recorded)
    error, differences = 0., 0
    for key, value in analytic.items():
        np.testing.assert_allclose(recorded[key], value, atol=3e-6, rtol=3e-4, err_msg=key)
        error = max(error, float(abs(recorded[key]-value).max()))
        objective = 'generator_loss' if key.startswith('0/') else 'discriminator_loss' if key.startswith('1/') else 'recognition_loss'
        for index in sorted({0, value.size//2, value.size-1}):
            old = p0[key].flat[index];h = 1e-5
            p0[key].flat[index] = old+h;plus = forward(p0, x, z, category, continuous, False)[0][objective]
            p0[key].flat[index] = old-h;minus = forward(p0, x, z, category, continuous, False)[0][objective];p0[key].flat[index] = old
            np.testing.assert_allclose((plus-minus)/(2*h), value.flat[index], atol=3e-9, rtol=1e-4)
            differences += 1
    out = arrays(directory/'outputs.npz')
    image_only = arrays(directory/'image_only_recognition.npz')
    for key in ['q_probability', 'q_mean']:exact(image_only[key], out[key]);values += out[key].size
    categorical = arrays(directory/'category_traversal.npz')
    exact(categorical['z'], np.tile(test['z'][:10], (10, 1)))
    exact(categorical['category'], np.repeat(np.arange(10, dtype=np.int32), 10))
    exact(categorical['continuous'], np.zeros((100, 2), np.float32))
    cont_effect = 0.
    for axis in [0, 1]:
        traversal = arrays(directory/('continuous'+str(axis)+'_traversal.npz'))
        exact(traversal['z'], np.repeat(test['z'][:5], 11, axis=0))
        exact(traversal['category'], np.zeros(55, np.int32))
        u = np.zeros((55, 2), np.float32)
        # Source explicitly performs the multiply and add in Float32.
        u[:, axis] = np.tile(np.float32(-1)+np.float32(.2)*np.arange(11, dtype=np.float32), 5)
        exact(traversal['continuous'], u)
        expected = generate(pf, traversal['z'].astype(np.float64), traversal['category'], u.astype(np.float64))[0]
        np.testing.assert_allclose(traversal['fake'], expected, atol=3e-6, rtol=4e-6);values += expected.size
        cont_effect = max(cont_effect, float(abs(traversal['fake'][10::11].astype(np.float64)-traversal['fake'][::11]).max()))
    expected = generate(pf, categorical['z'].astype(np.float64), categorical['category'], categorical['continuous'].astype(np.float64))[0]
    np.testing.assert_allclose(categorical['fake'], expected, atol=3e-6, rtol=4e-6);values += expected.size
    cat_effect = float(abs(categorical['fake'][10:].astype(np.float64)-np.tile(categorical['fake'][:10], (9, 1))).max())
    metrics = json.loads((directory/'metrics.json').read_text())
    assert metrics['steps_per_group'] == 96 and metrics['optimizer_updates'] == 288 and metrics['selected_epoch_fixed_final'] == 3
    for key in ['discriminator_loss','generator_loss','recognition_loss','categorical_loss','continuous_loss','mi_lower_bound','pixel_mmd']:
        assert metrics['test_'+key] == float(out[key])
    assert metrics['category_pixel_max_effect'] == cat_effect and metrics['continuous_pixel_max_effect'] == cont_effect
    assert metrics['generated_category_recognition_accuracy'] == float((out['q_probability'].argmax(1) == test['category']).mean())
    np.testing.assert_allclose(metrics['generated_continuous_recognition_mse'], ((test['continuous'].astype(np.float64)-out['q_mean'])**2).mean(), atol=1e-14, rtol=1e-14)
    return dict(passed=True,forward_values_checked=int(values),maximum_forward_error=largest,
                initial_native_derivatives_checked=77469,maximum_initial_derivative_error=error,
                independent_scalar_finite_differences=differences,exact_prior_and_noise_values=int(exact_prior_values),
                batch_rows_checked=3000,steps_per_group=96,optimizer_updates=288,fixed_final_epoch=3,
                Q_is_image_only=True,generated_prior_categories_are_not_digit_labels=True,
                code_effects_are_not_semantic_disentanglement=True,metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--native-gradients',type=Path,required=True);parser.add_argument('--first-epoch-prior',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args();result=check(args.directory,args.data,args.native_gradients,args.first_epoch_prior)
    if args.report:args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
