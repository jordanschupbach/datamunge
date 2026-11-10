#!/usr/bin/env python3
"""Independent NumPy BiGAN/ALI artifact check; no preparation or model training.

Replays procedural inputs, checks the joint-pair equations, separate G/E paths,
one analytic first D Adam step, and the stored G partial after that step.
Reconstruction and latent cycles are evaluated only as untrained diagnostics.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact, gaussian
from check_contractive_autoencoder import sigmoid


def points(count, seed):
    noise = gaussian((count, 2), seed)
    out = np.empty((count, 2), np.float32)
    for i in range(count):
        theta = 2*math.acos(-1)*(i % 8)/8
        out[i] = [2*math.cos(theta)+.08*noise[i, 0], 2*math.sin(theta)+.08*noise[i, 1]]
    return out


def mapping(p, group, x):
    hidden = np.tanh(x@p[group+'/hidden/W']+p[group+'/hidden/b'])
    return hidden@p[group+'/output/W']+p[group+'/output/b'], hidden


def joint(p, image, code):
    pair = np.concatenate([image, code], axis=1)
    hidden = np.tanh(pair@p['2/hidden/W']+p['2/hidden/b'])
    return hidden@p['2/output/W']+p['2/output/b'], hidden, pair


def forward(p, x, z):
    fake, gh = mapping(p, '0', z);encoded, eh = mapping(p, '1', x)
    real, rh, rp = joint(p, x, encoded);generated, fh, fp = joint(p, fake, z)
    reconstruction, _ = mapping(p, '0', encoded);cycle, _ = mapping(p, '1', fake)
    out = dict(fake=fake, encoded=encoded, real_score=real, fake_score=generated,
               discriminator_loss=np.logaddexp(0, -real).mean()+np.logaddexp(0, generated).mean(),
               eg_loss=np.logaddexp(0, real).mean()+np.logaddexp(0, -generated).mean(),
               reconstruction=reconstruction, latent_cycle=cycle)
    return out, (gh, eh, rh, fh, rp, fp)


def map_gradient(p, group, input_rows, hidden, delta):
    dh = (delta@p[group+'/output/W'].T)*(1-hidden*hidden)
    return {group+'/output/W': hidden.T@delta, group+'/output/b': delta.sum(axis=0),
            group+'/hidden/W': input_rows.T@dh, group+'/hidden/b': dh.sum(axis=0)}


def gradient(p, x, z, group):
    out, (gh, eh, rh, fh, rp, fp) = forward(p, x, z);count = len(x)
    if group == 2:
        dr = (sigmoid(out['real_score'])-1)/count;df = sigmoid(out['fake_score'])/count
        hr = (dr@p['2/output/W'].T)*(1-rh*rh);hf = (df@p['2/output/W'].T)*(1-fh*fh)
        return {'2/output/W': rh.T@dr+fh.T@df, '2/output/b': (dr+df).sum(axis=0),
                '2/hidden/W': rp.T@hr+fp.T@hf, '2/hidden/b': (hr+hf).sum(axis=0)}
    if group == 0:
        ds = (sigmoid(out['fake_score'])-1)/count
        dp = ((ds@p['2/output/W'].T)*(1-fh*fh))@p['2/hidden/W'].T
        return map_gradient(p, '0', z, gh, dp[:, :2])
    ds = sigmoid(out['real_score'])/count
    dp = ((ds@p['2/output/W'].T)*(1-rh*rh))@p['2/hidden/W'].T
    return map_gradient(p, '1', x, eh, dp[:, 2:])


def check(directory):
    data = arrays(directory/'data.npz');noise = arrays(directory/'fixed_noise.npz')
    for split, count, seed in [('fit', 400, 5507), ('validation', 128, 5519), ('test', 128, 5521)]:
        exact(data[split], points(count, seed))
    for split, seed in [('validation', 6043), ('test', 6047)]:
        exact(noise[split+'_z'], gaussian((128, 2), seed).astype(np.float32))
    p0 = {k: v.astype(np.float64) for k, v in arrays(directory/'initial_arrays.npz').items()}
    pf = {k: v.astype(np.float64) for k, v in arrays(directory/'final.npz').items()}
    assert len(p0) == 12 and sum(x.size for x in p0.values()) == 261
    batch = np.genfromtxt(directory/'batches.tsv', names=True, delimiter='\t', dtype=np.int64)
    for epoch in (1, 2, 3):
        rows = batch[batch['epoch'] == epoch];exact(np.sort(rows['row']), np.arange(400))
        assert len(np.unique(rows['step'])) == 13
    assert len(batch) == 1200 and len(np.unique(batch['step'])) == 39
    first = batch[batch['step'] == 0]['row'];x = data['fit'][first].astype(np.float64)
    z = gaussian((400, 2), 6068).astype(np.float32).astype(np.float64)[first]
    largest = 0.;values = 0
    for name, p, image, latent in [('first_forward', p0, x, z),
                                  ('initial_outputs', p0, data['test'].astype(np.float64), noise['test_z'].astype(np.float64)),
                                  ('outputs', pf, data['test'].astype(np.float64), noise['test_z'].astype(np.float64))]:
        expected, _ = forward(p, image, latent);recorded = arrays(directory/(name+'.npz'));assert set(expected) == set(recorded)
        for key, value in expected.items():
            np.testing.assert_allclose(recorded[key], value, atol=2e-6, rtol=3e-6, err_msg=name+'/'+key)
            values += np.size(value);largest = max(largest, float(abs(recorded[key]-value).max()))
    scalar_differences = 0
    for group in (0, 1, 2):
        g = gradient(p0, x, z, group);objective = 'discriminator_loss' if group == 2 else 'eg_loss'
        for key, derivative in g.items():
            for index in sorted({0, derivative.size//2, derivative.size-1}):
                old = p0[key].flat[index];h = 1e-5
                p0[key].flat[index] = old+h;plus = forward(p0, x, z)[0][objective]
                p0[key].flat[index] = old-h;minus = forward(p0, x, z)[0][objective];p0[key].flat[index] = old
                np.testing.assert_allclose((plus-minus)/(2*h), derivative.flat[index], atol=2e-9, rtol=1e-5)
                scalar_differences += 1
    # First Adam step only: this oracle does not fit the model. Float64 equations
    # followed by Float32 parameter rounding approximate native intermediate rounding.
    after_d = {k: v.copy() for k, v in p0.items()}
    for key, g in gradient(p0, x, z, 2).items():
        after_d[key] = (after_d[key]-.0002*g/(abs(g)+1e-8)).astype(np.float32).astype(np.float64)
    expected_g = gradient(after_d, x, z, 0);stored_g = arrays(directory/'g_gradient_before_e.npz');count = 0
    for key, value in expected_g.items():
        np.testing.assert_allclose(stored_g[key], value, atol=3e-7, rtol=2e-4, err_msg='after first D Adam/'+key);count += value.size
    changed_e = {k: (v+.7 if k.startswith('1/') else v.copy()) for k, v in after_d.items()}
    for key, value in gradient(changed_e, x, z, 0).items():
        exact(value, expected_g[key])
    controls = arrays(directory/'pair_controls.npz');output = arrays(directory/'outputs.npz')
    exact(controls['changed_z'], gaussian((128, 2), 6073).astype(np.float32))
    test = data['test'].astype(np.float64);encoded = output['encoded'].astype(np.float64)
    exact(controls['pair'], output['real_score'])
    for key, expected in [('zero_code', joint(pf, test, np.zeros_like(test))[0]),
                          ('zero_image', joint(pf, np.zeros_like(test), encoded)[0]),
                          ('changed_fake', mapping(pf, '0', controls['changed_z'].astype(np.float64))[0])]:
        np.testing.assert_allclose(controls[key], expected, atol=2e-6, rtol=3e-6);values += expected.size
        largest = max(largest, float(abs(controls[key]-expected).max()))
    metrics = json.loads((directory/'metrics.json').read_text())
    assert metrics['parameters'] == 261 and metrics['steps_per_group'] == 39 and metrics['optimizer_updates'] == 117 and metrics['selected_epoch_fixed_final'] == 3
    for key, field in [('test_discriminator_loss', 'discriminator_loss'), ('test_eg_loss', 'eg_loss')]:
        assert metrics[key] == float(output[field])
    for key, a, b in [('test_reconstruction_mse', output['reconstruction'], data['test']),
                       ('test_latent_cycle_mse', output['latent_cycle'], noise['test_z']),
                       ('fit_mean_reconstruction_mse', np.tile(data['fit'].astype(np.float64).mean(axis=0), (128, 1)), data['test'])]:
        np.testing.assert_allclose(np.mean((a.astype(np.float64)-b.astype(np.float64))**2), metrics[key], atol=1e-13, rtol=0)
    assert metrics['joint_code_score_max_effect'] == float(abs(controls['pair'].astype(np.float64)-controls['zero_code']).max())
    assert metrics['joint_image_score_max_effect'] == float(abs(controls['pair'].astype(np.float64)-controls['zero_image']).max())
    for key, a, b in [('outside_generated', data['test'], output['fake']), ('outside_encoded', noise['test_z'], output['encoded']), ('outside_reconstruction', data['test'], output['reconstruction'])]:
        assert metrics[key] == int((abs(a)>3).any(axis=1).sum()+(abs(b)>3).any(axis=1).sum()) == 0
    assert metrics['test_reconstruction_mse'] > metrics['fit_mean_reconstruction_mse']
    return dict(passed=True,forward_values_checked=values,maximum_forward_error=largest,
                procedural_point_values_exact=1312,fixed_and_changed_noise_values_exact=768,
                independent_first_D_Adam_then_stored_G_derivatives=count,first_D_Adam_uses_widened_equations_then_Float32_rounding=True,
                independent_scalar_finite_differences=scalar_differences,G_gradient_independent_of_E_with_fixed_D=True,
                batch_rows_checked=1200,steps_per_group=39,optimizer_updates=117,fixed_final_epoch=3,
                reconstruction_diagnostic_worse_than_mean=True,metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory', type=Path, required=True);parser.add_argument('--report', type=Path)
    args = parser.parse_args();r = check(args.directory)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True);args.report.write_text(json.dumps(r, indent=2)+'\n')
    print(json.dumps(r, indent=2))


if __name__ == '__main__':
    main()
