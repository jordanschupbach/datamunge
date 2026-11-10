#!/usr/bin/env python3
"""Independent NumPy logistic/hinge/least-squares GAN checks on captured artifacts; no fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact, gaussian
from check_contractive_autoencoder import sigmoid
from check_auxiliary_classifier_gan import coarse, mmd


def losses(real, fake, kind):
    if kind == 'vanilla_gan':
        return np.logaddexp(0,-real).mean()+np.logaddexp(0,fake).mean(),np.logaddexp(0,-fake).mean()
    if kind == 'lsgan':
        return .5*(np.mean((real-1)**2)+np.mean(fake**2)), .5*np.mean((fake-1)**2)
    return np.maximum(1-real, 0).mean()+np.maximum(1+fake, 0).mean(), -fake.mean()


def forward(p, image, noise, kind, diagnostic=True):
    hidden = np.tanh(noise@p['0/hidden/W']+p['0/hidden/b'])
    fake = sigmoid(hidden@p['0/output/W']+p['0/output/b'])
    rh = np.tanh(image@p['1/hidden/W']+p['1/hidden/b'])
    fh = np.tanh(fake@p['1/hidden/W']+p['1/hidden/b'])
    real = rh@p['1/output/W']+p['1/output/b']
    generated = fh@p['1/output/W']+p['1/output/b']
    dl, gl = losses(real, generated, kind)
    out = dict(fake=fake, real_score=real, fake_score=generated, discriminator_loss=dl, generator_loss=gl)
    if kind=='vanilla_gan':
        out['real_logit']=out.pop('real_score');out['fake_logit']=out.pop('fake_score');out['real_probability']=sigmoid(real);out['fake_probability']=sigmoid(generated)
    if diagnostic: out['pixel_mmd'] = mmd(coarse(image), coarse(fake))
    return out, (hidden, rh, fh)


def gradients(p, image, noise, kind):
    out, (hidden, rh, fh) = forward(p, image, noise, kind, False)
    real, fake = out['real_logit' if kind=='vanilla_gan' else 'real_score'], out['fake_logit' if kind=='vanilla_gan' else 'fake_score']
    if kind=='vanilla_gan':
        dr,df,gf=-sigmoid(-real)/len(image),sigmoid(fake)/len(noise),-sigmoid(-fake)/len(noise)
    elif kind == 'lsgan':
        dr, df, gf = (real-1)/len(image), fake/len(noise), (fake-1)/len(noise)
    else:
        dr, df, gf = -(real<1).astype(float)/len(image), (fake>-1).astype(float)/len(noise), -np.ones_like(fake)/len(noise)
    rdelta = (dr@p['1/output/W'].T)*(1-rh*rh)
    fdelta = (df@p['1/output/W'].T)*(1-fh*fh)
    d = {'1/hidden/W': image.T@rdelta+out['fake'].T@fdelta,
         '1/hidden/b': rdelta.sum(axis=0)+fdelta.sum(axis=0),
         '1/output/W': rh.T@dr+fh.T@df, '1/output/b': dr.sum(axis=0)+df.sum(axis=0)}
    pixels = ((gf@p['1/output/W'].T)*(1-fh*fh))@p['1/hidden/W'].T
    pixels *= out['fake']*(1-out['fake'])
    gh = (pixels@p['0/output/W'].T)*(1-hidden*hidden)
    g = {'0/hidden/W': noise.T@gh, '0/hidden/b': gh.sum(axis=0),
         '0/output/W': hidden.T@pixels, '0/output/b': pixels.sum(axis=0)}
    return dict(g, **d)


def check(directory, data, kind, native_gradients):
    raw = arrays(data)
    p0 = {k: v.astype(np.float64) for k, v in arrays(directory/'initial_arrays.npz').items()}
    pf = {k: v.astype(np.float64) for k, v in arrays(directory/'final.npz').items()}
    assert len(p0) == 8 and sum(v.size for v in p0.values()) == 51569
    images = {s: (raw[s+'_images'].astype(np.float32)/np.float32(255)).reshape(-1, 784).astype(np.float64) for s in ['fit', 'validation', 'test']}
    batch = np.genfromtxt(directory/'batches.tsv', names=True, delimiter='\t', dtype=np.int64)
    for epoch in [1, 2, 3]:
        rows = batch[batch['epoch'] == epoch];exact(np.sort(rows['row']), np.arange(1000));exact(rows['source_id'], raw['fit_source_ids'][rows['row']].astype(np.int64))
        assert len(np.unique(rows['step'])) == 32
    assert len(batch) == 3000 and len(np.unique(batch['step'])) == 96
    first = batch[batch['step'] == 0]['row'];x = images['fit'][first]
    z = gaussian((1000, 16), 4862).astype(np.float32).astype(np.float64)[first]
    tz = gaussian((200, 16), 4831).astype(np.float32).astype(np.float64)
    values, largest = 0, 0.
    for name, p, image, noise in [('first_forward', p0, x, z), ('initial_outputs', p0, images['test'], tz), ('outputs', pf, images['test'], tz)]:
        out, _ = forward(p, image, noise, kind);recorded = arrays(directory/(name+'.npz'));assert set(out) == set(recorded)
        for key, value in out.items():
            np.testing.assert_allclose(recorded[key], value, atol=2e-6, rtol=3e-6, err_msg=name+'/'+key)
            values += np.size(value);largest = max(largest, float(abs(recorded[key]-value).max()))
    analytic, stored = gradients(p0, x, z, kind), arrays(native_gradients)
    assert set(stored) == set(analytic)
    gradient_error, differences = 0., 0
    for key, value in analytic.items():
        np.testing.assert_allclose(stored[key], value, atol=2e-6, rtol=3e-4, err_msg=key)
        gradient_error = max(gradient_error, float(abs(stored[key]-value).max()))
        objective = 'generator_loss' if key.startswith('0/') else 'discriminator_loss'
        for index in sorted({0, value.size//2, value.size-1}):
            old = p0[key].flat[index];h = 1e-5
            p0[key].flat[index] = old+h;plus = forward(p0, x, z, kind, False)[0][objective]
            p0[key].flat[index] = old-h;minus = forward(p0, x, z, kind, False)[0][objective];p0[key].flat[index] = old
            np.testing.assert_allclose((plus-minus)/(2*h), value.flat[index], atol=2e-9, rtol=1e-4)
            differences += 1
    out, controls = arrays(directory/'outputs.npz'), arrays(directory/'controls.npz')
    exact(controls['changed_z'], gaussian((200, 16), 4877).astype(np.float32));exact(controls['zero_z'], np.zeros((200, 16), np.float32))
    for key, bank in [('changed_images', controls['changed_z']), ('zero_images', controls['zero_z'])]:
        expected = forward(pf, images['test'], bank.astype(np.float64), kind, False)[0]['fake']
        np.testing.assert_allclose(controls[key], expected, atol=2e-6, rtol=3e-6);values += expected.size
    algebra = arrays(directory/('logistic_algebra.npz' if kind=='vanilla_gan' else 'least_squares_algebra.npz' if kind=='lsgan' else 'hinge_algebra.npz'))
    r, f = algebra['real'].astype(np.float64), algebra['fake'].astype(np.float64)
    dl, gl = losses(r, f, kind);np.testing.assert_allclose(algebra['discriminator_loss'], dl, atol=2e-6, rtol=1e-6);np.testing.assert_allclose(algebra['generator_loss'], gl, atol=2e-6, rtol=1e-6)
    derivatives = [(r-1)/len(r), f/len(f), (f-1)/len(f)] if kind=='lsgan' else [-(r<1).astype(float)/len(r), (f>-1).astype(float)/len(f), -np.ones_like(f)/len(f)]
    if kind=='vanilla_gan':derivatives=[-sigmoid(-r)/len(r),sigmoid(f)/len(f),-sigmoid(-f)/len(f)]
    for key, value in zip(['d_real_gradient','d_fake_gradient','g_fake_gradient'], derivatives):np.testing.assert_allclose(algebra[key], value, atol=0, rtol=2e-6)
    if kind=='hinge_gan': assert algebra['d_real_gradient'][2,0] == 0 and algebra['d_fake_gradient'][2,0] == 0 and algebra['g_fake_gradient'][2,0] < 0
    metrics = json.loads((directory/'metrics.json').read_text());assert metrics['steps_per_group']==96 and metrics['optimizer_updates']==192 and metrics['selected_epoch_fixed_final']==3
    for key, field in [('test_discriminator_loss','discriminator_loss'),('test_generator_loss','generator_loss'),('test_coarse_pixel_mmd','pixel_mmd')]:assert metrics[key] == float(out[field])
    threshold = .5 if kind=='lsgan' else 0
    assert metrics['test_discriminator_real_accuracy']==float((out['real_logit' if kind=='vanilla_gan' else 'real_score']>=threshold).mean())
    assert metrics['test_discriminator_fake_accuracy']==float((out['fake_logit' if kind=='vanilla_gan' else 'fake_score']<threshold).mean())
    np.testing.assert_allclose(metrics['fit_mean_coarse_pixel_mmd'],mmd(coarse(images['test']),coarse(np.tile(images['fit'].mean(axis=0).astype(np.float32),(200,1)))),atol=3e-7,rtol=0)
    return dict(passed=True,kind=kind,forward_values_checked=int(values),maximum_forward_error=largest,
                initial_native_derivatives_checked=sum(v.size for v in analytic.values()),maximum_initial_derivative_error=gradient_error,
                independent_scalar_finite_differences=differences,batch_rows_checked=3000,steps_per_group=96,optimizer_updates=192,fixed_final_epoch=3,
                hinge_zero_margin_subgradient=(kind=='hinge_gan'),source_accuracy_threshold=threshold,coarse_MMD_is_not_image_quality=True,metrics=metrics)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--kind',choices=['vanilla_gan','lsgan','hinge_gan'],required=True)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--data',type=Path,required=True);parser.add_argument('--native-gradients',type=Path,required=True);parser.add_argument('--report',type=Path)
    args=parser.parse_args();result=check(args.directory,args.data,args.kind,args.native_gradients)
    if args.report:args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
