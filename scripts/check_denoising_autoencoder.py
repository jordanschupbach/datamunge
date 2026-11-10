#!/usr/bin/env python3
"""Optional independent NumPy check of native denoising-autoencoder artifacts.

Reads C++ archives only. It never prepares/downloads data or fits a model.
Checks raw MT19937/Box-Muller banks, every corruption, both complete first-step
parameter gradients/Adam updates and all five held-out conditions for both fits.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def arrays(path):
    """Load owned numeric arrays without pickle and close the archive immediately."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def gaussian(shape, seed):
    """Independent raw MT19937 reference stream and explicit cosine/sine pairs."""
    count = int(np.prod(shape))
    raw = np.random.RandomState(seed).randint(0, 2**32, 2*((count+1)//2), dtype=np.uint32).astype(np.float64)
    u, v = (raw[::2]+.5)/4294967296., (raw[1::2]+.5)/4294967296.
    radius, angle = np.sqrt(-2*np.log(u)), 6.283185307179586476925286766559*v
    values = np.column_stack((radius*np.cos(angle), radius*np.sin(angle))).reshape(-1)
    return values[:count].reshape(shape)


def corruption(clean, bank, sigma):
    """Compute in Float64, round the deviation to Float32, then round/clamp pixels."""
    return np.clip((clean.astype(np.float64)+float(np.float32(sigma))*bank.reshape(clean.shape)).astype(np.float32),0,1)


def forward(p, x):
    """Independent Float64 ReLU/sigmoid forward evaluation with untied weights."""
    code = np.maximum(x@p['encoder/W']+p['encoder/b'],0)
    reconstruction = 1/(1+np.exp(-(code@p['decoder/W']+p['decoder/b'])))
    return code, reconstruction


def check(directory):
    """Validate complete numeric artifacts and return error/count evidence."""
    data = arrays(directory/'denoising_data.npz')
    outputs = arrays(directory/'denoising_outputs.npz')
    standard_count = 0
    corruption_count = 0
    maximum_bank_error = 0.
    for key, seed in [('validation_standard_noise',1709),('test_standard_noise',1711),('second_standard_noise',1717)]:
        reference = gaussian(data[key].shape,seed)
        np.testing.assert_allclose(reference,data[key],atol=3e-15,rtol=3e-15)
        standard_count += reference.size
        maximum_bank_error = max(maximum_bank_error,float(np.max(np.abs(reference-data[key]))))
    for epoch in (1,2,3):
        saved = arrays(directory/f'denoising_train_epoch{epoch}.npz')
        reference = gaussian(saved['standard_noise'].shape,1671+epoch)
        np.testing.assert_allclose(reference,saved['standard_noise'],atol=3e-15,rtol=3e-15)
        np.testing.assert_array_equal(corruption(data['fit_images'],saved['standard_noise'],.35),saved['corrupted'])
        standard_count += reference.size
        corruption_count += saved['corrupted'].size
        maximum_bank_error = max(maximum_bank_error,float(np.max(np.abs(reference-saved['standard_noise']))))
    for key, clean_key, bank_key, sigma in [
        ('validation_corrupted','validation_images','validation_standard_noise',.35),
        ('test_corrupted','test_images','test_standard_noise',.35),
        ('test_low','test_images','test_standard_noise',.15),
        ('test_strong','test_images','test_standard_noise',.6),
        ('test_second','test_images','second_standard_noise',.35)]:
        np.testing.assert_array_equal(corruption(data[clean_key],data[bank_key],sigma),data[key])
        corruption_count += data[key].size
    records = []
    total_predictions = 0
    for variant in ('denoising','ordinary'):
        prefix = directory/('denoising_'+variant)
        initial = arrays(Path(str(prefix)+'_initial.npz'))
        saved = arrays(Path(str(prefix)+'_first_batch.npz'))
        gradients = arrays(Path(str(prefix)+'_first_gradients.npz'))
        updated = arrays(Path(str(prefix)+'_first_update.npz'))
        best = arrays(Path(str(prefix)+'_best.npz'))
        p = {key:value.astype(np.float64) for key,value in initial.items()}
        x = saved['input'].reshape(-1,784).astype(np.float64)
        target = saved['clean_target'].reshape(-1,784).astype(np.float64)
        h, prediction = forward(p,x)
        delta = 2*(prediction-target)*prediction*(1-prediction)/target.size
        back = (delta@p['decoder/W'].T)*(h>0)
        expected = {'encoder/W':x.T@back,'encoder/b':back.sum(axis=0),
                    'decoder/W':h.T@delta,'decoder/b':delta.sum(axis=0)}
        errors = {}
        for key,value in expected.items():
            np.testing.assert_allclose(value,gradients[key],atol=2e-8,rtol=3e-4)
            errors[key] = float(np.max(np.abs(value-gradients[key])))
            # Isolate Float32 optimizer rounding from the widened gradient check.
            g = gradients[key]
            first, second = g*np.float32(.1),(g*g)*np.float32(.001)
            direction = (first/np.float32(.1))/(np.sqrt(second/np.float32(.001))+np.float32(1e-8))
            np.testing.assert_array_equal(initial[key]-np.float32(.003)*direction,updated[key])
        p = {key:value.astype(np.float64) for key,value in best.items()}
        scores = {}
        target = data['test_images'].reshape(-1,784).astype(np.float64)
        maximum_prediction_error = 0.
        for name,key in [('clean','test_images'),('low','test_low'),('noisy','test_corrupted'),('second','test_second'),('strong','test_strong')]:
            h,prediction = forward(p,data[key].reshape(-1,784).astype(np.float64))
            stored = outputs[name+'_'+variant+'_reconstruction']
            np.testing.assert_allclose(prediction,stored,atol=1e-6,rtol=3e-6)
            if variant=='denoising': np.testing.assert_allclose(h,outputs[name+'_denoising_code'],atol=3e-6,rtol=3e-6)
            total_predictions += prediction.size
            scores[name] = float(np.mean((prediction-target)**2))
            maximum_prediction_error = max(maximum_prediction_error,float(np.max(np.abs(prediction-stored))))
        records.append(dict(variant=variant,complete_parameter_derivatives=sum(v.size for v in expected.values()),
                            first_gradient_errors=errors,Float32_Adam_step_bit_identical=True,
                            maximum_prediction_error=maximum_prediction_error,test_clean_target_MSE=scores))
    left = arrays(directory/'denoising_denoising_initial.npz')
    right = arrays(directory/'denoising_ordinary_initial.npz')
    for key in left: np.testing.assert_array_equal(left[key],right[key])
    fitting_mean = data['fit_images'].reshape(-1,784).astype(np.float64).mean(axis=0)
    np.testing.assert_allclose(outputs['mean'],np.broadcast_to(fitting_mean,outputs['mean'].shape),atol=3e-8,rtol=1e-7)
    for key,count in [('fit_source_ids',1000),('validation_source_ids',200),('test_source_ids',200)]:
        ids = data[key]
        lower,upper = (60000,70000) if key=='test_source_ids' else (0,60000)
        assert ids.shape==(count,) and len(np.unique(ids))==count and ((ids>=lower)&(ids<upper)).all()
    for a,b in [('fit_source_ids','validation_source_ids'),('fit_source_ids','test_source_ids'),('validation_source_ids','test_source_ids')]:
        assert not np.intersect1d(data[a],data[b]).size
    assert np.array_equal(np.bincount(data['test_labels'],minlength=10),np.full(10,20))
    return dict(passed=True,standard_noise_values=standard_count,maximum_bank_error=maximum_bank_error,
                bit_identical_corrupted_pixels=corruption_count,held_out_prediction_pixels=total_predictions,
                matched_initial_parameters=True,disjoint_official_source_ids=True,variants=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--report',type=Path)
    args = parser.parse_args()
    result = json.dumps(check(args.directory),indent=2)+'\n'
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(result)
    print(result,end='')


if __name__=='__main__':
    main()
