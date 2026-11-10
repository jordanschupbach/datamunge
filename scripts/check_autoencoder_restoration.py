#!/usr/bin/env python3
"""Optional independent NumPy oracle for frozen autoencoder restoration artifacts.

Reads native saved arrays only, without preparation, training or extra updates.
Recomputes corruption banks, both dense predictors, visible-only transformer
inference, patch/image mapping, composites and every reported error/PSNR.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from check_denoising_autoencoder import arrays, corruption, forward as dense_forward, gaussian
from check_masked_autoencoder import forward as masked_forward, parameters, selection


def image_from_patches(patches):
    """Restore row-major 7x7 tiles with explicit image coordinates."""
    result = np.empty((len(patches), 28, 28, 1), dtype=patches.dtype)
    for patch in range(16):
        row, col = 7*(patch//4), 7*(patch%4)
        result[:, row:row+7, col:col+7, 0] = patches[:, patch].reshape(-1, 7, 7)
    return result.reshape(-1, 784)


def mse(prediction, target):
    """One global pixel mean after promoting both operands to Float64."""
    return float(np.mean((prediction.reshape(len(prediction),-1).astype(np.float64)-target.reshape(len(target),-1).astype(np.float64))**2))


def check(directory):
    """Verify all saved restoration endpoints and return measured counts/errors."""
    a = arrays(directory/'analysis.npz')
    models = {name: arrays(directory/(name+'_arrays.npz')) for name in ('denoising', 'ordinary', 'masked')}
    assert sum(x.size for x in models['denoising'].values()) == 50992
    assert sum(x.size for x in models['ordinary'].values()) == 50992
    assert sum(x.size for x in models['masked'].values()) == 13649
    seen = set()
    for split, count in [('fit',1000),('validation',200),('test',200)]:
        assert a[split+'_images'].shape == (count,28,28,1)
        assert a[split+'_labels'].shape == (count,) and a[split+'_source_ids'].shape == (count,)
        np.testing.assert_array_equal(np.bincount(a[split+'_labels'],minlength=10),np.full(10,count//10))
        ids = a[split+'_source_ids'].astype(int).tolist()
        assert len(set(ids)) == count and not seen.intersection(ids)
        seen.update(ids)
        assert np.isfinite(a[split+'_images']).all() and np.all((a[split+'_images']>=0)&(a[split+'_images']<=1))
    raw_clean = a['test_images']
    clean = raw_clean.reshape(-1,784)
    bank = gaussian(a['standard_noise'].shape,4229)
    np.testing.assert_allclose(bank,a['standard_noise'],atol=3e-15,rtol=3e-15)
    mean = a['fit_images'].reshape(1000,784).astype(np.float64).mean(axis=0).astype(np.float32)
    np.testing.assert_array_equal(mean,a['fit_pixel_mean'])
    np.testing.assert_array_equal(a['mean_images'],np.broadcast_to(mean,clean.shape))
    with (directory/'denoising.tsv').open() as stream:
        rows = list(csv.DictReader(stream,delimiter='\t'))
    conditions = [('clean',0.),('low',.15),('trained_sigma',.35),('high',.6)]
    assert len(rows) == len(conditions)
    predictions = 0
    maximum_error = 0.

    def compare(actual, expected):
        nonlocal predictions, maximum_error
        np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=2e-5)
        maximum_error = max(maximum_error,float(np.max(np.abs(actual-expected))))
        predictions += expected.size

    for (name,sigma), row in zip(conditions,rows):
        assert row['condition'] == name and float(row['sigma']) == float(np.float32(sigma))
        x = corruption(raw_clean,a['standard_noise'],sigma)
        np.testing.assert_array_equal(x,a[name+'_input'])
        for model in ('denoising','ordinary'):
            p = {key:value.astype(np.float64) for key,value in models[model].items()}
            probability = dense_forward(p,x.reshape(-1,784).astype(np.float64))[1]
            compare(probability,a[name+'_'+model])
            np.testing.assert_allclose(mse(a[name+'_'+model],clean),float(row[model+'_mse']),atol=2e-14,rtol=0)
        np.testing.assert_allclose(mse(x,clean),float(row['noisy_input_mse']),atol=2e-14,rtol=0)
        np.testing.assert_allclose(mse(a['mean_images'],clean),float(row['fit_mean_mse']),atol=2e-14,rtol=0)
        expected_psnr=10*np.log10(1/mse(a[name+'_denoising'],clean))
        np.testing.assert_allclose(expected_psnr,float(row['denoising_global_psnr_db']),atol=2e-12,rtol=0)
    selector = selection(a['test_source_ids'],0,4243)
    np.testing.assert_array_equal(selector,a['selection'])
    hidden_patches = 1-selector.sum(axis=1)
    assert np.all(hidden_patches.sum(axis=1)==12)
    hidden = image_from_patches(np.broadcast_to(hidden_patches[:,:,None],(200,16,49))).astype(bool)
    np.testing.assert_array_equal(a['observed'].reshape(-1,784),np.where(hidden,0,clean))
    np.testing.assert_array_equal(a['hidden_changed'].reshape(-1,784),np.where(hidden,1,clean))
    outputs=[]
    for x in (a['observed'],a['hidden_changed'],clean):
        code, prediction, _, _ = masked_forward(parameters(models['masked']),x.reshape(-1,28,28,1),selector)
        outputs.append((code.v,prediction.v))
    for code,prediction in outputs[1:]:
        np.testing.assert_array_equal(code,outputs[0][0])
        np.testing.assert_array_equal(prediction,outputs[0][1])
    compare(outputs[0][0],a['mae_code'])
    compare(outputs[0][1],a['mae_patch_prediction'])
    np.testing.assert_array_equal(image_from_patches(a['mae_patch_prediction']),a['mae_hidden_prediction'])
    p = {key:value.astype(np.float64) for key,value in models['denoising'].items()}
    compare(dense_forward(p,a['observed'].reshape(-1,784).astype(np.float64))[1],a['denoising_masked_prediction'])
    for prediction,composite in [('mae_hidden_prediction','mae_composite'),('mean_images','mean_composite'),('denoising_masked_prediction','denoising_composite')]:
        np.testing.assert_array_equal(np.where(hidden,a[prediction],a['observed'].reshape(-1,784)),a[composite])
        np.testing.assert_array_equal(a[composite][~hidden],clean[~hidden])
    metrics = json.loads((directory/'metrics.json').read_text())
    assert metrics['additional_training_updates'] == 0
    assert metrics['denoising_parameters'] == metrics['ordinary_parameters'] == 50992 and metrics['masked_parameters'] == 13649
    assert metrics['test_records'] == 200 and metrics['visible_patches'] == 4 and metrics['hidden_patches'] == 12
    assert metrics['hidden_pixels_per_case'] == 588
    assert metrics['hidden_values_prediction_max_effect'] == metrics['observed_composite_max_error'] == 0
    for key,prediction in [('mae_hidden_mse','mae_composite'),('fit_mean_hidden_mse','mean_composite'),('denoising_hidden_mse','denoising_composite'),('zero_hidden_mse','observed')]:
        error=(a[prediction].reshape(-1,784).astype(np.float64)-clean.astype(np.float64))**2
        np.testing.assert_allclose(error[hidden].mean(),metrics[key],atol=3e-14,rtol=0)
    np.testing.assert_allclose(mse(a['mae_composite'],clean),metrics['mae_composite_full_mse'],atol=3e-14,rtol=0)
    np.testing.assert_allclose(metrics['mae_composite_full_mse'],.75*metrics['mae_hidden_mse'],atol=3e-14,rtol=0)
    return dict(passed=True,additional_training_updates=0,frozen_parameter_values=115633,
                Gaussian_values_checked=bank.size,corrupted_pixels_checked=4*clean.size,
                forward_values_checked=predictions,maximum_forward_error=maximum_error,
                masks_checked=200,hidden_pixel_predictions_independent=True,
                composite_visible_pixels_bit_exact=True,hidden_pixels_per_case=588,
                denoising_metric_conditions_checked=4,reported_hidden_MSEs_checked=4,
                global_PSNRs_checked=4,full_composite_MSE_equals_three_quarters_hidden_MSE=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--json',type=Path)
    args=parser.parse_args()
    result=check(args.directory)
    if args.json: args.json.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
