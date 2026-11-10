#!/usr/bin/env python3
"""Optional independent NumPy oracle for the native k-sparse report's arrays.

Never prepares/downloads data or fits. Checks stable signed selection, complete
first-step gradients/Adam updates, all pool predictions, code controls and IDs.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def arrays(path):
    """Own numeric archive arrays and close the input; pickle is disabled."""
    with np.load(path,allow_pickle=False) as archive:
        return {key:archive[key] for key in archive.files}


def selector(raw, plain=False):
    """Stable descending signed order in each row; lower indices win ties."""
    if plain:
        return np.ones_like(raw)
    result = np.zeros_like(raw)
    winners = np.argsort(-raw,axis=1,kind='stable')[:,:min(8,raw.shape[1])]
    np.put_along_axis(result,winners,1,axis=1)
    return result


def decode(p, code):
    """Untied Float64 sigmoid decoder, separate from all selection logic."""
    return 1/(1+np.exp(-(code@p['decoder/W']+p['decoder/b'])))


def forward(p, image, plain=False):
    """Independent linear activities, signed selector, gated code and decoder."""
    raw = image@p['encoder/W']+p['encoder/b']
    mask = selector(raw,plain)
    code = raw*mask
    return raw,mask,code,decode(p,code)


def check(directory):
    """Validate complete artifacts or raise; return measured error/count evidence."""
    data = arrays(directory/'k_sparse_data.npz')
    records=[]
    predicted_pixels=0
    for variant in ('competitive','plain'):
        plain = variant=='plain'
        prefix = directory/('k_sparse_'+variant)
        initial = arrays(Path(str(prefix)+'_initial.npz'))
        batch = arrays(Path(str(prefix)+'_first_batch.npz'))['images'].reshape(-1,784).astype(np.float64)
        first_probe = arrays(Path(str(prefix)+'_first_probe.npz'))
        saved_gradients = arrays(Path(str(prefix)+'_first_gradients.npz'))
        updated = arrays(Path(str(prefix)+'_first_update.npz'))
        best = arrays(Path(str(prefix)+'_best.npz'))
        outputs = arrays(Path(str(prefix)+'_outputs.npz'))
        p={key:value.astype(np.float64) for key,value in initial.items()}
        raw,mask,code,prediction=forward(p,batch,plain)
        np.testing.assert_allclose(raw,first_probe['raw'],atol=2e-6,rtol=3e-6)
        np.testing.assert_array_equal(mask,first_probe['mask'])
        np.testing.assert_allclose(prediction,first_probe['reconstruction'],atol=1e-6,rtol=3e-6)
        delta=2*(prediction-batch)*prediction*(1-prediction)/batch.size
        back=(delta@p['decoder/W'].T)*mask
        expected={'encoder/W':batch.T@back,'encoder/b':back.sum(axis=0),
                  'decoder/W':code.T@delta,'decoder/b':delta.sum(axis=0)}
        errors={}
        for key,value in expected.items():
            np.testing.assert_allclose(value,saved_gradients[key],atol=2e-8,rtol=3e-4)
            errors[key]=float(np.max(np.abs(value-saved_gradients[key])))
            # Check optimizer rounding separately using independently verified gradients.
            g=saved_gradients[key]
            first,second=g*np.float32(.1),(g*g)*np.float32(.001)
            direction=(first/np.float32(.1))/(np.sqrt(second/np.float32(.001))+np.float32(1e-8))
            np.testing.assert_array_equal((initial[key]-direction*np.float32(.003)).view(np.uint32),updated[key].view(np.uint32))
        for name in ('first_probe','partial_probe','selected_probe'):
            probe=arrays(Path(str(prefix)+'_'+name+'.npz'))
            np.testing.assert_array_equal(selector(probe['raw'],plain),probe['mask'])
            np.testing.assert_array_equal((probe['raw']*probe['mask']).view(np.uint32),probe['training_code'].view(np.uint32))
            assert np.array_equal(probe['mask'].sum(axis=1),np.full(len(probe['raw']),64 if plain else 8))
            if name=='partial_probe': assert probe['raw'].shape==(8,64)
        p={key:value.astype(np.float64) for key,value in best.items()}
        maximum_prediction_error=0.
        minimum_cutoff_margin=float('inf')
        for split in ('fit','validation','test'):
            image=data[split+'_images'].reshape(-1,784).astype(np.float64)
            raw,mask,code,prediction=forward(p,image,plain)
            np.testing.assert_allclose(raw,outputs[split+'_raw'],atol=3e-6,rtol=3e-6)
            np.testing.assert_array_equal(selector(outputs[split+'_raw'],plain),outputs[split+'_mask'])
            np.testing.assert_array_equal(mask,outputs[split+'_mask'])
            np.testing.assert_allclose(code,outputs[split+'_code'],atol=3e-6,rtol=3e-6)
            np.testing.assert_allclose(prediction,outputs[split+'_reconstruction'],atol=1e-6,rtol=3e-6)
            maximum_prediction_error=max(maximum_prediction_error,float(np.max(np.abs(prediction-outputs[split+'_reconstruction']))))
            predicted_pixels+=prediction.size
            if not plain:
                ordered=np.sort(outputs[split+'_raw'],axis=1)
                minimum_cutoff_margin=min(minimum_cutoff_margin,float(np.min(ordered[:,-8]-ordered[:,-9])))
        test=data['test_images'].reshape(-1,784).astype(np.float64)
        for key,control in [('zero_reconstruction',np.zeros_like(code)),('shuffled_reconstruction',np.roll(code,-1,axis=0))]:
            prediction=decode(p,control)
            np.testing.assert_allclose(prediction,outputs[key],atol=1e-6,rtol=3e-6)
            predicted_pixels+=prediction.size
        records.append(dict(variant=variant,complete_parameter_derivatives=sum(a.size for a in expected.values()),
            first_gradient_errors=errors,Float32_first_Adam_step_bit_identical=True,
            maximum_prediction_error=maximum_prediction_error,
            minimum_signed_rank_cutoff_margin=None if plain else minimum_cutoff_margin,
            test_mse=float(np.mean((outputs['test_reconstruction'].astype(np.float64)-test)**2))))
    left=arrays(directory/'k_sparse_competitive_initial.npz')
    right=arrays(directory/'k_sparse_plain_initial.npz')
    for key in left: np.testing.assert_array_equal(left[key].view(np.uint32),right[key].view(np.uint32))
    for key,count in [('fit_source_ids',1000),('validation_source_ids',200),('test_source_ids',200)]:
        ids=data[key]
        lower,upper=(60000,70000) if key=='test_source_ids' else (0,60000)
        assert ids.shape==(count,) and len(np.unique(ids))==count and ((ids>=lower)&(ids<upper)).all()
    for a,b in [('fit_source_ids','validation_source_ids'),('fit_source_ids','test_source_ids'),('validation_source_ids','test_source_ids')]:
        assert not np.intersect1d(data[a],data[b]).size
    assert np.array_equal(np.bincount(data['test_labels'],minlength=10),np.full(10,20))
    return dict(passed=True,prediction_pixels=predicted_pixels,matched_initial_parameters=True,
                disjoint_official_source_ids=True,variants=records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    result=json.dumps(check(args.directory),indent=2)+'\n'
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(result)
    print(result,end='')


if __name__=='__main__':
    main()
