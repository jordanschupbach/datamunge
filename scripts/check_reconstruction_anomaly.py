#!/usr/bin/env python3
"""Independent NumPy oracle for nominal-validation reconstruction stress tests.

Reads saved native artifacts only. Verifies all paired corruption pixels,
frozen model forwards, exact feature-order scores, nearest-rank calibration,
strict thresholding and pairwise/group-tied ROC/AP over every original case.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact
from check_contractive_autoencoder import sigmoid


def scores(images, reconstruction):
    """Source feature-order Double squared-residual sum, divided by784 per case."""
    x = images.reshape(len(images), 784).astype(np.float64)
    y = np.broadcast_to(reconstruction.reshape(-1, 784).astype(np.float64), x.shape)
    total = np.zeros(len(x), dtype=np.float64)
    for j in range(784):
        delta = x[:, j]-y[:, j]
        total += delta*delta
    return total/784


def ink(images, fit_mean):
    """Double row-order ink mean, then absolute deviation from the fitted scalar."""
    x = images.reshape(len(images), 784).astype(np.float64)
    total = np.zeros(len(x), dtype=np.float64)
    for j in range(784):
        total += x[:, j]
    return np.abs(total/784-fit_mean)


def ranking(normal, anomaly, threshold):
    """Independent pairwise half-ties AUC and descending score-group AP."""
    wins = float((anomaly[:, None] > normal[None, :]).sum())+.5*float((anomaly[:, None] == normal[None, :]).sum())
    values = np.concatenate([normal, anomaly]);labels = np.concatenate([np.zeros(len(normal), dtype=int), np.ones(len(anomaly), dtype=int)])
    order = np.argsort(-values, kind='stable');values, labels = values[order], labels[order]
    ap = 0.;seen = 0;positives = 0;i = 0
    while i < len(values):
        end = i
        while end < len(values) and values[end] == values[i]:
            end += 1
        added = int(labels[i:end].sum());positives += added;seen += end-i
        ap += (added/len(anomaly))*positives/seen
        i = end
    return dict(auroc=wins/(len(normal)*len(anomaly)), average_precision=ap,
                test_nominal_fpr=float((normal > threshold).mean()), corruption_tpr=float((anomaly > threshold).mean()))


def check(directory, upstream=None):
    """Raise on any changed source case/score/ranking; return independent measured evidence."""
    a, p = arrays(directory/'analysis.npz'), arrays(directory/'model_arrays.npz')
    if upstream:
        initial = arrays(upstream)
        for key in p:
            exact(p[key], initial[key])
    assert sum(v.size for v in p.values()) == 50992
    metrics = json.loads((directory/'metrics.json').read_text())
    identities = set()
    for split, count in [('fit', 1000), ('validation', 200), ('test', 200)]:
        ids, labels = a[split+'_source_ids'], a[split+'_labels']
        assert ids.shape == (count,) and len(set(ids)) == count and not identities.intersection(ids)
        identities.update(ids)
        lo, hi = (60000, 70000) if split == 'test' else (0, 60000)
        assert ((ids >= lo) & (ids < hi)).all()
        assert np.array_equal(np.bincount(labels, minlength=10), np.full(10, count//10))
    mean = a['fit_images'].reshape(1000, 784).astype(np.float64).mean(axis=0).astype(np.float32)
    exact(mean, a['fit_pixel_mean'])
    fit_ink = sum(float(v) for v in mean)/784
    assert fit_ink == metrics['fit_mean_ink']
    original = a['test_images'];patch = original.copy();patch[:, :8, :8, :] = np.float32(1)
    dark = original*np.float32(.35);shift = np.zeros_like(original);shift[:, :, 6:, :] = original[:, :, :-6, :]
    for name, image in [('patch', patch), ('darkening', dark), ('shift', shift)]:
        exact(image, a[name+'_images'])
    largest = 0.;forward_values = 0;score_values = 0
    for name in ['fit', 'validation', 'test', 'patch', 'darkening', 'shift', 'blank']:
        image = a['blank_image'] if name == 'blank' else a[name+'_images']
        x = image.reshape(len(image), 784).astype(np.float64)
        code = np.maximum(x@p['encoder/W'].astype(np.float64)+p['encoder/b'], 0)
        expected = sigmoid(code@p['decoder/W'].astype(np.float64)+p['decoder/b'])
        reconstruction = a['blank_reconstruction'] if name == 'blank' else a[name+'_reconstruction']
        np.testing.assert_allclose(expected, reconstruction, atol=1e-6, rtol=5e-6)
        largest = max(largest, float(abs(expected-reconstruction).max()));forward_values += expected.size
        actual = a['blank_ae_score'] if name == 'blank' else a[name+'_ae_score']
        exact(scores(image, reconstruction), actual);score_values += len(image)
        if name != 'blank':
            exact(scores(image, mean), a[name+'_mean_score']);exact(ink(image, fit_ink), a[name+'_ink_score']);score_values += 2*len(image)
    exact(a['blank_image'], np.zeros((1, 28, 28, 1), dtype=np.float32))
    assert a['blank_ae_score'][0] == metrics['blank_ae_score']
    controls = ranking(np.array([1., 2.]), np.array([2., 3.]), 2.)
    assert controls['auroc'] == .875
    assert math.isclose(controls['average_precision'], 5/6, abs_tol=1e-15, rel_tol=0)
    assert controls['test_nominal_fpr'] == 0 and controls['corruption_tpr'] == .5
    with (directory/'rankings.tsv').open() as stream:rows = list(csv.DictReader(stream, delimiter='\t'))
    assert len(rows) == 9
    seen = set()
    for row in rows:
        method, corruption = row['score'], row['corruption'];assert (method, corruption) not in seen;seen.add((method, corruption))
        assert method in ['ae', 'mean', 'ink'] and corruption in ['patch', 'darkening', 'shift']
        calibration = a['validation_'+method+'_score']
        threshold = float(np.sort(calibration)[math.ceil(.95*len(calibration))-1])
        assert threshold == metrics['thresholds'][method] == float(row['validation_nominal_threshold'])
        assert float((calibration > threshold).mean()) == float(row['validation_nominal_fpr'])
        out = ranking(a['test_'+method+'_score'], a[corruption+'_'+method+'_score'], threshold)
        for key, value in out.items():
            assert value == float(row[key]), (method, corruption, key, value, row[key])
    assert metrics['additional_training_updates'] == 0 and metrics['quantile'] == .95
    assert metrics['fit_records'] == 1000 and metrics['validation_nominal_records'] == metrics['test_nominal_records'] == metrics['each_corruption_records'] == 200
    return dict(passed=True, additional_training_updates=0, forward_values_checked=forward_values,
                maximum_forward_error=largest, exact_score_values=score_values, exact_corruption_pixels=3*200*784,
                nominal_validation_cases=200, fixed_test_sources=200, paired_corruption_cases=600,
                strict_thresholds=True, pairwise_ROC_and_group_tied_AP_exact=True, ranking_rows_checked=9,
                blank_control_excluded_from_rankings=True, upstream_parameters_exact=bool(upstream), metrics=metrics)


def main():
    """Read complete native outputs and optionally save an independent numerical receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args();result = check(args.directory, args.upstream)
    text = json.dumps(result, indent=2)+'\n'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True);args.report.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
