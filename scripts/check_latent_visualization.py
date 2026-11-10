#!/usr/bin/env python3
"""Independent NumPy checks of frozen autoencoder/PCA/retrieval artifacts.

Reads native arrays only: no preparation, neural-network training or downloads. Checks
fitting-only PCA/SVD, all projected pools, reconstruction controls, stable
row-order Euclidean neighbor lists, posthoc labels and display clipping counts.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact
from check_contractive_autoencoder import sigmoid


def nearest(fit, query, k=10):
    """Exact source feature-order double distance accumulation, with row-index ties."""
    fit = fit.reshape(len(fit), -1).astype(np.float64)
    query = query.reshape(len(query), -1).astype(np.float64)
    out = []
    for start in range(0, len(query), 10):
        chunk = query[start:start+10]
        distance = np.zeros((len(chunk), len(fit)), dtype=np.float64)
        for j in range(fit.shape[1]):
            delta = chunk[:, j, None]-fit[None, :, j]
            distance += delta*delta
        out.append(np.argsort(distance, axis=1, kind='stable')[:, :k].astype(np.int32))
    return np.concatenate(out)


def check(directory, upstream=None):
    """Check complete stored outputs and return independent numerical evidence."""
    a, p = arrays(directory/'analysis.npz'), arrays(directory/'model_arrays.npz')
    if upstream:
        original = arrays(upstream)
        for key in p:
            exact(p[key], original[key])
    assert sum(v.size for v in p.values()) == 50992
    metrics = json.loads((directory/'metrics.json').read_text())
    identities = set();largest = 0.;forward_values = 0
    for split, count in [('fit', 1000), ('validation', 200), ('test', 200)]:
        ids, labels = a[split+'_source_ids'], a[split+'_labels']
        assert ids.shape == (count,) and len(set(ids)) == count and not identities.intersection(ids)
        identities.update(ids)
        assert np.array_equal(np.bincount(labels, minlength=10), np.full(10, count//10))
        lo, hi = (60000, 70000) if split == 'test' else (0, 60000)
        assert ((ids >= lo) & (ids < hi)).all()
        x = a[split+'_images'].reshape(count, 784).astype(np.float64)
        code = np.maximum(x@p['encoder/W'].astype(np.float64)+p['encoder/b'], 0)
        decoded = sigmoid(code@p['decoder/W'].astype(np.float64)+p['decoder/b'])
        for expected, key in [(code, split+'_code'), (decoded, split+'_ae_reconstruction')]:
            np.testing.assert_allclose(expected, a[key], atol=5e-6, rtol=5e-6)
            largest = max(largest, float(abs(expected-a[key]).max()));forward_values += expected.size
    pca_values, svd_error = 0, 0.
    for prefix, input_suffix, retained in [('code_pca', 'code', 2), ('pixel_pca', 'images', 32)]:
        fit = a['fit_'+input_suffix].reshape(1000, -1).astype(np.float64)
        mean, components, variances = a[prefix+'_mean'], a[prefix+'_components'], a[prefix+'_variances']
        np.testing.assert_allclose(mean, fit.mean(axis=0, keepdims=True), atol=1e-13, rtol=0)
        np.testing.assert_allclose(components.T@components, np.eye(retained), atol=1e-12, rtol=0)
        centered = fit-mean
        _, singular, vh = np.linalg.svd(centered, full_matrices=False)
        expected = vh[:retained].T
        signs = np.where((expected*components).sum(axis=0) < 0, -1., 1.)
        np.testing.assert_allclose(expected*signs, components, atol=2e-10, rtol=2e-8)
        svd_error = max(svd_error, float(abs(expected*signs-components).max()))
        np.testing.assert_allclose(variances, singular[:retained]**2/999, atol=1e-12, rtol=1e-10)
        for split, count in [('fit', 1000), ('validation', 200), ('test', 200)]:
            raw = a[split+'_'+input_suffix].reshape(count, -1).astype(np.float64)
            score = (raw-mean)@components
            reconstruction = mean+score@components.T
            np.testing.assert_allclose(score, a[prefix+'_'+split], atol=2e-12, rtol=1e-10)
            np.testing.assert_allclose(reconstruction, a[prefix+'_'+split+'_reconstruction'], atol=2e-12, rtol=1e-10)
            pca_values += score.size+reconstruction.size
    pixel2 = a['pixel_pca_mean']+a['pixel_pca_test'][:, :2]@a['pixel_pca_components'][:, :2].T
    np.testing.assert_allclose(pixel2, a['pixel_pca2_test_reconstruction'], atol=2e-12, rtol=1e-10)
    projected = a['code_pca_test_reconstruction'].astype(np.float32).astype(np.float64)
    decoded = sigmoid(projected@p['decoder/W'].astype(np.float64)+p['decoder/b'])
    np.testing.assert_allclose(decoded, a['ae_projected2_decoder_reconstruction'], atol=1e-6, rtol=3e-6)
    largest = max(largest, float(abs(decoded-a['ae_projected2_decoder_reconstruction']).max()));forward_values += decoded.size
    inputs = {'pixel': (a['fit_images'], a['test_images']), 'ae32': (a['fit_code'], a['test_code']),
              'ae2': (a['code_pca_fit'], a['code_pca_test']), 'pca32': (a['pixel_pca_fit'], a['pixel_pca_test']),
              'pca2': (a['pixel_pca_fit'][:, :2], a['pixel_pca_test'][:, :2])}
    neighbors = {}
    for name, (fit, query) in inputs.items():
        neighbors[name] = nearest(fit, query);exact(neighbors[name], a[name+'_neighbors'])
    for name, rows in neighbors.items():
        accuracy = float(np.mean(a['fit_labels'][rows[:, 0]] == a['test_labels']))
        overlap = sum(len(set(r).intersection(s)) for r, s in zip(neighbors['pixel'], rows))/rows.size
        assert accuracy == metrics[name+'_test_1nn_label_accuracy']
        assert overlap == metrics[name+'_top10_pixel_neighbor_overlap']
    target = a['test_images'].reshape(200, 784).astype(np.float64)
    for name, raw, clip in [('ae_test_mse', a['test_ae_reconstruction'], False),
                           ('ae_projected2_decoder_test_mse', a['ae_projected2_decoder_reconstruction'], False),
                           ('pixel_pca32_test_mse', a['pixel_pca_test_reconstruction'], False),
                           ('pixel_pca32_clipped_test_mse', a['pixel_pca_test_reconstruction'], True),
                           ('pixel_pca2_test_mse', pixel2, False)]:
        value = raw.astype(np.float64)
        if clip:
            value = np.clip(value, 0, 1)
        np.testing.assert_allclose(np.mean((value-target)**2), metrics[name], atol=1e-12, rtol=0)
    for name, prefix in [('ae', 'code_pca'), ('pixel', 'pixel_pca')]:
        fit, test = a[prefix+'_fit'][:, :2], a[prefix+'_test'][:, :2]
        low, high = fit.min(axis=0), fit.max(axis=0)
        pad = np.maximum(1e-6, .05*(high-low));low -= pad;high += pad
        clipped = int(((test < low) | (test > high)).any(axis=1).sum())
        assert clipped == metrics[name+'_display_clipped_test_points']
    assert metrics['additional_training_updates'] == 0 and metrics['encoder_parameters'] == 50992
    assert metrics['code_dimensions'] == 32 and metrics['display_dimensions'] == 2
    return dict(passed=True, additional_training_updates=0, forward_values_checked=forward_values,
                maximum_forward_error=largest, PCA_projection_reconstruction_values_checked=pca_values,
                maximum_sign_aligned_SVD_basis_error=svd_error, neighbor_indices_exact=10000,
                PCA_fitting_rows_only=True, stable_row_order_neighbor_ties=True,
                upstream_parameters_exact=bool(upstream), metrics=metrics)


def main():
    """Read native outputs and optionally save a numerical verification receipt."""
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
