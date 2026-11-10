#!/usr/bin/env python3
"""Independent NumPy check of native frozen-encoder linear-probe artifacts.

Reads saved arrays only. Never prepares data, trains a model or repairs outputs.
Checks scaling, forwards, all head derivatives, first Adam, fixed encoders,
matched head starts/orders, every validation candidate and all held-out cases.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact
from check_factor_tc_variational_autoencoder import first_update


def forward(parameters, images, center, scale):
    """Widen saved parameters and evaluate ReLU, frozen scaling and softmax."""
    p = {k: v.astype(np.float64) for k, v in parameters.items()}
    image = images.reshape(len(images), -1).astype(np.float64)
    code = np.maximum(image @ p['encoder/W'] + p['encoder/b'], 0)
    features = (code-center.astype(np.float64))/scale.astype(np.float64)
    logits = features @ p['head/linear/W'] + p['head/linear/b']
    exp = np.exp(logits-logits.max(axis=1, keepdims=True))
    probability = exp/exp.sum(axis=1, keepdims=True)
    return dict(code=code, features=features, logits=logits, probability=probability)


def scaling(code):
    """Two explicit logical row-order double passes, with one final Float32 cast."""
    center, scale = [], []
    for j in range(code.shape[1]):
        mean = 0.
        for row in code:
            mean += float(row[j])
        mean /= len(code)
        variance = 0.
        for row in code:
            delta = float(row[j])-mean
            variance += delta*delta
        center.append(mean)
        scale.append(max(1e-6, math.sqrt(variance/len(code))))
    return np.array(center, dtype=np.float32), np.array(scale, dtype=np.float32)


def score(probability, labels):
    """Source diagnostic uses saved probabilities, not a re-softmax of logits."""
    nll = 0.
    for p, y in zip(probability, labels):
        nll -= math.log(max(1e-30, float(p[y])))
    return nll/len(labels), float(np.mean(probability.argmax(axis=1) == labels))


def table(path):
    """Read complete named TSV rows without filtering failures."""
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def check(directory, upstream=None):
    """Raise on contract violations; return measured independent evidence."""
    records, total_values, largest_forward, largest_gradient = [], 0, 0., 0.
    initial_heads, orders, data_sets = [], [], []
    for variant in ['pretrained', 'random']:
        root = directory/variant
        data = arrays(root/'data.npz');data_sets.append(data)
        initial = arrays(root/'initial_arrays.npz')
        outputs = arrays(root/'outputs.npz')
        center, scale = scaling(outputs['fit_code'])
        exact(data['feature_center'], center);exact(data['feature_scale'], scale)
        assert np.isfinite(scale).all() and (scale > 0).all()
        center, scale = data['feature_center'], data['feature_scale']
        assert sum(initial[k].size for k in ['encoder/W', 'encoder/b']) == 25120
        assert sum(initial[k].size for k in ['head/linear/W', 'head/linear/b']) == 330
        initial_heads.append({k: initial[k] for k in ['head/linear/W', 'head/linear/b']})
        if upstream:
            source = arrays(upstream/('model_arrays.npz' if variant == 'pretrained' else 'initial_arrays.npz'))
            for k in ['encoder/W', 'encoder/b']:
                exact(source[k], initial[k])
        for path in [root/'best.npz', root/'model_arrays.npz', root/'first_update_arrays.npz', *sorted(root.glob('epoch*_arrays.npz'))]:
            p = arrays(path)
            for k in ['encoder/W', 'encoder/b']:
                exact(p[k], initial[k])
        batch = arrays(root/'first_batch.npz')
        first = arrays(root/'first_forward.npz')
        widened = forward(initial, batch['images'], center, scale)
        for key, value in widened.items():
            np.testing.assert_allclose(value, first[key], atol=1e-5, rtol=3e-5)
            largest_forward = max(largest_forward, float(abs(value-first[key]).max()))
            total_values += value.size
        gradient_sets = []
        for shifted, filename in [(False, 'first_gradients.npz'), (True, 'wrong_label_gradients.npz')]:
            labels = (batch['labels']+int(shifted)) % 10
            delta = widened['probability'].copy()
            delta[np.arange(len(labels)), labels] -= 1
            delta /= len(labels)
            expected = {'head/linear/W': widened['features'].T@delta, 'head/linear/b': delta.sum(0)}
            saved = arrays(root/filename);gradient_sets.append(saved)
            for key, value in expected.items():
                np.testing.assert_allclose(value, saved[key], atol=5e-7, rtol=4e-5)
                largest_gradient = max(largest_gradient, float(abs(value-saved[key]).max()))
        assert np.linalg.norm(gradient_sets[0]['head/linear/W']-gradient_sets[1]['head/linear/W']) > .1
        first_update(initial, gradient_sets[0], arrays(root/'first_update_arrays.npz'))
        batches = table(root/'batches.tsv');orders.append(batches)
        assert len(batches) == 3000
        for epoch in range(1, 4):
            rows = [r for r in batches if int(r['epoch']) == epoch]
            assert sorted(int(r['row']) for r in rows) == list(range(1000))
            assert sorted(set(int(r['step']) for r in rows)) == list(range((epoch-1)*32, epoch*32))
            for r in rows:
                assert int(r['source_id']) == data['fit_source_ids'][int(r['row'])]
        first_rows = [int(r['row']) for r in batches if int(r['step']) == 0]
        exact(batch['images'], data['fit_images'][first_rows]);exact(batch['labels'], data['fit_labels'][first_rows])
        identities = set()
        for split, count in [('fit', 1000), ('validation', 200), ('test', 200)]:
            ids, labels = data[split+'_source_ids'], data[split+'_labels']
            assert ids.shape == (count,) and len(set(ids)) == count and not identities.intersection(ids)
            identities.update(ids)
            exact(np.bincount(labels, minlength=10), np.full(10, count//10, dtype=np.int64))
            low, high = (60000, 70000) if split == 'test' else (0, 60000)
            assert ((ids >= low) & (ids < high)).all()
        history = table(root/'history.tsv');assert len(history) == 3
        for row in history:
            epoch = int(row['epoch']);p = arrays(root/f'epoch{epoch}_arrays.npz')
            for split in ['fit', 'validation']:
                prediction = forward(p, data[split+'_images'], center, scale)['probability']
                nll, accuracy = score(prediction, data[split+'_labels'])
                np.testing.assert_allclose(nll, float(row[split+'_nll']), atol=2e-6, rtol=0)
                assert accuracy == float(row[split+'_accuracy'])
        selected = min(history, key=lambda r: float(r['validation_nll']))
        metrics = json.loads((root/'metrics.json').read_text())
        assert metrics['selected_epoch'] == int(selected['epoch']) == 3
        assert metrics['epochs'] == 3 and metrics['head_updates'] == 96
        assert metrics['frozen_encoder_max_change'] == 0 and metrics['single_batch_max_error'] < 2e-6
        best = arrays(root/'model_arrays.npz');selected_arrays = arrays(root/f"epoch{metrics['selected_epoch']}_arrays.npz")
        for key in best:
            exact(best[key], selected_arrays[key]);exact(best[key], arrays(root/'best.npz')[key])
        for split in ['fit', 'validation', 'test']:
            widened = forward(best, data[split+'_images'], center, scale)
            for key, value in widened.items():
                saved = outputs[split+'_'+key]
                np.testing.assert_allclose(value, saved, atol=1e-5, rtol=3e-5)
                largest_forward = max(largest_forward, float(abs(value-saved).max()));total_values += value.size
        probability, labels = outputs['test_probability'], data['test_labels']
        predicted = probability.argmax(axis=1).astype(np.int32)
        exact(predicted, outputs['test_predicted'])
        confusion = np.zeros((10, 10), dtype=np.int32)
        np.add.at(confusion, (labels, predicted), 1);exact(confusion, outputs['test_confusion'])
        nll, accuracy = score(probability, labels)
        assert accuracy == metrics['test_accuracy']
        np.testing.assert_allclose(nll, metrics['test_nll'], atol=2e-12, rtol=0)
        examples = table(root/'examples.tsv');assert len(examples) == 10
        for digit, row in enumerate(examples):
            b = int(np.flatnonzero(labels == digit)[0])
            assert int(row['true_digit']) == digit and int(row['predicted_digit']) == predicted[b]
            assert int(row['source_id']) == data['test_source_ids'][b]
        records.append(dict(variant=variant, selected_epoch=3, head_updates=96, test_nll=nll, test_accuracy=accuracy))
    for key in initial_heads[0]:
        exact(initial_heads[0][key], initial_heads[1][key])
    assert orders[0] == orders[1]
    for key in data_sets[0]:
        if not key.startswith('feature_'):
            exact(data_sets[0][key], data_sets[1][key])
    return dict(passed=True, variants=records, forward_values_checked=total_values,
                maximum_forward_error=largest_forward, maximum_gradient_error=largest_gradient,
                first_derivatives_checked=1320, exact_first_Adam=True, frozen_encoder_values_per_variant=25120,
                batch_rows_checked=6000, validation_candidates=6, matched_head_initialization_and_order=True,
                pretraining_extra_unsupervised_updates=96, head_updates_total=192,
                upstream_encoder_values_exact=bool(upstream))


def main():
    """Read artifact paths and optionally write a machine-readable receipt."""
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
