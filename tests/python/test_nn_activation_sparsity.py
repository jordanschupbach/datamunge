"""Native activation penalties and owning nested sparse-model handles."""
import gc
import math

import pytest
from pydatamunge import nn


def test_penalty_batch_aggregation_double_constants_and_clipping_gradients():
    h = nn.from_vector_double([0., .1, .2, 1., 0., .1, .2, 1.], [2, 4]).requires_grad_()
    penalty = nn.ActivationSparsity(h, .1, .2, .2, .1)
    expected = [0., .1 / 8 + .2 * (-.2 / .1 + .8 / .9) / 8, .1 / 8, .1 / 8] * 2
    assert list(nn.grad(penalty.total, [h])[0].values()) == pytest.approx(expected, abs=1e-14)
    q = nn.from_vector_double([.01, .99, .99, .01], [2, 2])
    whole = nn.ActivationSparsity(q, .1, .2, .2)
    halves = [nn.ActivationSparsity(nn.take_rows(q, [row]), .1, .2, .2) for row in range(2)]
    assert whole.kl.item() < sum(part.kl.item() for part in halves) / 2
    assert whole.l1.item() == sum(part.l1.item() for part in halves) / 2
    with pytest.raises(RuntimeError):
        nn.ActivationSparsity(nn.ones([1, 2]), .1, .2, .2, 1e-100)


def test_sparse_model_nested_penalty_and_tensor_outlive_parent_with_gradients():
    p = nn.ParameterStore(1601)
    x = nn.from_vector([.1, .6, .8, .3, .2, .7], [2, 3])
    result = nn.SparseAutoencoder(p, x, .1, .2, 3, 2)
    assert p.size() == 4 and p.count() == 17
    assert result.loss.item() == pytest.approx(result.mse.item() + result.penalty.total.item(), abs=1e-7)
    gradients = nn.grad(result.loss, p.parameters())
    assert len(gradients) == 4
    assert all(all(math.isfinite(v) for v in g.values()) and any(v != 0 for v in g.values()) for g in gradients)
    result = nn.SparseAutoencoder(p, x, .1, .2, 3, 2)
    penalty = result.penalty
    expected = tuple(penalty.unit_mean.values())
    del result
    gc.collect()
    assert tuple(penalty.unit_mean.values()) == expected
    derivatives = nn.grad(penalty.total, [p.at('encoder/W'), p.at('encoder/b')])
    assert all(any(v != 0 for v in g.values()) for g in derivatives)
    means = penalty.unit_mean
    del penalty
    gc.collect()
    assert tuple(means.values()) == expected
    empty = nn.ParameterStore(1)
    with pytest.raises(RuntimeError):
        nn.SparseAutoencoder(empty, x, -1, .2, 3, 2)
    assert empty.size() == 0
