"""Owning conditional Gaussian endpoints and host-owned optimization through native C++."""
import gc
import math
import pytest
from pydatamunge import nn


def test_conditional_result_inheritance_lifetime_concat_and_explicit_adam():
    p = nn.ParameterStore(2111)
    x, epsilon = nn.full([2, 3], .3), nn.full([2, 2], .7)
    labels = nn.cast(nn.from_vector([0, 2], [2]), nn.DType.Int64)
    out = nn.ConditionalVariationalAutoencoder(p, x, labels, epsilon, 3, 3, 2, 2)
    assert p.size() == 10 and p.count() == 47
    assert tuple(out.condition.values()) == (1, 0, 0, 0, 0, 1)
    assert tuple(out.encoder_input.shape()) == (2, 6)
    assert tuple(out.decoder_input.shape()) == (2, 5)
    probability, mu, z, loss = out.probability, out.mu, out.z, out.loss
    assert loss.item() == out.unweighted_loss.item()
    expected = tuple(probability.values())
    del out
    gc.collect()
    assert tuple(nn.sigmoid(nn.ConditionalVaeDecode(p, z, labels, 3, 3, 2, 2)).values()) == expected
    zero = nn.ConditionalVariationalAutoencoder(p, x, labels, nn.zeros([2, 2]), 3, 3, 2, 2)
    assert tuple(zero.mu.values()) == tuple(mu.values())
    assert tuple(zero.z.values()) == tuple(zero.mu.values())
    assert tuple(zero.z.values()) != tuple(z.values())
    gradients = nn.grad(loss, list(p.parameters()))
    assert len(gradients) == 10 and all(all(math.isfinite(v) for v in g.values()) for g in gradients)
    config = nn.OptimizerConfig()
    config.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(p, .003, config)
    before = tuple(p.flatten())
    optimizer.step(gradients)
    assert optimizer.iterations() == 1 and tuple(p.flatten()) != before


def test_supplied_class_enters_both_paths_hand_values_in_float64():
    dtype = nn.DType.Float64
    p = nn.ParameterStore(2111, dtype)
    x, epsilon, labels = nn.zeros([2, 1], dtype), nn.full([2, 1], .5, dtype), nn.from_vector_int64([0, 1], [2])
    nn.ConditionalVariationalAutoencoder(p, x, labels, epsilon, 1, 2, 1, 1)
    for name in p.names():
        p.set(name, nn.zeros(p.at(name).shape(), dtype))
    p.set('encoder/W', nn.from_vector_double([0, 1, 2], [3, 1]))
    p.set('mu/W', nn.ones([1, 1], dtype))
    p.set('decoder/W', nn.from_vector_double([1, 3, 5], [3, 1]))
    p.set('output/W', nn.ones([1, 1], dtype))
    out = nn.ConditionalVariationalAutoencoder(p, x, labels, epsilon, 1, 2, 1, 1)
    assert out.condition.dtype() == dtype
    assert tuple(out.mu.values()) == (1, 2)
    assert tuple(out.z.values()) == (1.5, 2.5)
    assert tuple(out.logits.values()) == (4.5, 7.5)
    assert tuple(out.kl.values()) == (.5, 2)
    assert tuple(nn.ConditionalVaeDecode(p, nn.ones([2, 1], dtype), labels, 1, 2, 1, 1).values()) == (4, 6)


def test_invalid_class_never_silently_encodes_all_off_or_creates_leaves():
    p = nn.ParameterStore(1)
    for labels in [nn.from_vector_int64([-1, 0], [2]), nn.from_vector_int64([0, 3], [2]), nn.zeros([2]), nn.zeros([1], nn.DType.Int32)]:
        with pytest.raises(RuntimeError):
            nn.ConditionalVariationalAutoencoder(p, nn.zeros([2, 3]), labels, nn.zeros([2, 2]), 3, 3, 2, 2)
        with pytest.raises(RuntimeError):
            nn.ConditionalVaeDecode(p, nn.zeros([2, 2]), labels, 3, 3, 2, 2)
    assert p.size() == 0


def test_decoder_needs_only_codes_and_classes_with_four_independent_leaves():
    p = nn.ParameterStore(9)
    logits = nn.ConditionalVaeDecode(p, nn.zeros([2, 2]), nn.from_vector_int64([0, 2], [2]), 3, 3, 2, 2)
    assert p.size() == 4 and p.count() == 21
    assert tuple(logits.shape()) == (2, 3)
