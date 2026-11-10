"""Native known-factor data, deterministic AAE endpoints and host-owned phase isolation."""
import gc
import math
import pytest
from pydatamunge import nn


def test_native_factor_lattice_owns_all_splits_and_preserves_known_ids():
    records = nn.KnownFactorLattice()
    data = dict(records)
    assert len(records) == 9
    all_ids = set()
    for split, count in [('fit', 210), ('validation', 42), ('test', 42)]:
        images, factors, ids = (data[split + suffix] for suffix in ['_images', '_factors', '_source_ids'])
        assert tuple(images.shape()) == (count, 28, 28, 1)
        assert images.dtype() == nn.DType.Float32 and not images.requires_grad()
        assert factors.dtype() == ids.dtype() == nn.DType.Int32
        values = list(ids.values())
        assert values == sorted(values) and not (all_ids & set(values))
        all_ids.update(values)
        assert set(images.values()) == {0, 1}
    image = data['test_images']
    expected = tuple(image.values())
    del records, data
    gc.collect()
    assert tuple(image.values()) == expected and len(all_ids) == 294


def test_aae_owns_fields_and_prior_does_not_enter_deterministic_reconstruction():
    e, g, d = nn.ParameterStore(3501), nn.ParameterStore(3511), nn.ParameterStore(3527)
    x = nn.full([2, 3], .3)
    prior = nn.from_vector([-.5, .4, .8, -.2, .1, .6], [3, 2])
    out = nn.AdversarialAutoencoder(e, g, d, x, prior, 3, 2, 2)
    assert e.count() + g.count() + d.count() == 38
    assert tuple(out.prior_logit.shape()) == (3, 1)
    probability, code, loss = out.probability, out.code, out.reconstruction_loss
    expected = tuple(probability.values())
    del out
    gc.collect()
    changed = nn.AdversarialAutoencoder(e, g, d, x, nn.full([4, 2], 2), 3, 2, 2)
    assert tuple(changed.probability.values()) == expected
    assert tuple(changed.code.values()) == tuple(code.values())
    assert tuple(nn.sigmoid(nn.AaeDecode(g, code, 3, 2, 2)).values()) == expected
    gradients = nn.grad(loss, list(e.parameters()) + list(g.parameters()))
    assert len(gradients) == 8 and all(all(math.isfinite(v) for v in a.values()) for a in gradients)


def test_host_owned_three_phases_preserve_unrelated_groups_and_four_states():
    e, g, d = nn.ParameterStore(3501), nn.ParameterStore(3511), nn.ParameterStore(3527)
    x, prior = nn.full([2, 3], .3), nn.from_vector([-.5, .4, .8, -.2], [2, 2])
    nn.AdversarialAutoencoder(e, g, d, x, prior, 3, 2, 2)
    cfg = nn.OptimizerConfig()
    cfg.kind = nn.OptimizerKind.Adam
    optimizers = [nn.Optimizer(p, .003, cfg) for p in [e, g, d, e]]
    snapshot = lambda: [tuple(p.flatten()) for p in [e, g, d]]
    before = snapshot()
    out = nn.AdversarialAutoencoder(e, g, d, x, prior, 3, 2, 2)
    gradients = nn.grad(out.reconstruction_loss, list(e.parameters()) + list(g.parameters()))
    optimizers[0].step(gradients[:e.size()])
    optimizers[1].step(gradients[e.size():])
    after = snapshot()
    assert before[0] != after[0] and before[1] != after[1] and before[2] == after[2]
    before = after
    optimizers[2].minimize(nn.AdversarialAutoencoder(e, g, d, x, prior, 3, 2, 2).discriminator_loss)
    after = snapshot()
    assert before[0] == after[0] and before[1] == after[1] and before[2] != after[2]
    before = after
    optimizers[3].minimize(nn.AdversarialAutoencoder(e, g, d, x, prior, 3, 2, 2).adversarial_loss)
    after = snapshot()
    assert before[0] != after[0] and before[1:] == after[1:]
    assert [o.iterations() for o in optimizers] == [1, 1, 1, 1]


def test_summed_bernoulli_nll_and_invalid_aae_options_before_creation():
    logits = nn.from_vector_double([1000, 1000, 0, -1000, -1000, 0], [2, 3]).requires_grad_()
    target = nn.from_vector_double([0, 1, .5, 1, 0, .5], [2, 3]).requires_grad_()
    values = nn.BernoulliPixelNll(logits, target)
    assert list(values.values()) == pytest.approx([1000 + math.log(2)] * 2, abs=1e-12)
    gs = nn.grad(nn.sum(values), [logits, target])
    assert list(gs[0].values()) == [1, 0, 0, -1, 0, 0]
    e, g, d = nn.ParameterStore(1), nn.ParameterStore(2), nn.ParameterStore(3)
    with pytest.raises(RuntimeError):
        nn.AdversarialAutoencoder(e, g, d, nn.full([1, 3], -1), nn.zeros([1, 2]), 3, 2, 2)
    with pytest.raises(RuntimeError):
        nn.AdversarialAutoencoder(e, e, d, nn.zeros([1, 3]), nn.zeros([1, 2]), 3, 2, 2)
    assert e.size() + g.size() + d.size() == 0
