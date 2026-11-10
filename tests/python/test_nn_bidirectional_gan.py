"""Native ordered ring, joint paths, owned fields and explicit Python BiGAN phases."""
import gc
import math
import pytest
from pydatamunge import nn

FIELDS = ['fake', 'encoded', 'real_score', 'fake_score', 'discriminator_loss',
          'eg_loss', 'reconstruction', 'latent_cycle']


def test_ordered_ring_centres_seed_reproducibility_final_cast_and_guards():
    t = nn.DType.Float64
    centre = nn.GaussianRingPoints(4, 5507, 4, 2, 0, t)
    assert list(centre.values()) == pytest.approx([2, 0, 0, 2, -2, 0, 0, -2], abs=1e-14)
    first = nn.GaussianRingPoints(19, 5507)
    assert list(first.values()) == list(nn.GaussianRingPoints(19, 5507).values())
    assert list(first.values()) != list(nn.GaussianRingPoints(19, 5519).values())
    widened = nn.GaussianRingPoints(19, 5507, 8, 2, .08, t)
    assert list(first.values()) == list(nn.cast(widened, nn.DType.Float32).values())
    assert not first.requires_grad()
    for args in [(0,), (3, 1, 0), (3, 1, 8, -1), (3, 1, 8, 2, math.inf),
                 (3, 1, 8, math.nan), (3, 1, 8, 2, .08, nn.DType.Int32), (3, 1, 1, 1e300, 0)]:
        with pytest.raises(RuntimeError): nn.GaussianRingPoints(*args)


def test_default_census_unequal_pools_and_standalone_pair_inputs():
    g, e, d = [nn.ParameterStore(seed) for seed in [6007, 6011, 6029]]
    x, z = nn.full([2, 2], .3), nn.full([3, 2], -.2)
    out = nn.BidirectionalGan(g, e, d, x, z)
    assert [p.count() for p in [g, e, d]] == [82, 82, 97]
    assert [p.size() for p in [g, e, d]] == [4, 4, 4]
    assert tuple(out.fake.shape()) == (3, 2) and tuple(out.encoded.shape()) == (2, 2)
    assert list(nn.BiganMap(g, z).values()) == list(out.fake.values())
    assert list(nn.BiganMap(e, x).values()) == list(out.encoded.values())
    pair = nn.BiganJointScore(d, x, out.encoded)
    assert list(pair.values()) == list(out.real_score.values())
    assert list(nn.BiganJointScore(d, x, nn.zeros([2, 2])).values()) != list(pair.values())
    assert list(nn.BiganJointScore(d, nn.zeros([2, 2]), out.encoded).values()) != list(pair.values())


def test_eight_live_fields_survive_gc_with_signed_data_and_prior_gradients():
    t = nn.DType.Float64
    g, e, d = [nn.ParameterStore(seed, t) for seed in [6007, 6011, 6029]]
    x, z = nn.full([2, 3], -.3, t).requires_grad_(), nn.full([4, 2], .2, t).requires_grad_()
    out = nn.BidirectionalGan(g, e, d, x, z, 3)
    fields = {name: getattr(out, name) for name in FIELDS}
    del out, g, e, d
    gc.collect()
    assert len(fields) == 8
    assert tuple(fields['reconstruction'].shape()) == (2, 3)
    assert tuple(fields['latent_cycle'].shape()) == (4, 2)
    assert all(all(math.isfinite(v) for v in a.values()) for a in fields.values())
    gradients = nn.grad(fields['eg_loss'], [x, z])
    assert all(any(abs(v) > 1e-12 for v in a.values()) for a in gradients)


def test_host_owned_d_then_e_then_g_writes_and_e_independent_g_partial():
    t = nn.DType.Float64
    stores = [nn.ParameterStore(seed, t) for seed in [6007, 6011, 6029]]
    x, z = nn.full([2, 3], .3, t), nn.full([3, 2], -.2, t)
    def forward(): return nn.BidirectionalGan(*stores, x, z, 3)
    forward()
    config = nn.OptimizerConfig()
    config.kind, config.beta1 = nn.OptimizerKind.Adam, .5
    optimizers = [nn.Optimizer(p, .0002, config) for p in stores]
    old_g = None
    for group in [2, 1, 0]:
        before = [tuple(p.flatten()) for p in stores]
        if group == 1:
            old_g = [tuple(a.values()) for a in nn.grad(forward().eg_loss, list(stores[0].parameters()))]
        if group == 0:
            assert [tuple(a.values()) for a in nn.grad(forward().eg_loss, list(stores[0].parameters()))] == old_g
        out = forward()
        objective = out.discriminator_loss if group == 2 else out.eg_loss
        optimizers[group].step(nn.grad(objective, list(stores[group].parameters())))
        assert all((tuple(p.flatten()) != before[k]) == (k == group) for k, p in enumerate(stores))
    assert [opt.iterations() for opt in optimizers] == [1, 1, 1]


def test_invalid_stores_pairs_and_geometry_fail_before_leaves():
    t = nn.DType.Float64
    g, e, d = [nn.ParameterStore(seed, t) for seed in [1, 2, 3]]
    x, z = nn.zeros([2, 3], t), nn.zeros([3, 2], t)
    for bad in [nn.zeros([0, 3], t), nn.zeros([2, 0], t), nn.zeros([2, 3]), nn.full([2, 3], math.nan, t)]:
        with pytest.raises(RuntimeError): nn.BidirectionalGan(g, e, d, bad, z, 2)
    for bad in [nn.zeros([0, 2], t), nn.zeros([3, 0], t), nn.zeros([3, 2]), nn.full([3, 2], math.inf, t)]:
        with pytest.raises(RuntimeError): nn.BidirectionalGan(g, e, d, x, bad, 2)
    with pytest.raises(RuntimeError): nn.BidirectionalGan(g, g, d, x, z, 2)
    with pytest.raises(RuntimeError): nn.BidirectionalGan(g, e, g, x, z, 2)
    with pytest.raises(RuntimeError): nn.BidirectionalGan(g, e, d, x, z, 0)
    with pytest.raises(RuntimeError): nn.BiganJointScore(d, x, z, 2)
    with pytest.raises(RuntimeError): nn.BiganMap(g, z, 0, 2)
    assert g.size()+e.size()+d.size() == 0
