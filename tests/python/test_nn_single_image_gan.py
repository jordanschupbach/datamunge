"""Native single-image noise, detached scale amplitude and owning patch game."""
import gc

import pytest
from pydatamunge import nn as n


def inputs():
    t = n.DType.Float64
    g, d = n.ParameterStore(7307, t), n.ParameterStore(7309, t)
    x = n.full([2, 5, 7, 3], .3, t)
    p = n.full([3, 5, 7, 3], -.2, t)
    z = n.SingleImageNoise(3, 5, 7, False, 7411, t)
    rp = n.full(x.shape(), .1, t)
    rz = n.SingleImageNoise(2, 5, 7, True, 7411, t)
    return g, d, x, p, z, rp, rz, n.full([1], .2, t)


def test_single_stream_coarse_shared_rgb_and_detached_double_rms():
    for shared in [False, True]:
        z = n.SingleImageNoise(2, 3, 5, shared, 7411)
        expected = n.cast(n.GaussianNoise([3, 5, 1 if shared else 3], 2, 7411), n.DType.Float32)
        if shared:
            expected = n.tile(expected, [1, 1, 1, 3])
        assert n.equal_bits(z, expected) and not z.requires_grad()
    x = n.full([1, 3, 5, 3], -2, n.DType.Float64).requires_grad_()
    p = n.ones(x.shape(), x.dtype()).requires_grad_()
    a = n.SingleImageAmplitude(x, p)
    assert list(a.shape()) == [1] and a.item() == pytest.approx(.3, abs=1e-15)
    assert not a.requires_grad() and n.SingleImageAmplitude(x, p, 0).item() == 0
    with pytest.raises(RuntimeError):
        n.SingleImageNoise(0, 2, 3, True, 1)
    with pytest.raises(RuntimeError):
        n.SingleImageAmplitude(x, p, -1)


def test_census_rectangles_unbounded_skip_and_full_generator_half_critic_loss():
    args = inputs()
    g, d, x, p, z, rp, rz, a = args
    o = n.SingleImageGan(*args)
    assert [g.count(), d.count(), g.size(), d.size()] == [371, 297, 6, 6]
    assert list(o.fake.shape()) == [3, 5, 7, 3]
    assert list(o.real_score.shape()) == [2, 5, 7, 1]
    assert o.g_loss.item() == pytest.approx(o.adversarial_loss.item() + 10 * o.reconstruction_loss.item(), abs=1e-14)
    expected_d = .5 * (n.mean(n.square(n.sub(o.real_score, 1))).item() + n.mean(n.square(o.fake_score)).item())
    assert o.d_loss.item() == pytest.approx(expected_d, abs=1e-15)
    for store in [g, d]:
        for name in store.names():
            store.set(name, n.zeros(store.at(name).shape(), x.dtype()))
    high = n.full(p.shape(), 4, x.dtype())
    assert n.equal_bits(n.SingleImageGenerate(g, high, z, a).image, high)
    scalar = n.full([], .2, x.dtype())
    assert n.equal_bits(n.SingleImageGenerate(g, p, z, scalar).image, n.SingleImageGenerate(g, p, z, a).image)


def test_d_then_fresh_g_keeps_owners_and_cold_warm_start_optimizers():
    args = inputs()
    g, d = args[:2]
    o = n.SingleImageGan(*args)
    keep = n.GradOptions()
    keep.retain_graph = True
    stale = n.grad(o.g_loss, g.parameters(), keep)
    dd = n.grad(o.d_loss, d.parameters())
    cfg = n.OptimizerConfig()
    cfg.kind = n.OptimizerKind.Adam
    cfg.beta1 = .5
    go, do = n.Optimizer(g, .0005, cfg), n.Optimizer(d, .0005, cfg)
    before = list(g.flatten())
    do.step(dd)
    assert list(g.flatten()) == before
    after = list(d.flatten())
    fresh = n.grad(n.SingleImageGan(*args).g_loss, g.parameters())
    assert max(n.max(n.abs(n.sub(a, b))).item() for a, b in zip(stale, fresh)) > 1e-9
    go.step(fresh)
    assert list(d.flatten()) == after and go.iterations() == do.iterations() == 1
    ng, nd = n.ParameterStore(7307, args[2].dtype()), n.ParameterStore(7309, args[2].dtype())
    n.SingleImageGan(ng, nd, *args[2:])
    for source, target in [(g, ng), (d, nd)]:
        for name in source.names():
            target.set(name, source.at(name).detach().clone())
        assert list(target.flatten()) == list(source.flatten())
    assert n.Optimizer(ng, .0005, cfg).iterations() == n.Optimizer(nd, .0005, cfg).iterations() == 0


def test_fields_own_graphs_for_both_noise_and_skip_after_parent_store_collection():
    t = n.DType.Float64
    x = n.full([1, 3, 5, 3], .2, t).requires_grad_()
    z = n.SingleImageNoise(1, 3, 5, False, 7411, t).requires_grad_()
    a = n.full([], .3, t).requires_grad_()
    g, d = n.ParameterStore(7307, t), n.ParameterStore(7309, t)
    out = n.SingleImageGan(g, d, x, x, z, x, z, a)
    image, loss, score = out.fake, out.g_loss, out.fake_score
    del out, g, d
    gc.collect()
    graph = n.GradOptions()
    graph.create_graph = True
    first = n.grad(n.add(loss, n.mean(n.square(image))), [x, z, a], graph)
    assert all(n.sum(n.abs(v)).item() > 0 for v in first)
    assert n.max(n.abs(n.grad(n.sum(n.square(first[0])), [z])[0])).item() > 1e-10
    assert list(score.shape()) == [1, 3, 5, 1]


def test_invalid_geometry_aliases_and_amplitudes_create_no_leaves():
    g, d, x, p, z, rp, rz, a = inputs()
    with pytest.raises(RuntimeError):
        n.SingleImageGan(g, g, x, p, z, rp, rz, a)
    with pytest.raises(RuntimeError):
        n.SingleImageGan(g, d, x, p, z, rp, n.zeros([1, 5, 7, 3], x.dtype()), a)
    with pytest.raises(RuntimeError):
        n.SingleImageGenerate(g, p, z, n.ones([1, 1], x.dtype()))
    with pytest.raises(RuntimeError):
        n.SingleImageDiscriminate(d, n.full(x.shape(), float('nan'), x.dtype()))
    assert g.size() == d.size() == 0
