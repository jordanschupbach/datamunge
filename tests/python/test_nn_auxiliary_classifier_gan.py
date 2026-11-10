"""Native conditional G, image-only D, stable losses and explicit Python update phases."""
import gc
import math
import pytest
from pydatamunge import nn

FIELDS = ['fake', 'real_logit', 'fake_logit', 'real_probability', 'fake_probability',
          'discriminator_loss', 'generator_loss', 'real_class_logits', 'fake_class_logits',
          'real_class_probability', 'fake_class_probability', 'real_class_loss',
          'fake_class_loss', 'source_discriminator_loss', 'source_generator_loss']


def test_stable_extreme_logits_keep_tiny_real_and_strong_nonsaturating_gradients():
    for dtype in [nn.DType.Float32, nn.DType.Float64]:
        real = nn.full([2, 1], 80., dtype).requires_grad_()
        fake = nn.full([3, 1], -80., dtype).requires_grad_()
        loss = nn.LogisticGanObjectives(real, fake)
        d, g = loss.discriminator, loss.generator
        del loss
        gc.collect()
        assert g.item() == pytest.approx(80)
        derivatives = nn.grad(d, [real, fake])
        assert list(derivatives[0].values()) == pytest.approx([-.5*math.exp(-80)]*2, rel=1e-6, abs=0)
        assert list(derivatives[1].values()) == pytest.approx([math.exp(-80)/3]*3, rel=1e-6, abs=0)
        assert list(nn.grad(g, [fake])[0].values()) == pytest.approx([-1/3]*3)


def test_default_census_image_only_discriminator_and_condition_changes():
    g, d = nn.ParameterStore(5209), nn.ParameterStore(5219)
    x, z = nn.full([2, 784], .3), nn.full([2, 16], -.2)
    labels = nn.zeros([2], nn.DType.Int32)
    first = nn.AuxiliaryClassifierGan(g, d, x, z, labels)
    other = nn.AuxiliaryClassifierGan(g, d, x, z, nn.ones([2], nn.DType.Int64))
    assert [p.count() for p in [g, d]] == [26736, 25483]
    assert list(d.names()) == ['hidden/W', 'hidden/b', 'class/W', 'class/b', 'source/W', 'source/b']
    assert list(first.real_logit.values()) == list(other.real_logit.values())
    assert list(first.real_class_probability.values()) == list(other.real_class_probability.values())
    assert list(first.fake.values()) != list(other.fake.values())
    assert first.real_class_loss.item() != other.real_class_loss.item()
    assert list(nn.ConditionalGanGenerate(g, z, labels).values()) == list(first.fake.values())
    out = nn.AuxiliaryGanDiscriminate(d, x)
    source, logits, probability = out.source, out.class_logits, out.class_probability
    del out, d
    gc.collect()
    assert list(source.values()) == list(first.real_logit.values())
    assert tuple(logits.shape()) == (2, 10)
    assert all(math.isfinite(v) for v in probability.values())


def test_fifteen_owning_fields_and_continuous_noise_gradients_survive_parent_gc():
    t = nn.DType.Float64
    g, d = nn.ParameterStore(5209, t), nn.ParameterStore(5219, t)
    x = nn.full([2, 3], .3, t).requires_grad_()
    z = nn.full([2, 2], -.2, t).requires_grad_()
    out = nn.AuxiliaryClassifierGan(g, d, x, z, nn.zeros([2], nn.DType.Int64), 3, 2, 2)
    fields = {name: getattr(out, name) for name in FIELDS}
    del out, g, d
    gc.collect()
    assert len(fields) == 15 and tuple(fields['fake'].shape()) == (2, 3)
    assert all(all(math.isfinite(v) for v in field.values()) for field in fields.values())
    derivative = nn.grad(fields['generator_loss'], [z])[0]
    assert any(abs(v) > 1e-12 for v in derivative.values())


def test_two_separate_adam_states_and_recomputed_d_then_g_write_isolation():
    t = nn.DType.Float64
    stores = [nn.ParameterStore(seed, t) for seed in [5209, 5219]]
    x, z, labels = nn.full([2, 3], .3, t), nn.full([2, 2], -.2, t), nn.zeros([2], nn.DType.Int32)
    def forward():
        return nn.AuxiliaryClassifierGan(*stores, x, z, labels, 3, 2, 2)
    forward()
    config = nn.OptimizerConfig()
    config.kind, config.beta1 = nn.OptimizerKind.Adam, .5
    optimizers = [nn.Optimizer(p, .0002, config) for p in stores]
    for group in [1, 0]:
        before = [tuple(p.flatten()) for p in stores]
        out = forward()
        objective = out.generator_loss if group == 0 else out.discriminator_loss
        optimizers[group].step(nn.grad(objective, list(stores[group].parameters())))
        assert all((tuple(p.flatten()) != before[k]) == (k == group) for k, p in enumerate(stores))
    assert [opt.iterations() for opt in optimizers] == [1, 1]


def test_coarse_pool_scale_gradients_and_invalid_inputs_before_leaves():
    t = nn.DType.Float64
    x = nn.ones([2, 784], t).requires_grad_()
    coarse = nn.GanCoarsePixels(x)
    assert tuple(coarse.shape()) == (2, 16)
    assert list(coarse.values()) == pytest.approx([.25]*32)
    assert list(nn.grad(nn.sum(coarse), [x])[0].values()) == pytest.approx([1/196]*1568)
    g, d = nn.ParameterStore(5209), nn.ParameterStore(5219)
    x, z, labels = nn.zeros([2, 3]), nn.zeros([2, 2]), nn.zeros([2], nn.DType.Int32)
    for bad in [nn.zeros([2]), nn.zeros([1], nn.DType.Int32), nn.full([2], -1, nn.DType.Int64), nn.full([2], 2, nn.DType.Int32)]:
        with pytest.raises(RuntimeError): nn.AuxiliaryClassifierGan(g, d, x, z, bad, 3, 2, 2)
        with pytest.raises(RuntimeError): nn.ConditionalGanGenerate(g, z, bad, 3, 2, 2)
        with pytest.raises(RuntimeError): nn.AuxiliaryClassLoss(nn.zeros([2, 2]), bad)
    with pytest.raises(RuntimeError): nn.AuxiliaryClassifierGan(g, g, x, z, labels, 3, 2, 2)
    with pytest.raises(RuntimeError): nn.AuxiliaryClassifierGan(g, d, x, nn.zeros([1, 2]), labels, 3, 2, 2)
    with pytest.raises(RuntimeError): nn.AuxiliaryClassifierGan(g, d, nn.full([2, 3], math.nan), z, labels, 3, 2, 2)
    with pytest.raises(RuntimeError): nn.AuxiliaryClassifierGan(g, d, x, z, labels, 3, 2, 0)
    with pytest.raises(RuntimeError): nn.LogisticGanObjectives(nn.zeros([2, 1]), nn.zeros([2, 2]))
    with pytest.raises(RuntimeError): nn.GanCoarsePixels(nn.zeros([2, 783]))
    assert g.size()+d.size() == 0
