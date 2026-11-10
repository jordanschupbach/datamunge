"""Native VAE-GAN endpoint ownership, explicit noise and isolated host-owned phases."""
import gc
import math
import pytest
from pydatamunge import nn


FIELDS = ['mu', 'log_variance', 'z', 'probability', 'prior_probability', 'real_feature',
          'reconstruction_feature', 'prior_feature', 'real_logit', 'reconstruction_logit',
          'prior_logit', 'feature_nll', 'kl', 'pixel_nll', 'mse', 'encoder_loss',
          'generator_loss', 'discriminator_loss', 'generator_adversarial']


def test_feature_distance_is_summed_and_real_target_is_detached():
    t = nn.DType.Float64
    real = nn.ones([2, 3], t).requires_grad_()
    rec = nn.zeros([2, 3], t).requires_grad_()
    loss = nn.VaeGanFeatureNll(real, rec)
    assert list(loss.values()) == [1.5, 1.5]
    assert list(nn.grad(nn.sum(loss), [rec])[0].values()) == [-1]*6
    with pytest.raises(RuntimeError):
        nn.grad(nn.sum(nn.VaeGanFeatureNll(real, rec)), [real])
    with pytest.raises(RuntimeError):
        nn.VaeGanFeatureNll(real, nn.zeros([2, 2], t))


def test_all_nineteen_fields_survive_gc_and_continuous_noise_paths_stay_live():
    t = nn.DType.Float64
    e, g, d = [nn.ParameterStore(seed, t) for seed in [3301, 3307, 3313]]
    x = nn.full([2, 3], .3, t).requires_grad_()
    epsilon = nn.full([2, 2], .2, t).requires_grad_()
    prior = nn.full([3, 2], -.4, t).requires_grad_()
    out = nn.VaeGan(e, g, d, x, epsilon, prior, 3, 2, 2, 2)
    fields = {name: getattr(out, name) for name in FIELDS}
    del out
    gc.collect()
    assert len(fields) == 19 and e.count()+g.count()+d.count() == 46
    assert tuple(fields['probability'].shape()) == (2, 3)
    assert tuple(fields['prior_probability'].shape()) == (3, 3)
    assert all(all(math.isfinite(v) for v in field.values()) for field in fields.values())
    gradients = nn.grad(fields['generator_loss'], [x, epsilon, prior])
    assert all(any(abs(v) > 1e-12 for v in field.values()) for field in gradients)
    assert list(nn.VaeGanDecode(g, fields['z'], 3, 2, 2).values()) == list(fields['probability'].values())
    other = nn.VaeGan(e, g, d, x, epsilon, nn.ones([4, 2], t), 3, 2, 2, 2)
    assert list(other.probability.values()) == list(fields['probability'].values())
    assert other.encoder_loss.item() == fields['encoder_loss'].item()


def test_standalone_discriminator_and_generator_own_fields_and_default_census():
    e, g, d = [nn.ParameterStore(seed) for seed in [3301, 3307, 3313]]
    nn.VaeGan(e, g, d, nn.zeros([2, 784]), nn.zeros([2, 4]), nn.zeros([3, 4]))
    assert [p.count() for p in [e, g, d]] == [12696, 13408, 6289]
    assert [p.size() for p in [e, g, d]] == [6, 4, 4]
    code = nn.full([2, 4], -.2).requires_grad_()
    generator = nn.ParameterStore(3307)
    probability = nn.VaeGanDecode(generator, code)
    assert generator.size() == 4 and tuple(probability.shape()) == (2, 784)
    assert any(abs(v) > 1e-12 for v in nn.grad(nn.sum(probability), [code])[0].values())
    image = nn.full([2, 3], -.2).requires_grad_()
    out = nn.VaeGanDiscriminator(nn.ParameterStore(3313), image, 3, 2)
    feature, logit = out.feature, out.logit
    del out
    gc.collect()
    assert tuple(feature.shape()) == (2, 2) and tuple(logit.shape()) == (2, 1)
    assert any(abs(v) > 1e-12 for v in nn.grad(nn.sum(logit), [image])[0].values())


def test_caller_owns_three_separate_adam_states_and_recomputed_phase_order():
    t = nn.DType.Float64
    stores = [nn.ParameterStore(seed, t) for seed in [3301, 3307, 3313]]
    x, epsilon, prior = nn.full([2, 3], .3, t), nn.full([2, 2], .2, t), nn.full([3, 2], -.4, t)
    nn.VaeGan(*stores, x, epsilon, prior, 3, 2, 2, 2)
    config = nn.OptimizerConfig()
    config.kind = nn.OptimizerKind.Adam
    optimizers = [nn.Optimizer(p, .003, config) for p in stores]
    for group in [2, 0, 1]:
        before = [tuple(p.flatten()) for p in stores]
        out = nn.VaeGan(*stores, x, epsilon, prior, 3, 2, 2, 2)
        objective = [out.encoder_loss, out.generator_loss, out.discriminator_loss][group]
        optimizers[group].step(nn.grad(objective, list(stores[group].parameters())))
        assert all((tuple(p.flatten()) != before[k]) == (k == group) for k, p in enumerate(stores))
    assert [opt.iterations() for opt in optimizers] == [1, 1, 1]


def test_invalid_inputs_stores_and_geometry_fail_before_leaf_creation():
    e, g, d = [nn.ParameterStore(seed) for seed in [3301, 3307, 3313]]
    x, epsilon, prior = nn.zeros([2, 3]), nn.zeros([2, 2]), nn.zeros([3, 2])
    for image in [nn.full([2, 3], 1.1), nn.zeros([2, 3], nn.DType.Float64)]:
        with pytest.raises(RuntimeError): nn.VaeGan(e, g, d, image, epsilon, prior, 3, 2, 2, 2)
    with pytest.raises(RuntimeError): nn.VaeGan(e, e, d, x, epsilon, prior, 3, 2, 2, 2)
    with pytest.raises(RuntimeError): nn.VaeGan(e, g, d, x, nn.zeros([1, 2]), prior, 3, 2, 2, 2)
    with pytest.raises(RuntimeError): nn.VaeGan(e, g, d, x, epsilon, nn.zeros([0, 2]), 3, 2, 2, 2)
    with pytest.raises(RuntimeError): nn.VaeGan(e, g, d, x, epsilon, prior, 3, 2, 2, 0)
    assert e.size()+g.size()+d.size() == 0
