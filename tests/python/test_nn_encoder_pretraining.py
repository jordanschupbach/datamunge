"""Frozen native feature state, fitting-only scaling and a caller-owned head."""
import gc
import math
import pytest
from pydatamunge import nn


def test_population_scaling_detaches_and_retains_owned_fields_through_gc():
    code = nn.from_vector_double([1, 5, 2, 5, 4, 5], [3, 2]).requires_grad_()
    fit = nn.FitTransferScaling(nn.transpose(nn.transpose(code).contiguous()))
    center, scale = fit.center, fit.scale
    del fit, code
    gc.collect()
    assert tuple(center.values()) == (7/3, 5)
    assert scale.values()[0] == math.sqrt(14/9)
    assert scale.values()[1] == 1e-6
    assert not center.requires_grad() and not scale.requires_grad()
    assert tuple(nn.FitTransferScaling(nn.ones([1, 2]), .25).scale.values()) == (.25, .25)


def test_feature_input_path_survives_while_encoder_paths_stop():
    t = nn.DType.Float64
    image = nn.full([2, 3], .4, t).requires_grad_()
    w = nn.full([3, 2], .5, t).requires_grad_()
    b = nn.full([2], .2, t).requires_grad_()
    out = nn.FrozenEncoderFeatures(image, w, b)
    objective = nn.add(nn.sum(out), nn.mul(nn.add(nn.sum(w), nn.sum(b)), 0.))
    g = nn.grad(objective, [image, w, b])
    assert tuple(g[0].values()) == (1,)*6
    assert tuple(g[1].values()) == (0,)*6
    assert tuple(g[2].values()) == (0,)*2


def test_classifier_deep_copy_exports_fields_and_head_updates():
    w, b = nn.ones([3, 2]), nn.zeros([2])
    center, scale = nn.zeros([2]), nn.ones([2])
    model = nn.FrozenEncoderClassifier(w, b, center, scale, 2)
    head = nn.ParameterStore(4703)
    x = nn.full([2, 3], .3).requires_grad_()
    labels = nn.cast(nn.from_vector_int64([0, 1], [2]), nn.DType.Int32)
    out = model.Forward(head, x, labels)
    before = tuple(out.logits.values())
    w.fill_(9); b.fill_(9); center.fill_(9); scale.fill_(9)
    model.encoder_weight().fill_(42)
    assert tuple(model.Forward(head, x).logits.values()) == before
    assert not model.Forward(head, x).loss.defined()
    assert head.size() == 2 and head.count() == 6
    assert nn.sum(nn.abs(nn.grad(out.loss, [x])[0])).item() > 0
    original = tuple(head.flatten())
    config = nn.OptimizerConfig(); config.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(head, .003, config)
    optimizer.step(nn.grad(model.Forward(head, x, labels).loss, list(head.parameters())))
    assert tuple(head.flatten()) != original
    assert tuple(model.encoder_weight().values()) == (1,)*6
    retained = model.Forward(head, x).probability
    del model, out, x, w, b, center, scale
    gc.collect()
    assert tuple(retained.shape()) == (2, 2)
    assert all(0 <= v <= 1 for v in retained.values())


def test_invalid_targets_fail_before_head_creation_and_default_census():
    model = nn.FrozenEncoderClassifier(nn.ones([784, 32]), nn.zeros([32]), nn.zeros([32]), nn.ones([32]))
    head = nn.ParameterStore(4703)
    for labels in [nn.zeros([2]), nn.zeros([1], nn.DType.Int32), nn.full([2], 10, nn.DType.Int32)]:
        with pytest.raises(RuntimeError):
            model.Forward(head, nn.ones([2, 784]), labels)
    assert head.size() == 0
    model.Forward(head, nn.ones([2, 784]))
    assert head.count() == 330 and head.size() == 2
    for floor in [0, -1, float('nan'), float('inf'), 1e100, 1e-100]:
        with pytest.raises(RuntimeError):
            nn.FitTransferScaling(nn.ones([2, 2]), floor)
    with pytest.raises(RuntimeError):
        nn.FrozenEncoderClassifier(nn.ones([3, 2]), nn.zeros([2]), nn.zeros([2]), nn.zeros([2]))
