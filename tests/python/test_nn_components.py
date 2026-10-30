"""Shared NN example components are bound through SWIG and grouped by C++ namespace."""
import math
import gc

import pytest

from pydatamunge import nn
from pydatamunge import nn_components


def test_every_component_namespace_is_present():
    for name in nn_components.NAMESPACES:
        namespace = getattr(nn_components, name)
        assert vars(namespace), name
    assert callable(nn_components.image_training.ImageBatch)
    assert callable(nn_components.resnet_demo.CompactResNet)
    assert callable(nn_components.check.Require)
    # C++ image_training::ImageTrainingConfig keeps its field names and defaults.
    cfg = nn_components.image_training.ImageTrainingConfig()
    assert (cfg.classes, cfg.epochs, cfg.batch, cfg.validation_rows) == (10, 10, 128, 5000)
    assert cfg.learning_rate == pytest.approx(0.001) and cfg.l2 == pytest.approx(0.0001)
    cfg.epochs = 3
    assert cfg.epochs == 3


def test_checks_and_runtime_helpers_behave_like_cpp(monkeypatch):
    check, runtime = nn_components.check, nn_components.example_runtime
    check.Require(True, "fine")
    with pytest.raises(RuntimeError, match="broken"):
        check.Require(False, "broken")
    check.Near(1.0, 1.0 + 1e-9, 1e-6, "close")
    with pytest.raises(RuntimeError):
        check.Near(1.0, 2.0, 1e-6, "far")
    monkeypatch.setenv("DATAMUNGE_EXAMPLES_FULL", "0")
    assert not runtime.Full() and runtime.Limit(12, 1) == 1 and runtime.LimitInt64(10000, 32) == 32
    monkeypatch.setenv("DATAMUNGE_EXAMPLES_FULL", "1")
    assert runtime.Full() and runtime.Limit(12, 1) == 12
    monkeypatch.setenv("DATAMUNGE_DATA_DIR", "/data/root")
    assert nn_components.paths.Dataset("cifar10") == "/data/root/cifar10"


def test_image_batch_and_objective_match_the_cpp_pieces():
    image_training = nn_components.image_training
    fixture = nn.ImageDataset()
    fixture.images = nn.ones([2, 32, 32, 3])
    fixture.labels = nn.from_vector_int64([2, 7], [2])
    x, y = image_training.ImageBatch(fixture, [1, 0])
    assert list(x.shape()) == [2, 32, 32, 3] and all(v == 1 for v in x.values())
    assert list(y.values()) == [7, 2]
    first, repeat = nn.Generator(151), nn.Generator(151)
    a = image_training.ImageBatch(fixture, [1, 0], first)[0].values()
    b = image_training.ImageBatch(fixture, [1, 0], repeat)[0].values()
    assert a == b and set(a) <= {-1.0, 1.0}
    params = nn.ParameterStore(17)
    logits = nn.dense(params, "output", nn.reshape(x, [2, 32 * 32 * 3]), 10)
    plain = image_training.TrainingCrossEntropy(logits, y, 10, 0.0)
    assert math.isfinite(plain.item())
    smoothed = image_training.TrainingCrossEntropy(logits, y, 10, 0.1)
    assert smoothed.item() != plain.item()
    penalty = image_training.L2Penalty(params)
    assert penalty.item() > 0
    objective = image_training.WithL2(plain, params, 0.5)
    assert objective.item() == pytest.approx(plain.item() + 0.25 * penalty.item(), rel=1e-6)
    assert image_training.WithL2(plain, params, 0.0).item() == plain.item()


def test_compact_resnet_handle_is_callable_like_the_cpp_builder():
    model = nn_components.resnet_demo.CompactResNet(18)
    params, state = nn.ParameterStore(17), nn.ParameterStore(17)
    with nn.no_grad():
        logits = model(params, state, nn.zeros([2, 32, 32, 3]), False)
    assert list(logits.shape()) == [2, 10]
    assert params.count() == 701466  # the resnet_basic example's compact ResNet-18
    with nn.no_grad():
        again = model.call(params, state, nn.zeros([2, 32, 32, 3]), False)
    assert again.values() == logits.values()
    empty = nn_components.nn_components.LogitsModel()
    with pytest.raises(RuntimeError, match="empty LogitsModel"):
        empty(params, state, nn.zeros([1, 32, 32, 3]), False)


def test_architecture_components_are_bound():
    cfg = nn_components.attention_cnn.AttentionResNetConfig()
    assert not cfg.has_conv2_hook()
    cfg.use_selective_kernel(32, 16)
    assert cfg.has_conv2_hook()
    assert nn_components.attention_cnn.ChannelAttention_kSqueezeExcitation != nn_components.attention_cnn.ChannelAttention_kNone
    layout = nn_components.dpn.DPNLayoutFor(92)
    assert layout.blocks(0) >= 1 and layout.inc(0) >= 1
    with pytest.raises(RuntimeError):
        layout.blocks(4)
    # %ignore on a fixed-array field must not also suppress its replacement
    # accessor. Check real forward results, including out-of-range handling.
    params = nn.ParameterStore(17)
    config = nn_components.vgg.VggConfig()
    config.depth, config.image_size, config.classes = 11, 32, 3
    config.width_divisor, config.dense_width = 16, 8
    with nn.no_grad():
        result = nn_components.vgg.Vgg(params, nn.zeros([1, 32, 32, 3]), config)
    assert list(result.pooling(4).shape()) == [1, 1, 1, 32]
    with pytest.raises(RuntimeError, match="index"):
        result.pooling(5)
    assert nn_components.mobilenet.MakeDivisible(37.0, 8) == 40


def test_image_resize_accepts_spatial_pairs_and_keeps_cpp_gradients():
    x = nn.from_vector([0, 10, 20, 30], [1, 2, 2, 1]).requires_grad_()
    resized = nn.resize_bilinear(x, [4, 4], False, True)
    assert list(resized.shape()) == [1, 4, 4, 1]
    assert list(resized.values()) == [0, 2.5, 7.5, 10, 5, 7.5, 12.5, 15,
                                     15, 17.5, 22.5, 25, 20, 22.5, 27.5, 30]
    assert list(nn.grad(nn.sum(resized), [x])[0].values()) == [4, 4, 4, 4]
    size = nn.ImageSize()
    size.height, size.width = 1, 1
    assert list(nn.resize_nearest(x, size).values()) == [0]
    with pytest.raises(ValueError):
        nn.resize_bilinear(x, [4])
    with pytest.raises(TypeError):
        nn.resize_bilinear(x, [4.5, 4])


def test_competitive_cpp_masks_preserve_ties_signs_and_gradients():
    ae = nn_components.ae
    x = nn.from_vector([3, 1, 1, 0, 2, 4, 0, 0, -1, -2, 5, 5, -4, -1, -2, -3], [4, 4]).requires_grad_()
    expected = [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0]
    mask = ae.CompetitiveMask(x.detach(), False, 2)
    assert list(mask.values()) == expected
    assert list(nn.grad(nn.sum(nn.mul(x, mask)), [x])[0].values()) == expected
    lifetime = ae.CompetitiveMask(x.detach(), True)
    assert list(lifetime.values()) == [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    ae.CompetitiveChecks()
    assert ae.SameTensorBits(nn.from_vector([0.0], [1]), nn.from_vector([0.0], [1]))
    assert not ae.SameTensorBits(nn.from_vector([0.0], [1]), nn.from_vector([-0.0], [1]))
    assert not ae.SameTensorBits(nn.zeros([1, 2]), nn.zeros([2, 1]))
    assert not ae.SameTensorBits(nn.from_vector_double([1.0], [1]),
                                 nn.from_vector_double([1.0 + 2.0 ** -30], [1]))


@pytest.mark.parametrize("lifetime", [False, True])
def test_competitive_cpp_forward_connects_to_host_optimizer(lifetime):
    ae = nn_components.ae
    params = nn.ParameterStore(1931)
    x = nn.from_vector([((i * 13) % 31) / 30.0 for i in range(4 * 784)], [4, 784])
    with nn.no_grad():
        ae.CompetitiveAutoencoder(params, x, lifetime, False)
    assert params.count() == 101200 and params.size() == 4
    before = list(params.at("encoder/W").values())
    config = nn.OptimizerConfig()
    config.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(params, 0.003, config)
    result = ae.CompetitiveAutoencoder(params, x, lifetime, False)
    # Reading a bound output struct must keep the C++ autograd graph alive.
    reconstruction = result.training_reconstruction
    del result
    loss = ae.ReconstructionMse(reconstruction, x)
    optimizer.minimize(loss)
    assert math.isfinite(loss.item()) and loss.item() > 0
    assert list(params.at("encoder/W").values()) != before
    with nn.no_grad():
        output = ae.CompetitiveAutoencoder(params, x, lifetime, False)
        assert list(output.code.shape()) == [4, 64]
        assert list(output.reconstruction.shape()) == [4, 784]
        ae.CheckCompetitiveSupport([output.raw, output.mask, output.training_code,
                                    output.training_reconstruction], lifetime, False)


def _memory_batch(copy_task):
    integer = lambda values, shape: nn.cast(nn.from_vector(values, shape), nn.DType.Int32)
    if not copy_task:
        return [("facts", integer([1, 8, 2, 9, 0, 0, 0, 0, 0, 0, 0, 0,
                                    1, 8, 2, 9, 3, 10, 0, 0, 0, 0, 0, 0], [2, 6, 2])),
                ("query", integer([1, 2], [2])), ("target", integer([0, 1], [2])),
                ("lengths", integer([2, 3], [2]))]
    inputs, targets = [0.0] * 72, [0.0] * 54
    for batch, length in enumerate([2, 3]):
        for step in range(length):
            for bit in range(3):
                value = float((batch + step + bit) % 2)
                inputs[(batch * 9 + step) * 4 + bit] = value
                targets[(batch * 9 + length + 1 + step) * 3 + bit] = value
        inputs[(batch * 9 + length) * 4 + 3] = 1
    return [("input", nn.from_vector(inputs, [2, 9, 4])),
            ("target", nn.from_vector(targets, [2, 9, 3])),
            ("lengths", integer([2, 3], [2]))]


@pytest.mark.parametrize("name,count,tensors", [
    ("Ntm", 851, 8), ("Dnc", 799, 7), ("MemNet", 312, 3), ("KeyValue", 264, 4),
])
def test_memory_cpp_models_support_host_gradients_adam_and_restore(tmp_path, name, count, tensors):
    component = nn_components.memory_components
    batch = _memory_batch(name in ("Ntm", "Dnc"))
    kind = getattr(component, "Kind_" + name)
    model = component.Model(kind, batch)
    parameters = model.parameters()
    assert parameters.count() == count and parameters.size() == tensors
    before = {key: tuple(tensor.values()) for key, tensor in model.Arrays()}
    cfg = nn.OptimizerConfig()
    cfg.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(parameters, 0.005 if name in ("Ntm", "Dnc") else 0.01, cfg)
    outputs = model.Forward(model.Leaves(batch), batch)
    loss = model.Loss(outputs, batch)
    del outputs
    gc.collect()
    options = nn.GradOptions()
    options.allow_unused = True
    gradients = nn.grad(loss, parameters.parameters(), options)
    assert all(gradient.defined() and all(math.isfinite(v) for v in gradient.values()) for gradient in gradients)
    optimizer.step(gradients)
    assert math.isfinite(loss.item()) and loss.item() > 0
    assert {key: tuple(tensor.values()) for key, tensor in model.Arrays()} != before
    diagnostic = dict(model.Evaluate(batch))
    assert math.isfinite(diagnostic["loss"].item())
    assert diagnostic["input_gradient"].defined()
    assert nn.equal_bits(diagnostic["logits"], model.Predict(batch))
    if name in ("Ntm", "Dnc"):
        state_names = list(model.StateNames())
        lengths = dict(batch)["lengths"].values()
        for time in range(9):
            state = [(key, nn.select(diagnostic[key], 1, time)) for key in state_names]
            step = dict(model.Step(nn.select(dict(batch)["input"], 1, time), state))
            active = [row for row, length in enumerate(lengths) if time <= 2 * length]
            for key, value in step.items():
                if active:
                    expected = nn.select(diagnostic[key], 1, time + int(key in state_names))
                    difference = nn.max(nn.abs(nn.sub(nn.take_rows(value, active), nn.take_rows(expected, active)))).item()
                    assert difference < 3e-5, (name, time, key)
    corrupt = dict(batch)
    corrupt["target"] = nn.zeros_like(corrupt["target"])
    assert nn.equal_bits(model.Predict(batch), model.Predict(list(corrupt.items())))
    checkpoint = tmp_path / (name + ".npz")
    nn.save_npz(str(checkpoint), model.Arrays())
    deployed = component.Model(kind, batch)
    deployed.Load(str(checkpoint))
    assert nn.equal_bits(model.Predict(batch), deployed.Predict(batch))


def test_memory_component_rejects_duplicate_record_names():
    component = nn_components.memory_components
    batch = _memory_batch(True)
    with pytest.raises(RuntimeError, match="duplicate memory batch field"):
        component.Model(component.Kind_Ntm, batch + [batch[0]])


def test_memory_data_generator_matches_copy_and_unambiguous_qa_protocols(tmp_path):
    import json
    component = nn_components.memory_components
    component.PrepareData(str(tmp_path))
    for task in ("ntm", "memnet"):
        manifest = json.loads((tmp_path / task / "manifest.json").read_text())
        assert manifest["external_selection"] is False
        arrays = dict(nn.load_npz(str(tmp_path / task / "data.npz")))
        source_ids = set()
        for split, count in (("fit", 128), ("validation", 32), ("test", 32)):
            lengths = [int(v) for v in arrays[split + "_lengths"].values()]
            identifiers = list(arrays[split + "_source_ids"].values())
            assert len(lengths) == count and len(set(identifiers)) == count
            assert not source_ids.intersection(identifiers)
            source_ids.update(identifiers)
            target = arrays[split + "_target"].values()
            if task == "ntm":
                inputs = arrays[split + "_input"].values()
                for row, length in enumerate(lengths):
                    assert 1 <= length <= 4 and inputs[(row * 9 + length) * 4 + 3] == 1
                    for step in range(length):
                        for bit in range(3):
                            assert target[(row * 9 + length + 1 + step) * 3 + bit] == inputs[(row * 9 + step) * 4 + bit]
            else:
                facts, queries = arrays[split + "_facts"].values(), arrays[split + "_query"].values()
                for row, length in enumerate(lengths):
                    pairs = [(facts[(row * 6 + slot) * 2], facts[(row * 6 + slot) * 2 + 1]) for slot in range(length)]
                    assert 1 <= length <= 6 and len({entity for entity, _ in pairs}) == length
                    assert dict(pairs)[queries[row]] == 8 + target[row]
        path = tmp_path / task / "data.npz"
        original = path.read_bytes()
        component.PrepareData(str(tmp_path), True)
        assert path.read_bytes() == original
    marker = tmp_path / "ntm/manifest.json"
    marker.write_text("user-provided provenance")
    component.PrepareData(str(tmp_path))
    assert marker.read_text() == "user-provided provenance"
