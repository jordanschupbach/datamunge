"""datamunge::nn binding surface: NnTensor and NnModel."""

import os
import tempfile

import pytest


def test_tensor_values_ops_and_gradients():
    from pydatamunge import datamunge as dm

    a = dm.NnTensor([1, 2, 3, 4, 5, 6], [2, 3])
    assert tuple(a.shape()) == (2, 3)
    assert a.dtype() == "float32"
    assert a.sum().item() == pytest.approx(21.0)
    assert tuple(a.matmul(a, False, True).values()) == (14.0, 32.0, 32.0, 77.0)
    assert "shape=[2, 3]" in repr(a)

    x = dm.NnTensor([1, 2, 3], [3], "float64")
    x.set_requires_grad(True)
    x.square().sum().backward()
    assert tuple(x.grad().values()) == (2.0, 4.0, 6.0)

    with pytest.raises(Exception):
        dm.NnTensor([1, 2, 3], [2, 2])


def test_model_trains_saves_and_loads():
    from pydatamunge import datamunge as dm

    x = dm.NnTensor.uniform([64, 2], 3, -1.0, 1.0)
    y = x.matmul(dm.NnTensor([2, -1], [2, 1])).add_scalar(0.5)
    model = dm.NnModel(11)
    model.add_dense(1)
    model.set_optimizer("adam", 0.05)
    first = model.train_step(x, y, "mse")
    for _ in range(300):
        last = model.train_step(x, y, "mse")
    assert last < 1e-3 < first
    assert list(model.parameter_names()) == ["dense0/W", "dense0/b"]
    assert model.parameter("dense0/W").values()[0] == pytest.approx(2.0, abs=0.05)

    path = os.path.join(tempfile.mkdtemp(), "model.npz")
    model.save(path)
    copy = dm.NnModel()
    copy.add_dense(1)
    copy.load(path)
    assert tuple(copy.predict(x).values()) == tuple(model.predict(x).values())


def test_conv_model_gradients():
    from pydatamunge import datamunge as dm

    model = dm.NnModel(5)
    model.add_reshape([6, 6, 1])
    model.add_conv2d(4, 3, 1, "same", "relu")
    model.add_max_pool2d(2, 2)
    model.add_flatten()
    model.add_dense(3)
    x = dm.NnTensor.randn([8, 36], 1)
    labels = dm.NnTensor([0, 1, 2, 0, 1, 2, 0, 1], [8], "int64")
    assert tuple(model.predict(x).shape()) == (8, 3)
    loss = model.compute_gradients(x, labels, "softmax_cross_entropy")
    assert loss > 0
    assert tuple(model.gradient("conv1/W").shape()) == (3, 3, 1, 4)
