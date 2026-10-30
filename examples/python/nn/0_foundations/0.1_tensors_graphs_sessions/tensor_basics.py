# Native Python NN example. Construction, views, broadcasting and checks run here.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/0_foundations/0.1_tensors_graphs_sessions/tensor_basics.py
"""Numeric Tensor basics, matching the canonical eager C++ demonstration."""
import math
from pydatamunge import nn


def show(name, tensor):
    print(f"{name}: {tensor.str()}")


def main(arguments=None):
    scalar = nn.scalar(3.14)
    vector = nn.arange(0, 40, 10, nn.DType.Int32)
    matrix = nn.reshape(nn.arange(0, 6), [2, 3])
    cube = nn.reshape(nn.arange(0, 4, 0.5, nn.DType.Float64), [2, 2, 2])
    flags = nn.cast(nn.from_vector([1, 0, 1]), nn.DType.Bool)
    for name, value in [
        ("scalar", scalar),
        ("vector", vector),
        ("matrix", matrix),
        ("cube", cube),
        ("flags", flags),
    ]:
        show(name, value)
    assert abs(scalar.item() - 3.14) < 1e-6
    assert list(vector.values()) == [0, 10, 20, 30] and vector.dtype() == nn.DType.Int32
    assert cube.at([1, 0, 1]) == 2.5
    assert (
        tuple(matrix.shape()) == (2, 3)
        and matrix.dim() == 2
        and matrix.size(1) == 3
        and matrix.numel() == 6
    )
    print(
        "dtype:",
        nn.dtype_name(matrix.dtype()),
        "bytes:",
        matrix.numel() * nn.dtype_size(matrix.dtype()),
    )
    undefined = nn.Tensor()
    assert not undefined.defined()
    show("default Tensor", undefined)

    matrix = nn.reshape(nn.arange(0, 12), [4, 3])
    rows = nn.slice_axis(matrix, 0, 1, 3)
    nn.select(nn.select(rows, 0, 0), 0, 0).fill_(-1)
    assert matrix.at([1, 0]) == -1
    show("select row 2", nn.select(matrix, 0, 2))
    reshaped = nn.reshape(matrix, [3, 4])
    show("reshaped view", reshaped)
    clone = matrix.clone()
    nn.select(nn.select(clone, 0, 0), 0, 0).fill_(100)
    assert clone.at([0, 0]) == 100 and matrix.at([0, 0]) == 0
    transpose = nn.transpose(matrix)
    assert not transpose.is_contiguous() and transpose.contiguous().is_contiguous()
    assert reshaped.is_same(matrix) == False

    a = nn.from_vector([1, 2, 3, 4, 5, 6], [2, 3])
    b = nn.from_vector([1, 0, 0, 1, 1, 1], [3, 2])
    assert nn.sum(a).item() == 21
    assert list(nn.sum(a, [1]).values()) == [6, 15]
    show("a*2+1", nn.add(nn.mul(a, 2), 1))
    show("a+bias", nn.add(a, nn.from_vector([10, 20, 30])))
    show("a@b", nn.matmul(a, b))
    show("a@a.T", nn.matmul(a, a, False, True))
    assert list(nn.matmul(a, b).values()) == [4, 5, 10, 11]
    assert abs(nn.norm(a).item() - math.sqrt(91)) < 1e-5

    x = nn.from_vector([1, 2, 3, 4, 5, 6], [2, 3])
    row = nn.from_vector([10, 20, 30])
    column = nn.from_vector([1, -1], [2, 1])
    results = [
        ("reshape 3x2", nn.reshape(x, [3, 2])),
        ("flatten", nn.reshape(x, [-1])),
        ("transpose", nn.transpose(x, [1, 0])),
        ("slice", nn.slice(x, [0, 1], [2, 2])),
        ("strided slice", nn.strided_slice(x, [0, 0], [2, 3], [1, 2])),
        ("select", nn.select(x, 0, 1)),
        ("expand dims", nn.expand_dims(x, 0)),
        ("squeeze", nn.squeeze(nn.expand_dims(x, 0))),
        ("cast truncates", nn.cast(nn.mul(x, 1.7), nn.DType.Int32)),
        ("x+row", nn.add(x, row)),
        ("x*column", nn.mul(x, column)),
        ("row+column", nn.add(row, column)),
        ("broadcast", nn.broadcast_to(row, [2, 3])),
        ("tile", nn.tile(row, [2])),
        ("range", nn.arange(0, 10, 3, nn.DType.Int32)),
        ("fill", nn.full([2, 2], 7, nn.DType.Int32)),
    ]
    for name, value in results:
        show(name, value)
    assert list(nn.cast(nn.mul(x, 1.7), nn.DType.Int32).values()) == [1, 3, 5, 6, 8, 10]
    assert tuple(nn.add(row, column).shape()) == (2, 3)
    try:
        nn.add(x, nn.from_vector([1, 2]))
    except RuntimeError as error:
        print("Expected shape error:", error)
    else:
        raise AssertionError("incompatible shapes must be rejected")
    print("All tensor basics checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
