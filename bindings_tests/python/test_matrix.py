"""Native dense/CSR binding expressions, ownership and deferred solve contracts."""
import gc

import pytest

import pydatamunge as dm


@pytest.mark.parametrize("matrix", [dm.Mat, dm.SpMat])
def test_normal_equations_and_qr(matrix):
    x = matrix([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]])
    y = matrix([[1.2], [3.3], [4.4]])
    beta = dm.solve(x.t() @ x) @ x.t() @ y
    assert beta.shape == (2, 1)
    assert [r[0] for r in beta.to_list()] == pytest.approx([7 / 33, 17 / 33])
    assert [r[0] for r in dm.solve(x, y).to_list()] == pytest.approx([7 / 33, 17 / 33])
    assert isinstance(x.t() @ x, matrix)
    assert type(beta.eval()) is dm.Mat
    assert "0.2121212" in str(beta)


def test_temporary_ownership_and_snapshot():
    expression = dm.solve(dm.Mat([[2, 0], [0, 4]])) @ dm.SpMat([[6], [8]])
    gc.collect()
    snapshot = expression.eval()
    del expression
    gc.collect()
    assert snapshot.to_list() == [[3], [2]]


def test_deferred_failure_and_shapes():
    # Formation must not factorise this singular coefficient matrix.
    expression = dm.solve(dm.Mat([[0]])) @ dm.Mat([[1]])
    assert expression.shape == (1, 1)
    with pytest.raises(RuntimeError):
        expression.eval()
    with pytest.raises(ValueError):
        dm.Mat([[1, 2]]) @ dm.Mat([[1, 2]])
    with pytest.raises(ValueError):
        dm.Mat([[1], [1, 2]])
    with pytest.raises(ValueError):
        dm.solve(dm.Mat([[1, 2]]))
    with pytest.raises(ValueError):
        dm.solve(dm.Mat([[float("nan")]]), dm.Mat([[1]]))


def test_sparse_zero_and_multi_rhs_fusion():
    a = dm.SpMat([[2, 0], [0, 4]])
    assert "nnz=2" in str(a)
    assert isinstance(a.t() @ a, dm.SpMat)
    result = dm.solve(a) @ dm.Mat([[1, 2, 3], [4, 5, 6]]) @ dm.Mat([[1], [0], [1]])
    assert result.to_list() == [[2], [2.5]]
