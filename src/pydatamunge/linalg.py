"""Double-precision dense/CSR matrices evaluated by datamunge's native kernels."""

from .datamunge import LinalgMatrix


class Mat:
    """Owned dense matrix from nested rows; @, t() and solve(A) defer computation.

    Expressions retain immutable native operands. Call eval() to keep a snapshot;
    printing or to_list() also evaluates. Binding scalars are double precision.

    Example: ``(Mat([[1, 2]]) @ Mat([[3], [4]])).to_list()`` returns ``[[11.0]]``.
    """

    def __init__(self, rows):
        rows = [list(row) for row in rows]
        cols = len(rows[0]) if rows else 0
        if any(len(row) != cols for row in rows):
            raise ValueError("matrix rows must have equal lengths")
        self._handle = LinalgMatrix(
            [float(value) for row in rows for value in row],
            len(rows), cols, isinstance(self, SpMat),
        )

    @staticmethod
    def _wrap(handle):
        result = object.__new__(SpMat if handle.is_sparse() else Mat)
        result._handle = handle
        return result

    @property
    def shape(self):
        """Result dimensions without evaluation."""
        return self._handle.rows(), self._handle.cols()

    def t(self):
        """Return a deferred transpose retaining this matrix's native storage."""
        return self._wrap(self._handle.transpose())

    def __matmul__(self, other):
        """Defer a native product, fusing preceding inverse actions into solves."""
        if not isinstance(other, Mat):
            return NotImplemented
        return self._wrap(self._handle.multiply(other._handle))

    def eval(self):
        """Evaluate once and return an owned dense/CSR result snapshot."""
        return self._wrap(self._handle.eval())

    def to_list(self):
        """Evaluate and copy coefficients into nested row lists."""
        values = list(self._handle.values())
        rows, cols = self.shape
        return [values[i * cols:(i + 1) * cols] for i in range(rows)]

    def __repr__(self):
        """Evaluate once and return the capped native matrix preview."""
        return self._handle.text()


class SpMat(Mat):
    """Owned CSR matrix from nested rows; zero entries consume no sparse storage."""


def solve(a, b=None):
    """Defer inverse action with one argument; solve by LU/QR with two.

    ``solve(A) @ B @ C`` solves A Z = B C without forming an inverse or
    an intermediate A^-1 B. ``solve(X, y)`` uses QR when X is tall.
    Bad shapes/nonfinite solve inputs raise ValueError, singular/rank-deficient
    systems raise RuntimeError on evaluation. Results use native double precision.
    """
    if not isinstance(a, Mat) or (b is not None and not isinstance(b, Mat)):
        raise TypeError("solve operands must be datamunge matrices")
    return Mat._wrap(a._handle.inverse_action() if b is None else a._handle.solve(b._handle))
