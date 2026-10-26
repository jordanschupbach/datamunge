# Native Python NN example. Eager feeds, fetches, side effects and timings.
"""Session concepts expressed as ordinary functions over the C++ tensor API."""
from time import perf_counter
from pydatamunge import nn


def batch(rows):
    return nn.from_vector([r + c for r in range(rows) for c in range(3)], [rows, 3])


def main(arguments=None):
    weights = nn.from_vector([1, 0, 0, 1, 1, 1], [3, 2])
    bias = nn.from_vector([0.5, -0.5])
    step = nn.scalar(0, nn.DType.Int64)

    def xw(x):
        if not x.defined():
            raise ValueError("x: no value supplied")
        if x.dim() != 2 or x.size(1) != 3:
            raise ValueError("x: expected shape [?, 3]")
        return nn.matmul(x, weights)

    def y_from_xw(value):
        return nn.add(value, bias)

    def y(x):
        return y_from_xw(xw(x))

    def increment():
        step.add_(nn.scalar(1, nn.DType.Int64))

    previous_threads = nn.get_num_threads()
    try:
        nn.set_num_threads(2)
        assert nn.get_num_threads() == 2
        print(
            "Eager CPU execution; threads:",
            nn.get_num_threads(),
            "GEMM:",
            nn.gemm_backend(),
        )
        result = y(batch(2))
        print("y (batch 2):", result.str(), "sum:", nn.sum(result).item())
        assert list(result.values()) == [2.5, 2.5, 4.5, 4.5]
        assert nn.sum(result).item() == 14
        assert tuple(y(batch(5)).shape()) == (5, 2)
        fed = y_from_xw(nn.from_vector([100, 200], [1, 2]))
        assert list(fed.values()) == [100.5, 199.5]
        print("Fed intermediate:", fed.str())
        for invalid in (nn.Tensor(), nn.zeros([2, 2])):
            try:
                y(invalid)
            except ValueError as error:
                print("Expected feed error:", error)
            else:
                raise AssertionError("invalid feed accepted")
        step.fill_(0)
        for _ in range(3):
            increment()
        assert step.item() == 3
        print("Target-only counter:", step.item())
        print("Combined fetch:", y(batch(1)).str())
        increment()

        x = batch(64)
        start = perf_counter()
        value = xw(x)
        mm = perf_counter()
        value = y_from_xw(value)
        add = perf_counter()
        total = nn.sum(value)
        end = perf_counter()
        assert total.item() == 8384
        print(
            "Stage timings (microseconds):",
            [(b - a) * 1e6 for a, b in ((start, mm), (mm, add), (add, end))],
        )

        def callable_step(x):
            total = nn.sum(y(x))
            increment()
            return total

        assert callable_step(batch(2)).item() == 14
        iterations, x = 2000, batch(8)
        start = perf_counter()
        for _ in range(iterations):
            nn.sum(y(x))
            increment()
        direct = perf_counter()
        for _ in range(iterations):
            callable_step(x)
        end = perf_counter()
        print(
            "Direct / callable microseconds per step:",
            (direct - start) * 1e6 / iterations,
            (end - direct) * 1e6 / iterations,
        )
        assert step.item() == 4005
        print("Final counter:", step.item(), "All session checks passed.")
    finally:
        nn.set_num_threads(previous_threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
