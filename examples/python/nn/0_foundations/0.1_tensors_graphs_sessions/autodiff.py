# Native Python NN example. Derivatives, Jacobians and gradient updates run here.
"""Reverse mode, differentiable gradients, finite differences and stop-gradient."""
import math
from pydatamunge import nn
from pydatamunge.nn_workflows import options


def matrix_loss(x, weights, bias, target):
    return nn.mean(
        nn.square(nn.sub(nn.add(nn.matmul(x, weights), bias), target)), [0, 1]
    )


def main(arguments=None):
    args = options(arguments, dict(epochs=5, full=False))
    if args["epochs"] < 1:
        raise ValueError("epochs must be positive")
    for value in [0.5, 1.0, 2.0]:
        x = nn.scalar(value).requires_grad_()
        y = nn.add(nn.pow(x, 3.0), nn.mul(x, 2.0))
        create = nn.GradOptions()
        create.create_graph = True
        dy = nn.grad(y, [x], create)[0]
        d2y = nn.grad(dy, [x])[0]
        assert abs(dy.item() - (3 * value * value + 2)) < 1e-5
        assert abs(d2y.item() - 6 * value) < 1e-5
        print(f"x={value}: y={y.item()}, dy={dy.item()}, d2y={d2y.item()}")
    for dtype, epsilon, tolerance in [
        (nn.DType.Float32, 0.01, 1e-3),
        (nn.DType.Float64, 1e-5, 1e-7),
    ]:
        x = nn.cast(
            nn.from_vector([1, 2, 3, 0, -1, 1, 2, 2, 2, -1, 0, 1], [4, 3]), dtype
        )
        target = nn.cast(nn.from_vector([1, 0, 0, 1, 1, 1, 0, 0], [4, 2]), dtype)
        weights = nn.cast(
            nn.from_vector([0.1, -0.2, 0.3, 0.4, -0.5, 0.6], [3, 2]), dtype
        ).requires_grad_()
        bias = nn.cast(nn.from_vector([0.05, -0.05]), dtype).requires_grad_()
        loss = matrix_loss(x, weights, bias, target)
        gradients = nn.grad(loss, [weights, bias])
        error = 0.0
        with nn.no_grad():
            for which, parameter in enumerate([weights, bias]):
                for index in range(parameter.numel()):
                    values = list(parameter.values())
                    samples = []
                    for sign in [-1, 1]:
                        changed = values.copy()
                        changed[index] += sign * epsilon
                        perturbed = nn.from_vector_double(
                            changed, parameter.shape()
                        ).to(dtype)
                        samples.append(
                            matrix_loss(
                                x,
                                perturbed if which == 0 else weights,
                                perturbed if which == 1 else bias,
                                target,
                            ).item()
                        )
                    error = max(
                        error,
                        abs(
                            (samples[1] - samples[0]) / (2 * epsilon)
                            - gradients[which].values()[index]
                        ),
                    )
        assert error < tolerance
        print(
            f"Matrix gradients {nn.dtype_name(dtype)}: maximum finite-difference error={error:.9g}"
        )
    x = nn.from_vector([2, 3]).requires_grad_()
    x0, x1 = nn.select(x, 0, 0), nn.select(x, 0, 1)
    y = nn.stack([nn.mul(x0, x1), nn.add(x0, x1), nn.sin(x0)])
    retain = nn.GradOptions()
    retain.retain_graph = True
    expected = [[3, 2], [1, 1], [math.cos(2), 0]]
    for row in range(3):
        seed = nn.one_hot(nn.scalar(row, nn.DType.Int64), 3)
        vjp = nn.grad([y], [x], [seed], retain)[0]
        assert max(abs(a - b) for a, b in zip(vjp.values(), expected[row])) < 1e-6
        print(f"Jacobian row {row}:", list(vjp.values()))
    x = nn.scalar(3).requires_grad_()
    stopped = nn.grad(nn.mul(x, nn.stop_gradient(x)), [x])[0].item()
    complete = nn.grad(nn.mul(x, x), [x])[0].item()
    assert stopped == 3 and complete == 6
    print("Stop-gradient derivative:", stopped, "full derivative:", complete)
    x = nn.from_vector([-2, -1, 0, 1, 2, 3])
    target = nn.from_vector([-7, -4, -1, 2, 5, 8])
    weight = nn.scalar(0).requires_grad_()
    intercept = nn.scalar(0).requires_grad_()
    initial = None
    for epoch in range(args["epochs"]):
        loss = nn.mean(
            nn.square(nn.sub(nn.add(nn.mul(x, weight), intercept), target)), [0]
        )
        if initial is None:
            initial = loss.item()
        gradients = nn.grad(loss, [weight, intercept])
        with nn.no_grad():
            weight.sub_(gradients[0], 0.05)
            intercept.sub_(gradients[1], 0.05)
        print(
            f"epoch{epoch+1}: pre-update loss={loss.item():.9g}, w={weight.item():.7g}, c={intercept.item():.7g}"
        )
    final = nn.mean(
        nn.square(nn.sub(nn.add(nn.mul(x, weight), intercept), target))
    ).item()
    assert math.isfinite(final) and final < initial
    print("All autodiff checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
