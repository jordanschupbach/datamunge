# Native Python NN example. Explicit weight/spectral reparameterization algorithms.
"""Gain/direction gradients, power iteration and spectral Lipschitz checks."""
import math
from pydatamunge import nn


def random_tensor(shape, rng):
    rng.reset_normal()
    return nn.from_vector([rng.normal32() for _ in range(math.prod(shape))], shape)


def weight_norm_dense(parameters, name, x, input_width, output_width):
    fresh = not parameters.contains(name + "/v")
    v = parameters.get(
        name + "/v", [input_width, output_width], nn.init.glorot_uniform()
    )
    if fresh:
        with nn.no_grad():
            parameters.set(name + "/g", nn.sqrt(nn.sum(nn.square(v), [0])))
    g = parameters.at(name + "/g")
    bias = parameters.get(name + "/b", [output_width], nn.init.zeros())
    norm = nn.sqrt(nn.sum(nn.square(v), [0]))
    weights = nn.mul(v, nn.div(g, norm))
    return nn.bias_add(nn.matmul(x, weights), bias), weights


def spectral_norm_dense(
    parameters, stats, name, x, input_width, output_width, iterations=1
):
    limit = nn.sqrt(nn.div(nn.scalar(6), input_width + output_width)).item()
    weights = parameters.get(
        name + "/W", [input_width, output_width], nn.init.uniform(limit)
    )
    bias = parameters.get(name + "/b", [output_width], nn.init.zeros())
    u0 = stats.get(name + "/u", [1, output_width], nn.init.normal(1))

    def l2(tensor):
        return nn.mul(
            tensor, nn.rsqrt(nn.add(nn.sum(nn.square(tensor), [1], True), 1e-12))
        )

    u = u0.detach()
    for _ in range(iterations):
        v = l2(nn.matmul(u, weights, False, True))
        u = l2(nn.matmul(v, weights))
    u, v = nn.stop_gradient(u), nn.stop_gradient(v)
    sigma = nn.squeeze(nn.matmul(nn.matmul(v, weights), u, False, True))
    normalized = nn.div(weights, sigma)
    return nn.bias_add(nn.matmul(x, normalized), bias), normalized, sigma, u0, u


def weight_norm():
    p = nn.ParameterStore(1)
    rng = nn.Mt19937_64(1)
    x = random_tensor([4, 5], rng)
    r = random_tensor([4, 3], rng)
    y, weights = weight_norm_dense(p, "d", x, 5, 3)
    v, g = p.at("d/v"), p.at("d/g")
    dv, dg, dw = nn.grad(nn.sum(nn.mul(y, r), [0, 1]), [v, g, weights])
    error_g = error_v = error_dot = error_norm = 0
    for j in range(3):
        vv = [v.at([i, j]) for i in range(5)]
        wv = [weights.at([i, j]) for i in range(5)]
        gradient = [dw.at([i, j]) for i in range(5)]
        direction = [dv.at([i, j]) for i in range(5)]
        nv = math.sqrt(sum(a * a for a in vv))
        nw = math.sqrt(sum(a * a for a in wv))
        expected_g = sum(a * b for a, b in zip(gradient, vv)) / nv
        expected_v = [
            g.at([j]) / nv * (a - expected_g * b / nv) for a, b in zip(gradient, vv)
        ]
        dot = sum(a * b for a, b in zip(vv, direction))
        error_g = max(error_g, abs(expected_g - dg.at([j])))
        error_v = max(error_v, max(abs(a - b) for a, b in zip(expected_v, direction)))
        error_dot = max(error_dot, abs(dot))
        error_norm = max(error_norm, abs(nw - g.at([j])))
        print(
            "Column",
            j,
            "v norm / gain / W norm / max W-v / v dot gradient:",
            nv,
            g.at([j]),
            nw,
            max(abs(a - b) for a, b in zip(wv, vv)),
            dot,
        )
    assert max(error_g, error_v, error_dot, error_norm) < 1e-5
    print(
        "Weight-norm errors gain/direction/orthogonality/norm:",
        error_g,
        error_v,
        error_dot,
        error_norm,
    )


def spectral_norm():
    none = nn.zeros([1, 64])
    sigma_true = 0
    last_error = 1
    for iterations in [1, 2, 3, 5, 10, 20]:
        p, stats = nn.ParameterStore(2), nn.ParameterStore(3)
        _, weights, sigma, _, _ = spectral_norm_dense(
            p, stats, "d", none, 64, 32, iterations
        )
        if iterations == 1:
            sigma_true = nn.svd(p.at("d/W"), False).s.values()[0]
        estimate = sigma.item()
        last_error = abs(estimate - sigma_true) / sigma_true
        assert estimate <= sigma_true * (1 + 1e-5)
        print(
            "Power iterations:",
            iterations,
            "sigma:",
            estimate,
            "relative error:",
            last_error,
        )
    assert last_error < 1e-2
    print("SVD largest singular value:", sigma_true)
    p, stats = nn.ParameterStore(2), nn.ParameterStore(3)
    for step in range(1, 21):
        _, _, sigma, u0, u = spectral_norm_dense(p, stats, "d", none, 64, 32)
        with nn.no_grad():
            u0.copy_(u)
        if step in [1, 2, 3, 5, 10, 20]:
            print(
                "Persistent-vector step:",
                step,
                "sigma:",
                sigma.item(),
                "relative error:",
                abs(sigma.item() - sigma_true) / sigma_true,
            )
    x = random_tensor([1000, 64], nn.Mt19937_64(4))
    y, weights, _, _, _ = spectral_norm_dense(p, stats, "d", x, 64, 32)
    normalized_sigma = nn.svd(weights, False).s.values()[0]
    maximum = 0
    for i in range(1000):
        nx = sum(x.at([i, j]) ** 2 for j in range(64))
        ny = sum(y.at([i, j]) ** 2 for j in range(32))
        maximum = max(maximum, math.sqrt(ny / nx))
    assert abs(normalized_sigma - 1) < 1e-2 and maximum <= normalized_sigma * (1 + 1e-5)
    print(
        "Normalized spectral norm:",
        normalized_sigma,
        "max output/input norm ratio over 1000 vectors:",
        maximum,
    )


def main(arguments=None):
    nn.set_num_threads(2)
    weight_norm()
    spectral_norm()
    print("All weight/spectral normalization checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
