# Native Python NN example. Batch statistics, moving state, deep stacks and gradients.
"""Training/inference batch normalization checked against independent moments."""
import math
from pydatamunge import nn
from pydatamunge.nn_workflows import options

EPS = 1e-3


def random_matrix(rows, cols, rng, means, stds):
    rng.reset_normal()
    z = nn.from_vector([rng.normal32() for _ in range(rows * cols)], [rows, cols])
    return nn.add(nn.mul(z, nn.from_vector(stds)), nn.from_vector(means))


def configuration(momentum):
    option = nn.BatchNormOptions()
    option.momentum = momentum
    option.eps = EPS
    return option


def hand_check():
    p, stats = nn.ParameterStore(), nn.ParameterStore()
    rng = nn.Mt19937_64(1)
    x = random_matrix(8, 4, rng, [1, -2, 0, 5], [1, 3, 0.1, 2])
    infer = nn.batch_norm(p, stats, "bn1", x, False, configuration(0.9))
    train = nn.batch_norm(p, stats, "bn1", x, True, configuration(0.9))
    fused = nn.batch_norm_train(x, p.at("bn1/gamma"), p.at("bn1/beta"), EPS)
    ey = em = eb = eu = ei = el = update = 0
    for c in range(4):
        values = [x.at([i, c]) for i in range(8)]
        mean = sum(values) / 8
        variance = sum((v - mean) ** 2 for v in values) / 8
        for i, v in enumerate(values):
            ey = max(
                ey, abs((v - mean) / math.sqrt(variance + EPS) - fused.y.at([i, c]))
            )
            el = max(el, abs(train.at([i, c]) - fused.y.at([i, c])))
            ei = max(ei, abs(v / math.sqrt(1 + EPS) - infer.at([i, c])))
        reported = fused.variance.at([c])
        em = max(em, abs(mean - fused.mean.at([c])))
        eb = max(eb, abs(variance - reported))
        eu = max(eu, abs(variance * 8 / 7 - reported))
        update = max(
            update,
            abs(stats.at("bn1/moving_mean").at([c]) - 0.1 * fused.mean.at([c])),
            abs(stats.at("bn1/moving_var").at([c]) - (0.9 + 0.1 * reported)),
        )
        print(
            "Channel",
            c,
            "mean / biased variance / unbiased variance / reported:",
            mean,
            variance,
            variance * 8 / 7,
            reported,
        )
    assert (
        ey < 1e-5
        and em < 1e-5
        and eu < 1e-5
        and eb > 0.1
        and ei < 1e-5
        and el == 0
        and update < 1e-6
    )
    print(
        "Hand errors: output, mean, biased/unbiased variance, inference, layer, moving update:",
        ey,
        em,
        eb,
        eu,
        ei,
        el,
        update,
    )


def moving_statistics(steps):
    p, stats = nn.ParameterStore(), nn.ParameterStore()
    rng = nn.Mt19937_64(2)
    means, stds = [1, 2, 3, 4], [1, 2, 0.5, 3]
    for step in range(1, steps + 1):
        with nn.no_grad():
            nn.batch_norm(
                p,
                stats,
                "bn1",
                random_matrix(32, 4, rng, means, stds),
                True,
                configuration(0.9),
            )
        if step in [1, 10, 30, 100, steps]:
            mm = list(stats.at("bn1/moving_mean").values())
            mv = list(stats.at("bn1/moving_var").values())
            print("Statistics step", step, "moving mean:", mm, "moving variance:", mv)
    if steps == 300:
        assert max(abs(a - b) for a, b in zip(mm, [1.021, 2.041, 2.990, 4.014])) < 2e-3
        assert max(abs(a - b) for a, b in zip(mv, [1.102, 4.113, 0.234, 8.317])) < 2e-3
    fresh = random_matrix(4096, 4, rng, means, stds)
    output = nn.batch_norm(p, stats, "bn1", fresh, False, configuration(0.9))
    for c in range(4):
        values = [output.at([i, c]) for i in range(4096)]
        mean = sum(values) / 4096
        std = math.sqrt(sum(v * v for v in values) / 4096 - mean * mean)
        error = max(
            abs(v - (fresh.at([i, c]) - mm[c]) / math.sqrt(mv[c] + EPS))
            for i, v in enumerate(values)
        )
        assert error < 1e-5
        if steps == 300:
            assert (
                abs(mean - [-0.041, -0.021, 0.028, 0.002][c]) < 2e-3
                and abs(std - [0.963, 0.999, 1.027, 1.046][c]) < 2e-3
            )
        print(
            "Inference channel",
            c,
            "mean/std:",
            mean,
            std,
            "moving-stat hand error:",
            error,
        )


def deep_stack():
    depth, width, batch = 24, 256, 128
    for use_bn in [False, True]:
        p, stats = nn.ParameterStore(3), nn.ParameterStore()
        rng = nn.Mt19937_64(4)
        h = random_matrix(batch, width, rng, [0] * width, [1] * width)
        acts = []
        for layer in range(depth):
            w = p.get(
                "l" + str(layer) + "/W",
                [width, width],
                nn.init.normal(0.5 / math.sqrt(width)),
            )
            z = nn.matmul(h, w)
            if use_bn:
                z = nn.batch_norm(
                    p, stats, "bn" + str(layer), z, True, configuration(0.99)
                )
            h = nn.relu(z)
            acts.append(h)
        gradient = nn.grad(nn.mean(nn.square(h)), [p.at("l0/W")])[0]
        norm = nn.sqrt(nn.sum(nn.square(gradient))).item()
        scales = []
        for i in [1, 4, 8, 12, 16, 20, 24]:
            values = list(acts[i - 1].values())
            scales.append(math.sqrt(sum(v * v for v in values) / (batch * width)))
        if use_bn:
            assert abs(scales[-1] - 0.70) < 0.05 and 1 < norm < 100
        else:
            assert scales[-1] < 1e-8 and norm < 1e-12
        print(
            "With BN" if use_bn else "Without BN",
            "RMS at layers 1,4,8,12,16,20,24:",
            scales,
            "first-weight gradient norm:",
            norm,
        )


def backward():
    p, stats = nn.ParameterStore(), nn.ParameterStore()
    rng = nn.Mt19937_64(5)
    n, c = 6, 3
    x = random_matrix(n, c, rng, [0, 1, -1], [1, 2, 0.5]).requires_grad_()
    r = random_matrix(n, c, rng, [0, 0, 0], [1, 1, 1])

    def loss(value):
        return nn.sum(
            nn.mul(nn.batch_norm(p, stats, "bn1", value, True, configuration(0.9)), r)
        )

    objective = loss(x)
    gx, gg, gb = nn.grad(objective, [x, p.at("bn1/gamma"), p.at("bn1/beta")])
    ex = eg = eb = 0
    for col in range(c):
        values = [x.at([i, col]) for i in range(n)]
        dy = [r.at([i, col]) for i in range(n)]
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        sigma = math.sqrt(variance + EPS)
        xhat = [(v - mean) / sigma for v in values]
        sdy = sum(dy)
        sdyx = sum(a * b for a, b in zip(dy, xhat))
        for i in range(n):
            ex = max(
                ex,
                abs((n * dy[i] - sdy - xhat[i] * sdyx) / (n * sigma) - gx.at([i, col])),
            )
        eg = max(eg, abs(sdyx - gg.at([col])))
        eb = max(eb, abs(sdy - gb.at([col])))
    values = list(x.values())
    dx = list(gx.values())
    eps = nn.scalar(1e-2).item()
    fd = 0
    with nn.no_grad():
        for i in range(n * c):
            plus = values.copy()
            minus = values.copy()
            plus[i] += eps
            minus[i] -= eps
            numeric = (
                loss(nn.from_vector(plus, [n, c])).item()
                - loss(nn.from_vector(minus, [n, c])).item()
            ) / (2 * eps)
            fd = max(fd, abs(numeric - dx[i]))
    gmax = max(abs(v) for v in dx)
    sums = [sum(gx.at([i, col]) for i in range(n)) for col in range(c)]
    assert (
        max(ex, eg, eb) < 1e-5
        and fd / gmax < 1e-3
        and abs(gmax - 2.40) < 0.01
        and max(abs(v) for v in sums) < 1e-5
    )
    print(
        "Backward errors dx/gamma/beta:",
        ex,
        eg,
        eb,
        "finite difference:",
        fd,
        "relative:",
        fd / gmax,
        "gradient max:",
        gmax,
        "column sums:",
        sums,
    )


def main(arguments=None):
    args = options(arguments, dict(epochs=5, full=False))
    if args["epochs"] < 1:
        raise ValueError("epochs must be positive")
    nn.set_num_threads(2)
    hand_check()
    moving_statistics(args["epochs"])
    deep_stack()
    backward()
    print("All batch normalization checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
