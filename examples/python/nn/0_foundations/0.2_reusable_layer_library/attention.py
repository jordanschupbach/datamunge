# Native Python NN example. Attention references, scaling, gradients and lookup training.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/0_foundations/0.2_reusable_layer_library/attention.py
"""Additive, scaled dot-product and multi-head attention through the bound API."""
import math
from pydatamunge import nn
from pydatamunge.nn_workflows import options


def random_tensor(shape, rng):
    rng.reset_normal()
    return nn.from_vector(rng.normal32_values(math.prod(shape)), shape)


def rows(tensor, example=0):
    return [
        [tensor.at([example, i, j]) for j in range(tensor.size(-1))]
        for i in range(tensor.size(-2))
    ]


def softmax(values):
    maximum = max(values)
    exponent = [math.exp(v - maximum) for v in values]
    total = sum(exponent)
    return [v / total for v in exponent]


def attend_reference(q, k, v, causal=False):
    weights = []
    output = []
    for i, query in enumerate(q):
        scores = [
            sum(a * b for a, b in zip(query, key)) / math.sqrt(len(query)) for key in k
        ]
        if causal:
            scores = [score if j <= i else -1e9 for j, score in enumerate(scores)]
        weight = softmax(scores)
        weights.append(weight)
        output.append(
            [sum(weight[j] * v[j][a] for j in range(len(k))) for a in range(len(v[0]))]
        )
    return weights, output


def dot_product():
    rng = nn.Mt19937_64(1)
    q, k, v = (
        random_tensor([1, 4, 8], rng),
        random_tensor([1, 4, 8], rng),
        random_tensor([1, 4, 3], rng),
    )
    plain = nn.scaled_dot_product_attention(q, k, v, nn.Tensor())
    causal = nn.scaled_dot_product_attention(q, k, v, nn.causal_mask(4))
    for is_causal, result in [(False, plain), (True, causal)]:
        weights, output = attend_reference(rows(q), rows(k), rows(v), is_causal)
        dw = max(
            abs(weights[i][j] - result.weights.at([0, i, j]))
            for i in range(4)
            for j in range(4)
        )
        dout = max(
            abs(output[i][j] - result.output.at([0, i, j]))
            for i in range(4)
            for j in range(3)
        )
        assert dw < 1e-6 and dout < 1e-6
        print("Causal" if is_causal else "Plain", "attention hand errors:", dw, dout)
        for i in range(4):
            row = [result.weights.at([0, i, j]) for j in range(4)]
            assert abs(sum(row) - 1) < 1e-5
            print("Query", i, "weights:", row)
    config = nn.AttentionOptions()
    config.impl = nn.AttentionImpl.Fused
    config.causal = True
    fused = nn.scaled_dot_product_attention(q, k, v, config)
    error = max(
        abs(a - b) for a, b in zip(fused.output.values(), causal.output.values())
    )
    assert error < 1e-6
    print("Fused causal output difference:", error)


def scaling():
    rng = nn.Mt19937_64(2)
    trials = 200
    for width in [4, 16, 64, 256, 1024]:
        raw = scaled = entropy_raw = entropy_scaled = 0.0
        for _ in range(trials):
            # Draw q, then all 64 keys, preserving the original scalar stream.
            data = nn.from_vector_double(rng.normal64_values(65 * width), [65, width])
            query = nn.slice_axis(data, 0, 0, 1)
            keys = nn.slice_axis(data, 0, 1, 65)
            scores = list(nn.matmul(keys, query, False, True).values())
            for score in scores:
                raw += score * score / (64 * trials)
                scaled += score * score / width / (64 * trials)
            for scale in [1, 1 / math.sqrt(width)]:
                probabilities = softmax([s * scale for s in scores])
                entropy = -sum(p * math.log(p) for p in probabilities if p > 0) / trials
                if scale == 1:
                    entropy_raw += entropy
                else:
                    entropy_scaled += entropy
        assert abs(scaled - 1) < 0.1
        print(
            "Scaling d=",
            width,
            "raw/scaled variance, raw/scaled entropy:",
            raw,
            scaled,
            entropy_raw,
            entropy_scaled,
        )


def additive():
    p = nn.ParameterStore(3)
    rng = nn.Mt19937_64(3)
    q, k, v = (
        random_tensor([2, 5], rng),
        random_tensor([2, 6, 3], rng),
        random_tensor([2, 6, 2], rng),
    )
    result = nn.additive_attention(p, "add", q, k, v, 4)
    wq, wk, vector = p.at("add/Wq"), p.at("add/Wk"), p.at("add/v")
    dw = dout = 0
    for n in range(2):
        scores = []
        for t in range(6):
            score = 0
            for u in range(4):
                h = sum(q.at([n, i]) * wq.at([i, u]) for i in range(5))
                h += sum(k.at([n, t, i]) * wk.at([i, u]) for i in range(3))
                score += math.tanh(h) * vector.at([u, 0])
            scores.append(score)
        weights = softmax(scores)
        dw = max(
            dw, max(abs(w - result.weights.at([n, t])) for t, w in enumerate(weights))
        )
        for i in range(2):
            dout = max(
                dout,
                abs(
                    sum(weights[t] * v.at([n, t, i]) for t in range(6))
                    - result.output.at([n, i])
                ),
            )
    assert dw < 1e-6 and dout < 1e-6 and p.count() == 36
    assert tuple(result.weights.shape()) == (2, 6) and tuple(result.output.shape()) == (
        2,
        2,
    )
    print("Additive attention hand errors:", dw, dout, "parameters:", p.count())


def multi_head():
    p = nn.ParameterStore(4)
    rng = nn.Mt19937_64(4)
    x = random_tensor([2, 5, 16], rng)
    config = nn.MultiHeadOptions()
    config.need_weights = True
    result = nn.multi_head_attention(p, "self", x, x, 4, config)
    assert (
        tuple(result.output.shape()) == (2, 5, 16)
        and tuple(result.weights.shape()) == (2, 4, 5, 5)
        and p.count() == 1024
    )
    wq, wk, wv, wo = [p.at("self/" + name) for name in ["Wq", "Wk", "Wv", "Wo"]]
    error = 0
    for n in range(2):
        concatenated = [[0.0] * 16 for _ in range(5)]
        for h in range(4):
            projected = []
            for weight in [wq, wk, wv]:
                projected.append(
                    [
                        [
                            sum(
                                x.at([n, t, i]) * weight.at([i, 4 * h + j])
                                for i in range(16)
                            )
                            for j in range(4)
                        ]
                        for t in range(5)
                    ]
                )
            _, output = attend_reference(*projected)
            for t in range(5):
                concatenated[t][4 * h : 4 * h + 4] = output[t]
        for t in range(5):
            for j in range(16):
                error = max(
                    error,
                    abs(
                        sum(concatenated[t][i] * wo.at([i, j]) for i in range(16))
                        - result.output.at([n, t, j])
                    ),
                )
    assert error < 1e-5
    print("Multi-head direct per-head error:", error, "parameters:", p.count())
    for head in range(4):
        print(
            "Head",
            head,
            "example-0/query-2 weights:",
            [result.weights.at([0, head, 2, j]) for j in range(5)],
        )
    scores = random_tensor([5], rng).requires_grad_()
    r = random_tensor([5], rng)
    alpha = nn.softmax(scores)
    gradient = nn.grad(nn.sum(nn.mul(alpha, r)), [scores])[0]
    dot = sum(a * b for a, b in zip(alpha.values(), r.values()))
    error = max(
        abs(a * (b - dot) - g)
        for a, b, g in zip(alpha.values(), r.values(), gradient.values())
    )
    assert error < 1e-6
    print("Softmax closed-form gradient error:", error)


def lookup(steps):
    length, ids, width, batch = 6, 10, 8, 64
    p = nn.ParameterStore(5)
    table = p.get("emb", [ids, width], nn.init.uniform(0.5))
    cfg = nn.OptimizerConfig()
    cfg.kind = nn.OptimizerKind.Adam
    adam = nn.Optimizer(p, 0.02, cfg)
    rng = nn.Mt19937_64(5)

    def make(n):
        keys = []
        queries = []
        values = []
        targets = []
        for _ in range(n):
            permutation = list(rng.shuffle(list(range(ids))))
            which = rng.modulo(length)
            value = list(rng.normal32_values(length))
            keys.extend(permutation[:length])
            values.extend(value)
            queries.append(permutation[which])
            targets.append(value[which])
        return dict(
            keys=nn.from_vector_int64(keys, [n, length]),
            query=nn.from_vector_int64(queries),
            values=nn.from_vector(values, [n, length]),
            target=nn.from_vector(targets),
        )

    def forward(data):
        k = nn.gather(table, data["keys"], 0)
        q = nn.expand_dims(nn.gather(table, data["query"], 0), 1)
        result = nn.scaled_dot_product_attention(
            q, k, nn.expand_dims(data["values"], -1), nn.Tensor()
        )
        prediction = nn.reshape(result.output, [-1])
        return nn.mean(nn.square(nn.sub(prediction, data["target"]))), result.weights

    test = make(512)
    final_weight = final_mse = 0
    for step in range(1, steps + 1):
        data = make(batch)
        adam.step(nn.grad(forward(data)[0], [table]))
        if step == 1 or step % 100 == 0 or step == steps:
            with nn.no_grad():
                loss, weights = forward(test)
                on_target = 0
                for i in range(512):
                    for t in range(length):
                        if test["keys"].at([i, t]) == test["query"].at([i]):
                            on_target += weights.at([i, 0, t]) / 512
                final_weight, final_mse = on_target, loss.item()
                print(
                    f"Lookup step {step}: test MSE={final_mse:.9g}, matching-key attention={on_target:.9g}"
                )
    assert (
        math.isfinite(final_mse)
        and final_mse >= 0
        and 0 < final_weight <= 1
        and adam.iterations() == steps
    )
    if steps >= 600:
        assert final_mse < 0.005 and final_weight > 0.95
    with nn.no_grad():
        _, weights = forward(test)
        cm = nn.ConfusionMatrix(length)
        for i in range(512):
            truth = next(
                t
                for t in range(length)
                if test["keys"].at([i, t]) == test["query"].at([i])
            )
            prediction = max(range(length), key=lambda t: weights.at([i, 0, t]))
            cm.add(truth, prediction)
        cells = list(cm.to_tensor().values())
        assert sum(cells) == 512
        print("Pointer accuracy:", cm.accuracy())
        print(
            "Confusion matrix (rows=matching key position, columns=predicted position; held-out 512 queries):"
        )
        for row in range(length):
            print(*[int(v) for v in cells[row * length : (row + 1) * length]])


def main(arguments=None):
    args = options(arguments, dict(epochs=5, full=False))
    if args["epochs"] < 1:
        raise ValueError("epochs must be positive")
    nn.set_num_threads(2)
    dot_product()
    scaling()
    additive()
    multi_head()
    lookup(args["epochs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
