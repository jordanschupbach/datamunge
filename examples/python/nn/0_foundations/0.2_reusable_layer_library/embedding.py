# Native Python NN example. Embedding lookup, sparse gradients, masking and skip-gram.
"""Train context-word embeddings directly with a final confusion matrix."""
import math
from pydatamunge import nn
from pydatamunge.nn_workflows import options


def lookup_and_gradient():
    p = nn.ParameterStore(1)
    ids = nn.from_vector_int64([3, 7, 3, 0, 9, 7], [2, 3])
    r = nn.from_vector([0.5 * (i % 5) - 1 for i in range(24)], [2, 3, 4])
    e = nn.embedding(p, "words", ids, 10, 4, 1.0)
    table = p.at("words/table")
    onehot = nn.reshape(
        nn.matmul(nn.one_hot(nn.reshape(ids, [-1]), 10), table), [2, 3, 4]
    )
    gradient = nn.grad(nn.sum(nn.mul(e, r)), [table])[0]
    assert tuple(e.shape()) == (2, 3, 4) and list(e.values()) == list(onehot.values())
    worst = 0
    for word in range(10):
        expected = [
            sum(
                r.at([i, j, k])
                for i in range(2)
                for j in range(3)
                if ids.at([i, j]) == word
            )
            for k in range(4)
        ]
        actual = [gradient.at([word, k]) for k in range(4)]
        worst = max(worst, max(abs(a - b) for a, b in zip(actual, expected)))
        print("Gradient row", word, actual, "expected:", expected)
    assert worst < 1e-6
    print(
        "Lookup equals one-hot product; repeated ids accumulate; unused rows have zero gradient."
    )


def masking():
    p = nn.ParameterStore(2)
    ids = nn.from_vector_int64([4, 2, 0, 0, 1, 5, 9, 3], [2, 4])
    e = nn.embedding(p, "words", ids, 10, 3, 1.0)
    pooled = nn.masked_mean(e, nn.not_equal(ids, 0))
    naive = nn.mean(e, [1])
    worst = 0
    for i in range(2):
        positions = [j for j in range(4) if ids.at([i, j]) != 0]
        expected = [
            sum(e.at([i, j, k]) for j in positions) / len(positions) for k in range(3)
        ]
        actual = [pooled.at([i, k]) for k in range(3)]
        worst = max(worst, max(abs(a - b) for a, b in zip(actual, expected)))
        print(
            "Sequence",
            i,
            "real tokens:",
            len(positions),
            "masked:",
            actual,
            "hand:",
            expected,
            "unmasked:",
            [naive.at([i, k]) for k in range(3)],
        )
    assert (
        worst < 1e-6
        and max(
            abs(v)
            for v in nn.sub(nn.select(pooled, 0, 1), nn.select(naive, 0, 1)).values()
        )
        < 1e-6
    )


def skip_gram(steps):
    vocab, dim, window = 12, 8, 2
    rng = nn.Mt19937_64(3)
    pairs = []
    for _ in range(400):
        topic = rng.modulo(3)
        sentence = [4 * topic + rng.modulo(4) for _ in range(8)]
        for i in range(8):
            for j in range(max(0, i - window), min(7, i + window) + 1):
                if i != j:
                    pairs.append((sentence[i], sentence[j]))
    assert len(pairs) == 10400
    print("Synthetic corpus:", len(pairs), "target/context pairs in three topics.")
    p = nn.ParameterStore(4)

    def logits(target):
        e = nn.embedding(p, "in", target, vocab, dim, 0.5)
        out = p.get("out/table", [vocab, dim], nn.init.uniform(0.5))
        return nn.matmul(e, out, False, True)

    with nn.no_grad():
        logits(nn.from_vector_int64([0] * 256))
    cfg = nn.OptimizerConfig()
    cfg.kind = nn.OptimizerKind.Adam
    adam = nn.Optimizer(p, 0.05, cfg)

    def evaluate():
        total = 0.0
        cm = nn.ConfusionMatrix(vocab)
        with nn.no_grad():
            for start in range(0, len(pairs), 512):
                batch = pairs[start : start + 512]
                target = nn.from_vector_int64([a for a, b in batch])
                context = nn.from_vector_int64([b for a, b in batch])
                scores = logits(target)
                total += (
                    len(batch) * nn.sparse_softmax_cross_entropy(scores, context).item()
                )
                cm.add(context, nn.argmax(scores, 1))
        return total / len(pairs), cm

    initial, _ = evaluate()
    last = math.inf
    for step in range(1, steps + 1):
        batch = [pairs[rng.modulo(len(pairs))] for _ in range(256)]
        target = nn.from_vector_int64([a for a, b in batch])
        context = nn.from_vector_int64([b for a, b in batch])
        loss = nn.sparse_softmax_cross_entropy(logits(target), context)
        adam.minimize(loss)
        last = loss.item()
        print(
            f"skip-gram step {step}/{steps}: NLL={last:.9g}; floor log(4)={math.log(4):.9g}"
        )
    table = p.at("in/table")
    within = []
    between = []
    print("Cosine similarity (words 0-11, topics 0-3 / 4-7 / 8-11):")
    for a in range(vocab):
        row = []
        for b in range(vocab):
            av = [table.at([a, k]) for k in range(dim)]
            bv = [table.at([b, k]) for k in range(dim)]
            cosine = sum(x * y for x, y in zip(av, bv)) / math.sqrt(
                sum(x * x for x in av) * sum(y * y for y in bv)
            )
            row.append(cosine)
            if a != b:
                (within if a // 4 == b // 4 else between).append(cosine)
        print(a, *[f"{v:.6f}" for v in row])
    mean_within, mean_between = sum(within) / len(within), sum(between) / len(between)
    print("Mean cosine within topic:", mean_within, "between topics:", mean_between)
    if steps >= 300:
        assert last < math.log(4) + 0.06 and mean_within > 0.8 and mean_between < -0.2
    final, cm = evaluate()
    assert math.isfinite(final) and final < initial and adam.iterations() == steps
    cells = list(cm.to_tensor().values())
    assert sum(cells) == len(pairs)
    print(f"Context-word evaluation NLL={final:.9g}; accuracy={cm.accuracy():.9g}")
    print(
        "Confusion matrix (rows=true context, columns=predicted context; synthetic corpus):"
    )
    for row in range(vocab):
        print(*[int(v) for v in cells[row * vocab : (row + 1) * vocab]])


def main(arguments=None):
    args = options(arguments, dict(epochs=5, full=False))
    if args["epochs"] < 1:
        raise ValueError("epochs must be positive")
    nn.set_num_threads(2)
    lookup_and_gradient()
    masking()
    skip_gram(args["epochs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
