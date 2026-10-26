# Native Python NN example. Binary and multinomial classification, including confusion matrices.
"""Stable logistic losses, explicit Adam updates and held-out evaluation."""
import math
from pydatamunge import nn
from pydatamunge.nn_workflows import options


def binary_cross_entropy(logits, labels):
    positive = nn.greater_equal(logits, 0)
    magnitude = nn.where(positive, nn.neg(logits), logits)
    linear = nn.where(
        positive, nn.mul(logits, nn.sub(1, labels)), nn.neg(nn.mul(logits, labels))
    )
    return nn.add(linear, nn.log1p(nn.exp(magnitude)))


class Adam:
    def __init__(self, parameters, rate=0.04):
        self.parameters, self.rate, self.t = parameters, rate, 0
        self.m = [nn.zeros_like(p) for p in parameters]
        self.v = [nn.zeros_like(p) for p in parameters]

    def step(self, gradients):
        self.t += 1
        c1, c2 = 1 - 0.9**self.t, 1 - 0.999**self.t
        with nn.no_grad():
            for i, (p, g) in enumerate(zip(self.parameters, gradients)):
                self.m[i] = nn.add(nn.mul(self.m[i], 0.9), nn.mul(g, 0.1))
                self.v[i] = nn.add(
                    nn.mul(self.v[i], 0.999), nn.mul(nn.square(g), 0.001)
                )
                update = nn.div(
                    nn.div(self.m[i], c1), nn.add(nn.sqrt(nn.div(self.v[i], c2)), 1e-8)
                )
                p.sub_(update, self.rate)


def print_confusion(labels, prediction, classes):
    confusion = nn.ConfusionMatrix(classes)
    confusion.add(labels, prediction)
    cells = list(confusion.to_tensor().values())
    assert sum(cells) == labels.numel()
    print("Confusion matrix (rows=truth, columns=prediction):")
    for row in range(classes):
        print(" ", *[int(v) for v in cells[row * classes : (row + 1) * classes]])
    return confusion.accuracy()


def binary(steps):
    extreme = nn.from_vector([-1000, -1000, 1000, 1000, 0, 0]).requires_grad_()
    labels = nn.from_vector([0, 1, 0, 1, 0, 1])
    stable = binary_cross_entropy(extreme, labels)
    derivative = nn.grad(nn.sum(stable), [extreme])[0]
    for actual, expected in zip(
        stable.values(), [0, 1000, 1000, 0, math.log(2), math.log(2)]
    ):
        assert abs(actual - expected) < 1e-6
    assert list(derivative.values()) == [0, -1, 1, 0, 0.5, -0.5]
    rng = nn.Mt19937(31)

    def dataset(n):
        features, targets = [], []
        for _ in range(n):
            a, b = rng.normal32(), rng.normal32()
            chance = 1 / (1 + math.exp(-(-0.4 + 1.5 * a - 2 * b)))
            features.extend([1, a, b])
            targets.append(float(rng.uniform32() < chance))
        return nn.from_vector(features, [n, 3]), nn.from_vector(targets, [n, 1])

    train_x, train_y = dataset(1200)
    test_x, test_y = dataset(600)
    w = nn.zeros([3, 1]).requires_grad_()
    mask = nn.from_vector([0, 1, 1], [3, 1])

    def objective(x, y):
        loss = nn.mean(binary_cross_entropy(nn.matmul(x, w), y), [0, 1])
        return loss, nn.add(
            loss, nn.mul(nn.sum(nn.mul(mask, nn.square(w)), [0, 1]), 0.005)
        )

    opt = Adam([w])
    initial = objective(train_x, train_y)[1].item()
    for epoch in range(1, steps + 1):
        loss = objective(train_x, train_y)[1]
        opt.step(nn.grad(loss, [w]))
        print(f"binary epoch {epoch}/{steps}: objective={loss.item():.9g}")
    assert objective(train_x, train_y)[1].item() < initial
    stationarity = nn.grad(objective(train_x, train_y)[1], [w])[0]
    if steps >= 400:
        assert max(abs(v) for v in stationarity.values()) < 1e-5
    with nn.no_grad():
        probability = nn.sigmoid(nn.matmul(test_x, w))
        tx = list(test_x.values())
        calibration = (
            sum(
                abs(
                    p
                    - 1
                    / (1 + math.exp(-(-0.4 + 1.5 * tx[i * 3 + 1] - 2 * tx[i * 3 + 2])))
                )
                for i, p in enumerate(probability.values())
            )
            / 600
        )
        loss = objective(test_x, test_y)[0].item()
        prediction = nn.reshape(
            nn.cast(nn.greater_equal(probability, 0.5), nn.DType.Int64), [-1]
        )
        truth = nn.reshape(nn.cast(test_y, nn.DType.Int64), [-1])
        accuracy = nn.accuracy(prediction, truth)
        print(
            f"Binary NLL={loss:.9g}; accuracy={accuracy:.9g}; calibration error={calibration:.9g}; weights={w.str()}"
        )
        if steps >= 400:
            assert loss < 0.5 and accuracy > 0.75 and calibration < 0.06
        print_confusion(truth, prediction, 2)


def multinomial(steps):
    rng = nn.Mt19937(47)

    def dataset(n):
        features, labels = [], []
        for i in range(n):
            c = i % 3
            angle = 2 * math.pi * c / 3
            features.extend(
                [
                    2 * math.cos(angle) + rng.normal32(0, 0.55),
                    2 * math.sin(angle) + rng.normal32(0, 0.55),
                ]
            )
            labels.append(c)
        return nn.from_vector(features, [n, 2]), nn.from_vector_int64(labels)

    train_x, train_y = dataset(450)
    test_x, test_y = dataset(300)
    limit = math.sqrt(6 / 5)
    w = nn.uniform([2, 3], -limit, limit, nn.Generator(19)).requires_grad_()
    b = nn.zeros([3]).requires_grad_()

    def nll(x, y):
        log_probs = nn.log_softmax(nn.add(nn.matmul(x, w), b), 1)
        return nn.neg(nn.mean(nn.sum(nn.mul(nn.one_hot(y, 3), log_probs), [1]), [0]))

    opt = Adam([w, b])
    initial = nll(train_x, train_y).item()
    for epoch in range(1, steps + 1):
        loss = nn.add(nll(train_x, train_y), nn.mul(nn.sum(nn.square(w)), 0.0005))
        opt.step(nn.grad(loss, [w, b]))
        print(f"multinomial epoch {epoch}/{steps}: objective={loss.item():.9g}")
    assert nll(train_x, train_y).item() < initial
    with nn.no_grad():
        logits = nn.add(nn.matmul(test_x, w), b)
        probabilities = nn.softmax(logits, 1)
        assert max(abs(v - 1) for v in nn.sum(probabilities, [1]).values()) < 1e-6
        loss = nll(test_x, test_y).item()
        accuracy = nn.accuracy(logits, test_y)
        if steps >= 300:
            assert accuracy > 0.97 and loss < 0.1
        print(f"Multinomial NLL={loss:.9g}; accuracy={accuracy:.9g}")
        print_confusion(test_y, nn.argmax(logits, 1), 3)


def main(arguments=None):
    args = options(arguments, dict(epochs=5, full=False))
    if args["epochs"] < 1:
        raise ValueError("epochs must be positive")
    binary(args["epochs"])
    multinomial(args["epochs"])
    print("All logistic regression checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
