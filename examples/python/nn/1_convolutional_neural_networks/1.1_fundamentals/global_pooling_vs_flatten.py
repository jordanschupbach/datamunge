# Native Python NN example. Global average pooling vs flatten heads, written out with datamunge::nn ops.
"""CMD: just python-example global_pooling_vs_flatten
Global average pooling vs flatten: invariance, location and learned heads.

Port of tflow's global_pooling_vs_flatten. For [N,H,W,C] feature maps and K
classes, a GAP-linear head learns C*K+K parameters while flatten-linear
learns H*W*C*K+K. GAP discards spatial arrangement: every permutation of a
map's spatial sites has the same logits. Verify uniform spatial input
gradients and the class-activation-map identity
logit_k = mean_hw(sum_c(feature_hwc * W_ck)) + bias_k.
These identities apply to the feature map. Padding and strided layers can
break translation equivariance of a preceding CNN, so image-level shift
invariance does not follow automatically.

Two controlled, head-only learning tasks use 8x8 two-channel feature maps.
Appearance: the active channel is the label; fit half the spatial sites and
test the other half. GAP shares evidence across space, while a zero-initialized
flatten head receives no gradient at unseen sites. Location: the label is
left/right and the active channel is fixed. Fit all sites and test a new
positive amplitude; GAP's two classes are indistinguishable, while flatten
retains location. These exact toy tasks isolate head behavior; they are not
an image benchmark or a comparison of jointly trained feature extractors.
Both heads use Adam(.05), the same examples and 300 full-batch steps
(mean sparse cross-entropy, nn.Optimizer Adam over a ParameterStore).
Each head's 2x2 test confusion matrix (truth rows) is printed.
"""
import math
import os
import struct

from pydatamunge import nn
from pydatamunge.nn_workflows import helper, options

checks = helper("checks")


def float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def g(value):
    return f"{value:.6g}"


def algebra_checks():
    weights = nn.from_vector([.4, -.7, .2, .3, -.5, .6], [3, 2])
    bias = nn.from_vector([.1, -.2], [2])
    # std::sin(int(i) * .17f): the product is a float32.
    values = [math.sin(float32(i * float32(.17))) for i in range(2 * 4 * 5 * 3)]
    x = nn.from_vector(values, [2, 4, 5, 3])
    x.requires_grad_()
    logits = nn.bias_add(nn.matmul(nn.global_avg_pool2d(x), weights), bias)
    maps = nn.reshape(nn.matmul(nn.reshape(x, [-1, 3]), weights), [2, 4, 5, 2])
    from_maps = nn.bias_add(nn.global_avg_pool2d(maps), bias)
    # Reversing both spatial axes is a nontrivial permutation that keeps channel identity.
    permuted = nn.bias_add(nn.matmul(nn.global_avg_pool2d(nn.reverse(x, [1, 2])), weights), bias)
    objective = nn.sum(nn.slice_axis(logits, 1, 0, 1))
    gradient = nn.grad(objective, [x])[0].values()
    a, b, c = logits.values(), from_maps.values(), permuted.values()
    for i in range(4):
        checks.near(a[i], b[i], 1e-7, "class-activation-map identity")
        checks.near(a[i], c[i], 1e-7, "spatial permutation invariance")
    w = weights.values()
    for i, value in enumerate(gradient):
        checks.near(value, w[(i % 3) * 2] / 20, 1e-8, "uniform spatial GAP gradient")
    print("  algebra: spatial permutation, class-activation maps and uniform input derivatives passed")


def head(parameters, inputs, global_average):
    width = 2 if global_average else 128
    features = nn.global_avg_pool2d(nn.reshape(inputs, [-1, 8, 8, 2])) if global_average else inputs
    weights = parameters.get("head/W", [width, 2], nn.init.zeros())
    bias = parameters.get("head/b", [2], nn.init.zeros())
    return nn.bias_add(nn.matmul(features, weights), bias)


def dataset(location, test):
    """An impulse of amplitude 64 gives one unit in the global average. Test
    location maps use amplitude 80; appearance uses exactly unseen sites."""
    examples = []
    for y in range(8):
        for x in range(8):
            if location:
                examples.append((y, x, 0, int(x >= 4)))
            elif (y + x) % 2 == int(test):
                for label in range(2):
                    examples.append((y, x, label, label))
    features = [0.0] * (len(examples) * 128)
    labels = []
    for i, (y, x, channel, label) in enumerate(examples):
        features[i * 128 + (y * 8 + x) * 2 + channel] = 80.0 if location and test else 64.0
        labels.append(label)
    n = len(examples)
    return nn.from_vector(features, [n, 128]), nn.from_vector_int64(labels, [n])


def learn_heads():
    for location in (False, True):
        train_x, train_y = dataset(location, False)
        test_x, test_y = dataset(location, True)
        # Explicitly prove that location labels have identical GAP features.
        if location:
            values = train_x.values()
            for row in range(int(train_x.shape()[0])):
                channel_zero = sum(values[row * 128 + site * 2] for site in range(64))
                channel_one = sum(values[row * 128 + site * 2 + 1] for site in range(64))
                checks.near(channel_zero, 64, 0, "location examples share GAP feature zero")
                checks.near(channel_one, 0, 0, "location examples share GAP feature one")
        for global_average in (False, True):
            model = nn.ParameterStore(13)
            with nn.no_grad():
                head(model, train_x, global_average)
            checks.require(model.count() == (6 if global_average else 258), "GAP vs flatten head parameter count")
            config = nn.OptimizerConfig()
            config.kind = nn.OptimizerKind.Adam
            optimizer = nn.Optimizer(model, .05, config)
            for step in range(300):
                loss = nn.sparse_softmax_cross_entropy(head(model, train_x, global_average), train_y)
                optimizer.minimize(loss)
            with nn.no_grad():
                def evaluate(x, y):
                    logits = head(model, x, global_average)
                    return nn.sparse_softmax_cross_entropy(logits, y).item(), nn.accuracy(logits, y)

                training, test = evaluate(train_x, train_y), evaluate(test_x, test_y)
                expected = (.5 if global_average else 1.) if location else (1. if global_average else .5)
                checks.near(test[1], expected, 0, "controlled head generalization result")
                if global_average and location:
                    checks.near(test[0], math.log(2.), 1e-6, "indistinguishable balanced classes have log(2) loss")
                else:
                    checks.near(training[1], 1, 0, "learnable training task fitted")
                if not global_average and not location:
                    weights = model.at("head/W").values()
                    for y in range(8):
                        for x in range(8):
                            if (y + x) % 2 == 1:
                                for channel in range(2):
                                    for label in range(2):
                                        checks.near(weights[((y * 8 + x) * 2 + channel) * 2 + label], 0, 0,
                                                    "unseen flatten coordinates receive no gradient")
                print(f"  {'location' if location else 'appearance'} {'GAP' if global_average else 'flatten'}: "
                      f"parameters={model.count()}; training accuracy={g(training[1])}; "
                      f"test accuracy={g(test[1])}; test NLL={g(test[0])}")
                confusion = nn.ConfusionMatrix(2)
                confusion.add(test_y, nn.argmax(head(model, test_x, global_average), 1))
                cells = [int(v) for v in confusion.to_tensor().values()]
                checks.near(confusion.accuracy(), test[1], 1e-12, "confusion matches test accuracy")
                print(f"    test confusion (truth rows): [{cells[0]} {cells[1]}; {cells[2]} {cells[3]}]")


def main(arguments=None):
    options(arguments, dict(full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    algebra_checks()
    learn_heads()
    print("All global-pooling vs flatten checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
