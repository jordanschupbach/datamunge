# Native Python NN example. Kim CNN-rand sentence classification with Adadelta written out in Python.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.9_1d_convolutions/kim_sentence_cnn.py
"""Kim2014 CNN-rand: learned embeddings,parallel3/4/5-ReLU word convolutions,
max-over-time pooling,dropout.5,softmax and Adadelta with head maxnorm3.
Real pinned MR v1.0 sentences;fit-only vocabulary,duplicate-group holdout.
Full architecture and independent every-coordinate host gradient fixtures.
Three compact default epochs;untrained/majority/fit-only unigram NB controls.

Port of tflow's kim_sentence_cnn. Data: tflow's `just sentence-assets`
(output/datasets/sentence/sentence.npz, found via checks.dataset).
Deviations: checkpoints are .npz (best.npz); tflow's GraphDef placeholder
audit has no eager counterpart (the inference model is built without labels
or an optimizer instead). Weights initialise with nn's generator, so trained
numbers match TF statistically; they match the C++ port exactly.
Options: epochs=3 data=PATH checks_only=0|1 threads=4.
Output: <output root>/models/kim_sentence_cnn/.
"""
import math
import os
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_workflows import helper, options

checks = helper("checks")
replay = helper("replay")
kim = helper("kim_cnn")
sentence = helper("sentence")
fmt, f32, ints, int_tensor = sentence.fmt, sentence.f32, sentence.ints, sentence.int_tensor
# The C++ reference is built by clang with -ffp-contract=on: a*b+c inside one
# expression (and x += a*b) is a fused multiply-add. Host fixtures mirror that
# with math.fma so even their rounding noise matches the C++ log.
fma = math.fma


def train_sentence(root, epochs):
    checks.require(epochs > 0, "positive short sentiment epochs")
    (fit, validation, test), config = sentence.load_sentences(root)
    model = sentence.SentenceModel(config)
    directory = checks.output_dir("models/kim_sentence_cnn")
    prefix = directory / "best.npz"
    probe = sentence.sentence_rows(fit, sentence.range_rows(0, 50))
    initial = model.loss(probe)
    untrained = model.predict_all(test)
    print(f"CNN-rand compact:{model.parameters.count()} parameters/{model.parameters.size()}"
          f" learned tensors,18 Adadelta slots;E32,F32,windows3/4/5,dropout.5,maxnorm3;{epochs}"
          f" epochs,batch50,rho.95,eps1e-6,initial fixed CE={fmt(initial)}", flush=True)
    checks.require(model.optimizer.slots.size() == 18, "two Adadelta slots per learned tensor")
    probe_loss = sentence.sentence_ce(model.forward(probe, True).logits, probe.labels)
    sentence.require_live_gradients(model.parameters, nn.grad(probe_loss, model.parameters.parameters()),
                                    "embedding/CNN/head")
    print(" all9 actual training gradient tensors finite/nonzero")
    order = nn.BatchOrder(fit.size(), 50, 331)
    best, selected = math.inf, 0
    with (directory / "history.tsv").open("w") as history:
        history.write("epoch\ttrain_dropout_ce\tvalidation_ce\n")
        for epoch in range(1, epochs + 1):
            order.begin_epoch()
            total, seen = 0.0, 0
            while True:
                rows = list(order.next())
                if not rows:
                    break
                # Dropout training graph, CE, all gradients and one Adadelta/max-norm step.
                batch = sentence.sentence_rows(fit, rows)
                loss = sentence.sentence_ce(model.forward(batch, True).logits, batch.labels)
                model.optimizer.step(nn.grad(loss, model.parameters.parameters()))
                value = loss.item()
                checks.require(math.isfinite(value) and value >= 0, "finite sentence CE")
                total += len(rows) * value
                seen += len(rows)
            checks.require(seen == fit.size(), "each fitting sentence once per epoch")
            score = model.loss(validation)
            if score < best:
                best, selected = score, epoch
                model.save(prefix)
            history.write(f"{epoch}\t{fmt(total / seen)}\t{fmt(score)}\n")
            history.flush()
            print(f" epoch {epoch}:trainCE={fmt(total / seen)},validationCE={fmt(score)}", flush=True)
    model.restore(prefix)
    checks.near(model.loss(validation), best, 1e-6, "restored selected validation CE")
    learned = model.loss(probe)
    checks.require(learned < initial, "short-run fixed fitting sentiment CE improves")
    # The deployed model is a fresh inference-only model (no labels, no optimizer).
    deployed = sentence.SentenceModel(config, False)
    deployed.restore(prefix)
    checks.require(deployed.parameters.size() == 9 and deployed.optimizer is None,
                   "inference model holds only the nine learned tensors")
    before = deployed.snapshot()
    logits, reference, repeat = deployed.predict_all(test), model.predict_all(test), deployed.predict_all(test)
    checks.near(replay.max_abs_diff(logits, reference), 0, 1e-6, "training/inference independent restore parity")
    checks.require(replay.max_abs_diff(logits, repeat) == 0, "dropout-free deterministic inference")
    one = sentence.sentence_rows(test, [0])
    single = deployed.predict(one)
    checks.near(replay.max_abs_diff(single, nn.slice_axis(logits, 0, 0, 1)), 0, 1e-6, "single/batch sentence agreement")
    length = ints(one.lengths)[0]
    tokens = ints(one.tokens)
    for t in range(length, 64):
        tokens[t] = 1
    one.tokens = int_tensor(tokens, [1, 64])
    checks.require(replay.max_abs_diff(deployed.predict(one), single) == 0,
                   "legal out-of-length token substitution invariant")
    one = sentence.sentence_rows(test, [0])
    tokens = ints(one.tokens)
    for t in range(length):
        tokens[t] = 1
    one.tokens = int_tensor(tokens, [1, 64])
    sensitivity = nn.sum(nn.abs(nn.sub(nn.cast(single, nn.DType.Float64),
                                       nn.cast(deployed.predict(one), nn.DType.Float64)))).item()
    checks.require(sensitivity > 1e-6, "prediction depends on observed word IDs")
    checks.require(replay.same_bits(before, deployed.snapshot()), "inference leaves all nine learned tensors unchanged")
    head = deployed.parameters.at("classifier/W").values()
    for c in range(2):
        norm = 0.0
        for i in range(3 * config.filters):
            norm = fma(head[2 * i + c], head[2 * i + c], norm)
        checks.require(norm <= 9 + 1e-5, "classifier column norm bounded by3")
    nb = sentence.sentence_nb(fit, test, config.vocabulary)
    majority = nn.zeros([test.size(), 2])
    sentence.sentence_metrics("trained CNN-rand", test.labels, logits)
    sentence.sentence_metrics("untrained", test.labels, untrained)
    sentence.sentence_metrics("fit unigram multinomial NB", test.labels, nb)
    sentence.sentence_metrics("majority/tie class0", test.labels, majority)
    sentence.print_confusion("trained CNN-rand test",
                             sentence.confusion(test.labels, sentence.positive(logits), 2), 2)
    ids, labels = ints(test.source_ids), ints(test.labels)
    z = logits.values()
    p, q = sentence.positive(logits), sentence.positive(nb)
    with (directory / "predictions.tsv").open("w") as predictions:
        predictions.write("source_id\tlabel\tprediction\tnb_prediction\tnegative_logit\tpositive_logit\n")
        for n in range(test.size()):
            predictions.write(f"{ids[n]}\t{labels[n]}\t{p[n]}\t{q[n]}\t{fmt(z[2 * n])}\t{fmt(z[2 * n + 1])}\n")
    nn.save_npz(str(directory / "test_outputs.npz"),
                [("tokens", test.tokens), ("lengths", test.lengths), ("labels", test.labels),
                 ("source_ids", test.source_ids), ("logits", logits), ("untrained", untrained), ("nb_logits", nb)])
    sentence.copy_if_exists(Path(root) / "manifest.json", directory / "data_manifest.json")
    print(f" selected={selected},fixed fitting CE {fmt(initial)} -> {fmt(learned)},word substitution logit change="
          f"{fmt(sensitivity)};restore/padding/single-batch/determinism/immutable/maxnorm passed\n"
          f"Artifacts:{sentence.quoted(directory)}\n"
          "Custom MR holdout,CNN-rand,not pretrained or published cross-validation accuracy.")


# ---- independent fixtures -------------------------------------------------------------------

HOST_IDS = [[1, 2, 3, 6, 5], [2, 4, 1, 5, 3]]
HOST_LENGTHS = [3, 5]


def host(weights):
    """Wide (pad 4) 3/4/5 convolutions, ReLU through max(0, .), max-over-time and head in double."""
    result = [0.0] * 4
    for n in range(2):
        pooled = [0.0] * 6
        for branch in range(3):
            kernel = branch + 3
            for f in range(2):
                maximum = 0.0
                for t in range(5 + 8 - kernel + 1):
                    value = weights[2 + 2 * branch][f]
                    for k in range(kernel):
                        position = t + k - 4
                        if position < 0 or position >= HOST_LENGTHS[n]:
                            continue
                        for e in range(2):
                            value = fma(weights[0][(HOST_IDS[n][position] - 1) * 2 + e],
                                        weights[1 + 2 * branch][(k * 2 + e) * 2 + f], value)
                    maximum = max(maximum, value)
                pooled[2 * branch + f] = maximum
        for c in range(2):
            result[2 * n + c] = weights[8][c]
            for f in range(6):
                result[2 * n + c] = fma(pooled[f], weights[7][2 * f + c], result[2 * n + c])
    return result


def objective(weights):
    z = host(weights)
    return fma(.4, z[3], fma(-.1, z[2], fma(.3, z[0], -(.2 * z[1]))))


def host_checks():
    p = nn.ParameterStore(337)
    config = kim.KimConfig(7, 5, 2, 2)
    tokens = int_tensor([1, 2, 3, 6, 5, 2, 4, 1, 5, 3], [2, 5])
    lengths = int_tensor([3, 5], [2])
    weight = nn.from_vector([.3, -.2, -.1, .4], [2, 2])
    kim.kim_cnn(p, tokens, lengths, config)

    def value(v, i, shape):
        if len(shape) == 1 and v != 8:
            return f32(fma(f32(.03), f32(math.sin(i + v)), f32(.17)))
        return f32(.23 * math.sin(.7 * i + 1.3 * v + .2))

    weights = sentence.fix_parameters(p, value)
    net = kim.kim_cnn(p, tokens, lengths, config)
    gradients = nn.grad(nn.sum(nn.mul(net.logits, weight)), p.parameters())
    expected = host(weights)
    z = net.logits.values()
    value_error = 0.0
    for i in range(4):
        value_error = max(value_error, abs(expected[i] - z[i]))
    for e in nn.slice_axis(net.embedding, 0, 0, 1).values():
        checks.near(e, 0, 0, "structural PAD embedding exactly zero")
    gradient_error, coordinates = sentence.host_gradient_error(weights, gradients, objective)
    checks.require(coordinates == 80 and p.size() == 9, "all80 tiny parameter coordinates covered")
    checks.near(value_error, 0, 1e-7, "independent host wide convolution/ReLU/max/concat/head")
    checks.near(gradient_error, 0, 2e-6, "EVERY embedding/filter/bias/head derivative versus host FD")
    t = ints(tokens)
    t[3], t[4] = 4, 1
    tokens = int_tensor(t, [2, 5])
    out = kim.kim_cnn(p, tokens, lengths, config).logits.values()
    for c in range(2):
        checks.near(out[c], expected[c], 1e-7, "padding values cannot affect sentence features")
    t[5], t[9] = t[9], t[5]
    tokens = int_tensor(t, [2, 5])
    out = kim.kim_cnn(p, tokens, lengths, config).logits.values()
    difference = 0.0
    for c in range(2):
        difference += abs(out[2 + c] - expected[2 + c])
    checks.require(difference > 1e-5, "window features depend on word order")
    print(f" independent wide3/4/5 conv/ReLU/max/head error={fmt(value_error)},ALL80 host finite-difference"
          f" derivative error={fmt(gradient_error)};zeroPAD/padding/order passed")


def optimizer_checks():
    p = nn.ParameterStore(347)
    p.get("classifier/W", [2, 2], nn.init.zeros())
    p.set("classifier/W", nn.from_vector([3.0, .1, 4.0, .2], [2, 2]))
    step = kim.KimAdadelta(p)
    host_w, eg, ed = [3, .1, 4, .2], [0.0] * 4, [0.0] * 4
    error = 0.0
    for t in range(3):
        gradient = []
        for i in range(4):
            value = f32(f32((i + 1) * f32(.2)) * (-1 if t == 1 else 1))
            gradient.append(value)
            eg[i] = fma(.95, eg[i], .05 * value * value)
            delta = math.sqrt(ed[i] + 1e-6) / math.sqrt(eg[i] + 1e-6) * value
            ed[i] = fma(.95, ed[i], .05 * delta * delta)
            host_w[i] -= delta
        for c in range(2):
            scale = max(1.0, math.sqrt(fma(host_w[c], host_w[c], host_w[2 + c] * host_w[2 + c])) / 3)
            host_w[c] /= scale
            host_w[2 + c] /= scale
        step.step([nn.from_vector(gradient, [2, 2])])
        w = p.at("classifier/W").values()
        g = step.slots.at("gradient0").values()
        d = step.slots.at("update0").values()
        for i in range(4):
            error = max(error, abs(host_w[i] - w[i]), abs(eg[i] - g[i]), abs(ed[i] - d[i]))
    checks.near(error, 0, 1e-6, "independent three-step Adadelta/both-slot/head-column projection")
    print(f" three signed Adadelta steps/all slots/maxnorm3 host error={fmt(error)}")


def dropout_checks():
    gen = nn.Generator(359)
    x = nn.full([4096], 1.0)
    x.requires_grad_(True)
    training = nn.dropout(x, .5, True, gen)
    inference = nn.dropout(x, 0, True, gen)
    gradient = nn.grad(nn.sum(training), [x])[0].values()
    values, identity = training.values(), inference.values()
    retained = 0
    for i in range(4096):
        checks.require(values[i] == 0 or values[i] == 2, "dropout.5 mask has only0 or2")
        retained += values[i] == 2
        checks.near(gradient[i], values[i], 0, "dropout derivative uses SAME sampled mask")
        checks.near(identity[i], 1, 0, "keep1 dropout identity")
    checks.require(1800 < retained < 2300, "seeded half-drop mask retention")
    print(f" dropout.5 values/shared-mask derivatives and keep1 identity passed,retained={retained}/4096")


def loss_checks():
    z = nn.from_vector([1000, 999, -2, 3], [2, 2])
    z.requires_grad_(True)
    loss = sentence.sentence_ce(z, int_tensor([1, 0], [2]))
    g = nn.grad(loss, [z])[0].values()
    logits = z.values()
    expected = 0.0
    for n in range(2):
        a, b = logits[2 * n], logits[2 * n + 1]
        m = max(a, b)
        total = math.exp(a - m) + math.exp(b - m)
        label = 1 - n
        expected += (m + math.log(total) - (b if label else a)) / 2
        for c in range(2):
            checks.near(g[2 * n + c], (math.exp((b if c else a) - m) / total - (c == label)) / 2, 1e-7,
                        "all stable mean CE logit derivatives")
    checks.near(loss.item(), expected, 2e-7, "independent stable mean binary CE")
    print(" extreme-logit stable CE/batch mean/all logit derivatives passed")


def full_checks():
    p = nn.ParameterStore(353)
    ids = [2 + (i * 17) % 7997 for i in range(128)]
    model = kim.kim_cnn(p, int_tensor(ids, [2, 64]), int_tensor([64, 37], [2]))
    norms = sentence.squared_norms(nn.grad(nn.sum(nn.square(model.logits)), p.parameters()))
    checks.require(p.count() == 2760602 and p.size() == 9, "independent full CNN-rand parameter formula")
    for i in range(3):
        checks.require(list(model.maps[i].shape()) == [2, 70 - i, 100], "window3/4/5 map shape")
    checks.require(list(model.features.shape()) == [2, 300] and list(model.logits.shape()) == [2, 2],
                   "pooled concatenation and binary output")
    for n in norms:
        checks.require(math.isfinite(n) and n > 0, "all9 full parameter tensor gradients finite/nonzero")
    print(" full CNN-rand2760602 parameters/9 tensors,E300,F100,map70/69/68,concat300;every learned tensor gradient passed")


def main(arguments=None):
    args = options(arguments, dict(epochs=3, data=str(checks.dataset("sentence")), checks_only=0, threads=4,
                                   full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    checks.require(args["checks_only"] in (0, 1), "expected epochs=N data=PATH checks_only=0|1 threads=N")
    nn.set_num_threads(args["threads"])
    host_checks()
    optimizer_checks()
    dropout_checks()
    loss_checks()
    full_checks()
    if not args["checks_only"]:
        train_sentence(Path(args["data"]), args["epochs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
