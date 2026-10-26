# Native Python NN example. Greedy layer-wise pretrained stacked sigmoid autoencoder with a scratch control.
"""CMD: just python-example stacked_autoencoder
Greedy 784->64 and 64->16 pretraining, exact transfer into 784->64->16->64->784.
Three epochs per stage/default; validation-only selection and same-stack scratch control.

Port of tflow's stacked_autoencoder (model: examples/nn/common/stacked_autoencoder.hpp).
Preparation (in ~/projects/tflow_examples): just autoencoder-assets; data are read from
$DATAMUNGE_DATA_DIR/saliency. Outputs (stage1/, stage2/, finetune/, scratch/ training records,
transferred_arrays.npz, stacked.npz, metrics.json, index.html and digit<k>.png galleries) go
to output/nn/models/stacked_autoencoder. Checkpoints are best.npz files instead of TF SaveV2
bundles; metrics.json gains "training_seconds". Options: epochs=N data=PATH out=PATH
init=DIR (a tflow stacked_autoencoder output directory: stage1/, stage2/ and scratch/
initial_arrays.npz are loaded into stage1, stage2 and both full stacks, to follow tflow's
trajectories exactly; by default nn's own initialiser is used, with the full stacks
identical because they share a seed).
"""
import math
import os
import struct
import time
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_workflows import data_root, helper, options, output_root

checks = helper("checks")
ae = helper("autoencoder")


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def sigmoid_autoencoder(p, image, widths):
    """Dense sigmoid layers "layer<i>" through mirrored, strictly decreasing widths; every layer output."""
    checks.require(len(widths) >= 3 and len(widths) % 2 == 1, "odd symmetric width list includes input and output")
    for i, width in enumerate(widths):
        checks.require(width > 0 and width == widths[len(widths) - 1 - i], "positive mirrored encoder/decoder dimensions")
        if i < len(widths) // 2:
            checks.require(width > widths[i + 1], "strictly decreasing encoder widths")
    outputs, value = [], image
    for i in range(1, len(widths)):
        value = nn.dense(p, f"layer{i - 1}", value, widths[i], nn.Activation.Sigmoid)
        outputs.append(value)
    return outputs


class StackedModel:
    """Parameters + optional Adam of one symmetric sigmoid stack (stacked_autoencoder.hpp)."""

    def __init__(self, widths, seed, training=True):
        self.widths = list(widths)
        self.features = self.widths[0] if self.widths else 0
        self.code_index = len(self.widths) // 2 - 1
        self.parameters = nn.ParameterStore(seed)
        with nn.no_grad():
            sigmoid_autoencoder(self.parameters, nn.zeros([1, max(self.features, 1)]), self.widths)
        self.optimizer = ae.make_adam(self.parameters) if training else None

    def net(self, x):
        """Every layer output, then the MSE (graph-building)."""
        checks.require(x.dim() == 2 and x.size(1) == self.features, "stage input dimensionality")
        layers = sigmoid_autoencoder(self.parameters, x, self.widths)
        return layers + [ae.reconstruction_mse(layers[-1], x)]

    def evaluate(self, x):
        with nn.no_grad():
            return [t.contiguous() for t in self.net(x)]

    def decode(self, code):
        """The layers after the code layer, from an explicit code."""
        with nn.no_grad():
            value = code
            for i in range(self.code_index + 2, len(self.widths)):
                value = nn.dense(self.parameters, f"layer{i - 1}", value, self.widths[i], nn.Activation.Sigmoid)
            return value.contiguous()

    def load(self, arrays):
        """Explicit whole-model weight transfer (optimizer state unchanged)."""
        checks.require(len(arrays) == self.parameters.size(), "transfer all model tensors")
        for name in self.parameters.names():
            value = ae.find(arrays, name)
            checks.require(value.dtype() == nn.DType.Float32
                           and list(value.shape()) == list(self.parameters.at(name).shape()),
                           "exact transfer shape and dtype")
            self.parameters.set(name, value)


def fixtures():
    toy = StackedModel([4, 3, 2, 3, 4], 1543)
    widths = [4, 3, 2, 3, 4]
    w, b = [[] for _ in range(4)], [[] for _ in range(4)]
    for l in range(4):
        w[l] = [f32(.1 * math.sin(i + l + .3)) for i in range(widths[l] * widths[l + 1])]
        b[l] = [f32(.01 * (i - 1)) for i in range(widths[l + 1])]
        toy.parameters.set(f"layer{l}/W", nn.from_vector(w[l], [widths[l], widths[l + 1]]))
        toy.parameters.set(f"layer{l}/b", nn.from_vector(b[l], [widths[l + 1]]))
    inputs = [f32(f32(.1) * (i + 1)) for i in range(8)]
    x = nn.from_vector(inputs, [2, 4])
    actual = toy.evaluate(x)
    # Explicit forward through every sigmoid layer, full-stack MSE, then reverse-mode gradients.
    toy_layers = sigmoid_autoencoder(toy.parameters, x, toy.widths)
    gradient = nn.grad(ae.reconstruction_mse(toy_layers[-1], x), toy.parameters.parameters())
    forward_error = 0.0
    activations = [list(inputs)] + [None] * 4
    for l in range(4):
        got = actual[l].values()
        following = [0.0] * (2 * widths[l + 1])
        for n in range(2):
            for j in range(widths[l + 1]):
                value = b[l][j]
                for i in range(widths[l]):
                    value += activations[l][n * widths[l] + i] * w[l][i * widths[l + 1] + j]
                following[n * widths[l + 1] + j] = 1 / (1 + math.exp(-value))
                forward_error = max(forward_error, abs(got[n * widths[l + 1] + j] - following[n * widths[l + 1] + j]))
        activations[l + 1] = following
    print(f"toy float32 sigmoid versus float64 maxerror={forward_error:g}")
    checks.near(forward_error, 0, 2e-7, "all24toy sigmoid activations across four layers,float32 tolerance")
    delta, loss = [0.0] * 8, 0.0
    for i in range(8):
        value = activations[4][i]
        error = value - activations[0][i]
        loss += error * error / 8
        delta[i] = 2 * error / 8 * value * (1 - value)
    checks.near(actual[-1].item(), loss, 2e-8, "known full-stack MSE")
    checked = 0
    for l in range(3, -1, -1):
        prior = [0.0] * (2 * widths[l])
        gw, gb = gradient[2 * l].values(), gradient[2 * l + 1].values()
        for j in range(widths[l + 1]):
            db = sum(delta[n * widths[l + 1] + j] for n in range(2))
            checks.near(gb[j], db, 2e-8, "all toy bias chain derivatives")
            checked += 1
            for i in range(widths[l]):
                dw = 0.0
                for n in range(2):
                    dw += activations[l][n * widths[l] + i] * delta[n * widths[l + 1] + j]
                    prior[n * widths[l] + i] += delta[n * widths[l + 1] + j] * w[l][i * widths[l + 1] + j]
                checks.near(gw[i * widths[l + 1] + j], dw, 2e-8, "all toy weight chain derivatives")
                checked += 1
        delta = [value * activations[l][i] * (1 - activations[l][i]) for i, value in enumerate(prior)]
    checks.require(checked == 48, "independent toy gradient inventory48")
    checks.require(ae.tensor_bytes(toy.decode(actual[1])) == ae.tensor_bytes(actual[3]),
                   "two-layer standalone decoder has same full result")
    checks.throws(lambda: StackedModel([4, 4, 4], 1543), "non-undercomplete stack rejected")
    print("toy four-layer sigmoid forward/MSE/ALL48analytic derivatives/transfer/decode/strictshape checks passed")


def train_stage(model, fit, validation, source, epochs, directory, include_zero):
    """Adam stage training with validation-only selection; returns (selected, initial_fit, fit, validation, seconds)."""
    directory.mkdir(parents=True, exist_ok=True)
    nn.save_npz(str(directory / "initial_arrays.npz"), ae.arrays(model.parameters))
    initial_fit = ae.mse(model.evaluate(fit)[-2], fit)
    epoch0 = ae.mse(model.evaluate(validation)[-2], validation)
    best, selected = epoch0, 0
    best_path = str(directory / "best.npz")
    if include_zero:
        model.parameters.save(best_path)  # epoch0 pretrained/scratch baseline eligible for validation selection
    else:
        best = math.inf
    nn.save_npz(str(directory / "epoch0_arrays.npz"), ae.arrays(model.parameters))
    ids = [int(v) for v in source.values()]
    order = nn.BatchOrder(1000, 32, 1559)
    step = 0
    started = time.perf_counter()
    with (directory / "history.tsv").open("w") as history, (directory / "batches.tsv").open("w") as batches:
        history.write("epoch\ttrain_preupdate_mse\tvalidation_mse\n")
        batches.write("epoch\tstep\trow\tsource_id\n")
        history.write(f"0\t{initial_fit:.17g}\t{epoch0:.17g}\n")
        for epoch in range(1, epochs + 1):
            order.begin_epoch()
            total, seen = 0.0, 0
            while True:
                rows = list(order.next())
                if not rows:
                    break
                x = nn.take_rows(fit, rows)
                # Forward the stage, reconstruction MSE against its own input.
                layers = sigmoid_autoencoder(model.parameters, x, model.widths)
                loss = ae.reconstruction_mse(layers[-1], x)
                if step == 0:
                    keep = nn.GradOptions()
                    keep.retain_graph = True
                    nn.save_npz(str(directory / "first_gradients.npz"),
                                ae.check_gradients(model.parameters, nn.grad(loss, model.parameters.parameters(), keep),
                                                   "all stage derivatives finite",
                                                   "every stage tensor receives reconstruction gradient"))
                    nn.save_npz(str(directory / "first_batch.npz"), [("images", x)])
                value = loss.item()
                checks.require(math.isfinite(value) and value >= 0, "finite stage MSE")
                model.optimizer.minimize(loss)
                total += len(rows) * value
                seen += len(rows)
                batches.write("".join(f"{epoch}\t{step}\t{row}\t{ids[row]}\n" for row in rows))
                if step == 0:
                    nn.save_npz(str(directory / "first_update_arrays.npz"), ae.arrays(model.parameters))
                step += 1
            checks.require(seen == 1000, "each fitting record once per stage epoch including short last batch")
            mse = ae.mse(model.evaluate(validation)[-2], validation)
            history.write(f"{epoch}\t{total / seen:.17g}\t{mse:.17g}\n")
            nn.save_npz(str(directory / f"epoch{epoch}_arrays.npz"), ae.arrays(model.parameters))
            if mse < best:
                best, selected = mse, epoch
                model.parameters.save(best_path)  # validation-only stage selection
            print(f"{directory.name} epoch{epoch}:fitPreMSE={total / seen:g},validationMSE={mse:g}", flush=True)
    seconds = time.perf_counter() - started
    model.parameters.load(best_path)  # restore selected greedy/joint model
    fit_mse = ae.mse(model.evaluate(fit)[-2], fit)
    checks.require(fit_mse < initial_fit or (include_zero and selected == 0),
                   "greedy stage learns target or eligible epoch0 is explicitly retained")
    nn.save_npz(str(directory / "model_arrays.npz"), ae.arrays(model.parameters))
    (directory / "stage.json").write_text(
        f'{{"epochs":{epochs},"updates":{step},"selected_epoch":{selected},'
        f'"include_epoch0":{"true" if include_zero else "false"},"initial_fit_mse":{initial_fit:.17g},'
        f'"fit_mse":{fit_mse:.17g},"validation_mse":{best:.17g},"training_seconds":{seconds:.17g}}}\n')
    return selected, initial_fit, fit_mse, best, seconds


def shift(codes):
    n = codes.size(0)
    return nn.take_rows(codes, [(b + 1) % n for b in range(n)]).contiguous()


def main(arguments=None):
    args = options(arguments, dict(epochs=3, data=str(data_root() / "saliency"), out="", init="",
                                   full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    epochs = args["epochs"]
    data = Path(args["data"])
    directory = Path(args["out"]) if args["out"] else output_root(args["full"]) / "models/stacked_autoencoder"
    init = Path(args["init"]) if args["init"] else None
    checks.require(0 < epochs <= 100, "three default epochs PER stage or explicit1..100")
    fixtures()

    pools = ae.load_saliency(data)
    fit, val, test = (ae.flatten(pools[k].images).contiguous() for k in range(3))
    ids = pools[0].source
    directory.mkdir(parents=True, exist_ok=True)
    stage1, stage2 = StackedModel([784, 64, 784], 1549), StackedModel([64, 16, 64], 1553)
    joint, scratch = StackedModel([784, 64, 16, 64, 784], 1553), StackedModel([784, 64, 16, 64, 784], 1553)
    if init is not None:
        stage1.parameters.load(str(init / "stage1" / "initial_arrays.npz"))
        stage2.parameters.load(str(init / "stage2" / "initial_arrays.npz"))
        joint.parameters.load(str(init / "scratch" / "initial_arrays.npz"))
        scratch.parameters.load(str(init / "scratch" / "initial_arrays.npz"))
    checks.require(stage1.parameters.count() == 101200 and stage2.parameters.count() == 2128
                   and joint.parameters.count() == 103328, "independent greedy/full parameter inventories")
    checks.require(ae.snapshot(joint.parameters) == ae.snapshot(scratch.parameters),
                   "same full graph/initial seed before transfer")

    first = train_stage(stage1, fit, val, ids, epochs, directory / "stage1", False)
    frozen = ae.snapshot(stage1.parameters)
    encoded_fit, encoded_val, encoded_test = (stage1.evaluate(x)[0] for x in (fit, val, test))
    second = train_stage(stage2, encoded_fit, encoded_val, ids, epochs, directory / "stage2", False)
    checks.require(frozen == ae.snapshot(stage1.parameters), "stage2 cannot mutate frozen stage1 encoder/decoder")
    checks.require(ae.tensor_bytes(stage1.evaluate(fit)[0]) == ae.tensor_bytes(encoded_fit),
                   "cached stage2 fitting targets remain exact frozen encoder outputs")

    a, b = ae.arrays(stage1.parameters), ae.arrays(stage2.parameters)
    transfer = []
    for target, source_name, source in [("layer0", "layer0", 1), ("layer3", "layer1", 1),
                                        ("layer1", "layer0", 2), ("layer2", "layer1", 2)]:
        for suffix in ("/W", "/b"):
            transfer.append((target + suffix, ae.find(a if source == 1 else b, source_name + suffix)))
    joint.load(transfer)
    loaded = ae.arrays(joint.parameters)
    for key, value in transfer:
        checks.require(ae.tensor_bytes(ae.find(loaded, key)) == ae.tensor_bytes(value),
                       "ALL8pretrained arrays transferred exactly including both decoders")
    nn.save_npz(str(directory / "transferred_arrays.npz"), loaded)
    pretrained = joint.evaluate(test)
    nested1 = stage1.decode(stage2.evaluate(encoded_test)[1])
    checks.require(ae.tensor_bytes(pretrained[3]) == ae.tensor_bytes(nested1),
                   "composed selected greedy decoders exactly match transferred full graph")

    fine = train_stage(joint, fit, val, ids, epochs, directory / "finetune", True)
    plain = train_stage(scratch, fit, val, ids, epochs, directory / "scratch", True)
    final, scratch_result = joint.evaluate(test), scratch.evaluate(test)
    deployed = StackedModel([784, 64, 16, 64, 784], 1553, False)
    deployed.parameters.load(str(directory / "finetune" / "best.npz"))  # fresh inference restore all8 tensors
    before = ae.snapshot(deployed.parameters)
    restored = deployed.evaluate(test)
    checks.require(ae.tensor_bytes(restored[3]) == ae.tensor_bytes(final[3])
                   and ae.tensor_bytes(restored[1]) == ae.tensor_bytes(final[1]),
                   "fresh restore exact all test reconstructions/codes")
    checks.require(ae.tensor_bytes(deployed.decode(restored[1])) == ae.tensor_bytes(restored[3]),
                   "full encode then two-layer decoder exact")
    checks.require(before == ae.snapshot(deployed.parameters), "deployed inference leaves all8parameters immutable")
    checks.require(frozen == ae.snapshot(stage1.parameters), "stage1 remains frozen during all later stages")

    shuffled = deployed.decode(shift(final[1]))
    zero = deployed.decode(nn.zeros([200, 16]))
    mean = ae.constant(ae.pixel_mean(fit), 200)
    checks.require(before == ae.snapshot(deployed.parameters),
                   "zero/shuffled code controls leave all8deployed parameters immutable")
    first_reconstruction = stage1.evaluate(test)[1]
    pre_mse, fine_mse = ae.mse(pretrained[3], test), ae.mse(final[3], test)
    scratch_mse, mean_mse = ae.mse(scratch_result[3], test), ae.mse(mean, test)
    nn.save_npz(str(directory / "stacked.npz"), [
        ("fit_images", fit), ("validation_images", val), ("test_images", test), ("fit_source_ids", ids),
        ("validation_source_ids", pools[1].source), ("test_source_ids", pools[2].source),
        ("test_labels", pools[2].labels), ("stage1_fit_code", encoded_fit), ("stage1_validation_code", encoded_val),
        ("stage1_test_code", encoded_test), ("stage1_test_reconstruction", first_reconstruction),
        ("pretrained_test_code", pretrained[1]), ("pretrained_test_reconstruction", pretrained[3]),
        ("finetuned_test_code", final[1]), ("finetuned_test_reconstruction", final[3]),
        ("scratch_test_code", scratch_result[1]), ("scratch_test_reconstruction", scratch_result[3]),
        ("mean_reconstruction", mean), ("shuffled_reconstruction", shuffled), ("zero_reconstruction", zero)])
    ae.copy_manifests(data, directory)
    seconds = first[4] + second[4] + fine[4] + plain[4]
    (directory / "metrics.json").write_text(
        f'{{"epochs_per_stage":{epochs},"staged_epochs":{epochs * 3},"scratch_epochs":{epochs},'
        f'"staged_updates":{epochs * 96},"scratch_updates":{epochs * 32},"pretrained_test_mse":{pre_mse:.17g},'
        f'"finetuned_test_mse":{fine_mse:.17g},"scratch_test_mse":{scratch_mse:.17g},"mean_test_mse":{mean_mse:.17g},'
        f'"shuffled_test_mse":{ae.mse(shuffled, test):.17g},"zero_test_mse":{ae.mse(zero, test):.17g},'
        f'"training_seconds":{seconds:.17g}}}\n')

    html = [
        "<!doctype html><meta charset='utf-8'><title>Greedy stacked autoencoder</title><style>body{font:16px "
        "system-ui;max-width:1100px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:"
        "25px 0}</style><h1>Stacked autoencoder with actual layer-wise pretraining</h1><p>Original,stage1 "
        "reconstruction,composed pretrained stack,pretrained plus fine-tuning,scratch joint training,fit-only mean,"
        "cyclic shuffled code,zero-code decoder. Fixed[0,1]grayscale/nearest4,all ten first heldout examples per "
        "true digit,no quality filtering.</p><p>784->64->16->64->784,ALL sigmoid layers,103328untied parameters. "
        "Stage1 learns784->64->784;selected encoder1 is frozen while stage2 learns64->16->64 from its actual cached "
        "representations. All8arrays including both decoders transfer EXACTLY before joint fine-tuning. No labels "
        "enter any loss or selection.</p><p>"
        f"{epochs}epochs PER stage;staged total{3 * epochs}epochs/{96 * epochs}"
        f"updates across different-sized models,versus scratch{epochs}epochs/{32 * epochs}"
        "full-model updates. Same full architecture and fitting order;equal joint phase,NOT equal total work or a "
        "causal proof that pretraining is better. Each phase selects by its own validation MSE;epoch0 is eligible "
        f"for fine-tuning/scratch.</p><p>Heldout MSE pretrained={pre_mse:g},fine-tuned={fine_mse:g},"
        f"scratch={scratch_mse:g},fit-mean={mean_mse:g}"
        ". Greedy stage2 reconstruction quality need not preserve raw-image detail. Blurry average-looking "
        "reconstructions and weak short-run results remain visible. This16float code is not a bit-rate compression "
        "claim,classification or a generative prior.</p>"]
    ae.digit_gallery(directory, pools[2],
                     [test, first_reconstruction, pretrained[3], final[3], scratch_result[3], mean, shuffled, zero], html)
    (directory / "index.html").write_text("".join(html))
    print(f"Stage selections={first[0]}/{second[0]}/{fine[0]},scratch={plain[0]}\n"
          f"HeldoutMSE pretrained/fine/scratch/mean={pre_mse:g}/{fine_mse:g}/{scratch_mse:g}/{mean_mse:g}\n"
          "Frozen stage1/cachedtargets/ALL8exacttransfer/nesteddecoder/freshrestore/encode-decode/"
          "immutable8weights passed. Staged work3x scratch epochs;no superiority claim.\n"
          f"training time {seconds:g} s\nArtifacts:{directory}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
