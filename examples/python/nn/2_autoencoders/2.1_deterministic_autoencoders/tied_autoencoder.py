# Native Python NN example. Tied-weight autoencoder (one live encoder matrix, transposed decoder) with an untied control.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/2_autoencoders/2.1_deterministic_autoencoders/tied_autoencoder.py
"""ONE live 784x32 parameter for encoder AND transposed decoder; two independent biases.
Three default epochs, matched-initial untied control, and explicit gradient-path checks.

Port of tflow's tied_autoencoder (model: examples/nn/common/tied_autoencoder.hpp).
Preparation (in ~/projects/tflow_examples): just autoencoder-assets; data are read from
$DATAMUNGE_DATA_DIR/saliency. Outputs (tied/ and untied/ training records, initial_forward.npz,
data.npz, index.html and digit<k>.png galleries) go to output/nn/models/tied_autoencoder.
Checkpoints are best.npz files instead of TF SaveV2 bundles; metrics.json gains
"training_seconds". In eager mode the tie is the decoder weight being recomputed as
transpose(encoder/W) on every forward pass (no decoder parameter exists); it is checked
coordinate-by-coordinate after every update as in tflow. Options: epochs=N data=PATH out=PATH
init=PATH (tied initial parameters, e.g. tflow's tied/initial_arrays.npz, to follow its
trajectories exactly; the untied control always starts from their exact transpose).
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


class TiedOutput:
    encoder_weight = decoder_weight = code = reconstruction = None


def tied_autoencoder(p, image, features=784, latent=32, tied=True):
    """code = ReLU(x W + eb); reconstruction = sigmoid(code D + db), D = transpose(W) when tied."""
    checks.require(features > 1 and 0 < latent < features, "positive strict undercomplete tied architecture")
    limit = math.sqrt(6.0 / (features + latent))
    out = TiedOutput()
    out.encoder_weight = p.get("encoder/W", [features, latent], nn.init.uniform(limit))
    eb = p.get("encoder/b", [latent], nn.init.zeros())
    out.code = nn.relu(nn.bias_add(nn.matmul(image, out.encoder_weight), eb))
    out.decoder_weight = (nn.transpose(out.encoder_weight, [1, 0]) if tied
                          else p.get("decoder/W", [latent, features], nn.init.uniform(limit)))
    db = p.get("decoder/b", [features], nn.init.zeros())
    out.reconstruction = nn.sigmoid(nn.bias_add(nn.matmul(out.code, out.decoder_weight), db))
    return out


class TiedModel:
    """Parameters (seed 1853) + optional Adam of the tied or untied model (tied_autoencoder.hpp)."""

    def __init__(self, tied=True, training=True):
        self.tied = tied
        self.parameters = nn.ParameterStore(1853)
        with nn.no_grad():
            tied_autoencoder(self.parameters, nn.zeros([1, 784]), 784, 32, tied)
        self.optimizer = ae.make_adam(self.parameters) if training else None

    def net(self, images):
        return tied_autoencoder(self.parameters, ae.flatten(images), 784, 32, self.tied)

    def evaluate(self, images):
        """[code, reconstruction, loss]"""
        with nn.no_grad():
            n = self.net(images)
            return [n.code, n.reconstruction, ae.reconstruction_mse(n.reconstruction, ae.flatten(images))]

    def decode(self, codes):
        with nn.no_grad():
            d = nn.transpose(self.parameters.at("encoder/W"), [1, 0]) if self.tied else self.parameters.at("decoder/W")
            return nn.sigmoid(nn.bias_add(nn.matmul(codes, d), self.parameters.at("decoder/b")))

    def load(self, arrays):
        """Loads every parameter by name (all must be supplied with matching shapes)."""
        checks.require(len(arrays) == self.parameters.size(), "all initial model tensors supplied")
        for name in self.parameters.names():
            value = ae.find(arrays, name)
            checks.require(value.dtype() == nn.DType.Float32
                           and list(value.shape()) == list(self.parameters.at(name).shape()),
                           "matching initial shape/dtype")
            self.parameters.set(name, value)

    def decoder_weight(self):
        """The decoder weight the forward pass actually uses."""
        with nn.no_grad():
            return self.net(nn.zeros([1, 784])).decoder_weight.contiguous()

    def check_tie(self):
        checks.require(not self.parameters.contains("decoder/W") and self.parameters.size() == 3,
                       "decoder graph directly transposes same encoder parameter")
        d = self.decoder_weight()
        checks.require(list(d.shape()) == [32, 784], "actual decoder transpose shape")
        dv, w = d.values(), self.parameters.at("encoder/W").contiguous().values()
        transposed = [w[i * 32 + j] for j in range(32) for i in range(784)]
        checks.require(list(dv) == transposed,
                       "ALL25088current decoder coordinates equal current encoder transpose")


def fixtures():
    p = nn.ParameterStore(1861)
    with nn.no_grad():
        tied_autoencoder(p, nn.zeros([1, 3]), 3, 2, True)
    w = [f32(v) for v in (.2, .1, -.3, .4, .5, -.2)]
    eb = [f32(v) for v in (.1, .2)]
    db = [f32(v) for v in (-.1, .05, .2)]
    inputs = [f32(v) for v in (.1, .2, .7, .8, .3, .4)]
    p.set("encoder/W", nn.from_vector(w, [3, 2]))
    p.set("encoder/b", nn.from_vector(eb, [2]))
    p.set("decoder/b", nn.from_vector(db, [3]))
    x = nn.from_vector(inputs, [2, 3])
    net = tied_autoencoder(p, x, 3, 2, True)
    loss = ae.reconstruction_mse(net.reconstruction, x)
    gradient = nn.grad(loss, p.parameters())
    code, recon = net.code.values(), net.reconstruction.values()
    gw, geb, gdb = (g.values() for g in gradient)
    encoder = [[0.0, 0.0] for _ in range(3)]
    decoder = [[0.0, 0.0] for _ in range(3)]
    ge, gd, mse = [0.0, 0.0], [0.0, 0.0, 0.0], 0.0
    for b in range(2):
        c = [eb[0], eb[1]]
        delta = [0.0] * 3
        for j in range(2):
            for i in range(3):
                c[j] += inputs[b * 3 + i] * w[i * 2 + j]
            checks.require(c[j] > 0, "toy avoids ReLU kink")
            checks.near(code[b * 2 + j], c[j], 5e-8, "all4toy codes")
        for i in range(3):
            value = db[i]
            for j in range(2):
                value += c[j] * w[i * 2 + j]
            y = 1 / (1 + math.exp(-value))
            error = y - inputs[b * 3 + i]
            checks.near(recon[b * 3 + i], y, 8e-8, "all6toy sigmoid predictions")
            mse += error * error / 6
            delta[i] = error / 3 * y * (1 - y)
            gd[i] += delta[i]
            for j in range(2):
                decoder[i][j] += delta[i] * c[j]
        for j in range(2):
            dz = 0.0
            for i in range(3):
                dz += delta[i] * w[i * 2 + j]
            ge[j] += dz
            for i in range(3):
                encoder[i][j] += inputs[b * 3 + i] * dz
    checks.require(p.size() == 3 and p.count() == 11, "one sharedmatrix plus two biases,not a copied decoder parameter")
    checks.near(loss.item(), mse, 2e-8, "known tied MSE")
    encoder_norm = decoder_norm = 0.0
    for i in range(3):
        checks.near(gdb[i], gd[i], 3e-8, "all3decoder bias derivatives")
        for j in range(2):
            checks.near(gw[i * 2 + j], encoder[i][j] + decoder[i][j], 3e-8,
                        "ALL6shared derivatives equal encoder-path PLUS decoder-path")
            encoder_norm += abs(encoder[i][j])
            decoder_norm += abs(decoder[i][j])
    for j in range(2):
        checks.near(geb[j], ge[j], 3e-8, "all2encoder bias derivatives")
    checks.require(encoder_norm > 1e-3 and decoder_norm > 1e-3,
                   "neither shared gradient contribution is silently missing")
    print("known tied toy4codes/6predictions/MSE/ALL11derivatives/BOTHsharedmatrix contributions passed")


def shift(codes):
    n = codes.size(0)
    return nn.take_rows(codes, [(b + 1) % n for b in range(n)]).contiguous()


def train_variant(tied, pools, epochs, root, initial):
    """Adam training of the tied or untied model from matched initial arrays, then a fresh-restore evaluation."""
    name = "tied" if tied else "untied"
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    model = TiedModel(tied)
    model.load(initial)
    checks.require(model.parameters.size() == (3 if tied else 4)
                   and model.parameters.count() == (25904 if tied else 50992),
                   "independent tied/untied parameter census")
    if tied:
        model.check_tie()
    nn.save_npz(str(directory / "initial_arrays.npz"), ae.arrays(model.parameters))
    order = nn.BatchOrder(1000, 32, 1867)
    fit_source = [int(v) for v in pools[0].source.values()]
    best_path = str(directory / "best.npz")
    best, selected, step = math.inf, 0, 0
    started = time.perf_counter()
    with (directory / "history.tsv").open("w") as history, (directory / "batches.tsv").open("w") as batches:
        history.write("epoch\tfit_preupdate_mse\tvalidation_mse\n")
        batches.write("epoch\tstep\trow\tsource_id\n")
        for epoch in range(1, epochs + 1):
            order.begin_epoch()
            total, seen = 0.0, 0
            while True:
                rows = list(order.next())
                if not rows:
                    break
                inputs = nn.take_rows(pools[0].images, rows)
                # Forward (live transposed decoder when tied), reconstruction MSE, gradients through BOTH paths.
                net = model.net(inputs)
                loss = ae.reconstruction_mse(net.reconstruction, ae.flatten(inputs))
                if step == 0:
                    keep = nn.GradOptions()
                    keep.retain_graph = True
                    nn.save_npz(str(directory / "first_gradients.npz"),
                                ae.check_gradients(model.parameters, nn.grad(loss, model.parameters.parameters(), keep),
                                                   "all model MSE derivatives finite",
                                                   "all model tensors receive full MSE gradients"))
                    nn.save_npz(str(directory / "first_batch.npz"), [("images", inputs)])
                value = loss.item()
                checks.require(math.isfinite(value) and value >= 0, "finite pre-update MSE")
                model.optimizer.minimize(loss)
                total += len(rows) * value
                seen += len(rows)
                batches.write("".join(f"{epoch}\t{step}\t{row}\t{fit_source[row]}\n" for row in rows))
                if tied:
                    model.check_tie()
                if step == 0:
                    nn.save_npz(str(directory / "first_update_arrays.npz"), ae.arrays(model.parameters))
                step += 1
            checks.require(seen == 1000, "all fitting records exactly once per epoch")
            mse = ae.mse(model.evaluate(pools[1].images)[1], pools[1].images)
            history.write(f"{epoch}\t{total / seen:.17g}\t{mse:.17g}\n")
            nn.save_npz(str(directory / f"epoch{epoch}_arrays.npz"), ae.arrays(model.parameters))
            if mse < best:
                best, selected = mse, epoch
                model.parameters.save(best_path)  # common validation reconstruction MSE selects each model
            print(f"{name} epoch{epoch}:validationMSE={mse:g}", flush=True)
    seconds = time.perf_counter() - started
    deployed = TiedModel(tied, False)
    deployed.parameters.load(best_path)  # fresh tied/untied inference restore
    if tied:
        deployed.check_tie()
    before = ae.snapshot(deployed.parameters)
    arrays = []
    for split, label in enumerate(("fit", "validation", "test")):
        out = deployed.evaluate(pools[split].images)
        arrays += [(label + "_code", out[0]), (label + "_reconstruction", out[1])]
        if split == 1:
            checks.near(ae.mse(out[1], pools[split].images), best, 1e-8, "fresh selected validation MSE")
    test, repeat = deployed.evaluate(pools[2].images), deployed.evaluate(pools[2].images)
    checks.require(ae.tensor_bytes(test[1]) == ae.tensor_bytes(repeat[1])
                   and ae.tensor_bytes(deployed.decode(test[0])) == ae.tensor_bytes(test[1]),
                   "deterministic exact encode-decode")
    arrays.append(("zero_reconstruction", deployed.decode(nn.zeros([200, 32]))))
    arrays.append(("shuffled_reconstruction", deployed.decode(shift(test[0]))))
    one = deployed.evaluate(ae.rows(pools[2], [0]).images)
    error = ae.max_abs_diff(one[1], test[1], 784)
    checks.require(error < 1e-6, "single/batch tied or untied inference parity")
    checks.require(before == ae.snapshot(deployed.parameters), "all deployed parameters immutable through code controls")
    nn.save_npz(str(directory / "outputs.npz"), arrays)
    nn.save_npz(str(directory / "model_arrays.npz"), ae.arrays(deployed.parameters))
    nn.save_npz(str(directory / "decoder_weight.npz"), [("decoder_weight", deployed.decoder_weight())])
    test_mse = ae.mse(test[1], pools[2].images)
    (directory / "metrics.json").write_text(
        f'{{"tied":{"true" if tied else "false"},"parameters":{model.parameters.count()},"epochs":{epochs},'
        f'"updates":{step},"selected_epoch":{selected},"test_mse":{test_mse:.17g},'
        f'"zero_mse":{ae.mse(ae.find(arrays, "zero_reconstruction"), pools[2].images):.17g},'
        f'"shuffled_mse":{ae.mse(ae.find(arrays, "shuffled_reconstruction"), pools[2].images):.17g},'
        f'"single_batch_max_error":{error:.17g},"training_seconds":{seconds:.17g}}}\n')
    print(f"{name}:selected{selected},parameters={model.parameters.count()},testMSE={test_mse:g},"
          "freshrestore/decode/repeat/immutable parameters passed")
    return seconds


def main(arguments=None):
    args = options(arguments, dict(epochs=3, data=str(data_root() / "saliency"), out="", init="",
                                   full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    epochs = args["epochs"]
    data = Path(args["data"])
    root = Path(args["out"]) if args["out"] else output_root(args["full"]) / "models/tied_autoencoder"
    checks.require(0 < epochs <= 100, "three default epochs or explicit1..100")
    fixtures()

    pools = ae.load_saliency(data)
    root.mkdir(parents=True, exist_ok=True)
    factory = TiedModel(True, False)
    if args["init"]:
        factory.parameters.load(args["init"])
    tied_initial = ae.arrays(factory.parameters)
    transposed = nn.transpose(ae.find(tied_initial, "encoder/W"), [1, 0]).contiguous()
    untied_initial = tied_initial + [("decoder/W", transposed)]
    initial_control = TiedModel(False, False)
    initial_control.load(untied_initial)
    initial_probe = ae.rows(pools[0], list(range(8))).images
    initial_t, initial_u = factory.evaluate(initial_probe), initial_control.evaluate(initial_probe)
    checks.require(ae.tensor_bytes(initial_t[0]) == ae.tensor_bytes(initial_u[0])
                   and ae.tensor_bytes(initial_t[1]) == ae.tensor_bytes(initial_u[1]),
                   "native matched initial codes AND reconstructions are BITEXACT")
    nn.save_npz(str(root / "initial_forward.npz"),
                [("images", initial_probe), ("tied_code", initial_t[0]), ("untied_code", initial_u[0]),
                 ("tied_reconstruction", initial_t[1]), ("untied_reconstruction", initial_u[1])])
    seconds = train_variant(True, pools, epochs, root, tied_initial)
    seconds += train_variant(False, pools, epochs, root, untied_initial)
    t_initial = nn.load_npz(str(root / "tied" / "initial_arrays.npz"))
    u_initial = nn.load_npz(str(root / "untied" / "initial_arrays.npz"))
    for key, value in t_initial:
        checks.require(ae.tensor_bytes(ae.find(u_initial, key)) == ae.tensor_bytes(value),
                       "matched initial encoder/bias bytes")
    checks.require(ae.tensor_bytes(ae.find(u_initial, "decoder/W")) == ae.tensor_bytes(transposed),
                   "untied starts EXACTLY same numerical transpose,then independently trains")
    ae.save_data(root / "data.npz", pools)
    ae.copy_manifests(data, root)
    t = nn.load_npz(str(root / "tied" / "outputs.npz"))
    u = nn.load_npz(str(root / "untied" / "outputs.npz"))
    mean = ae.constant(ae.pixel_mean(pools[0].images), 200)
    t_recon = ae.find(t, "test_reconstruction")
    tied_mse = ae.mse(t_recon, pools[2].images)
    untied_mse = ae.mse(ae.find(u, "test_reconstruction"), pools[2].images)
    mean_mse = ae.mse(mean, pools[2].images)
    html = [
        "<!doctype html><meta charset='utf-8'><title>Actual tied-weight autoencoder</title><style>body{font:16px "
        "system-ui;max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:"
        "25px 0}</style><h1>Tied weights:ONE parameter,two gradient paths</h1><p>Original,tied,untied,fit-only mean,"
        "absolute tied error,tied zero-code,tied shuffled-code. Fixed[0,1]grayscale/nearest4,no error amplification "
        "or quality filtering.</p><p>784->32ReLU->784sigmoid. Tied decoder is the LIVE transpose of the SAME encoder "
        "parameter,never a copied decoder parameter. Separate encoder/decoder biases;25904parameters versus50992"
        "untied. After EVERY tied update all25088decoder coordinates are checked EXACTLY against current encoder "
        "transpose. Shared MSE gradient sums encoder and decoder contributions.</p><p>Both models start numerically "
        "IDENTICAL,including untied decoder initialized to encoder transpose. Same source/order/Adam.003/"
        f"{epochs}"
        "epochs;each selects on common validation MSE. Parameter counts differ,so this is a shared-constraint "
        "comparison,not equal capacity or statistical proof of superiority. Continuous32float code is not bit-rate "
        f"compression. Blurry/distorted shapes can persist.</p><p>TestMSE tied={tied_mse:g},untied={untied_mse:g},"
        f"fitmean={mean_mse:g}"
        ". Zero/shuffle controls expose input-specific encoding;no classification/denoising/generative claim.</p>"]
    ae.digit_gallery(root, pools[2],
                     [pools[2].images, t_recon, ae.find(u, "test_reconstruction"), mean,
                      nn.abs(nn.sub(t_recon, ae.flatten(pools[2].images))), ae.find(t, "zero_reconstruction"),
                      ae.find(t, "shuffled_reconstruction")], html)
    (root / "index.html").write_text("".join(html))
    print("Actualsharedparameter/transpose afterEVERYupdate/BOTHgradientpaths/numericallymatched initial "
          f"untiedcontrol verified.\ntraining time {seconds:g} s\nArtifacts:{root}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
