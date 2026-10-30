# Native Python NN example. Contractive autoencoder with the exact sigmoid-encoder Jacobian penalty.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/2_autoencoders/2.1_deterministic_autoencoders/contractive_autoencoder.py
"""Exact sigmoid-encoder Jacobian-Frobenius penalty, plain and strong matched controls.
Three epochs/default; differentiate through activation slopes AND encoder weights.

Port of tflow's contractive_autoencoder. Preparation (in ~/projects/tflow_examples):
just autoencoder-assets; data are read from $DATAMUNGE_DATA_DIR/saliency. Outputs
(per-variant plain/contractive/strong histories, npz arrays and metrics.json, data.npz,
index.html and digit<k>.png galleries) go to output/nn/models/contractive_autoencoder.
Checkpoints are best.npz files instead of TF SaveV2 bundles. Options: epochs=N
data=PATH out=PATH init=PATH (start all three variants from saved parameters, e.g.
tflow's plain/initial_arrays.npz, to follow its trajectories exactly; by default nn's
own initialiser is used, identically for the three variants).
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


def float32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


# ---- model (examples/nn/common/contractive_autoencoder.hpp) ----------------------------------

class ContractivePenalty:
    def __init__(self, per_example, mean):
        self.per_example, self.mean = per_example, mean


def sigmoid_encoder_contraction(code, encoder_weight):
    """sum_j (h_j (1 - h_j))^2 sum_i W_ij^2 per example, and its batch mean."""
    slope = nn.mul(code, nn.sub(1.0, code))
    squared_column_norm = nn.sum(nn.square(encoder_weight), [0])
    per = nn.sum(nn.mul(nn.square(slope), squared_column_norm), [1])
    return ContractivePenalty(per, nn.mean(per, [0]))


class ContractiveOutput:
    code = reconstruction = loss = mse = penalty = None


class ContractiveAutoencoderModel:
    """784->32 sigmoid->784 sigmoid (seed 1753) with loss MSE + weight * exact Jacobian penalty."""

    def __init__(self, weight, training=True):
        self.weight = float32(weight)
        checks.require(math.isfinite(self.weight) and self.weight >= 0, "finite nonnegative Jacobian weight")
        self.parameters = nn.ParameterStore(1753)
        with nn.no_grad():
            self.forward(nn.zeros([1, 784]))
        self.optimizer = ae.make_adam(self.parameters) if training else None

    def decoder(self, code):
        return nn.dense(self.parameters, "decoder", code, 784, nn.Activation.Sigmoid)

    def forward(self, image):
        out = ContractiveOutput()
        out.code = nn.dense(self.parameters, "encoder", image, 32, nn.Activation.Sigmoid)
        out.reconstruction = self.decoder(out.code)
        out.mse = ae.reconstruction_mse(out.reconstruction, image)
        out.penalty = sigmoid_encoder_contraction(out.code, self.parameters.at("encoder/W"))
        out.loss = nn.add(out.mse, nn.mul(out.penalty.mean, self.weight))
        return out

    def evaluate(self, images):
        """[code, reconstruction, loss, mse, penalty mean, penalty per example]"""
        with nn.no_grad():
            n = self.forward(ae.flatten(images))
            return [n.code, n.reconstruction, n.loss, n.mse, n.penalty.mean, n.penalty.per_example]

    def jacobian(self, images):
        """[n, 32, 784] image Jacobian of the code, one reverse pass per code unit."""
        x = ae.flatten(images).contiguous().detach().requires_grad_()
        code = nn.dense(self.parameters, "encoder", x, 32, nn.Activation.Sigmoid)
        keep = nn.GradOptions()
        keep.retain_graph = True
        rows = [nn.grad(nn.sum(nn.slice_axis(code, 1, j, j + 1)), [x], keep)[0] for j in range(32)]
        return nn.stack(rows, 1).contiguous()

    def decode(self, codes):
        with nn.no_grad():
            return self.decoder(codes)


def fixtures():
    xv = [float32(v) for v in (.1, .2, .7, .8, .3, .4)]
    wv = [float32(v) for v in (.3, -.4, .7, .2, -.2, .5)]
    bv = [float32(v) for v in (.1, -.2)]
    x = nn.from_vector(xv, [2, 3]).requires_grad_()
    w = nn.from_vector(wv, [3, 2]).requires_grad_()
    b = nn.from_vector(bv, [2]).requires_grad_()
    h = nn.sigmoid(nn.bias_add(nn.matmul(x, w), b))
    p = sigmoid_encoder_contraction(h, w)
    keep = nn.GradOptions()
    keep.retain_graph = True
    grad = nn.grad(p.mean, [w, b], keep)
    rows = [nn.grad(nn.sum(nn.slice_axis(h, 1, j, j + 1)), [x], keep)[0] for j in range(2)]
    jac = nn.stack(rows, 1).values()  # [2, 2, 3]
    penalty = 0.0
    gw = [[0.0, 0.0] for _ in range(3)]
    gb = [0.0, 0.0]
    for n in range(2):
        for j in range(2):
            z, norm = bv[j], 0.0
            for i in range(3):
                z += xv[n * 3 + i] * wv[i * 2 + j]
                norm += wv[i * 2 + j] ** 2
            activation = 1 / (1 + math.exp(-z))
            slope = activation * (1 - activation)
            dz = slope * slope * (1 - 2 * activation) * norm
            penalty += slope * slope * norm / 2
            gb[j] += dz
            for i in range(3):
                checks.near(jac[(n * 2 + j) * 3 + i], slope * wv[i * 2 + j], 3e-8, "ALL12true encoder-Jacobian entries")
                gw[i][j] += xv[n * 3 + i] * dz + wv[i * 2 + j] * slope * slope
    checks.near(p.mean.item(), penalty, 3e-8, "SUMallJacobian coordinates,MEANbatch")
    gwv, gbv = grad[0].values(), grad[1].values()
    for j in range(2):
        checks.near(gbv[j], gb[j], 3e-8, "ALL2penalty bias derivatives")
        for i in range(3):
            checks.near(gwv[i * 2 + j], gw[i][j], 3e-8, "ALL6penalty weight derivatives include direct2W term")
    print("known encoder:ALL12Jacobian coordinates/Frobeniusnorm/ALL8penalty parameter derivatives passed")


def train_variant(name, weight, pools, epochs, root, init):
    """Trains one matched variant on MSE + weight * Jacobian penalty; returns training seconds."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    model = ContractiveAutoencoderModel(weight)
    if init:
        model.parameters.load(init)
    checks.require(model.parameters.count() == 50992 and model.parameters.size() == 4,
                   "independent784->32->784untied inventory")
    nn.save_npz(str(directory / "initial_arrays.npz"), ae.arrays(model.parameters))
    order = nn.BatchOrder(1000, 32, 1759)
    fit_source = [int(v) for v in pools[0].source.values()]
    best_path = str(directory / "best.npz")
    best, selected, step = math.inf, 0, 0
    started = time.perf_counter()
    with (directory / "history.tsv").open("w") as history, (directory / "batches.tsv").open("w") as batches:
        history.write("epoch\tfit_preupdate_loss\tvalidation_loss\tvalidation_mse\tvalidation_jacobian_squared\n")
        batches.write("epoch\tstep\trow\tsource_id\n")
        for epoch in range(1, epochs + 1):
            order.begin_epoch()
            total, seen = 0.0, 0
            while True:
                rows = list(order.next())
                if not rows:
                    break
                images = nn.take_rows(pools[0].images, rows)
                # Forward: MSE + weight * exact Jacobian penalty, differentiated through h AND W.
                loss = model.forward(ae.flatten(images)).loss
                if step == 0:
                    keep = nn.GradOptions()
                    keep.retain_graph = True
                    gradients = ae.check_gradients(model.parameters,
                                                   nn.grad(loss, model.parameters.parameters(), keep),
                                                   "every full contractive gradient finite",
                                                   "all4parameters receive complete gradients")
                    nn.save_npz(str(directory / "first_gradients.npz"), gradients)
                    nn.save_npz(str(directory / "first_batch.npz"), [("images", images)])
                value = loss.item()
                checks.require(math.isfinite(value) and value >= 0, "finite pre-update regularized loss")
                model.optimizer.minimize(loss)
                total += len(rows) * value
                seen += len(rows)
                batches.write("".join(f"{epoch}\t{step}\t{row}\t{fit_source[row]}\n" for row in rows))
                if step == 0:
                    nn.save_npz(str(directory / "first_update_arrays.npz"), ae.arrays(model.parameters))
                step += 1
            checks.require(seen == 1000, "every fit image once per variant epoch")
            result = model.evaluate(pools[1].images)
            value = result[2].item()
            history.write(f"{epoch}\t{total / seen:.17g}" + "".join(f"\t{result[i].item():.17g}" for i in range(2, 5))
                          + "\n")
            nn.save_npz(str(directory / f"epoch{epoch}_arrays.npz"), ae.arrays(model.parameters))
            if value < best:
                best, selected = value, epoch
                model.parameters.save(best_path)  # own full contractive validation objective selects checkpoint
            print(f"{name} epoch{epoch}:validation total/MSE/JacobianSquared={value:g}/{result[3].item():g}/"
                  f"{result[4].item():g}", flush=True)
    seconds = time.perf_counter() - started

    deployed = ContractiveAutoencoderModel(weight, False)
    deployed.parameters.load(best_path)  # fresh restore all contractive tensors
    before = ae.snapshot(deployed.parameters)
    arrays = []
    for split, label in enumerate(("fit", "validation", "test")):
        result = deployed.evaluate(pools[split].images)
        arrays += [(label + "_code", result[0]), (label + "_reconstruction", result[1]),
                   (label + "_jacobian_squared", result[5]),
                   (label + "_components", nn.stack([result[2], result[3], result[4]], 0).contiguous())]
        if split == 1:
            checks.near(result[2].item(), best, 2e-7, "fresh selected-loss restore")
    probe = ae.rows(pools[2], list(range(8)))
    actual_j = deployed.jacobian(probe.images)
    evaluation = deployed.evaluate(probe.images)
    jv, per = actual_j.values(), evaluation[5].values()
    norm_error = 0.0
    for n in range(8):
        total = 0.0
        for k in range(32 * 784):
            total += jv[n * 32 * 784 + k] ** 2
        norm_error = max(norm_error, abs(total - per[n]))
    checks.require(norm_error < 2e-5, "analytic contraction equals independent ALLautodiff Jacobian Frobenius sum")
    arrays += [("probe_images", probe.images), ("probe_jacobian", actual_j)]
    repeated = deployed.evaluate(pools[2].images)
    checks.require(ae.tensor_bytes(repeated[1]) == ae.tensor_bytes(ae.find(arrays, "test_reconstruction")),
                   "deterministic contractive reconstruction")
    checks.require(ae.tensor_bytes(deployed.decode(repeated[0])) == ae.tensor_bytes(repeated[1]),
                   "explicit contractive encode-decode exact")
    checks.require(before == ae.snapshot(deployed.parameters), "all4contractive parameters immutable")
    nn.save_npz(str(directory / "outputs.npz"), arrays)
    nn.save_npz(str(directory / "model_arrays.npz"), ae.arrays(deployed.parameters))
    (directory / "metrics.json").write_text(
        f'{{"weight":{deployed.weight:.17g},"epochs":{epochs},"updates":{step},"selected_epoch":{selected},'
        f'"test_mse":{ae.mse(repeated[1], pools[2].images):.17g},'
        f'"test_jacobian_squared":{repeated[4].item():.17g},"autodiff_norm_max_error":{norm_error:.17g},'
        f'"training_seconds":{seconds:.17g}}}\n')
    print(f"{name}:fullautodiff Jacobian-norm maxerror={norm_error:g},"
          f"freshrestore/decode/repeat/immutable4weights passed (training {seconds:g} s)")
    return seconds


def main(arguments=None):
    args = options(arguments, dict(epochs=3, data=str(data_root() / "saliency"), out="", init="",
                                   full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    epochs = args["epochs"]
    data = Path(args["data"])
    root = Path(args["out"]) if args["out"] else output_root(args["full"]) / "models/contractive_autoencoder"
    checks.require(0 < epochs <= 100, "three default epochs or explicit1..100")
    fixtures()

    pools = ae.load_saliency(data)
    root.mkdir(parents=True, exist_ok=True)
    variants = [("plain", 0.0), ("contractive", .01), ("strong", .1)]
    seconds = 0.0
    for name, weight in variants:
        seconds += train_variant(name, weight, pools, epochs, root, args["init"])
    base = nn.load_npz(str(root / "plain" / "initial_arrays.npz"))
    for name, _ in variants:
        initial = nn.load_npz(str(root / name / "initial_arrays.npz"))
        for key, value in base:
            checks.require(ae.tensor_bytes(ae.find(initial, key)) == ae.tensor_bytes(value),
                           "all3initializations byteidentical")
    ae.save_data(root / "data.npz", pools)
    ae.copy_manifests(data, root)
    mean = ae.constant(ae.pixel_mean(pools[0].images), 200)
    outputs = [nn.load_npz(str(root / name / "outputs.npz")) for name, _ in variants]
    html = [
        "<!doctype html><meta charset='utf-8'><title>Contractive autoencoder</title><style>body{font:16px "
        "system-ui;max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:"
        "25px 0}</style><h1>Contractive autoencoder:exact encoder Jacobian</h1><p>Original,plain,lambda.01,"
        "lambda.1,fit-only mean. Fixed[0,1]grayscale/nearest4,no quality filtering. 784->32sigmoid->784sigmoid/"
        f"50992parameters,{epochs}"
        "epochs each,matched initialization/order. Clean inputs reconstruct themselves;no corruption or class "
        "targets.</p><p>Loss=MSE+lambda*MEANexamples SUMunits,inputs (dh_j/dx_i)^2. For sigmoid h:Jacobian "
        "diag(h*(1-h))*W^T. Analytic exact norm differentiates through BOTH h and W,including direct weight "
        "dependence;it is not weight decay or a noisy-input proxy. Coordinates are original[0,1]pixels. "
        "Independent reverse-mode Jacobian rows validate the actual norm.</p><p>Each variant selects its OWN full "
        "validation objective;objective numbers across different lambdas are not a common-MSE ranking. A small "
        "encoder Jacobian can result from saturation or collapsed representations,not meaningful invariance. "
        "Reconstruction detail,latent variance and saturation accompany the metric. Local contraction does not "
        "prove robustness for finite changes or preservation of digit identity.</p><table><tr><th>Variant</th>"
        "<th>Test MSE</th><th>Mean Jacobian squared norm</th><th>Fraction saturated outside[.01,.99]</th><th>Mean "
        "code coordinate std</th></tr>"]
    for k, (name, _) in enumerate(variants):
        codes = ae.find(outputs[k], "test_code").values()
        components = ae.find(outputs[k], "test_components").values()
        saturated = sum(1 for value in codes if value < .01 or value > .99)
        standard_deviation = 0.0
        for j in range(32):
            mean_j = total = 0.0
            for n in range(200):
                mean_j += codes[n * 32 + j] / 200.
            for n in range(200):
                total += (codes[n * 32 + j] - mean_j) ** 2 / 200.
            standard_deviation += math.sqrt(total) / 32
        fraction = saturated / len(codes)
        html.append(f"<tr><td>{name}</td><td>{components[1]:g}</td><td>{components[2]:g}</td><td>{fraction:g}</td>"
                    f"<td>{standard_deviation:g}</td></tr>")
        print(f"{name}: test MSE={components[1]:g}, Jacobian squared={components[2]:g}, saturated={fraction:g}, "
              f"code std={standard_deviation:g}")
    html.append("</table><p>Short sigmoid runs may produce blurry average-looking outputs despite very small "
                "derivatives. Full-code variance/saturation and numerical evidence are saved. These are "
                "continuous32float coordinates,not a bit-rate compression or generative claim.</p>")
    ae.digit_gallery(root, pools[2],
                     [pools[2].images, ae.find(outputs[0], "test_reconstruction"),
                      ae.find(outputs[1], "test_reconstruction"), ae.find(outputs[2], "test_reconstruction"), mean],
                     html)
    (root / "index.html").write_text("".join(html))
    print(f"Three matched variants/{epochs} epochs,actual squared encoder Jacobian penalty;no semantic/"
          f"finite-robustness superiority claim.\ntraining time {seconds:g} s\nArtifacts:{root}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
