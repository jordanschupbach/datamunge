# Native Python NN example. Activation-L1 and Bernoulli-KL sparse autoencoders with matched controls.
"""CMD: just python-example sparse_autoencoder
BOTH activation-L1 and mean-unit Bernoulli-KL sparse autoencoders with matched controls.
Four matched 64-sigmoid variants, three epochs/default, no label targets.

Port of tflow's sparse_autoencoder (model and penalties: examples/nn/common/sparse_autoencoder.hpp).
Preparation (in ~/projects/tflow_examples): just autoencoder-assets; data are read from
$DATAMUNGE_DATA_DIR/saliency. Outputs (plain/, l1/, kl/, combined/ training records, data.npz,
index.html and digit<k>.png galleries) go to output/nn/models/sparse_autoencoder. Checkpoints
are best.npz files instead of TF SaveV2 bundles; metrics.json gains "training_seconds".
Options: epochs=N data=PATH out=PATH init=PATH (start all four variants from saved
parameters, e.g. tflow's plain/initial_arrays.npz, to follow its trajectories exactly; by
default nn's own initialiser is used, identically for every variant).
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


# ---- model and penalties (examples/nn/common/sparse_autoencoder.hpp) -------------------------

class SparsityPenalty:
    l1 = kl = unit_mean = clipped_mean = total = None


def activation_sparsity(code, l1_weight, kl_weight, rho=.05, epsilon=1e-5):
    """l1 = mean |code|; kl = mean over units of KL(Bernoulli(rho) || Bernoulli(clamped batch mean));
    total = l1_weight l1 + kl_weight kl. All constants are float32 values as in tflow."""
    l1_weight, kl_weight, rho, epsilon = (float32(v) for v in (l1_weight, kl_weight, rho, epsilon))
    checks.require(math.isfinite(l1_weight) and l1_weight >= 0 and math.isfinite(kl_weight) and kl_weight >= 0
                   and 0 < rho < 1 and 0 < epsilon < .5, "valid sparsity weights/target/clamp")
    out = SparsityPenalty()
    out.l1 = nn.mean(nn.abs(code))
    out.unit_mean = nn.mean(code, [0])
    out.clipped_mean = nn.clip(out.unit_mean, epsilon, float32(1 - epsilon))
    complement = float32(1 - rho)
    positive = nn.mul(nn.log(nn.div(rho, out.clipped_mean)), rho)
    negative = nn.mul(nn.log(nn.div(complement, nn.sub(1.0, out.clipped_mean))), complement)
    out.kl = nn.mean(nn.add(positive, negative), [0])
    out.total = nn.add(nn.mul(out.l1, l1_weight), nn.mul(out.kl, kl_weight))
    return out


class SparseAeOutput:
    code = reconstruction = mse = loss = penalty = None


def sparse_autoencoder(p, image, l1_weight, kl_weight):
    """784 -> 64 sigmoid -> 784 sigmoid; loss = MSE + sparsity penalties."""
    out = SparseAeOutput()
    out.code = nn.dense(p, "encoder", image, 64, nn.Activation.Sigmoid)
    out.reconstruction = nn.dense(p, "decoder", out.code, 784, nn.Activation.Sigmoid)
    out.mse = ae.reconstruction_mse(out.reconstruction, image)
    out.penalty = activation_sparsity(out.code, l1_weight, kl_weight)
    out.loss = nn.add(out.mse, out.penalty.total)
    return out


class SparseAutoencoderModel:
    def __init__(self, l1, kl, training=True):
        self.l1_weight, self.kl_weight = float32(l1), float32(kl)
        self.parameters = nn.ParameterStore(1601)
        with nn.no_grad():
            sparse_autoencoder(self.parameters, nn.zeros([1, 784]), self.l1_weight, self.kl_weight)
        self.optimizer = ae.make_adam(self.parameters) if training else None

    def net(self, images):
        return sparse_autoencoder(self.parameters, ae.flatten(images), self.l1_weight, self.kl_weight)

    def evaluate(self, images):
        """[code, reconstruction, loss, mse, l1, kl, unit_mean]"""
        with nn.no_grad():
            n = self.net(images)
            return [n.code, n.reconstruction, n.loss, n.mse, n.penalty.l1, n.penalty.kl, n.penalty.unit_mean]

    def decode(self, codes):
        with nn.no_grad():
            return nn.dense(self.parameters, "decoder", codes, 784, nn.Activation.Sigmoid)


def fixtures():
    values = [float32(v) for v in (0, .05, .2, 1, 0, .05, .6, 1, 0, .05, 1, 1)]
    h = nn.from_vector(values, [3, 4]).requires_grad_()
    p = activation_sparsity(h, .1, .2)
    gradient = nn.grad(p.total, [h])[0].values()
    low, high = float32(1e-5), float32(1 - float32(1e-5))
    rho, complement, w1, w2 = float32(.05), float32(1 - float32(.05)), float32(.1), float32(.2)
    l1 = kl = 0.0
    for v in values:
        l1 += abs(v) / 12
    for j in range(4):
        q = 0.0
        for b in range(3):
            q += values[b * 4 + j] / 3.
        clipped = min(max(q, low), high)
        kl += (rho * math.log(rho / clipped) + complement * math.log(complement / (1 - clipped))) / 4
        for b in range(3):
            expected = 0.0 if values[b * 4 + j] == 0 else w1 / 12
            if low < q < high:
                expected += w2 * (-rho / q + complement / (1 - q)) / 12
            checks.near(gradient[b * 4 + j], expected, 3e-7, "all12L1+KLpartials including low/high clamp and targetrho")
    checks.near(p.l1.item(), l1, 4e-8, "activation L1 mean not parameter penalty")
    checks.near(p.kl.item(), kl, 4e-6, "clamped perunit Bernoulli KL mean including extreme means")
    checks.near(p.total.item(), w1 * l1 + w2 * kl, 1e-6, "weighted sum of distinct penalties")
    checks.throws(lambda: activation_sparsity(h, .1, .2, 1.), "invalid targetrho rejected")
    print("analytic L1/KL means/ALL12derivatives/zero/targetrho/clamped0+1means/invalidrho fixtures passed")


def train_variant(name, l1, kl, pools, epochs, root, init):
    """Trains one matched sparse variant on its own regularized objective; returns training seconds."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    model = SparseAutoencoderModel(l1, kl)
    if init:
        model.parameters.load(init)
    checks.require(model.parameters.count() == 101200 and model.parameters.size() == 4,
                   "independent784x64+64+64x784+784 inventory")
    nn.save_npz(str(directory / "initial_arrays.npz"), ae.arrays(model.parameters))
    order = nn.BatchOrder(1000, 32, 1607)
    fit_source = [int(v) for v in pools[0].source.values()]
    best_path = str(directory / "best.npz")
    best, selected, step = math.inf, 0, 0
    started = time.perf_counter()
    with (directory / "history.tsv").open("w") as history, (directory / "batches.tsv").open("w") as batches:
        history.write("epoch\tfit_preupdate_loss\tvalidation_loss\tvalidation_mse\tvalidation_l1\tvalidation_kl\n")
        batches.write("epoch\tstep\trow\tsource_id\n")
        for epoch in range(1, epochs + 1):
            order.begin_epoch()
            total, seen = 0.0, 0
            while True:
                rows = list(order.next())
                if not rows:
                    break
                images = nn.take_rows(pools[0].images, rows)
                # Forward: reconstruction MSE + L1/KL activation penalties of this batch.
                loss = model.net(images).loss
                if step == 0:
                    keep = nn.GradOptions()
                    keep.retain_graph = True
                    nn.save_npz(str(directory / "first_gradients.npz"),
                                ae.check_gradients(model.parameters, nn.grad(loss, model.parameters.parameters(), keep),
                                                   "finite all penalized parameter gradients",
                                                   "all4tensors receive complete loss gradients"))
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
            checks.require(seen == 1000, "each fitting image once per variant epoch")
            validation = model.evaluate(pools[1].images)
            value = validation[2].item()
            history.write(f"{epoch}\t{total / seen:.17g}"
                          + "".join(f"\t{validation[i].item():.17g}" for i in range(2, 6)) + "\n")
            nn.save_npz(str(directory / f"epoch{epoch}_arrays.npz"), ae.arrays(model.parameters))
            if value < best:
                best, selected = value, epoch
                model.parameters.save(best_path)  # validation own full regularized objective selects checkpoint
            print(f"{name} epoch{epoch}:validation total/MSE/L1/KL={value:g}/{validation[3].item():g}/"
                  f"{validation[4].item():g}/{validation[5].item():g}", flush=True)
    seconds = time.perf_counter() - started

    deployed = SparseAutoencoderModel(l1, kl, False)
    deployed.parameters.load(best_path)  # fresh restore matched sparse variant
    before = ae.snapshot(deployed.parameters)
    arrays = []
    for split, label in enumerate(("fit", "validation", "test")):
        result = deployed.evaluate(pools[split].images)
        components = [result[i + 2].item() for i in range(4)]
        arrays += [(label + "_code", result[0]), (label + "_reconstruction", result[1]),
                   (label + "_unit_mean", result[6]), (label + "_components", nn.from_vector(components, [4]))]
        if split == 1:
            checks.near(result[2].item(), best, 1e-7, "selected regularized validation loss restored")
    test = deployed.evaluate(pools[2].images)
    checks.require(ae.tensor_bytes(test[1]) == ae.tensor_bytes(ae.find(arrays, "test_reconstruction")),
                   "deterministic sparse reconstruction")
    checks.require(ae.tensor_bytes(deployed.decode(test[0])) == ae.tensor_bytes(test[1]),
                   "explicit sparse encode-decode exact")
    one = deployed.evaluate(ae.rows(pools[2], [0]).images)
    error = ae.max_abs_diff(one[1], test[1], 784)
    checks.require(error < 1e-6, "inference single/batch parity despite batch-dependent KL objective")
    checks.require(before == ae.snapshot(deployed.parameters), "all4sparse parameters immutable during evaluation")
    nn.save_npz(str(directory / "outputs.npz"), arrays)
    nn.save_npz(str(directory / "model_arrays.npz"), ae.arrays(deployed.parameters))
    code = ae.find(arrays, "test_code").contiguous().values()
    threshold = float32(.1)
    mean = 0.0
    for v in code:
        mean += v
    low = sum(1 for v in code if v < threshold)
    mean /= len(code)
    fraction = low / len(code)
    mse = ae.mse(test[1], pools[2].images)
    (directory / "metrics.json").write_text(
        f'{{"l1_weight":{deployed.l1_weight:.17g},"kl_weight":{deployed.kl_weight:.17g},'
        f'"rho":{float32(.05):.17g},"epsilon":{float32(1e-5):.17g},"epochs":{epochs},'
        f'"selected_epoch":{selected},"updates":{step},"test_mse":{mse:.17g},"test_mean_activation":{mean:.17g},'
        f'"test_fraction_below_point1":{fraction:.17g},"single_batch_max_error":{error:.17g},'
        f'"training_seconds":{seconds:.17g}}}\n')
    print(f"{name} selected{selected}:testMSE={mse:g},meanactivation={mean:g},fractionbelow.1={fraction:g}"
          ";freshrestore/decode/repeat/batch/immutable4weights passed")
    return seconds


def main(arguments=None):
    args = options(arguments, dict(epochs=3, data=str(data_root() / "saliency"), out="", init="",
                                   full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    epochs = args["epochs"]
    data = Path(args["data"])
    directory = Path(args["out"]) if args["out"] else output_root(args["full"]) / "models/sparse_autoencoder"
    checks.require(0 < epochs <= 100, "three default epochs or explicit1..100")
    fixtures()

    pools = ae.load_saliency(data)
    variants = [("plain", 0.0, 0.0), ("l1", .05, 0.0), ("kl", 0.0, .05), ("combined", .05, .05)]
    seconds = 0.0
    for name, l1, kl in variants:
        seconds += train_variant(name, l1, kl, pools, epochs, directory, args["init"])
    original = nn.load_npz(str(directory / "plain" / "initial_arrays.npz"))
    for name, _, _ in variants:
        initial = nn.load_npz(str(directory / name / "initial_arrays.npz"))
        for key, value in original:
            checks.require(ae.tensor_bytes(ae.find(initial, key)) == ae.tensor_bytes(value),
                           "ALL4variants identical initial parameters")
    ae.save_data(directory / "data.npz", pools)
    ae.copy_manifests(data, directory)
    mean = ae.constant(ae.pixel_mean(pools[0].images), 200)
    html = [
        "<!doctype html><meta charset='utf-8'><title>L1 and KL sparse autoencoders</title><style>body{font:16px "
        "system-ui;max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:"
        "25px 0}</style><h1>Sparse autoencoders:activation L1 and KL</h1><p>Original,plain,L1,KL,combined,fit-only "
        f"mean. ALL use784->64sigmoid->784sigmoid/101200parameters,identical initialization and batch order,{epochs}"
        "epochs PER variant. Fixed[0,1]intensity/nearest4,no quality-filtering. Labels do not enter training or "
        "selection.</p><p>Total=MSE+lambda*mean(abs(code))+beta*mean_unit KL(rho||mean_batch(code)). rho=.05,"
        "lambda=.05 for L1,beta=.05 for KL. Unit means clamp to[1e-5,1-1e-5]before logs;clamp derivative is flat "
        "outside. KL is a penalty on each unit's BATCH mean,not each image's activation,not a softmax/class KL. "
        "Minibatch KL and full-pool KL differ. Each variant selects on its OWN full validation objective,not "
        "comparable objective numbers across penalties.</p><p>Sigmoid codes are generally positive:small activation "
        "is not exact zero sparsity. Mean activation and fraction below.1 are descriptive,threshold not used in "
        "optimization. Stronger sparsity can worsen reconstruction or collapse features;three short epochs do not "
        "guarantee rho is reached. MSE alone can favor blurry averages. This is not a proof of semantic features,"
        "denoising,generative modeling or bit-rate compression.</p>"]
    panels = [pools[2].images]
    for name, _, _ in variants:
        raw = nn.load_npz(str(directory / name / "outputs.npz"))
        c = ae.find(raw, "test_components").values()
        html.append(f"<p>{name}:test total/MSE/L1/KL={c[0]:g}/{c[1]:g}/{c[2]:g}/{c[3]:g}"
                    ". Mean activation equals L1 for these nonnegative sigmoid codes. rho=.05 remains a target rather "
                    "than an attained guarantee;per-unit means/codes and threshold metrics are saved in variant "
                    "artifacts.</p>")
        panels.append(ae.find(raw, "test_reconstruction"))
    panels.append(mean)
    ae.digit_gallery(directory, pools[2], panels, html)
    (directory / "index.html").write_text("".join(html))
    print("ALL4variants identicalinitial/sourceorder,96updates each at default3epochs. L1/KL tradeoffs saved "
          f"without superiority claim.\ntraining time {seconds:g} s\nArtifacts:{directory}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
