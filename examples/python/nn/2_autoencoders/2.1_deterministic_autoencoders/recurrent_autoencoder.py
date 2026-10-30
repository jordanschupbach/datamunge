# Native Python NN example. Sequence-to-sequence tanh recurrent autoencoder with a final-state-only bottleneck.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/2_autoencoders/2.1_deterministic_autoencoders/recurrent_autoencoder.py
"""Sequence-to-sequence tanh recurrent autoencoder with a final-state-only bottleneck.

Port of tflow's recurrent_autoencoder. Preparation (in ~/projects/tflow_examples):
just autoencoder-assets; data are read from $DATAMUNGE_DATA_DIR/saliency. Outputs
(history, npz arrays, states.npz, metrics.json, index.html and digit<k>.png galleries)
go to output/nn/models/recurrent_autoencoder. Checkpoints are best.npz files instead of
TF SaveV2 bundles. Options: epochs=N data=PATH out=PATH init=PATH (start from saved
parameters, e.g. tflow's initial_arrays.npz, to follow its trajectory exactly; by
default nn's own initialiser is used).
"""
import math
import os
import time
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_workflows import data_root, helper, options, output_root

checks = helper("checks")
ae = helper("autoencoder")


# ---- model (examples/nn/common/recurrent_autoencoder.hpp) --------------------------------

class RecurrentOutput:
    code = reconstruction = encoder_states = decoder_states = None


def recurrent_decode(p, code, steps=28, features=28):
    """Decoder only: repeat the code, NOT original/previous target rows; all steps share weights."""
    out = RecurrentOutput()
    out.code = code
    h = nn.zeros_like(code)
    states, rows = [], []
    for _ in range(steps):
        h = nn.rnn_step(p, "decoder", code, h)
        states.append(h)
        rows.append(nn.dense(p, "output", h, features, nn.Activation.Sigmoid))
    out.reconstruction = nn.stack(rows, 1)
    out.decoder_states = nn.stack(states, 1)
    return out


def recurrent_autoencoder(p, sequence, steps=28, features=28, units=32):
    """Tanh encoder over the rows of sequence [B, steps, features]; the final state is the only bridge."""
    checks.require(steps > 1 and features > 0 and 0 < units < steps * features, "strict sequence bottleneck")
    h = nn.zeros([sequence.size(0), units], sequence.dtype())
    states = []
    for t in range(steps):
        h = nn.rnn_step(p, "encoder", nn.select(sequence, 1, t), h)
        states.append(h)
    out = recurrent_decode(p, h, steps, features)
    out.encoder_states = nn.stack(states, 1)
    return out


def sequence_of(images):
    """[n, 28, 28, 1] viewed as an ordered 28-row sequence, without resize."""
    return nn.reshape(images, [images.size(0), 28, 28])


def sequence_mse(reconstruction, sequence):
    return nn.mean(nn.square(nn.sub(reconstruction, sequence)))


class RecurrentAutoencoderModel:
    def __init__(self, training=True):
        self.parameters = nn.ParameterStore(1871)
        with nn.no_grad():
            recurrent_autoencoder(self.parameters, nn.zeros([1, 28, 28]))
        self.optimizer = ae.make_adam(self.parameters) if training else None

    def evaluate(self, images):
        """[code, reconstruction [n, 28, 28], MSE]"""
        with nn.no_grad():
            x = sequence_of(images)
            n = recurrent_autoencoder(self.parameters, x)
            return [n.code, n.reconstruction, sequence_mse(n.reconstruction, x)]

    def decode(self, codes):
        with nn.no_grad():
            return recurrent_decode(self.parameters, codes).reconstruction

    def states(self, images):
        with nn.no_grad():
            n = recurrent_autoencoder(self.parameters, sequence_of(images))
            return [n.encoder_states, n.decoder_states]


# ---- experiment ------------------------------------------------------------------------------

def train_recurrent(pools, epochs, directory, init):
    """Trains, selects on validation MSE, and checks the restored model; returns training seconds."""
    directory.mkdir(parents=True, exist_ok=True)
    model = RecurrentAutoencoderModel()
    if init:
        model.parameters.load(init)
    checks.require(model.parameters.size() == 8 and model.parameters.count() == 4956,
                   "shared-over-time recurrent parameter census")
    nn.save_npz(str(directory / "initial_arrays.npz"), ae.arrays(model.parameters))
    initial = model.evaluate(pools[2].images)
    nn.save_npz(str(directory / "initial_outputs.npz"), [("code", initial[0]), ("reconstruction", initial[1])])
    order = nn.BatchOrder(1000, 32, 1873)
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
                # Full backpropagation through all 28 encoder + 28 decoder steps.
                sequence = sequence_of(inputs)
                loss = sequence_mse(recurrent_autoencoder(model.parameters, sequence).reconstruction, sequence)
                if step == 0:
                    keep = nn.GradOptions()
                    keep.retain_graph = True
                    gradients = ae.check_gradients(model.parameters,
                                                   nn.grad(loss, model.parameters.parameters(), keep),
                                                   "all model MSE derivatives finite",
                                                   "all model tensors receive full MSE gradients")
                    nn.save_npz(str(directory / "first_gradients.npz"), gradients)
                    nn.save_npz(str(directory / "first_batch.npz"), [("images", inputs)])
                value = loss.item()
                checks.require(math.isfinite(value) and value >= 0, "finite pre-update MSE")
                model.optimizer.minimize(loss)
                total += len(rows) * value
                seen += len(rows)
                batches.write("".join(f"{epoch}\t{step}\t{row}\t{fit_source[row]}\n" for row in rows))
                if step == 0:
                    nn.save_npz(str(directory / "first_update_arrays.npz"), ae.arrays(model.parameters))
                step += 1
            checks.require(seen == 1000, "all fitting records exactly once per epoch")
            value = model.evaluate(pools[1].images)
            mse = ae.mse(value[1], pools[1].images)
            history.write(f"{epoch}\t{total / seen:.17g}\t{mse:.17g}\n")
            nn.save_npz(str(directory / f"epoch{epoch}_arrays.npz"), ae.arrays(model.parameters))
            if mse < best:
                best, selected = mse, epoch
                model.parameters.save(best_path)  # common validation reconstruction MSE selects the model
            print(f"recurrent epoch{epoch}:validationMSE={mse:g}", flush=True)
    seconds = time.perf_counter() - started
    deployed = RecurrentAutoencoderModel(False)
    deployed.parameters.load(best_path)  # fresh recurrent inference restore
    before = ae.snapshot(deployed.parameters)
    probe = ae.rows(pools[2], range(8)).images
    states = deployed.states(probe)
    nn.save_npz(str(directory / "states.npz"),
                [("images", probe), ("encoder_states", states[0]), ("decoder_states", states[1])])
    arrays = []
    for split, name in enumerate(("fit", "validation", "test")):
        out = deployed.evaluate(pools[split].images)
        arrays += [(name + "_code", out[0]), (name + "_reconstruction", out[1])]
        if split == 1:
            checks.near(ae.mse(out[1], pools[split].images), best, 1e-8, "fresh selected validation MSE")
    test, repeat = deployed.evaluate(pools[2].images), deployed.evaluate(pools[2].images)
    checks.require(ae.tensor_bytes(test[1]) == ae.tensor_bytes(repeat[1])
                   and ae.tensor_bytes(deployed.decode(test[0])) == ae.tensor_bytes(test[1]),
                   "deterministic exact encode-decode")
    zero = deployed.decode(nn.zeros([200, 32]))
    shuffled = deployed.decode(nn.concat([nn.slice_axis(test[0], 0, 1, 200), nn.slice_axis(test[0], 0, 0, 1)], 0))
    arrays += [("zero_reconstruction", zero), ("shuffled_reconstruction", shuffled)]
    # Rows in reverse order (y -> 27 - y): the ordered input is disrupted, output positions unchanged.
    reversed_rows = nn.from_vector_int64([27 - y for y in range(28)], [28])
    reversed_images = nn.gather(sequence_of(pools[2].images), reversed_rows, 1).contiguous()
    reverse_out = deployed.evaluate(reversed_images)
    arrays += [("reversed_code", reverse_out[0]), ("reversed_reconstruction", reverse_out[1])]
    one = deployed.evaluate(nn.take_rows(pools[2].images, [0]))
    error = ae.max_abs_diff(one[1], test[1], 784)
    checks.require(error < 1e-6, "single/batch recurrent inference parity")
    checks.require(before == ae.snapshot(deployed.parameters), "all deployed parameters immutable through code controls")
    nn.save_npz(str(directory / "outputs.npz"), arrays)
    nn.save_npz(str(directory / "model_arrays.npz"), ae.arrays(deployed.parameters))
    test_mse = ae.mse(test[1], pools[2].images)
    (directory / "metrics.json").write_text(
        f'{{"parameters":{model.parameters.count()},"epochs":{epochs},"updates":{step},"selected_epoch":{selected},'
        f'"test_mse":{test_mse:.17g},"zero_mse":{ae.mse(zero, pools[2].images):.17g},'
        f'"shuffled_mse":{ae.mse(shuffled, pools[2].images):.17g},"single_batch_max_error":{error:.17g},'
        f'"training_seconds":{seconds:.17g}}}\n')
    print(f"recurrent:selected{selected},parameters={model.parameters.count()},testMSE={test_mse:g},"
          "freshrestore/decode/repeat/immutable parameters passed")
    return seconds


def main(arguments=None):
    args = options(arguments, dict(epochs=3, data=str(data_root() / "saliency"), out="", init="",
                                   full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    epochs = args["epochs"]
    data = Path(args["data"])
    root = Path(args["out"]) if args["out"] else output_root(args["full"]) / "models/recurrent_autoencoder"
    checks.require(0 < epochs <= 100, "three default epochs or explicit1..100")
    pools = ae.load_saliency(data)
    root.mkdir(parents=True, exist_ok=True)
    seconds = train_recurrent(pools, epochs, root, args["init"])
    ae.save_data(root / "data.npz", pools)
    ae.copy_manifests(data, root)
    t = nn.load_npz(str(root / "outputs.npz"))
    mean = ae.constant(ae.pixel_mean(pools[0].images), 200)
    reconstruction = ae.find(t, "test_reconstruction")
    html = [
        "<!doctype html><meta charset='utf-8'><title>Recurrent sequence autoencoder</title><style>body{font:16px "
        "system-ui;max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:"
        "25px 0}</style><h1>Sequence-to-sequence recurrent autoencoder</h1><p>Original,reconstruction,fit-only mean,"
        "absolute error,zero-code,shuffled-code,reversed encoder input. Fixed[0,1] grayscale,nearest4,no error "
        "amplification. First test record per digit,unfiltered.</p><p>MNIST is a sequence of28rows of28pixels. Tanh "
        "encoder32,final-state-only code32,repeated into tanh decoder32 with zero initial state,shared sigmoid row "
        "projection. Distinct encoder/decoder weights,shared across time;4956parameters. No teacher forcing,"
        "original-input bypass,attention or target rows in decoder. Full backpropagation through all28encoder+"
        "28decoder steps. This is fixed-length row-sequence reconstruction,not language translation or "
        f"variable-length prediction.</p><p>{epochs}"
        " epochs,Adam.003,batch32,1000fit/200validation/200test. Validation selects the completed epoch. Labels only "
        "balance/captions. Reversing encoder rows deliberately disrupts the ordered input while output positions stay "
        "unchanged;not an equivalent reconstruction task. Continuous code is not a compressed bitstream. Short "
        f"training may produce generic blurry rows or lose digit identity.</p><p>TestMSE="
        f"{ae.mse(reconstruction, pools[2].images):g},fitmean={ae.mse(mean, pools[2].images):g}.</p>"]
    ae.digit_gallery(root, pools[2],
                     [pools[2].images, reconstruction, mean,
                      nn.abs(nn.sub(nn.reshape(reconstruction, [200, 784]), ae.flatten(pools[2].images))),
                      ae.find(t, "zero_reconstruction"), ae.find(t, "shuffled_reconstruction"),
                      ae.find(t, "reversed_reconstruction")], html)
    (root / "index.html").write_text("".join(html))
    print(f"Full recurrent sequence reconstruction/fresh restore/latent controls verified.\ntraining time {seconds:g} s"
          f"\nArtifacts:{root}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
