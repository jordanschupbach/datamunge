# Native Python NN example. Visible-only transformer masked autoencoder (75 percent patch masking).
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/2_autoencoders/2.1_deterministic_autoencoders/masked_autoencoder.py
"""Visible-only transformer MAE with 75 percent masking and a smaller mask-token decoder.

Port of tflow's masked_autoencoder. Preparation (in ~/projects/tflow_examples):
just autoencoder-assets; data are read from $DATAMUNGE_DATA_DIR/saliency. Outputs
(history, per-epoch masks and arrays, probe.npz, outputs.npz, metrics.json, index.html
and digit<k>.png galleries) go to output/nn/models/masked_autoencoder. The patch masks
are the same hash-derived selections as tflow's (bit-identical). Checkpoints are
best.npz files instead of TF SaveV2 bundles. Options: epochs=N data=PATH out=PATH
init=PATH (start from saved parameters, e.g. tflow's initial_arrays.npz, to follow its
trajectory exactly; by default nn's own initialiser is used).
"""
import math
import os
import time
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_workflows import data_root, helper, options, output_root

checks = helper("checks")
ae = helper("autoencoder")
mae = helper("masked_autoencoder")


def loss_checks():
    mask = [0.0 if p < 4 else 1.0 for b in range(2) for p in range(16)]
    patches = nn.full([2, 16, 49], .7)
    y = [.3] * (2 * 16 * 49)
    prediction = nn.from_vector(y, [2, 16, 49]).requires_grad_()
    m = nn.from_vector(mask, [2, 16])
    loss = mae.mae_loss(prediction, patches, m)
    gradient = nn.grad(loss, [prediction])[0].values()
    checks.near(loss.item(), .16, 3e-7, "loss denominator counts ONLY12hidden patches per image")
    for i in range(2 * 16 * 49):
        expected = 0 if (i // 49) % 16 < 4 else -.8 / (2 * 12 * 49)
        checks.near(gradient[i], expected, 2e-10, "all1568prediction derivatives;visible EXACTzero")
    before = loss.item()
    for b in range(2):
        for p in range(4):
            for i in range(49):
                y[(b * 16 + p) * 49 + i] = .99
    checks.near(mae.mae_loss(nn.from_vector(y, [2, 16, 49]), patches, m).item(), before, 0,
                "visible predictions excluded exactly")
    print("masked loss/denominator/ALL1568derivatives/visible perturbation invariance passed")


def main(arguments=None):
    args = options(arguments, dict(epochs=3, data=str(data_root() / "saliency"), out="", init="",
                                   full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    epochs = args["epochs"]
    data = Path(args["data"])
    root = Path(args["out"]) if args["out"] else output_root(args["full"]) / "models/masked_autoencoder"
    checks.require(0 < epochs <= 100, "three default epochs or explicit1..100")
    loss_checks()

    pools = ae.load_saliency(data)
    root.mkdir(parents=True, exist_ok=True)
    model = mae.MaskedAutoencoderModel()
    if args["init"]:
        model.parameters.load(args["init"])
    checks.require(model.parameters.size() == 35 and model.parameters.count() == 13649,
                   "independent tiny transformer parameter census")
    nn.save_npz(str(root / "initial_arrays.npz"), ae.arrays(model.parameters))
    test_selection = mae.mae_selection(pools[2].source)
    initial = model.evaluate(pools[2].images, test_selection)
    nn.save_npz(str(root / "initial_outputs.npz"), [("code", initial[0]), ("prediction", initial[1])])
    order = nn.BatchOrder(1000, 32, 1889)
    fit_source = [int(v) for v in pools[0].source.values()]
    best_path = str(root / "best.npz")
    best, selected, step = math.inf, 0, 0
    started = time.perf_counter()
    with (root / "history.tsv").open("w") as history, (root / "batches.tsv").open("w") as batches:
        history.write("epoch\tfit_preupdate_masked_mse\tvalidation_masked_mse\n")
        batches.write("epoch\tstep\trow\tsource_id\n")
        for epoch in range(1, epochs + 1):
            order.begin_epoch()
            epoch_selection = mae.mae_selection(pools[0].source, epoch)
            nn.save_npz(str(root / f"epoch{epoch}_mask.npz"), [("selection", epoch_selection)])
            total, seen = 0.0, 0
            while True:
                rows = list(order.next())
                if not rows:
                    break
                inputs = nn.take_rows(pools[0].images, rows)
                selection = nn.take_rows(epoch_selection, rows)
                # Visible-only encode, mask-token decode, hidden-patch-only loss.
                net = mae.masked_autoencoder(model.parameters, inputs, selection)
                loss = mae.mae_loss(net.prediction, net.patches, net.mask)
                if step == 0:
                    keep = nn.GradOptions()
                    keep.retain_graph = True
                    gradients = ae.check_gradients(model.parameters,
                                                   nn.grad(loss, model.parameters.parameters(), keep),
                                                   "all transformer gradients finite",
                                                   "ALL35tensors receive nonzero masked-loss gradients")
                    nn.save_npz(str(root / "first_gradients.npz"), gradients)
                    nn.save_npz(str(root / "first_batch.npz"), [("images", inputs), ("selection", selection)])
                value = loss.item()
                checks.require(math.isfinite(value) and value >= 0, "finite masked MSE")
                model.optimizer.minimize(loss)
                total += len(rows) * value
                seen += len(rows)
                batches.write("".join(f"{epoch}\t{step}\t{row}\t{fit_source[row]}\n" for row in rows))
                if step == 0:
                    nn.save_npz(str(root / "first_update_arrays.npz"), ae.arrays(model.parameters))
                step += 1
            checks.require(seen == 1000, "every fitting record once per epoch")
            val_selection = mae.mae_selection(pools[1].source)
            value = model.evaluate(pools[1].images, val_selection)
            mse = mae.mae_mse(value[1], pools[1].images, val_selection)
            checks.near(value[2].item(), mse, 2e-7, "graph and independent native hidden-only MSE")
            history.write(f"{epoch}\t{total / seen:.17g}\t{mse:.17g}\n")
            nn.save_npz(str(root / f"epoch{epoch}_arrays.npz"), ae.arrays(model.parameters))
            if mse < best:
                best, selected = mse, epoch
                model.parameters.save(best_path)  # fixed masked validation MSE selects epoch
            print(f"MAE epoch{epoch}:validation masked MSE={mse:g}", flush=True)
    seconds = time.perf_counter() - started

    deployed = mae.MaskedAutoencoderModel(False)
    deployed.parameters.load(best_path)  # fresh inference MAE restore
    before = ae.snapshot(deployed.parameters)
    saved = []
    for split, name in enumerate(("fit", "validation", "test")):
        selection = mae.mae_selection(pools[split].source)
        out = deployed.evaluate(pools[split].images, selection)
        saved += [(name + "_code", out[0]), (name + "_prediction", out[1]), (name + "_selection", selection)]
        if split == 1:
            checks.near(mae.mae_mse(out[1], pools[split].images, selection), best, 1e-8,
                        "fresh selected validation metric")
    test = deployed.evaluate(pools[2].images, test_selection)
    repeat = deployed.evaluate(pools[2].images, test_selection)
    checks.require(ae.tensor_bytes(test[1]) == ae.tensor_bytes(repeat[1])
                   and ae.tensor_bytes(deployed.decode(test[0], test_selection)) == ae.tensor_bytes(test[1]),
                   "deterministic exact target-free code decoding")
    images = pools[2].images.values()
    sel = test_selection.values()
    altered, shifted, permuted = [0.0] * len(images), [0.0] * len(images), [0.0] * len(sel)
    for b in range(200):
        source = ((b + 1) % 200) * 784
        shifted[b * 784:(b + 1) * 784] = images[source:source + 784]
        for p in range(16):
            hidden_patch = mae.mae_hidden(sel, b, p)
            for j in mae.PIXEL[p]:
                index = b * 784 + j
                altered[index] = 1 - images[index] if hidden_patch else images[index]
        for k in range(4):
            for p in range(16):
                permuted[b * 64 + k * 16 + p] = sel[b * 64 + (3 - k) * 16 + p]
    image_shape = list(pools[2].images.shape())
    hidden = deployed.evaluate(nn.from_vector(altered, image_shape), test_selection)
    checks.require(ae.tensor_bytes(hidden[0]) == ae.tensor_bytes(test[0])
                   and ae.tensor_bytes(hidden[1]) == ae.tensor_bytes(test[1]),
                   "changing EVERY hidden input pixel leaves codes/predictions BITEXACT;no target leakage")
    shuffled = deployed.evaluate(nn.from_vector(shifted, image_shape), test_selection)
    permutation = deployed.evaluate(pools[2].images, nn.from_vector(permuted, [200, 4, 16]))
    second_selection = mae.mae_selection(pools[2].source, 0, 1913)
    second = deployed.evaluate(pools[2].images, second_selection)
    zero_prediction = deployed.decode(nn.zeros([200, 4, 32]), test_selection)
    saved += [("zero_prediction", zero_prediction), ("shuffled_prediction", shuffled[1]),
              ("second_selection", second_selection), ("second_prediction", second[1]),
              ("permuted_prediction", permutation[1])]
    single = deployed.evaluate(nn.take_rows(pools[2].images, [0]), nn.take_rows(test_selection, [0]))
    single_error = ae.max_abs_diff(single[1], test[1], 784)
    permutation_error = ae.max_abs_diff(permutation[1], test[1], 200 * 784)
    checks.require(single_error < 2e-6 and permutation_error < 2e-6,
                   "batch parity and visible token permutation/restoration equivariance")
    checks.require(before == ae.snapshot(deployed.parameters),
                   "inference and controls leave ALL35parameter tensors immutable")
    first8 = list(range(8))
    probe = nn.take_rows(pools[2].images, first8)
    probe_selection = nn.take_rows(test_selection, first8)
    stages = deployed.probe(probe, probe_selection)
    nn.save_npz(str(root / "probe.npz"), [("images", probe), ("selection", probe_selection), ("patches", stages[0]),
                                          ("visible", stages[1]), ("mask", stages[2]),
                                          ("encoder_attention", stages[3]), ("decoder_attention", stages[4])])
    nn.save_npz(str(root / "outputs.npz"), saved)
    nn.save_npz(str(root / "model_arrays.npz"), ae.arrays(deployed.parameters))
    ae.save_data(root / "data.npz", pools)
    ae.copy_manifests(data, root)
    test_mse = mae.mae_mse(test[1], pools[2].images, test_selection)
    zero_mse = mae.mae_mse(zero_prediction, pools[2].images, test_selection)
    shuffled_mse = mae.mae_mse(shuffled[1], pools[2].images, test_selection)
    second_mse = mae.mae_mse(second[1], pools[2].images, second_selection)
    (root / "metrics.json").write_text(
        f'{{"parameters":13649,"epochs":{epochs},"updates":{step},"selected_epoch":{selected},'
        f'"test_masked_mse":{test_mse:.17g},"zero_masked_mse":{zero_mse:.17g},'
        f'"shuffled_masked_mse":{shuffled_mse:.17g},"second_masked_mse":{second_mse:.17g},'
        f'"single_batch_max_error":{single_error:.17g},"visible_permutation_max_error":{permutation_error:.17g},'
        f'"training_seconds":{seconds:.17g}}}\n')

    # Gallery panels: original, visible input (gray hidden), raw prediction, visible-copy composite,
    # fit-only mean, hidden-only absolute error, zero-code raw prediction.
    predicted = mae.mae_image(test[1]).values()
    visible, composite, error = [0.0] * len(images), [0.0] * len(images), [0.0] * len(images)
    for b in range(200):
        for p in range(16):
            hidden_patch = mae.mae_hidden(sel, b, p)
            for j in mae.PIXEL[p]:
                index = b * 784 + j
                visible[index] = .5 if hidden_patch else images[index]
                composite[index] = predicted[index] if hidden_patch else images[index]
                # |float32 difference|: the exact double difference rounds to the same float.
                error[index] = abs(predicted[index] - images[index]) if hidden_patch else 0.0
    html = [
        "<!doctype html><meta charset='utf-8'><title>Tiny transformer MAE</title><style>body{font:16px system-ui;"
        "max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}</style><h1>Visible-only "
        "transformer masked autoencoder</h1><p>Original,visible input(gray hidden),raw prediction,visible-copy "
        "composite,fit-only mean,hidden-only absolute error,zero-code raw prediction. Fixed[0,1]gray/nearest4/no "
        "error amplification. First heldout example per digit,unfiltered. Composite copies known pixels;raw "
        "prediction does not.</p><p>16nonoverlapping7x7patches;75percent masked,encoder ONLY4visible tokens,width32/"
        "one2-head transformer block. Decoder width16/one2-head block receives restored projected codes and learned "
        "mask tokens plus fixed2D sin/cos positions. MLP ratio2,tanh-GELU,prenorm;no CLS token or dropout. Sigmoid "
        "raw-pixel targets;loss ONLY12hidden patches. Small educational adaptation of <a "
        "href='https://arxiv.org/abs/2111.06377'>He et al.MAE</a>,not an ImageNet reproduction or transfer-learning "
        f"benchmark.</p><p>{epochs}"
        "epochs,13649parameters,Adam.003,batch32. New reproducible per-example masks each training epoch;fixed "
        f"disjoint validation/test masks. Test masked MSE={test_mse:g}"
        ". Zero/shuffled content and second mask diagnostics remain in metrics. Hidden-input perturbation leaves "
        "codes/predictions BITEXACT. Short training can produce blurry generic patches;no classification/semantic or "
        "compression claim.</p>"]
    ae.digit_gallery(root, pools[2],
                     [pools[2].images, nn.from_vector(visible, [200, 784]), nn.from_vector(predicted, [200, 784]),
                      nn.from_vector(composite, [200, 784]), ae.constant(ae.pixel_mean(pools[0].images), 200),
                      nn.from_vector(error, [200, 784]), mae.mae_image(zero_prediction)], html)
    (root / "index.html").write_text("".join(html))
    print(f"MAE selected epoch{selected},test masked MSE={test_mse:g},zero={zero_mse:g},shuffled={shuffled_mse:g},"
          f"second mask={second_mse:g};fresh restore/exact decode/hidden leakage/permutation/batch/immutable checks "
          f"passed\ntraining time {seconds:g} s\nArtifacts:{root}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
