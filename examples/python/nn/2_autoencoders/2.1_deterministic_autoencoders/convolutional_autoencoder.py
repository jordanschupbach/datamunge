# Native Python NN example. Spatial 7x7x4-code convolutional autoencoder with latent and baseline controls.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/2_autoencoders/2.1_deterministic_autoencoders/convolutional_autoencoder.py
"""Actual spatial 7x7x4 code / nearest-repeat convolutional decoder, 741 learned parameters.
Three default epochs, mean/initial/pool-only/zero/shuffled-code controls.

Port of tflow's convolutional_autoencoder. Preparation (in ~/projects/tflow_examples):
just autoencoder-assets; data are read from $DATAMUNGE_DATA_DIR/saliency. Outputs
(history, npz arrays, metrics.json, index.html and digit<k>.png galleries) go to
output/nn/models/convolutional_autoencoder. Options: epochs=N data=PATH out=PATH
init=PATH (start from saved parameters, e.g. tflow's initial_arrays.npz, to
follow its trajectory exactly; by default nn's own initialiser is used).
"""
import math
import os
import time
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_workflows import data_root, helper, options, output_root

checks = helper("checks")
ae = helper("autoencoder")


def fixtures():
    pixels = nn.from_vector([i / 15 for i in range(16)], [16]).values()
    x = nn.from_vector(pixels, [1, 4, 4, 1]).requires_grad_()
    code = nn.avg_pool2d(x, 2, 2)
    enlarged = nn.upsample_nearest(code, 2)
    loss = nn.mean(nn.square(nn.sub(enlarged, x)))
    gradient = nn.grad(loss, [x])[0].values()
    code_values, enlarged_values = code.values(), enlarged.values()
    mse = 0.0
    for y in range(4):
        for a in range(4):
            mean = 0.0
            for dy in range(2):
                for dx in range(2):
                    mean += pixels[((y // 2) * 2 + dy) * 4 + (a // 2) * 2 + dx] / 4.
            checks.near(code_values[(y // 2) * 2 + a // 2], mean, 4e-8, "all4pool means")
            checks.near(enlarged_values[y * 4 + a], mean, 4e-8, "all16nearestrepeat pixels,not tile-whole-image")
            error = pixels[y * 4 + a] - mean
            mse += error * error / 16
            checks.near(gradient[y * 4 + a], error / 8, 2e-8, "all16analytic pool+repeat MSE input derivatives")
    checks.near(loss.item(), mse, 2e-8, "all16spatial reconstruction MSE terms")
    print("known4x4->2x2->4x4 allpool/repeatpixels/MSE/ALL16analytic gradients passed")


def main(arguments=None):
    args = options(arguments, dict(epochs=3, data=str(data_root() / "saliency"), out="", init="",
                                   full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    epochs = args["epochs"]
    data = Path(args["data"])
    root = Path(args["out"]) if args["out"] else output_root(args["full"]) / "models/convolutional_autoencoder"
    checks.require(0 < epochs <= 100, "three default epochs or explicit1..100")
    fixtures()

    pools = ae.load_saliency(data)
    model = ae.ConvolutionalAutoencoderModel()
    if args["init"]:
        model.parameters.load(args["init"])
    checks.require(model.parameters.size() == 8 and model.parameters.count() == 741,
                   "independent3x3kernels+biases741params/8tensors")
    root.mkdir(parents=True, exist_ok=True)
    nn.save_npz(str(root / "initial_arrays.npz"), ae.arrays(model.parameters))
    initial = model.evaluate(pools[2].images)
    order = nn.BatchOrder(1000, 32, 1807)
    fit_source = [int(v) for v in pools[0].source.values()]
    best_path = str(root / "best.npz")
    best, selected, step = math.inf, 0, 0
    started = time.perf_counter()
    with (root / "history.tsv").open("w") as history, (root / "batches.tsv").open("w") as batches:
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
                # Forward pass, reconstruction MSE, then (first step) every gradient and one Adam step.
                net = ae.convolutional_autoencoder(model.parameters, inputs)
                loss = model.loss(net.reconstruction, inputs)
                if step == 0:
                    keep = nn.GradOptions()
                    keep.retain_graph = True
                    gradients = ae.check_gradients(model.parameters,
                                                   nn.grad(loss, model.parameters.parameters(), keep),
                                                   "all convolutional parameter gradients finite",
                                                   "all8convolutional tensors receive reconstruction gradients")
                    nn.save_npz(str(root / "first_gradients.npz"), gradients)
                    nn.save_npz(str(root / "first_batch.npz"), [("images", inputs)])
                value = loss.item()
                checks.require(math.isfinite(value) and value >= 0, "finite pre-update MSE")
                model.optimizer.minimize(loss)
                total += len(rows) * value
                seen += len(rows)
                batches.write("".join(f"{epoch}\t{step}\t{row}\t{fit_source[row]}\n" for row in rows))
                if step == 0:
                    nn.save_npz(str(root / "first_update_arrays.npz"), ae.arrays(model.parameters))
                step += 1
            checks.require(seen == 1000, "each fitting record once per epoch including last8")
            validation = model.evaluate(pools[1].images)
            value = ae.mse(validation[1], pools[1].images)
            history.write(f"{epoch}\t{total / seen:.17g}\t{value:.17g}\n")
            nn.save_npz(str(root / f"epoch{epoch}_arrays.npz"), ae.arrays(model.parameters))
            if value < best:
                best, selected = value, epoch
                model.parameters.save(best_path)  # validation MSE alone selects the checkpoint
            print(f"epoch{epoch}:fitPreMSE={total / seen:g},validationMSE={value:g}", flush=True)
    train_seconds = time.perf_counter() - started
    model.parameters.load(best_path)  # restore selected weights for the checkpoint-gradient probes
    deployed = ae.ConvolutionalAutoencoderModel(False)
    deployed.parameters.load(best_path)  # fresh spatial inference restore
    before = ae.snapshot(deployed.parameters)
    arrays = []
    for split, name in enumerate(("fit", "validation", "test")):
        out = deployed.evaluate(pools[split].images)
        checks.require(list(out[0].shape()) == [pools[split].images.size(0), 7, 7, 4],
                       "actual spatial196-coordinate bottleneck")
        checks.require(list(out[1].shape()) == list(pools[split].images.shape()),
                       "decoder restores original28x28x1 geometry")
        arrays += [(name + "_images", pools[split].images), (name + "_source_ids", pools[split].source),
                   (name + "_code", out[0]), (name + "_reconstruction", out[1])]
        if split == 1:
            checks.near(ae.mse(out[1], pools[split].images), best, 1e-8, "fresh restored validation MSE")
    arrays += [("test_labels", pools[2].labels), ("initial_test_reconstruction", initial[1])]
    test, repeat = deployed.evaluate(pools[2].images), deployed.evaluate(pools[2].images)
    checks.require(ae.tensor_bytes(test[1]) == ae.tensor_bytes(repeat[1]), "deterministic spatial reconstruction")
    checks.require(ae.tensor_bytes(deployed.decode(test[0])) == ae.tensor_bytes(test[1]),
                   "standalone convolutional decoder EXACT encode/decode")
    probe = ae.rows(pools[2], range(8))
    stages = deployed.activations(probe.images)
    probe_net = ae.convolutional_autoencoder(model.parameters, probe.images)
    probe_gradients = nn.grad(model.loss(probe_net.reconstruction, probe.images), model.parameters.parameters())
    nn.save_npz(str(root / "probe_gradients.npz"), list(zip(model.parameters.names(), probe_gradients)))
    expected = [[8, 28, 28, 8], [8, 14, 14, 8], [8, 14, 14, 4], [8, 7, 7, 4], [8, 14, 14, 8], [8, 28, 28, 1]]
    for k in range(6):
        checks.require(list(stages[k].shape()) == expected[k], "all six actual intermediate spatial shapes")
        arrays.append((f"probe_stage{k}", stages[k]))
    arrays.append(("probe_images", probe.images))
    single = deployed.evaluate(ae.rows(pools[2], [0]).images)
    batch_error = ae.max_abs_diff(single[1], test[1], 784)
    checks.require(batch_error < 2e-6, "single/batch convolutional inference parity")
    # Decode an all-zero code and each image's code cyclically shifted by one record.
    zero = deployed.decode(nn.zeros([200, 7, 7, 4]))
    shuffled = deployed.decode(nn.concat([nn.slice_axis(test[0], 0, 1, 200), nn.slice_axis(test[0], 0, 0, 1)], 0))
    arrays += [("zero_reconstruction", zero), ("shuffled_reconstruction", shuffled)]
    mean4 = nn.reshape(ae.constant(ae.pixel_mean(pools[0].images), 200), [200, 28, 28, 1])
    arrays.append(("mean_reconstruction", mean4))
    # 4x4 average then nearest repeat: a 49-coordinate input-dependent baseline.
    pooled = nn.upsample_nearest(nn.avg_pool2d(pools[2].images, 4, 4), 4).contiguous()
    arrays.append(("pool_repeat_reconstruction", pooled))
    checks.require(before == ae.snapshot(deployed.parameters),
                   "all8parameters immutable through inference/zero/shuffledcode controls")
    nn.save_npz(str(root / "convolutional.npz"), arrays)
    nn.save_npz(str(root / "model_arrays.npz"), ae.arrays(deployed.parameters))
    ae.copy_manifests(data, root)
    mse = ae.mse(test[1], pools[2].images)
    mean_mse = ae.mse(mean4, pools[2].images)
    pool_mse = ae.mse(pooled, pools[2].images)
    (root / "metrics.json").write_text(
        f'{{"epochs":{epochs},"updates":{step},"selected_epoch":{selected},"parameters":741,'
        f'"test_mse":{mse:.17g},"mean_mse":{mean_mse:.17g},"pool_repeat_mse":{pool_mse:.17g},'
        f'"initial_mse":{ae.mse(initial[1], pools[2].images):.17g},"zero_mse":{ae.mse(zero, pools[2].images):.17g},'
        f'"shuffled_mse":{ae.mse(shuffled, pools[2].images):.17g},"single_batch_max_error":{batch_error:.17g}}}\n')
    html = [
        "<!doctype html><meta charset='utf-8'><title>Convolutional autoencoder</title><style>body{font:16px "
        "system-ui;max-width:1100px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:"
        "25px 0}</style><h1>Convolutional autoencoder: spatial bottleneck</h1><p>Original, trained, absolute "
        "error, fit-only mean, 4x4 average/nearest repeat, untrained, zero code, cyclic shuffled code. Fixed [0,1] "
        "grayscale/nearest 4x, no error amplification or quality filtering.</p><p>Conv3x3 same 1->8 ReLU, AvgPool2, "
        "Conv3x3 same 8->4 ReLU, AvgPool2 gives a 7x7x4 code. Decoder nearest-repeat 2, Conv3x3 same 4->8 ReLU, "
        "nearest-repeat 2, Conv3x3 same 8->1 sigmoid. No skips/dense bypass; 741 untied parameters, "
        f"{epochs} epochs, batch 32, Adam .003. Clean inputs are targets; no labels enter training or validation-MSE "
        "selection. The 196 continuous code values are 4x fewer coordinates, not a bit-rate/file compression "
        f"claim.</p><p>Held-out MSE trained={mse:g}, fit mean={mean_mse:g}, pool/repeat={pool_mse:g}"
        ". Pool/repeat is a 49-coordinate input-dependent baseline, not a same-capacity learned comparator. "
        "Latent zero/shuffle and initial-model controls expose whether learned spatial information matters. MSE "
        "alone does not establish shape fidelity, classification, denoising or a generative prior.</p>"]
    error = nn.abs(nn.sub(test[1], pools[2].images))
    labels = [int(v) for v in pools[2].labels.values()]
    source = [int(v) for v in pools[2].source.values()]
    for digit in range(10):
        b = labels.index(digit)
        panels = [ae.row(t, b) for t in (pools[2].images, test[1], error, mean4, pooled, initial[1], zero, shuffled)]
        filename = f"digit{digit}.png"
        ae.panel_row(root / filename, panels)
        html.append(f"<section><h2>Digit {digit}, source {source[b]}</h2><img src='{filename}'></section>")
    (root / "index.html").write_text("".join(html))
    print(f"Convolutional741params/8tensors,selectedepoch={selected},testMSE={mse:g},fitmean={mean_mse:g},"
          f"pool/repeat={pool_mse:g}\nSpatialall6shapes/freshrestore/exactdecode/repeat/single-batch/immutable8weights/"
          f"all8gradienttensors passed.\ntraining time {train_seconds:g} s\nArtifacts:{root}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
