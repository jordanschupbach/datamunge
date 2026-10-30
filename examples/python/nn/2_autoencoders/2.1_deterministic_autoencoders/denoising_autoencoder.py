# Native Python NN example. Corrupted-input/clean-target denoising autoencoder with a matched ordinary control.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/2_autoencoders/2.1_deterministic_autoencoders/denoising_autoencoder.py
"""Actual corrupted-input/CLEAN-target autoencoder with fresh epoch noise and matched ordinary AE.
Three epochs/default; fixed validation/test Gaussian banks, clean/low/strong/secondseed controls.

Port of tflow's denoising_autoencoder. Preparation (in ~/projects/tflow_examples):
just autoencoder-assets; data are read from $DATAMUNGE_DATA_DIR/saliency. Outputs
(denoising/ and ordinary/ training records, data.npz with every noise bank, the
per-epoch training corruptions, test_metrics.tsv, outputs.npz, index.html and
digit<k>.png galleries) go to output/nn/models/denoising_autoencoder. The Gaussian
banks (MT19937/Box-Muller) and therefore every corrupted image are bit-identical to
tflow's. Checkpoints are best.npz files instead of TF SaveV2 bundles. Options:
epochs=N data=PATH out=PATH init=PATH (start both models from saved parameters, e.g.
tflow's denoising/initial_arrays.npz, to follow its trajectories exactly; by default
nn's own initialiser is used, identically for both models).
"""
import math
import os
import time
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_workflows import data_root, helper, options, output_root

checks = helper("checks")
ae = helper("autoencoder")
DenoisingModel = helper("denoising_autoencoder").DenoisingModel

SIGMA = 0.35


def fixtures():
    clean = nn.from_vector([0, .2, .8, 1], [2, 2])
    z = nn.from_vector_double([-2., 1., -1., 2.], [1, 2, 2])
    noisy = ae.corrupt_for_denoising(clean, z, .3)
    expected = nn.from_vector([0, .5, .5, 1], [4]).values()
    nv = noisy.values()
    for i in range(4):
        checks.near(nv[i], expected[i], 6e-8, "known additiveGaussian/float32/clamp actual pixels")
    checks.require(ae.tensor_bytes(ae.corrupt_for_denoising(clean, z, 0)) == ae.tensor_bytes(clean),
                   "zero sigma exact identity")
    a, b = ae.gaussian_noise(clean.shape(), 1, 1673), ae.gaussian_noise(clean.shape(), 1, 1673)
    c = ae.gaussian_noise(clean.shape(), 1, 1674)
    checks.require(a.values() == b.values() and a.values() != c.values(), "Gaussian seed repeat/secondseed distinction")
    checks.throws(lambda: ae.corrupt_for_denoising(clean, z, -1), "negative corruption sigma rejected")
    prediction = noisy.clone().requires_grad_()
    loss = ae.reconstruction_mse(prediction, clean)
    gradient = nn.grad(loss, [prediction])[0].values()
    checks.near(loss.item(), .045, 2e-8, "nonzero error to CLEAN target despite prediction exactly noisy input")
    cv = clean.values()
    for i in range(4):
        checks.near(gradient[i], (nv[i] - cv[i]) / 2, 2e-8, "all4clean-target MSE derivatives")
    print("corruption/clamp/zero/seed/invalidsigma and CLEAN-vs-noisy target/MSE/4derivative fixtures passed")


def train_variant(name, pools, validation_noisy, epochs, root, init):
    """Trains the denoising (corrupted input) or ordinary (clean input) model on CLEAN targets."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    model = DenoisingModel()
    if init:
        model.parameters.load(init)
    checks.require(model.parameters.count() == 50992 and model.parameters.size() == 4,
                   "784->32ReLU->784sigmoid inventory")
    nn.save_npz(str(directory / "initial_arrays.npz"), ae.arrays(model.parameters))
    order = nn.BatchOrder(1000, 32, 1663)
    fit_source = [int(v) for v in pools[0].source.values()]
    best_path = str(directory / "best.npz")
    best, selected, step = math.inf, 0, 0
    started = time.perf_counter()
    with (directory / "history.tsv").open("w") as history, (directory / "batches.tsv").open("w") as batches:
        history.write("epoch\tfit_preupdate_clean_target_mse\tvalidation_denoising_mse\n")
        batches.write("epoch\tstep\trow\tsource_id\n")
        for epoch in range(1, epochs + 1):
            corruption = None
            if name == "denoising":
                bank = ae.gaussian_noise(pools[0].images.shape(), 1, 1671 + epoch)
                corruption = ae.corrupt_for_denoising(pools[0].images, bank, SIGMA)
                nn.save_npz(str(root / f"train_corruption_epoch{epoch}.npz"),
                            [("standard_noise", bank), ("corrupted", corruption)])
            order.begin_epoch()
            total, seen = 0.0, 0
            while True:
                rows = list(order.next())
                if not rows:
                    break
                clean = nn.take_rows(pools[0].images, rows)
                inputs = clean if name == "ordinary" else nn.take_rows(corruption, rows)
                # Forward the (corrupted or clean) input, score it against the CLEAN target.
                net = model.forward(inputs)
                loss = ae.reconstruction_mse(net.reconstruction, ae.flatten(clean))
                if step == 0:
                    keep = nn.GradOptions()
                    keep.retain_graph = True
                    gradients = ae.check_gradients(model.parameters,
                                                   nn.grad(loss, model.parameters.parameters(), keep),
                                                   "every actual clean-target parameter derivative finite",
                                                   "all4parameters receive CLEAN-target loss gradients")
                    nn.save_npz(str(directory / "first_gradients.npz"), gradients)
                    nn.save_npz(str(directory / "first_batch.npz"), [("input", inputs), ("clean_target", clean)])
                value = loss.item()
                checks.require(math.isfinite(value) and value >= 0, "finite pre-update MSE")
                model.optimizer.minimize(loss)
                total += len(rows) * value
                seen += len(rows)
                batches.write("".join(f"{epoch}\t{step}\t{row}\t{fit_source[row]}\n" for row in rows))
                if step == 0:
                    nn.save_npz(str(directory / "first_update_arrays.npz"), ae.arrays(model.parameters))
                step += 1
            checks.require(seen == 1000, "all1000fitting clean images once per epoch")
            value = model.loss(validation_noisy, pools[1].images)
            history.write(f"{epoch}\t{total / seen:.17g}\t{value:.17g}\n")
            nn.save_npz(str(directory / f"epoch{epoch}_arrays.npz"), ae.arrays(model.parameters))
            if value < best:
                best, selected = value, epoch
                model.parameters.save(best_path)  # both variants select on SAME fixed noisy validation-to-clean MSE
            print(f"{name} epoch{epoch}:fixed-noisy validation CLEAN-target MSE={value:g}", flush=True)
    seconds = time.perf_counter() - started
    nn.save_npz(str(directory / "last_arrays.npz"), ae.arrays(model.parameters))
    model.parameters.load(best_path)  # restore selected denoising-task checkpoint
    nn.save_npz(str(directory / "model_arrays.npz"), ae.arrays(model.parameters))
    (directory / "metrics.json").write_text(
        f'{{"epochs":{epochs},"updates":{step},"selected_epoch":{selected},'
        f'"validation_clean_target_mse":{best:.17g},'
        f'"training_corrupted_inputs":{"true" if name == "denoising" else "false"},'
        f'"training_seconds":{seconds:.17g}}}\n')
    return seconds


def main(arguments=None):
    args = options(arguments, dict(epochs=3, data=str(data_root() / "saliency"), out="", init="",
                                   full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    epochs = args["epochs"]
    data = Path(args["data"])
    root = Path(args["out"]) if args["out"] else output_root(args["full"]) / "models/denoising_autoencoder"
    checks.require(0 < epochs <= 100, "three default epochs or explicit1..100")
    fixtures()

    pools = ae.load_saliency(data)
    root.mkdir(parents=True, exist_ok=True)
    validation_bank = ae.gaussian_noise(pools[1].images.shape(), 1, 1709)
    test_bank = ae.gaussian_noise(pools[2].images.shape(), 1, 1711)
    second_bank = ae.gaussian_noise(pools[2].images.shape(), 1, 1717)
    validation_noisy = ae.corrupt_for_denoising(pools[1].images, validation_bank, SIGMA)
    test_noisy = ae.corrupt_for_denoising(pools[2].images, test_bank, SIGMA)
    test_low = ae.corrupt_for_denoising(pools[2].images, test_bank, .15)
    test_strong = ae.corrupt_for_denoising(pools[2].images, test_bank, .6)
    test_second = ae.corrupt_for_denoising(pools[2].images, second_bank, SIGMA)
    ae.save_data(root / "data.npz", pools,
                 [("validation_standard_noise", validation_bank), ("test_standard_noise", test_bank),
                  ("second_standard_noise", second_bank), ("validation_corrupted", validation_noisy),
                  ("test_corrupted", test_noisy), ("test_low", test_low), ("test_strong", test_strong),
                  ("test_second", test_second)])
    seconds = train_variant("denoising", pools, validation_noisy, epochs, root, args["init"])
    seconds += train_variant("ordinary", pools, validation_noisy, epochs, root, args["init"])
    initial_d = nn.load_npz(str(root / "denoising" / "initial_arrays.npz"))
    initial_o = nn.load_npz(str(root / "ordinary" / "initial_arrays.npz"))
    for key, value in initial_d:
        checks.require(ae.tensor_bytes(ae.find(initial_o, key)) == ae.tensor_bytes(value),
                       "matched ordinary and denoising initial parameters")

    deployed, ordinary = DenoisingModel(False), DenoisingModel(False)
    deployed.parameters.load(str(root / "denoising" / "best.npz"))  # fresh denoising inference restore
    ordinary.parameters.load(str(root / "ordinary" / "best.npz"))  # fresh ordinary control restore
    before, ordinary_before = ae.snapshot(deployed.parameters), ae.snapshot(ordinary.parameters)
    outputs = []
    cases = sorted({"noisy": test_noisy, "clean": pools[2].images, "low": test_low, "strong": test_strong,
                    "second": test_second}.items())  # std::map order
    scores = {}
    with (root / "test_metrics.tsv").open("w") as metrics:
        metrics.write("case\tcorrupted_to_clean_mse\tdenoising_to_clean_mse\tordinary_to_clean_mse\n")
        for name, inputs in cases:
            d, o = deployed.predict(inputs), ordinary.predict(inputs)
            outputs += [(name + "_denoising_code", d[0]), (name + "_denoising_reconstruction", d[1]),
                        (name + "_ordinary_reconstruction", o[1])]
            raw = ae.mse(inputs, pools[2].images)
            denoised = ae.mse(d[1], pools[2].images)
            control = ae.mse(o[1], pools[2].images)
            scores[name] = (raw, denoised, control)
            metrics.write(f"{name}\t{raw:.17g}\t{denoised:.17g}\t{control:.17g}\n")
            print(f"{name}:raw/denoised/ordinary CLEAN-target MSE={raw:g}/{denoised:g}/{control:g}")
    result, repeat = deployed.predict(test_noisy), deployed.predict(test_noisy)
    checks.require(ae.tensor_bytes(result[1]) == ae.tensor_bytes(repeat[1]),
                   "deterministic target-free denoising inference")
    checks.require(ae.tensor_bytes(deployed.decode(result[0])) == ae.tensor_bytes(result[1]),
                   "explicit encode-decode exact")
    one = deployed.predict(nn.take_rows(test_noisy, [0]))
    error = ae.max_abs_diff(one[1], result[1], 784)
    checks.require(error < 1e-6, "single/batch denoising parity")
    checks.require(before == ae.snapshot(deployed.parameters)
                   and ordinary_before == ae.snapshot(ordinary.parameters),
                   "all2x4model parameters immutable through test controls")
    nn.save_npz(str(root / "outputs.npz"), outputs)
    ae.copy_manifests(data, root)

    mean = ae.constant(ae.pixel_mean(pools[0].images), 200)
    baseline = ae.mse(mean, pools[2].images)
    html = [
        "<!doctype html><meta charset='utf-8'><title>Clean-target denoising autoencoder</title><style>body{font:"
        "16px system-ui;max-width:1100px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{"
        "margin:25px 0}</style><h1>Denoising:corrupted inputs,CLEAN targets</h1><p>Clean original,corrupted "
        "sigma.35,denoised,ordinary AE on SAME corruption,absolute denoising error,fit-only mean,denoiser on clean "
        "input,denoiser on sigma.6. Fixed[0,1]grayscale/nearest4,no error amplification or quality filtering.</p>"
        f"<p>784->32ReLU->784sigmoid/50992params,Adam.003,batch32,{epochs}"
        "epochs each,matched initial parameters and order1663. Denoiser receives NEW epoch-specific Gaussian "
        "draws,ordinary control trains clean->clean. BOTH select on SAME fixed noisy-validation-to-clean MSE. No "
        "labels enter training or selection;NO clean target is required for inference.</p><p>Noise MT19937/"
        "Box-Muller standardGaussian,sigma.35 in original unit-pixel coordinates,castfloat32 THEN clamp[0,1]. "
        "Clipping changes the corruption distribution and is NOT inverted. Epoch seeds1672onward;validation1709/"
        "test1711/second1717 independent. Low.15 and strong.6 share test standard draws for paired amplitude "
        "controls. Clean-input test has sigma0. Raw corruptions,allnoise banks and clean targets saved.</p><p>"
        f"Fit-only mean MSE={baseline:g}"
        ". All raw/denoised/ordinary CLEAN-target MSE values,including LOW-noise failure cases,in "
        "test_metrics.tsv. A short undercomplete model may blur digit identity or be WORSE than leaving mild noise "
        "unchanged. One chosen corruption family and tiny data budget do not demonstrate universal denoising,"
        "classification,generative quality or bit-rate compression.</p>",
        "<table><tr><th>Case</th><th>Leave input unchanged</th><th>Noise-trained AE</th><th>Ordinary AE</th></tr>"]
    for name, value in sorted(scores.items()):
        html.append(f"<tr><td>{name}</td><td>{value[0]:g}</td><td>{value[1]:g}</td><td>{value[2]:g}</td></tr>")
    verdict = ("noise-trained lower MSE" if scores["noisy"][1] < scores["noisy"][2]
               else "ordinary lower MSE in this short run")
    html.append(f"</table><p>All values compare against CLEAN targets. Noise-trained versus ordinary on main noise: "
                f"{verdict}. This baseline comparison remains visible regardless of result.</p>")
    denoised = ae.find(outputs, "noisy_denoising_reconstruction")
    ae.digit_gallery(root, pools[2],
                     [pools[2].images, test_noisy, denoised, ae.find(outputs, "noisy_ordinary_reconstruction"),
                      nn.abs(nn.sub(denoised, ae.flatten(pools[2].images))), mean,
                      ae.find(outputs, "clean_denoising_reconstruction"),
                      ae.find(outputs, "strong_denoising_reconstruction")], html)
    (root / "index.html").write_text("".join(html))
    print("Freshrestore/target-freeinference/encode-decode/repeat/single-batch/immutable2x4weights passed,"
          f"maxbatcherror={error:g}\nFit-onlymean CLEAN-target MSE={baseline:g};weak/mildnoise failures retained.\n"
          f"training time {seconds:g} s\nArtifacts:{root}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
