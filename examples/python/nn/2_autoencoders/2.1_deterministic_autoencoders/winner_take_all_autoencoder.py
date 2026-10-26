# Native Python NN example. Lifetime winner-take-all autoencoder: training-only per-feature batch winners, matched plain control.
"""CMD: just python-example winner_take_all_autoencoder
Lifetime WTA: training-only per-feature batch winners, full ReLU deployment.
Three default epochs per model; matched plain control.

Port of tflow's winner_take_all_autoencoder (the model is examples/nn/common/competitive_autoencoder.hpp,
mirrored by examples/python/nn/common/competitive_autoencoder.py and shared with
k_sparse_autoencoder). Preparation (in ~/projects/tflow_examples): just autoencoder-assets; data are
read from $DATAMUNGE_DATA_DIR/saliency. Outputs (competitive/ and plain/ training records,
data.npz, index.html and digit<k>.png galleries) go to output/nn/models/winner_take_all_autoencoder.
Checkpoints are best.npz files instead of TF SaveV2 bundles. Options: epochs=N data=PATH
out=PATH init=PATH (start both models from saved parameters, e.g. tflow's
competitive/initial_arrays.npz, to follow its trajectories exactly; by default nn's own
initialiser is used, identically for both models).
"""
import math
import os
import time
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_workflows import data_root, helper, options, output_root

checks = helper("checks")
ae = helper("autoencoder")
competitive = helper("competitive_autoencoder")

LIFETIME = True


def train_variant(lifetime, plain, pools, epochs, root, initial):
    """Trains the competitive (or matched plain) model; returns its training seconds."""
    directory = root / ("plain" if plain else "competitive")
    directory.mkdir(parents=True, exist_ok=True)
    model = competitive.CompetitiveAutoencoderModel(lifetime, plain)
    model.load(initial)
    checks.require(model.parameters.size() == 4 and model.parameters.count() == 101200,
                   "hard-competition and matched plain parameter census")
    nn.save_npz(str(directory / "initial_arrays.npz"), ae.arrays(model.parameters))
    initial_output = model.evaluate(pools[2].images)
    nn.save_npz(str(directory / "initial_outputs.npz"),
                [("code", initial_output[0]), ("reconstruction", initial_output[1])])
    order = nn.BatchOrder(1000, 32, 1933)
    fit_source = [int(v) for v in pools[0].source.values()]
    best_path = str(directory / "best.npz")
    label = "plain" if plain else "competitive"
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
                images = nn.take_rows(pools[0].images, rows)
                # Training forward (competition mask applied), reconstruction MSE against the input.
                x = ae.flatten(images)
                net = model.forward(x)
                loss = ae.reconstruction_mse(net.training_reconstruction, x)
                if step == 0:
                    probe = model.probe(images)
                    competitive.check_competitive_support(probe, lifetime, plain)
                    nn.save_npz(str(directory / "first_probe.npz"),
                                [("raw", probe[0]), ("mask", probe[1]), ("training_code", probe[2]),
                                 ("reconstruction", probe[3])])
                    keep = nn.GradOptions()
                    keep.retain_graph = True
                    gradients = ae.check_gradients(model.parameters,
                                                   nn.grad(loss, model.parameters.parameters(), keep),
                                                   "all model MSE derivatives finite",
                                                   "all model tensors receive full MSE gradients")
                    nn.save_npz(str(directory / "first_gradients.npz"), gradients)
                    nn.save_npz(str(directory / "first_batch.npz"), [("images", images)])
                if step == 31:
                    probe = model.probe(images)
                    competitive.check_competitive_support(probe, lifetime, plain)
                    nn.save_npz(str(directory / "partial_probe.npz"),
                                [("images", images), ("raw", probe[0]), ("mask", probe[1]),
                                 ("training_code", probe[2]), ("reconstruction", probe[3])])
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
            mse = ae.mse(model.evaluate(pools[1].images)[1], pools[1].images)
            history.write(f"{epoch}\t{total / seen:.17g}\t{mse:.17g}\n")
            nn.save_npz(str(directory / f"epoch{epoch}_arrays.npz"), ae.arrays(model.parameters))
            if mse < best:
                best, selected = mse, epoch
                model.parameters.save(best_path)  # common validation reconstruction MSE selects each model
            print(f"{label} epoch{epoch}:validationMSE={mse:g}", flush=True)
    seconds = time.perf_counter() - started

    deployed = competitive.CompetitiveAutoencoderModel(lifetime, plain, False)
    deployed.parameters.load(best_path)  # fresh competitive/plain inference restore
    before = ae.snapshot(deployed.parameters)
    probe_images = nn.take_rows(pools[2].images, list(range(32)))
    probe = deployed.probe(probe_images)
    competitive.check_competitive_support(probe, lifetime, plain)
    nn.save_npz(str(directory / "selected_probe.npz"),
                [("images", probe_images), ("raw", probe[0]), ("mask", probe[1]), ("training_code", probe[2]),
                 ("reconstruction", probe[3])])
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
    zero = deployed.decode(nn.zeros([200, 64]))
    shifted = deployed.decode(nn.concat([nn.slice_axis(test[0], 0, 1, 200), nn.slice_axis(test[0], 0, 0, 1)], 0))
    arrays += [("zero_reconstruction", zero), ("shuffled_reconstruction", shifted)]
    one = deployed.evaluate(nn.take_rows(pools[2].images, [0]))
    error = ae.max_abs_diff(one[1], test[1], 784)
    checks.require(error < 1e-6, "single/batch competitive/plain inference parity")
    checks.require(before == ae.snapshot(deployed.parameters),
                   "all deployed parameters immutable through code controls")
    nn.save_npz(str(directory / "outputs.npz"), arrays)
    nn.save_npz(str(directory / "model_arrays.npz"), ae.arrays(deployed.parameters))
    test_mse = ae.mse(test[1], pools[2].images)
    (directory / "metrics.json").write_text(
        f'{{"lifetime":{"true" if lifetime else "false"},"plain":{"true" if plain else "false"},'
        f'"parameters":{model.parameters.count()},"epochs":{epochs},"updates":{step},'
        f'"selected_epoch":{selected},"test_mse":{test_mse:.17g},'
        f'"zero_mse":{ae.mse(zero, pools[2].images):.17g},"shuffled_mse":{ae.mse(shifted, pools[2].images):.17g},'
        f'"single_batch_max_error":{error:.17g},"training_seconds":{seconds:.17g}}}\n')
    print(f"{label}:selected{selected},parameters={model.parameters.count()},testMSE={test_mse:g},"
          f"freshrestore/decode/repeat/immutable parameters passed (training {seconds:g} s)")
    return seconds


def main(arguments=None):
    args = options(arguments, dict(epochs=3, data=str(data_root() / "saliency"), out="", init="",
                                   full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    epochs = args["epochs"]
    data = Path(args["data"])
    root = Path(args["out"]) if args["out"] else output_root(args["full"]) / "models/winner_take_all_autoencoder"
    checks.require(0 < epochs <= 100, "three default epochs or explicit1..100")
    competitive.competitive_checks()

    lifetime = LIFETIME
    pools = ae.load_saliency(data)
    root.mkdir(parents=True, exist_ok=True)
    factory = competitive.CompetitiveAutoencoderModel(lifetime, False, False)
    if args["init"]:
        factory.parameters.load(args["init"])
    seconds = train_variant(lifetime, False, pools, epochs, root, factory.parameters)
    seconds += train_variant(lifetime, True, pools, epochs, root, factory.parameters)
    ae.save_data(root / "data.npz", pools)
    ae.copy_manifests(data, root)
    t = nn.load_npz(str(root / "competitive" / "outputs.npz"))
    u = nn.load_npz(str(root / "plain" / "outputs.npz"))
    mean = ae.constant(ae.pixel_mean(pools[0].images), 200)
    competitive_reconstruction = ae.find(t, "test_reconstruction")
    plain_reconstruction = ae.find(u, "test_reconstruction")
    competitive_mse = ae.mse(competitive_reconstruction, pools[2].images)
    plain_mse = ae.mse(plain_reconstruction, pools[2].images)
    mean_mse = ae.mse(mean, pools[2].images)
    html = [
        "<!doctype html><meta charset='utf-8'><title>Hard competition autoencoder</title><style>body{font:16px "
        "system-ui;max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:"
        "25px 0}</style><h1>"
        + ("Lifetime winner-take-all" if lifetime else "Per-image k-sparse")
        + " autoencoder</h1><p>Original,competitive,plain control,fit-only mean,absolute competitive error,zero-code,"
        "shuffled-code. Fixed[0,1]gray/nearest4/no error amplification. First test record of each digit,unfiltered."
        "</p><p>784->64->784sigmoid,101200parameters. "
        + ("ReLU encoder,TRAINING ONLY lifetime winners:each feature keeps ceil(batch/8) examples,4of32 "
           "or1of8. Deployed encoder is full ReLU with NO lifetime gate;deployed code is NOT guaranteed "
           "sparse. This is fully-connected WTA,not spatial/convolutional WTA." if lifetime else
           "LINEAR encoder,top8largest SIGNED activities per image,not top absolute magnitude;mask is "
           "retained in training AND inference. Exactly8selected coordinates,at most8nonzero values;"
           "negative winners remain signed. No ReLU,no sparsity penalty or straight-through gradient.")
        + " TopK ties select lower indices;derivatives pass ONLY selected coordinates. The discrete ranking has no "
        "derivative;at ties this is one deterministic subgradient convention.</p><p>Matched initial parameter bytes "
        f"and source orders;plain control removes competition. {epochs}"
        "epochs EACH,Adam.003,batch32/final8. Separate validation reconstruction MSE selects each deployed model. "
        "Sigmoid decoder/raw pixels/compact width are educational adaptations of the <a href='https://arxiv.org/abs/"
        + ("1409.2752" if lifetime else "1312.5663")
        + "'>hard sparsity approach</a>,not the original scale/decoder/classification experiment. Same weights do not "
        "imply same gated initial mapping. Labels only balance pools/captions. Blurry errors/weak quality remain "
        f"visible;no bit-rate or semantic/classification claim.</p><p>TestMSE competitive={competitive_mse:g}"
        f",plain={plain_mse:g},mean={mean_mse:g}.</p>"]
    ae.digit_gallery(root, pools[2],
                     [pools[2].images, competitive_reconstruction, plain_reconstruction, mean,
                      nn.abs(nn.sub(competitive_reconstruction, ae.flatten(pools[2].images))),
                      ae.find(t, "zero_reconstruction"), ae.find(t, "shuffled_reconstruction")], html)
    (root / "index.html").write_text("".join(html))
    print(f"Hard winner competition/matched plain controls/fresh deployment verified.\nTestMSE competitive="
          f"{competitive_mse:g},plain={plain_mse:g},mean={mean_mse:g}\ntraining time {seconds:g} s\n"
          f"Artifacts:{root}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
