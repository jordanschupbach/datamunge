# Native Python NN example. Per-gate per-example LayerNorm LSTM with raw cell carry and normalized readout.
"""CMD: just python-example layer_norm_lstm
Per-gate per-example LayerNorm LSTM with raw cell carry and normalized
hidden readout.

Port of tflow's layer_norm_lstm (common/ml/gated_sequence_demo.h with
GatedCell::kLayerNorm). No dataset: the marked sequences are generated with
tflow's std::mt19937 recipe (bit-identical data.npz). 818 parameters in 8
tensors, Adam .01, batch 32, three epochs. Same checks as lstm, with the
readout through the normalized cell, plus: row 0's prediction is bit-exact
whatever the other 127 batch rows contain (no batch statistics).
Weights initialise from nn's generator (statistical parity); init=PATH
(tflow's output/models/layer_norm_lstm/initial_arrays.npz) replays tflow's run.
Options: epochs=3 out=PATH init=PATH tf=PATH. Output: <output>/models/layer_norm_lstm/.
"""
import os
import time
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_workflows import helper, options

checks = helper("checks")
recurrent_model = helper("recurrent_model")
data = helper("marked_sequence_data")
demo = helper("marked_sequence_demo")
gated_sequence = helper("gated_sequence")
gated_demo = helper("gated_sequence_demo")

NAME, KIND = "layer_norm_lstm", gated_sequence.LAYER_NORM


def evaluate(model, feed):
    """Forward fields, loss and the full-BPTT gradients of the chosen logit with
    respect to the inputs and every zero initial state (tflow's Evaluate fetches)."""
    inputs = feed.inputs.detach().clone()
    initials = model.initials(feed.inputs.shape()[0])
    inputs.requires_grad_(True)
    for state in initials:
        state.requires_grad_(True)
    out = model.forward(model.store, inputs, feed.lengths, initials)
    out["probability"] = model.probability(out["logits"])
    loss, chosen = model.loss_fn(out, feed.labels)
    wrt = [inputs] + initials
    grad_options = nn.GradOptions()
    grad_options.retain_graph = True
    grad_options.allow_unused = True
    g = nn.grad([chosen], wrt, [], grad_options)
    result = {f: out[f].detach() for f in model.fields}
    result["loss"] = loss.detach()
    result["input_gradient"] = recurrent_model.defined_gradient(g[0], inputs)
    for k, name in enumerate(model.initial_gradient_names):
        if name:
            result[name] = recurrent_model.defined_gradient(g[k + 1], initials[k])
    return result


def loss_gradients(model, feed):
    """Loss gradient of every parameter in store order (zeros where unused)."""
    out = model.forward(model.store, feed.inputs, feed.lengths, model.initials(feed.inputs.shape()[0]))
    loss = model.loss_fn(out, feed.labels)[0]
    params = model.store.parameters()
    grad_options = nn.GradOptions()
    grad_options.allow_unused = True
    g = nn.grad([loss], params, [], grad_options)
    return [recurrent_model.defined_gradient(gi, p) for gi, p in zip(g, params)]


def named_gradients(model, gradients):
    return list(zip(model.store.names(), gradients))


def main(arguments=None):
    start = time.perf_counter()
    args = options(arguments, demo.demo_defaults(NAME, os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    demo.check_epochs(args["epochs"])
    nn.set_num_threads(4)
    spec = gated_demo.gated_demo_spec(KIND)
    root = Path(args["out"])
    root.mkdir(parents=True, exist_ok=True)

    # Data: 512 fit / 128 validation / 128 test marked sequences.
    arrays = []
    pools = data.marked_pools(arrays)
    nn.save_npz(str(root / "data.npz"), arrays)
    model = gated_sequence.gated_sequence_model(KIND, True)
    checks.require(model.store.count() == spec["parameters"] and model.store.size() == spec["tensors"],
                   spec["count_message"])
    if args["init"]:
        model.load_arrays(args["init"])
    initial_arrays = model.arrays()
    nn.save_npz(str(root / "initial_arrays.npz"), initial_arrays)
    demo.save_outputs(root / "initial_outputs.npz", evaluate(model, pools[2].feed()))

    # Training: Adam .01, batch 32, BatchOrder(512, 32, 26019).
    history = open(root / "history.tsv", "w")
    batches = open(root / "batches.tsv", "w")
    history.write("epoch\tvalidation_ce\tvalidation_accuracy\n")
    batches.write("epoch\tstep\trow\n")
    order = nn.BatchOrder(512, 32, 26019)
    updates = 0
    for epoch in range(1, args["epochs"] + 1):
        order.begin_epoch()
        while True:
            rows = list(order.next())
            if not rows:
                break
            x = data.marked_rows(pools[0], rows)
            if not updates:
                data.save_data(root / "first_batch.npz", x)
                demo.save_outputs(root / "first_forward.npz", evaluate(model, x.feed()))
                nn.save_npz(str(root / "initial_gradients.npz"), named_gradients(model, loss_gradients(model, x.feed())))
            # One Adam update: forward, loss, gradient of every parameter, step.
            out = model.forward(model.store, x.inputs, x.lengths, model.initials(x.size()))
            loss = model.loss_fn(out, x.labels)[0]
            params = model.store.parameters()
            grad_options = nn.GradOptions()
            grad_options.allow_unused = True
            g = nn.grad([loss], params, [], grad_options)
            model.optimizer.step([recurrent_model.defined_gradient(gi, p) for gi, p in zip(g, params)])
            if not updates:
                nn.save_npz(str(root / "first_after.npz"), model.arrays())
            for row in rows:
                batches.write(f"{epoch}\t{updates}\t{row}\n")
            updates += 1
        valid = evaluate(model, pools[1].feed())
        ce, accuracy = valid["loss"].item(), data.accuracy(valid["logits"], pools[1].labels)
        history.write(f"{epoch}\t{ce:.17g}\t{accuracy:.17g}\n")
        nn.save_npz(str(root / f"epoch{epoch}_arrays.npz"), model.arrays())
        demo.save_outputs(root / f"epoch{epoch}_validation.npz", valid)
        print(f"epoch{epoch} validationCE={ce:g} accuracy={accuracy:g}", flush=True)
    batches.flush()
    nn.save_npz(str(root / "final_gradients.npz"),
                named_gradients(model, loss_gradients(model, data.load_data(root / "first_batch.npz").feed())))

    # Save -> restore into a fresh inference model.
    model.save(root / "final.npz")
    deployed = gated_sequence.gated_sequence_model(KIND, False)
    deployed.restore(root / "final.npz")
    checks.require(recurrent_model.same_bits(model.arrays(), deployed.arrays()), spec["restore_message"])
    before = deployed.arrays()
    test = pools[2]
    output = evaluate(deployed, test.feed())
    prediction = deployed.predict(test.feed())
    for k, v in prediction.items():
        checks.require(recurrent_model.same_bits(v, output[k]), "labelFreePredictionMatchesEvaluation")
    nn.save_npz(str(root / "model_arrays.npz"), deployed.arrays())
    demo.save_outputs(root / "outputs.npz", output)
    lengths, labels = [int(v) for v in test.lengths.values()], [int(v) for v in test.labels.values()]
    gated_demo.gate_algebra_audit(KIND, output, lengths)
    repeat = deployed.predict(test.feed())
    for k, v in prediction.items():
        checks.require(recurrent_model.same_bits(v, repeat[k]), "RNNPredictRepeatBitexact")
    single_logits = deployed.predict(data.marked_rows(test, [0]).feed())["logits"].values()
    logits = output["logits"].values()
    for c in range(2):
        checks.near(single_logits[c], logits[c], 2e-6, "singleBatchRNNParityWithResetState")

    # Counterfactual copies: padding garbage, flipped marked value, flipped last valid value.
    test_inputs = list(test.inputs.values())
    input_gradient, states = output["input_gradient"].values(), output["states"].values()
    padded, cue_flip, last_flip = list(test_inputs), list(test_inputs), list(test_inputs)
    cue_labels = list(labels)
    oracle_correct = last_correct = 0

    def idx(b, t, c):
        return (b * 12 + t) * 2 + c

    def sidx(b, t, j):
        return (b * 13 + t) * 12 + j

    for b in range(128):
        length, selected = lengths[b], 0.0
        for t in range(12):
            if t >= length:
                padded[idx(b, t, 0)] = 123.0 + b + t
                padded[idx(b, t, 1)] = -57.0
                for c in range(2):
                    checks.require(input_gradient[idx(b, t, c)] == 0, "ALLpaddedInputLogitGradientExactlyZero")
                for j in range(12):
                    checks.require(states[sidx(b, t + 1, j)] == states[sidx(b, t, j)], "stateRetainedAcrossEveryPaddingStep")
            if test_inputs[idx(b, t, 1)] > 0:
                selected += test_inputs[idx(b, t, 0)]
                cue_flip[idx(b, t, 0)] *= -1
        oracle_correct += (selected > 0) == (labels[b] != 0)
        last_correct += (test_inputs[idx(b, length - 1, 0)] > 0) == (labels[b] != 0)
        cue_labels[b] = 1 - labels[b]
        last_flip[idx(b, length - 1, 0)] *= -1
    checks.require(oracle_correct == 128, "deterministicMarkerValueOracle")
    pad_data, cue_data, last_data = data.copy(test), data.copy(test), data.copy(test)
    pad_data.inputs = nn.from_vector(padded, [128, 12, 2])
    cue_data.inputs = nn.from_vector(cue_flip, [128, 12, 2])
    cue_data.labels = data.int32_tensor(cue_labels)
    last_data.inputs = nn.from_vector(last_flip, [128, 12, 2])
    pad_output = evaluate(deployed, pad_data.feed())
    cue_output = evaluate(deployed, cue_data.feed())
    last_output = evaluate(deployed, last_data.feed())
    for k, v in output.items():
        checks.require(recurrent_model.same_bits(v, pad_output[k]),
                       "arbitraryPaddingValuesCannotAffectAnyOutputOrGradient:" + k)
    last_states = last_output["states"].values()
    for b in range(128):
        for t in range(lengths[b]):
            for j in range(12):
                checks.require(last_states[sidx(b, t, j)] == states[sidx(b, t, j)],
                               "changingLastValidTokenCannotChangeEarlierHiddenStates")
    data.save_data(root / "padding_inputs.npz", pad_data)
    data.save_data(root / "cue_flip_inputs.npz", cue_data)
    data.save_data(root / "last_flip_inputs.npz", last_data)
    demo.save_outputs(root / "padding_outputs.npz", pad_output)
    demo.save_outputs(root / "cue_flip_outputs.npz", cue_output)
    demo.save_outputs(root / "last_flip_outputs.npz", last_output)
    empty = data.marked_rows(test, [0, 1, 2, 3])
    empty.lengths = nn.zeros([4], nn.DType.Int32)
    empty_output = evaluate(deployed, empty.feed())
    checks.require(all(v == 0 for v in empty_output["states"].values()), "zeroLengthSequenceRetainsZeroInitialState")
    checks.require(all(v == 0 for v in empty_output["input_gradient"].values()), "zeroLengthAllInputGradientsZero")
    data.save_data(root / "zero_length_inputs.npz", empty)
    demo.save_outputs(root / "zero_length_outputs.npz", empty_output)
    data.heatmap(root / "sequences.png", test.inputs)
    data.heatmap(root / "input_gradients.png", output["input_gradient"], True)
    predicted = data.predictions(output["logits"])
    ids = [int(v) for v in test.source_ids.values()]
    with open(root / "predictions.tsv", "w") as table:
        table.write("source_id\tlength\tlabel\tpredicted\tmarked_oracle\tlast_value_baseline\n")
        for b in range(128):
            table.write(f"{ids[b]}\t{lengths[b]}\t{labels[b]}\t{predicted[b]}\t{labels[b]}\t"
                        f"{int(test_inputs[idx(b, lengths[b] - 1, 0)] > 0)}\n")
    # No batch statistics: row 0's prediction is bit-exact whatever the other 127 rows contain.
    companions = data.copy(test)
    x = list(companions.inputs.values())
    for b in range(1, 128):
        for t in range(12):
            for c in range(2):
                x[(b * 12 + t) * 2 + c] = float((b + t + c) % 7 - 3) * 17
    companions.inputs = nn.from_vector(x, [128, 12, 2])
    altered = deployed.predict(companions.feed())
    a = altered["logits"].values()
    for c in range(2):
        checks.require(a[c] == logits[c], "layerNormFirstPredictionBitexactUnderAllOtherBatchRowChanges")
    data.save_data(root / "batch_companion_inputs.npz", companions)
    demo.save_outputs(root / "batch_companion_outputs.npz", altered)
    checks.require(recurrent_model.same_bits(before, deployed.arrays()),
                   "allRecurrentInferenceAndGradientControlsWeightsImmutable")

    # Test metrics, the confusion matrix and the report.
    test_ce, accuracy = output["loss"].item(), data.accuracy(output["logits"], test.labels)
    cue_accuracy = data.accuracy(cue_output["logits"], cue_data.labels)
    last_accuracy = data.accuracy(last_output["logits"], test.labels)
    confusion = [[0, 0], [0, 0]]
    for b in range(128):
        confusion[labels[b]][predicted[b]] += 1
    print("Confusion matrix (rows=truth, columns=prediction):\ntruth/pred     0     1")
    for r in range(2):
        print(f"{r:10d} {confusion[r][0]:5d} {confusion[r][1]:5d}")
    (root / "metrics.json").write_text(
        "{" + f"\"parameters\":{spec['parameters']},\"parameter_tensors\":{spec['tensors']},\"hidden_units\":12,"
        f"\"maximum_steps\":12,\"epochs\":{args['epochs']},\"optimizer_updates\":{updates},\"test_ce\":{test_ce:.17g},"
        f"\"test_accuracy\":{accuracy:.17g},\"cue_flip_accuracy\":{cue_accuracy:.17g},"
        f"\"last_distractor_flip_accuracy\":{last_accuracy:.17g},\"last_value_baseline_accuracy\":{last_correct / 128:.17g},"
        "\"majority_baseline_accuracy\":0.5,\"marked_value_oracle_accuracy\":1}\n")
    (root / "index.html").write_text(
        f"<!doctype html><meta charset='utf-8'><title>{spec['title']}</title><style>body{{font:16px system-ui;margin:30px;"
        f"max-width:1100px}}img{{max-width:100%;image-rendering:pixelated}}</style><h1>{spec['title']}</h1>"
        f"<p>{spec['description']}</p>"
        "<p>Synthetic marked-token memory task: 512 fit / 128 validation / 128 test, balanced labels, valid length 4..12, one "
        "marker before the last token whose +/-1 value sets the binary target; padding zeros. Three epochs, 48 updates. The task "
        "has a perfect deterministic marker-times-value oracle; it is not a natural-language benchmark.</p>"
        "<h2>First 24 held-out sequences</h2><p>12 time columns; blue=-1, red=+1, white=padding, black top strip=marked "
        "token.</p><img src='sequences.png'><h2>Target-logit input-value gradients</h2><p>Same cases, global symmetric colour "
        "peak. Masked padding derivatives are exactly zero.</p><img src='input_gradients.png'>"
        f"<p>Test CE {test_ce:g}, accuracy {accuracy:g}, cue-flip accuracy {cue_accuracy:g}, last-distractor-flip accuracy "
        f"{last_accuracy:g}, last-value baseline {last_correct / 128:g}. See metrics.json and predictions.tsv.</p>")
    history.close()
    batches.close()
    seconds = time.perf_counter() - start
    print(f"testCE={test_ce:g} accuracy={accuracy:g} cueFlip={cue_accuracy:g} lastBaseline={last_correct / 128:g};"
          f"{spec['pass_tag']} PASS ({seconds:g} s)\nArtifacts:{demo.quoted(root / 'index.html')}")
    demo.compare_with_tflow(root, args["epochs"], args["init"], args["tf"],
                            ["initial_outputs.npz", "first_forward.npz", "initial_gradients.npz", "first_after.npz",
                             "epoch1_arrays.npz", f"epoch{args['epochs']}_arrays.npz", "final_gradients.npz",
                             "outputs.npz", "cue_flip_outputs.npz", "last_flip_outputs.npz"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
