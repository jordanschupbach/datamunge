# Native Python NN example: ntm, explicit loop over canonical C++ memory components.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/4_recurrent_and_sequence_models/4.3_memory_augmented_networks/ntm.py
"""Options: epochs=N out=PATH init=PATH tf=PATH. Data: $DATAMUNGE_DATA_DIR/ntm.
"""
import json
import math
import shutil
import time
from pathlib import Path
from pydatamunge import nn
from pydatamunge.nn_components import memory_components as component, check
from pydatamunge.nn_workflows import data_root, output_root, options, helper
records = helper("memory_records")
NAME, DATASET = "ntm", "ntm"
KIND = component.Kind_Ntm
COUNT, TENSORS, SEED, LR = 851, 8, 45107, .005
FIRST_FORWARD = False
EMPTY_FIELDS = ['keys', 'values', 'attention', 'reads', 'input_gradient']
STREAM_TOL, PADDING = 2e-5, .9
DESCRIPTION = '<p>Binary copy: three-bit vectors, a delimiter, then recall. Canonical Ntm memory model, masked recall-bit cross-entropy, Adam and three default epochs.</p>'


def main(arguments=None):
    started = time.perf_counter()
    nn.set_num_threads(4)
    args = options(arguments, dict(epochs=3, out="", init="",
        tf=str(Path.home() / "projects/tflow_examples/output/models" / NAME)))
    epochs = args["epochs"]
    check.Require(0 < epochs <= 100, "three cheap default epochs")
    root = Path(args["out"]) if args["out"] else output_root() / "models" / NAME
    root.mkdir(parents=True, exist_ok=True)
    source = data_root() / DATASET
    component.PrepareDataset(str(source), True)
    for filename in ("data.npz", "manifest.json"):
        if (source / filename).exists():
            shutil.copyfile(source / filename, root / filename)
    component.Fixtures(KIND, str(root))
    all_arrays = dict(nn.load_npz(str(root / "data.npz")))
    keys = ['input', 'target', 'lengths', 'source_ids']
    fit, validation, test = [{key: all_arrays[split + "_" + key] for key in keys}
                              for split in ("fit", "validation", "test")]
    example = records.rows(fit, [0])
    model = component.Model(KIND, records.records(example))
    params = model.parameters()
    check.Require(params.count() == COUNT and params.size() == TENSORS, "canonical parameter census")
    if args["init"]:
        model.Load(args["init"])
    cfg = nn.OptimizerConfig()
    cfg.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(params, LR, cfg)

    def named_gradients(batch):
        values = model.Forward(model.Leaves(records.records(batch)), records.records(batch))
        loss = model.Loss(values, records.records(batch))
        grad_options = nn.GradOptions()
        grad_options.allow_unused = True
        tensors = list(params.parameters())
        gradients = nn.grad(loss, tensors, grad_options)
        return [(key, gradient.detach() if gradient.defined() else nn.zeros_like(tensors[i].detach()))
                for i, (key, gradient) in enumerate(zip(params.names(), gradients))]

    records.save(root / "initial_arrays.npz", model.Arrays())
    records.save(root / "initial_outputs.npz", records.evaluate(model, test))
    order = nn.BatchOrder(next(iter(fit.values())).size(0), 8, SEED)
    updates, first = 0, None
    with (root / "history.tsv").open("w") as history, (root / "batches.tsv").open("w") as batches:
        history.write("epoch\tvalidation_bit_ce\n")
        batches.write("epoch\tstep\trow\n")
        for epoch in range(1, epochs + 1):
            order.begin_epoch()
            while True:
                rows = list(order.next())
                if not rows:
                    break
                batch = records.rows(fit, rows)
                if not updates:
                    first = batch
                    records.save(root / "first_batch.npz", batch)
                    records.save(root / "initial_gradients.npz", named_gradients(batch))
                    if FIRST_FORWARD:
                        records.save(root / "first_forward.npz", records.evaluate(model, batch))
                leaves = model.Leaves(records.records(batch))
                values = model.Forward(leaves, records.records(batch))
                loss = model.Loss(values, records.records(batch))
                grad_options = nn.GradOptions()
                grad_options.allow_unused = True
                tensors = list(params.parameters())
                gradients = nn.grad(loss, tensors, grad_options)
                gradients = [gradient if gradient.defined() else nn.zeros_like(tensors[i].detach())
                             for i, gradient in enumerate(gradients)]
                check.Require(math.isfinite(loss.item()), "finite training loss")
                optimizer.step(gradients)
                if not updates:
                    records.save(root / "first_after.npz", model.Arrays())
                for row in rows:
                    batches.write(f"{epoch}\t{updates}\t{row}\n")
                updates += 1
            ce = records.evaluate(model, validation)["loss"].item()
            check.Require(math.isfinite(ce), "finite validation CE")
            history.write(f"{epoch}\t{ce:.6g}\n")
            print(f"epoch={epoch} validationBitCE={ce:.6g}", flush=True)
    records.save(root / "final_gradients.npz", named_gradients(first))
    if FIRST_FORWARD:
        records.save(root / "final_first_forward.npz", records.evaluate(model, first))
    records.save(root / "final.npz", model.Arrays())
    deployed = component.Model(KIND, records.records(example))
    deployed.Load(str(root / "final.npz"))
    weights = dict(deployed.Arrays())
    check.Require(records.same(dict(model.Arrays()), weights), "restore bitexact")
    out = records.evaluate(deployed, test)
    logits = deployed.Predict(records.records(test))
    check.Require(records.same(logits, out["logits"]) and
                  records.same(logits, deployed.Predict(records.records(test))), "label-free repeat inference")
    lengths = [int(v) for v in test["lengths"].values()]
    state_names = list(deployed.StateNames())
    for step_index in range(9):
        state = {key: nn.select(out[key], 1, step_index) for key in state_names}
        step = dict(deployed.Step(nn.select(test["input"], 1, step_index), records.records(state)))
        active = [row for row, length in enumerate(lengths) if step_index <= 2 * length]
        for key, value in step.items():
            expected = nn.select(out[key], 1, step_index + int(key in state_names))
            if active:
                error = nn.max(nn.abs(nn.sub(nn.take_rows(value, active), nn.take_rows(expected, active)))).item()
                check.Near(error, 0, STREAM_TOL, "native state/access streaming parity")
    normal_in = records.rows(test, [0, 1])
    normal = records.evaluate(deployed, normal_in)
    padded_values = list(normal_in["input"].values())
    for batch, length in enumerate(lengths[:2]):
        for step_index in range(2 * length + 1, 9):
            for channel in range(4):
                padded_values[(batch * 9 + step_index) * 4 + channel] = PADDING
    pad = dict(normal_in)
    pad["input"] = nn.from_vector(padded_values, [2, 9, 4])
    padout = records.evaluate(deployed, pad)
    check.Require(records.same(normal, padout), "all state/loss/gradient padding invariance")
    records.save(root / "padding_inputs.npz", pad)
    records.save(root / "padding_outputs.npz", padout)
    predictions, targets = logits.values(), test["target"].values()
    confusion, correct, total, exact = [[0, 0], [0, 0]], 0, 0, 0
    for batch, length in enumerate(lengths):
        match = True
        for step_index in range(length + 1, 2 * length + 1):
            for bit in range(3):
                index = (batch * 9 + step_index) * 3 + bit
                truth, predicted = int(targets[index] > 0), int(predictions[index] > 0)
                confusion[truth][predicted] += 1
                correct += truth == predicted
                total += 1
                match = match and truth == predicted
        exact += match
    check.Require(sum(map(sum, confusion)) == total and confusion[0][0] + confusion[1][1] == correct,
                  "all held-out recall bits counted")
    print("confusion (truth rows, predicted columns; recall bits):")
    for row in confusion:
        print(" ".join(str(v) for v in row))
    metrics = dict(parameters=COUNT, epochs=epochs, updates=updates, test_bit_ce=out["loss"].item(),
                   bit_accuracy=correct / total, exact=exact)
    summary = f"testBitCE={metrics['test_bit_ce']:.6g} bitAccuracy={correct / total:.6g} exact={exact}/32;restore/nativeStreaming/padding/labelFree/immutable PASS"

    records.save(root / "outputs.npz", out)
    records.save(root / "model_arrays.npz", deployed.Arrays())
    check.Require(records.same(dict(deployed.Arrays()), weights), "immutable inference")
    (root / "metrics.json").write_text(json.dumps(metrics) + "\n")
    (root / "index.html").write_text("<!doctype html><meta charset='utf-8'><title>" + NAME +
        "</title><style>body{font:16px system-ui;margin:30px;max-width:1000px}</style><h1>" + NAME +
        "</h1>" + DESCRIPTION + "<p>" + summary + "</p><p>See metrics.json, history.tsv and the .npz records.</p>")
    print(summary + f" ({time.perf_counter() - started:.3g} s)\nArtifacts:{root / 'index.html'}")
    comparison = ["first_batch.npz", "initial_arrays.npz", "initial_outputs.npz", "initial_gradients.npz",
                  "first_after.npz", "final_gradients.npz", "model_arrays.npz", "outputs.npz"]
    if FIRST_FORWARD:
        comparison += ["first_forward.npz", "final_first_forward.npz"]
    component.CompareRecords(str(root), args["tf"], args["init"], comparison, "test_bit_ce")
    for filename in ['address_fixture.npz']:
        if args["tf"] and (Path(args["tf"]) / filename).exists():
            difference = component.CompareNpz(str(root / filename), str(Path(args["tf"]) / filename))
            check.Require(difference <= 1e-5, filename + " matches recorded fixture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
