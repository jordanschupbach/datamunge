"""Records and tflow comparisons for the marked-token memory experiments of
tflow's 4.1 recurrent-cell examples (examples/nn/common/marked_sequence_demo.hpp).

The experiment itself (data pools, training loop, evaluations, counterfactual
audits, records) is written out in each example script; this module holds the
option defaults, .npz/JSON utilities and the comparison with tflow's run
directory. Parity with tflow: the data and batch order are bit-identical;
parameters initialise from nn's generator, so `init=PATH` (tflow's
initial_arrays.npz) is needed to replay tflow's run."""
import math
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_workflows import helper, output_root

recurrent_model = helper("recurrent_model")


def tflow_run_dir(name):
    """tflow's output/models/<name> (where its examples write their runs)."""
    return Path.home() / "projects" / "tflow_examples" / "output" / "models" / name


def demo_defaults(name, full=False):
    """Option defaults of ParseDemoOptions: epochs=3 out=PATH init=PATH tf=PATH."""
    return dict(epochs=3, out=str(output_root(full) / "models" / name), init="", tf=str(tflow_run_dir(name)), full=full)


def check_epochs(epochs):
    if not (0 < epochs <= 100):
        raise RuntimeError("threeCheapDefaultEpochs")


def quoted(path):
    """std::filesystem::path stream output (quoted)."""
    return '"' + str(path) + '"'


def save_outputs(path, outputs):
    nn.save_npz(str(path), recurrent_model.named(outputs))


def compare_npz(ours, theirs):
    """Max |ours - tflow| over the arrays both .npz files contain; -1 when tflow's file is missing."""
    if not Path(theirs).exists():
        return -1.0
    a, b = nn.load_npz(str(ours)), nn.load_npz(str(theirs))
    worst = 0.0
    for name, value in a:
        for other, tf_value in b:
            if name == other:
                worst = max(worst, recurrent_model.max_abs_diff(value, tf_value))
    return worst


def read_text(path):
    with open(path, "rb") as handle:
        return handle.read()


def json_number(text, key):
    """The number after "key": in a flat JSON object (NaN when absent)."""
    text = text.decode() if isinstance(text, bytes) else text
    at = text.find('"' + key + '":')
    if at < 0:
        return math.nan
    rest = text[at + len(key) + 3:]
    end = 0
    while end < len(rest) and rest[end] in "+-.0123456789eE":
        end += 1
    return float(rest[:end])


def format_g(x):
    """C++ default ostream formatting of a double (6 significant digits, %g)."""
    return f"{x:g}"


def compare_with_tflow(root, epochs, init, tf, records, test_key="test_ce"):
    """Compares this run with tflow's run directory: data and batch order always
    (identical; with another epoch budget the shorter batch record is a prefix of
    the longer), and with init= every replayed record."""
    root, tf = Path(root), Path(tf) if tf else None
    if tf is None or not (tf / "data.npz").exists():
        print(f"tflow run {quoted(tf or '')} not found; skipping the comparison")
        return
    if compare_npz(root / "data.npz", tf / "data.npz") != 0:
        raise RuntimeError("dataBitIdenticalToTflow")
    if (tf / "batches.tsv").exists():
        ours, theirs = read_text(root / "batches.tsv"), read_text(tf / "batches.tsv")
        if not (ours.startswith(theirs) or theirs.startswith(ours)):
            raise RuntimeError("batchOrderIdenticalToTflow")
    line = "vs tflow: data.npz and batches.tsv identical"
    if not init:
        print(line + f" (pass init={tf / 'initial_arrays.npz'} to replay tflow's run)")
        return
    worst_initial = worst_final = 0.0
    for name in records:
        d = compare_npz(root / name, tf / name)
        if d < 0:
            continue
        line += f"\n  {name} max|diff|={format_g(d)}"
        if name.startswith("initial") or name.startswith("first"):
            worst_initial = max(worst_initial, d)
        else:
            worst_final = max(worst_final, d)
    print(line)
    if worst_initial > 1e-5:
        raise RuntimeError("replayedInitialForwardGradientsAndFirstUpdateMatchTflow")
    ours = json_number(read_text(root / "metrics.json"), test_key)
    theirs = json_number(read_text(tf / "metrics.json"), test_key)
    if not math.isnan(theirs):
        print(f"  {test_key} ours={format_g(ours)} tflow={format_g(theirs)} (trained records max|diff|={format_g(worst_final)})")
        if abs(ours - theirs) > 5e-3:
            raise RuntimeError("replayedTrainedTestLossMatchesTflow")
