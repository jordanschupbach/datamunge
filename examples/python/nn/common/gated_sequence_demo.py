"""The gated-cell variant of the marked-token memory experiment
(examples/nn/common/gated_sequence_demo.hpp; tflow's common/ml/gated_sequence_demo.h,
used by lstm, gru, peephole_lstm and layer_norm_lstm): parameter counts, the
report text and the gate-algebra audit (c_t = f c + i g, h_t = o tanh(c_t) or the
layer-norm readout; GRU h_t = z h + (1 - z) n). The experiment loop is written
out in each example script."""
import math

from pydatamunge.nn_workflows import helper

checks = helper("checks")
gated_sequence = helper("gated_sequence")
LSTM, GRU, PEEPHOLE, LAYER_NORM = gated_sequence.LSTM, gated_sequence.GRU, gated_sequence.PEEPHOLE, gated_sequence.LAYER_NORM


def gated_demo_spec(kind):
    """Counts, check messages and report text of GatedDemo(kind)."""
    parameters = {LAYER_NORM: 818, PEEPHOLE: 782, LSTM: 746, GRU: 566}[kind]
    title = {LAYER_NORM: "Layer-normalized LSTM", PEEPHOLE: "Peephole LSTM", LSTM: "LSTM", GRU: "GRU"}[kind]
    if kind != GRU:
        cell = ("LSTM input/forget/candidate/output gates, forget bias 1, c_t=f*c_previous+i*g, h_t=o*tanh(c_t), separate "
                "hidden and cell state paths and initial-state gradients. The peephole variant adds diagonal old-cell "
                "input/forget and NEW-cell output connections. The layer-norm variant normalizes EACH gate affine across 12 "
                "units per example and time (eps 1e-5), then gain and gate bias (forget+1); the raw cell is carried and a "
                "separately normalized cell is used ONLY for the hidden readout; no batch statistics.")
    else:
        cell = ("Reset-before GRU, z update / r reset / candidate, h_t=z*h_previous+(1-z)*candidate, "
                "candidate=tanh(xWn+(r*h_previous)Whn+bn).")
    description = (f"{parameters} parameters across 12 steps, 12 hidden units, final 12->2 linear head. " + cell +
                   " Full BPTT through all valid steps with nn ops, no opaque RNN op. The explicit length mask retains the "
                   "state after each sequence ends; zero initial states per sequence; prediction feeds only inputs and lengths.")
    return dict(parameters=parameters, tensors=5 if kind == LSTM else 8,
                count_message="gatedCellExpectedParameterAndTensorCounts",
                restore_message="allGatedWeightsBitexactRestore",
                title=title + " from an explicit masked unroll", description=description,
                pass_tag="fullBPTT/masking/causality/zeroLength/restore/labelFree/immutable")


def gate_algebra_audit(kind, output, lengths):
    """Checks the cell algebra on every valid step of the 128 test rows from the recorded gates and states."""
    gate = output["gates"].values()
    cells, states = output["cell_states"].values(), output["states"].values()
    normalized = output["normalized_cell"].values() if kind == LAYER_NORM else None
    width = 36 if kind == GRU else 48
    for b in range(128):
        for t in range(lengths[b]):
            g0 = (b * 12 + t) * width
            s0, s1 = (b * 13 + t) * 12, (b * 13 + t + 1) * 12
            for j in range(12):
                if kind != GRU:
                    expected = gate[g0 + 12 + j] * cells[s0 + j] + gate[g0 + j] * gate[g0 + 24 + j]
                    checks.near(expected, cells[s1 + j], 2e-6, "actualLSTMcellRetentionAndInputGateAlgebra")
                    readout = normalized[(b * 12 + t) * 12 + j] if kind == LAYER_NORM else expected
                    checks.near(gate[g0 + 36 + j] * math.tanh(readout), states[s1 + j], 2e-6, "actualLSTMoutputGateAlgebra")
                else:
                    expected = gate[g0 + j] * states[s0 + j] + (1 - gate[g0 + j]) * gate[g0 + 24 + j]
                    checks.near(expected, states[s1 + j], 2e-6, "actualGRUupdateAndCandidateAlgebra")
