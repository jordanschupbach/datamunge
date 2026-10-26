# The gated-cell variant of the marked-token memory experiment
# (examples/nn/common/gated_sequence_demo.hpp; tflow's common/ml/gated_sequence_demo.h,
# used by lstm, gru, peephole_lstm and layer_norm_lstm): parameter counts, the
# report text and the gate-algebra audit (c_t = f c + i g, h_t = o tanh(c_t) or the
# layer-norm readout; GRU h_t = z h + (1 - z) n). The experiment loop is written
# out in each example script.
if (!exists("gated_sequence_model", inherits = TRUE)) nn_helper("gated_sequence")

# Counts, check messages and report text of GatedDemo(kind).
gated_sequence_demo_spec <- function(kind) {
    parameters <- c(layer_norm = 818, peephole = 782, lstm = 746, gru = 566)[[kind]]
    title <- c(layer_norm = "Layer-normalized LSTM", peephole = "Peephole LSTM", lstm = "LSTM", gru = "GRU")[[kind]]
    cell <- if (kind != "gru") {
        paste0("LSTM input/forget/candidate/output gates, forget bias 1, c_t=f*c_previous+i*g, h_t=o*tanh(c_t), separate ",
            "hidden and cell state paths and initial-state gradients. The peephole variant adds diagonal old-cell ",
            "input/forget and NEW-cell output connections. The layer-norm variant normalizes EACH gate affine across 12 ",
            "units per example and time (eps 1e-5), then gain and gate bias (forget+1); the raw cell is carried and a ",
            "separately normalized cell is used ONLY for the hidden readout; no batch statistics.")
    } else {
        paste0("Reset-before GRU, z update / r reset / candidate, h_t=z*h_previous+(1-z)*candidate, ",
            "candidate=tanh(xWn+(r*h_previous)Whn+bn).")
    }
    list(parameters = parameters, tensors = if (kind == "lstm") 5 else 8,
        count_message = "gatedCellExpectedParameterAndTensorCounts",
        restore_message = "allGatedWeightsBitexactRestore",
        title = paste(title, "from an explicit masked unroll"),
        description = paste0(parameters, " parameters across 12 steps, 12 hidden units, final 12->2 linear head. ", cell,
            " Full BPTT through all valid steps with nn ops, no opaque RNN op. The explicit length mask retains the ",
            "state after each sequence ends; zero initial states per sequence; prediction feeds only inputs and lengths."),
        pass_tag = "fullBPTT/masking/causality/zeroLength/restore/labelFree/immutable")
}

# Checks the cell algebra on every valid step of the 128 test rows from the recorded gates and states.
gated_sequence_demo_gate_algebra_audit <- function(kind, output, lengths) {
    width <- if (kind == "gru") 36 else 48
    gate <- output$gates$values()
    states <- output$states$values()
    cells <- output$cell_states$values()
    normalized <- if (kind == "layer_norm") output$normalized_cell$values() else NULL
    for (b in 0:127) {
        if (lengths[b + 1] < 1) next
        t <- rep(0:(lengths[b + 1] - 1), each = 12)
        j <- rep(0:11, lengths[b + 1])
        g <- function(k) gate[(b * 12 + t) * width + k + j + 1]
        before <- (b * 13 + t) * 12 + j + 1
        after <- (b * 13 + t + 1) * 12 + j + 1
        if (kind != "gru") {
            expected <- g(12) * cells[before] + g(0) * g(24)
            for (i in seq_along(expected))
                nn_near(expected[i], cells[after[i]], 2e-6, "actualLSTMcellRetentionAndInputGateAlgebra")
            readout <- if (kind == "layer_norm") normalized[(b * 12 + t) * 12 + j + 1] else expected
            output_gate <- g(36) * tanh(readout)
            for (i in seq_along(output_gate))
                nn_near(output_gate[i], states[after[i]], 2e-6, "actualLSTMoutputGateAlgebra")
        } else {
            expected <- g(0) * states[before] + (1 - g(0)) * g(24)
            for (i in seq_along(expected))
                nn_near(expected[i], states[after[i]], 2e-6, "actualGRUupdateAndCandidateAlgebra")
        }
    }
}
