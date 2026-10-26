// The gated-cell variant of the marked-token memory experiment: port of
// tflow's common/ml/gated_sequence_demo.h (used by lstm, gru, peephole_lstm and
// layer_norm_lstm). Adds the gate-algebra audit (c_t = f c + i g, h_t = o tanh(c_t)
// or the layer-norm readout; GRU h_t = z h + (1 - z) n) and, for the layer-norm
// LSTM, bit-exact independence of row 0 from every other batch row.
#pragma once

#include <cmath>
#include <string>

#include "common/gated_sequence.hpp"
#include "common/marked_sequence_demo.hpp"

namespace rnn_cells {

inline MarkedDemo GatedDemo(GatedCell kind, const std::string& name) {
  MarkedDemo d;
  d.name       = name;
  d.parameters = kind == GatedCell::kLayerNorm ? 818 : kind == GatedCell::kPeephole ? 782 : kind == GatedCell::kLstm ? 746 : 566;
  d.tensors    = kind == GatedCell::kLstm ? 5 : 8;
  d.count_message   = "gatedCellExpectedParameterAndTensorCounts";
  d.restore_message = "allGatedWeightsBitexactRestore";
  d.make            = [kind](bool training) { return GatedSequenceModel(kind, training); };
  d.title = std::string(kind == GatedCell::kLayerNorm ? "Layer-normalized LSTM"
                        : kind == GatedCell::kPeephole ? "Peephole LSTM"
                        : kind == GatedCell::kLstm     ? "LSTM"
                                                       : "GRU") +
            " from an explicit masked unroll";
  d.description =
      std::to_string(d.parameters) + " parameters across 12 steps, 12 hidden units, final 12->2 linear head. " +
      (kind != GatedCell::kGru
           ? "LSTM input/forget/candidate/output gates, forget bias 1, c_t=f*c_previous+i*g, h_t=o*tanh(c_t), separate hidden "
             "and cell state paths and initial-state gradients. The peephole variant adds diagonal old-cell input/forget and "
             "NEW-cell output connections. The layer-norm variant normalizes EACH gate affine across 12 units per example and "
             "time (eps 1e-5), then gain and gate bias (forget+1); the raw cell is carried and a separately normalized cell is "
             "used ONLY for the hidden readout; no batch statistics."
           : "Reset-before GRU, z update / r reset / candidate, h_t=z*h_previous+(1-z)*candidate, candidate=tanh(xWn+(r*h_previous)Whn+bn).") +
      " Full BPTT through all valid steps with nn ops, no opaque RNN op. The explicit length mask retains the state after each "
      "sequence ends; zero initial states per sequence; prediction feeds only inputs and lengths.";
  d.after_outputs = [kind](DemoContext& ctx) {
    const auto& out   = ctx.output;
    const auto  gate  = out.at("gates").to_vector<float>();
    const auto  cells = out.at("cell_states").to_vector<float>(), states = out.at("states").to_vector<float>();
    const auto  normalized =
        kind == GatedCell::kLayerNorm ? out.at("normalized_cell").to_vector<float>() : std::vector<float>{};
    const auto lengths = ctx.pools[2].lengths.to_vector<std::int32_t>();
    const int  width   = kind == GatedCell::kGru ? 36 : 48;
    auto       g = [&](int b, int t, int k) { return double(gate[(static_cast<std::size_t>(b) * 12 + t) * width + k]); };
    auto       s = [](const std::vector<float>& v, int b, int t, int j) { return double(v[(static_cast<std::size_t>(b) * 13 + t) * 12 + j]); };
    for (int b = 0; b < 128; ++b)
      for (int t = 0; t < lengths[b]; ++t)
        for (int j = 0; j < 12; ++j) {
          if (kind != GatedCell::kGru) {
            const double expected = g(b, t, 12 + j) * s(cells, b, t, j) + g(b, t, j) * g(b, t, 24 + j);
            check::Near(expected, s(cells, b, t + 1, j), 2e-6, "actualLSTMcellRetentionAndInputGateAlgebra");
            const double readout =
                kind == GatedCell::kLayerNorm ? double(normalized[(static_cast<std::size_t>(b) * 12 + t) * 12 + j]) : expected;
            check::Near(g(b, t, 36 + j) * std::tanh(readout), s(states, b, t + 1, j), 2e-6, "actualLSTMoutputGateAlgebra");
          } else {
            const double expected = g(b, t, j) * s(states, b, t, j) + (1 - g(b, t, j)) * g(b, t, 24 + j);
            check::Near(expected, s(states, b, t + 1, j), 2e-6, "actualGRUupdateAndCandidateAlgebra");
          }
        }
  };
  if (kind == GatedCell::kLayerNorm)
    d.after_predictions = [](DemoContext& ctx) {
      auto companions = Copy(ctx.pools[2]);
      auto x          = companions.inputs.to_vector<float>();
      for (int b = 1; b < 128; ++b)
        for (int t = 0; t < 12; ++t)
          for (int c = 0; c < 2; ++c) x[(static_cast<std::size_t>(b) * 12 + t) * 2 + c] = float((b + t + c) % 7 - 3) * 17;
      companions.inputs  = nn::from_vector(x, {128, 12, 2});
      const auto altered = ctx.deployed->Predict(companions.feed());
      const auto a = altered.at("logits").to_vector<float>(), o = ctx.output.at("logits").to_vector<float>();
      for (int c = 0; c < 2; ++c)
        check::Require(a[c] == o[c], "layerNormFirstPredictionBitexactUnderAllOtherBatchRowChanges");
      SaveData(ctx.root / "batch_companion_inputs.npz", companions);
      SaveOutputs(ctx.root / "batch_companion_outputs.npz", altered);
    };
  return d;
}

}  // namespace rnn_cells
