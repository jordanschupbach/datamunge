// CMD: just run-nn truncated_bptt
// Paired full and final-three-step BPTT, retaining all forward memory.
//
// Port of tflow's truncated_bptt (common/ml/truncated_bptt.h). No dataset: the
// 512/128/128 marked sequences are generated with tflow's std::mt19937 recipe
// (bit-identical data.npz). The same 206-parameter tanh RNN is trained twice
// from the same initial weights: full/ (full BPTT) and last3/ (each example
// stops gradients at max(length - 3, 0), immediately before its final three
// valid steps; the forward state is carried unchanged). Adam .01, batch 32,
// three epochs each. Checks: tflow's marked-sequence audit for both runs, a
// same-weights full-BPTT comparator proving truncation changes no forward
// value, exactly zero pre-boundary input signals in the truncated run while
// the full comparator's are nonzero, paired runs sharing bit-identical initial
// weights, data and batch order.
// Weights initialise from nn's generator (statistical parity); init=PATH
// (tflow's output/models/truncated_bptt/full/initial_arrays.npz) replays
// tflow's runs (tf=PATH is tflow's truncated_bptt directory).
// Options: epochs=3 out=PATH init=PATH tf=PATH. Output: output/nn/models/truncated_bptt/{full,last3}/.
#include <cmath>
#include <iostream>

#include "common/marked_sequence_demo.hpp"
#include "common/truncated_bptt.hpp"

// Experiment workflow from examples/nn/common/marked_sequence_demo.hpp.
namespace rnn_cells {
using namespace ::rnn_cells;
inline void RunMarkedDemo(const MarkedDemo& spec, const DemoOptions& options) {
  const auto start = std::chrono::steady_clock::now();
  const fs::path root = options.out;
  fs::create_directories(root);
  nn::NamedTensors data;
  auto             pools = MarkedPools(&data);
  nn::save_npz((root / "data.npz").string(), data);
  auto model = spec.make(true);
  check::Require(model->store.count() == spec.parameters && static_cast<std::int64_t>(model->store.size()) == spec.tensors,
                 spec.count_message);
  if (!options.init.empty()) model->LoadArrays(options.init);
  DemoContext ctx{root, pools, *model, nullptr, {}, {}, {}, model->Arrays(), {}};
  nn::save_npz((root / "initial_arrays.npz").string(), ctx.initial_arrays);
  SaveOutputs(root / "initial_outputs.npz", model->Evaluate(pools[2].feed()));
  std::ofstream history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tvalidation_ce\tvalidation_accuracy\n";
  batches << "epoch\tstep\trow\n";
  nn::BatchOrder order(512, 32, 26019);
  int            updates = 0;
  for (int epoch = 1; epoch <= options.epochs; ++epoch) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const auto x = MarkedRows(pools[0], rows);
      if (spec.before_batch) spec.before_batch(*model, x, root, updates);
      if (!updates) {
        SaveData(root / "first_batch.npz", x);
        SaveOutputs(root / "first_forward.npz", model->Evaluate(x.feed()));
        nn::save_npz((root / "initial_gradients.npz").string(), model->Gradient(x.feed()));
      }
      if (spec.before_update) spec.before_update(*model, x, root, updates);
      model->Update(x.feed());
      if (spec.after_update) spec.after_update(*model, x, root, updates);
      if (!updates) nn::save_npz((root / "first_after.npz").string(), model->Arrays());
      for (auto row : rows) batches << epoch << '\t' << updates << '\t' << row << '\n';
      ++updates;
    }
    const auto valid = model->Evaluate(pools[1].feed());
    const double ce = valid.at("loss").item<double>(), accuracy = Accuracy(valid.at("logits"), pools[1].labels);
    history << epoch << '\t' << float(ce) << '\t' << accuracy << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), model->Arrays());
    SaveOutputs(root / ("epoch" + std::to_string(epoch) + "_validation.npz"), valid);
    std::cout << "epoch" << epoch << " validationCE=" << float(ce) << " accuracy=" << accuracy << std::endl;
  }
  batches.flush();
  if (spec.after_training) spec.after_training(ctx);
  nn::save_npz((root / "final_gradients.npz").string(), model->Gradient(LoadData(root / "first_batch.npz").feed()));
  model->Save(root / "final.npz");
  auto deployed = spec.make(false);
  deployed->Restore(root / "final.npz");
  ctx.deployed = deployed.get();
  check::Require(SameBits(model->Arrays(), deployed->Arrays()), spec.restore_message);
  const auto before = deployed->Arrays();
  auto&      test   = pools[2];
  ctx.output              = deployed->Evaluate(test.feed());
  const auto  prediction  = deployed->Predict(test.feed());
  const auto& output      = ctx.output;
  for (const auto& [k, v] : prediction) check::Require(SameBits(v, output.at(k)), "labelFreePredictionMatchesEvaluation");
  nn::save_npz((root / "model_arrays.npz").string(), deployed->Arrays());
  SaveOutputs(root / "outputs.npz", output);
  if (spec.after_outputs) spec.after_outputs(ctx);
  const auto repeat = deployed->Predict(test.feed());
  for (const auto& [k, v] : prediction) check::Require(SameBits(v, repeat.at(k)), "RNNPredictRepeatBitexact");
  const auto single = deployed->Predict(MarkedRows(test, {0}).feed());
  const auto single_logits = single.at("logits").to_vector<float>(), logits = output.at("logits").to_vector<float>();
  for (int c = 0; c < 2; ++c) check::Near(single_logits[c], logits[c], 2e-6, "singleBatchRNNParityWithResetState");

  // Counterfactual copies: padding garbage, flipped marked value, flipped last valid value.
  const auto test_inputs = test.inputs.to_vector<float>();
  const auto lengths = test.lengths.to_vector<std::int32_t>(), labels = test.labels.to_vector<std::int32_t>();
  const auto input_gradient = output.at("input_gradient").to_vector<float>(), states = output.at("states").to_vector<float>();
  auto       padded = test_inputs, cue_flip = test_inputs, last_flip = test_inputs;
  auto       cue_labels = labels;
  int        oracle_correct = 0, last_correct = 0;
  auto       idx = [](int b, int t, int c) { return (static_cast<std::size_t>(b) * 12 + t) * 2 + c; };
  auto       sidx = [](int b, int t, int j) { return (static_cast<std::size_t>(b) * 13 + t) * 12 + j; };
  for (int b = 0; b < 128; ++b) {
    const int length   = lengths[b];
    double    selected = 0;
    for (int t = 0; t < 12; ++t) {
      if (t >= length) {
        padded[idx(b, t, 0)] = 123.f + b + t;
        padded[idx(b, t, 1)] = -57.f;
        for (int c = 0; c < 2; ++c) check::Require(input_gradient[idx(b, t, c)] == 0, "ALLpaddedInputLogitGradientExactlyZero");
        for (int j = 0; j < 12; ++j) check::Require(states[sidx(b, t + 1, j)] == states[sidx(b, t, j)], "stateRetainedAcrossEveryPaddingStep");
      }
      if (test_inputs[idx(b, t, 1)] > 0) {
        selected += test_inputs[idx(b, t, 0)];
        cue_flip[idx(b, t, 0)] *= -1;
      }
    }
    oracle_correct += (selected > 0) == (labels[b] != 0);
    last_correct += (test_inputs[idx(b, length - 1, 0)] > 0) == (labels[b] != 0);
    cue_labels[b] = 1 - labels[b];
    last_flip[idx(b, length - 1, 0)] *= -1;
  }
  check::Require(oracle_correct == 128, "deterministicMarkerValueOracle");
  MarkedSequences pad_data = Copy(test), cue_data = Copy(test), last_data = Copy(test);
  pad_data.inputs   = nn::from_vector(padded, {128, 12, 2});
  cue_data.inputs   = nn::from_vector(cue_flip, {128, 12, 2});
  cue_data.labels   = nn::from_vector(cue_labels);
  last_data.inputs  = nn::from_vector(last_flip, {128, 12, 2});
  const auto pad_output = deployed->Evaluate(pad_data.feed());
  ctx.cue_output        = deployed->Evaluate(cue_data.feed());
  ctx.last_output       = deployed->Evaluate(last_data.feed());
  for (const auto& [k, v] : output)
    check::Require(SameBits(v, pad_output.at(k)), "arbitraryPaddingValuesCannotAffectAnyOutputOrGradient:" + k);
  const auto last_states = ctx.last_output.at("states").to_vector<float>();
  for (int b = 0; b < 128; ++b)
    for (int t = 0; t < lengths[b]; ++t)
      for (int j = 0; j < 12; ++j)
        check::Require(last_states[sidx(b, t, j)] == states[sidx(b, t, j)], "changingLastValidTokenCannotChangeEarlierHiddenStates");
  if (spec.after_counterfactuals) spec.after_counterfactuals(ctx);
  SaveData(root / "padding_inputs.npz", pad_data);
  SaveData(root / "cue_flip_inputs.npz", cue_data);
  SaveData(root / "last_flip_inputs.npz", last_data);
  SaveOutputs(root / "padding_outputs.npz", pad_output);
  SaveOutputs(root / "cue_flip_outputs.npz", ctx.cue_output);
  SaveOutputs(root / "last_flip_outputs.npz", ctx.last_output);
  auto empty    = MarkedRows(test, {0, 1, 2, 3});
  empty.lengths = nn::zeros({4}, nn::DType::Int32);
  const auto empty_output = deployed->Evaluate(empty.feed());
  for (float v : empty_output.at("states").to_vector<float>()) check::Require(v == 0, "zeroLengthSequenceRetainsZeroInitialState");
  for (float v : empty_output.at("input_gradient").to_vector<float>()) check::Require(v == 0, "zeroLengthAllInputGradientsZero");
  SaveData(root / "zero_length_inputs.npz", empty);
  SaveOutputs(root / "zero_length_outputs.npz", empty_output);
  Heatmap(root / "sequences.png", test.inputs);
  Heatmap(root / "input_gradients.png", output.at("input_gradient"), true);
  std::ofstream predictions(root / "predictions.tsv");
  predictions << "source_id\tlength\tlabel\tpredicted\tmarked_oracle\tlast_value_baseline\n";
  const auto ids = test.source_ids.to_vector<std::int32_t>();
  for (int b = 0; b < 128; ++b)
    predictions << ids[b] << '\t' << lengths[b] << '\t' << labels[b] << '\t' << (logits[b * 2 + 1] > logits[b * 2]) << '\t' << labels[b]
                << '\t' << (test_inputs[idx(b, lengths[b] - 1, 0)] > 0) << '\n';
  if (spec.after_predictions) spec.after_predictions(ctx);
  check::Require(SameBits(before, deployed->Arrays()), "allRecurrentInferenceAndGradientControlsWeightsImmutable");
  const double test_ce = output.at("loss").item<double>(), accuracy = Accuracy(output.at("logits"), test.labels);
  const double cue_accuracy  = Accuracy(ctx.cue_output.at("logits"), cue_data.labels);
  const double last_accuracy = Accuracy(ctx.last_output.at("logits"), test.labels);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{" << spec.metrics_prefix << "\"parameters\":" << spec.parameters << ",\"parameter_tensors\":" << spec.tensors
       << ",\"hidden_units\":12,\"maximum_steps\":12,\"epochs\":" << options.epochs << ",\"optimizer_updates\":" << updates
       << ",\"test_ce\":" << float(test_ce) << ",\"test_accuracy\":" << accuracy << ",\"cue_flip_accuracy\":" << cue_accuracy
       << ",\"last_distractor_flip_accuracy\":" << last_accuracy << ",\"last_value_baseline_accuracy\":" << last_correct / 128.
       << ",\"majority_baseline_accuracy\":0.5,\"marked_value_oracle_accuracy\":1" << ctx.metrics.str() << "}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>" << spec.title
       << "</title><style>body{font:16px system-ui;margin:30px;max-width:1100px}img{max-width:100%;image-rendering:pixelated}</style>"
       << "<h1>" << spec.title << "</h1><p>" << spec.description << "</p>"
       << "<p>Synthetic marked-token memory task: 512 fit / 128 validation / 128 test, balanced labels, valid length 4..12, one "
          "marker before the last token whose +/-1 value sets the binary target; padding zeros. Three epochs, 48 updates. The task "
          "has a perfect deterministic marker-times-value oracle; it is not a natural-language benchmark.</p>"
       << "<h2>First 24 held-out sequences</h2><p>12 time columns; blue=-1, red=+1, white=padding, black top strip=marked "
          "token.</p><img src='sequences.png'><h2>Target-logit input-value gradients</h2><p>Same cases, global symmetric colour "
          "peak. Masked padding derivatives are exactly zero.</p><img src='input_gradients.png'>"
       << "<p>Test CE " << float(test_ce) << ", accuracy " << accuracy << ", cue-flip accuracy " << cue_accuracy
       << ", last-distractor-flip accuracy " << last_accuracy << ", last-value baseline " << last_correct / 128.
       << ". See metrics.json and predictions.tsv.</p>";
  check::Require(bool(meta) && bool(html) && bool(history) && bool(batches) && bool(predictions), "completeUnrolledRNNRecords");
  meta.close(), html.close(), history.close(), batches.close(), predictions.close();
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::cout << "testCE=" << float(test_ce) << " accuracy=" << accuracy << " cueFlip=" << cue_accuracy
            << " lastBaseline=" << last_correct / 128. << ";" << spec.pass_tag << " PASS (" << seconds << " s)\nArtifacts:"
            << (root / "index.html") << std::endl;
  CompareWithTflow(root, options,
                   {"initial_outputs.npz", "first_forward.npz", "initial_gradients.npz", "first_after.npz", "epoch1_arrays.npz",
                    "epoch" + std::to_string(options.epochs) + "_arrays.npz", "final_gradients.npz", "outputs.npz",
                    "cue_flip_outputs.npz", "last_flip_outputs.npz"});
}
}  // namespace rnn_cells



using namespace rnn_cells;

namespace {

MarkedDemo Spec(int window) {
  MarkedDemo d;
  d.name = "truncated_bptt", d.parameters = 206, d.tensors = 5;
  d.count_message   = "206SharedTanhRNNParameters5Tensors";
  d.restore_message = "all206WeightsBitexactRestore";
  d.metrics_prefix  = "\"window\":" + std::to_string(window) + ",";
  d.pass_tag        = "configuredBPTT/masking/causality/zeroLength/restore/labelFree/immutable";
  d.make            = [window](bool training) { return TruncatedBpttModel(training, window); };
  d.title           = "Truncated backpropagation through time";
  d.description =
      std::string(window ? "Truncated run (window 3). " : "Full-BPTT run (window 0). ") +
      "The full forward state is carried through every valid step. In the truncated case each example stops gradients at "
      "max(length-3,0), immediately before its final three valid steps; boundary values are retained. The input-gradient "
      "image therefore shows stopped-graph backward signals, not full sensitivities: early inputs still affect the forward "
      "state. Same 206-parameter tanh RNN, same initial weights, data and batches for both runs.";
  d.after_outputs = [window](DemoContext& ctx) {
    auto full = TruncatedBpttModel(false, 0);
    full->Restore(ctx.root / "final.npz");
    const auto full_output = full->Evaluate(ctx.pools[2].feed());
    SaveOutputs(ctx.root / "same_weights_full_outputs.npz", full_output);
    for (const char* key : {"logits", "probability", "states", "final_state", "loss"})
      check::Require(SameBits(full_output.at(key), ctx.output.at(key)), "truncationDoesNotChangeAnyForwardStateOrPrediction");
    if (window > 0) {
      const auto lengths = ctx.pools[2].lengths.to_vector<std::int32_t>();
      const auto ours = ctx.output.at("input_gradient").to_vector<float>(), reference = full_output.at("input_gradient").to_vector<float>();
      int        full_early_nonzero = 0;
      for (int b = 0; b < 128; ++b)
        for (int t = 0; t < std::max(0, lengths[b] - window); ++t)
          for (int c = 0; c < 2; ++c) {
            const std::size_t i = (static_cast<std::size_t>(b) * 12 + t) * 2 + c;
            check::Require(ours[i] == 0, "allPreBoundaryBackwardSignalsExactlyZero");
            full_early_nonzero += std::abs(reference[i]) > 1e-7;
          }
      check::Require(full_early_nonzero > 100, "earlyInputsStillAffectFORWARDmodelDespiteStoppedBackwardSignals");
    }
  };
  return d;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    nn::set_num_threads(4);
    const DemoOptions options = ParseDemoOptions(argc, argv, "truncated_bptt");
    for (const auto& [sub, window] : {std::pair<const char*, int>{"full", 0}, {"last3", 3}}) {
      DemoOptions o = options;
      o.out         = options.out / sub;
      o.tf          = options.tf / sub;
      std::cout << "== " << sub << " (window " << window << ")\n";
      RunMarkedDemo(Spec(window), o);
    }
    const auto a = nn::load_npz((options.out / "full/initial_arrays.npz").string());
    const auto b = nn::load_npz((options.out / "last3/initial_arrays.npz").string());
    check::Require(SameBits(a, b), "pairedInitialWeightsBitexact");
    for (const char* file : {"data.npz", "batches.tsv"})
      check::Require(ReadText(options.out / "full" / file) == ReadText(options.out / "last3" / file), "pairedSourceAndBatchFilesBitexact");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
