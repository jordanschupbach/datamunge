// CMD: just run-nn gradient_clipping
// Paired gradient clipping policies and actual vanishing/exploding chain
// derivatives; three epochs per case.
//
// Port of tflow's gradient_clipping. No dataset: the 512/128/128 marked
// sequences are generated with tflow's std::mt19937 recipe (bit-identical
// data.npz). First the derivative of linear and tanh chains (gains .5, 1, 1.5;
// lengths 1..64) through autograd, checked against gain^length, and the two
// policies on a fixed 4-vector. Then three paired runs of the 206-parameter
// unrolled tanh RNN with SGD .1: no policy, global-norm scaling to .05 and
// coordinate clipping to +-.01 (same initial weights, data and batch order;
// before/raw/applied/after records for every update), each with tflow's
// common marked-sequence audit (counts, restore, label-free, repeat/single
// parity, padding, last-token causality, zero length, immutability).
// Weights initialise from nn's generator (statistical parity); init=PATH
// (tflow's output/models/gradient_clipping/none/initial_arrays.npz, the same
// for all three cases) replays tflow's runs and compares them with tflow's
// run directories (tf=PATH, the parent of none/global/coordinate).
// Options: epochs=3 out=PATH init=PATH tf=PATH. Output: output/nn/models/gradient_clipping/.
#include <cmath>
#include <fstream>

#include <cstdio>

#include "common/gradient_clipping.hpp"
#include "common/marked_sequence_demo.hpp"

// Experiment workflow from examples/nn/common/marked_sequence_demo.hpp.
namespace rnn_cells {
using namespace ::rnn_cells;
// The eager equivalents of tflow's session fetches, written out in this
// example: forward pass, loss and nn::grad (RecurrentModel only holds the
// parameters, the forward function and the optimizer).

/// Forward fields, loss and the full-BPTT gradients of the chosen logit with
/// respect to the inputs and every zero initial state (tflow's Evaluate fetches).
inline Outputs Evaluate(RecurrentModel& m, const Feed& x) {
  Tensor              input    = x.inputs.detach().clone();
  std::vector<Tensor> initials = m.Initials(x.inputs.shape()[0]);
  input.requires_grad_(true);
  for (auto& s : initials) s.requires_grad_(true);
  Outputs out        = m.forward(m.store, input, x.lengths, initials);
  out["probability"] = m.probability(out.at("logits"));
  const auto [loss, chosen] = m.loss_fn(out, x.labels);
  std::vector<Tensor> wrt{input};
  wrt.insert(wrt.end(), initials.begin(), initials.end());
  nn::GradOptions options;
  options.retain_graph = true;
  options.allow_unused = true;
  const auto g = nn::grad(std::vector<Tensor>{chosen}, wrt, {}, options);
  Outputs    result;
  for (const auto& f : m.fields) result[f] = out.at(f).detach();
  result["loss"]           = loss.detach();
  result["input_gradient"] = DefinedGradient(g[0], input);
  for (std::size_t k = 0; k < m.initial_gradient_names.size(); ++k)
    if (!m.initial_gradient_names[k].empty()) result[m.initial_gradient_names[k]] = DefinedGradient(g[k + 1], initials[k]);
  for (const auto& probe : m.evaluate_probes) {
    const Tensor& target = wrt.at(static_cast<std::size_t>(probe.wrt));
    result[probe.name] = DefinedGradient(nn::grad(std::vector<Tensor>{probe.target(out)}, {target}, {}, options)[0], target);
  }
  return result;
}

/// Loss gradient of every parameter, in store order (zeros where unused).
inline std::vector<Tensor> LossGradients(RecurrentModel& m, const Feed& x) {
  const Outputs out    = m.forward(m.store, x.inputs, x.lengths, m.Initials(x.inputs.shape()[0]));
  const Tensor  loss   = m.loss_fn(out, x.labels).first;
  const auto    params = m.store.parameters();
  nn::GradOptions options;
  options.allow_unused = true;
  auto g = nn::grad(std::vector<Tensor>{loss}, params, {}, options);
  for (std::size_t i = 0; i < g.size(); ++i) g[i] = DefinedGradient(g[i], params[i]);
  return g;
}

/// Gradients by parameter name (the initial/final_gradients.npz records).
inline nn::NamedTensors NamedGradients(const RecurrentModel& m, const std::vector<Tensor>& g) {
  nn::NamedTensors result;
  for (std::size_t i = 0; i < g.size(); ++i) result.emplace_back(m.store.names()[i], g[i]);
  return result;
}

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
  SaveOutputs(root / "initial_outputs.npz", Evaluate(*model, pools[2].feed()));
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
        SaveOutputs(root / "first_forward.npz", Evaluate(*model, x.feed()));
        nn::save_npz((root / "initial_gradients.npz").string(), NamedGradients(*model, LossGradients(*model, x.feed())));
      }
      if (spec.before_update) spec.before_update(*model, x, root, updates);
      // One optimizer update written out: forward, loss, the gradient of every
      // parameter, the optional gradient policy, the optimizer step and the
      // optional projection.
      {
        const Outputs out    = model->forward(model->store, x.inputs, x.lengths, model->Initials(x.inputs.shape()[0]));
        const Tensor  loss   = model->loss_fn(out, x.labels).first;
        const auto    params = model->store.parameters();
        nn::GradOptions grad_options;
        grad_options.allow_unused = true;
        auto g = nn::grad(std::vector<Tensor>{loss}, params, {}, grad_options);
        for (std::size_t i = 0; i < g.size(); ++i) g[i] = DefinedGradient(g[i], params[i]);
        if (model->gradient_policy) g = model->gradient_policy(g);
        model->optimizer->step(g);
        if (model->after_update) model->after_update(model->store);
      }
      if (spec.after_update) spec.after_update(*model, x, root, updates);
      if (!updates) nn::save_npz((root / "first_after.npz").string(), model->Arrays());
      for (auto row : rows) batches << epoch << '\t' << updates << '\t' << row << '\n';
      ++updates;
    }
    const auto valid = Evaluate(*model, pools[1].feed());
    const double ce = valid.at("loss").item<double>(), accuracy = Accuracy(valid.at("logits"), pools[1].labels);
    history << epoch << '\t' << float(ce) << '\t' << accuracy << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), model->Arrays());
    SaveOutputs(root / ("epoch" + std::to_string(epoch) + "_validation.npz"), valid);
    std::cout << "epoch" << epoch << " validationCE=" << float(ce) << " accuracy=" << accuracy << std::endl;
  }
  batches.flush();
  if (spec.after_training) spec.after_training(ctx);
  nn::save_npz((root / "final_gradients.npz").string(), NamedGradients(*model, LossGradients(*model, LoadData(root / "first_batch.npz").feed())));
  model->Save(root / "final.npz");
  auto deployed = spec.make(false);
  deployed->Restore(root / "final.npz");
  ctx.deployed = deployed.get();
  check::Require(SameBits(model->Arrays(), deployed->Arrays()), spec.restore_message);
  const auto before = deployed->Arrays();
  auto&      test   = pools[2];
  ctx.output              = Evaluate(*deployed, test.feed());
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
  const auto pad_output = Evaluate(*deployed, pad_data.feed());
  ctx.cue_output        = Evaluate(*deployed, cue_data.feed());
  ctx.last_output       = Evaluate(*deployed, last_data.feed());
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
  const auto empty_output = Evaluate(*deployed, empty.feed());
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
  {
    // Binary confusion matrix over the whole test split (rows = truth, columns = prediction).
    const auto y = test.labels.to_vector<std::int32_t>();
    int        confusion[2][2] = {{0, 0}, {0, 0}};
    for (int b = 0; b < 128; ++b) ++confusion[y[b]][logits[b * 2 + 1] > logits[b * 2] ? 1 : 0];
    std::cout << "Confusion matrix (rows=truth, columns=prediction):\ntruth/pred     0     1\n";
    for (int r = 0; r < 2; ++r) std::printf("%10d %5d %5d\n", r, confusion[r][0], confusion[r][1]);
  }
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

void Dynamics(const fs::path& root) {
  fs::create_directories(root);
  const int    lengths[7] = {1, 2, 4, 8, 16, 32, 64};
  const float  gains[3]   = {.5f, 1.f, 1.5f};
  std::vector<float> state, gradient;
  for (float gain : gains)
    for (int length : lengths) {
      Tensor initial = nn::full({}, .1f);
      initial.requires_grad_(true);
      Tensor linear = initial, nonlinear = initial;
      for (int t = 0; t < length; ++t) {
        linear    = nn::mul(linear, double(gain));
        nonlinear = nn::tanh(nn::mul(nonlinear, double(gain)));
      }
      for (const Tensor& value : {linear, nonlinear}) {
        state.push_back(value.item<float>());
        gradient.push_back(nn::grad(value, {initial})[0].item<float>());
      }
    }
  const Tensor fixture    = nn::from_vector(std::vector<float>{3, 4, .01f, -.02f});
  const auto   global     = ControlGradients({fixture}, 1, 1.f, 1.f).values[0];
  const auto   coordinate = ControlGradients({fixture}, 2, 1.f, 1.f).values[0];
  const Tensor states = nn::from_vector(state, {3, 7, 2}), gradients = nn::from_vector(gradient, {3, 7, 2});
  nn::save_npz((root / "gradient_dynamics.npz").string(),
               {{"outputs", states}, {"gradients", gradients}, {"fixture", fixture}, {"global", global}, {"coordinate", coordinate}});
  std::ofstream table(root / "gradient_dynamics.tsv");
  table.precision(17);
  table << "gain\tlength\tlinear_gradient\ttanh_gradient\n";
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 7; ++j) {
      const double expected = std::pow(double(gains[i]), lengths[j]);
      check::Near(gradient[(i * 7 + j) * 2] / expected, 1, 2e-6, "actualLinearChainDerivativeGainToLength");
      table << gains[i] << '\t' << lengths[j] << '\t' << gradient[(i * 7 + j) * 2] << '\t' << gradient[(i * 7 + j) * 2 + 1] << '\n';
    }
  const auto g = global.to_vector<float>(), c = coordinate.to_vector<float>();
  std::cout << "chains: d/dx0 of 1.5^64 x0 = " << gradient[(2 * 7 + 6) * 2] << ", tanh chain gain 1.5 length 64 = "
            << gradient[(2 * 7 + 6) * 2 + 1] << "; fixture (3,4,.01,-.02): global-norm-1 -> (" << g[0] << "," << g[1] << "," << g[2]
            << "," << g[3] << "), coordinate-1 -> (" << c[0] << "," << c[1] << "," << c[2] << "," << c[3] << ")\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Gradient control</title><h1>Gradient control and temporal dynamics</h1><p>Three "
          "paired short runs: <a href='none/index.html'>none</a>, <a href='global/index.html'>global norm</a>, <a "
          "href='coordinate/index.html'>coordinate clipping</a>. Each uses three epochs; see case metrics and audits.</p><p>Autograd "
          "through linear and tanh chains with gains .5, 1, 1.5 and lengths 1..64 is saved in <a href='gradient_dynamics.tsv'>this "
          "table</a>. A linear derivative is gain^length: .5 vanishes, 1 stays 1, 1.5 explodes. Tanh saturation can make a gain 1.5 "
          "chain vanish instead. Clipping limits large update signals; it cannot recover lost derivatives.</p>";
}

MarkedDemo Case(int policy) {
  MarkedDemo d;
  d.name = "gradient_clipping", d.parameters = 206, d.tensors = 5;
  d.count_message   = "206SharedTanhRNNParameters5Tensors";
  d.restore_message = "all206WeightsBitexactRestore";
  d.make            = [policy](bool training) { return GradientClippingModel(training, policy); };
  d.metrics_prefix  = "\"policy\":" + std::to_string(policy) + ",";
  d.title           = "Gradient clipping and temporal dynamics";
  d.description     = std::string("Unrolled 12-unit tanh RNN, 206 parameters, plain SGD .1 with gradient policy: ") +
                  (policy == 0   ? "none."
                   : policy == 1 ? "one global scale min(1, .05 / ||g||) over all parameter gradients (preserves direction)."
                                 : "coordinate clipping of every gradient entry to [-.01, .01] (generally changes direction).") +
                  " Every update writes step*_before/raw/applied/after records.";
  d.before_update = [policy](RecurrentModel& m, const MarkedSequences& x, const fs::path& root, int step) {
    const std::string prefix = "step" + std::to_string(step);
    nn::save_npz((root / (prefix + "_before.npz")).string(), m.Arrays());
    const auto raw = LossGradients(m, x.feed());
    nn::save_npz((root / (prefix + "_raw.npz")).string(), NamedGradients(m, raw));
    nn::save_npz((root / (prefix + "_applied.npz")).string(), AppliedGradients(m, raw, policy));
  };
  d.after_update = [](RecurrentModel& m, const MarkedSequences&, const fs::path& root, int step) {
    nn::save_npz((root / ("step" + std::to_string(step) + "_after.npz")).string(), m.Arrays());
  };
  return d;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    nn::set_num_threads(4);
    const auto options = ParseDemoOptions(argc, argv, "gradient_clipping");
    Dynamics(options.out);
    const char* names[3] = {"none", "global", "coordinate"};
    for (int policy = 0; policy < 3; ++policy) {
      auto o = options;
      o.out  = options.out / names[policy];
      o.tf   = options.tf / names[policy];
      std::cout << "== policy " << names[policy] << "\n";
      RunMarkedDemo(Case(policy), o);
    }
    const auto a = nn::load_npz((options.out / "none/initial_arrays.npz").string());
    for (const char* name : {"global", "coordinate"}) {
      check::Require(SameBits(a, nn::load_npz((options.out / name / "initial_arrays.npz").string())), "all3PoliciesPairedInitialWeights");
      for (const char* file : {"data.npz", "batches.tsv"})
        check::Require(ReadText(options.out / "none" / file) == ReadText(options.out / name / file), "all3PoliciesPairedSourceBatches");
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
