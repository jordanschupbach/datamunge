// CMD: just run-nn conv_lstm
// Genuine spatial ConvLSTM on tiny moving-dot next-frame prediction; three
// cheap epochs.
//
// Port of tflow's conv_lstm. No dataset: 48 fit / 16 validation / 16 test
// moving-dot sequences on a 4x4 torus are generated with tflow's std::mt19937
// recipe (bit-identical data.npz). 3x3 SAME input-to-state and state-to-state
// convolutions, 4x4x2 hidden and cell maps, a 1x1 pixel head (227 parameters
// in 5 tensors); positive-weighted (x15) pixel BCE, Adam .01, batch 8, three
// epochs (18 updates).
// Checks (all of tflow's): parameter counts, bit-exact .npz restore,
// label-free inference == evaluation and repeatable, single-row parity, padded
// input partials exactly zero, the first spatial state depends only on the
// time-0 3x3 neighbourhood, hidden and cell maps held on padding, padding
// garbage cannot change any output or partial, last-frame ablation cannot
// change earlier maps, zero-length fixture, weights immutable.
// Weights initialise from nn's generator (statistical parity); init=PATH
// (tflow's output/models/conv_lstm/initial_arrays.npz) replays tflow's run and
// compares every record with tflow's run directory (tf=PATH).
// Options: epochs=3 out=PATH init=PATH tf=PATH. Output: output/nn/models/conv_lstm/.
#include "common/conv_lstm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>

#include "common/marked_sequence_demo.hpp"
#include "common/moving_dot_data.hpp"

using namespace rnn_cells;

namespace {

int Argmax(const std::vector<float>& x, int b) {
  return int(std::max_element(x.begin() + 16 * b, x.begin() + 16 * (b + 1)) - x.begin() - 16 * b);
}

double SpatialAccuracy(const Tensor& logits, const Tensor& labels) {
  const auto l = logits.to_vector<float>(), y = labels.to_vector<float>();
  const int  n       = int(labels.shape()[0]);
  int        correct = 0;
  for (int b = 0; b < n; ++b)
    correct += Argmax(l, b) == Argmax(y, b);
  return double(correct) / n;
}

// Each row: four input frames, the target frame and the predicted probability (fixed [0, 1] grey
// scale).
void Panels(const fs::path& file, const MovingDots& x, const Tensor& probability) {
  const int  n  = int(std::min<std::int64_t>(16, x.size()));
  const auto in = x.inputs.to_vector<float>(), lab = x.labels.to_vector<float>(),
             p = probability.to_vector<float>();
  std::vector<float> rgb(static_cast<std::size_t>(n) * 25 * 150 * 3, 180.f);
  for (int b = 0; b < n; ++b)
    for (int panel = 0; panel < 6; ++panel)
      for (int y = 0; y < 4; ++y)
        for (int col = 0; col < 4; ++col) {
          const double v     = panel < 4    ? in[b * 64 + panel * 16 + y * 4 + col]
                               : panel == 4 ? lab[b * 16 + y * 4 + col]
                                            : p[b * 16 + y * 4 + col];
          const int    shade = int(std::floor(255 * std::clamp(v, 0., 1.) + .5));
          for (int yy = 0; yy < 5; ++yy)
            for (int xx = 0; xx < 5; ++xx)
              for (int ch = 0; ch < 3; ++ch)
                rgb[(static_cast<std::size_t>(b * 25 + y * 6 + yy) * 150 + panel * 25 + col * 6 +
                     xx) *
                        3 +
                    ch] = float(shade);
        }
  nn::save_png(nn::from_vector(rgb, {n * 25, 150, 3}), file.string(), 0, 255);
}

// The eager equivalents of tflow's session fetches, written out in this
// example: forward pass, loss and nn::grad (RecurrentModel only holds the
// parameters, the forward function and the optimizer).

/// Forward fields, loss and the full-BPTT gradients of the chosen logit with
/// respect to the inputs and every zero initial state (tflow's Evaluate fetches).
inline Outputs Evaluate(RecurrentModel& m, const Feed& x) {
  Tensor              input    = x.inputs.detach().clone();
  std::vector<Tensor> initials = m.Initials(x.inputs.shape()[0]);
  input.requires_grad_(true);
  for (auto& s : initials)
    s.requires_grad_(true);
  Outputs out               = m.forward(m.store, input, x.lengths, initials);
  out["probability"]        = m.probability(out.at("logits"));
  const auto [loss, chosen] = m.loss_fn(out, x.labels);
  std::vector<Tensor> wrt{input};
  wrt.insert(wrt.end(), initials.begin(), initials.end());
  nn::GradOptions options;
  options.retain_graph = true;
  options.allow_unused = true;
  const auto g         = nn::grad(std::vector<Tensor>{chosen}, wrt, {}, options);
  Outputs    result;
  for (const auto& f : m.fields)
    result[f] = out.at(f).detach();
  result["loss"]           = loss.detach();
  result["input_gradient"] = DefinedGradient(g[0], input);
  for (std::size_t k = 0; k < m.initial_gradient_names.size(); ++k)
    if (!m.initial_gradient_names[k].empty())
      result[m.initial_gradient_names[k]] = DefinedGradient(g[k + 1], initials[k]);
  for (const auto& probe : m.evaluate_probes) {
    const Tensor& target = wrt.at(static_cast<std::size_t>(probe.wrt));
    result[probe.name]   = DefinedGradient(
        nn::grad(std::vector<Tensor>{probe.target(out)}, {target}, {}, options)[0], target);
  }
  return result;
}

/// Loss gradient of every parameter, in store order (zeros where unused).
inline std::vector<Tensor> LossGradients(RecurrentModel& m, const Feed& x) {
  const Outputs   out    = m.forward(m.store, x.inputs, x.lengths, m.Initials(x.inputs.shape()[0]));
  const Tensor    loss   = m.loss_fn(out, x.labels).first;
  const auto      params = m.store.parameters();
  nn::GradOptions options;
  options.allow_unused = true;
  auto g               = nn::grad(std::vector<Tensor>{loss}, params, {}, options);
  for (std::size_t i = 0; i < g.size(); ++i)
    g[i] = DefinedGradient(g[i], params[i]);
  return g;
}

/// Gradients by parameter name (the initial/final_gradients.npz records).
inline nn::NamedTensors NamedGradients(const RecurrentModel& m, const std::vector<Tensor>& g) {
  nn::NamedTensors result;
  for (std::size_t i = 0; i < g.size(); ++i)
    result.emplace_back(m.store.names()[i], g[i]);
  return result;
}

} // namespace

int main(int argc, char** argv) {
  try {
    nn::set_num_threads(4);
    const auto     options = ParseDemoOptions(argc, argv, "conv_lstm");
    const auto     start   = std::chrono::steady_clock::now();
    const fs::path root    = options.out;
    fs::create_directories(root);
    std::vector<MovingDots> pools;
    nn::NamedTensors        all;
    for (int k = 0; k < 3; ++k) {
      const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
      pools.push_back(MakeMovingDots(k ? 16 : 48, 37001 + 2 * k, 10000 * k));
      const auto& x = pools.back();
      all.insert(
          all.end(),
          {{split + "_inputs", x.inputs},
           {split + "_lengths", x.lengths},
           {split + "_labels", x.labels},
           {split + "_source_ids", x.source_ids},
           {split + "_uniforms", x.uniforms}});
    }
    nn::save_npz((root / "data.npz").string(), all);
    auto model = ConvLstmModel(true);
    check::Require(
        model->store.count() == 227 && model->store.size() == 5,
        "227RealSpatialKernelAndPixelHeadParameters");
    if (!options.init.empty())
      model->LoadArrays(options.init);
    nn::save_npz((root / "initial_arrays.npz").string(), model->Arrays());
    SaveOutputs(root / "initial_outputs.npz", Evaluate(*model, pools[2].feed()));
    std::ofstream history(root / "history.tsv"), batches(root / "batches.tsv");
    history.precision(17);
    history << "epoch\tvalidation_weighted_ce\tvalidation_top1_accuracy\n";
    batches << "epoch\tstep\trow\n";
    nn::BatchOrder order(48, 8, 37019);
    int            updates = 0;
    for (int epoch = 1; epoch <= options.epochs; ++epoch) {
      order.begin_epoch();
      for (;;) {
        const auto rows = order.next();
        if (rows.empty())
          break;
        const auto batch = MovingDotRows(pools[0], rows);
        if (!updates) {
          SaveMovingDots(root / "first_batch.npz", batch);
          SaveOutputs(root / "first_forward.npz", Evaluate(*model, batch.feed()));
          nn::save_npz(
              (root / "initial_gradients.npz").string(),
              NamedGradients(*model, LossGradients(*model, batch.feed())));
        }
        // One optimizer update written out: forward, loss, the gradient of every
        // parameter, the optional gradient policy, the optimizer step and the
        // optional projection.
        {
          const Outputs out = model->forward(
              model->store, batch.inputs, batch.lengths, model->Initials(batch.inputs.shape()[0]));
          const Tensor    loss   = model->loss_fn(out, batch.labels).first;
          const auto      params = model->store.parameters();
          nn::GradOptions grad_options;
          grad_options.allow_unused = true;
          auto g                    = nn::grad(std::vector<Tensor>{loss}, params, {}, grad_options);
          for (std::size_t i = 0; i < g.size(); ++i)
            g[i] = DefinedGradient(g[i], params[i]);
          if (model->gradient_policy)
            g = model->gradient_policy(g);
          model->optimizer->step(g);
          if (model->after_update)
            model->after_update(model->store);
        }
        if (!updates)
          nn::save_npz((root / "first_after.npz").string(), model->Arrays());
        for (auto row : rows)
          batches << epoch << '\t' << updates << '\t' << row << '\n';
        ++updates;
      }
      const auto   val  = Evaluate(*model, pools[1].feed());
      const double ce   = val.at("loss").item<double>(),
                   top1 = SpatialAccuracy(val.at("logits"), pools[1].labels);
      history << epoch << '\t' << float(ce) << '\t' << top1 << '\n';
      nn::save_npz(
          (root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), model->Arrays());
      SaveOutputs(root / ("epoch" + std::to_string(epoch) + "_validation.npz"), val);
      std::cout << "epoch" << epoch << " validationWeightedCE=" << float(ce) << " top1=" << top1
                << std::endl;
    }
    batches.flush();
    nn::save_npz(
        (root / "final_gradients.npz").string(),
        NamedGradients(
            *model, LossGradients(*model, LoadMovingDots(root / "first_batch.npz").feed())));
    model->Save(root / "final.npz");
    auto deployed = ConvLstmModel(false);
    deployed->Restore(root / "final.npz");
    check::Require(
        SameBits(model->Arrays(), deployed->Arrays()), "allSpatialWeightsRestoreBitexact");
    const auto before = deployed->Arrays();
    auto&      test   = pools[2];
    const auto out = Evaluate(*deployed, test.feed()), pred = deployed->Predict(test.feed()),
               repeat = deployed->Predict(test.feed());
    for (const auto& [k, v] : pred) {
      check::Require(SameBits(v, out.at(k)), "inferenceNoTargetFeedParity");
      check::Require(SameBits(v, repeat.at(k)), "repeatSpatialInferenceBitexact");
    }
    nn::save_npz((root / "model_arrays.npz").string(), deployed->Arrays());
    SaveOutputs(root / "outputs.npz", out);
    const auto single =
        deployed->Predict(MovingDotRows(test, {0}).feed()).at("logits").to_vector<float>();
    const auto logits = out.at("logits").to_vector<float>();
    for (int p = 0; p < 16; ++p)
      check::Near(single[p], logits[p], 2e-6, "singleAndBatchSpatialParity");
    const auto lengths        = test.lengths.to_vector<std::int32_t>();
    const auto input_gradient = out.at("input_gradient").to_vector<float>(),
               first_gate     = out.at("first_gate_input_gradient").to_vector<float>();
    const auto states         = out.at("states").to_vector<float>(),
               cells          = out.at("cell_states").to_vector<float>();
    auto padded = test.inputs.to_vector<float>(), last = padded;
    for (int b = 0; b < 16; ++b) {
      const int length = lengths[b];
      for (int t = 0; t < 4; ++t)
        for (int y = 0; y < 4; ++y)
          for (int x = 0; x < 4; ++x) {
            const std::size_t i = static_cast<std::size_t>(b) * 64 + t * 16 + y * 4 + x;
            if (t >= length) {
              padded[i] = float(123 + b + t + y + x);
              check::Require(input_gradient[i] == 0, "allPaddedSpatialInputPartialsExactlyZero");
            }
            if (t > 0 || y == 3 || x == 3)
              check::Require(first_gate[i] == 0, "firstSpatialStateOnlyTimeZero3x3Neighborhood");
          }
      for (int t = length; t < 4; ++t)
        for (int j = 0; j < 32; ++j) {
          check::Require(
              states[b * 160 + (t + 1) * 32 + j] == states[b * 160 + t * 32 + j],
              "spatialHiddenRetainedOnPadding");
          check::Require(
              cells[b * 160 + (t + 1) * 32 + j] == cells[b * 160 + t * 32 + j],
              "spatialCellRetainedOnPadding");
        }
      for (int j = 0; j < 16; ++j)
        last[static_cast<std::size_t>(b) * 64 + (length - 1) * 16 + j] = 0;
    }
    MovingDots pad_data =
                   MovingDotRows(test, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}),
               last_data = pad_data;
    pad_data.inputs      = nn::from_vector(padded, {16, 4, 4, 4, 1});
    last_data.inputs     = nn::from_vector(last, {16, 4, 4, 4, 1});
    const auto padout    = Evaluate(*deployed, pad_data.feed()),
               lastout   = Evaluate(*deployed, last_data.feed());
    for (const auto& [k, v] : out)
      check::Require(
          SameBits(v, padout.at(k)), "allOutputsAndSpatialPartialsInvariantToPaddingGarbage:" + k);
    const auto last_states = lastout.at("states").to_vector<float>();
    for (int b = 0; b < 16; ++b)
      for (int t = 0; t < lengths[b]; ++t)
        for (int j = 0; j < 32; ++j)
          check::Require(
              last_states[b * 160 + t * 32 + j] == states[b * 160 + t * 32 + j],
              "lastFrameAblationCannotChangeEarlierMaps");
    SaveMovingDots(root / "padding_inputs.npz", pad_data);
    SaveMovingDots(root / "last_frame_inputs.npz", last_data);
    SaveOutputs(root / "padding_outputs.npz", padout);
    SaveOutputs(root / "last_frame_outputs.npz", lastout);
    auto empty          = MovingDotRows(test, {0, 1});
    empty.lengths       = nn::zeros({2}, nn::DType::Int32);
    const auto emptyout = Evaluate(*deployed, empty.feed());
    for (const char* name : {"states", "cell_states", "input_gradient", "initial_cell_gradient"})
      for (float v : emptyout.at(name).to_vector<float>())
        check::Require(v == 0, "zeroLengthSpatialFixture");
    SaveMovingDots(root / "zero_length_inputs.npz", empty);
    SaveOutputs(root / "zero_length_outputs.npz", emptyout);
    Panels(root / "predictions.png", test, out.at("probability"));
    const auto probability = out.at("probability").to_vector<float>(),
               labels      = test.labels.to_vector<float>();
    double brier           = 0;
    for (int i = 0; i < 256; ++i)
      brier += std::pow(probability[i] - labels[i], 2) / 256.;
    const double test_ce = out.at("loss").item<double>(),
                 top1    = SpatialAccuracy(out.at("logits"), test.labels),
                 ablated = SpatialAccuracy(lastout.at("logits"), test.labels);
    std::ofstream metrics(root / "metrics.json");
    metrics.precision(17);
    metrics << "{\"parameters\":227,\"parameter_tensors\":5,\"epochs\":" << options.epochs
            << ",\"optimizer_updates\":" << updates << ",\"test_weighted_ce\":" << float(test_ce)
            << ",\"test_top1_accuracy\":" << top1
            << ",\"last_frame_ablated_top1_accuracy\":" << ablated << ",\"test_brier\":" << brier
            << ",\"uniform_top1_chance\":0.0625,\"uniform_probability_brier\":0.05859375,\"zero_"
               "frame_brier\":0.0625,"
               "\"motion_oracle_top1\":1,\"persistence_top1\":0}\n";
    std::ofstream predictions(root / "predictions.tsv");
    predictions << "source_id\tlength\ttarget_pixel\tpredicted_pixel\n";
    nn::ConfusionMatrix confusion(16);
    const auto          predicted_pixels = nn::argmax(nn::reshape(out.at("logits"), {16, 16}), 1);
    const auto          target_pixels    = nn::argmax(nn::reshape(test.labels, {16, 16}), 1);
    confusion.add(target_pixels, predicted_pixels);
    std::cout << "confusion (target pixel rows, predicted pixel columns):\n";
    for (const auto& row : confusion.cells()) {
      for (auto count : row)
        std::cout << count << ' ';
      std::cout << '\n';
    }
    const auto ids = test.source_ids.to_vector<std::int32_t>();
    for (int b = 0; b < 16; ++b)
      predictions << ids[b] << '\t' << lengths[b] << '\t' << Argmax(labels, b) << '\t'
                  << Argmax(logits, b) << '\n';
    std::ofstream html(root / "index.html");
    html << "<!doctype html><meta charset='utf-8'><title>Spatial "
            "ConvLSTM</title><style>body{font:16px system-ui;margin:30px;"
            "max-width:1000px}img{image-rendering:pixelated;width:600px;max-width:100%}</"
            "style><h1>ConvLSTM moving-dot next-frame "
            "prediction</h1><p>3x3 input-to-state and state-to-state SAME zero-padded "
            "convolutions; hidden and cell maps 4x4x2 stay "
            "spatial at every step; input 4x4x1. Four input steps, valid length 2..4, the mask "
            "retains both states. "
            "Input/forget/candidate/output gates, forget bias 1, no peepholes, final 1x1 pixel "
            "head; 227 parameters in 5 tensors.</p>"
            "<p>48 fit / 16 validation / 16 test synthetic sequences: one dot moves with a "
            "constant cardinal velocity on a 4x4 torus. "
            "Training uses zero-padded kernels although motion wraps at the boundaries, an "
            "explicit modelling mismatch. Three epochs, "
            "18 Adam .01 updates, positive-pixel BCE weight 15 to balance the 15 negative pixels. "
            "Top-1 measures the dot location.</p>"
            "<h2>Held-out sequences, target, predicted probability</h2><p>Each row: four input "
            "frames (black padding), the next-frame "
            "target and the probability heatmap on a fixed [0,1] grey scale.</p><img "
            "src='predictions.png'><p>Test weighted CE "
         << float(test_ce) << ", top-1 " << top1 << ", Brier " << brier
         << ", last-frame-ablated top-1 " << ablated
         << ". See metrics.json and predictions.tsv.</p>";
    check::Require(
        SameBits(before, deployed->Arrays()), "spatialInferenceControlsLeaveWeightsImmutable");
    check::Require(
        bool(metrics) && bool(html) && bool(history) && bool(batches) && bool(predictions),
        "completeSpatialRecords");
    metrics.close(), html.close(), history.close(), batches.close(), predictions.close();
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "testWeightedCE=" << float(test_ce) << " top1=" << top1 << " brier=" << brier
              << ";spatialKernels/maps/padding/causality/restore/labelFree PASS (" << seconds
              << " s)\nArtifacts:" << (root / "index.html") << std::endl;
    CompareWithTflow(
        root,
        options,
        {"initial_outputs.npz",
         "first_forward.npz",
         "initial_gradients.npz",
         "first_after.npz",
         "epoch1_arrays.npz",
         "epoch" + std::to_string(options.epochs) + "_arrays.npz",
         "final_gradients.npz",
         "outputs.npz",
         "last_frame_outputs.npz"},
        "test_weighted_ce");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
