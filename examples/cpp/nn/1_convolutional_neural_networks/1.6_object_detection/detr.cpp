// CMD: just run-nn detr
// DETR set prediction: object queries, Hungarian matching and transformer.
//
// Port of tflow's detr. Preparation (in ~/projects/tflow_examples):
// python3 scripts/fetch_datasets.py mnist --split both.
// https://arxiv.org/abs/2005.12872
// https://github.com/facebookresearch/detr
//
// Compare rectangular Hungarian assignments against exhaustive search on 100
// negative-cost matrices. Check invalid/empty assignments, normalized 2D sine
// positions, detached class/L1/GIoU matching, empty targets, weighted no-object
// CE and independent finite-difference GIoU/L1 derivatives. Construct the full
// ResNet-50 + six/six transformer (dim256,FF2048,eight heads,100 queries) and
// all parameter derivatives. Check compact forward/backward numerically.
//
// Train a compact ResNet-18/8 with small stem (stride8 features), dim64,FF128,
// four heads,two encoder/two decoder layers and ten learned queries. No dropout
// or pretrained weights. Shared heads supervise both decoder layers; re-match
// each layer independently. Loss: weighted CE (no-object .1),5*L1,2*GIoU.
// Train fifty epochs Adam(.001),batch32,L2 .0001 on 5500 fit/500 validation/
// 1000 test grayscale digit scenes, restore best validation AP50. Backbone BN
// is trained rather than the frozen pretrained BN of the reference. Inference
// uses one best foreground class per query, with no anchors or box NMS.
// Options: data=PATH epochs=50 warm_start=PREFIX lr=.001 threads=N.
//
// Eager port: each step runs the training forward pass once and matches each
// decoder layer on its detached outputs (tflow runs a separate forward pass
// with the same weights to feed the matcher). The full ResNet-50 model, which
// tflow builds as a 640-pixel graph without running it, runs forward and
// backward here at 128 pixels (parameter counts and gradient connectivity do
// not depend on the input size). Weights come from nn's generator, so trained
// numbers match TF statistically.
// Checkpoint (.npz) and history: output/nn/models/detr_digits/.

#include <functional>
#include <iostream>

#include "common/detr.hpp"

// Experiment workflow from examples/nn/common/detection.hpp.
namespace detector {
using namespace ::detector;
inline ApResult TrainDetector(const DetectorTrainingConfig& requested_cfg, const DetectionScenes& train,
                              const DetectionScenes& test, const DetectorHooks& hooks) {
  auto cfg = requested_cfg;
  cfg.epochs = example_runtime::Limit(cfg.epochs, 1);
  cfg.batch = example_runtime::Limit(cfg.batch, 2);
  cfg.validation = example_runtime::Limit(cfg.validation, 8);
  if (!example_runtime::Full()) cfg.fit = cfg.fit ? std::min(cfg.fit, 8) : 8;
  example_runtime::Describe();
  check::Require(cfg.validation > 0 && cfg.validation < train.size() && cfg.epochs > 0 && cfg.batch > 0,
                 "valid detector training configuration");
  check::Require(!example_runtime::Full() || cfg.accuracy || cfg.epochs >= 2, "short learning demonstration needs at least two epochs");
  const int fit_count =
      cfg.fit > 0 ? std::min(cfg.fit, train.size() - cfg.validation) : train.size() - cfg.validation;
  std::vector<int> fit(static_cast<std::size_t>(fit_count)), validation(static_cast<std::size_t>(cfg.validation)),
      test_rows(example_runtime::Limit<std::size_t>(test.size(), 8));
  std::iota(fit.begin(), fit.end(), 0);
  std::iota(validation.begin(), validation.end(), train.size() - cfg.validation);
  std::iota(test_rows.begin(), test_rows.end(), 0);
  nn::ParameterStore params(17), state(17);
  {
    nn::NoGradGuard guard;
    hooks.build(params, state);
  }
  std::vector<nn::Tensor> kernels;
  for (const auto& p : params.parameters())
    if (p.dim() >= 2) kernels.push_back(p);
  const auto directory = TrainingDirectory(cfg.name);
  std::filesystem::create_directories(directory);
  const auto prefix = (directory / "best").string();
  if (!cfg.warm_start.empty()) {
    check::Require(std::filesystem::absolute(cfg.warm_start).lexically_normal() !=
                       std::filesystem::absolute(prefix).lexically_normal(),
                   "warm start requires a separate output directory/run tag");
    RestoreCheckpoint(params, state, cfg.warm_start);
    std::cout << "  warm start: " << cfg.warm_start << "; fresh Adam moments" << std::endl;
  }
  nn::OptimizerConfig optimizer_config;
  optimizer_config.kind = nn::OptimizerKind::Adam;
  nn::Optimizer optimizer(params, cfg.learning_rate, optimizer_config);
  std::ofstream history(directory / "history.tsv");
  history << "epoch\ttraining_loss\tvalidation_ap50\tvalidation_coco_ap\n";
  std::cout << "== " << cfg.name << ": " << fit_count << " fit / " << cfg.validation << " validation / " << test_rows.size()
            << " test scenes (" << train.height << "x" << train.width << ")\n  parameters=" << params.count()
            << "; batch=" << cfg.batch << "; lr=" << cfg.learning_rate << std::endl;
  nn::BatchOrder      batches(std::size_t(fit_count), std::size_t(cfg.batch), 29);
  double              best = -1;
  int                 best_epoch = -1;
  std::vector<double> epoch_losses;
  if (!cfg.warm_start.empty()) {
    const auto baseline = EvaluateDetections(hooks.detect(params, state, train, validation), train, validation, cfg.classes);
    best       = baseline.ap50;
    best_epoch = 0;
    SaveCheckpoint(params, state, prefix);
    history << 0 << "\tnan\t" << baseline.ap50 << '\t' << baseline.coco << '\n';
    history.flush();
    std::cout << "  restored baseline validation " << ApSummary(baseline) << std::endl;
  }
  for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
    batches.begin_epoch();
    double      total = 0;
    std::size_t seen  = 0;
    for (;;) {
      const auto picked = batches.next();
      if (picked.empty()) break;
      std::vector<int> rows;
      for (auto k : picked) rows.push_back(fit[std::size_t(k)]);
      nn::Tensor objective = hooks.loss(params, state, train, rows);
      const double batch_loss = objective.item();
      check::Require(std::isfinite(batch_loss), "finite detector batch loss");
      if (cfg.l2 > 0 && !kernels.empty()) {
        nn::Tensor penalty = nn::sum(nn::square(kernels[0]));
        for (std::size_t i = 1; i < kernels.size(); ++i) penalty = penalty + nn::sum(nn::square(kernels[i]));
        objective = objective + 0.5 * cfg.l2 * penalty;
      }
      optimizer.minimize(objective);
      total += batch_loss * double(rows.size());
      seen += rows.size();
    }
    epoch_losses.push_back(total / double(seen));
    check::Require(std::isfinite(epoch_losses.back()), "finite detector epoch mean");
    const auto result = EvaluateDetections(hooks.detect(params, state, train, validation), train, validation, cfg.classes);
    check::Require(std::isfinite(result.ap50), "finite validation AP50");
    if (result.ap50 > best) {
      best       = result.ap50;
      best_epoch = epoch;
      SaveCheckpoint(params, state, prefix);
    }
    history << epoch << '\t' << total / double(seen) << '\t' << result.ap50 << '\t' << result.coco << '\n';
    history.flush();
    std::cout << "  epoch " << epoch << ": training loss=" << total / double(seen) << ", validation " << ApSummary(result)
              << std::endl;
  }
  RestoreCheckpoint(params, state, prefix);
  const auto restored = EvaluateDetections(hooks.detect(params, state, train, validation), train, validation, cfg.classes);
  check::Near(restored.ap50, best, 1e-6, "restored checkpoint reproduces selected validation AP50");
  std::cout << "  restored validation AP50=" << restored.ap50 << "; checkpoint agreement verified" << std::endl;
  const auto result = EvaluateDetections(hooks.detect(params, state, test, test_rows), test, test_rows, cfg.classes);
  std::cout << "  selected epoch=" << best_epoch << "; test " << ApSummary(result) << "\n  per-class test AP50:";
  for (double ap : result.per_class50) std::cout << ' ' << std::fixed << std::setprecision(3) << ap;
  std::cout << std::defaultfloat << std::setprecision(6) << "\n  best checkpoint and history: " << directory.string()
            << std::endl;
  if (hooks.inspect) hooks.inspect(params, state, test, test_rows, directory);
  if (example_runtime::Full()) CheckDetectorDemonstration(cfg, epoch_losses, result);
  else check::Require(std::isfinite(result.ap50), "finite quick detector evaluation");
  if (example_runtime::Full() && !cfg.accuracy)
    std::cout << "  short learning demonstration passed: mean loss " << epoch_losses.front() << " -> "
              << epoch_losses.back() << "; accuracy benchmark not requested (accuracy=1 enables AP50 gate "
              << cfg.minimum_test_ap50 << ")" << std::endl;
  return result;
}
}  // namespace detector



namespace nn = datamunge::nn;

namespace {
void AssignmentChecks() {
  std::mt19937                           rng(19);
  std::uniform_real_distribution<double> random(-5, 5);
  for (int trial = 0; trial < 100; ++trial) {
    const int                        rows = 1 + trial % 4, cols = rows + trial % 3;
    std::vector<std::vector<double>> cost(static_cast<std::size_t>(rows), std::vector<double>(static_cast<std::size_t>(cols)));
    for (auto& row : cost)
      for (double& c : row) c = random(rng);
    const auto        selected = hungarian::HungarianAssignment(cost);
    double            actual   = 0;
    std::vector<bool> used(static_cast<std::size_t>(cols));
    for (int r = 0; r < rows; ++r) {
      const int s = selected[std::size_t(r)];
      check::Require(s >= 0 && s < cols && !used[std::size_t(s)], "unique in-range assignment");
      used[std::size_t(s)] = true;
      actual += cost[std::size_t(r)][std::size_t(s)];
    }
    std::fill(used.begin(), used.end(), false);
    double                           best = INFINITY;
    std::function<void(int, double)> enumerate = [&](int row, double total) {
      if (row == rows) {
        best = std::min(best, total);
        return;
      }
      for (int c = 0; c < cols; ++c)
        if (!used[std::size_t(c)]) {
          used[std::size_t(c)] = true;
          enumerate(row + 1, total + cost[std::size_t(row)][std::size_t(c)]);
          used[std::size_t(c)] = false;
        }
    };
    enumerate(0, 0);
    check::Near(actual, best, 1e-10, "Hungarian equals exhaustive rectangular assignment");
  }
  check::Require(hungarian::HungarianAssignment({}).empty(), "empty targets need no assignment");
  for (const auto& cost : std::vector<std::vector<std::vector<double>>>{{{1}, {2}}, {{1, 2}, {3}}, {{NAN}}}) {
    bool rejected = false;
    try {
      hungarian::HungarianAssignment(cost);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    check::Require(rejected, "invalid assignment matrices rejected");
  }
  const auto   position = detr::DetrPositions(2, 3, 8).to_vector<float>();
  const double tau      = 2 * std::acos(-1.);
  check::Near(position[0], std::sin(tau / (2 + 1e-6)), 1e-7, "normalized row sine");
  check::Near(position[4], std::sin(tau / (3 + 1e-6)), 1e-7, "normalized column sine");
  for (int k = 0; k < 4; ++k) check::Near(position[std::size_t(k)], position[std::size_t(8 + k)], 0, "row positions independent of columns");
  std::cout << "  100 Hungarian/exhaustive rectangular assignments, invalid/empty inputs and 2D normalized sine positions passed"
            << std::endl;
}

void LossChecks() {
  std::vector<float> raw(6, 0.f), pred = {.25f, .3f, .2f, .2f, .2f, .3f, .4f, .5f};
  detr::DetrTargets  t;
  t.batch    = 1;
  t.queries  = 2;
  t.classes  = 2;
  t.labels   = {1, 0, 0, 0, 0, 1};
  t.boxes    = {.7f, .65f, .25f, .3f, 0, 0, 0, 0};
  t.positive = {1, 0};
  std::vector<std::vector<float>> out;
  auto run = [&] {
    auto       l    = nn::from_vector(raw, {1, 2, 3}).requires_grad_();
    auto       b    = nn::from_vector(pred, {1, 2, 4}).requires_grad_();
    const auto loss = detr::DetrLayerLoss(l, b, t);
    const auto g    = nn::grad(loss, {l, b});
    out             = {loss.to_vector<float>(), g[0].to_vector<float>(), g[1].to_vector<float>()};
  };
  auto box_loss = [&](const float* p) {
    double l1 = 0;
    for (int k = 0; k < 4; ++k) l1 += std::abs(p[k] - t.boxes[std::size_t(k)]);
    return 5 * l1 + 2 * (1 - detr::GeneralizedIou(detr::DetrCorners(p), detr::DetrCorners(t.boxes.data())));
  };
  run();
  check::Near(out[0][0], std::log(3.) + box_loss(pred.data()), 2e-6, "weighted CE plus L1 and disjoint GIoU");
  for (int q = 0; q < 2; ++q)
    for (int c = 0; c < 3; ++c)
      check::Near(out[1][std::size_t(q * 3 + c)], (1. / 3 - (c == (q ? 2 : 0))) * (q ? .1 : 1) / 1.1, 1e-7, "weighted no-object CE derivative");
  double error = 0;
  for (int k = 0; k < 4; ++k) {
    const float original = pred[std::size_t(k)], step = 1e-3f;
    pred[std::size_t(k)] = original + step;
    const double plus    = box_loss(pred.data());
    pred[std::size_t(k)] = original - step;
    const double minus   = box_loss(pred.data());
    pred[std::size_t(k)] = original;
    error = std::max(error, std::abs(out[2][std::size_t(k)] - (plus - minus) / (2 * step)));
  }
  check::Near(error, 0, 3e-4, "independent finite differences for disjoint GIoU plus L1");
  for (int k = 4; k < 8; ++k) check::Near(out[2][std::size_t(k)], 0, 0, "unmatched queries have no box loss");
  t.positive = {0, 0};
  t.labels   = {0, 0, 1, 0, 0, 1};
  run();
  check::Near(out[0][0], std::log(3.), 1e-6, "empty scenes train no-object class");
  for (int k = 0; k < 8; ++k) check::Near(out[2][std::size_t(k)], 0, 0, "empty scene box gradients");
  detector::DetectionScenes scene;
  scene.height = 64;
  scene.width  = 64;
  scene.boxes  = {{{8, 8, 24, 24}, {32, 32, 56, 56}}};
  scene.labels = {{0, 1}};
  const std::vector<float> match_logits = {0, 7, 0, 7, 0, 0, 0, 0, 7};
  const std::vector<float> match_boxes  = {.6875f, .6875f, .375f, .375f, .25f, .25f, .25f, .25f, .5f, .5f, .2f, .2f};
  const auto               m            = detr::MatchDetrTargets(match_logits, match_boxes, 3, scene, {0}, 2);
  check::Require(m.labels[0 * 3 + 1] == 1 && m.labels[1 * 3 + 0] == 1 && m.labels[2 * 3 + 2] == 1,
                 "matching uses box and class costs with no-object for unmatched queries");
  std::cout << "  class/no-object weighting, detached one-to-one targets, empty images and disjoint GIoU/L1 derivative error="
            << error << std::endl;
}

detr::DetrConfig CompactConfig() {
  detr::DetrConfig c;
  c.classes        = 10;
  c.image_size     = 64;
  c.backbone_depth = 18;
  c.width_divisor  = 8;
  c.dimension      = 64;
  c.heads          = 4;
  c.feedforward    = 128;
  c.encoder_layers = 2;
  c.decoder_layers = 2;
  c.queries        = 10;
  c.dropout        = 0;
  c.grayscale      = true;
  c.small_stem     = true;
  return c;
}
nn::Tensor Objective(const detr::DetrOutputs& n) {
  nn::Tensor total;
  for (std::size_t k = 0; k < n.logits.size(); ++k) {
    const auto term = nn::mean(nn::square(n.logits[k])) + nn::mean(nn::square(n.boxes[k]));
    total           = k ? total + term : term;
  }
  return total;
}
nn::Tensor SinImages(std::int64_t n, std::int64_t size, std::int64_t channels) {
  std::vector<float> v(std::size_t(n * size * size * channels));
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = std::sin(.13f * float(i));
  return nn::from_vector(v, {n, size, size, channels});
}
void ModelChecks() {
  {
    nn::ParameterStore p(17), state(17);
    detr::DetrConfig   full;
    full.image_size   = 128;
    full.update_stats = false;
    const auto net    = detr::Detr(p, state, SinImages(1, 128, 3), full, true);
    const auto gradients = nn::grad(Objective(net), p.parameters());
    check::Require(net.logits.size() == 6 && gradients.size() == p.size(), "full DETR auxiliary heads and parameter connections");
    std::cout << "  full DETR: parameters=" << p.count() << ", all " << gradients.size()
              << " parameter derivatives computed (128-pixel input)" << std::endl;
  }
  nn::ParameterStore p(17), state(17);
  auto               cfg = CompactConfig();
  cfg.update_stats       = false;
  const auto net         = detr::Detr(p, state, SinImages(2, 64, 1), cfg, true);
  const auto gradients   = nn::grad(Objective(net), p.parameters(), {false, false, true});
  check::Require(net.logits.back().shape() == nn::Shape({2, 10, 11}) && net.boxes.back().shape() == nn::Shape({2, 10, 4}) &&
                     net.memory.shape()[1] == 64,
                 "queries/classes/boxes and small-stem 8x8 memory shapes");
  for (float v : net.boxes.back().to_vector<float>()) check::Require(v > 0 && v < 1, "normalized positive predicted boxes");
  const auto names   = p.names();
  int        nonzero = 0, zero = 0;
  for (std::size_t k = 0; k < gradients.size(); ++k) {
    const double norm = gradients[k].defined() ? std::sqrt(nn::sum(nn::square(gradients[k])).item<float>()) : 0.0;
    check::Require(std::isfinite(norm), "finite DETR derivative: " + names[k]);
    if (norm > 0) ++nonzero;
    else {
      ++zero;
      const auto& name     = names[k];
      const bool  key_bias = name.size() >= 4 && name.substr(name.size() - 4) == "/k/b";
      check::Require(name.find("decoder0/self/") != std::string::npos || name == "decoder0/norm1/gamma" || key_bias,
                     "unexpected zero initial derivative: " + name);
    }
  }
  check::Require(nonzero > 100, "backbone, encoder, decoder and prediction heads receive gradients");
  std::cout << "  compact DETR parameters=" << p.count() << "; finite derivatives: " << nonzero << " nonzero, " << zero
            << " zero from zero-content self-attention/normalization or invariant key biases" << std::endl;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path            directory = paths::Dataset("mnist");
    detector::DetectorTrainingConfig cfg;
    cfg.name              = "detr_digits";
    cfg.epochs            = 50;
    cfg.minimum_test_ap50 = .7;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto        eq  = arg.find('=');
      if (eq == std::string::npos) throw std::invalid_argument("expected key=value option");
      const auto key = arg.substr(0, eq), value = arg.substr(eq + 1);
      if (key == "data") directory = value;
      else if (key == "epochs") cfg.epochs = std::stoi(value);
      else if (key == "warm_start") cfg.warm_start = value;
      else if (key == "lr") cfg.learning_rate = std::stof(value);
      else if (key == "threads") nn::set_num_threads(std::stoi(value));
      else throw std::invalid_argument("unknown option " + key);
    }
    AssignmentChecks();
    LossChecks();
    ModelChecks();
    const auto [train, test] = detector::StandardScenes(directory);
    const auto              model = CompactConfig();
    detector::DetectorHooks hooks;
    hooks.build = [&](nn::ParameterStore& p, nn::ParameterStore& state) { detr::Detr(p, state, nn::zeros({1, 64, 64, 1}), model, false); };
    hooks.loss  = [&](nn::ParameterStore& p, nn::ParameterStore& state, const detector::DetectionScenes& scenes,
                     const std::vector<int>& rows) {
      return detr::DetrLoss(detr::Detr(p, state, detector::SceneImages(scenes, rows), model, true), scenes, rows, 10);
    };
    hooks.detect = [&](nn::ParameterStore& p, nn::ParameterStore& state, const detector::DetectionScenes& scenes,
                       const std::vector<int>& rows) { return detr::DetectDetr(p, state, model, scenes, rows, 10); };
    detector::TrainDetector(cfg, train, test, hooks);
    std::cout << "All DETR checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
