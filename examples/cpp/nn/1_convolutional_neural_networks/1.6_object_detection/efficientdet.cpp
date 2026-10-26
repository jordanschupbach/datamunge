// CMD: just run-nn efficientdet
// EfficientDet: repeated weighted bidirectional feature fusion.
//
// Port of tflow's efficientdet. Preparation (in ~/projects/tflow_examples):
// python3 scripts/fetch_datasets.py mnist --split both.
// Short default: three training epochs; use epochs=N for longer accuracy experiments.
// https://arxiv.org/abs/1911.09070
// https://github.com/google/automl/tree/master/efficientdet
//
// Check normalized nonnegative fusion values and input/weight derivatives,
// including negative/all-zero weights; verify BiFPN odd grids and all eight
// top-down/bottom-up nodes. Check D0-D7 compound-scaling metadata, full D0
// parameter connectivity, shared separable heads with per-level BN, and
// every compact derivative. The EfficientNet backbone omits its classifier
// and final 1280-channel classification projection.
//
// Train a compact D0-derived model: quarter-width EfficientNet-B0, two BiFPN
// cells of width32, and two-layer shared separable prediction heads, keeping
// five pyramid levels and nine anchors per cell. Use 128-pixel grayscale digit
// scenes, 5500 fit/500 validation/1000 test, three epochs Adam(.001), batch32,
// L2 .0001 and best validation AP50. For this compact demonstration use shared
// RetinaNet focal(alpha=.25,gamma=2)/Smooth-L1(beta=1/9) targets and losses,
// rather than claiming the full EfficientDet reference training recipe.
// Anchor sides start16, scales1/2^(1/3)/2^(2/3), ratios .5/1/2. No pretrained
// weights or COCO accuracy are claimed.
// Options: data=PATH size=128 epochs=3 warm_start=PREFIX lr=.001 accuracy=0|1 threads=N.
// Digit cells scale with input size (32..64 at the 128-pixel default).
// size=64 reproduces the initial smaller-input geometry. Optional
// warm_start=PREFIX restores parameters/BN state (.npz) with fresh Adam moments.
// Structural/numerical fixtures retain their original input sizes, except the
// full D0 model: tflow builds its 512-pixel graph without running it; this
// eager port runs it forward and backward at 128 pixels (parameter counts and
// gradient connectivity do not depend on the input size; the 49104 anchors of
// a 512-pixel input are counted from its pyramid grids).
//
// Default accuracy=0 verifies finite, decreasing loss; accuracy=1 enforces
// the original test AP50 target. Report measured metrics in either mode.
// Weights come from nn's generator, so trained numbers match TF statistically.
// Checkpoint (.npz) and history: output/nn/models/efficientdet_digits/.

#include <iostream>

#include "common/efficientdet.hpp"

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
void FusionChecks() {
  const float values[3][2] = {{1, -2}, {3, 5}, {7, 11}}, initial0[] = {2, -1, 3};
  std::vector<float> initial(initial0, initial0 + 3);
  std::vector<std::vector<float>> out;
  auto run = [&] {
    std::vector<nn::Tensor> inputs, weights, wrt;
    for (int i = 0; i < 3; ++i) {
      inputs.push_back(nn::from_vector(std::vector<float>{values[i][0], values[i][1]}).requires_grad_());
      weights.push_back(nn::scalar(initial[std::size_t(i)]).requires_grad_());
    }
    const auto output = bifpn::FastFeatureFusion(inputs, weights);
    wrt               = inputs;
    wrt.insert(wrt.end(), weights.begin(), weights.end());
    const auto g = nn::grad(nn::sum(output), wrt);
    out          = {output.to_vector<float>()};
    for (const auto& t : g) out.push_back(t.to_vector<float>());
  };
  run();
  const double denominator = 5.0001;
  for (int k = 0; k < 2; ++k)
    check::Near(out[0][std::size_t(k)], (2 * values[0][k] + 3 * values[2][k]) / denominator, 2e-6, "independent weighted fusion values");
  for (int i = 0; i < 3; ++i) {
    double derivative = 0;
    for (int k = 0; k < 2; ++k) {
      check::Near(out[std::size_t(i + 1)][std::size_t(k)], std::max(0.f, initial[std::size_t(i)]) / denominator, 1e-7, "fusion input derivative");
      derivative += (values[i][k] * denominator - 2 * values[0][k] - 3 * values[2][k]) / (denominator * denominator);
    }
    check::Near(out[std::size_t(i + 4)][0], initial[std::size_t(i)] > 0 ? derivative : 0, 2e-6, "normalized positive/negative weight derivative");
  }
  std::fill(initial.begin(), initial.end(), 0.f);
  run();
  for (const auto& t : out)
    for (float v : t) check::Near(v, 0, 0, "epsilon keeps all-zero fusion values and derivatives finite");
  std::cout << "  normalized fusion values/input/weight derivatives and negative/all-zero edge weights passed" << std::endl;
}

void PyramidChecks() {
  nn::ParameterStore      p(17), state(17);
  std::vector<nn::Tensor> inputs;
  const int               sides[] = {17, 9, 5, 3, 2};
  for (int side : sides) inputs.push_back(nn::ones({2, side, side, 4}));
  const auto features = bifpn::BiFpnCell(p, state, "bifpn", inputs, 4);
  std::cout << "  odd-grid BiFPN parameter/state counts=" << p.count() << "/" << state.count() << std::endl;
  check::Require(p.count() == 531 && state.count() == 64, "eight separable nodes and nineteen learned edges");
  for (int i = 0; i < 5; ++i)
    check::Require(features[std::size_t(i)].shape() == nn::Shape({2, sides[i], sides[i], 4}), "odd grids preserved by top-down/bottom-up paths");
  const int widths[] = {64, 88, 112, 160, 224, 288, 384, 384}, backbones[] = {0, 1, 2, 3, 4, 5, 6, 6};
  for (int i = 0; i < 8; ++i) {
    const auto c = efficientdet::EfficientDetScaling(i);
    check::Require(c.width == widths[i] && c.backbone == backbones[i] && c.resolution % 128 == 0, "D0-D7 compound metadata");
  }
  std::cout << "  eight BiFPN nodes/nineteen edges, odd 17/9/5/3/2 grids and D0-D7 scaling metadata passed" << std::endl;
}

efficientdet::EfficientDetConfig CompactConfig() {
  efficientdet::EfficientDetConfig c;
  c.classes             = 10;
  c.image_size          = 64;
  c.width               = 32;
  c.repeats             = 2;
  c.head_repeats        = 2;
  c.backbone_multiplier = .25;
  c.grayscale           = true;
  return c;
}
nn::Tensor Objective(const retinanet::RetinaNetOutputs& n) { return nn::mean(nn::square(n.logits)) + nn::mean(nn::square(n.deltas)); }
nn::Tensor SinImages(std::int64_t n, std::int64_t size, std::int64_t channels) {
  std::vector<float> v(std::size_t(n * size * size * channels));
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = std::sin(.13f * float(i));
  return nn::from_vector(v, {n, size, size, channels});
}
void ModelChecks() {
  {
    nn::ParameterStore               p(17), state(17);
    efficientdet::EfficientDetConfig full;
    full.image_size   = 128;
    full.update_stats = false;
    const auto net    = efficientdet::EfficientDet(p, state, SinImages(1, 128, 3), full, true);
    int        anchors512 = 0;
    for (int level = 0; level < 5; ++level) {
      const int cells = (512 + (8 << level) - 1) / (8 << level);
      anchors512 += cells * cells * 9;
    }
    const auto gradients = nn::grad(Objective(net), p.parameters());
    check::Require(anchors512 == 49104 && gradients.size() == p.size(), "full D0 anchor count and every parameter connected");
    std::cout << "  full D0/80 classes: parameters=" << p.count() << ", state=" << state.count() << ", all " << gradients.size()
              << " parameter derivatives computed (128-pixel input)" << std::endl;
    check::Require(p.count() == 3874217 && state.count() == 47136 && gradients.size() == 493, "full D0 counts as in tflow");
  }
  nn::ParameterStore p(17), state(17);
  auto               cfg = CompactConfig();
  cfg.update_stats       = false;
  const auto net         = efficientdet::EfficientDet(p, state, SinImages(2, 64, 1), cfg, true);
  const auto gradients   = nn::grad(Objective(net), p.parameters());
  check::Require(net.probabilities.shape() == nn::Shape({2, 774, 10}), "compact EfficientDet anchor/head shapes");
  const auto names   = p.names();
  double     minimum = INFINITY, maximum = 0;
  std::string smallest;
  for (std::size_t k = 0; k < gradients.size(); ++k) {
    const double norm = std::sqrt(nn::sum(nn::square(gradients[k])).item<float>());
    // The last cell's P7 output feeds only the level-4 heads, whose
    // batch-statistics BN normalizes 2 values (batch 2, 1x1 grid): the
    // gradient reaching P7 sums to zero over the batch, so its BN beta has an
    // analytically zero derivative (tflow reports 6.6e-10 of rounding noise).
    if (names[k] == "bifpn1/node7/bn/beta") {
      check::Require(std::isfinite(norm) && norm < 1e-6, "analytically zero P7 beta derivative");
      std::cout << "  " << names[k] << ": derivative norm " << norm << " (analytically zero)" << std::endl;
      continue;
    }
    check::Require(std::isfinite(norm) && norm > 0, "finite nonzero EfficientDet derivative: " + names[k]);
    if (norm < minimum) smallest = names[k];
    minimum = std::min(minimum, norm);
    maximum = std::max(maximum, norm);
  }
  std::cout << "  compact EfficientDet: parameters=" << p.count() << ", state=" << state.count() << "; every other derivative norm ["
            << minimum << ',' << maximum << "] (smallest: " << smallest << ")" << std::endl;
  check::Require(p.count() == 342038 && state.count() == 13568, "compact EfficientDet counts as in tflow");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    int                              scene_size = 128;
    std::filesystem::path            directory  = paths::Dataset("mnist");
    detector::DetectorTrainingConfig cfg;
    cfg.name              = "efficientdet_digits";
    cfg.epochs            = 3;
    cfg.accuracy          = false;
    cfg.minimum_test_ap50 = .7;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto        eq  = arg.find('=');
      if (eq == std::string::npos) throw std::invalid_argument("expected key=value option");
      const auto key = arg.substr(0, eq), value = arg.substr(eq + 1);
      if (key == "size") {
        scene_size = std::stoi(value);
        if (scene_size < 64 || scene_size % 32) throw std::invalid_argument("size must be at least64 and divisible by32");
      } else if (key == "data") directory = value;
      else if (key == "epochs") cfg.epochs = std::stoi(value);
      else if (key == "accuracy" && (value == "0" || value == "1")) cfg.accuracy = value == "1";
      else if (key == "warm_start") cfg.warm_start = value;
      else if (key == "lr") cfg.learning_rate = std::stof(value);
      else if (key == "threads") nn::set_num_threads(std::stoi(value));
      else throw std::invalid_argument("unknown option " + key);
    }
    FusionChecks();
    PyramidChecks();
    ModelChecks();
    const auto [train, test] = detector::StandardScenes(directory, scene_size);
    auto model               = CompactConfig();
    model.image_size         = scene_size;
    std::vector<detector::Box> anchors;
    detector::DetectorHooks    hooks;
    hooks.build = [&](nn::ParameterStore& p, nn::ParameterStore& state) {
      const auto net = efficientdet::EfficientDet(p, state, nn::zeros({1, scene_size, scene_size, 1}), model, false);
      anchors        = retinanet::RetinaAnchors(net.cells, float(scene_size) / 8);
    };
    hooks.loss = [&](nn::ParameterStore& p, nn::ParameterStore& state, const detector::DetectionScenes& scenes,
                     const std::vector<int>& rows) {
      const auto net = efficientdet::EfficientDet(p, state, detector::SceneImages(scenes, rows), model, true);
      return retinanet::RetinaLoss(net.logits, net.deltas, retinanet::MakeRetinaTargets(anchors, scenes, rows, 10));
    };
    hooks.detect = [&](nn::ParameterStore& p, nn::ParameterStore& state, const detector::DetectionScenes& scenes,
                       const std::vector<int>& rows) {
      auto forward = [&](const nn::Tensor& images) { return efficientdet::EfficientDet(p, state, images, model, false); };
      return retinanet::DetectRetina(forward, anchors, scenes, rows, 10);
    };
    detector::TrainDetector(cfg, train, test, hooks);
    std::cout << "All EfficientDet checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
