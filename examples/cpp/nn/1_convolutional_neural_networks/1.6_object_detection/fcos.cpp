// CMD: just run-nn fcos
// FCOS anchor-free detection: point targets, centerness and shared GN towers.
//
// Port of tflow's fcos. Preparation (in ~/projects/tflow_examples):
// python3 scripts/fetch_datasets.py mnist --split both.
// Short default: three training epochs; use epochs=N for longer accuracy experiments.
// https://arxiv.org/abs/1904.01355
// https://github.com/tianzhi0549/FCOS
//
// Check smallest-area assignment, strict box interiors, pyramid range bounds,
// optional centre sampling, centerness ratios, analytic focal/BCE/IoU loss
// values and derivatives and empty images. Construct the full ResNet-50
// P3-P7 model and all parameter derivatives; check every compact parameter
// derivative is finite and nonzero. The full model uses GN32, original
// classification tower centerness and exp(l,t,r,b) with a learned scalar at
// each level.
//
// Train a compact ResNet-18 (1/8 widths), 32-channel FPN and shared four-layer
// GN8 towers on 128-pixel grayscale scenes. Keep P3-P7 grids 16/8/4/2/1 and
// one prediction per point. Adapt ranges by .25 for small digit boxes and
// predict exp distances times stride; use radius1.5 centre sampling. These
// are explicit compact-training adaptations. Focal classification, positive
// centerness BCE and centerness-weighted smoothed -log IoU replace anchor
// matching/mining. Train three epochs, Adam(.001), batch32, L2 .0001, 5500
// fit/500 validation/1000 test scenes, retaining best validation AP50.
// No pretrained weights or full-scale dataset performance are claimed.
// Options: data=PATH size=128 epochs=3 warm_start=PREFIX lr=.001 accuracy=0|1 threads=N.
// Digit cells scale with input size (32..64 at the 128-pixel default).
// size=64 reproduces the initial smaller-input geometry. Optional
// warm_start=PREFIX restores parameters/BN state (.npz) with fresh Adam moments.
// Structural/numerical fixtures retain their original input sizes, except the
// full ResNet-50 model: tflow builds its 640-pixel graph without running it;
// this eager port runs it forward and backward at 128 pixels (parameter
// counts and gradient connectivity do not depend on the input size; the
// 8525 points of a 640-pixel input are counted from its pyramid grids).
//
// Default accuracy=0 verifies finite, decreasing loss; accuracy=1 enforces
// the original test AP50 target. Report measured metrics in either mode.
// Weights come from nn's generator, so trained numbers match TF statistically.
// Checkpoint (.npz) and history: output/nn/models/fcos_digits/.

#include <iostream>

#include "common/fcos.hpp"

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
  const auto points = fcos::FcosPoints({8, 4, 2, 1, 1});
  check::Require(points.size() == 86, "one FCOS prediction per location");
  check::Near(points[0].y, 4, 0, "half-stride points");
  check::Near(points[64].stride, 16, 0, "pyramid strides");
  check::Near(points[64].minimum, 64, 0, "P4 lower range");
  check::Near(points[64].maximum, 128, 0, "P4 upper range");
  check::Near(fcos::FcosCenterness({2, 3, 8, 12}), .25, 1e-7, "asymmetric centerness ratios");
  check::Near(fcos::FcosCenterness({4, 5, 4, 5}), 1, 0, "box centre centerness");
  check::Near(fcos::FcosCenterness({0, 2, 4, 2}), 0, 0, "boundary centerness");
  const std::vector<fcos::FcosPoint> locations = {{8, 8, 8, 0, 16}, {4, 8, 8, 0, 24}, {8, 8, 8, 16, 32}, {2, 2, 1, 0, 32}, {8, 8, 8, 17, 32}};
  detector::DetectionScenes          scenes;
  scenes.boxes  = {{{0, 0, 24, 24}, {4, 4, 12, 12}}};
  scenes.labels = {{0, 1}};
  auto t        = fcos::MakeFcosTargetArrays(locations, scenes, {0}, 2);
  // classes [1, 5, 2]: point p, class c at p * 2 + c; positive [1, 5]
  check::Require(t.classes[0 * 2 + 1] == 1 && t.classes[1 * 2 + 0] == 1 && t.classes[2 * 2 + 0] == 1,
                 "smallest area, strict interior and inclusive range boundary");
  check::Near(t.positive[3], 0, 0, "centre sampling removes distant interior point");
  check::Near(t.positive[4], 0, 0, "level range excludes all boxes");
  t = fcos::MakeFcosTargetArrays(locations, scenes, {0}, 2, 0);
  check::Near(t.positive[3], 1, 0, "original no-centre-sampling option");
  std::cout << "  point centres/strides, scale ranges, strict interiors, smallest-area overlaps, centre sampling and "
               "centerness passed"
            << std::endl;
}

void LossChecks() {
  std::vector<float> logits(4, 0.f), distances = {.8f, 1.3f, 2.1f, 2.7f, 2, 3, 4, 5}, center(2, 0.f), labels = {1, 0, 0, 0};
  std::vector<float> target = {1, 2, 3, 4, 0, 0, 0, 0}, positive = {1, 0}, centerness = {.6f, 0};
  std::vector<std::vector<float>> out;
  auto run = [&] {
    auto l = nn::from_vector(logits, {1, 2, 2}).requires_grad_();
    auto d = nn::from_vector(distances, {1, 2, 4}).requires_grad_();
    auto c = nn::from_vector(center, {1, 2}).requires_grad_();
    const fcos::FcosTargets t{nn::from_vector(labels, {1, 2, 2}), nn::from_vector(target, {1, 2, 4}),
                              nn::from_vector(positive, {1, 2}), nn::from_vector(centerness, {1, 2})};
    const auto loss = fcos::FcosLoss(l, d, c, t);
    const auto g    = nn::grad(loss, {l, d, c});
    out = {loss.to_vector<float>(), g[0].to_vector<float>(), g[1].to_vector<float>(), g[2].to_vector<float>()};
  };
  run();
  check::Near(out[0][0], 1.625 * std::log(2.) + std::log(25. / 12.6), 2e-6, "independent focal, BCE and smoothed IoU value");
  const double a = .125 + .25 * std::log(2.);
  for (int k = 0; k < 4; ++k) check::Near(out[1][std::size_t(k)], k == 0 ? -.25 * a : .75 * a, 2e-6, "analytic focal derivative");
  for (int k = 0; k < 8; ++k)
    check::Near(out[2][std::size_t(k)], k >= 4 ? 0 : k % 2 ? -2.9 / 12.6 : -4 / 12.6, 2e-6, "centerness-normalized smoothed IoU derivative");
  check::Near(out[3][0], -.1, 1e-7, "soft centerness BCE derivative");
  check::Near(out[3][1], 0, 0, "negative point no centerness learning");
  std::fill(positive.begin(), positive.end(), 0.f);
  std::fill(centerness.begin(), centerness.end(), 0.f);
  std::fill(labels.begin(), labels.end(), 0.f);
  run();
  check::Near(out[0][0], .75 * std::log(2.), 2e-6, "empty image background focal loss");
  for (int k : {2, 3})
    for (float v : out[std::size_t(k)]) check::Near(v, 0, 0, "empty image zero box/centerness derivatives");
  std::cout << "  focal/BCE/centerness-weighted IoU values and analytic derivatives, negative masking and empty images passed"
            << std::endl;
}

fcos::FcosConfig CompactConfig() {
  fcos::FcosConfig c;
  c.classes                 = 10;
  c.image_size              = 64;
  c.backbone_depth          = 18;
  c.width_divisor           = 8;
  c.pyramid_width           = 32;
  c.groups                  = 8;
  c.grayscale               = true;
  c.stride_scaled_distances = true;
  return c;
}
nn::Tensor Objective(const fcos::FcosOutputs& n) {
  return nn::mean(nn::square(n.logits)) + (nn::mean(nn::square(n.distances)) + nn::mean(nn::square(n.centerness)));
}
nn::Tensor SinImages(std::int64_t n, std::int64_t size, std::int64_t channels) {
  std::vector<float> v(std::size_t(n * size * size * channels));
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = std::sin(.13f * float(i));
  return nn::from_vector(v, {n, size, size, channels});
}
void ModelChecks() {
  {
    nn::ParameterStore p(17), state(17);
    fcos::FcosConfig   full;
    full.image_size   = 128;
    full.update_stats = false;
    const auto net    = fcos::Fcos(p, state, SinImages(1, 128, 3), full, true);
    int        points640 = 0;
    for (int level = 0; level < 5; ++level) {
      const int cells = (640 + (8 << level) - 1) / (8 << level);
      points640 += cells * cells;
    }
    check::Require(p.count() == 36426138 && state.count() == 53120 && points640 == 8525, "full FCOS parameter/state/location counts");
    const auto gradients = nn::grad(Objective(net), p.parameters());
    check::Require(gradients.size() == 218, "shared towers and five per-level scales connected");
    std::cout << "  full FCOS ResNet-50/GN32: 36426138 parameters, 8525 points at 640 pixels, all 218 parameter "
                 "derivatives computed (128-pixel input)"
              << std::endl;
  }
  nn::ParameterStore p(17), state(17);
  auto               cfg = CompactConfig();
  cfg.update_stats       = false;
  const auto net         = fcos::Fcos(p, state, SinImages(2, 64, 1), cfg, true);
  check::Require(p.count() == 314684 && state.count() == 1200 && net.points == 86, "compact FCOS counts");
  const auto gradients = nn::grad(Objective(net), p.parameters());
  check::Require(net.distances.shape() == nn::Shape({2, 86, 4}), "one four-distance vector per point");
  for (float v : net.distances.to_vector<float>()) check::Require(std::isfinite(v) && v > 0, "positive finite distances");
  const auto names   = p.names();
  double     minimum = INFINITY, maximum = 0;
  for (std::size_t k = 0; k < gradients.size(); ++k) {
    const double norm = std::sqrt(nn::sum(nn::square(gradients[k])).item<float>());
    check::Require(std::isfinite(norm) && norm > 0, "finite nonzero FCOS derivative: " + names[k]);
    minimum = std::min(minimum, norm);
    maximum = std::max(maximum, norm);
  }
  std::cout << "  compact FCOS: 314684 parameters, 86 locations, shared GN8 heads, all derivative norms [" << minimum << ','
            << maximum << "]" << std::endl;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    int                              scene_size = 128;
    std::filesystem::path            directory  = paths::Dataset("mnist");
    detector::DetectorTrainingConfig cfg;
    cfg.name              = "fcos_digits";
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
    AssignmentChecks();
    LossChecks();
    ModelChecks();
    const auto [train, test] = detector::StandardScenes(directory, scene_size);
    auto model               = CompactConfig();
    model.image_size         = scene_size;
    std::vector<fcos::FcosPoint> points;
    detector::DetectorHooks      hooks;
    hooks.build = [&](nn::ParameterStore& p, nn::ParameterStore& state) {
      const auto net = fcos::Fcos(p, state, nn::zeros({1, scene_size, scene_size, 1}), model, false);
      points         = fcos::FcosPoints(net.cells, float(scene_size) / 512);
    };
    hooks.loss = [&](nn::ParameterStore& p, nn::ParameterStore& state, const detector::DetectionScenes& scenes,
                     const std::vector<int>& rows) {
      const auto net = fcos::Fcos(p, state, detector::SceneImages(scenes, rows), model, true);
      return fcos::FcosLoss(net, fcos::MakeFcosTargets(points, scenes, rows, 10));
    };
    hooks.detect = [&](nn::ParameterStore& p, nn::ParameterStore& state, const detector::DetectionScenes& scenes,
                       const std::vector<int>& rows) { return fcos::DetectFcos(p, state, model, points, scenes, rows, 10); };
    detector::TrainDetector(cfg, train, test, hooks);
    std::cout << "All FCOS checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
