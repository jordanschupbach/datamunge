// CMD: just run-nn yolo_v1
// YOLO v1: one grid, two responsible box predictors and conditional class scores.
//
// Port of tflow's yolo_v1. Preparation (in ~/projects/tflow_examples):
// python3 scripts/fetch_datasets.py mnist --split both.
// Short default: three training epochs; use epochs=N for longer accuracy experiments.
//
// Check the complete squared-error loss and analytic derivatives on exact,
// imperfect, swapped-responsibility and empty-cell fixtures. Confidence
// learns the current IoU, whose target is detached from differentiation.
// Check collision handling in grid assignment. Construct the full paper
// architecture (24 convolutions, dense 4096, linear output 7x7x30) and
// verify 271,703,550 learned parameters. The compact model is checked
// numerically with every parameter derivative.
// https://arxiv.org/abs/1506.02640
//
// Train all 24 convolutions on grayscale 56-pixel digit scenes with 1/8
// widths and a 256-unit dense head. Disable three downsamplings to retain a
// 7x7 feature map. The compact default follows the original no-BN blocks;
// batch_norm=1 selects the optional BN experiment. Compact dropout defaults
// to zero; dropout=.5 selects the paper's rate. Use 5500 fit / 500
// validation / 1000 test scenes, three epochs of Adam(.0005), batch 32 and
// kernel L2 .0001. The best validation AP50 checkpoint is restored before
// separate test evaluation. Report the number of scene objects discarded
// because another object's centre shares its cell. No ImageNet pretraining.
// Options: data=PATH epochs=3 batch_norm=0|1 (default 0) dropout=0..1 (default 0),
// warm_start=PREFIX lr=.0005 dense=256 accuracy=0|1 threads=N. checks_only=1
// runs fixtures without training. diagnose=PREFIX compares train/inference
// paths on fit/validation scenes without updates, checkpoint writes or test
// evaluation; expect_ap50=VALUE then verifies the restored validation AP50.
// augment=1 adds training-only random uniform scale [.8,1.2], translation
// within +/-20% of image size, and exposure [1/1.5,1.5] (common/
// detection_affine.hpp; grayscale, no flips). pretrain=N trains the first 20
// convolutions on fitting-scene digit crops with a pooled classifier before
// detector fine-tuning; pretrain_only=1 saves the bootstrap only.
// Default accuracy=0 verifies finite, decreasing loss; accuracy=1 enforces
// the original test AP50 target (.5).
//
// Eager port (common/yolo_v1.hpp). Differences from tflow: the full paper
// model's 24 convolutions (60.0M parameters) are allocated and
// differentiated at 448 pixels, but its dense head (fc1 50176x4096 and the
// 4096x1470 output, 211.5M parameters, 846 MB) is counted from its shapes and
// not allocated, on this shared 16 GB machine (tflow builds the whole model as
// a graph only and also never allocates the head). Checkpoints and
// bootstraps are .npz parameter/state pairs. The augmentation fixture that
// rejects scenes carrying instance masks has no counterpart (detector scenes
// have no masks). Weights come from nn's generator, so trained numbers match
// TF statistically. Output: output/nn/models/yolo_v1_digits/.

#include <chrono>
#include <iostream>

#include "common/yolo_v1.hpp"

// Experiment workflow from examples/nn/common/yolo_v1.hpp.
namespace yolo {
using namespace ::yolo;
inline std::string PretrainYoloV1(const DetectionScenes& scenes, const YoloV1Config& detector_cfg, int epochs) {
  check::Require(epochs > 0 && scenes.size() > 500 && detector_cfg.compact && detector_cfg.width_divisor == 8 && detector_cfg.classes == 10,
                 "pretraining epochs, scene split and compact configuration");
  epochs = example_runtime::Limit(epochs, 1);
  const auto fit        = YoloV1PretrainingCrops(scenes, 0, example_runtime::Limit(scenes.size() - 500, 16));
  const auto validation = YoloV1PretrainingCrops(scenes, scenes.size() - 500,
                                               example_runtime::Full() ? scenes.size() : scenes.size() - 484);
  check::Require(fit.size() > 0 && validation.size() > 0, "nonempty pretraining fit/validation crops");
  auto cfg             = detector_cfg;
  cfg.keep_probability = 1;
  YoloV1Classifier model(cfg);
  check::Require(model.parameters.count() == (cfg.batch_norm ? 353714 : 352578), "first-20 classifier parameter count");
  const auto directory = detector::TrainingDirectory("yolo_v1_digits_pretrain");
  std::filesystem::create_directories(directory);
  const auto    best_prefix = (directory / "best_classifier").string();
  std::ofstream history(directory / "history.tsv");
  check::Require(bool(history), "open pretraining history");
  history << "epoch\tvalidation_loss\tvalidation_accuracy\n";
  std::cout << "== YOLO v1 first-20 classifier pretraining: " << fit.size() << " fit / " << validation.size()
            << " validation crops; no test data; parameters=" << model.parameters.count() << std::endl;
  nn::BatchOrder                          batches(std::size_t(fit.size()), example_runtime::Limit(64, 8), 29);
  std::mt19937_64                         random(73);
  std::uniform_real_distribution<double> scale(.8, 1.2), shift(-.2, .2), exposure(-std::log(1.5), std::log(1.5));
  double                                  best     = INFINITY;
  int                                     selected = -1;
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    batches.begin_epoch();
    std::size_t seen = 0;
    for (;;) {
      const auto picked = batches.next();
      if (picked.empty()) break;
      const std::vector<int>                    rows(picked.begin(), picked.end());
      std::vector<detector::DetectionTransform> transforms(rows.size());
      for (auto& t : transforms) {
        t.scale    = scale(random);
        t.shift_y  = 56 * shift(random);
        t.shift_x  = 56 * shift(random);
        t.exposure = std::exp(exposure(random));
      }
      const auto       augmented = detector::TransformDetectionScenes(fit, rows, transforms);
      std::vector<int> local(rows.size());
      std::iota(local.begin(), local.end(), 0);
      const auto [x, y] = YoloV1PretrainingBatch(augmented, local);
      model.Train(x, y);
      seen += rows.size();
    }
    check::Require(seen == std::size_t(fit.size()), "each fitting crop used once per pretraining epoch");
    const auto [loss, accuracy] = EvaluateYoloV1Pretraining(model, validation);
    if (loss < best) {
      best     = loss;
      selected = epoch;
      detector::SaveCheckpoint(model.parameters, model.state, best_prefix);
    }
    history << epoch << '\t' << loss << '\t' << accuracy << '\n';
    history.flush();
    std::cout << "  pretrain epoch " << epoch << ": validation NLL=" << loss << ", accuracy=" << accuracy << std::endl;
  }
  detector::RestoreCheckpoint(model.parameters, model.state, best_prefix);
  const auto [loss, accuracy] = EvaluateYoloV1Pretraining(model, validation);
  check::Near(loss, best, 1e-6, "restored pretraining validation NLL");
  std::cout << "  selected pretrain epoch=" << selected << "; validation NLL=" << loss << "; accuracy=" << accuracy << std::endl;
  check::TrainingQuality(accuracy >= .9, "pretraining validation accuracy at least .9");
  const auto bootstrap = directory / "detector_initial";
  WriteYoloV1Bootstrap(CaptureYoloV1Pretrained(model), detector_cfg, bootstrap);
  std::cout << "  verified pretrained prefix transfer; detector bootstrap: " << bootstrap.string() << std::endl;
  return bootstrap.string();
}
}  // namespace yolo



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
using detector::Box;
using detector::DetectionScenes;

namespace {
nn::Tensor SinImages(std::int64_t n, std::int64_t height, std::int64_t width, std::int64_t channels) {
  std::vector<float> v(std::size_t(n * height * width * channels));
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = std::sin(.13f * float(i));
  return nn::from_vector(v, {n, height, width, channels});
}

void PretrainingChecks() {
  DetectionScenes source;
  source.height = source.width = 4;
  source.boxes                 = {{{1, 1, 3, 3}}, {{1, 1, 3, 3}}};
  source.labels                = {{3}, {7}};
  source.pixels.resize(32, 128);
  std::fill(source.pixels.begin() + 16, source.pixels.end(), 200);
  const auto fit = yolo::YoloV1PretrainingCrops(source, 0, 1), held = yolo::YoloV1PretrainingCrops(source, 1, 2);
  check::Require(fit.size() == 1 && held.size() == 1 && fit.labels[0][0] == 3 && held.labels[0][0] == 7,
                 "pretraining split keeps source rows and labels separate");
  check::Near(fit.pixels[0], 0, 0, "pretraining canvas padding");
  check::Near(fit.pixels[28 * 56 + 28], 128, 0, "aspect-preserving centre crop pixel");
  check::Near(held.pixels[28 * 56 + 28], 200, 0, "validation crop pixel");
  auto changed         = source;
  changed.labels[1][0] = 9;
  std::fill(changed.pixels.begin() + 16, changed.pixels.end(), 255);
  const auto unchanged = yolo::YoloV1PretrainingCrops(changed, 0, 1);
  check::Require(unchanged.pixels == fit.pixels && unchanged.labels == fit.labels, "held-out row changes cannot affect fitting crops");
  const auto crops  = yolo::YoloV1PretrainingCrops(source, 0, 2);
  const auto [x, y] = yolo::YoloV1PretrainingBatch(crops, {0, 1});
  const auto xv     = x.to_vector<float>();
  const auto yv     = y.to_vector<std::int32_t>();
  check::Require(x.shape() == nn::Shape{2, 56 * 56} && yv[0] == 3 && yv[1] == 7, "pretraining batch dimensions and class pairing");
  check::Near(xv[0], -1, 0, "pretraining normalized black padding");
  check::Near(xv[28 * 56 + 28], 128 / 127.5 - 1, 1e-7, "pretraining normalized crop intensity");
  const auto directory = detector::TrainingDirectory("yolo_v1_pretrain_transfer_fixture") /
                         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  for (bool bn : {false, true}) {
    yolo::YoloV1Config cfg;
    cfg.compact          = true;
    cfg.classes          = 10;
    cfg.width_divisor    = 8;
    cfg.dense_width      = 8;
    cfg.keep_probability = 1;
    cfg.batch_norm       = bn;
    yolo::YoloV1Classifier classifier(cfg);
    check::Require(classifier.parameters.count() == (bn ? 353714 : 352578), "first-20 plus pooled classifier parameter count");
    classifier.Train(x, y);
    check::Require(std::isfinite(classifier.Evaluate(x, y).first), "pretraining classifier forward and optimizer step finite");
    auto weights = yolo::CaptureYoloV1Pretrained(classifier);
    check::Require(weights.parameters.size() == std::size_t(bn ? 60 : 40) && weights.state.size() == std::size_t(bn ? 40 : 0),
                   "capture excludes classification head and includes prefix BN state");
    // Distinct sentinels make every tensor differ from fresh initialization.
    int index = 0;
    for (auto& [name, value] : weights.parameters) value = nn::full(value.shape(), .001 * ++index);
    index = 0;
    for (auto& [name, value] : weights.state) value = nn::full(value.shape(), 1 + .001 * ++index);
    nn::ParameterStore p(17), state(17);
    {
      nn::NoGradGuard guard;
      yolo::YoloV1(p, &state, nn::zeros({1, 56, 56, 1}), cfg, false);
    }
    const std::vector<std::string>  tail = {"conv21/W", "fc1/W", "output/W"};
    std::vector<std::vector<float>> before;
    for (const auto& name : tail) before.push_back(p.at(name).to_vector<float>());
    const auto prefix = directory / (bn ? "bn" : "plain") / "bootstrap";
    yolo::WriteYoloV1Bootstrap(weights, cfg, prefix);
    detector::RestoreCheckpoint(p, state, prefix.string());
    for (std::size_t k = 0; k < tail.size(); ++k) {
      const auto after = p.at(tail[k]).to_vector<float>();
      for (std::size_t i = 0; i < after.size(); ++i)
        check::Near(after[i], before[k][i], 0, "final convolutions and dense head stay at original initialization");
    }
    auto verify = [&](const std::map<std::string, nn::Tensor>& expected, const nn::ParameterStore& actual) {
      for (const auto& [name, value] : expected) {
        const auto a = actual.at(name).to_vector<float>(), b = value.to_vector<float>();
        for (std::size_t i = 0; i < a.size(); ++i) check::Near(a[i], b[i], 0, "all pretrained prefix tensors survive checkpoint transfer");
      }
    };
    verify(weights.parameters, p);
    verify(weights.state, state);
  }
  std::cout << "  pretraining: disjoint source crops, classifier forward/update, exact first-20/BN checkpoint transfer and "
               "untouched detector tail passed"
            << std::endl;
}

void AugmentationChecks() {
  DetectionScenes source;
  source.height = 4;
  source.width  = 6;
  source.boxes  = {{{1, 2, 3, 4}}, {{0, 0, 1, 2}}};
  source.labels = {{3}, {7}};
  source.pixels.resize(48, 200);
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 6; ++c) source.pixels[std::size_t(r * 6 + c)] = std::uint8_t(4 + 10 * r + c);
  const auto original = source;
  using T             = detector::DetectionTransform;
  const auto identity = detector::TransformDetectionScenes(source, {0, 1}, {T{}, T{}});
  check::Require(identity.pixels == source.pixels && identity.boxes == source.boxes && identity.labels == source.labels,
                 "identity augmentation is exact");
  const auto shifted = detector::TransformDetectionScenes(source, {0}, {T{1, 1, -1, 1}});
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 6; ++c) {
      const int expected = r > 0 && c < 5 ? source.pixels[std::size_t((r - 1) * 6 + c + 1)] : 0;
      check::Near(shifted.pixels[std::size_t(r * 6 + c)], expected, 0, "integer translation pixel reference and zero padding");
    }
  check::Require(shifted.boxes[0] == std::vector<Box>{{2, 1, 4, 3}} && shifted.labels[0] == std::vector<int>{3},
                 "translated boxes and labels follow pixels");
  const auto scaled = detector::TransformDetectionScenes(source, {0}, {T{2, 0, 0, 1}});
  check::Require(scaled.boxes[0] == std::vector<Box>{{0, 1, 4, 5}}, "scale uses the rectangular image centre");
  check::Near(scaled.pixels[1 * 6 + 2], 19, 0, "scale inverse at (1.25,2.25) interpolates 18.75 and rounds to 19");
  const auto fractional = detector::TransformDetectionScenes(source, {0}, {T{1, .5, .5, 1}});
  check::Near(fractional.pixels[1 * 6 + 1], 10, 0, "half-pixel translation averages 4/5/14/15");
  check::Near(fractional.pixels[0], 1, 0, "bilinear border gets one quarter of the corner pixel");
  check::Require(fractional.boxes[0] == std::vector<Box>{{1.5, 2.5, 3.5, 4.5}}, "fractional box edges use the same transform");
  const auto clipped = detector::TransformDetectionScenes(source, {0}, {T{1, -2, 0, 1}});
  check::Require(clipped.boxes[0] == std::vector<Box>{{0, 2, 1, 4}} && clipped.labels[0] == std::vector<int>{3},
                 "partially visible target clipped with label preserved");
  const auto removed = detector::TransformDetectionScenes(source, {0}, {T{1, 0, 10, 1}});
  check::Require(removed.boxes[0].empty() && removed.labels[0].empty() &&
                     std::all_of(removed.pixels.begin(), removed.pixels.end(), [](std::uint8_t p) { return p == 0; }),
                 "fully off-image objects and pixels removed");
  const auto reordered = detector::TransformDetectionScenes(source, {1, 0}, {T{1, 0, 0, 1.5}, T{}});
  check::Require(reordered.labels == std::vector<std::vector<int>>{{7}, {3}} && reordered.boxes[0] == source.boxes[1],
                 "explicit row order preserves label pairing");
  for (std::size_t i = 0; i < 24; ++i) {
    check::Near(reordered.pixels[i], 255, 0, "exposure clamps at 255");
    check::Near(reordered.pixels[24 + i], source.pixels[i], 0, "other image remains unchanged");
  }
  const auto dimmed = detector::TransformDetectionScenes(source, {1}, {T{1, 0, 0, .5}});
  check::Near(dimmed.pixels[0], 100, 0, "exposure scales grayscale intensity");
  check::Require(source.pixels == original.pixels && source.boxes == original.boxes && source.labels == original.labels,
                 "augmentation never mutates its source");
  auto rejects = [](const DetectionScenes& input, const std::vector<int>& rows, const std::vector<T>& transforms) {
    bool rejected = false;
    try {
      detector::TransformDetectionScenes(input, rows, transforms);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    check::Require(rejected, "invalid augmentation input rejected");
  };
  for (const auto& t : std::vector<T>{{0, 0, 0, 1}, {-1, 0, 0, 1}, {1, 0, 0, 0}, {1, INFINITY, 0, 1}, {NAN, 0, 0, 1}})
    rejects(source, {0}, {t});
  rejects(source, {2}, {T{}});
  rejects(source, {0}, {});
  auto invalid = source;
  invalid.pixels.pop_back();
  rejects(invalid, {0}, {T{}});
  invalid = source;
  invalid.labels[0].clear();
  rejects(invalid, {0}, {T{}});
  invalid               = source;
  invalid.boxes[0][0][0] = NAN;
  rejects(invalid, {0}, {T{}});
  std::cout << "  affine augmentation: exact identity/translation/scale/bilinear/exposure fixtures, clipped/removed targets, "
               "row/label pairing and invalid inputs passed"
            << std::endl;
}

void TrainingPolicyChecks() {
  detector::DetectorTrainingConfig cfg;
  cfg.accuracy = false;
  detector::ApResult early;
  early.ap50       = .03;
  early.ap50_voc07 = .04;
  early.ap75       = .001;
  early.coco       = .008;
  detector::CheckDetectorDemonstration(cfg, {5, 4, 3}, early);
  auto rejects = [&](const std::vector<double>& losses, const detector::ApResult& metrics) {
    bool rejected = false;
    try {
      detector::CheckDetectorDemonstration(cfg, losses, metrics);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    check::Require(rejected, "training policy rejects invalid learning/metrics");
  };
  rejects({3, 4, 5}, early);
  rejects({3, 3, 3}, early);
  rejects({3}, early);
  rejects({NAN, 4, 3}, early);
  rejects({5, INFINITY, 3}, early);
  auto invalid = early;
  invalid.ap50 = NAN;
  rejects({5, 4, 3}, invalid);
  invalid      = early;
  invalid.coco = 1.1;
  rejects({5, 4, 3}, invalid);
  cfg.accuracy          = true;
  cfg.minimum_test_ap50 = .5;
  rejects({5, 4, 3}, early);
  auto achieved = early;
  achieved.ap50 = .5;
  detector::CheckDetectorDemonstration(cfg, {5, 4, 3}, achieved);
  std::cout << "  short-learning and strict-accuracy policies: falling loss accepted; flat/rising/nonfinite loss, invalid AP and "
               "unmet strict threshold rejected"
            << std::endl;
}

void LossChecks() {
  std::vector<float> value = {.2f, .8f, .5f, .5f, .5f, .5f, .4f, .2f, .2f, .3f, .3f, .2f};
  yolo::YoloV1Targets t;
  t.object = {1};
  t.labels = {1};
  t.boxes  = {.5f, .5f, .5f, .5f};
  auto run = [&](double expected_loss, const std::vector<double>& expected_gradient, int responsible) {
    auto       prediction = nn::from_vector(value, {1, 1, 1, 12}).requires_grad_();
    const auto out        = yolo::YoloV1Loss(prediction, t, 2, 1);
    const auto gradient   = nn::grad(out.loss, {prediction})[0].to_vector<float>();
    check::Near(out.loss.item(), expected_loss, 2e-6, "independent YOLO loss");
    for (std::size_t i = 0; i < 12; ++i) check::Near(gradient[i], expected_gradient[i], 2e-6, "analytic YOLO derivative");
    const auto r = out.responsible.to_vector<float>();
    for (int i = 0; i < 2; ++i) check::Near(r[std::size_t(i)], i == responsible ? 1 : 0, 0, "responsible predictor");
  };
  run(.46, {.4, -.4, 0, 0, 0, 0, -1.2, 0, 0, 0, 0, .2}, 0);
  value[4] = value[5] = .4f;
  run(.20009216, {.4, -.4, 0, 0, -1, -1, -.0192, 0, 0, 0, 0, .2}, 0);
  for (std::size_t i = 0; i < 4; ++i) {
    value[2 + i] = .1f;
    value[7 + i] = .5f;
  }
  run(.8, {.4, -.4, 0, 0, 0, 0, .4, 0, 0, 0, 0, -1.6}, 1);
  t.object[0] = 0;
  run(.1, {0, 0, 0, 0, 0, 0, .4, 0, 0, 0, 0, .2}, -1);
  std::cout << "  exact/imperfect/swapped/empty-cell losses and every analytic derivative passed; IoU targets detached" << std::endl;
}

void AssignmentChecks() {
  DetectionScenes scene;
  scene.height = scene.width = 56;
  scene.boxes                = {{{8, 8, 16, 16}, {7, 7, 17, 17}}};
  scene.labels               = {{3, 4}};
  const auto t               = yolo::MakeYoloV1Targets(scene, {0});
  check::Require(t.discarded == 1, "report one grid collision");
  const std::size_t slot = 1 * 7 + 1;
  check::Require(t.labels[slot] == 4, "keep the larger colliding object");
  check::Near(t.boxes[slot * 4], .5, 1e-6, "centre offset within assigned cell");
  check::Near(t.boxes[slot * 4 + 2], std::sqrt(10. / 56), 1e-6, "root height encoding");
  std::cout << "  centre assignment keeps the larger of two colliding boxes and reports the discarded target" << std::endl;
}

void DecodingChecks() {
  DetectionScenes scenes;
  scenes.height = 16;
  scenes.width  = 24;
  scenes.pixels.assign(2 * 16 * 24, 0);
  scenes.boxes.resize(2);
  scenes.labels.resize(2);
  std::vector<float> values(2 * 2 * 2 * 12, 0);
  auto v = [&](int i, int r, int c, int k) -> float& { return values[std::size_t(((i * 2 + r) * 2 + c) * 12 + k)]; };
  // Image 0, top-right cell: centre (4,15), height 4, width 6.
  // Both box predictors coincide, so NMS keeps only the stronger of each class.
  v(0, 0, 1, 0) = .8f;
  v(0, 0, 1, 1) = .2f;
  for (int b = 0; b < 2; ++b) {
    const int k       = 2 + 5 * b;
    v(0, 0, 1, k)     = .5f;
    v(0, 0, 1, k + 1) = .25f;
    v(0, 0, 1, k + 2) = .5f;
    v(0, 0, 1, k + 3) = .5f;
    v(0, 0, 1, k + 4) = b ? .5f : .75f;
  }
  // Image 1, bottom-left cell: centre (10,6), height 4, width 6.
  v(1, 1, 0, 1) = .9f;
  v(1, 1, 0, 2) = .25f;
  v(1, 1, 0, 3) = .5f;
  v(1, 1, 0, 4) = .5f;
  v(1, 1, 0, 5) = .5f;
  v(1, 1, 0, 6) = .8f;
  // Substitute known predictions for a model output while exercising the
  // real image batching, coordinate decoder, score filtering and NMS.
  auto decode = [&] {
    return yolo::DetectYoloV1([&](const nn::Tensor&) { return nn::from_vector(values, {2, 2, 2, 12}); }, scenes, {0, 1}, 2, 2);
  };
  const auto detections = decode();
  check::Require(detections.size() == 3, "YOLO decoding preserves two classes and independent images after duplicate suppression");
  bool seen[2][2] = {};
  for (const auto& d : detections) {
    const Box expected = d.image == 0 ? Box{2, 12, 6, 18} : Box{8, 3, 12, 9};
    check::Require(d.image == 0 || d.image == 1, "decoded image index");
    check::Require(d.label >= 0 && d.label < 2 && !seen[d.image][d.label], "one detection per expected image/class");
    seen[d.image][d.label] = true;
    if (d.image == 1) check::Require(d.label == 1, "second image class");
    for (std::size_t k = 0; k < 4; ++k) check::Near(d.box[k], expected[k], 1e-6, "rectangular-image YOLO box decoding");
    check::Near(d.score, d.image == 1 ? .72 : d.label == 0 ? .6 : .15, 1e-6, "conditional class times box confidence");
  }
  std::fill(values.begin(), values.end(), 0.f);
  v(0, 0, 0, 0)  = -1;
  v(0, 0, 0, 1)  = .7f;
  v(0, 0, 0, 4)  = .5f;
  v(0, 0, 0, 5)  = .5f;
  v(0, 0, 0, 6)  = 1.2f;
  v(0, 0, 0, 9)  = .5f;
  v(0, 0, 0, 10) = .5f;
  v(0, 0, 0, 11) = -1;
  const auto clipped = decode();
  check::Require(clipped.size() == 1 && clipped[0].image == 0 && clipped[0].label == 1, "negative class/confidence scores are filtered");
  const Box expected = {0, 0, 2, 3};
  for (std::size_t k = 0; k < 4; ++k) check::Near(clipped[0].box[k], expected[k], 0, "decoded corner clipping");
  check::Near(clipped[0].score, .7, 1e-6, "confidence capped at one");
  std::cout << "  rectangular-image decoding, class-confidence scores, clipping and per-image/class NMS fixtures passed" << std::endl;
}

void ModelChecks(bool batch_norm, double keep_probability, int dense_width) {
  {
    nn::ParameterStore p(17);
    yolo::YoloV1Config full;
    const auto         net = yolo::YoloV1Backbone(p, nullptr, SinImages(1, 448, 448, 3), full, false);
    check::Require(net.convolutions.size() == 24 && net.features.shape() == nn::Shape{1, 7, 7, 1024}, "full paper YOLO v1 7x7x1024 features");
    const std::int64_t head = (7 * 7 * 1024) * 4096 + 4096 + 4096 * (7 * 7 * 30) + 7 * 7 * 30;
    check::Require(p.count() + head == 271703550, "full paper YOLO v1 architecture and parameter count");
    const auto gradients = nn::grad(nn::mean(nn::square(net.features)), p.parameters());
    for (const auto& g : gradients) check::Require(g.defined(), "every full convolution parameter connected");
    check::Require(gradients.size() == 48, "all 24 convolution kernel/bias derivatives computed");
    std::cout << "  full paper architecture: 24 convolutions (" << p.count() << " parameters, 48 derivatives computed at 448 px) + dense head "
              << head << " counted from shapes = 271703550 parameters" << std::endl;
  }
  nn::ParameterStore p(17), state(17);
  yolo::YoloV1Config cfg;
  cfg.classes          = 10;
  cfg.compact          = true;
  cfg.width_divisor    = 8;
  cfg.dense_width      = dense_width;
  cfg.batch_norm       = batch_norm;
  cfg.keep_probability = keep_probability;
  cfg.update_stats     = false;
  nn::Generator dropout(37);
  const auto    image = SinImages(2, 56, 56, 1);
  const auto    net   = yolo::YoloV1(p, &state, image, cfg, true, &dropout);
  check::Require(p.count() == 942604 + 7253LL * dense_width + (batch_norm ? 1648 : 0), "compact convolution and dense-head parameter count");
  check::Require(p.at("fc1/W").shape() == nn::Shape{7 * 7 * 128, dense_width} && p.at("output/W").shape() == nn::Shape{dense_width, 7 * 7 * 20},
                 "selected compact dense width reaches both head layers");
  check::Require(net.features.shape() == nn::Shape{2, 7, 7, 128} && net.prediction.shape() == nn::Shape{2, 7, 7, 20}, "compact YOLO grids");
  const auto gradients = nn::grad(nn::mean(nn::square(net.prediction)), p.parameters());
  double     minimum = INFINITY, maximum = 0;
  for (std::size_t i = 0; i < gradients.size(); ++i) {
    const double norm = std::sqrt(nn::sum(nn::square(gradients[i])).item());
    check::Require(std::isfinite(norm) && norm > 0, "nonzero finite compact YOLO derivative: " + p.names()[i]);
    minimum = std::min(minimum, norm);
    maximum = std::max(maximum, norm);
  }
  if (batch_norm) {
    nn::NoGradGuard guard;
    const auto      constant = nn::full({2, 56, 56, 1}, .5);
    auto            calibrate = cfg;
    calibrate.update_stats    = true;
    yolo::YoloV1(p, &state, constant, calibrate, true, &dropout);
    nn::Conv2DOptions o = nn::conv_options(1, nn::Padding::Valid);
    const auto first = nn::mean(nn::conv2d(nn::pad(constant, {{0, 0}, {3, 3}, {3, 3}, {0, 0}}), p.at("conv1/W"), o), {0, 1, 2}).to_vector<float>();
    const auto moving = state.at("conv1/bn/moving_mean").to_vector<float>();
    for (std::size_t c = 0; c < 8; ++c) check::Near(moving[c], .1 * first[c], 2e-6, "compact BN updates mean with .1 batch weight");
    std::cout << "  compact BN momentum .9 verified: each moving mean gets .1 of independently recomputed batch mean" << std::endl;
  }
  std::cout << "  compact model (BN=" << batch_norm << ", dropout=" << 1 - keep_probability << ", dense=" << dense_width << "): " << p.count()
            << " parameters, 7x7x128 features, prediction 7x7x20; derivative norms [" << minimum << ',' << maximum << "]" << std::endl;
}

void DiagnoseCheckpoint(const DetectionScenes& scenes, const std::string& prefix, bool batch_norm, int dense_width, double expected_ap50) {
  nn::ParameterStore p(17), state(17);
  yolo::YoloV1Config cfg;
  cfg.classes          = 10;
  cfg.compact          = true;
  cfg.width_divisor    = 8;
  cfg.dense_width      = dense_width;
  cfg.batch_norm       = batch_norm;
  cfg.keep_probability = 1;
  cfg.update_stats     = false;
  {
    nn::NoGradGuard guard;
    yolo::YoloV1(p, &state, nn::zeros({1, 56, 56, 1}), cfg, true);
  }
  detector::RestoreCheckpoint(p, state, prefix);
  nn::NoGradGuard guard;
  std::cout << "== YOLO v1 checkpoint diagnosis: " << prefix
            << "\n  batch=32; no updates, checkpoint writes or test evaluation; all objects including grid collisions" << std::endl;
  for (bool validation : {false, true}) {
    struct Summary {
      double loss = 0, iou = 0, confidence = 0;
      int    correct = 0, localized = 0, objects = 0;
    } summaries[2];
    const int start = validation ? scenes.size() - 500 : 0;
    for (int offset = 0; offset < 128; offset += 32) {
      std::vector<int> rows(32);
      std::iota(rows.begin(), rows.end(), start + offset);
      const auto images  = detector::SceneImages(scenes, rows);
      const auto targets = yolo::MakeYoloV1Targets(scenes, rows);
      for (int mode = 0; mode < 2; ++mode) {
        const auto prediction = yolo::YoloV1(p, &state, images, cfg, mode == 0).prediction;
        auto&      summary    = summaries[mode];
        summary.loss += yolo::YoloV1Loss(prediction, targets, 10).loss.item() / 4;
        const auto values = prediction.to_vector<float>();
        auto       at     = [&](int n, int r, int c, int k) { return values[std::size_t(((n * 7 + r) * 7 + c) * 20 + k)]; };
        for (int n = 0; n < 32; ++n)
          for (std::size_t object = 0; object < scenes.boxes[std::size_t(rows[std::size_t(n)])].size(); ++object) {
            const auto& target = scenes.boxes[std::size_t(rows[std::size_t(n)])][object];
            const int   r      = std::clamp(int((target[0] + target[2]) * .5f / float(scenes.height) * 7), 0, 6);
            const int   c      = std::clamp(int((target[1] + target[3]) * .5f / float(scenes.width) * 7), 0, 6);
            int         label  = 0;
            for (int k = 1; k < 10; ++k)
              if (at(n, r, c, k) > at(n, r, c, label)) label = k;
            summary.correct += label == scenes.labels[std::size_t(rows[std::size_t(n)])][object];
            float best = -1, confidence = 0;
            for (int box = 0; box < 2; ++box) {
              const int   k = 10 + 5 * box;
              const float y = (float(r) + at(n, r, c, k)) / 7 * float(scenes.height), x = (float(c) + at(n, r, c, k + 1)) / 7 * float(scenes.width);
              const float h = std::pow(at(n, r, c, k + 2), 2.f) * float(scenes.height), w = std::pow(at(n, r, c, k + 3), 2.f) * float(scenes.width);
              const float iou = detector::BoxIoU({y - h / 2, x - w / 2, y + h / 2, x + w / 2}, target);
              if (iou > best) {
                best       = iou;
                confidence = at(n, r, c, k + 4);
              }
            }
            summary.iou += best;
            summary.confidence += confidence;
            summary.localized += best >= .5f;
            ++summary.objects;
          }
      }
    }
    for (int mode = 0; mode < 2; ++mode) {
      const auto& summary = summaries[mode];
      check::Require(summary.objects > 0, "diagnostic objects exist");
      std::cout << "  " << (validation ? "validation" : "fit") << ' ' << (mode ? "inference" : "training") << ": loss=" << summary.loss
                << "; objects=" << summary.objects << "; cell class accuracy=" << double(summary.correct) / summary.objects
                << "; mean best-box IoU=" << summary.iou / summary.objects
                << "; IoU>=.5 fraction=" << double(summary.localized) / summary.objects
                << "; mean best-box confidence=" << summary.confidence / summary.objects << std::endl;
    }
  }
  std::vector<int> validation(500);
  std::iota(validation.begin(), validation.end(), scenes.size() - 500);
  const auto restored = detector::EvaluateDetections(
      yolo::DetectYoloV1([&](const nn::Tensor& images) { return yolo::YoloV1(p, &state, images, cfg, false).prediction; }, scenes, validation, 10),
      scenes, validation, 10);
  std::cout << "  restored full validation " << detector::ApSummary(restored) << std::endl;
  if (std::isfinite(expected_ap50)) {
    check::Near(restored.ap50, expected_ap50, 1e-6, "restored validation AP50 matches the recorded selected checkpoint");
    std::cout << "  restored validation AP50 agrees with recorded history within 1e-6" << std::endl;
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path directory = paths::Dataset("mnist");
    std::string           diagnose;
    bool                  batch_norm = false, checks_only = false, augment = false, pretrain_only = false;
    double                keep_probability = 1, expected_ap50 = NAN;
    int                   dense_width = 256, pretrain_epochs = 0;
    detector::DetectorTrainingConfig cfg;
    cfg.name              = "yolo_v1_digits";
    cfg.epochs            = 3;
    cfg.accuracy          = false;
    cfg.learning_rate     = .0005f;
    cfg.minimum_test_ap50 = .5;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto        eq  = arg.find('=');
      if (eq == std::string::npos) throw std::invalid_argument("expected key=value option");
      const auto key = arg.substr(0, eq), value = arg.substr(eq + 1);
      if (key == "data") directory = value;
      else if (key == "epochs") cfg.epochs = std::stoi(value);
      else if (key == "accuracy" && (value == "0" || value == "1")) cfg.accuracy = value == "1";
      else if (key == "warm_start") cfg.warm_start = value;
      else if (key == "lr") cfg.learning_rate = std::stof(value);
      else if (key == "batch_norm" && (value == "0" || value == "1")) batch_norm = value == "1";
      else if (key == "checks_only" && (value == "0" || value == "1")) checks_only = value == "1";
      else if (key == "augment" && (value == "0" || value == "1")) augment = value == "1";
      else if (key == "dense") {
        dense_width = std::stoi(value);
        if (dense_width < 1) throw std::invalid_argument("dense must be positive");
      } else if (key == "pretrain") {
        pretrain_epochs = std::stoi(value);
        if (pretrain_epochs < 1) throw std::invalid_argument("pretrain epochs must be positive");
      } else if (key == "pretrain_only" && (value == "0" || value == "1")) pretrain_only = value == "1";
      else if (key == "diagnose") diagnose = value;
      else if (key == "expect_ap50") {
        expected_ap50 = std::stod(value);
        if (!std::isfinite(expected_ap50) || expected_ap50 < 0 || expected_ap50 > 1)
          throw std::invalid_argument("expected AP50 must be finite and in [0,1]");
      } else if (key == "dropout") {
        const double rate = std::stod(value);
        if (!(rate >= 0 && rate < 1)) throw std::invalid_argument("dropout must be in [0,1)");
        keep_probability = 1 - rate;
      } else if (key == "threads") nn::set_num_threads(std::stoi(value));
      else throw std::invalid_argument("unknown option " + key);
    }
    if (std::isfinite(expected_ap50) && diagnose.empty()) throw std::invalid_argument("expect_ap50 requires diagnose=PREFIX");
    if (pretrain_only && !pretrain_epochs) throw std::invalid_argument("pretrain_only requires pretrain=N");
    if (pretrain_epochs && (!cfg.warm_start.empty() || !diagnose.empty() || checks_only))
      throw std::invalid_argument("choose pretraining, warm_start, diagnose or checks_only");
    TrainingPolicyChecks();
    PretrainingChecks();
    AugmentationChecks();
    LossChecks();
    AssignmentChecks();
    DecodingChecks();
    ModelChecks(batch_norm, keep_probability, dense_width);
    if (checks_only) {
      std::cout << "All YOLO v1 fixtures passed (training not requested).\n";
      return 0;
    }
    detector::SceneConfig train_cfg;
    train_cfg.scenes = 6000;
    train_cfg.seed   = 11;
    train_cfg.height = train_cfg.width = 56;
    detector::SceneConfig test_cfg     = train_cfg;
    test_cfg.scenes                    = 1000;
    test_cfg.seed                      = 22;
    const auto train = detector::MakeDetectionScenes(detector::LoadDigits(directory, "train"), train_cfg);
    if (!diagnose.empty()) {
      check::Require(keep_probability == 1, "checkpoint diagnosis requires dropout disabled");
      DiagnoseCheckpoint(train, diagnose, batch_norm, dense_width, expected_ap50);
      return 0;
    }
    yolo::YoloV1Config model;
    model.classes          = 10;
    model.compact          = true;
    model.width_divisor    = 8;
    model.dense_width      = dense_width;
    model.batch_norm       = batch_norm;
    model.keep_probability = keep_probability;
    if (pretrain_epochs) {
      cfg.warm_start = yolo::PretrainYoloV1(train, model, pretrain_epochs);
      if (pretrain_only) {
        std::cout << "Classifier pretraining and bootstrap checks passed (detector training not requested).\n";
        return 0;
      }
    }
    const auto test = detector::MakeDetectionScenes(detector::LoadDigits(directory, "t10k"), test_cfg);
    std::mt19937_64                        augment_random(73);
    std::uniform_real_distribution<double> scale(.8, 1.2), shift(-.2, .2), log_exposure(-std::log(1.5), std::log(1.5));
    nn::Generator                          dropout(37);
    std::cout << "  training affine augmentation=" << augment << "; validation/test inputs unchanged" << std::endl;
    detector::DetectorHooks hooks;
    hooks.build = [&](nn::ParameterStore& p, nn::ParameterStore& state) {
      auto frozen         = model;
      frozen.update_stats = false;
      yolo::YoloV1(p, &state, nn::zeros({1, 56, 56, 1}), frozen, true, &dropout);
    };
    hooks.loss = [&](nn::ParameterStore& p, nn::ParameterStore& state, const DetectionScenes& scenes, const std::vector<int>& rows) {
      if (augment) {
        std::vector<detector::DetectionTransform> transforms(rows.size());
        for (auto& t : transforms) {
          t.scale    = scale(augment_random);
          t.shift_y  = shift(augment_random) * scenes.height;
          t.shift_x  = shift(augment_random) * scenes.width;
          t.exposure = std::exp(log_exposure(augment_random));
        }
        const auto       batch = detector::TransformDetectionScenes(scenes, rows, transforms);
        std::vector<int> picked(rows.size());
        std::iota(picked.begin(), picked.end(), 0);
        const auto prediction = yolo::YoloV1(p, &state, detector::SceneImages(batch, picked), model, true, &dropout).prediction;
        return yolo::YoloV1Loss(prediction, yolo::MakeYoloV1Targets(batch, picked), 10).loss;
      }
      const auto prediction = yolo::YoloV1(p, &state, detector::SceneImages(scenes, rows), model, true, &dropout).prediction;
      return yolo::YoloV1Loss(prediction, yolo::MakeYoloV1Targets(scenes, rows), 10).loss;
    };
    hooks.detect = [&](nn::ParameterStore& p, nn::ParameterStore& state, const DetectionScenes& scenes, const std::vector<int>& rows) {
      return yolo::DetectYoloV1([&](const nn::Tensor& images) { return yolo::YoloV1(p, &state, images, model, false).prediction; }, scenes,
                                rows, 10);
    };
    {  // Count assignment collisions without inspecting test labels or tuning on them.
      std::vector<int> all(std::size_t(train.size()));
      std::iota(all.begin(), all.end(), 0);
      std::cout << "  prepared (unaugmented) training scenes: " << yolo::MakeYoloV1Targets(train, all).discarded
                << " object targets discarded by same-cell collisions" << std::endl;
    }
    detector::TrainDetector(cfg, train, test, hooks);
    std::cout << "All YOLO v1 checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
