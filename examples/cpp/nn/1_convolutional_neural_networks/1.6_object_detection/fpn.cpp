// CMD: just run-nn fpn
// Feature pyramid networks: top-down semantics and lateral connections at every scale.
//
// Port of tflow's fpn. Preparation (in ~/projects/tflow_examples):
// python3 scripts/fetch_datasets.py mnist --split both.
//
// Check nearest-neighbour top-down fusion on odd rectangular maps against
// an independent host calculation, including every input derivative. Give
// each smoothing filter a different central coefficient to detect accidental
// propagation of the smoothed maps. Negative outputs must remain negative.
// Check P6 subsampling and the paper's RoI level assignment with constant
// feature maps at levels 2..5, including both clamped extremes.
// https://arxiv.org/abs/1612.03144
// https://github.com/pytorch/vision/blob/main/torchvision/ops/feature_pyramid_network.py
//
// Train a compact Faster R-CNN with 64-channel P2..P5 and pooled P6 on
// deterministic MNIST digit scenes (5500 fit / 500 validation / 1000 test).
// The shared RPN uses three ratios and one anchor size per level, and RoIs
// select a pyramid level before the shared head. The 64-pixel scene adaptation
// uses anchor sides 8/16/32/64/128 and canonical RoI side 32 at level 4.
// The backbone is a small convolutional pyramid, not an ImageNet-pretrained
// ResNet. RoI pooling uses bilinear crops plus max pooling, as documented in
// two_stage.hpp. Optimize the combined proposal/head loss for ten epochs,
// select by validation AP50, then evaluate the separate test scenes once.
//
// Eager port: the level assignment reuses panoptic::PanopticLevels (job
// 1.7a's pyramid routing) and the pixel-RoI box head maskrcnn::PooledRoiHead;
// each RoI is pooled only on its own level and the original order restored
// with a gather (tflow pools every RoI on every level and masks; same values
// and gradients). Each step runs the backbone and RPN once and takes the
// detached proposals from that pass (tflow runs a separate proposal pass with
// the same weights). Weights come from nn's generator, so trained numbers
// match TF statistically. Checkpoint (.npz) and history:
// output/nn/models/fpn_digits/.
// Options: data=PATH epochs=10 threads=N.

#include <iostream>

#include "common/panoptic_fpn.hpp"

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
/// One pyramid level for RoI pooling: features [N, size, size, C] at `stride`.
struct PyramidLevel {
  nn::Tensor features;
  int        stride, size;
};
/// tflow's ml::PyramidRoiPool (first level 2, k0 = 4): crop_and_resize to
/// 14x14 then 2x2 max pooling on the level clamp(floor(4 + log2(sqrt(wh) /
/// canonical)), 2, 2 + levels - 1) of each pixel box -> [K, 7, 7, C].
nn::Tensor PyramidRoiPool(const std::vector<PyramidLevel>& levels, const std::vector<Box>& boxes,
                          const std::vector<int>& indices, float canonical) {
  if (levels.size() != 4 || !(canonical > 0)) throw std::invalid_argument("invalid pyramid RoI configuration");
  for (const auto& lv : levels)
    if (lv.stride < 1 || lv.size < 2) throw std::invalid_argument("pyramid RoI levels require positive stride and at least two cells");
  const auto              assigned = panoptic::PanopticLevels(boxes, canonical);
  std::vector<nn::Tensor> parts;
  std::vector<int>        order;
  for (std::size_t l = 0; l < levels.size(); ++l) {
    std::vector<std::array<float, 4>> rois;
    std::vector<int>                  selected;
    for (std::size_t k = 0; k < boxes.size(); ++k)
      if (assigned[k] == int(l) + 2) {
        rois.push_back(detector::FeatureBox(boxes[k], levels[l].stride, levels[l].size));
        selected.push_back(indices[k]);
        order.push_back(int(k));
      }
    if (rois.empty()) continue;
    parts.push_back(nn::max_pool2d(nn::crop_and_resize(levels[l].features, detector::RoiTensor(rois), detector::IndexTensor(selected), {14, 14}),
                                   2, 2, nn::Padding::Valid));
  }
  if (parts.empty()) return nn::zeros({0, 7, 7, levels[0].features.shape()[3]});
  const nn::Tensor          grouped = parts.size() == 1 ? parts[0] : nn::concat(parts, 0);
  std::vector<std::int32_t> inverse(order.size());
  for (std::size_t j = 0; j < order.size(); ++j) inverse[std::size_t(order[j])] = std::int32_t(j);
  return nn::gather(grouped, nn::from_vector(inverse), 0);
}

void FusionChecks() {
  const int          height[] = {7, 4, 2}, width[] = {9, 5, 3};
  std::vector<std::vector<float>> values(3);
  for (int l = 0; l < 3; ++l)
    for (int i = 0; i < height[l] * width[l]; ++i) values[std::size_t(l)].push_back(std::sin(.4f * float(i) + float(l)));
  nn::ParameterStore      p(17);
  std::vector<nn::Tensor> input;
  for (int l = 0; l < 3; ++l) input.push_back(nn::from_vector(values[std::size_t(l)], {1, height[l], width[l], 1}).requires_grad_());
  {
    nn::NoGradGuard guard;
    fpn::FeaturePyramid(p, "pyramid", input, 1);
  }
  for (int l = 0; l < 3; ++l) {
    const auto         prefix = "pyramid/level" + std::to_string(l + 2);
    std::vector<float> weight(9, 0);
    weight[4] = float(l + 2);
    p.set(prefix + "/smooth/W", nn::from_vector(weight, {3, 3, 1, 1}));
    p.set(prefix + "/lateral/W", nn::ones({1, 1, 1, 1}));
  }
  const auto pyramid = fpn::FeaturePyramid(p, "pyramid", input, 1);
  nn::Tensor loss    = nn::sum(pyramid.levels[0]);
  for (int l = 1; l < 3; ++l) loss = loss + nn::sum(pyramid.levels[std::size_t(l)]);
  const auto                       gradients = nn::grad(loss, input);
  std::vector<std::vector<double>> expected_gradient;
  for (int l = 0; l < 3; ++l) expected_gradient.emplace_back(std::size_t(height[l] * width[l]), 0);
  double error = 0, derivative_error = 0;
  int    negative = 0;
  for (int l = 0; l < 3; ++l) {
    const auto& level = pyramid.levels[std::size_t(l)];
    check::Require(level.shape() == nn::Shape{1, height[l], width[l], 1}, "FPN preserves each rectangular lateral grid");
    const auto out = level.to_vector<float>();
    for (int r = 0; r < height[l]; ++r)
      for (int c = 0; c < width[l]; ++c) {
        int    rr = r, cc = c;
        double expected = 0;
        for (int j = l; j < 3; ++j) {
          if (j > l) {
            rr = rr * height[j] / height[j - 1];
            cc = cc * width[j] / width[j - 1];
          }
          expected += values[std::size_t(j)][std::size_t(rr * width[j] + cc)];
          expected_gradient[std::size_t(j)][std::size_t(rr * width[j] + cc)] += l + 2;
        }
        const float actual = out[std::size_t(r * width[l] + c)];
        error              = std::max(error, std::abs(actual - (l + 2) * expected));
        negative += actual < 0;
      }
  }
  for (int l = 0; l < 3; ++l) {
    const auto g = gradients[std::size_t(l)].to_vector<float>();
    for (int i = 0; i < height[l] * width[l]; ++i)
      derivative_error = std::max(derivative_error, std::abs(g[std::size_t(i)] - expected_gradient[std::size_t(l)][std::size_t(i)]));
  }
  const auto p6 = pyramid.levels[3].to_vector<float>(), p4 = pyramid.levels[2].to_vector<float>();
  check::Require(pyramid.levels[3].shape() == nn::Shape{1, 1, 2, 1}, "P6 uses stride-two 1x1 pooling, including odd widths");
  check::Near(p6[0], p4[0], 0, "P6 first sample");
  check::Near(p6[1], p4[2], 0, "P6 second sample");
  check::Near(error, 0, 2e-6, "independent FPN top-down and smoothing reference");
  check::Near(derivative_error, 0, 0, "independent FPN derivative accumulation");
  check::Require(negative > 0, "linear FPN maps preserve negative values");
  std::cout << "  odd maps 7x9/4x5/2x3: fusion error=" << error << ", input derivative error=" << derivative_error << "; P6 samples exact; "
            << negative << " negative outputs preserved" << std::endl;
}

void RoiLevelChecks() {
  std::vector<PyramidLevel> levels;
  for (int k = 2; k <= 5; ++k) {
    const int cells = 2048 / (1 << k);
    levels.push_back({nn::full({1, cells, cells, 1}, double(k)), 1 << k, cells});
  }
  const int        sides[] = {32, 64, 112, 224, 448, 896}, expected[] = {2, 2, 3, 4, 5, 5};
  std::vector<Box> boxes;
  for (int side : sides) boxes.push_back({64, 64, 64 + float(side), 64 + float(side)});
  const auto pooled = PyramidRoiPool(levels, boxes, std::vector<int>(6, 0), 224);
  check::Require(pooled.shape() == nn::Shape{6, 7, 7, 1}, "pyramid RoI output geometry");
  const auto out = pooled.to_vector<float>();
  for (int i = 0; i < 6; ++i)
    for (int pixel = 0; pixel < 49; ++pixel) check::Near(out[std::size_t(49 * i + pixel)], expected[i], 1e-6, "RoI selects its assigned pyramid level");
  std::cout << "  RoI sides 32/64/112/224/448/896 select P2/P2/P3/P4/P5/P5 exactly" << std::endl;
}

std::vector<Box> Anchors() {
  std::vector<Box> anchors;
  for (int k = 2; k <= 6; ++k) {
    const auto level = detector::AnchorGrid(64 / (1 << k), 64 / (1 << k), float(1 << k), {float(2 << k)}, {.5f, 1, 2});
    anchors.insert(anchors.end(), level.begin(), level.end());
  }
  return anchors;
}

struct Forward {
  std::vector<PyramidLevel> roi_levels;
  detector::Rpn             rpn;
};
Forward Build(nn::ParameterStore& p, const nn::Tensor& images) {
  std::vector<nn::Tensor> inputs;
  auto                    h = detector::StrideFourBackbone(p, images);
  inputs.push_back(h);
  std::int64_t         in = 64;
  nn::ConvLayerOptions conv;
  conv.init       = nn::init::he_uniform();
  conv.activation = nn::Activation::Relu;
  conv.stride     = 2;
  for (int stage = 3; stage <= 5; ++stage) {
    const std::int64_t channels = std::min<std::int64_t>(256, 2 * in);
    h                           = nn::conv2d(p, "backbone/c" + std::to_string(stage), h, channels, 3, conv);
    inputs.push_back(h);
    in = channels;
  }
  const auto                     pyramid = fpn::FeaturePyramid(p, "fpn", inputs, 64);
  std::vector<detector::RpnLevel> rpn_levels;
  Forward                        f;
  for (int k = 2; k <= 6; ++k) {
    const int stride = 1 << k, cells = 64 / stride;
    rpn_levels.push_back({pyramid.levels[std::size_t(k - 2)], 64, cells, 3});
    if (k <= 5) f.roi_levels.push_back({pyramid.levels[std::size_t(k - 2)], stride, cells});
  }
  f.rpn = detector::BuildRpn(p, rpn_levels);
  return f;
}
detector::RoiHeadOutputs Head(nn::ParameterStore& p, const Forward& f, const std::vector<Box>& boxes, const std::vector<int>& indices) {
  return maskrcnn::PooledRoiHead(p, PyramidRoiPool(f.roi_levels, boxes, indices, 32), 11);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path            directory = paths::Dataset("mnist");
    detector::DetectorTrainingConfig cfg;
    cfg.name              = "fpn_digits";
    cfg.epochs            = 10;
    cfg.batch             = 2;
    cfg.minimum_test_ap50 = .7;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto        eq  = arg.find('=');
      if (eq == std::string::npos) throw std::invalid_argument("expected key=value option");
      const auto key = arg.substr(0, eq), value = arg.substr(eq + 1);
      if (key == "data") directory = value;
      else if (key == "epochs") cfg.epochs = std::stoi(value);
      else if (key == "threads") nn::set_num_threads(std::stoi(value));
      else throw std::invalid_argument("unknown option " + key);
    }
    FusionChecks();
    RoiLevelChecks();
    const auto [train, test] = detector::StandardScenes(directory);
    const auto      anchors  = Anchors();
    std::mt19937_64 random(7);
    detector::DetectorHooks hooks;
    hooks.build = [&](nn::ParameterStore& p, nn::ParameterStore&) {
      const auto f = Build(p, nn::zeros({1, 64, 64, 1}));
      check::Require(f.rpn.anchors == int(anchors.size()), "pyramid anchor order/count");
      Head(p, f, {{0, 0, 8, 8}}, {0});
    };
    hooks.loss = [&](nn::ParameterStore& p, nn::ParameterStore&, const DetectionScenes& scenes, const std::vector<int>& rows) {
      const auto                    f = Build(p, detector::SceneImages(scenes, rows));
      std::vector<std::vector<Box>> proposals;
      {
        nn::NoGradGuard guard;
        proposals = detector::ComputeProposals(f.rpn, anchors, scenes.height, scenes.width);
      }
      detector::RoiBatch batch;
      for (std::size_t i = 0; i < rows.size(); ++i)
        detector::SampleRois(scenes, rows[i], int(i), proposals[i], 64, 10, 4, 16, random, batch);
      const auto targets = detector::MakeRpnTargets(anchors, scenes, rows, random);
      std::vector<Box> pixels;
      for (const auto& b : batch.pixels) pixels.push_back(b);
      const auto out = Head(p, f, pixels, batch.indices);
      return detector::RpnLoss(f.rpn, targets.labels, targets.weights, targets.targets, targets.positives) +
             detector::RoiLoss(out, batch, 11);
    };
    hooks.detect = [&](nn::ParameterStore& p, nn::ParameterStore&, const DetectionScenes& scenes, const std::vector<int>& rows) {
      nn::NoGradGuard                  guard;
      std::vector<detector::Detection> found;
      for (std::size_t start = 0; start < rows.size(); start += 16) {
        const std::vector<int> chunk(rows.begin() + std::ptrdiff_t(start), rows.begin() + std::ptrdiff_t(std::min(rows.size(), start + 16)));
        const auto             f         = Build(p, detector::SceneImages(scenes, chunk));
        const auto             proposals = detector::ComputeProposals(f.rpn, anchors, scenes.height, scenes.width);
        std::vector<std::pair<int, Box>> boxes;
        for (std::size_t i = 0; i < chunk.size(); ++i)
          for (const auto& b : proposals[i]) boxes.push_back({int(i), b});
        const auto detections = maskrcnn::DetectPixelRois(
            [&](const std::vector<Box>& rois, const std::vector<int>& indices) { return Head(p, f, rois, indices); }, chunk, boxes, 11);
        found.insert(found.end(), detections.begin(), detections.end());
      }
      return detector::ClassNms(found, .3f);
    };
    detector::TrainDetector(cfg, train, test, hooks);
    std::cout << "All FPN checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
