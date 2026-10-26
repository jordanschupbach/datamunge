// CMD: just run-nn fast_rcnn
// Fast R-CNN: one convolutional pass per image and RoI pooling of every proposal.
//
// Port of tflow's fast_rcnn. Preparation (in ~/projects/tflow_examples):
// python3 scripts/fetch_datasets.py mnist --split both.
//
// On the synthetic digit scenes with the same tight-biased proposals as rcnn
// (connected components at thresholds 48 to 230), compute a stride-4 feature
// map once per image (two 3x3 convolutions, pool, two 3x3 convolutions, pool)
// and pool every proposal from it: crop_and_resize to 14x14 bilinear samples,
// then 2x2 max pooling to the paper's 7x7 grid (RoI pooling's max over
// bins, with bilinear sampling in place of rounding), two 256-unit dense
// layers, a softmax over eleven classes and class-specific box deltas. As in
// the paper, each training image contributes 64 RoIs, a quarter foreground
// (IoU >= 0.5, ground-truth boxes included) and the rest background with IoU
// in [0.1, 0.5) where available, and the loss is cross-entropy plus smooth-L1
// box regression on foreground RoIs (multi-task, end to end). Check the
// RoI pooling geometry on a feature map holding each cell's coordinates.
// https://arxiv.org/abs/1504.08083
//
// Train with 5500 / 500 / 1000 scenes, two images (128 RoIs) per step,
// Adam(.001), kernel L2(.0001) and the best validation AP50 checkpoint, then
// report test AP and the per-image cost: one backbone pass instead of one per
// proposal. Training is eager (common/detection.hpp TrainDetector); weights
// come from nn's generator, so trained numbers match TF statistically.
// Checkpoint (.npz) and history: output/nn/models/fast_rcnn_digits/.
// Options: data=PATH epochs=10 threads=N.

#include <iomanip>
#include <iostream>

#include "common/two_stage.hpp"

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
constexpr int kClasses = 11, kBackground = 10, kStride = 4, kFeature = 16, kRois = 64;
std::vector<Box> Proposals(const DetectionScenes& scenes, int row) {
  return detector::ComponentProposals(scenes, row, {48, 64, 96, 128, 160, 200, 230}, 3);
}

void PoolingChecks() {
  std::vector<float> grid(kFeature * kFeature * 2);
  for (int r = 0; r < kFeature; ++r)
    for (int c = 0; c < kFeature; ++c) {
      grid[std::size_t((r * kFeature + c) * 2)]     = float(r);
      grid[std::size_t((r * kFeature + c) * 2 + 1)] = float(c);
    }
  // Pixels [8, 40) x [12, 36): feature rows 1.5..9.5 and columns 2.5..8.5 in cell-centre coordinates.
  const Box  box = {8, 12, 40, 36};
  const auto f   = detector::FeatureBox(box, kStride, kFeature);
  const auto pooled =
      nn::max_pool2d(nn::crop_and_resize(nn::from_vector(grid, {1, kFeature, kFeature, 2}), detector::RoiTensor({f}),
                                         detector::IndexTensor({0}), {14, 14}),
                     2, 2, nn::Padding::Valid)
          .to_vector<float>();
  // Bin (i, j) of 7 takes the maximum of two sample rows/columns; the larger
  // one sits at y0 + (2i + 1) * (y1 - y0) / 13 in feature coordinates.
  double      error = 0;
  const float y0 = (box[0] - 2) / kStride, y1 = (box[2] - 2) / kStride, x0 = (box[1] - 2) / kStride,
              x1 = (box[3] - 2) / kStride;
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 7; ++j) {
      error = std::max(error, double(std::abs(pooled[std::size_t((i * 7 + j) * 2)] - (y0 + (2 * i + 1) * (y1 - y0) / 13))));
      error = std::max(error, double(std::abs(pooled[std::size_t((i * 7 + j) * 2 + 1)] - (x0 + (2 * j + 1) * (x1 - x0) / 13))));
    }
  check::Near(error, 0, 1e-4, "RoI pooling samples the box on the feature grid");
  std::cout << "  RoI pooling of pixels [8,40)x[12,36) on the stride-4 grid: 7x7 bins sample feature rows " << y0 << ".."
            << y1 << " and columns " << x0 << ".." << x1 << " (error " << error << ")" << std::endl;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path            directory = paths::Dataset("mnist");
    detector::DetectorTrainingConfig cfg;
    cfg.name              = "fast_rcnn_digits";
    cfg.epochs            = 10;
    cfg.batch             = 2;
    cfg.validation        = 500;
    cfg.minimum_test_ap50 = .65;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      const auto        equal    = argument.find('=');
      if (equal == std::string::npos) throw std::invalid_argument("expected key=value option");
      const auto key = argument.substr(0, equal), value = argument.substr(equal + 1);
      if (key == "data") directory = value;
      else if (key == "epochs") cfg.epochs = std::stoi(value);
      else if (key == "threads") nn::set_num_threads(std::stoi(value));
      else throw std::invalid_argument("unknown option " + key);
    }
    PoolingChecks();
    const auto [train, test] = detector::StandardScenes(directory);
    std::mt19937_64         random(7);
    detector::DetectorHooks hooks;
    auto head = [](nn::ParameterStore& p) {
      return [&p](const nn::Tensor& features, const nn::Tensor& rois, const nn::Tensor& indices) {
        return detector::RoiHead(p, features, rois, indices, kClasses);
      };
    };
    hooks.build = [&](nn::ParameterStore& p, nn::ParameterStore&) {
      const auto features = detector::StrideFourBackbone(p, nn::zeros({1, 64, 64, 1}));
      detector::RoiHead(p, features, detector::RoiTensor({{0, 0, 1, 1}}), detector::IndexTensor({0}), kClasses);
    };
    hooks.loss = [&](nn::ParameterStore& p, nn::ParameterStore&, const DetectionScenes& scenes, const std::vector<int>& rows) {
      detector::RoiBatch batch;
      for (std::size_t i = 0; i < rows.size(); ++i)
        detector::SampleRois(scenes, rows[i], int(i), Proposals(scenes, rows[i]), kRois, kBackground, kStride, kFeature,
                             random, batch);
      const auto features = detector::StrideFourBackbone(p, detector::SceneImages(scenes, rows));
      const auto out = detector::RoiHead(p, features, detector::RoiTensor(batch.rois), detector::IndexTensor(batch.indices), kClasses);
      return detector::RoiLoss(out, batch, kClasses);
    };
    hooks.detect = [&](nn::ParameterStore& p, nn::ParameterStore&, const DetectionScenes& scenes, const std::vector<int>& rows) {
      nn::NoGradGuard                  guard;
      std::vector<detector::Detection> dets;
      for (std::size_t start = 0; start < rows.size(); start += 64) {
        const std::vector<int> chunk(rows.begin() + std::ptrdiff_t(start), rows.begin() + std::ptrdiff_t(std::min(rows.size(), start + 64)));
        std::vector<std::pair<int, Box>> boxes;
        for (std::size_t i = 0; i < chunk.size(); ++i)
          for (const auto& b : Proposals(scenes, chunk[i])) boxes.push_back({int(i), b});
        const auto features = detector::StrideFourBackbone(p, detector::SceneImages(scenes, chunk));
        const auto found    = detector::DetectRois(head(p), features, chunk, boxes, kClasses, kStride, kFeature);
        dets.insert(dets.end(), found.begin(), found.end());
      }
      return detector::ClassNms(dets, .3f);
    };
    const auto result    = detector::TrainDetector(cfg, train, test, hooks);
    double     proposals = 0;
    for (int row = 0; row < test.size(); ++row) proposals += double(Proposals(test, row).size());
    std::cout << "  per test scene: one backbone pass for " << std::fixed << std::setprecision(1) << proposals / test.size()
              << " proposals (R-CNN runs its CNN once per proposal); test AP50 " << std::setprecision(4) << result.ap50
              << std::defaultfloat << std::endl;
    std::cout << "All Fast R-CNN checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
