// CMD: just run-nn faster_rcnn
// Faster R-CNN: a region proposal network shares the backbone with the Fast R-CNN head.
//
// Port of tflow's faster_rcnn. Preparation (in ~/projects/tflow_examples):
// python3 scripts/fetch_datasets.py mnist --split both.
//
// On the synthetic digit scenes, replace the classical proposals of
// fast_rcnn with a learned region proposal network on the shared stride-4
// feature map: a 3x3 convolution (128 channels, ReLU) and two 1x1 layers
// scoring and refining 9 anchors per position (sizes 12, 20 and 30 pixels,
// height/width 0.5, 1 and 2; 2304 anchors). Anchors with IoU >= 0.7 (and each
// target's best anchor) are positive and below 0.3 negative; 128 anchors per
// image are sampled, at most half positive, for a sigmoid objectness loss
// and smooth-L1 box loss. Proposals are the 300 best decoded anchors, reduced
// by NMS at IoU 0.7 to 100 per image. Training is approximate joint training:
// each step computes proposals with the current RPN, samples 64 RoIs from
// them (a quarter foreground) and minimizes the RPN and head losses together.
// Check anchor matching on a fixture, then train with 5500 / 500 / 1000
// scenes, two images per step, Adam(.001), kernel L2(.0001) and the best
// validation AP50 checkpoint; report RPN proposal recall and test AP.
// https://arxiv.org/abs/1506.01497
//
// Eager port: each step runs the backbone and RPN once and takes the
// (detached) proposals from that same forward pass; tflow runs a separate
// proposal pass with the same, not yet updated, weights. Proposal recall is
// measured on the test scenes only (tflow's counter also pools the 500
// restored-validation scenes into its printed recall). Weights come from nn's
// generator, so trained numbers match TF statistically.
// Checkpoint (.npz) and history: output/nn/models/faster_rcnn_digits/.
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
constexpr int kClasses = 11, kBackground = 10, kStride = 4, kFeature = 16, kAnchorsPerCell = 9, kRois = 64;
const auto    kAnchors     = detector::AnchorGrid(kFeature, kFeature, kStride, {12, 20, 30}, {.5f, 1, 2});
const int     kAnchorCount = int(kAnchors.size());

void MatchingChecks() {
  // A target equal to an anchor is matched to it; distant anchors are negatives.
  const std::size_t index  = std::size_t((5 * kFeature + 7) * kAnchorsPerCell + 4);
  const Box         target = kAnchors[index];
  const auto        match  = detector::MatchAnchors(kAnchors, {target}, .7f, .3f);
  int               positives = 0, negatives = 0, ignored = 0;
  for (int m : match) {
    positives += m == 0;
    negatives += m == -1;
    ignored += m == -2;
  }
  check::Require(match[index] == 0 && positives >= 1 && positives + negatives + ignored == kAnchorCount, "anchor matching");
  const auto delta = detector::EncodeBox(target, target);
  check::Near(std::abs(delta[0]) + std::abs(delta[1]) + std::abs(delta[2]) + std::abs(delta[3]), 0, 1e-6,
              "zero deltas for a matching anchor");
  std::cout << "  anchors: " << kAnchorCount << " (16x16 cells x 3 sizes x 3 ratios); a target equal to one anchor has "
            << positives << " positive, " << ignored << " ignored and " << negatives << " negative anchors" << std::endl;
}

nn::Tensor RoiImages(const DetectionScenes& scenes, const std::vector<int>& rows) { return detector::SceneImages(scenes, rows); }
}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path            directory = paths::Dataset("mnist");
    detector::DetectorTrainingConfig cfg;
    cfg.name              = "faster_rcnn_digits";
    cfg.epochs            = 10;
    cfg.batch             = 2;
    cfg.validation        = 500;
    cfg.minimum_test_ap50 = .7;
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
    MatchingChecks();
    const auto [train, test] = detector::StandardScenes(directory);
    std::mt19937_64 random(7);
    auto forward = [](nn::ParameterStore& p, const nn::Tensor& images) {
      const auto features = detector::StrideFourBackbone(p, images);
      return std::make_pair(features, detector::BuildRpn(p, {{features, 64, kFeature, kAnchorsPerCell}}));
    };
    auto head = [](nn::ParameterStore& p) {
      return [&p](const nn::Tensor& features, const nn::Tensor& rois, const nn::Tensor& indices) {
        return detector::RoiHead(p, features, rois, indices, kClasses);
      };
    };
    detector::DetectorHooks hooks;
    hooks.build = [&](nn::ParameterStore& p, nn::ParameterStore&) {
      const auto [features, rpn] = forward(p, nn::zeros({1, 64, 64, 1}));
      detector::RoiHead(p, features, detector::RoiTensor({{0, 0, 1, 1}}), detector::IndexTensor({0}), kClasses);
    };
    hooks.loss = [&](nn::ParameterStore& p, nn::ParameterStore&, const DetectionScenes& scenes, const std::vector<int>& rows) {
      const auto [features, rpn] = forward(p, RoiImages(scenes, rows));
      std::vector<std::vector<Box>> proposals;
      {
        nn::NoGradGuard guard;
        proposals = detector::ComputeProposals(rpn, kAnchors, scenes.height, scenes.width);
      }
      detector::RoiBatch batch;
      for (std::size_t i = 0; i < rows.size(); ++i)
        detector::SampleRois(scenes, rows[i], int(i), proposals[i], kRois, kBackground, kStride, kFeature, random, batch);
      const auto targets = detector::MakeRpnTargets(kAnchors, scenes, rows, random);
      const auto out = detector::RoiHead(p, features, detector::RoiTensor(batch.rois), detector::IndexTensor(batch.indices), kClasses);
      return detector::RpnLoss(rpn, targets.labels, targets.weights, targets.targets, targets.positives) +
             detector::RoiLoss(out, batch, kClasses);
    };
    // The trainer evaluates validation once per epoch, again after restoring
    // the best checkpoint, then the test rows: measure RPN recall on that last call.
    const auto* test_scenes = &test;
    int    objects = 0;
    double recall50 = 0, recall70 = 0;
    hooks.detect = [&](nn::ParameterStore& p, nn::ParameterStore&, const DetectionScenes& scenes, const std::vector<int>& rows) {
      nn::NoGradGuard                  guard;
      const bool                       final_call = &scenes == test_scenes;
      std::vector<detector::Detection> dets;
      for (std::size_t start = 0; start < rows.size(); start += 32) {
        const std::vector<int> chunk(rows.begin() + std::ptrdiff_t(start), rows.begin() + std::ptrdiff_t(std::min(rows.size(), start + 32)));
        const auto [features, rpn] = forward(p, RoiImages(scenes, chunk));
        const auto proposals       = detector::ComputeProposals(rpn, kAnchors, scenes.height, scenes.width);
        std::vector<std::pair<int, Box>> boxes;
        for (std::size_t i = 0; i < chunk.size(); ++i) {
          for (const auto& b : proposals[i]) boxes.push_back({int(i), b});
          if (final_call)
            for (const auto& g : scenes.boxes[std::size_t(chunk[i])]) {
              float best = 0;
              for (const auto& b : proposals[i]) best = std::max(best, detector::BoxIoU(b, g));
              ++objects;
              recall50 += best >= .5f;
              recall70 += best >= .7f;
            }
        }
        const auto found = detector::DetectRois(head(p), features, chunk, boxes, kClasses, kStride, kFeature);
        dets.insert(dets.end(), found.begin(), found.end());
      }
      return detector::ClassNms(dets, .3f);
    };
    const auto result = detector::TrainDetector(cfg, train, test, hooks);
    std::cout << "  RPN proposals (at most 100 per test scene): recall " << std::fixed << std::setprecision(3)
              << recall50 / objects << " at IoU 0.5 and " << recall70 / objects << " at 0.7; test AP50 "
              << std::setprecision(4) << result.ap50 << std::defaultfloat << std::endl;
    check::Require(objects > 0 && std::isfinite(recall50 / objects), "finite RPN proposal recall on evaluated test objects");
    check::TrainingQuality(recall50 / objects > .9, "RPN proposal recall");
    std::cout << "All Faster R-CNN checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
