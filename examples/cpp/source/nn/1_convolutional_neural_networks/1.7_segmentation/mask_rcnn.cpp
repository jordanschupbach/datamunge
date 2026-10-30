// CMD: just run-nn mask_rcnn
// Mask R-CNN: learned RPN, aligned box head and class-specific mask branch.
// https://arxiv.org/abs/1703.06870
// Preparation (tflow): dm::datasets::fetch("mnist")
//
// Port of tflow's mask_rcnn. Unquantized aligned RoI features use four
// regular bilinear samples per bin, averaged with replicated borders.
// Independently check fractional/border interpolation and all feature
// gradients; verify only the matched mask class receives sigmoid-BCE
// gradients. Full256-channel four3x3/deconv mask head derivatives; compact
// 32-channel derivatives.28px masks.
//
// Compact stride4 CNN/RPN with2304anchors and approximate joint training;
// seven-bin aligned256-unit box head,14-bin aligned four-conv32 mask head.
// Each step samples64RoIs/image,at most1/4foreground; only positives train
// their matched class mask. Training targets are individual MNIST support
// masks (>32),not union foreground. No box overlap in these teaching scenes.
// Default128fit/32validation/64test,Adam(.001),batch2,L2 .0001,3epochs.
// Keep every learned branch and best-validation boxAP50 checkpoint. This is
// an early-learning demonstration, not an instance segmentation benchmark.
// After restore report boxAP and class-specific pasted-mask integral AP50/75,
// plus mask AP averaged over IoU.50:.95 and image/truth/predicted-mask PNG.
// Default accuracy=0 verifies finite/decreasing training loss and finite AP.
// accuracy=1 restores5500/500/1000 scene defaults and the original boxAP50 .7
// and maskAP50 .6 targets; use epochs=10 to request the previous duration.
//
// TF -> datamunge: eager training; the RPN proposals of a step come from the
// same forward pass that is differentiated (TF runs the RPN once more before
// the step, with the same weights, so the proposals are the same). MNIST
// comes from tflow's dataset cache (data=PATH overrides); the scenes and their
// instance supports share the tensor sampler with plain scenes. Borderline
// interpolation rounding may differ by one byte from the original host sampler. Weights come from datamunge's
// generator, so trained numbers match TF statistically. The full mask head is
// executed (TF builds it as a graph only).
// Options: data=PATH epochs=3 fit=N validation=N test=N accuracy=0|1 threads=N.
// Artifacts: output/nn/models/mask_rcnn_digits/.

#include "common/mask_rcnn.hpp"

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/instance_segmentation.hpp"
#include "common/paths.hpp"
#include "common/two_stage.hpp"

namespace nn = datamunge::nn;
using detector::Box;
using detector::Detection;
using nn::Tensor;

namespace {

constexpr int kClasses = 10, kCells = 16, kStride = 4;

void AlignChecks() {
  std::vector<float> values(25);
  for (int i = 0; i < 25; ++i)
    values[std::size_t(i)] = std::sin(.3f * i);
  Tensor input = nn::from_vector(values, {1, 5, 5, 1});
  input.requires_grad_(true);
  const std::vector<Box> boxes    = {{0, 0, 20, 20}, {2.3f, 3.1f, 15.7f, 18.6f}};
  const Tensor           aligned  = maskrcnn::RoiAlign(input, boxes, {0, 0}, 5, 4, 3);
  const auto             gradient = nn::grad(nn::sum(aligned), {input})[0].to_vector<float>();
  const auto             out      = aligned.to_vector<float>();
  std::vector<double>    derivatives(25, 0);
  double                 error = 0;
  for (int b = 0; b < 2; ++b)
    for (int y = 0; y < 3; ++y)
      for (int x = 0; x < 3; ++x) {
        double     expected = 0;
        const Box& box      = boxes[std::size_t(b)];
        for (int a = 0; a < 2; ++a)
          for (int z = 0; z < 2; ++z) {
            const double sy = std::clamp(
                             (box[0] + (y + (a + .5) / 2) / 3 * (box[2] - box[0])) / 4 - .5,
                             0.,
                             4.),
                         sx = std::clamp(
                             (box[1] + (x + (z + .5) / 2) / 3 * (box[3] - box[1])) / 4 - .5,
                             0.,
                             4.);
            const int iy = int(sy), ix = int(sx);
            for (int dy = 0; dy < 2; ++dy)
              for (int dx = 0; dx < 2; ++dx) {
                const int    i = std::min(iy + dy, 4) * 5 + std::min(ix + dx, 4);
                const double w =
                    (dy ? sy - iy : 1 - (sy - iy)) * (dx ? sx - ix : 1 - (sx - ix)) * .25;
                expected += w * values[std::size_t(i)];
                derivatives[std::size_t(i)] += w;
              }
          }
        error = std::max(error, std::abs(expected - out[std::size_t((b * 3 + y) * 3 + x)]));
      }
  for (int i = 0; i < 25; ++i)
    error = std::max(error, std::abs(derivatives[std::size_t(i)] - gradient[std::size_t(i)]));
  check::Near(error, 0, 3e-6, "fractional RoIAlign every value and feature derivative");
  std::cout << "  aligned2x2 samples/bin: independent border/fractional value/gradient error="
            << error << std::endl;
}

void MaskChecks() {
  {
    Tensor logits = nn::zeros({2, 2, 2, 3});
    logits.requires_grad_(true);
    const std::vector<float> targets = {0, 1, 0, 1, 1, 1, 0, 0};
    const Tensor             loss =
        maskrcnn::SelectedMaskLoss(logits, {0, 2}, nn::from_vector(targets, {2, 2, 2}), 3);
    const auto g = nn::grad(loss, {logits})[0].to_vector<float>();
    check::Near(loss.item<double>(), std::log(2.), 1e-7, "zero-logit mask BCE");
    for (int n = 0; n < 2; ++n)
      for (int pixel = 0; pixel < 4; ++pixel)
        for (int c = 0; c < 3; ++c)
          check::Near(
              g[std::size_t((n * 4 + pixel) * 3 + c)],
              c == (n ? 2 : 0) ? (.5 - targets[std::size_t(n * 4 + pixel)]) / 8 : 0,
              1e-7,
              "only matched class gets mask gradient");
  }
  for (bool full : {true, false}) {
    nn::ParameterStore p(17);
    const int          in = full ? 256 : 64;
    std::vector<float> values(std::size_t(2 * 14 * 14 * in));
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = std::sin(.13f * float(i));
    const Tensor output = maskrcnn::MaskHead(
        p, nn::from_vector(values, {2, 14, 14, in}), full ? 256 : 32, full ? 80 : 10);
    const auto grads = nn::grad(nn::mean(nn::square(output)), p.parameters());
    check::Require(grads.size() == 12, "four conv,upconv,score parameters all connected");
    check::Require(
        output.shape() == nn::Shape({2, 28, 28, full ? 80 : 10}),
        "learned upconv produces28px masks");
    for (const auto& g : grads) {
      const double norm = nn::sum(nn::square(g)).item<double>();
      check::Require(
          std::isfinite(norm) && (full || norm > 0), "finite nonzero mask parameter derivative");
    }
    std::cout << "  " << (full ? "full" : "compact") << " mask head: " << p.count()
              << " parameters, " << grads.size() << " derivatives" << std::endl;
  }
  instance::InstanceScenes scene;
  scene.height = scene.width                       = 2;
  scene.boxes                                      = {{{0, 0, 2, 2}}};
  scene.labels                                     = {{0}};
  scene.instance_masks                             = {{{1, 0, 0, 1}}};
  std::vector<instance::MaskDetection> predictions = {
      {{0, 0, .9f, {0, 0, 2, 2}}, {1, 0, 0, 1}}, {{0, 0, .8f, {0, 0, 2, 2}}, {1, 0, 0, 1}}};
  check::Near(
      instance::MaskAveragePrecision(predictions, scene, 1, .5),
      1,
      0,
      "duplicate after TP leaves AP1");
  predictions[0].mask = {0, 1, 1, 0};
  check::Near(
      instance::MaskAveragePrecision(predictions, scene, 1, .5),
      .5,
      0,
      "higher-scoring disjoint mask reduces AP");
  check::Near(instance::MaskIoU({1, 0}, {1, 1}), .5, 0, "binary support IoU");
  std::cout << "  selected-class BCE,wrong-class zeros,maskIoU and scored AP fixtures passed"
            << std::endl;
}

/// Compact Mask R-CNN: stride-4 backbone, one-level RPN, aligned box and mask heads.
struct Model {
  nn::ParameterStore& p;
  Tensor              features;
  detector::Rpn       rpn;
  Model(nn::ParameterStore& params, const Tensor& images) : p(params) {
    features = detector::StrideFourBackbone(p, images);
    rpn      = detector::BuildRpn(p, {{features, 64, kCells, 9}});
  }
  detector::RoiHeadOutputs Head(const std::vector<Box>& rois, const std::vector<int>& indices)
      const {
    return maskrcnn::PooledRoiHead(
        p, maskrcnn::RoiAlign(features, rois, indices, kCells, kStride, 7), 11);
  }
  Tensor MaskLogits(const std::vector<Box>& boxes, const std::vector<int>& indices) const {
    return maskrcnn::MaskHead(
        p, maskrcnn::RoiAlign(features, boxes, indices, kCells, kStride, 14), 32, kClasses);
  }
};

} // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3, fit_count = -1, validation_count = -1, test_count = -1;
    bool                  accuracy  = false;
    std::filesystem::path directory = paths::Dataset("mnist");
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("epochs=", 0) == 0)
        epochs = std::stoi(arg.substr(7));
      else if (arg.rfind("fit=", 0) == 0)
        fit_count = std::stoi(arg.substr(4));
      else if (arg.rfind("validation=", 0) == 0)
        validation_count = std::stoi(arg.substr(11));
      else if (arg.rfind("test=", 0) == 0)
        test_count = std::stoi(arg.substr(5));
      else if (arg == "accuracy=0" || arg == "accuracy=1")
        accuracy = arg == "accuracy=1";
      else if (arg.rfind("data=", 0) == 0)
        directory = arg.substr(5);
      else if (arg.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(arg.substr(8)));
      else
        throw std::invalid_argument(
            "expected data=PATH, epochs=N, fit=N, validation=N, test=N, accuracy=0|1 or threads=N");
    }
    if (fit_count == -1)
      fit_count = accuracy ? 5500 : 128;
    if (validation_count == -1)
      validation_count = accuracy ? 500 : 32;
    if (test_count == -1)
      test_count = accuracy ? 1000 : 64;
    check::Require(
        fit_count > 0 && validation_count > 0 && validation_count <= 100000 && test_count > 0 &&
            test_count <= 100000 && fit_count <= 100000 - validation_count,
        "positive bounded fit/validation/test counts");
    check::Require(epochs > 0 && (accuracy || epochs >= 2), "valid Mask R-CNN training duration");
    AlignChecks();
    MaskChecks();
    const auto anchors = detector::AnchorGrid(kCells, kCells, kStride, {12, 20, 30}, {.5f, 1, 2});
    detector::SceneConfig train_cfg;
    train_cfg.scenes      = fit_count + validation_count;
    train_cfg.seed        = 11;
    train_cfg.max_overlap = 0;
    auto test_cfg         = train_cfg;
    test_cfg.scenes       = test_count;
    test_cfg.seed         = 22;
    const auto train =
        instance::MakeInstanceScenes(detector::LoadDigits(directory, "train"), train_cfg);
    const auto test =
        instance::MakeInstanceScenes(detector::LoadDigits(directory, "t10k"), test_cfg);
    std::mt19937_64 random(7);
    std::size_t     mask_training_rois = 0;
    const auto      instances =
        [](const detector::DetectionScenes& scenes) -> const instance::InstanceScenes& {
      return static_cast<const instance::InstanceScenes&>(scenes);
    };
    detector::DetectorHooks hooks;
    hooks.build = [&](nn::ParameterStore& p, nn::ParameterStore&) {
      const Model net(p, detector::SceneImages(train, {0}));
      net.Head({{0, 0, 16, 16}}, {0});
      net.MaskLogits({{0, 0, 16, 16}}, {0});
    };
    hooks.loss = [&](nn::ParameterStore& p,
                     nn::ParameterStore&,
                     const detector::DetectionScenes& scenes,
                     const std::vector<int>&          rows) {
      const Model net(p, detector::SceneImages(scenes, rows));
      const auto  candidates =
          detector::ComputeProposals(net.rpn, anchors, scenes.height, scenes.width);
      detector::RoiBatch batch;
      for (std::size_t i = 0; i < rows.size(); ++i)
        detector::SampleRois(scenes, rows[i], int(i), candidates[i], 64, 10, 4, 16, random, batch);
      const auto targets  = detector::MakeRpnTargets(anchors, scenes, rows, random);
      const auto rpn_loss = detector::RpnLoss(
          net.rpn, targets.labels, targets.weights, targets.targets, targets.positives);
      const auto head  = net.Head(batch.pixels, batch.indices);
      const auto masks = maskrcnn::PositiveMaskTargets(instances(scenes), rows, batch);
      check::Require(!masks.boxes.empty(), "ground truth supplies positive mask RoIs");
      mask_training_rois += masks.boxes.size();
      const auto mask_loss = maskrcnn::SelectedMaskLoss(
          net.MaskLogits(masks.boxes, masks.indices), masks.labels, masks.targets, kClasses);
      return rpn_loss + detector::RoiLoss(head, batch, 11) + mask_loss;
    };
    hooks.detect = [&](nn::ParameterStore& p,
                       nn::ParameterStore&,
                       const detector::DetectionScenes& scenes,
                       const std::vector<int>&          rows) {
      nn::NoGradGuard        guard;
      std::vector<Detection> found;
      for (std::size_t start = 0; start < rows.size(); start += 32) {
        const std::vector<int> chunk(
            rows.begin() + std::ptrdiff_t(start),
            rows.begin() + std::ptrdiff_t(std::min(rows.size(), start + 32)));
        const Model net(p, detector::SceneImages(scenes, chunk));
        const auto  proposed =
            detector::ComputeProposals(net.rpn, anchors, scenes.height, scenes.width);
        std::vector<std::pair<int, Box>> boxes;
        for (std::size_t i = 0; i < chunk.size(); ++i)
          for (const auto& b : proposed[i])
            boxes.push_back({int(i), b});
        const auto detected = maskrcnn::DetectPixelRois(
            [&](const std::vector<Box>& r, const std::vector<int>& ix) { return net.Head(r, ix); },
            chunk,
            boxes,
            11);
        found.insert(found.end(), detected.begin(), detected.end());
      }
      return detector::ClassNms(found, .3f);
    };
    hooks.inspect = [&](nn::ParameterStore&              p,
                        nn::ParameterStore&              state,
                        const detector::DetectionScenes& scenes,
                        const std::vector<int>&          rows,
                        const std::filesystem::path&     outdir) {
      nn::NoGradGuard                      guard;
      const auto&                          inst = instances(scenes);
      std::vector<instance::MaskDetection> detections;
      for (std::size_t start = 0; start < rows.size(); start += 16) {
        const std::vector<int> chunk(
            rows.begin() + std::ptrdiff_t(start),
            rows.begin() + std::ptrdiff_t(std::min(rows.size(), start + 16)));
        std::vector<Detection> selected;
        for (const auto& b : hooks.detect(p, state, scenes, chunk))
          if (b.score > .05f)
            selected.push_back(b);
        if (selected.empty())
          continue;
        std::vector<Box> rois;
        std::vector<int> indices;
        for (const auto& d : selected) {
          rois.push_back(d.box);
          indices.push_back(int(std::find(chunk.begin(), chunk.end(), d.image) - chunk.begin()));
        }
        const Model net(p, detector::SceneImages(scenes, chunk));
        const auto  probabilities = nn::sigmoid(net.MaskLogits(rois, indices)).to_vector<float>();
        for (std::size_t i = 0; i < selected.size(); ++i)
          detections.push_back(
              {selected[i],
               instance::PasteMask(
                   probabilities.data() + i * 28 * 28 * kClasses,
                   28,
                   kClasses,
                   selected[i].label,
                   selected[i].box,
                   64,
                   64)});
      }
      const double ap50  = instance::MaskAveragePrecision(detections, inst, kClasses, .5),
                   ap75  = instance::MaskAveragePrecision(detections, inst, kClasses, .75);
      double        mean = 0;
      std::ofstream report(outdir / "mask_ap.tsv");
      report << "iou\tintegral_mask_ap\n";
      for (int i = 0; i < 10; ++i) {
        const double t  = .5 + .05 * i,
                     ap = instance::MaskAveragePrecision(detections, inst, kClasses, t);
        check::Require(std::isfinite(ap) && ap >= 0 && ap <= 1, "finite mask AP in [0,1]");
        mean += ap / 10;
        report << t << '\t' << ap << '\n';
      }
      std::cout << "  held-out predicted-box mask AP50=" << ap50 << ", AP75=" << ap75
                << ", AP50:95=" << mean << "; " << detections.size() << " scored masks, all "
                << scenes.size() << " test images" << std::endl;
      check::Require(
          std::isfinite(ap50) && std::isfinite(ap75) && std::isfinite(mean), "finite mask metrics");
      instance::ExportInstanceMasks(inst, detections, outdir / "mask_preview.png");
      if (accuracy)
        check::Require(ap50 >= .6, "instance-mask AP50 demonstration threshold");
    };
    detector::DetectorTrainingConfig cfg;
    cfg.name              = "mask_rcnn_digits";
    cfg.epochs            = epochs;
    cfg.accuracy          = accuracy;
    cfg.batch             = 2;
    cfg.validation        = validation_count;
    cfg.minimum_test_ap50 = .7;
    {
      using namespace detector;
      auto training_cfg    = cfg;
      bool explicit_epochs = false;
      for (int option = 1; option < argc; ++option)
        explicit_epochs = explicit_epochs || std::string(argv[option]).rfind("epochs=", 0) == 0;
      if (!explicit_epochs)
        training_cfg.epochs = example_runtime::Limit(training_cfg.epochs, 1);
      training_cfg.batch      = example_runtime::Limit(training_cfg.batch, 2);
      training_cfg.validation = example_runtime::Limit(training_cfg.validation, 8);
      if (!example_runtime::Full())
        training_cfg.fit = training_cfg.fit ? std::min(training_cfg.fit, 8) : 8;
      example_runtime::Describe();
      check::Require(
          training_cfg.validation > 0 && training_cfg.validation < train.size() &&
              training_cfg.epochs > 0 && training_cfg.batch > 0,
          "valid detector training configuration");
      check::Require(
          !example_runtime::Full() || training_cfg.accuracy || training_cfg.epochs >= 2,
          "short learning demonstration needs at least two epochs");
      const int        fit_count = training_cfg.fit > 0
                                       ? std::min(training_cfg.fit, train.size() - training_cfg.validation)
                                       : train.size() - training_cfg.validation;
      std::vector<int> fit(static_cast<std::size_t>(fit_count)),
          validation(static_cast<std::size_t>(training_cfg.validation)),
          test_rows(example_runtime::Limit<std::size_t>(test.size(), 8));
      std::iota(fit.begin(), fit.end(), 0);
      std::iota(validation.begin(), validation.end(), train.size() - training_cfg.validation);
      std::iota(test_rows.begin(), test_rows.end(), 0);
      nn::ParameterStore params(17), state(17);
      {
        nn::NoGradGuard guard;
        hooks.build(params, state);
      }
      std::vector<nn::Tensor> kernels;
      for (const auto& p : params.parameters())
        if (p.dim() >= 2)
          kernels.push_back(p);
      const auto directory = TrainingDirectory(training_cfg.name);
      std::filesystem::create_directories(directory);
      const auto prefix = (directory / "best").string();
      if (!training_cfg.warm_start.empty()) {
        check::Require(
            std::filesystem::absolute(training_cfg.warm_start).lexically_normal() !=
                std::filesystem::absolute(prefix).lexically_normal(),
            "warm start requires a separate output directory/run tag");
        RestoreCheckpoint(params, state, training_cfg.warm_start);
        std::cout << "  warm start: " << training_cfg.warm_start << "; fresh Adam moments"
                  << std::endl;
      }
      nn::OptimizerConfig optimizer_config;
      optimizer_config.kind = nn::OptimizerKind::Adam;
      nn::Optimizer optimizer(params, training_cfg.learning_rate, optimizer_config);
      std::ofstream history(directory / "history.tsv");
      history << "epoch\ttraining_loss\tvalidation_ap50\tvalidation_coco_ap\n";
      std::cout << "== " << training_cfg.name << ": " << fit_count << " fit / "
                << training_cfg.validation << " validation / " << test_rows.size()
                << " test scenes (" << train.height << "x" << train.width
                << ")\n  parameters=" << params.count() << "; batch=" << training_cfg.batch
                << "; lr=" << training_cfg.learning_rate << std::endl;
      nn::BatchOrder      batches(std::size_t(fit_count), std::size_t(training_cfg.batch), 29);
      double              best       = -1;
      int                 best_epoch = -1;
      std::vector<double> epoch_losses;
      if (!training_cfg.warm_start.empty()) {
        const auto baseline = EvaluateDetections(
            hooks.detect(params, state, train, validation),
            train,
            validation,
            training_cfg.classes);
        best       = baseline.ap50;
        best_epoch = 0;
        SaveCheckpoint(params, state, prefix);
        history << 0 << "\tnan\t" << baseline.ap50 << '\t' << baseline.coco << '\n';
        history.flush();
        std::cout << "  restored baseline validation " << ApSummary(baseline) << std::endl;
      }
      for (int epoch = 1; epoch <= training_cfg.epochs; ++epoch) {
        batches.begin_epoch();
        double      total = 0;
        std::size_t seen  = 0;
        for (;;) {
          const auto picked = batches.next();
          if (picked.empty())
            break;
          std::vector<int> rows;
          for (auto k : picked)
            rows.push_back(fit[std::size_t(k)]);
          nn::Tensor   objective  = hooks.loss(params, state, train, rows);
          const double batch_loss = objective.item();
          check::Require(std::isfinite(batch_loss), "finite detector batch loss");
          if (training_cfg.l2 > 0 && !kernels.empty()) {
            nn::Tensor penalty = nn::sum(nn::square(kernels[0]));
            for (std::size_t i = 1; i < kernels.size(); ++i)
              penalty = penalty + nn::sum(nn::square(kernels[i]));
            objective = objective + 0.5 * training_cfg.l2 * penalty;
          }
          optimizer.minimize(objective);
          total += batch_loss * double(rows.size());
          seen += rows.size();
        }
        epoch_losses.push_back(total / double(seen));
        check::Require(std::isfinite(epoch_losses.back()), "finite detector epoch mean");
        const auto result = EvaluateDetections(
            hooks.detect(params, state, train, validation),
            train,
            validation,
            training_cfg.classes);
        check::Require(std::isfinite(result.ap50), "finite validation AP50");
        if (result.ap50 > best) {
          best       = result.ap50;
          best_epoch = epoch;
          SaveCheckpoint(params, state, prefix);
        }
        history << epoch << '\t' << total / double(seen) << '\t' << result.ap50 << '\t'
                << result.coco << '\n';
        history.flush();
        std::cout << "  epoch " << epoch << ": training loss=" << total / double(seen)
                  << ", validation " << ApSummary(result) << std::endl;
      }
      RestoreCheckpoint(params, state, prefix);
      const auto restored = EvaluateDetections(
          hooks.detect(params, state, train, validation), train, validation, training_cfg.classes);
      check::Near(
          restored.ap50, best, 1e-6, "restored checkpoint reproduces selected validation AP50");
      std::cout << "  restored validation AP50=" << restored.ap50
                << "; checkpoint agreement verified" << std::endl;
      const auto result = EvaluateDetections(
          hooks.detect(params, state, test, test_rows), test, test_rows, training_cfg.classes);
      std::cout << "  selected epoch=" << best_epoch << "; test " << ApSummary(result)
                << "\n  per-class test AP50:";
      for (double ap : result.per_class50)
        std::cout << ' ' << std::fixed << std::setprecision(3) << ap;
      std::cout << std::defaultfloat << std::setprecision(6)
                << "\n  best checkpoint and history: " << directory.string() << std::endl;
      if (hooks.inspect)
        hooks.inspect(params, state, test, test_rows, directory);
      if (example_runtime::Full())
        CheckDetectorDemonstration(training_cfg, epoch_losses, result);
      else
        check::Require(std::isfinite(result.ap50), "finite quick detector evaluation");
      if (example_runtime::Full() && !training_cfg.accuracy)
        std::cout << "  short learning demonstration passed: mean loss " << epoch_losses.front()
                  << " -> " << epoch_losses.back()
                  << "; accuracy benchmark not requested (accuracy=1 enables AP50 gate "
                  << training_cfg.minimum_test_ap50 << ")" << std::endl;
    }
    std::cout << "  positive matched mask training RoIs=" << mask_training_rois
              << "\nAll Mask R-CNN checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
