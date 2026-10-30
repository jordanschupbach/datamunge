// CMD: just run-nn ssd
// SSD: multiscale single-shot detection with hard negative mining.
//
// Port of tflow's ssd. Preparation (in ~/projects/tflow_examples):
// dm::datasets::fetch("mnist").
//
// Build full VGG-16 SSD300: normalized conv4_3, dilated fc6, fc7, four
// extra blocks and six independent classification/regression heads. Check
// 26,285,486 parameters (21 classes including background), feature grids
// 38/19/10/5/3/1, all 8732 predictions and every parameter derivative.
// Independently check default-box geometry, collision-free forced matches,
// per-image 3:1 hard-negative selection, the complete loss and its analytic
// logits/box derivatives. An empty-positive batch must have zero loss and
// zero gradients. Check learned channel L2 rescaling on a 3-4-5 fixture.
// https://arxiv.org/abs/1512.02325
// https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py
//
// Train a compact six-level SSD on 64-pixel MNIST digit scenes, with grids
// 16/8/4/2/1/1 and default-box sizes 8/16/24/32/48/64 plus geometric-mean
// square boxes. This backbone is an explicit small-image adaptation; the
// full VGG model is checked separately. Use 5500 fit / 500 validation /
// 1000 test scenes, twelve epochs of Adam(.001), batch 32, kernel L2 .0001,
// validation AP50 checkpoint selection and separate test evaluation.
//
// Eager port (common/ssd.hpp, common/detection.hpp TrainDetector): the full
// SSD300 forward/backward pass runs eagerly at 300 pixels. Weights come from
// nn's generator, so trained numbers match TF statistically.
// Checkpoint (.npz) and history: output/nn/models/ssd_digits/.
// Options: data=PATH epochs=12 threads=N.

#include "common/ssd.hpp"

#include <iostream>

namespace nn = datamunge::nn;
using detector::Box;
using detector::DetectionScenes;

namespace {
void DefaultBoxChecks() {
  const auto boxes = ssd::SsdDefaultBoxes(
      {38, 19, 10, 5, 3, 1},
      {21, 45, 99, 153, 207, 261, 315},
      {8, 16, 32, 64, 100, 300},
      {4, 6, 6, 6, 4, 4});
  check::Require(boxes.size() == 8732, "SSD300 has 8732 default boxes");
  check::Near((boxes[0][0] + boxes[0][2]) / 2, 4, 1e-6, "first default-box centre");
  check::Near(boxes[0][2] - boxes[0][0], 21, 1e-6, "first square default-box side");
  check::Near(boxes[1][2] - boxes[1][0], std::sqrt(21. * 45), 2e-6, "geometric-mean square side");
  check::Near(
      (boxes[2][3] - boxes[2][1]) / (boxes[2][2] - boxes[2][0]),
      2,
      1e-6,
      "wide default-box aspect ratio");
  check::Near(
      (boxes[3][3] - boxes[3][1]) / (boxes[3][2] - boxes[3][0]),
      .5,
      1e-6,
      "tall default-box aspect ratio");
  // Both targets prefer the first anchor, so forced matching must resolve the collision.
  const std::vector<Box> anchors = {
      {0, 0, 10, 10}, {0, 1, 10, 11}, {30, 30, 40, 40}, {50, 50, 60, 60}};
  const auto match = detector::MatchSsd(anchors, {{0, 0, 10, 10}, {0, 0, 10, 10}});
  check::Require(
      match[0] == 0 && match[1] == 1 && match[2] == -1 && match[3] == -1,
      "distinct forced matches and background assignment");
  check::Require(
      detector::MatchSsd(anchors, {}) == std::vector<int>(4, -1), "empty target matching");
  std::cout << "  8732 default boxes: square/geometric-mean/reciprocal ratios checked; "
               "forced-match collisions resolved"
            << std::endl;
}

void NormalizationChecks() {
  nn::ParameterStore p(17);
  const auto         y =
      ssd::SsdNormalize(p, "fixture", nn::from_vector(std::vector<float>{3, 4, 0, 0}, {1, 1, 2, 2}))
          .to_vector<float>();
  const float expected[] = {12, 16, 0, 0};
  for (std::size_t i = 0; i < 4; ++i)
    check::Near(y[i], expected[i], 1e-6, "SSD L2 channel normalization");
  std::cout << "  L2 rescaling: [3,4] -> [12,16], zero vector stays finite and zero" << std::endl;
}

void LossChecks() {
  constexpr int      count = 8, classes = 3;
  std::vector<float> x(2 * count * classes), d(2 * count * 4, .3f), targets(2 * count * 4, 0);
  std::vector<int>   labels(2 * count, 2);
  labels[0]     = 0;
  labels[count] = labels[count + 1] = 1;
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] = .8f * std::sin(.7f * float(i));
  auto run = [&](std::vector<double>& gl, std::vector<double>& gd, std::vector<char>& negatives) {
    auto       logits = nn::from_vector(x, {2, count, classes}).requires_grad_();
    auto       deltas = nn::from_vector(d, {2, count, 4}).requires_grad_();
    const auto out    = ssd::SsdLoss(logits, deltas, labels, targets, classes);
    const auto g      = nn::grad(out.loss, {logits, deltas});
    const auto a = g[0].to_vector<float>(), b = g[1].to_vector<float>();
    gl.assign(a.begin(), a.end());
    gd.assign(b.begin(), b.end());
    negatives = out.selected_negatives;
    return out.loss.item();
  };
  std::vector<double> gl, gd;
  std::vector<char>   negatives;
  const double        loss          = run(gl, gd, negatives);
  double              expected_loss = 0, gradient_error = 0;
  for (int row = 0; row < 2; ++row) {
    std::vector<double>                ce(count);
    std::vector<std::array<double, 3>> probability(count);
    std::vector<int>                   negative_list;
    int                                positives = 0;
    for (int a = 0; a < count; ++a) {
      double total = 0;
      for (int c = 0; c < classes; ++c) {
        probability[std::size_t(a)][std::size_t(c)] =
            std::exp(double(x[std::size_t((row * count + a) * classes + c)]));
        total += probability[std::size_t(a)][std::size_t(c)];
      }
      for (int c = 0; c < classes; ++c)
        probability[std::size_t(a)][std::size_t(c)] /= total;
      const int label    = labels[std::size_t(row * count + a)];
      ce[std::size_t(a)] = -std::log(probability[std::size_t(a)][std::size_t(label)]);
      if (label == 2)
        negative_list.push_back(a);
      else
        ++positives;
    }
    std::stable_sort(negative_list.begin(), negative_list.end(), [&](int a, int b) {
      return ce[std::size_t(a)] > ce[std::size_t(b)];
    });
    negative_list.resize(std::min<std::size_t>(negative_list.size(), std::size_t(3 * positives)));
    for (int a = 0; a < count; ++a) {
      const int  label    = labels[std::size_t(row * count + a)];
      const bool positive = label != 2;
      const bool negative =
          std::find(negative_list.begin(), negative_list.end(), a) != negative_list.end();
      check::Require(
          bool(negatives[std::size_t(row * count + a)]) == negative,
          "per-image hardest negatives, capped at 3:1");
      if (positive || negative)
        expected_loss += ce[std::size_t(a)] / 3;
      if (positive)
        expected_loss += 4 * .5 * double(.3f) * double(.3f) / 3;
      for (int c = 0; c < classes; ++c) {
        const double expected =
            positive || negative ? (probability[std::size_t(a)][std::size_t(c)] - (c == label)) / 3
                                 : 0;
        gradient_error = std::max(
            gradient_error, std::abs(gl[std::size_t((row * count + a) * classes + c)] - expected));
      }
      for (int c = 0; c < 4; ++c)
        gradient_error = std::max(
            gradient_error,
            std::abs(
                gd[std::size_t((row * count + a) * 4 + c)] - (positive ? double(.3f) / 3 : 0)));
    }
  }
  check::Near(loss, expected_loss, 1e-5, "independent multibox loss");
  check::Near(gradient_error, 0, 1e-6, "analytic softmax and smooth-L1 gradients");
  std::fill(labels.begin(), labels.end(), 2);
  check::Near(run(gl, gd, negatives), 0, 0, "zero loss without positives");
  for (double g : gl)
    check::Near(g, 0, 0, "zero gradient without positives");
  for (double g : gd)
    check::Near(g, 0, 0, "zero gradient without positives");
  std::cout << "  per-image hard negatives 3/6 for 1/2 positives; loss matches host; analytic "
               "derivative error="
            << gradient_error << "; empty-positive loss and gradients exact zero" << std::endl;
}

void FullModelChecks() {
  nn::ParameterStore p(17);
  std::vector<float> x(300 * 300 * 3);
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] = std::sin(.13f * float(i));
  const auto net = ssd::SSD(p, nn::from_vector(x, {1, 300, 300, 3}));
  check::Require(
      p.count() == 26285486 && net.anchors == 8732, "full SSD300 parameter/default-box counts");
  for (std::size_t i = 0; i < 6; ++i) {
    const auto& s = net.features[i].shape();
    check::Require(
        s[0] == 1 && s[1] == net.cells[i] && s[2] == net.cells[i] && s[3] == net.channels[i],
        "canonical SSD300 feature dimensions");
  }
  check::Require(
      net.logits.shape() == nn::Shape{1, 8732, 21} && net.deltas.shape() == nn::Shape{1, 8732, 4},
      "SSD300 predictions");
  const auto objective = nn::mean(nn::square(net.logits)) + nn::mean(nn::square(net.deltas));
  const auto gradients = nn::grad(objective, p.parameters());
  double     minimum = INFINITY, maximum = 0;
  for (std::size_t i = 0; i < gradients.size(); ++i) {
    const double norm = std::sqrt(nn::sum(nn::square(gradients[i])).item());
    check::Require(
        std::isfinite(norm) && norm > 0, "finite nonzero SSD300 derivative: " + p.names()[i]);
    minimum = std::min(minimum, norm);
    maximum = std::max(maximum, norm);
  }
  std::cout
      << "  SSD300: " << p.count()
      << " parameters, grids 38/19/10/5/3/1, 8732 predictions; every derivative nonzero, norms ["
      << minimum << ',' << maximum << "]" << std::endl;
}
} // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path            directory = paths::Dataset("mnist");
    detector::DetectorTrainingConfig cfg;
    cfg.name              = "ssd_digits";
    cfg.epochs            = 12;
    cfg.minimum_test_ap50 = .7;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto        eq  = arg.find('=');
      if (eq == std::string::npos)
        throw std::invalid_argument("expected key=value option");
      const auto key = arg.substr(0, eq), value = arg.substr(eq + 1);
      if (key == "data")
        directory = value;
      else if (key == "epochs")
        cfg.epochs = std::stoi(value);
      else if (key == "threads")
        nn::set_num_threads(std::stoi(value));
      else
        throw std::invalid_argument("unknown option " + key);
    }
    DefaultBoxChecks();
    NormalizationChecks();
    LossChecks();
    FullModelChecks();
    const auto [train, test] = detector::StandardScenes(directory);
    const auto anchors       = ssd::SsdDefaultBoxes(
        {16, 8, 4, 2, 1, 1},
        {8, 16, 24, 32, 48, 64, 80},
        {4, 8, 16, 32, 64, 64},
        {4, 6, 6, 6, 4, 4});
    detector::DetectorHooks hooks;
    hooks.build = [&](nn::ParameterStore& p, nn::ParameterStore&) {
      const auto net = ssd::SSD(p, nn::zeros({1, 64, 64, 1}), 11, true);
      check::Require(net.anchors == int(anchors.size()), "compact SSD default-box alignment");
    };
    hooks.loss = [&](nn::ParameterStore& p,
                     nn::ParameterStore&,
                     const DetectionScenes&  scenes,
                     const std::vector<int>& rows) {
      const auto net     = ssd::SSD(p, detector::SceneImages(scenes, rows), 11, true);
      const auto targets = ssd::MakeSsdTargets(anchors, scenes, rows, 10);
      return ssd::SsdLoss(net.logits, net.deltas, targets.labels, targets.targets, 11).loss;
    };
    hooks.detect = [&](nn::ParameterStore& p,
                       nn::ParameterStore&,
                       const DetectionScenes&  scenes,
                       const std::vector<int>& rows) {
      return ssd::DetectSsd(
          [&](const nn::Tensor& images) { return ssd::SSD(p, images, 11, true); },
          anchors,
          scenes,
          rows,
          11);
    };
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
    std::cout << "All SSD checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
