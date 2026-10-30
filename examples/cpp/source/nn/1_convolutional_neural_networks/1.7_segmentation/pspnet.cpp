// CMD: just run-nn pspnet
// PSPNet: ResNet101/output stride 8 with pyramid pooling bins 1/2/3/6.
// https://arxiv.org/abs/1612.01105
//
// Port of tflow's pspnet. Four adaptive average branches project 2048->512,
// resize and concatenate with the backbone, then 3x3/512, dropout .1,
// class 21 and bilinear input resize. An auxiliary stage-C4 classifier uses
// 3x3/256 and relative loss weight .4. Pooling uses floor starts/ceil ends,
// including overlapping odd-size bins. Independently check every bin value
// and input derivative on odd, rectangular and smaller-than-bin grids.
// Construct all full-model gradients; execute every compact derivative and
// verify four pyramid/auxiliary outputs.
//
// Compact ResNet18/8, pool 16/head 32/aux 16, no dropout; 600 fit/120
// validation/120 test independent procedural 96px shape scenes, four classes.
// 3 epochs, Adam(.001), batch 16, L2 .0001, fit-frequency balanced pixel CE.
// Main + .4 aux CE normalized by 1.4; select best validation mIoU, restore
// checkpoint and report test confusion and input/truth/prediction PNG.
// Scratch teaching experiment, not pretrained VOC/Cityscapes performance.
//
// TF -> datamunge: TF builds the full 473x473 model as a graph only; here the
// full model executes forward and backward on a 33x33 image (same parameter,
// state and derivative census). Weights and shapes use tensor generators;
// shape pixels differ from TF's fixed-seed mt19937 implementation.
// Options: epochs=3 accuracy=0|1 (default 0; accuracy=1 retains .55 mIoU) threads=N.
// Artifacts: output/nn/models/pspnet_shapes/.

#include "common/pspnet.hpp"

#include <array>
#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/segmentation.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

Tensor Wave(const nn::Shape& shape, double scale, bool cosine = false) {
  std::int64_t n = 1;
  for (auto d : shape)
    n *= d;
  std::vector<float> v(static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = cosine ? std::cos(float(scale) * float(i)) : std::sin(float(scale) * float(i));
  return nn::from_vector(v, shape);
}

std::pair<double, double> GradientNorms(
    const nn::ParameterStore& p,
    const Tensor&             objective,
    bool                      nonzero,
    const std::string&        what) {
  const auto params = p.parameters();
  const auto grads  = nn::grad(objective, params);
  check::Require(grads.size() == params.size(), what + ": every parameter has a derivative");
  double low = INFINITY, high = 0;
  for (std::size_t k = 0; k < grads.size(); ++k) {
    const double norm =
        std::sqrt(nn::sum(nn::square(grads[k].to(nn::DType::Float64))).item<double>());
    check::Require(
        std::isfinite(norm) && (!nonzero || norm > 0),
        "finite nonzero " + what + " derivative: " + p.names()[k]);
    low  = std::min(low, norm);
    high = std::max(high, norm);
  }
  return {low, high};
}

void PoolChecks() {
  for (const auto shape :
       std::vector<std::array<int, 3>>{{7, 9, 3}, {2, 3, 6}, {12, 12, 2}, {1, 1, 1}}) {
    const int h = shape[0], w = shape[1], bins = shape[2];
    Tensor    input = Wave({2, h, w, 2}, .17);
    input.requires_grad_(true);
    const Tensor weights  = Wave({2, bins, bins, 2}, .23, true);
    const Tensor pool     = pspnet::AdaptiveAveragePool(input, bins);
    const auto   gradient = nn::grad(nn::sum(pool * weights), {input})[0].to_vector<float>();
    const auto   value = input.to_vector<float>(), out = pool.to_vector<float>(),
               weight = weights.to_vector<float>();
    std::vector<double> expected_gradient(value.size(), 0);
    double              error = 0;
    for (int n = 0; n < 2; ++n)
      for (int by = 0; by < bins; ++by)
        for (int bx = 0; bx < bins; ++bx)
          for (int c = 0; c < 2; ++c) {
            const int y0 = by * h / bins, y1 = ((by + 1) * h + bins - 1) / bins, x0 = bx * w / bins,
                      x1    = ((bx + 1) * w + bins - 1) / bins,
                      index = ((n * bins + by) * bins + bx) * 2 + c;
            double mean     = 0;
            for (int y = y0; y < y1; ++y)
              for (int x = x0; x < x1; ++x) {
                const int i = ((n * h + y) * w + x) * 2 + c;
                mean += value[std::size_t(i)] / double((y1 - y0) * (x1 - x0));
                expected_gradient[std::size_t(i)] +=
                    weight[std::size_t(index)] / double((y1 - y0) * (x1 - x0));
              }
            error = std::max(error, std::abs(mean - out[std::size_t(index)]));
          }
    for (std::size_t i = 0; i < value.size(); ++i)
      error = std::max(error, std::abs(expected_gradient[i] - gradient[i]));
    check::Near(
        error, 0, 2e-6, "every independent adaptive-pool mean and overlapping pixel gradient");
    std::cout << "  adaptive pool " << h << 'x' << w << " -> " << bins << 'x' << bins
              << ": host value/gradient error=" << error << std::endl;
  }
}

pspnet::PSPNetConfig Config(bool compact) {
  pspnet::PSPNetConfig c;
  if (compact) {
    c.classes          = 4;
    c.depth            = 18;
    c.width_divisor    = 8;
    c.pool_width       = 16;
    c.head_width       = 32;
    c.aux_width        = 16;
    c.keep_probability = 1;
  }
  return c;
}

Tensor Objective(const pspnet::PSPNetOutputs& net) {
  return nn::mean(nn::square(net.logits)) + .4 * nn::mean(nn::square(net.auxiliary));
}

void ModelChecks() {
  {
    nn::ParameterStore p(17), state(17);
    const auto net = pspnet::PSPNet(p, state, Wave({1, 33, 33, 3}, .13), Config(false), true);
    GradientNorms(p, Objective(net), false, "full PSPNet");
    check::Require(net.pools.size() == 4, "full PSPNet every parameter connected");
    check::Require(
        p.at("head/W").shape() == nn::Shape({3, 3, 4096, 512}),
        "full pyramid channels concatenate to 4096");
    check::Require(
        p.count() == 67949930 && state.count() == 110976 && p.size() == 334,
        "full PSPNet101 census (TF: 67949930 parameters, 110976 BN state, 334 gradients)");
    std::cout << "  full PSPNet101: " << p.count() << " parameters, " << state.count()
              << " BN state, " << p.size() << " gradients (executed on 33x33)" << std::endl;
  }
  for (int size : {96, 97}) {
    nn::ParameterStore p(17), state(17);
    const auto net = pspnet::PSPNet(p, state, Wave({2, size, size, 3}, .13), Config(true), true);
    check::Require(
        net.logits.shape() == nn::Shape({2, size, size, 4}) &&
            net.auxiliary.shape() == net.logits.shape(),
        "main and auxiliary logits match input grid");
    const auto norms  = GradientNorms(p, Objective(net), true, "PSPNet");
    const int  bins[] = {1, 2, 3, 6};
    for (int k = 0; k < 4; ++k)
      check::Require(
          net.pools[std::size_t(k)].shape()[1] == bins[k] &&
              net.pools[std::size_t(k)].shape()[2] == bins[k],
          "four distinct pyramid sizes");
    std::cout << "  compact PSPNet " << size << "px: " << p.count()
              << " parameters; every gradient norm [" << norms.first << ',' << norms.second << "]"
              << std::endl;
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    int  epochs   = 3;
    bool accuracy = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("epochs=", 0) == 0)
        epochs = std::stoi(arg.substr(7));
      else if (arg == "accuracy=0" || arg == "accuracy=1")
        accuracy = arg == "accuracy=1";
      else if (arg.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(arg.substr(8)));
      else
        throw std::invalid_argument("expected epochs=N, accuracy=0|1 or threads=N");
    }
    seg::SegmentationTrainingChecks();
    seg::SegmentationLossChecks();
    PoolChecks();
    ModelChecks();
    const auto fit        = seg::SegmentationShapes(600, 96, 11),
               validation = seg::SegmentationShapes(120, 96, 22),
               test       = seg::SegmentationShapes(120, 96, 33);
    seg::SegmentationConfig cfg;
    cfg.name     = "pspnet_shapes";
    cfg.epochs   = epochs;
    cfg.accuracy = accuracy;
    const auto builder =
        [](nn::ParameterStore& p, nn::ParameterStore& state, const Tensor& images, bool training) {
          const auto net = pspnet::PSPNet(p, state, images, Config(true), training);
          return seg::SegmentationModel{net.logits, {net.auxiliary}, .4f};
        };
    {
      using namespace seg;
      const auto& source_fit        = fit;
      const auto& source_validation = validation;
      const auto& source_test       = test;
      auto        training_cfg      = cfg;
      bool        explicit_epochs   = false;
      for (int option = 1; option < argc; ++option)
        explicit_epochs = explicit_epochs || std::string(argv[option]).rfind("epochs=", 0) == 0;
      if (!explicit_epochs)
        training_cfg.epochs = example_runtime::Limit(training_cfg.epochs, 1);
      training_cfg.batch = example_runtime::Limit(training_cfg.batch, 2);
      auto subset        = [](const SegmentationData& data, int count) {
        SegmentationData result;
        result.count             = std::min(data.count, count);
        result.height            = data.height;
        result.width             = data.width;
        result.classes           = data.classes;
        const std::size_t pixels = std::size_t(result.count) * data.height * data.width;
        result.images.assign(data.images.begin(), data.images.begin() + 3 * pixels);
        result.labels.assign(data.labels.begin(), data.labels.begin() + pixels);
        return result;
      };
      SegmentationData quick_fit, quick_validation, quick_test;
      if (!example_runtime::Full()) {
        quick_fit        = subset(source_fit, 8);
        quick_validation = subset(source_validation, 4);
        quick_test       = subset(source_test, 4);
      }
      const auto& fit        = example_runtime::Full() ? source_fit : quick_fit;
      const auto& validation = example_runtime::Full() ? source_validation : quick_validation;
      const auto& test       = example_runtime::Full() ? source_test : quick_test;
      example_runtime::Describe();
      check::Require(
          training_cfg.epochs > 0 && training_cfg.batch > 0 && fit.classes == validation.classes &&
              fit.classes == test.classes,
          "valid segmentation configuration");
      check::Require(
          !example_runtime::Full() || training_cfg.accuracy || training_cfg.epochs >= 2,
          "short segmentation demonstration needs at least two epochs");
      nn::ParameterStore p(training_cfg.seed), state(training_cfg.seed);
      {
        nn::NoGradGuard guard; // create every parameter (no statistics update) before the optimizer
        const auto      probe = SegmentationBatch(fit, {0});
        builder(p, state, probe.first, false);
      }
      const LogitsFn infer = [&](const nn::Tensor& images) {
        return builder(p, state, images, false).logits;
      };
      std::vector<std::int64_t> frequencies(std::size_t(fit.classes), 1);
      for (int label : fit.labels)
        ++frequencies.at(std::size_t(label));
      std::vector<float> weights(std::size_t(fit.classes));
      for (int c = 0; c < fit.classes; ++c)
        weights[std::size_t(c)] =
            1 / std::sqrt(float(frequencies[std::size_t(c)]) / float(fit.labels.size()));
      const auto              class_weights = nn::from_vector(weights, {fit.classes});
      std::vector<nn::Tensor> kernels;
      for (const auto& t : p.parameters())
        if (t.dim() > 1)
          kernels.push_back(t);
      nn::OptimizerConfig oc;
      oc.kind = nn::OptimizerKind::Adam;
      nn::Optimizer optimizer(p, training_cfg.learning_rate, oc);
      const auto    directory = SegmentationDirectory(training_cfg.name);
      const auto    prefix    = (directory / "best").string();
      std::ofstream history(directory / "history.tsv");
      history << "epoch\tloss\tvalidation_mean_iou\n";
      std::cout << "== " << training_cfg.name << ": " << fit.count << " fit / " << validation.count
                << " validation / " << test.count << " test; " << p.count() << " parameters; "
                << fit.height << 'x' << fit.width << ", classes=" << fit.classes << std::endl;
      nn::BatchOrder      batches(std::size_t(fit.count), std::size_t(training_cfg.batch), 29);
      double              best       = -1;
      int                 best_epoch = -1;
      std::vector<double> epoch_losses;
      if (!training_cfg.warm_start.empty()) {
        check::Require(
            std::filesystem::absolute(training_cfg.warm_start).lexically_normal() !=
                std::filesystem::absolute(prefix).lexically_normal(),
            "segmentation warm start requires a separate output tag");
        p.load(training_cfg.warm_start + "_params.npz");
        state.load(training_cfg.warm_start + "_state.npz");
        best       = SegmentationScore(infer, validation).mean_iou;
        best_epoch = 0;
        p.save(prefix + "_params.npz");
        state.save(prefix + "_state.npz");
        history << 0 << "\tnan\t" << best << '\n';
        history.flush();
        std::cout << "  warm start: " << training_cfg.warm_start
                  << "; fresh Adam moments; restored validation mIoU=" << best << std::endl;
      }
      double train_seconds = 0;
      for (int epoch = 1; epoch <= training_cfg.epochs; ++epoch) {
        const auto started = std::chrono::steady_clock::now();
        batches.begin_epoch();
        double total = 0;
        int    seen  = 0;
        for (;;) {
          const auto picked = batches.next();
          if (picked.empty())
            break;
          const std::vector<int> rows(picked.begin(), picked.end());
          const auto             batch = SegmentationBatch(fit, rows);
          const auto             model = builder(p, state, batch.first, true);
          nn::Tensor             loss =
              SegmentationCrossEntropy(model.logits, batch.second, fit.classes, class_weights);
          if (!model.auxiliary.empty()) {
            nn::Tensor sum = loss;
            for (const auto& output : model.auxiliary)
              sum = sum +
                    double(model.auxiliary_weight) *
                        SegmentationCrossEntropy(output, batch.second, fit.classes, class_weights);
            loss = sum / (1.0 + double(model.auxiliary_weight) * double(model.auxiliary.size()));
          }
          const double value = loss.item<double>();
          check::Require(std::isfinite(value), "finite segmentation training loss");
          nn::Tensor objective = loss;
          if (training_cfg.l2 > 0 && !kernels.empty()) {
            nn::Tensor penalty = nn::sum(nn::square(kernels[0]));
            for (std::size_t i = 1; i < kernels.size(); ++i)
              penalty = penalty + nn::sum(nn::square(kernels[i]));
            objective = objective + .5 * double(training_cfg.l2) * penalty;
          }
          optimizer.minimize(objective);
          total += value * double(rows.size());
          seen += int(rows.size());
        }
        train_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        epoch_losses.push_back(total / seen);
        check::Require(std::isfinite(epoch_losses.back()), "finite segmentation epoch mean");
        const auto score = SegmentationScore(infer, validation);
        check::Require(std::isfinite(score.mean_iou), "finite validation mIoU");
        if (score.mean_iou > best) {
          best       = score.mean_iou;
          best_epoch = epoch;
          p.save(prefix + "_params.npz");
          state.save(prefix + "_state.npz");
        }
        history << epoch << '\t' << total / seen << '\t' << score.mean_iou << '\n';
        history.flush();
        std::cout << "  epoch " << epoch << ": loss=" << total / seen
                  << ", validation mIoU=" << score.mean_iou
                  << ", pixel accuracy=" << score.pixel_accuracy << std::endl;
      }
      p.load(prefix + "_params.npz");
      state.load(prefix + "_state.npz");
      check::Near(
          SegmentationScore(infer, validation).mean_iou,
          best,
          1e-10,
          "restored validation score matches selected checkpoint");
      std::cout << "  restored validation mIoU=" << best << "; checkpoint agreement verified"
                << std::endl;
      if (training_cfg.inspect)
        training_cfg.inspect(infer, test, directory);
      const auto result = SegmentationScore(infer, test);
      std::cout << "  selected epoch=" << best_epoch << "; test mIoU=" << result.mean_iou
                << ", pixel accuracy=" << result.pixel_accuracy
                << "\n  test confusion (truth rows, predicted columns):\n";
      for (int c = 0; c < fit.classes; ++c) {
        std::cout << "   ";
        for (int k = 0; k < fit.classes; ++k)
          std::cout << ' ' << result.confusion[std::size_t(c * fit.classes + k)];
        std::cout << '\n';
      }
      SegmentationPreview(infer, test, directory / "preview.png");
      std::cout << "  checkpoint, history and image/truth/prediction preview: "
                << directory.string() << std::endl;
      std::printf(
          "  training time %.1f s (%.2f s/epoch, excluding validation)\n",
          train_seconds,
          train_seconds / training_cfg.epochs);
      if (example_runtime::Full())
        CheckSegmentationDemonstration(training_cfg, epoch_losses, result);
      else
        check::Require(
            std::isfinite(result.mean_iou) && result.mean_iou >= 0 && result.mean_iou <= 1,
            "finite quick segmentation evaluation");
      if (example_runtime::Full() && !training_cfg.accuracy)
        std::cout << "  short learning demonstration passed: mean loss " << epoch_losses.front()
                  << " -> " << epoch_losses.back() << "; accuracy=1 enables mIoU target "
                  << training_cfg.minimum_test_iou << std::endl;
    }
    std::cout << "All PSPNet checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
