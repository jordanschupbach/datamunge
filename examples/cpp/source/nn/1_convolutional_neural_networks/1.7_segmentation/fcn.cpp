// CMD: just run-nn fcn
// FCN-32s/16s/8s: dense VGG predictions, bilinear deconvolution and skip fusion.
// https://arxiv.org/abs/1411.4038
// https://github.com/shelhamer/fcn.berkeleyvision.org
//
// Port of tflow's fcn. Construct the original 21-class VGG graphs, including
// pad100/ceil pooling, valid7x7 fc6, pool4/pool3 crop offsets5/9, and final
// crop19/27/31. Check independent trainable parameter counts
// 134346581/134357354/134362751 and every derivative. Bilinear filters are
// fixed diagonal constants, matching the release's frozen upsampling; learned
// skip projections retain signed scores. Check deconvolution/skip/crop values
// and input derivatives against independent host scatter calculations,
// including class isolation. Check every compact derivative and output sizes
// 95/96/97.
//
// Compare compact versions on procedural circles/rectangles/triangles with
// class-independent intensities/tints, occlusion and noise, requiring spatial
// shape recognition. This is explicitly a teaching dataset, not Pascal VOC.
// All thirteen VGG convolutions remain; widths/8, fc6 3x3/64, SAME padding,
// dropout0 and crop0 reduce computation. Train from scratch with inverse-sqrt
// fit-frequency pixel CE, Adam(.001), batch16,L2 .0001,twenty epochs and best
// validation mean IoU; 600 fit/120 validation/120 test images,96x96. Report
// confusion and export image/truth/prediction PNGs. No pretrained weights.
//
// TF -> datamunge: TF builds the full models as graphs only; here the full
// models execute forward and backward on an 8x8 image (pad 100 keeps every
// layer valid), so the counts and derivatives come from a real run. The
// shapes use tensor RNG and geometry, so fixed-seed pixels differ from TF's
// mt19937 implementation. Weights also use datamunge's generator.
// Options: stride=32|16|8 (default all) epochs=20 threads=N.
// Artifacts: output/nn/models/fcn{32,16,8}_shapes/.

#include "common/fcn.hpp"

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

std::vector<float> Wave(std::size_t n, double scale, double offset = 0, bool cosine = false) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i)
    v[i] = float(offset + (cosine ? std::cos(scale * double(i)) : std::sin(scale * double(i))));
  return v;
}

/// Max/min gradient norm (and count) over every parameter; requires finite (and nonzero when
/// `nonzero`).
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

void FusionChecks() {
  Tensor a           = nn::from_vector(Wave(24, .7), {1, 3, 4, 2}),
         b           = nn::from_vector(Wave(160, .31, -.2, true), {1, 8, 10, 2});
  const auto weights = Wave(96, .13);
  a.requires_grad_(true);
  b.requires_grad_(true);
  const Tensor fused   = fcn::FcnFuseSkip(a, b, 2, false, 1);
  const auto gradients = nn::grad(nn::sum(fused * nn::from_vector(weights, {1, 6, 8, 2})), {a, b});
  const auto av = a.to_vector<float>(), bv = b.to_vector<float>(), out = fused.to_vector<float>();
  const auto ga_out = gradients[0].to_vector<float>(), gb_out = gradients[1].to_vector<float>();
  std::vector<double> expected(6 * 8 * 2), ga(3 * 4 * 2), gb(8 * 10 * 2);
  for (int r = 0; r < 6; ++r)
    for (int c = 0; c < 8; ++c)
      for (int k = 0; k < 2; ++k) {
        const int index              = (r * 8 + c) * 2 + k;
        expected[std::size_t(index)] = bv[std::size_t(((r + 1) * 10 + c + 1) * 2 + k)];
        gb[std::size_t(((r + 1) * 10 + c + 1) * 2 + k)] = weights[std::size_t(index)];
      }
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 4; ++c)
      for (int dy = 0; dy < 4; ++dy)
        for (int dx = 0; dx < 4; ++dx) {
          const int y = 2 * r + dy - 1, x = 2 * c + dx - 1;
          if (y < 0 || x < 0 || y >= 6 || x >= 8)
            continue;
          const double factor = (1 - std::abs(dy - 1.5) / 2) * (1 - std::abs(dx - 1.5) / 2);
          for (int k = 0; k < 2; ++k) {
            const int destination = (y * 8 + x) * 2 + k, source = (r * 4 + c) * 2 + k;
            expected[std::size_t(destination)] += factor * av[std::size_t(source)];
            ga[std::size_t(source)] += factor * weights[std::size_t(destination)];
          }
        }
  check::Require(fused.shape() == nn::Shape({1, 6, 8, 2}), "SAME bilinear fusion shape");
  double error = 0;
  for (std::size_t i = 0; i < out.size(); ++i)
    error = std::max(error, std::abs(out[i] - expected[i]));
  for (std::size_t i = 0; i < ga_out.size(); ++i)
    error = std::max(error, std::abs(ga_out[i] - ga[i]));
  for (std::size_t i = 0; i < gb_out.size(); ++i)
    error = std::max(error, std::abs(gb_out[i] - gb[i]));
  check::Near(error, 0, 1e-6, "host bilinear scatter, signed skip crop and every input gradient");
  const auto kernel = fcn::BilinearKernel(2, 2).to_vector<float>(); // [4, 4, 2, 2]
  check::Near(kernel[((1 * 4 + 1) * 2 + 0) * 2 + 0], .5625, 0, "bilinear kernel centre");
  check::Near(kernel[((1 * 4 + 1) * 2 + 0) * 2 + 1], 0, 0, "no cross-class mixing");
  std::cout
      << "  signed bilinear/skip/crop values and every input derivative match host scatter, error="
      << error << std::endl;
}

fcn::FcnConfig CompactConfig(int stride) {
  fcn::FcnConfig c;
  c.stride           = stride;
  c.classes          = 4;
  c.width_divisor    = 8;
  c.fc_width         = 64;
  c.compact          = true;
  c.keep_probability = 1;
  return c;
}

void ModelChecks() {
  const int          strides[]        = {32, 16, 8};
  const std::int64_t full_counts[]    = {134346581, 134357354, 134362751};
  const std::int64_t compact_counts[] = {271916, 272176, 272308};
  for (int v = 0; v < 3; ++v) {
    nn::ParameterStore p(17);
    fcn::FcnConfig     c;
    c.stride       = strides[v];
    const auto net = fcn::Fcn(p, nn::from_vector(Wave(8 * 8 * 3, .13), {1, 8, 8, 3}), c, true);
    check::Require(
        p.count() == full_counts[v], "full FCN trainable count excludes frozen bilinear filters");
    check::Require(
        net.logits.shape() == nn::Shape({1, 8, 8, 21}), "full FCN crops logits to the input grid");
    GradientNorms(p, nn::mean(nn::square(net.logits)), false, "full FCN");
    check::Require(p.size() == std::size_t(32 + 2 * v), "full FCN parameter connections");
    std::cout << "  full FCN-" << strides[v] << "s: " << p.count() << " parameters, " << p.size()
              << " derivatives (executed on an 8x8 image)" << std::endl;
  }
  for (int v = 0; v < 3; ++v) {
    nn::ParameterStore p(17);
    double             low = INFINITY, high = 0;
    for (int side : {95, 96, 97}) {
      const auto net = fcn::Fcn(
          p,
          nn::from_vector(Wave(std::size_t(2 * side * side * 3), .13), {2, side, side, 3}),
          CompactConfig(strides[v]),
          true);
      check::Require(p.count() == compact_counts[v], "compact FCN parameter count");
      check::Require(
          net.logits.shape() == nn::Shape({2, side, side, 4}),
          "FCN logits align to original image dimensions");
      if (side != 96)
        continue;
      const auto norms = GradientNorms(p, nn::mean(nn::square(net.logits)), true, "FCN");
      low              = std::min(low, norms.first);
      high             = std::max(high, norms.second);
    }
    std::cout << "  compact FCN-" << strides[v] << "s: " << p.count()
              << " parameters; exact95/96/97 output grids; gradient norms [" << low << ',' << high
              << "]" << std::endl;
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    int              epochs  = 20;
    std::vector<int> strides = {32, 16, 8};
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("epochs=", 0) == 0)
        epochs = std::stoi(arg.substr(7));
      else if (arg.rfind("stride=", 0) == 0) {
        const int stride = std::stoi(arg.substr(7));
        if (stride != 32 && stride != 16 && stride != 8)
          throw std::invalid_argument("stride must be32,16or8");
        strides = {stride};
      } else if (arg.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(arg.substr(8)));
      else
        throw std::invalid_argument("expected epochs=N, stride=32|16|8 or threads=N");
    }
    seg::SegmentationLossChecks();
    FusionChecks();
    ModelChecks();
    const auto fit        = seg::SegmentationShapes(600, 96, 11),
               validation = seg::SegmentationShapes(120, 96, 22),
               test       = seg::SegmentationShapes(120, 96, 33);
    for (int stride : strides) {
      seg::SegmentationConfig cfg;
      cfg.name   = "fcn" + std::to_string(stride) + "_shapes";
      cfg.epochs = epochs;
      const auto builder =
          [stride](
              nn::ParameterStore& p, nn::ParameterStore&, const Tensor& images, bool training) {
            return seg::SegmentationModel{
                fcn::Fcn(p, images, CompactConfig(stride), training).logits, {}};
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
            training_cfg.epochs > 0 && training_cfg.batch > 0 &&
                fit.classes == validation.classes && fit.classes == test.classes,
            "valid segmentation configuration");
        check::Require(
            !example_runtime::Full() || training_cfg.accuracy || training_cfg.epochs >= 2,
            "short segmentation demonstration needs at least two epochs");
        nn::ParameterStore p(training_cfg.seed), state(training_cfg.seed);
        {
          nn::NoGradGuard
                     guard; // create every parameter (no statistics update) before the optimizer
          const auto probe = SegmentationBatch(fit, {0});
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
        std::cout << "== " << training_cfg.name << ": " << fit.count << " fit / "
                  << validation.count << " validation / " << test.count << " test; " << p.count()
                  << " parameters; " << fit.height << 'x' << fit.width
                  << ", classes=" << fit.classes << std::endl;
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
                sum = sum + double(model.auxiliary_weight) *
                                SegmentationCrossEntropy(
                                    output, batch.second, fit.classes, class_weights);
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
    }
    std::cout << "All FCN-32s/16s/8s checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
