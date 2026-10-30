// CMD: just run-nn dpn
// Dual path networks: a shared bottleneck feeding residual and dense paths.
//
// Port of tflow's dpn. Follow the released DPN units: pre-activation 1x1 /
// grouped 3x3 / 1x1 whose output is split into a residual update (added) and
// inc new dense channels (concatenated); each stage starts with a projection
// of the joined input. With the last 1x1 zeroed, ordinary units must pass the
// residual path and old dense channels through unchanged with identity
// derivatives, and strided projection units must reproduce an explicit
// BN/ReLU/1x1 reference. Across every stage, the residual path must equal the
// projection plus the sum of all residual updates (bitwise, accumulating
// float32 in the same order), and the dense path the concatenation of the
// projection's dense part and every unit's new channels, in order.
// Full DPN-68/92/98/107/131 have 12,611,602 / 37,668,392 / 61,570,728 /
// 86,917,800 / 79,254,504 learned parameters (an independent count of the
// reference layout); DPN-92 is also checked for stage dimensions and every
// parameter derivative at 224 pixels in training mode (moving averages are
// not updated). Parameter counts do not depend on the image size, so the
// count-only models are built on 64-pixel images (tflow only builds graphs).
// https://arxiv.org/abs/1707.01629
// https://github.com/rwightman/pytorch-dpn-pretrained/blob/master/dpn.py
//
// Train all 92 layers on CIFAR-10 with quarter widths and increments,
// cardinality 8 (quarter-width bottlenecks of 24 channels are not divisible
// by 32) and a 3x3 stride-one stem without max pooling. Use 12k fit / 5k
// validation, twelve epochs, crop/flip, Adam(.001), kernel L2(.0001) and the
// best-validation checkpoint before evaluating all 10k test images. This is a
// compact CPU adaptation, not an ImageNet reproduction. The grouped 3x3s use
// one grouped filter per layer (see common/dpn.hpp). CIFAR-10 is read from the
// tflow dataset cache (common/paths.hpp) or data=.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

#include "common/dpn.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>

#include "common/architecture_checks.hpp"
#include "common/checks.hpp"
#include "common/image_training.hpp"

namespace nn = datamunge::nn;

namespace {

std::vector<float> PatternValues(std::int64_t n, float scale, float phase) {
  std::vector<float> v(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i)
    v[static_cast<std::size_t>(i)] = scale * std::sin(.37f * static_cast<float>(i) + phase);
  return v;
}
nn::Tensor Pattern(const nn::Shape& shape, float scale, float phase) {
  std::int64_t n = 1;
  for (auto d : shape)
    n *= d;
  return nn::from_vector(PatternValues(n, scale, phase), shape);
}
double MaxDifference(const std::vector<float>& a, const std::vector<float>& b) {
  check::Require(a.size() == b.size(), "matching DPN comparison shapes");
  double error = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    error = std::max(error, std::abs(double(a[i]) - b[i]) / std::max(1., std::abs(double(b[i]))));
  return error;
}

// Zeroed final 1x1: identity paths for ordinary units, explicit projection for
// strided first units, and the corresponding input derivatives.
void BlockChecks() {
  nn::ParameterStore p(17), state(17);
  const auto         r = Pattern({2, 5, 5, 4}, .8f, .3f), d = Pattern({2, 5, 5, 3}, .8f, 1.3f);
  auto               run = [&](const nn::Tensor& residual, const nn::Tensor& dense) {
    const dpn::DPNPaths x{residual, dense, 4, 3};
    return std::make_pair(
        dpn::DPNBlock(p, state, "normal", x, 4, 4, 2, 2, false),
        dpn::DPNBlock(p, state, "down", x, 6, 4, 2, 2, true, 2));
  };
  {
    nn::NoGradGuard guard;
    run(r, d);
    for (const auto* name : {"normal/conv3/W", "down/conv3/W"})
      p.set(name, nn::zeros_like(p.at(name)));
    for (const auto& name : state.names()) {
      const bool variance = name.find("moving_var") != std::string::npos;
      const auto shape    = state.at(name).shape();
      auto       values   = PatternValues(shape[0], variance ? .2f : .3f, variance ? 0.f : 1.f);
      if (variance)
        for (auto& v : values)
          v += .8f;
      state.set(name, nn::from_vector(values, shape));
    }
  }
  nn::Tensor residual = r.clone(), dense = d.clone();
  residual.requires_grad_(true);
  dense.requires_grad_(true);
  const auto [normal, down] = run(residual, dense);
  // Explicit projection: inference BN, ReLU, strided 1x1, split 6 + 4.
  nn::Tensor projection;
  {
    nn::NoGradGuard   guard;
    const nn::Tensor  joined = nn::concat({r, d}, 3);
    const std::string n      = "down/projection/bn";
    const nn::Tensor  bn     = (joined - state.at(n + "/moving_mean")) *
                              nn::rsqrt(state.at(n + "/moving_var") + 1e-3) * p.at(n + "/gamma") +
                          p.at(n + "/beta");
    projection = nn::conv2d(
        nn::relu(bn), p.at("down/projection/W"), nn::conv_options(2, nn::Padding::Valid));
  }
  // Weighted sums of each output; their input derivatives must be the weights.
  const auto wr = Pattern({2, 5, 5, 4}, 1, .1f), wd = Pattern({2, 5, 5, 5}, 1, .7f);
  const auto loss      = nn::sum(normal.output.residual * wr) + nn::sum(normal.output.dense * wd);
  const auto gradients = nn::grad(loss, {residual, dense});
  check::Require(
      normal.output.residual.shape() == nn::Shape({2, 5, 5, 4}) &&
          normal.output.dense.shape() == nn::Shape({2, 5, 5, 5}),
      "ordinary DPN unit widths");
  check::Require(
      down.output.residual.shape() == nn::Shape({2, 3, 3, 6}) &&
          down.output.dense.shape() == nn::Shape({2, 3, 3, 6}),
      "strided DPN projection widths");
  const auto nr = normal.output.residual.to_vector<float>(),
             nd = normal.output.dense.to_vector<float>();
  const auto dr = down.output.residual.to_vector<float>(),
             dd = down.output.dense.to_vector<float>();
  const auto pr = projection.to_vector<float>(), gr = gradients[0].to_vector<float>(),
             gd = gradients[1].to_vector<float>();
  const auto rv = r.to_vector<float>(), dv = d.to_vector<float>(), wrv = wr.to_vector<float>(),
             wdv  = wd.to_vector<float>();
  double identity = MaxDifference(nr, rv), gradient = MaxDifference(gr, wrv), projected = 0,
         zeros = 0;
  for (std::size_t i = 0; i < nd.size(); ++i) {
    const std::size_t channel = i % 5, pixel = i / 5;
    if (channel < 3) {
      identity = std::max(identity, std::abs(double(nd[i]) - dv[pixel * 3 + channel]));
      gradient = std::max(gradient, std::abs(double(gd[pixel * 3 + channel]) - wdv[i]));
    } else {
      zeros = std::max(zeros, std::abs(double(nd[i])));
    }
  }
  for (std::size_t i = 0; i < dr.size(); ++i) {
    const std::size_t channel = i % 6, pixel = i / 6;
    projected = std::max(projected, std::abs(double(dr[i]) - pr[pixel * 10 + channel]));
    if (channel < 4)
      projected = std::max(projected, std::abs(double(dd[i]) - pr[pixel * 10 + 6 + channel]));
    else
      zeros = std::max(zeros, std::abs(double(dd[i])));
  }
  check::Near(identity, 0, 0, "ordinary unit passes residual and old dense channels unchanged");
  check::Near(zeros, 0, 0, "zeroed final 1x1 appends zero dense channels");
  check::Near(gradient, 0, 0, "identity derivatives through both paths");
  check::Near(projected, 0, 1e-6, "strided projection splits into residual and dense paths");
  std::cout << "  zeroed updates: identity paths and derivatives exact; strided projection error="
            << projected << std::endl;
}

// Whole-network path unrolling in a compact model with nontrivial weights.
void StageChecks() {
  nn::ParameterStore p(17), state(17);
  dpn::DPNConfig     cfg;
  cfg.width_divisor = 4;
  cfg.groups        = 8;
  cfg.classes       = 10;
  cfg.small_stem    = true;
  nn::NoGradGuard guard;
  const auto      net       = dpn::DPN(p, state, Pattern({2, 32, 32, 3}, 1, 0), cfg);
  double          sum_error = 0, concat_error = 0;
  std::string     widths;
  for (std::size_t stage = 0; stage < net.blocks.size(); ++stage) {
    const auto& blocks       = net.blocks[stage];
    const auto  res          = blocks.back().output.residual.to_vector<float>(),
               den           = blocks.back().output.dense.to_vector<float>();
    const auto res0          = blocks.front().input.residual.to_vector<float>(),
               den0          = blocks.front().input.dense.to_vector<float>();
    const auto&        shape = blocks.back().output.residual.shape();
    const std::int64_t rc = shape[3], pixels = static_cast<std::int64_t>(res.size()) / rc;
    const std::int64_t dc  = blocks.back().output.dense.shape()[3],
                       dc0 = blocks.front().input.dense.shape()[3];
    std::vector<float> total(res0); // float, in the same addition order
    std::int64_t       offset = dc0;
    for (const auto& block : blocks) {
      const auto ru          = block.residual_update.to_vector<float>(),
                 du          = block.dense_update.to_vector<float>();
      const std::int64_t inc = block.dense_update.shape()[3];
      for (std::size_t i = 0; i < ru.size(); ++i)
        total[i] += ru[i];
      for (std::int64_t q = 0; q < pixels; ++q)
        for (std::int64_t c = 0; c < inc; ++c)
          concat_error =
              std::max(concat_error, std::abs(double(den[q * dc + offset + c]) - du[q * inc + c]));
      offset += inc;
    }
    for (std::int64_t q = 0; q < pixels; ++q)
      for (std::int64_t c = 0; c < dc0; ++c)
        concat_error =
            std::max(concat_error, std::abs(double(den[q * dc + c]) - den0[q * dc0 + c]));
    check::Require(
        offset == dc, "dense path holds exactly the projection and every unit's new channels");
    for (std::size_t i = 0; i < res.size(); ++i)
      sum_error = std::max(sum_error, std::abs(double(res[i]) - total[i]));
    widths += (stage ? ", " : "") + std::to_string(rc) + "+" + std::to_string(dc) + " at " +
              std::to_string(shape[1]);
  }
  check::Near(
      sum_error, 0, 0, "residual path is the projection plus the sum of all residual updates");
  check::Near(concat_error, 0, 0, "dense path is the ordered concatenation of all new features");
  std::cout << "  compact DPN-92 paths (residual+dense channels at pixels): " << widths
            << "; residual sum and dense concatenation exact" << std::endl;
}

void FullChecks() {
  const std::map<int, std::pair<std::int64_t, std::int64_t>> counts = {
      {68, {12611602, 65116}},
      {92, {37668392, 125216}},
      {98, {61570728, 164896}},
      {107, {86917800, 213824}},
      {131, {79254504, 223360}}};
  for (const auto& [depth, expected] : counts) {
    nn::ParameterStore p(17), state(17);
    dpn::DPNConfig     cfg;
    cfg.depth = depth;
    {
      nn::NoGradGuard guard;
      dpn::DPN(p, state, Pattern({1, 64, 64, 3}, .1f, 0), cfg);
    }
    check::Require(
        p.count() == expected.first && state.count() == expected.second,
        "DPN-" + std::to_string(depth) + " parameter/state counts");
    std::cout << "  DPN-" << depth << ": parameters=" << p.count() << ", state=" << state.count()
              << std::endl;
  }
  nn::ParameterStore p(17), state(17);
  dpn::DPNConfig     cfg;
  cfg.update_stats              = false;
  const auto         net        = dpn::DPN(p, state, Pattern({2, 224, 224, 3}, .1f, 0), cfg, true);
  const std::int64_t spatial[5] = {56, 56, 28, 14, 7}, channels[5] = {64, 336, 704, 1552, 2688};
  for (int stage = 0; stage < 5; ++stage) {
    const nn::Shape shape =
        stage == 0 ? net.stem.shape() : dpn::DPNJoin(net.stages[stage - 1]).shape();
    check::Require(
        shape == nn::Shape({2, spatial[stage], spatial[stage], channels[stage]}),
        "canonical DPN-92 stage dimensions");
  }
  check::Require(net.logits.shape() == nn::Shape({2, 1000}), "DPN classifier dimensions");
  const auto [minimum, maximum] =
      architecture::GradientNormRange(p, nn::mean(nn::square(net.logits)), "DPN");
  std::cout << "  DPN-92 at 224: joined channels 64/336/704/1552/2688 at 56/56/28/14/7; "
               "training-mode gradient norms ["
            << minimum << ',' << maximum << "]" << std::endl;
}

} // namespace

int main(int argc, char** argv) {
  try {
    using namespace image_training;
    for (int i = 1; i < argc; ++i) // threads= also applies to the checks
      if (std::string(argv[i]).rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(std::string(argv[i]).substr(8)));
    BlockChecks();
    StageChecks();
    FullChecks();
    std::filesystem::path dir = paths::Dataset("cifar10");
    ImageTrainingConfig   cfg;
    cfg.name                  = "dpn92_compact_cifar10";
    cfg.maximum_fit_rows      = 12000;
    cfg.epochs                = 12;
    cfg.minimum_test_accuracy = .4;
    bool explicit_epochs      = false;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      const auto        equal    = argument.find('=');
      if (equal == std::string::npos)
        throw std::invalid_argument("expected key=value option");
      explicit_epochs = explicit_epochs || argument.substr(0, equal) == "epochs";
      if (!ParseCifarOption(argument.substr(0, equal), argument.substr(equal + 1), dir, cfg))
        throw std::invalid_argument("unknown option: " + argument);
    }
    const auto [development, test] = LoadCifar10(dir);
    // Quarter-width DPN-92 with a small stem for CIFAR-10; the count check runs on every forward
    // pass.
    const auto model = [](nn::ParameterStore& p,
                          nn::ParameterStore& state,
                          const nn::Tensor&   image,
                          bool                training) {
      dpn::DPNConfig arch;
      arch.width_divisor = 4;
      arch.groups        = 8;
      arch.classes       = 10;
      arch.small_stem    = true;
      const auto logits  = dpn::DPN(p, state, image, arch, training).logits;
      check::Require(
          p.count() == 2474330 && state.count() == 31304, "compact DPN parameter/state counts");
      return logits;
    };

    // Quick mode bounds the data and batch; an explicit epoch budget is honored.
    if (!explicit_epochs)
      cfg.epochs = example_runtime::Limit(cfg.epochs, 1);
    cfg.batch = example_runtime::Limit(cfg.batch, 8);
    if (!example_runtime::Full())
      cfg.maximum_fit_rows = cfg.maximum_fit_rows ? std::min(cfg.maximum_fit_rows, 32) : 32;
    example_runtime::Describe();
    const std::int64_t development_rows = development.images.shape()[0];
    const std::int64_t test_size = example_runtime::Limit<std::int64_t>(test.images.shape()[0], 32);
    check::Require(
        development.images.dim() == 4 && test.images.dim() == 4 && cfg.validation_rows > 0 &&
            cfg.validation_rows < development_rows && cfg.epochs > 0 && cfg.batch > 0 &&
            cfg.classes > 1,
        "valid image training dimensions/config");
    for (int d = 1; d < 4; ++d)
      check::Require(
          development.images.shape()[d] == test.images.shape()[d],
          "training/test image dimensions match");
    const std::size_t available = static_cast<std::size_t>(development_rows - cfg.validation_rows);
    check::Require(
        cfg.maximum_fit_rows >= 0 && static_cast<std::size_t>(cfg.maximum_fit_rows) <= available,
        "valid optional fitting-row limit");
    const std::size_t fit_count =
        cfg.maximum_fit_rows ? static_cast<std::size_t>(cfg.maximum_fit_rows) : available;
    check::Require(fit_count >= 2, "at least two fitting rows for model initialization");

    // Split the development rows once; keep the validation split fixed when limiting the fitting
    // subset.
    std::vector<std::size_t> order(static_cast<std::size_t>(development_rows));
    std::iota(order.begin(), order.end(), 0);
    std::mt19937_64 split_rng(2026);
    std::shuffle(order.begin(), order.end(), split_rng);
    const std::vector<std::int64_t> fit(
        order.begin(), order.begin() + static_cast<std::ptrdiff_t>(fit_count));
    std::vector<std::int64_t> validation(
        order.begin() + static_cast<std::ptrdiff_t>(available), order.end());
    validation.resize(example_runtime::Limit<std::size_t>(validation.size(), 32));
    std::vector<std::int64_t> test_rows(static_cast<std::size_t>(test_size));
    std::iota(test_rows.begin(), test_rows.end(), 0);

    nn::ParameterStore params(cfg.seed), state(cfg.seed);
    {
      nn::NoGradGuard
          guard; // create every parameter (and batch-norm state) before building the optimizer
      model(params, state, ImageBatch(development, {fit[0], fit[1]}).first, false);
    }
    nn::OptimizerConfig config;
    config.kind = nn::OptimizerKind::Adam;
    nn::Optimizer optimizer(params, cfg.learning_rate, config);

    // Inference-mode cross-entropy, accuracy and confusion over `rows`, in batches of 256.
    auto evaluate = [&](const nn::ImageDataset& dataset, const std::vector<std::int64_t>& rows) {
      nn::NoGradGuard     guard;
      nn::WeightedMean    loss, accuracy;
      nn::ConfusionMatrix confusion(static_cast<std::size_t>(cfg.classes));
      for (std::size_t start = 0; start < rows.size(); start += 256) {
        const std::size_t               end = std::min(rows.size(), start + 256);
        const std::vector<std::int64_t> ids(
            rows.begin() + static_cast<std::ptrdiff_t>(start),
            rows.begin() + static_cast<std::ptrdiff_t>(end));
        const auto [x, y]           = ImageBatch(dataset, ids);
        const nn::Tensor logits     = model(params, state, x, false);
        const nn::Tensor prediction = nn::argmax(logits, 1);
        loss.add(nn::sparse_softmax_cross_entropy(logits, y).item(), end - start);
        accuracy.add(nn::accuracy(prediction, y), end - start);
        confusion.add(y, prediction);
      }
      return std::make_tuple(loss.value(), accuracy.value(), confusion);
    };

    const auto directory   = paths::OutputDir("models/" + cfg.name);
    const auto params_path = (directory / "best_params.npz").string(),
               state_path  = (directory / "best_state.npz").string();
    std::ofstream history(directory / "history.tsv");
    check::Require(bool(history), "open image training history");
    history << "epoch\tvalidation_loss\tvalidation_accuracy\n";
    std::cout << "== " << cfg.name << ": " << fit_count << " fit / " << validation.size()
              << " validation / " << test_size << " test\n  parameters=" << params.count()
              << "; augmentation=" << cfg.augment << "; batch=" << cfg.batch
              << "; lr=" << cfg.learning_rate << "; threads=" << nn::get_num_threads() << std::endl;
    nn::BatchOrder      batches(fit_count, static_cast<std::size_t>(cfg.batch), 29);
    nn::Generator       augmentation(151);
    double              best       = std::numeric_limits<double>::infinity();
    int                 best_epoch = -1;
    std::vector<double> epoch_seconds;
    for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
      const auto epoch_start = std::chrono::steady_clock::now();
      batches.begin_epoch();
      std::size_t used = 0;
      for (;;) {
        auto rows = batches.next();
        if (rows.empty())
          break;
        used += rows.size();
        for (auto& row : rows)
          row = fit[static_cast<std::size_t>(row)];
        // Augmented batch; training-mode forward; cross-entropy plus 0.5 * l2 * sum(W^2); one Adam
        // step.
        const auto [x, y] = ImageBatch(development, rows, cfg.augment ? &augmentation : nullptr);
        const nn::Tensor logits    = model(params, state, x, true);
        const nn::Tensor objective = WithL2(
            TrainingCrossEntropy(logits, y, cfg.classes, cfg.label_smoothing), params, cfg.l2);
        optimizer.minimize(objective);
      }
      check::Require(used == fit_count, "each fit image used once per epoch");
      const double train_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch_start).count();
      const auto [loss, accuracy, confusion] = evaluate(development, validation);
      if (loss < best) {
        best       = loss;
        best_epoch = epoch;
        params.save(params_path);
        if (state.size())
          state.save(state_path);
      }
      history << epoch << '\t' << loss << '\t' << accuracy << '\n';
      history.flush();
      const double seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch_start).count();
      epoch_seconds.push_back(seconds);
      std::cout << "  epoch " << epoch << ": validation NLL=" << loss << ", accuracy=" << accuracy
                << " (" << std::setprecision(3) << train_seconds << " s train, " << seconds
                << " s total)" << std::setprecision(6) << std::endl;
    }
    params.load(params_path);
    if (state.size())
      state.load(state_path);
    check::Near(
        std::get<0>(evaluate(development, validation)),
        best,
        1e-6,
        "best image checkpoint validation loss");
    const auto [test_loss, test_accuracy, confusion] = evaluate(test, test_rows);
    std::cout << "  selected epoch=" << best_epoch << "; test NLL=" << test_loss
              << "; accuracy=" << test_accuracy << "\n  confusion (truth rows):\n";
    std::uint64_t total = 0, correct = 0;
    const auto&   cells = confusion.cells();
    for (int i = 0; i < cfg.classes; ++i) {
      std::cout << "   ";
      for (int j = 0; j < cfg.classes; ++j) {
        std::cout << ' ' << std::setw(4) << cells[i][j];
        total += cells[i][j];
      }
      correct += cells[i][i];
      std::cout << '\n';
    }
    check::Require(total == static_cast<std::uint64_t>(test_size), "all test images evaluated");
    check::Near(test_accuracy, double(correct) / total, 1e-7, "confusion agrees with accuracy");
    check::TrainingQuality(
        test_accuracy >= cfg.minimum_test_accuracy,
        "image classifier demonstration accuracy threshold");
    check::Require(bool(history), "write image training history");
    double train_total = 0;
    for (double seconds : epoch_seconds)
      train_total += seconds;
    std::printf("  training time %.1f s (%.2f s/epoch)\n", train_total, train_total / cfg.epochs);
    std::cout << "  best checkpoint and history: " << directory.string() << "\nAll " << cfg.name
              << " checks passed.\n";
    std::cout << "All DPN checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
