// CMD: just run-nn regnet
// RegNet: X and Y models generated from a quantized linear width rule.
//
// Generate all fourteen released RegNetX/Y models (400MF to 32GF) from
// (depth, w0, wa, wm, group width) and check their stage widths and depths,
// learned parameters (5,495,976 for X-400MF to 145,046,770 for Y-32GF,
// torchvision's counts) and multiply-adds (0.414 to 31.736 GFLOPs for X and
// 0.402 to 32.280 for Y, torchvision's figures) at 224 pixels. Show the rule
// at work for Y-400MF. Check every derivative of Y-400MF in training mode
// with shifted BN betas and SE biases; with a ReLU after every addition, none
// vanish.
// https://arxiv.org/abs/2003.13678
// https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py
//
// Train Y-400MF on CIFAR-10 with a stride-one stem (all sixteen blocks and
// the four stage downsamplings). Use 12k fit / 5k validation, twelve epochs,
// crop/flip, Adam(.001), kernel L2(.0001) and the best-validation checkpoint
// before evaluating all 10k test images.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1.

// Eager port: native initialization gives statistical training parity.
#include "common/regnet.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>

#include "common/image_training.hpp"
#include "common/mobile_checks.hpp"

namespace nn = datamunge::nn;
namespace {
std::string Join(const std::vector<int>& values) {
  std::string text;
  for (size_t i = 0; i < values.size(); ++i)
    text += (i ? "/" : "") + std::to_string(values[i]);
  return text;
}

// Compare the packed grouped kernel with independent per-group convolutions,
// including every filter, bias and input derivative in float64.
void FusedGroupChecks() {
  for (bool bias : {false, true})
    for (auto activation : {nn::Activation::None, nn::Activation::Relu}) {
      nn::ParameterStore  p(17, nn::DType::Float64);
      std::vector<double> values(2 * 7 * 6 * 12);
      for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = std::sin(.13 * i);
      auto                 x = nn::from_vector(values, {2, 7, 6, 12}).requires_grad_();
      nn::ConvLayerOptions o;
      o.bias       = bias;
      o.activation = activation;
      o.stride     = 2;
      o.dilation   = 2;
      auto split   = grouped::GroupedConv2D(p, "fixture", x, 15, 3, 3, o);
      auto inputs  = p.parameters();
      inputs.push_back(x);
      auto a     = nn::grad(nn::sum(nn::square(split)), inputs);
      auto fused = grouped::FusedGroupedConv2D(p, "fixture", x, 15, 3, 3, o);
      auto b     = nn::grad(nn::sum(nn::square(fused)), inputs);
      check::Near(
          nn::max(nn::abs(split - fused)).item(), 0, 1e-12, "packed grouped convolution values");
      for (std::size_t i = 0; i < a.size(); ++i)
        check::Near(
            nn::max(nn::abs(a[i] - b[i])).item(),
            0,
            1e-11,
            "every packed grouped filter/bias/input derivative");
    }
}
void DesignChecks() {
  const auto d = regnet::RegNetDesign(regnet::RegNetModel("y400mf"));
  std::cout << "  Y-400MF (d=16, w0=48, wa=27.89, wm=2.09): u_j = 48, " << d.continuous[1] << ", "
            << d.continuous[2] << ", ..., " << d.continuous.back() << " quantize to "
            << Join(d.quantized) << " -> stages " << Join(d.widths) << " with depths "
            << Join(d.depths) << std::endl;
}

void CostChecks() {
  struct Case {
    const char*      name;
    std::int64_t     params, state, macs;
    std::vector<int> widths, depths;
  };
  const std::vector<Case> cases = {
      {"x400mf", 5495976, 37856, 413812608, {32, 64, 160, 400}, {1, 2, 7, 12}},
      {"x800mf", 7259656, 37312, 799699712, {64, 128, 288, 672}, {1, 3, 7, 5}},
      {"x1.6gf", 9190136, 43504, 1602849792, {72, 168, 408, 912}, {2, 4, 10, 2}},
      {"x3.2gf", 15296552, 62560, 3176621952, {96, 192, 432, 1008}, {2, 6, 15, 2}},
      {"x8gf", 39572648, 90464, 7995132416, {80, 240, 720, 1920}, {2, 5, 15, 1}},
      {"x16gf", 54278536, 111168, 15940755456, {256, 512, 896, 2048}, {2, 6, 13, 1}},
      {"x32gf", 107811560, 162016, 31735929792, {336, 672, 1344, 2520}, {2, 7, 13, 1}},
      {"y400mf", 4344144, 27152, 401842848, {48, 104, 208, 440}, {1, 3, 6, 6}},
      {"y800mf", 6432512, 30432, 833856256, {64, 144, 320, 784}, {1, 3, 8, 2}},
      {"y1.6gf", 11202430, 52672, 1612076160, {48, 120, 336, 888}, {2, 6, 17, 2}},
      {"y3.2gf", 19436338, 66160, 3176482752, {72, 216, 576, 1512}, {2, 5, 13, 1}},
      {"y8gf", 39381472, 86528, 8473070592, {224, 448, 896, 2016}, {2, 4, 10, 1}},
      {"y16gf", 83590140, 122816, 15911510272, {224, 448, 1232, 3024}, {2, 4, 11, 1}},
      {"y32gf", 145046770, 158288, 32279552832, {232, 696, 1392, 3712}, {2, 5, 12, 1}}};
  for (const auto& c : cases) {
    auto design = regnet::RegNetDesign(regnet::RegNetModel(c.name));
    check::Require(
        design.widths == c.widths && design.depths == c.depths,
        std::string("RegNet stage design ") + c.name);
    nn::NoGradGuard      guard;
    nn::ParameterStore   p(17), state(17);
    regnet::RegNetConfig cfg;
    cfg.model         = c.name;
    auto         net  = regnet::RegNet(p, state, nn::zeros({1, 224, 224, 3}), cfg);
    std::int64_t macs = net.classifier_macs;
    for (const auto& l : net.layers)
      macs += l.Macs();
    check::Require(
        p.count() == c.params && state.count() == c.state && macs == c.macs,
        std::string("RegNet cost ") + c.name);
    std::cout << c.name << ":widths=" << Join(design.widths) << " depths=" << Join(design.depths)
              << " parameters=" << p.count() << " MACs=" << macs << std::endl;
  }
}
void GradientChecks() {
  nn::ParameterStore   p(17), state(17);
  regnet::RegNetConfig cfg;
  cfg.update_stats = false;
  auto x           = mobile_checks::Pattern({2, 224, 224, 3}, 1, 0);
  {
    nn::NoGradGuard guard;
    regnet::RegNet(p, state, x, cfg, true);
    mobile_checks::ShiftBetas(p);
  }
  auto net   = regnet::RegNet(p, state, x, cfg, true);
  auto norms = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  for (auto norm : norms)
    check::Require(std::isfinite(norm), "finite RegNet derivative");
  auto range = std::minmax_element(norms.begin(), norms.end());
  check::Require(
      *range.first > 1e-5 * *range.second, "nonzero RegNet gradients with shifted BN/SE");
  std::cout << "Y-400MF gradient norms " << *range.first << " to " << *range.second << std::endl;
}
} // namespace
int main(int argc, char** argv) {
  try {
    using namespace image_training;
    nn::set_num_threads(2);
    FusedGroupChecks();
    DesignChecks();
    CostChecks();
    GradientChecks();
    std::filesystem::path dir = paths::Dataset("cifar10");
    ImageTrainingConfig   cfg;
    cfg.name                  = "regnet_y400mf_cifar10";
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
    // CIFAR RegNet Y-400MF: stride-one stem and ten classes; the parameter count is checked on
    // every pass.
    const auto model = [](nn::ParameterStore& p,
                          nn::ParameterStore& state,
                          const nn::Tensor&   image,
                          bool                training) {
      regnet::RegNetConfig options;
      options.classes    = 10;
      options.small_stem = true;
      const auto net     = regnet::RegNet(p, state, image, options, training);
      check::Require(p.count() == 3907554, "CIFAR RegNet Y-400MF count");
      return net.logits;
    };

    // The training workflow is written out below so the R and Python examples can
    // follow it line by line over the same bound components. Quick mode bounds the
    // data and batch; an explicit epoch budget is honored.
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
    std::cout << "All RegNet checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
