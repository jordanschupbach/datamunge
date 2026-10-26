// CMD: just run-nn non_local
// Non-local neural networks: self-attention over all positions of a feature map.
//
// Port of tflow's non_local. Check all four pairwise functions (embedded
// Gaussian, Gaussian, dot product, concatenation), with and without 2x2
// subsampling of phi and g, against a host computation of
// y_i = sum_j f(x_i,x_j) g(x_j) / C(x), z = BN(W_z y) + x. Check that a
// freshly built block is the identity (BN scale zero) and that, once active,
// one output position depends on every input position for the smooth
// pairwise functions without subsampling (a 3x3 convolution reaches nine;
// max-pooling and the concatenation ReLU pass derivatives to fewer). Count
// ResNet-50 with the paper's five blocks (two in res3, three in res4):
// 32,915,496 learned parameters. Check every derivative in training mode
// (moving averages not updated) at 224 pixels: all are clearly nonzero except
// the biases of phi, g and W_z, which only shift softmax rows or channels
// that training-mode BN normalizes.
// https://arxiv.org/abs/1711.07971
// https://github.com/facebookresearch/video-nonlocal-net
//
// Train the compact ResNet-50 of resnet_bottleneck with five embedded-Gaussian
// blocks (at 16x16 and 8x8 resolution, subsampled) on CIFAR-10, for
// comparison with the plain and channel-attention versions. Use 12k fit / 5k
// validation, twelve epochs, crop/flip, Adam(.001), kernel L2(.0001) and the
// best-validation checkpoint before evaluating all 10k test images. The
// blocks attach through AttentionResNetConfig::after_block (tflow's
// cfg.non_local). Weights come from nn's generator, so trained numbers match
// TF only statistically. CIFAR-10 is read from the tflow dataset cache
// (common/paths.hpp) or data=.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>

#include "common/attention_cnn.hpp"
#include "common/checks.hpp"
#include "common/image_training.hpp"
#include "common/mobile_checks.hpp"
#include "common/non_local.hpp"

// Experiment workflow from examples/nn/common/image_training.hpp.
namespace image_training {
using namespace ::image_training;
inline void TrainImages(const nn::ImageDataset& development, const nn::ImageDataset& test,
                        const LogitsBuilder& model, const ImageTrainingConfig& requested_cfg,
                        const TrainingBuilder& training = {}) {
  auto cfg = requested_cfg;
  cfg.epochs = example_runtime::Limit(cfg.epochs, 1);
  cfg.batch = example_runtime::Limit(cfg.batch, 8);
  if (!example_runtime::Full()) cfg.maximum_fit_rows = cfg.maximum_fit_rows ? std::min(cfg.maximum_fit_rows, 32) : 32;
  example_runtime::Describe();
  const std::int64_t development_rows = development.images.shape()[0];
  const std::int64_t test_size = example_runtime::Limit<std::int64_t>(test.images.shape()[0], 32);
  check::Require(development.images.dim() == 4 && test.images.dim() == 4 && cfg.validation_rows > 0 &&
                     cfg.validation_rows < development_rows && cfg.epochs > 0 && cfg.batch > 0 && cfg.classes > 1,
                 "valid image training dimensions/config");
  for (int d = 1; d < 4; ++d)
    check::Require(development.images.shape()[d] == test.images.shape()[d], "training/test image dimensions match");
  const std::size_t available = static_cast<std::size_t>(development_rows - cfg.validation_rows);
  check::Require(cfg.maximum_fit_rows >= 0 && static_cast<std::size_t>(cfg.maximum_fit_rows) <= available,
                 "valid optional fitting-row limit");
  const std::size_t        fit_count = cfg.maximum_fit_rows ? static_cast<std::size_t>(cfg.maximum_fit_rows) : available;
  std::vector<std::size_t> order(static_cast<std::size_t>(development_rows));
  std::iota(order.begin(), order.end(), 0);
  std::mt19937_64 split_rng(2026);
  std::shuffle(order.begin(), order.end(), split_rng);
  // Keep the validation split fixed when limiting the fitting subset.
  const std::vector<std::int64_t> fit(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(fit_count));
  std::vector<std::int64_t>
      validation(order.begin() + static_cast<std::ptrdiff_t>(available), order.end());
  validation.resize(example_runtime::Limit<std::size_t>(validation.size(), 32));
  std::vector<std::int64_t> test_rows(static_cast<std::size_t>(test_size));
  std::iota(test_rows.begin(), test_rows.end(), 0);

  nn::ParameterStore params(cfg.seed), state(cfg.seed);
  {
    nn::NoGradGuard guard;  // create every parameter (and state entry) before building the optimizer
    const nn::Tensor x = ImageBatch(development, {fit[0], fit[1]}).first;
    model(params, state, x, false);
    if (training) {
      // Training-only parameters (auxiliary heads); their batch-norm statistics
      // go to a scratch store so this pass updates no real moving averages.
      nn::ParameterStore scratch(cfg.seed);
      training(params, scratch, x);
    }
  }
  nn::OptimizerConfig config;
  config.kind = nn::OptimizerKind::Adam;
  nn::Optimizer optimizer(params, cfg.learning_rate, config);

  const auto directory = paths::OutputDir("models/" + cfg.name);
  const auto params_path = (directory / "best_params.npz").string(), state_path = (directory / "best_state.npz").string();
  std::ofstream history(directory / "history.tsv");
  check::Require(bool(history), "open image training history");
  history << "epoch\tvalidation_loss\tvalidation_accuracy\n";
  std::cout << "== " << cfg.name << ": " << fit_count << " fit / " << validation.size() << " validation / " << test_size
            << " test\n  parameters=" << params.count() << "; augmentation=" << cfg.augment << "; batch=" << cfg.batch
            << "; lr=" << cfg.learning_rate << "; threads=" << nn::get_num_threads() << std::endl;
  nn::BatchOrder batches(fit_count, static_cast<std::size_t>(cfg.batch), 29);
  nn::Generator  augmentation(151);
  double         best = std::numeric_limits<double>::infinity();
  int            best_epoch = -1;
  EpochSeconds().clear();
  for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
    const auto epoch_start = std::chrono::steady_clock::now();
    batches.begin_epoch();
    std::size_t used = 0;
    for (;;) {
      auto rows = batches.next();
      if (rows.empty()) break;
      used += rows.size();
      for (auto& row : rows) row = fit[static_cast<std::size_t>(row)];
      const auto [x, y] = ImageBatch(development, rows, cfg.augment ? &augmentation : nullptr);
      optimizer.minimize(TrainingObjective(params, state, model, x, y, cfg, training));
    }
    check::Require(used == fit_count, "each fit image used once per epoch");
    const double train_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch_start).count();
    const auto   result        = EvaluateImages(params, state, model, development, validation, cfg.classes);
    const double loss = result.loss.value(), accuracy = result.accuracy.value();
    if (loss < best) {
      best       = loss;
      best_epoch = epoch;
      params.save(params_path);
      if (state.size()) state.save(state_path);
    }
    history << epoch << '\t' << loss << '\t' << accuracy << '\n';
    history.flush();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch_start).count();
    EpochSeconds().push_back(seconds);
    std::cout << "  epoch " << epoch << ": validation NLL=" << loss << ", accuracy=" << accuracy << " ("
              << std::setprecision(3) << train_seconds << " s train, " << seconds << " s total)" << std::setprecision(6)
              << std::endl;
  }
  params.load(params_path);
  if (state.size()) state.load(state_path);
  check::Near(EvaluateImages(params, state, model, development, validation, cfg.classes).loss.value(), best, 1e-6,
              "best image checkpoint validation loss");
  const auto result = EvaluateImages(params, state, model, test, test_rows, cfg.classes);
  std::cout << "  selected epoch=" << best_epoch << "; test NLL=" << result.loss.value()
            << "; accuracy=" << result.accuracy.value() << "\n  confusion (truth rows):\n";
  std::uint64_t total = 0, correct = 0;
  const auto&   cells = result.confusion.cells();
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
  check::Near(result.accuracy.value(), double(correct) / total, 1e-7, "confusion agrees with accuracy");
  check::TrainingQuality(result.accuracy.value() >= cfg.minimum_test_accuracy, "image classifier demonstration accuracy threshold");
  check::Require(bool(history), "write image training history");
  double train_total = 0;
  for (double s : EpochSeconds()) train_total += s;
  std::printf("  training time %.1f s (%.2f s/epoch)\n", train_total, train_total / cfg.epochs);
  std::cout << "  best checkpoint and history: " << directory.string() << "\nAll " << cfg.name << " checks passed.\n";
}
}  // namespace image_training

// Experiment workflow from examples/nn/common/image_training.hpp.
namespace image_training {
using namespace ::image_training;
inline int RunCifar10(int argc, char** argv, ImageTrainingConfig cfg, const LogitsBuilder& architecture,
                      const TrainingBuilder& training = {}) {
  try {
    std::filesystem::path directory = paths::Dataset("cifar10");
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      const auto        equal    = argument.find('=');
      if (equal == std::string::npos) throw std::invalid_argument("expected key=value option");
      if (!ParseCifarOption(argument.substr(0, equal), argument.substr(equal + 1), directory, cfg))
        throw std::invalid_argument("unknown option: " + argument);
    }
    check::Require(cfg.classes == 10, "CIFAR-10 runner requires ten output classes");
    const auto [development, test] = LoadCifar10(directory);
    TrainImages(development, test, architecture, cfg, training);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
}  // namespace image_training



namespace nn = datamunge::nn;
using mobile_checks::EndsWith;
using mobile_checks::Pattern;
using non_local::Pairwise;

namespace {

using Matrix = std::vector<std::vector<double>>;
// Rows times a [from, to] kernel plus bias.
Matrix Project(const Matrix& rows, const std::vector<float>& w, const std::vector<float>& b, int from, int to) {
  Matrix out(rows.size(), std::vector<double>(to));
  for (std::size_t r = 0; r < rows.size(); ++r)
    for (int j = 0; j < to; ++j) {
      double v = b[j];
      for (int i = 0; i < from; ++i) v += rows[r][i] * w[i * to + j];
      out[r][j] = v;
    }
  return out;
}

void BlockChecks() {
  const int   size = 6, channels = 8, inner = 4;
  const char* labels[4] = {"embedded Gaussian", "Gaussian", "dot product", "concatenation"};
  std::string summary;
  for (int mode = 0; mode < 4; ++mode)
    for (bool subsample : {false, true}) {
      if (subsample && mode != 0) continue;
      const auto         f = Pairwise(mode);
      nn::ParameterStore p(17), state(17);
      nn::Tensor         x     = Pattern({1, size, size, channels}, 1, .2f);
      const auto         value = x.to_vector<float>();
      double             identity = 0;
      {
        const auto fresh = non_local::NonLocalBlock(p, state, "nl", x, f, subsample, false).output.to_vector<float>();
        for (std::size_t i = 0; i < value.size(); ++i) identity = std::max(identity, double(std::abs(fresh[i] - value[i])));
      }
      check::Near(identity, 0, 0, "a new non-local block is the identity");
      {
        nn::NoGradGuard guard;
        for (const auto& name : p.names()) {
          nn::Tensor values = Pattern(p.at(name).shape(),
                                      name.find("/W") != std::string::npos && name.find("concat") == std::string::npos ? .4f : .3f,
                                      float(name.size()));
          if (name == "nl/bn/gamma") values = values + 1.0;
          p.set(name, values);
        }
        for (const auto& name : state.names()) {
          nn::Tensor values = Pattern(state.at(name).shape(), .2f, float(name.size()));
          if (name.find("moving_var") != std::string::npos) values = values + 1.0;
          state.set(name, values);
        }
      }
      x.requires_grad_(true);
      const auto block = non_local::NonLocalBlock(p, state, "nl", x, f, subsample, false);
      const auto support =
          nn::grad(nn::sum(nn::slice(block.output, {0, 0, 0, 0}, {1, 1, 1, channels})), {x})[0].to_vector<float>();
      const auto                                 output = block.output.to_vector<float>();
      std::map<std::string, std::vector<float>> v;
      for (const auto& name : p.names()) v[name] = p.at(name).to_vector<float>();
      const auto mean = state.at("nl/bn/moving_mean").to_vector<float>(), var = state.at("nl/bn/moving_var").to_vector<float>();
      // Host reference.
      Matrix rows(size * size, std::vector<double>(channels)), pooled;
      for (int i = 0; i < size * size; ++i)
        for (int c = 0; c < channels; ++c) rows[i][c] = value[i * channels + c];
      if (subsample) {
        for (int r = 0; r < size; r += 2)
          for (int c = 0; c < size; c += 2) {
            std::vector<double> m(channels, -std::numeric_limits<double>::infinity());
            for (int dr = 0; dr < 2; ++dr)
              for (int dc = 0; dc < 2; ++dc)
                for (int ch = 0; ch < channels; ++ch) m[ch] = std::max(m[ch], rows[(r + dr) * size + c + dc][ch]);
            pooled.push_back(m);
          }
      } else
        pooled = rows;
      const auto g = Project(pooled, v["nl/g/W"], v["nl/g/b"], channels, inner);
      Matrix     a(rows.size(), std::vector<double>(pooled.size()));
      if (f == Pairwise::kGaussian) {
        for (std::size_t i = 0; i < rows.size(); ++i)
          for (std::size_t j = 0; j < pooled.size(); ++j)
            for (int c = 0; c < channels; ++c) a[i][j] += rows[i][c] * pooled[j][c];
      } else {
        const auto theta = Project(rows, v["nl/theta/W"], v["nl/theta/b"], channels, inner),
                   phi   = Project(pooled, v["nl/phi/W"], v["nl/phi/b"], channels, inner);
        for (std::size_t i = 0; i < rows.size(); ++i)
          for (std::size_t j = 0; j < pooled.size(); ++j) {
            if (f == Pairwise::kConcatenation) {
              double z = 0;
              for (int k = 0; k < inner; ++k) z += theta[i][k] * v["nl/concat/W1"][k] + phi[j][k] * v["nl/concat/W2"][k];
              a[i][j] = std::max(z, 0.);
            } else {
              for (int k = 0; k < inner; ++k) a[i][j] += theta[i][k] * phi[j][k];
              if (f == Pairwise::kEmbeddedGaussian) a[i][j] /= std::sqrt(double(inner));
            }
          }
      }
      for (auto& row : a) {
        if (f == Pairwise::kEmbeddedGaussian || f == Pairwise::kGaussian) {
          const double top   = *std::max_element(row.begin(), row.end());
          double       total = 0;
          for (auto& e : row) {
            e = std::exp(e - top);
            total += e;
          }
          for (auto& e : row) e /= total;
        } else
          for (auto& e : row) e /= row.size();
      }
      Matrix y(rows.size(), std::vector<double>(inner, 0));
      for (std::size_t i = 0; i < rows.size(); ++i)
        for (std::size_t j = 0; j < pooled.size(); ++j)
          for (int k = 0; k < inner; ++k) y[i][k] += a[i][j] * g[j][k];
      const auto w     = Project(y, v["nl/W/W"], v["nl/W/b"], inner, channels);
      double     error = 0;
      for (std::size_t i = 0; i < rows.size(); ++i)
        for (int c = 0; c < channels; ++c) {
          const double bn = (w[i][c] - mean[c]) / std::sqrt(var[c] + 1e-5) * v["nl/bn/gamma"][c] + v["nl/bn/beta"][c];
          error           = std::max(error, std::abs(output[i * channels + c] - (bn + rows[i][c])));
        }
      int reached = 0;
      for (int i = 0; i < size * size; ++i) {
        double m = 0;
        for (int c = 0; c < channels; ++c) m += std::abs(support[i * channels + c]);
        reached += m > 0;
      }
      check::Near(error, 0, 2e-5, std::string("non-local block: ") + labels[mode]);
      // Max-pooling passes derivatives only to window maxima; a concatenation
      // pair whose ReLU is off contributes none.
      if (!subsample && f != Pairwise::kConcatenation)
        check::Require(reached == size * size, "one output position depends on every input position");
      std::ostringstream e;
      e << error;
      summary += std::string("\n    ") + labels[mode] +
                 (subsample ? ", subsampled (36 x 9 affinities)" : " (36 x 36 affinities)") + ": error " + e.str() +
                 ", output 0 depends on " + std::to_string(reached) + " of 36 positions";
    }
  std::cout << "  non-local blocks on 6x6x8 (identity when new; a 3x3 convolution would reach 9 positions):" << summary
            << std::endl;
}

/// tflow's cfg.non_local: embedded-Gaussian blocks after alternate blocks of res3 and res4.
attention_cnn::AfterBlockHook NonLocalHook(double momentum, bool update_stats, bool subsample = true) {
  return [=](nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name, const nn::Tensor& h, int stage,
             int block, bool training) {
    if ((stage == 1 || stage == 2) && block % 2 == 0)
      return non_local::NonLocalBlock(p, state, name + "/non_local", h, Pairwise::kEmbeddedGaussian, subsample, training,
                                      momentum, update_stats)
          .output;
    return h;
  };
}

void FullChecks() {
  nn::ParameterStore                   p(17), state(17);
  attention_cnn::AttentionResNetConfig cfg;
  cfg.update_stats = false;
  cfg.after_block  = NonLocalHook(cfg.momentum, false);
  {
    nn::NoGradGuard guard;
    attention_cnn::AttentionResNet(p, state, nn::zeros({1, 64, 64, 3}), cfg);  // create the parameters
  }
  check::Require(p.count() == 32915496, "ResNet-50 with five non-local blocks");
  {
    // Activate the blocks (BN scale one) so their inner layers receive derivatives.
    nn::NoGradGuard guard;
    for (const auto& name : p.names()) {
      if (name.find("non_local/bn/gamma") != std::string::npos)
        p.set(name, nn::ones(p.at(name).shape()));
      else if (EndsWith(name, "/beta"))
        p.set(name, Pattern(p.at(name).shape(), .5f, float(name.size())));
    }
  }
  const auto net   = attention_cnn::AttentionResNet(p, state, Pattern({2, 224, 224, 3}, 1, 0), cfg, true);
  const auto norms = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  // Embedded Gaussian: phi's bias shifts every logit of a softmax row by
  // theta_i . b, g's bias passes through rows summing to one, and W_z's bias is
  // per-channel; the training-mode BN removes all three exactly.
  double      minimum = std::numeric_limits<double>::infinity(), maximum = 0, removed = 0;
  int         removed_count = 0;
  std::string smallest;
  for (double norm : norms) maximum = std::max(maximum, norm);
  for (std::size_t i = 0; i < norms.size(); ++i) {
    const auto& name = p.names()[i];
    check::Require(std::isfinite(norms[i]), "finite non-local ResNet derivative");
    if (EndsWith(name, "non_local/phi/b") || EndsWith(name, "non_local/g/b") || EndsWith(name, "non_local/W/b")) {
      removed = std::max(removed, norms[i]);
      ++removed_count;
    } else if (norms[i] < minimum) {
      minimum  = norms[i];
      smallest = name;
    }
  }
  check::Require(removed_count == 15 && removed < 1e-7 * maximum, "vanishing non-local bias derivatives");
  check::Require(minimum > 1e-6 * maximum, "clearly nonzero non-local ResNet derivatives");
  std::cout << "  ResNet-50 + 5 non-local blocks: " << p.count()
            << " parameters; training-mode gradient norms with active blocks [" << minimum << ',' << maximum
            << "] (smallest " << smallest << "); the 15 phi, g and W_z bias derivatives vanish (largest " << removed << ")"
            << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i)  // threads= also applies to the checks
      if (std::string(argv[i]).rfind("threads=", 0) == 0) nn::set_num_threads(std::stoi(std::string(argv[i]).substr(8)));
    BlockChecks();
    FullChecks();
    image_training::ImageTrainingConfig cfg;
    cfg.name                  = "nonlocal_resnet50_compact_cifar10";
    cfg.maximum_fit_rows      = 12000;
    cfg.epochs                = 12;
    cfg.minimum_test_accuracy = .4;
    const image_training::LogitsBuilder builder = [](nn::ParameterStore& p, nn::ParameterStore& state,
                                                     const nn::Tensor& image, bool training) {
      attention_cnn::AttentionResNetConfig model;
      model.classes       = 10;
      model.width_divisor = 4;
      model.small_stem    = true;
      model.stride_on_3x3 = false;
      model.after_block   = NonLocalHook(model.momentum, true);
      return attention_cnn::AttentionResNet(p, state, image, model, training).logits;
    };
    if (image_training::RunCifar10(argc, argv, cfg, builder)) return 1;
    std::cout << "All non-local checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
