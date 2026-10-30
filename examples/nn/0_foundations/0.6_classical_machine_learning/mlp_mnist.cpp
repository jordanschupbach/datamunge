// CMD: just run-nn mlp_mnist
// A multilayer perceptron on the original MNIST handwritten digits.
//
// Port of tflow's mlp_mnist. Preparation (in ~/projects/tflow_examples):
// dm::datasets::fetch("mnist"); the data are read from
// $DATAMUNGE_DATA_DIR/mnist (default ~/projects/tflow_examples/output/datasets).
// Architecture: flattened 784 pixels -> 256 ReLU -> 128 ReLU -> 10 logits,
// with 235,146 learned parameters, Adam and small L2 weight regularization.
// Pool the official train/test files, shuffle with seed 2026, and use a
// configurable proportion of all 70k images (default: 1.0). Split that subset
// into 40% training, 40% testing and 20% validation by default. Keep the lowest
// validation cross-entropy checkpoint and evaluate the held-out test rows once.
// Print the confusion matrix and save the checkpoint (.npz) and validation
// history under output/nn/models/mlp_mnist. Require at least 96% test accuracy
// at the default training budget. No augmentation or access to test data
// during training. Options: data=PATH epochs=1000 batch=1024 lr=.0001 threads=N
// train_proportion=.4 test_proportion=.4 validation_proportion=.2 dataset_proportion=1
// Split proportions must be positive and sum to 1; dataset_proportion is in [0,1]
// and must select enough images for three nonempty splits. Counts round down
// for training/testing, with the remainder assigned to validation. Quick mode
// caps the selected subset at 2000 images before splitting; use
// `just run-nn-full mlp_mnist` to use the entire requested subset.
// After restoring the best checkpoint, explain lrp_images=6 held-out test
// images with epsilon LRP (lrp_epsilon=.001). Save original/relevance pairs
// to lrp.svg beside the checkpoint and print its absolute path at the end.
// (tflow's device= option is dropped; nn runs on the CPU).

#include <datamunge/plot/plot.hpp>

#include "common/dataset_assets.hpp"
#include "common/mnist.hpp"

namespace {

namespace nn   = datamunge::nn;
namespace graphics = datamunge::graphics;

// Epsilon LRP for a dense layer, with z = input * weights + bias:
// R_i = input_i * sum_j(weights_ij * R_j / (z_j + eps * sign(z_j))).
// Treat sign(0) as +1. Bias and the stabilizer absorb relevance, so input
// relevance need not sum exactly to the explained logit. ReLUs pass relevance
// through unchanged; inactive units contribute zero through their activation.
// Bach et al. (2015), https://doi.org/10.1371/journal.pone.0130140, Eq. 58.
nn::Tensor DenseRelevance(
    const nn::Tensor& input,
    const nn::Tensor& weights,
    const nn::Tensor& preactivation,
    const nn::Tensor& relevance,
    double            epsilon) {
  const auto stabilizer = nn::where(
      nn::greater_equal(preactivation, 0.0), nn::scalar_like(epsilon, preactivation), -epsilon);
  return input * nn::matmul(relevance / (preactivation + stabilizer), nn::transpose(weights));
}

struct Explanation {
  nn::Tensor logits, relevance;
};

// Replay this example's dense/ReLU architecture using the restored parameters.
// Keep each layer's input and preactivation for reverse relevance propagation.
Explanation ExplainMlp(
    const nn::ParameterStore&        store,
    const nn::Tensor&                images,
    const std::vector<std::int64_t>& hidden,
    double                           epsilon) {
  nn::NoGradGuard         guard;
  std::vector<nn::Tensor> activations{images}, weights, preactivations;
  for (std::size_t i = 0; i <= hidden.size(); ++i) {
    const bool output = i == hidden.size();
    const auto name   = output ? std::string("output") : "hidden/dense" + std::to_string(i);
    weights.push_back(store.at(name + "/W"));
    const auto z =
        nn::bias_add(nn::matmul(activations.back(), weights.back()), store.at(name + "/b"));
    preactivations.push_back(z);
    activations.push_back(output ? z : nn::relu(z));
  }
  const auto logits = activations.back();
  // Explain each image's predicted pre-softmax class score, not its label.
  auto relevance =
      logits * nn::one_hot(nn::argmax(logits, 1), logits.size(1), 1, 0, logits.dtype());
  for (std::size_t i = weights.size(); i-- > 0;)
    relevance = DenseRelevance(activations[i], weights[i], preactivations[i], relevance, epsilon);
  check::Require(nn::all(nn::is_finite(relevance)).item<bool>(), "finite LRP pixel relevance");
  return {logits, relevance};
}

graphics::RPlot
PixelPanel(const std::vector<double>& pixels, const std::string& title, bool relevance) {
  auto panel = graphics::RPlot::create();
  // Compensate for the plot margins and caption space to keep pixels square.
  panel.size(360, 454).title(title).hide_axes().show_grid(false).x_limits(0, 28).y_limits(-6, 28);
  double scale = 0;
  for (const auto value : pixels)
    scale = std::max(scale, std::abs(value));
  for (std::size_t i = 0; i < pixels.size(); ++i) {
    graphics::RGB color;
    if (relevance) {
      const double value = scale > 0 ? pixels[i] / scale : 0;
      const auto   fade  = static_cast<int>(std::lround(255 * (1 - std::abs(value))));
      color              = value >= 0 ? graphics::RGB{255, fade, fade} : graphics::RGB{fade, fade, 255};
    } else {
      const auto gray = static_cast<int>(std::lround(255 * std::clamp(pixels[i], 0.0, 1.0)));
      color           = {gray, gray, gray};
    }
    const double x = i % 28, y = 27 - i / 28;
    panel.polygon({x, x + 1, x + 1, x}, {y, y, y + 1, y + 1}, color);
  }
  panel.text(
      14,
      -2,
      relevance ? "red: positive / blue: negative" : "MNIST input (28 x 28)",
      {17, 24, 39},
      12);
  panel.text(
      14,
      -4.5,
      relevance ? "white: zero; scaled per image" : "held-out test image",
      {17, 24, 39},
      12);
  return panel;
}

void SaveRelevancePlot(
    nn::ParameterStore&              store,
    const mnist::LogitsBuilder&      builder,
    const std::vector<std::int64_t>& hidden,
    const nn::Tensor&                images,
    const nn::Tensor&                labels,
    const std::vector<std::int64_t>& test_rows,
    int                              requested_images,
    double                           epsilon,
    const std::filesystem::path&     path) {
  nn::NoGradGuard guard;
  const auto      count = std::min(test_rows.size(), static_cast<std::size_t>(requested_images));
  const std::vector<std::int64_t> rows(test_rows.begin(), test_rows.begin() + count);
  const auto                      x           = nn::take_rows(images, rows);
  const auto                      explanation = ExplainMlp(store, x, hidden, epsilon);
  check::Near(
      nn::max(nn::abs(explanation.logits - builder(store, x, false))).item(),
      0,
      1e-6,
      "LRP forward pass matches the restored classifier");
  const auto truth       = nn::take_rows(labels, rows).to_vector<std::int64_t>();
  const auto predicted   = nn::argmax(explanation.logits, 1).to_vector<std::int64_t>();
  const auto columns     = std::min<std::size_t>(count, 3);
  const auto layout_rows = (count + columns - 1) / columns;
  auto       figure      = graphics::RLayout::create(layout_rows, 2 * columns);
  figure.size(720 * columns, 454 * layout_rows);
  for (std::size_t i = 0; i < count; ++i) {
    const auto title = "row " + std::to_string(rows[i]) + ": true " + std::to_string(truth[i]) +
                       ", predicted " + std::to_string(predicted[i]);
    figure.add(PixelPanel(nn::select(x, 0, i).to_vector<double>(), title, false));
    figure.add(PixelPanel(
        nn::select(explanation.relevance, 0, i).to_vector<double>(),
        "LRP: predicted " + std::to_string(predicted[i]),
        true));
  }
  figure.save_svg(path.string());
}

} // namespace

int main(int argc, char** argv) {
  {
    // Dataset setup, batching, fitting and checkpoint selection are explicit here.
    using namespace mnist;
    const std::string name = "mlp_mnist";
    // const std::vector<std::int64_t> hidden                = {128, 64};
    const std::vector<std::int64_t> hidden                = {256, 128};
    const int                       default_epochs        = 1000;
    const double                    default_rate          = .0001;
    const double                    minimum_test_accuracy = .96;
    LogitsBuilder                   builder               = {};

    try {
      std::filesystem::path dir    = paths::Dataset("mnist");
      int                   epochs = default_epochs, batch = 1024;
      double                rate = default_rate;
      // Edit these proportions here, or override them with key=value arguments.
      double train_proportion = 0.40, test_proportion = 0.40, validation_proportion = 0.20;
      double dataset_proportion = 1.0;
      int    lrp_images         = 6;
      double lrp_epsilon        = 0.001;
      for (int i = 1; i < argc; ++i) {
        const std::string arg   = argv[i];
        const auto        equal = arg.find('=');
        if (equal == std::string::npos)
          throw std::invalid_argument("expected key=value argument");
        const auto key = arg.substr(0, equal), value = arg.substr(equal + 1);
        if (key == "data")
          dir = value;
        else if (key == "epochs")
          epochs = std::stoi(value);
        else if (key == "batch")
          batch = std::stoi(value);
        else if (key == "lr")
          rate = std::stod(value);
        else if (key == "train_proportion")
          train_proportion = std::stod(value);
        else if (key == "test_proportion")
          test_proportion = std::stod(value);
        else if (key == "validation_proportion")
          validation_proportion = std::stod(value);
        else if (key == "dataset_proportion")
          dataset_proportion = std::stod(value);
        else if (key == "lrp_images")
          lrp_images = std::stoi(value);
        else if (key == "lrp_epsilon")
          lrp_epsilon = std::stod(value);
        else if (key == "threads")
          nn::set_num_threads(std::stoi(value));
        else
          throw std::invalid_argument("unknown argument: " + key);
      }
      check::Require(
          epochs > 0 && batch > 0 && std::isfinite(rate) && rate > 0,
          "positive epochs, batch and learning rate required");
      check::Require(
          lrp_images > 0 && std::isfinite(lrp_epsilon) && lrp_epsilon > 0,
          "positive lrp_images and finite positive lrp_epsilon required");
      for (const double proportion : {train_proportion, test_proportion, validation_proportion})
        check::Require(
            std::isfinite(proportion) && proportion > 0 && proportion < 1,
            "train, test and validation proportions must be finite and in (0,1)");
      check::Require(
          std::abs(train_proportion + test_proportion + validation_proportion - 1.0) <= 1e-9,
          "train, test and validation proportions must sum to 1");
      check::Require(
          std::isfinite(dataset_proportion) && dataset_proportion >= 0 && dataset_proportion <= 1,
          "dataset_proportion must be finite and in [0,1]");
      batch = example_runtime::Limit(batch, 16);
      example_runtime::Describe();
      dataset_assets::EnsureMnist(dir.string());
      const auto train = dataset_assets::LoadMnist(dir.string(), "train");
      const auto test  = dataset_assets::LoadMnist(dir.string(), "test");
      check::Require(
          train.images.shape() == nn::Shape({60000, 28, 28, 1}) &&
              test.images.shape() == nn::Shape({10000, 28, 28, 1}),
          "official MNIST dimensions required");
      const std::size_t total_rows    = 70000;
      const std::size_t selected_rows = example_runtime::Limit<std::size_t>(
          static_cast<std::size_t>(total_rows * dataset_proportion), 2000);
      const auto fit_count  = static_cast<std::size_t>(selected_rows * train_proportion);
      const auto test_count = static_cast<std::size_t>(selected_rows * test_proportion);
      check::Require(
          fit_count > 0 && test_count > 0 && fit_count + test_count < selected_rows,
          "dataset_proportion must select enough images for nonempty train, test and validation "
          "splits");
      const nn::Tensor images =
          nn::reshape(nn::concat({train.images, test.images}, 0), {70000, 784});
      const nn::Tensor labels = nn::concat({train.labels, test.labels}, 0);

      std::vector<std::int64_t> order(total_rows);
      std::iota(order.begin(), order.end(), 0);
      std::mt19937_64 split_rng(2026);
      std::shuffle(order.begin(), order.end(), split_rng);
      order.resize(selected_rows);
      const auto test_begin       = order.begin() + static_cast<std::ptrdiff_t>(fit_count);
      const auto validation_begin = test_begin + static_cast<std::ptrdiff_t>(test_count);
      const std::vector<std::int64_t> fit_rows(order.begin(), test_begin),
          test_rows(test_begin, validation_begin), validation_rows(validation_begin, order.end());

      if (!builder)
        builder = [hidden](nn::ParameterStore& store, const nn::Tensor& x, bool) {
          return nn::dense(store, "output", nn::mlp(store, "hidden", x, hidden), 10);
        };
      const double       l2 = 1e-4;
      nn::ParameterStore store(17);
      {
        nn::NoGradGuard guard; // create every parameter before building the optimizer
        builder(store, nn::slice_axis(images, 0, 0, 1), true);
      }
      nn::OptimizerConfig config;
      config.kind = nn::OptimizerKind::Adam;
      nn::Optimizer           optimizer(store, rate, config);
      std::vector<nn::Tensor> weights;
      for (const auto& p : store.parameters())
        if (p.dim() >= 2)
          weights.push_back(p);

      const auto    model_dir  = paths::OutputDir("models/" + name);
      const auto    checkpoint = (model_dir / "best.npz").string();
      std::ofstream history(model_dir / "history.tsv");
      check::Require(bool(history), "open training history");
      history << "epoch\tvalidation_loss\tvalidation_accuracy\n";
      std::cout << "== " << name << ": " << fit_rows.size() << " fit / " << validation_rows.size()
                << " validation / " << test_rows.size() << " test\n"
                << "  parameters=" << store.count() << "; batch=" << batch << "; lr=" << rate
                << '\n';

      nn::BatchOrder batches(fit_rows.size(), static_cast<std::size_t>(batch), 29);
      double         best       = std::numeric_limits<double>::infinity();
      int            best_epoch = -1;
      const auto     started    = std::chrono::steady_clock::now();
      for (int epoch = 1; epoch <= epochs; ++epoch) {
        const auto epoch_start = std::chrono::steady_clock::now();
        batches.begin_epoch();
        std::size_t samples = 0;
        for (;;) {
          auto ids = batches.next();
          if (ids.empty())
            break;
          samples += ids.size();
          for (auto& index : ids)
            index = fit_rows[static_cast<std::size_t>(index)];
          const nn::Tensor x = nn::take_rows(images, ids), y = nn::take_rows(labels, ids);
          nn::Tensor       objective = nn::sparse_softmax_cross_entropy(builder(store, x, true), y);
          if (l2 > 0) {
            nn::Tensor penalty = nn::sum(nn::square(weights[0]));
            for (std::size_t i = 1; i < weights.size(); ++i)
              penalty = penalty + nn::sum(nn::square(weights[i]));
            objective = objective + 0.5 * l2 * penalty;
          }
          optimizer.minimize(objective);
        }
        check::Require(
            samples == fit_rows.size(), "each selected training row used once per epoch");
        const auto   validation = Evaluate(store, builder, images, labels, validation_rows);
        const double loss = validation.loss.value(), accuracy = validation.accuracy.value();
        if (loss < best) {
          best       = loss;
          best_epoch = epoch;
          store.save(checkpoint);
        }
        history << epoch << '\t' << loss << '\t' << accuracy << '\n';
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch_start).count();
        std::cout << "  epoch " << epoch << ": validation NLL=" << loss << ", accuracy=" << accuracy
                  << " (" << std::setprecision(3) << seconds << std::setprecision(6) << " s)"
                  << std::endl;
      }
      const double train_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      store.load(checkpoint);
      const auto restored = Evaluate(store, builder, images, labels, validation_rows);
      check::Near(
          restored.loss.value(), best, 1e-7, "best checkpoint matches selected validation loss");
      const auto result = Evaluate(store, builder, images, labels, test_rows);
      std::cout << "  selected epoch=" << best_epoch << "; test NLL=" << result.loss.value()
                << "; accuracy=" << result.accuracy.value() << "\n  test confusion (truth rows):\n";
      std::uint64_t total = 0, correct = 0;
      const auto&   cells = result.confusion.cells();
      for (std::size_t i = 0; i < 10; ++i) {
        std::cout << "   ";
        for (std::size_t j = 0; j < 10; ++j) {
          std::cout << ' ' << std::setw(4) << cells[i][j];
          total += cells[i][j];
        }
        correct += cells[i][i];
        std::cout << '\n';
      }
      check::Require(total == test_rows.size(), "all selected test images evaluated");
      check::Near(
          result.accuracy.value(),
          double(correct) / test_rows.size(),
          1e-7,
          "confusion matches accuracy");
      const auto relevance_path = std::filesystem::absolute(model_dir / "lrp.svg");
      SaveRelevancePlot(
          store,
          builder,
          hidden,
          images,
          labels,
          test_rows,
          lrp_images,
          lrp_epsilon,
          relevance_path);
      check::Require(bool(history), "write training history");
      std::printf("  training time %.1f s (%.2f s/epoch)\n", train_seconds, train_seconds / epochs);
      std::cout << "  best checkpoint and history: " << model_dir.string()
                << "\n  LRP plot (restored best checkpoint): " << relevance_path.string() << '\n';
      check::TrainingQuality(
          result.accuracy.value() >= minimum_test_accuracy,
          "classifier did not reach demonstration accuracy threshold");
      std::cout << "All " << name << " checks passed.\n";
      return 0;
    } catch (const std::exception& e) {
      std::cerr << e.what() << '\n';
      return 1;
    }
  }
}
