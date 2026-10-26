// CMD: just run-nn mlp_mnist
// A multilayer perceptron on the original MNIST handwritten digits.
//
// Port of tflow's mlp_mnist. Preparation (in ~/projects/tflow_examples):
// python3 scripts/fetch_datasets.py mnist --split both; the data are read from
// $DATAMUNGE_DATA_DIR/mnist (default ~/projects/tflow_examples/output/datasets).
// Architecture: flattened 784 pixels -> 128 ReLU -> 64 ReLU -> 10 logits,
// with 109,386 learned parameters, Adam and small L2 weight regularization.
// Reuse the softmax example's 55k/5k fit/validation split, keep the lowest
// validation cross-entropy checkpoint, and evaluate all 10k test images once.
// Print the confusion matrix and save the checkpoint (.npz) and validation
// history under output/nn/models/mlp_mnist. Require at least 96% test accuracy
// at the default training budget. No augmentation or access to test data
// during training. Options: data=PATH epochs=12 batch=256 lr=.001 threads=N
// (tflow's device= option is dropped; nn runs on the CPU).

#include "common/mnist.hpp"

int main(int argc, char** argv) {
  {
    // Dataset setup, batching, fitting and checkpoint selection are explicit here.
    using namespace mnist;
    const std::string               name                  = "mlp_mnist";
    const std::vector<std::int64_t> hidden                = {128, 64};
    const int                       default_epochs        = 12;
    const double                    default_rate          = .001;
    const double                    minimum_test_accuracy = .96;
    LogitsBuilder                   builder               = {};

    try {
      std::filesystem::path dir    = paths::Dataset("mnist");
      int                   epochs = default_epochs, batch = 256;
      double                rate = default_rate;
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
        else if (key == "threads")
          nn::set_num_threads(std::stoi(value));
        else
          throw std::invalid_argument("unknown argument: " + key);
      }
      check::Require(
          epochs > 0 && batch > 0 && std::isfinite(rate) && rate > 0,
          "positive epochs, batch and learning rate required");
      batch  = example_runtime::Limit(batch, 16);
      example_runtime::Describe();
      check::Require(
          std::filesystem::exists(dir / "train-images-idx3-ubyte") ||
              std::filesystem::exists(dir / "train-images-idx3-ubyte.gz"),
          "MNIST missing in " + dir.string() +
              "; run python3 scripts/fetch_datasets.py mnist --split both in tflow_examples, or "
              "set data=");
      const auto train = nn::load_mnist(dir.string(), "train");
      const auto test  = nn::load_mnist(dir.string(), "test");
      check::Require(
          train.images.shape() == nn::Shape({60000, 28, 28, 1}) &&
              test.images.shape() == nn::Shape({10000, 28, 28, 1}),
          "official MNIST dimensions required");
      const nn::Tensor train_x = nn::reshape(train.images, {60000, 784}),
                       test_x  = nn::reshape(test.images, {10000, 784});

      std::vector<std::size_t> order(60000);
      std::iota(order.begin(), order.end(), 0);
      std::mt19937_64 split_rng(2026);
      std::shuffle(order.begin(), order.end(), split_rng);
      std::vector<std::int64_t> fit_rows(order.begin(), order.begin() + 55000),
          validation_rows(order.begin() + 55000, order.end());
      fit_rows.resize(example_runtime::Limit<std::size_t>(fit_rows.size(), 64));
      validation_rows.resize(example_runtime::Limit<std::size_t>(validation_rows.size(), 32));
      std::vector<std::int64_t> test_rows(example_runtime::Limit<std::size_t>(10000, 32));
      std::iota(test_rows.begin(), test_rows.end(), 0);

      if (!builder)
        builder = [hidden](nn::ParameterStore& store, const nn::Tensor& x, bool) {
          return nn::dense(store, "output", nn::mlp(store, "hidden", x, hidden), 10);
        };
      const double       l2 = 1e-4;
      nn::ParameterStore store(17);
      {
        nn::NoGradGuard guard; // create every parameter before building the optimizer
        builder(store, nn::slice_axis(train_x, 0, 0, 1), true);
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
          const nn::Tensor x = nn::take_rows(train_x, ids), y = nn::take_rows(train.labels, ids);
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
        const auto   validation = Evaluate(store, builder, train_x, train.labels, validation_rows);
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
      const auto restored = Evaluate(store, builder, train_x, train.labels, validation_rows);
      check::Near(
          restored.loss.value(), best, 1e-7, "best checkpoint matches selected validation loss");
      const auto result = Evaluate(store, builder, test_x, test.labels, test_rows);
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
      check::TrainingQuality(
          result.accuracy.value() >= minimum_test_accuracy,
          "classifier did not reach demonstration accuracy threshold");
      check::Require(bool(history), "write training history");
      std::printf("  training time %.1f s (%.2f s/epoch)\n", train_seconds, train_seconds / epochs);
      std::cout << "  best checkpoint and history: " << model_dir.string() << "\nAll " << name
                << " checks passed.\n";
      return 0;
    } catch (const std::exception& e) {
      std::cerr << e.what() << '\n';
      return 1;
    }
  }
}
