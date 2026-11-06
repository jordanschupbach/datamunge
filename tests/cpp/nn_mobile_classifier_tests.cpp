#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/nn/mobile_classifier.hpp>
#include <datamunge/nn/nn.hpp>
#include <datamunge/nn/parallel.hpp>
#include <filesystem>
namespace n = datamunge::nn;
TEST(MobileImageClassifier, RealModelLearnsUpdatesAndRestoresMomentsWithWeights) {
  n::set_num_threads(1);
  n::MobileNetClassifier model(2, 16, .0625, 17);
  std::vector<float>     rgb(4 * 16 * 16 * 3);
  for (int row = 0; row < 4; row++)
    for (int px = 0; px < 16 * 16; px++) {
      rgb[(row * 256 + px) * 3]     = .8f;
      rgb[(row * 256 + px) * 3 + 1] = row % 2 ? .8f : .1f;
      rgb[(row * 256 + px) * 3 + 2] = .05f;
    }
  auto x      = n::from_vector(rgb, {4, 16, 16, 3});
  auto y      = n::from_vector(std::vector<std::int64_t>{0, 1, 0, 1});
  auto before = model.predict(x);
  EXPECT_GT(model.parameter_count(), 1000);
  n::MobileClassifierOptions options;
  options.epochs     = 2;
  options.batch_size = 4;
  std::vector<std::size_t> epochs;
  options.progress = [&](const n::MobileClassifierEpoch& epoch) {
    epochs.push_back(epoch.epoch);
    EXPECT_TRUE(std::isfinite(epoch.training_loss));
    EXPECT_TRUE(std::isfinite(epoch.validation_loss));
  };
  auto learned = model.fit(x, y, x, y, options);
  EXPECT_EQ(learned.history.size(), 2);
  EXPECT_EQ(epochs, (std::vector<std::size_t>{1, 2}));
  EXPECT_GE(learned.selected_epoch, 1);
  auto after = model.predict(x);
  EXPECT_GT(n::max(n::abs(after - before)).item(), 1e-8);
  auto path = (std::filesystem::temp_directory_path() / "datamunge-mobile-test.npz").string();
  model.save(path);
  auto loaded = n::MobileNetClassifier::load(path);
  EXPECT_EQ(after.to_vector<float>(), loaded.predict(x).to_vector<float>());
  auto arrays = n::load_npz(path);
  arrays.erase(
      std::remove_if(
          arrays.begin(),
          arrays.end(),
          [](const auto& p) { return p.first.rfind("moments/", 0) == 0; }),
      arrays.end());
  n::save_npz(path, arrays);
  EXPECT_THROW(n::MobileNetClassifier::load(path), std::runtime_error);
  std::filesystem::remove(path);
}
TEST(MobileImageClassifier, InvalidPixelsLabelsAndOptionsFailBeforeWeightMutation) {
  n::set_num_threads(1);
  n::MobileNetClassifier model(2, 16, .0625, 17);
  auto                   x      = n::zeros({2, 16, 16, 3});
  auto                   y      = n::from_vector(std::vector<std::int64_t>{0, 1});
  auto                   before = model.predict(x).to_vector<float>();
  EXPECT_THROW(model.predict(x - .1), std::invalid_argument);
  EXPECT_THROW(model.predict(n::zeros({1, 16, 16, 4})), std::invalid_argument);
  EXPECT_THROW(model.predict(x.to(n::DType::Float64)), std::invalid_argument);
  EXPECT_THROW(
      model.fit(x, n::from_vector(std::vector<std::int64_t>{0, 2}), x, y), std::invalid_argument);
  n::MobileClassifierOptions bad;
  bad.epochs = 0;
  EXPECT_THROW(model.fit(x, y, x, y, bad), std::invalid_argument);
  EXPECT_EQ(before, model.predict(x).to_vector<float>());
  EXPECT_THROW(n::MobileNetClassifier(1, 16, .0625), std::invalid_argument);
  EXPECT_THROW(n::MobileNetClassifier(2, 17, .0625), std::invalid_argument);
}
TEST(MobileImageClassifier, GrayscaleAugmentationIsSeededAndValidatesBeforeMutation) {
  n::set_num_threads(1);
  n::MobileNetClassifier a(2, 16, .0625, 19), b(2, 16, .0625, 19);
  std::vector<float>     colored(2 * 16 * 16 * 3);
  for (std::size_t i = 0; i < colored.size(); ++i)
    colored[i] = (i % 3) * .3f;
  auto                       x = n::from_vector(colored, {2, 16, 16, 3});
  auto                       y = n::from_vector(std::vector<std::int64_t>{0, 1});
  n::MobileClassifierOptions opts;
  opts.epochs                = 1;
  opts.grayscale_probability = 1;
  a.fit(x, y, x, y, opts);
  b.fit(x, y, x, y, opts);
  EXPECT_EQ(a.predict(x).to_vector<float>(), b.predict(x).to_vector<float>());
  auto before                = a.predict(x).to_vector<float>();
  opts.grayscale_probability = 1.1;
  EXPECT_THROW(a.fit(x, y, x, y, opts), std::invalid_argument);
  EXPECT_EQ(before, a.predict(x).to_vector<float>());
}
