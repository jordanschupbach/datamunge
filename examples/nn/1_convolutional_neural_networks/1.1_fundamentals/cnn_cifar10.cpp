// CMD: just run-nn cnn_cifar10
// A small CIFAR-10 CNN trained with independent random crops and flips.
//
// Port of tflow's cnn_cifar10. Preparation (in ~/projects/tflow_examples):
// python3 scripts/fetch_datasets.py cifar10 --split both.
// Architecture: Conv(32,3)-ReLU-pool -> Conv(64,3)-ReLU-pool -> Dense(128)-
// ReLU -> 10 logits. Train on 45k original images with four-pixel zero padding,
// random 32x32 crops and horizontal flips; normalize RGB pixels to [-1,1].
// Select by NLL on 5k untouched validation images, restore that checkpoint,
// then evaluate all 10k original test images once. Adam/L2; no pretrained data.
// Options: data=PATH epochs=10 batch=128 lr=.001 augment=0|1 threads=N.
// The default run requires 55% test accuracy as a training sanity check.
// Checkpoint/history: output/nn/models/cnn_cifar10/ (or cnn_cifar10_no_augmentation).

#include <iostream>

#include "common/image_training.hpp"

namespace nn = datamunge::nn;

void AugmentationChecks() {
  nn::ImageDataset fixture;
  fixture.images = nn::ones({2, 32, 32, 3});
  fixture.labels = nn::from_vector(std::vector<std::int64_t>{2, 7}, {2});
  nn::Generator first(151), repeat(151);
  const auto    plain      = image_training::ImageBatch(fixture, {1, 0});
  const auto    augmented  = image_training::ImageBatch(fixture, {1, 0}, &first);
  const auto    reproduced = image_training::ImageBatch(fixture, {1, 0}, &repeat);
  const auto    p = plain.first.to_vector<float>(), a = augmented.first.to_vector<float>(),
             r = reproduced.first.to_vector<float>();
  std::size_t padded = 0, interior = 0;
  for (std::size_t i = 0; i < p.size(); ++i) {
    check::Near(p[i], 1, 0, "evaluation only normalizes pixels");
    check::Near(a[i], r[i], 0, "seeded per-image augmentation");
    check::Require(a[i] == -1 || a[i] == 1, "crop preserves white image or zero padding");
    padded += a[i] == -1;
    interior += a[i] == 1;
  }
  check::Require(padded > 0 && interior > 0, "random crop includes padded and original pixels");
  const auto labels = augmented.second.to_vector<std::int64_t>();
  check::Require(labels[0] == 7 && labels[1] == 2, "augmentation preserves labels and row order");
  std::cout << "  augmentation checks: reproducible crops, normalized padding and paired labels passed\n";
}

int main(int argc, char** argv) {
  try {
    std::filesystem::path                       dir = paths::Dataset("cifar10");
    image_training::ImageTrainingConfig cfg;
    cfg.name = "cnn_cifar10";
    for (int i = 1; i < argc; ++i) {
      const std::string arg   = argv[i];
      const auto        equal = arg.find('=');
      if (equal == std::string::npos) throw std::invalid_argument("expected key=value argument");
      if (!image_training::ParseCifarOption(arg.substr(0, equal), arg.substr(equal + 1), dir, cfg))
        throw std::invalid_argument("unknown argument: " + arg);
    }
    if (!cfg.augment) cfg.name += "_no_augmentation";
    AugmentationChecks();
    const auto [development, test] = image_training::LoadCifar10(dir);
    auto architecture = [](nn::ParameterStore& p, nn::ParameterStore&, const nn::Tensor& image, bool) {
      nn::ConvLayerOptions options;
      options.activation = nn::Activation::Relu;
      auto first  = nn::max_pool2d(nn::conv2d(p, "conv1", image, 32, 3, options), 2, 2);
      auto second = nn::max_pool2d(nn::conv2d(p, "conv2", first, 64, 3, options), 2, 2);
      auto hidden = nn::dense(p, "dense", nn::reshape(second, {-1, 8 * 8 * 64}), 128, nn::Activation::Relu);
      return nn::dense(p, "output", hidden, 10);
    };
    image_training::TrainImages(development, test, architecture, cfg);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
