// CMD: just run-nn global_pooling_vs_flatten
// Global average pooling vs flatten: invariance, location and learned heads.
//
// Port of tflow's global_pooling_vs_flatten. For [N,H,W,C] feature maps and K
// classes, a GAP-linear head learns C*K+K parameters while flatten-linear
// learns H*W*C*K+K. GAP discards spatial arrangement: every permutation of a
// map's spatial sites has the same logits. Verify uniform spatial input
// gradients and the class-activation-map identity
// logit_k = mean_hw(sum_c(feature_hwc * W_ck)) + bias_k.
// These identities apply to the feature map. Padding and strided layers can
// break translation equivariance of a preceding CNN, so image-level shift
// invariance does not follow automatically.
//
// Two controlled, head-only learning tasks use 8x8 two-channel feature maps.
// Appearance: the active channel is the label; fit half the spatial sites and
// test the other half. GAP shares evidence across space, while a zero-initialized
// flatten head receives no gradient at unseen sites. Location: the label is
// left/right and the active channel is fixed. Fit all sites and test a new
// positive amplitude; GAP's two classes are indistinguishable, while flatten
// retains location. These exact toy tasks isolate head behavior; they are not
// an image benchmark or a comparison of jointly trained feature extractors.
// Both heads use Adam(.05), the same examples and 300 full-batch steps
// (tflow's ml::Classifier with l2=0 is written out here: mean sparse
// cross-entropy, nn::Optimizer Adam over a ParameterStore).

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <tuple>

#include "common/checks.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

void AlgebraChecks() {
  const Tensor weights = nn::from_vector(std::vector<float>{.4f, -.7f, .2f, .3f, -.5f, .6f}, {3, 2});
  const Tensor bias    = nn::from_vector(std::vector<float>{.1f, -.2f}, {2});
  std::vector<float> values(2 * 4 * 5 * 3);
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = std::sin(int(i) * .17f);
  Tensor x = nn::from_vector(values, {2, 4, 5, 3});
  x.requires_grad_();
  auto logits    = nn::bias_add(nn::matmul(nn::global_avg_pool2d(x), weights), bias);
  auto maps      = nn::reshape(nn::matmul(nn::reshape(x, {-1, 3}), weights), {2, 4, 5, 2});
  auto from_maps = nn::bias_add(nn::global_avg_pool2d(maps), bias);
  // Reverse spatial axes is a nontrivial permutation, preserving channel identity.
  auto permuted  = nn::bias_add(nn::matmul(nn::global_avg_pool2d(nn::reverse(x, {1, 2})), weights), bias);
  auto objective = nn::sum(nn::slice_axis(logits, 1, 0, 1));
  const auto gradient = nn::grad(objective, {x})[0].to_vector<float>();
  const auto a = logits.to_vector<float>(), b = from_maps.to_vector<float>(), c = permuted.to_vector<float>();
  for (int i = 0; i < 4; ++i) {
    check::Near(a[i], b[i], 1e-7, "class-activation-map identity");
    check::Near(a[i], c[i], 1e-7, "spatial permutation invariance");
  }
  const auto w = weights.to_vector<float>();
  for (std::size_t i = 0; i < gradient.size(); ++i)
    check::Near(gradient[i], w[(i % 3) * 2] / 20, 1e-8, "uniform spatial GAP gradient");
  std::cout << "  algebra: spatial permutation, class-activation maps and uniform input derivatives passed\n";
}

Tensor Head(nn::ParameterStore& parameters, const Tensor& input, bool global_average) {
  const std::int64_t width    = global_average ? 2 : 128;
  const Tensor       features = global_average ? nn::global_avg_pool2d(nn::reshape(input, {-1, 8, 8, 2})) : input;
  auto               weights  = parameters.get("head/W", {width, 2}, nn::init::zeros());
  auto               bias     = parameters.get("head/b", {2}, nn::init::zeros());
  return nn::bias_add(nn::matmul(features, weights), bias);
}

std::pair<Tensor, Tensor> Dataset(bool location, bool test) {
  // An impulse of amplitude 64 gives one unit in the global average. Test
  // location maps use amplitude 80; appearance uses exactly unseen sites.
  std::vector<std::tuple<int, int, int, int>> examples;
  for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 8; ++x) {
      if (location)
        examples.emplace_back(y, x, 0, x >= 4);
      else if ((y + x) % 2 == int(test))
        for (int label = 0; label < 2; ++label) examples.emplace_back(y, x, label, label);
    }
  std::vector<float>        features(examples.size() * 128, 0.f);
  std::vector<std::int64_t> labels(examples.size());
  for (std::size_t i = 0; i < examples.size(); ++i) {
    const auto [y, x, channel, label]               = examples[i];
    features[i * 128 + (y * 8 + x) * 2 + channel] = location && test ? 80.f : 64.f;
    labels[i]                                       = label;
  }
  const auto n = static_cast<std::int64_t>(examples.size());
  return {nn::from_vector(features, {n, 128}), nn::from_vector(labels, {n})};
}

void LearnHeads() {
  for (bool location : {false, true}) {
    const auto [train_x, train_y] = Dataset(location, false);
    const auto [test_x, test_y]   = Dataset(location, true);
    // Explicitly prove that location labels have identical GAP features.
    if (location) {
      const auto values = train_x.to_vector<float>();
      for (std::int64_t row = 0; row < train_x.size(0); ++row) {
        double channel_zero = 0, channel_one = 0;
        for (int site = 0; site < 64; ++site) {
          channel_zero += values[static_cast<std::size_t>(row * 128 + site * 2)];
          channel_one += values[static_cast<std::size_t>(row * 128 + site * 2 + 1)];
        }
        check::Near(channel_zero, 64, 0, "location examples share GAP feature zero");
        check::Near(channel_one, 0, 0, "location examples share GAP feature one");
      }
    }
    for (bool global_average : {false, true}) {
      nn::ParameterStore model(13);
      {
        nn::NoGradGuard guard;
        Head(model, train_x, global_average);
      }
      check::Require(model.count() == (global_average ? 6 : 258), "GAP vs flatten head parameter count");
      nn::OptimizerConfig config;
      config.kind = nn::OptimizerKind::Adam;
      nn::Optimizer optimizer(model, .05, config);
      for (int step = 0; step < 300; ++step)
        optimizer.minimize(nn::sparse_softmax_cross_entropy(Head(model, train_x, global_average), train_y));
      nn::NoGradGuard guard;
      auto evaluate = [&](const Tensor& x, const Tensor& y) {
        const Tensor logits = Head(model, x, global_average);
        return std::pair<double, double>{nn::sparse_softmax_cross_entropy(logits, y).item(), nn::accuracy(logits, y)};
      };
      const auto training = evaluate(train_x, train_y), test = evaluate(test_x, test_y);
      const double expected = location ? (global_average ? .5 : 1.) : (global_average ? 1. : .5);
      check::Near(test.second, expected, 0, "controlled head generalization result");
      if (global_average && location)
        check::Near(test.first, std::log(2.), 1e-6, "indistinguishable balanced classes have log(2) loss");
      else
        check::Near(training.second, 1, 0, "learnable training task fitted");
      if (!global_average && !location) {
        const auto weights = model.at("head/W").to_vector<float>();
        for (int y = 0; y < 8; ++y)
          for (int x = 0; x < 8; ++x)
            if ((y + x) % 2 == 1)
              for (int channel = 0; channel < 2; ++channel)
                for (int label = 0; label < 2; ++label)
                  check::Near(weights[static_cast<std::size_t>(((y * 8 + x) * 2 + channel) * 2 + label)], 0, 0,
                              "unseen flatten coordinates receive no gradient");
      }
      std::cout << "  " << (location ? "location" : "appearance") << ' ' << (global_average ? "GAP" : "flatten")
                << ": parameters=" << model.count() << "; training accuracy=" << training.second
                << "; test accuracy=" << test.second << "; test NLL=" << test.first << '\n';
      nn::ConfusionMatrix confusion(2);
      confusion.add(test_y, nn::argmax(Head(model, test_x, global_average), 1));
      const auto& cells = confusion.cells();
      check::Near(confusion.accuracy(), test.second, 1e-12, "confusion matches test accuracy");
      std::cout << "    test confusion (truth rows): [" << cells[0][0] << ' ' << cells[0][1] << "; " << cells[1][0]
                << ' ' << cells[1][1] << "]\n";
    }
  }
}

}  // namespace

int main() {
  try {
    AlgebraChecks();
    LearnHeads();
    std::cout << "All global-pooling vs flatten checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
