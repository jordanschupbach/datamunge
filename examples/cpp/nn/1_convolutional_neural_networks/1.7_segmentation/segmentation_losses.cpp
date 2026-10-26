// CMD: just run-nn segmentation_losses
// Dice, Tversky, Lovasz-Softmax and signed-distance boundary losses.
// https://arxiv.org/abs/1706.05721
// https://arxiv.org/abs/1705.08790
// https://arxiv.org/abs/1812.07032
//
// Port of tflow's segmentation_losses. Check closed-form soft overlap
// values/derivatives, FP/FN weighting, sorted Jaccard increments and every
// Lovasz subgradient, perfect/absent/void classes, detached signed-distance
// derivatives and empty masks. Compare exact linear-time squared Euclidean
// transforms with exhaustive nearest-site search on 100 random rectangular
// grids. Check interior/boundary/exterior signed distances and zero maps for
// classes with no internal boundary.
//
// Optimize four independent 24x24 probability fields for a known circle using
// Dice, Tversky, Lovasz and Dice + .1 * normalized boundary loss, Adam(.1),
// 200 steps. This demonstrates loss optimization, not held-out image
// segmentation. Boundary loss can be negative and supplements a region loss;
// distances are divided by 24 only in this optimization demonstration. Report
// each loss/foreground IoU and export input/truth/prediction PNG. No dataset
// needed. Field logits initialise from datamunge's generator (normal, .25), so
// the loss trajectory matches TF's statistically.
// Options: threads=N.
// Artifacts: output/nn/models/segmentation_loss_fields/preview.png.

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/segmentation.hpp"
#include "common/segmentation_losses.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

void LossChecks() {
  std::vector<float>        p_values = {.2f, .8f, .7f, .3f, .6f, .4f, .9f, .1f};
  std::vector<std::int32_t> y_values = {1, 0, 1, 0};
  const Tensor              distance = nn::from_vector(std::vector<float>{-2, 0, 1, 3}, {1, 1, 4, 1});
  std::vector<double>       values;
  std::vector<std::vector<float>> gradients;
  auto run = [&] {
    Tensor probabilities = nn::from_vector(p_values, {1, 1, 4, 2});
    probabilities.requires_grad_(true);
    const Tensor              labels = nn::from_vector(y_values, {1, 1, 4});
    const std::vector<Tensor> losses = {
        seg::SoftDiceLoss(probabilities, labels, 2), seg::SoftTverskyLoss(probabilities, labels, 2),
        seg::LovaszSoftmaxLoss(probabilities, labels, 2, 1),
        seg::BoundaryLoss(nn::slice(probabilities, {0, 0, 0, 1}, {1, 1, 4, 1}), distance)};
    values.clear();
    gradients.clear();
    for (const auto& loss : losses) {
      values.push_back(loss.item<double>());
      nn::GradOptions options;
      options.allow_unused = true;  // a constant (all-void/absent) loss does not depend on the probabilities
      const Tensor g = loss.requires_grad() ? nn::grad(loss, {probabilities}, options)[0] : Tensor();
      gradients.push_back(g.defined() ? g.to_vector<float>() : std::vector<float>(8, 0.f));
    }
  };
  run();
  constexpr double epsilon = 1e-6;
  check::Near(values[0], 1 - (1.2 + epsilon) / (1.8 + epsilon), 1e-7, "soft Dice overlap value");
  check::Near(values[1], 1 - (1.2 + epsilon) / (1.88 + epsilon), 1e-7, "asymmetric Tversky overlap value");
  check::Near(values[2], 5. / 12, 1e-7, "Lovasz sorted errors [.6,.3,.2,.1] with Jaccard increments [.5,1/6,1/3,0]");
  check::Near(values[3], -.225, 1e-7, "boundary loss may be negative");
  const double lovasz[] = {-1. / 3, 1. / 6, -.5, 0}, boundary[] = {-.5, 0, .25, .75};
  for (int i = 0; i < 4; ++i) {
    const double target = i % 2 == 0 ? 1 : 0;
    for (int which = 0; which < 2; ++which) {
      const double alpha = which ? .3 : .5, beta = which ? .7 : .5, den = which ? 1.88 + epsilon : 1.8 + epsilon,
                   numerator  = 1.2 + epsilon;
      const double derivative = -(target * den - numerator * ((1 - beta) * target + alpha * (1 - target))) / (den * den);
      check::Near(gradients[std::size_t(which)][std::size_t(2 * i + 1)], derivative, 2e-7,
                  "independent overlap quotient-rule derivative");
    }
    check::Near(gradients[2][std::size_t(2 * i + 1)], lovasz[i], 1e-7, "every Lovasz fixed-sort subgradient");
    check::Near(gradients[3][std::size_t(2 * i + 1)], boundary[i], 1e-7, "signed-distance boundary derivative");
    for (int k = 0; k < 4; ++k)
      check::Near(gradients[std::size_t(k)][std::size_t(2 * i)], 0, 0, "background omitted from foreground loss");
  }
  y_values.assign(4, 255);
  run();
  for (int k = 0; k < 3; ++k) {
    check::Near(values[std::size_t(k)], 0, 0, "all-void image has zero region loss");
    for (int i = 0; i < 8; ++i)
      check::Near(gradients[std::size_t(k)][std::size_t(i)], 0, 0, "all-void image has zero region derivatives");
  }
  y_values.assign(4, 0);
  run();
  check::Near(values[2], 0, 0, "Lovasz skips absent foreground class");
  y_values = {1, 0, 1, 0};
  p_values = {0, 1, 1, 0, 0, 1, 1, 0};
  run();
  for (int k = 0; k < 3; ++k) check::Near(values[std::size_t(k)], 0, 0, "perfect probabilities give zero region loss");
  std::cout << "  Dice/Tversky values and analytic gradients; Lovasz sorted subgradients; boundary/absent/perfect/void "
               "cases passed"
            << std::endl;
}

void DistanceChecks() {
  std::mt19937                rng(23);
  std::bernoulli_distribution pick(.2);
  for (int trial = 0; trial < 100; ++trial) {
    const int         h = 1 + trial % 8, w = 1 + (trial / 8) % 9;
    std::vector<bool> seeds(std::size_t(h * w));
    for (int i = 0; i < h * w; ++i) seeds[std::size_t(i)] = trial % 10 ? pick(rng) : false;
    const auto distance = seg::SquaredDistance2D(seeds, h, w);
    for (int r = 0; r < h; ++r)
      for (int c = 0; c < w; ++c) {
        double expected = INFINITY;
        for (int y = 0; y < h; ++y)
          for (int x = 0; x < w; ++x)
            if (seeds[std::size_t(y * w + x)]) expected = std::min(expected, double((r - y) * (r - y) + (c - x) * (c - x)));
        if (std::isfinite(expected))
          check::Near(distance[std::size_t(r * w + c)], expected, 1e-12, "linear-time EDT equals exhaustive nearest site");
        else
          check::Require(std::isinf(distance[std::size_t(r * w + c)]), "no-seed EDT is infinite");
      }
  }
  std::vector<std::int32_t> labels(5 * 7, 0);
  for (int r = 1; r < 4; ++r)
    for (int c = 2; c < 5; ++c) labels[std::size_t(r * 7 + c)] = 1;
  auto distance = seg::SignedBoundaryDistances(nn::from_vector(labels, {1, 5, 7}), 2).to_vector<float>();
  check::Near(distance[2 * 7 + 3], -1, 0, "interior distance uses -(EDT-1)");
  check::Near(distance[1 * 7 + 2], 0, 0, "foreground boundary distance zero");
  check::Near(distance[0], std::sqrt(5.), 1e-7, "exterior Euclidean distance");
  check::Near(distance[2 * 7 + 0], 2, 0, "exterior horizontal distance");
  for (int fill : {0, 1}) {
    distance = seg::SignedBoundaryDistances(nn::from_vector(std::vector<std::int32_t>(35, fill), {1, 5, 7}), 2)
                   .to_vector<float>();
    for (float v : distance)
      check::Near(v, 0, 0, fill ? "full foreground has no internal boundary" : "empty foreground boundary map zero");
  }
  std::cout << "  100 exact EDT/exhaustive rectangular comparisons and signed interior/boundary/exterior/empty/full maps "
               "passed"
            << std::endl;
}

void OptimizeFields() {
  constexpr int             side = 24;
  std::vector<std::int32_t> label_values(side * side);
  for (int r = 0; r < side; ++r)
    for (int c = 0; c < side; ++c)
      label_values[std::size_t(r * side + c)] = ((r - 12) * (r - 12) + (c - 12) * (c - 12) < 49) ? 1 : 0;
  const Tensor target   = nn::from_vector(label_values, {1, side, side});
  const Tensor distance = seg::SignedBoundaryDistances(target, 2) / double(side);
  nn::ParameterStore p(17);
  for (int i = 0; i < 4; ++i) p.get("field" + std::to_string(i), {1, side, side, 2}, nn::init::normal(.25));
  auto probabilities = [&] {
    std::vector<Tensor> out;
    for (int i = 0; i < 4; ++i) out.push_back(nn::softmax(p.at("field" + std::to_string(i)), -1));
    return out;
  };
  auto losses = [&](const std::vector<Tensor>& prob) {
    return std::vector<Tensor>{
        seg::SoftDiceLoss(prob[0], target, 2), seg::SoftTverskyLoss(prob[1], target, 2),
        seg::LovaszSoftmaxLoss(prob[2], target, 2, 1),
        seg::SoftDiceLoss(prob[3], target, 2) +
            .1 * seg::BoundaryLoss(nn::slice(prob[3], {0, 0, 0, 1}, {1, side, side, 1}), distance)};
  };
  nn::OptimizerConfig config;
  config.kind = nn::OptimizerKind::Adam;
  nn::Optimizer       optimizer(p, .1, config);
  std::vector<double> initial;
  {
    nn::NoGradGuard guard;
    for (const auto& loss : losses(probabilities())) initial.push_back(loss.item<double>());
  }
  for (int step = 1; step <= 200; ++step) {
    const auto terms = losses(probabilities());
    optimizer.minimize(terms[0] + terms[1] + terms[2] + terms[3]);
    if (step % 50 == 0) {
      nn::NoGradGuard guard;
      std::cout << "  step" << step << " losses:";
      for (const auto& loss : losses(probabilities())) std::cout << ' ' << loss.item<double>();
      std::cout << std::endl;
    }
  }
  nn::NoGradGuard guard;
  const auto      prob  = probabilities();
  const auto      final = losses(prob);
  const char*     titles[] = {"Dice", "Tversky", "Lovasz", "Dice+boundary"};
  for (int field = 0; field < 4; ++field) {
    const auto predictions = nn::argmax(prob[std::size_t(field)], 3).to_vector<std::int64_t>();
    int        intersection = 0, combined = 0;
    for (int i = 0; i < side * side; ++i) {
      const bool a = predictions[std::size_t(i)] == 1, b = label_values[std::size_t(i)] == 1;
      intersection += a && b;
      combined += a || b;
    }
    const double iou   = double(intersection) / combined;
    const double value = final[std::size_t(field)].item<double>();
    check::Require(value < initial[std::size_t(field)] && iou > .98, "each loss reduces error and reconstructs known mask");
    std::cout << "  " << titles[field] << ": loss" << initial[std::size_t(field)] << " -> " << value
              << ", foreground IoU=" << iou << std::endl;
  }
  seg::SegmentationData preview{4, side, side, 2, {}, {}};
  preview.images.resize(4 * side * side * 3);
  preview.labels.resize(4 * side * side);
  for (int n = 0; n < 4; ++n)
    for (int i = 0; i < side * side; ++i) {
      preview.labels[std::size_t(n * side * side + i)] = label_values[std::size_t(i)];
      for (int k = 0; k < 3; ++k)
        preview.images[std::size_t((n * side * side + i) * 3 + k)] = label_values[std::size_t(i)] ? .85f : .1f;
    }
  const auto directory = seg::SegmentationDirectory("segmentation_loss_fields");
  seg::SegmentationPreview([&](const Tensor&) { return nn::concat(prob, 0); }, preview, directory / "preview.png");
  std::cout << "  loss-field target/prediction preview: " << (directory / "preview.png").string() << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(arg.substr(8)));
      else
        throw std::invalid_argument("expected threads=N");
    }
    LossChecks();
    DistanceChecks();
    OptimizeFields();
    std::cout << "All Dice/Tversky/Lovasz/boundary checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
