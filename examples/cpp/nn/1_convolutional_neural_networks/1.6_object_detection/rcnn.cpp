// CMD: just run-nn rcnn
// R-CNN: classify warped region proposals with a CNN and refine their boxes.
//
// Port of tflow's rcnn. Preparation (in ~/projects/tflow_examples):
// python3 scripts/fetch_datasets.py mnist --split both.
//
// On the synthetic digit scenes (6000 training, 1000 test; see
// detection_baselines), propose regions classically from connected
// components of the image thresholded at 48, 64, 96, 128, 160, 200 and 230
// (and merges of nearby components). High thresholds separate touching
// digits but give boxes systematically tighter than the ground truth (defined
// at 32), so localization needs correcting. As in R-CNN, each
// proposal is padded by 7% on every side (the paper's 16 pixels at 227) and
// warped anisotropically to 28x28; the CNN is fine-tuned on proposals and
// ground-truth boxes, with proposals of IoU >= 0.5 labelled by their best
// target and the rest background, and a class-specific box regressor learns
// the Faster R-CNN deltas of proposals with IoU >= 0.6. A softmax over
// eleven classes stands in for the paper's per-class SVMs, and the regressor
// is a second head of the same network instead of ridge regression on pool5
// features. Check the box coding round trip, then report proposal recall and
// test AP with and without box regression.
// https://arxiv.org/abs/1311.2524
//
// Eager port: the fine-tuning loop (Adam(.001), batch 128, 5e-5 * sum of
// squared kernels) runs directly; weights come from nn's generator, so trained
// numbers match TF statistically. Scenes and proposals are identical to TF's.
// Options: data=PATH epochs=8 threads=N.

#include <iomanip>
#include <iostream>

#include "common/two_stage.hpp"
#include "common/detection_mosaic.hpp"

namespace nn = datamunge::nn;
using detector::Box;
using detector::DetectionScenes;

namespace {
constexpr int kPatch = 28, kClasses = 11, kBackground = 10;
Box Pad(const Box& b, float fraction) {
  const float h = (b[2] - b[0]) * fraction, w = (b[3] - b[1]) * fraction;
  return {b[0] - h, b[1] - w, b[2] + h, b[3] + w};
}
std::vector<Box> Proposals(const DetectionScenes& scenes, int row) {
  return detector::ComponentProposals(scenes, row, {48, 64, 96, 128, 160, 200, 230}, 3);
}

struct Samples {
  std::vector<float>                pixels;
  std::vector<std::array<float, 4>> deltas;
  std::vector<int>                  labels;
  std::vector<float>                regress;
};
Samples MakeSamples(const DetectionScenes& scenes, int first, int last) {
  Samples out;
  auto add = [&](int row, const Box& box) {
    int         label  = kBackground, target = -1;
    float       best   = 0;
    const auto& truth  = scenes.boxes[std::size_t(row)];
    for (std::size_t k = 0; k < truth.size(); ++k) {
      const float iou = detector::BoxIoU(box, truth[k]);
      if (iou > best) {
        best   = iou;
        target = int(k);
      }
    }
    if (best >= .5f) label = scenes.labels[std::size_t(row)][std::size_t(target)];
    std::array<float, 4> delta   = {0, 0, 0, 0};
    const bool           regress = best >= .6f;
    if (regress) delta = detector::EncodeBox(box, truth[std::size_t(target)]);
    out.pixels.resize(out.pixels.size() + kPatch * kPatch);
    detector::CropScene(scenes, row, Pad(box, .07f), kPatch, out.pixels.data() + out.pixels.size() - kPatch * kPatch);
    out.labels.push_back(label);
    out.regress.push_back(regress);
    out.deltas.push_back(delta);
  };
  for (int row = first; row < last; ++row) {
    for (const auto& b : Proposals(scenes, row)) add(row, b);
    for (const auto& b : scenes.boxes[std::size_t(row)]) add(row, b);
  }
  return out;
}

/// conv(32)/pool, conv(64)/pool, fc1 128, an 11-way classifier and 10 class-specific box regressors.
detector::RoiHeadOutputs Network(nn::ParameterStore& p, const nn::Tensor& input) {
  nn::ConvLayerOptions cfg;
  cfg.init       = nn::init::he_uniform();
  cfg.activation = nn::Activation::Relu;
  auto h = nn::max_pool2d(nn::conv2d(p, "conv1", nn::reshape(input, {-1, kPatch, kPatch, 1}), 32, 3, cfg), 2, 2, nn::Padding::Same);
  h      = nn::max_pool2d(nn::conv2d(p, "conv2", h, 64, 3, cfg), 2, 2, nn::Padding::Same);
  const auto f = nn::relu(nn::dense(p, "fc1", nn::reshape(h, {-1, 7 * 7 * 64}), 128));
  detector::RoiHeadOutputs n;
  n.logits        = nn::dense(p, "classifier", f, kClasses);
  n.probabilities = nn::softmax(n.logits, -1);
  n.deltas        = nn::reshape(nn::dense(p, "regressor", f, 4 * kBackground), {-1, kBackground, 4});
  return n;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path directory = paths::Dataset("mnist");
    int                   epochs    = 8;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      const auto        equal    = argument.find('=');
      if (equal == std::string::npos) throw std::invalid_argument("expected key=value option");
      const auto key = argument.substr(0, equal), value = argument.substr(equal + 1);
      if (key == "data") directory = value;
      else if (key == "epochs") epochs = std::stoi(value);
      else if (key == "threads") nn::set_num_threads(std::stoi(value));
      else throw std::invalid_argument("unknown option " + key);
    }
    // Box coding round trip.
    const Box  anchor = {10, 12, 30, 40}, box = {13, 9, 35, 47};
    const auto code   = detector::EncodeBox(anchor, box);
    const auto back   = detector::DecodeBox(anchor, code.data());
    double     coding = 0;
    for (std::size_t i = 0; i < 4; ++i) coding = std::max(coding, double(std::abs(back[i] - box[i])));
    check::Near(coding, 0, 1e-4, "box encoding round trip");
    std::cout << "  box deltas (dy, dx, dh, dw) / (0.1, 0.1, 0.2, 0.2) = (" << code[0] << ", " << code[1] << ", " << code[2]
              << ", " << code[3] << "); decoding restores the box (error " << coding << ")" << std::endl;
    epochs = example_runtime::Limit(epochs, 1);
    example_runtime::Describe();
    auto [train, test] = detector::StandardScenes(directory);
    if (!example_runtime::Full()) {
      std::vector<int> selected(std::min(test.size(), 8));
      std::iota(selected.begin(), selected.end(), 0);
      test = detector::CopyDetectionRows(test, selected);
    }
    const auto samples       = MakeSamples(train, 0, example_runtime::Limit(5500, 16));
    int        positives = 0, regressed = 0;
    for (std::size_t i = 0; i < samples.labels.size(); ++i) {
      positives += samples.labels[i] != kBackground;
      regressed += samples.regress[i] > 0;
    }
    std::cout << "  fine-tuning set: " << samples.labels.size() << " warped regions from "
              << example_runtime::Limit(5500, 16) << " scenes (" << positives
              << " digit, " << samples.labels.size() - std::size_t(positives) << " background; " << regressed
              << " with regression targets)" << std::endl;
    nn::ParameterStore p(17);
    {
      nn::NoGradGuard guard;
      Network(p, nn::zeros({1, kPatch * kPatch}));
    }
    std::vector<nn::Tensor> kernels;
    for (const auto& t : p.parameters())
      if (t.dim() >= 2) kernels.push_back(t);
    nn::OptimizerConfig oc;
    oc.kind = nn::OptimizerKind::Adam;
    nn::Optimizer  optimizer(p, 1e-3, oc);
    nn::BatchOrder order(samples.labels.size(), example_runtime::Limit(128, 16), 29);
    for (int epoch = 1; epoch <= epochs; ++epoch) {
      order.begin_epoch();
      double      total = 0;
      std::size_t seen  = 0;
      for (auto rows = order.next(); !rows.empty(); rows = order.next()) {
        const auto         n = std::int64_t(rows.size());
        std::vector<float> x(std::size_t(n) * kPatch * kPatch);
        detector::RoiBatch batch;
        for (std::int64_t i = 0; i < n; ++i) {
          const auto r = std::size_t(rows[std::size_t(i)]);
          std::copy_n(samples.pixels.begin() + std::ptrdiff_t(r * kPatch * kPatch), kPatch * kPatch,
                      x.begin() + std::ptrdiff_t(i * kPatch * kPatch));
          batch.labels.push_back(samples.labels[r]);
          batch.mask.push_back(samples.regress[r]);
          batch.targets.push_back(samples.deltas[r]);
        }
        const auto net  = Network(p, nn::from_vector(x, {n, kPatch * kPatch}));
        const auto loss = detector::RoiLoss(net, batch, kClasses);
        nn::Tensor penalty = nn::sum(nn::square(kernels[0]));
        for (std::size_t k = 1; k < kernels.size(); ++k) penalty = penalty + nn::sum(nn::square(kernels[k]));
        const double value = loss.item();
        check::Require(std::isfinite(value), "finite R-CNN loss");
        optimizer.minimize(loss + 5e-5 * penalty);
        total += value * double(n);
        seen += std::size_t(n);
      }
      std::cout << "  epoch " << epoch << ": training loss " << total / double(seen) << std::endl;
    }
    // Test: classify every proposal, with and without class-specific box regression.
    std::vector<std::pair<int, Box>> proposals;
    int                              objects = 0, covered50 = 0, covered70 = 0;
    for (int row = 0; row < test.size(); ++row) {
      const auto boxes = Proposals(test, row);
      for (const auto& b : boxes) proposals.push_back({row, b});
      for (const auto& g : test.boxes[std::size_t(row)]) {
        ++objects;
        float best = 0;
        for (const auto& b : boxes) best = std::max(best, detector::BoxIoU(b, g));
        covered50 += best >= .5f;
        covered70 += best >= .7f;
      }
    }
    std::vector<detector::Detection> plain, refined;
    {
      nn::NoGradGuard guard;
      for (std::size_t start = 0; start < proposals.size(); start += 4096) {
        const std::size_t  n = std::min<std::size_t>(4096, proposals.size() - start);
        std::vector<float> x(n * kPatch * kPatch);
        for (std::size_t i = 0; i < n; ++i)
          detector::CropScene(test, proposals[start + i].first, Pad(proposals[start + i].second, .07f), kPatch,
                              x.data() + i * kPatch * kPatch);
        const auto out           = Network(p, nn::from_vector(x, {std::int64_t(n), kPatch * kPatch}));
        const auto probabilities = out.probabilities.to_vector<float>(), deltas = out.deltas.to_vector<float>();
        for (std::size_t i = 0; i < n; ++i)
          for (int c = 0; c < kBackground; ++c) {
            const float score = probabilities[i * kClasses + std::size_t(c)];
            if (score <= .05f) continue;
            const auto& [row, b] = proposals[start + i];
            plain.push_back({row, c, score, b});
            refined.push_back({row, c, score, detector::DecodeBox(b, deltas.data() + (i * kBackground + std::size_t(c)) * 4)});
          }
      }
    }
    std::vector<int> rows(std::size_t(test.size()));
    std::iota(rows.begin(), rows.end(), 0);
    const auto without = detector::EvaluateDetections(detector::ClassNms(plain, .3f), test, rows, 10);
    const auto with    = detector::EvaluateDetections(detector::ClassNms(refined, .3f), test, rows, 10);
    std::cout << "  proposals: " << std::fixed << std::setprecision(1) << double(proposals.size()) / test.size()
              << " per test scene, recall " << std::setprecision(3) << double(covered50) / objects << " at IoU 0.5 and "
              << double(covered70) / objects << " at 0.7" << std::defaultfloat << std::setprecision(6)
              << "\n  without box regression: test " << detector::ApSummary(without)
              << "\n  with box regression:    test " << detector::ApSummary(with) << std::endl;
    check::TrainingQuality(with.ap50 > .7 && with.ap75 > without.ap75, "box regression improves localization");
    std::cout << "All R-CNN checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
