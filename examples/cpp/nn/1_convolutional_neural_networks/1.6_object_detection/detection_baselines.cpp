// CMD: just run-nn detection_baselines
// Detection by classification: sliding windows versus region proposals.
//
// Port of tflow's detection_baselines. Preparation (in ~/projects/tflow_examples):
// python3 scripts/fetch_datasets.py mnist --split both.
//
// Build synthetic detection scenes (common/detection_scenes.hpp, identical to
// tflow's scenes): one to four MNIST digits, 16-32 pixels, scattered on a
// 64x64 canvas, with tight boxes; 6000 training scenes from MNIST training
// digits and 1000 test scenes from MNIST test digits. Train an 11-way patch
// classifier (10 digits and background) on 28x28 crops: jittered squares
// around training boxes and random squares that overlap no box by more than
// IoU 0.2. Detect two ways: (1) slide square windows of 16, 22, 28 and 34
// pixels with stride 4 over every scene and classify each; (2) propose
// regions classically, from the connected components of the image
// thresholded at 16, 32, 64, 96, 128, 160 and 200 and the merged boxes of
// components lying within three pixels of each other, and classify a square
// crop around each. Both keep every class scoring above 0.05, then apply
// per-class NMS (IoU 0.3). Report windows or proposals per scene, proposal
// recall at IoU 0.5 / 0.7, and test AP (VOC AP50, AP75, COCO AP).
// Square windows rarely fit a digit's tight box, which caps their AP75.
// Proposals fit isolated digits almost exactly (blank background), but digits
// whose strokes touch merge into one component at every threshold, which caps
// proposal recall well below one; faster_rcnn learns proposals instead.
// The patch classifier trains eagerly (Adam .001, batch 128, L2 .0001 on
// kernels, nn's generator with seed 17), so its numbers match TF statistically.
// Options: data=PATH epochs=6 threads=N.

#include <iomanip>
#include <iostream>
#include <random>

#include "common/detection_scenes.hpp"
#include "common/detection_mosaic.hpp"

namespace nn = datamunge::nn;
using detector::Box;
using detector::DetectionScenes;

namespace {
constexpr int kPatch = 28, kClasses = 11, kBackground = 10;

struct Patches {
  std::vector<float> pixels;
  std::vector<int>   labels;
};
Patches MakePatches(const DetectionScenes& scenes, int first, int last, std::uint64_t seed) {
  Patches                               p;
  std::mt19937_64                       random(seed);
  std::uniform_real_distribution<float> unit(0, 1);
  auto add = [&](int row, const Box& box, int label) {
    p.pixels.resize(p.pixels.size() + kPatch * kPatch);
    detector::CropScene(scenes, row, box, kPatch, p.pixels.data() + p.pixels.size() - kPatch * kPatch);
    p.labels.push_back(label);
  };
  for (int row = first; row < last; ++row) {
    const auto& boxes = scenes.boxes[std::size_t(row)];
    for (std::size_t k = 0; k < boxes.size(); ++k)
      for (int copy = 0; copy < 3; ++copy) {
        const auto& b    = boxes[k];
        const float side = std::max(b[2] - b[0], b[3] - b[1]) * (1 + .3f * unit(random));
        const float cy = (b[0] + b[2]) / 2 + (unit(random) - .5f) * .2f * side;
        const float cx = (b[1] + b[3]) / 2 + (unit(random) - .5f) * .2f * side;
        add(row, {cy - side / 2, cx - side / 2, cy + side / 2, cx + side / 2}, scenes.labels[std::size_t(row)][k]);
      }
    for (int copy = 0, tries = 0; copy < 3 * int(boxes.size()) && tries < 200; ++tries) {
      const float side = 12 + 28 * unit(random);
      const float y = unit(random) * (scenes.height - side), x = unit(random) * (scenes.width - side);
      const Box   window = {y, x, y + side, x + side};
      bool        clear  = true;
      for (const auto& b : boxes)
        if (detector::BoxIoU(window, b) > .2f) clear = false;
      if (clear) {
        add(row, window, kBackground);
        ++copy;
      }
    }
  }
  return p;
}

nn::Tensor PatchLogits(nn::ParameterStore& p, const nn::Tensor& x) {
  nn::ConvLayerOptions cfg;
  cfg.init       = nn::init::he_uniform();
  cfg.activation = nn::Activation::Relu;
  auto h = nn::max_pool2d(nn::conv2d(p, "conv1", nn::reshape(x, {-1, kPatch, kPatch, 1}), 32, 3, cfg), 2, 2, nn::Padding::Same);
  h      = nn::max_pool2d(nn::conv2d(p, "conv2", h, 64, 3, cfg), 2, 2, nn::Padding::Same);
  auto f = nn::dense(p, "fc1", nn::reshape(h, {-1, 7 * 7 * 64}), 128, nn::Activation::Relu);
  return nn::dense(p, "output", f, kClasses);
}

nn::Tensor Batch(const Patches& p, const std::vector<std::int64_t>& rows, std::vector<std::int64_t>* labels) {
  nn::Tensor x   = nn::empty({std::int64_t(rows.size()), kPatch * kPatch});
  float*     dst = x.data_ptr<float>();
  for (std::size_t i = 0; i < rows.size(); ++i) {
    std::copy_n(p.pixels.begin() + rows[i] * kPatch * kPatch, kPatch * kPatch, dst + i * kPatch * kPatch);
    if (labels) labels->push_back(p.labels[std::size_t(rows[i])]);
  }
  return x;
}

// Class probabilities [boxes x 11] of square crops around boxes.
std::vector<std::array<float, kClasses>> Classify(nn::ParameterStore& model, const DetectionScenes& scenes,
                                                  const std::vector<std::pair<int, Box>>& crops) {
  nn::NoGradGuard                          guard;
  std::vector<std::array<float, kClasses>> result(crops.size());
  for (std::size_t start = 0; start < crops.size(); start += 4096) {
    const std::size_t n   = std::min<std::size_t>(4096, crops.size() - start);
    nn::Tensor        x   = nn::empty({std::int64_t(n), kPatch * kPatch});
    float*            dst = x.data_ptr<float>();
    for (std::size_t i = 0; i < n; ++i)
      detector::CropScene(scenes, crops[start + i].first, crops[start + i].second, kPatch, dst + i * kPatch * kPatch);
    const auto probabilities = nn::softmax(PatchLogits(model, x), -1).to_vector<float>();
    for (std::size_t i = 0; i < n; ++i)
      for (int c = 0; c < kClasses; ++c) result[start + i][std::size_t(c)] = probabilities[i * kClasses + std::size_t(c)];
  }
  return result;
}
std::vector<detector::Detection> Detect(const std::vector<std::pair<int, Box>>& crops, const std::vector<Box>& reported,
                                        const std::vector<std::array<float, kClasses>>& probabilities) {
  std::vector<detector::Detection> dets;
  for (std::size_t i = 0; i < crops.size(); ++i)
    for (int c = 0; c < kBackground; ++c)
      if (probabilities[i][std::size_t(c)] > .05f) dets.push_back({crops[i].first, c, probabilities[i][std::size_t(c)], reported[i]});
  return detector::ClassNms(dets, .3f);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path directory = paths::Dataset("mnist");
    int                   epochs    = 6;
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
    epochs = example_runtime::Limit(epochs, 1);
    example_runtime::Describe();
    auto [train, test] = detector::StandardScenes(directory);
    if (!example_runtime::Full()) {
      std::vector<int> selected(std::min(test.size(), 8));
      std::iota(selected.begin(), selected.end(), 0);
      test = detector::CopyDetectionRows(test, selected);
    }
    int objects = 0;
    for (const auto& b : test.boxes) objects += int(b.size());
    std::cout << "  scenes: " << train.size() << " training (" << example_runtime::Limit(5500, 16)
              << " for patches, " << example_runtime::Limit(500, 16) << " validation), " << test.size()
              << " test with " << objects << " digits"
              << std::endl;
    // Patch classifier.
    const auto         fit = MakePatches(train, 0, example_runtime::Limit(5500, 16), 3), validation = MakePatches(train, 5500, example_runtime::Full() ? 6000 : 5516, 4);
    nn::ParameterStore model(17);
    {
      nn::NoGradGuard guard;
      PatchLogits(model, nn::zeros({1, kPatch * kPatch}));
    }
    std::vector<nn::Tensor> kernels;
    for (const auto& p : model.parameters())
      if (p.dim() >= 2) kernels.push_back(p);
    nn::OptimizerConfig adam;
    adam.kind = nn::OptimizerKind::Adam;
    nn::Optimizer  optimizer(model, .001, adam);
    nn::BatchOrder order(fit.labels.size(), example_runtime::Limit(128, 16), 29);
    for (int epoch = 1; epoch <= epochs; ++epoch) {
      order.begin_epoch();
      for (auto rows = order.next(); !rows.empty(); rows = order.next()) {
        std::vector<std::int64_t> labels;
        const auto                x    = Batch(fit, rows, &labels);
        nn::Tensor                loss = nn::sparse_softmax_cross_entropy(PatchLogits(model, x), nn::from_vector(labels));
        nn::Tensor                penalty = nn::sum(nn::square(kernels[0]));
        for (std::size_t k = 1; k < kernels.size(); ++k) penalty = penalty + nn::sum(nn::square(kernels[k]));
        optimizer.minimize(loss + .5 * .0001 * penalty);
      }
      std::vector<std::int64_t> all(validation.labels.size()), labels;
      std::iota(all.begin(), all.end(), 0);
      nn::NoGradGuard guard;
      const auto      predicted = nn::argmax(PatchLogits(model, Batch(validation, all, &labels)), -1).to_vector<std::int64_t>();
      int             correct   = 0;
      for (std::size_t i = 0; i < labels.size(); ++i) correct += predicted[i] == labels[i];
      std::cout << "  patch classifier epoch " << epoch << ": " << fit.labels.size()
                << " training patches, validation accuracy " << double(correct) / labels.size() << " on "
                << validation.labels.size() << std::endl;
    }
    // Sliding windows.
    std::vector<std::pair<int, Box>> windows;
    for (int row = 0; row < test.size(); ++row)
      for (int side : {16, 22, 28, 34})
        for (int y = 0; y + side <= test.height; y += 4)
          for (int x = 0; x + side <= test.width; x += 4)
            windows.push_back({row, {float(y), float(x), float(y + side), float(x + side)}});
    std::vector<Box> window_boxes;
    for (const auto& w : windows) window_boxes.push_back(w.second);
    const auto       sliding = Detect(windows, window_boxes, Classify(model, test, windows));
    std::vector<int> rows(std::size_t(test.size()));
    std::iota(rows.begin(), rows.end(), 0);
    const auto sliding_ap = detector::EvaluateDetections(sliding, test, rows, 10);
    // Region proposals.
    std::vector<std::pair<int, Box>> crops;
    std::vector<Box>                 proposals;
    int                              covered50 = 0, covered70 = 0;
    for (int row = 0; row < test.size(); ++row) {
      const auto boxes = detector::ComponentProposals(test, row, {16, 32, 64, 96, 128, 160, 200}, 3);
      for (const auto& b : boxes) {
        crops.push_back({row, detector::Square(b, 1.15f)});
        proposals.push_back(b);
      }
      for (const auto& g : test.boxes[std::size_t(row)]) {
        float best = 0;
        for (const auto& b : boxes) best = std::max(best, detector::BoxIoU(b, g));
        covered50 += best >= .5f;
        covered70 += best >= .7f;
      }
    }
    const auto proposed    = Detect(crops, proposals, Classify(model, test, crops));
    const auto proposal_ap = detector::EvaluateDetections(proposed, test, rows, 10);
    std::cout << "  sliding windows: " << windows.size() / std::size_t(test.size()) << " per scene; test "
              << detector::ApSummary(sliding_ap) << "\n  region proposals: " << std::fixed << std::setprecision(1)
              << double(crops.size()) / test.size() << " per scene, recall " << std::setprecision(3)
              << double(covered50) / objects << " at IoU 0.5 and " << double(covered70) / objects << " at 0.7; test "
              << detector::ApSummary(proposal_ap) << std::defaultfloat << std::setprecision(6) << std::endl;
    check::TrainingQuality(proposal_ap.ap50 > .6 && proposal_ap.ap50 > sliding_ap.ap50 && proposal_ap.ap75 > sliding_ap.ap75 &&
                       sliding_ap.ap50 > .2,
                   "detection baselines");
    std::cout << "All detection baseline checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
