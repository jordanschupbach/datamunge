// CMD: just run-nn yolo_v5_style
// YOLO v4/v5-style detection: CSP, SPPF, a PAN neck, CIoU and mosaic augmentation.
// Preparation: dm::datasets::fetch("mnist")
// Short default: three training epochs; use epochs=N for longer accuracy experiments.
//
// Follow the v6.0 YOLOv5s topology: 7,235,389 learned parameters for COCO,
// CSP splits and identity bottlenecks, serial spatial-pyramid pooling and
// a top-down/bottom-up PAN neck. Independently check CSP values and input
// derivatives with controlled filters, SPPF's equivalence to 5/9/13 pooling
// including negative borders, four-way mosaic pixels/boxes and flip
// round trips. Compare CIoU values with host calculations and derivatives
// with central differences holding its detached alpha fixed. Check expanded
// centre decoding, ratio/neighbor assignments and empty/positive BCE
// derivatives. Full topology derivatives are constructed symbolically;
// compact parameter derivatives are evaluated at 64/96/128 pixels.
// https://github.com/ultralytics/yolov5/blob/v6.0/models/yolov5s.yaml
// https://arxiv.org/abs/1911.08287
//
// Train the full topology at quarter YOLOv5s widths on grayscale digit
// scenes. Use a fixed-centre four-image mosaic in half of training batches,
// random horizontal flips, nine fit-only dimension priors, ratio/neighbor
// assignment and CIoU regression. Box/class/objectness loss weights 5/5/1
// are a compact-training adaptation. Cycle 128/160/192 training resolutions every
// ten batches. Use 5500 fit / 500 validation / 1000 test scenes, three
// epochs of Adam(.001), batch 32 and kernel L2 .0001, with best validation
// AP50 checkpoint selection. No pretrained weights or COCO accuracy are
// claimed. Export an annotated mosaic PNG under output/visualizations/.
// Options: data=PATH size=128 epochs=3 mosaic=0|1 (default 1).
// checks_only=1 runs all fixtures without loading data or training.
//
// Digit cells scale with input size (32..64 at the 128-pixel default).
// size=64 reproduces the initial smaller-input geometry. Optional
// warm_start=PREFIX restores parameters/BN state with fresh Adam moments.
// Structural/numerical fixtures retain their original input sizes.
//
// Default accuracy=0 verifies finite, decreasing loss; accuracy=1 enforces
// the original test AP50 target. Report measured metrics in either mode.

// Eager port: full connectivity is computed at128 pixels; compact checks
// retain64/96/128 grids. Model checkpoints are NPZ. Matching uses the same
// training forward pass, with detached target construction.
#include <datamunge/image/png.hpp>

#include "common/detection.hpp"
#include "common/detection_mosaic.hpp"
#include "common/yolo_v5.hpp"

namespace nn = datamunge::nn;
namespace {
using detector::DetectionScenes;
using yolo::Dimension;
double Silu(double x) {
  return x / (1 + std::exp(-x));
}
double SiluDerivative(double x) {
  double p = 1 / (1 + std::exp(-x));
  return p + x * p * (1 - p);
}
void CspChecks() {
  nn::ParameterStore p(17), state;
  yolo::CspContext   c{p, state, false, false};
  auto               x = nn::zeros({1, 3, 3, 4});
  {
    nn::NoGradGuard guard;
    yolo::CspBlock(c, "csp", x, 4, 1, true);
  }
  for (auto part : {"left", "right", "fuse"}) {
    auto               name  = std::string("csp/") + part + "/W";
    auto               shape = p.at(name).shape();
    std::vector<float> w(nn::shape_numel(shape));
    if (std::string(part) == "fuse") {
      for (int ch = 0; ch < 4; ++ch)
        w[ch * 4 + ch] = 1;
    } else
      for (int ch = 0; ch < 2; ++ch)
        w[(ch + (std::string(part) == "right" ? 2 : 0)) * 2 + ch] = 1;
    p.set(name, nn::from_vector(w, shape));
  }
  for (auto part : {"reduce", "expand"}) {
    auto name = std::string("csp/unit1/") + part + "/W";
    p.set(name, nn::zeros(p.at(name).shape()));
  }
  std::vector<float> values(36);
  for (int i = 0; i < 36; ++i)
    values[i] = .1f * i - 1.8f;
  x            = nn::from_vector(values, {1, 3, 3, 4}).requires_grad_();
  double error = 0, derivative_error = 0, normalization = 1 / std::sqrt(1.001);
  for (bool residual : {true, false}) {
    auto y   = yolo::CspBlock(c, "csp", x, 4, 1, residual).output;
    auto g   = nn::grad(nn::sum(y), {x})[0].to_vector<float>();
    auto out = y.to_vector<float>();
    for (int i = 0; i < 36; ++i) {
      double z = values[i], projected = Silu(normalization * z),
             expected   = Silu(normalization * projected),
             derivative = normalization * normalization * SiluDerivative(normalization * z) *
                          SiluDerivative(normalization * projected);
      bool connected   = residual || i % 4 >= 2;
      error            = std::max(error, std::abs(out[i] - (connected ? expected : 0)));
      derivative_error = std::max(derivative_error, std::abs(g[i] - (connected ? derivative : 0)));
    }
  }
  check::Near(error, 0, 2e-5, "CSP split,bypass,residual paths");
  check::Near(derivative_error, 0, 2e-5, "analytic CSP input derivatives");
  std::cout << "CSP value=" << error << " derivative=" << derivative_error << std::endl;
}
void PoolingChecks() {
  std::vector<float> input(15 * 17);
  for (int i = 0; i < 255; ++i)
    input[i] = std::sin(.37f * i) - 2;
  auto x = nn::from_vector(input, {1, 15, 17, 1}), previous = x;
  for (int scale = 0; scale < 3; ++scale) {
    previous      = nn::max_pool2d(previous, 5, 1, nn::Padding::Same);
    auto serial   = previous.to_vector<float>(),
         parallel = nn::max_pool2d(x, 5 + 4 * scale, 1, nn::Padding::Same).to_vector<float>();
    for (int r = 0; r < 15; ++r)
      for (int c = 0; c < 17; ++c) {
        float expected = -INFINITY;
        int   radius   = 2 + 2 * scale;
        for (int y = std::max(0, r - radius); y <= std::min(14, r + radius); ++y)
          for (int z = std::max(0, c - radius); z <= std::min(16, c + radius); ++z)
            expected = std::max(expected, input[y * 17 + z]);
        check::Near(serial[r * 17 + c], expected, 0, "SPPF serial host windows");
        check::Near(parallel[r * 17 + c], expected, 0, "SPP parallel negative borders");
      }
  }
}
void MosaicChecks() {
  detector::DetectionScenes source;
  source.height = source.width = 4;
  source.pixels.resize(4 * 16);
  for (int row = 0; row < 4; ++row) {
    std::fill_n(source.pixels.data() + 16 * row, 16, 10 * (row + 1));
    source.boxes.push_back({{0, 0, 4, 4}});
    source.labels.push_back({row});
  }
  auto                             mosaic   = detector::MosaicScenes(source, {{0, 1, 2, 3}});
  const std::vector<detector::Box> expected = {
      {0, 0, 2, 2}, {0, 2, 2, 4}, {2, 0, 4, 2}, {2, 2, 4, 4}};
  check::Require(
      mosaic.boxes[0] == expected && mosaic.labels[0] == std::vector<int>({0, 1, 2, 3}),
      "mosaic scales/translates boxes with their labels");
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c)
      check::Require(
          mosaic.pixels[r * 4 + c] == 10 * ((r / 2) * 2 + c / 2 + 1), "mosaic pixel quadrants");
  const auto original = mosaic;
  detector::FlipDetectionScene(mosaic, 0);
  detector::FlipDetectionScene(mosaic, 0);
  check::Require(
      mosaic.pixels == original.pixels && mosaic.boxes == original.boxes,
      "detection flip inverse preserves pixels and boxes");
  const auto copied = detector::CopyDetectionRows(source, {3, 1});
  check::Require(copied.pixels[0] == 40 && copied.labels[1][0] == 1, "detection row copying");
  // Nonconstant rectangular images exercise the actual 2x2 resampling.
  // All expected bytes/boxes below are independent, hand-computed values.
  detector::DetectionScenes rectangular;
  rectangular.height = 2;
  rectangular.width  = 4;
  rectangular.pixels = {0, 1, 10, 11, 2, 3, 12, 13, 20, 22, 30, 34, 24, 26, 38, 42};
  rectangular.boxes  = {{{0, 1, 2, 3}}, {{.5f, 0, 1.5f, 2}, {0, 2, 2, 4}}};
  rectangular.labels = {{7}, {3, 9}};
  const auto frozen  = rectangular;
  const auto reduced = detector::MosaicScenes(rectangular, {{1, 0, 0, 1}, {0, 0, 0, 0}});
  check::Require(
      reduced.height == 2 && reduced.width == 4 && reduced.size() == 2,
      "rectangular mosaic batch geometry");
  check::Require(
      reduced.pixels ==
          std::vector<uint8_t>({23, 36, 2, 12, 2, 12, 23, 36, 2, 12, 2, 12, 2, 12, 2, 12}),
      "rectangular mosaic bilinear averages, rounding and selected-row order");
  const std::vector<detector::Box> rectangular_boxes = {
      {.25f, 0, .75f, 1},
      {0, 1, 1, 2},
      {0, 2.5f, 1, 3.5f},
      {1, .5f, 2, 1.5f},
      {1.25f, 2, 1.75f, 3},
      {1, 3, 2, 4}};
  check::Require(
      reduced.boxes[0] == rectangular_boxes &&
          reduced.labels[0] == std::vector<int>({3, 9, 7, 7, 3, 9}),
      "rectangular mosaic transforms every box and preserves multiple labels");
  check::Require(
      reduced.labels[1] == std::vector<int>({7, 7, 7, 7}) && rectangular.pixels == frozen.pixels &&
          rectangular.boxes == frozen.boxes && rectangular.labels == frozen.labels,
      "mosaic repeated selections preserve the source");
  detector::FlipDetectionScene(rectangular, 1);
  check::Require(
      rectangular.pixels ==
              std::vector<uint8_t>({0, 1, 10, 11, 2, 3, 12, 13, 34, 30, 22, 20, 42, 38, 26, 24}) &&
          rectangular.boxes[1] == std::vector<detector::Box>({{.5f, 2, 1.5f, 4}, {0, 0, 2, 2}}) &&
          rectangular.boxes[0] == frozen.boxes[0] && rectangular.labels == frozen.labels,
      "single horizontal flip reverses pixels and reflects asymmetric boxes only in its row");
  detector::FlipDetectionScene(rectangular, 1);
  check::Require(
      rectangular.pixels == frozen.pixels && rectangular.boxes == frozen.boxes,
      "rectangular flip round trip");
  std::cout << "  mosaic quadrants, rectangular bilinear averages/rounding, multiple labels, "
               "repeated rows and asymmetric single/double flips exact"
            << std::endl;
}

std::array<double, 4> HostQuality(
    const std::array<double, 4>& p,
    const std::array<double, 4>& t,
    double                       fixed_alpha = -1) {
  constexpr double eps = 1e-7;
  const double     py0 = p[0] - p[2] / 2, px0 = p[1] - p[3] / 2, py1 = p[0] + p[2] / 2,
               px1 = p[1] + p[3] / 2;
  const double ty0 = t[0] - t[2] / 2, tx0 = t[1] - t[3] / 2, ty1 = t[0] + t[2] / 2,
               tx1          = t[1] + t[3] / 2;
  const double intersection = std::max(0., std::min(py1, ty1) - std::max(py0, ty0)) *
                              std::max(0., std::min(px1, tx1) - std::max(px0, tx0));
  const double iou =
      std::min(1., intersection / std::max(eps, p[2] * p[3] + t[2] * t[3] - intersection));
  const double h        = std::max(py1, ty1) - std::min(py0, ty0),
               w        = std::max(px1, tx1) - std::min(px0, tx0);
  const double distance = (p[0] - t[0]) * (p[0] - t[0]) + (p[1] - t[1]) * (p[1] - t[1]);
  const double angle    = std::atan(t[3] / t[2]) - std::atan(p[3] / p[2]),
               v        = 4 / (std::acos(-1.) * std::acos(-1.)) * angle * angle;
  const double alpha    = fixed_alpha >= 0 ? fixed_alpha : v / (1 - iou + v + eps),
               diou     = iou - distance / (h * h + w * w + eps);
  return {iou, diou, diou - alpha * v, alpha};
}

void QualityChecks() {
  std::array<double, 4> p{.53, .48, .3, .2}, t{.5, .5, .4, .32};
  auto input  = nn::from_vector(std::vector<double>(p.begin(), p.end()), {1, 4}).requires_grad_(),
       target = nn::from_vector(std::vector<double>(t.begin(), t.end()), {1, 4});
  auto                    quality = yolo::CompleteBoxQuality(input, target);
  auto                    g = nn::grad(nn::sum(1. - quality.ciou), {input})[0].to_vector<double>();
  auto                    expected = HostQuality(p, t);
  std::vector<nn::Tensor> values{quality.iou, quality.diou, quality.ciou, quality.alpha};
  for (int i = 0; i < 4; ++i)
    check::Near(values[i].item(), expected[i], 5e-6, "host IoU/DIoU/CIoU/alpha");
  double error = 0;
  for (int i = 0; i < 4; ++i) {
    auto plus = p, minus = p;
    plus[i] += 1e-5;
    minus[i] -= 1e-5;
    double d =
        -(HostQuality(plus, t, expected[3])[2] - HostQuality(minus, t, expected[3])[2]) / 2e-5;
    error = std::max(error, std::abs(g[i] - d));
  }
  check::Near(error, 0, 5e-4, "CIoU frozen-alpha central differences");
  check::Near(yolo::CompleteBoxQuality(target, target).ciou.item(), 1, 1e-6, "identical unit CIoU");
  auto disjoint   = nn::from_vector(std::vector<double>{1.2, 1.2, .2, .2}, {1, 4}).requires_grad_();
  auto q          = yolo::CompleteBoxQuality(disjoint, target);
  auto derivative = nn::grad(nn::sum(1. - q.ciou), {disjoint})[0].to_vector<double>();
  check::Near(q.iou.item(), 0, 0, "disjoint IoU");
  check::Require(
      q.ciou.item() < 0 && std::isfinite(derivative[0]) && derivative[0] > 0,
      "disjoint CIoU pulls center toward target");
}
void LossAndAssignmentChecks() {
  std::vector<Dimension> priors(9, {32, 32});
  for (bool positive : {false, true}) {
    std::array<nn::Tensor, 3> raw;
    yolo::YoloV5Targets       targets;
    for (int l = 0; l < 3; ++l) {
      raw[l]  = nn::zeros({1, 1, 1, 3, 7}).requires_grad_();
      auto& t = targets.levels[l];
      t.geometry.resize(12);
      t.classes.resize(6);
      t.positive.resize(3);
    }
    if (positive) {
      targets.levels[0].geometry   = {.5, .5, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
      targets.levels[0].classes[0] = targets.levels[0].positive[0] = 1;
    }
    auto loss      = yolo::YoloV5Loss(raw, targets, 2, priors);
    auto gradients = nn::grad(loss, {raw[0], raw[1], raw[2]});
    check::Near(
        loss.item(),
        (positive ? 10.4 : 5.4) * std::log(2.),
        3e-5,
        "balanced objectness/class host loss");
    const double balance[] = {.4, 1, 4};
    for (int l = 0; l < 3; ++l) {
      auto g = gradients[l].to_vector<float>();
      for (int a = 0; a < 3; ++a)
        for (int k = 0; k < 7; ++k) {
          double expected = k == 4 ? balance[l] / 6 : 0;
          if (positive && l == 0 && a == 0) {
            if (k == 4)
              expected *= -1;
            if (k == 5)
              expected = -1.25;
            if (k == 6)
              expected = 1.25;
          }
          check::Near(g[a * 7 + k], expected, 2e-5, "empty/perfect analytic head derivatives");
        }
    }
  }
  std::vector<Dimension> dimensions{
      {4, 4}, {6, 6}, {8, 8}, {12, 12}, {16, 16}, {20, 20}, {24, 24}, {32, 32}, {48, 48}};
  std::array<nn::Shape, 3> shapes;
  for (int l = 0; l < 3; ++l)
    shapes[l] = {1, 2 << l, 2 << l, 3, 7};
  DetectionScenes scene;
  scene.height = scene.width = 64;
  scene.boxes                = {{{22, 22, 30, 30}}};
  scene.labels               = {{1}};
  auto targets               = yolo::MakeYoloV5Targets(shapes, dimensions, scene, {0});
  check::Require(targets.collisions == 0, "one target ratio/neighbor no conflicts");
  int assigned = 0;
  for (int l = 0; l < 3; ++l) {
    int  cells = 2 << l;
    auto mask  = targets.levels[l].positive;
    for (int i = 0; i < int(mask.size()); ++i)
      if (mask[i]) {
        ++assigned;
        int   cell = i / 3, row = cell / cells, col = cell % cells;
        float y = 26.f / 64 * cells - row, x = 26.f / 64 * cells - col;
        check::Require(
            y > -.5 && y < 1.5 && x > -.5 && x < 1.5, "neighbor centers within expanded range");
      }
  }
  check::Require(assigned == 21, "seven ratio priors each base plus two neighbors");
}
nn::Tensor Images(int n, int side, int channels) {
  std::vector<float> v(n * side * side * channels);
  for (size_t i = 0; i < v.size(); ++i)
    v[i] = std::sin(.13f * i);
  return nn::from_vector(v, {n, side, side, channels});
}
nn::Tensor Objective(const yolo::YoloV5Outputs& net) {
  auto loss = nn::mean(nn::square(net.raw[0]));
  for (int i = 1; i < 3; ++i)
    loss = loss + nn::mean(nn::square(net.raw[i]));
  return loss;
}
void ModelChecks() {
  nn::ParameterStore full(17), full_state;
  yolo::YoloV5Config original;
  original.update_stats = false;
  auto net              = yolo::YoloV5(full, full_state, Images(1, 128, 3), original, true);
  check::Require(
      full.count() == 7235389 && full_state.count() == 19008,
      "YOLOv5s full parameter/state counts");
  auto g = nn::grad(Objective(net), full.parameters());
  check::Require(g.size() == 177, "57 BN feature convolutions/three biased heads");
  for (auto& value : g)
    check::Require(value.defined(), "every full YOLOv5 parameter connected");
  nn::ParameterStore p(17), state;
  yolo::YoloV5Config cfg;
  cfg.classes          = 10;
  cfg.width_divisor    = 4;
  cfg.grayscale        = true;
  cfg.reference_size   = 64;
  cfg.expected_objects = 3;
  cfg.update_stats     = false;
  double minimum = INFINITY, maximum = 0;
  for (int resolution : {64, 96, 128}) {
    auto compact = yolo::YoloV5(p, state, Images(2, resolution, 1), cfg, true);
    for (int l = 0; l < 3; ++l)
      check::Require(
          compact.raw[l].shape() ==
              nn::Shape({2, resolution / (32 >> l), resolution / (32 >> l), 3, 15}),
          "dynamic CSP/PAN three-scale grid");
    if (resolution == 64) {
      auto derivatives = nn::grad(Objective(compact), p.parameters());
      for (size_t i = 0; i < derivatives.size(); ++i) {
        double norm = std::sqrt(nn::sum(nn::square(derivatives[i])).item());
        check::Require(
            std::isfinite(norm) && norm > 0, "finite nonzero CSP/PAN derivative " + p.names()[i]);
        minimum = std::min(minimum, norm);
        maximum = std::max(maximum, norm);
      }
    }
  }
  std::cout << "full7235389params/177 derivatives;compact" << p.count()
            << " multiscale64/96/128 norms[" << minimum << ',' << maximum << "] PASS" << std::endl;
}
void ExportMosaic(const detector::DetectionScenes& scenes) {
  const auto              mosaic = detector::MosaicScenes(scenes, {{0, 1, 2, 3}});
  constexpr int           scale  = 4;
  const int               h = mosaic.height * scale, w = mosaic.width * scale;
  datamunge::image::Image rgb(w, h, datamunge::image::ImageMode::RGB);
  std::vector<uint8_t>    bytes(h * w * 3);
  for (int r = 0; r < h; ++r)
    for (int c = 0; c < w; ++c)
      for (int k = 0; k < 3; ++k)
        bytes[(r * w + c) * 3 + k] = mosaic.pixels[(r / scale) * mosaic.width + c / scale];
  for (size_t n = 0; n < mosaic.boxes[0].size(); ++n) {
    const auto&   box     = mosaic.boxes[0][n];
    const int     label   = mosaic.labels[0][n];
    const uint8_t color[] = {
        uint8_t(80 + (37 * label) % 176),
        uint8_t(80 + (83 * label + 90) % 176),
        uint8_t(80 + (113 * label + 30) % 176)};
    const int y0 = std::clamp(int(box[0] * scale), 0, h - 1),
              x0 = std::clamp(int(box[1] * scale), 0, w - 1),
              y1 = std::clamp(int(std::ceil(box[2] * scale)) - 1, 0, h - 1),
              x1 = std::clamp(int(std::ceil(box[3] * scale)) - 1, 0, w - 1);
    for (int r = y0; r <= y1; ++r)
      for (int k = 0; k < 3; ++k)
        bytes[(r * w + x0) * 3 + k] = bytes[(r * w + x1) * 3 + k] = color[k];
    for (int c = x0; c <= x1; ++c)
      for (int k = 0; k < 3; ++k)
        bytes[(y0 * w + c) * 3 + k] = bytes[(y1 * w + c) * 3 + k] = color[k];
  }
  for (int r = 0; r < h; ++r)
    for (int c = 0; c < w; ++c)
      rgb.set_pixel(
          c,
          r,
          {bytes[(r * w + c) * 3], bytes[(r * w + c) * 3 + 1], bytes[(r * w + c) * 3 + 2], 255});
  auto root = paths::OutputDir("visualizations");
  datamunge::image::write_png(rgb, (root / "detection_mosaic.png").string());
  std::cout << "annotated mosaic: " << (root / "detection_mosaic.png") << std::endl;
}
} // namespace
int main(int argc, char** argv) {
  try {
    nn::set_num_threads(2);
    int                              scene_size = 128;
    auto                             directory  = paths::Dataset("mnist");
    bool                             mosaic = true, checks_only = false;
    detector::DetectorTrainingConfig cfg;
    cfg.name              = "yolo_v5_style_digits";
    cfg.epochs            = 3;
    cfg.accuracy          = false;
    cfg.minimum_test_ap50 = .7;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      auto        eq  = arg.find('=');
      if (eq == std::string::npos)
        throw std::invalid_argument("key=value option");
      auto key = arg.substr(0, eq), value = arg.substr(eq + 1);
      if (key == "size") {
        scene_size = std::stoi(value);
        check::Require(scene_size >= 64 && scene_size % 32 == 0, "size>=64 divisible32");
      } else if (key == "data")
        directory = value;
      else if (key == "epochs")
        cfg.epochs = std::stoi(value);
      else if (key == "accuracy" && (value == "0" || value == "1"))
        cfg.accuracy = value == "1";
      else if (key == "warm_start")
        cfg.warm_start = value;
      else if (key == "lr")
        cfg.learning_rate = std::stof(value);
      else if (key == "threads")
        nn::set_num_threads(std::stoi(value));
      else if (key == "checks_only" && (value == "0" || value == "1"))
        checks_only = value == "1";
      else if (key == "mosaic" && (value == "0" || value == "1"))
        mosaic = value == "1";
      else
        throw std::invalid_argument("unknown option " + key);
    }
    CspChecks();
    PoolingChecks();
    MosaicChecks();
    QualityChecks();
    LossAndAssignmentChecks();
    ModelChecks();
    if (checks_only) {
      std::cout << "All CSP/PAN/mosaic fixtures passed (training not requested).\n";
      return 0;
    }
    auto [train, test] = detector::StandardScenes(directory, scene_size);
    ExportMosaic(train);
    std::vector<Dimension> dimensions;
    for (int row = 0; row < 5500; ++row)
      for (const auto& b : train.boxes[row])
        dimensions.push_back({b[2] - b[0], b[3] - b[1]});
    auto               priors = yolo::ClusterDimensions(dimensions, 9);
    yolo::YoloV5Config model;
    model.classes          = 10;
    model.width_divisor    = 4;
    model.grayscale        = true;
    model.reference_size   = scene_size;
    model.expected_objects = 3;
    std::mt19937                       random(41);
    int                                step = 0, mosaic_batches = 0, collisions = 0;
    std::uniform_int_distribution<int> coin(0, 1), extra(0, 5499);
    detector::DetectorHooks            hooks;
    hooks.build = [&](nn::ParameterStore& p, nn::ParameterStore& state) {
      auto frozen         = model;
      frozen.update_stats = false;
      yolo::YoloV5(p, state, nn::zeros({1, scene_size, scene_size, 1}), frozen, true);
    };
    hooks.loss = [&](nn::ParameterStore&     p,
                     nn::ParameterStore&     state,
                     const DetectionScenes&  scenes,
                     const std::vector<int>& rows) {
      DetectionScenes batch;
      if (mosaic && coin(random)) {
        std::vector<std::array<int, 4>> selections;
        for (int row : rows)
          selections.push_back({row, extra(random), extra(random), extra(random)});
        batch = detector::MosaicScenes(scenes, selections);
        ++mosaic_batches;
      } else
        batch = detector::CopyDetectionRows(scenes, rows);
      for (int row = 0; row < batch.size(); ++row)
        if (coin(random))
          detector::FlipDetectionScene(batch, row);
      std::vector<int> indices(batch.size());
      std::iota(indices.begin(), indices.end(), 0);
      int  resolution = scene_size + 32 * ((step++ / 10) % 3);
      auto net        = yolo::YoloV5(
          p,
          state,
          nn::resize_bilinear(detector::SceneImages(batch, indices), {resolution, resolution}),
          model,
          true);
      std::array<nn::Shape, 3> shapes;
      for (int l = 0; l < 3; ++l)
        shapes[l] = net.raw[l].shape();
      auto targets = yolo::MakeYoloV5Targets(shapes, priors, batch, indices);
      collisions += targets.collisions;
      return yolo::YoloV5Loss(net.raw, targets, 10, priors);
    };
    hooks.detect = [&](nn::ParameterStore&     p,
                       nn::ParameterStore&     state,
                       const DetectionScenes&  scenes,
                       const std::vector<int>& rows) {
      return yolo::DetectYoloV5(
          [&](const nn::Tensor& images) {
            return yolo::YoloV5(p, state, images, model, false).raw;
          },
          priors,
          scenes,
          rows);
    };
    {
      using namespace detector;
      auto training_cfg    = cfg;
      bool explicit_epochs = false;
      for (int option = 1; option < argc; ++option)
        explicit_epochs = explicit_epochs || std::string(argv[option]).rfind("epochs=", 0) == 0;
      if (!explicit_epochs)
        training_cfg.epochs = example_runtime::Limit(training_cfg.epochs, 1);
      training_cfg.batch      = example_runtime::Limit(training_cfg.batch, 2);
      training_cfg.validation = example_runtime::Limit(training_cfg.validation, 8);
      if (!example_runtime::Full())
        training_cfg.fit = training_cfg.fit ? std::min(training_cfg.fit, 8) : 8;
      example_runtime::Describe();
      check::Require(
          training_cfg.validation > 0 && training_cfg.validation < train.size() &&
              training_cfg.epochs > 0 && training_cfg.batch > 0,
          "valid detector training configuration");
      check::Require(
          !example_runtime::Full() || training_cfg.accuracy || training_cfg.epochs >= 2,
          "short learning demonstration needs at least two epochs");
      const int        fit_count = training_cfg.fit > 0
                                       ? std::min(training_cfg.fit, train.size() - training_cfg.validation)
                                       : train.size() - training_cfg.validation;
      std::vector<int> fit(static_cast<std::size_t>(fit_count)),
          validation(static_cast<std::size_t>(training_cfg.validation)),
          test_rows(example_runtime::Limit<std::size_t>(test.size(), 8));
      std::iota(fit.begin(), fit.end(), 0);
      std::iota(validation.begin(), validation.end(), train.size() - training_cfg.validation);
      std::iota(test_rows.begin(), test_rows.end(), 0);
      nn::ParameterStore params(17), state(17);
      {
        nn::NoGradGuard guard;
        hooks.build(params, state);
      }
      std::vector<nn::Tensor> kernels;
      for (const auto& p : params.parameters())
        if (p.dim() >= 2)
          kernels.push_back(p);
      const auto directory = TrainingDirectory(training_cfg.name);
      std::filesystem::create_directories(directory);
      const auto prefix = (directory / "best").string();
      if (!training_cfg.warm_start.empty()) {
        check::Require(
            std::filesystem::absolute(training_cfg.warm_start).lexically_normal() !=
                std::filesystem::absolute(prefix).lexically_normal(),
            "warm start requires a separate output directory/run tag");
        RestoreCheckpoint(params, state, training_cfg.warm_start);
        std::cout << "  warm start: " << training_cfg.warm_start << "; fresh Adam moments"
                  << std::endl;
      }
      nn::OptimizerConfig optimizer_config;
      optimizer_config.kind = nn::OptimizerKind::Adam;
      nn::Optimizer optimizer(params, training_cfg.learning_rate, optimizer_config);
      std::ofstream history(directory / "history.tsv");
      history << "epoch\ttraining_loss\tvalidation_ap50\tvalidation_coco_ap\n";
      std::cout << "== " << training_cfg.name << ": " << fit_count << " fit / "
                << training_cfg.validation << " validation / " << test_rows.size()
                << " test scenes (" << train.height << "x" << train.width
                << ")\n  parameters=" << params.count() << "; batch=" << training_cfg.batch
                << "; lr=" << training_cfg.learning_rate << std::endl;
      nn::BatchOrder      batches(std::size_t(fit_count), std::size_t(training_cfg.batch), 29);
      double              best       = -1;
      int                 best_epoch = -1;
      std::vector<double> epoch_losses;
      if (!training_cfg.warm_start.empty()) {
        const auto baseline = EvaluateDetections(
            hooks.detect(params, state, train, validation),
            train,
            validation,
            training_cfg.classes);
        best       = baseline.ap50;
        best_epoch = 0;
        SaveCheckpoint(params, state, prefix);
        history << 0 << "\tnan\t" << baseline.ap50 << '\t' << baseline.coco << '\n';
        history.flush();
        std::cout << "  restored baseline validation " << ApSummary(baseline) << std::endl;
      }
      for (int epoch = 1; epoch <= training_cfg.epochs; ++epoch) {
        batches.begin_epoch();
        double      total = 0;
        std::size_t seen  = 0;
        for (;;) {
          const auto picked = batches.next();
          if (picked.empty())
            break;
          std::vector<int> rows;
          for (auto k : picked)
            rows.push_back(fit[std::size_t(k)]);
          nn::Tensor   objective  = hooks.loss(params, state, train, rows);
          const double batch_loss = objective.item();
          check::Require(std::isfinite(batch_loss), "finite detector batch loss");
          if (training_cfg.l2 > 0 && !kernels.empty()) {
            nn::Tensor penalty = nn::sum(nn::square(kernels[0]));
            for (std::size_t i = 1; i < kernels.size(); ++i)
              penalty = penalty + nn::sum(nn::square(kernels[i]));
            objective = objective + 0.5 * training_cfg.l2 * penalty;
          }
          optimizer.minimize(objective);
          total += batch_loss * double(rows.size());
          seen += rows.size();
        }
        epoch_losses.push_back(total / double(seen));
        check::Require(std::isfinite(epoch_losses.back()), "finite detector epoch mean");
        const auto result = EvaluateDetections(
            hooks.detect(params, state, train, validation),
            train,
            validation,
            training_cfg.classes);
        check::Require(std::isfinite(result.ap50), "finite validation AP50");
        if (result.ap50 > best) {
          best       = result.ap50;
          best_epoch = epoch;
          SaveCheckpoint(params, state, prefix);
        }
        history << epoch << '\t' << total / double(seen) << '\t' << result.ap50 << '\t'
                << result.coco << '\n';
        history.flush();
        std::cout << "  epoch " << epoch << ": training loss=" << total / double(seen)
                  << ", validation " << ApSummary(result) << std::endl;
      }
      RestoreCheckpoint(params, state, prefix);
      const auto restored = EvaluateDetections(
          hooks.detect(params, state, train, validation), train, validation, training_cfg.classes);
      check::Near(
          restored.ap50, best, 1e-6, "restored checkpoint reproduces selected validation AP50");
      std::cout << "  restored validation AP50=" << restored.ap50
                << "; checkpoint agreement verified" << std::endl;
      const auto result = EvaluateDetections(
          hooks.detect(params, state, test, test_rows), test, test_rows, training_cfg.classes);
      std::cout << "  selected epoch=" << best_epoch << "; test " << ApSummary(result)
                << "\n  per-class test AP50:";
      for (double ap : result.per_class50)
        std::cout << ' ' << std::fixed << std::setprecision(3) << ap;
      std::cout << std::defaultfloat << std::setprecision(6)
                << "\n  best checkpoint and history: " << directory.string() << std::endl;
      if (hooks.inspect)
        hooks.inspect(params, state, test, test_rows, directory);
      if (example_runtime::Full())
        CheckDetectorDemonstration(training_cfg, epoch_losses, result);
      else
        check::Require(std::isfinite(result.ap50), "finite quick detector evaluation");
      if (example_runtime::Full() && !training_cfg.accuracy)
        std::cout << "  short learning demonstration passed: mean loss " << epoch_losses.front()
                  << " -> " << epoch_losses.back()
                  << "; accuracy benchmark not requested (accuracy=1 enables AP50 gate "
                  << training_cfg.minimum_test_ap50 << ")" << std::endl;
    }
    std::cout << "mosaic batches=" << mosaic_batches
              << ";ratio/neighbor collisions resolved=" << collisions
              << "\nAll CSP/PAN/mosaic detection checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
