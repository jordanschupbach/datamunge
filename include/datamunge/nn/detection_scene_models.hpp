#pragma once
/// @file detection_scene_models.hpp
/// @brief Tensor MNIST detection scenes, pixel-centre crops and discrete component proposals.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/boxes.hpp"
#include "datamunge/nn/datasets.hpp"
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/random.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Configure quantized MNIST digits placed on grayscale detection canvases.
///
/// Integer settings describe data geometry and sampling. No learned parameters or
/// fitting. A placement can fail after thirty attempts, so a scene can contain
/// fewer than the requested random object count, including zero for blank digits.
///
/// @par Example
/// @code
/// dm::nn::models::DetectionSceneConfig cfg;
/// cfg.scenes = 8;
/// cfg.seed = 11;
/// @endcode
/// @par References
/// - LeCun et al. (1998), Gradient-based learning applied to document recognition, https://doi.org/10.1109/5.726791; synthetic scene convention from examples/nn/common/detection_scenes.hpp
/// @see MakeTensorDetectionScenes()
struct DetectionSceneConfig {
  int scenes = 1000;   ///< Positive number of scenes.
  int height = 64;     ///< Positive canvas height.
  int width = 64;      ///< Positive canvas width.
  int max_objects = 4; ///< Positive inclusive maximum number of attempted objects per scene.
  int min_size = 16;   ///< Minimum sampled square digit-cell side, at least four.
  int max_size = 32;   ///< Maximum cell side, at least min_size and no larger than the canvas.
  double max_overlap =
      .3; ///< Finite allowed maximum IoU with every earlier accepted box, in [0,1].
  std::uint64_t seed =
      1; ///< Seed for mt19937_64 discrete count/digit/size/placement metadata.
};
/// @brief Own scene pixels and per-image variable-length tensor annotations.
///
/// Images and boxes use the source image Float32/64 dtype; labels use Int64.
/// All data are detached. One annotation list exists per image, including empty
/// [0,4] boxes and [0] labels when placements fail.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::DetectionSceneConfig cfg;
/// cfg.scenes = 1;
/// const auto scenes = n::models::MakeTensorDetectionScenes(n::ones({1, 28, 28, 1}),
///                                                          n::zeros({1}, n::DType::Int64), cfg);
/// @endcode
/// @par References
/// - LeCun et al. (1998), Gradient-based learning applied to document recognition, https://doi.org/10.1109/5.726791; synthetic scene convention from examples/nn/common/detection_scenes.hpp
/// @see MakeTensorDetectionScenes()
struct TensorDetectionScenes {
  nn::Tensor images; ///< [N,H,W,1] grayscale pixels in [0,1], quantized to multiples of 1/255.
  std::vector<nn::Tensor>
      boxes; ///< N tensors [Ki,4], tight half-open pixel boxes above byte threshold 32.
  std::vector<nn::Tensor> labels; ///< N Int64 tensors [Ki] of digit classes 0 through9.
};
/// @brief Internal shared scene sampler for boxes and optional per-object supports.
namespace detection_scene_detail {
/// @brief Generate canonical scene pixels/annotations,optionally retaining every accepted support mask.
/// @param digits Floating[N,28,28,1] normalized MNIST pixels.
/// @param labels Aligned Int64 digit labels in[0,10).
/// @param cfg Valid scene metadata/discrete RNG settings.
/// @param retained_masks Optional output;replaces its contents only on success with Bool[Ki,H,W] per scene.
/// @return Detached scene pixels/boxes/labels,identical whether support retention is requested or not.
/// @throws std::invalid_argument For invalid data/configuration,as in MakeTensorDetectionScenes().
inline TensorDetectionScenes Generate(const nn::Tensor &digits, const nn::Tensor &labels,
                                      const DetectionSceneConfig &cfg,
                                      std::vector<nn::Tensor> *retained_masks) {
  if (digits.dim() != 4 || digits.shape()[0] < 1 ||
      digits.shape()[0] > std::numeric_limits<int>::max() || digits.shape()[1] != 28 ||
      digits.shape()[2] != 28 || digits.shape()[3] != 1 ||
      (digits.dtype() != nn::DType::Float32 && digits.dtype() != nn::DType::Float64) ||
      labels.dtype() != nn::DType::Int64 || labels.shape() != nn::Shape{digits.shape()[0]} ||
      !nn::all(nn::logical_and(nn::greater_equal(digits, 0.), nn::less_equal(digits, 1.)))
           .item<bool>() ||
      !nn::all(nn::logical_and(nn::greater_equal(labels, 0.), nn::less(labels, 10.)))
           .item<bool>() ||
      cfg.scenes < 1 || cfg.height < 1 || cfg.width < 1 || cfg.max_objects < 1 ||
      cfg.min_size < 4 || cfg.max_size < cfg.min_size ||
      cfg.max_size > std::min(cfg.height, cfg.width) || !std::isfinite(cfg.max_overlap) ||
      cfg.max_overlap < 0 || cfg.max_overlap > 1)
    throw std::invalid_argument("invalid tensor detection scene inputs/configuration");
  nn::NoGradGuard guard;
  const auto type = nn::DType::Float64;
  const auto bytes = nn::floor(digits.to(type) * 255. + .5);
  std::mt19937_64 rng(cfg.seed);
  std::uniform_int_distribution<int> count(1, cfg.max_objects),
      pick(0, int(digits.shape()[0]) - 1), size(cfg.min_size, cfg.max_size);
  TensorDetectionScenes out;
  std::vector<nn::Tensor> canvases, all_masks;
  for (int scene = 0; scene < cfg.scenes; ++scene) {
    auto canvas = nn::zeros({1, cfg.height, cfg.width, 1}, type);
    std::vector<nn::Tensor> boxes, classes, masks;
    const int objects = count(rng);
    for (int object = 0; object < objects; ++object)
      for (int attempt = 0; attempt < 30; ++attempt) {
        const int digit = pick(rng), cell = size(rng);
        std::uniform_int_distribution<int> top(0, cfg.height - cell),
            left(0, cfg.width - cell);
        const int y = top(rng), x = left(rng);
        const auto coord = (nn::arange(0, cell).to(type) + .5) * 28. / double(cell) - .5;
        const auto xx = nn::ones({cell, cell}, type) * coord.reshape({1, cell}),
                   yy = nn::ones({cell, cell}, type) * coord.reshape({cell, 1});
        const auto patch = nn::floor(
            nn::sample_pixels(nn::slice_axis(bytes, 0, digit, digit + 1),
                              nn::stack({xx, yy}, 2).reshape({1, cell, cell, 2}),
                              nn::GridSampleMode::Bilinear, nn::GridSamplePadding::Border) +
            .5);
        const auto mask = nn::greater(patch.reshape({cell, cell}), 32.);
        if (!nn::any(mask).item<bool>())
          continue;
        const auto rr = nn::arange(0, cell).to(type).reshape({cell, 1}) *
                        nn::ones({cell, cell}, type),
                   cc = nn::arange(0, cell).to(type).reshape({1, cell}) *
                        nn::ones({cell, cell}, type);
        const auto ymin = nn::min(nn::where(mask, rr, nn::full({}, double(cell), type))) + y,
                   xmin = nn::min(nn::where(mask, cc, nn::full({}, double(cell), type))) + x;
        const auto ymax = nn::max(nn::where(mask, rr + 1., nn::zeros({}, type))) + y,
                   xmax = nn::max(nn::where(mask, cc + 1., nn::zeros({}, type))) + x;
        const auto box = nn::stack({ymin, xmin, ymax, xmax}, 0).reshape({1, 4});
        if (!boxes.empty() &&
            nn::any(nn::greater(nn::box_iou(nn::concat(boxes, 0), box), cfg.max_overlap))
                .item<bool>())
          continue;
        canvas = nn::maximum(
            canvas,
            nn::pad(patch,
                    {{0, 0}, {y, cfg.height - y - cell}, {x, cfg.width - x - cell}, {0, 0}}));
        if (retained_masks)
          masks.push_back(
              nn::pad(mask, {{y, cfg.height - y - cell}, {x, cfg.width - x - cell}}));
        boxes.push_back(box);
        classes.push_back(nn::slice_axis(labels, 0, digit, digit + 1));
        break;
      }
    if (retained_masks)
      all_masks.push_back(masks.empty()
                              ? nn::zeros({0, cfg.height, cfg.width}, nn::DType::Bool)
                              : nn::stack(masks, 0));
    canvases.push_back(canvas);
    out.boxes.push_back(boxes.empty() ? nn::zeros({0, 4}, digits.dtype())
                                      : nn::concat(boxes, 0).to(digits.dtype()));
    out.labels.push_back(classes.empty() ? nn::zeros({0}, nn::DType::Int64)
                                         : nn::concat(classes, 0));
  }
  out.images = (nn::concat(canvases, 0) / 255.).to(digits.dtype());
  if (retained_masks)
    *retained_masks = std::move(all_masks);
  return out;
}
} // namespace detection_scene_detail
/// @brief Generate detection scenes using tensor interpolation, quantization and compositing.
///
/// Each object gets up to thirty placement attempts. Cells use half-pixel bilinear
/// sampling with border clamping, then positive half-up byte rounding. Boxes bound
/// pixels strictly above32. Accepted patches combine by maximum, not addition.
/// Discrete draws preserve legacy mt19937_64 draw order. Float64 tensor arithmetic
/// replaces host interpolation; borderline rounding may differ by one byte.
/// Input graphs are not retained; the routine neither fits nor mutates inputs.
/// Work is proportional to scene pixels and attempted cell pixels.
///
/// @param digits Float32/64 [N,28,28,1] in [0,1], N positive; restored to nearest source byte before resampling.
/// @param labels Int64 [N] digit classes in [0,9], matching digits.
/// @param cfg Positive valid scene dimensions, placement limits, overlap and seed.
/// @return Detached normalized scene images and aligned per-image box/label tensors.
/// @throws std::invalid_argument For invalid shapes/dtypes, image/label ranges or configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::DetectionSceneConfig cfg;
/// cfg.scenes = 2;
/// const auto scenes = n::models::MakeTensorDetectionScenes(n::ones({1, 28, 28, 1}),
///                                                          n::zeros({1}, n::DType::Int64), cfg);
/// @endcode
/// @par References
/// - LeCun et al. (1998), Gradient-based learning applied to document recognition, https://doi.org/10.1109/5.726791; synthetic scene convention from examples/nn/common/detection_scenes.hpp
/// @see CropDetectionPatches(), ComponentDetectionProposals()
inline TensorDetectionScenes MakeTensorDetectionScenes(const nn::Tensor &digits,
                                                       const nn::Tensor &labels,
                                                       const DetectionSceneConfig &cfg = {}) {
  return detection_scene_detail::Generate(digits, labels, cfg, nullptr);
}
/// @brief Crop pixel-centre boxes with bilinear zero padding using tensor sampling.
///
/// Output (r,c) samples (ymin+(r+.5)*height/side-.5, xmin+(c+.5)*width/side-.5).
/// Outside taps read zero. This is pixel-centre cropping, not TF endpoint-aligned
/// crop_and_resize. It neither normalizes nor fits. Work O(B*side²*C). Empty B
/// returns an empty tensor; nonempty calls preserve both image and box gradients.
///
/// @param images Float32/64 [N,H,W,C], all dimensions positive; values are not rescaled or range-checked.
/// @param rows Int64 [B] valid source-image indices; repeats are allowed.
/// @param boxes Floating [B,4] finite ordered pixel-edge coordinates; input dtype is converted to images dtype.
/// @param side Positive square output side, default28.
/// @return [B,side,side,C] sampled images, retaining gradients to images and boxes.
/// @throws std::invalid_argument For malformed shapes/dtypes, invalid rows, nonfinite/reversed boxes or side.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto patches = n::models::CropDetectionPatches(
///     n::ones({1, 32, 32, 1}), n::from_vector(std::vector<std::int64_t>{0}),
///     n::tensor({0., 0., 32., 32.}, {1, 4}));
/// @endcode
/// @par References
/// - LeCun et al. (1998), Gradient-based learning applied to document recognition, https://doi.org/10.1109/5.726791; synthetic scene convention from examples/nn/common/detection_scenes.hpp
/// @see SquareDetectionBoxes()
inline nn::Tensor CropDetectionPatches(const nn::Tensor &images, const nn::Tensor &rows,
                                       const nn::Tensor &boxes, int side = 28) {
  if (images.dim() != 4 || images.shape()[0] < 1 || images.shape()[1] < 1 ||
      images.shape()[2] < 1 || images.shape()[3] < 1 ||
      (images.dtype() != nn::DType::Float32 && images.dtype() != nn::DType::Float64) ||
      rows.dim() != 1 || rows.dtype() != nn::DType::Int64 || boxes.dim() != 2 ||
      boxes.shape()[0] != rows.numel() || boxes.shape()[1] != 4 ||
      (boxes.dtype() != nn::DType::Float32 && boxes.dtype() != nn::DType::Float64) || side < 1)
    throw std::invalid_argument("invalid detection crop shapes/dtypes");
  const auto B = rows.numel();
  if (!B)
    return nn::zeros({0, side, side, images.shape()[3]}, images.dtype());
  if (!nn::all(nn::logical_and(nn::greater_equal(rows, 0.),
                               nn::less(rows, double(images.shape()[0]))))
           .item<bool>() ||
      !nn::all(nn::less(nn::abs(boxes), std::numeric_limits<double>::infinity())).item<bool>())
    throw std::invalid_argument("invalid crop rows or nonfinite boxes");
  const auto b = boxes.to(images.dtype());
  const auto y1 = nn::slice_axis(b, 1, 0, 1).reshape({B, 1, 1}),
             x1 = nn::slice_axis(b, 1, 1, 2).reshape({B, 1, 1}),
             y2 = nn::slice_axis(b, 1, 2, 3).reshape({B, 1, 1}),
             x2 = nn::slice_axis(b, 1, 3, 4).reshape({B, 1, 1});
  if (nn::any(nn::logical_or(nn::less(y2, y1), nn::less(x2, x1))).item<bool>())
    throw std::invalid_argument("reversed crop boxes");
  const auto t = (nn::arange(0, side).to(images.dtype()) + .5) / double(side);
  const auto yy = y1 + t.reshape({1, side, 1}) * (y2 - y1) - .5,
             xx = x1 + t.reshape({1, 1, side}) * (x2 - x1) - .5;
  const auto full = nn::ones({B, side, side}, images.dtype());
  return nn::sample_pixels(nn::gather(images, rows), nn::stack({xx * full, yy * full}, 3));
}
/// @brief Expand pixel boxes to centred squares for patch classification.
///
/// Squares may extend outside an image. No clipping, resampling, mutation or fitting.
/// The original centre is preserved and side=max(height,width)*margin.
///
/// @param boxes Float32/64 [...,4] finite ordered coordinates; zero extent is allowed.
/// @param margin Finite strictly positive multiplier of the larger side, default1.15.
/// @return Same-shaped tensor of square boxes; gradients reach source boxes.
/// @throws std::invalid_argument For invalid shape/dtype, nonfinite/reversed boxes or margin.
/// @par Example
/// @code
/// const auto square =
///     dm::nn::models::SquareDetectionBoxes(dm::nn::tensor({0., 0., 10., 20.}, {1, 4}), 1.);
/// @endcode
/// @par References
/// - LeCun et al. (1998), Gradient-based learning applied to document recognition, https://doi.org/10.1109/5.726791; synthetic scene convention from examples/nn/common/detection_scenes.hpp
/// @see CropDetectionPatches()
inline nn::Tensor SquareDetectionBoxes(const nn::Tensor &boxes, double margin = 1.15) {
  if (boxes.dim() < 1 || boxes.shape().back() != 4 ||
      (boxes.dtype() != nn::DType::Float32 && boxes.dtype() != nn::DType::Float64) ||
      !std::isfinite(margin) || margin <= 0 ||
      (boxes.numel() &&
       !nn::all(nn::less(nn::abs(boxes), std::numeric_limits<double>::infinity()))
            .item<bool>()))
    throw std::invalid_argument("invalid square box input");
  const auto c = nn::boxes_to_center(boxes);
  const auto cy = nn::slice_axis(c, -1, 0, 1), cx = nn::slice_axis(c, -1, 1, 2),
             h = nn::slice_axis(c, -1, 2, 3), w = nn::slice_axis(c, -1, 3, 4);
  if (nn::any(nn::logical_or(nn::less(h, 0.), nn::less(w, 0.))).item<bool>())
    throw std::invalid_argument("reversed square boxes");
  const auto side = nn::maximum(h, w) * margin;
  return nn::boxes_from_center(nn::concat({cy, cx, side, side}, -1));
}
/// @brief Propose tight connected-component boxes and nearby component-pair unions.
///
/// Quantizes pixels to nearest byte and thresholds through tensor ops. A host
/// queue traverses discrete 8-connected foreground indices; all box extrema,
/// areas, pair gaps and unions use tensor reductions. Components with box area
/// less than9 are omitted. Only original component pairs are merged; no recursive
/// union search or selective-search hierarchy. Returns empty [0,4] for blank
/// images or no thresholds. No gradients or fitting. Work O(T*H*W+T*C²+P²)
/// including pair unions and duplicate removal; host indices are graph metadata.
///
/// @param image Float32/64 [H,W,1] image, positive H/W, finite pixels in [0,1].
/// @param thresholds Byte cutoffs in [0,255], default16/32/64/96/128/160/200; strict foreground comparison.
/// @param gap Nonnegative maximum vertical/horizontal edge gap for component-pair unions, default3.
/// @return Detached [P,4] pixel boxes in image dtype, with exact duplicates removed in first-occurrence order.
/// @throws std::invalid_argument For invalid image shape/range/dtype, thresholds or gap.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto proposals =
///     n::models::ComponentDetectionProposals(n::ones({12, 12, 1}), {32}, 3); // [1,4]
/// @endcode
/// @par References
/// - Synthetic scene and proposal conventions from examples/nn/common/detection_scenes.hpp; Rosenfeld and Pfaltz (1966), Sequential Operations in Digital Picture Processing, https://doi.org/10.1145/321356.321357
/// @see MakeTensorDetectionScenes(), SquareDetectionBoxes()
inline nn::Tensor ComponentDetectionProposals(
    const nn::Tensor &image,
    const std::vector<int> &thresholds = {16, 32, 64, 96, 128, 160, 200}, int gap = 3) {
  if (image.dim() != 3 || image.shape()[0] < 1 || image.shape()[1] < 1 ||
      image.shape()[2] != 1 ||
      (image.dtype() != nn::DType::Float32 && image.dtype() != nn::DType::Float64) ||
      !nn::all(nn::logical_and(nn::greater_equal(image, 0.), nn::less_equal(image, 1.)))
           .item<bool>() ||
      gap < 0)
    throw std::invalid_argument("invalid component image or gap");
  for (int t : thresholds)
    if (t < 0 || t > 255)
      throw std::invalid_argument("component thresholds must be bytes");
  nn::NoGradGuard guard;
  const auto H = image.shape()[0], W = image.shape()[1];
  const auto type = image.dtype();
  const auto bytes = nn::floor(image * 255. + .5).reshape({H * W});
  std::vector<nn::Tensor> all;
  for (int threshold : thresholds) {
    const auto foreground = nn::greater(bytes, double(threshold)).to_vector<bool>();
    std::vector<bool> visited(std::size_t(H * W), false);
    std::vector<nn::Tensor> boxes;
    for (std::int64_t start = 0; start < H * W; ++start) {
      if (!foreground[start] || visited[start])
        continue;
      std::deque<std::int64_t> queue{start};
      visited[start] = true;
      std::vector<std::int64_t> rows, cols;
      while (!queue.empty()) {
        const auto index = queue.front();
        queue.pop_front();
        const auto row = index / W, col = index % W;
        rows.push_back(row);
        cols.push_back(col);
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            const auto y = row + dy, x = col + dx;
            if (y < 0 || y >= H || x < 0 || x >= W)
              continue;
            const auto next = y * W + x;
            if (foreground[next] && !visited[next]) {
              visited[next] = true;
              queue.push_back(next);
            }
          }
      }
      const auto rr = nn::from_vector(rows).to(type), cc = nn::from_vector(cols).to(type);
      const auto box =
          nn::stack({nn::min(rr), nn::min(cc), nn::max(rr) + 1., nn::max(cc) + 1.}, 0)
              .reshape({1, 4});
      if (nn::greater_equal(nn::box_area(box), 9.).item<bool>())
        boxes.push_back(box);
    }
    all.insert(all.end(), boxes.begin(), boxes.end());
    for (std::size_t i = 0; i < boxes.size(); ++i)
      for (std::size_t j = i + 1; j < boxes.size(); ++j) {
        // Tensor coordinate accessor; borrows the immutable per-threshold box list.
        const auto coord = [&](std::size_t k, int c) {
          return nn::slice_axis(boxes[k], 1, c, c + 1);
        };
        const auto zero = nn::zeros({1, 1}, type);
        const auto dy = nn::maximum(zero, nn::maximum(coord(i, 0) - coord(j, 2),
                                                      coord(j, 0) - coord(i, 2))),
                   dx = nn::maximum(zero, nn::maximum(coord(i, 1) - coord(j, 3),
                                                      coord(j, 1) - coord(i, 3)));
        if (nn::logical_and(nn::less_equal(dy, double(gap)), nn::less_equal(dx, double(gap)))
                .item<bool>())
          all.push_back(nn::concat({nn::minimum(nn::slice_axis(boxes[i], 1, 0, 2),
                                                nn::slice_axis(boxes[j], 1, 0, 2)),
                                    nn::maximum(nn::slice_axis(boxes[i], 1, 2, 4),
                                                nn::slice_axis(boxes[j], 1, 2, 4))},
                                   1));
      }
  }
  std::vector<nn::Tensor> unique;
  for (const auto &box : all) {
    bool duplicate = false;
    for (const auto &b : unique)
      if (nn::all(nn::equal(box, b)).item<bool>()) {
        duplicate = true;
        break;
      }
    if (!duplicate)
      unique.push_back(box);
  }
  return unique.empty() ? nn::zeros({0, 4}, type) : nn::concat(unique, 0);
}
/// @brief Overlay one-pixel colored box outlines on grayscale tensor scene pixels.
///
/// Uses tensor coordinate grids and where masks; boxes use half-open edges, so
/// the far outline is at ymax-1/xmax-1. Coordinates are rounded only by pixel
/// coverage tolerance .51. Out-of-image lines naturally clip. Images and boxes
/// are not mutated; gradients to image pixels outside outlines remain available.
/// This is presentation, not model fitting. Work O(H*W*sum(Ki)).
///
/// @param images Float32/64 [N,H,W,1] pixels; values are displayed unchanged, typically [0,1].
/// @param boxes N tensors [Ki,4] of finite ordered pixel boxes; no labels or scores are drawn.
/// @param colour Three finite RGB values in [0,1], default red.
/// @return [N,H,W,3] tensor, grayscale expanded to RGB with colored one-pixel centre-aligned outlines.
/// @throws std::invalid_argument For invalid image shape/dtype, box count/shape/coordinates or colour.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto rgb = n::models::DrawDetectionBoxes(n::zeros({1, 16, 16, 1}),
///                                                {n::tensor({2., 3., 12., 13.}, {1, 4})});
/// @endcode
/// @see MakeTensorDetectionScenes(), CropDetectionPatches()
inline nn::Tensor DrawDetectionBoxes(const nn::Tensor &images,
                                     const std::vector<nn::Tensor> &boxes,
                                     const std::array<double, 3> &colour = {1., .15, .15}) {
  if (images.dim() != 4 || images.shape()[0] < 1 || images.shape()[1] < 1 ||
      images.shape()[2] < 1 || images.shape()[3] != 1 ||
      std::int64_t(boxes.size()) != images.shape()[0] ||
      (images.dtype() != nn::DType::Float32 && images.dtype() != nn::DType::Float64))
    throw std::invalid_argument("invalid detection drawing images/boxes");
  for (double v : colour)
    if (!std::isfinite(v) || v < 0 || v > 1)
      throw std::invalid_argument("invalid RGB colour");
  const auto type = images.dtype();
  const auto H = images.shape()[1], W = images.shape()[2];
  const auto yy = nn::arange(0, H).to(type).reshape({1, H, 1, 1}),
             xx = nn::arange(0, W).to(type).reshape({1, 1, W, 1});
  const auto rgb = nn::from_vector(std::vector<double>(colour.begin(), colour.end()))
                       .to(type)
                       .reshape({1, 1, 1, 3});
  std::vector<nn::Tensor> out;
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    const auto &b = boxes[i];
    SquareDetectionBoxes(b, 1.);
    if (b.dim() != 2)
      throw std::invalid_argument("drawing boxes must have rank two");
    auto image = nn::slice_axis(images, 0, std::int64_t(i), std::int64_t(i) + 1) *
                 nn::ones({1, 1, 1, 3}, type);
    for (std::int64_t j = 0; j < b.shape()[0]; ++j) {
      const auto box =
          nn::slice_axis(b, 0, j, j + 1)
              .to(type); // Tensor coordinate accessor; borrows this box for the current iteration.
      const auto c = [&](int index) {
        return nn::slice_axis(box, 1, index, index + 1).reshape({1, 1, 1, 1});
      };
      const auto ymin = c(0), xmin = c(1), ymax = c(2) - 1., xmax = c(3) - 1.;
      const auto horizontal = nn::logical_and(
          nn::logical_and(nn::greater_equal(xx, xmin - .51), nn::less_equal(xx, xmax + .51)),
          nn::logical_or(nn::less_equal(nn::abs(yy - ymin), .51),
                         nn::less_equal(nn::abs(yy - ymax), .51)));
      const auto vertical = nn::logical_and(
          nn::logical_and(nn::greater_equal(yy, ymin - .51), nn::less_equal(yy, ymax + .51)),
          nn::logical_or(nn::less_equal(nn::abs(xx - xmin), .51),
                         nn::less_equal(nn::abs(xx - xmax), .51)));
      image = nn::where(nn::logical_or(horizontal, vertical), rgb, image);
    }
    out.push_back(image);
  }
  return nn::concat(out, 0);
}
/// @brief Sample foreground jitter crops and background windows for an eleven-way patch classifier.
///
/// For each object, creates three centred squares with side multiplied by1+.3u
/// and centre jitter of±.1 side. Then accepts up to three backgrounds per object
/// from side12..40 windows with maximum target IoU at most.2, trying at most200
/// windows per scene. Small canvases may put some window edges outside the image;
/// zero padding applies. This matches source ranges but tensor RNG draws differ
/// from legacy std::uniform_real_distribution. No learned state, augmentation
/// during fitting or optimizer. Empty selections/objects return zero patches.
///
/// @param scenes Valid normalized tensor scene images and aligned floating boxes/Int64 digit labels.
/// @param rows Unique or repeated valid scene indices; only these scenes supply patches.
/// @param seed Seed for tensor uniform jitter/window draws, default3.
/// @return ImageDataset with detached [P,28,28,1] crops in [0,1] and Int64 digit/background labels; background is10.
/// @throws std::invalid_argument For invalid scene shapes/annotations or row indices.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::DetectionSceneConfig cfg;
/// cfg.scenes = 1;
/// const auto scenes = n::models::MakeTensorDetectionScenes(n::ones({1, 28, 28, 1}),
///                                                          n::zeros({1}, n::DType::Int64), cfg);
/// const auto patches = n::models::MakeDetectionPatches(scenes, {0}, 3);
/// @endcode
/// @see CropDetectionPatches(), SquareDetectionBoxes()
inline nn::ImageDataset MakeDetectionPatches(const TensorDetectionScenes &scenes,
                                             const std::vector<std::int64_t> &rows,
                                             std::uint64_t seed = 3) {
  const auto &images = scenes.images;
  if (images.dim() != 4 || images.shape()[0] < 1 || images.shape()[1] < 1 ||
      images.shape()[2] < 1 || images.shape()[3] != 1 ||
      std::int64_t(scenes.boxes.size()) != images.shape()[0] ||
      scenes.labels.size() != scenes.boxes.size() ||
      (images.dtype() != nn::DType::Float32 && images.dtype() != nn::DType::Float64))
    throw std::invalid_argument("invalid patch source scene shapes");
  nn::NoGradGuard guard;
  nn::Generator rng(seed);
  std::vector<nn::Tensor> crops, labels;
  std::vector<std::int64_t> source;
  const auto type = images.dtype();
  for (auto row : rows) {
    if (row < 0 || row >= images.shape()[0])
      throw std::invalid_argument("patch scene row outside source");
    const auto &boxes = scenes.boxes[row];
    const auto &classes = scenes.labels[row];
    if (boxes.dim() != 2 || boxes.shape()[1] != 4 || classes.dtype() != nn::DType::Int64 ||
        classes.shape() != nn::Shape{boxes.shape()[0]} ||
        !nn::all(nn::logical_and(nn::greater_equal(classes, 0.), nn::less(classes, 10.)))
             .item<bool>())
      throw std::invalid_argument("invalid patch scene annotations");
    SquareDetectionBoxes(boxes, 1.);
    for (std::int64_t object = 0; object < boxes.shape()[0]; ++object)
      for (int copy = 0; copy < 3; ++copy) {
        const auto c =
            nn::boxes_to_center(nn::slice_axis(boxes, 0, object, object + 1).to(type))
                .reshape({4});
        const auto u = nn::uniform({3}, 0., 1., rng, type);
        // Tensor coordinate accessor; borrows its immutable argument for one slice.
        const auto at = [&](const nn::Tensor &t, int k) {
          return nn::slice_axis(t, 0, k, k + 1);
        };
        const auto side = nn::maximum(at(c, 2), at(c, 3)) * (1. + .3 * at(u, 0));
        const auto cy = at(c, 0) + (at(u, 1) - .5) * .2 * side,
                   cx = at(c, 1) + (at(u, 2) - .5) * .2 * side;
        crops.push_back(
            nn::concat({cy - side * .5, cx - side * .5, cy + side * .5, cx + side * .5}, 0)
                .reshape({1, 4}));
        labels.push_back(nn::slice_axis(classes, 0, object, object + 1));
        source.push_back(row);
      }
    for (std::int64_t accepted = 0, attempt = 0;
         accepted < 3 * boxes.shape()[0] && attempt < 200; ++attempt) {
      const auto u = nn::uniform(
          {3}, 0., 1., rng,
          type); // Tensor uniform-draw accessor; borrows this attempt's draw tensor.
      const auto at = [&](int k) {
        return nn::slice_axis(u, 0, k, k + 1);
      };
      const auto side = 12. + 28. * at(0), y = at(1) * (double(images.shape()[1]) - side),
                 x = at(2) * (double(images.shape()[2]) - side);
      const auto box = nn::concat({y, x, y + side, x + side}, 0).reshape({1, 4});
      if (boxes.shape()[0] &&
          nn::any(nn::greater(nn::box_iou(box, boxes.to(type)), .2)).item<bool>())
        continue;
      crops.push_back(box);
      labels.push_back(nn::full({1}, 10., nn::DType::Int64));
      source.push_back(row);
      ++accepted;
    }
  }
  if (crops.empty())
    return {nn::zeros({0, 28, 28, 1}, type), nn::zeros({0}, nn::DType::Int64)};
  return {CropDetectionPatches(images, nn::from_vector(source), nn::concat(crops, 0), 28),
          nn::concat(labels, 0)};
}
/// @brief Enumerate fully contained square sliding windows as tensor pixel boxes.
///
/// Each side contributes (floor((height-side)/stride)+1)*(floor((width-side)/stride)+1)
/// windows. No anchor clipping, model computation, random state or fitting. Integer
/// metadata chooses geometry; all coordinate construction uses tensor ops.
///
/// @param height Positive image height.
/// @param width Positive image width.
/// @param sides Nonempty positive square sides, each fitting both image dimensions; default16/22/28/34.
/// @param stride Positive integer pixel displacement, default4.
/// @return Float32 [A,4] half-open boxes in side, row, column order, beginning at pixel edge zero.
/// @throws std::invalid_argument For invalid dimensions/sides/stride or more than2^24 coordinate pixels.
/// @par Example
/// @code
/// const auto windows = dm::nn::models::SlidingDetectionWindows(64, 64); // [454,4]
/// @endcode
/// @see CropDetectionPatches(), ComponentDetectionProposals()
inline nn::Tensor SlidingDetectionWindows(int height, int width,
                                          const std::vector<int> &sides = {16, 22, 28, 34},
                                          int stride = 4) {
  if (height < 1 || width < 1 || height > (1 << 24) || width > (1 << 24) || stride < 1 ||
      sides.empty())
    throw std::invalid_argument("invalid sliding window geometry");
  std::vector<nn::Tensor> boxes;
  for (int side : sides) {
    if (side < 1 || side > std::min(height, width))
      throw std::invalid_argument("invalid sliding window side");
    const auto H = (height - side) / stride + 1, W = (width - side) / stride + 1;
    const auto yy = nn::arange(0, H).reshape({H, 1}) * double(stride) * nn::ones({H, W}),
               xx = nn::arange(0, W).reshape({1, W}) * double(stride) * nn::ones({H, W});
    boxes.push_back(
        nn::stack({yy, xx, yy + side, xx + side}, 2).reshape({std::int64_t(H) * W, 4}));
  }
  return nn::concat(boxes, 0);
}
} // namespace datamunge::nn::models
