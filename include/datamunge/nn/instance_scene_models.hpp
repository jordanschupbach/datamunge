#pragma once
/// @file instance_scene_models.hpp
/// @brief Tensor instance support scenes,binary mask overlap and pixel-centre mask pasting.
#include "datamunge/nn/detection_inference.hpp"
#include "datamunge/nn/detection_scene_models.hpp"
#include <utility>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Extend detection pixels/boxes/labels with aligned full-canvas binary instance supports.
///
/// Base TensorDetectionScenes APIs accept this derived data by const reference.
/// Each accepted rescaled digit retains its own thresholded support,including
/// overlap with other objects;support is not trimmed by later compositing.
/// No learned state,hidden optimizer or ownership of source MNIST pixels.
///
/// @throws std::invalid_argument For invalid shapes,dtypes,classes or scene geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::DetectionSceneConfig c;
/// c.scenes = 2;
/// const auto data = n::models::MakeTensorInstanceScenes(n::ones({1, 28, 28, 1}),
///                                                       n::zeros({1}, n::DType::Int64), c);
/// @endcode
/// @par References
/// - He et al.(2017), Mask R-CNN, https://arxiv.org/abs/1703.06870; synthetic support/paste/AP conventions from examples/nn/common/instance_segmentation.hpp. MNIST source LeCun et al.(1998), https://doi.org/10.1109/5.726791.
/// @see MakeTensorInstanceScenes()
struct TensorInstanceScenes : TensorDetectionScenes {
  std::vector<nn::Tensor>
      masks; ///< N Bool[Ki,H,W] tensors,aligned with each scene's boxes/labels,empty Ki allowed.
};
/// @brief Generate canonical MNIST scenes while retaining every accepted instance support.
///
/// Shares one canonical sampler with MakeTensorDetectionScenes,so the same
/// inputs/settings give identical pixels,boxes and labels regardless of support
/// retention. Supports mark resampled bytes strictly>32. Tensor interpolation,
/// quantization,overlap tests and compositing preserve existing conventions.
/// No numerical host image loops,fitting,source mutation or additional RNG draws.
/// Retaining supports costs O(total_accepted_objects*H*W) Boolean storage.
///
/// @param digits Float32/64[N,28,28,1] in[0,1],N positive.
/// @param labels Aligned Int64[N] digit classes in[0,10).
/// @param cfg Valid detection canvas/object/overlap/discrete RNG settings.
/// @return Detached scene pixels/boxes/labels plus Bool[Ki,H,W] supports.
/// @throws std::invalid_argument For invalid shapes,dtypes,classes or scene geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::DetectionSceneConfig c;
/// c.scenes = 2;
/// const auto data = n::models::MakeTensorInstanceScenes(n::ones({1, 28, 28, 1}),
///                                                       n::zeros({1}, n::DType::Int64), c);
/// @endcode
/// @par References
/// - He et al.(2017), Mask R-CNN, https://arxiv.org/abs/1703.06870; synthetic support/paste/AP conventions from examples/nn/common/instance_segmentation.hpp. MNIST source LeCun et al.(1998), https://doi.org/10.1109/5.726791.
/// @see TensorInstanceScenes, MakeTensorDetectionScenes()
inline TensorInstanceScenes MakeTensorInstanceScenes(const nn::Tensor &digits,
                                                     const nn::Tensor &labels,
                                                     const DetectionSceneConfig &cfg = {}) {
  TensorInstanceScenes out;
  static_cast<TensorDetectionScenes &>(out) =
      detection_scene_detail::Generate(digits, labels, cfg, &out.masks);
  return out;
}
/// @brief Compute detached binary-mask intersection over union using tensor logical counts.
///
/// Every nonzero support is encoded as true before calling. This metric has
/// no gradients or fitted state,and does not mutate inputs.
///
/// @param a Bool tensor of any shape,including empty masks.
/// @param b Same-shape Bool tensor.
/// @return Float64 scalar IoU,zero when union is empty.
/// @throws std::invalid_argument For invalid shapes,dtypes,classes or scene geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto a = n::greater(n::arange(0, 6).reshape({2, 3}), 2.);
/// const auto overlap = n::models::InstanceMaskIoU(a, a);
/// @endcode
/// @par References
/// - He et al.(2017), Mask R-CNN, https://arxiv.org/abs/1703.06870; synthetic support/paste/AP conventions from examples/nn/common/instance_segmentation.hpp. MNIST source LeCun et al.(1998), https://doi.org/10.1109/5.726791.
/// @see PasteInstanceMask()
inline nn::Tensor InstanceMaskIoU(const nn::Tensor &a, const nn::Tensor &b) {
  if (a.shape() != b.shape() || a.dtype() != nn::DType::Bool || b.dtype() != nn::DType::Bool)
    throw std::invalid_argument("binary mask geometry/dtypes differ");
  nn::NoGradGuard guard;
  const auto intersection = nn::count_nonzero(nn::logical_and(a, b)).to(nn::DType::Float64),
             united = nn::count_nonzero(nn::logical_or(a, b)).to(nn::DType::Float64);
  return intersection / nn::maximum(united, nn::ones({}, nn::DType::Float64));
}
/// @brief Paste one class probability map into a pixel box using border-clamped bilinear sampling.
///
/// A canvas pixel contributes only if its centre lies inside the half-open
/// box. It samples ((centre-box_start)/extent*S-.5),clamped to[0,S-1],
/// then thresholds bilinear probability>=.5. Boxes may extend outside the
/// canvas;outside-box centres are false. No optimizer,gradients or mutation.
///
/// @param probabilities Float32/64[S,S,C] finite probabilities in[0,1],S/C positive.
/// @param label In-range class channel.
/// @param box Floating[4] finite ordered pixel edges,ymin,xmin,ymax,xmax;converted to probability dtype.
/// @param height Positive output canvas height.
/// @param width Positive output canvas width.
/// @return Detached Bool[height,width] support after inclusive probability threshold.5.
/// @throws std::invalid_argument For invalid shapes,dtypes,classes or scene geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto mask = n::models::PasteInstanceMask(n::full({2, 2, 1}, .75), 0,
///                                                n::tensor({1., 1., 5., 5.}), 6, 6);
/// @endcode
/// @par References
/// - He et al.(2017), Mask R-CNN, https://arxiv.org/abs/1703.06870; synthetic support/paste/AP conventions from examples/nn/common/instance_segmentation.hpp. MNIST source LeCun et al.(1998), https://doi.org/10.1109/5.726791.
/// @see InstanceMaskIoU()
inline nn::Tensor PasteInstanceMask(const nn::Tensor &probabilities, int label,
                                    const nn::Tensor &box, int height, int width) {
  if (probabilities.dim() != 3 || probabilities.shape()[0] < 1 ||
      probabilities.shape()[1] != probabilities.shape()[0] || probabilities.shape()[2] < 1 ||
      label < 0 || label >= probabilities.shape()[2] || box.shape() != nn::Shape{4} ||
      (probabilities.dtype() != nn::DType::Float32 &&
       probabilities.dtype() != nn::DType::Float64) ||
      (box.dtype() != nn::DType::Float32 && box.dtype() != nn::DType::Float64) || height < 1 ||
      width < 1 ||
      !nn::all(nn::logical_and(nn::greater_equal(probabilities, 0.),
                               nn::less_equal(probabilities, 1.)))
           .item<bool>() ||
      !nn::all(nn::less(nn::abs(box), std::numeric_limits<double>::infinity())).item<bool>())
    throw std::invalid_argument("invalid pasted mask inputs");
  nn::NoGradGuard guard;
  const auto type = probabilities.dtype();
  const auto b = nn::unstack(box.to(type), 0);
  if (!(nn::greater(b[2], b[0]).item<bool>() && nn::greater(b[3], b[1]).item<bool>()))
    throw std::invalid_argument("pasted mask box must have positive area");
  const auto S = probabilities.shape()[0];
  const auto y = nn::arange(0, double(height)).to(type).reshape({height, 1}) + .5,
             x = nn::arange(0, double(width)).to(type).reshape({1, width}) + .5,
             inside = nn::logical_and(
                 nn::logical_and(nn::greater_equal(y, b[0]), nn::less(y, b[2])),
                 nn::logical_and(nn::greater_equal(x, b[1]), nn::less(x, b[3]))),
             sy = nn::clip((y - b[0]) / (b[2] - b[0]) * double(S) - .5, 0., double(S - 1)) *
                  nn::ones({1, width}, type),
             sx = nn::clip((x - b[1]) / (b[3] - b[1]) * double(S) - .5, 0., double(S - 1)) *
                  nn::ones({height, 1}, type);
  const auto selected =
                 nn::slice_axis(probabilities, 2, label, label + 1).reshape({1, S, S, 1}),
             values = nn::sample_pixels(
                          selected, nn::stack({sx, sy}, 2).reshape({1, height, width, 2}),
                          nn::GridSampleMode::Bilinear, nn::GridSamplePadding::Border)
                          .reshape({height, width});
  return nn::logical_and(inside, nn::greater_equal(values, .5));
}
/// @brief Own scored object metadata and aligned binary full-canvas predicted masks.
///
/// Masks are detached Bool supports. Image IDs index the accompanying scene
/// dataset. Boxes are retained for presentation;mask AP uses masks/scores/
/// labels/image IDs,not box overlap.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorInstanceScenes data;
/// data.images = n::zeros({1, 2, 2, 1});
/// data.boxes = {n::tensor({0., 0., 2., 2.}, {1, 4})};
/// data.labels = {n::zeros({1}, n::DType::Int64)};
/// data.masks = {n::ones({1, 2, 2}, n::DType::Bool)};
/// n::models::TensorMaskDetections predictions{{n::zeros({1, 4}), n::ones({1}),
///                                              n::zeros({1}, n::DType::Int64),
///                                              n::zeros({1}, n::DType::Int64)},
///                                             n::ones({1, 2, 2}, n::DType::Bool)};
/// @endcode
/// @par References
/// - Instance support/all-point mask-AP conventions from examples/nn/common/instance_segmentation.hpp; He et al.(2017), Mask R-CNN, https://arxiv.org/abs/1703.06870. This is integral all-point class-mean mask AP,not the complete official COCO protocol.
/// @see InstanceMaskAveragePrecision()
struct TensorMaskDetections {
  TensorDetections
      objects;      ///< Boxes[M,4],floating scores[M],Int64 labels[M] and scene IDs[M].
  nn::Tensor masks; ///< Bool[M,H,W] support for every scored object.
};
/// @brief Evaluate class-mean integral all-point AP of scored binary instance supports.
///
/// Predictions sort by descending confidence with stable original-order ties.
/// Each selects its highest-IoU same-class/same-image truth,first on ties,even
/// if that truth was already used;duplicates become false positives rather than
/// rematching another target. True/false-positive cumsums and the suffix precision
/// envelope integrate all recall changes. Absent target classes are excluded.
/// Mask intersections use exact Float64 products/counts of Boolean indicators.
/// This is the source small-scene integral metric,not official COCO101-point/
/// crowd/area/maxDets evaluation. No fitting,randomness or input mutation.
///
/// @param predictions Aligned finite floating scores,Int64 class/image IDs,Bool[M,H,W] masks;boxes[M,4] are ignored geometrically.
/// @param scenes Aligned grayscale scene metadata and Bool[Ki,H,W] instance supports,including empty objects.
/// @param classes Positive class count;all truth/prediction labels in[0,classes).
/// @param threshold Finite mask IoU threshold in[0,1],default.5;matching is inclusive.
/// @return Detached Float64 scalar mean AP across classes with targets,or0 if none.
/// @throws std::invalid_argument For invalid shapes,dtypes,labels/IDs,nonfinite scores or threshold.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorInstanceScenes data;
/// data.images = n::zeros({1, 2, 2, 1});
/// data.boxes = {n::tensor({0., 0., 2., 2.}, {1, 4})};
/// data.labels = {n::zeros({1}, n::DType::Int64)};
/// data.masks = {n::ones({1, 2, 2}, n::DType::Bool)};
/// n::models::TensorMaskDetections predictions{{n::zeros({1, 4}), n::ones({1}),
///                                              n::zeros({1}, n::DType::Int64),
///                                              n::zeros({1}, n::DType::Int64)},
///                                             n::ones({1, 2, 2}, n::DType::Bool)};
/// const auto ap = n::models::InstanceMaskAveragePrecision(predictions, data, 1);
/// @endcode
/// @par References
/// - Instance support/all-point mask-AP conventions from examples/nn/common/instance_segmentation.hpp; He et al.(2017), Mask R-CNN, https://arxiv.org/abs/1703.06870. This is integral all-point class-mean mask AP,not the complete official COCO protocol.
/// @see InstanceMaskIoU(), TensorMaskDetections
inline nn::Tensor InstanceMaskAveragePrecision(const TensorMaskDetections &predictions,
                                               const TensorInstanceScenes &scenes, int classes,
                                               double threshold = .5) {
  const auto &objects = predictions.objects;
  if (classes < 1 || !std::isfinite(threshold) || threshold < 0 || threshold > 1 ||
      scenes.images.dim() != 4 || scenes.images.shape()[0] < 0 ||
      scenes.images.shape()[1] < 1 || scenes.images.shape()[2] < 1 ||
      scenes.images.shape()[3] != 1 ||
      scenes.boxes.size() != std::size_t(scenes.images.shape()[0]) ||
      scenes.labels.size() != scenes.boxes.size() ||
      scenes.masks.size() != scenes.boxes.size())
    throw std::invalid_argument("invalid instance AP scenes/settings");
  const auto N = scenes.images.shape()[0], H = scenes.images.shape()[1],
             W = scenes.images.shape()[2], M = objects.scores.numel();
  if (objects.scores.shape() != nn::Shape{M} ||
      (objects.scores.dtype() != nn::DType::Float32 &&
       objects.scores.dtype() != nn::DType::Float64) ||
      objects.boxes.shape() != nn::Shape{M, 4} || objects.labels.shape() != nn::Shape{M} ||
      objects.images.shape() != nn::Shape{M} || objects.labels.dtype() != nn::DType::Int64 ||
      objects.images.dtype() != nn::DType::Int64 ||
      predictions.masks.shape() != nn::Shape{M, H, W} ||
      predictions.masks.dtype() != nn::DType::Bool ||
      !nn::all(nn::logical_and(nn::greater_equal(objects.labels, 0.),
                               nn::less(objects.labels, double(classes))))
           .item<bool>() ||
      !nn::all(nn::logical_and(nn::greater_equal(objects.images, 0.),
                               nn::less(objects.images, double(N))))
           .item<bool>() ||
      !nn::all(nn::less(nn::abs(objects.scores), std::numeric_limits<double>::infinity()))
           .item<bool>())
    throw std::invalid_argument("invalid scored instance masks");
  nn::NoGradGuard guard;
  const auto type = nn::DType::Float64;
  std::vector<nn::Tensor> mask_parts, label_parts, image_parts;
  for (std::int64_t row = 0; row < N; ++row) {
    const auto K = scenes.boxes[row].shape().empty() ? -1 : scenes.boxes[row].shape()[0];
    if (K < 0 || scenes.boxes[row].shape() != nn::Shape{K, 4} ||
        scenes.labels[row].shape() != nn::Shape{K} ||
        scenes.labels[row].dtype() != nn::DType::Int64 ||
        scenes.masks[row].shape() != nn::Shape{K, H, W} ||
        scenes.masks[row].dtype() != nn::DType::Bool ||
        !nn::all(nn::logical_and(nn::greater_equal(scenes.labels[row], 0.),
                                 nn::less(scenes.labels[row], double(classes))))
             .item<bool>())
      throw std::invalid_argument("invalid instance truth alignment");
    mask_parts.push_back(scenes.masks[row]);
    label_parts.push_back(scenes.labels[row]);
    image_parts.push_back(nn::full({K}, double(row), nn::DType::Int64));
  }
  if (N == 0)
    return nn::zeros({}, type);
  const auto truth = nn::concat(mask_parts, 0), labels = nn::concat(label_parts, 0),
             images = nn::concat(image_parts, 0);
  const auto T = labels.numel();
  if (!T || M == 0)
    return nn::zeros({}, type);
  const auto pred_flat = predictions.masks.reshape({M, H * W}).to(type),
             truth_flat = truth.reshape({T, H * W}).to(type),
             intersection = nn::matmul(pred_flat, nn::transpose(truth_flat)),
             united = nn::sum(pred_flat, {1}).reshape({M, 1}) +
                      nn::sum(truth_flat, {1}).reshape({1, T}) - intersection,
             overlaps = intersection / nn::maximum(united, nn::ones({}, type));
  auto total = nn::zeros({}, type);
  int present = 0;
  for (int c = 0; c < classes; ++c) {
    const auto class_truth = nn::equal(labels, double(c)),
               positives = nn::count_nonzero(class_truth).to(type);
    if (positives.item() == 0)
      continue;
    ++present;
    const auto membership = nn::equal(objects.labels, double(c));
    const auto candidates =
                   nn::slice_axis(nn::argsort(membership.to(nn::DType::Int64), true), 0, 0,
                                  nn::count_nonzero(membership).item<std::int64_t>()),
               sorted = nn::gather(candidates,
                                   nn::argsort(nn::gather(objects.scores, candidates), true));
    if (sorted.numel() == 0)
      continue;
    auto used = nn::zeros({T}, nn::DType::Bool);
    std::vector<nn::Tensor> tp;
    for (auto k : sorted.to_vector<std::int64_t>()) {
      const auto eligible = nn::logical_and(
                     class_truth,
                     nn::equal(images,
                               nn::slice_axis(objects.images, 0, k, k + 1).reshape({}))),
                 iou = nn::where(eligible, nn::slice_axis(overlaps, 0, k, k + 1).reshape({T}),
                                 nn::full({}, -1., type)),
                 best = nn::argmax(iou),
                 hit = nn::logical_and(nn::greater_equal(nn::max(iou), threshold),
                                       nn::logical_not(nn::gather(used, best)));
      used = nn::logical_or(
          used, nn::logical_and(nn::equal(nn::arange(0, double(T), 1, nn::DType::Int64), best),
                                hit));
      tp.push_back(hit.to(type));
    }
    const auto indicators = nn::stack(tp, 0), cum_tp = nn::cumsum(indicators, 0),
               precision = cum_tp / nn::arange(1, double(tp.size() + 1)).to(type),
               recall = cum_tp / positives;
    std::vector<nn::Tensor> reversed_envelope;
    auto best_precision = nn::zeros({}, type);
    for (std::int64_t i = precision.numel(); i > 0; --i) {
      best_precision =
          nn::maximum(best_precision, nn::slice_axis(precision, 0, i - 1, i).reshape({}));
      reversed_envelope.push_back(best_precision);
    }
    const auto envelope = nn::reverse(nn::stack(reversed_envelope, 0), {0}),
               previous = nn::concat(
                   {nn::zeros({1}, type), nn::slice_axis(recall, 0, 0, recall.numel() - 1)},
                   0);
    total = total + nn::sum((recall - previous) * envelope);
  }
  return total / double(std::max(present, 1));
}
/// @brief Render image,individual truth supports and optional predicted supports as RGB triptychs.
///
/// Palette entry c is(80+(37c)%176,80+(83c+90)%176,80+(113c+30)%176)/255.
/// Truth objects paint in annotation order. Predictions paint ascending score
/// with stable input ties,so greater scores overlay earlier masks. All numerical
/// painting/order uses tensors;host loops only discrete object IDs. No model,
/// fitting,mutation or retained gradients.
///
/// @param scenes Valid floating grayscale instance scenes,ten digit classes0..9.
/// @param predictions Optional aligned Bool[D,H,W] masks with scored dataset image/class IDs;nullptr(default) omits prediction panel.
/// @param count Positive maximum number of first scene rows,default4.
/// @return Floating [min(count,N),H,(predictions?3:2)*W,3] RGB in[0,1].
/// @throws std::invalid_argument For invalid input/settings,unaligned scenes or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::DetectionSceneConfig c;
/// c.scenes = 1;
/// const auto data = n::models::MakeTensorInstanceScenes(n::ones({1, 28, 28, 1}),
///                                                       n::zeros({1}, n::DType::Int64), c);
/// const auto preview = n::models::InstanceMaskPreview(data);
/// @endcode
/// @par References
/// - He et al.(2017), Mask R-CNN, https://arxiv.org/abs/1703.06870; the compact64px single-level network is a source teaching adaptation,not full ResNet/FPN COCO Mask R-CNN.
/// @see TensorMaskDetections, MakeTensorInstanceScenes()
inline nn::Tensor InstanceMaskPreview(const TensorInstanceScenes &scenes,
                                      const TensorMaskDetections *predictions = nullptr,
                                      int count = 4) {
  if (count < 1 || scenes.images.dim() != 4 || scenes.images.shape()[0] < 1 ||
      scenes.images.shape()[1] < 1 || scenes.images.shape()[2] < 1 ||
      scenes.images.shape()[3] != 1 ||
      scenes.masks.size() != std::size_t(scenes.images.shape()[0]) ||
      scenes.labels.size() != scenes.masks.size())
    throw std::invalid_argument("invalid instance preview scenes");
  nn::NoGradGuard guard;
  const auto N = scenes.images.shape()[0], H = scenes.images.shape()[1],
             W = scenes.images.shape()[2], take = std::min<std::int64_t>(count, N);
  const auto type = scenes.images.dtype();
  std::vector<double> colors;
  for (int c = 0; c < 10; ++c) {
    colors.push_back((80 + (37 * c) % 176) / 255.);
    colors.push_back((80 + (83 * c + 90) % 176) / 255.);
    colors.push_back((80 + (113 * c + 30) % 176) / 255.);
  }
  const auto palette = nn::from_vector(colors, {10, 3}).to(type);
  std::vector<nn::Tensor> result;
  nn::Tensor order;
  if (predictions) {
    const auto &d = predictions->objects;
    const auto D = d.scores.numel();
    if (d.scores.dim() != 1 || d.labels.shape() != nn::Shape{D} ||
        d.images.shape() != nn::Shape{D} || d.labels.dtype() != nn::DType::Int64 ||
        d.images.dtype() != nn::DType::Int64 ||
        predictions->masks.shape() != nn::Shape{D, H, W} ||
        predictions->masks.dtype() != nn::DType::Bool ||
        !nn::all(nn::logical_and(nn::greater_equal(d.labels, 0.), nn::less(d.labels, 10.)))
             .item<bool>() ||
        !nn::all(
             nn::logical_and(nn::greater_equal(d.images, 0.), nn::less(d.images, double(N))))
             .item<bool>() ||
        !nn::all(nn::less(nn::abs(d.scores), std::numeric_limits<double>::infinity()))
             .item<bool>())
      throw std::invalid_argument("invalid instance preview detections");
    order = nn::argsort(d.scores, false);
  }
  for (std::int64_t row = 0; row < take; ++row) {
    const auto K = scenes.labels[row].numel();
    if (scenes.labels[row].shape() != nn::Shape{K} ||
        scenes.labels[row].dtype() != nn::DType::Int64 ||
        scenes.masks[row].shape() != nn::Shape{K, H, W} ||
        scenes.masks[row].dtype() != nn::DType::Bool ||
        !nn::all(nn::logical_and(nn::greater_equal(scenes.labels[row], 0.),
                                 nn::less(scenes.labels[row], 10.)))
             .item<bool>())
      throw std::invalid_argument("invalid preview truth masks");
    const auto image =
        nn::slice_axis(scenes.images, 0, row, row + 1) * nn::ones({1, 1, 1, 3}, type);
    auto truth = nn::zeros({1, H, W, 3}, type), pred = nn::zeros({1, H, W, 3}, type);
    for (std::int64_t k = 0; k < K; ++k)
      truth = nn::where(nn::slice_axis(scenes.masks[row], 0, k, k + 1).reshape({1, H, W, 1}),
                        nn::gather(palette, nn::slice_axis(scenes.labels[row], 0, k, k + 1))
                            .reshape({1, 1, 1, 3}),
                        truth);
    if (predictions) {
      for (const auto k : order.to_vector<std::int64_t>())
        if (nn::slice_axis(predictions->objects.images, 0, k, k + 1).item<std::int64_t>() ==
            row)
          pred = nn::where(
              nn::slice_axis(predictions->masks, 0, k, k + 1).reshape({1, H, W, 1}),
              nn::gather(palette, nn::slice_axis(predictions->objects.labels, 0, k, k + 1))
                  .reshape({1, 1, 1, 3}),
              pred);
      result.push_back(nn::concat({image, truth, pred}, 2));
    } else
      result.push_back(nn::concat({image, truth}, 2));
  }
  return nn::concat(result, 0);
}
} // namespace datamunge::nn::models
