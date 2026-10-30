#pragma once
/// @file mask_rcnn_models.hpp
/// @brief Aligned region sampling,class-specific mask objectives and compact Mask R-CNN tensor components.
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/instance_scene_models.hpp"
#include "datamunge/nn/two_stage_detection_models.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Map pixel boxes to normalized coordinates for aligned regular2x2 samples per bin.
///
/// Near/far fractions are .25/bins and1-.25/bins. Transform each pixel
/// coordinate to (pixel/stride+.5)/(feature_extent+1). Together with2*bins
/// aligned-corner samples and average pooling,this subtracts .5 feature pixels.
/// No fitting or mutation;O(K) work. Boxes receive no gradients.
///
/// @param boxes Float32/64 [K,4] finite positive-area pixel boxes,possibly beyond the canvas.
/// @param height Feature rows,at least2.
/// @param width Feature columns,at least2.
/// @param stride Positive number of image pixels per feature cell.
/// @param bins Positive number of averaged output bins per axis.
/// @return Detached floating [K,4] crop coordinates for a symmetric one-pixel-padded feature map.
/// @throws std::invalid_argument For invalid tensor shapes,dtypes,settings,indices,targets or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto b = n::tensor({0., 0., 20., 20.}, {1, 4});
/// const auto crops = n::models::MaskAlignmentBoxes(b, 5, 5, 4, 3);
/// @endcode
/// @par References
/// - He, Gkioxari, Dollár and Girshick (2017), Mask R-CNN, https://arxiv.org/abs/1703.06870. Aligned regular2x2 sampling and nearest support targets follow the source teaching example,not an official pretrained implementation.
/// @see MaskRoiAlign()
inline nn::Tensor MaskAlignmentBoxes(const nn::Tensor &boxes, int height, int width,
                                     int stride, int bins) {
  if (boxes.dim() != 2 || boxes.shape()[1] != 4 || height < 2 || width < 2 || stride < 1 ||
      bins < 1 || bins > std::numeric_limits<int>::max() / 2 ||
      (boxes.dtype() != nn::DType::Float32 && boxes.dtype() != nn::DType::Float64) ||
      !nn::all(nn::less(nn::abs(boxes), std::numeric_limits<double>::infinity())).item<bool>())
    throw std::invalid_argument("invalid aligned mask crop geometry");
  nn::NoGradGuard guard;
  const auto b = nn::unstack(boxes, 1);
  const auto dy = b[2] - b[0], dx = b[3] - b[1];
  if (!nn::all(nn::logical_and(nn::greater(dy, 0.), nn::greater(dx, 0.))).item<bool>())
    throw std::invalid_argument("mask boxes must have positive area");
  const double near = .25 / bins, far = 1 - near;
  return nn::stack({((b[0] + dy * near) / double(stride) + .5) / double(height + 1),
                    ((b[1] + dx * near) / double(stride) + .5) / double(width + 1),
                    ((b[0] + dy * far) / double(stride) + .5) / double(height + 1),
                    ((b[1] + dx * far) / double(stride) + .5) / double(width + 1)},
                   1);
}
/// @brief Sample unquantized aligned image regions with four bilinear samples averaged per bin.
///
/// Symmetric one-cell mirror padding replicates feature borders. Fixed regular
/// 2x2 samples per bin match the source teaching convention. No coordinate
/// quantization,optimizer or feature mutation. WorkO(K*bins²*C).
///
/// @param feature Positive Float32/64 NHWC features with H/W at least2.
/// @param boxes Same-dtype [K,4] finite positive-area pixel boxes;coordinates detached.
/// @param indices Int64 [K] valid feature-batch positions.
/// @param stride Positive pixel spacing between feature cells.
/// @param bins Positive output bins per axis.
/// @return [K,bins,bins,C] differentiable features,including graph-connected empty K=0.
/// @throws std::invalid_argument For invalid tensor shapes,dtypes,settings,indices,targets or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// auto f = n::ones({1, 5, 7, 2});
/// f.requires_grad_(true);
/// const auto y = n::models::MaskRoiAlign(f, n::tensor({0., 0., 20., 28.}, {1, 4}),
///                                        n::zeros({1}, n::DType::Int64), 4, 3);
/// const auto g = n::grad(n::sum(y), {f});
/// @endcode
/// @par References
/// - He, Gkioxari, Dollár and Girshick (2017), Mask R-CNN, https://arxiv.org/abs/1703.06870. Aligned regular2x2 sampling and nearest support targets follow the source teaching example,not an official pretrained implementation.
/// @see MaskAlignmentBoxes(), MaskRcnnHead()
inline nn::Tensor MaskRoiAlign(const nn::Tensor &feature, const nn::Tensor &boxes,
                               const nn::Tensor &indices, int stride, int bins) {
  if (feature.dim() != 4 || feature.shape()[0] < 1 || feature.shape()[1] < 2 ||
      feature.shape()[2] < 2 || feature.shape()[3] < 1 || boxes.dtype() != feature.dtype() ||
      indices.dtype() != nn::DType::Int64 || boxes.dim() != 2 ||
      indices.shape() != nn::Shape{boxes.shape()[0]} ||
      !nn::all(nn::logical_and(nn::greater_equal(indices, 0.),
                               nn::less(indices, double(feature.shape()[0]))))
           .item<bool>())
    throw std::invalid_argument("invalid RoIAlign feature/indices");
  const auto crops = MaskAlignmentBoxes(boxes, int(feature.shape()[1]),
                                        int(feature.shape()[2]), stride, bins);
  if (boxes.shape()[0] == 0)
    return nn::zeros({0, bins, bins, feature.shape()[3]}, feature.dtype()) +
           nn::sum(feature) * 0.;
  return nn::avg_pool2d(
      nn::crop_and_resize(nn::mirror_pad(feature, 1, 1, nn::PadMode::Symmetric), crops,
                          indices, {2 * bins, 2 * bins}),
      2, 2, nn::Padding::Valid);
}
/// @brief Build four3x3 mask convolutions,a learned2x up-convolution and per-class linear logits.
///
/// Four biased He/ReLU convolutions precede biased He/ReLU2x2 stride2 VALID
/// transpose convolution and biased Glorot1x1 classifier. No BN,dropout or hidden
/// fitting. Full256/80 head on256-channel inputs has2643280 scalars;compact
/// 32/10 head on64-channel inputs has50666. Warm before optimizer creation.
///
/// @param p Same-dtype learned parameter store,creates/reuses mask/* leaves.
/// @param pooled Positive floating square NHWC aligned features.
/// @param width Positive hidden channels,default256.
/// @param classes Positive number of foreground mask channels,default80.
/// @return [K,2*S,2*S,classes] logits for pooled spatial side S.
/// @throws std::invalid_argument For invalid tensor shapes,dtypes,settings,indices,targets or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto masks = n::models::MaskRcnnHead(p, n::ones({1, 14, 14, 64}), 32, 10);
/// @endcode
/// @par References
/// - He, Gkioxari, Dollár and Girshick (2017), Mask R-CNN, https://arxiv.org/abs/1703.06870. Aligned regular2x2 sampling and nearest support targets follow the source teaching example,not an official pretrained implementation.
/// @see SelectedInstanceMaskLoss(), MaskRoiAlign()
inline nn::Tensor MaskRcnnHead(nn::ParameterStore &p, const nn::Tensor &pooled,
                               int width = 256, int classes = 80) {
  if (width < 1 || classes < 1 || pooled.dim() != 4 || pooled.shape()[0] < 1 ||
      pooled.shape()[1] < 1 || pooled.shape()[2] != pooled.shape()[1] ||
      pooled.shape()[3] < 1 || pooled.dtype() != p.dtype())
    throw std::invalid_argument("invalid mask head widths");
  nn::ConvLayerOptions relu;
  relu.activation = nn::Activation::Relu;
  relu.init = nn::init::he_uniform();
  nn::Tensor h = pooled;
  for (int i = 0; i < 4; ++i)
    h = nn::conv2d(p, "mask/conv" + std::to_string(i + 1), h, width, 3, relu);
  auto up = relu;
  up.stride = 2;
  up.padding = nn::Padding::Valid;
  h = nn::conv2d_transpose(p, "mask/up", h, width, 2, up);
  return nn::conv2d(p, "mask/classifier", h, classes, 1);
}

/// @brief Average stable sigmoid BCE over only each positive region matched foreground class.
///
/// No background channel and no cross-class softmax. Every unselected class
/// has exactly zero derivative. Target/label gradients are disabled. All
/// selected pixels share1/(K*S*S) normalization;no fitting or mutation.
///
/// @param logits Floating [K,S,S,C] class-specific mask scores,S/C positive.
/// @param labels Int64 [K] foreground class IDs in[0,C).
/// @param targets Same-dtype [K,S,S] finite target probabilities in[0,1],detached.
/// @return Scalar mean selected softplus(z)-z*target;empty K returns a graph-connected zero.
/// @throws std::invalid_argument For invalid tensor shapes,dtypes,settings,indices,targets or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// auto z = n::zeros({2, 2, 2, 3});
/// z.requires_grad_(true);
/// const auto loss = n::models::SelectedInstanceMaskLoss(
///     z, n::from_vector(std::vector<std::int64_t>{0, 2}), n::ones({2, 2, 2}));
/// const auto g = n::grad(loss, {z});
/// @endcode
/// @par References
/// - He, Gkioxari, Dollár and Girshick (2017), Mask R-CNN, https://arxiv.org/abs/1703.06870. Aligned regular2x2 sampling and nearest support targets follow the source teaching example,not an official pretrained implementation.
/// @see MaskRcnnHead(), PositiveInstanceMaskTargets()
inline nn::Tensor SelectedInstanceMaskLoss(const nn::Tensor &logits, const nn::Tensor &labels,
                                           const nn::Tensor &targets) {
  if (logits.dim() != 4 || logits.shape()[1] < 1 || logits.shape()[2] != logits.shape()[1] ||
      logits.shape()[3] < 1 ||
      (logits.dtype() != nn::DType::Float32 && logits.dtype() != nn::DType::Float64) ||
      labels.shape() != nn::Shape{logits.shape()[0]} || labels.dtype() != nn::DType::Int64 ||
      targets.shape() != nn::Shape{logits.shape()[0], logits.shape()[1], logits.shape()[2]} ||
      targets.dtype() != logits.dtype() ||
      !nn::all(nn::logical_and(nn::greater_equal(labels, 0.),
                               nn::less(labels, double(logits.shape()[3]))))
           .item<bool>() ||
      !nn::all(nn::logical_and(nn::greater_equal(targets, 0.), nn::less_equal(targets, 1.)))
           .item<bool>())
    throw std::invalid_argument("invalid selected mask BCE inputs");
  if (!logits.shape()[0])
    return nn::sum(logits) * 0.;
  const auto selected =
      nn::sum(logits * nn::one_hot(labels, logits.shape()[3])
                           .to(logits.dtype())
                           .reshape({logits.shape()[0], 1, 1, logits.shape()[3]}),
              {3});
  return nn::mean(nn::softplus(selected) - selected * targets.detach());
}
/// @brief Own selected positive pixel boxes,feature-batch indices,classes and cropped support targets.
///
/// Detached tensors have common leading K. Target side defaults28;no learned
/// parameters or independent store.
///
/// @throws std::invalid_argument For invalid tensor shapes,dtypes,settings,indices,targets or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::DetectionSceneConfig c;
/// c.scenes = 1;
/// c.max_objects = 1;
/// const auto data = n::models::MakeTensorInstanceScenes(n::ones({1, 28, 28, 1}),
///                                                       n::zeros({1}, n::DType::Int64), c);
/// n::Generator rng(7);
/// const auto batch = n::models::SampleDetectionRois(data, {0}, {data.boxes[0]}, rng);
/// const auto target = n::models::PositiveInstanceMaskTargets(data, {0}, batch);
/// @endcode
/// @par References
/// - He, Gkioxari, Dollár and Girshick (2017), Mask R-CNN, https://arxiv.org/abs/1703.06870. Aligned regular2x2 sampling and nearest support targets follow the source teaching example,not an official pretrained implementation.
/// @see PositiveInstanceMaskTargets()
struct TensorInstanceMaskTargets {
  nn::Tensor boxes;   ///< Floating [K,4] positive sampled pixel boxes.
  nn::Tensor indices; ///< Int64 [K] feature-minibatch positions.
  nn::Tensor labels;  ///< Int64 [K] foreground class IDs.
  nn::Tensor targets; ///< Floating [K,S,S] nearest-sampled binary object supports.
};
/// @brief Crop the highest-IoU truth support for each sampled positive region using tensor nearest indexing.
///
/// First truth wins tied box IoU;its class must equal the sampled class. Each
/// axis samples floor(start+(j+.5)*extent/side),clamped to scene bounds. Truth
/// masks retain individual object support even under overlap. Host loops select
/// discrete regions/truth only;IoU and pixel gathering use tensors. Empty
/// positives return aligned empty tensors. No fitting or source mutation.
///
/// @param scenes Valid tensor instance scenes with aligned boxes,classes and Bool supports.
/// @param rows Source scene IDs in feature-minibatch order.
/// @param batch Consistent sampled regions,positive mask nonzero;indices refer to rows.
/// @param side Positive mask target side,default28.
/// @return Detached selected boxes/indices/labels plus same-image-dtype[K,side,side] supports.
/// @throws std::invalid_argument For invalid tensor shapes,dtypes,settings,indices,targets or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::DetectionSceneConfig c;
/// c.scenes = 1;
/// c.max_objects = 1;
/// const auto data = n::models::MakeTensorInstanceScenes(n::ones({1, 28, 28, 1}),
///                                                       n::zeros({1}, n::DType::Int64), c);
/// n::Generator rng(7);
/// const auto batch = n::models::SampleDetectionRois(data, {0}, {data.boxes[0]}, rng);
/// const auto target = n::models::PositiveInstanceMaskTargets(data, {0}, batch);
/// @endcode
/// @par References
/// - He, Gkioxari, Dollár and Girshick (2017), Mask R-CNN, https://arxiv.org/abs/1703.06870. Aligned regular2x2 sampling and nearest support targets follow the source teaching example,not an official pretrained implementation.
/// @see SelectedInstanceMaskLoss(), SampleDetectionRois()
inline TensorInstanceMaskTargets
PositiveInstanceMaskTargets(const TensorInstanceScenes &scenes,
                            const std::vector<std::int64_t> &rows, const TensorRoiBatch &batch,
                            int side = 28) {
  if (side < 1 || scenes.images.dim() != 4 || scenes.images.shape()[1] < 1 ||
      scenes.images.shape()[2] < 1 || scenes.masks.size() != scenes.boxes.size() ||
      batch.mask.dim() != 1)
    throw std::invalid_argument("invalid instance targets");
  const auto K = batch.mask.numel();
  const auto type = scenes.images.dtype();
  if (batch.pixels.shape() != nn::Shape{K, 4} || batch.pixels.dtype() != type ||
      batch.indices.shape() != nn::Shape{K} || batch.indices.dtype() != nn::DType::Int64 ||
      batch.labels.shape() != nn::Shape{K} || batch.labels.dtype() != nn::DType::Int64 ||
      !nn::all(nn::logical_and(nn::greater_equal(batch.indices, 0.),
                               nn::less(batch.indices, double(rows.size()))))
           .item<bool>())
    throw std::invalid_argument("inconsistent positive region tensors");
  for (const auto row : rows)
    if (row < 0 || row >= scenes.images.shape()[0] ||
        scenes.boxes.size() != std::size_t(scenes.images.shape()[0]) ||
        scenes.labels.size() != scenes.boxes.size() || scenes.boxes[row].dim() != 2 ||
        scenes.boxes[row].shape()[1] != 4 ||
        scenes.labels[row].shape() != nn::Shape{scenes.boxes[row].shape()[0]} ||
        scenes.masks[row].shape() != nn::Shape{scenes.boxes[row].shape()[0],
                                               scenes.images.shape()[1],
                                               scenes.images.shape()[2]} ||
        scenes.masks[row].dtype() != nn::DType::Bool)
      throw std::invalid_argument("invalid instance target scene alignment");
  nn::NoGradGuard guard;
  const auto selected = two_stage_detection_detail::Indices(nn::not_equal(batch.mask, 0.));
  TensorInstanceMaskTargets out{
      nn::gather(batch.pixels, selected), nn::gather(batch.indices, selected),
      nn::gather(batch.labels, selected), nn::zeros({selected.numel(), side, side}, type)};
  std::vector<nn::Tensor> masks;
  const auto index = out.indices.to_vector<std::int64_t>();
  for (std::int64_t k = 0; k < selected.numel(); ++k) {
    const auto row = rows[index[k]], G = scenes.boxes[row].shape()[0];
    if (!G)
      throw std::invalid_argument("positive region without truth");
    const auto box = nn::slice_axis(out.boxes, 0, k, k + 1);
    const auto best = nn::argmax(nn::box_iou(box, scenes.boxes[row]).reshape({G}), 0);
    if (nn::gather(scenes.labels[row], best).item<std::int64_t>() !=
        nn::slice_axis(out.labels, 0, k, k + 1).item<std::int64_t>())
      throw std::invalid_argument("positive mask class mismatch");
    const auto b = nn::unstack(box.reshape({4}), 0);
    if (!(nn::greater(b[2], b[0]).item<bool>() && nn::greater(b[3], b[1]).item<bool>()))
      throw std::invalid_argument("positive mask box area");
    const auto unit = (nn::arange(0, double(side)).to(type) + .5) / double(side);
    const auto y = nn::clip(nn::floor(b[0] + unit * (b[2] - b[0])), 0.,
                            double(scenes.images.shape()[1] - 1))
                       .to(nn::DType::Int64)
                       .reshape({side, 1}),
               x = nn::clip(nn::floor(b[1] + unit * (b[3] - b[1])), 0.,
                            double(scenes.images.shape()[2] - 1))
                       .to(nn::DType::Int64)
                       .reshape({1, side});
    const auto pixels =
        y * nn::full({}, double(scenes.images.shape()[2]), nn::DType::Int64) + x;
    const auto support = nn::gather(scenes.masks[row], best)
                             .reshape({scenes.images.shape()[1] * scenes.images.shape()[2]});
    masks.push_back(
        nn::gather(support, pixels.reshape({side * side})).to(type).reshape({1, side, side}));
  }
  if (!masks.empty())
    out.targets = nn::concat(masks, 0);
  return out;
}
/// @brief Own shared stride4 features and RPN outputs for the compact64px Mask R-CNN.
///
/// The learned store remains caller-owned. These tensors retain its graph;
/// use the same features for both aligned heads before optimizer mutation.
///
/// @throws std::invalid_argument For invalid input/settings,unaligned scenes or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto net = n::models::CompactMaskRcnnForward(p, n::zeros({1, 64, 64, 1}));
/// @endcode
/// @par References
/// - He et al.(2017), Mask R-CNN, https://arxiv.org/abs/1703.06870; the compact64px single-level network is a source teaching adaptation,not full ResNet/FPN COCO Mask R-CNN.
/// @see CompactMaskRcnnForward()
struct CompactMaskRcnnOutputs {
  nn::Tensor features; ///< Floating [N,16,16,64] shared convolutional features.
  Rpn rpn;             ///< Shared hidden128 RPN with9 anchors per cell,2304 total.
};
/// @brief Build the complete compact stride4 backbone and one-level proposal network.
///
/// No BN,dropout or fitting. The box head is PooledDetectionRoiHead on7-bin
/// MaskRoiAlign;the mask head is width32/10 classes on14-bin MaskRoiAlign.
/// Warm both before constructing the optimizer. Source anchors are16x16,
/// stride4,sides12/20/30,ratios.5/1/2.
///
/// @param p Floating learned store,creates/reuses backbone/* and rpn/* leaves.
/// @param images Same-dtype [N,64,64,1] input,usually rescaled from[0,1] to[-1,1].
/// @return Graph-connected shared features and RPN outputs;heads are built separately.
/// @throws std::invalid_argument For invalid input/settings,unaligned scenes or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto net = n::models::CompactMaskRcnnForward(p, n::zeros({1, 64, 64, 1}));
/// @endcode
/// @par References
/// - He et al.(2017), Mask R-CNN, https://arxiv.org/abs/1703.06870; the compact64px single-level network is a source teaching adaptation,not full ResNet/FPN COCO Mask R-CNN.
/// @see MaskRoiAlign(), MaskRcnnHead(), PooledDetectionRoiHead()
inline CompactMaskRcnnOutputs CompactMaskRcnnForward(nn::ParameterStore &p,
                                                     const nn::Tensor &images) {
  if (images.dim() != 4 || images.shape()[0] < 1 || images.shape()[1] != 64 ||
      images.shape()[2] != 64 || images.shape()[3] != 1 || images.dtype() != p.dtype())
    throw std::invalid_argument("compact Mask R-CNN needs64px gray images");
  const auto features = StrideFourDetectionBackbone(p, images);
  return {features, BuildRpn(p, {{features, 64, 16, 9}})};
}
/// @brief Decode compact Mask R-CNN proposals,boxes and selected-class pasted instance masks.
///
/// One shared forward per selected image. Decode2304 anchors,keep top300,
/// clip/filter minimum side4,NMS.7/100 proposals per image. Aligned7-bin box
/// head uses11 classes/hidden256;class NMS.3 and strict score>.05 cap100
/// detections per image. Aligned14-bin masks use four32-channel convolutions
/// and10 class logits. Paste selected sigmoid masks at pixel centres with
/// bilinear border sampling and threshold>=.5. No gradients or hidden optimizer.
/// Memory scales with all rows;call in bounded chunks for larger datasets.
///
/// @param p Learned store for all compact branches;no fitting or mutation of existing leaves.
/// @param scenes Valid64x64 floating grayscale scenes in[0,1].
/// @param rows Nonempty selected scene IDs,each unique and valid.
/// @return Detached dataset-ID box detections with Bool[D,64,64] pasted masks.
/// @throws std::invalid_argument For invalid input/settings,unaligned scenes or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::DetectionSceneConfig c;
/// c.scenes = 1;
/// const auto data = n::models::MakeTensorInstanceScenes(n::ones({1, 28, 28, 1}),
///                                                       n::zeros({1}, n::DType::Int64), c);
/// const auto detection = n::models::DetectCompactMaskRcnn(p, data, {0});
/// @endcode
/// @par References
/// - He et al.(2017), Mask R-CNN, https://arxiv.org/abs/1703.06870; the compact64px single-level network is a source teaching adaptation,not full ResNet/FPN COCO Mask R-CNN.
/// @see PasteInstanceMask(), DecodeDetectionRoiOutputs()
inline TensorMaskDetections DetectCompactMaskRcnn(nn::ParameterStore &p,
                                                  const TensorDetectionScenes &scenes,
                                                  const std::vector<std::int64_t> &rows) {
  if (rows.empty() || scenes.images.dim() != 4 || scenes.images.shape()[1] != 64 ||
      scenes.images.shape()[2] != 64 || scenes.images.shape()[3] != 1 ||
      scenes.images.dtype() != p.dtype())
    throw std::invalid_argument("invalid compact inference scenes/rows");
  std::vector<std::int64_t> seen;
  for (const auto row : rows) {
    if (row < 0 || row >= scenes.images.shape()[0] ||
        std::find(seen.begin(), seen.end(), row) != seen.end())
      throw std::invalid_argument("inference rows must be distinct and valid");
    seen.push_back(row);
  }
  nn::NoGradGuard guard;
  const auto ids = nn::from_vector(rows);
  const auto net = CompactMaskRcnnForward(p, 2. * nn::gather(scenes.images, ids) - 1.);
  const auto proposals = ComputeDetectionProposals(
      net.rpn, nn::anchor_grid(16, 16, 4, {12, 20, 30}, {.5, 1, 2}).to(p.dtype()), 64, 64);
  const auto batch = PackDetectionRois(proposals);
  if (!batch.indices.numel())
    return {{nn::zeros({0, 4}, p.dtype()), nn::zeros({0}, p.dtype()),
             nn::zeros({0}, nn::DType::Int64), nn::zeros({0}, nn::DType::Int64)},
            nn::zeros({0, 64, 64}, nn::DType::Bool)};
  const auto head =
      PooledDetectionRoiHead(p, MaskRoiAlign(net.features, batch.pixels, batch.indices, 4, 7));
  const auto objects =
      DecodeDetectionRoiOutputs(head, batch.pixels, nn::gather(ids, batch.indices));
  if (!objects.scores.numel())
    return {objects, nn::zeros({0, 64, 64}, nn::DType::Bool)};
  std::vector<std::int64_t> local;
  for (const auto image : objects.images.to_vector<std::int64_t>())
    local.push_back(std::find(rows.begin(), rows.end(), image) - rows.begin());
  const auto probabilities = nn::sigmoid(MaskRcnnHead(
      p, MaskRoiAlign(net.features, objects.boxes, nn::from_vector(local), 4, 14), 32, 10));
  std::vector<nn::Tensor> masks;
  for (std::int64_t k = 0; k < objects.scores.numel(); ++k)
    masks.push_back(PasteInstanceMask(
                        nn::slice_axis(probabilities, 0, k, k + 1).reshape({28, 28, 10}),
                        int(nn::slice_axis(objects.labels, 0, k, k + 1).item<std::int64_t>()),
                        nn::slice_axis(objects.boxes, 0, k, k + 1).reshape({4}), 64, 64)
                        .reshape({1, 64, 64}));
  return {objects, nn::concat(masks, 0)};
}
} // namespace datamunge::nn::models
