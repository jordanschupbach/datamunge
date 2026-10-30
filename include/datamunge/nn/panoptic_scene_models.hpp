#pragma once
/// @file panoptic_scene_models.hpp
/// @brief Tensor MNIST instance supports over textured stuff,aligned semantic/panoptic targets and preview colours.
#include "datamunge/nn/panoptic_models.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Own textured instance scenes and paired semantic and panoptic ground truth.
///
/// No fitting or independent random state after construction. Original object
/// supports and boxes/classes remain aligned through background synthesis.
///
/// @throws std::invalid_argument For invalid source tensor shapes/dtypes,count,seed or annotation geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto data = n::models::MakeTensorPanopticScenes(n::ones({1, 28, 28, 1}),
///                                                       n::zeros({1}, n::DType::Int64), 2, 11);
/// @endcode
/// @par References
/// - Kirillov et al.(2019),Panoptic Feature Pyramid Networks, https://arxiv.org/abs/1901.02446. MNIST objects over two deterministic textured materials are teaching data from examples/nn/common/panoptic_scenes.hpp,not a panoptic benchmark.
/// @see MakeTensorPanopticScenes()
struct TensorPanopticScenes {
  TensorInstanceScenes
      instances; ///< Floating grayscale scenes with retained per-object supports.
  nn::Tensor
      semantic; ///< Int64[N,64,64],0 all things,1/2 material classes mapped to categories10/11.
  std::vector<TensorPanopticMap>
      panoptic; ///< N exact segment-ID grids/catalogs for thing instances and stuff.
};
/// @brief Generate digit instances over two deterministic textured background materials with aligned targets.
///
/// Use shared MakeTensorInstanceScenes with nonoverlapping boxes. Each image
/// has a near-midpoint material split,byte bases40/100 plus deterministic
/// 0..10 texture. Restore original digit intensity only at its retained support
/// pixels;semantic all-things label0 and panoptic object IDs use those supports,
/// never a threshold of the textured image. Empty stuff metadata is removed.
/// All texture,painting,areas and targets use tensors;host loops handle
/// scene/object metadata only. No fitting or input mutation.
///
/// @param digits Float32/64[N,28,28,1] MNIST source pixels in[0,1].
/// @param labels Int64[N] digit classes0..9.
/// @param count Positive number of64px scenes.
/// @param seed Nonnegative deterministic source-scene/texture seed.
/// @return Detached textured instance scenes,semantic labels and complete panoptic maps.
/// @throws std::invalid_argument For invalid source tensor shapes/dtypes,count,seed or annotation geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto data = n::models::MakeTensorPanopticScenes(n::ones({1, 28, 28, 1}),
///                                                       n::zeros({1}, n::DType::Int64), 2, 11);
/// @endcode
/// @par References
/// - Kirillov et al.(2019),Panoptic Feature Pyramid Networks, https://arxiv.org/abs/1901.02446. MNIST objects over two deterministic textured materials are teaching data from examples/nn/common/panoptic_scenes.hpp,not a panoptic benchmark.
/// @see MakeTensorInstanceScenes(), PanopticAreas()
inline TensorPanopticScenes MakeTensorPanopticScenes(const nn::Tensor &digits,
                                                     const nn::Tensor &labels, int count,
                                                     int seed) {
  if (count < 1 || seed < 0)
    throw std::invalid_argument("positive panoptic count/nonnegative seed required");
  DetectionSceneConfig cfg;
  cfg.scenes = count;
  cfg.seed = std::uint64_t(seed);
  cfg.max_overlap = 0;
  TensorPanopticScenes out;
  out.instances = MakeTensorInstanceScenes(digits, labels, cfg);
  nn::NoGradGuard guard;
  const auto type = digits.dtype();
  const auto y = nn::arange(0, 64, 1., nn::DType::Int64).reshape({64, 1}),
             x = nn::arange(0, 64, 1., nn::DType::Int64).reshape({1, 64});
  std::vector<nn::Tensor> images, semantics;
  for (int row = 0; row < count; ++row) {
    const auto objects = out.instances.labels[row].numel();
    const auto offset = nn::full({}, double(seed) + 7. * row, nn::DType::Float64);
    const auto split = 32. + offset - nn::floor(offset / 9.) * 9. - 4.;
    auto semantic = nn::where(nn::less(y.to(nn::DType::Float64), split),
                              nn::ones({64, 64}, nn::DType::Int64),
                              nn::full({64, 64}, 2., nn::DType::Int64));
    const auto q = 13. * y.to(nn::DType::Float64) + 7. * x.to(nn::DType::Float64) + 3. * row;
    const auto texture = q - nn::floor(q / 11.) * 11.;
    auto image =
        (nn::where(nn::equal(semantic, 1.), nn::full({64, 64}, 40., nn::DType::Float64),
                   nn::full({64, 64}, 100., nn::DType::Float64)) +
         texture)
            .to(type) /
        255.;
    TensorPanopticMap map{
        semantic + nn::full({}, double(objects), nn::DType::Int64),
        {{int(objects + 1), {10, false, false}}, {int(objects + 2), {11, false, false}}}};
    const auto original =
        nn::slice_axis(out.instances.images, 0, row, row + 1).reshape({64, 64});
    for (std::int64_t k = 0; k < objects; ++k) {
      const auto support =
          nn::slice_axis(out.instances.masks[row], 0, k, k + 1).reshape({64, 64});
      image = nn::where(support, original, image);
      semantic = nn::where(support, nn::zeros({}, nn::DType::Int64), semantic);
      map.ids = nn::where(support, nn::full({}, double(k + 1), nn::DType::Int64), map.ids);
      map.segments[int(k + 1)] = {
          int(nn::slice_axis(out.instances.labels[row], 0, k, k + 1).item<std::int64_t>()),
          true, false};
    }
    for (auto it = map.segments.begin(); it != map.segments.end();)
      if (!nn::count_nonzero(nn::equal(map.ids, double(it->first))).item<std::int64_t>())
        it = map.segments.erase(it);
      else
        ++it;
    PanopticAreas(map);
    out.panoptic.push_back(map);
    images.push_back(image.reshape({1, 64, 64, 1}));
    semantics.push_back(semantic.reshape({1, 64, 64}));
  }
  out.instances.images = nn::concat(images, 0);
  out.semantic = nn::concat(semantics, 0);
  return out;
}
/// @brief Colour category or segment labels using the panoptic teaching palette.
///
/// Digit palette matches InstanceMaskPreview. Stuff10 uses(80,80,120)/255
/// and stuff11(170,160,100)/255. Presentation only,no fitting/mutation.
///
/// @param labels Int64 tensor of category IDs0..11;negative values denote void.
/// @return Float32[labels.shape(),3] RGB in[0,1],void black.
/// @throws std::invalid_argument For invalid source tensor shapes/dtypes,count,seed or annotation geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto rgb = n::models::ColorPanopticCategories(
///     n::from_vector(std::vector<std::int64_t>{-1, 0, 10, 11}));
/// @endcode
/// @par References
/// - Kirillov et al.(2019),Panoptic Feature Pyramid Networks, https://arxiv.org/abs/1901.02446. MNIST objects over two deterministic textured materials are teaching data from examples/nn/common/panoptic_scenes.hpp,not a panoptic benchmark.
/// @see ColorPanopticMap()
inline nn::Tensor ColorPanopticCategories(const nn::Tensor &labels) {
  if (labels.dtype() != nn::DType::Int64 || !nn::all(nn::less(labels, 12.)).item<bool>())
    throw std::invalid_argument("unknown panoptic palette category");
  nn::NoGradGuard guard;
  std::vector<float> colors;
  for (int c = 0; c < 12; ++c) {
    colors.push_back(float(c == 10 ? 80 : c == 11 ? 170 : 80 + (37 * c) % 176) / 255);
    colors.push_back(float(c == 10 ? 80 : c == 11 ? 160 : 80 + (83 * c + 90) % 176) / 255);
    colors.push_back(float(c == 10 ? 120 : c == 11 ? 100 : 80 + (113 * c + 30) % 176) / 255);
  }
  auto shape = labels.shape();
  shape.push_back(3);
  const auto rgb =
      nn::gather(
          nn::from_vector(colors, {12, 3}),
          nn::maximum(labels, nn::zeros({}, nn::DType::Int64)).reshape({labels.numel()}))
          .reshape(shape);
  auto mask_shape = labels.shape();
  mask_shape.push_back(1);
  return rgb * nn::greater_equal(labels, 0.).to(nn::DType::Float32).reshape(mask_shape);
}
/// @brief Render segment categories while keeping void pixels black.
///
/// Colours encode category rather than unique instance ID;metadata/ID grids
/// retain instance separation. No optimization or source mutation.
///
/// @param map Valid Int64 panoptic grid/catalog with categories0..11.
/// @return Float32 RGB[map.ids.shape(),3] with teaching category colours.
/// @throws std::invalid_argument For invalid source tensor shapes/dtypes,count,seed or annotation geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto data = n::models::MakeTensorPanopticScenes(n::ones({1, 28, 28, 1}),
///                                                       n::zeros({1}, n::DType::Int64), 2, 11);
/// const auto rgb = n::models::ColorPanopticMap(data.panoptic[0]);
/// @endcode
/// @par References
/// - Kirillov et al.(2019),Panoptic Feature Pyramid Networks, https://arxiv.org/abs/1901.02446. MNIST objects over two deterministic textured materials are teaching data from examples/nn/common/panoptic_scenes.hpp,not a panoptic benchmark.
/// @see ColorPanopticCategories(), PanopticAreas()
inline nn::Tensor ColorPanopticMap(const TensorPanopticMap &map) {
  PanopticAreas(map);
  nn::NoGradGuard guard;
  auto labels = nn::full(map.ids.shape(), -1., nn::DType::Int64);
  for (const auto &[id, segment] : map.segments)
    labels = nn::where(nn::equal(map.ids, double(id)),
                       nn::full({}, double(segment.label), nn::DType::Int64), labels);
  return ColorPanopticCategories(labels);
}
} // namespace datamunge::nn::models
