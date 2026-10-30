#pragma once
/// @file johnson_losses.hpp
/// @brief Tensor feed-forward style objectives,image preparation,paired generator checkpoint and actual image previews.
#include "datamunge/nn/johnson_models.hpp"
#include "datamunge/nn/style_losses.hpp"
#include "datamunge/nn/training.hpp"
#include <filesystem>
#include <set>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Compute one channel-and-spatial normalized Gram per image.
///
/// Differs from Gatys by an additional channel divisor. Batch images never mix.
///
/// @param x Floating nonempty NHWC features.
/// @return Graph-preserving[N,C,C] F^T F/(H*W*C).
/// @throws std::invalid_argument For invalid geometry,row metadata or incompatible checkpoint inventory.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto g = m::JohnsonGram(n::ones({2, 3, 4, 5}));
/// @endcode
/// @par References
/// - Johnson,Alahi and Fei-Fei(2016),Perceptual Losses for Real-Time Style Transfer and Super-Resolution, https://arxiv.org/abs/1603.08155. Source choices:channel/spatial-normalized Grams,Frobenius style distance,quadratic joint-interior TV,BN and fixed initial loss denominators;teaching CIFAR32->64 contents,not paper-quality or real-time claims.
/// @see JohnsonTransform(), FrozenVgg16, JohnsonGram(), JohnsonTV()
inline nn::Tensor JohnsonGram(const nn::Tensor &x) {
  return StyleGram(x) / double(x.size(3));
}

/// @brief Measure mean squared frozen content-feature reconstruction.
///
/// Uses relu2_2 in the report. Caller caches targets without a graph;no hidden fit.
///
/// @param features Generated feature tensor.
/// @param target Compatible fixed content target,same shape recommended.
/// @return Scalar mean((features-target)^2).
/// @throws std::invalid_argument For invalid geometry,row metadata or incompatible checkpoint inventory.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto l = m::JohnsonContent(n::ones({1, 2, 3, 4}), n::zeros({1, 2, 3, 4}));
/// @endcode
/// @par References
/// - Johnson,Alahi and Fei-Fei(2016),Perceptual Losses for Real-Time Style Transfer and Super-Resolution, https://arxiv.org/abs/1603.08155. Source choices:channel/spatial-normalized Grams,Frobenius style distance,quadratic joint-interior TV,BN and fixed initial loss denominators;teaching CIFAR32->64 contents,not paper-quality or real-time claims.
/// @see JohnsonTransform(), FrozenVgg16, JohnsonGram(), JohnsonTV()
inline nn::Tensor JohnsonContent(const nn::Tensor &features, const nn::Tensor &target) {
  return nn::mean(nn::squared_difference(features, target));
}

/// @brief Sum four batch-mean Frobenius normalized-Gram distances.
///
/// No implicit channel mean beyond JohnsonGram normalization. Style layer
/// coefficients and initial balancing are caller policy. Retains image derivatives.
///
/// @param features Generated relu1_2,2_2,3_3,4_3 activations.
/// @param targets Four fixed normalized Gram targets.
/// @return Scalar sum over layers of mean_batch(sum_channels((G-A)^2)).
/// @throws std::invalid_argument For invalid geometry,row metadata or incompatible checkpoint inventory.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// std::array<n::Tensor, 4> f, t;
/// for (int i = 0; i < 4; ++i) {
///   f[i] = n::ones({2, 3, 4, 5});
///   t[i] = m::JohnsonGram(f[i]);
/// }
/// const auto l = m::JohnsonStyle(f, t);
/// @endcode
/// @par References
/// - Johnson,Alahi and Fei-Fei(2016),Perceptual Losses for Real-Time Style Transfer and Super-Resolution, https://arxiv.org/abs/1603.08155. Source choices:channel/spatial-normalized Grams,Frobenius style distance,quadratic joint-interior TV,BN and fixed initial loss denominators;teaching CIFAR32->64 contents,not paper-quality or real-time claims.
/// @see JohnsonTransform(), FrozenVgg16, JohnsonGram(), JohnsonTV()
inline nn::Tensor JohnsonStyle(const std::array<nn::Tensor, 4> &features,
                               const std::array<nn::Tensor, 4> &targets) {
  nn::Tensor total;
  for (std::size_t i = 0; i < 4; ++i) {
    const nn::Tensor term = nn::mean(
        nn::sum(nn::squared_difference(JohnsonGram(features[i]), targets[i]), {1, 2}));
    total = i ? total + term : term;
  }
  return total;
}

/// @brief Compute quadratic total variation over joint-interior neighbor pairs.
///
/// Both directions use the shared(H-1)x(W-1) interior. This is quadratic TV,
/// not L1 TV or a full-image denominator. No resizing or clipping.
///
/// @param image Floating[N,H,W,C] generated pixels,H/W>=2.
/// @return Scalar.5*mean((center-right)^2+(center-down)^2).
/// @throws std::invalid_argument For invalid geometry,row metadata or incompatible checkpoint inventory.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto l = m::JohnsonTV(n::ones({2, 3, 4, 3}));
/// @endcode
/// @par References
/// - Johnson,Alahi and Fei-Fei(2016),Perceptual Losses for Real-Time Style Transfer and Super-Resolution, https://arxiv.org/abs/1603.08155. Source choices:channel/spatial-normalized Grams,Frobenius style distance,quadratic joint-interior TV,BN and fixed initial loss denominators;teaching CIFAR32->64 contents,not paper-quality or real-time claims.
/// @see JohnsonTransform(), FrozenVgg16, JohnsonGram(), JohnsonTV()
inline nn::Tensor JohnsonTV(const nn::Tensor &image) {
  if (image.dim() != 4 || image.size(0) < 1 || image.size(1) < 2 || image.size(2) < 2 ||
      image.size(3) < 1)
    throw std::invalid_argument("positive TV interior required");
  const std::int64_t b = image.size(0), h = image.size(1) - 1, w = image.size(2) - 1,
                     c = image.size(3);
  const nn::Tensor center = nn::slice(image, {0, 0, 0, 0}, {b, h, w, c}),
                   right = nn::slice(image, {0, 0, 1, 0}, {b, h, w, c}),
                   down = nn::slice(image, {0, 1, 0, 0}, {b, h, w, c});
  return .5 * nn::mean(nn::squared_difference(center, right) +
                       nn::squared_difference(center, down));
}

/// @brief Gather rank4 content or cached features by original row metadata.
///
/// Host loops only serialize/validate row metadata;no pixel arithmetic.
///
/// @param input Rank4 tensor with leading batch axis.
/// @param rows Bounded source rows,duplicates/reordering/empty output allowed.
/// @return Contiguous gathered tensor,preserving graph and dtype.
/// @throws std::invalid_argument For invalid geometry,row metadata or incompatible checkpoint inventory.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto b = m::FastStyleBatch(n::ones({3, 2, 3, 4}), {2, 0});
/// @endcode
/// @par References
/// - Johnson,Alahi and Fei-Fei(2016),Perceptual Losses for Real-Time Style Transfer and Super-Resolution, https://arxiv.org/abs/1603.08155. Source choices:channel/spatial-normalized Grams,Frobenius style distance,quadratic joint-interior TV,BN and fixed initial loss denominators;teaching CIFAR32->64 contents,not paper-quality or real-time claims.
/// @see JohnsonTransform(), FrozenVgg16, JohnsonGram(), JohnsonTV()
inline nn::Tensor FastStyleBatch(const nn::Tensor &input,
                                 const std::vector<std::size_t> &rows) {
  if (input.dim() != 4)
    throw std::invalid_argument("rank4 cached style input required");
  std::vector<std::int64_t> ids;
  for (auto r : rows) {
    if (r >= static_cast<std::size_t>(input.size(0)))
      throw std::invalid_argument("style row bounds");
    ids.push_back(static_cast<std::int64_t>(r));
  }
  return nn::take_rows(input, ids).contiguous();
}
/// @brief Prepare observed CIFAR content images using tensor normalization and legacy resize.
///
/// Casts original bytes to Float32,divides by255,then legacy bilinear32->64.
/// No target-derived statistics,external resize,hidden training or image replacement.
///
/// @param images Original Int32[N,32,32,3] encoded RGB bytes.
/// @param rows Nonempty bounded original row metadata.
/// @return Float32[Nselected,64,64,3] unit RGB.
/// @throws std::invalid_argument For invalid geometry,row metadata or incompatible checkpoint inventory.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto x = m::FastStyleImages(n::zeros({2, 32, 32, 3}, n::DType::Int32), {1});
/// @endcode
/// @par References
/// - Johnson,Alahi and Fei-Fei(2016),Perceptual Losses for Real-Time Style Transfer and Super-Resolution, https://arxiv.org/abs/1603.08155. Source choices:channel/spatial-normalized Grams,Frobenius style distance,quadratic joint-interior TV,BN and fixed initial loss denominators;teaching CIFAR32->64 contents,not paper-quality or real-time claims.
/// @see JohnsonTransform(), FrozenVgg16, JohnsonGram(), JohnsonTV()
inline nn::Tensor FastStyleImages(const nn::Tensor &images,
                                  const std::vector<std::size_t> &rows) {
  if (images.dim() != 4 || images.shape() != nn::Shape{images.size(0), 32, 32, 3} ||
      images.dtype() != nn::DType::Int32 || rows.empty())
    throw std::invalid_argument("original Int32 CIFAR image rows required");
  const auto input = FastStyleBatch(images, rows).to(nn::DType::Float32) / 255.;
  if (!nn::all(nn::logical_and(nn::greater_equal(input, 0.), nn::less_equal(input, 1.)))
           .item<bool>())
    throw std::invalid_argument("CIFAR source byte range invalid");
  return nn::resize_bilinear(input, {64, 64}).contiguous();
}
/// @brief Save generator learned values and moving BN statistics together.
///
/// Archive prefixes transform/ and statistics/ preserve complete selection.
/// Frozen VGG,style/content targets and optimizer state are excluded. No mutation.
///
/// @param p Warm learned generator store.
/// @param state Warm separate moving BN store.
/// @param path Output named-tensor NPZ path.
/// @throws std::invalid_argument For invalid geometry,row metadata or incompatible checkpoint inventory.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(17), state(18);
/// p.get("w", {1}, n::init::ones());
/// state.get("mean", {1}, n::init::zeros());
/// const auto path =
///     (std::filesystem::temp_directory_path() / "dm-johnson-save-doc.npz").string();
/// m::SaveJohnson(p, state, path);
/// std::filesystem::remove(path);
/// @endcode
/// @par References
/// - Johnson,Alahi and Fei-Fei(2016),Perceptual Losses for Real-Time Style Transfer and Super-Resolution, https://arxiv.org/abs/1603.08155. Source choices:channel/spatial-normalized Grams,Frobenius style distance,quadratic joint-interior TV,BN and fixed initial loss denominators;teaching CIFAR32->64 contents,not paper-quality or real-time claims.
/// @see JohnsonTransform(), FrozenVgg16, JohnsonGram(), JohnsonTV()
inline void SaveJohnson(const nn::ParameterStore &p, const nn::ParameterStore &state,
                        const std::string &path) {
  nn::NamedTensors arrays;
  for (std::size_t i = 0; i < p.size(); ++i)
    arrays.push_back({"transform/" + p.names()[i], p.parameters()[i]});
  for (std::size_t i = 0; i < state.size(); ++i)
    arrays.push_back({"statistics/" + state.names()[i], state.parameters()[i]});
  nn::save_npz(path, arrays);
}
/// @brief Restore a complete validated generator/BN checkpoint into warm stores.
///
/// Validates complete counts,unique names,known prefixes,exact leaf names,shapes
/// and dtypes BEFORE any copy. Does not lazily add missing leaves or restore
/// optimizer state. Validation failure leaves both stores unchanged.
///
/// @param p Warm mutable learned store,leaf identities preserved.
/// @param state Warm mutable separate BN store.
/// @param path Readable paired named-tensor checkpoint.
/// @throws std::invalid_argument For invalid geometry,row metadata or incompatible checkpoint inventory.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(17), state(18);
/// p.get("w", {1}, n::init::ones());
/// state.get("mean", {1}, n::init::zeros());
/// const auto path =
///     (std::filesystem::temp_directory_path() / "dm-johnson-restore-doc.npz").string();
/// m::SaveJohnson(p, state, path);
/// m::RestoreJohnson(p, state, path);
/// std::filesystem::remove(path);
/// @endcode
/// @par References
/// - Johnson,Alahi and Fei-Fei(2016),Perceptual Losses for Real-Time Style Transfer and Super-Resolution, https://arxiv.org/abs/1603.08155. Source choices:channel/spatial-normalized Grams,Frobenius style distance,quadratic joint-interior TV,BN and fixed initial loss denominators;teaching CIFAR32->64 contents,not paper-quality or real-time claims.
/// @see JohnsonTransform(), FrozenVgg16, JohnsonGram(), JohnsonTV()
inline void RestoreJohnson(nn::ParameterStore &p, nn::ParameterStore &state,
                           const std::string &path) {
  const auto arrays = nn::load_npz(path);
  if (arrays.size() != p.size() + state.size())
    throw std::invalid_argument("complete warm Johnson checkpoint required");
  std::set<std::string> names;
  for (const auto &[name, value] : arrays) {
    if (!names.insert(name).second)
      throw std::invalid_argument("duplicate Johnson checkpoint name");
    const bool stats = name.rfind("statistics/", 0) == 0,
               model = name.rfind("transform/", 0) == 0;
    if (!stats && !model)
      throw std::invalid_argument("unexpected Johnson checkpoint prefix");
    const auto &store = stats ? state : p;
    const auto key = name.substr(stats ? 11 : 10);
    const auto found = std::find(store.names().begin(), store.names().end(), key);
    if (found == store.names().end())
      throw std::invalid_argument("unknown Johnson checkpoint leaf");
    const auto &current = store.at(key);
    if (value.shape() != current.shape() || value.dtype() != current.dtype())
      throw std::invalid_argument("Johnson checkpoint shape/dtype mismatch");
  }
  nn::NoGradGuard guard;
  for (const auto &[name, value] : arrays)
    if (name.rfind("statistics/", 0) == 0)
      state.set(name.substr(11), value);
    else
      p.set(name.substr(10), value);
}
/// @brief Build actual content|fixed painting|generated RGB preview tiles.
///
/// Tensor broadcasting,concatenation and Float64 half-up display quantization.
/// No generated image is substituted for original source data.
///
/// @param content Unit RGB[N,64,64,3] original contents.
/// @param style Unit RGB[1,64,64,3] fixed painting.
/// @param prediction Unit RGB matching content.
/// @return Float32[N,64,192,3] source half-up byte-quantized tiles.
/// @throws std::invalid_argument For invalid geometry,row metadata or incompatible checkpoint inventory.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto x = n::full({2, 64, 64, 3}, .5), s = n::full({1, 64, 64, 3}, .3);
/// const auto tiles = m::JohnsonPreviewTiles(x, s, x);
/// @endcode
/// @par References
/// - Johnson,Alahi and Fei-Fei(2016),Perceptual Losses for Real-Time Style Transfer and Super-Resolution, https://arxiv.org/abs/1603.08155. Source choices:channel/spatial-normalized Grams,Frobenius style distance,quadratic joint-interior TV,BN and fixed initial loss denominators;teaching CIFAR32->64 contents,not paper-quality or real-time claims.
/// @see JohnsonTransform(), FrozenVgg16, JohnsonGram(), JohnsonTV()
inline nn::Tensor JohnsonPreviewTiles(const nn::Tensor &content, const nn::Tensor &style,
                                      const nn::Tensor &prediction) {
  if (content.dim() != 4 || content.size(0) < 1 ||
      content.shape() != nn::Shape{content.size(0), 64, 64, 3} ||
      prediction.shape() != content.shape() || style.shape() != nn::Shape{1, 64, 64, 3})
    throw std::invalid_argument("aligned64x64 style preview required");
  const auto tiles = nn::concat({content, nn::ones_like(content) * style, prediction}, 2);
  return (nn::floor(tiles.to(nn::DType::Float64) * 255. + .5) / 255.).to(nn::DType::Float32);
}
} // namespace datamunge::nn::models
