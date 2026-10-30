#pragma once
/// @file style_losses.hpp
/// @brief Tensor Gatys feature objectives and reproducible JPEG/PNG image ingress/display.
#include "datamunge/image/jpeg.hpp"
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/image_io.hpp"
#include "datamunge/nn/perceptual_models.hpp"
#include <filesystem>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Compute per-image channel Gram matrices divided by spatial count.
///
/// No centering or channel normalization. Spatial permutations preserve Grams;
/// batch images never mix. Differentiable for arbitrary nonsymmetric upstreams.
///
/// @param features Nonempty floating NHWC activations.
/// @return Graph-preserving[N,C,C] per-image F^T F/(H*W).
/// @throws std::invalid_argument For incompatible shapes,nonpositive size or invalid projected pixels.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto g = m::StyleGram(n::arange(0, 24).reshape({2, 2, 3, 2}));
/// @endcode
/// @par References
/// - Gatys,Ecker and Bethge(2015),A Neural Algorithm of Artistic Style, https://arxiv.org/abs/1508.06576. M-normalized per-image Grams yield.25*mean((G-A)^2),equivalent to1/(4*N²*M²) for unnormalized Grams. Equal.2 style-layer weights and half-summed content follow this teaching source. Original images and attribution: https://www.tensorflow.org/tutorials/generative/style_transfer.
/// @see StyleGram(), GatysContentLoss(), GatysStyleLoss(), FrozenVgg19
inline nn::Tensor StyleGram(const nn::Tensor &features) {
  if (features.dim() != 4 || features.numel() < 1)
    throw std::invalid_argument("nonempty NHWC Gram features required");
  const nn::Tensor flat = nn::reshape(features, {features.size(0), -1, features.size(3)});
  return nn::matmul(flat, flat, true) / double(flat.size(1));
}

/// @brief Measure half-summed squared feature reconstruction.
///
/// Retains spatial arrangement. Caller caches detached targets and owns optimization.
///
/// @param features Floating generated features.
/// @param target Compatible fixed content features,same shape recommended.
/// @return Scalar.5*sum((features-target)^2).
/// @throws std::invalid_argument For incompatible shapes,nonpositive size or invalid projected pixels.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto l = m::GatysContentLoss(n::ones({1, 2, 3, 2}), n::zeros({1, 2, 3, 2}));
/// @endcode
/// @par References
/// - Gatys,Ecker and Bethge(2015),A Neural Algorithm of Artistic Style, https://arxiv.org/abs/1508.06576. M-normalized per-image Grams yield.25*mean((G-A)^2),equivalent to1/(4*N²*M²) for unnormalized Grams. Equal.2 style-layer weights and half-summed content follow this teaching source. Original images and attribution: https://www.tensorflow.org/tutorials/generative/style_transfer.
/// @see StyleGram(), GatysContentLoss(), GatysStyleLoss(), FrozenVgg19
inline nn::Tensor GatysContentLoss(const nn::Tensor &features, const nn::Tensor &target) {
  return .5 * nn::sum(nn::squared_difference(features, target));
}

/// @brief Measure the exact Gatys normalized squared-Gram layer objective.
///
/// Equivalent to original1/(4*C²*M²) coefficient for unnormalized Grams.
/// This is channel correlation matching,not pixel correspondence or centered covariance.
///
/// @param features Floating NHWC generated features.
/// @param target_gram Compatible fixed per-image M-normalized[N,C,C] style Grams.
/// @return Scalar.25*mean((StyleGram(features)-target_gram)^2).
/// @throws std::invalid_argument For incompatible shapes,nonpositive size or invalid projected pixels.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto x = n::ones({1, 2, 3, 2});
/// const auto l = m::GatysStyleLayer(x, m::StyleGram(x));
/// @endcode
/// @par References
/// - Gatys,Ecker and Bethge(2015),A Neural Algorithm of Artistic Style, https://arxiv.org/abs/1508.06576. M-normalized per-image Grams yield.25*mean((G-A)^2),equivalent to1/(4*N²*M²) for unnormalized Grams. Equal.2 style-layer weights and half-summed content follow this teaching source. Original images and attribution: https://www.tensorflow.org/tutorials/generative/style_transfer.
/// @see StyleGram(), GatysContentLoss(), GatysStyleLoss(), FrozenVgg19
inline nn::Tensor GatysStyleLayer(const nn::Tensor &features, const nn::Tensor &target_gram) {
  return .25 * nn::mean(nn::squared_difference(StyleGram(features), target_gram));
}

/// @brief Sum five equally weighted style-layer objectives.
///
/// Layers and fixed target construction are caller policy;no extractor,optimizer
/// or adaptive reweighting is hidden. All generated-feature gradients remain live.
///
/// @param features Generated conv1_1..conv5_1 floating activations.
/// @param targets Fixed M-normalized Gram matrix for each selected style layer.
/// @return Scalar.2*sum of five GatysStyleLayer values.
/// @throws std::invalid_argument For incompatible shapes,nonpositive size or invalid projected pixels.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// std::array<n::Tensor, 5> f, t;
/// for (int i = 0; i < 5; ++i) {
///   f[i] = n::ones({1, 2, 3, 2});
///   t[i] = m::StyleGram(f[i]);
/// }
/// const auto l = m::GatysStyleLoss(f, t);
/// @endcode
/// @par References
/// - Gatys,Ecker and Bethge(2015),A Neural Algorithm of Artistic Style, https://arxiv.org/abs/1508.06576. M-normalized per-image Grams yield.25*mean((G-A)^2),equivalent to1/(4*N²*M²) for unnormalized Grams. Equal.2 style-layer weights and half-summed content follow this teaching source. Original images and attribution: https://www.tensorflow.org/tutorials/generative/style_transfer.
/// @see StyleGram(), GatysContentLoss(), GatysStyleLoss(), FrozenVgg19
inline nn::Tensor GatysStyleLoss(const std::array<nn::Tensor, 5> &features,
                                 const std::array<nn::Tensor, 5> &targets) {
  nn::Tensor total;
  for (std::size_t i = 0; i < 5; ++i) {
    const nn::Tensor layer = GatysStyleLayer(features[i], targets[i]);
    total = i ? total + layer : layer;
  }
  return .2 * total;
}

/// @brief Decode a JPEG and prepare square unit RGB with tensor operations.
///
/// JPEG decoding is file ingress. Tensor center-square slicing uses integer
/// floor offsets,then legacy bilinear resize and/255. No half-pixel variant,
/// external image processing or fitting. Decoder may throw runtime I/O errors.
///
/// @param path Original readable JPEG file.
/// @param side Positive square output extent.
/// @return Contiguous Float32[1,side,side,3] encoded RGB/255.
/// @throws std::invalid_argument For incompatible shapes,nonpositive size or invalid projected pixels.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// try {
///   const auto x = m::LoadStyleImage("missing-style-doc-example.jpg", 64);
/// } catch (const std::exception &) {
/// }
/// @endcode
/// @par References
/// - Gatys,Ecker and Bethge(2015),A Neural Algorithm of Artistic Style, https://arxiv.org/abs/1508.06576. M-normalized per-image Grams yield.25*mean((G-A)^2),equivalent to1/(4*N²*M²) for unnormalized Grams. Equal.2 style-layer weights and half-summed content follow this teaching source. Original images and attribution: https://www.tensorflow.org/tutorials/generative/style_transfer.
/// @see StyleGram(), GatysContentLoss(), GatysStyleLoss(), FrozenVgg19
inline nn::Tensor LoadStyleImage(const std::filesystem::path &path, int side) {
  if (side < 1)
    throw std::invalid_argument("positive style image size required");
  datamunge::image::JpegDecodeOptions options;
  options.channels = 3;
  const nn::Tensor decoded =
      nn::from_image(datamunge::image::read_jpeg(path.string(), options)); // [H, W, 3] 0..255
  const std::int64_t h = decoded.size(0), w = decoded.size(1), square = std::min(h, w);
  const nn::Tensor crop =
      nn::slice(decoded, {(h - square) / 2, (w - square) / 2, 0}, {square, square, 3});
  const nn::Tensor resized =
      nn::resize_bilinear(nn::expand_dims(crop, 0), {side, side}) / 255.0;
  PerceptualRequire(resized.shape() == nn::Shape({1, side, side, 3}),
                    "style-transfer RGB image shape");
  return resized.contiguous();
}

/// @brief Write one projected unit RGB tensor with exact half-up byte display.
///
/// Tensor Float64 floor(255*x+.5) preserves source std::lround half-byte
/// quantization for nonnegative projected pixels. No silent numeric clipping.
///
/// @param rgb Floating[1,H,W,3] finite values within[0,1].
/// @param filename Output PNG path,parent must exist.
/// @throws std::invalid_argument For incompatible shapes,nonpositive size or invalid projected pixels.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto path = std::filesystem::temp_directory_path() / "dm-style-doc-example.png";
/// m::WriteStylePng(n::full({1, 2, 3, 3}, .5), path);
/// std::filesystem::remove(path);
/// @endcode
/// @par References
/// - Gatys,Ecker and Bethge(2015),A Neural Algorithm of Artistic Style, https://arxiv.org/abs/1508.06576. M-normalized per-image Grams yield.25*mean((G-A)^2),equivalent to1/(4*N²*M²) for unnormalized Grams. Equal.2 style-layer weights and half-summed content follow this teaching source. Original images and attribution: https://www.tensorflow.org/tutorials/generative/style_transfer.
/// @see StyleGram(), GatysContentLoss(), GatysStyleLoss(), FrozenVgg19
inline void WriteStylePng(const nn::Tensor &rgb, const std::filesystem::path &filename) {
  PerceptualRequire(rgb.dim() == 4 && rgb.size(0) == 1 && rgb.size(3) == 3,
                    "single RGB image PNG");
  PerceptualRequire(
      nn::all(nn::logical_and(nn::is_finite(rgb), nn::logical_and(nn::greater_equal(rgb, 0.),
                                                                  nn::less_equal(rgb, 1.))))
          .item<bool>(),
      "finite projected RGB unit pixels required");
  const auto bytes = nn::floor(rgb.to(nn::DType::Float64) * 255. + .5);
  nn::save_png(nn::select(bytes, 0, 0), filename.string(), 0, 255);
}
} // namespace datamunge::nn::models
