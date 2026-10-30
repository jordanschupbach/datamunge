#pragma once
/// @file perceptual_models.hpp
/// @brief Frozen full-width VGG19 convolution features with tensor image preprocessing.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/checkpoint.hpp"
#include "datamunge/nn/layers.hpp"
#include <array>
#include <map>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Require an internal perceptual contract without mutating values.
///
/// Internal validation only;invalid_argument on failure.
///
/// @param valid Whether the contract holds.
/// @param message Error context if false.
/// @throws std::invalid_argument For an invalid perceptual tensor contract or changed frozen inventory.

/// @par References
/// - Simonyan and Zisserman(2015),Very Deep Convolutional Networks for Large-Scale Image Recognition, https://arxiv.org/abs/1409.1556;Gatys,Ecker and Bethge(2015),A Neural Algorithm of Artistic Style, https://arxiv.org/abs/1508.06576;official Keras VGG19 source, https://github.com/keras-team/keras/blob/master/keras/src/applications/vgg19.py. ImageNet Keras convolution weights are retained,with average pooling for this teaching Gatys extractor;not original-paper weight rescaling/optimization reproduction.
/// @see FrozenVgg19, VggImageNetInput(), LoadTensorMap()
inline void PerceptualRequire(bool valid, const std::string &message) {
  if (!valid)
    throw std::invalid_argument(message);
}
/// @brief Own named tensor handles for frozen feature archives.
///
/// Map copies retain tensor storage. FrozenVgg19 separately detaches/clones constants.
///
/// @throws std::invalid_argument For an invalid perceptual tensor contract or changed frozen inventory.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::TensorMap arrays;
/// @endcode
/// @par References
/// - Simonyan and Zisserman(2015),Very Deep Convolutional Networks for Large-Scale Image Recognition, https://arxiv.org/abs/1409.1556;Gatys,Ecker and Bethge(2015),A Neural Algorithm of Artistic Style, https://arxiv.org/abs/1508.06576;official Keras VGG19 source, https://github.com/keras-team/keras/blob/master/keras/src/applications/vgg19.py. ImageNet Keras convolution weights are retained,with average pooling for this teaching Gatys extractor;not original-paper weight rescaling/optimization reproduction.
/// @see FrozenVgg19, VggImageNetInput(), LoadTensorMap()
using TensorMap = std::map<std::string, nn::Tensor>;

/// @brief Load an NPZ tensor archive by unique member names.
///
/// Rejects duplicate names;archive decoding may throw I/O errors. No fitting or normalization.
///
/// @param path Readable named-tensor NPZ file.
/// @return Owning map of typed tensors.
/// @throws std::invalid_argument For an invalid perceptual tensor contract or changed frozen inventory.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// try {
///   const auto arrays = m::LoadTensorMap("missing-perceptual-doc-example.npz");
/// } catch (const std::exception &) {
/// }
/// @endcode
/// @par References
/// - Simonyan and Zisserman(2015),Very Deep Convolutional Networks for Large-Scale Image Recognition, https://arxiv.org/abs/1409.1556;Gatys,Ecker and Bethge(2015),A Neural Algorithm of Artistic Style, https://arxiv.org/abs/1508.06576;official Keras VGG19 source, https://github.com/keras-team/keras/blob/master/keras/src/applications/vgg19.py. ImageNet Keras convolution weights are retained,with average pooling for this teaching Gatys extractor;not original-paper weight rescaling/optimization reproduction.
/// @see FrozenVgg19, VggImageNetInput(), LoadTensorMap()
inline TensorMap LoadTensorMap(const std::string &path) {
  TensorMap out;
  for (auto &[name, value] : nn::load_npz(path))
    if (!out.emplace(name, value).second)
      throw std::invalid_argument("duplicate tensor archive name");
  return out;
}

/// @brief Convert encoded RGB unit values to ImageNet-centered BGR byte units.
///
/// Differentiable channel gather and affine tensor arithmetic. Does not clip,
/// linearize gamma,resize,fit statistics or mutate input.
///
/// @param rgb Floating Float32/64[N,H,W,3] encoded RGB.
/// @return Matching dtype/shape,BGR*255 minus103.939/116.779/123.68.
/// @throws std::invalid_argument For an invalid perceptual tensor contract or changed frozen inventory.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto bgr = m::VggImageNetInput(n::tensor({1., 0., .5}, {1, 1, 1, 3}));
/// @endcode
/// @par References
/// - Simonyan and Zisserman(2015),Very Deep Convolutional Networks for Large-Scale Image Recognition, https://arxiv.org/abs/1409.1556;Gatys,Ecker and Bethge(2015),A Neural Algorithm of Artistic Style, https://arxiv.org/abs/1508.06576;official Keras VGG19 source, https://github.com/keras-team/keras/blob/master/keras/src/applications/vgg19.py. ImageNet Keras convolution weights are retained,with average pooling for this teaching Gatys extractor;not original-paper weight rescaling/optimization reproduction.
/// @see FrozenVgg19, VggImageNetInput(), LoadTensorMap()
inline nn::Tensor VggImageNetInput(const nn::Tensor &rgb) {
  if (rgb.dim() != 4 || rgb.size(3) != 3 ||
      (rgb.dtype() != nn::DType::Float32 && rgb.dtype() != nn::DType::Float64))
    throw std::invalid_argument("floating NHWC RGB required");
  const nn::Tensor bgr = nn::gather(rgb, nn::tensor({2, 1, 0}, {3}, nn::DType::Int64), 3);
  return bgr * 255.0 - nn::tensor({103.939f, 116.779f, 123.68f}, {3}).to(rgb.dtype());
}

/// @brief Own graph-preserving Gatys content and five style feature handles.
///
/// Selected activations retain image gradients through frozen convolutions.
///
/// @throws std::invalid_argument For an invalid perceptual tensor contract or changed frozen inventory.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::TensorMap weights;
/// const int depths[] = {2, 2, 4, 4, 4}, widths[] = {64, 128, 256, 512, 512};
/// int input = 3;
/// // Zero weights are a shape-only documentation fixture,not pretrained features.
/// for (int b = 0; b < 5; ++b)
///   for (int l = 0; l < depths[b]; ++l) {
///     const auto name = "block" + std::to_string(b + 1) + "_conv" + std::to_string(l + 1);
///     weights.emplace(name + "_W", n::zeros({3, 3, input, widths[b]}));
///     weights.emplace(name + "_b", n::zeros({widths[b]}));
///     input = widths[b];
///   }
/// m::FrozenVgg19 vgg(weights);
/// const auto f = vgg.Extract(n::ones({1, 16, 16, 3}));
/// @endcode
/// @par References
/// - Simonyan and Zisserman(2015),Very Deep Convolutional Networks for Large-Scale Image Recognition, https://arxiv.org/abs/1409.1556;Gatys,Ecker and Bethge(2015),A Neural Algorithm of Artistic Style, https://arxiv.org/abs/1508.06576;official Keras VGG19 source, https://github.com/keras-team/keras/blob/master/keras/src/applications/vgg19.py. ImageNet Keras convolution weights are retained,with average pooling for this teaching Gatys extractor;not original-paper weight rescaling/optimization reproduction.
/// @see FrozenVgg19, VggImageNetInput(), LoadTensorMap()
struct PerceptualFeatures {
  nn::Tensor content;              ///< Differentiable block4_conv2 features[N,H/8,W/8,512].
  std::array<nn::Tensor, 5> style; ///< First ReLU activation from each of five blocks.
};

/// @brief Import all16 genuine-shape VGG19 convolution pairs as frozen constants.
///
/// Requires32 Float32 arrays and20,024,384 scalar values;only12,944,960
/// values through block5_conv1 execute. Actual reports must load genuine
/// verified pretrained weights. Shape/dtype/finite validation cannot establish
/// provenance;all-zero documentation fixtures are explicitly untrained.
/// Constants are detached cloned tensors outside optimizer stores. Input map
/// handles are owned for later frozen equality checks;the map object may die.
///
/// @throws std::invalid_argument For an invalid perceptual tensor contract or changed frozen inventory.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::TensorMap weights;
/// const int depths[] = {2, 2, 4, 4, 4}, widths[] = {64, 128, 256, 512, 512};
/// int input = 3;
/// // Zero weights are a shape-only documentation fixture,not pretrained features.
/// for (int b = 0; b < 5; ++b)
///   for (int l = 0; l < depths[b]; ++l) {
///     const auto name = "block" + std::to_string(b + 1) + "_conv" + std::to_string(l + 1);
///     weights.emplace(name + "_W", n::zeros({3, 3, input, widths[b]}));
///     weights.emplace(name + "_b", n::zeros({widths[b]}));
///     input = widths[b];
///   }
/// m::FrozenVgg19 vgg(weights);
/// @endcode
/// @par References
/// - Simonyan and Zisserman(2015),Very Deep Convolutional Networks for Large-Scale Image Recognition, https://arxiv.org/abs/1409.1556;Gatys,Ecker and Bethge(2015),A Neural Algorithm of Artistic Style, https://arxiv.org/abs/1508.06576;official Keras VGG19 source, https://github.com/keras-team/keras/blob/master/keras/src/applications/vgg19.py. ImageNet Keras convolution weights are retained,with average pooling for this teaching Gatys extractor;not original-paper weight rescaling/optimization reproduction.
/// @see FrozenVgg19, VggImageNetInput(), LoadTensorMap()
class FrozenVgg19 {
public:
  static constexpr std::array<int, 5> Depths = {
      2, 2, 4, 4, 4}; ///< Convolutions per canonical VGG19 block.
  static constexpr std::array<int, 5> Widths = {64, 128, 256, 512,
                                                512}; ///< Output channels per block.

  /// @brief Validate and freeze a complete named VGG19 convolution inventory.
  ///
  /// Copies source handles and independently detaches/clones every constant.
  /// Nonfinite,missing,mis-shaped or wrong-dtype entries throw. No optimizer.
  ///
  /// @param tensors Exact32 Float32 members block<b>_conv<l>_W[3,3,in,out] and _b[out].
  /// @throws std::invalid_argument For an invalid perceptual tensor contract or changed frozen inventory.
  /// @par Example
  /// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::TensorMap weights;
/// const int depths[] = {2, 2, 4, 4, 4}, widths[] = {64, 128, 256, 512, 512};
/// int input = 3;
/// // Zero weights are a shape-only documentation fixture,not pretrained features.
/// for (int b = 0; b < 5; ++b)
///   for (int l = 0; l < depths[b]; ++l) {
///     const auto name = "block" + std::to_string(b + 1) + "_conv" + std::to_string(l + 1);
///     weights.emplace(name + "_W", n::zeros({3, 3, input, widths[b]}));
///     weights.emplace(name + "_b", n::zeros({widths[b]}));
///     input = widths[b];
///   }
/// m::FrozenVgg19 vgg(weights);
/// @endcode
  /// @par References
  /// - Simonyan and Zisserman(2015),Very Deep Convolutional Networks for Large-Scale Image Recognition, https://arxiv.org/abs/1409.1556;Gatys,Ecker and Bethge(2015),A Neural Algorithm of Artistic Style, https://arxiv.org/abs/1508.06576;official Keras VGG19 source, https://github.com/keras-team/keras/blob/master/keras/src/applications/vgg19.py. ImageNet Keras convolution weights are retained,with average pooling for this teaching Gatys extractor;not original-paper weight rescaling/optimization reproduction.
  /// @see FrozenVgg19, VggImageNetInput(), LoadTensorMap()
  explicit FrozenVgg19(const TensorMap &tensors) : source(tensors) {
    PerceptualRequire(tensors.size() == 32, "exact16 VGG-19 kernel/bias pairs required");
    std::int64_t channels = 3, count = 0;
    for (int block = 0; block < 5; ++block)
      for (int layer = 0; layer < Depths[std::size_t(block)]; ++layer) {
        const auto name = Name(block, layer);
        const std::int64_t out = Widths[std::size_t(block)];
        for (const auto &entry : std::vector<std::pair<std::string, nn::Shape>>{
                 {name + "_W", {3, 3, channels, out}}, {name + "_b", {out}}}) {
          const auto it = tensors.find(entry.first);
          PerceptualRequire(it != tensors.end() && it->second.dtype() == nn::DType::Float32 &&
                                it->second.shape() == entry.second,
                            "VGG-19 pretrained tensor inventory/dtype/shape: " + entry.first);
          if (!nn::all(nn::is_finite(it->second)).item<bool>())
            throw std::invalid_argument("non-finite pretrained VGG-19 tensor");
          count += it->second.numel();
          constants.emplace(entry.first, it->second.detach().clone());
        }
        channels = out;
      }
    PerceptualRequire(count == 20024384,
                      "full pretrained VGG-19 convolution parameter census");
  }
  /// @brief Extract conv4_2 content and conv1_1..conv5_1 style activations.
  ///
  /// Input preprocesses to BGR255 minus fixed ImageNet means. SAME3x3
  /// convolutions/ReLU and VALID2x2 average pools preserve the source teaching
  /// policy. Odd spatial extents floor at pools. No weights,state or optimizer
  /// mutation;gradients reach image pixels but imported constants require no grad.
  ///
  /// @param rgb Float32[N,H,W,3],N>0,H/W>=16,encoded unit RGB.
  /// @return Six graph-preserving selected ReLU activations.
  /// @throws std::invalid_argument For an invalid perceptual tensor contract or changed frozen inventory.
  /// @par Example
  /// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::TensorMap weights;
/// const int depths[] = {2, 2, 4, 4, 4}, widths[] = {64, 128, 256, 512, 512};
/// int input = 3;
/// // Zero weights are a shape-only documentation fixture,not pretrained features.
/// for (int b = 0; b < 5; ++b)
///   for (int l = 0; l < depths[b]; ++l) {
///     const auto name = "block" + std::to_string(b + 1) + "_conv" + std::to_string(l + 1);
///     weights.emplace(name + "_W", n::zeros({3, 3, input, widths[b]}));
///     weights.emplace(name + "_b", n::zeros({widths[b]}));
///     input = widths[b];
///   }
/// m::FrozenVgg19 vgg(weights);
/// const auto f = vgg.Extract(n::ones({1, 16, 16, 3}));
/// @endcode
  /// @par References
  /// - Simonyan and Zisserman(2015),Very Deep Convolutional Networks for Large-Scale Image Recognition, https://arxiv.org/abs/1409.1556;Gatys,Ecker and Bethge(2015),A Neural Algorithm of Artistic Style, https://arxiv.org/abs/1508.06576;official Keras VGG19 source, https://github.com/keras-team/keras/blob/master/keras/src/applications/vgg19.py. ImageNet Keras convolution weights are retained,with average pooling for this teaching Gatys extractor;not original-paper weight rescaling/optimization reproduction.
  /// @see FrozenVgg19, VggImageNetInput(), LoadTensorMap()
  [[nodiscard]] PerceptualFeatures Extract(const nn::Tensor &rgb) const {
    if (rgb.dim() != 4 || rgb.size(0) < 1 || rgb.size(1) < 16 || rgb.size(2) < 16 ||
        rgb.size(3) != 3 || rgb.dtype() != nn::DType::Float32)
      throw std::invalid_argument("Float32 RGB NHWC with spatial dimensions>=16 required");
    PerceptualFeatures output;
    nn::Tensor x = VggImageNetInput(rgb);
    for (int block = 0; block < 5; ++block) {
      for (int layer = 0; layer < Depths[std::size_t(block)]; ++layer) {
        const auto name = Name(block, layer);
        x = nn::relu(
            nn::bias_add(nn::conv2d(x, constants.at(name + "_W")), constants.at(name + "_b")));
        if (layer == 0)
          output.style[std::size_t(block)] = x;
        if (block == 3 && layer == 1)
          output.content = x;
        if (block == 4 && layer == 0)
          return output;
      }
      x = nn::avg_pool2d(x, 2, 2, nn::Padding::Valid);
    }
    return output;
  }
  /// @brief Require every imported frozen value to equal its retained source bitwise.
  ///
  /// Tensor equality checks all32 arrays,shapes and requires_grad flags. Throws
  /// if exposed constants or retained caller source storage changed. No mutation.
  ///
  /// @throws std::invalid_argument For an invalid perceptual tensor contract or changed frozen inventory.
  /// @par Example
  /// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::TensorMap weights;
/// const int depths[] = {2, 2, 4, 4, 4}, widths[] = {64, 128, 256, 512, 512};
/// int input = 3;
/// // Zero weights are a shape-only documentation fixture,not pretrained features.
/// for (int b = 0; b < 5; ++b)
///   for (int l = 0; l < depths[b]; ++l) {
///     const auto name = "block" + std::to_string(b + 1) + "_conv" + std::to_string(l + 1);
///     weights.emplace(name + "_W", n::zeros({3, 3, input, widths[b]}));
///     weights.emplace(name + "_b", n::zeros({widths[b]}));
///     input = widths[b];
///   }
/// m::FrozenVgg19 vgg(weights);
/// vgg.CheckFrozen();
/// @endcode
  /// @par References
  /// - Simonyan and Zisserman(2015),Very Deep Convolutional Networks for Large-Scale Image Recognition, https://arxiv.org/abs/1409.1556;Gatys,Ecker and Bethge(2015),A Neural Algorithm of Artistic Style, https://arxiv.org/abs/1508.06576;official Keras VGG19 source, https://github.com/keras-team/keras/blob/master/keras/src/applications/vgg19.py. ImageNet Keras convolution weights are retained,with average pooling for this teaching Gatys extractor;not original-paper weight rescaling/optimization reproduction.
  /// @see FrozenVgg19, VggImageNetInput(), LoadTensorMap()
  void CheckFrozen() const {
    for (const auto &[name, value] : source) {
      const auto it = constants.find(name);
      PerceptualRequire(it != constants.end() && !it->second.requires_grad() &&
                            it->second.shape() == value.shape() &&
                            nn::all(nn::equal(it->second, value)).item<bool>(),
                        "all pretrained VGG-19 values remain bitwise unchanged");
    }
  }
  std::map<std::string, nn::Tensor>
      constants; ///< Detached cloned frozen kernels/biases,not optimizer leaves.

private:
  /// @brief Construct the canonical Keras layer name from zero-based metadata.
  ///
  /// Internal string metadata only.
  ///
  /// @param block Zero-based block index.
  /// @param layer Zero-based convolution index.
  /// @return block<b+1>_conv<l+1>.
  /// @throws std::invalid_argument For an invalid perceptual tensor contract or changed frozen inventory.

  /// @par References
  /// - Simonyan and Zisserman(2015),Very Deep Convolutional Networks for Large-Scale Image Recognition, https://arxiv.org/abs/1409.1556;Gatys,Ecker and Bethge(2015),A Neural Algorithm of Artistic Style, https://arxiv.org/abs/1508.06576;official Keras VGG19 source, https://github.com/keras-team/keras/blob/master/keras/src/applications/vgg19.py. ImageNet Keras convolution weights are retained,with average pooling for this teaching Gatys extractor;not original-paper weight rescaling/optimization reproduction.
  /// @see FrozenVgg19, VggImageNetInput(), LoadTensorMap()
  static std::string Name(int block, int layer) {
    return "block" + std::to_string(block + 1) + "_conv" + std::to_string(layer + 1);
  }
  TensorMap source; ///< Owning imported comparison handles retained for frozen checks.
};

} // namespace datamunge::nn::models
