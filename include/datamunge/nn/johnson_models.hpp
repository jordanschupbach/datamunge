#pragma once
/// @file johnson_models.hpp
/// @brief Feed-forward Johnson transform with explicit BN state and genuine-shape frozen VGG16 features.
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/perceptual_models.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Reflect-pad both image axes with no duplicated boundary sample.
///
/// Matches MirrorPad REFLECT. Integer indexing is handled by canonical tensor padding.
///
/// @param x Floating NHWC image.
/// @param pad Nonnegative reflection padding,default40,less than H/W.
/// @return Graph-preserving padded NHWC image.
/// @throws std::invalid_argument For incompatible dimensions,metadata,parameters or changed frozen tensors.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto y = m::JohnsonReflect(n::ones({1, 64, 64, 3}));
/// @endcode
/// @par References
/// - Johnson,Alahi and Fei-Fei(2016),Perceptual Losses for Real-Time Style Transfer and Super-Resolution, https://arxiv.org/abs/1603.08155;original architecture supplement, https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf;Keras VGG16 source, https://github.com/keras-team/keras/blob/master/keras/src/applications/vgg16.py. Source teaching variant keeps BN and fixed initial loss normalization,base8 compact width and64px CIFAR contents;no paper-quality or real-time claim.
/// @see JohnsonTransform(), FrozenVgg16, JohnsonNorm()
inline nn::Tensor JohnsonReflect(const nn::Tensor &x, std::int64_t pad = 40) {
  return nn::mirror_pad(x, pad, pad, nn::PadMode::Reflect);
}

/// @brief Remove two pixels from each spatial side of a residual shortcut.
///
/// Matches two VALID3x3 convolutions without interpolating shortcut pixels.
///
/// @param x NHWC tensor with H/W>4.
/// @return Contiguous-coordinate view[N,H-4,W-4,C].
/// @throws std::invalid_argument For incompatible dimensions,metadata,parameters or changed frozen tensors.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto y = m::JohnsonCrop(n::ones({1, 12, 14, 2}));
/// @endcode
/// @par References
/// - Johnson,Alahi and Fei-Fei(2016),Perceptual Losses for Real-Time Style Transfer and Super-Resolution, https://arxiv.org/abs/1603.08155;original architecture supplement, https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf;Keras VGG16 source, https://github.com/keras-team/keras/blob/master/keras/src/applications/vgg16.py. Source teaching variant keeps BN and fixed initial loss normalization,base8 compact width and64px CIFAR contents;no paper-quality or real-time claim.
/// @see JohnsonTransform(), FrozenVgg16, JohnsonNorm()
inline nn::Tensor JohnsonCrop(const nn::Tensor &x) {
  return nn::slice(x, {0, 2, 2, 0}, {x.size(0), x.size(1) - 4, x.size(2) - 4, x.size(3)});
}

/// @brief Apply He-normal convolution with zero bias and explicit stride/padding.
///
/// Creates/reuses name/W and name/b. No activation,normalization or optimizer hidden.
///
/// @param p Mutable learned store.
/// @param name Unique learned-layer prefix.
/// @param x Floating NHWC features matching p dtype.
/// @param out Positive output channels.
/// @param kernel Positive square kernel size.
/// @param stride Positive stride,default1.
/// @param padding Default Same;Valid for cropped residual blocks.
/// @return Differentiable convolution result.
/// @throws std::invalid_argument For incompatible dimensions,metadata,parameters or changed frozen tensors.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(17);
/// const auto y = m::JohnsonConv(p, "c", n::ones({1, 4, 5, 2}), 3, 3);
/// @endcode
/// @par References
/// - Johnson,Alahi and Fei-Fei(2016),Perceptual Losses for Real-Time Style Transfer and Super-Resolution, https://arxiv.org/abs/1603.08155;original architecture supplement, https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf;Keras VGG16 source, https://github.com/keras-team/keras/blob/master/keras/src/applications/vgg16.py. Source teaching variant keeps BN and fixed initial loss normalization,base8 compact width and64px CIFAR contents;no paper-quality or real-time claim.
/// @see JohnsonTransform(), FrozenVgg16, JohnsonNorm()
inline nn::Tensor JohnsonConv(nn::ParameterStore &p, const std::string &name,
                              const nn::Tensor &x, std::int64_t out, std::int64_t kernel,
                              std::int64_t stride = 1,
                              nn::Padding padding = nn::Padding::Same) {
  const nn::Tensor w =
      p.get(name + "/W", {kernel, kernel, x.size(3), out}, nn::init::he_normal());
  const nn::Tensor b = p.get(name + "/b", {out}, nn::init::zeros());
  return nn::bias_add(nn::conv2d(x, w, nn::conv_options(stride, padding)), b);
}

/// @brief Reproduce Torch3x3/stride2/pad1 encoder coordinates.
///
/// Explicit symmetric zero padding differs from generic TensorFlow SAME phase.
///
/// @param p Mutable learned store.
/// @param name Unique layer prefix.
/// @param x Floating NHWC input.
/// @param out Positive output channels.
/// @return Zero-pad1 then VALID stride2 convolution.
/// @throws std::invalid_argument For incompatible dimensions,metadata,parameters or changed frozen tensors.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(17);
/// const auto y = m::JohnsonDown(p, "d", n::ones({1, 6, 8, 2}), 4);
/// @endcode
/// @par References
/// - Johnson,Alahi and Fei-Fei(2016),Perceptual Losses for Real-Time Style Transfer and Super-Resolution, https://arxiv.org/abs/1603.08155;original architecture supplement, https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf;Keras VGG16 source, https://github.com/keras-team/keras/blob/master/keras/src/applications/vgg16.py. Source teaching variant keeps BN and fixed initial loss normalization,base8 compact width and64px CIFAR contents;no paper-quality or real-time claim.
/// @see JohnsonTransform(), FrozenVgg16, JohnsonNorm()
inline nn::Tensor JohnsonDown(nn::ParameterStore &p, const std::string &name,
                              const nn::Tensor &x, std::int64_t out) {
  return JohnsonConv(p, name, nn::pad(x, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}), out, 3, 2,
                     nn::Padding::Valid);
}

/// @brief Reproduce Torch3x3/stride2/pad1/adjustment1 learned upsampling.
///
/// VALID transpose produces2H+1/2W+1,then drops first row/column and last
/// excess edge. He-normal true fan-in is9*Cin,not the stored output-channel axis.
///
/// @param p Mutable learned store.
/// @param name Unique layer prefix.
/// @param x Floating[N,H,W,Cin] input.
/// @param out Positive output channels.
/// @return Differentiable[N,2H,2W,out] biased output.
/// @throws std::invalid_argument For incompatible dimensions,metadata,parameters or changed frozen tensors.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(17);
/// const auto y = m::JohnsonUp(p, "u", n::ones({1, 2, 3, 4}), 2);
/// @endcode
/// @par References
/// - Johnson,Alahi and Fei-Fei(2016),Perceptual Losses for Real-Time Style Transfer and Super-Resolution, https://arxiv.org/abs/1603.08155;original architecture supplement, https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf;Keras VGG16 source, https://github.com/keras-team/keras/blob/master/keras/src/applications/vgg16.py. Source teaching variant keeps BN and fixed initial loss normalization,base8 compact width and64px CIFAR contents;no paper-quality or real-time claim.
/// @see JohnsonTransform(), FrozenVgg16, JohnsonNorm()
inline nn::Tensor JohnsonUp(nn::ParameterStore &p, const std::string &name,
                            const nn::Tensor &x, std::int64_t out) {
  const std::int64_t b = x.size(0), h = x.size(1), w = x.size(2);
  // He normal over the true fan-in 9 * in (nn::init::he_normal would read [3,3,out,in] as fan-in 9 * out).
  const nn::Tensor weights = p.get(name + "/W", {3, 3, out, x.size(3)},
                                   nn::init::normal(std::sqrt(2.0 / double(9 * x.size(3)))));
  const nn::Tensor transposed = nn::conv2d_transpose(
      x, weights, {b, 2 * h + 1, 2 * w + 1, out}, nn::conv_options(2, nn::Padding::Valid));
  const nn::Tensor cropped = nn::slice(transposed, {0, 1, 1, 0}, {b, 2 * h, 2 * w, out});
  return nn::bias_add(cropped, p.get(name + "/b", {out}, nn::init::zeros()));
}

/// @brief Apply learned BN with explicit training and moving-state mutation policy.
///
/// Momentum.9,epsilon1e-5. Caller warms learned and state leaves before
/// optimizer construction. Validation/inference must use training=false.
///
/// @param p Mutable gamma/beta store.
/// @param state Separate moving mean/variance store.
/// @param name Unique norm prefix.
/// @param x Floating NHWC features.
/// @param training True uses batch moments,false uses moving statistics.
/// @param update Mutate moving statistics only when training&&update.
/// @param relu Apply ReLU after BN,default true.
/// @return Normalized optional-ReLU features.
/// @throws std::invalid_argument For incompatible dimensions,metadata,parameters or changed frozen tensors.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(17), state(18);
/// const auto y = m::JohnsonNorm(p, state, "bn", n::ones({2, 3, 4, 2}), true, false);
/// @endcode
/// @par References
/// - Johnson,Alahi and Fei-Fei(2016),Perceptual Losses for Real-Time Style Transfer and Super-Resolution, https://arxiv.org/abs/1603.08155;original architecture supplement, https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf;Keras VGG16 source, https://github.com/keras-team/keras/blob/master/keras/src/applications/vgg16.py. Source teaching variant keeps BN and fixed initial loss normalization,base8 compact width and64px CIFAR contents;no paper-quality or real-time claim.
/// @see JohnsonTransform(), FrozenVgg16, JohnsonNorm()
inline nn::Tensor JohnsonNorm(nn::ParameterStore &p, nn::ParameterStore &state,
                              const std::string &name, const nn::Tensor &x, bool training,
                              bool update, bool relu = true) {
  nn::BatchNormOptions o;
  o.momentum = .9;
  o.eps = 1e-5;
  o.update_stats = update;
  const nn::Tensor y = nn::batch_norm(p, state, name, x, training, o);
  return relu ? nn::relu(y) : y;
}

/// @brief Add two VALID convolution/BN transforms to a cropped feature shortcut.
///
/// First conv BN/ReLU,second conv BN;no ReLU after addition. No optimizer hidden.
///
/// @param p Mutable learned store.
/// @param state Separate moving BN statistics.
/// @param name Unique residual prefix.
/// @param x Floating NHWC input with H/W>4.
/// @param training Use batch moments if true.
/// @param update Update moving state only during training,default true.
/// @return Input crop2 plus residual,NHWC spatial dimensions reduced by4.
/// @throws std::invalid_argument For incompatible dimensions,metadata,parameters or changed frozen tensors.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(17), state(18);
/// const auto y = m::JohnsonResidual(p, state, "r", n::ones({2, 12, 14, 2}), true, false);
/// @endcode
/// @par References
/// - Johnson,Alahi and Fei-Fei(2016),Perceptual Losses for Real-Time Style Transfer and Super-Resolution, https://arxiv.org/abs/1603.08155;original architecture supplement, https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf;Keras VGG16 source, https://github.com/keras-team/keras/blob/master/keras/src/applications/vgg16.py. Source teaching variant keeps BN and fixed initial loss normalization,base8 compact width and64px CIFAR contents;no paper-quality or real-time claim.
/// @see JohnsonTransform(), FrozenVgg16, JohnsonNorm()
inline nn::Tensor JohnsonResidual(nn::ParameterStore &p, nn::ParameterStore &state,
                                  const std::string &name, const nn::Tensor &x, bool training,
                                  bool update = true) {
  const std::int64_t channels = x.size(3);
  const nn::Tensor a = JohnsonConv(p, name + "/conv1", x, channels, 3, 1, nn::Padding::Valid);
  const nn::Tensor h = JohnsonNorm(p, state, name + "/bn1", a, training, update);
  const nn::Tensor b = JohnsonConv(p, name + "/conv2", h, channels, 3, 1, nn::Padding::Valid);
  return JohnsonCrop(x) + JohnsonNorm(p, state, name + "/bn2", b, training, update, false);
}

/// @brief Own generated RGB and every retained transformation stage.
///
/// Image is.5*(tanh(logits)+1). Eleven stages retain input/parameter graphs.
///
/// @throws std::invalid_argument For incompatible dimensions,metadata,parameters or changed frozen tensors.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(17), state(18);
/// const auto y = m::JohnsonTransform(p, state, n::ones({1, 64, 64, 3}), 8);
/// @endcode
/// @par References
/// - Johnson,Alahi and Fei-Fei(2016),Perceptual Losses for Real-Time Style Transfer and Super-Resolution, https://arxiv.org/abs/1603.08155;original architecture supplement, https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf;Keras VGG16 source, https://github.com/keras-team/keras/blob/master/keras/src/applications/vgg16.py. Source teaching variant keeps BN and fixed initial loss normalization,base8 compact width and64px CIFAR contents;no paper-quality or real-time claim.
/// @see JohnsonTransform(), FrozenVgg16, JohnsonNorm()
struct JohnsonOutputs {
  nn::Tensor image; ///< Bounded generated encoded RGB[N,H,W,3].
  std::vector<nn::Tensor>
      stages; ///< Eleven reflection,encoder,residual and learned-upsample activations.
};

/// @brief Stylize observed RGB with a complete feed-forward Johnson transformation net.
///
/// RGB*2-1,reflection40,9x9 encoder,two explicit stride2 downs,five VALID
/// cropped residual blocks,two learned transpose upsamples,9x9 output and
/// scaled tanh. BN follows every convolution except output;residual sums have
/// no post-add activation. Full base32 has1679235 scalars/3200 BN values;
/// compact base8 has108771/800. Neither VGG,style targets nor optimizer is
/// an inference input. Repeated inference leaves learned/BN values unchanged.
///
/// @param p Mutable learned store;warm all62 leaves before optimizer construction.
/// @param state Separate moving BN mean/variance store,30 tensors.
/// @param rgb Floating[N,H,W,3],H/W>40 and divisible by4,matching p dtype.
/// @param base Positive first feature width,default32;teaching compact8.
/// @param training Use BN batch moments if true,default false.
/// @param update Update BN statistics iff training&&update,default true.
/// @return Bounded RGB at original resolution and eleven stages.
/// @throws std::invalid_argument For incompatible dimensions,metadata,parameters or changed frozen tensors.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(17), state(18);
/// const auto y = m::JohnsonTransform(p, state, n::ones({1, 64, 64, 3}), 8);
/// @endcode
/// @par References
/// - Johnson,Alahi and Fei-Fei(2016),Perceptual Losses for Real-Time Style Transfer and Super-Resolution, https://arxiv.org/abs/1603.08155;original architecture supplement, https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf;Keras VGG16 source, https://github.com/keras-team/keras/blob/master/keras/src/applications/vgg16.py. Source teaching variant keeps BN and fixed initial loss normalization,base8 compact width and64px CIFAR contents;no paper-quality or real-time claim.
/// @see JohnsonTransform(), FrozenVgg16, JohnsonNorm()
inline JohnsonOutputs JohnsonTransform(nn::ParameterStore &p, nn::ParameterStore &state,
                                       const nn::Tensor &rgb, std::int64_t base = 32,
                                       bool training = false, bool update = true) {
  if (base < 1 || rgb.dim() != 4 || rgb.size(0) < 1 || rgb.size(1) <= 40 ||
      rgb.size(2) <= 40 || rgb.size(1) % 4 || rgb.size(2) % 4 || rgb.size(3) != 3 ||
      rgb.dtype() != p.dtype())
    throw std::invalid_argument("positive Johnson width required");
  JohnsonOutputs net;
  nn::Tensor x = JohnsonReflect(rgb * 2.0 - 1.0);
  net.stages.push_back(x);
  x = JohnsonNorm(p, state, "enc1/bn", JohnsonConv(p, "enc1", x, base, 9), training, update);
  net.stages.push_back(x);
  x = JohnsonNorm(p, state, "enc2/bn", JohnsonDown(p, "enc2", x, base * 2), training, update);
  net.stages.push_back(x);
  x = JohnsonNorm(p, state, "enc3/bn", JohnsonDown(p, "enc3", x, base * 4), training, update);
  net.stages.push_back(x);
  for (int i = 0; i < 5; ++i) {
    x = JohnsonResidual(p, state, "res" + std::to_string(i + 1), x, training, update);
    net.stages.push_back(x);
  }
  x = JohnsonNorm(p, state, "up1/bn", JohnsonUp(p, "up1", x, base * 2), training, update);
  net.stages.push_back(x);
  x = JohnsonNorm(p, state, "up2/bn", JohnsonUp(p, "up2", x, base), training, update);
  net.stages.push_back(x);
  net.image = .5 * (nn::tanh(JohnsonConv(p, "output", x, 3, 9)) + 1.0);
  return net;
}

/// @brief Own selected frozen VGG16 content and style activation handles.
///
/// Content is relu2_2;style is relu1_2,2_2,3_3,4_3. Requested blocks control
/// which handles are defined. Graphs retain observed-image derivatives.
///
/// @throws std::invalid_argument For incompatible dimensions,metadata,parameters or changed frozen tensors.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::TensorMap weights;
/// const int depths[] = {2, 2, 3, 3, 3}, widths[] = {64, 128, 256, 512, 512};
/// int input = 3;
/// // Zero weights are a shape-only documentation fixture,not pretrained features.
/// for (int b = 0; b < 5; ++b)
///   for (int l = 0; l < depths[b]; ++l) {
///     const auto name = "block" + std::to_string(b + 1) + "_conv" + std::to_string(l + 1);
///     weights.emplace(name + "_W", n::zeros({3, 3, input, widths[b]}));
///     weights.emplace(name + "_b", n::zeros({widths[b]}));
///     input = widths[b];
///   }
/// m::FrozenVgg16 vgg(weights);
/// const auto f = vgg.Extract(n::ones({1, 8, 8, 3}));
/// @endcode
/// @par References
/// - Johnson,Alahi and Fei-Fei(2016),Perceptual Losses for Real-Time Style Transfer and Super-Resolution, https://arxiv.org/abs/1603.08155;original architecture supplement, https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf;Keras VGG16 source, https://github.com/keras-team/keras/blob/master/keras/src/applications/vgg16.py. Source teaching variant keeps BN and fixed initial loss normalization,base8 compact width and64px CIFAR contents;no paper-quality or real-time claim.
/// @see JohnsonTransform(), FrozenVgg16, JohnsonNorm()
struct JohnsonFeatures {
  nn::Tensor content; ///< Differentiable relu2_2 content features.
  std::array<nn::Tensor, 4>
      style; ///< Last ReLU from blocks1..4,undefined beyond requested block count.
};

/// @brief Own all13 imported VGG16 convolution pairs as detached fixed constants.
///
/// Exact26 Float32 arrays/14,714,688 values are required. Real reports load
/// verified official ImageNet weights. Shape/finite checks alone do not prove
/// provenance. The source map object may die because handles are owned;every
/// constant is separately cloned and remains outside optimizer stores.
/// Selected features use original max pools,not Gatys average pooling.
///
/// @throws std::invalid_argument For incompatible dimensions,metadata,parameters or changed frozen tensors.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::TensorMap weights;
/// const int depths[] = {2, 2, 3, 3, 3}, widths[] = {64, 128, 256, 512, 512};
/// int input = 3;
/// // Zero weights are a shape-only documentation fixture,not pretrained features.
/// for (int b = 0; b < 5; ++b)
///   for (int l = 0; l < depths[b]; ++l) {
///     const auto name = "block" + std::to_string(b + 1) + "_conv" + std::to_string(l + 1);
///     weights.emplace(name + "_W", n::zeros({3, 3, input, widths[b]}));
///     weights.emplace(name + "_b", n::zeros({widths[b]}));
///     input = widths[b];
///   }
/// m::FrozenVgg16 vgg(weights);
/// @endcode
/// @par References
/// - Johnson,Alahi and Fei-Fei(2016),Perceptual Losses for Real-Time Style Transfer and Super-Resolution, https://arxiv.org/abs/1603.08155;original architecture supplement, https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf;Keras VGG16 source, https://github.com/keras-team/keras/blob/master/keras/src/applications/vgg16.py. Source teaching variant keeps BN and fixed initial loss normalization,base8 compact width and64px CIFAR contents;no paper-quality or real-time claim.
/// @see JohnsonTransform(), FrozenVgg16, JohnsonNorm()
class FrozenVgg16 {
public:
  /// @brief Validate and freeze the complete named VGG16 convolution inventory.
  ///
  /// Copies retained source handles and detaches/clones constants. Missing,
  /// nonfinite,mis-shaped or wrong-dtype entries throw. No optimizer or fitting.
  ///
  /// @param tensors Exact26 Float32 block<b>_conv<l>_W[3,3,in,out]/_b[out] members.
  /// @throws std::invalid_argument For incompatible dimensions,metadata,parameters or changed frozen tensors.
  /// @par Example
  /// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::TensorMap weights;
/// const int depths[] = {2, 2, 3, 3, 3}, widths[] = {64, 128, 256, 512, 512};
/// int input = 3;
/// // Zero weights are a shape-only documentation fixture,not pretrained features.
/// for (int b = 0; b < 5; ++b)
///   for (int l = 0; l < depths[b]; ++l) {
///     const auto name = "block" + std::to_string(b + 1) + "_conv" + std::to_string(l + 1);
///     weights.emplace(name + "_W", n::zeros({3, 3, input, widths[b]}));
///     weights.emplace(name + "_b", n::zeros({widths[b]}));
///     input = widths[b];
///   }
/// m::FrozenVgg16 vgg(weights);
/// @endcode
  /// @par References
  /// - Johnson,Alahi and Fei-Fei(2016),Perceptual Losses for Real-Time Style Transfer and Super-Resolution, https://arxiv.org/abs/1603.08155;original architecture supplement, https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf;Keras VGG16 source, https://github.com/keras-team/keras/blob/master/keras/src/applications/vgg16.py. Source teaching variant keeps BN and fixed initial loss normalization,base8 compact width and64px CIFAR contents;no paper-quality or real-time claim.
  /// @see JohnsonTransform(), FrozenVgg16, JohnsonNorm()
  explicit FrozenVgg16(const TensorMap &tensors) : source(tensors) {
    PerceptualRequire(tensors.size() == 26, "all26 VGG-16 weight/bias tensors required");
    std::int64_t in = 3, count = 0;
    const int depth[] = {2, 2, 3, 3, 3}, width[] = {64, 128, 256, 512, 512};
    for (int b = 0; b < 5; ++b) {
      for (int l = 0; l < depth[b]; ++l) {
        const auto name = Name(b, l);
        for (const auto &entry : std::vector<std::pair<std::string, nn::Shape>>{
                 {name + "_W", {3, 3, in, width[b]}}, {name + "_b", {width[b]}}}) {
          const auto it = tensors.find(entry.first);
          PerceptualRequire(it != tensors.end() && it->second.dtype() == nn::DType::Float32 &&
                                it->second.shape() == entry.second,
                            "VGG-16 tensor inventory/dtype/shape: " + entry.first);
          if (!nn::all(nn::is_finite(it->second)).item<bool>())
            throw std::invalid_argument("nonfinite VGG16 tensor");
          count += it->second.numel();
          constants.emplace(entry.first, it->second.detach().clone());
        }
        in = width[b];
      }
    }
    PerceptualRequire(count == 14714688, "VGG-16 convolution parameter census");
  }
  /// @brief Extract the last ReLU from requested VGG16 blocks with max pooling.
  ///
  /// Fixed centered BGR255 preprocessing,SAME3x3/ReLU and VALID2x2 max pools
  /// between blocks. H/W must be>=2^(blocks-1). No constant mutation or fitting;
  /// image gradients pass through frozen operations. Partial inpainting separately
  /// pools the last block features and rescales by255 as an explicit source policy.
  ///
  /// @param rgb Float32[N,H,W,3],positive batch and sufficient pool geometry.
  /// @param blocks Requested block count1..4,default4.
  /// @return Content relu2_2 if reached,and style handles for each requested block.
  /// @throws std::invalid_argument For incompatible dimensions,metadata,parameters or changed frozen tensors.
  /// @par Example
  /// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::TensorMap weights;
/// const int depths[] = {2, 2, 3, 3, 3}, widths[] = {64, 128, 256, 512, 512};
/// int input = 3;
/// // Zero weights are a shape-only documentation fixture,not pretrained features.
/// for (int b = 0; b < 5; ++b)
///   for (int l = 0; l < depths[b]; ++l) {
///     const auto name = "block" + std::to_string(b + 1) + "_conv" + std::to_string(l + 1);
///     weights.emplace(name + "_W", n::zeros({3, 3, input, widths[b]}));
///     weights.emplace(name + "_b", n::zeros({widths[b]}));
///     input = widths[b];
///   }
/// m::FrozenVgg16 vgg(weights);
/// const auto f = vgg.Extract(n::ones({1, 8, 8, 3}), 4);
/// @endcode
  /// @par References
  /// - Johnson,Alahi and Fei-Fei(2016),Perceptual Losses for Real-Time Style Transfer and Super-Resolution, https://arxiv.org/abs/1603.08155;original architecture supplement, https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf;Keras VGG16 source, https://github.com/keras-team/keras/blob/master/keras/src/applications/vgg16.py. Source teaching variant keeps BN and fixed initial loss normalization,base8 compact width and64px CIFAR contents;no paper-quality or real-time claim.
  /// @see JohnsonTransform(), FrozenVgg16, JohnsonNorm()
  [[nodiscard]] JohnsonFeatures Extract(const nn::Tensor &rgb, int blocks = 4) const {
    if (blocks < 1 || blocks > 4 || rgb.dim() != 4 || rgb.size(0) < 1 ||
        rgb.size(1) < (1LL << (blocks - 1)) || rgb.size(2) < (1LL << (blocks - 1)) ||
        rgb.size(3) != 3 || rgb.dtype() != nn::DType::Float32)
      throw std::invalid_argument("VGG16 Float32 RGB shape/requested blocks invalid");
    JohnsonFeatures net;
    nn::Tensor x = VggImageNetInput(rgb);
    const int depth[] = {2, 2, 3, 3};
    for (int b = 0; b < blocks; ++b) {
      for (int l = 0; l < depth[b]; ++l) {
        const auto name = Name(b, l);
        x = nn::relu(
            nn::bias_add(nn::conv2d(x, constants.at(name + "_W")), constants.at(name + "_b")));
      }
      net.style[std::size_t(b)] = x;
      if (b == 1)
        net.content = x;
      if (b < 3 && b + 1 < blocks)
        x = nn::max_pool2d(x, 2, 2, nn::Padding::Valid);
    }
    return net;
  }
  /// @brief Require every frozen VGG16 tensor to equal its retained source bitwise.
  ///
  /// Checks all26 tensors,shape and requires_grad flags with tensor equality.
  /// Throws if exposed constants or retained source storage changed.
  ///
  /// @throws std::invalid_argument For incompatible dimensions,metadata,parameters or changed frozen tensors.
  /// @par Example
  /// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::TensorMap weights;
/// const int depths[] = {2, 2, 3, 3, 3}, widths[] = {64, 128, 256, 512, 512};
/// int input = 3;
/// // Zero weights are a shape-only documentation fixture,not pretrained features.
/// for (int b = 0; b < 5; ++b)
///   for (int l = 0; l < depths[b]; ++l) {
///     const auto name = "block" + std::to_string(b + 1) + "_conv" + std::to_string(l + 1);
///     weights.emplace(name + "_W", n::zeros({3, 3, input, widths[b]}));
///     weights.emplace(name + "_b", n::zeros({widths[b]}));
///     input = widths[b];
///   }
/// m::FrozenVgg16 vgg(weights);
/// vgg.CheckFrozen();
/// @endcode
  /// @par References
  /// - Johnson,Alahi and Fei-Fei(2016),Perceptual Losses for Real-Time Style Transfer and Super-Resolution, https://arxiv.org/abs/1603.08155;original architecture supplement, https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf;Keras VGG16 source, https://github.com/keras-team/keras/blob/master/keras/src/applications/vgg16.py. Source teaching variant keeps BN and fixed initial loss normalization,base8 compact width and64px CIFAR contents;no paper-quality or real-time claim.
  /// @see JohnsonTransform(), FrozenVgg16, JohnsonNorm()
  void CheckFrozen() const {
    for (const auto &[name, value] : source) {
      const auto it = constants.find(name);
      PerceptualRequire(it != constants.end() && !it->second.requires_grad() &&
                            it->second.shape() == value.shape() &&
                            nn::all(nn::equal(it->second, value)).item<bool>(),
                        "all pretrained VGG-16 values unchanged bitwise");
    }
  }
  std::map<std::string, nn::Tensor>
      constants; ///< Independent detached cloned frozen kernels/biases.

private:
  /// @brief Construct a canonical Keras VGG16 layer name from zero-based metadata.
  ///
  /// Private string metadata;no numeric image work.
  ///
  /// @param b Zero-based block index.
  /// @param l Zero-based convolution index.
  /// @return block<b+1>_conv<l+1>.
  /// @throws std::invalid_argument For incompatible dimensions,metadata,parameters or changed frozen tensors.

  /// @par References
  /// - Johnson,Alahi and Fei-Fei(2016),Perceptual Losses for Real-Time Style Transfer and Super-Resolution, https://arxiv.org/abs/1603.08155;original architecture supplement, https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf;Keras VGG16 source, https://github.com/keras-team/keras/blob/master/keras/src/applications/vgg16.py. Source teaching variant keeps BN and fixed initial loss normalization,base8 compact width and64px CIFAR contents;no paper-quality or real-time claim.
  /// @see JohnsonTransform(), FrozenVgg16, JohnsonNorm()
  static std::string Name(int b, int l) {
    return "block" + std::to_string(b + 1) + "_conv" + std::to_string(l + 1);
  }
  TensorMap source; ///< Owning original imported tensor handles for frozen checks.
};

} // namespace datamunge::nn::models
