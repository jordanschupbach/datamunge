#pragma once
/// @file visual_features.hpp
/// @brief Complete frozen original-maxpool VGG19tap inspection and honest channel display scales.
#include "datamunge/nn/perceptual_models.hpp"
#include <stdexcept>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Own actual frozen VGG19activation taps and first pre-activation map.
///
/// Owning tensor handles retain all five chosen feature endpoints. Includes
/// original centered input and first raw convolution for independent numeric
/// checks. No learned classifier head or optimizer state belongs to this value.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const m::FrozenVgg19 model(m::LoadTensorMap("build/cnn-datasets/style_transfer/vgg19.npz"));
/// const auto out = m::VggVisualFeatures(model, n::zeros({1, 64, 64, 3}));
/// @endcode
/// @par References
/// - Simonyan and Zisserman(2014/2015),Very Deep Convolutional Networks for Large-Scale Image Recognition, https://arxiv.org/abs/1409.1556;official Keras VGG19no-top ImageNet weights. This path executes all16convolutions with ORIGINALMAXpooling,not the average-pooling style-transfer variant. Activation/weight inspection is not class attribution,classification-head prediction or image optimization.
/// @see VggVisualFeatures(),FrozenVgg19
struct VisualFeatures {
  Tensor centered;  ///< Float32[N,H,W,3]RGB*255 reversed toBGR-minusImageNetmeans.
  Tensor first_pre; ///< Unrectified first3x3convolution[N,H,W,64].
  std::vector<Tensor> activation; ///< Five post-ReLUtaps:firstconv,lastconvblocks2..5.
  std::vector<std::string> names; ///< Canonical tap names in exact activation order.
};

/// @brief Run every frozen VGG19convolution and original MAXpool to inspect all five tap levels.
///
/// RGB*255->BGR-minus[103.939,116.779,123.68],all16SAME3x3convs/ReLUs,
/// four VALIDmaxpool2betweenblocks. Tapblock1_conv1,block2_conv2,block3_conv4,
/// block4_conv4,block5_conv4;at64pxsides64/32/16/8/4,widths64/128/256/512/512.
/// Unlike style-transfer features,this preserves original MAXpooling. No fitting,
/// random substitute,classification head or learned-constant mutation. All numeric
/// preprocessing/forward values remain tensors;caller chooses graph/no-grad mode.
///
/// @param model Validated immutable complete32array/20024384value original VGG19convolution constants.
/// @param rgb Float32[N,H,W,3],Npositive,H/W>=16,normally[0,1].
/// @return Centered input,first raw convolution and five canonical post-ReLUactivation taps.
/// @throws std::invalid_argument For invalid RGBgeometry/dtype or missing/incompatible model constants.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const m::FrozenVgg19 model(m::LoadTensorMap("build/cnn-datasets/style_transfer/vgg19.npz"));
/// const auto out = m::VggVisualFeatures(model, n::zeros({1, 64, 64, 3}));
/// @endcode
/// @par References
/// - Simonyan and Zisserman(2014/2015),Very Deep Convolutional Networks for Large-Scale Image Recognition, https://arxiv.org/abs/1409.1556;official Keras VGG19no-top ImageNet weights. This path executes all16convolutions with ORIGINALMAXpooling,not the average-pooling style-transfer variant. Activation/weight inspection is not class attribution,classification-head prediction or image optimization.
/// @see VggVisualFeatures(),FrozenVgg19
inline VisualFeatures VggVisualFeatures(const FrozenVgg19 &model, const Tensor &rgb) {
  if (rgb.dim() != 4 || rgb.size(0) < 1 || rgb.size(1) < 16 || rgb.size(2) < 16 ||
      rgb.size(3) != 3 || rgb.dtype() != nn::DType::Float32)
    throw std::invalid_argument("Float32NHWCRGBwithspatialaxes>=16required");
  VisualFeatures out;
  out.centered = VggImageNetInput(rgb);
  Tensor x = out.centered;
  for (int block = 0; block < 5; ++block) {
    const int depth = FrozenVgg19::Depths[std::size_t(block)];
    for (int layer = 0; layer < depth; ++layer) {
      const auto name =
          "block" + std::to_string(block + 1) + "_conv" + std::to_string(layer + 1);
      const Tensor pre = nn::bias_add(nn::conv2d(x, model.constants.at(name + "_W")),
                                      model.constants.at(name + "_b"));
      x = nn::relu(pre);
      if (block == 0 && layer == 0) {
        out.first_pre = pre;
        out.names.push_back(name);
        out.activation.push_back(x);
      } else if (block > 0 && layer == depth - 1) {
        out.names.push_back(name);
        out.activation.push_back(x);
      }
    }
    if (block < 4)
      x = nn::max_pool2d(x, 2, 2, nn::Padding::Valid);
  }
  return out;
}

/// @brief Convert every activation channel into a normalized grayscale tensor image.
///
/// Display normalization only;does not alter rawfeatures. Shared scale retains
/// within-layer relative amplitudes across both images/allchannels. Per-channel
/// scale can make weak responses look strong. Zero maps stay0;positive constant
/// channels remain visible. Different layers have different response units.
///
/// @param activation Finite nonnegative Float32[N,H,W,C]post-ReLUfeature tensor.
/// @param per_channel False uses ONEmaximum across all observations/channels of this layer;true uses each observation/channel maximum.
/// @return Float32[N*C,H,W,1]images,observation-major/channel-minor order.
/// @throws std::invalid_argument For empty/non-rank4,non-Float32,nonfinite or negative activations.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto images = m::VisualChannelImages(n::ones({2, 4, 4, 8}), false);
/// @endcode
/// @par References
/// - Simonyan and Zisserman(2014/2015),Very Deep Convolutional Networks for Large-Scale Image Recognition, https://arxiv.org/abs/1409.1556;official Keras VGG19no-top ImageNet weights. This path executes all16convolutions with ORIGINALMAXpooling,not the average-pooling style-transfer variant. Activation/weight inspection is not class attribution,classification-head prediction or image optimization.
/// @see VggVisualFeatures(),FrozenVgg19
inline Tensor VisualChannelImages(const Tensor &activation, bool per_channel = false) {
  if (activation.dim() != 4 || activation.numel() < 1 ||
      activation.dtype() != nn::DType::Float32 ||
      !nn::all(nn::is_finite(activation)).item<bool>() || nn::min(activation).item() < 0)
    throw std::invalid_argument("finite nonnegativeFloat32NHWCactivation required");
  const auto maps = nn::transpose(activation, {0, 3, 1, 2})
                        .reshape({activation.size(0) * activation.size(3), activation.size(1),
                                  activation.size(2), 1});
  const auto maximum = per_channel ? nn::max(maps, {1, 2, 3}, true) : nn::max(maps);
  return maps / nn::maximum(maximum, nn::full(maximum.shape(), 1e-30));
}
} // namespace datamunge::nn::models
