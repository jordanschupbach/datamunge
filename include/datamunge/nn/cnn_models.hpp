#pragma once
/// @file cnn_models.hpp
/// @brief Shared MNIST CNN and sparse LeNet-5 architecture definitions; no training loops.
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <array>
#include <string>
#include <vector>

/// @brief Reusable tensor model definitions, separate from fitting and evaluation.
namespace datamunge::nn::models {
namespace nn = datamunge::nn;

/// @brief Intermediate activations of the two-block MNIST CNN.
/// @par Example
/// @code
/// dm::nn::ParameterStore p(17);
/// const auto stages=dm::nn::models::MnistCnn(p,dm::nn::zeros({1,784}));
/// @endcode
/// @see MnistCnn()
struct MnistCnnOutputs {
  nn::Tensor image;  ///< Reshaped input [N,28,28,1].
  nn::Tensor conv1;  ///< First ReLU map [N,28,28,16].
  nn::Tensor pool1;  ///< First pooled map [N,14,14,16].
  nn::Tensor conv2;  ///< Second ReLU map [N,14,14,32].
  nn::Tensor pool2;  ///< Second pooled map [N,7,7,32].
  nn::Tensor hidden; ///< Dense ReLU features [N,64].
  nn::Tensor logits; ///< Unnormalised class scores [N,10].
};

/// @brief Build the shared two-block 28x28 MNIST CNN.
///
/// Two SAME 3x3 convolutions (16 then 32 channels) each use ReLU and 2x2
/// max-pooling, followed by Dense(64)-ReLU and ten logits. Creates/reuses
/// parameters without fitting. Input and store dtypes must agree.
/// @param p Mutable parameter store; prefixes conv1, conv2, dense, output.
/// @param input Floating [N,784] or [N,28,28,1] input pixels.
/// @return Every intermediate activation and [N,10] logits.
/// @throws std::invalid_argument On incompatible tensor shapes or parameter-store entries.
///
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::models;
/// n::ParameterStore p(17);
/// const auto stages=m::MnistCnn(p,n::zeros({1,784}));
/// @endcode
///
/// @par References
/// - LeCun et al. (1998), Gradient-Based Learning Applied to Document Recognition,
/// https://doi.org/10.1109/5.726791
/// @see LeNet5()
inline MnistCnnOutputs MnistCnn(nn::ParameterStore &p, const nn::Tensor &input) {
  MnistCnnOutputs out;
  nn::ConvLayerOptions options;
  options.activation = nn::Activation::Relu;
  out.image = nn::reshape(input, {-1, 28, 28, 1});
  out.conv1 = nn::conv2d(p, "conv1", out.image, 16, 3, options);
  out.pool1 = nn::max_pool2d(out.conv1, 2, 2);
  out.conv2 = nn::conv2d(p, "conv2", out.pool1, 32, 3, options);
  out.pool2 = nn::max_pool2d(out.conv2, 2, 2);
  out.hidden = nn::dense(p, "dense", nn::reshape(out.pool2, {-1, 7 * 7 * 32}), 64,
                         nn::Activation::Relu);
  out.logits = nn::dense(p, "output", out.hidden, 10);
  return out;
}

/// @brief Return the original sparse C3 connection table.
///
/// Lifetime is the process lifetime; no allocation after first initialisation.
/// @return Static immutable lists for 16 output maps, totalling 60 input connections.
///
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::models;
/// const auto& table=m::LeNetC3Connections(); // 16 output maps
/// @endcode
///
/// @par References
/// - LeCun et al. (1998), Gradient-Based Learning Applied to Document Recognition,
/// https://doi.org/10.1109/5.726791
/// @see LeNet5()
inline const std::array<std::vector<std::int64_t>, 16> &LeNetC3Connections() {
  // Immutable sparse connectivity, initialised once for the process.
  static const std::array<std::vector<std::int64_t>, 16> table = {{{0, 1, 2},
                                                                   {1, 2, 3},
                                                                   {2, 3, 4},
                                                                   {3, 4, 5},
                                                                   {4, 5, 0},
                                                                   {5, 0, 1},
                                                                   {0, 1, 2, 3},
                                                                   {1, 2, 3, 4},
                                                                   {2, 3, 4, 5},
                                                                   {3, 4, 5, 0},
                                                                   {4, 5, 0, 1},
                                                                   {5, 0, 1, 2},
                                                                   {0, 1, 3, 4},
                                                                   {1, 2, 4, 5},
                                                                   {0, 2, 3, 5},
                                                                   {0, 1, 2, 3, 4, 5}}};
  return table;
}

/// @brief Apply the scaled LeNet activation, 1.7159 tanh(2x/3).
///
/// @param x Floating tensor of any shape.
/// @return Same-shape differentiable tensor.
/// @throws std::invalid_argument On incompatible tensor shapes or parameter-store entries.
///
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::models;
/// const auto zero=m::LeNetTanh(n::zeros({1})); // zero
/// @endcode
///
/// @par References
/// - LeCun et al. (1998), Gradient-Based Learning Applied to Document Recognition,
/// https://doi.org/10.1109/5.726791
/// @see LeNet5()
inline nn::Tensor LeNetTanh(const nn::Tensor &x) {
  return 1.7159 * nn::tanh(x * (2. / 3.));
}

/// @brief Apply learned 2x2 sum-pooling, scale, bias and scaled tanh.
///
/// Creates/reuses two trainable scalars per input channel; the average-pool sum
/// is multiplied by the learned scale. Parameter dtype follows p.
/// @param p Mutable learnable parameter store.
/// @param name Parameter prefix; /scale initialises at .25 and /bias at zero.
/// @param x Floating NHWC tensor, both spatial axes at least two.
/// @return NHWC tensor with halved spatial sizes (VALID pooling).
/// @throws std::invalid_argument On incompatible tensor shapes or parameter-store entries.
///
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::models;
/// n::ParameterStore p(17);
/// const auto y=m::LeNetSubsample(p,"s",n::ones({1,4,4,2}));
/// @endcode
///
/// @par References
/// - LeCun et al. (1998), Gradient-Based Learning Applied to Document Recognition,
/// https://doi.org/10.1109/5.726791
/// @see LeNet5()
inline nn::Tensor LeNetSubsample(nn::ParameterStore &p, const std::string &name,
                                 const nn::Tensor &x) {
  const std::int64_t channels = x.size(-1);
  auto sum = 4. * nn::avg_pool2d(x, 2, 2);
  auto scale = p.get(name + "/scale", {channels}, nn::init::constant(.25));
  auto bias = p.get(name + "/bias", {channels}, nn::init::zeros());
  return LeNetTanh(sum * scale + bias);
}

/// @brief Apply the original sparsely connected LeNet C3 stage.
///
/// Only the 60 connected 5x5 filters are stored; absent-channel derivatives
/// are exactly zero. The stage has 1516 learned scalars including biases.
/// @param p Mutable learnable parameter store.
/// @param name Prefix for 16 independently stored output-map filters.
/// @param x Floating [N,H,W,6], H,W at least five.
/// @return Scaled-tanh output [N,H-4,W-4,16].
/// @throws std::invalid_argument On incompatible tensor shapes or parameter-store entries.
///
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::models;
/// n::ParameterStore p(17);
/// const auto y=m::LeNetC3(p,"c3",n::ones({1,14,14,6}));
/// @endcode
///
/// @par References
/// - LeCun et al. (1998), Gradient-Based Learning Applied to Document Recognition,
/// https://doi.org/10.1109/5.726791
/// @see LeNet5()
inline nn::Tensor LeNetC3(nn::ParameterStore &p, const std::string &name,
                          const nn::Tensor &x) {
  nn::ConvLayerOptions options;
  options.padding = nn::Padding::Valid;
  std::vector<nn::Tensor> maps;
  for (std::size_t i = 0; i < LeNetC3Connections().size(); ++i) {
    const auto &channels = LeNetC3Connections()[i];
    auto selected =
        nn::gather(x, nn::from_vector(channels, {std::int64_t(channels.size())}), 3);
    maps.push_back(nn::conv2d(p, name + "/map" + std::to_string(i), selected, 1, 5, options));
  }
  return LeNetTanh(nn::concat(maps, 3));
}

/// @brief Construct fixed seven-segment digit prototypes for the RBF readout.
///
/// These are our seven-segment codes, not the original paper's digit art.
/// The return owns its storage and no learned parameters.
/// @return Float32 [10,84] codes in {-1,+1}, flattened 12x7 pictures.
/// @throws std::invalid_argument On incompatible tensor shapes or parameter-store entries.
///
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::models;
/// const auto codes=m::LeNetDigitCodes(); // [10,84]
/// @endcode
///
/// @par References
/// - LeCun et al. (1998), Gradient-Based Learning Applied to Document Recognition,
/// https://doi.org/10.1109/5.726791
/// @see LeNet5()
inline nn::Tensor LeNetDigitCodes() {
  std::vector<float> codes(10 * 84, -1.f);
  // Segment order: top, upper right, lower right, bottom, lower left,
  // upper left, middle. Each code has unit magnitude at every coordinate.
  const std::array<std::string, 10> segments = {
      "abcdef", "bc", "abdeg", "abcdg", "bcfg", "acdfg", "acdefg", "abc", "abcdefg", "abcdfg"};
  for (int digit = 0; digit < 10; ++digit)
    for (char segment : segments[digit]) {
      auto pixel = [&](int y, int x) {
        codes[static_cast<std::size_t>(digit * 84 + y * 7 + x)] = 1.f;
      };
      if (segment == 'a' || segment == 'd' || segment == 'g') {
        int y = segment == 'a' ? 1 : segment == 'd' ? 10 : 5;
        for (int x = 1; x <= 5; ++x)
          pixel(y, x);
      } else {
        const bool upper = segment == 'b' || segment == 'f';
        const int x = segment == 'b' || segment == 'c' ? 5 : 1;
        for (int y = upper ? 2 : 6; y <= (upper ? 4 : 9); ++y)
          pixel(y, x);
      }
    }
  return nn::from_vector(codes, {10, 84});
}

/// @brief Compute negative squared distances to fixed digit prototypes.
///
/// Prototype constants cast to the feature dtype; no learned output weights.
/// The softmax-cross-entropy training loss is a modern adaptation.
/// @param features Floating [N,84] LeNet F6 activations; Float32 or Float64.
/// @return Differentiable [N,10] logits, zero for a matching prototype.
/// @throws std::invalid_argument On incompatible tensor shapes or parameter-store entries.
///
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::models;
/// const auto logits=m::LeNetRbfLogits(m::LeNetDigitCodes());
/// @endcode
///
/// @par References
/// - LeCun et al. (1998), Gradient-Based Learning Applied to Document Recognition,
/// https://doi.org/10.1109/5.726791
/// @see LeNet5()
inline nn::Tensor LeNetRbfLogits(const nn::Tensor &features) {
  auto difference =
      nn::expand_dims(features, 1) - nn::cast(LeNetDigitCodes(), features.dtype());
  return -nn::sum(nn::square(difference), {2});
}

/// @brief Intermediate tensors from LeNet-5, retaining their autodiff graphs.
/// @par Example
/// @code
/// dm::nn::ParameterStore p(17);
/// const auto stages=dm::nn::models::LeNet5(p,dm::nn::zeros({1,784}));
/// @endcode
/// @see LeNet5()
struct LeNetOutputs {
  nn::Tensor image;  ///< Padded, rescaled [N,32,32,1] input.
  nn::Tensor c1;     ///< First valid convolution [N,28,28,6].
  nn::Tensor s2;     ///< Learned subsampling [N,14,14,6].
  nn::Tensor c3;     ///< Sparse valid convolution [N,10,10,16].
  nn::Tensor s4;     ///< Learned subsampling [N,5,5,16].
  nn::Tensor c5;     ///< Convolution [N,1,1,120].
  nn::Tensor f6;     ///< Dense scaled-tanh features [N,84].
  nn::Tensor logits; ///< Fixed-prototype negative squared distances [N,10].
};

/// @brief Build all stages of the 60000-parameter LeNet-5 network.
///
/// Pads to 32x32, maps intensities to [-.1,1.175], uses sparse C3 and learned
/// subsampling, then fixed seven-segment prototypes. Does not fit or update
/// parameters. Gradients remain available through every trainable stage.
/// @param p Mutable floating parameter store, using c1/s2/c3/s4/c5/f6 prefixes.
/// @param flat_pixels Floating [N,784] or reshape-compatible MNIST pixels in [0,1].
/// @return All intermediate stages and ten negative squared-distance logits.
/// @throws std::invalid_argument On incompatible tensor shapes or parameter-store entries.
///
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::models;
/// n::ParameterStore p(17);
/// const auto outputs=m::LeNet5(p,n::zeros({1,784})); // p.count()==60000
/// @endcode
///
/// @par References
/// - LeCun et al. (1998), Gradient-Based Learning Applied to Document Recognition,
/// https://doi.org/10.1109/5.726791
/// @see LeNet5()
inline LeNetOutputs LeNet5(nn::ParameterStore &p, const nn::Tensor &flat_pixels) {
  LeNetOutputs out;
  auto padded =
      nn::pad(nn::reshape(flat_pixels, {-1, 28, 28, 1}), {{0, 0}, {2, 2}, {2, 2}, {0, 0}});
  out.image = 1.275 * padded - .1;
  nn::ConvLayerOptions options;
  options.padding = nn::Padding::Valid;
  out.c1 = LeNetTanh(nn::conv2d(p, "c1", out.image, 6, 5, options));
  out.s2 = LeNetSubsample(p, "s2", out.c1);
  out.c3 = LeNetC3(p, "c3", out.s2);
  out.s4 = LeNetSubsample(p, "s4", out.c3);
  out.c5 = LeNetTanh(nn::conv2d(p, "c5", out.s4, 120, 5, options));
  out.f6 = LeNetTanh(nn::dense(p, "f6", nn::reshape(out.c5, {-1, 120}), 84));
  out.logits = LeNetRbfLogits(out.f6);
  return out;
}

} // namespace datamunge::nn::models
