#pragma once
/// @file bilinear_models.hpp
/// @brief Two independent convolution streams,aligned bilinear pooling and smoothed unit descriptors.
#include "datamunge/nn/angular_margin_models.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Mean aligned spatial outer products between two feature maps.
///
/// Flatten aligned H*W positions before batch matrix multiplication. Joint
/// spatial permutations preserve the result;independent permutations generally
/// do not. It is not the outer product of separate spatial means. No fitting.
///
/// @param first Float32/64[B,H,W,Ca],all dimensions positive.
/// @param second Same-type[B,H,W,Cb],Cb positive.
/// @return Same-type[B,Ca,Cb] mean spatial outer product,with gradients through both streams.
/// @throws std::invalid_argument For incompatible rank,shape,dtype,nonpositive dimensions or smoothing epsilon.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto matrix = n::models::SpatialBilinear(n::ones({1, 2, 2, 3}), n::ones({1, 2, 2, 4}));
/// @endcode
/// @par References
/// - Lin,RoyChowdhury and Maji(2015),Bilinear CNNs for Fine-grained Visual Recognition, https://arxiv.org/abs/1504.07889v3. Two independent VGG16-style streams,smoothed signed root,compact widths and from-scratch Flowers subset follow the explicit source teaching adaptation.
/// @see SpatialBilinear(), BilinearCNN()
inline nn::Tensor SpatialBilinear(const nn::Tensor &first, const nn::Tensor &second) {
  if (first.dim() != 4 || second.dim() != 4 || first.dtype() != second.dtype() ||
      (first.dtype() != nn::DType::Float32 && first.dtype() != nn::DType::Float64) ||
      first.shape()[0] < 1 || first.shape()[1] < 1 || first.shape()[2] < 1 ||
      first.shape()[3] < 1 || second.shape()[3] < 1 || first.shape()[0] != second.shape()[0] ||
      first.shape()[1] != second.shape()[1] || first.shape()[2] != second.shape()[2])
    throw std::invalid_argument("invalid bilinear feature geometry");
  const std::int64_t batch = first.size(0);
  const nn::Tensor a = nn::reshape(first, {batch, -1, first.size(-1)}),
                   b = nn::reshape(second, {batch, -1, second.size(-1)});
  return nn::matmul(a, b, true) / double(a.size(1));
}

/// @brief Apply a differentiable smoothed signed square root.
///
/// Derivative is(.5*abs(x)+epsilon)/(abs(x)+epsilon)^(3/2),including
/// 1/sqrt(epsilon) at zero. No mutation,parameters or hidden fitting.
///
/// @param input Float32/64 tensor of any shape,empty allowed.
/// @param epsilon Finite positive smoothing constant,default1e-5.
/// @return Same-type x/sqrt(abs(x)+epsilon),retaining input gradients.
/// @throws std::invalid_argument For incompatible rank,shape,dtype,nonpositive dimensions or smoothing epsilon.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto root = n::models::BilinearRoot(n::tensor({-4., 0., 9.}));
/// @endcode
/// @par References
/// - Lin,RoyChowdhury and Maji(2015),Bilinear CNNs for Fine-grained Visual Recognition, https://arxiv.org/abs/1504.07889v3. Two independent VGG16-style streams,smoothed signed root,compact widths and from-scratch Flowers subset follow the explicit source teaching adaptation.
/// @see SpatialBilinear(), BilinearCNN()
inline nn::Tensor BilinearRoot(const nn::Tensor &input, float epsilon = 1e-5f) {
  if ((input.dtype() != nn::DType::Float32 && input.dtype() != nn::DType::Float64) ||
      !std::isfinite(epsilon) || epsilon <= 0)
    throw std::invalid_argument("floating bilinear root and positive epsilon required");
  return input * nn::rsqrt(nn::abs(input) + double(epsilon));
}

/// @brief Normalize bilinear descriptor rows with a squared-norm floor.
///
/// Zero rows stay zero with finite floor-region gradients. Division/sqrt
/// retain the source arithmetic order. No learned state or fitting.
///
/// @param input Float32/64[B,D],D positive,empty B allowed.
/// @return Same-type[B,D] divided by sqrt(max(row squared norm,1e-12)).
/// @throws std::invalid_argument For incompatible rank,shape,dtype,nonpositive dimensions or smoothing epsilon.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto unit = n::models::BilinearUnit(n::tensor({3., 4.}, {1, 2}));
/// @endcode
/// @par References
/// - Lin,RoyChowdhury and Maji(2015),Bilinear CNNs for Fine-grained Visual Recognition, https://arxiv.org/abs/1504.07889v3. Two independent VGG16-style streams,smoothed signed root,compact widths and from-scratch Flowers subset follow the explicit source teaching adaptation.
/// @see SpatialBilinear(), BilinearCNN()
inline nn::Tensor BilinearUnit(const nn::Tensor &input) {
  if (input.dim() != 2 || input.shape()[1] < 1 ||
      (input.dtype() != nn::DType::Float32 && input.dtype() != nn::DType::Float64))
    throw std::invalid_argument("floating bilinear descriptor matrix required");
  return input / nn::sqrt(nn::maximum(nn::sum(nn::square(input), {1}, true),
                                      nn::scalar_like(double(1e-12f), input)));
}

/// @brief Own both stream maps,outer-product matrix,unit descriptor and class logits.
///
/// Tensor handles retain input/store gradients. Stage ordering is17 maps per
/// stream,then matrix,descriptor and logits,37 total. No optimizer or BN state.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(233);
/// const auto out = n::models::BilinearCNN(p, n::ones({2, 64, 64, 3}), 16, 6);
/// @endcode
/// @par References
/// - Lin,RoyChowdhury and Maji(2015),Bilinear CNNs for Fine-grained Visual Recognition, https://arxiv.org/abs/1504.07889v3. Two independent VGG16-style streams,smoothed signed root,compact widths and from-scratch Flowers subset follow the explicit source teaching adaptation.
/// @see BilinearCNN()
struct BilinearResult {
  nn::Tensor feature_a;           ///< Floating[B,Hf,Wf,512/divisor] first stream features.
  nn::Tensor feature_b;           ///< Same-shaped independent second stream features.
  nn::Tensor matrix;              ///< Floating[B,C,C] spatial mean outer products.
  nn::Tensor descriptor;          ///< Floating[B,C*C] smoothed-root unit representation.
  nn::Tensor logits;              ///< Floating[B,classes] linear classifier output.
  std::vector<nn::Tensor> stages; ///< All37 recorded feature/output maps.
};
/// @brief Run two independent VGG16-style feature streams and a bilinear classifier.
///
/// Subtract.5 from observed RGB. Each independent stream has13 biased
/// He-normal SAME ReLU convolutions and four2x2 stride2 VALID max pools.
/// Mean outer-product pooling,smoothed signed root,unit normalization and a
/// Glorot biased dense head follow. No pretraining,BN,dropout or optimizer;
/// warm all leaves before constructing the caller optimizer.
///
/// @param p Same-dtype learned store,creates/reuses54 biased conv/dense leaves.
/// @param rgb Float32/64[B,H,W,3],B positive,H/W at least16.
/// @param divisor Positive exact divisor of64,default1;compact16.
/// @param classes At least two output classes,default102.
/// @return Both feature maps,outer-product matrix,unit descriptor,logits and37 stages.
/// @throws std::invalid_argument For incompatible rank,shape,dtype,nonpositive dimensions or smoothing epsilon.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(233);
/// const auto out = n::models::BilinearCNN(p, n::ones({2, 64, 64, 3}), 16, 6);
/// @endcode
/// @par References
/// - Lin,RoyChowdhury and Maji(2015),Bilinear CNNs for Fine-grained Visual Recognition, https://arxiv.org/abs/1504.07889v3. Two independent VGG16-style streams,smoothed signed root,compact widths and from-scratch Flowers subset follow the explicit source teaching adaptation.
/// @see SpatialBilinear(), BilinearCNN()
inline BilinearResult BilinearCNN(nn::ParameterStore &p, const nn::Tensor &rgb,
                                  int divisor = 1, int classes = 102) {
  if (divisor < 1 || 64 % divisor || classes < 2)
    throw std::invalid_argument("bilinear CNN divisor must divide64 and classes>=2");
  if (rgb.dim() != 4 || rgb.shape()[0] < 1 || rgb.shape()[1] < 16 || rgb.shape()[2] < 16 ||
      rgb.shape()[3] != 3 || rgb.dtype() != p.dtype())
    throw std::invalid_argument("invalid bilinear RGB input");
  BilinearResult result;
  const int widths[] = {64, 128, 256, 512, 512}, layers[] = {2, 2, 3, 3, 3};
  nn::ConvLayerOptions config;
  config.activation = nn::Activation::Relu;
  config.init = nn::init::he_normal();
  for (int stream = 0; stream < 2; ++stream) {
    nn::Tensor x = rgb - .5;
    for (int block = 0; block < 5; ++block) {
      for (int layer = 0; layer < layers[block]; ++layer) {
        x = nn::conv2d(p,
                       "stream" + std::to_string(stream) + "/block" + std::to_string(block) +
                           "/conv" + std::to_string(layer),
                       x, widths[block] / divisor, 3, config);
        result.stages.push_back(x);
      }
      if (block < 4) {
        x = nn::max_pool2d(x, 2, 2);
        result.stages.push_back(x);
      }
    }
    (stream == 0 ? result.feature_a : result.feature_b) = x;
  }
  const std::int64_t width = 512 / divisor;
  result.matrix = SpatialBilinear(result.feature_a, result.feature_b);
  result.stages.push_back(result.matrix);
  result.descriptor =
      BilinearUnit(BilinearRoot(nn::reshape(result.matrix, {-1, width * width})));
  result.stages.push_back(result.descriptor);
  result.logits = nn::dense(p, "head", result.descriptor, classes);
  result.stages.push_back(result.logits);
  return result;
}

/// @brief Compute stable multiclass cross entropy for bilinear classifier logits.
///
/// Forwards the canonical validated softmax CE. No fitting,margin transform
/// or source mutation.
///
/// @param logits Float32/64[B,classes],B positive.
/// @param labels Int32/64[B] valid class IDs.
/// @param classes Number of classes,at least two.
/// @return Same-type scalar negative mean labelled log-softmax with logit gradients.
/// @throws std::invalid_argument For incompatible rank,shape,dtype,nonpositive dimensions or smoothing epsilon.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto loss = n::models::FlowerCrossEntropy(
///     n::zeros({2, 6}), n::from_vector(std::vector<std::int64_t>{0, 1}), 6);
/// @endcode
/// @par References
/// - Lin,RoyChowdhury and Maji(2015),Bilinear CNNs for Fine-grained Visual Recognition, https://arxiv.org/abs/1504.07889v3. Two independent VGG16-style streams,smoothed signed root,compact widths and from-scratch Flowers subset follow the explicit source teaching adaptation.
/// @see AngularCrossEntropy(), BilinearCNN()
inline nn::Tensor FlowerCrossEntropy(const nn::Tensor &logits, const nn::Tensor &labels,
                                     int classes) {
  return AngularCrossEntropy(logits, labels, classes);
}
} // namespace datamunge::nn::models
