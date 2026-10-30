#pragma once
/// @file dense_crf_models.hpp
/// @brief Tensor exact bounded Gaussian/bilateral Potts CRF kernels and differentiable mean-field inference.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Configure exact small-grid Gaussian/bilateral Potts mean-field inference.
///
/// Weights and bandwidths are fixed caller settings,not learned leaves.
/// No optimizer or statistical parameter fitting is hidden.
///
/// @throws std::invalid_argument For invalid shapes,dtypes,nonfinite values or configuration;kernel overflow also rejects.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto kernel = n::models::DenseCrfKernel(n::zeros({2, 3, 3}));
/// @endcode
/// @par References
/// - Krahenbuhl and Koltun(2011), Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials, NeurIPS24,109-117, https://arxiv.org/abs/1210.5644. This implementation is exact bounded quadratic mean-field reference inference,not the paper's fast permutohedral approximation.
/// @see DenseCrfKernel(), DenseCrfMeanField()
struct DenseCrfConfig {
  float spatial_sigma = 3;     ///< Finite positive spatial Gaussian bandwidth in pixels.
  float appearance_sigma = 10; ///< Finite positive bilateral spatial bandwidth in pixels.
  float color_sigma =
      .1f; ///< Finite positive bilateral RGB bandwidth in input intensity units.
  float spatial_weight = .5f;  ///< Finite nonnegative spatial component strength.
  float appearance_weight = 2; ///< Finite nonnegative bilateral component strength.
  int iterations =
      5; ///< Positive default iteration count for callers;kernel construction does not use it.
};
/// @brief Construct detached exact spatial-plus-bilateral symmetrically normalized Potts affinities.
///
/// Off-diagonal Gaussian weights normalize independently per component by
/// sqrt(max(degree_i*degree_j,1e-30));weighted components then sum. Image
/// values first cast Float32,including RGB differences,then distances/exp/degree
/// normalization use Float64 tensors before each component casts Float32.
/// This preserves the source convention. A single pixel yields zero affinity.
/// O(P²) work/storage;bounded4096 pixels is an explicit reference limit,not
/// fast dense-CRF filtering. No gradients to image or fitted preprocessing.
///
/// @param image Float32/64[H,W,3] finite image,1<=H*W<=4096.
/// @param cfg Positive finite bandwidths and nonnegative finite component weights.
/// @return Detached Float32[P,P] kernel,nonnegative,symmetric,zero diagonal.
/// @throws std::invalid_argument For invalid shapes,dtypes,nonfinite values or configuration;kernel overflow also rejects.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto kernel = n::models::DenseCrfKernel(n::zeros({2, 3, 3}));
/// @endcode
/// @par References
/// - Krahenbuhl and Koltun(2011), Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials, NeurIPS24,109-117, https://arxiv.org/abs/1210.5644. This implementation is exact bounded quadratic mean-field reference inference,not the paper's fast permutohedral approximation.
/// @see DenseCrfMeanField()
inline nn::Tensor DenseCrfKernel(const nn::Tensor &image, const DenseCrfConfig &cfg = {}) {
  if (image.dim() != 3 || image.shape()[0] < 1 || image.shape()[1] < 1 ||
      image.shape()[2] != 3 || image.shape()[0] > 4096 / image.shape()[1] ||
      (image.dtype() != nn::DType::Float32 && image.dtype() != nn::DType::Float64) ||
      !(cfg.spatial_sigma > 0 && cfg.appearance_sigma > 0 && cfg.color_sigma > 0 &&
        cfg.spatial_weight >= 0 && cfg.appearance_weight >= 0) ||
      !std::isfinite(double(cfg.spatial_sigma) + cfg.appearance_sigma + cfg.color_sigma +
                     cfg.spatial_weight + cfg.appearance_weight) ||
      !nn::all(nn::less(nn::abs(image), std::numeric_limits<double>::infinity())).item<bool>())
    throw std::invalid_argument("invalid exact dense CRF image/settings");
  nn::NoGradGuard guard;
  const auto H = image.shape()[0], W = image.shape()[1], P = H * W;
  const auto type = nn::DType::Float64;
  const auto
      rows =
          (nn::arange(0, double(H)).reshape({H, 1}) * nn::ones({1, W})).reshape({P}).to(type),
      cols =
          (nn::ones({H, 1}) * nn::arange(0, double(W)).reshape({1, W})).reshape({P}).to(type),
      spatial = nn::square(rows.reshape({P, 1}) - rows.reshape({1, P})) +
                nn::square(cols.reshape({P, 1}) - cols.reshape({1, P}));
  const auto pixels = image.to(nn::DType::Float32).reshape({P, 3}),
             delta = (pixels.reshape({P, 1, 3}) - pixels.reshape({1, P, 3})).to(type),
             color_distance = nn::sum(nn::square(delta), {2}),
             off_diagonal = 1. - nn::eye(P, type);
  auto result = nn::zeros({P, P});
  for (int component = 0; component < 2; ++component) {
    const auto sigma = nn::full(
        {}, component ? double(cfg.appearance_sigma) : double(cfg.spatial_sigma), type);
    auto exponent = spatial / (2. * nn::square(sigma));
    if (component) {
      const auto color_sigma = nn::full({}, double(cfg.color_sigma), type);
      exponent = exponent + color_distance / (2. * nn::square(color_sigma));
    }
    const auto weights = nn::exp(-exponent) * off_diagonal, degree = nn::sum(weights, {1}),
               denominator =
                   nn::sqrt(nn::maximum(degree.reshape({P, 1}) * degree.reshape({1, P}),
                                        nn::full({}, 1e-30, type)));
    result =
        result +
        (weights * (component ? double(cfg.appearance_weight) : double(cfg.spatial_weight)) /
         denominator)
            .to(nn::DType::Float32);
  }
  if (!nn::all(nn::less(nn::abs(result), std::numeric_limits<double>::infinity()))
           .item<bool>())
    throw std::invalid_argument("dense CRF weight overflow");
  return result;
}
/// @brief Iterate differentiable exact Potts mean-field probabilities for fixed unary logits and affinities.
///
/// Q starts softmax(logits). Each update is softmax(logits+kernel*Q),because
/// the all-class Potts message is a per-pixel constant removed by softmax.
/// Preserves gradients through every iteration to unary logits and to a caller
/// explicitly differentiable kernel. No detach of the iterative probabilities,
/// optimizer,hidden fitting or input mutation. Cost O(iterations*P²*C).
///
/// @param logits Finite Float32/64[P,C],1<=P<=4096,C>=2.
/// @param kernel Same-dtype finite[P,P] affinity matrix;DenseCrfKernel gives detached nonnegative zero-self affinities.
/// @param iterations Positive update count,default5.
/// @return [P,C] probabilities after initial softmax and requested updates.
/// @throws std::invalid_argument For invalid shapes,dtypes,nonfinite values or configuration;kernel overflow also rejects.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto kernel = n::models::DenseCrfKernel(n::zeros({2, 3, 3}));
/// const auto probabilities = n::models::DenseCrfMeanField(n::zeros({6, 2}), kernel);
/// @endcode
/// @par References
/// - Krahenbuhl and Koltun(2011), Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials, NeurIPS24,109-117, https://arxiv.org/abs/1210.5644. This implementation is exact bounded quadratic mean-field reference inference,not the paper's fast permutohedral approximation.
/// @see DenseCrfKernel()
inline nn::Tensor DenseCrfMeanField(const nn::Tensor &logits, const nn::Tensor &kernel,
                                    int iterations = 5) {
  if (iterations < 1 || logits.dim() != 2 || logits.shape()[0] < 1 ||
      logits.shape()[0] > 4096 || logits.shape()[1] < 2 ||
      kernel.shape() != nn::Shape{logits.shape()[0], logits.shape()[0]} ||
      kernel.dtype() != logits.dtype() ||
      (logits.dtype() != nn::DType::Float32 && logits.dtype() != nn::DType::Float64) ||
      !nn::all(nn::less(nn::abs(logits), std::numeric_limits<double>::infinity()))
           .item<bool>() ||
      !nn::all(nn::less(nn::abs(kernel), std::numeric_limits<double>::infinity()))
           .item<bool>())
    throw std::invalid_argument("invalid dense CRF unary/kernel/iterations");
  auto probabilities = nn::softmax(logits, -1);
  for (int i = 0; i < iterations; ++i)
    probabilities = nn::softmax(logits + nn::matmul(kernel, probabilities), -1);
  return probabilities;
}
/// @brief Sample NHWC pixels at floor((target_index+.5)*source_size/target_size) centres.
///
/// Uses tensor index arithmetic;no image interpolation,quantization or mutation.
/// At96->32 selects coordinates1,4,7,...,94,matching the source CRF preview.
/// This pixel-centre convention differs from floor(index*scale) nearest resize.
/// No fitting,randomness or gradient to index metadata.
///
/// @param input Positive NHWC tensor,including singleton-channel integer masks.
/// @param height Positive output height,default32.
/// @param width Positive output width,default32.
/// @return Gathered same-dtype[N,height,width,C] values,with source gradients when floating.
/// @throws std::invalid_argument For invalid shapes,dtypes,nonfinite values or configuration;kernel overflow also rejects.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto sampled = n::models::SampleSegmentationGrid(n::ones({1, 96, 96, 3}));
/// @endcode
/// @par References
/// - Krahenbuhl and Koltun(2011), Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials, NeurIPS24,109-117, https://arxiv.org/abs/1210.5644. This implementation is exact bounded quadratic mean-field reference inference,not the paper's fast permutohedral approximation.
/// @see DenseCrfKernel()
inline nn::Tensor SampleSegmentationGrid(const nn::Tensor &input, int height = 32,
                                         int width = 32) {
  if (input.dim() != 4 || input.shape()[0] < 1 || input.shape()[1] < 1 ||
      input.shape()[2] < 1 || input.shape()[3] < 1 || height < 1 || width < 1)
    throw std::invalid_argument("invalid segmentation sample grid");
  const auto y = nn::clip(
                     nn::floor((nn::arange(0, double(height)).to(nn::DType::Float64) + .5) *
                               (double(input.shape()[1]) / height)),
                     0., double(input.shape()[1] - 1))
                     .to(nn::DType::Int64),
             x = nn::clip(
                     nn::floor((nn::arange(0, double(width)).to(nn::DType::Float64) + .5) *
                               (double(input.shape()[2]) / width)),
                     0., double(input.shape()[2] - 1))
                     .to(nn::DType::Int64);
  return nn::gather(nn::gather(input, y, 1), x, 2);
}
} // namespace datamunge::nn::models
