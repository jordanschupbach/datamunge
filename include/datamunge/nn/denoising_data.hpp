#pragma once
/// @file denoising_data.hpp
/// @brief Tensor grayscale ingress,seeded Box-Muller corruption,reflected reference and denoising scores.
#include "datamunge/nn/inpainting_data.hpp"
#include <cstdint>
#include <random>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Convert one RGB presentation triple to weighted encoded grayscale.
///
/// No gamma linearization or fitting. This preserves the source grayscale convention.
///
/// @param r Red channel in common input units.
/// @param g Green channel in the same units.
/// @param b Blue channel in the same units.
/// @return Double.299r+.587g+.114b,computed by tensors.
/// @throws std::invalid_argument For invalid geometry,dtype,range or incompatible stored parameters.
/// @par Example
/// @code
/// const auto gray = dm::nn::models::DnGray(1., 0., 0.);
/// @endcode
/// @par References
/// - Zhang et al.(2017),Beyond a Gaussian Denoiser:Residual Learning of Deep CNN for Image Denoising, https://arxiv.org/abs/1608.03981;author training https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_pytorch/main_train.py. Natural CIFAR32/sigma25/255 compact teaching fit is not Train400/BSD68 evaluation.
/// @see DnCNN(), DnCorrupt(), ScoreTensorDn()
inline double DnGray(double r, double g, double b) {
  const auto x = nn::from_vector(std::vector<double>{r, g, b}, {3});
  return nn::sum(x * nn::from_vector(std::vector<double>{.299, .587, .114}, {3})).item();
}
/// @brief Prepare selected natural32px grayscale images directly from byte tensors.
///
/// Float64/255 and weighted RGB arithmetic precede one Float32 cast.
/// No source labels,cropping,augmentation or learned parameters.
///
/// @param images Int32[N,32,32,3] RGB0..255 source.
/// @param rows Nonempty bounded source indices,in output order.
/// @return Float32[rows.size(),32,32,1] grayscale[0,1].
/// @throws std::invalid_argument For invalid geometry,dtype,range or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto clean = n::models::DnImages(n::zeros({2, 32, 32, 3}, n::DType::Int32), {1, 0});
/// @endcode
/// @par References
/// - Zhang et al.(2017),Beyond a Gaussian Denoiser:Residual Learning of Deep CNN for Image Denoising, https://arxiv.org/abs/1608.03981;author training https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_pytorch/main_train.py. Natural CIFAR32/sigma25/255 compact teaching fit is not Train400/BSD68 evaluation.
/// @see DnCNN(), DnCorrupt(), ScoreTensorDn()
inline nn::Tensor DnImages(const nn::Tensor &images, const std::vector<std::size_t> &rows) {
  if (images.dim() != 4 || images.size(0) < 1 || images.size(1) != 32 ||
      images.size(2) != 32 || images.size(3) != 3 || images.dtype() != nn::DType::Int32 ||
      rows.empty())
    throw std::invalid_argument("original32px Int32 RGB denoising images required");
  std::vector<std::int64_t> ids;
  for (auto r : rows) {
    if (r >= static_cast<std::size_t>(images.size(0)))
      throw std::invalid_argument("denoising row bounds");
    ids.push_back(static_cast<std::int64_t>(r));
  }
  const auto rgb = nn::take_rows(images, ids).to(nn::DType::Float64) / 255.;
  return (.299 * nn::slice_axis(rgb, 3, 0, 1) + .587 * nn::slice_axis(rgb, 3, 1, 2) +
          .114 * nn::slice_axis(rgb, 3, 2, 3))
      .to(nn::DType::Float32);
}
/// @brief Gather floating denoising image rows.
///
/// Host iteration serializes row metadata only;all pixel work uses tensors.
///
/// @param x Positive-batch Float32 rank4 input.
/// @param rows Bounded metadata indices;duplicates and empty output allowed.
/// @return Contiguous gathered tensor,graph preserved.
/// @throws std::invalid_argument For invalid geometry,dtype,range or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto x = n::models::DnBatch(n::ones({2, 3, 4, 1}), {1, 0, 1});
/// @endcode
/// @par References
/// - Zhang et al.(2017),Beyond a Gaussian Denoiser:Residual Learning of Deep CNN for Image Denoising, https://arxiv.org/abs/1608.03981;author training https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_pytorch/main_train.py. Natural CIFAR32/sigma25/255 compact teaching fit is not Train400/BSD68 evaluation.
/// @see DnCNN(), DnCorrupt(), ScoreTensorDn()
inline nn::Tensor DnBatch(const nn::Tensor &x, const std::vector<std::size_t> &rows) {
  if (x.dtype() != nn::DType::Float32 || x.dim() != 4 || x.size(0) < 1)
    throw std::invalid_argument("positive float NHWC denoising batch required");
  std::vector<std::int64_t> ids;
  for (auto r : rows) {
    if (r >= static_cast<std::size_t>(x.size(0)))
      throw std::invalid_argument("denoising batch row bounds");
    ids.push_back(static_cast<std::int64_t>(r));
  }
  return nn::take_rows(x, ids).contiguous();
}
/// @brief Own aligned clean,noisy and signed residual tensors.
///
/// Handles retain their own storage;the clean tensor is shared without mutation.
///
/// @throws std::invalid_argument For invalid geometry,dtype,range or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::DnCorrupt(n::full({1, 3, 4, 1}, .5), 25. / 255, 41);
/// @endcode
/// @par References
/// - Zhang et al.(2017),Beyond a Gaussian Denoiser:Residual Learning of Deep CNN for Image Denoising, https://arxiv.org/abs/1608.03981;author training https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_pytorch/main_train.py. Natural CIFAR32/sigma25/255 compact teaching fit is not Train400/BSD68 evaluation.
/// @see DnCNN(), DnCorrupt(), ScoreTensorDn()
struct DnData {
  nn::Tensor clean; ///< Float32[N,H,W,1] clean intensities in[0,1].
  nn::Tensor noisy; ///< Float32 matching-shape unbounded corrupted intensities.
  nn::Tensor noise; ///< Float32 actual noisy-clean residual after corruption rounding.
};
/// @brief Add reproducibly seeded unbounded Gaussian noise using tensor Box-Muller arithmetic.
///
/// Host RNG emits only raw53-bit integer draws in source order. Uniform scaling,
/// Box-Muller radius/angle/trigonometry,noise addition and residual subtraction
/// use tensors. Final odd draw is truncated. Sigma0 gives exact identity,inputs
/// are not clipped. Fresh fitting seeds should change each epoch;keep test fixed.
///
/// @param clean Nonempty Float32[N,H,W,1] finite intensities[0,1].
/// @param sigma Finite nonnegative AWGN standard deviation,in intensity units.
/// @param seed mt19937_64 stream seed,independent for fitting/validation/test.
/// @return Clean handle,Float32 unbounded noisy image and actual rounded noisy-clean target.
/// @throws std::invalid_argument For invalid geometry,dtype,range or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::DnCorrupt(n::full({2, 3, 4, 1}, .5), 25. / 255, 41);
/// @endcode
/// @par References
/// - Zhang et al.(2017),Beyond a Gaussian Denoiser:Residual Learning of Deep CNN for Image Denoising, https://arxiv.org/abs/1608.03981;author training https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_pytorch/main_train.py. Natural CIFAR32/sigma25/255 compact teaching fit is not Train400/BSD68 evaluation.
/// @see DnCNN(), DnCorrupt(), ScoreTensorDn()
inline DnData DnCorrupt(const nn::Tensor &clean, double sigma, std::uint64_t seed) {
  if (clean.dtype() != nn::DType::Float32 || clean.dim() != 4 || clean.size(3) != 1 ||
      clean.numel() < 1 || !std::isfinite(sigma) || sigma < 0 ||
      !nn::all(nn::logical_and(nn::greater_equal(clean, 0.), nn::less_equal(clean, 1.)))
           .item<bool>())
    throw std::invalid_argument("finite clean grayscale[0,1] and nonnegative sigma required");
  const auto count = clean.numel(), pairs = count / 2 + count % 2;
  std::vector<std::int64_t> u(pairs), v(pairs);
  std::mt19937_64 rng(seed);
  for (std::int64_t i = 0; i < pairs; ++i) {
    u[i] = static_cast<std::int64_t>(rng() >> 11);
    v[i] = static_cast<std::int64_t>(rng() >> 11);
  }
  const auto a = (nn::from_vector(u, {pairs}).to(nn::DType::Float64) + 1.) / 9007199254740992.,
             b = (nn::from_vector(v, {pairs}).to(nn::DType::Float64) + 1.) / 9007199254740992.;
  const auto radius = nn::sqrt(-2. * nn::log(a)), angle = 6.2831853071795864769 * b;
  const auto z =
      nn::slice_axis(nn::stack({radius * nn::cos(angle), radius * nn::sin(angle)}, 1)
                         .reshape({pairs * 2}),
                     0, 0, count)
          .reshape(clean.shape());
  const auto noisy = (clean.to(nn::DType::Float64) + sigma * z).to(nn::DType::Float32);
  return {clean, noisy, noisy - clean};
}
/// @brief Gather clean,noisy and residual rows together.
///
/// No corruption rerun or numerical host-array work.
///
/// @param source Aligned denoising tensor handles.
/// @param rows Requested bounded row metadata,duplicates/empty allowed.
/// @return Aligned gathered clean/noisy/noise tensors.
/// @throws std::invalid_argument For invalid geometry,dtype,range or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::DnCorrupt(n::full({2, 3, 4, 1}, .5), .1, 41);
/// const auto batch = n::models::DnBatch(d, {1, 0});
/// @endcode
/// @par References
/// - Zhang et al.(2017),Beyond a Gaussian Denoiser:Residual Learning of Deep CNN for Image Denoising, https://arxiv.org/abs/1608.03981;author training https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_pytorch/main_train.py. Natural CIFAR32/sigma25/255 compact teaching fit is not Train400/BSD68 evaluation.
/// @see DnCNN(), DnCorrupt(), ScoreTensorDn()
inline DnData DnBatch(const DnData &source, const std::vector<std::size_t> &rows) {
  return {DnBatch(source.clean, rows), DnBatch(source.noisy, rows),
          DnBatch(source.noise, rows)};
}
/// @brief Reflect one index without duplicating the boundary sample.
///
/// Integer geometry metadata only;64-bit period avoids signed-int overflow.
///
/// @param i Signed logical coordinate,including outside bounds.
/// @param length Positive axis length.
/// @return Index0..length-1,or0 for a singleton axis.
/// @throws std::invalid_argument For invalid geometry,dtype,range or incompatible stored parameters.
/// @par Example
/// @code
/// const auto i = dm::nn::models::DnReflect(-1, 3); // 1
/// @endcode
/// @par References
/// - Zhang et al.(2017),Beyond a Gaussian Denoiser:Residual Learning of Deep CNN for Image Denoising, https://arxiv.org/abs/1608.03981;author training https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_pytorch/main_train.py. Natural CIFAR32/sigma25/255 compact teaching fit is not Train400/BSD68 evaluation.
/// @see DnCNN(), DnCorrupt(), ScoreTensorDn()
inline int DnReflect(int i, int length) {
  if (length < 1)
    throw std::invalid_argument("positive reflected axis length required");
  if (length == 1)
    return 0;
  const std::int64_t period = 2 * (std::int64_t(length) - 1);
  auto k = std::int64_t(i) % period;
  if (k < 0)
    k += period;
  return static_cast<int>(k < length ? k : period - k);
}
/// @brief Apply a fixed3x3 mean filter with reflected edges.
///
/// Reflection indices are geometry metadata;gather and Float64 convolution
/// perform all image arithmetic. Singleton axes are supported. The filter has
/// no fitted coefficients or optimizer;it is a deterministic reference.
///
/// @param image Positive[N,H,W,1] grayscale image,finite values supported.
/// @return Float32 matching-shape averaged image,graph preserved.
/// @throws std::invalid_argument For invalid geometry,dtype,range or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto y = n::models::DnBoxFilter(n::tensor({0., 1., 2., 3., 4., 5.}, {1, 2, 3, 1}));
/// @endcode
/// @par References
/// - Zhang et al.(2017),Beyond a Gaussian Denoiser:Residual Learning of Deep CNN for Image Denoising, https://arxiv.org/abs/1608.03981;author training https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_pytorch/main_train.py. Natural CIFAR32/sigma25/255 compact teaching fit is not Train400/BSD68 evaluation.
/// @see DnCNN(), DnCorrupt(), ScoreTensorDn()
inline nn::Tensor DnBoxFilter(const nn::Tensor &image) {
  if (image.dim() != 4 || image.size(0) < 1 || image.size(1) < 1 || image.size(2) < 1 ||
      image.size(3) != 1 || image.size(1) > std::numeric_limits<int>::max() - 1 ||
      image.size(2) > std::numeric_limits<int>::max() - 1)
    throw std::invalid_argument("positive grayscale box-filter geometry required");
  std::vector<std::int64_t> ys, xs;
  for (int i = -1; i <= image.size(1); ++i)
    ys.push_back(DnReflect(i, int(image.size(1))));
  for (int i = -1; i <= image.size(2); ++i)
    xs.push_back(DnReflect(i, int(image.size(2))));
  const auto padded =
      nn::gather(nn::gather(image.to(nn::DType::Float64),
                            nn::from_vector(ys, {static_cast<std::int64_t>(ys.size())}), 1),
                 nn::from_vector(xs, {static_cast<std::int64_t>(xs.size())}), 2);
  return (nn::conv2d(padded, nn::ones({3, 3, 1, 1}, nn::DType::Float64),
                     nn::conv_options(1, nn::Padding::Valid)) /
          9.)
      .to(nn::DType::Float32);
}
/// @brief Own raw and clipped denoising diagnostics.
///
/// Perfect-image PSNR is positive infinity. Numerical work uses Float64 tensors,
/// without updating a learned model,BN statistics or optimizer.
///
/// @throws std::invalid_argument For invalid geometry,dtype,range or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto clean = n::full({2, 3, 4, 1}, .5);
/// const auto score = n::models::ScoreTensorDn(clean, clean);
/// @endcode
/// @par References
/// - Zhang et al.(2017),Beyond a Gaussian Denoiser:Residual Learning of Deep CNN for Image Denoising, https://arxiv.org/abs/1608.03981;author training https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_pytorch/main_train.py. Natural CIFAR32/sigma25/255 compact teaching fit is not Train400/BSD68 evaluation.
/// @see DnCNN(), DnCorrupt(), ScoreTensorDn()
struct TensorDnMetrics {
  nn::Tensor mse;          ///< Pooled raw MSE over every sample.
  nn::Tensor psnr;         ///< Mean per-image raw PSNR,dB.
  nn::Tensor clipped_psnr; ///< Mean per-image clipped-estimate PSNR,dB.
  nn::Tensor outside;      ///< Fraction of raw estimate samples outside[0,1].
};
/// @brief Own raw and clipped denoising diagnostics. Host presentation view.
///
/// Perfect-image PSNR is positive infinity. Numerical work uses Float64 tensors,
/// without updating a learned model,BN statistics or optimizer.
///
/// @throws std::invalid_argument For invalid geometry,dtype,range or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto clean = n::full({2, 3, 4, 1}, .5);
/// const auto score = n::models::DnScore(clean, clean);
/// @endcode
/// @par References
/// - Zhang et al.(2017),Beyond a Gaussian Denoiser:Residual Learning of Deep CNN for Image Denoising, https://arxiv.org/abs/1608.03981;author training https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_pytorch/main_train.py. Natural CIFAR32/sigma25/255 compact teaching fit is not Train400/BSD68 evaluation.
/// @see DnCNN(), DnCorrupt(), ScoreTensorDn()
struct DnMetrics {
  double mse = 0;          ///< Pooled raw MSE over every sample.
  double psnr = 0;         ///< Mean per-image raw PSNR,dB.
  double clipped_psnr = 0; ///< Mean per-image clipped-estimate PSNR,dB.
  double outside = 0;      ///< Fraction of raw estimate samples outside[0,1].
};
/// @brief Compute scalar unit-range denoising PSNR.
///
/// Delegates numerical work and validation to the shared tensor PSNR implementation.
///
/// @param mse Finite nonnegative MSE;zero means perfect reconstruction.
/// @return Double-10log10(MSE),positive infinity for zero.
/// @throws std::invalid_argument For invalid geometry,dtype,range or incompatible stored parameters.
/// @par Example
/// @code
/// const double db = dm::nn::models::DnPSNR(.01); // 20
/// @endcode
/// @par References
/// - Zhang et al.(2017),Beyond a Gaussian Denoiser:Residual Learning of Deep CNN for Image Denoising, https://arxiv.org/abs/1608.03981;author training https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_pytorch/main_train.py. Natural CIFAR32/sigma25/255 compact teaching fit is not Train400/BSD68 evaluation.
/// @see DnCNN(), DnCorrupt(), ScoreTensorDn()
inline double DnPSNR(double mse) {
  return InpaintPSNR(mse);
}
/// @brief Score pooled denoising MSE and mean per-image raw/clipped PSNR.
///
/// Difference is promoted to Float64 before squaring. Only clipped PSNR projects
/// estimates to[0,1];raw MSE and outside fraction retain the full output range.
/// Mean log-PSNR is not PSNR of pooled MSE.
///
/// @param estimate Nonempty Float32[N,H,W,C] raw estimates,finite.
/// @param clean Same-shaped finite clean intensities,usually[0,1].
/// @return Detached Float64 scalar MSE,raw/clipped PSNR and outside fraction.
/// @throws std::invalid_argument For invalid geometry,dtype,range or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto score =
///     n::models::ScoreTensorDn(n::tensor({.1, .2}, {2, 1, 1, 1}), n::zeros({2, 1, 1, 1}));
/// @endcode
/// @par References
/// - Zhang et al.(2017),Beyond a Gaussian Denoiser:Residual Learning of Deep CNN for Image Denoising, https://arxiv.org/abs/1608.03981;author training https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_pytorch/main_train.py. Natural CIFAR32/sigma25/255 compact teaching fit is not Train400/BSD68 evaluation.
/// @see DnCNN(), DnCorrupt(), ScoreTensorDn()
inline TensorDnMetrics ScoreTensorDn(const nn::Tensor &estimate, const nn::Tensor &clean) {
  nn::NoGradGuard guard;
  if (estimate.dtype() != nn::DType::Float32 || estimate.dim() != 4 || estimate.size(0) < 1 ||
      estimate.numel() < 1 || estimate.shape() != clean.shape() ||
      !nn::all(nn::less(nn::abs(estimate), std::numeric_limits<double>::infinity()))
           .item<bool>() ||
      !nn::all(nn::less(nn::abs(clean), std::numeric_limits<double>::infinity())).item<bool>())
    throw std::invalid_argument("aligned finite Float32 denoising estimates required");
  const auto x = estimate.to(nn::DType::Float64), y = clean.to(nn::DType::Float64);
  const auto mse = nn::mean(nn::square(x - y), {1, 2, 3}),
             clipped = nn::mean(nn::square(nn::clip(x, 0., 1.) - y), {1, 2, 3});
  return {
      nn::mean(mse), nn::mean(InpaintPSNR(mse)), nn::mean(InpaintPSNR(clipped)),
      nn::mean(nn::logical_or(nn::less(x, 0.), nn::greater(x, 1.)).to(nn::DType::Float64))};
}
/// @brief Present tensor denoising diagnostics as four host scalars.
///
/// Compatibility egress only;all numerical work delegates to ScoreTensorDn.
///
/// @param estimate Finite nonempty Float32 raw estimate.
/// @param clean Matching finite clean tensor.
/// @return Four doubles with canonical scorer units and infinity policy.
/// @throws std::invalid_argument For invalid geometry,dtype,range or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto m = n::models::DnScore(n::ones({1, 3, 4, 1}), n::ones({1, 3, 4, 1}));
/// @endcode
/// @par References
/// - Zhang et al.(2017),Beyond a Gaussian Denoiser:Residual Learning of Deep CNN for Image Denoising, https://arxiv.org/abs/1608.03981;author training https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_pytorch/main_train.py. Natural CIFAR32/sigma25/255 compact teaching fit is not Train400/BSD68 evaluation.
/// @see DnCNN(), DnCorrupt(), ScoreTensorDn()
inline DnMetrics DnScore(const nn::Tensor &estimate, const nn::Tensor &clean) {
  const auto m = ScoreTensorDn(estimate, clean);
  return {m.mse.item(), m.psnr.item(), m.clipped_psnr.item(), m.outside.item()};
}
/// @brief Render noisy,box-filter,restored,clean and four-times red error tiles.
///
/// Numerical clipping is presentation-only. Float64 source-style lround(255*x)
/// quantizes every pixel. Use save_image_grid for layout/scaling;no optimizer.
///
/// @param data Aligned clean/noisy grayscale tensors.
/// @param restored Matching raw learned clean estimate.
/// @param box Matching fixed-filter reference.
/// @return Float32[N,H,5W,3] display-clipped/byte-quantized RGB tiles.
/// @throws std::invalid_argument For invalid geometry,dtype,range or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::DnCorrupt(n::full({1, 3, 4, 1}, .5), .1, 41);
/// const auto box = n::models::DnBoxFilter(d.noisy);
/// const auto tiles = n::models::DnPreviewTiles(d, box, box);
/// @endcode
/// @par References
/// - Zhang et al.(2017),Beyond a Gaussian Denoiser:Residual Learning of Deep CNN for Image Denoising, https://arxiv.org/abs/1608.03981;author training https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_pytorch/main_train.py. Natural CIFAR32/sigma25/255 compact teaching fit is not Train400/BSD68 evaluation.
/// @see DnCNN(), DnCorrupt(), ScoreTensorDn()
inline nn::Tensor DnPreviewTiles(const DnData &data, const nn::Tensor &restored,
                                 const nn::Tensor &box) {
  if (data.clean.dim() != 4 || data.clean.size(0) < 1 || data.clean.size(3) != 1 ||
      data.noisy.shape() != data.clean.shape() || restored.shape() != data.clean.shape() ||
      box.shape() != data.clean.shape())
    throw std::invalid_argument("aligned grayscale denoising preview required");
  const auto rgb = nn::ones({1, 1, 1, 3}, nn::DType::Float64),
             red = nn::from_vector(std::vector<double>{1., 0., 0.}, {1, 1, 1, 3});
  const auto tiles = nn::concat(
      {data.noisy.to(nn::DType::Float64) * rgb, box.to(nn::DType::Float64) * rgb,
       restored.to(nn::DType::Float64) * rgb, data.clean.to(nn::DType::Float64) * rgb,
       4. * nn::abs(restored.to(nn::DType::Float64) - data.clean.to(nn::DType::Float64)) *
           red},
      2);
  return (nn::floor(255. * nn::clip(tiles, 0., 1.) + .5) / 255.).to(nn::DType::Float32);
}
} // namespace datamunge::nn::models
