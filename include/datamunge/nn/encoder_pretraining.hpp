#pragma once
/// @file encoder_pretraining.hpp
/// @brief Frozen ReLU features, fitting-only population scaling and caller-owned linear probes.
#include <limits>

#include "datamunge/nn/adversarial_autoencoder.hpp"
#include "datamunge/nn/autograd.hpp"

namespace datamunge::nn::models {
/// @brief Validation shared by frozen feature extraction and linear probing.
namespace transfer_detail {
/// @brief Check a finite matching floating vector, optionally positive.
/// @param value Float32/64 vector[width].
/// @param width Required positive width.
/// @param type Required floating dtype.
/// @param positive Require every value>0 when true.
/// @throws std::invalid_argument For invalid shape/dtype/values.
inline void Vector(const Tensor& value, std::int64_t width, DType type, bool positive = false) {
  if (!value.defined() || value.shape() != Shape{width} || width <= 0 || value.dtype() != type ||
      (type != DType::Float32 && type != DType::Float64))
    throw std::invalid_argument("transfer state needs matching floating vectors");
  for (double v : value.contiguous().to_vector<double>())
    if (!std::isfinite(v) || (positive && v <= 0))
      throw std::invalid_argument("transfer vectors must be finite and scales positive");
}
/// @brief Check representable feature, head and batch products.
/// @param batch Positive image count.
/// @param features Positive input width.
/// @param latent Positive frozen code width.
/// @param classes Positive class count.
/// @throws std::invalid_argument For invalid dimensions or int64 product overflow.
inline void
Geometry(std::int64_t batch, std::int64_t features, std::int64_t latent, std::int64_t classes) {
  const auto limit = std::numeric_limits<std::int64_t>::max();
  if (batch <= 0 || features <= 0 || latent <= 0 || classes <= 0 || features > limit / latent ||
      latent > limit / classes || batch > limit / std::max(features, std::max(latent, classes)))
    throw std::invalid_argument("transfer dimensions must have representable products");
}
/// @brief Validate optional-free classification targets before head leaves are created.
/// @param labels Int32/Int64 vector[batch].
/// @param batch Positive input row count.
/// @param classes Positive class count.
/// @throws std::invalid_argument For invalid labels or IDs outside[0,classes).
inline void Labels(const Tensor& labels, std::int64_t batch, std::int64_t classes) {
  if (!labels.defined() || labels.shape() != Shape{batch} ||
      (labels.dtype() != DType::Int32 && labels.dtype() != DType::Int64))
    throw std::invalid_argument("transfer labels must be an aligned integer vector");
  for (auto id : labels.contiguous().to_vector<std::int64_t>())
    if (id < 0 || id >= classes)
      throw std::invalid_argument("transfer label outside class vocabulary");
}
} // namespace transfer_detail

/// @brief Extract ReLU features with explicit frozen weights and a live input path.
///
/// Computes relu(image*weight+bias). The supplied weight/bias are detached, so
/// only image derivatives remain live; this is frozen transfer, not fine-tuning.
/// Finite signed matching Float32/64 matrices accept strides. No targets,
/// standardization, parameter creation, fitting or RNG is involved. Result owns
/// lifetime, inputs unchanged; callers synchronize concurrent input mutation.
/// ReLU uses the core's zero subgradient at zero. Computed overflow is not
/// repaired. Work O(B*F*L), activation storage O(B*L).
/// @param image Finite matching pixel/feature rows[B,F], B,F positive.
/// @param weight Finite frozen matrix[F,L], L positive.
/// @param bias Finite matching frozen vector[L].
/// @return Owning code[B,L], with input gradients and stopped weight/bias paths.
/// @throws std::invalid_argument For invalid inputs or dimension products.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto code=n::models::FrozenEncoderFeatures(n::ones({2,3}),n::ones({3,2}),n::zeros({2}));
/// // code[2,2] contains threes; supplied encoder leaves are not trained.
/// @endcode
/// @par References
/// - Vincent et al. (2010), Stacked Denoising Autoencoders,
///   https://www.jmlr.org/papers/v11/vincent10a.html . This migrated teaching
///   variant uses one frozen ReLU encoder rather than stacked supervised fine-tuning.
/// @see FitTransferScaling(), FrozenEncoderClassifier
inline Tensor FrozenEncoderFeatures(const Tensor& image, const Tensor& weight, const Tensor& bias) {
  if (!weight.defined() || weight.dim() != 2)
    throw std::invalid_argument("frozen encoder weight must be a matrix");
  const auto features = weight.size(0), latent = weight.size(1);
  aae_detail::Matrix(weight, latent, weight.dtype(), false);
  transfer_detail::Vector(bias, latent, weight.dtype());
  aae_detail::Matrix(image, features, weight.dtype(), false);
  transfer_detail::Geometry(image.size(0), features, latent, 1);
  return relu(matmul(image, weight.detach()) + bias.detach());
}

/// @brief Own detached fitting-code population means and floored population deviations.
///
/// Both vectors have code dtype and width L. They own storage and retain no
/// autograd path to fitting rows. No fitting provenance can be inferred from
/// the arrays themselves; the caller chooses fitting data explicitly.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto scaling=n::models::FitTransferScaling(n::ones({3,2}));
/// // center=[1,1]; scale=[float(1e-6),float(1e-6)].
/// @endcode
/// @see FitTransferScaling(), FrozenEncoderClassifier
struct TransferScaling {
  Tensor center; ///< Owning detached per-coordinate population means[L].
  Tensor scale;  ///< Owning detached max(floor,population standard deviation)[L].
};

/// @brief Fit detached per-feature population scaling from explicit fitting codes only.
///
/// For each coordinate, accumulates the mean in logical row order in double,
/// then a second-pass sum of squared differences from that double mean; divides
/// variance by B, takes sqrt and floors before casting both vectors to code
/// dtype. This preserves original Float32 rounding. A constant/singleton column
/// gets floor, not zero or sample-standard-deviation NaN. Finite signed nonempty
/// Float32/64[B,L] inputs accept strides and gradient flags; outputs are detached.
/// The positive finite floor must not overflow or underflow to zero in the
/// output dtype. Computed accumulation overflow is rejected. No labels, other
/// splits, mutation or shared state. Work O(B*L), host validation/storage O(B*L).
/// @param code Finite fitting-only matrix[B,L], B,L positive.
/// @param floor Positive finite standard-deviation floor, default1e-6.
/// @return Owning detached center/scale vectors[L] in code dtype.
/// @throws std::invalid_argument For invalid code/floor or computed/cast overflow.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto fit=n::models::FitTransferScaling(n::tensor({1.,5.,3.,5.},{2,2}));
/// // center=[2,5], scale=[1,float(1e-6)]; population, not sample deviation.
/// @endcode
/// @see TransferScaling, FrozenEncoderFeatures(), FrozenEncoderClassifier
inline TransferScaling FitTransferScaling(const Tensor& code, double floor = 1e-6) {
  if (!code.defined() || code.dim() != 2)
    throw std::invalid_argument("transfer scaling needs fitting code rows");
  aae_detail::Matrix(code, code.size(1), code.dtype(), false);
  if (!std::isfinite(floor) || floor <= 0 ||
      (code.dtype() == DType::Float32 && (!std::isfinite(float(floor)) || float(floor) == 0)))
    throw std::invalid_argument("transfer scaling floor must be positive and representable");
  const auto          rows = code.size(0), width = code.size(1);
  const auto          values = code.contiguous().to_vector<double>();
  std::vector<double> center(width), scale(width);
  for (std::int64_t j = 0; j < width; ++j) {
    double mean = 0, variance = 0;
    for (std::int64_t i = 0; i < rows; ++i)
      mean += values[i * width + j];
    mean /= double(rows);
    for (std::int64_t i = 0; i < rows; ++i) {
      const double d = values[i * width + j] - mean;
      variance += d * d;
    }
    center[j] = mean;
    scale[j]  = std::max(floor, std::sqrt(variance / double(rows)));
    if (!std::isfinite(center[j]) || !std::isfinite(scale[j]) ||
        (code.dtype() == DType::Float32 &&
         (!std::isfinite(float(center[j])) || !std::isfinite(float(scale[j])))))
      throw std::invalid_argument("transfer scaling accumulation overflowed");
  }
  return {
      cast(from_vector(center, {width}), code.dtype()),
      cast(from_vector(scale, {width}), code.dtype())};
}

/// @brief Own frozen-code, standardized-feature and linear-head endpoints.
///
/// All tensors own lifetime and retain applicable input/head paths. loss is
/// undefined for image-only inference; otherwise it is mean categorical CE.
/// Frozen encoder and scaling state never enter the caller's head store.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore head(4703);
/// n::models::FrozenEncoderClassifier
/// model(n::ones({3,2}),n::zeros({2}),n::zeros({2}),n::ones({2}),2); const auto
/// out=model.Forward(head,n::ones({4,3}));
/// // code/features[4,2], logits/probability[4,2]; loss is undefined.
/// @endcode
/// @see FrozenEncoderClassifier
struct EncoderClassifierOutput {
  Tensor code;        ///< Owning frozen-ReLU features[B,L], with a live input path.
  Tensor features;    ///< Owning(code-center)/scale[B,L], with frozen normalization.
  Tensor logits;      ///< Owning linear head scores[B,C].
  Tensor probability; ///< Owning row-softmax probabilities[B,C].
  Tensor loss;        ///< Owning mean categorical CE scalar, or undefined without labels.
};

/// @brief Copy frozen encoder/scaling state and evaluate a caller-owned linear classifier head.
///
/// Stores detached deep copies of weight[F,L],bias[L],center[L],scale[L]; exports
/// are independent copies. No optimizer, fitted head, training loop, selection
/// or RNG state is held. Caller supplies a matching floating ParameterStore and
/// owns head updates. Forward creates/reuses only linear/W[L,C] and linear/b[C],
/// ordinary Glorot-uniform/zero initialization; defaults L32/C10 give330 scalars.
/// The migrated source uses seed4703. Encoder/scaler do not receive derivatives,
/// while image and head continuous paths remain live. Const calls may share this
/// frozen object; synchronize shared mutable head stores. ReLU at zero uses the
/// core subgradient. No label or evaluation pool enters scaling implicitly.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore head(4703);
/// n::models::FrozenEncoderClassifier
/// model(n::zeros({784,32}),n::zeros({32}),n::zeros({32}),n::ones({32})); const auto
/// out=model.Forward(head,n::full({2,784},.3),n::zeros({2},n::DType::Int32)); const auto
/// gradient=n::grad(out.loss,head.parameters());
/// // Only linear/W,b derivatives are returned: two leaves/330 scalars.
/// @endcode
/// @par References
/// - Vincent et al. (2010), https://www.jmlr.org/papers/v11/vincent10a.html .
///   Uses a compact frozen-encoder linear-probe adaptation, not stacked fine-tuning.
/// @see FrozenEncoderFeatures(), FitTransferScaling(), EncoderClassifierOutput
class FrozenEncoderClassifier {
  Tensor       encoder_weight_; ///< Private detached independently owned frozen weights[F,L].
  Tensor       encoder_bias_;   ///< Private detached independently owned frozen bias[L].
  Tensor       center_;         ///< Private detached independently owned fitting means[L].
  Tensor       scale_;          ///< Private detached independently owned positive deviations[L].
  std::int64_t classes_;        ///< Positive configured number of classes.
 public:
  /// @brief Validate and copy the frozen feature/scaling tensors without creating head leaves.
  /// @param weight Finite Float32/64 frozen matrix[F,L], F,L positive.
  /// @param bias Finite matching frozen vector[L].
  /// @param center Finite matching fitting-only population means[L].
  /// @param scale Finite matching strictly positive population deviations[L].
  /// @param classes Positive class count, default10.
  /// @throws std::invalid_argument For invalid tensors/classes or head product overflow.
  /// @par Example
  /// @code
  /// #include <dm/dm.hpp>
  /// namespace n=dm::nn;
  /// n::models::FrozenEncoderClassifier
  /// model(n::ones({3,2}),n::zeros({2}),n::zeros({2}),n::ones({2}),2);
  /// // No ParameterStore or optimizer is created by construction.
  /// @endcode
  /// @see FrozenEncoderClassifier, Forward()
  FrozenEncoderClassifier(
      const Tensor& weight,
      const Tensor& bias,
      const Tensor& center,
      const Tensor& scale,
      std::int64_t  classes = 10)
      : classes_(classes) {
    if (!weight.defined() || weight.dim() != 2)
      throw std::invalid_argument("frozen classifier needs matrix encoder weights");
    aae_detail::Matrix(weight, weight.size(1), weight.dtype(), false);
    transfer_detail::Geometry(1, weight.size(0), weight.size(1), classes);
    transfer_detail::Vector(bias, weight.size(1), weight.dtype());
    transfer_detail::Vector(center, weight.size(1), weight.dtype());
    transfer_detail::Vector(scale, weight.size(1), weight.dtype(), true);
    encoder_weight_ = weight.detach().clone();
    encoder_bias_   = bias.detach().clone();
    center_         = center.detach().clone();
    scale_          = scale.detach().clone();
  }
  /// @brief Export independent frozen weights, without a gradient path to private state.
  /// @return Owning detached matrix[F,L]; modifying it leaves this model unchanged.
  Tensor encoder_weight() const { return encoder_weight_.clone(); }
  /// @brief Export an independent frozen encoder bias.
  /// @return Owning detached vector[L]; modifying it leaves this model unchanged.
  Tensor encoder_bias() const { return encoder_bias_.clone(); }
  /// @brief Export independent fitting means.
  /// @return Owning detached center[L].
  Tensor center() const { return center_.clone(); }
  /// @brief Export independent fitting deviations.
  /// @return Owning detached positive scale[L].
  Tensor scale() const { return scale_.clone(); }

  /// @brief Apply frozen features/scaling and a live head, optionally scoring supplied labels.
  ///
  /// Accepts finite signed strided matching Float32/64 image[B,F], B>0. Optional
  /// Int32/Int64[B] labels in[0,C) are validated before head leaf creation. CE is
  /// mean(-sum(one_hot(labels)*log_softmax(logits),classes)), with no clipping,
  /// smoothing, averaging over classes or implicit updates. Without labels,
  /// inference needs images only and loss is undefined. Inputs unchanged; all
  /// fields own lifetime. Head and image derivatives remain live; frozen state
  /// is stopped. Invalid inputs fail before leaves; incompatible existing leaves
  /// or computed overflow may fail later, without rollback or repair. Work
  /// O(B*F*L+B*L*C), activation storage O(B*(F+L+C)).
  /// @param head Mutable matching floating store, containing/creating linear/W,b only.
  /// @param image Finite matrix[B,F], B positive.
  /// @param labels Optional aligned integer class IDs[B], undefined by default.
  /// @return Owning five-field result; loss defined exactly when labels are supplied.
  /// @throws std::invalid_argument For invalid inputs/options or incompatible head leaves.
  /// @par Example
  /// @code
  /// #include <dm/dm.hpp>
  /// namespace n=dm::nn;
  /// n::ParameterStore head(4703);
  /// n::models::FrozenEncoderClassifier
  /// model(n::ones({3,2}),n::zeros({2}),n::zeros({2}),n::ones({2}),2); const auto
  /// prediction=model.Forward(head,n::ones({1,3}));
  /// // probability[1,2] is target-free; no head optimizer is invoked.
  /// @endcode
  /// @see FrozenEncoderClassifier, EncoderClassifierOutput
  EncoderClassifierOutput
  Forward(ParameterStore& head, const Tensor& image, const Tensor& labels = Tensor{}) const {
    aae_detail::Matrix(image, encoder_weight_.size(0), encoder_weight_.dtype(), false);
    if (head.dtype() != encoder_weight_.dtype())
      throw std::invalid_argument("classifier head and frozen state dtype must match");
    transfer_detail::Geometry(
        image.size(0), encoder_weight_.size(0), encoder_weight_.size(1), classes_);
    if (labels.defined())
      transfer_detail::Labels(labels, image.size(0), classes_);
    EncoderClassifierOutput out;
    out.code        = relu(matmul(image, encoder_weight_) + encoder_bias_);
    out.features    = (out.code - center_) / scale_;
    out.logits      = dense(head, "linear", out.features, classes_);
    out.probability = softmax(out.logits, 1);
    if (labels.defined())
      out.loss = mean(
          -sum(one_hot(labels, classes_, 1, 0, head.dtype()) * log_softmax(out.logits, 1), {1}),
          {0});
    return out;
  }
};
} // namespace datamunge::nn::models
