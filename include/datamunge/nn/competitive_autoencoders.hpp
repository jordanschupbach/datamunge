#pragma once
/// @file competitive_autoencoders.hpp
/// @brief Hard signed-activity competition and differentiable autoencoder forward components.

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace datamunge::nn::models {

/// @brief Select largest signed activities per observation or lifetime winners per feature.
///
/// For lifetime=false, each row keeps min(k,F) features of scores[B,F]. For
/// lifetime=true, each column keeps ceil(B/8) observations; k is ignored after
/// validation. Ranking uses signed values, not magnitude; equal values select
/// lower column/row indices respectively. Selected zero values still count as
/// selected slots. Empty axes and strided inputs are supported. Only finite
/// Float32/64 matrices are accepted, avoiding undefined NaN ranking semantics.
/// Returns detached binary values of the same dtype/shape, without changing
/// scores. Differentiating scores*mask passes gradients at winners only; this
/// does not differentiate the discrete ranking or define a derivative at ties.
/// Uses stable sorting: O(B*F*log(F)) per-row or O(F*B*log(B)) lifetime work and
/// O(B*F+max(B,F)) auxiliary storage. No mutable shared state is used.
///
/// @param scores Finite floating activation matrix [B,F]; it need not be detached.
/// @param lifetime Whether to compete across observations in each feature.
/// @param k Nonnegative requested per-row count, default 8; larger counts saturate at F.
/// @return Newly owned contiguous binary selector [B,F], without autograd history.
/// @throws std::invalid_argument For undefined/nonmatrix/nonfloating/nonfinite scores or k<0.
///
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n = dm::nn;
/// const auto scores = n::tensor({-4.,-1.,-2.,-3.},{1,4});
/// const auto mask = n::models::CompetitiveMask(scores,false,2);
/// // mask is [[0,1,1,0]]; winners are -1 and -2, not largest magnitudes.
/// @endcode
///
/// @par References
/// - Makhzani and Frey (2013/2014), k-Sparse Autoencoders, https://arxiv.org/abs/1312.5663 .
/// - Makhzani and Frey (2014/2015), Winner-Take-All Autoencoders,
///   https://arxiv.org/abs/1409.2752 . This lifetime variant fixes retention at one eighth.
/// @see CompetitiveAutoencoder(), top_k()
inline Tensor CompetitiveMask(const Tensor& scores, bool lifetime, std::int64_t k = 8) {
    if (!scores.defined() || scores.dim() != 2 || k < 0 ||
        (scores.dtype() != DType::Float32 && scores.dtype() != DType::Float64))
        throw std::invalid_argument("competitive mask requires floating [B,F] and nonnegative k");
    const auto oriented = lifetime ? transpose(scores) : scores;
    const auto rows = oriented.size(0), columns = oriented.size(1);
    const auto values = oriented.contiguous().to_vector<double>();
    for (double value : values)
        if (!std::isfinite(value)) throw std::invalid_argument("competitive scores must be finite");
    std::vector<double> selected(values.size(), 0);
    std::vector<std::int64_t> order(static_cast<std::size_t>(columns));
    const auto keep = lifetime ? columns / 8 + (columns % 8 != 0) : std::min(k, columns);
    for (std::int64_t row = 0; row < rows; ++row) {
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](std::int64_t a, std::int64_t b) {
            return values[row * columns + a] > values[row * columns + b];
        });
        for (std::int64_t i = 0; i < keep; ++i) selected[row * columns + order[i]] = 1;
    }
    const auto result = cast(from_vector(selected, oriented.shape()), scores.dtype());
    return lifetime ? transpose(result).contiguous() : result;
}

/// @brief Own training and deployment endpoints of a hard-competition autoencoder.
///
/// All handles retain autograd history except the detached binary mask. The
/// parameter store owns learned leaves; this value owns no optimizer or fit.
///
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n = dm::nn;
/// n::ParameterStore p(1931);
/// const auto net = n::models::CompetitiveAutoencoder(p,n::zeros({2,784}),false);
/// // code is [2,64]; training_reconstruction and reconstruction are the same path.
/// @endcode
/// @see CompetitiveAutoencoder()
struct CompetitiveAeOutput {
    Tensor raw;                     ///< [B,latent] linear activities (per-row) or ReLU activities (lifetime).
    Tensor mask;                    ///< Detached [B,latent] binary winners, or all ones for plain=true.
    Tensor training_code;           ///< raw*mask, retaining winner-only activation gradients.
    Tensor code;                    ///< Deployment code: training_code per-row, raw for lifetime.
    Tensor training_reconstruction; ///< Sigmoid decoder of training_code, [B,features].
    Tensor reconstruction;          ///< Sigmoid decoder of deployment code, [B,features].
};

/// @brief Evaluate signed k-sparse or lifetime winner-take-all encoder/decoder paths without fitting.
///
/// Creates/reuses encoder/W,b and decoder/W,b with independent Glorot-uniform
/// weights and zero biases. Parameter count is 2*features*latent+latent+features
/// (101200 by default). Per-row competition uses a linear encoder and preserves
/// its hard mask at deployment. Lifetime competition uses ReLU, keeps ceil(B/8)
/// examples per feature during training and bypasses the gate at deployment.
/// plain=true bypasses either competition but retains the corresponding encoder
/// activation, parameter layout and both result paths. No target, optimizer,
/// training loop or global training-mode switch is hidden here. Losses are the
/// caller's choice. Supports strided inputs, input/parameter gradients and higher
/// derivatives on fixed support. Discrete support changes/ties have no promised
/// derivative. Input is not clipped or mutated. Nonfinite raw activities cause
/// CompetitiveMask() to throw unless plain=true, where they propagate normally.
/// Invalid input geometry/dtype/options are rejected before parameter creation;
/// incompatible existing leaves/nonfinite activities may fail after creation.
/// Shared stores require external synchronization. Dense work is O(B*features*latent),
/// plus the selector's sorting cost; training/deployment share decoder leaves.
///
/// @param parameters Mutable Float32/64 store matching input's dtype.
/// @param input Nonempty floating matrix [B,features].
/// @param lifetime Select lifetime ReLU competition rather than per-row linear competition.
/// @param plain Use the matched ungated control, default false.
/// @param features Positive input/output width, default 784.
/// @param latent Positive code width, default 64; strict undercompleteness is not required.
/// @param k Nonnegative per-row retention count, default 8; ignored by lifetime/plain paths.
/// @return Owning training/deployment endpoints with differentiable continuous paths.
/// @throws std::invalid_argument For invalid input/options, incompatible stored leaves or nonfinite ranked activities.
///
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n = dm::nn;
/// n::ParameterStore p(1931);
/// const auto x = n::full({2,3},.25);
/// const auto net = n::models::CompetitiveAutoencoder(p,x,false,false,3,4,2);
/// const auto loss = n::mse(net.training_reconstruction,x);
/// // The caller owns Adam; 31 learned scalars and two selected slots per row.
/// @endcode
///
/// @par References
/// - Makhzani and Frey (2013/2014), k-Sparse Autoencoders, https://arxiv.org/abs/1312.5663 .
/// - Makhzani and Frey (2014/2015), Winner-Take-All Autoencoders,
///   https://arxiv.org/abs/1409.2752 . These small untied sigmoid/MSE models on raw
///   pixels are educational variants, not the original decoder, scale or classification experiments.
/// @see CompetitiveAeOutput, CompetitiveMask(), mse()
inline CompetitiveAeOutput CompetitiveAutoencoder(ParameterStore& parameters, const Tensor& input, bool lifetime,
                                                   bool plain = false, std::int64_t features = 784,
                                                   std::int64_t latent = 64, std::int64_t k = 8) {
    if (!input.defined() || input.dim() != 2 || input.size(0) <= 0 || input.size(1) != features ||
        features <= 0 || latent <= 0 || k < 0 ||
        (input.dtype() != DType::Float32 && input.dtype() != DType::Float64) || parameters.dtype() != input.dtype())
        throw std::invalid_argument("competitive autoencoder requires matching floating [B,features], positive widths and nonnegative k");
    CompetitiveAeOutput out;
    out.raw = dense(parameters, "encoder", input, latent, lifetime ? Activation::Relu : Activation::None);
    out.mask = plain ? ones_like(out.raw) : CompetitiveMask(out.raw, lifetime, k);
    out.training_code = out.raw * out.mask;
    out.code = lifetime ? out.raw : out.training_code;
    out.training_reconstruction = dense(parameters, "decoder", out.training_code, features, Activation::Sigmoid);
    out.reconstruction = lifetime ? dense(parameters, "decoder", out.code, features, Activation::Sigmoid)
                                  : out.training_reconstruction;
    return out;
}

}  // namespace datamunge::nn::models
