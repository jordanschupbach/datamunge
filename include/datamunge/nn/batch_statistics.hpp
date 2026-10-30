#pragma once

/// @file batch_statistics.hpp
/// @brief Batch renormalization, exact population statistics for batch-norm layers ("precise BN") and
///        folding an inference-mode batch norm into the preceding linear layer.
///
/// nn::batch_norm() (layers.hpp) normalises with the batch's statistics while training and with exponential
/// moving averages at inference. Three standard companions are here. batch_renorm() corrects the training-time
/// normalisation towards the moving statistics, so that small or non-i.i.d. batches train the same function that
/// inference uses (Ioffe 2017). BatchStatsAccumulator computes the exact mean and variance of a layer's input
/// over many batches, to replace the moving averages after training (Wu and Johnson 2021). fold_batch_norm()
/// absorbs the inference-mode affine map into the weights and bias of a dense or convolution layer.
/// All three use the stats store layout of nn::batch_norm(): "<name>/moving_mean" and "<name>/moving_var".

#include "datamunge/nn/params.hpp"
#include "datamunge/nn/tensor.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace datamunge::nn {

/// @brief Options of batch_renorm().
struct BatchRenormOptions {
    /// Moving-average momentum m: stat <- m stat + (1 - m) batch_stat.
    double momentum{0.99};
    /// Added to the variance before the square root.
    double eps{1e-3};
    /// Clip of the scale correction r to \f$[1/r_{\max},r_{\max}]\f$; 1 turns the correction off.
    double r_max{3.0};
    /// Clip of the shift correction d to \f$[-d_{\max},d_{\max}]\f$; 0 turns the correction off.
    double d_max{5.0};
    /// Update the moving averages on a training-mode call.
    bool update_stats{true};
};

/// @brief Batch renormalization (Ioffe 2017) over every axis but the last.
///
/// While training, with batch mean \f$\mu_B\f$, biased batch variance \f$\sigma_B^2\f$, \f$s_B=\sqrt{\sigma_B^2+\varepsilon}\f$
/// and moving statistics \f$\mu,\ s=\sqrt{\text{var}+\varepsilon}\f$,
/// \f[ \hat x=\frac{x-\mu_B}{s_B}\,r+d,\qquad r=\operatorname{clip}\Big(\frac{s_B}{s}\Big),\quad d=\operatorname{clip}\Big(\frac{\mu_B-\mu}{s}\Big), \f]
/// with r and d treated as constants by autograd, and \f$y=\gamma\hat x+\beta\f$. Unclipped, \f$\hat x=(x-\mu)/s\f$: the
/// inference-mode output, with the batch statistics still in the gradient. With r_max = 1 and d_max = 0 it is
/// nn::batch_norm(). The moving averages are updated as in nn::batch_norm() (unbiased batch variance), after the
/// corrections are computed. Inference uses the moving statistics. Parameters "<name>/gamma" and "<name>/beta" live
/// in @p params, statistics in @p stats.
///
/// @param params  Store of the learned scale and shift.
/// @param stats  Store of the moving statistics.
/// @param name  Layer name.
/// @param x  Input [..., C]; statistics are taken over every axis but the last.
/// @param training  Normalise with the corrected batch statistics (and update them) or with the moving ones.
/// @param options  Momentum, epsilon and clips.
/// @return The normalised output, same shape as x.
/// @throws std::invalid_argument  If x is a scalar, r_max < 1 or d_max < 0.
[[nodiscard]] Tensor batch_renorm(ParameterStore& params, ParameterStore& stats, const std::string& name, const Tensor& x,
                                  bool training, const BatchRenormOptions& options = {});

/// @brief Exact per-channel mean and variance over any number of batches, merged with Chan et al.'s pairwise
///        update in double precision.
///
/// Feed it a layer's inputs batch by batch (in inference mode, with the trained weights fixed), then write() the
/// result into the stats store in place of the moving averages ("precise BN"). Each add() merges the batch's count,
/// mean and sum of squared deviations, so the result equals the statistics of all rows concatenated.
class BatchStatsAccumulator {
  public:
    /// @brief Adds a batch [..., C]; statistics run over every axis but the last.
    /// @param x  The batch; C must match earlier batches.
    /// @throws std::invalid_argument  If x is a scalar or its channel count changed.
    void add(const Tensor& x);
    /// @brief Rows seen so far.
    /// @return The count.
    [[nodiscard]] std::int64_t count() const noexcept { return count_; }
    /// @brief Per-channel mean of every row seen.
    /// @return [C] float64 tensor.
    /// @throws std::logic_error  Before the first add().
    [[nodiscard]] Tensor mean() const;
    /// @brief Per-channel variance of every row seen.
    /// @param unbiased  Divide by count - 1 (default, matching nn::batch_norm's moving variance) or by count.
    /// @return [C] float64 tensor.
    /// @throws std::logic_error  Before the first add(), or with one row and unbiased.
    [[nodiscard]] Tensor variance(bool unbiased = true) const;
    /// @brief Stores mean() and variance() as "<name>/moving_mean" and "<name>/moving_var" in @p stats, in its dtype.
    /// @param stats  The batch-norm stats store.
    /// @param name  Layer name.
    void write(ParameterStore& stats, const std::string& name) const;
    /// @brief Forgets every batch.
    void reset();

  private:
    /// Rows merged so far.
    std::int64_t count_{0};
    /// Per-channel running mean.
    std::vector<double> mean_;
    /// Per-channel sum of squared deviations from the running mean.
    std::vector<double> m2_;
};

/// @brief A linear layer's weight and bias after folding a batch norm into it.
struct FoldedLinear {
    /// Folded weight, same shape as the original.
    Tensor weight;
    /// Folded bias [C].
    Tensor bias;
};

/// @brief Folds an inference-mode batch norm that follows a linear layer into that layer:
///        \f$\mathrm{BN}(xW+b)=xW'+b'\f$ with \f$W'=W\operatorname{diag}(s)\f$, \f$b'=(b-\mu)\odot s+\beta\f$, \f$s=\gamma/\sqrt{\text{var}+\varepsilon}\f$.
///
/// Works for any weight whose last axis is the output channel: dense [in, C] and convolution [kh, kw, in, C] kernels.
///
/// @param weight  Weight [..., C].
/// @param bias  Bias [C], or undefined for none.
/// @param gamma  Batch-norm scale [C].
/// @param beta  Batch-norm shift [C].
/// @param mean  Moving mean [C].
/// @param variance  Moving variance [C].
/// @param eps  The batch norm's epsilon (default 1e-3).
/// @return The folded weight and bias.
/// @throws std::invalid_argument  If a [C] vector does not match the weight's last axis.
[[nodiscard]] FoldedLinear fold_batch_norm(const Tensor& weight, const Tensor& bias, const Tensor& gamma, const Tensor& beta,
                                           const Tensor& mean, const Tensor& variance, double eps = 1e-3);

}  // namespace datamunge::nn
