#pragma once

/// @file word_embeddings.hpp
/// @brief Bags of embeddings, co-occurrence counts, pointwise mutual information, and skip-gram with negative
///        sampling.
///
/// nn::embedding() (layers.hpp) looks up rows of a table. This header adds what word-embedding models build on it.
/// embedding_bag() pools the rows of variable-length bags given as flat ids plus offsets, without padding.
/// cooccurrence_matrix() and pmi() give the count-based view: (shifted, positive) pointwise mutual information,
/// whose truncated SVD is a classical embedding. negative_sampling_loss() and sample_negatives() give the predictive
/// view, word2vec's skip-gram with negative sampling (Mikolov et al. 2013). The two views meet: with enough dimensions
/// the SGNS optimum has \f$w\cdot c=\mathrm{PMI}(w,c)-\log k\f$ (Levy and Goldberg 2014).

#include "datamunge/nn/random.hpp"
#include "datamunge/nn/tensor.hpp"

#include <cstdint>
#include <vector>

namespace datamunge::nn {

/// @brief How embedding_bag() pools a bag.
enum class BagMode {
    /// Sum of the rows (weighted by per_sample_weights when given).
    Sum,
    /// Mean of the rows (weighted mean when per_sample_weights is given).
    Mean,
    /// Elementwise maximum of the rows.
    Max,
};

/// @brief Pools rows of @p table over bags given as flat ids and start offsets (PyTorch's EmbeddingBag).
///
/// Bag b holds ids[offsets[b] .. offsets[b+1]) (the last bag runs to the end). An empty bag gives zeros.
/// Differentiable with respect to the table (and the weights for Sum and Mean).
///
/// @param table  Embedding table [vocab, dim].
/// @param ids  Flat int64 ids [n].
/// @param offsets  Int64 start of each bag [B], non-decreasing, offsets[0] = 0.
/// @param mode  Sum, Mean or Max.
/// @param per_sample_weights  Optional weights [n] for Sum and Mean.
/// @return [B, dim].
/// @throws std::invalid_argument  On bad ranks or offsets, or weights with Max.
[[nodiscard]] Tensor embedding_bag(const Tensor& table, const Tensor& ids, const Tensor& offsets, BagMode mode = BagMode::Mean,
                                   const Tensor& per_sample_weights = Tensor());

/// @brief Symmetric co-occurrence counts within a window: entry (w, c) counts the positions i != j with
///        |i - j| <= window, word w at i and word c at j, summed over sentences.
/// @param sentences  Sentences as lists of ids in [0, vocab).
/// @param vocab  Vocabulary size.
/// @param window  Window radius >= 1.
/// @param harmonic  Weight each pair by 1/|i - j| (as GloVe does) instead of 1.
/// @return Float64 [vocab, vocab].
/// @throws std::invalid_argument  If window < 1 or an id is out of range.
[[nodiscard]] Tensor cooccurrence_matrix(const std::vector<std::vector<std::int64_t>>& sentences, std::int64_t vocab, std::int64_t window,
                                         bool harmonic = false);

/// @brief Pointwise mutual information of a count matrix,
///        \f$\log\frac{\#(w,c)\,\sum_{c'}\#(c')^{\alpha}}{\#(w)\,\#(c)^{\alpha}}-s\f$, with row sums \f$\#(w)\f$ and column sums \f$\#(c)\f$.
///
/// The context distribution is smoothed by the power α (word2vec uses 0.75); α = 1 gives the usual PMI. Zero counts
/// give -infinity, or 0 when @p positive (shifted positive PMI, \f$\max(\mathrm{PMI}-s,0)\f$).
///
/// @param counts  Non-negative [W, C] counts.
/// @param shift  The shift s, e.g. log k for k negative samples.
/// @param positive  Clip at zero.
/// @param context_power  The smoothing power α > 0.
/// @return Float64 [W, C].
/// @throws std::invalid_argument  If counts is not rank 2, has a negative entry, or α <= 0.
[[nodiscard]] Tensor pmi(const Tensor& counts, double shift = 0.0, bool positive = false, double context_power = 1.0);

/// @brief Skip-gram negative-sampling loss, averaged over the batch:
///        \f$-\log\sigma(t\cdot c)-\sum_{k}\log\sigma(-t\cdot n_k)\f$.
/// @param target  Target-word vectors [B, d].
/// @param context  Observed context vectors [B, d].
/// @param negatives  Negative context vectors [B, K, d].
/// @return Scalar loss, differentiable in all three inputs.
/// @throws std::invalid_argument  If the shapes do not match.
[[nodiscard]] Tensor negative_sampling_loss(const Tensor& target, const Tensor& context, const Tensor& negatives);

/// @brief Draws negative samples with probability proportional to \f$\mathrm{count}^{\alpha}\f$.
/// @param counts  Non-negative word counts [vocab].
/// @param shape  Shape of the result.
/// @param power  The smoothing power α (default 0.75).
/// @param gen  Random stream.
/// @return Int64 ids of the given shape.
/// @throws std::invalid_argument  If counts is not rank 1 or has no positive entry.
[[nodiscard]] Tensor sample_negatives(const Tensor& counts, const Shape& shape, double power, Generator& gen);

}  // namespace datamunge::nn
