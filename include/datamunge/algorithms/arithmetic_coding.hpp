#pragma once

/// \file arithmetic_coding.hpp
/// \brief Arithmetic coding: near-optimal entropy coding to a fractional bit budget.
///
/// Arithmetic coding represents an entire message as a single number in \f$[0,1)\f$. Starting
/// from the full interval, each symbol *narrows* it to the sub-interval whose width is the
/// symbol's probability; the final interval identifies the message, and any number inside it
/// (written in as few bits as possible) is the code. Unlike Huffman coding -- which must spend a
/// whole bit per symbol -- arithmetic coding approaches the Shannon entropy arbitrarily closely,
/// spending *fractional* bits per symbol, which matters most for skewed alphabets.
///
/// The module separates the *coder* from the *model*. ArithmeticEncoder and ArithmeticDecoder
/// are the classic 32-bit integer coder with carry-free renormalization (E1/E2/E3 scaling and
/// the bit-plus-follow rule); they accept any cumulative-frequency slice. The models are a
/// static frequency table (arithmetic_encode()), an adaptive count model that learns as it
/// codes (AdaptiveFrequencyModel, arithmetic_encode_adaptive()), and a context model that keeps
/// one adaptive table per context chosen from the already-coded history.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// \brief Incremental 32-bit integer arithmetic encoder.
///
/// Each call to encode() narrows the current interval \f$[\text{low},\text{high}]\f$ to the
/// slice \f$[c_\ell/T,\,c_h/T)\f$ of it and emits every bit that is already settled. finish()
/// flushes the two bits needed to pin a number inside the final interval. A message of
/// probability \f$P\f$ costs at most \f$\lceil -\log_2 P\rceil + 2\f$ bits, apart from the
/// integer rounding of the slices.
class ArithmeticEncoder {
public:
    /// \brief Narrows the interval to the slice [cum_low, cum_high) of a total of \p total.
    /// \param cum_low   cumulative count of all symbols before the coded one.
    /// \param cum_high  cum_low plus the coded symbol's count; must exceed cum_low.
    /// \param total     total count; at most 2^30 so every slice keeps a non-empty interval.
    /// \throws std::invalid_argument if the slice is empty or outside [0, total].
    void encode(std::uint64_t cum_low, std::uint64_t cum_high, std::uint64_t total);

    /// \brief Flushes the final interval and returns the complete bit stream.
    ///
    /// The encoder must not be used again after finish().
    std::vector<bool> finish();

    /// \brief Number of bits committed so far, counting the deferred (pending) bits.
    [[nodiscard]] std::size_t bits_committed() const { return out_.size() + pending_; }

private:
    /// \brief Appends \p bit followed by the deferred opposite bits.
    void emit(int bit);

    std::uint64_t     low_     = 0;            ///< Lower end of the current interval.
    std::uint64_t     high_    = 0xFFFFFFFFull;  ///< Upper end of the current interval (inclusive).
    std::size_t       pending_ = 0;            ///< Bits deferred by middle-straddling (E3) scaling.
    std::vector<bool> out_;                    ///< Bits emitted so far.
};

/// \brief Incremental decoder matching ArithmeticEncoder.
///
/// For each symbol call target() with the model's total, look up the symbol whose slice
/// contains the returned count, then call consume() with that slice.
class ArithmeticDecoder {
public:
    /// \brief Starts decoding \p bits (missing trailing bits read as zero).
    explicit ArithmeticDecoder(std::vector<bool> bits);

    /// \brief Returns the count in [0, total) that the next symbol's slice must contain.
    [[nodiscard]] std::uint64_t target(std::uint64_t total) const;

    /// \brief Removes the decoded symbol's slice [cum_low, cum_high) and renormalizes.
    void consume(std::uint64_t cum_low, std::uint64_t cum_high, std::uint64_t total);

private:
    /// \brief Returns the next input bit, or 0 past the end.
    std::uint64_t read_bit();

    std::vector<bool> bits_;                     ///< The coded bit stream.
    std::size_t       pos_   = 0;                ///< Index of the next unread bit.
    std::uint64_t     low_   = 0;                ///< Lower end of the current interval.
    std::uint64_t     high_  = 0xFFFFFFFFull;    ///< Upper end of the current interval (inclusive).
    std::uint64_t     value_ = 0;                ///< The 32 bits of the code currently in view.
};

/// \brief Adaptive frequency model: counts start at one and grow as symbols are coded.
///
/// Before the \f$(N+1)\f$-th symbol the model assigns symbol \f$s\f$ the probability
/// \f$(n_s+1)/(N+K)\f$, Laplace's rule of succession, where \f$n_s\f$ counts earlier
/// occurrences and \f$K\f$ is the alphabet size. Encoder and decoder update identically, so no
/// table is transmitted. When the total would exceed \p max_total all counts are halved
/// (rounding up), which keeps the coder exact and lets the model forget old statistics.
class AdaptiveFrequencyModel {
public:
    /// \brief Creates a model over \p alphabet_size symbols, each with count one.
    /// \param alphabet_size  number of symbols K (at least 1).
    /// \param max_total      rescaling threshold; at most 2^30.
    /// \throws std::invalid_argument if the alphabet is empty or max_total is out of range.
    explicit AdaptiveFrequencyModel(std::size_t alphabet_size, std::uint64_t max_total = 1u << 24);

    /// \brief Cumulative count of symbols below \p symbol.
    [[nodiscard]] std::uint64_t cum_low(int symbol) const;
    /// \brief Cumulative count of symbols up to and including \p symbol.
    [[nodiscard]] std::uint64_t cum_high(int symbol) const;
    /// \brief Sum of all counts.
    [[nodiscard]] std::uint64_t total() const { return total_; }
    /// \brief Current probability (count / total) the model gives \p symbol.
    [[nodiscard]] double probability(int symbol) const;
    /// \brief The symbol whose slice contains cumulative count \p target.
    [[nodiscard]] int find(std::uint64_t target) const;
    /// \brief Records one occurrence of \p symbol (and rescales if needed).
    void update(int symbol);
    /// \brief Number of symbols in the alphabet.
    [[nodiscard]] std::size_t alphabet_size() const { return counts_.size(); }

private:
    std::vector<std::uint64_t> counts_;     ///< Count of each symbol (all at least one).
    std::uint64_t              total_;      ///< Sum of counts_.
    std::uint64_t              max_total_;  ///< Halve all counts when total_ exceeds this.
};

/// \brief Chooses the context (model index) for position \p history.size() from the symbols
///        already coded. The decoder calls it with the symbols decoded so far.
using ArithmeticContextFunction = std::function<int(const std::vector<int>& history)>;

/// \brief Encode a symbol sequence under a static frequency model to a bit stream.
///
/// \param symbols  symbol indices (0..freq.size()-1).
/// \param freq     frequency (weight) of each symbol; the total must be at most 2^30.
/// \throws std::invalid_argument for an out-of-range symbol or a symbol with zero frequency.
std::vector<bool> arithmetic_encode(const std::vector<int>& symbols, const std::vector<std::uint32_t>& freq);

/// \brief Decode \p count symbols from a bit stream under the same static frequency model.
std::vector<int> arithmetic_decode(const std::vector<bool>& bits, const std::vector<std::uint32_t>& freq,
                                   int count);

/// \brief Encode with adaptive (Laplace) models, one per context.
///
/// With the default arguments a single adaptive model codes every symbol. Otherwise
/// \p context returns, for each position, which of the \p num_contexts models to use.
/// \param symbols        symbol indices in [0, alphabet_size).
/// \param alphabet_size  number of symbols K.
/// \param context        context chooser, or empty for a single model.
/// \param num_contexts   number of models; context() must return values in [0, num_contexts).
/// \throws std::invalid_argument for an out-of-range symbol or context.
std::vector<bool> arithmetic_encode_adaptive(const std::vector<int>& symbols, std::size_t alphabet_size,
                                             const ArithmeticContextFunction& context = {},
                                             std::size_t num_contexts = 1);

/// \brief Decode \p count symbols coded by arithmetic_encode_adaptive() with the same model.
std::vector<int> arithmetic_decode_adaptive(const std::vector<bool>& bits, std::size_t alphabet_size, int count,
                                            const ArithmeticContextFunction& context = {},
                                            std::size_t num_contexts = 1);

/// \brief Bits committed by the encoder after each symbol of a static-model encoding.
///
/// Entry i is ArithmeticEncoder::bits_committed() after symbol i, so it can be compared with
/// the cumulative information content \f$-\sum_{j\le i}\log_2 p_{s_j}\f$.
std::vector<std::size_t> arithmetic_bits_trace(const std::vector<int>& symbols, const std::vector<std::uint32_t>& freq);

/// \brief The real interval [low, high) after each symbol, in double precision and without
///        renormalization, so it is useful for short messages only.
std::vector<std::pair<double, double>> arithmetic_intervals(const std::vector<int>& symbols,
                                                            const std::vector<std::uint32_t>& freq);

/// \brief Information content \f$-\sum_i \log_2 p_{s_i}\f$ of a message under a static model,
///        in bits: the length an ideal coder would reach.
double ideal_code_length(const std::vector<int>& symbols, const std::vector<std::uint32_t>& freq);

/// \brief Counts of each symbol in [0, alphabet_size).
std::vector<std::uint64_t> symbol_counts(const std::vector<int>& symbols, std::size_t alphabet_size);

/// \brief Shannon entropy \f$-\sum_s p_s\log_2 p_s\f$ (bits per symbol) of a count vector.
double shannon_entropy_bits(const std::vector<std::uint64_t>& counts);

/// \brief Exact ideal length (bits) of coding a message with the adaptive Laplace model.
///
/// The product of the sequential probabilities depends only on the final counts:
/// \f$P=\frac{(K-1)!\,\prod_s n_s!}{(N+K-1)!}\f$, so the length is
/// \f$\log_2\binom{N+K-1}{K-1}+\log_2\frac{N!}{\prod_s n_s!}\f$.
double adaptive_code_length(const std::vector<std::uint64_t>& counts);

/// \brief Asymptotic redundancy (bits) of the adaptive Laplace model over the empirical entropy.
///
/// Stirling's formula applied to adaptive_code_length() gives
/// \f$\text{len}-N\hat H\approx\frac{K-1}{2}\log_2\frac{N}{2\pi}-\frac12\sum_s\log_2\hat p_s
/// -\log_2 (K-1)!\f$ with \f$\hat p_s=n_s/N\f$. All counts must be positive.
/// \throws std::invalid_argument if a count is zero.
double adaptive_redundancy_approx(const std::vector<std::uint64_t>& counts);

/// \brief Symbol counts within each context, as the context model would see them.
///
/// Row c holds the counts of the symbols coded while \p context returned c. An empty
/// \p context puts every symbol in row 0.
std::vector<std::vector<std::uint64_t>> context_counts(const std::vector<int>& symbols, std::size_t alphabet_size,
                                                       const ArithmeticContextFunction& context,
                                                       std::size_t num_contexts);

}  // namespace datamunge::algorithms
