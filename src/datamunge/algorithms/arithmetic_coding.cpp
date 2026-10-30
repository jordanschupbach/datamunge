/// @file arithmetic_coding.cpp
/// @brief Integer arithmetic coder, adaptive frequency model and code-length helpers.

#include <datamunge/algorithms/arithmetic_coding.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

namespace datamunge::algorithms {

/// @brief Constants and helpers shared by the coder and the models.
namespace {

constexpr std::uint64_t kMask      = 0xFFFFFFFFull;  ///< 32-bit register mask.
constexpr std::uint64_t kHalf      = 0x80000000ull;  ///< One half of the register range.
constexpr std::uint64_t kQuarter   = 0x40000000ull;  ///< One quarter of the register range.
constexpr std::uint64_t kThreeQ    = 0xC0000000ull;  ///< Three quarters of the register range.
constexpr std::uint64_t kMaxTotal  = 1ull << 30;     ///< Largest model total the coder accepts.

/// @brief Prefix sums of @p freq (size freq.size()+1, starting at zero).
std::vector<std::uint64_t> cumulative(const std::vector<std::uint32_t>& freq) {
    std::vector<std::uint64_t> cum(freq.size() + 1, 0);
    for (std::size_t i = 0; i < freq.size(); ++i) cum[i + 1] = cum[i] + freq[i];
    if (cum.back() == 0 || cum.back() > kMaxTotal)
        throw std::invalid_argument("arithmetic coding: frequency total must be in [1, 2^30]");
    return cum;
}

/// @brief Throws unless @p s is a symbol with non-zero frequency.
void check_symbol(int s, const std::vector<std::uint64_t>& cum) {
    if (s < 0 || static_cast<std::size_t>(s) + 1 >= cum.size() || cum[s + 1] == cum[s])
        throw std::invalid_argument("arithmetic coding: symbol " + std::to_string(s) +
                                    " is out of range or has zero frequency");
}

/// @brief Calls @p context for the next position and checks the result.
std::size_t pick_context(const ArithmeticContextFunction& context, const std::vector<int>& history,
                         std::size_t num_contexts) {
    if (!context) return 0;
    const int c = context(history);
    if (c < 0 || static_cast<std::size_t>(c) >= num_contexts)
        throw std::invalid_argument("arithmetic coding: context " + std::to_string(c) + " out of range");
    return static_cast<std::size_t>(c);
}

/// @brief One adaptive model per context.
std::vector<AdaptiveFrequencyModel> make_models(std::size_t alphabet_size, std::size_t num_contexts) {
    if (num_contexts == 0) throw std::invalid_argument("arithmetic coding: need at least one context");
    return std::vector<AdaptiveFrequencyModel>(num_contexts, AdaptiveFrequencyModel(alphabet_size));
}

}  // namespace

// ---- ArithmeticEncoder -----------------------------------------------------

void ArithmeticEncoder::emit(int bit) {
    out_.push_back(bit != 0);
    for (; pending_ > 0; --pending_) out_.push_back(bit == 0);  // bit-plus-follow
}

void ArithmeticEncoder::encode(std::uint64_t cum_low, std::uint64_t cum_high, std::uint64_t total) {
    if (cum_low >= cum_high || cum_high > total || total > kMaxTotal)
        throw std::invalid_argument("ArithmeticEncoder::encode: empty or invalid slice");
    const std::uint64_t range = high_ - low_ + 1;
    high_ = low_ + range * cum_high / total - 1;
    low_  = low_ + range * cum_low / total;
    for (;;) {
        if (high_ < kHalf) {
            emit(0);
        } else if (low_ >= kHalf) {
            emit(1);
            low_ -= kHalf;
            high_ -= kHalf;
        } else if (low_ >= kQuarter && high_ < kThreeQ) {
            ++pending_;
            low_ -= kQuarter;
            high_ -= kQuarter;
        } else {
            break;
        }
        low_  = (low_ << 1) & kMask;
        high_ = ((high_ << 1) | 1) & kMask;
    }
}

std::vector<bool> ArithmeticEncoder::finish() {
    ++pending_;  // one more bit disambiguates the final interval
    emit(low_ < kQuarter ? 0 : 1);
    return std::move(out_);
}

// ---- ArithmeticDecoder -----------------------------------------------------

ArithmeticDecoder::ArithmeticDecoder(std::vector<bool> bits) : bits_(std::move(bits)) {
    for (int i = 0; i < 32; ++i) value_ = (value_ << 1) | read_bit();
}

std::uint64_t ArithmeticDecoder::read_bit() { return pos_ < bits_.size() ? (bits_[pos_++] ? 1 : 0) : 0; }

std::uint64_t ArithmeticDecoder::target(std::uint64_t total) const {
    const std::uint64_t range = high_ - low_ + 1;
    return ((value_ - low_ + 1) * total - 1) / range;
}

void ArithmeticDecoder::consume(std::uint64_t cum_low, std::uint64_t cum_high, std::uint64_t total) {
    const std::uint64_t range = high_ - low_ + 1;
    high_ = low_ + range * cum_high / total - 1;
    low_  = low_ + range * cum_low / total;
    for (;;) {
        if (high_ < kHalf) {
            // settled in the lower half: nothing to subtract
        } else if (low_ >= kHalf) {
            low_ -= kHalf;
            high_ -= kHalf;
            value_ -= kHalf;
        } else if (low_ >= kQuarter && high_ < kThreeQ) {
            low_ -= kQuarter;
            high_ -= kQuarter;
            value_ -= kQuarter;
        } else {
            break;
        }
        low_   = (low_ << 1) & kMask;
        high_  = ((high_ << 1) | 1) & kMask;
        value_ = ((value_ << 1) | read_bit()) & kMask;
    }
}

// ---- AdaptiveFrequencyModel ------------------------------------------------

AdaptiveFrequencyModel::AdaptiveFrequencyModel(std::size_t alphabet_size, std::uint64_t max_total)
    : counts_(alphabet_size, 1), total_(alphabet_size), max_total_(max_total) {
    if (alphabet_size == 0) throw std::invalid_argument("AdaptiveFrequencyModel: empty alphabet");
    if (max_total < 2 * alphabet_size || max_total > kMaxTotal)
        throw std::invalid_argument("AdaptiveFrequencyModel: max_total must be in [2K, 2^30]");
}

std::uint64_t AdaptiveFrequencyModel::cum_low(int symbol) const {
    std::uint64_t c = 0;
    for (int i = 0; i < symbol; ++i) c += counts_[i];
    return c;
}

std::uint64_t AdaptiveFrequencyModel::cum_high(int symbol) const { return cum_low(symbol) + counts_.at(symbol); }

double AdaptiveFrequencyModel::probability(int symbol) const {
    return static_cast<double>(counts_.at(symbol)) / static_cast<double>(total_);
}

int AdaptiveFrequencyModel::find(std::uint64_t target) const {
    std::uint64_t c = 0;
    for (std::size_t s = 0; s < counts_.size(); ++s) {
        c += counts_[s];
        if (target < c) return static_cast<int>(s);
    }
    return static_cast<int>(counts_.size()) - 1;
}

void AdaptiveFrequencyModel::update(int symbol) {
    ++counts_.at(symbol);
    ++total_;
    if (total_ > max_total_) {
        total_ = 0;
        for (auto& c : counts_) {
            c = (c + 1) / 2;
            total_ += c;
        }
    }
}

// ---- Static model ----------------------------------------------------------

std::vector<bool> arithmetic_encode(const std::vector<int>& symbols, const std::vector<std::uint32_t>& freq) {
    const auto        cum = cumulative(freq);
    ArithmeticEncoder enc;
    for (int s : symbols) {
        check_symbol(s, cum);
        enc.encode(cum[s], cum[s + 1], cum.back());
    }
    return enc.finish();
}

std::vector<int> arithmetic_decode(const std::vector<bool>& bits, const std::vector<std::uint32_t>& freq,
                                   int count) {
    const auto        cum = cumulative(freq);
    ArithmeticDecoder dec(bits);
    std::vector<int>  out;
    out.reserve(count > 0 ? static_cast<std::size_t>(count) : 0);
    for (int c = 0; c < count; ++c) {
        const std::uint64_t t = dec.target(cum.back());
        int                 s = 0;
        while (s + 2 < static_cast<int>(cum.size()) && cum[s + 1] <= t) ++s;
        out.push_back(s);
        dec.consume(cum[s], cum[s + 1], cum.back());
    }
    return out;
}

std::vector<std::size_t> arithmetic_bits_trace(const std::vector<int>& symbols,
                                               const std::vector<std::uint32_t>& freq) {
    const auto               cum = cumulative(freq);
    ArithmeticEncoder        enc;
    std::vector<std::size_t> trace;
    trace.reserve(symbols.size());
    for (int s : symbols) {
        check_symbol(s, cum);
        enc.encode(cum[s], cum[s + 1], cum.back());
        trace.push_back(enc.bits_committed());
    }
    return trace;
}

std::vector<std::pair<double, double>> arithmetic_intervals(const std::vector<int>& symbols,
                                                            const std::vector<std::uint32_t>& freq) {
    const auto                             cum   = cumulative(freq);
    const double                           total = static_cast<double>(cum.back());
    double                                 low = 0.0, width = 1.0;
    std::vector<std::pair<double, double>> out;
    out.reserve(symbols.size());
    for (int s : symbols) {
        check_symbol(s, cum);
        low += width * static_cast<double>(cum[s]) / total;
        width *= static_cast<double>(freq[s]) / total;
        out.emplace_back(low, low + width);
    }
    return out;
}

double ideal_code_length(const std::vector<int>& symbols, const std::vector<std::uint32_t>& freq) {
    const auto   cum   = cumulative(freq);
    const double total = static_cast<double>(cum.back());
    double       bits  = 0.0;
    for (int s : symbols) {
        check_symbol(s, cum);
        bits -= std::log2(static_cast<double>(freq[s]) / total);
    }
    return bits;
}

// ---- Adaptive and context models -------------------------------------------

std::vector<bool> arithmetic_encode_adaptive(const std::vector<int>& symbols, std::size_t alphabet_size,
                                             const ArithmeticContextFunction& context, std::size_t num_contexts) {
    auto              models = make_models(alphabet_size, num_contexts);
    ArithmeticEncoder enc;
    std::vector<int>  history;
    history.reserve(symbols.size());
    for (int s : symbols) {
        if (s < 0 || static_cast<std::size_t>(s) >= alphabet_size)
            throw std::invalid_argument("arithmetic_encode_adaptive: symbol out of range");
        auto& m = models[pick_context(context, history, num_contexts)];
        enc.encode(m.cum_low(s), m.cum_high(s), m.total());
        m.update(s);
        history.push_back(s);
    }
    return enc.finish();
}

std::vector<int> arithmetic_decode_adaptive(const std::vector<bool>& bits, std::size_t alphabet_size, int count,
                                            const ArithmeticContextFunction& context, std::size_t num_contexts) {
    auto              models = make_models(alphabet_size, num_contexts);
    ArithmeticDecoder dec(bits);
    std::vector<int>  history;
    history.reserve(count > 0 ? static_cast<std::size_t>(count) : 0);
    for (int c = 0; c < count; ++c) {
        auto&     m = models[pick_context(context, history, num_contexts)];
        const int s = m.find(dec.target(m.total()));
        dec.consume(m.cum_low(s), m.cum_high(s), m.total());
        m.update(s);
        history.push_back(s);
    }
    return history;
}

// ---- Code-length helpers ---------------------------------------------------

std::vector<std::uint64_t> symbol_counts(const std::vector<int>& symbols, std::size_t alphabet_size) {
    std::vector<std::uint64_t> counts(alphabet_size, 0);
    for (int s : symbols) {
        if (s < 0 || static_cast<std::size_t>(s) >= alphabet_size)
            throw std::invalid_argument("symbol_counts: symbol out of range");
        ++counts[s];
    }
    return counts;
}

double shannon_entropy_bits(const std::vector<std::uint64_t>& counts) {
    double n = 0.0;
    for (auto c : counts) n += static_cast<double>(c);
    if (n == 0.0) return 0.0;
    double h = 0.0;
    for (auto c : counts)
        if (c > 0) {
            const double p = static_cast<double>(c) / n;
            h -= p * std::log2(p);
        }
    return h;
}

double adaptive_code_length(const std::vector<std::uint64_t>& counts) {
    if (counts.empty()) throw std::invalid_argument("adaptive_code_length: empty alphabet");
    const double k = static_cast<double>(counts.size());
    double       n = 0.0, log_p = std::lgamma(k);  // ln (K-1)!
    for (auto c : counts) {
        n += static_cast<double>(c);
        log_p += std::lgamma(static_cast<double>(c) + 1.0);
    }
    log_p -= std::lgamma(n + k);  // ln (N+K-1)!
    return -log_p / std::log(2.0);
}

/// Stirling approximation of adaptive_code_length() minus the empirical code length.
double adaptive_redundancy_approx(const std::vector<std::uint64_t>& counts) {
    if (counts.empty()) throw std::invalid_argument("adaptive_redundancy_approx: empty alphabet");
    const double pi = std::acos(-1.0);
    const double k  = static_cast<double>(counts.size());
    double       n  = 0.0;
    for (auto c : counts) {
        if (c == 0) throw std::invalid_argument("adaptive_redundancy_approx: counts must be positive");
        n += static_cast<double>(c);
    }
    double r = 0.5 * (k - 1.0) * std::log2(n / (2.0 * pi)) - std::lgamma(k) / std::log(2.0);
    for (auto c : counts) r -= 0.5 * std::log2(static_cast<double>(c) / n);
    return r;
}

std::vector<std::vector<std::uint64_t>> context_counts(const std::vector<int>& symbols, std::size_t alphabet_size,
                                                       const ArithmeticContextFunction& context,
                                                       std::size_t num_contexts) {
    if (num_contexts == 0) throw std::invalid_argument("context_counts: need at least one context");
    std::vector<std::vector<std::uint64_t>> counts(num_contexts, std::vector<std::uint64_t>(alphabet_size, 0));
    std::vector<int>                        history;
    history.reserve(symbols.size());
    for (int s : symbols) {
        if (s < 0 || static_cast<std::size_t>(s) >= alphabet_size)
            throw std::invalid_argument("context_counts: symbol out of range");
        ++counts[pick_context(context, history, num_contexts)][s];
        history.push_back(s);
    }
    return counts;
}

}  // namespace datamunge::algorithms
