#pragma once

/// \file approximate_counting.hpp
/// \brief Morris's approximate counting algorithm: count up to \f$n\f$ events using
///        only about \f$\log_2\log_2 n\f$ bits (Morris 1978).
///
/// To count \f$n\f$ events exactly needs \f$\lceil\log_2 n\rceil\f$ bits. When billions
/// of events must be counted in tiny registers (early hardware, per-flow counters,
/// sketches), that is too much. *Morris's algorithm* stores not the count but an
/// *exponent* \f$c\f$, and increments \f$c\f$ only *probabilistically*: on each event it
/// bumps \f$c\f$ with probability \f$2^{-c}\f$ (base 2), so \f$c\f$ grows like
/// \f$\log_2 n\f$ and the estimate \f$\hat n = 2^{c}-1\f$ tracks the true count. The
/// register therefore holds \f$\approx\log_2\log_2 n\f$ bits -- doubly logarithmic.
///
/// The estimator \f$\hat n = 2^c - 1\f$ is *unbiased*: \f$\mathbb{E}[\hat n]=n\f$. Its
/// relative variance is fixed (about \f$1/2\f$ for base 2), which is large for a single
/// counter but is tuned down by using a base \f$a=1+2\varepsilon\f$ closer to 1: the
/// generalized update increments with probability \f$a^{-c}\f$ and estimates
/// \f$\hat n = (a^{c}-1)/(a-1)\f$, trading more bits for accuracy
/// \f$\operatorname{Var}(\hat n)/n^2 \approx \varepsilon\f$.

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace datamunge::algorithms {

/// A probabilistic (Morris) counter with a tunable base for accuracy/space trade-off.
class MorrisCounter {
 public:
    /// \param base The base \f$a>1\f$. Base 2 is the classic counter; values closer to 1
    ///             (e.g. 1.08) shrink the variance at the cost of more bits in \f$c\f$.
    /// \throws std::invalid_argument if \p base is not greater than 1.
    explicit MorrisCounter(double base = 2.0);

    /// Process one event: increment the exponent with probability \f$a^{-c}\f$.
    /// \param rng Random engine for the coin flip.
    void increment(std::mt19937_64& rng);

    /// \brief Process \p events events at once, with the same distribution as calling
    ///        increment() that many times.
    ///
    /// Instead of one coin per event it draws the geometric waiting time until the next
    /// increment, so the cost is proportional to the number of increments, about
    /// \f$\log_a n\f$, not to \p events.
    /// \param events Number of events to add.
    /// \param rng Random engine.
    void add(std::uint64_t events, std::mt19937_64& rng);

    /// The stored exponent \f$c\f$ (this, in bits, is all the counter occupies).
    /// \return The exponent, a whole number stored as a double.
    double exponent() const { return exponent_; }

    /// The base \f$a\f$.
    /// \return The base passed to the constructor.
    double base() const { return base_; }

    /// Bits needed to store the current exponent, \f$\lceil\log_2(c+1)\rceil\f$ (0 for \f$c=0\f$).
    /// \return The register width in bits.
    int register_bits() const;

    /// Unbiased estimate of the number of events seen: \f$(a^{c}-1)/(a-1)\f$ (\f$2^c-1\f$ for base 2).
    /// \return The estimate.
    double estimate() const;

 private:
    double base_     = 2.0;  ///< the base a.
    double exponent_ = 0.0;  ///< the stored exponent c.
};

/// \brief Exact variance of the Morris estimate after \p n events: \f$(a-1)\,n(n-1)/2\f$.
/// \param n Number of events.
/// \param base The base \f$a>1\f$.
/// \return \f$\operatorname{Var}(\hat n)\f$.
double morris_variance(double n, double base);

/// \brief Exact distribution of the exponent after \p n events, by dynamic programming over events.
///
/// Entry \f$c\f$ is \f$P(c_n=c)\f$ for \f$c=0,\dots,\f$ \p max_level; mass above \p max_level
/// is added to the last entry. The cost is \f$O(n\cdot\f$ \p max_level\f$)\f$.
/// \param n Number of events.
/// \param base The base \f$a>1\f$.
/// \param max_level Largest exponent tracked.
/// \return The probability of each exponent.
/// \throws std::invalid_argument if \p base is not greater than 1.
std::vector<double> morris_exponent_pmf(std::uint64_t n, double base, std::size_t max_level);

}  // namespace datamunge::algorithms
