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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

namespace datamunge::algorithms {

/// A probabilistic (Morris) counter with a tunable base for accuracy/space trade-off.
class MorrisCounter {
 public:
    /// \param base The base \f$a>1\f$. Base 2 is the classic counter; values closer to 1
    ///             (e.g. 1.08) shrink the variance at the cost of more bits in \f$c\f$.
    explicit MorrisCounter(double base = 2.0) : base_(base) {}

    /// Process one event: increment the exponent with probability \f$a^{-c}\f$.
    void increment(std::mt19937_64& rng) {
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        if (unit(rng) < std::pow(base_, -exponent_)) exponent_ += 1.0;
    }

    /// The stored exponent \f$c\f$ (this, in bits, is all the counter occupies).
    double exponent() const { return exponent_; }

    /// Unbiased estimate of the number of events seen: \f$(a^{c}-1)/(a-1)\f$ (\f$2^c-1\f$ for base 2).
    double estimate() const {
        return base_ == 2.0 ? std::pow(2.0, exponent_) - 1.0
                            : (std::pow(base_, exponent_) - 1.0) / (base_ - 1.0);
    }

 private:
    double base_     = 2.0;
    double exponent_ = 0.0;
};

}  // namespace datamunge::algorithms
