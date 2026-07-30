#pragma once

/// \file online_statistics.hpp
/// \brief Numerically stable, single-pass computation of mean, variance, and
///        covariance -- Welford's algorithm and its covariance extension.
///
/// The textbook "sum of squares minus square of sum" variance formula,
/// \f$\operatorname{Var}=\frac1n\sum x_i^2 - \bar{x}^2\f$, is a *single pass* but
/// *numerically catastrophic*: for data with a large mean and small spread it
/// subtracts two huge, nearly equal numbers and loses all precision (it can even
/// return a negative variance). *Welford's algorithm* (1962) computes the same
/// quantities in one pass while staying accurate, by updating running estimates with
/// each new value:
/// \f[
///   \bar{x}_n = \bar{x}_{n-1} + \frac{x_n - \bar{x}_{n-1}}{n},\qquad
///   M_{2,n} = M_{2,n-1} + (x_n-\bar{x}_{n-1})(x_n-\bar{x}_n),
/// \f]
/// after which \f$M_2/n\f$ is the population variance and \f$M_2/(n-1)\f$ the sample
/// variance. The update never forms large intermediate sums, so it is stable
/// regardless of the data's offset. The same idea extends to a running covariance
/// (needed for streaming correlation and PCA) and the accumulators are *mergeable*,
/// enabling parallel/chunked reduction.

#include <cmath>
#include <cstddef>

namespace datamunge::algorithms {

/// Streaming mean and variance via Welford's recurrence.
class OnlineVariance {
 public:
    /// Incorporate one observation.
    void add(double x) {
        ++count_;
        const double delta = x - mean_;
        mean_ += delta / static_cast<double>(count_);
        const double delta2 = x - mean_;
        m2_ += delta * delta2;
    }

    std::size_t count() const { return count_; }
    double      mean() const { return mean_; }
    /// Population variance \f$M_2/n\f$.
    double      variance() const { return count_ > 0 ? m2_ / static_cast<double>(count_) : 0.0; }
    /// Unbiased sample variance \f$M_2/(n-1)\f$.
    double      sample_variance() const { return count_ > 1 ? m2_ / static_cast<double>(count_ - 1) : 0.0; }
    double      sample_stddev() const { return std::sqrt(sample_variance()); }
    /// Raw \f$M_2 = \sum (x_i-\bar x)^2\f$ (the sum of squared deviations).
    double      sum_of_squares() const { return m2_; }

    /// Merge another accumulator (Chan et al.'s parallel combination), for chunked reduction.
    void merge(const OnlineVariance& other) {
        if (other.count_ == 0) return;
        if (count_ == 0) {
            *this = other;
            return;
        }
        const double na = static_cast<double>(count_), nb = static_cast<double>(other.count_);
        const double delta = other.mean_ - mean_;
        const double n     = na + nb;
        mean_              = mean_ + delta * nb / n;
        m2_                = m2_ + other.m2_ + delta * delta * na * nb / n;
        count_ += other.count_;
    }

 private:
    std::size_t count_ = 0;
    double      mean_  = 0.0;
    double      m2_    = 0.0;
};

/// Streaming covariance (and correlation) of paired observations, Welford-style.
class OnlineCovariance {
 public:
    void add(double x, double y) {
        ++count_;
        const double dx = x - mean_x_;
        mean_x_ += dx / static_cast<double>(count_);
        mean_y_ += (y - mean_y_) / static_cast<double>(count_);
        c_ += dx * (y - mean_y_);  // uses updated mean_y_, old mean_x via dx
        vx_.add(x);
        vy_.add(y);
    }

    std::size_t count() const { return count_; }
    double      mean_x() const { return mean_x_; }
    double      mean_y() const { return mean_y_; }
    /// Population covariance.
    double      covariance() const { return count_ > 0 ? c_ / static_cast<double>(count_) : 0.0; }
    /// Sample covariance.
    double      sample_covariance() const { return count_ > 1 ? c_ / static_cast<double>(count_ - 1) : 0.0; }
    /// Pearson correlation coefficient.
    double correlation() const {
        const double sx = vx_.sample_stddev(), sy = vy_.sample_stddev();
        return (sx > 0.0 && sy > 0.0) ? sample_covariance() / (sx * sy) : 0.0;
    }

 private:
    std::size_t    count_  = 0;
    double         mean_x_ = 0.0;
    double         mean_y_ = 0.0;
    double         c_      = 0.0;  // running sum of products of deviations
    OnlineVariance vx_, vy_;
};

}  // namespace datamunge::algorithms
