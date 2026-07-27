#pragma once

#include <datamunge/stats/resample_types.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::stats {

struct UStatisticOptions {
    /// @brief If 0, the *complete* U-statistic is computed over all \(\binom{n}{m}\) subsets.
    ///        If > 0, an *incomplete* U-statistic is computed from this many randomly drawn
    ///        m-subsets -- the practical route when \(\binom{n}{m}\) is astronomically large
    ///        (as it is for the "infinite-order" kernels underlying subbagging / random forests).
    std::size_t max_terms{0};
    std::uint64_t seed{42};
};

struct UStatisticResult {
    double estimate{0.0};        ///< the U-statistic U_n
    double zeta1{0.0};           ///< estimated first-order Hoeffding variance component zeta_1
    double variance{0.0};        ///< estimated Var(U_n), the leading-order m^2 * zeta_1 / n
    double standard_error{0.0};
    std::size_t degree{0};       ///< kernel degree m
    std::size_t num_terms{0};    ///< number of kernel evaluations used
    bool complete{false};        ///< true iff all C(n,m) subsets were used
};

/// @brief A U-statistic (Hoeffding 1948): the minimum-variance unbiased estimator of an
///        estimand \(\theta = \mathbb{E}[h(X_1,\dots,X_m)]\) defined by a symmetric kernel
///        \(h\) of degree \(m\). It averages the kernel over every size-m subset of the sample.
///        Many familiar estimators are U-statistics (the sample mean, the unbiased variance,
///        Gini's mean difference, Kendall's tau). The variance is estimated to leading order
///        via the first Hoeffding projection, \(\mathrm{Var}(U_n) \approx m^2 \zeta_1 / n\).
///        When \(\binom{n}{m}\) is too large to enumerate, an *incomplete* U-statistic over a
///        random subset of tuples is used instead (see UStatisticOptions::max_terms) -- the
///        same device that turns bagged predictors into "infinite-order" U-statistics.
class UStatistic {
public:
    /// @param degree the kernel degree m (m >= 1).
    explicit UStatistic(std::size_t degree, UStatisticOptions options = {});

    [[nodiscard]] UStatisticResult run(const std::vector<double>& sample, const SymmetricKernel& kernel) const;

private:
    std::size_t degree_;
    UStatisticOptions options_;
};

} // namespace datamunge::stats
