#pragma once

#include <datamunge/stats/resample_types.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::stats {

struct VStatisticOptions {
    /// @brief If 0, the V-statistic is computed exactly over all \(n^m\) index tuples (feasible
    ///        only for small \(n^m\)). If > 0, it is Monte Carlo-estimated from this many tuples
    ///        drawn uniformly with replacement.
    std::size_t max_terms{0};
    std::uint64_t seed{42};
};

struct VStatisticResult {
    double estimate{0.0};      ///< the V-statistic V_n
    std::size_t degree{0};     ///< kernel degree m
    std::size_t num_terms{0};  ///< number of kernel evaluations used
    bool exact{false};         ///< true iff all n^m tuples were enumerated
};

/// @brief A V-statistic (von Mises 1947): the *plug-in* estimator of
///        \(\theta = \mathbb{E}[h(X_1,\dots,X_m)]\), obtained by applying the kernel's
///        expectation to the empirical distribution \(\hat F_n\). Equivalently it averages the
///        symmetric kernel \(h\) over all \(n^m\) tuples drawn *with replacement* -- the same
///        kernel as the corresponding U-statistic, but including the "diagonal" tuples with
///        repeated indices. Those diagonal terms make the V-statistic biased at finite n
///        (\(V_n - U_n = O(1/n)\)), but it is often the natural plug-in quantity and shares the
///        U-statistic's asymptotic distribution. Example: with the kernel
///        \(h(a,b) = (a-b)^2/2\), \(V_n\) is the biased (\(\div n\)) sample variance while the
///        U-statistic is the unbiased (\(\div (n-1)\)) one.
class VStatistic {
public:
    explicit VStatistic(std::size_t degree, VStatisticOptions options = {});

    [[nodiscard]] VStatisticResult run(const std::vector<double>& sample, const SymmetricKernel& kernel) const;

private:
    std::size_t degree_;
    VStatisticOptions options_;
};

} // namespace datamunge::stats
