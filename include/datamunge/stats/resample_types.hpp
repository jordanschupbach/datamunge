#pragma once

#include <functional>
#include <vector>

namespace datamunge::stats {

/// @brief A scalar statistic computed from a univariate sample -- e.g. the mean, median,
///        trimmed mean, or standard deviation. The resampling methods (bootstrap, jackknife)
///        call this on resampled versions of the data.
using SampleStatistic = std::function<double(const std::vector<double>&)>;

/// @brief A symmetric kernel of some fixed degree m: a function of an m-tuple of observations,
///        the building block of U- and V-statistics. "Symmetric" means its value does not
///        depend on the order of its arguments (the caller supplies them as a vector of length
///        m). Example: the degree-2 kernel h(a,b) = (a-b)^2 / 2 yields the sample variance.
using SymmetricKernel = std::function<double(const std::vector<double>&)>;

} // namespace datamunge::stats
