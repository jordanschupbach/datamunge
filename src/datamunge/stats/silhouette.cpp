/// @file silhouette.cpp
/// @brief Cluster separation diagnostics for a supplied partition and dissimilarities.

#include <datamunge/stats/silhouette.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Validation of pairwise dissimilarities.
namespace {

/// @brief Checks symmetry to floating-point tolerance and rejects invalid entries.
void validate_dissimilarities(const linalg::DenseMatrix<double>& distances) {
    for (std::size_t i = 0; i < distances.rows(); ++i) {
        if (distances(i,i) != 0)
            throw std::invalid_argument("silhouette analysis: dissimilarity diagonal must be zero");
        for (std::size_t j = 0; j < i; ++j) {
            const auto a = distances(i,j), b = distances(j,i);
            if (!(a >= 0 && b >= 0) || !std::isfinite(a) || !std::isfinite(b) ||
                std::abs(a-b) > 64*std::numeric_limits<double>::epsilon()*std::max(a,b))
                throw std::invalid_argument("silhouette analysis: dissimilarities must be nonnegative, finite and symmetric");
        }
    }
}

} // namespace

SilhouetteResult silhouette_analysis(const linalg::DenseMatrix<double>& dissimilarities,
    const std::vector<std::size_t>& labels) {
    const auto n = dissimilarities.rows();
    if (n < 2 || dissimilarities.cols() != n || labels.size() != n)
        throw std::invalid_argument("silhouette analysis: nonempty dissimilarity matrix and row labels must agree");
    validate_dissimilarities(dissimilarities);
    auto groups = labels;
    std::ranges::sort(groups);
    groups.erase(std::unique(groups.begin(),groups.end()),groups.end());
    const auto k = groups.size();
    if (k < 2)
        throw std::invalid_argument("silhouette analysis: at least two clusters are required");
    std::vector<std::size_t> index(n), count(k);
    for (std::size_t i = 0; i < n; ++i) {
        index[i] = std::ranges::lower_bound(groups,labels[i])-groups.begin();
        ++count[index[i]];
    }

    SilhouetteResult result;
    result.observations.reserve(n);
    result.clusters.reserve(k);
    std::vector<long double> means(k), width_sum(k);
    for (std::size_t i = 0; i < n; ++i) {
        const auto own = index[i];
        std::ranges::fill(means,0.0L);
        for (std::size_t j = 0; j < n; ++j)
            if (i != j) {
                const auto other = index[j];
                const auto denominator = count[other] - (other == own);
                const auto value = std::midpoint(static_cast<long double>(dissimilarities(i,j)),
                    static_cast<long double>(dissimilarities(j,i)));
                means[other] += value/denominator;
            }
        auto neighbor = own == 0 ? std::size_t{1} : std::size_t{0};
        for (std::size_t j = 0; j < k; ++j)
            if (j != own && means[j] < means[neighbor])
                neighbor = j;
        const auto a = means[own], b = means[neighbor];
        const auto denominator = std::max(a,b);
        const auto width = count[own] > 1 && denominator > 0 ? (b-a)/denominator : 0;
        width_sum[own] += width;
        result.observations.push_back({groups[own],groups[neighbor],
            static_cast<double>(a),static_cast<double>(b),static_cast<double>(width)});
    }

    long double total = 0;
    for (std::size_t j = 0; j < k; ++j) {
        total += width_sum[j];
        result.clusters.push_back({groups[j],count[j],static_cast<double>(width_sum[j]/count[j])});
    }
    result.mean_width = static_cast<double>(total/n);
    return result;
}

} // namespace datamunge::stats
