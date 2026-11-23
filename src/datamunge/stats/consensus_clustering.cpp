/// @file consensus_clustering.cpp
/// @brief Subsample selection and conditional co-assignment frequencies.

#include <datamunge/stats/consensus_clustering.hpp>

#include "clustering_kmeans_detail.hpp"
#include "subsample_detail.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {
/// @brief Validation and sampling helpers for the pairwise estimate.
namespace {

/// @brief Calculates the supported subsample size and validates pairwise storage.
std::size_t sample_size(const linalg::DenseMatrix<double>& data,
    const ConsensusClusteringOptions& options) {
    const auto n = data.rows();
    if (n == 0 || data.cols() == 0 || options.clusters == 0 || options.subsamples == 0 ||
        !(options.sample_fraction > 0 && options.sample_fraction <= 1) ||
        !std::ranges::all_of(data.data(),[](double value) { return std::isfinite(value); }))
        throw std::invalid_argument("consensus clustering: invalid observations or controls");
    if (n > std::vector<std::uint64_t>{}.max_size()/n ||
        n > std::vector<double>{}.max_size()/n)
        throw std::length_error("consensus clustering: dense pairwise matrices exceed storage capacity");
    const auto retained = options.sample_fraction == 1 ? n
        : static_cast<std::size_t>(static_cast<double>(n)*options.sample_fraction);
    if (retained < options.clusters)
        throw std::invalid_argument("consensus clustering: fewer sampled rows than requested clusters");
    return retained;
}

} // namespace

ConsensusClusteringResult consensus_clustering(const linalg::DenseMatrix<double>& observations,
    const ConsensusClusteringOptions& options, const KMeansOptions& clustering) {
    if (clustering.n_init == 0 || clustering.max_iterations == 0 ||
        !(clustering.tolerance > 0) || !std::isfinite(clustering.tolerance))
        throw std::invalid_argument("consensus clustering: invalid k-means controls");
    const ConsensusPartitioner partition = [clustering](const auto& sample,std::size_t k) {
        const auto scaled = clustering_detail::scale_clustering_data(sample);
        return clustering_detail::matrix_kmeans(scaled,k,clustering);
    };
    return consensus_clustering(observations,partition,options);
}

ConsensusClusteringResult consensus_clustering(const linalg::DenseMatrix<double>& observations,
    const ConsensusPartitioner& partition, const ConsensusClusteringOptions& options) {
    if (!partition)
        throw std::invalid_argument("consensus clustering: partitioner must be callable");
    const auto n = observations.rows(), size = sample_size(observations,options);
    ConsensusClusteringResult fit{linalg::DenseMatrix<std::uint64_t>(n,n),
        linalg::DenseMatrix<std::uint64_t>(n,n),linalg::DenseMatrix<double>(n,n),size};
    linalg::DenseMatrix<double> subset(size,observations.cols());
    std::vector<std::size_t> sampled(n);
    std::mt19937_64 random(options.seed);
    for (std::size_t run = 0; run < options.subsamples; ++run) {
        subsample_detail::draw_rows(sampled,size,random);
        for (std::size_t i = 0; i < size; ++i)
            for (std::size_t j = 0; j < subset.cols(); ++j)
                subset(i,j) = observations(sampled[i],j);
        const auto labels = partition(subset,options.clusters);
        if (labels.size() != size)
            throw std::invalid_argument("consensus clustering: partition size must match the subsample");
        auto groups = labels;
        std::ranges::sort(groups);
        if (static_cast<std::size_t>(std::unique(groups.begin(),groups.end())-groups.begin()) != options.clusters)
            throw std::domain_error("consensus clustering: partition has the wrong number of nonempty clusters");
        for (std::size_t i = 0; i < size; ++i)
            for (std::size_t j = i; j < size; ++j) {
                ++fit.co_sampled(sampled[i],sampled[j]);
                fit.co_assigned(sampled[i],sampled[j]) += labels[i] == labels[j];
            }
    }

    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i; j < n; ++j) {
            const auto available = fit.co_sampled(i,j);
            fit.co_sampled(j,i) = available;
            fit.co_assigned(j,i) = fit.co_assigned(i,j);
            fit.probability(i,j) = fit.probability(j,i) = available > 0
                ? static_cast<double>(static_cast<long double>(fit.co_assigned(i,j))/available)
                : std::numeric_limits<double>::quiet_NaN();
        }
    return fit;
}

} // namespace datamunge::stats
