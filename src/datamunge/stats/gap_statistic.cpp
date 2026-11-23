/// @file gap_statistic.cpp
/// @brief Reference-box simulation and scale-stable within-cluster dispersion.

#include <datamunge/stats/gap_statistic.hpp>

#include <datamunge/linalg/svd.hpp>

#include "clustering_kmeans_detail.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {
/// @brief Data normalization, reference simulation and dispersion calculations.
namespace {

/// @brief Working coordinates in extended precision.
using Matrix = linalg::DenseMatrix<long double>;
/// @brief One vector of working coordinates.
using Vector = std::vector<long double>;

/// @brief Stable running mean and centered sum of squares for simulated log dispersions.
struct Moments {
    std::size_t count{};     ///< Number of values received.
    long double mean{};      ///< Arithmetic mean.
    long double squared{};  ///< Sum of squared deviations from the mean.

    /// @brief Incorporates one additional value.
    void add(long double value) {
        const auto delta = value-mean;
        mean += delta/++count;
        squared += delta*(value-mean);
    }
};

/// @brief Computes log SSE from a complete partition, using cluster means of the actual assignments.
long double dispersion(const clustering_detail::ScaledData& scaled, const std::vector<std::size_t>& labels,
    std::size_t k) {
    const auto& data = scaled.values;
    if (labels.size() != data.rows())
        throw std::invalid_argument("gap statistic: partition size does not match observations");
    auto values = labels;
    std::ranges::sort(values);
    values.erase(std::unique(values.begin(),values.end()),values.end());
    if (values.size() != k)
        throw std::domain_error("gap statistic: partition must contain the requested number of nonempty clusters");
    Matrix centers(k,data.cols());
    std::vector<std::size_t> count(k);
    long double total = 0;
    for (std::size_t i = 0; i < data.rows(); ++i) {
        const auto cluster = static_cast<std::size_t>(std::ranges::lower_bound(values,labels[i])-values.begin());
        ++count[cluster];
        for (std::size_t j = 0; j < data.cols(); ++j) {
            const auto delta = data(i,j)-centers(cluster,j);
            centers(cluster,j) += delta/count[cluster];
            total += delta*(data(i,j)-centers(cluster,j));
        }
    }
    if (!(total > 0))
        throw std::domain_error("gap statistic: within-cluster dispersion must be positive");
    return std::log(total)+2*std::log(scaled.scale);
}

/// @brief Reference ranges, centering offsets and optional principal axes for inverse rotation.
struct Box {
    Vector lower; ///< Minimum coordinate along each sampled axis.
    Vector upper; ///< Maximum coordinate along each sampled axis.
    Vector center; ///< Centering offset in normalized original coordinates.
    Matrix axes; ///< Original coordinates by retained PCA coordinates, empty for axis-aligned sampling.
};

/// @brief Reads coordinate bounds, using thin SVD for the principal-component box when requested.
Box reference_box(Matrix data, GapReference reference) {
    const auto n = data.rows(), d = data.cols();
    Box box;
    box.center.resize(d);
    if (reference == GapReference::PrincipalComponentBox) {
        for (std::size_t j = 0; j < d; ++j) {
            for (std::size_t i = 0; i < n; ++i)
                box.center[j] += data(i,j)/n;
            for (std::size_t i = 0; i < n; ++i)
                data(i,j) -= box.center[j];
        }
        const auto cutoff = static_cast<long double>(std::max(n,d))*std::numeric_limits<double>::epsilon();
        const auto decomposition = linalg::svd(data,cutoff);
        if (!decomposition.converged)
            throw std::runtime_error("gap statistic: reference principal axes did not converge");
        const auto rank = decomposition.rank(cutoff);
        if (rank == 0)
            throw std::domain_error("gap statistic: observations have zero dispersion");
        box.axes = Matrix(d,rank);
        for (std::size_t j = 0; j < d; ++j)
            for (std::size_t k = 0; k < rank; ++k)
                box.axes(j,k) = decomposition.V(j,k);
        data = data*box.axes;
    }
    box.lower = data.row(0);
    box.upper = box.lower;
    for (std::size_t j = 0; j < data.cols(); ++j)
        for (std::size_t i = 1; i < n; ++i) {
            box.lower[j] = std::min(box.lower[j],data(i,j));
            box.upper[j] = std::max(box.upper[j],data(i,j));
        }
    return box;
}

/// @brief Samples the reference box and restores the feature coordinates for the partitioner.
linalg::DenseMatrix<double> sample_box(const Box& box, const linalg::DenseMatrix<double>& original,
    long double scale, std::mt19937_64& random) {
    linalg::DenseMatrix<double> sample(original.rows(),original.cols());
    Vector rotated(box.lower.size());
    for (std::size_t i = 0; i < sample.rows(); ++i) {
        for (std::size_t j = 0; j < rotated.size(); ++j) {
            const auto uniform = static_cast<long double>(random() >> 11)*0x1p-53L;
            rotated[j] = std::lerp(box.lower[j],box.upper[j],uniform);
        }
        for (std::size_t j = 0; j < sample.cols(); ++j) {
            long double value = box.center[j];
            if (box.axes.cols() == 0)
                value = rotated[j];
            else
                for (std::size_t k = 0; k < rotated.size(); ++k)
                    value += box.axes(j,k)*rotated[k];
            value = std::fma(value,scale,static_cast<long double>(original(0,j)));
            if (std::abs(value) > std::numeric_limits<double>::max())
                throw std::overflow_error("gap statistic: sampled feature exceeds finite double range");
            sample(i,j) = static_cast<double>(value);
        }
    }
    return sample;
}

/// @brief Evaluates all requested candidate counts on one dataset.
Vector evaluate(const linalg::DenseMatrix<double>& data, const GapPartitioner& partition,
    std::size_t maximum) {
    const auto scaled = clustering_detail::scale_clustering_data(data);
    Vector scores;
    scores.push_back(dispersion(scaled,std::vector<std::size_t>(data.rows()),1));
    for (std::size_t k = 2; k <= maximum; ++k)
        scores.push_back(dispersion(scaled,partition(data,k),k));
    return scores;
}

} // namespace

GapStatisticResult gap_statistic(const linalg::DenseMatrix<double>& observations,
    const GapStatisticOptions& options, const KMeansOptions& clustering) {
    if (clustering.n_init == 0 || clustering.max_iterations == 0 ||
        !(clustering.tolerance > 0) || !std::isfinite(clustering.tolerance))
        throw std::invalid_argument("gap statistic: invalid k-means controls");
    const GapPartitioner partition = [clustering](const linalg::DenseMatrix<double>& data,std::size_t k) {
        const auto scaled = clustering_detail::scale_clustering_data(data);
        return clustering_detail::matrix_kmeans(scaled,k,clustering);
    };
    return gap_statistic(observations,partition,options);
}

GapStatisticResult gap_statistic(const linalg::DenseMatrix<double>& observations,
    const GapPartitioner& partition, const GapStatisticOptions& options) {
    if (observations.rows() < 2 || observations.cols() == 0 || !partition ||
        options.max_clusters == 0 || options.max_clusters >= observations.rows() ||
        options.reference_samples < 2 ||
        (options.reference != GapReference::PrincipalComponentBox &&
         options.reference != GapReference::AxisAlignedBox) ||
        !std::ranges::all_of(observations.data(),[](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("gap statistic: invalid observations, partitioner or simulation controls");
    const auto observed = evaluate(observations,partition,options.max_clusters);
    auto scaled = clustering_detail::scale_clustering_data(observations);
    const auto box = reference_box(std::move(scaled.values),options.reference);
    std::mt19937_64 random(options.seed);
    std::vector<Moments> reference(options.max_clusters);
    for (std::size_t b = 0; b < options.reference_samples; ++b) {
        const auto sample = sample_box(box,observations,scaled.scale,random);
        const auto scores = evaluate(sample,partition,options.max_clusters);
        for (std::size_t k = 0; k < scores.size(); ++k)
            reference[k].add(scores[k]);
    }

    GapStatisticResult result;
    result.selected_clusters = options.max_clusters;
    for (std::size_t i = 0; i < observed.size(); ++i) {
        const auto& stats = reference[i];
        const auto error = std::sqrt(std::max(0.0L,stats.squared)/(stats.count-1)
            * (1+1.0L/stats.count));
        result.entries.push_back({i+1,static_cast<double>(observed[i]),static_cast<double>(stats.mean),
            static_cast<double>(stats.mean-observed[i]),static_cast<double>(error)});
    }
    for (std::size_t i = 1; i < result.entries.size(); ++i) {
        const auto& next = result.entries[i];
        if (result.entries[i-1].gap >= next.gap-next.simulation_error) {
            result.selected_clusters = i;
            break;
        }
    }
    return result;
}

} // namespace datamunge::stats
