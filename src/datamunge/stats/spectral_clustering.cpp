/// @file spectral_clustering.cpp
/// @brief Weighted normalized graph embedding and k-means partitions.

#include <datamunge/stats/spectral_clustering.hpp>

#include <datamunge/linalg/eigen.hpp>
#include <datamunge/stats/knn_graph.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace datamunge::stats {
/// @brief Affinity checks and embedding helpers.
namespace {

/// @brief Working graph matrix for degrees and eigendecomposition.
using Matrix = linalg::DenseMatrix<long double>;

/// @brief Checks parameter ranges and dense workspace capacity.
void check_options(std::size_t n, const KMeansOptions& options) {
    if (options.n_clusters == 0 || options.n_clusters > n || options.n_init == 0 ||
        options.max_iterations == 0 || !(options.tolerance > 0) || !std::isfinite(options.tolerance))
        throw std::invalid_argument("spectral clustering: invalid clustering controls");
    if (n > std::vector<long double>{}.max_size()/n)
        throw std::length_error("spectral clustering: affinity workspace exceeds storage capacity");
}

/// @brief Checks symmetric nonnegative weights and mirrors each off-diagonal average.
Matrix validated_affinity(const linalg::DenseMatrix<double>& input) {
    const auto n = input.rows();
    Matrix weights(n,n);
    for (std::size_t i = 0; i < n; ++i) {
        if (input(i,i) != 0)
            throw std::invalid_argument("spectral clustering: affinity diagonal must be zero");
        for (std::size_t j = 0; j < i; ++j) {
            const auto a = input(i,j), b = input(j,i);
            if (!std::isfinite(a) || !std::isfinite(b) || a < 0 || b < 0 ||
                std::abs(a-b) > 64*std::numeric_limits<double>::epsilon()*std::max(a,b))
                throw std::invalid_argument("spectral clustering: affinities must be finite, nonnegative and symmetric");
            weights(i,j) = weights(j,i) = std::midpoint(static_cast<long double>(a),
                static_cast<long double>(b));
        }
    }
    return weights;
}

/// @brief Labels connected components by graph traversal in vertex order.
std::size_t component_labels(const Matrix& weights, std::vector<std::size_t>& labels) {
    const auto n = weights.rows();
    labels.assign(n,n);
    std::vector<std::size_t> queue;
    std::size_t count = 0;
    for (std::size_t start = 0; start < n; ++start) {
        if (labels[start] != n)
            continue;
        labels[start] = count;
        queue.push_back(start);
        for (std::size_t head = 0; head < queue.size(); ++head) {
            const auto current = queue[head];
            for (std::size_t next = 0; next < n; ++next)
                if (labels[next] == n && weights(current,next) > 0) {
                    labels[next] = count;
                    queue.push_back(next);
                }
        }
        queue.clear();
        ++count;
    }
    return count;
}

/// @brief Converts weights in place to the symmetric normalized Laplacian.
void normalize_laplacian(Matrix& matrix) {
    const auto n = matrix.rows();
    std::vector<long double> root_scale(n), root_sum(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto row = matrix.row(i);
        const auto largest = *std::max_element(row.begin(),row.end());
        if (largest > 0) {
            root_scale[i] = std::sqrt(largest);
            long double sum = 0;
            for (const auto weight : row)
                sum += weight/largest;
            root_sum[i] = std::sqrt(sum);
            matrix(i,i) = 1;
        }
    }
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j) {
            if (matrix(i,j) == 0)
                continue;
            const auto root = std::sqrt(matrix(i,j));
            matrix(i,j) = matrix(j,i) = -((root/root_scale[i])/root_sum[i]) *
                ((root/root_scale[j])/root_sum[j]);
        }
}

/// @brief Partitions embedding rows through the existing k-means implementation.
std::vector<std::size_t> partition(const linalg::DenseMatrix<double>& embedding,
    const KMeansOptions& options) {
    dstruct::DataFrame frame;
    std::vector<std::string> columns;
    for (std::size_t j = 0; j < embedding.cols(); ++j) {
        columns.push_back("eigenvector_" + std::to_string(j));
        std::vector<double> values(embedding.rows());
        for (std::size_t i = 0; i < embedding.rows(); ++i)
            values[i] = embedding(i,j);
        frame.add_column(columns.back(),std::move(values));
    }
    const KMeans fit(frame,columns,options);
    auto labels = fit.labels();
    std::vector<std::size_t> order(options.n_clusters,options.n_clusters);
    std::size_t count = 0;
    for (auto& label : labels) {
        if (order[label] == options.n_clusters)
            order[label] = count++;
        label = order[label];
    }
    if (count != options.n_clusters)
        throw std::runtime_error("spectral clustering: k-means produced an empty cluster");
    return labels;
}

} // namespace

SpectralClusteringFit fit_spectral_clustering(const linalg::DenseMatrix<double>& affinity,
    const KMeansOptions& options) {
    const auto n = affinity.rows(), k = options.n_clusters;
    if (n == 0 || n != affinity.cols())
        throw std::invalid_argument("spectral clustering: affinity must be nonempty and square");
    check_options(n,options);
    auto matrix = validated_affinity(affinity);
    SpectralClusteringFit fit;
    fit.connected_components = component_labels(matrix,fit.labels);
    if (fit.connected_components > k)
        throw std::invalid_argument("spectral clustering: more connected components than clusters");
    fit.eigenvalues.resize(k);
    fit.embedding = linalg::DenseMatrix<double>(n,k);
    if (fit.connected_components == k) {
        for (std::size_t i = 0; i < n; ++i)
            fit.embedding(i,fit.labels[i]) = 1;
        return fit;
    }

    normalize_laplacian(matrix);
    const auto eig = linalg::symmetric_eigen(matrix);
    for (std::size_t j = 0; j < k; ++j) {
        const auto index = n-1-j;
        fit.eigenvalues[j] = j < fit.connected_components ? 0
            : static_cast<double>(std::clamp(eig.eigenvalues[index],0.0L,2.0L));
        long double largest = 0;
        for (std::size_t i = 0; i < n; ++i)
            if (std::abs(eig.eigenvectors(i,index)) > std::abs(largest))
                largest = eig.eigenvectors(i,index);
        const auto orientation = std::signbit(largest) ? -1 : 1;
        for (std::size_t i = 0; i < n; ++i)
            fit.embedding(i,j) = static_cast<double>(orientation*eig.eigenvectors(i,index));
    }
    for (std::size_t i = 0; i < n; ++i) {
        long double norm = 0;
        for (const double value : fit.embedding.row(i))
            norm = std::hypot(norm,value);
        if (!(norm > 0))
            throw std::runtime_error("spectral clustering: embedding contains a zero row");
        for (std::size_t j = 0; j < k; ++j)
            fit.embedding(i,j) = static_cast<double>(fit.embedding(i,j)/norm);
    }
    fit.labels = partition(fit.embedding,options);
    return fit;
}

SpectralClusteringFit fit_spectral_clustering_knn(const linalg::DenseMatrix<double>& observations,
    std::size_t neighbors, const KMeansOptions& options) {
    const auto n = observations.rows();
    if (observations.cols() == 0 || neighbors == 0 || neighbors >= n)
        throw std::invalid_argument("spectral clustering: invalid observations or neighbor count");
    check_options(n,options);
    double scale = 0;
    for (const double value : observations.data()) {
        if (!std::isfinite(value))
            throw std::invalid_argument("spectral clustering: observations must be finite");
        scale = std::max(scale,std::abs(value));
    }
    if (scale == 0)
        scale = 1;
    std::vector<std::vector<double>> points(n,std::vector<double>(observations.cols()));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < observations.cols(); ++j)
            points[i][j] = observations(i,j)/scale;
    const KnnGraph graph(points,neighbors);
    linalg::DenseMatrix<double> affinity(n,n);
    for (std::size_t i = 0; i < n; ++i)
        for (const auto j : graph.neighbors(i))
            affinity(i,j) = 1;
    return fit_spectral_clustering(affinity,options);
}

} // namespace datamunge::stats
