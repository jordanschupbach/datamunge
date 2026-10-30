/// @file mds.cpp
/// @brief Classical scaling, spectral diagnostics, distance residuals and embedding plots.

#include <datamunge/stats/mds.hpp>

#include <datamunge/linalg/eigen.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Internal group-colour and feature-distance helpers.
namespace {

/// @brief Group colours, recycled in sorted label order.
const graphics::RGB kSeriesColors[] = {
    {37, 99, 235}, {220, 38, 38}, {22, 163, 74}, {217, 119, 6}, {124, 58, 237}, {8, 145, 178},
};

/// @brief Copies distinct labels in sorted order.
/// @param labels Group labels.
/// @return Sorted unique labels.
std::vector<std::string> unique_sorted(const std::vector<std::string>& labels) {
    std::vector<std::string> result(labels.begin(), labels.end());
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

/// @brief Distance between two numeric feature rows.
/// @param X Feature matrix.
/// @param i First row.
/// @param j Second row.
/// @param metric Euclidean or Manhattan.
/// @return Distance in feature units.
double row_distance(const linalg::DenseMatrix<double>& X, const std::size_t i, const std::size_t j,
                    const DistanceMetric metric) {
    double sum = 0.0;
    if (metric == DistanceMetric::Euclidean) {
        for (std::size_t k = 0; k < X.cols(); ++k) {
            const double d = X(i, k) - X(j, k);
            sum += d * d;
        }
        return std::sqrt(sum);
    }
    for (std::size_t k = 0; k < X.cols(); ++k) sum += std::abs(X(i, k) - X(j, k));
    return sum;
}

} // namespace

MDS::MDS(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns, MDSOptions options)
    : feature_columns_(feature_columns), options_(options) {
    fit(data);
}

void MDS::fit(const dstruct::DataFrame& data) {
    if (options_.metric != DistanceMetric::Euclidean && options_.metric != DistanceMetric::Manhattan)
        throw std::invalid_argument("MDS: metric must be Euclidean or Manhattan");
    if (feature_columns_.empty()) throw std::invalid_argument("MDS: feature_columns must not be empty");
    if (options_.n_components == 0) throw std::invalid_argument("MDS: n_components must be at least 1");
    for (const auto& col : feature_columns_) {
        if (!data.has_column(col)) throw std::invalid_argument("MDS: column '" + col + "' not found");
        if (data.column_type(col) != dstruct::DataFrame::ColumnType::Numeric)
            throw std::invalid_argument("MDS: column '" + col + "' must be numeric");
    }

    kept_row_indices_.clear();
    for (std::size_t i = 0; i < data.nrows(); ++i) {
        bool ok = true;
        for (const auto& col : feature_columns_) {
            if (data.is_null(col, i)) {
                ok = false;
                break;
            }
        }
        if (ok) kept_row_indices_.push_back(i);
    }

    const std::size_t n = kept_row_indices_.size();
    const std::size_t p = feature_columns_.size();
    if (options_.n_components >= n)
        throw std::invalid_argument("MDS: n_components must be less than the number of complete observations");

    linalg::DenseMatrix<double> X(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) {
            X(i, j) = data.double_at(feature_columns_[j], kept_row_indices_[i]);
            if (!std::isfinite(X(i,j))) throw std::invalid_argument("MDS: retained feature values must be finite");
        }

    linalg::DenseMatrix<double> distances(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) for (std::size_t j = i+1; j < n; ++j)
        distances(i,j) = distances(j,i) = row_distance(X,i,j,options_.metric);
    fit_distances(distances);
}

MDS MDS::from_distances(const linalg::DenseMatrix<double>& dissimilarities, std::size_t n_components) {
    MDS model;
    model.options_.n_components = n_components;
    model.fit_distances(dissimilarities);
    model.kept_row_indices_.resize(model.observations_);
    std::iota(model.kept_row_indices_.begin(),model.kept_row_indices_.end(),0);
    return model;
}

void MDS::fit_distances(const linalg::DenseMatrix<double>& distances) {
    const std::size_t n = distances.rows();
    if (n < 2 || distances.cols() != n)
        throw std::invalid_argument("MDS: dissimilarities must be square with at least two rows");
    if (!options_.n_components || options_.n_components >= n)
        throw std::invalid_argument("MDS: n_components must be in [1, observations-1]");
    double largest = 0.0;
    for (std::size_t i=0;i<n;++i) for (std::size_t j=0;j<n;++j) {
        const double v = distances(i,j);
        if (!std::isfinite(v) || v < 0)
            throw std::invalid_argument("MDS: dissimilarities must be finite and non-negative");
        largest = std::max(largest,v);
    }
    const double tolerance = 1e-12*largest;
    dissimilarities_ = linalg::DenseMatrix<double>(n,n,0.0);
    linalg::DenseMatrix<double> D2(n,n,0.0);
    for (std::size_t i=0;i<n;++i) {
        if (distances(i,i)>tolerance) throw std::invalid_argument("MDS: dissimilarity diagonal must be zero");
        for (std::size_t j=i+1;j<n;++j) {
            if (std::abs(distances(i,j)-distances(j,i))>tolerance)
                throw std::invalid_argument("MDS: dissimilarities must be symmetric");
            const double d = 0.5*distances(i,j)+0.5*distances(j,i);
            if (!std::isfinite(d*d)) throw std::invalid_argument("MDS: squared dissimilarities exceed finite range");
            dissimilarities_(i,j) = dissimilarities_(j,i) = d;
            D2(i,j) = D2(j,i) = d*d;
        }
    }
    // Wider accumulation avoids overflow in a row sum of otherwise finite squares.
    std::vector<long double> row_mean(n,0.0L);
    long double grand_mean = 0.0L;
    for (std::size_t i=0;i<n;++i) {
        for (std::size_t j=0;j<n;++j) row_mean[i] += D2(i,j);
        row_mean[i] /= n;
        grand_mean += row_mean[i]/n;
    }
    gram_ = linalg::DenseMatrix<double>(n,n,0.0);
    for (std::size_t i=0;i<n;++i) for (std::size_t j=i;j<n;++j) {
        const double v = static_cast<double>(-0.5L*(D2(i,j)-row_mean[i]-row_mean[j]+grand_mean));
        if (!std::isfinite(v)) throw std::invalid_argument("MDS: centred Gram entries exceed finite range");
        gram_(i,j) = gram_(j,i) = v;
    }
    const auto eig = linalg::jacobi_eigen(gram_);
    eigenvalues_ = eig.eigenvalues;

    double positive_total = 0.0;
    for (const auto v : eigenvalues_)
        if (v > 0.0) positive_total += v;

    embedding_ = linalg::DenseMatrix<double>(n, options_.n_components, 0.0);
    double used_total = 0.0;
    for (std::size_t c = 0; c < options_.n_components; ++c) {
        const double value = std::max(eigenvalues_[c], 0.0);
        used_total += value;
        const double scale = std::sqrt(value);
        for (std::size_t i = 0; i < n; ++i) embedding_(i, c) = eig.eigenvectors(i, c) * scale;
    }
    goodness_of_fit_ = positive_total > 0.0 ? used_total / positive_total : 0.0;
    observations_ = n;
}

std::vector<double> MDS::dimension(const std::size_t index) const {
    if (index >= embedding_.cols()) throw std::out_of_range("MDS::dimension index out of range");
    return embedding_.col(index);
}

std::string MDS::summary() const {
    std::ostringstream oss;
    print_summary(oss);
    return oss.str();
}

void MDS::print_summary() const { print_summary(std::cout); }

void MDS::print_summary(std::ostream& os) const {
    os << "Classical (metric) Multidimensional Scaling\n";
    os << (feature_columns_.empty() ? "Input: precomputed dissimilarities" : "Features: ");
    for (std::size_t j = 0; j < feature_columns_.size(); ++j) os << (j ? ", " : "") << feature_columns_[j];
    os << "\n";
    if (!feature_columns_.empty()) os << "Distance metric: " << (options_.metric == DistanceMetric::Euclidean ? "euclidean" : "manhattan") << "\n";
    os << "Number of obs: " << observations_ << ", dimensions: " << n_components() << "\n";
    os << std::fixed << std::setprecision(4);
    os << "Goodness of fit (positive inertia): " << goodness_of_fit_ << "\n\n";

    os << "Eigenvalues used:\n";
    for (std::size_t c = 0; c < n_components(); ++c) os << "  Dim" << (c + 1) << ": " << eigenvalues_[c] << "\n";
}

graphics::RPlot MDS::plot_embedding(const std::size_t dimension_x, const std::size_t dimension_y) const {
    if (dimension_x >= n_components() || dimension_y >= n_components())
        throw std::out_of_range("MDS::plot_embedding dimension index out of range");

    auto plot = graphics::RPlot::create();
    plot.points(embedding_.col(dimension_x), embedding_.col(dimension_y), "", kSeriesColors[0]);
    plot.title("Multidimensional Scaling")
        .x_label("Dim" + std::to_string(dimension_x + 1))
        .y_label("Dim" + std::to_string(dimension_y + 1));
    return plot;
}

graphics::RPlot MDS::plot_embedding(const std::vector<std::string>& group_labels, const std::size_t dimension_x,
                                const std::size_t dimension_y) const {
    if (dimension_x >= n_components() || dimension_y >= n_components())
        throw std::out_of_range("MDS::plot_embedding dimension index out of range");
    if (group_labels.size() != observations_)
        throw std::invalid_argument("MDS::plot_embedding: group_labels must have one entry per fitted observation");

    auto plot = graphics::RPlot::create();
    const auto groups = unique_sorted(group_labels);
    for (std::size_t g = 0; g < groups.size(); ++g) {
        std::vector<double> xs, ys;
        for (std::size_t i = 0; i < observations_; ++i) {
            if (group_labels[i] != groups[g]) continue;
            xs.push_back(embedding_(i, dimension_x));
            ys.push_back(embedding_(i, dimension_y));
        }
        plot.points(xs, ys, groups[g], kSeriesColors[g % (sizeof(kSeriesColors) / sizeof(kSeriesColors[0]))]);
    }
    plot.title("Multidimensional Scaling")
        .x_label("Dim" + std::to_string(dimension_x + 1))
        .y_label("Dim" + std::to_string(dimension_y + 1));
    return plot;
}

dstruct::DataFrame MDS::spectrum() const {
    double positive=0.0, absolute=0.0, largest=0.0;
    for (double v : eigenvalues_) {
        positive += std::max(0.0,v); absolute += std::abs(v); largest=std::max(largest,std::abs(v));
    }
    std::vector<double> dimension, pos_share, abs_share, cumulative, retained, negative;
    double running=0.0;
    for (std::size_t i=0;i<eigenvalues_.size();++i) {
        const double v=eigenvalues_[i];
        running += std::max(0.0,v);
        dimension.push_back(i+1); pos_share.push_back(positive>0 ? std::max(0.0,v)/positive : 0.0);
        abs_share.push_back(absolute>0 ? std::abs(v)/absolute : 0.0);
        cumulative.push_back(positive>0 ? running/positive : 0.0);
        retained.push_back(i<n_components() ? 1.0 : 0.0);
        negative.push_back(v < -1e-10*largest ? 1.0 : 0.0);
    }
    dstruct::DataFrame table;
    table.add_column("dimension",dimension); table.add_column("eigenvalue",eigenvalues_);
    table.add_column("positive_share",pos_share); table.add_column("absolute_share",abs_share);
    table.add_column("cumulative_positive_share",cumulative); table.add_column("retained",retained);
    table.add_column("significant_negative",negative);
    return table;
}

dstruct::DataFrame MDS::dimension_diagnostics() const {
    const std::size_t n=observations_;
    double positive=0.0, absolute=0.0;
    long double spectral_norm2=0.0L, original_norm2=0.0L;
    for (double v : eigenvalues_) {
        positive += std::max(0.0,v); absolute += std::abs(v);
        spectral_norm2 += static_cast<long double>(v)*v;
    }
    for (std::size_t i=0;i<n;++i) for (std::size_t j=i+1;j<n;++j)
        original_norm2 += static_cast<long double>(dissimilarities_(i,j))*dissimilarities_(i,j);
    linalg::DenseMatrix<double> embedded2(n,n,0.0);
    std::vector<double> dimensions,gof_positive,gof_absolute,strain,relative,rmse,stress;
    double used=0.0;
    for (std::size_t c=0;c<n_components();++c) {
        used += std::max(0.0,eigenvalues_[c]);
        long double residual=0.0L, spectral_error=0.0L;
        for (std::size_t i=0;i<n;++i) for (std::size_t j=i+1;j<n;++j) {
            const double delta=embedding_(i,c)-embedding_(j,c);
            embedded2(i,j) += delta*delta;
            const double error=std::sqrt(embedded2(i,j))-dissimilarities_(i,j);
            residual += static_cast<long double>(error)*error;
        }
        for (std::size_t i=0;i<eigenvalues_.size();++i)
            if (i>c || eigenvalues_[i]<0) spectral_error += static_cast<long double>(eigenvalues_[i])*eigenvalues_[i];
        dimensions.push_back(c+1); gof_positive.push_back(positive>0 ? used/positive : 0);
        gof_absolute.push_back(absolute>0 ? used/absolute : 0);
        strain.push_back(static_cast<double>(spectral_error));
        relative.push_back(spectral_norm2>0 ? std::sqrt(spectral_error/spectral_norm2) : 0);
        rmse.push_back(std::sqrt(residual/(static_cast<long double>(n)*(n-1)/2)));
        stress.push_back(original_norm2>0 ? std::sqrt(residual/original_norm2) : 0);
    }
    dstruct::DataFrame table;
    table.add_column("dimensions",dimensions); table.add_column("gof_positive",gof_positive);
    table.add_column("gof_absolute",gof_absolute); table.add_column("strain_squared",strain);
    table.add_column("relative_strain",relative); table.add_column("distance_rmse",rmse);
    table.add_column("relative_distance_error",stress);
    return table;
}

dstruct::DataFrame MDS::pair_distances(std::size_t dimensions) const {
    if (!dimensions) dimensions=n_components();
    if (dimensions>n_components()) throw std::out_of_range("MDS::pair_distances: too many dimensions");
    std::vector<double> row_i,row_j,original,embedded,residual;
    for (std::size_t i=0;i<observations_;++i) for (std::size_t j=i+1;j<observations_;++j) {
        double d2=0.0;
        for (std::size_t c=0;c<dimensions;++c) {
            const double delta=embedding_(i,c)-embedding_(j,c); d2 += delta*delta;
        }
        row_i.push_back(kept_row_indices_[i]); row_j.push_back(kept_row_indices_[j]);
        original.push_back(dissimilarities_(i,j)); embedded.push_back(std::sqrt(d2));
        residual.push_back(embedded.back()-original.back());
    }
    dstruct::DataFrame table;
    table.add_column("row_i",row_i); table.add_column("row_j",row_j); table.add_column("dissimilarity",original);
    table.add_column("embedded_distance",embedded); table.add_column("residual",residual);
    return table;
}

graphics::RPlot MDS::plot_shepard(std::size_t dimensions) const {
    const auto pairs=pair_distances(dimensions);
    const auto original=pairs.numeric_column("dissimilarity").values();
    const auto embedded=pairs.numeric_column("embedded_distance").values();
    const double largest=std::max(*std::max_element(original.begin(),original.end()),
                                  *std::max_element(embedded.begin(),embedded.end()));
    const double upper=largest>0 ? 1.05*largest : 1.0;
    auto plot=graphics::RPlot::create();
    plot.points(original,embedded,"",kSeriesColors[0],2.5)
        .segments({0},{0},{upper},{upper},{90,90,90},1.0)
        .x_limits(0,upper).y_limits(0,upper).x_label("Input dissimilarity").y_label("Embedded distance")
        .title("Shepard plot");
    return plot;
}

} // namespace datamunge::stats
