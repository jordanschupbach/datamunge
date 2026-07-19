#include <datamunge/stats/dbscan.hpp>

#include <datamunge/linalg/dense_matrix.hpp>

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace datamunge::stats {

DBSCAN::DBSCAN(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns, DBSCANOptions options)
    : feature_columns_(feature_columns), options_(options) {
    fit(data);
}

void DBSCAN::fit(const dstruct::DataFrame& data) {
    if (feature_columns_.empty()) throw std::invalid_argument("DBSCAN: feature_columns must not be empty");
    if (options_.eps <= 0.0) throw std::invalid_argument("DBSCAN: eps must be positive");
    if (options_.min_samples == 0) throw std::invalid_argument("DBSCAN: min_samples must be at least 1");
    for (const auto& col : feature_columns_) {
        if (!data.has_column(col)) throw std::invalid_argument("DBSCAN: column '" + col + "' not found");
        if (data.column_type(col) != dstruct::DataFrame::ColumnType::Numeric)
            throw std::invalid_argument("DBSCAN: column '" + col + "' must be numeric");
    }

    std::vector<std::size_t> rows;
    for (std::size_t i = 0; i < data.nrows(); ++i) {
        bool ok = true;
        for (const auto& col : feature_columns_) {
            if (data.is_null(col, i)) {
                ok = false;
                break;
            }
        }
        if (ok) rows.push_back(i);
    }
    const std::size_t n = rows.size();
    const std::size_t d = feature_columns_.size();
    if (n == 0) throw std::invalid_argument("DBSCAN: no complete observations to fit");

    linalg::DenseMatrix<double> X(n, d, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < d; ++j) X(i, j) = data.double_at(feature_columns_[j], rows[i]);

    linalg::DenseMatrix<double> D(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            double dist;
            if (options_.metric == DistanceMetric::Euclidean) {
                double s = 0.0;
                for (std::size_t c = 0; c < d; ++c) {
                    const double diff = X(i, c) - X(j, c);
                    s += diff * diff;
                }
                dist = std::sqrt(s);
            } else {
                double s = 0.0;
                for (std::size_t c = 0; c < d; ++c) s += std::abs(X(i, c) - X(j, c));
                dist = s;
            }
            D(i, j) = dist;
            D(j, i) = dist;
        }
    }

    const auto region_query = [&](const std::size_t p) {
        std::vector<std::size_t> neighbors;
        for (std::size_t i = 0; i < n; ++i)
            if (i == p || D(p, i) <= options_.eps) neighbors.push_back(i);
        return neighbors;
    };

    constexpr int kUndefined = -2;
    std::vector<int> label(n, kUndefined);
    int cluster_id = -1;

    for (std::size_t p = 0; p < n; ++p) {
        if (label[p] != kUndefined) continue;
        auto neighbors = region_query(p);
        if (neighbors.size() < options_.min_samples) {
            label[p] = -1; // noise; may still be reclaimed as a border point below
            continue;
        }
        ++cluster_id;
        label[p] = cluster_id;

        std::vector<std::size_t> seeds = std::move(neighbors);
        for (std::size_t idx = 0; idx < seeds.size(); ++idx) {
            const std::size_t q = seeds[idx];
            if (label[q] == -1) label[q] = cluster_id;
            if (label[q] != kUndefined) continue;
            label[q] = cluster_id;
            auto q_neighbors = region_query(q);
            if (q_neighbors.size() >= options_.min_samples)
                for (const auto nb : q_neighbors) seeds.push_back(nb);
        }
    }

    labels_ = std::move(label);
    observations_ = n;
    n_clusters_ = static_cast<std::size_t>(cluster_id + 1);
    n_noise_ = 0;
    for (const auto l : labels_)
        if (l == -1) ++n_noise_;
}

std::string DBSCAN::summary() const {
    std::ostringstream oss;
    print_summary(oss);
    return oss.str();
}

void DBSCAN::print_summary() const { print_summary(std::cout); }

void DBSCAN::print_summary(std::ostream& os) const {
    os << "DBSCAN clustering (eps=" << options_.eps << ", min_samples=" << options_.min_samples << ")\n";
    os << "Features: ";
    for (std::size_t j = 0; j < feature_columns_.size(); ++j) os << (j ? ", " : "") << feature_columns_[j];
    os << "\n";
    os << "Number of obs: " << observations_ << "\n";
    os << "Clusters found: " << n_clusters_ << "\n";
    os << "Noise points: " << n_noise_ << "\n\n";

    std::vector<std::size_t> counts(n_clusters_, 0);
    for (const auto l : labels_)
        if (l >= 0) ++counts[static_cast<std::size_t>(l)];
    os << "Cluster sizes:\n";
    for (std::size_t c = 0; c < n_clusters_; ++c) os << "  cluster " << c << ": " << counts[c] << " points\n";
}

} // namespace datamunge::stats
