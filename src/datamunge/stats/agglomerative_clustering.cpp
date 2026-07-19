#include <datamunge/stats/agglomerative_clustering.hpp>

#include <datamunge/linalg/dense_matrix.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace datamunge::stats {

namespace {

// Lance-Williams recurrence for the distance between a newly merged cluster (a union b) and
// another cluster c. For Ward, operates on SQUARED distances throughout (the caller is
// responsible for reporting sqrt(distance) back to the user); for the others, on the actual
// (non-squared) distance.
double lance_williams(const LinkageCriterion linkage, const std::size_t n_a, const std::size_t n_b,
                       const std::size_t n_c, const double d_ac, const double d_bc, const double d_ab) {
    switch (linkage) {
        case LinkageCriterion::Single: return std::min(d_ac, d_bc);
        case LinkageCriterion::Complete: return std::max(d_ac, d_bc);
        case LinkageCriterion::Average: {
            const auto total = static_cast<double>(n_a + n_b);
            return (static_cast<double>(n_a) * d_ac + static_cast<double>(n_b) * d_bc) / total;
        }
        case LinkageCriterion::Ward: {
            const auto total = static_cast<double>(n_a + n_b + n_c);
            return (static_cast<double>(n_a + n_c) * d_ac + static_cast<double>(n_b + n_c) * d_bc -
                    static_cast<double>(n_c) * d_ab) /
                   total;
        }
    }
    return 0.0;
}

std::string linkage_name(const LinkageCriterion linkage) {
    switch (linkage) {
        case LinkageCriterion::Single: return "single";
        case LinkageCriterion::Complete: return "complete";
        case LinkageCriterion::Average: return "average";
        case LinkageCriterion::Ward: return "ward";
    }
    return "unknown";
}

} // namespace

AgglomerativeClustering::AgglomerativeClustering(const dstruct::DataFrame& data,
                                                  const std::vector<std::string>& feature_columns,
                                                  AgglomerativeClusteringOptions options)
    : feature_columns_(feature_columns), options_(options) {
    fit(data);
}

void AgglomerativeClustering::fit(const dstruct::DataFrame& data) {
    if (feature_columns_.empty()) throw std::invalid_argument("AgglomerativeClustering: feature_columns must not be empty");
    if (options_.linkage == LinkageCriterion::Ward && options_.metric != DistanceMetric::Euclidean)
        throw std::invalid_argument("AgglomerativeClustering: ward linkage requires Euclidean distance");
    for (const auto& col : feature_columns_) {
        if (!data.has_column(col)) throw std::invalid_argument("AgglomerativeClustering: column '" + col + "' not found");
        if (data.column_type(col) != dstruct::DataFrame::ColumnType::Numeric)
            throw std::invalid_argument("AgglomerativeClustering: column '" + col + "' must be numeric");
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
    if (options_.n_clusters == 0 || options_.n_clusters > n)
        throw std::invalid_argument("AgglomerativeClustering: n_clusters out of range");
    if (n < 2) throw std::invalid_argument("AgglomerativeClustering: need at least 2 complete observations");

    linalg::DenseMatrix<double> X(n, d, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < d; ++j) X(i, j) = data.double_at(feature_columns_[j], rows[i]);

    const bool use_squared = options_.linkage == LinkageCriterion::Ward;
    const std::size_t total = 2 * n - 1;
    linalg::DenseMatrix<double> D(total, total, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            double dist;
            if (options_.metric == DistanceMetric::Euclidean) {
                double s = 0.0;
                for (std::size_t c = 0; c < d; ++c) {
                    const double diff = X(i, c) - X(j, c);
                    s += diff * diff;
                }
                dist = use_squared ? s : std::sqrt(s);
            } else {
                double s = 0.0;
                for (std::size_t c = 0; c < d; ++c) s += std::abs(X(i, c) - X(j, c));
                dist = s;
            }
            D(i, j) = dist;
            D(j, i) = dist;
        }
    }

    std::vector<std::size_t> sizes(total, 0);
    for (std::size_t i = 0; i < n; ++i) sizes[i] = 1;
    std::vector<bool> alive(total, false);
    for (std::size_t i = 0; i < n; ++i) alive[i] = true;

    merges_.clear();
    merges_.reserve(n - 1);
    std::size_t next_id = n;
    for (std::size_t step = 0; step < n - 1; ++step) {
        std::size_t best_a = 0, best_b = 0;
        double best_d = std::numeric_limits<double>::max();
        for (std::size_t i = 0; i < next_id; ++i) {
            if (!alive[i]) continue;
            for (std::size_t j = i + 1; j < next_id; ++j) {
                if (!alive[j]) continue;
                if (D(i, j) < best_d) {
                    best_d = D(i, j);
                    best_a = i;
                    best_b = j;
                }
            }
        }

        const double reported = use_squared ? std::sqrt(std::max(0.0, best_d)) : best_d;
        merges_.push_back(Merge{best_a, best_b, reported, sizes[best_a] + sizes[best_b]});

        for (std::size_t c = 0; c < next_id; ++c) {
            if (!alive[c] || c == best_a || c == best_b) continue;
            const double d_ac = D(best_a, c);
            const double d_bc = D(best_b, c);
            const double updated =
                lance_williams(options_.linkage, sizes[best_a], sizes[best_b], sizes[c], d_ac, d_bc, best_d);
            D(next_id, c) = updated;
            D(c, next_id) = updated;
        }

        sizes[next_id] = sizes[best_a] + sizes[best_b];
        alive[best_a] = false;
        alive[best_b] = false;
        alive[next_id] = true;
        ++next_id;
    }

    observations_ = n;
    labels_ = cut(options_.n_clusters);
}

std::vector<std::size_t> AgglomerativeClustering::cut(const std::size_t n_clusters) const {
    const std::size_t n = observations_;
    if (n_clusters == 0 || n_clusters > n)
        throw std::invalid_argument("AgglomerativeClustering::cut: n_clusters out of range");
    const std::size_t m = n - n_clusters;

    std::vector<std::size_t> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    std::function<std::size_t(std::size_t)> find = [&](std::size_t x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    std::vector<std::size_t> representative(2 * n - 1);
    for (std::size_t i = 0; i < n; ++i) representative[i] = i;
    for (std::size_t j = 0; j < m; ++j) {
        const auto& merge = merges_[j];
        const std::size_t ra = find(representative[merge.cluster_a]);
        const std::size_t rb = find(representative[merge.cluster_b]);
        parent[rb] = ra;
        representative[n + j] = ra;
    }

    std::unordered_map<std::size_t, std::size_t> relabel;
    std::vector<std::size_t> labels(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t r = find(i);
        const auto it = relabel.find(r);
        if (it == relabel.end()) {
            const std::size_t new_label = relabel.size();
            relabel.emplace(r, new_label);
            labels[i] = new_label;
        } else {
            labels[i] = it->second;
        }
    }
    return labels;
}

std::string AgglomerativeClustering::summary() const {
    std::ostringstream oss;
    print_summary(oss);
    return oss.str();
}

void AgglomerativeClustering::print_summary() const { print_summary(std::cout); }

void AgglomerativeClustering::print_summary(std::ostream& os) const {
    os << "Agglomerative clustering (" << linkage_name(options_.linkage) << " linkage) with " << options_.n_clusters
       << " clusters\n";
    os << "Features: ";
    for (std::size_t j = 0; j < feature_columns_.size(); ++j) os << (j ? ", " : "") << feature_columns_[j];
    os << "\n";
    os << "Number of obs: " << observations_ << "\n\n";

    std::vector<std::size_t> counts(options_.n_clusters, 0);
    for (const auto l : labels_) ++counts[l];
    os << "Cluster sizes:\n";
    for (std::size_t c = 0; c < options_.n_clusters; ++c) os << "  cluster " << c << ": " << counts[c] << " points\n";

    os << "\nLast few merges (lowest to highest distance overall; showing the tail nearest the requested cut):\n";
    os << std::fixed << std::setprecision(4);
    const std::size_t show = std::min<std::size_t>(5, merges_.size());
    for (std::size_t i = merges_.size() - show; i < merges_.size(); ++i)
        os << "  merge " << i << ": clusters " << merges_[i].cluster_a << " + " << merges_[i].cluster_b
           << " -> size " << merges_[i].size << " at distance " << merges_[i].distance << "\n";
}

} // namespace datamunge::stats
