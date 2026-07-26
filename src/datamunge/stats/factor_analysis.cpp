#include <datamunge/stats/factor_analysis.hpp>

#include <datamunge/linalg/eigen.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace datamunge::stats {

namespace {

const plot::RGB kSeriesColors[] = {
    {37, 99, 235}, {220, 38, 38}, {22, 163, 74}, {217, 119, 6}, {124, 58, 237}, {8, 145, 178},
};

std::vector<std::string> unique_sorted(const std::vector<std::string>& labels) {
    std::vector<std::string> result(labels.begin(), labels.end());
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

// Inverse of a symmetric positive-definite matrix via its eigendecomposition:
// A^{-1} = V diag(1/lambda) V^T. Eigenvalues below `floor` are treated as `floor` for stability
// (Tikhonov-style), which keeps a near-singular correlation matrix invertible.
linalg::DenseMatrix<double> symmetric_inverse(const linalg::DenseMatrix<double>& A, double floor = 1e-8) {
    const auto eig = linalg::jacobi_eigen(A);
    const std::size_t n = A.rows();
    linalg::DenseMatrix<double> inv(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t c = 0; c < n; ++c) {
                const double lambda = std::max(eig.eigenvalues[c], floor);
                sum += eig.eigenvectors(i, c) * eig.eigenvectors(j, c) / lambda;
            }
            inv(i, j) = sum;
        }
    return inv;
}

// Kaiser-normalized varimax rotation of a p x k loading matrix, in place. Rotates every pair of
// factor columns by the angle that maximizes the varimax simple-structure criterion, sweeping until
// the criterion stops improving. A no-op when k < 2.
void varimax(linalg::DenseMatrix<double>& L, std::size_t max_iter, double tol) {
    const std::size_t p = L.rows();
    const std::size_t k = L.cols();
    if (k < 2) return;

    // Kaiser normalization: scale each row to unit length so high- and low-communality variables
    // influence the rotation equally, then restore afterward.
    std::vector<double> norm(p, 1.0);
    for (std::size_t j = 0; j < p; ++j) {
        double ss = 0.0;
        for (std::size_t c = 0; c < k; ++c) ss += L(j, c) * L(j, c);
        norm[j] = std::sqrt(ss);
        if (norm[j] > 0.0)
            for (std::size_t c = 0; c < k; ++c) L(j, c) /= norm[j];
    }

    double prev_crit = 0.0;
    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        for (std::size_t a = 0; a < k; ++a) {
            for (std::size_t b = a + 1; b < k; ++b) {
                double sum_u = 0.0, sum_v = 0.0, sum_u2mv2 = 0.0, sum_2uv = 0.0;
                for (std::size_t j = 0; j < p; ++j) {
                    const double x = L(j, a), y = L(j, b);
                    const double u = x * x - y * y;
                    const double v = 2.0 * x * y;
                    sum_u += u;
                    sum_v += v;
                    sum_u2mv2 += u * u - v * v;
                    sum_2uv += 2.0 * u * v;
                }
                const double num = sum_2uv - 2.0 * sum_u * sum_v / static_cast<double>(p);
                const double den = sum_u2mv2 - (sum_u * sum_u - sum_v * sum_v) / static_cast<double>(p);
                const double theta = 0.25 * std::atan2(num, den);
                if (std::abs(theta) < 1e-12) continue;
                const double c = std::cos(theta), s = std::sin(theta);
                for (std::size_t j = 0; j < p; ++j) {
                    const double x = L(j, a), y = L(j, b);
                    L(j, a) = c * x + s * y;
                    L(j, b) = -s * x + c * y;
                }
            }
        }

        double crit = 0.0;
        for (std::size_t c = 0; c < k; ++c) {
            double s1 = 0.0, s2 = 0.0;
            for (std::size_t j = 0; j < p; ++j) {
                const double q = L(j, c) * L(j, c);
                s1 += q;
                s2 += q * q;
            }
            crit += s2 - s1 * s1 / static_cast<double>(p);
        }
        if (iter > 0 && std::abs(crit - prev_crit) < tol) break;
        prev_crit = crit;
    }

    for (std::size_t j = 0; j < p; ++j)
        for (std::size_t c = 0; c < k; ++c) L(j, c) *= norm[j];
}

} // namespace

FactorAnalysis::FactorAnalysis(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns,
                               FactorAnalysisOptions options)
    : feature_columns_(feature_columns), options_(options) {
    fit(data);
}

void FactorAnalysis::fit(const dstruct::DataFrame& data) {
    if (feature_columns_.empty()) throw std::invalid_argument("FactorAnalysis: feature_columns must not be empty");
    if (options_.n_factors == 0) throw std::invalid_argument("FactorAnalysis: n_factors must be at least 1");
    if (options_.rotation != "none" && options_.rotation != "varimax")
        throw std::invalid_argument("FactorAnalysis: rotation must be 'none' or 'varimax'");
    for (const auto& col : feature_columns_) {
        if (!data.has_column(col)) throw std::invalid_argument("FactorAnalysis: column '" + col + "' not found");
        if (data.column_type(col) != dstruct::DataFrame::ColumnType::Numeric)
            throw std::invalid_argument("FactorAnalysis: column '" + col + "' must be numeric");
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
    const std::size_t k = options_.n_factors;
    if (n < 2) throw std::invalid_argument("FactorAnalysis: fewer than 2 complete observations");
    if (k > p) throw std::invalid_argument("FactorAnalysis: n_factors must not exceed the number of features");

    // Center (and, by default, standardize) the data.
    linalg::DenseMatrix<double> Xc(n, p, 0.0);
    std::vector<double> mean(p, 0.0), scale(p, 1.0);
    for (std::size_t j = 0; j < p; ++j) {
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) sum += data.double_at(feature_columns_[j], kept_row_indices_[i]);
        mean[j] = sum / static_cast<double>(n);
    }
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j)
            Xc(i, j) = data.double_at(feature_columns_[j], kept_row_indices_[i]) - mean[j];
    if (options_.standardize) {
        for (std::size_t j = 0; j < p; ++j) {
            double ss = 0.0;
            for (std::size_t i = 0; i < n; ++i) ss += Xc(i, j) * Xc(i, j);
            const double sd = std::sqrt(ss / static_cast<double>(n - 1));
            scale[j] = sd > 0.0 ? sd : 1.0;
        }
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < p; ++j) Xc(i, j) /= scale[j];
    }

    // Observed covariance / correlation matrix C (correlation when standardized).
    linalg::DenseMatrix<double> C(p, p, 0.0);
    for (std::size_t a = 0; a < p; ++a)
        for (std::size_t b = a; b < p; ++b) {
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) sum += Xc(i, a) * Xc(i, b);
            const double value = sum / static_cast<double>(n - 1);
            C(a, b) = value;
            C(b, a) = value;
        }

    const double total_variance = [&] {
        double t = 0.0;
        for (std::size_t j = 0; j < p; ++j) t += C(j, j);
        return t;
    }();

    // Initial communality estimate: squared multiple correlation, h_j^2 = 1 - 1/(C^{-1})_{jj}
    // (relative to that variable's own variance), the standard PAF starting point.
    const auto Cinv = symmetric_inverse(C);
    std::vector<double> comm(p, 0.0);
    for (std::size_t j = 0; j < p; ++j) {
        const double smc = 1.0 - 1.0 / (Cinv(j, j) * C(j, j));
        comm[j] = std::clamp(smc, 0.0, C(j, j));
    }

    // Principal axis factoring: repeatedly eigendecompose C with its diagonal replaced by the
    // current communalities, rebuild the loadings from the top-k eigenpairs, and refresh the
    // communalities as the loadings' row sums of squares, until they stop moving.
    linalg::DenseMatrix<double> L(p, k, 0.0);
    converged_ = false;
    iterations_ = 0;
    for (std::size_t iter = 0; iter < options_.max_iter; ++iter) {
        linalg::DenseMatrix<double> reduced = C;
        for (std::size_t j = 0; j < p; ++j) reduced(j, j) = comm[j];

        const auto eig = linalg::jacobi_eigen(reduced);
        for (std::size_t c = 0; c < k; ++c) {
            const double lambda = std::max(eig.eigenvalues[c], 0.0);
            const double s = std::sqrt(lambda);
            for (std::size_t j = 0; j < p; ++j) L(j, c) = eig.eigenvectors(j, c) * s;
        }

        double max_change = 0.0;
        for (std::size_t j = 0; j < p; ++j) {
            double h2 = 0.0;
            for (std::size_t c = 0; c < k; ++c) h2 += L(j, c) * L(j, c);
            h2 = std::clamp(h2, 0.0, C(j, j));  // guard Heywood cases
            max_change = std::max(max_change, std::abs(h2 - comm[j]));
            comm[j] = h2;
        }
        iterations_ = iter + 1;
        if (max_change < options_.tol) {
            converged_ = true;
            break;
        }
    }

    // Optional varimax rotation toward simple structure (leaves communalities and fit unchanged).
    if (options_.rotation == "varimax") varimax(L, options_.max_iter, options_.tol);

    // Order the (rotated) factors by descending variance explained, for a stable, PCA-like report.
    std::vector<double> ss(k, 0.0);
    for (std::size_t c = 0; c < k; ++c)
        for (std::size_t j = 0; j < p; ++j) ss[c] += L(j, c) * L(j, c);
    std::vector<std::size_t> order(k);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return ss[a] > ss[b]; });

    loadings_ = linalg::DenseMatrix<double>(p, k, 0.0);
    variance_explained_.assign(k, 0.0);
    for (std::size_t c = 0; c < k; ++c) {
        // Fix each factor's sign so its largest-magnitude loading is positive (a convention).
        std::size_t argmax = 0;
        double best = 0.0;
        for (std::size_t j = 0; j < p; ++j)
            if (std::abs(L(j, order[c])) > best) {
                best = std::abs(L(j, order[c]));
                argmax = j;
            }
        const double sign = (L(argmax, order[c]) < 0.0) ? -1.0 : 1.0;
        for (std::size_t j = 0; j < p; ++j) loadings_(j, c) = sign * L(j, order[c]);
        variance_explained_[c] = ss[order[c]];
    }

    communalities_.assign(p, 0.0);
    uniquenesses_.assign(p, 0.0);
    for (std::size_t j = 0; j < p; ++j) {
        double h2 = 0.0;
        for (std::size_t c = 0; c < k; ++c) h2 += loadings_(j, c) * loadings_(j, c);
        communalities_[j] = h2;
        uniquenesses_[j] = std::max(C(j, j) - h2, 0.0);
    }

    // Regression (Thomson) factor scores: F = Z C^{-1} Lambda.
    linalg::DenseMatrix<double> B = Cinv * loadings_;  // p x k
    scores_ = Xc * B;                                   // n x k
    observations_ = n;

    (void)total_variance;
}

std::vector<double> FactorAnalysis::factor_loadings(const std::size_t factor_index) const {
    if (factor_index >= loadings_.cols()) throw std::out_of_range("FactorAnalysis::factor_loadings index out of range");
    return loadings_.col(factor_index);
}

std::vector<double> FactorAnalysis::factor_scores(const std::size_t factor_index) const {
    if (factor_index >= scores_.cols()) throw std::out_of_range("FactorAnalysis::factor_scores index out of range");
    return scores_.col(factor_index);
}

std::vector<double> FactorAnalysis::proportion_variance() const {
    double total = 0.0;
    for (std::size_t j = 0; j < feature_columns_.size(); ++j) total += options_.standardize ? 1.0 : 0.0;
    // For the correlation matrix total variance == p; for covariance, use communalities + uniqueness.
    if (!options_.standardize) {
        total = 0.0;
        for (std::size_t j = 0; j < communalities_.size(); ++j) total += communalities_[j] + uniquenesses_[j];
    }
    std::vector<double> out(variance_explained_.size(), 0.0);
    if (total <= 0.0) return out;
    for (std::size_t c = 0; c < variance_explained_.size(); ++c) out[c] = variance_explained_[c] / total;
    return out;
}

std::string FactorAnalysis::summary() const {
    std::ostringstream oss;
    print_summary(oss);
    return oss.str();
}

void FactorAnalysis::print_summary() const { print_summary(std::cout); }

void FactorAnalysis::print_summary(std::ostream& os) const {
    os << "Factor Analysis (principal axis factoring";
    if (options_.rotation != "none") os << ", " << options_.rotation << " rotation";
    os << ")\n";
    os << "Number of obs: " << observations_ << ", factors: " << num_factors() << "\n";
    os << (options_.standardize ? "Fit on correlation matrix\n" : "Fit on covariance matrix\n");
    os << "Communalities " << (converged_ ? "converged" : "did NOT converge") << " in " << iterations_
       << " iterations\n";

    os << std::fixed << std::setprecision(4);
    const std::size_t k = num_factors();
    os << "\nLoadings:\n" << std::left << std::setw(16) << "";
    for (std::size_t c = 0; c < k; ++c) os << std::right << std::setw(10) << ("Factor" + std::to_string(c + 1));
    os << std::right << std::setw(12) << "Communality" << std::setw(11) << "Uniqueness\n";
    for (std::size_t j = 0; j < feature_columns_.size(); ++j) {
        os << std::left << std::setw(16) << feature_columns_[j];
        for (std::size_t c = 0; c < k; ++c) os << std::right << std::setw(10) << loadings_(j, c);
        os << std::right << std::setw(12) << communalities_[j] << std::setw(11) << uniquenesses_[j] << "\n";
    }

    const auto prop = proportion_variance();
    os << "\n" << std::left << std::setw(16) << "SS loadings";
    for (std::size_t c = 0; c < k; ++c) os << std::right << std::setw(10) << variance_explained_[c];
    os << "\n" << std::left << std::setw(16) << "Prop. variance";
    for (std::size_t c = 0; c < k; ++c) os << std::right << std::setw(10) << prop[c];
    os << "\n";
}

plot::RPlot FactorAnalysis::plot_scores(const std::vector<std::string>& group_labels, const std::size_t factor_x,
                                        const std::size_t factor_y) const {
    if (factor_x >= num_factors() || factor_y >= num_factors())
        throw std::out_of_range("FactorAnalysis::plot_scores factor index out of range");
    if (group_labels.size() != observations_)
        throw std::invalid_argument("FactorAnalysis::plot_scores: group_labels must have one entry per fitted observation");

    auto plot = plot::RPlot::create();
    const auto groups = unique_sorted(group_labels);
    for (std::size_t g = 0; g < groups.size(); ++g) {
        std::vector<double> xs, ys;
        for (std::size_t i = 0; i < observations_; ++i) {
            if (group_labels[i] != groups[g]) continue;
            xs.push_back(scores_(i, factor_x));
            ys.push_back(scores_(i, factor_y));
        }
        plot.points(xs, ys, groups[g], kSeriesColors[g % (sizeof(kSeriesColors) / sizeof(kSeriesColors[0]))]);
    }
    plot.title("Factor Analysis Scores")
        .x_label("Factor " + std::to_string(factor_x + 1))
        .y_label("Factor " + std::to_string(factor_y + 1));
    return plot;
}

plot::RPlot FactorAnalysis::plot_loadings(const std::size_t factor_x, const std::size_t factor_y) const {
    if (factor_x >= num_factors() || factor_y >= num_factors())
        throw std::out_of_range("FactorAnalysis::plot_loadings factor index out of range");

    auto plot = plot::RPlot::create();
    // Unit circle for reference (communality == 1 lies on it).
    std::vector<double> cx, cy;
    for (std::size_t t = 0; t <= 100; ++t) {
        const double a = 2.0 * M_PI * static_cast<double>(t) / 100.0;
        cx.push_back(std::cos(a));
        cy.push_back(std::sin(a));
    }
    plot.lines(cx, cy, "", {203, 213, 225});
    plot.abline_h(0.0, {203, 213, 225});
    plot.abline_v(0.0, {203, 213, 225});

    std::vector<double> lx, ly;
    for (std::size_t j = 0; j < feature_columns_.size(); ++j) {
        lx.push_back(loadings_(j, factor_x));
        ly.push_back(loadings_(j, factor_y));
    }
    plot.points(lx, ly, "", kSeriesColors[0], 6.0);
    for (std::size_t j = 0; j < feature_columns_.size(); ++j)
        plot.text(loadings_(j, factor_x), loadings_(j, factor_y) + 0.04, feature_columns_[j], {17, 24, 39}, 12.0);

    plot.title("Factor Loadings")
        .x_label("Factor " + std::to_string(factor_x + 1))
        .y_label("Factor " + std::to_string(factor_y + 1));
    return plot;
}

} // namespace datamunge::stats
