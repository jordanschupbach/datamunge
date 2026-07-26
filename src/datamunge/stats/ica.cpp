#include <datamunge/stats/ica.hpp>

#include <datamunge/linalg/eigen.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
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

// A contrast's first derivative g and its own derivative g', evaluated together.
struct ContrastValue {
    double g;
    double gp;
};
using Contrast = ContrastValue (*)(double);

ContrastValue logcosh_contrast(double u) {
    const double t = std::tanh(u);
    return {t, 1.0 - t * t};
}
ContrastValue exp_contrast(double u) {
    const double e = std::exp(-0.5 * u * u);
    return {u * e, (1.0 - u * u) * e};
}
ContrastValue cube_contrast(double u) { return {u * u * u, 3.0 * u * u}; }

} // namespace

ICA::ICA(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns, ICAOptions options)
    : feature_columns_(feature_columns), options_(options) {
    fit(data);
}

void ICA::fit(const dstruct::DataFrame& data) {
    if (feature_columns_.empty()) throw std::invalid_argument("ICA: feature_columns must not be empty");
    if (options_.n_components == 0) throw std::invalid_argument("ICA: n_components must be at least 1");
    Contrast contrast = nullptr;
    if (options_.contrast == "logcosh")
        contrast = logcosh_contrast;
    else if (options_.contrast == "exp")
        contrast = exp_contrast;
    else if (options_.contrast == "cube")
        contrast = cube_contrast;
    else
        throw std::invalid_argument("ICA: contrast must be 'logcosh', 'exp', or 'cube'");
    for (const auto& col : feature_columns_) {
        if (!data.has_column(col)) throw std::invalid_argument("ICA: column '" + col + "' not found");
        if (data.column_type(col) != dstruct::DataFrame::ColumnType::Numeric)
            throw std::invalid_argument("ICA: column '" + col + "' must be numeric");
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
    const std::size_t m = options_.n_components;
    if (n < 2) throw std::invalid_argument("ICA: fewer than 2 complete observations");
    if (m > p) throw std::invalid_argument("ICA: n_components must not exceed the number of features");

    // Center each feature.
    linalg::DenseMatrix<double> Xc(n, p, 0.0);
    mean_.assign(p, 0.0);
    for (std::size_t j = 0; j < p; ++j) {
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) sum += data.double_at(feature_columns_[j], kept_row_indices_[i]);
        mean_[j] = sum / static_cast<double>(n);
    }
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j)
            Xc(i, j) = data.double_at(feature_columns_[j], kept_row_indices_[i]) - mean_[j];

    // Covariance and its eigendecomposition, for PCA whitening.
    linalg::DenseMatrix<double> C(p, p, 0.0);
    for (std::size_t a = 0; a < p; ++a)
        for (std::size_t b = a; b < p; ++b) {
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) sum += Xc(i, a) * Xc(i, b);
            const double value = sum / static_cast<double>(n - 1);
            C(a, b) = value;
            C(b, a) = value;
        }
    const auto eig = linalg::jacobi_eigen(C);
    for (std::size_t c = 0; c < m; ++c)
        if (eig.eigenvalues[c] <= 1e-12)
            throw std::invalid_argument("ICA: whitening failed -- fewer than n_components non-degenerate dimensions");

    // Whitening matrix Kw (p x m), Kw[:,c] = E[:,c] / sqrt(lambda_c); Z = Xc Kw has identity
    // covariance. (Scaling by sqrt(n-1) is absorbed since the sources are reported standardized.)
    linalg::DenseMatrix<double> Kw(p, m, 0.0);
    for (std::size_t j = 0; j < p; ++j)
        for (std::size_t c = 0; c < m; ++c) Kw(j, c) = eig.eigenvectors(j, c) / std::sqrt(eig.eigenvalues[c]);
    linalg::DenseMatrix<double> Z = Xc * Kw;  // n x m, unit-variance whitened data

    // Deflationary FastICA: extract one unit-norm direction w at a time, orthogonalizing each
    // against those already found.
    std::mt19937_64 rng(options_.seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::vector<std::vector<double>> W(m, std::vector<double>(m, 0.0));
    iterations_.assign(m, 0);
    converged_ = true;

    std::vector<double> y(n), w(m), w_new(m);
    for (std::size_t c = 0; c < m; ++c) {
        for (std::size_t j = 0; j < m; ++j) w[j] = gauss(rng);
        double norm = 0.0;
        for (std::size_t j = 0; j < m; ++j) norm += w[j] * w[j];
        norm = std::sqrt(norm);
        for (std::size_t j = 0; j < m; ++j) w[j] /= norm;

        bool component_converged = false;
        std::size_t it = 0;
        for (; it < options_.max_iter; ++it) {
            for (std::size_t i = 0; i < n; ++i) {
                double s = 0.0;
                for (std::size_t j = 0; j < m; ++j) s += Z(i, j) * w[j];
                y[i] = s;
            }
            std::vector<double> mean_g(m, 0.0);
            double mean_gp = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const ContrastValue cv = contrast(y[i]);
                for (std::size_t j = 0; j < m; ++j) mean_g[j] += Z(i, j) * cv.g;
                mean_gp += cv.gp;
            }
            for (std::size_t j = 0; j < m; ++j) mean_g[j] /= static_cast<double>(n);
            mean_gp /= static_cast<double>(n);
            for (std::size_t j = 0; j < m; ++j) w_new[j] = mean_g[j] - mean_gp * w[j];

            // Gram-Schmidt deflation against previously extracted components.
            for (std::size_t prev = 0; prev < c; ++prev) {
                double dot = 0.0;
                for (std::size_t j = 0; j < m; ++j) dot += w_new[j] * W[prev][j];
                for (std::size_t j = 0; j < m; ++j) w_new[j] -= dot * W[prev][j];
            }
            double nn = 0.0;
            for (std::size_t j = 0; j < m; ++j) nn += w_new[j] * w_new[j];
            nn = std::sqrt(nn);
            for (std::size_t j = 0; j < m; ++j) w_new[j] /= nn;

            double conv = 0.0;
            for (std::size_t j = 0; j < m; ++j) conv += w_new[j] * w[j];
            w = w_new;
            if (std::abs(conv) > 1.0 - options_.tol) {
                component_converged = true;
                ++it;
                break;
            }
        }
        iterations_[c] = it;
        if (!component_converged) converged_ = false;
        W[c] = w;
    }

    // Recover the sources S = Z W^T (n x m), fix each component's arbitrary sign so its
    // largest-magnitude value is positive, and record excess kurtosis.
    sources_ = linalg::DenseMatrix<double>(n, m, 0.0);
    std::vector<double> sign(m, 1.0);
    for (std::size_t c = 0; c < m; ++c) {
        double max_abs = 0.0, at = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < m; ++j) s += Z(i, j) * W[c][j];
            sources_(i, c) = s;
            if (std::abs(s) > max_abs) {
                max_abs = std::abs(s);
                at = s;
            }
        }
        sign[c] = (at < 0.0) ? -1.0 : 1.0;
        if (sign[c] < 0.0)
            for (std::size_t i = 0; i < n; ++i) sources_(i, c) = -sources_(i, c);
    }

    source_kurtosis_.assign(m, 0.0);
    for (std::size_t c = 0; c < m; ++c) {
        double m2 = 0.0, m4 = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double s = sources_(i, c);
            m2 += s * s;
            m4 += s * s * s * s;
        }
        m2 /= static_cast<double>(n);
        m4 /= static_cast<double>(n);
        source_kurtosis_[c] = (m2 > 0.0) ? m4 / (m2 * m2) - 3.0 : 0.0;
    }

    // Unmixing matrix U (m x p): source row = U (x - mean). U = W Kw^T, with the sign fix applied.
    unmixing_ = linalg::DenseMatrix<double>(m, p, 0.0);
    for (std::size_t c = 0; c < m; ++c)
        for (std::size_t l = 0; l < p; ++l) {
            double s = 0.0;
            for (std::size_t j = 0; j < m; ++j) s += Kw(l, j) * W[c][j];
            unmixing_(c, l) = sign[c] * s;
        }

    // Mixing matrix A (p x m), the pseudo-inverse of the whitening+rotation: A = E_m D_m^{1/2} W^T.
    mixing_ = linalg::DenseMatrix<double>(p, m, 0.0);
    for (std::size_t l = 0; l < p; ++l)
        for (std::size_t c = 0; c < m; ++c) {
            double s = 0.0;
            for (std::size_t j = 0; j < m; ++j)
                s += eig.eigenvectors(l, j) * std::sqrt(eig.eigenvalues[j]) * W[c][j];
            mixing_(l, c) = sign[c] * s;
        }

    observations_ = n;
}

std::vector<double> ICA::component(const std::size_t index) const {
    if (index >= sources_.cols()) throw std::out_of_range("ICA::component index out of range");
    return sources_.col(index);
}

linalg::DenseMatrix<double> ICA::transform(const dstruct::DataFrame& newdata) const {
    for (const auto& col : feature_columns_)
        if (!newdata.has_column(col)) throw std::invalid_argument("ICA::transform: column '" + col + "' not found");
    const std::size_t mrows = newdata.nrows();
    const std::size_t p = feature_columns_.size();
    const std::size_t k = n_components();
    linalg::DenseMatrix<double> out(mrows, k, 0.0);
    for (std::size_t i = 0; i < mrows; ++i)
        for (std::size_t c = 0; c < k; ++c) {
            double s = 0.0;
            for (std::size_t l = 0; l < p; ++l) {
                if (newdata.is_null(feature_columns_[l], i))
                    throw std::invalid_argument("ICA::transform: null value in feature column '" + feature_columns_[l] +
                                                "'");
                s += unmixing_(c, l) * (newdata.double_at(feature_columns_[l], i) - mean_[l]);
            }
            out(i, c) = s;
        }
    return out;
}

std::string ICA::summary() const {
    std::ostringstream oss;
    print_summary(oss);
    return oss.str();
}

void ICA::print_summary() const { print_summary(std::cout); }

void ICA::print_summary(std::ostream& os) const {
    os << "Independent Component Analysis (FastICA, " << options_.contrast << " contrast)\n";
    os << "Number of obs: " << observations_ << ", components: " << n_components() << "\n";
    os << "PCA-whitened to " << n_components() << " dimensions\n";
    os << (converged_ ? "All components converged\n" : "At least one component did NOT converge\n");
    os << std::fixed << std::setprecision(4);
    os << "\n" << std::left << std::setw(8) << "IC" << std::right << std::setw(12) << "iterations" << std::setw(18)
       << "excess kurtosis\n";
    for (std::size_t c = 0; c < n_components(); ++c)
        os << std::left << std::setw(8) << ("IC" + std::to_string(c + 1)) << std::right << std::setw(12)
           << iterations_[c] << std::setw(18) << source_kurtosis_[c] << "\n";
}

plot::RPlot ICA::plot_sources(const std::vector<std::string>& group_labels, const std::size_t component_x,
                              const std::size_t component_y) const {
    if (component_x >= n_components() || component_y >= n_components())
        throw std::out_of_range("ICA::plot_sources component index out of range");
    if (group_labels.size() != observations_)
        throw std::invalid_argument("ICA::plot_sources: group_labels must have one entry per fitted observation");

    auto plot = plot::RPlot::create();
    const auto groups = unique_sorted(group_labels);
    for (std::size_t g = 0; g < groups.size(); ++g) {
        std::vector<double> xs, ys;
        for (std::size_t i = 0; i < observations_; ++i) {
            if (group_labels[i] != groups[g]) continue;
            xs.push_back(sources_(i, component_x));
            ys.push_back(sources_(i, component_y));
        }
        plot.points(xs, ys, groups[g], kSeriesColors[g % (sizeof(kSeriesColors) / sizeof(kSeriesColors[0]))]);
    }
    plot.title("Independent Component Analysis")
        .x_label("IC" + std::to_string(component_x + 1))
        .y_label("IC" + std::to_string(component_y + 1));
    return plot;
}

} // namespace datamunge::stats
