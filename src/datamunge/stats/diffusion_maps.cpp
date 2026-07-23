#include <datamunge/stats/diffusion_maps.hpp>

#include <datamunge/linalg/eigen.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
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

double row_distance_squared(const linalg::DenseMatrix<double>& X, const std::size_t i, const std::size_t j) {
    double sum = 0.0;
    for (std::size_t k = 0; k < X.cols(); ++k) {
        const double d = X(i, k) - X(j, k);
        sum += d * d;
    }
    return sum;
}

// Sign-preserving fractional power: lambda^t for possibly-negative lambda and non-integer t.
double signed_pow(const double lambda, const double t) {
    if (lambda < 0.0) return -std::pow(-lambda, t);
    return std::pow(lambda, t);
}

} // namespace

DiffusionMaps::DiffusionMaps(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns,
                             DiffusionMapsOptions options)
    : feature_columns_(feature_columns), options_(options) {
    fit(data);
}

void DiffusionMaps::fit(const dstruct::DataFrame& data) {
    if (feature_columns_.empty()) throw std::invalid_argument("DiffusionMaps: feature_columns must not be empty");
    if (options_.n_components == 0) throw std::invalid_argument("DiffusionMaps: n_components must be at least 1");
    if (options_.heat_kernel_epsilon <= 0.0)
        throw std::invalid_argument("DiffusionMaps: heat_kernel_epsilon must be positive");
    if (options_.alpha < 0.0 || options_.alpha > 1.0)
        throw std::invalid_argument("DiffusionMaps: alpha must be in [0, 1]");
    for (const auto& col : feature_columns_) {
        if (!data.has_column(col)) throw std::invalid_argument("DiffusionMaps: column '" + col + "' not found");
        if (data.column_type(col) != dstruct::DataFrame::ColumnType::Numeric)
            throw std::invalid_argument("DiffusionMaps: column '" + col + "' must be numeric");
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
    // One eigenvalue/eigenvector pair (the trivial ~1 one) is always discarded, so we need
    // n_components() usable pairs beyond it.
    if (n < 3 || options_.n_components >= n - 1)
        throw std::invalid_argument(
            "DiffusionMaps: n_components must be less than (number of complete observations - 1)");

    linalg::DenseMatrix<double> X(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) X(i, j) = data.double_at(feature_columns_[j], kept_row_indices_[i]);

    // Dense all-pairs squared Euclidean distance matrix (hardcoded Euclidean, per the algorithm's
    // standard literature definition).
    linalg::DenseMatrix<double> D2(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double d2 = row_distance_squared(X, i, j);
            D2(i, j) = d2;
            D2(j, i) = d2;
        }
    }

    // Dense heat kernel: K(i,j) = exp(-D2(i,j) / epsilon). K(i,i) = exp(0) = 1 automatically.
    linalg::DenseMatrix<double> K(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) K(i, j) = std::exp(-D2(i, j) / options_.heat_kernel_epsilon);

    // Alpha-normalization: removes the influence of sampling density.
    std::vector<double> q(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) sum += K(i, j);
        q[i] = sum;
    }
    std::vector<double> q_pow_alpha(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) q_pow_alpha[i] = std::pow(q[i], options_.alpha);

    linalg::DenseMatrix<double> Kp(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) Kp(i, j) = K(i, j) / (q_pow_alpha[i] * q_pow_alpha[j]);

    // Row-normalize into a (generally non-symmetric) row-stochastic Markov matrix P; d(i) is its
    // row-sum normalizer.
    std::vector<double> d(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) sum += Kp(i, j);
        d[i] = sum;
    }

    // Symmetrize for eigendecomposition: P_sym(i,j) = Kp(i,j) / (sqrt(d(i)) * sqrt(d(j))).
    // P_sym has the same eigenvalues as P; if v is an eigenvector of P_sym, D^{-1/2} v is the
    // corresponding right eigenvector of P.
    std::vector<double> sqrt_d(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) sqrt_d[i] = std::sqrt(d[i]);

    linalg::DenseMatrix<double> Psym(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) Psym(i, j) = Kp(i, j) / (sqrt_d[i] * sqrt_d[j]);

    const auto eig = linalg::jacobi_eigen(Psym);
    eigenvalues_ = eig.eigenvalues;

    // Discard the top (trivial, ~1) eigenpair; use the next n_components pairs.
    embedding_ = linalg::DenseMatrix<double>(n, options_.n_components, 0.0);
    for (std::size_t c = 0; c < options_.n_components; ++c) {
        const std::size_t eig_index = c + 1;
        const double lambda = eigenvalues_[eig_index];
        const double scale = signed_pow(lambda, options_.diffusion_time);
        for (std::size_t i = 0; i < n; ++i) {
            const double psi_k_i = eig.eigenvectors(i, eig_index) / sqrt_d[i];
            embedding_(i, c) = scale * psi_k_i;
        }
    }

    observations_ = n;
}

std::vector<double> DiffusionMaps::dimension(const std::size_t index) const {
    if (index >= embedding_.cols()) throw std::out_of_range("DiffusionMaps::dimension index out of range");
    return embedding_.col(index);
}

std::string DiffusionMaps::summary() const {
    std::ostringstream oss;
    print_summary(oss);
    return oss.str();
}

void DiffusionMaps::print_summary() const { print_summary(std::cout); }

void DiffusionMaps::print_summary(std::ostream& os) const {
    os << "Diffusion Maps\n";
    os << "Features: ";
    for (std::size_t j = 0; j < feature_columns_.size(); ++j) os << (j ? ", " : "") << feature_columns_[j];
    os << "\n";
    os << "Number of obs: " << observations_ << ", dimensions: " << n_components() << "\n";
    os << std::fixed << std::setprecision(4);
    os << "Heat kernel epsilon: " << options_.heat_kernel_epsilon << ", alpha: " << options_.alpha
       << ", diffusion time: " << options_.diffusion_time << "\n\n";

    os << "Eigenvalues of symmetrized Markov matrix (top is trivial, discarded):\n";
    for (std::size_t c = 0; c < eigenvalues_.size() && c < n_components() + 1; ++c)
        os << "  lambda" << c << ": " << eigenvalues_[c] << (c == 0 ? " (trivial, discarded)\n" : "\n");
}

plot::RPlot DiffusionMaps::plot_embedding(const std::size_t dimension_x, const std::size_t dimension_y) const {
    if (dimension_x >= n_components() || dimension_y >= n_components())
        throw std::out_of_range("DiffusionMaps::plot_embedding dimension index out of range");

    auto plot = plot::RPlot::create();
    plot.points(embedding_.col(dimension_x), embedding_.col(dimension_y), "", kSeriesColors[0]);
    plot.title("Diffusion Maps")
        .x_label("Dim" + std::to_string(dimension_x + 1))
        .y_label("Dim" + std::to_string(dimension_y + 1));
    return plot;
}

plot::RPlot DiffusionMaps::plot_embedding(const std::vector<std::string>& group_labels, const std::size_t dimension_x,
                                          const std::size_t dimension_y) const {
    if (dimension_x >= n_components() || dimension_y >= n_components())
        throw std::out_of_range("DiffusionMaps::plot_embedding dimension index out of range");
    if (group_labels.size() != observations_)
        throw std::invalid_argument(
            "DiffusionMaps::plot_embedding: group_labels must have one entry per fitted observation");

    auto plot = plot::RPlot::create();
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
    plot.title("Diffusion Maps")
        .x_label("Dim" + std::to_string(dimension_x + 1))
        .y_label("Dim" + std::to_string(dimension_y + 1));
    return plot;
}

} // namespace datamunge::stats
