/// @file distance_correlation.cpp
/// @brief Double-centred (V) and U-centred distance matrices, estimates and tests.
#include <datamunge/stats/distance_correlation.hpp>

#include <datamunge/random/distributions.hpp>
#include <datamunge/random/random.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

namespace datamunge::stats {

/// @brief Implementation helpers for distance correlation.
namespace {
/// @brief Dense matrix alias.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Pairwise Euclidean distances raised to a power, as a row-major n x n array.
/// @param x Observations in rows.
/// @param index Exponent.
/// @return Distances.
std::vector<double> distances(const Matrix& x, double index) {
    const std::size_t n = x.rows(), p = x.cols();
    std::vector<double> d(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j) {
            double s = 0;
            for (std::size_t k = 0; k < p; ++k) {
                const double t = x(i, k) - x(j, k);
                s += t * t;
            }
            const double v = index == 1.0 ? std::sqrt(s) : std::pow(s, index / 2);
            d[i * n + j] = d[j * n + i] = v;
        }
    return d;
}

/// @brief Double-centres a distance matrix in place: \f$a_{ij}-\bar a_{i\cdot}-\bar a_{\cdot j}+\bar a\f$.
/// @param d Distances (row-major, n x n).
/// @param n Size.
void double_centre(std::vector<double>& d, std::size_t n) {
    std::vector<double> row(n, 0.0);
    double all = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j)
            row[i] += d[i * n + j];
        all += row[i];
        row[i] /= static_cast<double>(n);
    }
    all /= static_cast<double>(n * n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            d[i * n + j] += all - row[i] - row[j];
}

/// @brief U-centres a distance matrix in place (Székely and Rizzo 2014), with a zero diagonal.
/// @param d Distances (row-major, n x n), n >= 4.
/// @param n Size.
void u_centre(std::vector<double>& d, std::size_t n) {
    std::vector<double> row(n, 0.0);
    double all = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j)
            row[i] += d[i * n + j];
        all += row[i];
    }
    const double nd = static_cast<double>(n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            d[i * n + j] = i == j ? 0.0
                                  : d[i * n + j] - row[i] / (nd - 2) - row[j] / (nd - 2) +
                                        all / ((nd - 1) * (nd - 2));
}

/// @brief U-centred inner product \f$\frac{1}{n(n-3)}\sum_{i\ne j}A_{ij}B_{ij}\f$.
/// @param a First U-centred matrix.
/// @param b Second.
/// @param n Size.
/// @return Inner product.
double u_inner(const std::vector<double>& a, const std::vector<double>& b, std::size_t n) {
    double s = 0;
    for (std::size_t k = 0; k < n * n; ++k)
        s += a[k] * b[k];
    return s / (static_cast<double>(n) * (static_cast<double>(n) - 3));
}

/// @brief Validates inputs and the exponent.
/// @param x First.
/// @param y Second.
/// @param index Exponent.
/// @param fn Caller name.
void validate(const Matrix& x, const Matrix& y, double index, const char* fn) {
    if (x.rows() != y.rows())
        throw std::invalid_argument(std::string(fn) + ": samples need the same number of rows");
    if (x.rows() < 4)
        throw std::invalid_argument(std::string(fn) + ": need at least 4 observations");
    if (!(index > 0 && index < 2))
        throw std::invalid_argument(std::string(fn) + ": index must be in (0, 2)");
    for (const Matrix* m : {&x, &y})
        for (std::size_t i = 0; i < m->rows(); ++i)
            for (std::size_t j = 0; j < m->cols(); ++j)
                if (!std::isfinite((*m)(i, j)))
                    throw std::invalid_argument(std::string(fn) + ": values must be finite");
}

/// @brief Bias-corrected distance correlation from U-centred matrices.
/// @param a U-centred x distances.
/// @param b U-centred y distances.
/// @param n Size.
/// @return \f$\mathcal R^*\f$ (0 if either variance is not positive).
double bias_corrected(const std::vector<double>& a, const std::vector<double>& b, std::size_t n) {
    const double aa = u_inner(a, a, n), bb = u_inner(b, b, n);
    return aa > 0 && bb > 0 ? u_inner(a, b, n) / std::sqrt(aa * bb) : 0.0;
}
} // namespace

DistanceCorrelation distance_correlation(const linalg::DenseMatrix<double>& x, const linalg::DenseMatrix<double>& y,
                                         const DistanceCorrelationOptions& options) {
    validate(x, y, options.index, "distance_correlation");
    const std::size_t n = x.rows();
    const double nn = static_cast<double>(n) * static_cast<double>(n);
    const auto dx = distances(x, options.index), dy = distances(y, options.index);
    auto a = dx, b = dy;
    double_centre(a, n), double_centre(b, n);
    double sab = 0, saa = 0, sbb = 0;
    for (std::size_t k = 0; k < n * n; ++k)
        sab += a[k] * b[k], saa += a[k] * a[k], sbb += b[k] * b[k];
    DistanceCorrelation r;
    r.n = n;
    const double v2 = std::max(sab / nn, 0.0);
    r.dcov = std::sqrt(v2), r.dvar_x = std::sqrt(saa / nn), r.dvar_y = std::sqrt(sbb / nn);
    r.dcor = r.dvar_x > 0 && r.dvar_y > 0 ? std::sqrt(v2 / (r.dvar_x * r.dvar_y)) : 0.0;
    auto ua = dx, ub = dy;
    u_centre(ua, n), u_centre(ub, n);
    r.dcov_squared_unbiased = u_inner(ua, ub, n);
    r.dcor_bias_corrected = bias_corrected(ua, ub, n);
    const double m = static_cast<double>(n) * (static_cast<double>(n) - 3) / 2;
    r.t_df = m - 1;
    const double rc = r.dcor_bias_corrected;
    r.t_statistic = std::sqrt(r.t_df) * rc / std::sqrt(1 - rc * rc);
    r.t_p_value = 1 - random::student_t_cdf(r.t_statistic, r.t_df);
    r.permutation_statistic = static_cast<double>(n) * v2;
    if (options.permutations > 0) {
        // Permuting y's labels permutes rows and columns of its centred matrix together.
        random::SplitMix64 rng(options.seed);
        std::vector<std::size_t> perm(n);
        std::iota(perm.begin(), perm.end(), std::size_t{0});
        std::size_t exceed = 0;
        for (std::size_t rep = 0; rep < options.permutations; ++rep) {
            for (std::size_t i = n - 1; i > 0; --i)
                std::swap(perm[i], perm[static_cast<std::size_t>(rng.next_u64() % (i + 1))]);
            double s = 0;
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    s += a[i * n + j] * b[perm[i] * n + perm[j]];
            exceed += s >= sab * (1 - 1e-12);
        }
        r.permutations = options.permutations;
        r.permutation_p_value =
            (1.0 + static_cast<double>(exceed)) / (static_cast<double>(options.permutations) + 1);
    }
    return r;
}

DistanceCorrelation distance_correlation(const std::vector<double>& x, const std::vector<double>& y,
                                         const DistanceCorrelationOptions& options) {
    Matrix mx(x.size(), 1, 0.0), my(y.size(), 1, 0.0);
    for (std::size_t i = 0; i < x.size(); ++i)
        mx(i, 0) = x[i];
    for (std::size_t i = 0; i < y.size(); ++i)
        my(i, 0) = y[i];
    return distance_correlation(mx, my, options);
}

double partial_distance_correlation(const linalg::DenseMatrix<double>& x, const linalg::DenseMatrix<double>& y,
                                    const linalg::DenseMatrix<double>& z, double index) {
    validate(x, y, index, "partial_distance_correlation");
    validate(x, z, index, "partial_distance_correlation");
    const std::size_t n = x.rows();
    auto a = distances(x, index), b = distances(y, index), c = distances(z, index);
    u_centre(a, n), u_centre(b, n), u_centre(c, n);
    const double rxy = bias_corrected(a, b, n), rxz = bias_corrected(a, c, n), ryz = bias_corrected(b, c, n);
    const double den = std::sqrt((1 - rxz * rxz) * (1 - ryz * ryz));
    return den > 1e-14 ? (rxy - rxz * ryz) / den : 0.0;
}

} // namespace datamunge::stats
