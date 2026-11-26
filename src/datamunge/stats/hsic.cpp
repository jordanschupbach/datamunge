/// @file hsic.cpp
/// @brief Gram matrices, the dHSIC V-statistic, gamma moment matching and permutation testing.
#include <datamunge/stats/hsic.hpp>

#include <datamunge/random/distributions.hpp>
#include <datamunge/random/random.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Implementation helpers for HSIC.
namespace {
/// @brief Dense matrix alias.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Squared Euclidean distance between rows i and j.
/// @param x Observations in rows.
/// @param i Row.
/// @param j Row.
/// @return Squared distance.
double squared_distance(const Matrix& x, std::size_t i, std::size_t j) {
    double s = 0;
    for (std::size_t k = 0; k < x.cols(); ++k) {
        const double t = x(i, k) - x(j, k);
        s += t * t;
    }
    return s;
}

/// @brief Median heuristic \f$\sqrt{\operatorname{median}_{i<j}\lVert x_i-x_j\rVert^2/2}\f$ (upper median; 0.001 if
/// zero).
/// @param x Observations in rows.
/// @return Bandwidth.
double median_bandwidth(const Matrix& x) {
    const std::size_t n = x.rows();
    std::vector<double> d;
    d.reserve(n * (n - 1) / 2);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j)
            d.push_back(squared_distance(x, i, j));
    const std::size_t h = d.size() / 2;
    std::ranges::nth_element(d, d.begin() + static_cast<std::ptrdiff_t>(h));
    const double bw = std::sqrt(d[h] / 2); // upper median for an even count, as in dHSIC
    return bw == 0 ? 0.001 : bw;
}

/// @brief Gram matrix (row-major n x n) of one variable.
/// @param x Observations.
/// @param kernel Kernel.
/// @param bw Gaussian bandwidth.
/// @return Gram matrix.
std::vector<double> gram(const Matrix& x, HsicKernel kernel, double bw) {
    const std::size_t n = x.rows();
    std::vector<double> k(n * n, 1.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j) {
            const double d = squared_distance(x, i, j);
            const double v = kernel == HsicKernel::discrete ? (d == 0 ? 1.0 : 0.0) : std::exp(-d / (2 * bw * bw));
            k[i * n + j] = k[j * n + i] = v;
        }
    return k;
}

/// @brief dHSIC V-statistic \f$\frac1{n^2}\sum\prod_jK^j+\prod_j\frac{1}{n^2}\sum K^j-\frac2n\sum_i\prod_j
/// \frac1n\sum_kK^j_{ik}\f$ with variables j >= 1 permuted.
/// @param k Gram matrices.
/// @param perm Permutation per variable (perm[0] unused).
/// @param n Size.
/// @return Statistic.
double dhsic(const std::vector<std::vector<double>>& k, const std::vector<std::vector<std::size_t>>& perm,
             std::size_t n) {
    const double nd = static_cast<double>(n);
    double term1 = 0, term2 = 1;
    std::vector<double> term3(n, 2 / nd);
    std::vector<double> prod(n * n, 1.0);
    for (std::size_t v = 0; v < k.size(); ++v) {
        const auto& p = perm[v];
        double total = 0;
        for (std::size_t i = 0; i < n; ++i) {
            double row = 0;
            for (std::size_t j = 0; j < n; ++j) {
                const double e = k[v][p[i] * n + p[j]];
                prod[i * n + j] *= e;
                row += e;
            }
            total += row;
            term3[i] *= row / nd;
        }
        term2 *= total / (nd * nd);
    }
    for (const double e : prod)
        term1 += e;
    return term1 / (nd * nd) + term2 - std::accumulate(term3.begin(), term3.end(), 0.0);
}
} // namespace

Hsic hsic(const std::vector<linalg::DenseMatrix<double>>& variables, const HsicOptions& options) {
    const std::size_t d = variables.size();
    if (d < 2)
        throw std::invalid_argument("hsic: need at least two variables");
    const std::size_t n = variables[0].rows();
    for (const auto& v : variables) {
        if (v.rows() != n || v.cols() == 0)
            throw std::invalid_argument("hsic: variables need the same number of rows");
        for (std::size_t i = 0; i < v.rows(); ++i)
            for (std::size_t j = 0; j < v.cols(); ++j)
                if (!std::isfinite(v(i, j)))
                    throw std::invalid_argument("hsic: values must be finite");
    }
    if (n < 2 * d)
        throw std::invalid_argument("hsic: need n >= 2d observations");
    if (!(options.bandwidth > 0) || !(options.bandwidth_scale > 0))
        throw std::invalid_argument("hsic: bandwidth and bandwidth_scale must be positive");
    Hsic r;
    r.n = n;
    std::vector<std::vector<double>> k;
    for (const auto& v : variables) {
        const double bw = options.kernel == HsicKernel::gaussian         ? options.bandwidth_scale * median_bandwidth(v)
                          : options.kernel == HsicKernel::gaussian_fixed ? options.bandwidth
                                                                         : std::numeric_limits<double>::quiet_NaN();
        r.bandwidths.push_back(bw);
        k.push_back(gram(v, options.kernel, bw));
    }
    std::vector<std::vector<std::size_t>> perm(d, std::vector<std::size_t>(n));
    for (auto& p : perm)
        std::iota(p.begin(), p.end(), std::size_t{0});
    r.statistic = dhsic(k, perm, n);
    const double nd = static_cast<double>(n);
    r.scaled_statistic = nd * r.statistic;

    // Gamma approximation (Pfister et al. 2018, Proposition 4).
    std::vector<double> a(d), b(d), c(d);
    for (std::size_t j = 0; j < d; ++j) {
        double s = 0, s2 = 0, rs2 = 0;
        for (std::size_t i = 0; i < n; ++i) {
            double row = 0;
            for (std::size_t l = 0; l < n; ++l) {
                const double e = k[j][i * n + l];
                row += e, s2 += e * e;
            }
            s += row, rs2 += row * row;
        }
        a[j] = s / (nd * nd), b[j] = s2 / (nd * nd), c[j] = rs2 / (nd * nd * nd);
    }
    const auto product = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 1.0, std::multiplies<>());
    };
    const double pa = product(a), pb = product(b), pc = product(c), pd = pa * pa;
    const double dd = static_cast<double>(d);
    double one_out_a = 0;
    for (std::size_t j = 0; j < d; ++j)
        one_out_a += pa / a[j];
    const double mean = (1 - one_out_a + (dd - 1) * pa) / nd;
    double t4 = 0, t5 = 0, t6 = 0, t7 = 0;
    for (std::size_t j = 0; j < d; ++j) {
        const double od = pd / (a[j] * a[j]), oc = pc / c[j];
        t4 += b[j] * od, t5 += b[j] * oc, t6 += c[j] * od;
        for (std::size_t s = j + 1; s < d; ++s)
            t7 += 2 * c[j] * c[s] * od / (a[s] * a[s]);
    }
    double f1 = nd - 2 * dd, f2 = nd * (nd - 1) * (nd - 2);
    for (std::size_t j = 1; j <= 2 * d - 3; ++j) {
        f1 *= nd - 2 * dd - static_cast<double>(j);
        f2 *= nd - 2 - static_cast<double>(j);
    }
    const double var = 2 * f1 / f2 *
                       (pb + (dd - 1) * (dd - 1) * pd + 2 * (dd - 1) * pc + t4 - 2 * t5 - 2 * (dd - 1) * t6 + t7);
    r.gamma_shape = mean * mean / var;
    r.gamma_scale = nd * var / mean;
    r.gamma_p_value = random::gamma_cdf(r.scaled_statistic, r.gamma_shape, r.gamma_scale, false);

    if (options.permutations > 0) {
        random::SplitMix64 rng(options.seed);
        std::size_t exceed = 0;
        for (std::size_t rep = 0; rep < options.permutations; ++rep) {
            for (std::size_t v = 1; v < d; ++v)
                for (std::size_t i = n - 1; i > 0; --i)
                    std::swap(perm[v][i], perm[v][static_cast<std::size_t>(rng.next_u64() % (i + 1))]);
            exceed += dhsic(k, perm, n) >= r.statistic * (1 - 1e-12);
        }
        r.permutations = options.permutations;
        r.permutation_p_value =
            (1.0 + static_cast<double>(exceed)) / (static_cast<double>(options.permutations) + 1);
    }
    return r;
}

Hsic hsic(const std::vector<double>& x, const std::vector<double>& y, const HsicOptions& options) {
    Matrix mx(x.size(), 1, 0.0), my(y.size(), 1, 0.0);
    for (std::size_t i = 0; i < x.size(); ++i)
        mx(i, 0) = x[i];
    for (std::size_t i = 0; i < y.size(); ++i)
        my(i, 0) = y[i];
    return hsic(std::vector<Matrix>{mx, my}, options);
}

} // namespace datamunge::stats
