/// @file compositional.cpp
/// @brief Log-ratio transforms, Aitchison-geometry summaries and zero replacement.
#include <datamunge/stats/compositional.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Implementation helpers for compositional data analysis.
namespace {
/// @brief Dense matrix alias.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Throws unless every entry is strictly positive and finite.
/// @param x Matrix.
/// @param who Caller name for the message.
void require_positive(const Matrix& x, const char* who) {
    for (std::size_t i = 0; i < x.rows(); ++i)
        for (std::size_t j = 0; j < x.cols(); ++j)
            if (!(x(i, j) > 0) || !std::isfinite(x(i, j)))
                throw std::invalid_argument(std::string(who) +
                                            ": parts must be strictly positive and finite (replace zeros first)");
}

/// @brief Throws unless every entry is finite.
/// @param x Matrix.
/// @param who Caller name for the message.
void require_finite(const Matrix& x, const char* who) {
    for (std::size_t i = 0; i < x.rows(); ++i)
        for (std::size_t j = 0; j < x.cols(); ++j)
            if (!std::isfinite(x(i, j)))
                throw std::invalid_argument(std::string(who) + ": input must be finite");
}

/// @brief Resolves the alr reference index.
/// @param reference Requested index or last_part.
/// @param parts D.
/// @return Index.
std::size_t resolve_reference(std::size_t reference, std::size_t parts) {
    const std::size_t r = reference == last_part ? parts - 1 : reference;
    if (r >= parts)
        throw std::invalid_argument("alr: reference part out of range");
    return r;
}

/// @brief clr of one row given as a vector.
/// @param v Strictly positive parts.
/// @return clr vector.
std::vector<double> clr_row(const std::vector<double>& v) {
    double mean = 0;
    for (const double a : v)
        mean += std::log(a) / static_cast<double>(v.size());
    std::vector<double> z(v.size());
    for (std::size_t j = 0; j < v.size(); ++j)
        z[j] = std::log(v[j]) - mean;
    return z;
}
} // namespace

linalg::DenseMatrix<double> closure(const linalg::DenseMatrix<double>& x, double total) {
    if (!(total > 0) || !std::isfinite(total))
        throw std::invalid_argument("closure: total must be positive");
    Matrix out(x.rows(), x.cols(), 0.0);
    for (std::size_t i = 0; i < x.rows(); ++i) {
        double s = 0;
        for (std::size_t j = 0; j < x.cols(); ++j) {
            if (!(x(i, j) >= 0) || !std::isfinite(x(i, j)))
                throw std::invalid_argument("closure: entries must be nonnegative and finite");
            s += x(i, j);
        }
        if (!(s > 0))
            throw std::invalid_argument("closure: every row needs a positive sum");
        for (std::size_t j = 0; j < x.cols(); ++j)
            out(i, j) = total * x(i, j) / s;
    }
    return out;
}

linalg::DenseMatrix<double> clr(const linalg::DenseMatrix<double>& x) {
    require_positive(x, "clr");
    Matrix z(x.rows(), x.cols(), 0.0);
    for (std::size_t i = 0; i < x.rows(); ++i) {
        const auto r = clr_row(x.row(i));
        for (std::size_t j = 0; j < x.cols(); ++j)
            z(i, j) = r[j];
    }
    return z;
}

linalg::DenseMatrix<double> clr_inverse(const linalg::DenseMatrix<double>& z) {
    require_finite(z, "clr_inverse");
    Matrix x(z.rows(), z.cols(), 0.0);
    for (std::size_t i = 0; i < z.rows(); ++i) {
        // Subtracting the row maximum avoids overflow; closure removes the shift.
        double m = -std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < z.cols(); ++j)
            m = std::max(m, z(i, j));
        double s = 0;
        for (std::size_t j = 0; j < z.cols(); ++j)
            s += (x(i, j) = std::exp(z(i, j) - m));
        for (std::size_t j = 0; j < z.cols(); ++j)
            x(i, j) /= s;
    }
    return x;
}

linalg::DenseMatrix<double> alr(const linalg::DenseMatrix<double>& x, std::size_t reference) {
    require_positive(x, "alr");
    if (x.cols() < 2)
        throw std::invalid_argument("alr: need at least 2 parts");
    const std::size_t r = resolve_reference(reference, x.cols());
    Matrix z(x.rows(), x.cols() - 1, 0.0);
    for (std::size_t i = 0; i < x.rows(); ++i)
        for (std::size_t j = 0, k = 0; j < x.cols(); ++j)
            if (j != r)
                z(i, k++) = std::log(x(i, j) / x(i, r));
    return z;
}

linalg::DenseMatrix<double> alr_inverse(const linalg::DenseMatrix<double>& z, std::size_t reference) {
    require_finite(z, "alr_inverse");
    const std::size_t parts = z.cols() + 1;
    const std::size_t r = resolve_reference(reference, parts);
    Matrix c(z.rows(), parts, 0.0);
    for (std::size_t i = 0; i < z.rows(); ++i)
        for (std::size_t j = 0, k = 0; j < parts; ++j)
            c(i, j) = j == r ? 0.0 : z(i, k++);
    return clr_inverse(c);
}

linalg::DenseMatrix<double> ilr_basis(std::size_t parts, IlrBasis basis) {
    if (parts < 2)
        throw std::invalid_argument("ilr_basis: need at least 2 parts");
    const double d = static_cast<double>(parts);
    Matrix v(parts, parts - 1, 0.0);
    for (std::size_t j = 0; j + 1 < parts; ++j) {
        const double k = static_cast<double>(j);
        if (basis == IlrBasis::helmert) {
            const double norm = std::sqrt((k + 1) * (k + 2));
            for (std::size_t i = 0; i <= j; ++i)
                v(i, j) = -1 / norm;
            v(j + 1, j) = (k + 1) / norm;
        } else {
            const double m = d - k; // parts j..D-1
            v(j, j) = std::sqrt((m - 1) / m);
            for (std::size_t i = j + 1; i < parts; ++i)
                v(i, j) = -1 / std::sqrt(m * (m - 1));
        }
    }
    return v;
}

linalg::DenseMatrix<double> balance_basis(const linalg::DenseMatrix<double>& partition) {
    const std::size_t parts = partition.rows();
    if (parts < 2 || partition.cols() != parts - 1)
        throw std::invalid_argument("balance_basis: partition must be D x (D - 1)");
    Matrix v(parts, parts - 1, 0.0);
    for (std::size_t j = 0; j + 1 < parts; ++j) {
        double r = 0, s = 0;
        for (std::size_t i = 0; i < parts; ++i) {
            const double p = partition(i, j);
            if (p != 1 && p != -1 && p != 0)
                throw std::invalid_argument("balance_basis: entries must be -1, 0 or 1");
            r += p == 1, s += p == -1;
        }
        if (r == 0 || s == 0)
            throw std::invalid_argument("balance_basis: each balance needs parts on both sides");
        const double scale = std::sqrt(r * s / (r + s));
        for (std::size_t i = 0; i < parts; ++i)
            v(i, j) = partition(i, j) == 1 ? scale / r : partition(i, j) == -1 ? -scale / s : 0.0;
    }
    return v;
}

linalg::DenseMatrix<double> ilr(const linalg::DenseMatrix<double>& x, const linalg::DenseMatrix<double>& basis) {
    if (basis.rows() != x.cols() || basis.cols() + 1 != x.cols())
        throw std::invalid_argument("ilr: basis must be D x (D - 1)");
    const Matrix c = clr(x);
    Matrix z(x.rows(), basis.cols(), 0.0);
    for (std::size_t i = 0; i < x.rows(); ++i)
        for (std::size_t k = 0; k < basis.cols(); ++k)
            for (std::size_t j = 0; j < x.cols(); ++j)
                z(i, k) += c(i, j) * basis(j, k);
    return z;
}

linalg::DenseMatrix<double> ilr(const linalg::DenseMatrix<double>& x) { return ilr(x, ilr_basis(x.cols())); }

linalg::DenseMatrix<double> ilr_inverse(const linalg::DenseMatrix<double>& z,
                                        const linalg::DenseMatrix<double>& basis) {
    if (basis.cols() != z.cols() || basis.rows() != z.cols() + 1)
        throw std::invalid_argument("ilr_inverse: basis must be D x (D - 1) with D - 1 = coordinates");
    require_finite(z, "ilr_inverse");
    Matrix c(z.rows(), basis.rows(), 0.0);
    for (std::size_t i = 0; i < z.rows(); ++i)
        for (std::size_t j = 0; j < basis.rows(); ++j)
            for (std::size_t k = 0; k < z.cols(); ++k)
                c(i, j) += z(i, k) * basis(j, k);
    return clr_inverse(c);
}

double aitchison_distance(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size() || a.empty())
        throw std::invalid_argument("aitchison_distance: compositions must have the same nonzero length");
    for (std::size_t j = 0; j < a.size(); ++j)
        if (!(a[j] > 0) || !(b[j] > 0) || !std::isfinite(a[j]) || !std::isfinite(b[j]))
            throw std::invalid_argument("aitchison_distance: parts must be strictly positive and finite");
    const auto za = clr_row(a), zb = clr_row(b);
    double s = 0;
    for (std::size_t j = 0; j < a.size(); ++j)
        s += (za[j] - zb[j]) * (za[j] - zb[j]);
    return std::sqrt(s);
}

linalg::DenseMatrix<double> aitchison_distances(const linalg::DenseMatrix<double>& x) {
    const Matrix c = clr(x);
    Matrix d(x.rows(), x.rows(), 0.0);
    for (std::size_t i = 0; i < x.rows(); ++i)
        for (std::size_t k = i + 1; k < x.rows(); ++k) {
            double s = 0;
            for (std::size_t j = 0; j < x.cols(); ++j)
                s += (c(i, j) - c(k, j)) * (c(i, j) - c(k, j));
            d(i, k) = d(k, i) = std::sqrt(s);
        }
    return d;
}

std::vector<double> compositional_center(const linalg::DenseMatrix<double>& x) {
    if (x.rows() == 0)
        throw std::invalid_argument("compositional_center: no rows");
    const Matrix c = clr(x);
    Matrix mean(1, x.cols(), 0.0);
    for (std::size_t i = 0; i < x.rows(); ++i)
        for (std::size_t j = 0; j < x.cols(); ++j)
            mean(0, j) += c(i, j) / static_cast<double>(x.rows());
    return clr_inverse(mean).row(0);
}

linalg::DenseMatrix<double> variation_matrix(const linalg::DenseMatrix<double>& x) {
    require_positive(x, "variation_matrix");
    if (x.rows() < 2)
        throw std::invalid_argument("variation_matrix: need at least 2 rows");
    const std::size_t n = x.rows(), d = x.cols();
    Matrix t(d, d, 0.0);
    for (std::size_t j = 0; j < d; ++j)
        for (std::size_t k = j + 1; k < d; ++k) {
            double m = 0, s = 0;
            for (std::size_t i = 0; i < n; ++i)
                m += std::log(x(i, j) / x(i, k)) / static_cast<double>(n);
            for (std::size_t i = 0; i < n; ++i) {
                const double e = std::log(x(i, j) / x(i, k)) - m;
                s += e * e;
            }
            t(j, k) = t(k, j) = s / static_cast<double>(n - 1);
        }
    return t;
}

double total_variance(const linalg::DenseMatrix<double>& x) {
    const Matrix t = variation_matrix(x);
    double s = 0;
    for (std::size_t j = 0; j < t.rows(); ++j)
        for (std::size_t k = 0; k < t.cols(); ++k)
            s += t(j, k);
    return s / (2 * static_cast<double>(t.rows()));
}

linalg::DenseMatrix<double> multiplicative_replacement(const linalg::DenseMatrix<double>& x,
                                                       const std::vector<double>& detection_limits,
                                                       double fraction) {
    if (!(fraction > 0 && fraction <= 1))
        throw std::invalid_argument("multiplicative_replacement: fraction must be in (0, 1]");
    if (detection_limits.size() != x.cols())
        throw std::invalid_argument("multiplicative_replacement: one detection limit per column");
    const std::size_t n = x.rows(), d = x.cols();
    std::vector<double> totals(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < d; ++j) {
            if (!(x(i, j) >= 0) || !std::isfinite(x(i, j)))
                throw std::invalid_argument("multiplicative_replacement: data must be nonnegative and finite");
            totals[i] += x(i, j);
        }
    // Closed data (all row totals equal, as zCompositions tests) keep their total.
    const double mean_total = n ? std::accumulate(totals.begin(), totals.end(), 0.0) / static_cast<double>(n) : 0;
    const bool closed = std::ranges::all_of(
        totals, [&](double t) { return std::abs(t - mean_total) < std::pow(2.0, -52 * 0.3); });
    Matrix out = x;
    for (std::size_t i = 0; i < n; ++i) {
        double sum_delta = 0;
        for (std::size_t j = 0; j < d; ++j)
            if (x(i, j) == 0) {
                if (!(detection_limits[j] > 0) || !std::isfinite(detection_limits[j]))
                    throw std::invalid_argument("multiplicative_replacement: detection limits for zeros must be positive");
                sum_delta += fraction * detection_limits[j];
            }
        const double adjustment = 1 - sum_delta / totals[i];
        if (!(adjustment > 0))
            throw std::invalid_argument("multiplicative_replacement: replacements reach the row total");
        double s = 0;
        for (std::size_t j = 0; j < d; ++j) {
            if (x(i, j) == 0)
                out(i, j) = fraction * detection_limits[j] / adjustment;
            s += out(i, j);
        }
        if (closed)
            for (std::size_t j = 0; j < d; ++j)
                out(i, j) *= totals[0] / s;
    }
    return out;
}

linalg::DenseMatrix<double> count_zero_replacement(const linalg::DenseMatrix<double>& counts,
                                                   const CountZeroOptions& options) {
    const std::size_t n = counts.rows(), d = counts.cols();
    if (n < 2 || d < 2)
        throw std::invalid_argument("count_zero_replacement: need at least 2 rows and 2 columns");
    std::vector<double> row_sum(n, 0.0), col_sum(d, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < d; ++j) {
            const double v = counts(i, j);
            if (!(v >= 0) || !std::isfinite(v))
                throw std::invalid_argument("count_zero_replacement: counts must be nonnegative and finite");
            row_sum[i] += v, col_sum[j] += v;
        }
    if (std::ranges::any_of(row_sum, [](double s) { return !(s > 0); }))
        throw std::invalid_argument("count_zero_replacement: every row needs a positive total");
    if (std::ranges::any_of(col_sum, [](double s) { return !(s > 0); }))
        throw std::invalid_argument("count_zero_replacement: a column is entirely zero");
    // Minimum nonzero proportion per column.
    std::vector<double> colmin(d, std::numeric_limits<double>::infinity());
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < d; ++j)
            if (counts(i, j) > 0)
                colmin[j] = std::min(colmin[j], counts(i, j) / row_sum[i]);
    Matrix out(n, d, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        // Leave-one-out prior proportions t_ij and prior strength s_i.
        std::vector<double> t(d);
        double tsum = 0;
        for (std::size_t j = 0; j < d; ++j)
            tsum += (t[j] = col_sum[j] - counts(i, j));
        for (auto& v : t)
            v /= tsum;
        double s = 0;
        switch (options.method) {
        case CountZeroMethod::geometric_bayesian: {
            double lg = 0;
            for (const double v : t) {
                if (!(v > 0))
                    throw std::invalid_argument("count_zero_replacement: GBM needs positive prior proportions");
                lg += std::log(v) / static_cast<double>(d);
            }
            s = std::exp(-lg);
            break;
        }
        case CountZeroMethod::square_root:
            s = std::sqrt(row_sum[i]);
            break;
        case CountZeroMethod::bayes_laplace:
            s = static_cast<double>(d);
            break;
        case CountZeroMethod::count_zero_multiplicative:
            break;
        }
        double replaced = 0;
        bool any_zero = false;
        for (std::size_t j = 0; j < d; ++j) {
            if (counts(i, j) > 0)
                continue;
            any_zero = true;
            double r = options.method == CountZeroMethod::count_zero_multiplicative
                           ? options.fraction * options.threshold / row_sum[i]
                           : t[j] * s / (row_sum[i] + s);
            if (options.adjust && r > colmin[j])
                r = options.fraction * colmin[j];
            out(i, j) = r, replaced += r;
        }
        for (std::size_t j = 0; j < d; ++j)
            if (counts(i, j) > 0)
                out(i, j) = (any_zero ? 1 - replaced : 1.0) * counts(i, j) / row_sum[i];
    }
    return out;
}

} // namespace datamunge::stats
