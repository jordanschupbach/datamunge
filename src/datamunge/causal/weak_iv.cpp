/// @file weak_iv.cpp
/// @brief Anderson-Rubin test inversion, 2SLS and LIML after partialling out exogenous controls.
#include <datamunge/causal/weak_iv.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace datamunge::causal {
/// @brief Implementation helpers local to this translation unit.
namespace {
/// @brief Dense row-major matrix of doubles.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Least-squares projection onto the columns of a matrix.
///
/// Holds a modified Gram-Schmidt orthonormal basis, so that fitted values and residuals of any
/// vector follow from inner products.
class Projector {
public:
    /// @brief Builds an orthonormal basis of the columns.
    /// @param a n-by-m matrix (no columns gives the zero projection).
    /// @param what Name used in the error message.
    /// @throws std::runtime_error When the columns are collinear.
    Projector(const Matrix& a, const char* what) {
        const std::size_t n = a.rows();
        for (std::size_t c = 0; c < a.cols(); ++c) {
            std::vector<double> v(n);
            double norm0 = 0;
            for (std::size_t i = 0; i < n; ++i) {
                v[i] = a(i, c);
                norm0 += v[i] * v[i];
            }
            for (int pass = 0; pass < 2; ++pass)
                for (const auto& q : basis_) {
                    double dot = 0;
                    for (std::size_t i = 0; i < n; ++i)
                        dot += q[i] * v[i];
                    for (std::size_t i = 0; i < n; ++i)
                        v[i] -= dot * q[i];
                }
            double norm = 0;
            for (const double x : v)
                norm += x * x;
            if (!(norm > 1e-20 * std::max(norm0, 1e-300)))
                throw std::runtime_error(std::string("weak_iv_inference: collinear ") + what);
            norm = std::sqrt(norm);
            for (double& x : v)
                x /= norm;
            basis_.push_back(std::move(v));
        }
    }
    /// @brief Residual of a vector after projection.
    /// @param v Vector of length n.
    /// @return v minus its projection.
    [[nodiscard]] std::vector<double> residual(std::vector<double> v) const {
        for (const auto& q : basis_) {
            double dot = 0;
            for (std::size_t i = 0; i < v.size(); ++i)
                dot += q[i] * v[i];
            for (std::size_t i = 0; i < v.size(); ++i)
                v[i] -= dot * q[i];
        }
        return v;
    }
    /// @brief Rank of the projection.
    /// @return Number of basis vectors.
    [[nodiscard]] std::size_t rank() const { return basis_.size(); }

private:
    std::vector<std::vector<double>> basis_; ///< Orthonormal basis vectors.
};

/// @brief Inner product.
/// @param a First vector.
/// @param b Second vector.
/// @return Sum of products.
double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        s += a[i] * b[i];
    return s;
}

/// @brief A k-class estimate from partialled-out moments.
/// @param yy y'y.
/// @param yd y'd.
/// @param dd d'd.
/// @param myd y'M_Z d.
/// @param mdd d'M_Z d.
/// @param k The k-class parameter.
/// @param dof n - p - 1.
/// @param t Critical t value.
/// @return The estimate.
KClassEstimate k_class(double yy, double yd, double dd, double myd, double mdd, double k, double dof, double t) {
    const double denom = dd - k * mdd;
    KClassEstimate e;
    e.k = k;
    e.estimate = (yd - k * myd) / denom;
    const double rss = yy - 2 * e.estimate * yd + e.estimate * e.estimate * dd;
    e.standard_error = std::sqrt(rss / dof / denom);
    e.lower = e.estimate - t * e.standard_error;
    e.upper = e.estimate + t * e.standard_error;
    return e;
}
} // namespace

WeakIvInference weak_iv_inference(const std::vector<double>& outcome, const std::vector<double>& endogenous,
                                  const linalg::DenseMatrix<double>& instruments,
                                  const linalg::DenseMatrix<double>& controls, const WeakIvOptions& options) {
    const std::size_t n = outcome.size();
    if (endogenous.size() != n || instruments.rows() != n || instruments.cols() == 0 ||
        (controls.cols() > 0 && controls.rows() != n))
        throw std::invalid_argument("weak_iv_inference: input sizes differ or no instruments");
    if (!(options.alpha > 0 && options.alpha < 1) || !std::isfinite(options.null_value))
        throw std::invalid_argument("weak_iv_inference: alpha must be in (0, 1) and the null value finite");
    const auto finite = [](const std::vector<double>& v) {
        return std::ranges::all_of(v, [](double x) { return std::isfinite(x); });
    };
    if (!finite(outcome) || !finite(endogenous) || !finite(instruments.data()) || !finite(controls.data()))
        throw std::invalid_argument("weak_iv_inference: inputs must be finite");
    const std::size_t p = controls.cols() + (options.intercept ? 1 : 0);
    Matrix x(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t c = 0;
        if (options.intercept)
            x(i, c++) = 1;
        for (std::size_t j = 0; j < controls.cols(); ++j)
            x(i, c++) = controls(i, j);
    }
    const Projector px(x, "controls");
    const std::vector<double> y = px.residual(outcome), d = px.residual(endogenous);
    Matrix z(n, instruments.cols(), 0.0);
    for (std::size_t j = 0; j < instruments.cols(); ++j) {
        std::vector<double> col(n);
        for (std::size_t i = 0; i < n; ++i)
            col[i] = instruments(i, j);
        col = px.residual(col);
        for (std::size_t i = 0; i < n; ++i)
            z(i, j) = col[i];
    }
    const Projector pz(z, "instruments");
    const std::size_t l = pz.rank();
    if (n <= l + p + 1)
        throw std::invalid_argument("weak_iv_inference: too few observations");
    const std::vector<double> my = pz.residual(y), md = pz.residual(d);
    // Partialled-out moments: totals, and residual (M_Z) parts; projected parts are differences.
    const double yy = dot(y, y), yd = dot(y, d), dd = dot(d, d);
    const double myy = dot(my, my), myd = dot(my, md), mdd = dot(md, md);
    const double pyy = yy - myy, pyd = yd - myd, pdd = dd - mdd;
    const double df2 = static_cast<double>(n - l - p), df1 = static_cast<double>(l);
    WeakIvInference out;
    out.df1 = l;
    out.df2 = n - l - p;
    out.first_stage_f = (pdd / df1) / (mdd / df2);
    const double b0 = options.null_value;
    const double proj = pyy - 2 * b0 * pyd + b0 * b0 * pdd;
    const double resid = myy - 2 * b0 * myd + b0 * b0 * mdd;
    out.ar_statistic = (proj / df1) / (resid / df2);
    out.ar_p_value = 1 - random::f_cdf(out.ar_statistic, df1, df2);
    // Accept beta when c (r'r) - (c + 1) r'P r >= 0 with r = y - beta d: a quadratic in beta.
    const double c = random::f_quantile(1 - options.alpha, df1, df2) * df1 / df2;
    const double qa = c * dd - (c + 1) * pdd;
    const double qb = -2 * c * yd + 2 * (c + 1) * pyd;
    const double qc = c * yy - (c + 1) * pyy;
    const double inf = std::numeric_limits<double>::infinity();
    auto& set = out.ar_set;
    if (qa == 0) {
        if (qb > 0) {
            set.shape = ConfidenceSetShape::two_rays; // degenerate: a single ray
            set.pieces = {{-qc / qb, inf}};
        } else if (qb < 0) {
            set.shape = ConfidenceSetShape::two_rays;
            set.pieces = {{-inf, -qc / qb}};
        } else if (qc >= 0) {
            set.shape = ConfidenceSetShape::whole_line;
            set.pieces = {{-inf, inf}};
        } else
            set.shape = ConfidenceSetShape::empty;
    } else {
        const double disc = qb * qb - 4 * qa * qc;
        if (disc <= 0) {
            if (qa > 0) {
                set.shape = ConfidenceSetShape::whole_line;
                set.pieces = {{-inf, inf}};
            } else
                set.shape = ConfidenceSetShape::empty;
        } else {
            const double r1 = (-qb + std::sqrt(disc)) / (2 * qa), r2 = (-qb - std::sqrt(disc)) / (2 * qa);
            const double lo = std::min(r1, r2), hi = std::max(r1, r2);
            if (qa < 0) {
                set.shape = ConfidenceSetShape::interval;
                set.pieces = {{lo, hi}};
            } else {
                set.shape = ConfidenceSetShape::two_rays;
                set.pieces = {{-inf, lo}, {hi, inf}};
            }
        }
    }
    // LIML k: the smaller eigenvalue of A B^{-1}, A = [y d]'[y d], B = [y d]'M_Z[y d].
    const double det_b = myy * mdd - myd * myd;
    const double m11 = (yy * mdd - yd * myd) / det_b, m12 = (-yy * myd + yd * myy) / det_b;
    const double m21 = (yd * mdd - dd * myd) / det_b, m22 = (-yd * myd + dd * myy) / det_b;
    const double tr = m11 + m22, det = m11 * m22 - m12 * m21;
    const double k_liml = 0.5 * (tr - std::sqrt(std::max(tr * tr - 4 * det, 0.0)));
    const double dof = static_cast<double>(n - p - 1);
    const double t = random::student_t_quantile(1 - options.alpha / 2, dof);
    out.tsls = k_class(yy, yd, dd, myd, mdd, 1.0, dof, t);
    out.liml = k_class(yy, yd, dd, myd, mdd, k_liml, dof, t);
    return out;
}

} // namespace datamunge::causal
