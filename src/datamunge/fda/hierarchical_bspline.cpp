/// @file hierarchical_bspline.cpp
/// @brief Implementation of tensor B-splines, knot refinement, tensor P-spline fits and the
///        HB / THB hierarchical splines declared in fda/hierarchical_bspline.hpp.

#include <datamunge/fda/hierarchical_bspline.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/csr.hpp>
#include <datamunge/linalg/preconditioners.hpp>
#include <datamunge/linalg/solvers.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace datamunge::fda {
/// @brief File-local helpers: Oslo refinement rows, tensor design rows, penalties, solvers.
namespace {

/// @brief One sparse design/normal-equation row: (column, value) pairs.
using SparseRow = std::vector<std::pair<std::size_t, double>>;

/// @brief A symmetric matrix entry (row, column, value) used to add penalties.
struct Entry {
    std::size_t row; ///< Row index.
    std::size_t col; ///< Column index.
    double      val; ///< Value.
};

/// @brief Whether a basis has clamped (open) end knots of multiplicity degree + 1.
/// @param basis Basis to test. @return Clamped flag.
bool is_clamped(const BSplineBasis& basis) {
    const auto&       t = basis.knots();
    const std::size_t p = basis.degree();
    for (std::size_t i = 1; i <= p; ++i) {
        if (t[i] != t[0] || t[t.size() - 1 - i] != t.back()) return false;
    }
    return true;
}

/// @brief Throws std::invalid_argument unless @p basis is clamped.
/// @param basis Basis to test. @param who Caller name for the message.
void require_clamped(const BSplineBasis& basis, const char* who) {
    if (!is_clamped(basis))
        throw std::invalid_argument(std::string(who) +
                                    ": basis must be clamped (end knots repeated degree + 1 times)");
}

/// @brief Throws std::invalid_argument unless @p x is finite.
/// @param x Value. @param who Caller name for the message.
void require_finite(double x, const char* who) {
    if (!std::isfinite(x)) throw std::invalid_argument(std::string(who) + ": values must be finite");
}

/// @brief Oslo algorithm: rows of the operator mapping coefficients on clamped @p coarse to
///        the refined knot vector @p tau (which must contain every coarse knot).
/// @param coarse Clamped coarse basis. @param tau Refined knot vector.
/// @return rows[i] = (coarse j, alpha_j(i)) with nonzero weights.
detail::RefinementRows oslo_rows(const BSplineBasis& coarse, const std::vector<double>& tau) {
    const auto&       t = coarse.knots();
    const std::size_t p = coarse.degree();
    const std::size_t n = coarse.basis_size();
    const std::size_t m = tau.size() - p - 1;

    detail::RefinementRows rows(m);
    std::vector<double>    b, nb;
    for (std::size_t i = 0; i < m; ++i) {
        // mu: coarse knot interval containing tau[i], restricted to the valid spans [p, n-1].
        auto        it = std::upper_bound(t.begin(), t.end(), tau[i]);
        std::size_t mu = static_cast<std::size_t>(it - t.begin()) - 1;
        mu             = std::clamp(mu, p, n - 1);
        while (mu > p && t[mu] == t[mu + 1]) --mu;

        b.assign(1, 1.0);
        for (std::size_t k = 1; k <= p; ++k) {
            const double x = tau[i + k];
            nb.assign(k + 1, 0.0);
            for (std::size_t r = 1; r <= k; ++r) {
                const double hi    = t[mu + r];
                const double lo    = t[mu + r - k];
                const double denom = hi - lo;
                if (denom == 0.0) continue;
                nb[r - 1] += b[r - 1] * (hi - x) / denom;
                nb[r] += b[r - 1] * (x - lo) / denom;
            }
            b.swap(nb);
        }
        for (std::size_t s = 0; s <= p; ++s) {
            if (b[s] != 0.0) rows[i].emplace_back(static_cast<int>(mu - p + s), b[s]);
        }
    }
    return rows;
}

/// @brief Builds the transpose (column) view of refinement rows.
/// @param rows Operator rows. @param coarse_size Number of coarse functions.
/// @return cols[j] = (refined i, weight).
detail::RefinementRows transpose_rows(const detail::RefinementRows& rows, std::size_t coarse_size) {
    detail::RefinementRows cols(coarse_size);
    for (std::size_t i = 0; i < rows.size(); ++i)
        for (const auto& [j, w] : rows[i]) cols[static_cast<std::size_t>(j)].emplace_back(static_cast<int>(i), w);
    return cols;
}

/// @brief Refined knot vector and Oslo rows for @p basis with @p inserted knots.
/// @param basis Clamped basis. @param inserted Knots to insert (validated).
/// @param who Caller name for messages.
/// @return (refined basis, rows).
std::pair<BSplineBasis, detail::RefinementRows> refine_rows(const BSplineBasis&        basis,
                                                            const std::vector<double>& inserted,
                                                            const char*                who) {
    require_clamped(basis, who);
    const auto& t = basis.knots();
    for (double x : inserted) {
        if (!std::isfinite(x) || !(x > t.front() && x < t.back()))
            throw std::invalid_argument(std::string(who) + ": inserted knots must lie strictly inside the domain");
    }
    std::vector<double> tau = t;
    tau.insert(tau.end(), inserted.begin(), inserted.end());
    std::sort(tau.begin(), tau.end());
    const std::size_t limit = basis.degree() + 1;
    for (std::size_t i = 0; i < tau.size();) {
        std::size_t j = i;
        while (j < tau.size() && tau[j] == tau[i]) ++j;
        if (j - i > limit)
            throw std::invalid_argument(std::string(who) + ": knot multiplicity would exceed degree + 1");
        i = j;
    }
    auto rows = oslo_rows(basis, tau);
    return {BSplineBasis(basis.degree(), std::move(tau)), std::move(rows)};
}

/// @brief Midpoints of every positive-length knot span of @p basis.
/// @param basis Basis. @return Midpoints in increasing order.
std::vector<double> span_midpoints(const BSplineBasis& basis) {
    const auto&         t = basis.knots();
    std::vector<double> mids;
    for (std::size_t i = 0; i + 1 < t.size(); ++i)
        if (t[i] < t[i + 1]) mids.push_back(0.5 * (t[i] + t[i + 1]));
    return mids;
}

/// @brief Row-major strides for @p shape (last axis stride 1).
/// @param shape Per-axis sizes. @return Strides.
std::vector<std::size_t> strides_for(const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> strides(shape.size());
    std::size_t              s = 1;
    for (std::size_t k = shape.size(); k-- > 0;) {
        strides[k] = s;
        s *= shape[k];
    }
    return strides;
}

/// @brief Applies a per-axis operator (given by rows) along @p axis of a row-major net.
/// @param net Input net. @param shape Input shape (updated to the new axis size).
/// @param axis Axis to transform. @param rows Operator rows (new size = rows.size()).
/// @return Transformed net.
std::vector<double> apply_axis(const std::vector<double>& net, std::vector<std::size_t>& shape, std::size_t axis,
                               const detail::RefinementRows& rows) {
    const auto  old_strides = strides_for(shape);
    std::size_t outer       = 1;
    for (std::size_t k = 0; k < axis; ++k) outer *= shape[k];
    const std::size_t inner    = old_strides[axis];
    const std::size_t new_size = rows.size();

    std::vector<double> out(outer * new_size * inner, 0.0);
    for (std::size_t o = 0; o < outer; ++o) {
        const std::size_t old_base = o * shape[axis] * inner;
        const std::size_t new_base = o * new_size * inner;
        for (std::size_t i = 0; i < new_size; ++i)
            for (const auto& [j, w] : rows[i]) {
                const double* src = &net[old_base + static_cast<std::size_t>(j) * inner];
                double*       dst = &out[new_base + i * inner];
                for (std::size_t r = 0; r < inner; ++r) dst[r] += w * src[r];
            }
    }
    shape[axis] = new_size;
    return out;
}

/// @brief Local tensor evaluation: nonzero (flat index, value) pairs of all tensor basis
///        functions at @p point, with each coordinate clamped into its basis domain.
/// @param bases Per-axis bases. @param strides Row-major strides.
/// @param point Point (length checked by caller). @return Sparse row.
SparseRow tensor_row(const std::vector<BSplineBasis>& bases, const std::vector<std::size_t>& strides,
                     const std::vector<double>& point) {
    SparseRow row{{0, 1.0}};
    SparseRow next;
    for (std::size_t k = 0; k < bases.size(); ++k) {
        const auto [lo, hi] = bases[k].domain();
        const auto local    = bases[k].evaluate_point(std::clamp(point[k], lo, hi));
        next.clear();
        next.reserve(row.size() * local.indices.size());
        for (const auto& [f, v] : row)
            for (std::size_t j = 0; j < local.indices.size(); ++j)
                next.emplace_back(f + local.indices[j] * strides[k], v * local.values[j]);
        row.swap(next);
    }
    return row;
}

/// @brief Validates a point list for a d-dimensional fit/evaluation.
/// @param points Points. @param values Values (same count; may be null to skip).
/// @param d Dimension. @param who Caller name for messages.
void validate_points(const std::vector<std::vector<double>>& points, const std::vector<double>* values,
                     std::size_t d, const char* who) {
    if (points.empty()) throw std::invalid_argument(std::string(who) + ": requires at least one point");
    if (values && values->size() != points.size())
        throw std::invalid_argument(std::string(who) + ": points and values must have the same length");
    for (const auto& p : points) {
        if (p.size() != d) throw std::invalid_argument(std::string(who) + ": point dimension mismatch");
        for (double x : p) require_finite(x, who);
    }
    if (values)
        for (double y : *values) require_finite(y, who);
}

/// @brief Validates per-axis smoothing weights.
/// @param lambdas Weights. @param d Expected count. @param who Caller name.
void validate_lambdas(const std::vector<double>& lambdas, std::size_t d, const char* who) {
    if (lambdas.size() != d)
        throw std::invalid_argument(std::string(who) + ": lambdas must have one entry per dimension");
    for (double l : lambdas)
        if (!std::isfinite(l) || l < 0.0)
            throw std::invalid_argument(std::string(who) + ": lambdas must be finite and nonnegative");
}

/// @brief Binomial coefficient C(n, k) as a double. @param n Top. @param k Bottom. @return C(n, k).
double binomial(int n, int k) {
    double r = 1.0;
    for (int i = 1; i <= k; ++i) r = r * static_cast<double>(n - k + i) / static_cast<double>(i);
    return r;
}

/// @brief Entry (a, b) of D_q^T D_q for n coefficients (D_q the order-q difference matrix).
/// @param n Coefficient count (> q). @param q Order. @param a Row. @param b Column. @return Entry.
double difference_penalty_entry(int n, int q, int a, int b) {
    if (q == 0) return a == b ? 1.0 : 0.0;
    if (std::abs(a - b) > q) return 0.0;
    const int lo  = std::max({0, a - q, b - q});
    const int hi  = std::min({n - q - 1, a, b});
    double    sum = 0.0;
    for (int i = lo; i <= hi; ++i) {
        const int    ja = a - i, jb = b - i;
        const double da = ((q - ja) % 2 ? -1.0 : 1.0) * binomial(q, ja);
        const double db = ((q - jb) % 2 ? -1.0 : 1.0) * binomial(q, jb);
        sum += da * db;
    }
    return sum;
}

/// @brief Calls @p emit(flat_a, flat_b, value) for every nonzero entry of the tensor
///        difference penalty on a net of @p shape, restricted to rows in @p row_flats.
/// @param shape Per-axis sizes. @param lambdas Per-axis weights. @param order Difference order.
/// @param row_flats Flat indices whose rows are generated.
/// @param emit Callback receiving (row flat, column flat, value).
template <typename Emit>
void penalty_rows(const std::vector<std::size_t>& shape, const std::vector<double>& lambdas, int order,
                  const std::vector<std::size_t>& row_flats, Emit&& emit) {
    const auto strides = strides_for(shape);
    for (std::size_t k = 0; k < shape.size(); ++k) {
        if (lambdas[k] != 0.0 && static_cast<int>(shape[k]) <= order)
            throw std::invalid_argument("difference penalty: each penalized axis needs more than order coefficients");
    }
    for (std::size_t f : row_flats) {
        for (std::size_t k = 0; k < shape.size(); ++k) {
            if (lambdas[k] == 0.0) continue;
            const int n = static_cast<int>(shape[k]);
            const int a = static_cast<int>((f / strides[k]) % shape[k]);
            for (int b = std::max(0, a - order); b <= std::min(n - 1, a + order); ++b) {
                const double v = difference_penalty_entry(n, order, a, b);
                if (v == 0.0) continue;
                const std::size_t g = f + static_cast<std::size_t>(b) * strides[k] - static_cast<std::size_t>(a) * strides[k];
                emit(f, g, lambdas[k] * v);
            }
        }
    }
}

/// @brief Solves the (optionally penalized) least-squares normal equations
///        (sum_r w_r w_r^T + P) c = sum_r y_r w_r.
/// @param n Unknown count. @param rows Sparse design rows. @param y Targets (one per row).
/// @param penalty Penalty entries (both triangles). @param options Solver settings.
/// @param who Caller name for messages. @return Solution c.
std::vector<double> solve_normal(std::size_t n, const std::vector<SparseRow>& rows, const std::vector<double>& y,
                                 const std::vector<Entry>& penalty, const SplineSolveOptions& options,
                                 const char* who) {
    if (n == 0) throw std::invalid_argument(std::string(who) + ": no unknowns to fit");
    std::vector<double> rhs(n, 0.0);
    for (std::size_t r = 0; r < rows.size(); ++r)
        for (const auto& [c, v] : rows[r]) rhs[c] += v * y[r];

    bool direct = options.method == SplineSolver::Cholesky ||
                  (options.method == SplineSolver::Automatic && n <= options.direct_limit);
    if (direct) {
        linalg::DenseMatrix<double> a(n, n, 0.0);
        for (const auto& row : rows)
            for (const auto& [i, vi] : row)
                for (const auto& [j, vj] : row) a(i, j) += vi * vj;
        for (const auto& e : penalty) a(e.row, e.col) += e.val;
        double max_diag = 0.0;
        for (std::size_t i = 0; i < n; ++i) max_diag = std::max(max_diag, a(i, i));
        const auto chol = linalg::cholesky(a);
        bool       ok   = chol.ok && max_diag > 0.0;
        for (std::size_t i = 0; ok && i < n; ++i)
            ok = chol.L(i, i) * chol.L(i, i) > 1e-12 * max_diag;
        if (!ok)
            throw std::runtime_error(std::string(who) +
                                     ": normal equations are singular (too little data for some basis "
                                     "function; add a penalty or use fewer functions)");
        return chol.solve(rhs);
    }

    linalg::SparseCOO<double> coo(n, n);
    for (const auto& row : rows)
        for (const auto& [i, vi] : row)
            for (const auto& [j, vj] : row) coo.set(i, j, vi * vj);
    for (const auto& e : penalty) coo.set(e.row, e.col, e.val);
    coo.compress();
    const auto csr = linalg::to_csr(coo);
    const auto pre  = linalg::make_jacobi(csr);
    const auto diag = linalg::diagonal(csr);
    for (std::size_t i = 0; i < n; ++i)
        if (diag[i] <= 0.0)
            throw std::runtime_error(std::string(who) + ": normal equations are singular (an unknown has no data)");
    linalg::SolverOptions so;
    so.tol      = options.tolerance;
    so.max_iter = options.max_iterations ? options.max_iterations : 10 * n;
    auto res    = linalg::pcg(csr, rhs, pre, {}, so);
    if (!res.converged)
        throw std::runtime_error(std::string(who) + ": conjugate gradient did not converge (system may be singular)");
    return res.x;
}

/// @brief Shared row-major Cartesian-grid evaluation.
/// @param d Dimension. @param axes Per-axis sample lists. @param eval Point evaluator.
/// @param who Caller name for messages. @return Row-major values.
template <typename Eval>
std::vector<double> grid_eval(std::size_t d, const std::vector<std::vector<double>>& axes, Eval&& eval,
                              const char* who) {
    if (axes.size() != d) throw std::invalid_argument(std::string(who) + ": need one axis per dimension");
    std::size_t total = 1;
    for (const auto& a : axes) {
        if (a.empty()) throw std::invalid_argument(std::string(who) + ": each axis needs at least one value");
        total *= a.size();
    }
    std::vector<double>      out;
    std::vector<std::size_t> idx(d, 0);
    std::vector<double>      point(d);
    out.reserve(total);
    for (std::size_t flat = 0; flat < total; ++flat) {
        for (std::size_t k = 0; k < d; ++k) point[k] = axes[k][idx[k]];
        out.push_back(eval(point));
        for (std::size_t k = d; k-- > 0;) {
            if (++idx[k] < axes[k].size()) break;
            idx[k] = 0;
        }
    }
    return out;
}

/// @brief Per-axis basis sizes of @p bases. @param bases Bases. @return Sizes.
std::vector<std::size_t> shape_of(const std::vector<BSplineBasis>& bases) {
    std::vector<std::size_t> shape;
    for (const auto& b : bases) shape.push_back(b.basis_size());
    return shape;
}

/// @brief Shared worker for fit_tensor_bspline() and fit_tensor_pspline().
/// @param bases Bases. @param points Points. @param values Values.
/// @param lambdas Penalty weights or null. @param order Difference order.
/// @param options Solver settings. @param who Caller name. @return Fitted spline.
TensorBSpline fit_tensor_impl(std::vector<BSplineBasis> bases, const std::vector<std::vector<double>>& points,
                              const std::vector<double>& values, const std::vector<double>* lambdas, int order,
                              const SplineSolveOptions& options, const char* who) {
    if (bases.empty()) throw std::invalid_argument(std::string(who) + ": at least one basis is required");
    validate_points(points, &values, bases.size(), who);
    const auto shape   = shape_of(bases);
    const auto strides = strides_for(shape);
    std::size_t n = 1;
    for (auto s : shape) n *= s;

    std::vector<SparseRow> rows;
    rows.reserve(points.size());
    for (const auto& p : points) rows.push_back(tensor_row(bases, strides, p));

    std::vector<Entry> penalty;
    if (lambdas) {
        validate_lambdas(*lambdas, bases.size(), who);
        if (order < 0) throw std::invalid_argument(std::string(who) + ": order must be nonnegative");
        std::vector<std::size_t> all(n);
        for (std::size_t i = 0; i < n; ++i) all[i] = i;
        penalty_rows(shape, *lambdas, order, all,
                     [&](std::size_t a, std::size_t b, double v) { penalty.push_back({a, b, v}); });
    }
    auto coef = solve_normal(n, rows, values, penalty, options, who);
    return TensorBSpline(std::move(bases), std::move(coef));
}

/// @brief Checks a box (lo, hi) for a hierarchical region argument.
/// @param lo Lower corner. @param hi Upper corner. @param d Dimension. @param who Caller name.
void validate_box(const std::vector<double>& lo, const std::vector<double>& hi, std::size_t d, const char* who) {
    if (lo.size() != d || hi.size() != d)
        throw std::invalid_argument(std::string(who) + ": region corners must have one entry per dimension");
    for (std::size_t k = 0; k < d; ++k)
        if (!std::isfinite(lo[k]) || !std::isfinite(hi[k]) || !(lo[k] < hi[k]))
            throw std::invalid_argument(std::string(who) + ": region bounds must be finite with min < max");
}

} // namespace

// ── BSplineRefinement / refinement ──────────────────────────────────────────

std::vector<double> BSplineRefinement::apply(const std::vector<double>& coefficients) const {
    if (coefficients.size() != matrix.cols())
        throw std::invalid_argument("BSplineRefinement::apply: coefficient count must equal the coarse basis size");
    std::vector<double> out(matrix.rows(), 0.0);
    const auto&         r = matrix.row_indices();
    const auto&         c = matrix.col_indices();
    const auto&         v = matrix.values();
    for (std::size_t k = 0; k < v.size(); ++k) out[r[k]] += v[k] * coefficients[c[k]];
    return out;
}

BSplineRefinement refine_basis(const BSplineBasis& basis, const std::vector<double>& inserted_knots) {
    auto [refined, rows] = refine_rows(basis, inserted_knots, "refine_basis");
    linalg::SparseCOO<double> m(refined.basis_size(), basis.basis_size());
    for (std::size_t i = 0; i < rows.size(); ++i)
        for (const auto& [j, w] : rows[i]) m.set(i, static_cast<std::size_t>(j), w);
    m.compress();
    return BSplineRefinement{std::move(refined), std::move(m)};
}

BSplineRefinement dyadic_refine(const BSplineBasis& basis) {
    require_clamped(basis, "dyadic_refine");
    return refine_basis(basis, span_midpoints(basis));
}

// ── TensorBSpline ───────────────────────────────────────────────────────────

TensorBSpline::TensorBSpline(BSplineBasis basis, std::vector<double> coefficients)
    : TensorBSpline(std::vector<BSplineBasis>{std::move(basis)}, std::move(coefficients)) {}

TensorBSpline::TensorBSpline(std::vector<BSplineBasis> bases, std::vector<double> coefficients)
    : bases_(std::move(bases)), coefficients_(std::move(coefficients)) {
    if (bases_.empty()) throw std::invalid_argument("TensorBSpline: at least one basis is required");
    const auto  shape = shape_of(bases_);
    std::size_t total = 1;
    for (auto s : shape) {
        if (total > std::numeric_limits<std::size_t>::max() / s)
            throw std::invalid_argument("TensorBSpline: coefficient net is too large");
        total *= s;
    }
    if (coefficients_.size() != total)
        throw std::invalid_argument("TensorBSpline: coefficient count must equal the product of the basis sizes");
    strides_ = strides_for(shape);
}

std::vector<int> TensorBSpline::shape() const {
    std::vector<int> out;
    for (const auto& b : bases_) out.push_back(static_cast<int>(b.basis_size()));
    return out;
}

std::vector<double> TensorBSpline::domain_min() const {
    std::vector<double> out;
    for (const auto& b : bases_) out.push_back(b.domain().first);
    return out;
}

std::vector<double> TensorBSpline::domain_max() const {
    std::vector<double> out;
    for (const auto& b : bases_) out.push_back(b.domain().second);
    return out;
}

std::size_t TensorBSpline::flat_index(const std::vector<int>& index) const {
    if (index.size() != bases_.size())
        throw std::invalid_argument("TensorBSpline: multi-index must have one entry per dimension");
    std::size_t f = 0;
    for (std::size_t k = 0; k < index.size(); ++k) {
        if (index[k] < 0 || static_cast<std::size_t>(index[k]) >= bases_[k].basis_size())
            throw std::out_of_range("TensorBSpline: multi-index component out of range");
        f += static_cast<std::size_t>(index[k]) * strides_[k];
    }
    return f;
}

double TensorBSpline::coefficient(const std::vector<int>& index) const { return coefficients_[flat_index(index)]; }

void TensorBSpline::set_coefficient(const std::vector<int>& index, double value) {
    coefficients_[flat_index(index)] = value;
}

double TensorBSpline::evaluate(double x) const {
    if (bases_.size() != 1)
        throw std::invalid_argument("TensorBSpline::evaluate: scalar input requires a one-dimensional spline");
    return evaluate(std::vector<double>{x});
}

double TensorBSpline::evaluate(const std::vector<double>& point) const {
    if (point.size() != bases_.size())
        throw std::invalid_argument("TensorBSpline::evaluate: point dimension mismatch");
    for (double x : point) require_finite(x, "TensorBSpline::evaluate");
    double sum = 0.0;
    for (const auto& [f, v] : tensor_row(bases_, strides_, point)) sum += v * coefficients_[f];
    return sum;
}

std::vector<double> TensorBSpline::evaluate_points(const std::vector<std::vector<double>>& points) const {
    std::vector<double> out;
    out.reserve(points.size());
    for (const auto& p : points) out.push_back(evaluate(p));
    return out;
}

std::vector<double> TensorBSpline::evaluate_grid(const std::vector<std::vector<double>>& axes) const {
    return grid_eval(
        bases_.size(), axes, [&](const std::vector<double>& p) { return evaluate(p); },
        "TensorBSpline::evaluate_grid");
}

TensorBSpline TensorBSpline::dyadic_refine() const {
    std::vector<BSplineBasis> refined;
    auto                      shape = shape_of(bases_);
    std::vector<double>       net   = coefficients_;
    for (std::size_t k = 0; k < bases_.size(); ++k) {
        auto [basis, rows] = refine_rows(bases_[k], span_midpoints(bases_[k]), "TensorBSpline::dyadic_refine");
        net                = apply_axis(net, shape, k, rows);
        refined.push_back(std::move(basis));
    }
    return TensorBSpline(std::move(refined), std::move(net));
}

// ── Penalties and fits ──────────────────────────────────────────────────────

linalg::SparseCOO<double> tensor_difference_penalty(const std::vector<int>& shape, const std::vector<double>& lambdas,
                                                    int order) {
    if (shape.empty()) throw std::invalid_argument("tensor_difference_penalty: shape must not be empty");
    validate_lambdas(lambdas, shape.size(), "tensor_difference_penalty");
    if (order < 0) throw std::invalid_argument("tensor_difference_penalty: order must be nonnegative");
    std::vector<std::size_t> s;
    std::size_t              n = 1;
    for (int v : shape) {
        if (v <= 0) throw std::invalid_argument("tensor_difference_penalty: sizes must be positive");
        s.push_back(static_cast<std::size_t>(v));
        n *= s.back();
    }
    std::vector<std::size_t> all(n);
    for (std::size_t i = 0; i < n; ++i) all[i] = i;
    linalg::SparseCOO<double> out(n, n);
    penalty_rows(s, lambdas, order, all, [&](std::size_t a, std::size_t b, double v) { out.set(a, b, v); });
    out.compress();
    return out;
}

TensorBSpline fit_tensor_bspline(std::vector<BSplineBasis> bases, const std::vector<std::vector<double>>& points,
                                 const std::vector<double>& values, const SplineSolveOptions& options) {
    return fit_tensor_impl(std::move(bases), points, values, nullptr, 0, options, "fit_tensor_bspline");
}

TensorBSpline fit_tensor_pspline(std::vector<BSplineBasis> bases, const std::vector<std::vector<double>>& points,
                                 const std::vector<double>& values, const std::vector<double>& lambdas, int order,
                                 const SplineSolveOptions& options) {
    return fit_tensor_impl(std::move(bases), points, values, &lambdas, order, options, "fit_tensor_pspline");
}

// ── OrthantCellTree ─────────────────────────────────────────────────────────

namespace detail {

OrthantCellTree::OrthantCellTree(std::vector<double> lo, std::vector<double> hi)
    : root_lo_(std::move(lo)), root_hi_(std::move(hi)) {
    if (root_lo_.empty() || root_lo_.size() != root_hi_.size())
        throw std::invalid_argument("hierarchical spline: domain corners must be non-empty and equal-sized");
    if (root_lo_.size() > 20) throw std::invalid_argument("hierarchical spline: at most 20 dimensions are supported");
    for (std::size_t k = 0; k < root_lo_.size(); ++k)
        if (!std::isfinite(root_lo_[k]) || !std::isfinite(root_hi_[k]) || !(root_lo_[k] < root_hi_[k]))
            throw std::invalid_argument("hierarchical spline: domain bounds must be finite with min < max");
    lo_.push_back(root_lo_);
    hi_.push_back(root_hi_);
    depth_.push_back(0);
    parent_.push_back(-1);
    children_.emplace_back();
    by_depth_.push_back({0});
}

void OrthantCellTree::check(int cell) const {
    if (cell < 0 || cell >= size()) throw std::out_of_range("hierarchical spline: invalid cell id");
}

void OrthantCellTree::check_slot(int slot) const {
    if (slot < 0 || slot >= arity()) throw std::out_of_range("hierarchical spline: child slot out of range");
}

int OrthantCellTree::child(int cell, int slot) const {
    const auto& c = children_[static_cast<std::size_t>(cell)];
    return c.empty() ? -1 : c[static_cast<std::size_t>(slot)];
}

std::pair<std::vector<double>, std::vector<double>> OrthantCellTree::child_box(int cell, int slot) const {
    std::vector<double> lo = lo_[static_cast<std::size_t>(cell)];
    std::vector<double> hi = hi_[static_cast<std::size_t>(cell)];
    for (std::size_t k = 0; k < lo.size(); ++k) {
        const double mid = 0.5 * (lo[k] + hi[k]);
        if ((slot >> k) & 1)
            lo[k] = mid;
        else
            hi[k] = mid;
    }
    return {std::move(lo), std::move(hi)};
}

int OrthantCellTree::ensure_child(int cell, int slot) {
    auto* c = &children_[static_cast<std::size_t>(cell)];
    if (c->empty()) c->assign(static_cast<std::size_t>(arity()), -1);
    if ((*c)[static_cast<std::size_t>(slot)] >= 0) return (*c)[static_cast<std::size_t>(slot)];
    auto [lo, hi]   = child_box(cell, slot);
    const int id    = size();
    const int depth = depth_[static_cast<std::size_t>(cell)] + 1;
    lo_.push_back(std::move(lo));
    hi_.push_back(std::move(hi));
    depth_.push_back(depth);
    parent_.push_back(cell);
    children_.emplace_back();
    children_[static_cast<std::size_t>(cell)][static_cast<std::size_t>(slot)] = id;
    if (static_cast<int>(by_depth_.size()) <= depth) by_depth_.resize(static_cast<std::size_t>(depth) + 1);
    by_depth_[static_cast<std::size_t>(depth)].push_back(id);
    max_depth_ = std::max(max_depth_, depth);
    return id;
}

int OrthantCellTree::slot_for(int cell, const std::vector<double>& point) const {
    const auto& lo   = lo_[static_cast<std::size_t>(cell)];
    const auto& hi   = hi_[static_cast<std::size_t>(cell)];
    int         slot = 0;
    for (std::size_t k = 0; k < lo.size(); ++k) {
        const double mid = 0.5 * (lo[k] + hi[k]);
        if (point[k] > mid || point[k] == hi[k]) slot |= 1 << k;
    }
    return slot;
}

bool OrthantCellTree::contains(int cell, const std::vector<double>& point) const {
    const auto& lo = lo_[static_cast<std::size_t>(cell)];
    const auto& hi = hi_[static_cast<std::size_t>(cell)];
    for (std::size_t k = 0; k < lo.size(); ++k)
        if (!(point[k] >= lo[k] && point[k] <= hi[k])) return false;
    return true;
}

int OrthantCellTree::locate(const std::vector<double>& point) const {
    if (point.size() != dimensions())
        throw std::invalid_argument("hierarchical spline: point dimension mismatch");
    if (!contains(0, point)) return -1;
    int cell = 0;
    while (true) {
        const int c = child(cell, slot_for(cell, point));
        if (c < 0) return cell;
        cell = c;
    }
}

const std::vector<int>& OrthantCellTree::cells_at_depth(int depth) const {
    static const std::vector<int> empty;
    if (depth < 0 || depth >= static_cast<int>(by_depth_.size())) return empty;
    return by_depth_[static_cast<std::size_t>(depth)];
}

bool OrthantCellTree::overlaps(const std::vector<double>& lo, const std::vector<double>& hi,
                               const std::vector<double>& blo, const std::vector<double>& bhi) {
    for (std::size_t k = 0; k < lo.size(); ++k)
        if (hi[k] <= blo[k] || lo[k] >= bhi[k]) return false;
    return true;
}

bool OrthantCellTree::covered_to_depth(int target_depth, const std::vector<double>& blo,
                                       const std::vector<double>& bhi) const {
    if (target_depth > max_depth_) return !overlaps(root_lo_, root_hi_, blo, bhi);
    return covered_impl(0, target_depth, blo, bhi);
}

bool OrthantCellTree::covered_impl(int cell, int target_depth, const std::vector<double>& blo,
                                   const std::vector<double>& bhi) const {
    if (!overlaps(lo(cell), hi(cell), blo, bhi)) return true;
    if (depth(cell) >= target_depth) return true;
    for (int slot = 0; slot < arity(); ++slot) {
        auto [clo, chi] = child_box(cell, slot);
        if (!overlaps(clo, chi, blo, bhi)) continue;
        const int c = child(cell, slot);
        if (c < 0 || !covered_impl(c, target_depth, blo, bhi)) return false;
    }
    return true;
}

} // namespace detail

// ── HierarchicalBSpline ─────────────────────────────────────────────────────

HierarchicalBSpline::HierarchicalBSpline(std::vector<double> domain_min, std::vector<double> domain_max)
    : tree_(std::move(domain_min), std::move(domain_max)), patches_(1) {}

int HierarchicalBSpline::parent(int cell) const {
    tree_.check(cell);
    return tree_.parent(cell);
}

int HierarchicalBSpline::child(int cell, int slot) const {
    tree_.check(cell);
    tree_.check_slot(slot);
    return tree_.child(cell, slot);
}

int HierarchicalBSpline::depth(int cell) const {
    tree_.check(cell);
    return tree_.depth(cell);
}

std::vector<double> HierarchicalBSpline::cell_min(int cell) const {
    tree_.check(cell);
    return tree_.lo(cell);
}

std::vector<double> HierarchicalBSpline::cell_max(int cell) const {
    tree_.check(cell);
    return tree_.hi(cell);
}

int HierarchicalBSpline::locate(const std::vector<double>& point) const { return tree_.locate(point); }

bool HierarchicalBSpline::has_spline(int cell) const {
    tree_.check(cell);
    return patches_[static_cast<std::size_t>(cell)].has_value();
}

const TensorBSpline& HierarchicalBSpline::spline(int cell) const {
    tree_.check(cell);
    const auto& p = patches_[static_cast<std::size_t>(cell)];
    if (!p) throw std::invalid_argument("HierarchicalBSpline::spline: cell has no patch");
    return *p;
}

void HierarchicalBSpline::check_patch(const TensorBSpline& spline) const {
    if (spline.dimensions() != dimensions())
        throw std::invalid_argument("HierarchicalBSpline: patch dimension does not match the hierarchy");
}

void HierarchicalBSpline::set_spline(int cell, TensorBSpline spline) {
    tree_.check(cell);
    check_patch(spline);
    patches_[static_cast<std::size_t>(cell)] = std::move(spline);
}

void HierarchicalBSpline::clear_spline(int cell) {
    tree_.check(cell);
    patches_[static_cast<std::size_t>(cell)].reset();
}

int HierarchicalBSpline::make_child(int cell, int slot) {
    const int c = tree_.ensure_child(cell, slot);
    if (static_cast<int>(patches_.size()) < tree_.size()) patches_.resize(static_cast<std::size_t>(tree_.size()));
    return c;
}

std::vector<int> HierarchicalBSpline::subdivide(int cell) {
    tree_.check(cell);
    std::vector<int> out;
    for (int s = 0; s < tree_.arity(); ++s) out.push_back(make_child(cell, s));
    return out;
}

int HierarchicalBSpline::refine(int cell, int slot) {
    tree_.check(cell);
    tree_.check_slot(slot);
    return make_child(cell, slot);
}

int HierarchicalBSpline::refine(int cell, int slot, TensorBSpline spline) {
    tree_.check(cell);
    tree_.check_slot(slot);
    check_patch(spline);
    const int c                           = make_child(cell, slot);
    patches_[static_cast<std::size_t>(c)] = std::move(spline);
    return c;
}

void HierarchicalBSpline::refine_region(const std::vector<double>& region_min, const std::vector<double>& region_max,
                                        int depth) {
    validate_box(region_min, region_max, dimensions(), "HierarchicalBSpline::refine_region");
    refine_region_impl(0, region_min, region_max, depth, nullptr);
}

void HierarchicalBSpline::refine_region(const std::vector<double>& region_min, const std::vector<double>& region_max,
                                        int depth, const SplineFactory& factory) {
    validate_box(region_min, region_max, dimensions(), "HierarchicalBSpline::refine_region");
    refine_region_impl(0, region_min, region_max, depth, &factory);
}

void HierarchicalBSpline::refine_region_impl(
    int cell, const std::vector<double>& lo, const std::vector<double>& hi, int depth,
    const std::function<std::optional<TensorBSpline>(const std::vector<double>&, const std::vector<double>&)>*
        factory) {
    if (tree_.depth(cell) >= depth) return;
    for (int s = 0; s < tree_.arity(); ++s) {
        auto [clo, chi] = tree_.child_box(cell, s);
        if (!detail::OrthantCellTree::overlaps(clo, chi, lo, hi)) continue;
        const int c = make_child(cell, s);
        if (factory && *factory) {
            if (auto patch = (*factory)(clo, chi)) {
                check_patch(*patch);
                patches_[static_cast<std::size_t>(c)] = std::move(*patch);
            }
        }
        refine_region_impl(c, lo, hi, depth, factory);
    }
}

std::vector<double> HierarchicalBSpline::to_local(int cell, const std::vector<double>& point) const {
    const auto&         lo = tree_.lo(cell);
    const auto&         hi = tree_.hi(cell);
    std::vector<double> local(point.size());
    for (std::size_t k = 0; k < point.size(); ++k) local[k] = (point[k] - lo[k]) / (hi[k] - lo[k]);
    return local;
}

double HierarchicalBSpline::evaluate(const std::vector<double>& point) const {
    if (point.size() != dimensions())
        throw std::invalid_argument("HierarchicalBSpline::evaluate: point dimension mismatch");
    if (!tree_.contains(0, point))
        throw std::invalid_argument("HierarchicalBSpline::evaluate: point lies outside the domain");
    double value = 0.0;
    int    cell  = 0;
    while (cell >= 0) {
        const auto& p = patches_[static_cast<std::size_t>(cell)];
        if (p) value += p->evaluate(to_local(cell, point));
        const int c = tree_.child(cell, tree_.slot_for(cell, point));
        cell        = (c >= 0 && tree_.contains(c, point)) ? c : -1;
    }
    return value;
}

std::vector<double> HierarchicalBSpline::evaluate_points(const std::vector<std::vector<double>>& points) const {
    std::vector<double> out;
    out.reserve(points.size());
    for (const auto& p : points) out.push_back(evaluate(p));
    return out;
}

std::vector<double> HierarchicalBSpline::evaluate_grid(const std::vector<std::vector<double>>& axes) const {
    return grid_eval(
        dimensions(), axes, [&](const std::vector<double>& p) { return evaluate(p); },
        "HierarchicalBSpline::evaluate_grid");
}

std::pair<std::vector<std::vector<double>>, std::vector<double>> HierarchicalBSpline::prepare_fit(
    int cell, const std::vector<std::vector<double>>& points, const std::vector<double>& values,
    const char* who) const {
    tree_.check(cell);
    if (patches_[static_cast<std::size_t>(cell)])
        throw std::invalid_argument(std::string(who) + ": cell already has a patch; clear_spline() first to refit");
    validate_points(points, &values, dimensions(), who);
    std::vector<std::vector<double>> local;
    std::vector<double>              residual;
    local.reserve(points.size());
    residual.reserve(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (!tree_.contains(cell, points[i]))
            throw std::invalid_argument(std::string(who) + ": point lies outside the cell");
        local.push_back(to_local(cell, points[i]));
        residual.push_back(values[i] - evaluate(points[i]));
    }
    return {std::move(local), std::move(residual)};
}

void HierarchicalBSpline::fit_cell(int cell, std::vector<BSplineBasis> bases,
                                   const std::vector<std::vector<double>>& points, const std::vector<double>& values,
                                   const SplineSolveOptions& options) {
    auto [local, residual] = prepare_fit(cell, points, values, "HierarchicalBSpline::fit_cell");
    set_spline(cell, fit_tensor_impl(std::move(bases), local, residual, nullptr, 0, options,
                                     "HierarchicalBSpline::fit_cell"));
}

void HierarchicalBSpline::fit_cell_penalized(int cell, std::vector<BSplineBasis> bases,
                                             const std::vector<std::vector<double>>& points,
                                             const std::vector<double>& values, const std::vector<double>& lambdas,
                                             int order, const SplineSolveOptions& options) {
    auto [local, residual] = prepare_fit(cell, points, values, "HierarchicalBSpline::fit_cell_penalized");
    set_spline(cell, fit_tensor_impl(std::move(bases), local, residual, &lambdas, order, options,
                                     "HierarchicalBSpline::fit_cell_penalized"));
}

// ── TruncatedHierarchicalBSpline ────────────────────────────────────────────

TruncatedHierarchicalBSpline::TruncatedHierarchicalBSpline(TensorBSpline base)
    : tree_(base.domain_min(), base.domain_max()) {
    for (const auto& b : base.bases()) require_clamped(b, "TruncatedHierarchicalBSpline");
    levels_.push_back(std::move(base));
    inside_cache_.resize(1);
}

void TruncatedHierarchicalBSpline::check_level(int level) const {
    if (level < 0 || level >= level_count())
        throw std::out_of_range("TruncatedHierarchicalBSpline: level out of range");
}

const TensorBSpline& TruncatedHierarchicalBSpline::level(int level) const {
    check_level(level);
    return levels_[static_cast<std::size_t>(level)];
}

int TruncatedHierarchicalBSpline::parent(int cell) const {
    tree_.check(cell);
    return tree_.parent(cell);
}

int TruncatedHierarchicalBSpline::child(int cell, int slot) const {
    tree_.check(cell);
    tree_.check_slot(slot);
    return tree_.child(cell, slot);
}

int TruncatedHierarchicalBSpline::depth(int cell) const {
    tree_.check(cell);
    return tree_.depth(cell);
}

std::vector<double> TruncatedHierarchicalBSpline::cell_min(int cell) const {
    tree_.check(cell);
    return tree_.lo(cell);
}

std::vector<double> TruncatedHierarchicalBSpline::cell_max(int cell) const {
    tree_.check(cell);
    return tree_.hi(cell);
}

int TruncatedHierarchicalBSpline::locate(const std::vector<double>& point) const { return tree_.locate(point); }

void TruncatedHierarchicalBSpline::invalidate() {
    for (auto& c : inside_cache_) c.clear();
}

void TruncatedHierarchicalBSpline::ensure_level(int level) {
    while (level_count() <= level) {
        const auto&                         top = levels_.back();
        std::vector<detail::AxisRefinement> ops;
        std::vector<BSplineBasis>           bases;
        auto                                shape = shape_of(top.bases());
        std::vector<double>                 net   = top.coefficients();
        for (std::size_t k = 0; k < top.dimensions(); ++k) {
            const auto& b      = top.bases()[k];
            auto [basis, rows] = refine_rows(b, span_midpoints(b), "TruncatedHierarchicalBSpline");
            net                = apply_axis(net, shape, k, rows);
            detail::AxisRefinement op;
            op.cols = transpose_rows(rows, b.basis_size());
            op.rows = std::move(rows);
            ops.push_back(std::move(op));
            bases.push_back(std::move(basis));
        }
        refine_ops_.push_back(std::move(ops));
        levels_.emplace_back(std::move(bases), std::move(net));
        inside_cache_.emplace_back();
    }
}

int TruncatedHierarchicalBSpline::refine(int cell, int slot) {
    tree_.check(cell);
    tree_.check_slot(slot);
    ensure_level(tree_.depth(cell) + 1);
    const int c = tree_.ensure_child(cell, slot);
    invalidate();
    return c;
}

std::vector<int> TruncatedHierarchicalBSpline::subdivide(int cell) {
    tree_.check(cell);
    ensure_level(tree_.depth(cell) + 1);
    std::vector<int> out;
    for (int s = 0; s < tree_.arity(); ++s) out.push_back(tree_.ensure_child(cell, s));
    invalidate();
    return out;
}

void TruncatedHierarchicalBSpline::refine_region(const std::vector<double>& region_min,
                                                 const std::vector<double>& region_max, int depth) {
    validate_box(region_min, region_max, dimensions(), "TruncatedHierarchicalBSpline::refine_region");
    // Iterative depth-first walk; levels are materialized before cells reach them.
    std::vector<int> stack{0};
    while (!stack.empty()) {
        const int cell = stack.back();
        stack.pop_back();
        if (tree_.depth(cell) >= depth) continue;
        for (int s = 0; s < tree_.arity(); ++s) {
            auto [clo, chi] = tree_.child_box(cell, s);
            if (!detail::OrthantCellTree::overlaps(clo, chi, region_min, region_max)) continue;
            ensure_level(tree_.depth(cell) + 1);
            stack.push_back(tree_.ensure_child(cell, s));
        }
    }
    invalidate();
}

std::size_t TruncatedHierarchicalBSpline::flat(int level, const std::vector<int>& index) const {
    check_level(level);
    return levels_[static_cast<std::size_t>(level)].flat_index(index);
}

std::vector<int> TruncatedHierarchicalBSpline::unflat(int level, std::size_t f) const {
    const auto&      s = levels_[static_cast<std::size_t>(level)];
    std::vector<int> idx(s.dimensions());
    for (std::size_t k = 0; k < idx.size(); ++k)
        idx[k] = static_cast<int>((f / s.strides_[k]) % s.bases_[k].basis_size());
    return idx;
}

std::vector<double> TruncatedHierarchicalBSpline::support_min(int level, const std::vector<int>& index) const {
    const auto          idx = unflat(level, flat(level, index));
    const auto&         s   = levels_[static_cast<std::size_t>(level)];
    std::vector<double> out;
    for (std::size_t k = 0; k < idx.size(); ++k) out.push_back(s.bases_[k].knots()[static_cast<std::size_t>(idx[k])]);
    return out;
}

std::vector<double> TruncatedHierarchicalBSpline::support_max(int level, const std::vector<int>& index) const {
    const auto          idx = unflat(level, flat(level, index));
    const auto&         s   = levels_[static_cast<std::size_t>(level)];
    std::vector<double> out;
    for (std::size_t k = 0; k < idx.size(); ++k)
        out.push_back(s.bases_[k].knots()[static_cast<std::size_t>(idx[k]) + s.bases_[k].degree() + 1]);
    return out;
}

bool TruncatedHierarchicalBSpline::inside(int level, std::size_t f) const {
    if (level == 0) return true;
    auto& cache = inside_cache_[static_cast<std::size_t>(level)];
    auto  it    = cache.find(f);
    if (it != cache.end()) return it->second;
    const auto          idx = unflat(level, f);
    const auto&         s   = levels_[static_cast<std::size_t>(level)];
    std::vector<double> lo, hi;
    for (std::size_t k = 0; k < idx.size(); ++k) {
        const auto& t = s.bases_[k].knots();
        lo.push_back(t[static_cast<std::size_t>(idx[k])]);
        hi.push_back(t[static_cast<std::size_t>(idx[k]) + s.bases_[k].degree() + 1]);
    }
    const bool in = tree_.covered_to_depth(level, lo, hi);
    cache.emplace(f, in);
    return in;
}

bool TruncatedHierarchicalBSpline::active_flat(int level, std::size_t f) const {
    if (!inside(level, f)) return false;
    if (level + 1 > tree_.max_depth()) return true;
    // The support (a union of level-l knot spans) lies in Omega^(l+1) iff each of its
    // level-(l+1) children used by the refinement does; test the box directly instead.
    const auto          idx = unflat(level, f);
    const auto&         s   = levels_[static_cast<std::size_t>(level)];
    std::vector<double> lo, hi;
    for (std::size_t k = 0; k < idx.size(); ++k) {
        const auto& t = s.bases_[k].knots();
        lo.push_back(t[static_cast<std::size_t>(idx[k])]);
        hi.push_back(t[static_cast<std::size_t>(idx[k]) + s.bases_[k].degree() + 1]);
    }
    return !tree_.covered_to_depth(level + 1, lo, hi);
}

bool TruncatedHierarchicalBSpline::is_active(int level, const std::vector<int>& index) const {
    return active_flat(level, flat(level, index));
}

std::vector<std::size_t> TruncatedHierarchicalBSpline::active_flat_indices(int level) const {
    check_level(level);
    const auto&              s = levels_[static_cast<std::size_t>(level)];
    std::vector<std::size_t> out;
    if (level == 0) {
        for (std::size_t f = 0; f < s.coefficient_count(); ++f)
            if (active_flat(0, f)) out.push_back(f);
        return out;
    }
    // Candidates: functions whose support overlaps some cell of depth `level`.
    std::unordered_set<std::size_t> seen;
    const std::size_t               d = s.dimensions();
    for (int cell : tree_.cells_at_depth(level)) {
        std::vector<std::size_t> first(d), last(d);
        bool                     any = true;
        for (std::size_t k = 0; k < d; ++k) {
            const auto&       t = s.bases_[k].knots();
            const std::size_t p = s.bases_[k].degree();
            const std::size_t n = s.bases_[k].basis_size();
            const double      lo = tree_.lo(cell)[k], hi = tree_.hi(cell)[k];
            // i overlaps (lo, hi) iff t[i] < hi and t[i + p + 1] > lo.
            std::size_t a = static_cast<std::size_t>(std::upper_bound(t.begin(), t.end(), lo) - t.begin());
            a             = a >= p + 1 ? a - p - 1 : 0;
            while (a < n && !(t[a + p + 1] > lo)) ++a;
            std::size_t b = a;
            while (b < n && t[b] < hi) ++b;
            if (a >= b) {
                any = false;
                break;
            }
            first[k] = a;
            last[k]  = b;
        }
        if (!any) continue;
        std::vector<std::size_t> idx = first;
        while (true) {
            std::size_t f = 0;
            for (std::size_t k = 0; k < d; ++k) f += idx[k] * s.strides_[k];
            seen.insert(f);
            std::size_t k = d;
            while (k-- > 0) {
                if (++idx[k] < last[k]) break;
                idx[k] = first[k];
            }
            if (k == static_cast<std::size_t>(-1)) break;
        }
    }
    for (std::size_t f : seen)
        if (active_flat(level, f)) out.push_back(f);
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<std::vector<int>> TruncatedHierarchicalBSpline::active_indices(int level) const {
    std::vector<std::vector<int>> out;
    for (std::size_t f : active_flat_indices(level)) out.push_back(unflat(level, f));
    return out;
}

std::size_t TruncatedHierarchicalBSpline::basis_size() const {
    std::size_t n = 0;
    for (int l = 0; l < level_count(); ++l) n += active_flat_indices(l).size();
    return n;
}

double TruncatedHierarchicalBSpline::coefficient(int level, const std::vector<int>& index) const {
    return levels_[static_cast<std::size_t>(level)].coefficients_[flat(level, index)];
}

void TruncatedHierarchicalBSpline::propagate(int level, const SparseLevelVector& delta) {
    SparseLevelVector cur = delta;
    for (int k = level; k + 1 < level_count() && !cur.empty(); ++k) {
        const auto&       coarse = levels_[static_cast<std::size_t>(k)];
        auto&             fine   = levels_[static_cast<std::size_t>(k) + 1];
        const auto&       ops    = refine_ops_[static_cast<std::size_t>(k)];
        const std::size_t d      = coarse.dimensions();
        SparseLevelVector next;
        std::vector<std::size_t> pos(d);
        for (const auto& [f, v] : cur) {
            const auto idx = unflat(k, f);
            // Enumerate the tensor product of per-axis refinement columns.
            std::vector<const std::vector<std::pair<int, double>>*> cols(d);
            bool                                                    empty = false;
            for (std::size_t a = 0; a < d; ++a) {
                cols[a] = &ops[a].cols[static_cast<std::size_t>(idx[a])];
                empty   = empty || cols[a]->empty();
            }
            if (empty) continue;
            std::fill(pos.begin(), pos.end(), 0);
            while (true) {
                std::size_t g = 0;
                double      w = v;
                for (std::size_t a = 0; a < d; ++a) {
                    const auto& [i, r] = (*cols[a])[pos[a]];
                    g += static_cast<std::size_t>(i) * fine.strides_[a];
                    w *= r;
                }
                next[g] += w;
                std::size_t a = d;
                while (a-- > 0) {
                    if (++pos[a] < cols[a]->size()) break;
                    pos[a] = 0;
                }
                if (a == static_cast<std::size_t>(-1)) break;
            }
        }
        // Truncation: finer functions inside Omega^(k+1) keep their own coefficients.
        for (auto it = next.begin(); it != next.end();) {
            if (inside(k + 1, it->first))
                it = next.erase(it);
            else {
                fine.coefficients_[it->first] += it->second;
                ++it;
            }
        }
        cur = std::move(next);
    }
}

void TruncatedHierarchicalBSpline::set_coefficient(int level, const std::vector<int>& index, double value) {
    require_finite(value, "TruncatedHierarchicalBSpline::set_coefficient");
    const std::size_t f = flat(level, index);
    if (!active_flat(level, f))
        throw std::invalid_argument(
            "TruncatedHierarchicalBSpline::set_coefficient: function is not active at this level");
    auto&        c     = levels_[static_cast<std::size_t>(level)].coefficients_[f];
    const double delta = value - c;
    c                  = value;
    propagate(level, {{f, delta}});
}

std::vector<TruncatedHierarchicalBSpline::SparseLevelVector> TruncatedHierarchicalBSpline::truncated_rows(
    const std::vector<double>& point, int min_level) const {
    const int cell = tree_.locate(point);
    if (cell < 0) throw std::invalid_argument("TruncatedHierarchicalBSpline: point lies outside the domain");
    const int d = tree_.depth(cell);
    if (d < min_level) return {};
    std::vector<SparseLevelVector> out(static_cast<std::size_t>(d - min_level) + 1);
    const auto&                    top = levels_[static_cast<std::size_t>(d)];
    SparseLevelVector              w;
    for (const auto& [f, v] : tensor_row(top.bases_, top.strides_, point)) w[f] += v;
    out.back() = w;
    const std::size_t        dims = top.dimensions();
    std::vector<std::size_t> pos(dims);
    for (int k = d; k > min_level; --k) {
        const auto&       coarse = levels_[static_cast<std::size_t>(k) - 1];
        const auto&       ops    = refine_ops_[static_cast<std::size_t>(k) - 1];
        SparseLevelVector up;
        for (const auto& [f, v] : w) {
            if (inside(k, f)) continue; // truncated away at level k
            const auto idx = unflat(k, f);
            std::vector<const std::vector<std::pair<int, double>>*> rows(dims);
            for (std::size_t a = 0; a < dims; ++a) rows[a] = &ops[a].rows[static_cast<std::size_t>(idx[a])];
            std::fill(pos.begin(), pos.end(), 0);
            while (true) {
                std::size_t g = 0;
                double      x = v;
                for (std::size_t a = 0; a < dims; ++a) {
                    const auto& [j, r] = (*rows[a])[pos[a]];
                    g += static_cast<std::size_t>(j) * coarse.strides_[a];
                    x *= r;
                }
                up[g] += x;
                std::size_t a = dims;
                while (a-- > 0) {
                    if (++pos[a] < rows[a]->size()) break;
                    pos[a] = 0;
                }
                if (a == static_cast<std::size_t>(-1)) break;
            }
        }
        w                                                   = std::move(up);
        out[static_cast<std::size_t>(k - 1 - min_level)] = w;
    }
    return out;
}

double TruncatedHierarchicalBSpline::truncated_basis(int level, const std::vector<int>& index,
                                                     const std::vector<double>& point) const {
    const std::size_t f = flat(level, index);
    if (point.size() != dimensions())
        throw std::invalid_argument("TruncatedHierarchicalBSpline::truncated_basis: point dimension mismatch");
    if (!active_flat(level, f)) {
        if (tree_.locate(point) < 0)
            throw std::invalid_argument("TruncatedHierarchicalBSpline: point lies outside the domain");
        return 0.0;
    }
    const auto rows = truncated_rows(point, level);
    if (rows.empty()) return 0.0;
    auto it = rows.front().find(f);
    return it == rows.front().end() ? 0.0 : it->second;
}

void TruncatedHierarchicalBSpline::fit_level_impl(int level, const std::vector<std::vector<double>>& points,
                                                  const std::vector<double>& values,
                                                  const std::vector<double>* lambdas, int order,
                                                  const SplineSolveOptions& options) {
    const char* who = lambdas ? "TruncatedHierarchicalBSpline::fit_level_penalized"
                              : "TruncatedHierarchicalBSpline::fit_level";
    check_level(level);
    validate_points(points, &values, dimensions(), who);
    if (lambdas) {
        validate_lambdas(*lambdas, dimensions(), who);
        if (order < 0) throw std::invalid_argument(std::string(who) + ": order must be nonnegative");
    }
    const auto active = active_flat_indices(level);
    if (active.empty()) throw std::invalid_argument(std::string(who) + ": level has no active functions");
    std::unordered_map<std::size_t, std::size_t> column;
    for (std::size_t c = 0; c < active.size(); ++c) column.emplace(active[c], c);

    std::vector<SparseRow> rows;
    std::vector<double>    targets;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const auto tr = truncated_rows(points[i], level);
        if (tr.empty()) continue; // shallower point: unaffected by this level
        SparseRow row;
        for (const auto& [f, v] : tr.front()) {
            auto it = column.find(f);
            if (it != column.end() && v != 0.0) row.emplace_back(it->second, v);
        }
        rows.push_back(std::move(row));
        targets.push_back(values[i] - evaluate(points[i]));
    }
    if (rows.empty())
        throw std::invalid_argument(std::string(who) + ": no point lies in the region refined to this level");

    std::vector<Entry> penalty;
    if (lambdas) {
        const auto shape = shape_of(levels_[static_cast<std::size_t>(level)].bases_);
        penalty_rows(shape, *lambdas, order, active, [&](std::size_t a, std::size_t b, double v) {
            auto it = column.find(b);
            if (it != column.end()) penalty.push_back({column.at(a), it->second, v});
        });
    }
    const auto        delta = solve_normal(active.size(), rows, targets, penalty, options, who);
    SparseLevelVector change;
    auto&             net = levels_[static_cast<std::size_t>(level)].coefficients_;
    for (std::size_t c = 0; c < active.size(); ++c) {
        net[active[c]] += delta[c];
        change[active[c]] = delta[c];
    }
    propagate(level, change);
}

void TruncatedHierarchicalBSpline::fit_level(int level, const std::vector<std::vector<double>>& points,
                                             const std::vector<double>& values, const SplineSolveOptions& options) {
    fit_level_impl(level, points, values, nullptr, 0, options);
}

void TruncatedHierarchicalBSpline::fit_level_penalized(int level, const std::vector<std::vector<double>>& points,
                                                       const std::vector<double>& values,
                                                       const std::vector<double>& lambdas, int order,
                                                       const SplineSolveOptions& options) {
    fit_level_impl(level, points, values, &lambdas, order, options);
}

void TruncatedHierarchicalBSpline::repropagate_all() {
    for (int k = 0; k + 1 < level_count(); ++k) {
        const auto& coarse = levels_[static_cast<std::size_t>(k)];
        auto        shape  = shape_of(coarse.bases_);
        auto        net    = coarse.coefficients_;
        const auto& ops    = refine_ops_[static_cast<std::size_t>(k)];
        for (std::size_t a = 0; a < coarse.dimensions(); ++a) net = apply_axis(net, shape, a, ops[a].rows);
        auto& fine = levels_[static_cast<std::size_t>(k) + 1].coefficients_;
        for (std::size_t f = 0; f < fine.size(); ++f)
            if (!inside(k + 1, f)) fine[f] = net[f];
    }
}

void TruncatedHierarchicalBSpline::fit_all_impl(const std::vector<std::vector<double>>& points,
                                                const std::vector<double>& values, const std::vector<double>* lambdas,
                                                int order, const SplineSolveOptions& options) {
    const char* who = lambdas ? "TruncatedHierarchicalBSpline::fit_penalized" : "TruncatedHierarchicalBSpline::fit";
    validate_points(points, &values, dimensions(), who);
    if (lambdas) {
        validate_lambdas(*lambdas, dimensions(), who);
        if (order < 0) throw std::invalid_argument(std::string(who) + ": order must be nonnegative");
    }
    std::vector<std::vector<std::size_t>>                     active(static_cast<std::size_t>(level_count()));
    std::vector<std::unordered_map<std::size_t, std::size_t>> column(static_cast<std::size_t>(level_count()));
    std::size_t                                               n = 0;
    for (int l = 0; l < level_count(); ++l) {
        active[static_cast<std::size_t>(l)] = active_flat_indices(l);
        for (std::size_t f : active[static_cast<std::size_t>(l)]) column[static_cast<std::size_t>(l)].emplace(f, n++);
    }

    std::vector<SparseRow> rows;
    rows.reserve(points.size());
    for (const auto& p : points) {
        const auto tr = truncated_rows(p, 0);
        SparseRow  row;
        for (std::size_t l = 0; l < tr.size(); ++l)
            for (const auto& [f, v] : tr[l]) {
                auto it = column[l].find(f);
                if (it != column[l].end() && v != 0.0) row.emplace_back(it->second, v);
            }
        rows.push_back(std::move(row));
    }

    // Penalize only order-q difference rows whose q + 1 coefficients (consecutive along one
    // axis of one level) are all active, so the penalty's polynomial null space (constants
    // for q = 1, ...) is unpenalized even though inactive neighbours are absent.
    std::vector<Entry> penalty;
    if (lambdas) {
        std::vector<double> w(static_cast<std::size_t>(order) + 1);
        for (int j = 0; j <= order; ++j) w[static_cast<std::size_t>(j)] = ((order - j) % 2 ? -1.0 : 1.0) * binomial(order, j);
        for (int l = 0; l < level_count(); ++l) {
            const auto& cols = column[static_cast<std::size_t>(l)];
            const auto& s    = levels_[static_cast<std::size_t>(l)];
            for (std::size_t k = 0; k < s.dimensions(); ++k) {
                const double lambda = (*lambdas)[k];
                const auto   n      = s.bases_[k].basis_size();
                if (lambda == 0.0) continue;
                if (n <= static_cast<std::size_t>(order))
                    throw std::invalid_argument(std::string(who) +
                                                ": each penalized axis needs more than order coefficients");
                for (std::size_t f : active[static_cast<std::size_t>(l)]) {
                    const std::size_t a = (f / s.strides_[k]) % n;
                    if (a + static_cast<std::size_t>(order) >= n) continue;
                    std::vector<std::size_t> run;
                    for (int j = 0; j <= order; ++j) {
                        auto it = cols.find(f + static_cast<std::size_t>(j) * s.strides_[k]);
                        if (it == cols.end()) break;
                        run.push_back(it->second);
                    }
                    if (run.size() != w.size()) continue;
                    for (std::size_t i = 0; i < run.size(); ++i)
                        for (std::size_t j = 0; j < run.size(); ++j) penalty.push_back({run[i], run[j], lambda * w[i] * w[j]});
                }
            }
        }
    }
    const auto c = solve_normal(n, rows, values, penalty, options, who);
    for (int l = 0; l < level_count(); ++l) {
        auto& net = levels_[static_cast<std::size_t>(l)].coefficients_;
        for (std::size_t f : active[static_cast<std::size_t>(l)]) net[f] = c[column[static_cast<std::size_t>(l)].at(f)];
    }
    repropagate_all();
}

void TruncatedHierarchicalBSpline::fit(const std::vector<std::vector<double>>& points,
                                       const std::vector<double>& values, const SplineSolveOptions& options) {
    fit_all_impl(points, values, nullptr, 0, options);
}

void TruncatedHierarchicalBSpline::fit_penalized(const std::vector<std::vector<double>>& points,
                                                 const std::vector<double>& values,
                                                 const std::vector<double>& lambdas, int order,
                                                 const SplineSolveOptions& options) {
    fit_all_impl(points, values, &lambdas, order, options);
}

double TruncatedHierarchicalBSpline::evaluate(const std::vector<double>& point) const {
    if (point.size() != dimensions())
        throw std::invalid_argument("TruncatedHierarchicalBSpline::evaluate: point dimension mismatch");
    for (double x : point) require_finite(x, "TruncatedHierarchicalBSpline::evaluate");
    const int cell = tree_.locate(point);
    if (cell < 0) throw std::invalid_argument("TruncatedHierarchicalBSpline::evaluate: point lies outside the domain");
    return levels_[static_cast<std::size_t>(tree_.depth(cell))].evaluate(point);
}

std::vector<double> TruncatedHierarchicalBSpline::evaluate_points(const std::vector<std::vector<double>>& points) const {
    std::vector<double> out;
    out.reserve(points.size());
    for (const auto& p : points) out.push_back(evaluate(p));
    return out;
}

std::vector<double> TruncatedHierarchicalBSpline::evaluate_grid(const std::vector<std::vector<double>>& axes) const {
    return grid_eval(
        dimensions(), axes, [&](const std::vector<double>& p) { return evaluate(p); },
        "TruncatedHierarchicalBSpline::evaluate_grid");
}

} // namespace datamunge::fda
