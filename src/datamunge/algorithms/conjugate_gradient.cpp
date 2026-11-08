/// @file conjugate_gradient.cpp
/// @brief One executing checked CG kernel, forward stationary baseline and fixtures.
#include <datamunge/algorithms/conjugate_gradient.hpp>
#include <datamunge/algorithms/gauss_seidel.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace datamunge::algorithms {
/// @brief Implementation details for the checked iterative solvers.
namespace {
using Matrix = std::vector<std::vector<double>>; ///< Dense owned row vectors.
using Vector = std::vector<double>;             ///< Double vector storage.

/// @brief Check and narrow a computed value to double.
/// @param x Working value; finite underflow rounds normally.
/// @return Rounded finite double.
/// @throws std::overflow_error Nonfinite value or magnitude beyond double.
double narrow(long double x) {
    if (!std::isfinite(x) || std::abs(x) > std::numeric_limits<double>::max())
        throw std::overflow_error("linear iteration: computed value cannot fit double");
    return static_cast<double>(x);
}

/// @brief Validate all entries of a vector before arithmetic.
/// @param x Vector to inspect without mutation.
/// @throws std::invalid_argument Nonfinite entry.
void finite_vector(const Vector& x) {
    for (double v : x)
        if (!std::isfinite(v)) throw std::invalid_argument("linear iteration: nonfinite vector");
}

/// @brief Validate common absolute solve controls and RHS.
/// @param b RHS to inspect completely.
/// @param iterations Nonnegative budget.
/// @param tolerance Finite nonnegative absolute cutoff.
/// @throws std::invalid_argument Invalid controls or RHS.
void controls(const Vector& b, int iterations, double tolerance) {
    finite_vector(b);
    if (iterations < 0 || !std::isfinite(tolerance) || tolerance < 0)
        throw std::invalid_argument("linear iteration: invalid budget/tolerance");
}

/// @brief Validate full matrix shape/entries and the selected solver's structural contract.
/// @param a Matrix to inspect.
/// @param n Required square dimension.
/// @param symmetric Require exact symmetry for CG, otherwise nonzero diagonal for GS.
/// @throws std::invalid_argument Bad shape, entry, symmetry or diagonal.
void matrix_input(const Matrix& a, std::size_t n, bool symmetric) {
    if (a.size() != n) throw std::invalid_argument("linear iteration: matrix/RHS dimension");
    for (const auto& row : a) {
        if (row.size() != n) throw std::invalid_argument("linear iteration: nonsquare matrix");
        finite_vector(row);
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (!symmetric && a[i][i] == 0)
            throw std::invalid_argument("Gauss-Seidel: zero diagonal");
        if (symmetric)
            for (std::size_t j = 0; j < i; ++j)
                if (a[i][j] != a[j][i]) throw std::invalid_argument("CG: matrix is not exactly symmetric");
    }
}

/// @brief Widened checked inner product of matching internal vectors.
/// @param a First vector.
/// @param b Equally sized second vector.
/// @return Finite working inner product.
/// @throws std::overflow_error Working sum/product is nonfinite.
long double inner(const Vector& a, const Vector& b) {
    long double s = 0;
    for (std::size_t i = 0; i < a.size(); ++i) s += static_cast<long double>(a[i]) * b[i];
    if (!std::isfinite(s)) throw std::overflow_error("CG: working inner product overflow");
    return s;
}

/// @brief Scaled Euclidean norm without squared-double intermediates.
/// @param x Finite vector.
/// @return Working norm.
/// @throws std::overflow_error Working norm is nonfinite.
long double norm(const Vector& x) {
    long double s = 0;
    for (double v : x) s = std::hypot(s, static_cast<long double>(v));
    if (!std::isfinite(s)) throw std::overflow_error("linear iteration: working norm overflow");
    return s;
}

/// @brief Apply a validated dense matrix with widened row sums.
/// @param a Validated matrix.
/// @param x Matching finite vector.
/// @return Owned finite product rounded to double.
/// @throws std::overflow_error A row product cannot fit double.
Vector dense_apply(const Matrix& a, const Vector& x) {
    Vector y(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) y[i] = narrow(inner(a[i], x));
    return y;
}

/// @brief Recompute a residual norm from a validated operator result.
/// @param ax Rounded finite A x product.
/// @param b Matching finite RHS.
/// @return Working norm of b - A x with widened differences.
/// @throws std::overflow_error Working norm is nonfinite.
long double true_norm(const Vector& ax, const Vector& b) {
    long double s = 0;
    for (std::size_t i = 0; i < b.size(); ++i)
        s = std::hypot(s, static_cast<long double>(b[i]) - ax[i]);
    if (!std::isfinite(s)) throw std::overflow_error("linear iteration: true norm overflow");
    return s;
}

/// @brief Execute the single checked CG recurrence with optional actual-state capture.
/// @param apply Nonempty operator; its returned vectors are validated on every call.
/// @param b Validated finite RHS.
/// @param o Validated controls.
/// @param record Capture four actual vectors per completed step when true.
/// @return Owned result; no replay is performed for diagnostics or capture.
/// @throws std::invalid_argument Invalid operator output size/finiteness.
/// @throws std::length_error Online snapshot cap or product counter overflows.
/// @throws std::overflow_error Nonfinite/unrepresentable recurrence arithmetic.
ConjugateGradientRun run(const LinearOperator& apply, const Vector& b,
                        ConjugateGradientOptions o, bool record) {
    ConjugateGradientRun out;
    const auto n = b.size();
    auto& result = out.solution;
    result.x.assign(n, 0);
    out.initial_residual = out.recursive_residual = norm(b);
    result.residual = narrow(out.initial_residual);
    if (out.initial_residual <= o.tolerance) {
        result.converged = true;
        out.stop = CGStop::converged;
        return out;
    }
    if (o.max_iterations == 0) return out;
    Vector r = b, p = r;
    long double rs = inner(r, r);
    if (rs == 0) throw std::overflow_error("CG: working residual square underflow");
    std::size_t stored = 0;
    /// @brief Count and validate one actual operator request.
    /// @param v Finite argument vector.
    /// @return Owned matching finite output.
    /// @throws std::length_error Product counter cannot increment.
    /// @throws std::invalid_argument Callback returns an invalid vector.
    const auto product = [&](const Vector& v) {
        if (out.products == std::numeric_limits<std::size_t>::max())
            throw std::length_error("CG: product counter overflow");
        ++out.products;
        auto y = apply(v);
        if (y.size() != n) throw std::invalid_argument("CG: wrong operator result size");
        finite_vector(y);
        return y;
    };
    for (int k = 0; k < o.max_iterations; ++k) {
        auto ap = product(p);
        const long double curvature = inner(p, ap);
        if (curvature <= 0) { out.stop = CGStop::nonpositive_curvature; break; }
        if (record) {
            if (n > (std::numeric_limits<std::size_t>::max() - stored) / 4)
                throw std::length_error("CG: snapshot size overflow");
            const auto next = stored + 4*n;
            if (o.max_trace_values && next > o.max_trace_values)
                throw std::length_error("CG: actual snapshot cap");
            stored = next;
        }
        const long double alpha = rs / curvature;
        if (!std::isfinite(alpha)) throw std::overflow_error("CG: nonfinite step");
        for (std::size_t i = 0; i < n; ++i) {
            result.x[i] = narrow(static_cast<long double>(result.x[i]) + alpha*p[i]);
            r[i] = narrow(static_cast<long double>(r[i]) - alpha*ap[i]);
        }
        result.iterations = k + 1;
        out.recursive_residual = norm(r);
        const auto actual = true_norm(product(result.x), b);
        result.residual = narrow(actual);
        const long double next_rs = inner(r, r);
        long double beta = 0;
        bool next_direction = false;
        if (actual <= o.tolerance) {
            result.converged = true;
            out.stop = CGStop::converged;
        } else if (out.recursive_residual <= o.tolerance || next_rs == 0) {
            out.stop = CGStop::residual_gap;
        } else if (k + 1 < o.max_iterations) {
            beta = next_rs / rs;
            if (!std::isfinite(beta)) throw std::overflow_error("CG: nonfinite direction coefficient");
            next_direction = true;
        }
        if (record) out.steps.push_back({k + 1, result.x, r, p, ap, alpha, beta,
            out.recursive_residual, actual, next_direction});
        if (result.converged || out.stop == CGStop::residual_gap) break;
        if (next_direction)
            for (std::size_t i = 0; i < n; ++i)
                p[i] = narrow(static_cast<long double>(r[i]) + beta*p[i]);
        rs = next_rs;
    }
    return out;
}

/// @brief Add checked history entries or actual arithmetic counters.
/// @param a Current count.
/// @param b Increment.
/// @return Representable sum.
/// @throws std::length_error Count overflows size_t.
std::size_t seidel_add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::length_error("Gauss-Seidel: count overflow");
    return a + b;
}

/// @brief Multiply checked history dimensions or actual product counts.
/// @param a First count.
/// @param b Second count.
/// @return Representable product.
/// @throws std::length_error Count overflows size_t.
std::size_t seidel_multiply(std::size_t a, std::size_t b) {
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::length_error("Gauss-Seidel: count overflow");
    return a * b;
}

/// @brief Execute forward in-place Gauss-Seidel, optionally keeping actual residual norms.
/// @param a Validated finite matrix with nonzero diagonal.
/// @param b Validated finite RHS.
/// @param max_iter Nonnegative budget.
/// @param tol Validated absolute tolerance.
/// @param max_history Actual entry cap, zero unlimited.
/// @param record Keep initial and post-sweep residuals when true.
/// @param work Optional actual work output, initially zero.
/// @param history Optional full state recorder admitted before this call.
/// @return Owned solution/history from this execution.
/// @throws std::length_error Actual history cap exceeded.
/// @throws std::overflow_error Computed arithmetic cannot fit.
GaussSeidelTrace stationary(const Matrix& a, const Vector& b, int max_iter,
                           double tol, std::size_t max_history, bool record,
                           GaussSeidelWork* work = nullptr,
                           GaussSeidelHistory* history = nullptr) {
    GaussSeidelTrace out;
    auto& s = out.solution;
    s.x.assign(b.size(), 0);
    s.residual = narrow(norm(b));
    if (work) work->residual_evaluations = 1;
    if (history) {
        history->initial = s.x;
        history->initial_residual = s.residual;
        history->captured_entries = seidel_add(b.size(), 1);
    }
    if (record) out.residuals.push_back(s.residual);
    s.converged = s.residual <= tol;
    for (int k = 0; k < max_iter && !s.converged; ++k) {
        if (record && max_history && out.residuals.size() >= max_history)
            throw std::length_error("Gauss-Seidel: actual history cap");
        for (std::size_t i = 0; i < b.size(); ++i) {
            long double value = b[i];
            for (std::size_t j = 0; j < b.size(); ++j)
                if (i != j) value -= static_cast<long double>(a[i][j])*s.x[j];
            GaussSeidelCoordinate event;
            if (history) {
                event.sweep = k + 1;
                event.row = i;
                event.numerator = value;
                event.previous = s.x[i];
                event.before = s.x;
            }
            s.x[i] = narrow(value/a[i][i]);
            if (work) {
                work->coordinate_updates = seidel_add(work->coordinate_updates, 1);
                work->divisions = seidel_add(work->divisions, 1);
                work->off_diagonal_products = seidel_add(work->off_diagonal_products, b.size()-1);
            }
            if (history) {
                event.value = s.x[i];
                event.after = s.x;
                history->coordinates.push_back(std::move(event));
                history->captured_entries = seidel_add(history->captured_entries,
                    seidel_add(1, seidel_multiply(2, b.size())));
            }
        }
        s.iterations = k + 1;
        s.residual = narrow(true_norm(dense_apply(a, s.x), b));
        if (work) {
            work->residual_evaluations = seidel_add(work->residual_evaluations, 1);
            work->residual_products = seidel_add(work->residual_products,
                seidel_multiply(b.size(), b.size()));
        }
        if (history) {
            history->sweeps.push_back({k + 1, s.x, s.residual, *work});
            history->captured_entries = seidel_add(history->captured_entries,
                seidel_add(1, b.size()));
        }
        if (record) out.residuals.push_back(s.residual);
        s.converged = s.residual <= tol;
    }
    return out;
}
} // namespace

ConjugateGradientRun conjugate_gradient_solve(const std::vector<std::vector<double>>& a, const std::vector<double>& b, ConjugateGradientOptions options) {
    controls(b, options.max_iterations, options.tolerance);
    matrix_input(a, b.size(), true);
    return run([&](const Vector& v) { return dense_apply(a, v); }, b, options, false);
}
ConjugateGradientRun conjugate_gradient_trace(const std::vector<std::vector<double>>& a, const std::vector<double>& b, ConjugateGradientOptions options) {
    controls(b, options.max_iterations, options.tolerance);
    matrix_input(a, b.size(), true);
    return run([&](const Vector& v) { return dense_apply(a, v); }, b, options, true);
}
ConjugateGradientRun conjugate_gradient_solve(const LinearOperator& apply, const std::vector<double>& b, ConjugateGradientOptions options) {
    controls(b, options.max_iterations, options.tolerance);
    if (!apply) throw std::invalid_argument("CG: empty operator");
    return run(apply, b, options, false);
}
ConjugateGradientRun conjugate_gradient_trace(const LinearOperator& apply, const std::vector<double>& b, ConjugateGradientOptions options) {
    controls(b, options.max_iterations, options.tolerance);
    if (!apply) throw std::invalid_argument("CG: empty operator");
    return run(apply, b, options, true);
}
IterativeSolution conjugate_gradient(const std::vector<std::vector<double>>& a, const std::vector<double>& b, int max_iter, double tol) {
    return conjugate_gradient_solve(a, b, {max_iter, tol, 0}).solution;
}
GaussSeidelTrace gauss_seidel_trace(const std::vector<std::vector<double>>& a, const std::vector<double>& b, int max_iter, double tol, std::size_t max_history) {
    controls(b, max_iter, tol);
    matrix_input(a, b.size(), false);
    return stationary(a, b, max_iter, tol, max_history, true);
}
IterativeSolution gauss_seidel(const std::vector<std::vector<double>>& a, const std::vector<double>& b, int max_iter, double tol) {
    controls(b, max_iter, tol);
    matrix_input(a, b.size(), false);
    return stationary(a, b, max_iter, tol, 0, false).solution;
}
std::size_t gauss_seidel_history_entries(std::size_t n, int max_iter) {
    if (max_iter < 0) throw std::invalid_argument("Gauss-Seidel: negative budget");
    const auto initial = seidel_add(n, 1);
    if (!n || !max_iter) return initial;
    const auto coordinate = seidel_add(1, seidel_multiply(2, n));
    const auto sweep = seidel_add(seidel_multiply(n, coordinate), seidel_add(1, n));
    return seidel_add(initial, seidel_multiply(static_cast<std::size_t>(max_iter), sweep));
}
GaussSeidelRun gauss_seidel_solve(const std::vector<std::vector<double>>& a, const std::vector<double>& b, int max_iter, double tol) {
    controls(b, max_iter, tol);
    matrix_input(a, b.size(), false);
    GaussSeidelRun out;
    out.solution = stationary(a, b, max_iter, tol, 0, false, &out.work).solution;
    return out;
}
GaussSeidelHistory gauss_seidel_history(const std::vector<std::vector<double>>& a, const std::vector<double>& b, int max_iter,
                                      double tol, std::size_t max_entries) {
    controls(b, max_iter, tol);
    matrix_input(a, b.size(), false);
    const auto entries = gauss_seidel_history_entries(b.size(), max_iter);
    if (max_entries && entries > max_entries)
        throw std::length_error("Gauss-Seidel: whole history cap");
    if (b.size() > Vector{}.max_size() || (b.size() &&
        (seidel_multiply(b.size(), static_cast<std::size_t>(max_iter)) >
         std::vector<GaussSeidelCoordinate>{}.max_size() ||
         static_cast<std::size_t>(max_iter) > std::vector<GaussSeidelSweep>{}.max_size())))
        throw std::length_error("Gauss-Seidel: history vector capacity");
    GaussSeidelHistory out;
    out.run.solution = stationary(a, b, max_iter, tol, 0, false, &out.run.work, &out).solution;
    return out;
}
Matrix laplacian_1d_matrix(std::size_t n, std::size_t cap) {
    if (n && n > std::numeric_limits<std::size_t>::max()/n)
        throw std::length_error("Laplacian: matrix size overflow");
    if (cap && n*n > cap) throw std::length_error("Laplacian: complete matrix cap");
    Matrix a(n, Vector(n));
    for (std::size_t i = 0; i < n; ++i) {
        a[i][i] = 2;
        if (i) a[i][i-1] = -1;
        if (i+1 < n) a[i][i+1] = -1;
    }
    return a;
}
std::vector<double> laplacian_1d_apply(const std::vector<double>& x) {
    finite_vector(x);
    Vector y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        long double s = 2*static_cast<long double>(x[i]);
        if (i) s -= x[i-1];
        if (i+1 < x.size()) s -= x[i+1];
        y[i] = narrow(s);
    }
    return y;
}
Vector laplacian_1d_unit_solution(std::size_t n, std::size_t cap) {
    if (cap && n > cap) throw std::length_error("Laplacian: complete dimension cap");
    Vector x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = narrow((static_cast<long double>(i)+1)*(n-i)/2);
    return x;
}
} // namespace datamunge::algorithms
