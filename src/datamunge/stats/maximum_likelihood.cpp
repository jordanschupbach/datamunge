/// @file maximum_likelihood.cpp
/// @brief Bound transforms, BFGS maximization and finite-difference information for maximum_likelihood().

#include <datamunge/stats/maximum_likelihood.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Parameter transforms, numerical derivatives and dense helpers.
namespace {

/// @brief Dense double matrix used by the helpers.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Positive infinity.
constexpr double infinity = std::numeric_limits<double>::infinity();

/// @brief Problem definition shared by the helpers.
struct Problem {
    std::size_t n;                          ///< Number of observations.
    std::size_t p;                          ///< Number of parameters.
    const ObservationLogLikelihood& loglik; ///< Per-observation log likelihood.
    const ObservationScore& score;          ///< Optional analytic score.
    std::vector<double> lower;              ///< Lower bounds (negative infinity when absent).
    std::vector<double> upper;              ///< Upper bounds (positive infinity when absent).
    double relative_step;                   ///< Relative finite-difference step.
};

/// @brief Maps an unconstrained coordinate to the parameter scale.
/// @param u Unconstrained value.
/// @param lo Lower bound.
/// @param hi Upper bound.
/// @return Parameter value.
double to_theta(double u, double lo, double hi) {
    if (std::isfinite(lo) && std::isfinite(hi))
        return lo+(hi-lo)/(1+std::exp(-u));
    if (std::isfinite(lo))
        return lo+std::exp(u);
    if (std::isfinite(hi))
        return hi-std::exp(u);
    return u;
}

/// @brief Inverse of to_theta() for a value strictly inside the bounds.
/// @param theta Parameter value.
/// @param lo Lower bound.
/// @param hi Upper bound.
/// @return Unconstrained value.
double to_u(double theta, double lo, double hi) {
    if (std::isfinite(lo) && std::isfinite(hi))
        return std::log((theta-lo)/(hi-theta));
    if (std::isfinite(lo))
        return std::log(theta-lo);
    if (std::isfinite(hi))
        return std::log(hi-theta);
    return theta;
}

/// @brief Derivative \f$d\theta/du\f$ at a parameter value.
/// @param theta Parameter value.
/// @param lo Lower bound.
/// @param hi Upper bound.
/// @return Jacobian entry.
double jacobian(double theta, double lo, double hi) {
    if (std::isfinite(lo) && std::isfinite(hi))
        return (theta-lo)*(hi-theta)/(hi-lo);
    if (std::isfinite(lo))
        return theta-lo;
    if (std::isfinite(hi))
        return -(hi-theta);
    return 1;
}

/// @brief Total log likelihood, throwing on NaN.
/// @param problem Problem definition.
/// @param theta Parameters.
/// @return Sum of contributions (may be negative infinity).
double total(const Problem& problem, const std::vector<double>& theta) {
    double sum = 0;
    for (std::size_t i = 0; i < problem.n; ++i) {
        const double v = problem.loglik(theta,i);
        if (std::isnan(v) || v == infinity)
            throw std::invalid_argument("maximum_likelihood: log likelihood returned NaN or +infinity");
        sum += v;
    }
    return sum;
}

/// @brief Finite-difference step for parameter j that keeps both sides inside the bounds when possible.
/// @param problem Problem definition.
/// @param theta Parameters.
/// @param j Parameter index.
/// @param scale Multiplier on the relative step.
/// @param forward Set to +1/-1 for a one-sided step, 0 for central.
/// @return Step size.
double step_for(const Problem& problem, const std::vector<double>& theta, std::size_t j, double scale, int& forward) {
    const double h = scale*problem.relative_step*std::max(1.0,std::abs(theta[j]));
    const bool up = theta[j]+h <= problem.upper[j], down = theta[j]-h >= problem.lower[j];
    forward = up && down ? 0 : (up ? 1 : -1);
    return h;
}

/// @brief Score contributions \f$s_i(\theta)\f$ as an n×p matrix, analytic or by finite differences.
/// @param problem Problem definition.
/// @param theta Parameters.
/// @return Row i holds observation i's score.
Matrix score_matrix(const Problem& problem, const std::vector<double>& theta) {
    Matrix s(problem.n,problem.p,0.0);
    if (problem.score) {
        for (std::size_t i = 0; i < problem.n; ++i) {
            const auto g = problem.score(theta,i);
            if (g.size() != problem.p)
                throw std::invalid_argument("maximum_likelihood: score must have one entry per parameter");
            for (std::size_t j = 0; j < problem.p; ++j) {
                if (std::isnan(g[j]))
                    throw std::invalid_argument("maximum_likelihood: score returned NaN");
                s(i,j) = g[j];
            }
        }
        return s;
    }
    auto shifted = theta;
    for (std::size_t j = 0; j < problem.p; ++j) {
        int forward;
        const double h = step_for(problem,theta,j,1.0,forward);
        for (std::size_t i = 0; i < problem.n; ++i) {
            const double centre = problem.loglik(theta,i);
            shifted[j] = forward >= 0 ? theta[j]+h : theta[j];
            const double above = forward >= 0 ? problem.loglik(shifted,i) : centre;
            shifted[j] = forward <= 0 ? theta[j]-h : theta[j];
            const double below = forward <= 0 ? problem.loglik(shifted,i) : centre;
            shifted[j] = theta[j];
            s(i,j) = (above-below)/(forward == 0 ? 2*h : h);
        }
    }
    return s;
}

/// @brief Column sums of the score matrix.
/// @param problem Problem definition.
/// @param theta Parameters.
/// @return Total score.
std::vector<double> total_score(const Problem& problem, const std::vector<double>& theta) {
    const auto s = score_matrix(problem,theta);
    std::vector<double> g(problem.p,0.0);
    for (std::size_t i = 0; i < problem.n; ++i)
        for (std::size_t j = 0; j < problem.p; ++j)
            g[j] += s(i,j);
    return g;
}

/// @brief Inverse of a symmetric matrix by Gauss–Jordan elimination with partial pivoting.
/// @param a Matrix to invert.
/// @param ok Set false when a pivot is negligible.
/// @return The inverse (unspecified when not ok).
Matrix invert(Matrix a, bool& ok) {
    const std::size_t p = a.rows();
    Matrix inv(p,p,0.0);
    for (std::size_t j = 0; j < p; ++j)
        inv(j,j) = 1;
    double scale = 0;
    for (const double v : a.data())
        scale = std::max(scale,std::abs(v));
    ok = scale > 0 && std::isfinite(scale);
    for (std::size_t c = 0; c < p && ok; ++c) {
        std::size_t pivot = c;
        for (std::size_t r = c+1; r < p; ++r)
            if (std::abs(a(r,c)) > std::abs(a(pivot,c)))
                pivot = r;
        if (std::abs(a(pivot,c)) <= 1e-12*scale) {
            ok = false;
            break;
        }
        for (std::size_t k = 0; k < p; ++k) {
            std::swap(a(c,k),a(pivot,k));
            std::swap(inv(c,k),inv(pivot,k));
        }
        const double d = a(c,c);
        for (std::size_t k = 0; k < p; ++k) {
            a(c,k) /= d;
            inv(c,k) /= d;
        }
        for (std::size_t r = 0; r < p; ++r)
            if (r != c && a(r,c) != 0) {
                const double f = a(r,c);
                for (std::size_t k = 0; k < p; ++k) {
                    a(r,k) -= f*a(c,k);
                    inv(r,k) -= f*inv(c,k);
                }
            }
    }
    return inv;
}

/// @brief Fills estimate-dependent fields: scores, information, covariances and bound flags.
/// @param problem Problem definition.
/// @param theta Parameters at which to evaluate.
/// @param log_likelihood Total log likelihood at theta.
/// @param fit Result to complete.
void complete_fit(const Problem& problem, const std::vector<double>& theta, double log_likelihood,
                  MaximumLikelihoodFit& fit) {
    const std::size_t n = problem.n, p = problem.p;
    const bool score = static_cast<bool>(problem.score);
    fit.estimate = theta;
    fit.log_likelihood = log_likelihood;
    fit.score_contributions = score_matrix(problem,theta);
    fit.outer_product = Matrix(p,p,0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t r = 0; r < p; ++r)
            for (std::size_t c = 0; c < p; ++c)
                fit.outer_product(r,c) += fit.score_contributions(i,r)*fit.score_contributions(i,c);
    // Observed information: negated Jacobian of the total score, by (one-sided near bounds) differences.
    Matrix info(p,p,0.0);
    auto shifted = theta;
    for (std::size_t j = 0; j < p; ++j) {
        int forward;
        const double step = step_for(problem,theta,j,score ? 1.0 : 10.0,forward);
        shifted[j] = forward >= 0 ? theta[j]+step : theta[j];
        const auto above = total_score(problem,shifted);
        shifted[j] = forward <= 0 ? theta[j]-step : theta[j];
        const auto below = total_score(problem,shifted);
        shifted[j] = theta[j];
        for (std::size_t r = 0; r < p; ++r)
            info(r,j) = -(above[r]-below[r])/(forward == 0 ? 2*step : step);
    }
    fit.observed_information = Matrix(p,p,0.0);
    for (std::size_t r = 0; r < p; ++r)
        for (std::size_t c = 0; c < p; ++c)
            fit.observed_information(r,c) = .5*(info(r,c)+info(c,r));
    bool ok;
    const auto inverse = invert(fit.observed_information,ok);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    fit.covariance = ok ? inverse : Matrix(p,p,nan);
    fit.robust_covariance = ok ? inverse*fit.outer_product*inverse : Matrix(p,p,nan);
    fit.standard_errors.resize(p);
    fit.at_bound.resize(p);
    for (std::size_t j = 0; j < p; ++j) {
        fit.standard_errors[j] = ok && fit.covariance(j,j) >= 0 ? std::sqrt(fit.covariance(j,j)) : nan;
        const auto near = [&](double b) { return std::isfinite(b) && std::abs(theta[j]-b) <= 1e-6*std::max(1.0,std::abs(b)); };
        fit.at_bound[j] = near(problem.lower[j]) || near(problem.upper[j]);
    }
}

/// @brief Validates inputs shared by maximum_likelihood() and evaluate_likelihood() and builds the problem.
/// @param n Number of observations.
/// @param theta Parameter vector (start or evaluation point).
/// @param log_likelihood Per-observation log likelihood.
/// @param options Bounds and steps.
/// @param score Optional analytic score.
/// @param strict Whether theta must be strictly inside the bounds (otherwise inclusive).
/// @return The problem.
/// @throws std::invalid_argument On invalid input.
Problem make_problem(std::size_t n, const std::vector<double>& theta, const ObservationLogLikelihood& log_likelihood,
                     const MaximumLikelihoodOptions& options, const ObservationScore& score, bool strict) {
    const std::size_t p = theta.size();
    if (n == 0 || p == 0 || !log_likelihood)
        throw std::invalid_argument("maximum_likelihood: need observations, parameters and a log likelihood");
    if ((!options.lower.empty() && options.lower.size() != p) || (!options.upper.empty() && options.upper.size() != p))
        throw std::invalid_argument("maximum_likelihood: bounds must be empty or have one entry per parameter");
    if (!(options.gradient_tolerance > 0) || !(options.relative_step > 0) || options.relative_step > .1)
        throw std::invalid_argument("maximum_likelihood: tolerance and step must be positive (step at most 0.1)");
    Problem problem{n,p,log_likelihood,score,options.lower.empty() ? std::vector<double>(p,-infinity) : options.lower,
                    options.upper.empty() ? std::vector<double>(p,infinity) : options.upper,options.relative_step};
    for (std::size_t j = 0; j < p; ++j) {
        const bool inside = strict ? theta[j] > problem.lower[j] && theta[j] < problem.upper[j]
                                   : theta[j] >= problem.lower[j] && theta[j] <= problem.upper[j];
        if (std::isnan(problem.lower[j]) || std::isnan(problem.upper[j]) || problem.lower[j] == infinity ||
            problem.upper[j] == -infinity || !(problem.lower[j] < problem.upper[j]) || !std::isfinite(theta[j]) || !inside)
            throw std::invalid_argument("maximum_likelihood: parameters must be finite and inside lower < upper");
    }
    if (!std::isfinite(total(problem,theta)))
        throw std::invalid_argument("maximum_likelihood: log likelihood must be finite at the supplied parameters");
    return problem;
}

} // namespace

MaximumLikelihoodFit maximum_likelihood(std::size_t n, const std::vector<double>& start,
                                        const ObservationLogLikelihood& log_likelihood,
                                        const MaximumLikelihoodOptions& options, const ObservationScore& score) {
    const std::size_t p = start.size();
    const Problem problem = make_problem(n,start,log_likelihood,options,score,true);

    std::vector<double> u(p), theta(start);
    for (std::size_t j = 0; j < p; ++j)
        u[j] = to_u(start[j],problem.lower[j],problem.upper[j]);
    const auto to_parameters = [&](const std::vector<double>& coords) {
        std::vector<double> t(p);
        for (std::size_t j = 0; j < p; ++j)
            t[j] = to_theta(coords[j],problem.lower[j],problem.upper[j]);
        return t;
    };
    // Minimize f(u) = -loglik(theta(u)) with gradient -diag(dtheta/du) * score(theta).
    const auto gradient_u = [&](const std::vector<double>& t) {
        auto g = total_score(problem,t);
        for (std::size_t j = 0; j < p; ++j)
            g[j] = -g[j]*jacobian(t[j],problem.lower[j],problem.upper[j]);
        return g;
    };
    double f = -total(problem,theta);
    auto g = gradient_u(theta);
    Matrix h(p,p,0.0);
    for (std::size_t j = 0; j < p; ++j)
        h(j,j) = 1/std::max(1.0,std::abs(g[j]));
    MaximumLikelihoodFit fit;
    const auto max_abs = [](const std::vector<double>& v) {
        double m = 0;
        for (const double x : v)
            m = std::max(m,std::abs(x));
        return m;
    };
    for (; fit.iterations < options.max_iterations; ++fit.iterations) {
        if (max_abs(g) <= options.gradient_tolerance) {
            fit.converged = true;
            break;
        }
        std::vector<double> d(p,0.0);
        double slope = 0;
        for (std::size_t r = 0; r < p; ++r) {
            for (std::size_t c = 0; c < p; ++c)
                d[r] -= h(r,c)*g[c];
            slope += d[r]*g[r];
        }
        if (!(slope < 0)) { // Not a descent direction: restart from steepest descent.
            for (std::size_t r = 0; r < p; ++r)
                for (std::size_t c = 0; c < p; ++c)
                    h(r,c) = r == c ? 1/std::max(1.0,max_abs(g)) : 0;
            for (std::size_t r = 0; r < p; ++r)
                d[r] = -h(r,r)*g[r];
            slope = 0;
            for (std::size_t r = 0; r < p; ++r)
                slope += d[r]*g[r];
        }
        double alpha = 1, f_new = infinity;
        std::vector<double> u_new(p), theta_new;
        for (int trial = 0; trial < 60; ++trial, alpha *= .5) {
            for (std::size_t j = 0; j < p; ++j)
                u_new[j] = u[j]+alpha*d[j];
            theta_new = to_parameters(u_new);
            bool inside = true;
            for (std::size_t j = 0; j < p; ++j)
                inside = inside && std::isfinite(theta_new[j]) && theta_new[j] > problem.lower[j] &&
                         theta_new[j] < problem.upper[j];
            if (!inside)
                continue;
            f_new = -total(problem,theta_new);
            if (std::isfinite(f_new) && f_new <= f+1e-4*alpha*slope)
                break;
            f_new = infinity;
        }
        if (!std::isfinite(f_new))
            break; // Line search failed; report non-convergence unless the gradient is already small.
        const auto g_new = gradient_u(theta_new);
        std::vector<double> s(p), y(p);
        double sy = 0, yy = 0;
        for (std::size_t j = 0; j < p; ++j) {
            s[j] = u_new[j]-u[j];
            y[j] = g_new[j]-g[j];
            sy += s[j]*y[j];
            yy += y[j]*y[j];
        }
        if (sy > 1e-12*std::sqrt(yy)*max_abs(s)) {
            if (fit.iterations == 0)
                for (std::size_t j = 0; j < p; ++j)
                    for (std::size_t k = 0; k < p; ++k)
                        h(j,k) = j == k ? sy/yy : 0;
            std::vector<double> hy(p,0.0);
            double yhy = 0;
            for (std::size_t r = 0; r < p; ++r) {
                for (std::size_t c = 0; c < p; ++c)
                    hy[r] += h(r,c)*y[c];
                yhy += y[r]*hy[r];
            }
            const double rho = 1/sy;
            for (std::size_t r = 0; r < p; ++r)
                for (std::size_t c = 0; c < p; ++c)
                    h(r,c) += -rho*(hy[r]*s[c]+s[r]*hy[c])+(rho*rho*yhy+rho)*s[r]*s[c];
        }
        u = u_new;
        theta = theta_new;
        f = f_new;
        g = g_new;
    }
    if (!fit.converged && max_abs(g) <= options.gradient_tolerance)
        fit.converged = true;

    complete_fit(problem,theta,-f,fit);
    return fit;
}

MaximumLikelihoodFit evaluate_likelihood(std::size_t n, const std::vector<double>& theta,
                                         const ObservationLogLikelihood& log_likelihood,
                                         const MaximumLikelihoodOptions& options, const ObservationScore& score) {
    const Problem problem = make_problem(n,theta,log_likelihood,options,score,false);
    MaximumLikelihoodFit fit;
    complete_fit(problem,theta,total(problem,theta),fit);
    return fit;
}

} // namespace datamunge::stats
