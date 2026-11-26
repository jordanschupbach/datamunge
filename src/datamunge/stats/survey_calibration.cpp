/// @file survey_calibration.cpp
/// @brief Newton calibration with linear, raking and logit distances, and iterative proportional fitting.

#include <datamunge/stats/survey_calibration.hpp>

#include <datamunge/linalg/cholesky.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Internal helpers for weight calibration.
namespace {

/// @brief Adjustment \f$g=F(u)\f$ and its derivative for a distance, following `survey`'s calfun objects.
/// @param distance Distance.
/// @param u Linear predictor \f$x^\top\lambda\f$.
/// @param lower Lower bound on g.
/// @param upper Upper bound on g.
/// @param derivative Receives \f$F'(u)\f$ (0 where a bound is active).
/// @return \f$g\f$.
double adjustment(CalibrationDistance distance, double u, double lower, double upper, double& derivative) {
    switch (distance) {
    case CalibrationDistance::linear:
        derivative = (u+1 < upper && u+1 > lower) ? 1.0 : 0.0;
        return std::clamp(u+1,lower,upper);
    case CalibrationDistance::raking: {
        const double e = std::exp(u);
        derivative = (u < upper-1 && u > lower-1) ? e : 0.0;
        return std::clamp(e,lower,upper);
    }
    case CalibrationDistance::logit: {
        const double a = (upper-lower)/((upper-1)*(1-lower));
        const double e = std::exp(a*u);
        const double den = upper-1+(1-lower)*e;
        const double g = (lower*(upper-1)+upper*(1-lower)*e)/den;
        derivative = upper*(1-lower)*e*a/den-g*(1-lower)*e*a/den;
        return g;
    }
    }
    throw std::invalid_argument("calibrate_weights: unknown distance");
}

/// @brief Numerically stable \f$\log(1+e^x)\f$.
/// @param x Argument.
/// @return Softplus of x.
double softplus(double x) {
    return x > 0 ? x+std::log1p(std::exp(-x)) : std::log1p(std::exp(x));
}

/// @brief Antiderivative \f$\Phi(u)=\int_0^uF(s)\,ds\f$ of the adjustment function.
///
/// The calibration equations are the stationarity conditions of the convex dual
/// \f$D(\lambda)=\sum_iw_i\Phi(x_i^\top\lambda)-\lambda^\top T\f$, which the line search decreases.
/// @param distance Distance.
/// @param u Upper limit of integration.
/// @param lower Lower bound on g.
/// @param upper Upper bound on g.
/// @return \f$\Phi(u)\f$.
double antiderivative(CalibrationDistance distance, double u, double lower, double upper) {
    switch (distance) {
    case CalibrationDistance::linear: {
        const double knot = u >= 0 ? upper-1 : lower-1, bound = u >= 0 ? upper : lower;
        if (std::abs(u) <= std::abs(knot))
            return u+u*u/2;
        return knot+knot*knot/2+bound*(u-knot);
    }
    case CalibrationDistance::raking: {
        const double bound = u >= 0 ? upper : lower;
        const double knot = bound > 0 ? std::log(bound) : -std::numeric_limits<double>::infinity();
        if (u >= 0 ? u <= knot : u >= knot)
            return std::expm1(u);
        return bound-1+bound*(u-knot);
    }
    case CalibrationDistance::logit: {
        // F(s) = L + (U-L) * logistic(A s + c) with c = log((1-L)/(U-1)).
        const double a = (upper-lower)/((upper-1)*(1-lower));
        const double c = std::log((1-lower)/(upper-1));
        return lower*u+(upper-lower)/a*(softplus(a*u+c)-softplus(c));
    }
    }
    return 0;
}

} // namespace

CalibrationResult calibrate_weights(const std::vector<double>& weights, const linalg::DenseMatrix<double>& auxiliary,
                                    const std::vector<double>& targets, const CalibrationOptions& options) {
    const std::size_t n = weights.size(), p = auxiliary.cols();
    if (n == 0 || auxiliary.rows() != n || p == 0 || targets.size() != p)
        throw std::invalid_argument("calibrate_weights: need one auxiliary row per weight and one target per column");
    if (options.distance != CalibrationDistance::linear && options.distance != CalibrationDistance::raking &&
        options.distance != CalibrationDistance::logit)
        throw std::invalid_argument("calibrate_weights: unknown distance");
    if (!(options.lower < 1 && options.upper > 1) || std::isnan(options.lower) || std::isnan(options.upper))
        throw std::invalid_argument("calibrate_weights: bounds must satisfy lower < 1 < upper");
    if (options.distance == CalibrationDistance::logit && !(std::isfinite(options.lower) && std::isfinite(options.upper)))
        throw std::invalid_argument("calibrate_weights: logit calibration needs finite bounds");
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(weights[i]) || weights[i] <= 0)
            throw std::invalid_argument("calibrate_weights: weights must be finite and positive");
        for (std::size_t a = 0; a < p; ++a)
            if (!std::isfinite(auxiliary(i,a)))
                throw std::invalid_argument("calibrate_weights: auxiliary values must be finite");
    }
    for (const double t : targets)
        if (!std::isfinite(t))
            throw std::invalid_argument("calibrate_weights: targets must be finite");

    std::vector<double> lambda(p,0.0), g(n,1.0), deriv(n,1.0);
    const auto update = [&](const std::vector<double>& l) {
        bool finite = true;
        for (std::size_t i = 0; i < n; ++i) {
            double u = 0;
            for (std::size_t a = 0; a < p; ++a)
                u += auxiliary(i,a)*l[a];
            g[i] = adjustment(options.distance,u,options.lower,options.upper,deriv[i]);
            finite = finite && std::isfinite(g[i]) && std::isfinite(deriv[i]);
        }
        return finite;
    };
    const auto misfit = [&] {
        std::vector<double> m(targets);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t a = 0; a < p; ++a)
                m[a] -= weights[i]*g[i]*auxiliary(i,a);
        return m;
    };
    const auto relative = [&](const std::vector<double>& m) {
        double worst = 0;
        for (std::size_t a = 0; a < p; ++a)
            worst = std::max(worst,std::abs(m[a])/(1+std::abs(targets[a])));
        return worst;
    };
    // Line-search objective: the convex dual, whose gradient is minus the misfit and whose
    // Hessian is the Newton matrix, so damped Newton converges whenever a solution exists.
    const auto dual = [&](const std::vector<double>& l) {
        double s = 0;
        for (std::size_t i = 0; i < n; ++i) {
            double u = 0;
            for (std::size_t a = 0; a < p; ++a)
                u += auxiliary(i,a)*l[a];
            s += weights[i]*antiderivative(options.distance,u,options.lower,options.upper);
        }
        for (std::size_t a = 0; a < p; ++a)
            s -= l[a]*targets[a];
        return s;
    };
    (void)update(lambda);
    CalibrationResult r;
    auto m = misfit();
    while (relative(m) >= options.tolerance && r.iterations < options.max_iterations) {
        linalg::DenseMatrix<double> jacobian(p,p);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t a = 0; a < p; ++a)
                for (std::size_t b = 0; b <= a; ++b)
                    jacobian(a,b) += weights[i]*deriv[i]*auxiliary(i,a)*auxiliary(i,b);
        for (std::size_t a = 0; a < p; ++a)
            for (std::size_t b = 0; b < a; ++b)
                jacobian(b,a) = jacobian(a,b);
        // Jacobi scaling keeps columns on very different scales well conditioned. The scale uses the
        // unweighted design as a floor so that columns whose units all sit at a bound stay usable.
        std::vector<double> scale(p,0.0), rhs(p);
        for (std::size_t a = 0; a < p; ++a) {
            double reference = 0;
            for (std::size_t i = 0; i < n; ++i)
                reference += weights[i]*auxiliary(i,a)*auxiliary(i,a);
            if (!(reference > 0))
                throw std::runtime_error("calibrate_weights: an auxiliary column is identically zero");
            scale[a] = 1/std::sqrt(std::max(jacobian(a,a),1e-12*reference));
        }
        for (std::size_t a = 0; a < p; ++a) {
            rhs[a] = scale[a]*m[a];
            for (std::size_t b = 0; b < p; ++b)
                jacobian(a,b) *= scale[a]*scale[b];
        }
        // Levenberg damping when units at the bounds make the matrix singular after the first step.
        std::vector<double> step;
        for (double damping = 0;; damping = damping == 0 ? 1e-10 : damping*10) {
            auto damped = jacobian;
            for (std::size_t a = 0; a < p; ++a)
                damped(a,a) += damping;
            const auto factor = linalg::cholesky(damped);
            if (factor.ok) {
                step = factor.solve(rhs);
                break;
            }
            if (r.iterations == 0)
                throw std::runtime_error("calibrate_weights: singular constraint matrix (linearly dependent columns)");
            if (damping > 1e10)
                break;
        }
        if (step.empty())
            break;
        for (std::size_t a = 0; a < p; ++a)
            step[a] *= scale[a];
        std::vector<double> candidate(p);
        // Armijo backtracking on the dual. Near convergence the dual is dominated by rounding,
        // so a step that strictly reduces the largest relative misfit is also accepted.
        const double before = dual(lambda), before_misfit = relative(m);
        double slope = 0;
        for (std::size_t a = 0; a < p; ++a)
            slope -= m[a]*step[a];
        std::vector<double> trial;
        double t = 1;
        bool accepted = false;
        for (int halving = 0; halving <= 60 && !accepted; ++halving, t /= 2) {
            for (std::size_t a = 0; a < p; ++a)
                candidate[a] = lambda[a]+t*step[a];
            if (!update(candidate))
                continue;
            trial = misfit();
            const double value = dual(candidate);
            const bool rounding = std::abs(t*slope) < 1e-12*(1+std::abs(before)); // dual change lost to rounding
            accepted = (std::isfinite(value) && value <= before+1e-4*t*slope) ||
                       (rounding && relative(trial) < before_misfit);
        }
        if (!accepted) {
            (void)update(lambda); // no acceptable step: keep the best weights found
            break;
        }
        lambda = candidate;
        m = std::move(trial);
        ++r.iterations;
    }
    r.max_relative_residual = relative(m);
    r.converged = r.max_relative_residual < options.tolerance;
    r.g = g;
    for (std::size_t i = 0; i < n; ++i)
        r.weights.push_back(weights[i]*g[i]);
    for (std::size_t a = 0; a < p; ++a) {
        r.achieved.push_back(targets[a]-m[a]);
        r.residual.push_back(-m[a]);
    }
    return r;
}

CalibrationResult rake_weights(const std::vector<double>& weights, const std::vector<RakingMargin>& margins,
                               const RakingOptions& options) {
    const std::size_t n = weights.size();
    if (n == 0 || margins.empty())
        throw std::invalid_argument("rake_weights: need weights and at least one margin");
    for (const double w : weights)
        if (!std::isfinite(w) || w <= 0)
            throw std::invalid_argument("rake_weights: weights must be finite and positive");
    double grand = -1;
    for (const auto& margin : margins) {
        if (margin.categories.size() != n || margin.totals.empty())
            throw std::invalid_argument("rake_weights: each margin needs a category per unit and category totals");
        double sum = 0;
        std::vector<bool> present(margin.totals.size(),false);
        for (const std::size_t c : margin.categories) {
            if (c >= margin.totals.size())
                throw std::invalid_argument("rake_weights: category index out of range");
            present[c] = true;
        }
        for (std::size_t c = 0; c < margin.totals.size(); ++c) {
            if (!std::isfinite(margin.totals[c]) || margin.totals[c] < 0 || (margin.totals[c] > 0 && !present[c]))
                throw std::invalid_argument("rake_weights: totals must be nonnegative, and positive only for sampled categories");
            sum += margin.totals[c];
        }
        if (grand >= 0 && std::abs(sum-grand) > 1e-9*std::max(1.0,grand))
            throw std::invalid_argument("rake_weights: margins must have the same grand total");
        grand = sum;
    }
    CalibrationResult r;
    r.weights = weights;
    const auto margin_sums = [&](const RakingMargin& margin) {
        std::vector<double> s(margin.totals.size(),0.0);
        for (std::size_t i = 0; i < n; ++i)
            s[margin.categories[i]] += r.weights[i];
        return s;
    };
    const auto worst = [&] {
        double w = 0;
        for (const auto& margin : margins) {
            const auto s = margin_sums(margin);
            for (std::size_t c = 0; c < s.size(); ++c)
                w = std::max(w,std::abs(s[c]-margin.totals[c])/(1+std::abs(margin.totals[c])));
        }
        return w;
    };
    while (worst() >= options.tolerance && r.iterations < options.max_iterations) {
        for (const auto& margin : margins) {
            const auto s = margin_sums(margin);
            for (std::size_t i = 0; i < n; ++i)
                r.weights[i] *= margin.totals[margin.categories[i]]/s[margin.categories[i]];
        }
        ++r.iterations;
    }
    r.max_relative_residual = worst();
    r.converged = r.max_relative_residual < options.tolerance;
    for (std::size_t i = 0; i < n; ++i)
        r.g.push_back(r.weights[i]/weights[i]);
    for (const auto& margin : margins) {
        const auto s = margin_sums(margin);
        for (std::size_t c = 0; c < s.size(); ++c) {
            r.achieved.push_back(s[c]);
            r.residual.push_back(s[c]-margin.totals[c]);
        }
    }
    return r;
}

} // namespace datamunge::stats
