#pragma once

/// @file frank_wolfe.hpp
/// @brief The Frank-Wolfe (conditional gradient) method for smooth convex minimisation over a
///        compact convex set, with the duality-gap certificate, three step rules, away-step and
///        pairwise variants, linear minimisation oracles, and Gaussian-location NPMLE.
///
/// Frank-Wolfe never projects onto the feasible set \f$C\f$. Each step asks a *linear
/// minimisation oracle* (LMO) for \f$s_k=\arg\min_{s\in C}\langle\nabla f(x_k),s\rangle\f$, a
/// vertex of \f$C\f$, and moves towards it: \f$x_{k+1}=x_k+\gamma_k(s_k-x_k)\f$. Iterates are
/// convex combinations of vertices, so they stay feasible and, over the simplex or an
/// \f$\ell_1\f$ ball, sparse. The *Frank-Wolfe gap*
/// \f$g_k=\langle\nabla f(x_k),x_k-s_k\rangle\f$ bounds \f$f(x_k)-f^\star\f$ from above at no
/// extra cost.
///
/// The vanilla method converges at rate \f$O(1/k)\f$, and no faster when the optimum lies on a
/// face of a polytope: it zig-zags towards the face. The away-step and pairwise variants
/// (Wolfe 1970; Lacoste-Julien and Jaggi 2015) keep the iterate's active set of vertices and
/// can move weight *away* from a bad vertex, which gives linear convergence for strongly
/// convex objectives over polytopes.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace datamunge::algorithms {

/// @brief Step-size rule for `frank_wolfe`.
enum class FrankWolfeStep {
    Standard,  ///< \f$\gamma_k=2/(k+2)\f$; vanilla variant only.
    ShortStep, ///< \f$\gamma=\min(\gamma_{\max}, \langle-\nabla f,d\rangle/(L\|d\|^2))\f$; needs `lipschitz`.
    LineSearch ///< Exact line search: safeguarded regula falsi on \f$\langle\nabla f(x+\gamma d),d\rangle\f$ over \f$[0,\gamma_{\max}]\f$.
};

/// @brief Which directions `frank_wolfe` may take.
enum class FrankWolfeVariant {
    Vanilla,  ///< Always towards the LMO vertex.
    AwayStep, ///< Towards the LMO vertex, or away from the worst active vertex, whichever descends faster.
    Pairwise  ///< Move weight from the worst active vertex straight to the LMO vertex.
};

/// @brief What one Frank-Wolfe iteration did.
enum class FrankWolfeMove : char {
    Forward  = 'F', ///< Step towards the LMO vertex.
    Away     = 'A', ///< Step away from an active vertex, which stays active.
    Drop     = 'D', ///< Away or pairwise step that removed an active vertex.
    Pairwise = 'P'  ///< Pairwise step that kept both vertices active.
};

/// @brief Options for `frank_wolfe`.
struct FrankWolfeOptions {
    int               max_iter          = 1000;                     ///< Maximum number of steps.
    double            tolerance         = 0.0;                      ///< Stop when the Frank-Wolfe gap is at most this.
    FrankWolfeVariant variant           = FrankWolfeVariant::Vanilla; ///< Directions allowed.
    FrankWolfeStep    step              = FrankWolfeStep::Standard; ///< Step-size rule.
    double            lipschitz         = 0.0;                      ///< Gradient Lipschitz constant (ShortStep).
    int               line_search_iters = 60;                       ///< Maximum slope evaluations per line search.
    bool              record_path       = false;                    ///< Keep every iterate in `path`.
};

/// @brief Result and trace of `frank_wolfe`.
struct FrankWolfeResult {
    std::vector<double>              x;             ///< Final iterate.
    std::vector<double>              objective;     ///< \f$f(x_k)\f$ for \f$k=0,\dots,K\f$ (size iterations+1).
    std::vector<double>              gap;           ///< Frank-Wolfe gap \f$g_k\f$ for \f$k=0,\dots,K\f$.
    std::vector<double>              step_size;     ///< \f$\gamma_k\f$ of each step.
    std::vector<FrankWolfeMove>      moves;         ///< Kind of each step.
    std::vector<int>                 active_atoms;  ///< Active vertices after each step.
    std::vector<std::vector<double>> path;          ///< \f$x_0,\dots,x_K\f$ (if `record_path`).
    std::vector<std::vector<double>> atoms;         ///< Final active vertices.
    std::vector<double>              weights;       ///< Their convex weights (sum 1).
    int                              iterations{};  ///< Steps taken, \f$K\f$.
    bool                             converged{};   ///< Whether the gap reached `tolerance`.
};

/// @brief Helpers for `frank_wolfe`.
namespace detail {

/// @brief Inner product of two equal-length vectors.
inline double fw_dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

/// @brief Index of @p atom in @p atoms, or -1.
inline int fw_find(const std::vector<std::vector<double>>& atoms, const std::vector<double>& atom) {
    for (std::size_t i = 0; i < atoms.size(); ++i)
        if (atoms[i] == atom) return static_cast<int>(i);
    return -1;
}

} // namespace detail

/// @brief Minimise a smooth convex @p f over a compact convex set given by its LMO, starting
///        from a convex combination of known points of the set.
///
/// The iterate is kept as a convex combination of "atoms": the starting atoms and the vertices
/// the LMO returns. Away-step and pairwise moves shift weight between them, so they need no
/// knowledge of the set beyond the oracle, and can drop any starting atom.
/// @param f Objective, `double(const std::vector<double>&)`.
/// @param grad Gradient, `std::vector<double>(const std::vector<double>&)`.
/// @param lmo Linear minimisation oracle: given a gradient @c g, returns a point of the set
///        minimising \f$\langle g,s\rangle\f$ (a vertex, so that equal vertices compare equal).
/// @param atoms Starting points of the set (all the same length).
/// @param weights Their convex weights (nonnegative, summing to 1).
/// @param options Iterations, tolerance, variant, step rule, path recording.
/// @return The final iterate with its objective, gap, step and active-set trace.
/// @throws std::invalid_argument If @p atoms is empty or does not match @p weights, if a
///         Standard step is asked for with away or pairwise moves, or ShortStep without a
///         positive `lipschitz`.
template <class F, class Grad, class Lmo>
FrankWolfeResult frank_wolfe(F f, Grad grad, Lmo lmo, std::vector<std::vector<double>> atoms,
                             std::vector<double> weights, const FrankWolfeOptions& options) {
    if (atoms.empty() || atoms.size() != weights.size())
        throw std::invalid_argument("frank_wolfe: need as many weights as starting atoms");
    if (options.step == FrankWolfeStep::Standard && options.variant != FrankWolfeVariant::Vanilla)
        throw std::invalid_argument("frank_wolfe: away-step and pairwise variants need ShortStep or LineSearch");
    if (options.step == FrankWolfeStep::ShortStep && !(options.lipschitz > 0))
        throw std::invalid_argument("frank_wolfe: ShortStep needs a positive lipschitz constant");

    FrankWolfeResult  res;
    const std::size_t n = atoms.front().size();
    res.x.assign(n, 0.0);
    for (std::size_t a = 0; a < atoms.size(); ++a)
        for (std::size_t i = 0; i < n; ++i) res.x[i] += weights[a] * atoms[a][i];
    res.atoms   = std::move(atoms);
    res.weights = std::move(weights);
    if (options.record_path) res.path.push_back(res.x);

    for (int k = 0;; ++k) {
        res.objective.push_back(f(res.x));
        const std::vector<double> g = grad(res.x);
        const std::vector<double> s = lmo(g);
        std::vector<double>       d_fw(n);
        for (std::size_t i = 0; i < n; ++i) d_fw[i] = s[i] - res.x[i];
        const double gap = -detail::fw_dot(g, d_fw);
        res.gap.push_back(gap);
        if (gap <= options.tolerance) {
            res.converged = true;
            break;
        }
        if (k >= options.max_iter) break;

        // Worst active vertex (largest <g, v>) for away and pairwise moves.
        int away = -1;
        if (options.variant != FrankWolfeVariant::Vanilla) {
            double worst = -std::numeric_limits<double>::infinity();
            for (std::size_t a = 0; a < res.atoms.size(); ++a) {
                const double v = detail::fw_dot(g, res.atoms[a]);
                if (v > worst) worst = v, away = static_cast<int>(a);
            }
        }

        std::vector<double> d = d_fw;
        double              gamma_max = 1.0;
        FrankWolfeMove      move      = FrankWolfeMove::Forward;
        if (options.variant == FrankWolfeVariant::AwayStep) {
            const double alpha = res.weights[away];
            std::vector<double> d_a(n);
            for (std::size_t i = 0; i < n; ++i) d_a[i] = res.x[i] - res.atoms[away][i];
            if (-detail::fw_dot(g, d_a) > gap && alpha < 1.0) {
                d         = std::move(d_a);
                gamma_max = alpha / (1.0 - alpha);
                move      = FrankWolfeMove::Away;
            }
        } else if (options.variant == FrankWolfeVariant::Pairwise) {
            for (std::size_t i = 0; i < n; ++i) d[i] = s[i] - res.atoms[away][i];
            gamma_max = res.weights[away];
            move      = FrankWolfeMove::Pairwise;
        }

        double gamma = 0;
        if (options.step == FrankWolfeStep::Standard) {
            gamma = 2.0 / (k + 2.0);
        } else {
            const double slope = -detail::fw_dot(g, d), dd = detail::fw_dot(d, d);
            if (dd > 0 && slope > 0) {
                if (options.step == FrankWolfeStep::ShortStep) {
                    gamma = std::min(gamma_max, slope / (options.lipschitz * dd));
                } else {
                    // Root of the slope phi'(t) = <grad f(x + t d), d> by the Illinois variant of
                    // regula falsi, kept inside a bracket. Unlike comparing f values, this
                    // resolves t to machine precision, in a handful of gradient calls.
                    const auto slope_at = [&](double t) {
                        std::vector<double> y(n);
                        for (std::size_t i = 0; i < n; ++i) y[i] = res.x[i] + t * d[i];
                        return detail::fw_dot(grad(y), d);
                    };
                    double s_hi = slope_at(gamma_max);
                    if (s_hi <= 0) {
                        gamma = gamma_max; // still descending at the end: full step, a vertex may drop
                    } else {
                        double lo = 0, hi = gamma_max, s_lo = -slope; // phi'(0) = -slope < 0
                        int    side = 0;
                        gamma       = hi;
                        for (int it = 0; it < options.line_search_iters; ++it) {
                            double t = (lo * s_hi - hi * s_lo) / (s_hi - s_lo);
                            if (!(t > lo && t < hi)) t = (lo + hi) / 2;
                            const double st = slope_at(t);
                            const bool   still = t == gamma;
                            gamma              = t;
                            // Stop once the slope is negligible against the starting slope, or t stalls.
                            if (std::abs(st) <= 1e-13 * slope || still ||
                                hi - lo <= 4 * std::numeric_limits<double>::epsilon() * gamma_max)
                                break;
                            if (st < 0) {
                                lo = t, s_lo = st;
                                if (side == -1) s_hi /= 2;
                                side = -1;
                            } else {
                                hi = t, s_hi = st;
                                if (side == 1) s_lo /= 2;
                                side = 1;
                            }
                        }
                    }
                }
            }
        }

        // Move x and the convex weights of the active atoms.
        for (std::size_t i = 0; i < n; ++i) res.x[i] += gamma * d[i];
        const auto add_to = [&](const std::vector<double>& atom, double w) {
            const int j = detail::fw_find(res.atoms, atom);
            if (j >= 0) res.weights[j] += w;
            else res.atoms.push_back(atom), res.weights.push_back(w);
        };
        const auto drop = [&](int j) {
            res.atoms.erase(res.atoms.begin() + j);
            res.weights.erase(res.weights.begin() + j);
        };
        if (move == FrankWolfeMove::Forward) {
            if (gamma >= 1.0) {
                res.atoms   = {s};
                res.weights = {1.0};
            } else {
                for (double& w : res.weights) w *= 1.0 - gamma;
                add_to(s, gamma);
            }
        } else if (move == FrankWolfeMove::Away) {
            for (double& w : res.weights) w *= 1.0 + gamma;
            res.weights[away] -= gamma;
            if (gamma >= gamma_max) {
                drop(away);
                move = FrankWolfeMove::Drop;
            }
        } else {
            res.weights[away] -= gamma;
            add_to(s, gamma);
            if (gamma >= gamma_max) {
                drop(away); // add_to appends, so index away is unchanged
                move = FrankWolfeMove::Drop;
            }
        }
        res.step_size.push_back(gamma);
        res.moves.push_back(move);
        res.active_atoms.push_back(static_cast<int>(res.atoms.size()));
        if (options.record_path) res.path.push_back(res.x);
        ++res.iterations;
    }
    return res;
}

/// @brief Minimise a smooth convex @p f over a compact convex set given by its LMO, from the
///        feasible point @p x0 (a single starting atom).
/// @param f Objective, `double(const std::vector<double>&)`.
/// @param grad Gradient, `std::vector<double>(const std::vector<double>&)`.
/// @param lmo Linear minimisation oracle.
/// @param x0 Feasible starting point.
/// @param options Iterations, tolerance, variant, step rule, path recording.
/// @return The final iterate with its objective, gap, step and active-set trace.
/// @throws std::invalid_argument As for the atoms overload.
template <class F, class Grad, class Lmo>
FrankWolfeResult frank_wolfe(F f, Grad grad, Lmo lmo, std::vector<double> x0, const FrankWolfeOptions& options) {
    return frank_wolfe(f, grad, lmo, std::vector<std::vector<double>>{std::move(x0)}, std::vector<double>{1.0},
                       options);
}

/// @brief Vanilla Frank-Wolfe with the standard step \f$2/(k+2)\f$ for exactly @p iters steps.
/// @param f Objective.
/// @param grad Gradient.
/// @param lmo Linear minimisation oracle.
/// @param x0 Feasible starting point.
/// @param iters Number of steps; `objective` has @p iters + 1 entries.
/// @return The final iterate and its trace.
template <class F, class Grad, class Lmo>
FrankWolfeResult frank_wolfe(F f, Grad grad, Lmo lmo, std::vector<double> x0, int iters) {
    FrankWolfeOptions options;
    options.max_iter  = iters;
    options.tolerance = -std::numeric_limits<double>::infinity();
    return frank_wolfe(f, grad, lmo, std::move(x0), options);
}

/// @brief LMO for the probability simplex \f$\{x\ge0,\sum x_i=1\}\f$: the unit vector on the
///        coordinate with the smallest gradient.
/// @param g Gradient.
/// @return The minimising vertex.
inline std::vector<double> simplex_lmo(const std::vector<double>& g) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < g.size(); ++i)
        if (g[i] < g[best]) best = i;
    std::vector<double> s(g.size(), 0.0);
    if (!g.empty()) s[best] = 1.0;
    return s;
}

/// @brief LMO for the \f$\ell_1\f$ ball \f$\{\|x\|_1\le r\}\f$: \f$-r\,\mathrm{sign}(g_i)e_i\f$ on
///        the coordinate with the largest \f$|g_i|\f$.
/// @param g Gradient.
/// @param radius Ball radius \f$r\f$.
/// @return The minimising vertex.
inline std::vector<double> l1_ball_lmo(const std::vector<double>& g, double radius) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < g.size(); ++i)
        if (std::abs(g[i]) > std::abs(g[best])) best = i;
    std::vector<double> s(g.size(), 0.0);
    if (!g.empty()) s[best] = g[best] > 0 ? -radius : radius;
    return s;
}

/// @brief LMO for the convex hull of a finite list of vertices: the vertex minimising
///        \f$\langle g,v\rangle\f$ (the first on ties).
/// @param g Gradient.
/// @param vertices The polytope's vertices.
/// @return The minimising vertex.
/// @throws std::invalid_argument If @p vertices is empty.
inline std::vector<double> vertex_lmo(const std::vector<double>& g, const std::vector<std::vector<double>>& vertices) {
    if (vertices.empty()) throw std::invalid_argument("vertex_lmo: no vertices");
    std::size_t best = 0;
    double      bv   = detail::fw_dot(g, vertices[0]);
    for (std::size_t i = 1; i < vertices.size(); ++i) {
        const double v = detail::fw_dot(g, vertices[i]);
        if (v < bv) bv = v, best = i;
    }
    return vertices[best];
}

/// @brief Fit of mixing weights on a fixed grid of Gaussian locations.
struct MixingWeightsFit {
    std::vector<double> grid;           ///< Component locations \f$\mu_j\f$.
    std::vector<double> weights;        ///< Mixing weights \f$\pi_j\f$ (sum 1).
    double              log_likelihood{}; ///< Final \f$\sum_i\log\sum_j\pi_j\varphi_\sigma(x_i-\mu_j)\f$.
    std::vector<double> objective;      ///< Mean negative log-likelihood per iteration.
    std::vector<double> gap;            ///< Frank-Wolfe gap \f$\max_j D_j-1\f$ per iteration.
    std::vector<int>    support;        ///< Grid points with positive weight per iteration (EM: above \f$10^{-8}\f$).
    int                 iterations{};   ///< Iterations run.
    bool                converged{};    ///< Whether the gap reached the tolerance.
};

/// @brief Kiefer-Wolfowitz nonparametric MLE of a Gaussian location mixture with known
///        \f$\sigma\f$, on a fixed grid of locations, by Frank-Wolfe over the simplex.
///
/// Minimises \f$f(\pi)=-\frac1n\sum_i\log\sum_j\pi_j\varphi_{ij}\f$ with
/// \f$\varphi_{ij}=\varphi_\sigma(x_i-\mu_j)\f$. The gradient is \f$-D_j\f$ with
/// \f$D_j=\frac1n\sum_i\varphi_{ij}/m_i\f$ and \f$m_i=\sum_k\pi_k\varphi_{ik}\f$, so the
/// Frank-Wolfe gap is \f$\max_jD_j-1\f$: zero exactly at the optimum (Lindsay 1983). Starts
/// from equal weights on every grid point, each its own atom, so `support` is the number of
/// grid points with positive weight: vanilla steps never remove one, away and pairwise steps
/// can. The variant and step come from @p options.
/// @param data Observations \f$x_i\f$.
/// @param grid Locations \f$\mu_j\f$.
/// @param sigma Known component standard deviation.
/// @param options Frank-Wolfe options.
/// @return The weights and the per-iteration trace.
/// @throws std::invalid_argument If data or grid is empty or sigma is not positive.
MixingWeightsFit gaussian_location_npmle(const std::vector<double>& data, const std::vector<double>& grid,
                                         double sigma, const FrankWolfeOptions& options);

/// @brief The same mixing-weights problem solved by EM, \f$\pi_j\leftarrow\pi_jD_j\f$, from the
///        uniform weights, for comparison with `gaussian_location_npmle`.
///
/// EM increases the likelihood at every step and keeps every weight positive, so it never
/// reaches a sparse solution in finite time.
/// @param data Observations.
/// @param grid Locations.
/// @param sigma Known component standard deviation.
/// @param max_iter Maximum iterations.
/// @param tolerance Stop when the gap \f$\max_jD_j-1\f$ is at most this.
/// @return The weights and trace (`support` counts weights above \f$10^{-8}\f$).
/// @throws std::invalid_argument If data or grid is empty or sigma is not positive.
MixingWeightsFit gaussian_location_em(const std::vector<double>& data, const std::vector<double>& grid,
                                      double sigma, int max_iter, double tolerance);

} // namespace datamunge::algorithms
