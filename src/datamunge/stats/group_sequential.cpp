/// @file group_sequential.cpp
/// @brief Spending functions and the Armitage–McPherson–Rowe recursion for one-sided group-sequential designs.

#include <datamunge/stats/group_sequential.hpp>

#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Normal helpers and the stagewise density recursion.
namespace {

/// @brief Simpson grid intervals per stage (even).
constexpr std::size_t intervals = 2000;

/// @brief Standard normal density.
/// @param x Argument.
/// @return \f$\phi(x)\f$.
double phi(double x) {
    return std::exp(-.5*x*x)/std::sqrt(2*std::numbers::pi);
}

/// @brief Upper normal tail.
/// @param x Argument.
/// @return \f$1-\Phi(x)\f$.
double upper(double x) {
    return .5*std::erfc(x/std::numbers::sqrt2);
}

/// @brief Validates information fractions.
/// @param t Fractions.
/// @throws std::invalid_argument When empty, not strictly increasing, or outside (0, 1].
void check_fractions(const std::vector<double>& t) {
    if (t.empty())
        throw std::invalid_argument("group sequential: need at least one look");
    for (std::size_t k = 0; k < t.size(); ++k)
        if (!(t[k] > 0 && t[k] <= 1) || (k > 0 && !(t[k] > t[k-1])))
            throw std::invalid_argument("group sequential: fractions must increase strictly within (0, 1]");
}

/// @brief Sub-density of the score \f$S_k=Z_k\sqrt{t_k}\f$ on the continuation region, on a Simpson grid.
struct Stage {
    std::vector<double> s;       ///< Grid points.
    std::vector<double> weight;  ///< Simpson weight times density at each point.
};

/// @brief Probability of first crossing at the next look and the continuation density there.
/// @param previous Weighted density at the previous look (empty for the first look).
/// @param t_prev Previous fraction (0 at the first look).
/// @param t Current fraction.
/// @param bound Upper boundary on the Z scale (may be infinite).
/// @param drift Drift.
/// @param next When not null, set to the continuation density at this look.
/// @return \f$P(\text{continue to here and } Z\ge b)\f$.
double advance(const Stage* previous, double t_prev, double t, double bound, double drift, Stage* next) {
    const double dt = t-t_prev, sd = std::sqrt(dt), mean = drift*dt;
    const double c = bound*std::sqrt(t);
    double cross = 0;
    if (!previous) {
        cross = std::isinf(c) ? 0.0 : upper((c-mean)/sd);
    } else {
        for (std::size_t i = 0; i < previous->s.size(); ++i)
            if (!std::isinf(c))
                cross += previous->weight[i]*upper((c-previous->s[i]-mean)/sd);
    }
    if (next) {
        const double centre = drift*t, spread = std::sqrt(t);
        const double lo = centre-9*spread;
        const double hi = std::isinf(c) ? centre+9*spread : std::max(c,lo+1e-9*spread);
        const double h = (hi-lo)/static_cast<double>(intervals);
        next->s.resize(intervals+1);
        next->weight.resize(intervals+1);
        for (std::size_t j = 0; j <= intervals; ++j) {
            const double x = lo+h*static_cast<double>(j);
            double density = 0;
            if (!previous) {
                density = phi((x-mean)/sd)/sd;
            } else {
                for (std::size_t i = 0; i < previous->s.size(); ++i)
                    density += previous->weight[i]*phi((x-previous->s[i]-mean)/sd)/sd;
            }
            const double simpson = (j == 0 || j == intervals) ? 1.0 : (j%2 ? 4.0 : 2.0);
            next->s[j] = x;
            next->weight[j] = density*simpson*h/3;
        }
        if (c <= lo)
            std::ranges::fill(next->weight,0.0);
    }
    return cross;
}

} // namespace

double alpha_spent(double t, const AlphaSpending& spending) {
    if (!(t >= 0) || !std::isfinite(t))
        throw std::invalid_argument("alpha_spent: t must be finite and nonnegative");
    const double a = spending.alpha;
    if (!(a > 0 && a < .5))
        throw std::invalid_argument("alpha_spent: alpha must be in (0, 0.5)");
    t = std::min(t,1.0);
    if (t == 0)
        return 0;
    switch (spending.function) {
    case SpendingFunction::obrien_fleming:
        return 2*upper(random::normal_quantile(1-a/2)/std::sqrt(t));
    case SpendingFunction::pocock:
        return a*std::log1p((std::numbers::e-1)*t);
    case SpendingFunction::power:
        if (!(spending.parameter > 0) || !std::isfinite(spending.parameter))
            throw std::invalid_argument("alpha_spent: power family needs rho > 0");
        return a*std::pow(t,spending.parameter);
    case SpendingFunction::hwang_shih_decani: {
        const double g = spending.parameter;
        if (!std::isfinite(g))
            throw std::invalid_argument("alpha_spent: gamma must be finite");
        if (std::abs(g) < 1e-12)
            return a*t;
        return a*std::expm1(-g*t)/std::expm1(-g);
    }
    }
    return 0;
}

GroupSequentialBoundary group_sequential_boundaries(const std::vector<double>& fractions,
                                                    const AlphaSpending& spending) {
    check_fractions(fractions);
    GroupSequentialBoundary b;
    b.fractions = fractions;
    Stage current, next;
    double spent = 0;
    for (std::size_t k = 0; k < fractions.size(); ++k) {
        const double cumulative = alpha_spent(fractions[k],spending);
        const double target = cumulative-spent;
        const double t_prev = k ? fractions[k-1] : 0.0;
        double bound;
        if (k == 0) {
            bound = random::normal_quantile(1-target);
        } else if (!(target > 0)) {
            bound = std::numeric_limits<double>::infinity();
        } else {
            // Crossing probability decreases in the boundary: bracket and use Illinois false position.
            double lo = -2, hi = 12;
            double flo = advance(&current,t_prev,fractions[k],lo,0,nullptr)-target;
            double fhi = advance(&current,t_prev,fractions[k],hi,0,nullptr)-target;
            int side = 0;
            bound = hi;
            for (int it = 0; it < 200 && hi-lo > 1e-12; ++it) {
                bound = (lo*fhi-hi*flo)/(fhi-flo);
                const double f = advance(&current,t_prev,fractions[k],bound,0,nullptr)-target;
                if (f > 0) {
                    lo = bound;
                    flo = f;
                    if (side == 1)
                        fhi /= 2;
                    side = 1;
                } else {
                    hi = bound;
                    fhi = f;
                    if (side == -1)
                        flo /= 2;
                    side = -1;
                }
                if (std::abs(f) < 1e-15)
                    break;
            }
        }
        (void)advance(k ? &current : nullptr,t_prev,fractions[k],bound,0,&next);
        std::swap(current,next);
        spent = cumulative;
        b.z.push_back(bound);
        b.nominal_alpha.push_back(std::isinf(bound) ? 0.0 : upper(bound));
        b.cumulative_alpha.push_back(cumulative);
    }
    return b;
}

CrossingProbabilities boundary_crossing_probabilities(const std::vector<double>& fractions,
                                                      const std::vector<double>& z, double drift) {
    check_fractions(fractions);
    if (z.size() != fractions.size())
        throw std::invalid_argument("boundary_crossing_probabilities: need one boundary per look");
    if (std::ranges::any_of(z,[](double v) { return std::isnan(v); }) || !std::isfinite(drift))
        throw std::invalid_argument("boundary_crossing_probabilities: NaN boundary or nonfinite drift");
    CrossingProbabilities p;
    Stage current, next;
    double running = 0;
    for (std::size_t k = 0; k < fractions.size(); ++k) {
        const double t_prev = k ? fractions[k-1] : 0.0;
        const bool last = k+1 == fractions.size();
        const double cross = advance(k ? &current : nullptr,t_prev,fractions[k],z[k],drift,last ? nullptr : &next);
        if (!last)
            std::swap(current,next);
        running += cross;
        p.stage.push_back(cross);
        p.cumulative.push_back(running);
        p.expected_fraction += fractions[k]*cross;
    }
    p.total = running;
    p.expected_fraction += fractions.back()*(1-running);
    return p;
}

double drift_for_power(const std::vector<double>& fractions, const std::vector<double>& z, double power) {
    const double size = boundary_crossing_probabilities(fractions,z,0).total;
    if (!(power > size && power < 1))
        throw std::invalid_argument("drift_for_power: power must lie between the size and 1");
    double lo = 0, hi = 1;
    while (boundary_crossing_probabilities(fractions,z,hi).total < power) {
        lo = hi;
        hi *= 2;
        if (hi > 1e3)
            throw std::invalid_argument("drift_for_power: power not reachable");
    }
    // Illinois false position on power(drift) - target, which increases in the drift.
    double flo = boundary_crossing_probabilities(fractions,z,lo).total-power;
    double fhi = boundary_crossing_probabilities(fractions,z,hi).total-power;
    int side = 0;
    for (int it = 0; it < 100 && hi-lo > 1e-11; ++it) {
        const double mid = (lo*fhi-hi*flo)/(fhi-flo);
        const double f = boundary_crossing_probabilities(fractions,z,mid).total-power;
        if (std::abs(f) < 1e-14)
            return mid;
        if (f < 0) {
            lo = mid;
            flo = f;
            if (side == 1)
                fhi /= 2;
            side = 1;
        } else {
            hi = mid;
            fhi = f;
            if (side == -1)
                flo /= 2;
            side = -1;
        }
    }
    return .5*(lo+hi);
}

} // namespace datamunge::stats
