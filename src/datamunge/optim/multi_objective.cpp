#include <datamunge/optim/multi_objective.hpp>

#include <cmath>

namespace datamunge::optim {

bool dominates(const std::vector<double>& a, const std::vector<double>& b) {
    // Minimization: `a` dominates `b` iff a is <= b in every objective and < b in at least one.
    bool strictly_better_somewhere = false;
    const std::size_t m = a.size();
    for (std::size_t k = 0; k < m; ++k) {
        if (a[k] > b[k]) return false;
        if (a[k] < b[k]) strictly_better_somewhere = true;
    }
    return strictly_better_somewhere;
}

std::vector<ParetoPoint> non_dominated_front(const std::vector<ParetoPoint>& points) {
    std::vector<ParetoPoint> front;
    for (std::size_t i = 0; i < points.size(); ++i) {
        bool dominated = false;
        for (std::size_t j = 0; j < points.size(); ++j) {
            if (i != j && dominates(points[j].objectives, points[i].objectives)) {
                dominated = true;
                break;
            }
        }
        if (!dominated) front.push_back(points[i]);
    }
    return front;
}

namespace detail {

namespace {
constexpr double kEps = 1e-14;
double clamp(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }
} // namespace

void sbx_crossover(const std::vector<double>& p1, const std::vector<double>& p2, std::vector<double>& c1,
                   std::vector<double>& c2, const std::vector<double>& lower, const std::vector<double>& upper,
                   double crossover_rate, double eta_c, std::mt19937_64& rng) {
    const std::size_t n = p1.size();
    c1 = p1;
    c2 = p2;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    if (u(rng) > crossover_rate) return; // parents pass through unchanged

    const double exp1 = 1.0 / (eta_c + 1.0);
    for (std::size_t j = 0; j < n; ++j) {
        if (u(rng) > 0.5) continue;                    // per-variable crossover probability
        if (std::fabs(p1[j] - p2[j]) <= kEps) continue; // identical genes: nothing to spread
        const double lo = lower[j], hi = upper[j];
        const double y1 = std::min(p1[j], p2[j]);
        const double y2 = std::max(p1[j], p2[j]);
        const double rand = u(rng);

        // Child 1 (spread toward the lower bound).
        double beta = 1.0 + 2.0 * (y1 - lo) / (y2 - y1);
        double alpha = 2.0 - std::pow(beta, -(eta_c + 1.0));
        double betaq = (rand <= 1.0 / alpha) ? std::pow(rand * alpha, exp1)
                                             : std::pow(1.0 / (2.0 - rand * alpha), exp1);
        double child1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1));

        // Child 2 (spread toward the upper bound).
        beta = 1.0 + 2.0 * (hi - y2) / (y2 - y1);
        alpha = 2.0 - std::pow(beta, -(eta_c + 1.0));
        betaq = (rand <= 1.0 / alpha) ? std::pow(rand * alpha, exp1)
                                      : std::pow(1.0 / (2.0 - rand * alpha), exp1);
        double child2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1));

        child1 = clamp(child1, lo, hi);
        child2 = clamp(child2, lo, hi);
        if (u(rng) <= 0.5) std::swap(child1, child2); // random parent-to-child assignment
        c1[j] = child1;
        c2[j] = child2;
    }
}

void polynomial_mutation(std::vector<double>& x, const std::vector<double>& lower, const std::vector<double>& upper,
                         double mutation_rate, double eta_m, std::mt19937_64& rng) {
    const std::size_t n = x.size();
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const double mut_pow = 1.0 / (eta_m + 1.0);
    for (std::size_t j = 0; j < n; ++j) {
        if (u(rng) > mutation_rate) continue;
        const double lo = lower[j], hi = upper[j];
        const double span = hi - lo;
        if (span <= kEps) continue;
        const double y = x[j];
        const double delta1 = (y - lo) / span;
        const double delta2 = (hi - y) / span;
        const double rnd = u(rng);
        double deltaq;
        if (rnd < 0.5) {
            const double xy = 1.0 - delta1;
            const double val = 2.0 * rnd + (1.0 - 2.0 * rnd) * std::pow(xy, eta_m + 1.0);
            deltaq = std::pow(val, mut_pow) - 1.0;
        } else {
            const double xy = 1.0 - delta2;
            const double val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * std::pow(xy, eta_m + 1.0);
            deltaq = 1.0 - std::pow(val, mut_pow);
        }
        x[j] = clamp(y + deltaq * span, lo, hi);
    }
}

} // namespace detail

} // namespace datamunge::optim
