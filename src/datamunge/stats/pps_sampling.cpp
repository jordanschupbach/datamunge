/// @file pps_sampling.cpp
/// @brief Systematic, Sampford, Brewer and pivotal unequal-probability samplers and exact joint inclusion probabilities.

#include <datamunge/stats/pps_sampling.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Internal helpers for unequal-probability sampling.
namespace {

/// @brief Tolerance for treating a probability as 0 or 1.
constexpr double edge = 1e-12;

/// @brief Validates inclusion probabilities and returns the sample size they imply.
/// @param pi Inclusion probabilities.
/// @param who Function name for messages.
/// @return The integer sample size.
std::size_t sample_size(const std::vector<double>& pi, const char* who) {
    if (pi.empty())
        throw std::invalid_argument(std::string(who)+": need at least one unit");
    double total = 0;
    for (const double p : pi) {
        if (!std::isfinite(p) || p < 0 || p > 1+edge)
            throw std::invalid_argument(std::string(who)+": inclusion probabilities must lie in [0, 1]");
        total += p;
    }
    const double n = std::round(total);
    if (std::abs(total-n) > 1e-6)
        throw std::invalid_argument(std::string(who)+": inclusion probabilities must sum to an integer");
    return static_cast<std::size_t>(n);
}

/// @brief Systematic selection along a given order of units.
/// @param pi Inclusion probabilities.
/// @param order Unit indices in list order.
/// @param u Random start in [0, 1).
/// @return Selected indices (unsorted).
std::vector<std::size_t> systematic(const std::vector<double>& pi, const std::vector<std::size_t>& order, double u) {
    std::vector<std::size_t> s;
    double cumulative = 0;
    for (const std::size_t k : order) {
        const double before = cumulative;
        cumulative += pi[k];
        if (pi[k] >= 1-edge || (pi[k] > edge && std::floor(cumulative-u) > std::floor(before-u)))
            s.push_back(k);
    }
    return s;
}

/// @brief Elementary symmetric polynomials \f$e_0..e_m\f$ of the values, by stable products.
/// @param r Positive values.
/// @param m Highest degree needed.
/// @param skip Index to omit (or r.size() to keep all).
/// @return Coefficients \f$e_0..e_m\f$.
std::vector<double> elementary(const std::vector<double>& r, std::size_t m, std::size_t skip) {
    std::vector<double> e(m+1,0.0);
    e[0] = 1;
    std::size_t used = 0;
    for (std::size_t k = 0; k < r.size(); ++k) {
        if (k == skip)
            continue;
        ++used;
        for (std::size_t s = std::min(m,used); s >= 1; --s)
            e[s] += r[k]*e[s-1];
    }
    return e;
}

} // namespace

std::vector<double> inclusion_probabilities(const std::vector<double>& sizes, std::size_t n) {
    std::size_t positive = 0;
    for (const double x : sizes) {
        if (!std::isfinite(x) || x < 0)
            throw std::invalid_argument("inclusion_probabilities: sizes must be finite and nonnegative");
        positive += x > 0;
    }
    if (n == 0 || n > positive)
        throw std::invalid_argument("inclusion_probabilities: need 0 < n <= number of positive sizes");
    std::vector<double> pi(sizes.size(),0.0);
    std::vector<bool> certain(sizes.size(),false);
    for (std::size_t certainties = 0;;) {
        double rest = 0;
        for (std::size_t k = 0; k < sizes.size(); ++k)
            if (!certain[k])
                rest += sizes[k];
        std::size_t found = certainties;
        for (std::size_t k = 0; k < sizes.size(); ++k) {
            if (certain[k])
                continue;
            pi[k] = static_cast<double>(n-certainties)*sizes[k]/rest;
            if (pi[k] >= 1) {
                certain[k] = true;
                pi[k] = 1;
                ++found;
            }
        }
        if (found == certainties)
            break;
        certainties = found;
    }
    return pi;
}

std::vector<std::size_t> pps_sample(const std::vector<double>& inclusion, PpsMethod method, random::RandomGenerator& rng) {
    const std::size_t n = sample_size(inclusion,"pps_sample"), N = inclusion.size();
    std::vector<std::size_t> s;
    switch (method) {
    case PpsMethod::systematic:
    case PpsMethod::randomized_systematic: {
        std::vector<std::size_t> order(N);
        std::iota(order.begin(),order.end(),std::size_t{0});
        if (method == PpsMethod::randomized_systematic)
            for (std::size_t k = N; k > 1; --k)
                std::swap(order[k-1],order[rng.uniform_u64(0,k-1)]);
        s = systematic(inclusion,order,rng.uniform01());
        break;
    }
    case PpsMethod::sampford: {
        std::vector<std::size_t> open;
        for (std::size_t k = 0; k < N; ++k) {
            if (inclusion[k] >= 1-edge)
                s.push_back(k);
            else if (inclusion[k] > edge)
                open.push_back(k);
        }
        const std::size_t m = n-s.size();
        if (m == 0)
            break;
        // First draw proportional to pi, the rest with replacement proportional to pi/(1-pi);
        // accept when all m draws are distinct.
        std::vector<double> first, rest;
        double cf = 0, cr = 0;
        for (const std::size_t k : open) {
            cf += inclusion[k];
            cr += inclusion[k]/(1-inclusion[k]);
            first.push_back(cf);
            rest.push_back(cr);
        }
        const auto draw = [&](const std::vector<double>& cumulative) {
            const double u = rng.uniform01()*cumulative.back();
            return static_cast<std::size_t>(std::upper_bound(cumulative.begin(),cumulative.end()-1,u)-cumulative.begin());
        };
        std::vector<std::size_t> pick;
        std::vector<bool> taken(open.size());
        for (std::size_t attempt = 0;; ++attempt) {
            if (attempt == 1000000)
                throw std::runtime_error("pps_sample: Sampford rejection did not accept a sample in 1e6 attempts");
            pick.assign(1,draw(first));
            std::fill(taken.begin(),taken.end(),false);
            taken[pick[0]] = true;
            bool distinct = true;
            while (distinct && pick.size() < m) {
                const std::size_t j = draw(rest);
                distinct = !taken[j];
                taken[j] = true;
                pick.push_back(j);
            }
            if (distinct)
                break;
        }
        for (const std::size_t j : pick)
            s.push_back(open[j]);
        break;
    }
    case PpsMethod::brewer: {
        std::vector<std::size_t> open;
        double m = 0;
        for (std::size_t k = 0; k < N; ++k) {
            if (inclusion[k] >= 1-edge)
                s.push_back(k);
            else if (inclusion[k] > edge) {
                open.push_back(k);
                m += inclusion[k];
            }
        }
        const std::size_t draws = static_cast<std::size_t>(std::llround(m));
        std::vector<bool> taken(open.size(),false);
        double a = 0;
        std::vector<double> weight(open.size());
        for (std::size_t i = 1; i <= draws; ++i) {
            double total = 0;
            for (std::size_t j = 0; j < open.size(); ++j) {
                const double p = inclusion[open[j]];
                weight[j] = taken[j] ? 0.0 : std::max(0.0,p*(m-a-p)/(m-a-p*static_cast<double>(draws-i+1)));
                total += weight[j];
            }
            double u = rng.uniform01()*total;
            std::size_t chosen = open.size();
            for (std::size_t j = 0; j < open.size(); ++j) {
                if (weight[j] <= 0)
                    continue;
                chosen = j;
                if (u < weight[j])
                    break;
                u -= weight[j];
            }
            taken[chosen] = true;
            a += inclusion[open[chosen]];
            s.push_back(open[chosen]);
        }
        break;
    }
    case PpsMethod::pivotal: {
        // Ordered pivotal method: the open unit duels with the next one until one of them is decided.
        std::vector<double> p(inclusion);
        std::size_t current = N;
        for (std::size_t k = 0; k < N; ++k) {
            if (p[k] <= edge || p[k] >= 1-edge)
                continue;
            if (current == N) {
                current = k;
                continue;
            }
            const double a = p[current], b = p[k], sum = a+b;
            if (sum < 1) {
                if (rng.uniform01() < b/sum) {
                    p[current] = 0;
                    p[k] = sum;
                } else {
                    p[current] = sum;
                    p[k] = 0;
                }
            } else {
                if (rng.uniform01() < (1-b)/(2-sum)) {
                    p[current] = 1;
                    p[k] = sum-1;
                } else {
                    p[current] = sum-1;
                    p[k] = 1;
                }
            }
            // Keep whichever unit is still undecided open.
            if (p[current] <= edge || p[current] >= 1-edge)
                current = (p[k] <= edge || p[k] >= 1-edge) ? N : k;
        }
        if (current != N) // rounding residue: resolve the last open unit
            p[current] = std::round(p[current]);
        for (std::size_t k = 0; k < N; ++k)
            if (p[k] >= 1-edge)
                s.push_back(k);
        break;
    }
    default:
        throw std::invalid_argument("pps_sample: unknown method");
    }
    std::ranges::sort(s);
    return s;
}

linalg::DenseMatrix<double> systematic_joint_inclusion(const std::vector<double>& inclusion) {
    (void)sample_size(inclusion,"systematic_joint_inclusion");
    const std::size_t N = inclusion.size();
    std::vector<double> start(N);
    double cumulative = 0;
    for (std::size_t k = 0; k < N; ++k) {
        start[k] = cumulative-std::floor(cumulative);
        cumulative += inclusion[k];
    }
    // Overlap of the circular arcs [start, start + length) on the unit circle.
    const auto overlap = [](double a, double la, double b, double lb) {
        double total = 0;
        for (int shift = -1; shift <= 1; ++shift)
            total += std::max(0.0,std::min(a+la,b+lb+shift)-std::max(a,b+shift));
        return total;
    };
    linalg::DenseMatrix<double> joint(N,N);
    for (std::size_t k = 0; k < N; ++k) {
        joint(k,k) = inclusion[k];
        for (std::size_t l = 0; l < k; ++l) {
            const double v = std::min({overlap(start[k],std::min(1.0,inclusion[k]),start[l],std::min(1.0,inclusion[l])),
                                       inclusion[k],inclusion[l]});
            joint(k,l) = joint(l,k) = v;
        }
    }
    return joint;
}

linalg::DenseMatrix<double> sampford_joint_inclusion(const std::vector<double>& inclusion) {
    const std::size_t n = sample_size(inclusion,"sampford_joint_inclusion"), N = inclusion.size();
    std::vector<std::size_t> open;
    for (std::size_t k = 0; k < N; ++k)
        if (inclusion[k] > edge && inclusion[k] < 1-edge)
            open.push_back(k);
    std::size_t m = 0;
    {
        double t = 0;
        for (const std::size_t k : open)
            t += inclusion[k];
        m = static_cast<std::size_t>(std::llround(t));
    }
    // Certainty and zero units: pi_kl = pi_k pi_l.
    linalg::DenseMatrix<double> joint(N,N);
    for (std::size_t k = 0; k < N; ++k)
        for (std::size_t l = 0; l < N; ++l)
            joint(k,l) = k == l ? inclusion[k] : inclusion[k]*inclusion[l];
    (void)n;
    if (m < 2)
        return joint; // at most one random draw: pairs among open units are never both selected
    std::vector<double> r, p;
    for (const std::size_t k : open) {
        p.push_back(inclusion[k]);
        r.push_back(inclusion[k]/(1-inclusion[k]));
    }
    const std::size_t M = open.size();
    const auto all = elementary(r,m,M);
    double denominator = 0;
    for (std::size_t s = 0; s < m; ++s)
        denominator += static_cast<double>(m-s)*all[s];
    if (!std::isfinite(denominator) || !(denominator > 0))
        throw std::runtime_error("sampford_joint_inclusion: elementary symmetric polynomials overflowed");
    for (std::size_t i = 0; i < M; ++i) {
        const auto without_i = elementary(r,m,i);
        for (std::size_t j = 0; j < M; ++j) {
            if (j == i || r[j] > r[i] || (r[j] == r[i] && j > i))
                continue; // handle each pair once, deflating the smaller value from e^(-i)
            // e^(-i,-j)_s = e^(-i)_s - r_j e^(-i,-j)_{s-1}
            double previous = 0, sum = 0;
            for (std::size_t s = 0; s + 2 <= m; ++s) {
                const double e = without_i[s]-r[j]*previous;
                sum += (static_cast<double>(m-s)-p[i]-p[j])*e;
                previous = e;
            }
            const double v = r[i]*r[j]*sum/denominator;
            joint(open[i],open[j]) = joint(open[j],open[i]) = v;
        }
    }
    return joint;
}

} // namespace datamunge::stats
