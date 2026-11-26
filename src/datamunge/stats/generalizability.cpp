/// @file generalizability.cpp
/// @brief ANOVA (expected mean square) variance components for crossed designs, and D-studies.
#include <datamunge/stats/generalizability.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Implementation helpers for generalizability theory.
namespace {
/// @brief Factor subsets are bit masks over at most 31 factors.
using Mask = std::uint32_t;

/// @brief Marginal means over the factors in a mask (others averaged), indexed row-major over those factors.
/// @param x Scores (row-major over all factors).
/// @param levels Factor levels.
/// @param mask Factors kept.
/// @return Marginal means.
std::vector<double> marginal_means(const std::vector<double>& x, const std::vector<std::size_t>& levels, Mask mask) {
    const std::size_t m = levels.size();
    std::size_t size = 1;
    for (std::size_t f = 0; f < m; ++f)
        if (mask >> f & 1U)
            size *= levels[f];
    std::vector<double> sum(size, 0.0);
    std::vector<std::size_t> idx(m, 0);
    for (std::size_t cell = 0; cell < x.size(); ++cell) {
        std::size_t key = 0;
        for (std::size_t f = 0; f < m; ++f)
            if (mask >> f & 1U)
                key = key * levels[f] + idx[f];
        sum[key] += x[cell];
        for (std::size_t f = m; f-- > 0;) { // advance the row-major counter
            if (++idx[f] < levels[f])
                break;
            idx[f] = 0;
        }
    }
    const double per = static_cast<double>(x.size()) / static_cast<double>(size);
    for (double& s : sum)
        s /= per;
    return sum;
}

/// @brief Sum of squares of the effect of a factor subset by inclusion-exclusion over marginal means.
/// @param x Scores.
/// @param levels Factor levels.
/// @param mask Effect.
/// @return SS.
double sum_of_squares(const std::vector<double>& x, const std::vector<std::size_t>& levels, Mask mask) {
    const std::size_t m = levels.size();
    std::vector<std::size_t> fs;
    for (std::size_t f = 0; f < m; ++f)
        if (mask >> f & 1U)
            fs.push_back(f);
    std::size_t size = 1;
    for (const std::size_t f : fs)
        size *= levels[f];
    std::vector<double> effect(size, 0.0);
    // alpha_S = sum over T subset of S of (-1)^{|S|-|T|} mean_T, each mean_T broadcast over S's cells.
    for (Mask t = mask;; t = (t - 1) & mask) {
        const auto mt = marginal_means(x, levels, t);
        const double sign = (std::popcount(mask) - std::popcount(t)) % 2 == 0 ? 1.0 : -1.0;
        std::vector<std::size_t> idx(fs.size(), 0);
        for (std::size_t c = 0; c < size; ++c) {
            std::size_t key = 0;
            for (std::size_t a = 0; a < fs.size(); ++a)
                if (t >> fs[a] & 1U)
                    key = key * levels[fs[a]] + idx[a];
            effect[c] += sign * mt[key];
            for (std::size_t a = fs.size(); a-- > 0;) {
                if (++idx[a] < levels[fs[a]])
                    break;
                idx[a] = 0;
            }
        }
        if (t == 0)
            break;
    }
    double ss = 0;
    for (const double e : effect)
        ss += e * e;
    return ss * static_cast<double>(x.size()) / static_cast<double>(size);
}
} // namespace

GStudy g_study(const std::vector<double>& scores, const std::vector<std::size_t>& levels) {
    const std::size_t m = levels.size();
    if (m < 2 || m > 16)
        throw std::invalid_argument("g_study: need between 2 and 16 factors");
    std::size_t n = 1;
    for (const std::size_t l : levels) {
        if (l < 2)
            throw std::invalid_argument("g_study: every factor needs at least two levels");
        n *= l;
    }
    if (scores.size() != n || !std::ranges::all_of(scores, [](double v) { return std::isfinite(v); }))
        throw std::invalid_argument("g_study: need one finite score per cell of the crossed design");
    GStudy g;
    g.levels = levels;
    for (const double v : scores)
        g.grand_mean += v / static_cast<double>(n);
    const Mask full = (Mask{1} << m) - 1;
    std::vector<Mask> masks;
    for (Mask s = 1; s <= full; ++s)
        masks.push_back(s);
    std::ranges::stable_sort(masks, [](Mask a, Mask b) {
        return std::popcount(a) != std::popcount(b) ? std::popcount(a) < std::popcount(b) : a < b;
    });
    const auto pos = [&](Mask s) { return static_cast<std::size_t>(std::ranges::find(masks, s) - masks.begin()); };
    const std::size_t e = masks.size();
    std::vector<double> ms(e), df(e), ss(e);
    for (std::size_t i = 0; i < e; ++i) {
        ss[i] = sum_of_squares(scores, levels, masks[i]);
        df[i] = 1;
        for (std::size_t f = 0; f < m; ++f)
            if (masks[i] >> f & 1U)
                df[i] *= static_cast<double>(levels[f] - 1);
        ms[i] = ss[i] / df[i];
    }
    // Coefficients of each raw variance component on the mean squares, from the top down.
    std::vector<std::vector<double>> coef(e, std::vector<double>(e, 0.0));
    const auto not_in = [&](Mask s) {
        double p = 1;
        for (std::size_t f = 0; f < m; ++f)
            if (!(s >> f & 1U))
                p *= static_cast<double>(levels[f]);
        return p;
    };
    for (std::size_t i = e; i-- > 0;) {
        const Mask s = masks[i];
        std::vector<double> c(e, 0.0);
        c[i] = 1;
        for (Mask t = full; t > 0; --t)
            if ((t & s) == s && t != s) {
                const std::size_t j = pos(t);
                for (std::size_t a = 0; a < e; ++a)
                    c[a] -= not_in(t) * coef[j][a];
            }
        for (double& v : c)
            v /= not_in(s);
        coef[i] = c;
    }
    double total = 0;
    for (std::size_t i = 0; i < e; ++i) {
        VarianceComponent vc;
        for (std::size_t f = 0; f < m; ++f)
            if (masks[i] >> f & 1U)
                vc.factors.push_back(f);
        vc.df = df[i];
        vc.sum_of_squares = ss[i];
        vc.mean_square = ms[i];
        double se2 = 0;
        for (std::size_t a = 0; a < e; ++a) {
            vc.raw_variance += coef[i][a] * ms[a];
            se2 += 2 * coef[i][a] * coef[i][a] * ms[a] * ms[a] / (df[a] + 2);
        }
        vc.variance = std::max(0.0, vc.raw_variance);
        vc.standard_error = std::sqrt(se2);
        total += vc.variance;
        g.components.push_back(vc);
    }
    for (auto& vc : g.components)
        vc.percent = total > 0 ? 100 * vc.variance / total : 0.0;
    return g;
}

DStudy d_study(const GStudy& g, const std::vector<double>& facet_sizes) {
    const std::size_t m = g.levels.size();
    if (facet_sizes.size() + 1 != m || !std::ranges::all_of(facet_sizes, [](double v) { return v > 0 && std::isfinite(v); }))
        throw std::invalid_argument("d_study: need one positive size per facet");
    DStudy d;
    d.facet_sizes = facet_sizes;
    for (const auto& vc : g.components) {
        const bool persons = vc.factors.front() == 0;
        double divisor = 1;
        for (const std::size_t f : vc.factors)
            if (f != 0)
                divisor *= facet_sizes[f - 1];
        if (persons && vc.factors.size() == 1)
            d.universe_variance = vc.variance;
        else if (persons)
            d.relative_error += vc.variance / divisor;
        else
            d.absolute_error += vc.variance / divisor;
    }
    d.absolute_error += d.relative_error;
    const double p = d.universe_variance;
    d.generalizability = p + d.relative_error > 0 ? p / (p + d.relative_error) : 0.0;
    d.dependability = p + d.absolute_error > 0 ? p / (p + d.absolute_error) : 0.0;
    d.relative_sem = std::sqrt(d.relative_error);
    d.absolute_sem = std::sqrt(d.absolute_error);
    return d;
}

} // namespace datamunge::stats
