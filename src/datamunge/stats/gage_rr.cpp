/// @file gage_rr.cpp
/// @brief Two-way crossed random-effects ANOVA for gage R&R studies.
#include <datamunge/stats/gage_rr.hpp>

#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace datamunge::stats {

GageStudy gage_rr(const std::vector<double>& values, std::size_t parts, std::size_t operators,
                  std::size_t replicates, const GageOptions& options) {
    const std::size_t p = parts, o = operators, r = replicates;
    if (p < 2 || o < 2 || r < 2)
        throw std::invalid_argument("gage_rr: need at least 2 parts, 2 operators and 2 replicates");
    if (values.size() != p * o * r || !std::ranges::all_of(values, [](double v) { return std::isfinite(v); }))
        throw std::invalid_argument("gage_rr: need parts x operators x replicates finite values");
    if (!(options.study_multiplier > 0) || (!std::isnan(options.tolerance) && !(options.tolerance > 0)))
        throw std::invalid_argument("gage_rr: multiplier and tolerance must be positive");
    const double pd = static_cast<double>(p), od = static_cast<double>(o), rd = static_cast<double>(r);
    std::vector<double> part(p, 0.0), op(o, 0.0), cell(p * o, 0.0);
    double grand = 0;
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j < o; ++j)
            for (std::size_t k = 0; k < r; ++k) {
                const double y = values[(i * o + j) * r + k];
                part[i] += y / (od * rd), op[j] += y / (pd * rd), cell[i * o + j] += y / rd;
                grand += y / (pd * od * rd);
            }
    double ssp = 0, sso = 0, sspo = 0, sse = 0;
    for (std::size_t i = 0; i < p; ++i)
        ssp += od * rd * (part[i] - grand) * (part[i] - grand);
    for (std::size_t j = 0; j < o; ++j)
        sso += pd * rd * (op[j] - grand) * (op[j] - grand);
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j < o; ++j) {
            const double d = cell[i * o + j] - part[i] - op[j] + grand;
            sspo += rd * d * d;
            for (std::size_t k = 0; k < r; ++k) {
                const double e = values[(i * o + j) * r + k] - cell[i * o + j];
                sse += e * e;
            }
        }
    const double dfp = pd - 1, dfo = od - 1, dfpo = dfp * dfo, dfe = pd * od * (rd - 1);
    const double msp = ssp / dfp, mso = sso / dfo, mspo = sspo / dfpo, mse = sse / dfe;
    if (!(msp + mso + mspo + mse > 0))
        throw std::invalid_argument("gage_rr: the measurements do not vary");
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const auto f_p = [nan](double f, double d1, double d2) {
        return std::isfinite(f) ? 1 - random::f_cdf(f, d1, d2) : (f > 0 ? 0.0 : nan);
    };
    GageStudy g;
    g.parts = p, g.operators = o, g.replicates = r;
    const double f_int = mspo / mse;
    const double p_int = mse > 0 ? f_p(f_int, dfpo, dfe) : 0.0;
    g.interaction_pooled = p_int > options.pool_interaction_above;
    double var_e = 0, var_po = 0, var_o = 0, var_p = 0;
    if (!g.interaction_pooled) {
        g.anova = {{"Part", dfp, ssp, msp, msp / mspo, f_p(msp / mspo, dfp, dfpo)},
                   {"Operator", dfo, sso, mso, mso / mspo, f_p(mso / mspo, dfo, dfpo)},
                   {"Part x Operator", dfpo, sspo, mspo, f_int, p_int},
                   {"Repeatability", dfe, sse, mse, nan, nan}};
        var_e = mse;
        var_po = std::max(0.0, (mspo - mse) / rd);
        var_o = std::max(0.0, (mso - mspo) / (pd * rd));
        var_p = std::max(0.0, (msp - mspo) / (od * rd));
    } else {
        const double dfr = dfpo + dfe, msr = (sspo + sse) / dfr;
        g.anova = {{"Part", dfp, ssp, msp, msp / msr, f_p(msp / msr, dfp, dfr)},
                   {"Operator", dfo, sso, mso, mso / msr, f_p(mso / msr, dfo, dfr)},
                   {"Repeatability", dfr, sspo + sse, msr, nan, nan}};
        var_e = msr;
        var_o = std::max(0.0, (mso - msr) / (pd * rd));
        var_p = std::max(0.0, (msp - msr) / (od * rd));
    }
    const double reproducibility = var_o + var_po, gage = var_e + reproducibility, total = gage + var_p;
    const std::vector<std::pair<const char*, double>> parts_list{
        {"Total Gage R&R", gage}, {"Repeatability", var_e},  {"Reproducibility", reproducibility},
        {"Operator", var_o},      {"Operator x Part", var_po}, {"Part-to-Part", var_p},
        {"Total Variation", total}};
    for (const auto& [name, v] : parts_list) {
        GageComponent c;
        c.source = name;
        c.variance = v;
        c.percent_contribution = 100 * v / total;
        c.sd = std::sqrt(v);
        c.study_variation = options.study_multiplier * c.sd;
        c.percent_study_variation = 100 * c.sd / std::sqrt(total);
        c.percent_tolerance = std::isnan(options.tolerance) ? nan : 100 * c.study_variation / options.tolerance;
        g.components.push_back(c);
    }
    g.distinct_categories =
        gage > 0 ? std::max(1.0, std::floor(1.41 * std::sqrt(var_p) / std::sqrt(gage))) : std::numeric_limits<double>::infinity();
    return g;
}

} // namespace datamunge::stats
