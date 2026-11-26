/// @file diebold_mariano.cpp
/// @brief Autocovariance-based long-run variance and the corrected t reference.

#include <datamunge/stats/diebold_mariano.hpp>

#include <datamunge/random/distributions.hpp>

#include <cmath>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Tail probabilities of Student's t.
namespace {

/// @brief Upper tail \f$P(T>x)\f$ of \f$t_{\nu}\f$ without cancellation.
/// @param x Value.
/// @param df Degrees of freedom.
/// @return Tail probability.
double upper_t(double x, double df) {
    const double half = .5*random::detail::regularized_incomplete_beta(.5*df,.5,df/(df+x*x));
    return x >= 0 ? half : 1-half;
}

} // namespace

DieboldMarianoTest diebold_mariano_test_differentials(const std::vector<double>& d, const DieboldMarianoOptions& options) {
    const std::size_t n = d.size();
    if (n < 3)
        throw std::invalid_argument("diebold_mariano_test: need at least three forecast errors");
    for (const double v : d)
        if (!std::isfinite(v))
            throw std::invalid_argument("diebold_mariano_test: values must be finite");
    std::size_t h = options.horizon;
    if (h < 1 || h > n)
        throw std::invalid_argument("diebold_mariano_test: horizon must be between 1 and the number of errors");
    double mean = 0;
    for (const double v : d)
        mean += v;
    mean /= static_cast<double>(n);
    const auto gamma = [&](std::size_t lag) {
        double s = 0;
        for (std::size_t t = lag; t < n; ++t)
            s += (d[t]-mean)*(d[t-lag]-mean);
        return s/static_cast<double>(n);
    };
    const auto long_run = [&](std::size_t horizon) {
        double v = gamma(0);
        for (std::size_t j = 1; j < horizon; ++j) {
            const double w = options.variance == DmVariance::bartlett
                                 ? 1-static_cast<double>(j)/static_cast<double>(horizon) : 1.0;
            v += 2*w*gamma(j);
        }
        return v/static_cast<double>(n);
    };
    double v = long_run(h);
    if (!(v > 0)) {
        if (h == 1)
            throw std::invalid_argument("diebold_mariano_test: variance of the loss differential is zero");
        h = 1;
        v = long_run(1);
        if (!(v > 0))
            throw std::invalid_argument("diebold_mariano_test: variance of the loss differential is zero");
    }
    const double nn = static_cast<double>(n), hh = static_cast<double>(h);
    DieboldMarianoTest r;
    r.observations = n;
    r.horizon_used = h;
    r.mean_differential = mean;
    r.variance = v;
    r.correction = std::sqrt((nn+1-2*hh+(hh/nn)*(hh-1))/nn);
    r.statistic = r.correction*mean/std::sqrt(v);
    const double df = nn-1;
    switch (options.alternative) {
    case Alternative::TwoSided: r.p_value = 2*upper_t(std::abs(r.statistic),df); break;
    case Alternative::Less: r.p_value = upper_t(-r.statistic,df); break;
    case Alternative::Greater: r.p_value = upper_t(r.statistic,df); break;
    }
    return r;
}

DieboldMarianoTest diebold_mariano_test(const std::vector<double>& e1, const std::vector<double>& e2,
                                        const DieboldMarianoOptions& options) {
    if (e1.size() != e2.size())
        throw std::invalid_argument("diebold_mariano_test: error series must have equal length");
    if (!(options.power > 0) || !std::isfinite(options.power))
        throw std::invalid_argument("diebold_mariano_test: loss power must be positive");
    std::vector<double> d(e1.size());
    for (std::size_t t = 0; t < d.size(); ++t)
        d[t] = std::pow(std::abs(e1[t]),options.power)-std::pow(std::abs(e2[t]),options.power);
    return diebold_mariano_test_differentials(d,options);
}

} // namespace datamunge::stats
