/// @file variates.cpp
/// @brief Public gamma and chi-square variates built on the internal Marsaglia–Tsang sampler.

#include <datamunge/random/variates.hpp>

#include "gamma_sampling.hpp"

#include <cmath>
#include <stdexcept>

namespace datamunge::random {

double gamma_variate(RandomGenerator& rng, double shape, double scale) {
    if (!(shape > 0) || !std::isfinite(shape) || !(scale > 0) || !std::isfinite(scale))
        throw std::invalid_argument("gamma_variate: shape and scale must be positive and finite");
    return scale*static_cast<double>(std::exp(detail::gamma_draw_log(rng,shape,1.0L)));
}

double chi_squared_variate(RandomGenerator& rng, double df) {
    if (!(df > 0) || !std::isfinite(df))
        throw std::invalid_argument("chi_squared_variate: degrees of freedom must be positive and finite");
    return gamma_variate(rng,df/2,2.0);
}

} // namespace datamunge::random
