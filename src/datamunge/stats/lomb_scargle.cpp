/// @file lomb_scargle.cpp
/// @brief Tau-shifted weighted sums for the floating-mean periodogram, the automatic grid and Baluev-type calibration.

#include <datamunge/stats/lomb_scargle.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Input checks, weights and false-alarm formulas.
namespace {

/// @brief Validates the data and returns normalized weights \f$w_i\propto\sigma_i^{-2}\f$, \f$\sum w_i=1\f$.
/// @param t Times.
/// @param y Values.
/// @param dy Uncertainties or empty.
/// @return Weights.
std::vector<double> weights(const std::vector<double>& t, const std::vector<double>& y, const std::vector<double>& dy) {
    if (t.size() != y.size() || (!dy.empty() && dy.size() != y.size()))
        throw std::invalid_argument("lomb_scargle: t, y and dy must have equal lengths");
    if (t.size() < 3)
        throw std::invalid_argument("lomb_scargle: need at least three observations");
    for (std::size_t i = 0; i < t.size(); ++i)
        if (!std::isfinite(t[i]) || !std::isfinite(y[i]) || (!dy.empty() && !(dy[i] > 0 && std::isfinite(dy[i]))))
            throw std::invalid_argument("lomb_scargle: inputs must be finite with positive uncertainties");
    const auto [lo, hi] = std::ranges::minmax_element(t);
    if (!(*hi > *lo))
        throw std::invalid_argument("lomb_scargle: times must not all be equal");
    std::vector<double> w(t.size(),1.0);
    if (!dy.empty())
        for (std::size_t i = 0; i < w.size(); ++i)
            w[i] = 1/(dy[i]*dy[i]);
    double total = 0;
    for (const double v : w)
        total += v;
    for (double& v : w)
        v /= total;
    return w;
}

/// @brief Weighted mean of y.
/// @param y Values.
/// @param w Normalized weights.
/// @return \f$\sum w_iy_i\f$.
double weighted_mean(const std::vector<double>& y, const std::vector<double>& w) {
    double m = 0;
    for (std::size_t i = 0; i < y.size(); ++i)
        m += w[i]*y[i];
    return m;
}

/// @brief Grid limits \f$(f_{\min},\delta f,N_f)\f$ of the automatic grid.
struct Grid {
    double minimum{}; ///< First frequency.
    double step{};    ///< Spacing.
    std::size_t count{}; ///< Number of frequencies.
};

/// @brief Astropy's autofrequency rule.
/// @param t Times.
/// @param options Grid options.
/// @return Grid.
Grid grid(const std::vector<double>& t, const LombScargleOptions& options) {
    if (t.size() < 2)
        throw std::invalid_argument("lomb_scargle: need at least two times");
    const auto [lo, hi] = std::ranges::minmax_element(t);
    const double baseline = *hi-*lo;
    if (!(baseline > 0) || !std::isfinite(baseline))
        throw std::invalid_argument("lomb_scargle: times must be finite and not all equal");
    if (!(options.samples_per_peak > 0) || !(options.nyquist_factor > 0))
        throw std::invalid_argument("lomb_scargle: samples per peak and Nyquist factor must be positive");
    Grid g;
    g.step = 1/baseline/options.samples_per_peak;
    g.minimum = options.minimum_frequency.value_or(.5*g.step);
    const double maximum = options.maximum_frequency.value_or(options.nyquist_factor*.5*static_cast<double>(t.size())/baseline);
    if (!(g.minimum > 0) || !(maximum >= g.minimum))
        throw std::invalid_argument("lomb_scargle: need 0 < minimum frequency <= maximum frequency");
    g.count = 1+static_cast<std::size_t>(std::llround((maximum-g.minimum)/g.step));
    return g;
}

/// @brief Single-frequency false-alarm probability with \f$N_K=N-3\f$.
/// @param z Power.
/// @param n Observations.
/// @param norm Normalization.
/// @return Tail probability.
double fap_single(double z, std::size_t n, LombScargleNormalization norm) {
    const double nk = static_cast<double>(n)-3;
    switch (norm) {
    case LombScargleNormalization::standard: return z >= 1 ? 0.0 : std::pow(1-z,.5*nk);
    case LombScargleNormalization::model: return std::pow(1+z,-.5*nk);
    case LombScargleNormalization::log: return std::exp(-.5*nk*z);
    case LombScargleNormalization::psd: return std::exp(-z);
    }
    return 1;
}

/// @brief \f$\sqrt{2/N}\,\Gamma(N/2)/\Gamma((N-1)/2)\f$.
/// @param n Degrees of freedom.
/// @return Value.
double gamma_ratio(double n) {
    return std::sqrt(2/n)*std::exp(std::lgamma(n/2)-std::lgamma((n-1)/2));
}

/// @brief Davies' expected number of upcrossings \f$\tau\f$.
/// @param z Power.
/// @param n Observations.
/// @param w Bandwidth factor \f$f_{\max}T_{\text{eff}}\f$.
/// @param norm Normalization.
/// @return \f$\tau\f$.
double tau_davies(double z, std::size_t n, double w, LombScargleNormalization norm) {
    const double nh = static_cast<double>(n)-1, nk = static_cast<double>(n)-3;
    switch (norm) {
    case LombScargleNormalization::standard:
        return z >= 1 ? 0.0 : gamma_ratio(nh)*w*std::pow(1-z,.5*(nk-1))*std::sqrt(.5*nh*z);
    case LombScargleNormalization::model:
        return gamma_ratio(nk)*w*std::pow(1+z,-.5*nk)*std::sqrt(.5*nk*z);
    case LombScargleNormalization::log:
        return gamma_ratio(nk)*w*std::exp(-.5*z*(nk-.5))*std::sqrt(nk*std::sinh(.5*z));
    case LombScargleNormalization::psd:
        return w*std::exp(-z)*std::sqrt(z);
    }
    return 0;
}

/// @brief Solves the 2x2 or 3x3 weighted least-squares fit of a sinusoid (and offset).
/// @param t Times.
/// @param y Centered values.
/// @param w Weights.
/// @param frequency Frequency.
/// @param fit_mean Whether to include an offset.
/// @return Coefficients (offset, sine, cosine); offset 0 when not fitted.
std::array<double,3> sinusoid_fit(const std::vector<double>& t, const std::vector<double>& y,
                                  const std::vector<double>& w, double frequency, bool fit_mean) {
    const std::size_t p = fit_mean ? 3 : 2;
    double a[3][3] = {}, b[3] = {};
    for (std::size_t i = 0; i < t.size(); ++i) {
        const double phase = 2*std::numbers::pi*frequency*t[i];
        double x[3];
        std::size_t c = 0;
        if (fit_mean)
            x[c++] = 1;
        x[c++] = std::sin(phase);
        x[c++] = std::cos(phase);
        for (std::size_t r = 0; r < p; ++r) {
            b[r] += w[i]*x[r]*y[i];
            for (std::size_t s = 0; s < p; ++s)
                a[r][s] += w[i]*x[r]*x[s];
        }
    }
    // Gaussian elimination with partial pivoting.
    std::size_t order[3] = {0,1,2};
    for (std::size_t k = 0; k < p; ++k) {
        std::size_t best = k;
        for (std::size_t r = k+1; r < p; ++r)
            if (std::abs(a[order[r]][k]) > std::abs(a[order[best]][k]))
                best = r;
        std::swap(order[k],order[best]);
        const double pivot = a[order[k]][k];
        if (pivot == 0)
            throw std::invalid_argument("lomb_scargle_model: sinusoid is not identifiable at this frequency");
        for (std::size_t r = k+1; r < p; ++r) {
            const double f = a[order[r]][k]/pivot;
            for (std::size_t s = k; s < p; ++s)
                a[order[r]][s] -= f*a[order[k]][s];
            b[order[r]] -= f*b[order[k]];
        }
    }
    double x[3] = {};
    for (std::size_t k = p; k-- > 0;) {
        double v = b[order[k]];
        for (std::size_t s = k+1; s < p; ++s)
            v -= a[order[k]][s]*x[s];
        x[k] = v/a[order[k]][k];
    }
    return fit_mean ? std::array<double,3>{x[0],x[1],x[2]} : std::array<double,3>{0,x[0],x[1]};
}

} // namespace

std::vector<double> lomb_scargle_power(const std::vector<double>& t, const std::vector<double>& y,
                                       const std::vector<double>& dy, const std::vector<double>& frequencies,
                                       const LombScargleOptions& options) {
    const auto w = weights(t,y,dy);
    for (const double f : frequencies)
        if (!(f > 0) || !std::isfinite(f))
            throw std::invalid_argument("lomb_scargle_power: frequencies must be positive and finite");
    const std::size_t n = t.size();
    std::vector<double> yc = y;
    if (options.fit_mean || options.center_data) {
        const double m = weighted_mean(y,w);
        for (double& v : yc)
            v -= m;
    }
    const double ybar = weighted_mean(yc,w);
    double yy = 0;
    for (std::size_t i = 0; i < n; ++i)
        yy += w[i]*yc[i]*yc[i];
    double total_inverse_variance = 0;
    for (std::size_t i = 0; i < n; ++i)
        total_inverse_variance += dy.empty() ? 1.0 : 1/(dy[i]*dy[i]);
    std::vector<double> sn(n), cs(n), out;
    out.reserve(frequencies.size());
    for (const double f : frequencies) {
        const double omega = 2*std::numbers::pi*f;
        double s2 = 0, c2 = 0, s = 0, c = 0;
        for (std::size_t i = 0; i < n; ++i) {
            sn[i] = std::sin(omega*t[i]);
            cs[i] = std::cos(omega*t[i]);
            s2 += 2*w[i]*sn[i]*cs[i];
            c2 += w[i]*(cs[i]*cs[i]-sn[i]*sn[i]);
            s += w[i]*sn[i];
            c += w[i]*cs[i];
        }
        if (options.fit_mean) {
            s2 -= 2*s*c;
            c2 -= c*c-s*s;
        }
        const double theta = .5*std::atan2(s2,c2), ct = std::cos(theta), st = std::sin(theta);
        double yc_tau = 0, ys_tau = 0, cc = 0, ss = 0, c_tau = 0, s_tau = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const double cos_tau = cs[i]*ct+sn[i]*st, sin_tau = sn[i]*ct-cs[i]*st;
            yc_tau += w[i]*yc[i]*cos_tau;
            ys_tau += w[i]*yc[i]*sin_tau;
            cc += w[i]*cos_tau*cos_tau;
            ss += w[i]*sin_tau*sin_tau;
            c_tau += w[i]*cos_tau;
            s_tau += w[i]*sin_tau;
        }
        if (options.fit_mean) {
            yc_tau -= ybar*c_tau;
            ys_tau -= ybar*s_tau;
            cc -= c_tau*c_tau;
            ss -= s_tau*s_tau;
        }
        double p = yc_tau*yc_tau/cc+ys_tau*ys_tau/ss;
        switch (options.normalization) {
        case LombScargleNormalization::standard: p /= yy; break;
        case LombScargleNormalization::model: p /= yy-p; break;
        case LombScargleNormalization::log: p = -std::log(1-p/yy); break;
        case LombScargleNormalization::psd: p *= .5*total_inverse_variance; break;
        }
        out.push_back(p);
    }
    return out;
}

std::vector<double> lomb_scargle_frequencies(const std::vector<double>& t, const LombScargleOptions& options) {
    const auto g = grid(t,options);
    std::vector<double> f(g.count);
    for (std::size_t k = 0; k < g.count; ++k)
        f[k] = g.minimum+g.step*static_cast<double>(k);
    return f;
}

double lomb_scargle_false_alarm_probability(double power, const std::vector<double>& t, const std::vector<double>& y,
                                            const std::vector<double>& dy, const LombScargleOptions& options) {
    if (!options.fit_mean && !options.center_data)
        throw std::invalid_argument("lomb_scargle_false_alarm_probability: data must be centered or the mean fitted");
    const auto w = weights(t,y,dy);
    const std::size_t n = t.size();
    if (n < 4)
        throw std::invalid_argument("lomb_scargle_false_alarm_probability: need at least four observations");
    if (!(power >= 0))
        throw std::invalid_argument("lomb_scargle_false_alarm_probability: power must be nonnegative");
    const double single = fap_single(power,n,options.normalization);
    if (options.method == FalseAlarmMethod::single)
        return single;
    const auto g = grid(t,options);
    const double fmax = g.minimum+g.step*static_cast<double>(g.count-1);
    if (options.method == FalseAlarmMethod::naive) {
        const auto [lo, hi] = std::ranges::minmax_element(t);
        const double effective = fmax*(*hi-*lo);
        return -std::expm1(effective*std::log1p(-single));
    }
    // Weighted variance of the times (population form).
    double mt = 0, mt2 = 0;
    for (std::size_t i = 0; i < n; ++i) {
        mt += w[i]*t[i];
        mt2 += w[i]*t[i]*t[i];
    }
    const double bandwidth = fmax*std::sqrt(4*std::numbers::pi*(mt2-mt*mt));
    const double tau = tau_davies(power,n,bandwidth,options.normalization);
    if (options.method == FalseAlarmMethod::davies)
        return single+tau;
    return -std::expm1(-tau)+single*std::exp(-tau);
}

double lomb_scargle_false_alarm_level(double probability, const std::vector<double>& t, const std::vector<double>& y,
                                      const std::vector<double>& dy, const LombScargleOptions& options) {
    if (!(probability > 0 && probability < 1))
        throw std::invalid_argument("lomb_scargle_false_alarm_level: probability must be in (0, 1)");
    const auto fap = [&](double z) { return lomb_scargle_false_alarm_probability(z,t,y,dy,options); };
    double lo = 0, hi = 1;
    if (options.normalization == LombScargleNormalization::standard) {
        hi = 1;
    } else {
        while (fap(hi) >= probability) {
            lo = hi;
            hi *= 2;
            if (hi > 1e8)
                throw std::invalid_argument("lomb_scargle_false_alarm_level: level not found");
        }
    }
    for (int it = 0; it < 200 && hi-lo > 1e-15*std::max(1.0,hi); ++it) {
        const double mid = .5*(lo+hi);
        (fap(mid) >= probability ? lo : hi) = mid;
    }
    return .5*(lo+hi);
}

std::vector<double> lomb_scargle_model(const std::vector<double>& t, const std::vector<double>& y,
                                       const std::vector<double>& dy, double frequency,
                                       const std::vector<double>& t_fit, const LombScargleOptions& options) {
    const auto w = weights(t,y,dy);
    if (!(frequency > 0) || !std::isfinite(frequency))
        throw std::invalid_argument("lomb_scargle_model: frequency must be positive and finite");
    const double mean = options.center_data ? weighted_mean(y,w) : 0.0;
    std::vector<double> yc = y;
    for (double& v : yc)
        v -= mean;
    const auto c = sinusoid_fit(t,yc,w,frequency,options.fit_mean);
    std::vector<double> out;
    for (const double s : t_fit) {
        const double phase = 2*std::numbers::pi*frequency*s;
        out.push_back(mean+c[0]+c[1]*std::sin(phase)+c[2]*std::cos(phase));
    }
    return out;
}

LombScargleResult lomb_scargle(const std::vector<double>& t, const std::vector<double>& y,
                               const std::vector<double>& dy, const LombScargleOptions& options) {
    LombScargleResult r;
    r.frequencies = lomb_scargle_frequencies(t,options);
    r.power = lomb_scargle_power(t,y,dy,r.frequencies,options);
    const auto best = static_cast<std::size_t>(std::ranges::max_element(r.power)-r.power.begin());
    r.best_frequency = r.frequencies[best];
    r.best_power = r.power[best];
    r.false_alarm_probability = lomb_scargle_false_alarm_probability(r.best_power,t,y,dy,options);
    const auto w = weights(t,y,dy);
    const double mean = options.center_data ? weighted_mean(y,w) : 0.0;
    std::vector<double> yc = y;
    for (double& v : yc)
        v -= mean;
    const auto c = sinusoid_fit(t,yc,w,r.best_frequency,options.fit_mean);
    r.offset = mean+c[0];
    r.amplitude = std::hypot(c[1],c[2]);
    return r;
}

} // namespace datamunge::stats
