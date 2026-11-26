/// @file spectral_density.cpp
/// @brief Segment periodograms, Slepian tapers by bisection and inverse iteration, and adaptive multitaper weights.

#include <datamunge/stats/spectral_density.hpp>

#include <datamunge/algorithms/fft.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace datamunge::stats {
/// @brief Windows, transforms, tridiagonal eigen-solvers and band helpers.
namespace {

/// @brief Complex vector alias.
using Complex = std::vector<std::complex<double>>;

/// @brief Periodic window of length n.
/// @param window Window type.
/// @param n Length.
/// @return Window values.
std::vector<double> make_window(SpectralWindow window, std::size_t n) {
    std::vector<double> w(n,1.0);
    const double step = 2*std::numbers::pi/static_cast<double>(n);
    for (std::size_t k = 0; k < n; ++k) {
        const double c = std::cos(step*static_cast<double>(k));
        switch (window) {
        case SpectralWindow::hann: w[k] = .5-.5*c; break;
        case SpectralWindow::hamming: w[k] = .54-.46*c; break;
        case SpectralWindow::blackman: w[k] = .42-.5*c+.08*std::cos(2*step*static_cast<double>(k)); break;
        case SpectralWindow::rectangular: break;
        }
    }
    return w;
}

/// @brief Squared moduli of the first nfft/2+1 DFT bins of a zero-padded real sequence.
/// @param x Sequence (length at most nfft).
/// @param nfft Transform length.
/// @return \f$|X_k|^2\f$ for \f$k=0..\lfloor n_{\text{fft}}/2\rfloor\f$.
std::vector<double> power(const std::vector<double>& x, std::size_t nfft) {
    Complex a(nfft);
    for (std::size_t t = 0; t < x.size(); ++t)
        a[t] = x[t];
    a = algorithms::dft_any_length(std::move(a));
    std::vector<double> p(nfft/2+1);
    for (std::size_t k = 0; k < p.size(); ++k)
        p[k] = std::norm(a[k]);
    return p;
}

/// @brief Whether bin k is one of the self-conjugate bins (0 or Nyquist).
/// @param k Bin.
/// @param nfft Transform length.
/// @return True at 0 and at nfft/2 for even nfft.
bool self_conjugate(std::size_t k, std::size_t nfft) {
    return k == 0 || (nfft%2 == 0 && 2*k == nfft);
}

/// @brief Fills chi-square bands for every frequency.
/// @param s Estimate with density and degrees of freedom set.
/// @param confidence Coverage.
void bands(SpectralEstimate& s, double confidence) {
    const double a = 1-confidence;
    s.lower.resize(s.density.size());
    s.upper.resize(s.density.size());
    for (std::size_t k = 0; k < s.density.size(); ++k) {
        const double nu = s.degrees_of_freedom[k];
        if (!(nu > 0) || !std::isfinite(nu)) {
            s.lower[k] = s.upper[k] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        s.lower[k] = nu*s.density[k]/random::chi_squared_quantile(1-a/2,nu);
        s.upper[k] = nu*s.density[k]/random::chi_squared_quantile(a/2,nu);
    }
}

/// @brief Checks sampling rate and confidence.
/// @param rate Sampling rate.
/// @param confidence Coverage.
/// @param name Caller name.
void check_common(double rate, double confidence, const char* name) {
    if (!(rate > 0) || !std::isfinite(rate))
        throw std::invalid_argument(std::string(name)+": sampling rate must be positive");
    if (!(confidence > 0 && confidence < 1))
        throw std::invalid_argument(std::string(name)+": confidence must be in (0, 1)");
}

/// @brief Number of eigenvalues of the symmetric tridiagonal matrix below x (Sturm count).
/// @param d Diagonal.
/// @param e Off-diagonal.
/// @param x Shift.
/// @return Count.
std::size_t sturm_count(const std::vector<double>& d, const std::vector<double>& e, double x) {
    std::size_t count = 0;
    double q = d[0]-x;
    const double tiny = std::numeric_limits<double>::min();
    if (q < 0)
        ++count;
    for (std::size_t i = 1; i < d.size(); ++i) {
        if (std::abs(q) < tiny)
            q = -tiny;
        q = d[i]-x-e[i-1]*e[i-1]/q;
        if (q < 0)
            ++count;
    }
    return count;
}

/// @brief Solves \f$(T-\mu I)y=b\f$ for symmetric tridiagonal T by LU with partial pivoting (LAPACK dgttrf/dgttrs).
/// @param d Diagonal of T.
/// @param e Off-diagonal of T.
/// @param mu Shift.
/// @param b Right-hand side, overwritten with the solution.
void shifted_solve(const std::vector<double>& d, const std::vector<double>& e, double mu, std::vector<double>& b) {
    const std::size_t n = d.size();
    std::vector<double> dl(e), dd(n), du(e), du2(n,0.0);
    std::vector<bool> swapped(n,false);
    for (std::size_t i = 0; i < n; ++i)
        dd[i] = d[i]-mu;
    for (std::size_t i = 0; i+1 < n; ++i) {
        if (std::abs(dd[i]) >= std::abs(dl[i])) {
            if (dd[i] == 0)
                dd[i] = std::numeric_limits<double>::epsilon()*(std::abs(mu)+1);
            const double fact = dl[i]/dd[i];
            dl[i] = fact;
            dd[i+1] -= fact*du[i];
        } else {
            const double fact = dd[i]/dl[i];
            dd[i] = dl[i];
            dl[i] = fact;
            const double temp = du[i];
            du[i] = dd[i+1];
            dd[i+1] = temp-fact*dd[i+1];
            if (i+2 < n) {
                du2[i] = du[i+1];
                du[i+1] = -fact*du[i+1];
            }
            swapped[i] = true;
        }
    }
    if (dd[n-1] == 0)
        dd[n-1] = std::numeric_limits<double>::epsilon()*(std::abs(mu)+1);
    for (std::size_t i = 0; i+1 < n; ++i) {
        if (!swapped[i]) {
            b[i+1] -= dl[i]*b[i];
        } else {
            const double temp = b[i];
            b[i] = b[i+1];
            b[i+1] = temp-dl[i]*b[i];
        }
    }
    b[n-1] /= dd[n-1];
    if (n >= 2)
        b[n-2] = (b[n-2]-du[n-2]*b[n-1])/dd[n-2];
    for (std::size_t i = n-2; i-- > 0;)
        b[i] = (b[i]-du[i]*b[i+1]-du2[i]*b[i+2])/dd[i];
}

} // namespace

SpectralEstimate welch_spectrum(const std::vector<double>& x, const WelchOptions& options) {
    check_common(options.sampling_rate,options.confidence,"welch_spectrum");
    const std::size_t n = x.size();
    if (n < 2)
        throw std::invalid_argument("welch_spectrum: need at least two values");
    if (std::ranges::any_of(x,[](double v) { return !std::isfinite(v); }))
        throw std::invalid_argument("welch_spectrum: values must be finite");
    const std::size_t length = std::min(options.segment_length,n);
    if (length < 2)
        throw std::invalid_argument("welch_spectrum: segment length must be at least 2");
    const std::size_t overlap = options.overlap.value_or(length/2);
    if (overlap >= length)
        throw std::invalid_argument("welch_spectrum: overlap must be smaller than the segment length");
    const std::size_t nfft = options.fft_length ? options.fft_length : length;
    if (nfft < length)
        throw std::invalid_argument("welch_spectrum: FFT length must be at least the segment length");
    const std::size_t step = length-overlap, segments = (n-length)/step+1;
    const auto w = make_window(options.window,length);
    double sum = 0, sumsq = 0;
    for (const double v : w) {
        sum += v;
        sumsq += v*v;
    }
    const double scale = options.scaling == SpectralScaling::density ? 1/(options.sampling_rate*sumsq) : 1/(sum*sum);
    const std::size_t bins = nfft/2+1;
    std::vector<std::vector<double>> periodograms(bins,std::vector<double>(segments));
    std::vector<double> seg(length);
    for (std::size_t s = 0; s < segments; ++s) {
        std::copy_n(x.begin()+static_cast<std::ptrdiff_t>(s*step),length,seg.begin());
        if (options.detrend == SegmentDetrend::constant) {
            double mean = 0;
            for (const double v : seg)
                mean += v;
            mean /= static_cast<double>(length);
            for (double& v : seg)
                v -= mean;
        } else if (options.detrend == SegmentDetrend::linear) {
            const double tbar = (static_cast<double>(length)-1)/2;
            double ybar = 0, sxy = 0, sxx = 0;
            for (const double v : seg)
                ybar += v;
            ybar /= static_cast<double>(length);
            for (std::size_t t = 0; t < length; ++t) {
                const double dt = static_cast<double>(t)-tbar;
                sxy += dt*(seg[t]-ybar);
                sxx += dt*dt;
            }
            const double slope = sxy/sxx;
            for (std::size_t t = 0; t < length; ++t)
                seg[t] -= ybar+slope*(static_cast<double>(t)-tbar);
        }
        for (std::size_t t = 0; t < length; ++t)
            seg[t] *= w[t];
        const auto p = power(seg,nfft);
        for (std::size_t k = 0; k < bins; ++k)
            periodograms[k][s] = p[k]*scale*(self_conjugate(k,nfft) ? 1.0 : 2.0);
    }

    SpectralEstimate out;
    out.averaged = segments;
    double nu = std::numeric_limits<double>::quiet_NaN();
    if (options.average == SegmentAverage::mean) {
        // Percival and Walden (1993), eq. 292b, with the unit-energy window.
        double denominator = 1;
        for (std::size_t m = 1; m < segments && m*step < length; ++m) {
            double rho = 0;
            for (std::size_t t = 0; t+m*step < length; ++t)
                rho += w[t]*w[t+m*step];
            rho /= sumsq;
            denominator += 2*(1-static_cast<double>(m)/static_cast<double>(segments))*rho*rho;
        }
        nu = 2*static_cast<double>(segments)/denominator;
    }
    double bias = 1;
    if (options.average == SegmentAverage::median)
        for (std::size_t i = 1; i <= (segments-1)/2; ++i)
            bias += 1/(2.0*static_cast<double>(i)+1)-1/(2.0*static_cast<double>(i));
    for (std::size_t k = 0; k < bins; ++k) {
        out.frequencies.push_back(static_cast<double>(k)*options.sampling_rate/static_cast<double>(nfft));
        auto& p = periodograms[k];
        double value = 0;
        if (options.average == SegmentAverage::mean) {
            for (const double v : p)
                value += v;
            value /= static_cast<double>(segments);
        } else {
            std::ranges::sort(p);
            value = segments%2 ? p[segments/2] : .5*(p[segments/2-1]+p[segments/2]);
            value /= bias;
        }
        out.density.push_back(value);
        out.degrees_of_freedom.push_back(self_conjugate(k,nfft) ? nu/2 : nu);
    }
    bands(out,options.confidence);
    return out;
}

DpssTapers dpss_tapers(std::size_t n, double time_bandwidth, std::size_t count) {
    if (n < 2)
        throw std::invalid_argument("dpss_tapers: length must be at least 2");
    if (!(time_bandwidth > 0) || !(time_bandwidth < static_cast<double>(n)/2))
        throw std::invalid_argument("dpss_tapers: time-bandwidth must be in (0, n/2)");
    if (count < 1 || count > n)
        throw std::invalid_argument("dpss_tapers: taper count must be between 1 and n");
    const double w = time_bandwidth/static_cast<double>(n);
    std::vector<double> d(n), e(n-1);
    for (std::size_t t = 0; t < n; ++t) {
        const double c = (static_cast<double>(n)-1-2*static_cast<double>(t))/2;
        d[t] = c*c*std::cos(2*std::numbers::pi*w);
    }
    for (std::size_t t = 1; t < n; ++t)
        e[t-1] = static_cast<double>(t)*(static_cast<double>(n)-static_cast<double>(t))/2;
    // Gershgorin interval.
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    for (std::size_t t = 0; t < n; ++t) {
        const double r = (t ? std::abs(e[t-1]) : 0)+(t+1 < n ? std::abs(e[t]) : 0);
        lo = std::min(lo,d[t]-r);
        hi = std::max(hi,d[t]+r);
    }
    DpssTapers out;
    out.tapers = linalg::DenseMatrix<double>(n,count);
    const std::size_t padded = [&] {
        std::size_t m = 1;
        while (m < 2*n)
            m <<= 1;
        return m;
    }();
    for (std::size_t k = 0; k < count; ++k) {
        // The (k+1)-th largest eigenvalue: the smallest x with at least n-k eigenvalues below it.
        double a = lo, b = hi;
        for (int it = 0; it < 200 && b-a > 4*std::numeric_limits<double>::epsilon()*std::max(std::abs(a),std::abs(b)); ++it) {
            const double mid = .5*(a+b);
            (sturm_count(d,e,mid) >= n-k ? b : a) = mid;
        }
        const double mu = .5*(a+b);
        std::vector<double> v(n);
        for (std::size_t t = 0; t < n; ++t)
            v[t] = 1+.01*std::sin(static_cast<double>(t*(k+1)));
        for (int it = 0; it < 4; ++it) {
            shifted_solve(d,e,mu,v);
            double norm = 0;
            for (const double x : v)
                norm += x*x;
            norm = std::sqrt(norm);
            for (double& x : v)
                x /= norm;
        }
        // SciPy's sign convention.
        if (k%2 == 0) {
            double s = 0;
            for (const double x : v)
                s += x;
            if (s < 0)
                for (double& x : v)
                    x = -x;
        } else {
            const double threshold = std::max(1e-7,1/static_cast<double>(n));
            for (const double x : v)
                if (x*x > threshold) {
                    if (x < 0)
                        for (double& y : v)
                            y = -y;
                    break;
                }
        }
        for (std::size_t t = 0; t < n; ++t)
            out.tapers(t,k) = v[t];
        // Concentration from the autocorrelation: sum over lags of r_vv(m) sin(2 pi W m)/(pi m).
        Complex a2(padded);
        for (std::size_t t = 0; t < n; ++t)
            a2[t] = v[t];
        a2 = algorithms::fft(std::move(a2));
        for (auto& z : a2)
            z = std::norm(z);
        a2 = algorithms::fft(std::move(a2),true);
        double lambda = 2*w*a2[0].real();
        for (std::size_t m = 1; m < n; ++m)
            lambda += a2[m].real()*2*std::sin(2*std::numbers::pi*w*static_cast<double>(m))/(std::numbers::pi*static_cast<double>(m));
        out.concentrations.push_back(lambda);
    }
    return out;
}

SpectralEstimate multitaper_spectrum(const std::vector<double>& x, const MultitaperOptions& options) {
    check_common(options.sampling_rate,options.confidence,"multitaper_spectrum");
    const std::size_t n = x.size();
    if (n < 4)
        throw std::invalid_argument("multitaper_spectrum: need at least four values");
    if (std::ranges::any_of(x,[](double v) { return !std::isfinite(v); }))
        throw std::invalid_argument("multitaper_spectrum: values must be finite");
    const double nw = options.time_bandwidth;
    const std::size_t k = options.tapers ? options.tapers
                                         : static_cast<std::size_t>(std::max(1.0,std::floor(2*nw)-1));
    std::size_t nfft = options.fft_length;
    if (nfft == 0) {
        nfft = 2;
        while (nfft < n)
            nfft <<= 1;
        nfft *= 2;
    }
    if (nfft < n)
        throw std::invalid_argument("multitaper_spectrum: FFT length must be at least n");
    const auto dpss = dpss_tapers(n,nw,k);
    std::vector<double> y = x;
    double mean = 0;
    for (const double v : y)
        mean += v;
    mean /= static_cast<double>(n);
    if (options.demean)
        for (double& v : y)
            v -= mean;
    double variance = 0;
    for (const double v : x)
        variance += (v-mean)*(v-mean);
    variance /= static_cast<double>(n);
    const double dt = 1/options.sampling_rate;
    const std::size_t bins = nfft/2+1;
    std::vector<std::vector<double>> eigen(k);
    std::vector<double> tapered(n);
    for (std::size_t j = 0; j < k; ++j) {
        for (std::size_t t = 0; t < n; ++t)
            tapered[t] = dpss.tapers(t,j)*y[t];
        eigen[j] = power(tapered,nfft);
        for (double& v : eigen[j])
            v *= dt;
    }
    SpectralEstimate out;
    out.averaged = k;
    const auto& lambda = dpss.concentrations;
    const double noise = variance*dt;
    for (std::size_t f = 0; f < bins; ++f) {
        out.frequencies.push_back(static_cast<double>(f)*options.sampling_rate/static_cast<double>(nfft));
        double s = 0, nu = 2*static_cast<double>(k);
        if (!options.adaptive || k == 1) {
            for (std::size_t j = 0; j < k; ++j)
                s += eigen[j][f];
            s /= static_cast<double>(k);
        } else {
            s = .5*(eigen[0][f]+eigen[1][f]);
            double num = 0, den = 0, den4 = 0;
            for (std::size_t it = 0; it < options.max_iterations; ++it) {
                num = den = den4 = 0;
                for (std::size_t j = 0; j < k; ++j) {
                    const double b = s/(lambda[j]*s+(1-lambda[j])*noise);
                    const double weight = b*b*lambda[j];
                    num += weight*eigen[j][f];
                    den += weight;
                    den4 += weight*weight;
                }
                const double next = den > 0 ? num/den : 0;
                const bool done = std::abs(next-s) <= 1e-12*std::max(next,std::numeric_limits<double>::min());
                s = next;
                if (done)
                    break;
            }
            // Degrees of freedom at the final weights.
            den = den4 = 0;
            for (std::size_t j = 0; j < k; ++j) {
                const double b = s/(lambda[j]*s+(1-lambda[j])*noise);
                den += b*b*lambda[j];
                den4 += b*b*b*b*lambda[j]*lambda[j];
            }
            nu = den4 > 0 ? 2*den*den/den4 : 2*static_cast<double>(k);
        }
        const bool edge = self_conjugate(f,nfft);
        out.density.push_back(edge ? s : 2*s);
        out.degrees_of_freedom.push_back(edge ? nu/2 : nu);
    }
    bands(out,options.confidence);
    return out;
}

} // namespace datamunge::stats
