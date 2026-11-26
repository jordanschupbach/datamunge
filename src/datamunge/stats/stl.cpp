/// @file stl.cpp
/// @brief Port of the STL reference Fortran (as used by R's stats::stl) with zero-based storage.

#include <datamunge/stats/stl.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Loess primitives, moving averages and the inner and outer loops of STL.
namespace {

/// @brief Read-only view of a contiguous series.
struct View {
    const double* data; ///< First element.
    std::size_t size;   ///< Length.
};

/// @brief Local (degree 0 or 1) tricube-weighted fit at position xs from points nleft..nright (1-based).
/// @param y Series.
/// @param len Span.
/// @param degree 0 or 1.
/// @param xs Position (1-based, may lie outside 1..n).
/// @param nleft First point used.
/// @param nright Last point used.
/// @param w Scratch weights, length at least n.
/// @param rw Robustness weights, or null.
/// @param ys Set to the fitted value when the weights are not all zero.
/// @return Whether a fit was possible.
bool loess_point(View y, std::size_t len, int degree, double xs, std::size_t nleft, std::size_t nright,
                 std::vector<double>& w, const double* rw, double& ys) {
    const std::size_t n = y.size;
    const double range = static_cast<double>(n)-1;
    double h = std::max(xs-static_cast<double>(nleft),static_cast<double>(nright)-xs);
    if (len > n)
        h += static_cast<double>((len-n)/2);
    const double h9 = .999*h, h1 = .001*h;
    double a = 0;
    for (std::size_t j = nleft; j <= nright; ++j) {
        w[j-1] = 0;
        const double r = std::abs(static_cast<double>(j)-xs);
        if (r <= h9) {
            if (r <= h1) {
                w[j-1] = 1;
            } else {
                const double q = r/h;
                const double c = 1-q*q*q;
                w[j-1] = c*c*c;
            }
            if (rw)
                w[j-1] *= rw[j-1];
            a += w[j-1];
        }
    }
    if (a <= 0)
        return false;
    for (std::size_t j = nleft; j <= nright; ++j)
        w[j-1] /= a;
    if (h > 0 && degree > 0) {
        a = 0;
        for (std::size_t j = nleft; j <= nright; ++j)
            a += w[j-1]*static_cast<double>(j);
        double b = xs-a, c = 0;
        for (std::size_t j = nleft; j <= nright; ++j) {
            const double d = static_cast<double>(j)-a;
            c += w[j-1]*d*d;
        }
        if (std::sqrt(c) > .001*range) {
            b /= c;
            for (std::size_t j = nleft; j <= nright; ++j)
                w[j-1] *= b*(static_cast<double>(j)-a)+1;
        }
    }
    ys = 0;
    for (std::size_t j = nleft; j <= nright; ++j)
        ys += w[j-1]*y.data[j-1];
    return true;
}

/// @brief Loess smooth of a whole series, evaluated every `jump` points and linearly interpolated.
/// @param y Series.
/// @param len Span.
/// @param degree 0 or 1.
/// @param jump Stride.
/// @param rw Robustness weights, or null.
/// @param ys Output, length n.
/// @param w Scratch, length at least n.
void loess_smooth(View y, std::size_t len, int degree, std::size_t jump, const double* rw, double* ys,
                  std::vector<double>& w) {
    const std::size_t n = y.size;
    if (n < 2) {
        ys[0] = y.data[0];
        return;
    }
    const std::size_t nj = std::min(jump,n-1);
    std::size_t nleft = 1, nright = n;
    const auto fit = [&](std::size_t i) {
        if (!loess_point(y,len,degree,static_cast<double>(i),nleft,nright,w,rw,ys[i-1]))
            ys[i-1] = y.data[i-1];
    };
    if (len >= n) {
        for (std::size_t i = 1; i <= n; i += nj)
            fit(i);
    } else if (nj == 1) {
        const std::size_t nsh = (len+1)/2;
        nright = len;
        for (std::size_t i = 1; i <= n; ++i) {
            if (i > nsh && nright != n) {
                ++nleft;
                ++nright;
            }
            fit(i);
        }
    } else {
        const std::size_t nsh = (len+1)/2;
        for (std::size_t i = 1; i <= n; i += nj) {
            if (i < nsh) {
                nleft = 1;
                nright = len;
            } else if (i >= n-nsh+1) {
                nleft = n-len+1;
                nright = n;
            } else {
                nleft = i-nsh+1;
                nright = len+i-nsh;
            }
            fit(i);
        }
    }
    if (nj != 1) {
        for (std::size_t i = 1; i+nj <= n; i += nj) {
            const double delta = (ys[i+nj-1]-ys[i-1])/static_cast<double>(nj);
            for (std::size_t j = i+1; j < i+nj; ++j)
                ys[j-1] = ys[i-1]+delta*static_cast<double>(j-i);
        }
        const std::size_t k = ((n-1)/nj)*nj+1;
        if (k != n) {
            // Uses the window of the last evaluated point, as the reference implementation does.
            fit(n);
            if (k != n-1) {
                const double delta = (ys[n-1]-ys[k-1])/static_cast<double>(n-k);
                for (std::size_t j = k+1; j < n; ++j)
                    ys[j-1] = ys[k-1]+delta*static_cast<double>(j-k);
            }
        }
    }
}

/// @brief Running mean of length len (n - len + 1 outputs), updated incrementally as the reference.
/// @param x Input.
/// @param n Input length.
/// @param len Window.
/// @param ave Output.
void moving_average(const double* x, std::size_t n, std::size_t len, double* ave) {
    const std::size_t m = n-len+1;
    const double flen = static_cast<double>(len);
    double v = 0;
    for (std::size_t i = 0; i < len; ++i)
        v += x[i];
    ave[0] = v/flen;
    for (std::size_t j = 1; j < m; ++j) {
        v = v-x[j-1]+x[len+j-1];
        ave[j] = v/flen;
    }
}

/// @brief Resolved spans, degrees and strides.
struct Settings {
    std::size_t np; ///< Period.
    std::size_t ns; ///< Seasonal span (odd, at least 3).
    std::size_t nt; ///< Trend span.
    std::size_t nl; ///< Low-pass span.
    int sdeg;       ///< Seasonal degree.
    int tdeg;       ///< Trend degree.
    int ldeg;       ///< Low-pass degree.
    std::size_t sjump; ///< Seasonal stride.
    std::size_t tjump; ///< Trend stride.
    std::size_t ljump; ///< Low-pass stride.
    std::size_t inner;            ///< Inner passes.
};

/// @brief One run of the inner loop.
/// @param y Series.
/// @param s Settings.
/// @param rw Robustness weights, or null.
/// @param season Seasonal component, updated.
/// @param trend Trend component, updated.
void inner_loop(const std::vector<double>& y, const Settings& s, const double* rw, std::vector<double>& season,
                std::vector<double>& trend) {
    const std::size_t n = y.size(), np = s.np, m = n+2*np;
    std::vector<double> detrended(m), cycle(m), lowpass(m), work(m), w(m), sub(m), subw(m), smooth(m);
    for (std::size_t pass = 0; pass < s.inner; ++pass) {
        for (std::size_t i = 0; i < n; ++i)
            detrended[i] = y[i]-trend[i];
        // Cycle-subseries smoothing, extended one cycle at each end.
        for (std::size_t j = 1; j <= np; ++j) {
            const std::size_t k = (n-j)/np+1;
            for (std::size_t i = 0; i < k; ++i) {
                sub[i] = detrended[i*np+j-1];
                if (rw)
                    subw[i] = rw[i*np+j-1];
            }
            const View v{sub.data(),k};
            const double* sw = rw ? subw.data() : nullptr;
            loess_smooth(v,s.ns,s.sdeg,s.sjump,sw,smooth.data()+1,w);
            if (!loess_point(v,s.ns,s.sdeg,0,1,std::min(s.ns,k),w,sw,smooth[0]))
                smooth[0] = smooth[1];
            const std::size_t nleft = k >= s.ns ? k-s.ns+1 : 1;
            if (!loess_point(v,s.ns,s.sdeg,static_cast<double>(k+1),nleft,k,w,sw,smooth[k+1]))
                smooth[k+1] = smooth[k];
            for (std::size_t i = 0; i < k+2; ++i)
                cycle[i*np+j-1] = smooth[i];
        }
        // Low-pass filter of the cycle-subseries: MA(np), MA(np), MA(3), then loess.
        moving_average(cycle.data(),m,np,lowpass.data());
        moving_average(lowpass.data(),m-np+1,np,work.data());
        moving_average(work.data(),m-2*np+2,3,lowpass.data());
        loess_smooth({lowpass.data(),n},s.nl,s.ldeg,s.ljump,nullptr,work.data(),w);
        for (std::size_t i = 0; i < n; ++i) {
            season[i] = cycle[np+i]-work[i];
            detrended[i] = y[i]-season[i];
        }
        loess_smooth({detrended.data(),n},s.nt,s.tdeg,s.tjump,rw,trend.data(),w);
    }
}

/// @brief Bisquare robustness weights from residuals, with scale six times the median absolute residual.
///
/// Uses the exact median (as statsmodels); R's Fortran partial sort can return slightly different
/// middle order statistics, so robust fits can differ from R's after several outer iterations.
/// @param y Series.
/// @param fit Seasonal plus trend.
/// @param rw Output weights.
void robustness_weights(const std::vector<double>& y, const std::vector<double>& fit, std::vector<double>& rw) {
    const std::size_t n = y.size();
    std::vector<double> r(n);
    for (std::size_t i = 0; i < n; ++i)
        r[i] = std::abs(y[i]-fit[i]);
    auto sorted = r;
    const std::size_t m1 = n/2, m2 = n-m1-1;
    std::ranges::nth_element(sorted,sorted.begin()+static_cast<std::ptrdiff_t>(m1));
    const double a = sorted[m1];
    std::ranges::nth_element(sorted,sorted.begin()+static_cast<std::ptrdiff_t>(m2));
    const double cmad = 3*(a+sorted[m2]);
    const double c9 = .999*cmad, c1 = .001*cmad;
    for (std::size_t i = 0; i < n; ++i) {
        if (r[i] <= c1) {
            rw[i] = 1;
        } else if (r[i] <= c9) {
            const double q = r[i]/cmad;
            rw[i] = (1-q*q)*(1-q*q);
        } else {
            rw[i] = 0;
        }
    }
}

/// @brief Next odd integer at or above round(x), as R's nextodd.
/// @param x Value.
/// @return Odd integer.
std::size_t next_odd(double x) {
    auto v = static_cast<std::size_t>(std::llround(x));
    return v%2 ? v : v+1;
}

/// @brief Rounds a span up to an odd value of at least 3.
/// @param v Span.
/// @return Adjusted span.
std::size_t odd_span(std::size_t v) {
    v = std::max<std::size_t>(3,v);
    return v%2 ? v : v+1;
}

} // namespace

StlDecomposition stl_decomposition(const std::vector<double>& y, const StlOptions& options) {
    const std::size_t n = y.size(), period = options.period;
    if (period < 2 || n <= 2*period)
        throw std::invalid_argument("stl_decomposition: need period >= 2 and more than two periods");
    if (!std::ranges::all_of(y,[](double v) { return std::isfinite(v); }))
        throw std::invalid_argument("stl_decomposition: series must be finite");
    const bool periodic = !options.seasonal_window;
    const std::size_t ns = periodic ? 10*n+1 : *options.seasonal_window;
    const int sdeg = periodic ? 0 : options.seasonal_degree;
    const int ldeg = options.lowpass_degree.value_or(options.trend_degree);
    for (const int d : {sdeg,options.trend_degree,ldeg})
        if (d != 0 && d != 1)
            throw std::invalid_argument("stl_decomposition: degrees must be 0 or 1");
    if (ns == 0)
        throw std::invalid_argument("stl_decomposition: seasonal window must be positive");
    const double pd = static_cast<double>(period);
    const std::size_t nt = options.trend_window.value_or(
        next_odd(std::ceil(1.5*pd/(1-1.5/static_cast<double>(ns)))));
    const std::size_t nl = options.lowpass_window.value_or(next_odd(pd));
    const auto ceil10 = [](std::size_t v) { return (v+9)/10; };
    const Settings s{period,odd_span(ns),odd_span(nt),odd_span(nl),sdeg,options.trend_degree,ldeg,
                     options.seasonal_jump.value_or(ceil10(ns)),options.trend_jump.value_or(ceil10(nt)),
                     options.lowpass_jump.value_or(ceil10(nl)),options.inner.value_or(options.robust ? 1 : 2)};
    const std::size_t outer = options.outer.value_or(options.robust ? 15 : 0);
    if (s.sjump == 0 || s.tjump == 0 || s.ljump == 0 || s.inner == 0)
        throw std::invalid_argument("stl_decomposition: strides and inner passes must be positive");

    StlDecomposition d;
    d.seasonal.assign(n,0);
    d.trend.assign(n,0);
    d.weights.assign(n,1);
    std::vector<double> fit(n);
    bool use_weights = false;
    for (std::size_t k = 0;; ++k) {
        inner_loop(y,s,use_weights ? d.weights.data() : nullptr,d.seasonal,d.trend);
        if (k >= outer)
            break;
        for (std::size_t i = 0; i < n; ++i)
            fit[i] = d.trend[i]+d.seasonal[i];
        robustness_weights(y,fit,d.weights);
        use_weights = true;
    }
    if (outer == 0)
        std::ranges::fill(d.weights,1.0);
    if (periodic) {
        std::vector<double> sum(period,0), count(period,0);
        for (std::size_t i = 0; i < n; ++i) {
            sum[i%period] += d.seasonal[i];
            count[i%period] += 1;
        }
        for (std::size_t i = 0; i < n; ++i)
            d.seasonal[i] = sum[i%period]/count[i%period];
    }
    d.remainder.resize(n);
    for (std::size_t i = 0; i < n; ++i)
        d.remainder[i] = y[i]-d.seasonal[i]-d.trend[i];
    d.seasonal_window = ns;
    d.trend_window = nt;
    d.lowpass_window = nl;
    return d;
}

} // namespace datamunge::stats
