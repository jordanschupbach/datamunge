#include <datamunge/algorithms/fft.hpp>
#include <datamunge/stats/spectral_density.hpp>

#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;

#include "stats_spectral_data.inc"

void expect_relative(const std::vector<double>& got, const std::vector<double>& want, double tol) {
    ASSERT_EQ(got.size(),want.size());
    for (std::size_t i = 0; i < want.size(); ++i)
        EXPECT_NEAR(got[i],want[i],tol*std::max(std::abs(want[i]),1e-300)) << i;
}

TEST(Spectral, BluesteinMatchesDirectDft) {
    for (const std::size_t n : {3u, 5u, 12u, 97u, 150u, 1000u}) {
        std::vector<std::complex<double>> a(n);
        for (std::size_t t = 0; t < n; ++t)
            a[t] = {std::sin(.3*double(t*t%17)), std::cos(1.7*double(t))};
        const auto got = datamunge::algorithms::dft_any_length(a);
        for (std::size_t k = 0; k < n; k += std::max<std::size_t>(1,n/13)) {
            std::complex<double> want;
            for (std::size_t t = 0; t < n; ++t)
                want += a[t]*std::polar(1.0,-2*std::numbers::pi*double(k*t%n)/double(n));
            EXPECT_NEAR(std::abs(got[k]-want),0,1e-10*double(n)) << n << " " << k;
        }
        const auto back = datamunge::algorithms::dft_any_length(got,true);
        for (std::size_t t = 0; t < n; ++t)
            EXPECT_NEAR(std::abs(back[t]-a[t]),0,1e-12*double(n));
    }
}

// References: scipy.signal.welch (SciPy 1.17.1).
TEST(Spectral, WelchMatchesScipy) {
    auto s = welch_spectrum(series);
    expect_relative(s.frequencies,welch_default_f,1e-14);
    expect_relative(s.density,welch_default_p,1e-11);
    EXPECT_EQ(s.averaged,6u);
    EXPECT_NEAR(s.degrees_of_freedom[5],welch_default_nu[0],1e-12);
    expect_relative(s.lower,welch_default_lower,1e-8);
    expect_relative(s.upper,welch_default_upper,1e-8);

    s = welch_spectrum(series,{.sampling_rate=10,.segment_length=100,.overlap=25,.fft_length=150,
                               .window=SpectralWindow::hamming,.detrend=SegmentDetrend::linear,
                               .scaling=SpectralScaling::spectrum});
    expect_relative(s.frequencies,welch_hamming_f,1e-14);
    expect_relative(s.density,welch_hamming_p,1e-10);

    s = welch_spectrum(series,{.segment_length=128,.window=SpectralWindow::blackman,.average=SegmentAverage::median});
    expect_relative(s.density,welch_median_p,1e-10);
    EXPECT_TRUE(std::isnan(s.lower[3]));

    s = welch_spectrum(series,{.segment_length=99,.overlap=0,.fft_length=101,.window=SpectralWindow::rectangular,
                               .detrend=SegmentDetrend::none});
    expect_relative(s.frequencies,welch_boxcar_odd_f,1e-14);
    expect_relative(s.density,welch_boxcar_odd_p,1e-10);
    // Non-overlapping rectangular segments: 2 per segment exactly.
    EXPECT_DOUBLE_EQ(s.degrees_of_freedom[7],2.0*double(s.averaged));
}

TEST(Spectral, DpssMatchesScipyAndR) {
    const auto d = dpss_tapers(200,4,7);
    for (std::size_t k = 0; k < 7; ++k) {
        // R's multitaper starts odd tapers negative; SciPy (and dpss_tapers) start them positive.
        double dot = 0;
        for (std::size_t t = 0; t < 200; ++t)
            dot += dpss200_r[k*200+t]*dpss200[k*200+t];
        // R's multitaper fixes taper signs differently from SciPy for some odd tapers.
        const double r_sign = dot < 0 ? -1.0 : 1.0;
        for (std::size_t t = 0; t < 200; ++t) {
            EXPECT_NEAR(d.tapers(t,k),dpss200[k*200+t],1e-10) << k << "," << t;
            EXPECT_NEAR(d.tapers(t,k),r_sign*dpss200_r[k*200+t],1e-9) << k << "," << t;
        }
    }
    expect_relative(d.concentrations,dpss200_ratio,1e-12);
    for (std::size_t k = 0; k < 7; ++k)
        EXPECT_NEAR(d.concentrations[k],dpss200_r_eigen[k],1e-9);
    const auto e = dpss_tapers(1001,2.5,4);
    for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t t = 0; t < 1001; ++t)
            EXPECT_NEAR(e.tapers(t,k),dpss1001[k*1001+t],1e-10) << k << "," << t;
    expect_relative(e.concentrations,dpss1001_ratio,1e-11);
}

// Reference: R multitaper::spec.mtm(ts(x, deltat = .5), nw = 4, k = 7, centre = "arithMean"),
// which reports Delta t |sum v x e|^2 on (0, Nyquist); the one-sided density doubles interior bins.
TEST(Spectral, MultitaperMatchesR) {
    const std::vector<double> x(series.begin(),series.begin()+200);
    const auto s = multitaper_spectrum(x,{.sampling_rate=2,.time_bandwidth=4,.tapers=7,.adaptive=false});
    expect_relative(s.frequencies,mtm_r_freq,1e-14);
    ASSERT_EQ(s.density.size(),mtm_r_spec.size());
    for (std::size_t f = 0; f < s.density.size(); ++f) {
        const bool edge = f == 0 || f+1 == s.density.size();
        EXPECT_NEAR(s.density[f],(edge ? 1 : 2)*mtm_r_spec[f],1e-8*mtm_r_spec[f]+1e-14) << f;
        EXPECT_DOUBLE_EQ(s.degrees_of_freedom[f],edge ? 7.0 : 14.0);
    }
    // R's adaptive weighting differs in form and stops at a 3% tolerance, so agreement is rough.
    const auto a = multitaper_spectrum(x,{.sampling_rate=2,.time_bandwidth=4,.tapers=7});
    for (std::size_t f = 1; f+1 < a.density.size(); f += 16)
        EXPECT_NEAR(a.density[f]/(2*mtm_r_adaptive[f]),1,.1) << f;
}

TEST(Spectral, AdaptiveMultitaperMatchesReference) {
    const std::vector<double> x(series.begin(),series.begin()+300);
    const auto s = multitaper_spectrum(x,{.sampling_rate=4,.time_bandwidth=3,.tapers=5,.fft_length=1024,.confidence=.9});
    expect_relative(s.density,adaptive_density,1e-8);
    expect_relative(s.degrees_of_freedom,adaptive_nu,1e-7);
    expect_relative(s.lower,adaptive_lower,1e-7);
    expect_relative(s.upper,adaptive_upper,1e-7);
}

TEST(Spectral, Validation) {
    EXPECT_THROW((void)welch_spectrum({1}),std::invalid_argument);
    EXPECT_THROW((void)welch_spectrum(series,{.segment_length=64,.overlap=64}),std::invalid_argument);
    EXPECT_THROW((void)welch_spectrum(series,{.segment_length=64,.fft_length=32}),std::invalid_argument);
    EXPECT_THROW((void)welch_spectrum(series,{.sampling_rate=0}),std::invalid_argument);
    EXPECT_THROW((void)welch_spectrum({1,NAN,3}),std::invalid_argument);
    EXPECT_THROW((void)dpss_tapers(10,5,2),std::invalid_argument);
    EXPECT_THROW((void)dpss_tapers(10,2,11),std::invalid_argument);
    EXPECT_THROW((void)multitaper_spectrum({1,2,3}),std::invalid_argument);
    EXPECT_THROW((void)multitaper_spectrum(series,{.confidence=1}),std::invalid_argument);
    EXPECT_THROW((void)multitaper_spectrum(series,{.fft_length=500}),std::invalid_argument);
    // Short series: the segment shrinks to the series.
    EXPECT_EQ(welch_spectrum({1,3,2,5,4,6}).averaged,1u);
}

} // namespace
