#include <gtest/gtest.h>

#include <datamunge/algorithms/fft.hpp>
#include <datamunge/algorithms/goertzel.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

using datamunge::algorithms::fft;
using datamunge::algorithms::fft_real;
using datamunge::algorithms::goertzel_power_bin;

TEST(FFT, MatchesDirectDFT) {
    std::vector<std::complex<double>> x = {{1, 0}, {2, -1}, {0, -1}, {-1, 2}, {3, 0}, {0, 0}, {-2, 1}, {1, 1}};
    auto                              X = fft(x, false);
    const std::size_t                 n = x.size();
    for (std::size_t k = 0; k < n; ++k) {
        std::complex<double> acc(0, 0);
        for (std::size_t t = 0; t < n; ++t) {
            double ang = -2.0 * M_PI * k * t / n;
            acc += x[t] * std::complex<double>(std::cos(ang), std::sin(ang));
        }
        EXPECT_NEAR(X[k].real(), acc.real(), 1e-9);
        EXPECT_NEAR(X[k].imag(), acc.imag(), 1e-9);
    }
}

TEST(FFT, InverseRecoversInput) {
    std::vector<std::complex<double>> x = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    auto                              rt = fft(fft(x, false), true);
    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(rt[i].real(), x[i].real(), 1e-9);
        EXPECT_NEAR(rt[i].imag(), 0.0, 1e-9);
    }
}

TEST(FFT, DetectsAPureTone) {
    // A cosine at bin 3 should put all energy in bins 3 and N-3.
    const std::size_t   n = 64;
    std::vector<double> x(n);
    for (std::size_t t = 0; t < n; ++t) x[t] = std::cos(2.0 * M_PI * 3.0 * t / n);
    auto X = fft_real(x);
    std::size_t argmax = 1;
    double      best   = 0.0;
    for (std::size_t k = 1; k < n / 2; ++k) {
        double mag = std::abs(X[k]);
        if (mag > best) { best = mag; argmax = k; }
    }
    EXPECT_EQ(argmax, 3u);
}

TEST(Goertzel, MatchesFFTBinMagnitude) {
    const std::size_t   n = 64;
    std::vector<double> x(n);
    for (std::size_t t = 0; t < n; ++t)
        x[t] = std::cos(2.0 * M_PI * 5.0 * t / n) + 0.5 * std::sin(2.0 * M_PI * 11.0 * t / n);
    auto X = fft_real(x);
    for (std::size_t k : {3u, 5u, 11u, 20u}) {
        double fftpow  = std::norm(X[k]);  // |X_k|^2
        double goertz  = goertzel_power_bin(x, k);
        EXPECT_NEAR(goertz, fftpow, 1e-6 * (fftpow + 1.0));
    }
    // Present tones (5, 11) have far more power than an absent bin (20).
    EXPECT_GT(goertzel_power_bin(x, 5), 100.0 * goertzel_power_bin(x, 20));
}
