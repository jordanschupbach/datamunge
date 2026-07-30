#pragma once

/// \file goertzel.hpp
/// \brief The Goertzel algorithm: efficiently detect a single frequency component in
///        a signal (Goertzel 1958).
///
/// When only *one* (or a few) frequency bins are needed -- not the whole spectrum --
/// a full FFT is wasteful. The Goertzel algorithm computes a single DFT term with a
/// second-order recurrence that costs \f$O(N)\f$ per frequency and needs only a couple
/// of state variables, making it ideal for embedded tone detection (its classic use
/// is decoding DTMF touch-tone digits). For a target bin \f$k\f$ with
/// \f$\omega = 2\pi k/N\f$ and coefficient \f$c = 2\cos\omega\f$, it runs the IIR filter
/// \f[
///   s_n = x_n + c\,s_{n-1} - s_{n-2},
/// \f]
/// over the \f$N\f$ samples, after which the bin's squared magnitude ("power") is
/// \f$|X_k|^2 = s_{N-1}^2 + s_{N-2}^2 - c\,s_{N-1}s_{N-2}\f$. It is exactly the DFT
/// coefficient's magnitude, obtained without any complex arithmetic in the loop and
/// without storing the whole signal transform.

#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// \brief Goertzel power (squared magnitude) of the DFT bin nearest \p target_freq.
///
/// \param x            Real samples.
/// \param target_freq  Frequency of interest (Hz).
/// \param sample_rate  Sampling rate (Hz).
/// \return \f$|X_k|^2\f$ for the bin \f$k=\mathrm{round}(N\,f/f_s)\f$.
inline double goertzel_power(const std::vector<double>& x, double target_freq, double sample_rate) {
    const std::size_t n = x.size();
    if (n == 0) return 0.0;
    const int    k     = static_cast<int>(0.5 + static_cast<double>(n) * target_freq / sample_rate);
    const double omega = 2.0 * M_PI * static_cast<double>(k) / static_cast<double>(n);
    const double coeff = 2.0 * std::cos(omega);

    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        s0 = x[i] + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    return s1 * s1 + s2 * s2 - coeff * s1 * s2;
}

/// \brief Goertzel power for an integer DFT bin index \p k directly.
inline double goertzel_power_bin(const std::vector<double>& x, std::size_t k) {
    const std::size_t n = x.size();
    if (n == 0) return 0.0;
    const double omega = 2.0 * M_PI * static_cast<double>(k) / static_cast<double>(n);
    const double coeff = 2.0 * std::cos(omega);
    double       s1 = 0.0, s2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double s0 = x[i] + coeff * s1 - s2;
        s2              = s1;
        s1              = s0;
    }
    return s1 * s1 + s2 * s2 - coeff * s1 * s2;
}

}  // namespace datamunge::algorithms
