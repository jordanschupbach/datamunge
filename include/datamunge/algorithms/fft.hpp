#pragma once

/// \file fft.hpp
/// \brief The Cooley-Tukey fast Fourier transform (radix-2) and its inverse
///        (Cooley & Tukey 1965).
///
/// The discrete Fourier transform (DFT) of a length-\f$N\f$ sequence,
/// \f$X_k=\sum_{n=0}^{N-1} x_n e^{-2\pi i kn/N}\f$, costs \f$O(N^2)\f$ computed
/// directly. The *fast* Fourier transform reduces this to \f$O(N\log N)\f$ by
/// recursively splitting the transform into its even- and odd-indexed subsequences
/// (the *divide-and-conquer* Cooley-Tukey factorization). For \f$N\f$ a power of two,
/// the split repeats \f$\log_2 N\f$ times, and each level does \f$N/2\f$ *butterfly*
/// operations \f$(a,b)\mapsto(a+wb,\,a-wb)\f$ with twiddle factor
/// \f$w=e^{-2\pi i k/N}\f$. This implementation is the classic iterative version: a
/// bit-reversal permutation followed by \f$\log_2 N\f$ butterfly passes. The inverse
/// transform reuses the same butterflies with conjugated twiddles and a \f$1/N\f$
/// scaling. The FFT is arguably the most important numerical algorithm of the 20th
/// century -- the engine of digital signal processing, fast convolution, and spectral
/// methods.

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// True iff \p n is a power of two (and nonzero).
inline bool is_power_of_two(std::size_t n) { return n != 0 && (n & (n - 1)) == 0; }

/// \brief In-place-style radix-2 Cooley-Tukey FFT (returns the transformed vector).
///
/// \param a        Input samples (length must be a power of two).
/// \param inverse  If true, computes the inverse DFT (conjugate twiddles, \f$1/N\f$ scaling).
/// \return the DFT (or inverse DFT) of \p a.
inline std::vector<std::complex<double>> fft(std::vector<std::complex<double>> a, bool inverse = false) {
    const std::size_t n = a.size();
    if (n == 0) return a;
    if (!is_power_of_two(n)) throw std::invalid_argument("fft: length must be a power of two");

    // Bit-reversal permutation.
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    const double sign = inverse ? 1.0 : -1.0;
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double            theta = sign * 2.0 * M_PI / static_cast<double>(len);
        const std::complex<double> wlen(std::cos(theta), std::sin(theta));
        for (std::size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (std::size_t k = 0; k < len / 2; ++k) {
                const std::complex<double> u = a[i + k];
                const std::complex<double> v = a[i + k + len / 2] * w;
                a[i + k]           = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (inverse)
        for (auto& z : a) z /= static_cast<double>(n);
    return a;
}

/// Convenience: forward FFT of a real signal (length a power of two).
inline std::vector<std::complex<double>> fft_real(const std::vector<double>& x) {
    std::vector<std::complex<double>> a(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) a[i] = std::complex<double>(x[i], 0.0);
    return fft(std::move(a), false);
}

}  // namespace datamunge::algorithms
