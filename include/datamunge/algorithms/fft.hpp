#pragma once

/// @file fft.hpp
/// @brief Copy-owned radix-2 Cooley-Tukey FFT, with an unnormalized forward and 1/N inverse.
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief Tests the nonzero power-of-two domain accepted by the radix-2 transform.
/// @param n Integer length, including zero.
/// @return True exactly for 1,2,4,8,...; false for zero. Work/storage O(1).
/// @par Example
/// @code
/// std::print("{} {}\n", dm::algorithms::is_power_of_two(8),
///            dm::algorithms::is_power_of_two(0)); // true false
/// @endcode
inline bool is_power_of_two(std::size_t n) {
  return n != 0 && (n & (n - 1)) == 0;
}

/// @brief Forward or inverse complex DFT using iterative radix-2 butterflies.
/// The forward uses exp(-2*pi*i*k*t/N) with no scaling. Inverse uses the positive
/// exponent and divides by N. Bit reversal followed by log2(N) passes costs O(N log N)
/// time; the by-value working vector uses O(N) storage. Input lvalues are copied and
/// unchanged, rvalues can donate storage; no shared mutable state. Empty returns empty
/// and a singleton is identity. Finite arithmetic has roundoff; nonfinite values or
/// overflow are propagated, not validated by this low-level transform.
/// @param a Owned working samples; nonempty length must be a power of two.
/// @param inverse False (default) for forward; true for normalized inverse.
/// @return Owned N complex coefficients in standard unshifted bin order, or inverse samples.
/// @throws std::invalid_argument Nonempty length is not a power of two.
/// @par Example
/// @code
/// const auto x=dm::algorithms::fft({{1,0},{2,0},{3,0},{4,0}});
/// std::print("{:.0f} {:.0f}\n",x[1].real(),x[1].imag()); // -2 2
/// @endcode
/// @par References
/// - Cooley and Tukey (1965), An algorithm for the machine calculation of complex Fourier series.
///   https://doi.org/10.1090/S0025-5718-1965-0178586-1
/// @see fft_real(), fft_diagnostics()
inline std::vector<std::complex<double>> fft(
    std::vector<std::complex<double>> a,
    bool                              inverse = false) {
  const std::size_t n = a.size();
  if (n == 0)
    return a;
  if (!is_power_of_two(n))
    throw std::invalid_argument("fft: length must be a power of two");

  // Bit-reversal permutation.
  for (std::size_t i = 1, j = 0; i < n; ++i) {
    std::size_t bit = n >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j)
      std::swap(a[i], a[j]);
  }

  const double sign = inverse ? 1.0 : -1.0;
  for (std::size_t len = 2; len <= n; len <<= 1) {
    const double theta = sign * 2.0 * std::numbers::pi_v<double> / static_cast<double>(len);
    const std::complex<double> wlen(std::cos(theta), std::sin(theta));
    for (std::size_t i = 0; i < n; i += len) {
      std::complex<double> w(1.0, 0.0);
      for (std::size_t k = 0; k < len / 2; ++k) {
        const std::complex<double> u = a[i + k];
        const std::complex<double> v = a[i + k + len / 2] * w;
        a[i + k]                     = u + v;
        a[i + k + len / 2]           = u - v;
        w *= wlen;
      }
    }
    if (len == n)
      break; // Avoid overflowing the stage length after the final pass.
  }
  if (inverse)
    for (auto& z : a)
      z /= static_cast<double>(n);
  return a;
}

/// @brief Forward FFT of real samples, returning the full N-bin complex transform.
/// Unlike a packed rfft, this does not discard the conjugate negative-frequency half.
/// Empty input returns empty. Samples are borrowed unchanged; output owns its storage.
/// Work O(N log N), storage O(N); no shared mutable state or nonfinite-value validation.
/// @param x Real samples, length zero or a power of two.
/// @return N complex coefficients; for finite real input X[N-k]=conj(X[k]) up to roundoff.
/// @throws std::invalid_argument Nonempty length is not a power of two.
/// @par Example
/// @code
/// const auto x=dm::algorithms::fft_real({1,0,-1,0});
/// std::print("{} {:.0f}\n",x.size(),x[1].real()); // 4 2
/// @endcode
/// @par References
/// - Cooley and Tukey (1965). https://doi.org/10.1090/S0025-5718-1965-0178586-1
/// @see fft(), fft_spectrum()
inline std::vector<std::complex<double>> fft_real(const std::vector<double>& x) {
  std::vector<std::complex<double>> a(x.size());
  for (std::size_t i = 0; i < x.size(); ++i)
    a[i] = std::complex<double>(x[i], 0.0);
  return fft(std::move(a), false);
}

} // namespace datamunge::algorithms
