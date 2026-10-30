#pragma once
/// @file fft_diagnostics.hpp
/// @brief Small direct-DFT references, calibrated one-sided spectra and FFT convolution.
#include <datamunge/algorithms/fft.hpp>
#include <datamunge/dstruct/dataframe.hpp>
namespace datamunge::algorithms {
/// @brief Window applied to M observed samples before any zero padding.
/// Both windows are deterministic and nonnegative. Periodic Hann uses denominator M,
/// as in scipy.signal.windows.hann(M,sym=False), not the symmetric M-1 convention.
/// @par Example
/// See fft_spectrum() for both modes and calibrated amplitudes.
/// @par References
/// - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.hann.html
/// @see fft_spectrum()
enum class FftWindow {
  Rectangular, ///< Every observed sample receives weight one.
  PeriodicHann ///< Weight .5-.5*cos(2*pi*t/M); requires M>=2.
};
/// @brief Evaluates the defining complex DFT directly with extended-precision accumulation.
/// An O(N^2) small-signal reference, not a fast transform. Unlike fft(), arbitrary lengths
/// are accepted. Input is borrowed unchanged; result owns its values; no shared state.
/// @param samples Finite complex samples; an empty sequence gives empty output.
/// @param inverse False (default): negative exponent, no scale; true: positive exponent divided by
/// N.
/// @return Owned N coefficients/samples, rounded to finite double.
/// @throws std::invalid_argument Nonfinite input or unrepresentable output.
/// @par Example
/// @code
/// const auto x=dm::algorithms::dft_reference({{1,0},{1,0},{1,0}});
/// std::print("{:.0f}\n",x[0].real()); // 3
/// @endcode
/// @par References
/// - https://www.fftw.org/fftw3_doc/The-1d-Discrete-Fourier-Transform-_0028DFT_0029.html
/// @see fft(), fft_diagnostics()
[[nodiscard]] std::vector<std::complex<double>> dft_reference(
    const std::vector<std::complex<double>>& samples,
    bool                                     inverse = false);
/// @brief Checks FFT coefficients, full complex inverse residual and Parseval energy against a
/// direct DFT. Quadratic work O(N^2) is intentional; output is numerical evidence, not a timing
/// benchmark or a guarantee for every dynamic range. Empty input gives a single zero row.
/// No mutation/shared state; auxiliary storage O(N). Ratios with zero scale are zero.
/// @param samples Finite complex samples, length zero or a power of two.
/// @return One-row frame: length, max_dft_error, relative_dft_error, max_inverse_error,
/// relative_inverse_error, time_energy, frequency_energy_over_n, relative_energy_error.
/// DFT error divides by the largest direct coefficient; inverse error by largest sample;
/// energy error by the larger energy. Complex magnitudes include imaginary residuals.
/// @throws std::invalid_argument Bad length, nonfinite samples or unrepresentable result/energy.
/// @par Example
/// @code
/// const auto d=dm::algorithms::fft_diagnostics({{1,0},{2,0},{3,0},{4,0}});
/// std::print("{:.0f}\n",d.double_at("time_energy",0)); // 30
/// @endcode
/// @par References
/// - Cooley and Tukey (1965). https://doi.org/10.1090/S0025-5718-1965-0178586-1
/// - https://www.fftw.org/fftw3_doc/The-1d-Discrete-Fourier-Transform-_0028DFT_0029.html
/// @see dft_reference()
[[nodiscard]] dstruct::DataFrame fft_diagnostics(const std::vector<std::complex<double>>& samples);
/// @brief One-sided real-input spectrum with explicit sample rate, window and padding
/// normalization. Window M observed samples, append zeros to N, then call the full radix-2 FFT.
/// Define W=sum(w), U=sum(w^2), a_k=1 at DC and even-N Nyquist, 2 elsewhere. Returned amplitude is
/// a_k*abs(X_k)/W: coherent-tone magnitude calibration, not an exact off-bin amplitude estimate.
/// Returned mean_square is a_k*abs(X_k)^2/(N*U); its sum is sum(w^2*x^2)/U. It is a mean-square
/// contribution per bin, not a spectral density per Hz. Padding changes bin spacing, not observed
/// duration or intrinsic resolving power; W/U exclude padded zeros. Borrows input unchanged; owns
/// columns; no shared state. Work O(N log N), storage O(N).
/// @param samples Nonempty finite real observations, length M.
/// @param sample_rate Finite positive samples per time unit, default one; frequencies use its
/// units.
/// @param fft_length Zero (default) uses M. Otherwise a power of two N>=M; no implicit cropping.
/// @param window Rectangular (default) or PeriodicHann; the latter requires M>=2.
/// @return Rows k=0..floor(N/2): bin/frequency/real/imaginary/magnitude/amplitude/mean_square,
/// including DC and Nyquist for even N; a singleton produces only DC.
/// @throws std::invalid_argument Empty/nonfinite input, invalid enum/rate/length or unrepresentable
/// output.
/// @par Example
/// @code
/// const auto s=dm::algorithms::fft_spectrum({2,2,2,2},4);
/// const auto
/// h=dm::algorithms::fft_spectrum({2,2,2,2},4,8,dm::algorithms::FftWindow::PeriodicHann);
/// std::print("{:.0f} {:.0f}\n",s.double_at("amplitude",0),h.double_at("amplitude",0)); // 2 2
/// @endcode
/// @par References
/// - https://docs.scipy.org/doc/scipy/tutorial/signal.html (spectral analysis and window
/// normalization).
/// - https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html (Hermitian bins).
/// @see fft_real(), fft_diagnostics()
[[nodiscard]] dstruct::DataFrame fft_spectrum(
    const std::vector<double>& samples,
    double                     sample_rate = 1,
    std::size_t                fft_length  = 0,
    FftWindow                  window      = FftWindow::Rectangular);
/// @brief Full linear convolution via zero padding, spectral multiplication and normalized inverse
/// FFT. Pads to the smallest power of two >= a.size()+b.size()-1, preventing circular wrap.
/// Discards the inverse's roundoff imaginary residual and returns real coefficients.
/// Work O(N log N), storage O(N); inputs borrowed unchanged, owned result, no shared state.
/// @param a Finite real input. If either input is empty, returns empty without examining the other.
/// @param b Finite real input.
/// @return a.size()+b.size()-1 full convolution samples, or empty for an empty operand.
/// @throws std::invalid_argument Nonfinite input or unrepresentable output.
/// @throws std::length_error Required convolution/padded length exceeds size_t range.
/// @par Example
/// @code
/// const auto y=dm::algorithms::fft_linear_convolution({1,2,3,4},{1,1});
/// std::print("{} {:.0f}\n",y.size(),y[4]); // 5 4
/// @endcode
/// @par References
/// - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html
/// @see fft()
[[nodiscard]] std::vector<double> fft_linear_convolution(
    const std::vector<double>& a,
    const std::vector<double>& b);
} // namespace datamunge::algorithms
