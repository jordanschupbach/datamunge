#pragma once

/// @file spectral_density.hpp
/// @brief Welch averaged-periodogram and Thomson multitaper spectral density estimates with chi-square bands.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <optional>
#include <vector>

namespace datamunge::stats {

/// @brief Data window applied to each Welch segment (periodic forms, as SciPy's `get_window`).
/// @par Example
/// @code
/// const auto w = dm::stats::SpectralWindow::hann;
/// std::println("{}", w == dm::stats::SpectralWindow::rectangular); // false
/// @endcode
enum class SpectralWindow {
    hann,        ///< \f$0.5-0.5\cos(2\pi k/N)\f$.
    hamming,     ///< \f$0.54-0.46\cos(2\pi k/N)\f$.
    blackman,    ///< \f$0.42-0.5\cos(2\pi k/N)+0.08\cos(4\pi k/N)\f$.
    rectangular, ///< All ones (Bartlett's method when segments do not overlap).
};

/// @brief Trend removed from each segment before tapering.
/// @par Example
/// @code
/// const auto d = dm::stats::SegmentDetrend::linear;
/// std::println("{}", d == dm::stats::SegmentDetrend::none); // false
/// @endcode
enum class SegmentDetrend {
    none,     ///< Use the segment as is.
    constant, ///< Subtract the segment mean.
    linear,   ///< Subtract the least-squares line.
};

/// @brief Units of the Welch estimate.
/// @par Example
/// @code
/// const auto s = dm::stats::SpectralScaling::spectrum;
/// std::println("{}", s == dm::stats::SpectralScaling::density); // false
/// @endcode
enum class SpectralScaling {
    density,  ///< Power per unit frequency; integrates over \f$[0,f_s/2]\f$ to the variance.
    spectrum, ///< Power per bin; a sinusoid of amplitude A gives a peak near \f$A^2/2\f$.
};

/// @brief How segment periodograms are combined.
/// @par Example
/// @code
/// const auto a = dm::stats::SegmentAverage::median;
/// std::println("{}", a == dm::stats::SegmentAverage::mean); // false
/// @endcode
enum class SegmentAverage {
    mean,   ///< Arithmetic mean (Welch's method).
    median, ///< Median divided by its bias for exponential variables; robust to bursts.
};

/// @brief Options for welch_spectrum().
/// @par Example
/// @code
/// const dm::stats::WelchOptions options{.sampling_rate = 100, .segment_length = 128};
/// std::println("{}", options.segment_length); // 128
/// @endcode
struct WelchOptions {
    double sampling_rate{1};                    ///< Samples per unit time \f$f_s\f$.
    std::size_t segment_length{256};            ///< Samples per segment (reduced to the series length if longer).
    std::optional<std::size_t> overlap{};       ///< Overlapping samples (default half a segment).
    std::size_t fft_length{0};                  ///< Zero-padded transform length (0 means the segment length).
    SpectralWindow window{SpectralWindow::hann}; ///< Segment window.
    SegmentDetrend detrend{SegmentDetrend::constant}; ///< Per-segment detrending.
    SpectralScaling scaling{SpectralScaling::density}; ///< Output units.
    SegmentAverage average{SegmentAverage::mean}; ///< Averaging rule.
    double confidence{0.95};                    ///< Coverage of the chi-square bands.
};

/// @brief A one-sided spectral estimate on a frequency grid.
/// @par Example
/// @code
/// const dm::stats::SpectralEstimate s;
/// std::println("{}", s.frequencies.size()); // 0
/// @endcode
struct SpectralEstimate {
    std::vector<double> frequencies; ///< Grid \f$k f_s/n_{\text{fft}}\f$ from 0 to the Nyquist frequency.
    std::vector<double> density;     ///< Estimate at each frequency (one-sided: doubled except at 0 and Nyquist).
    std::vector<double> lower;       ///< Lower confidence band \f$\nu\hat S/\chi^2_{\nu,1-\alpha/2}\f$ (NaN when undefined).
    std::vector<double> upper;       ///< Upper confidence band \f$\nu\hat S/\chi^2_{\nu,\alpha/2}\f$ (NaN when undefined).
    std::vector<double> degrees_of_freedom; ///< Equivalent degrees of freedom \f$\nu\f$ at each frequency.
    std::size_t averaged{};          ///< Number of segments (Welch) or tapers (multitaper) combined.
};

/// @brief Welch's averaged modified periodogram, as SciPy's `scipy.signal.welch`.
///
/// Splits the series into segments of `segment_length` with the given overlap (trailing samples
/// that do not fill a segment are dropped), detrends and windows each one, and averages the
/// periodograms. With mean averaging the bands use the equivalent degrees of freedom of
/// Percival and Walden for overlapping segments,
/// \f$\nu=2N_S\big/\big(1+2\sum_{m=1}^{N_S-1}(1-m/N_S)\,|\sum_t h_th_{t+mD}|^2\big)\f$ with
/// unit-energy window \f$h\f$ and step \f$D\f$ (halved at 0 and Nyquist, where periodograms
/// are \f$\chi^2_1\f$). They are pointwise and assume a smooth spectrum across the window's
/// bandwidth. With median averaging the bands are NaN.
/// @param x Finite series with at least two values.
/// @param options Segmenting, window, detrending, units and averaging.
/// @return Frequencies, estimate, bands and degrees of freedom.
/// @throws std::invalid_argument For fewer than two values, nonfinite values, a segment shorter
/// than 2, overlap not below the segment length, an FFT length below the segment length, a
/// nonpositive sampling rate or confidence outside (0, 1).
/// @par Example
/// @code
/// std::vector<double> x;
/// for (int t = 0; t < 1024; ++t)
///     x.push_back(std::sin(2 * std::numbers::pi * .125 * t));
/// const auto s = dm::stats::welch_spectrum(x, {.segment_length = 64});
/// const auto peak = std::ranges::max_element(s.density) - s.density.begin();
/// std::println("{} {}", s.frequencies[peak], s.averaged); // 0.125 31
/// @endcode
/// @par References
/// - P. D. Welch (1967). "The use of fast Fourier transform for the estimation of power spectra."
///   IEEE Transactions on Audio and Electroacoustics 15(2), 70-73. doi:10.1109/TAU.1967.1161901
/// - D. B. Percival and A. T. Walden (1993). Spectral Analysis for Physical Applications.
///   Cambridge University Press, section 6.17. doi:10.1017/CBO9780511622762
/// @see multitaper_spectrum()
[[nodiscard]] SpectralEstimate welch_spectrum(const std::vector<double>& x, const WelchOptions& options = {});

/// @brief Discrete prolate spheroidal (Slepian) sequences.
/// @par Example
/// @code
/// const dm::stats::DpssTapers d;
/// std::println("{}", d.concentrations.size()); // 0
/// @endcode
struct DpssTapers {
    linalg::DenseMatrix<double> tapers;   ///< n by K matrix of unit-energy tapers, one per column.
    std::vector<double> concentrations;   ///< Fraction of each taper's energy inside \f$|f|\le W\f$.
};

/// @brief The K most band-concentrated sequences of length n with half-bandwidth \f$W=NW/n\f$.
///
/// Eigenvectors of the commuting tridiagonal matrix with diagonal
/// \f$\big((n-1-2t)/2\big)^2\cos 2\pi W\f$ and off-diagonal \f$t(n-t)/2\f$, found by Sturm
/// bisection and inverse iteration in \f$O(nK)\f$ memory. Signs follow SciPy's `windows.dpss`
/// (even tapers have positive sum, odd tapers start positive). Concentrations come from the
/// taper autocorrelations, as SciPy's `return_ratios`.
/// @param n Taper length (at least 2).
/// @param time_bandwidth \f$NW\f$, in (0, n/2).
/// @param count Number of tapers K (1 to n).
/// @return Tapers and concentrations, most concentrated first.
/// @throws std::invalid_argument For invalid sizes or bandwidth.
/// @par Example
/// @code
/// const auto d = dm::stats::dpss_tapers(128, 3, 5);
/// std::println("{:.4f} {:.4f}", d.concentrations[0], d.concentrations[4]); // 1.0000 0.9462
/// @endcode
/// @par References
/// - D. Slepian (1978). "Prolate spheroidal wave functions, Fourier analysis, and uncertainty V:
///   the discrete case." Bell System Technical Journal 57(5), 1371-1430.
///   doi:10.1002/j.1538-7305.1978.tb02104.x
/// @see multitaper_spectrum()
[[nodiscard]] DpssTapers dpss_tapers(std::size_t n, double time_bandwidth, std::size_t count);

/// @brief Options for multitaper_spectrum().
/// @par Example
/// @code
/// const dm::stats::MultitaperOptions options{.time_bandwidth = 3, .tapers = 5};
/// std::println("{}", options.tapers); // 5
/// @endcode
struct MultitaperOptions {
    double sampling_rate{1};      ///< Samples per unit time \f$f_s\f$.
    double time_bandwidth{4};     ///< \f$NW\f$; the resolution half-bandwidth is \f$NW f_s/n\f$.
    std::size_t tapers{0};        ///< Number of tapers K (0 means \f$\lfloor 2NW\rfloor-1\f$).
    bool adaptive{true};          ///< Thomson's adaptive weights instead of a plain average.
    std::size_t fft_length{0};    ///< Zero-padded transform length (0 means \f$2^{\lceil\log_2 n\rceil+1}\f$, as R's multitaper).
    bool demean{true};            ///< Subtract the sample mean first.
    double confidence{0.95};      ///< Coverage of the chi-square bands.
    std::size_t max_iterations{100}; ///< Adaptive iterations per frequency.
};

/// @brief Thomson's multitaper spectral estimate.
///
/// Computes eigenspectra \f$\hat S_k(f)=\Delta t\,|\sum_t v_{k,t}x_te^{-2\pi ift\Delta t}|^2\f$ with DPSS
/// tapers. The plain estimate averages them with \f$\nu=2K\f$. The adaptive estimate iterates
/// \f$\hat S=\sum_k b_k^2\lambda_k\hat S_k/\sum_k b_k^2\lambda_k\f$ with
/// \f$b_k=\hat S/(\lambda_k\hat S+(1-\lambda_k)\hat\sigma^2\Delta t)\f$ to convergence, which down-weights
/// leaky high-order tapers where the spectrum is low, and reports
/// \f$\nu=2(\sum b_k^2\lambda_k)^2/\sum b_k^4\lambda_k^2\f$ (Percival and Walden, eq. 370a-b). Output is
/// one-sided. The plain estimate matches R's `multitaper::spec.mtm(centre = "arithMean",
/// adaptiveWeighting = FALSE)` after doubling. R's adaptive option uses a different weight and a
/// loose stopping rule.
/// @param x Finite series with at least four values.
/// @param options Bandwidth, taper count, weighting, padding and band coverage.
/// @return Frequencies, estimate, bands and degrees of freedom.
/// @throws std::invalid_argument For short or nonfinite input, invalid bandwidth or taper count,
/// an FFT length below n, a nonpositive sampling rate or confidence outside (0, 1).
/// @par Example
/// @code
/// std::vector<double> x;
/// for (int t = 0; t < 500; ++t)
///     x.push_back(std::cos(2 * std::numbers::pi * .2 * t) + (t % 7) / 7.0);
/// const auto s = dm::stats::multitaper_spectrum(x, {.time_bandwidth = 3});
/// const auto peak = std::ranges::max_element(s.density) - s.density.begin();
/// std::println("{:.2f} {}", s.frequencies[peak], s.averaged); // 0.20 5
/// @endcode
/// @par References
/// - D. J. Thomson (1982). "Spectrum estimation and harmonic analysis." Proceedings of the IEEE
///   70(9), 1055-1096. doi:10.1109/PROC.1982.12433
/// - D. B. Percival and A. T. Walden (1993). Spectral Analysis for Physical Applications.
///   Cambridge University Press, chapter 7. doi:10.1017/CBO9780511622762
/// @see dpss_tapers(), welch_spectrum()
[[nodiscard]] SpectralEstimate multitaper_spectrum(const std::vector<double>& x, const MultitaperOptions& options = {});

} // namespace datamunge::stats
