/// @file fft_diagnostics.cpp
/// @brief Checked direct DFT, window/amplitude normalization and linear FFT convolution.
#include <algorithm>
#include <datamunge/algorithms/fft_diagnostics.hpp>
#include <limits>
#include <numbers>
namespace datamunge::algorithms {
/// @brief Private finite-arithmetic and window helpers.
namespace {
/// @brief Rejects a nonfinite or nonrepresentable double result, permitting underflow.
/// @param x Extended arithmetic value.
/// @return Finite double.
/// @throws std::invalid_argument Nonfinite/unrepresentable result.
double finite_fft_value(long double x) {
  const double result = static_cast<double>(x);
  if (!std::isfinite(result))
    throw std::invalid_argument("FFT diagnostics: nonfinite or unrepresentable value");
  return result;
}
/// @brief Validates every real and imaginary component without changing the input.
/// @param x Borrowed complex samples.
/// @throws std::invalid_argument A component is nonfinite.
void check_complex(const std::vector<std::complex<double>>& x) {
  for (auto z : x) {
    (void)finite_fft_value(z.real());
    (void)finite_fft_value(z.imag());
  }
}
/// @brief Extended squared complex magnitude, avoiding premature double overflow.
/// @param x Finite double complex value.
/// @return Extended real squared magnitude.
long double square_magnitude(std::complex<double> x) {
  return static_cast<long double>(x.real()) * x.real() +
         static_cast<long double>(x.imag()) * x.imag();
}
/// @brief Produces the declared periodic or rectangular observed-sample weights.
/// @param m Positive observed length.
/// @param window Valid window enumerator; periodic Hann requires m>=2.
/// @return Owned m weights.
/// @throws std::invalid_argument Bad enum or invalid Hann length.
std::vector<double> observed_window(std::size_t m, FftWindow window) {
  if (window != FftWindow::Rectangular && window != FftWindow::PeriodicHann)
    throw std::invalid_argument("fft_spectrum: invalid window");
  if (window == FftWindow::PeriodicHann && m < 2)
    throw std::invalid_argument("fft_spectrum: Hann requires at least two observations");
  std::vector<double> w(m, 1);
  if (window == FftWindow::PeriodicHann)
    for (std::size_t i = 0; i < m; ++i)
      w[i] = .5 - .5 * std::cos(2 * std::numbers::pi_v<double> * static_cast<double>(i) / m);
  return w;
}
} // namespace
std::vector<std::complex<double>> dft_reference(
    const std::vector<std::complex<double>>& x,
    bool                                     inverse) {
  check_complex(x);
  const auto                        n = x.size();
  std::vector<std::complex<double>> out(n);
  for (std::size_t k = 0; k < n; ++k) {
    std::complex<long double> sum = 0;
    for (std::size_t t = 0; t < n; ++t) {
      const long double angle =
          (inverse ? 2.L : -2.L) * std::numbers::pi_v<long double> * k * t / n;
      sum += std::complex<long double>(x[t].real(), x[t].imag()) *
             std::complex<long double>(std::cos(angle), std::sin(angle));
    }
    if (inverse)
      sum /= n;
    out[k] = {finite_fft_value(sum.real()), finite_fft_value(sum.imag())};
  }
  return out;
}
dstruct::DataFrame fft_diagnostics(const std::vector<std::complex<double>>& x) {
  check_complex(x);
  const auto transformed = fft(x), direct = dft_reference(x), restored = fft(transformed, true);
  check_complex(transformed);
  check_complex(restored);
  long double time = 0, frequency = 0, error = 0, inverse_error = 0, largest = 0, input_largest = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    time += square_magnitude(x[i]);
    frequency += square_magnitude(transformed[i]);
    error = std::max(
        error,
        std::abs(
            std::complex<long double>(transformed[i].real(), transformed[i].imag()) -
            std::complex<long double>(direct[i].real(), direct[i].imag())));
    inverse_error = std::max(
        inverse_error,
        std::abs(
            std::complex<long double>(restored[i].real(), restored[i].imag()) -
            std::complex<long double>(x[i].real(), x[i].imag())));
    largest       = std::max(largest, std::sqrt(square_magnitude(direct[i])));
    input_largest = std::max(input_largest, std::sqrt(square_magnitude(x[i])));
  }
  if (!x.empty())
    frequency /= x.size();
  const auto         energy_scale = std::max(time, frequency);
  dstruct::DataFrame out;
  out.add_column("length", std::vector<double>{double(x.size())});
  out.add_column("max_dft_error", std::vector<double>{finite_fft_value(error)});
  out.add_column(
      "relative_dft_error", std::vector<double>{finite_fft_value(largest ? error / largest : 0)});
  out.add_column("max_inverse_error", std::vector<double>{finite_fft_value(inverse_error)});
  out.add_column(
      "relative_inverse_error",
      std::vector<double>{finite_fft_value(input_largest ? inverse_error / input_largest : 0)});
  out.add_column("time_energy", std::vector<double>{finite_fft_value(time)});
  out.add_column("frequency_energy_over_n", std::vector<double>{finite_fft_value(frequency)});
  out.add_column(
      "relative_energy_error",
      std::vector<double>{
          finite_fft_value(energy_scale ? std::abs(time - frequency) / energy_scale : 0)});
  return out;
}
dstruct::DataFrame
fft_spectrum(const std::vector<double>& x, double fs, std::size_t n, FftWindow window) {
  if (x.empty() || !std::isfinite(fs) || fs <= 0)
    throw std::invalid_argument("fft_spectrum: nonempty samples and positive finite rate required");
  if (n == 0)
    n = x.size();
  if (n < x.size() || !is_power_of_two(n))
    throw std::invalid_argument(
        "fft_spectrum: power-of-two length at least the observed length required");
  const auto          w = observed_window(x.size(), window);
  std::vector<double> padded(n, 0);
  long double         sum = 0, squares = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    (void)finite_fft_value(x[i]);
    padded[i] = finite_fft_value(static_cast<long double>(x[i]) * w[i]);
    sum += w[i];
    squares += static_cast<long double>(w[i]) * w[i];
  }
  const auto transformed = fft_real(padded);
  check_complex(transformed);
  std::vector<double> bins, freq, re, im, mag, amplitude, power;
  for (std::size_t k = 0; k <= n / 2; ++k) {
    const long double a         = (k == 0 || (n > 1 && k == n / 2)) ? 1 : 2;
    const auto        magnitude = std::sqrt(square_magnitude(transformed[k]));
    bins.push_back(k);
    freq.push_back(finite_fft_value(static_cast<long double>(k) / n * fs));
    re.push_back(transformed[k].real());
    im.push_back(transformed[k].imag());
    mag.push_back(finite_fft_value(magnitude));
    amplitude.push_back(finite_fft_value(a * magnitude / sum));
    power.push_back(finite_fft_value(a * square_magnitude(transformed[k]) / (n * squares)));
  }
  dstruct::DataFrame out;
  out.add_column("bin", bins);
  out.add_column("frequency", freq);
  out.add_column("real", re);
  out.add_column("imaginary", im);
  out.add_column("magnitude", mag);
  out.add_column("amplitude", amplitude);
  out.add_column("mean_square", power);
  return out;
}
std::vector<double> fft_linear_convolution(
    const std::vector<double>& a,
    const std::vector<double>& b) {
  if (a.empty() || b.empty())
    return {};
  if (a.size() > std::numeric_limits<std::size_t>::max() - b.size() + 1)
    throw std::length_error("FFT convolution length overflow");
  const auto  length = a.size() + b.size() - 1;
  std::size_t n      = 1;
  while (n < length) {
    if (n > std::numeric_limits<std::size_t>::max() / 2)
      throw std::length_error("FFT padded length overflow");
    n *= 2;
  }
  std::vector<double> x(n, 0), y(n, 0);
  for (std::size_t i = 0; i < a.size(); ++i)
    x[i] = finite_fft_value(a[i]);
  for (std::size_t i = 0; i < b.size(); ++i)
    y[i] = finite_fft_value(b[i]);
  auto       X = fft_real(x);
  const auto Y = fft_real(y);
  for (std::size_t i = 0; i < n; ++i)
    X[i] *= Y[i];
  const auto result = fft(std::move(X), true);
  check_complex(result);
  std::vector<double> out(length);
  for (std::size_t i = 0; i < length; ++i)
    out[i] = finite_fft_value(result[i].real());
  return out;
}
} // namespace datamunge::algorithms
