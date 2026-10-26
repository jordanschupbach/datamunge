#pragma once

/// @file fft.hpp
/// @brief Differentiable fast Fourier transforms on `nn::Tensor`.
///
/// `nn::Tensor` has no complex dtype, so complex values travel as a
/// `ComplexTensor`: a pair of float tensors (real and imaginary parts) with
/// the same shape and dtype. Every transform works along the last axis and
/// is batched (and threaded) over the leading axes. Lengths can be anything:
/// lengths whose prime factors are all <= 103 use a mixed-radix Stockham FFT
/// (radix 4, 2, 3, 5 and a generic odd-prime butterfly); other lengths use
/// Bluestein's chirp-z algorithm over a padded 2^a 3^b 5^c length, so every
/// transform is O(n log n). Computation is in double for both dtypes.
///
/// Semantics follow TensorFlow (tf.signal):
///   fft / ifft      complex -> complex, ifft scaled by 1/n (TF FFT / IFFT)
///   rfft            real n -> complex n/2+1 bins (TF RFFT, incl. fft_length)
///   irfft           complex n/2+1 bins -> real n (TF IRFFT, incl. fft_length;
///                   the imaginary parts of the DC and Nyquist bins are ignored)
///   fft2d, rfft2d   the same over the last two axes
///
/// Differentiation: the transforms are built from three linear primitives
/// (`dft`, `rdft` and `rdft_adjoint`) whose backward passes are each other
/// (the adjoint of a DFT is the DFT with the opposite sign), so gradients of
/// every order flow through them. Gradients of a ComplexTensor are the pair
/// (dL/dreal, dL/dimag), as with any real tensors.
///
/// Computation is in double precision internally for both float32 and
/// float64 tensors.

#include <array>
#include <complex>
#include <cstdint>

#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

/// A complex tensor stored as two real float tensors of equal shape and dtype.
struct ComplexTensor {
  Tensor real;
  Tensor imag;

  /// Shape of either part.
  [[nodiscard]] const Shape& shape() const { return real.shape(); }
  /// Dtype of either part.
  [[nodiscard]] DType dtype() const { return real.dtype(); }
  /// Rank of either part.
  [[nodiscard]] std::int64_t dim() const { return real.dim(); }
  /// Size of one axis (negative axes count from the end).
  [[nodiscard]] std::int64_t size(std::int64_t axis) const { return real.size(axis); }
};

// ---- complex helpers (all differentiable compositions of nn ops) ---------------------------------

/// Pairs a real and an imaginary part (broadcast to a common shape and float
/// dtype). An undefined `imag` means zeros.
ComplexTensor make_complex(const Tensor& real, const Tensor& imag = Tensor());
/// a + b (broadcasting).
ComplexTensor complex_add(const ComplexTensor& a, const ComplexTensor& b);
/// a - b (broadcasting).
ComplexTensor complex_sub(const ComplexTensor& a, const ComplexTensor& b);
/// a * b (broadcasting).
ComplexTensor complex_mul(const ComplexTensor& a, const ComplexTensor& b);
/// a / b (broadcasting).
ComplexTensor complex_div(const ComplexTensor& a, const ComplexTensor& b);
/// a scaled by a real tensor (broadcasting).
ComplexTensor complex_scale(const ComplexTensor& a, const Tensor& factor);
/// a scaled by a real number.
ComplexTensor complex_scale(const ComplexTensor& a, double factor);
/// Complex conjugate.
ComplexTensor complex_conj(const ComplexTensor& a);
/// -a.
ComplexTensor complex_neg(const ComplexTensor& a);
/// |a| (TF ComplexAbs). The gradient at 0 is 0 rather than NaN.
Tensor complex_abs(const ComplexTensor& a);
/// |a|^2 = real^2 + imag^2 (smooth everywhere).
Tensor complex_abs_squared(const ComplexTensor& a);
/// arg(a) = atan2(imag, real) in (-pi, pi] (TF Angle). The gradient at 0 is 0.
Tensor complex_angle(const ComplexTensor& a);
/// magnitude * exp(i * phase) (TF Polar).
ComplexTensor complex_from_polar(const Tensor& magnitude, const Tensor& phase);
/// exp(a) = exp(real) * (cos(imag) + i sin(imag)).
ComplexTensor complex_exp(const ComplexTensor& a);
/// Sum over axes of both parts.
ComplexTensor complex_sum(const ComplexTensor& a, const std::vector<std::int64_t>& axes = {}, bool keepdims = false);
/// Python-style slice of both parts along one axis.
ComplexTensor complex_slice_axis(const ComplexTensor& a, std::int64_t axis, std::int64_t start, std::int64_t stop,
                                 std::int64_t step = 1);
/// Both parts reshaped.
ComplexTensor complex_reshape(const ComplexTensor& a, const Shape& shape);
/// Both parts with two axes swapped.
ComplexTensor complex_swapaxes(const ComplexTensor& a, std::int64_t axis0, std::int64_t axis1);
/// Both parts converted to another float dtype.
ComplexTensor complex_cast(const ComplexTensor& a, DType dtype);

// ---- primitives (unnormalized; differentiable, mutually adjoint) --------------------------------

/// Unnormalized DFT along the last axis:
/// y_k = sum_j x_j exp(s 2 pi i j k / n) with s = -1 (forward) or +1 (inverse).
/// Its backward pass is `dft` with the opposite sign.
ComplexTensor dft(const ComplexTensor& x, bool inverse = false);
/// Unnormalized half-spectrum DFT of a real signal of length n along the last
/// axis: bins k = 0 .. n/2. Backward: `rdft_adjoint`.
ComplexTensor rdft(const Tensor& x);
/// The adjoint of `rdft` for length `n`: y_j = Re sum_{k <= n/2} x_k exp(2 pi i j k / n),
/// mapping [..., n/2+1] complex to [..., n] real. Backward: `rdft`.
Tensor rdft_adjoint(const ComplexTensor& x, std::int64_t n);

// ---- transforms --------------------------------------------------------------------------------

/// Forward DFT along the last axis (TF FFT).
ComplexTensor fft(const ComplexTensor& x);
/// Inverse DFT along the last axis, scaled by 1/n (TF IFFT).
ComplexTensor ifft(const ComplexTensor& x);
/// 2-D DFT over the last two axes (TF FFT2D).
ComplexTensor fft2d(const ComplexTensor& x);
/// 2-D inverse DFT over the last two axes (TF IFFT2D).
ComplexTensor ifft2d(const ComplexTensor& x);
/// Real-input FFT along the last axis (TF RFFT): the input is cropped or
/// zero-padded to `fft_length` (default: its own length) and the
/// fft_length/2+1 non-negative-frequency bins are returned.
ComplexTensor rfft(const Tensor& x, std::int64_t fft_length = -1);
/// Inverse of rfft (TF IRFFT): the input's last axis is cropped or
/// zero-padded to fft_length/2+1 bins, treated as the half of a Hermitian
/// spectrum, and transformed to `fft_length` real samples (scaled by 1/n).
/// The default fft_length is 2 * (bins - 1).
Tensor irfft(const ComplexTensor& x, std::int64_t fft_length = -1);
/// Real-input 2-D FFT over the last two axes (TF RFFT2D); fft_length entries
/// of -1 keep the input size.
ComplexTensor rfft2d(const Tensor& x, std::array<std::int64_t, 2> fft_length = {-1, -1});
/// Inverse of rfft2d (TF IRFFT2D). A -1 entry defaults to the input rows and
/// to 2 * (bins - 1) columns.
Tensor irfft2d(const ComplexTensor& x, std::array<std::int64_t, 2> fft_length = {-1, -1});

/// Swaps the two halves of the last axis so frequency 0 is in the middle (numpy fftshift, 1-D).
Tensor fftshift(const Tensor& x);
/// Sample frequencies of an n-point DFT with sample spacing d (numpy fftfreq), float64.
Tensor fftfreq(std::int64_t n, double d = 1.0, DType dtype = DType::Float64);
/// Frequencies of the n/2+1 rfft bins (numpy rfftfreq), float64.
Tensor rfftfreq(std::int64_t n, double d = 1.0, DType dtype = DType::Float64);

// ---- convolution via FFT ----------------------------------------------------------------------

/// The smallest power of two >= n (FFT lengths for zero-padded convolution).
std::int64_t next_power_of_two(std::int64_t n);
/// Full linear convolution along the last axis, (a * b)[t] = sum_s a[s] b[t - s],
/// with output length La + Lb - 1. Leading axes broadcast. Computed with
/// zero-padded real FFTs; differentiable in both inputs.
Tensor fft_convolve(const Tensor& a, const Tensor& b);
/// Causal long convolution as used by S4 and Hyena: u [..., L], kernel
/// [..., Lk] (leading axes broadcast). y[t] = sum_{s <= t} k[s] u[t - s] for
/// t < L, i.e. the first L samples of the full convolution. Kernel taps
/// beyond L are ignored. Differentiable in both inputs.
Tensor fft_causal_conv(const Tensor& u, const Tensor& kernel);
/// Circular convolution of period n = last-axis length (both inputs must
/// have it): y[t] = sum_s a[s] b[(t - s) mod n].
Tensor fft_circular_convolve(const Tensor& a, const Tensor& b);

// ---- raw kernels (no autograd) ---------------------------------------------------------------

/// In-place unnormalized DFT of `n` complex values (any n >= 1), the same
/// kernel the tensor transforms use. Thread-safe.
void fft_inplace(std::complex<double>* data, std::int64_t n, bool inverse = false);

}  // namespace datamunge::nn
