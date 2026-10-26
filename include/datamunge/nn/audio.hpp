#pragma once

/// @file audio.hpp
/// @brief Audio features on `nn::Tensor` (tf.signal semantics) and WAV I/O.
///
/// Framing, windows, STFT / inverse STFT, mel filter banks, log-mel
/// spectrograms and MFCCs follow tf.signal exactly (argument names, defaults
/// and conventions), so ports of TF code line up value for value. All feature
/// functions are differentiable compositions of nn ops and the FFT
/// primitives in fft.hpp.
///
/// WAV: PCM 8/16/24/32-bit and IEEE float 32/64-bit files (incl.
/// WAVE_FORMAT_EXTENSIBLE) are read into float32 [frames, channels] in
/// [-1, 1]. Writing defaults to 16-bit PCM using TF EncodeWav's conversion
/// (round(x * 32768), clamped), and reading uses TF DecodeWav's (x / 32768).

#include <cstdint>
#include <string>
#include <vector>

#include "datamunge/nn/fft.hpp"
#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

// ---- framing ------------------------------------------------------------------------------------

/// Slides a window over `axis` (TF signal.frame): the axis of length n is
/// replaced by [num_frames, frame_length]. Without `pad_end`, num_frames =
/// 1 + (n - frame_length) / frame_step (0 if negative); with it, num_frames =
/// ceil(n / frame_step) and the end is padded with `pad_value`.
Tensor frame(const Tensor& signal, std::int64_t frame_length, std::int64_t frame_step, bool pad_end = false,
             double pad_value = 0.0, std::int64_t axis = -1);
/// Inverse of framing (TF signal.overlap_and_add): [..., frames, frame_length]
/// -> [..., (frames - 1) * frame_step + frame_length], summing overlaps.
Tensor overlap_and_add(const Tensor& frames, std::int64_t frame_step);

// ---- windows ------------------------------------------------------------------------------------

/// Hann window (TF signal.hann_window): 0.5 - 0.5 cos(2 pi n / N) with N =
/// length when `periodic` (the TF default, for spectral analysis) and length - 1
/// otherwise. A length-1 window is [1].
Tensor hann_window(std::int64_t length, bool periodic = true, DType dtype = DType::Float32);
/// Hamming window (TF signal.hamming_window): 0.54 - 0.46 cos(2 pi n / N).
Tensor hamming_window(std::int64_t length, bool periodic = true, DType dtype = DType::Float32);

// ---- STFT ---------------------------------------------------------------------------------------

/// Short-time Fourier transform along the last axis (TF signal.stft):
/// signals [..., samples] -> [..., frames, fft_length / 2 + 1]. `fft_length`
/// defaults (-1) to the smallest power of two >= frame_length; `window`
/// defaults (undefined) to a periodic Hann window of frame_length samples.
ComplexTensor stft(const Tensor& signals, std::int64_t frame_length, std::int64_t frame_step,
                   std::int64_t fft_length = -1, const Tensor& window = Tensor(), bool pad_end = false);
/// Inverse STFT (TF signal.inverse_stft): irfft each frame, keep frame_length
/// samples, multiply by `window` (default periodic Hann) and overlap-add.
/// For perfect reconstruction pass `inverse_stft_window(...)` as `window`.
Tensor inverse_stft(const ComplexTensor& stfts, std::int64_t frame_length, std::int64_t frame_step,
                    std::int64_t fft_length = -1, const Tensor& window = Tensor());
/// The synthesis window that makes inverse_stft invert stft (TF
/// signal.inverse_stft_window_fn): forward_window / sum of the squared
/// windows overlapping each sample. Default forward window: periodic Hann.
Tensor inverse_stft_window(std::int64_t frame_length, std::int64_t frame_step, const Tensor& forward_window = Tensor());
/// |stft|^2 (power spectrogram, no normalization).
Tensor power_spectrogram(const ComplexTensor& stfts);
/// |stft| (magnitude spectrogram; gradient 0 at exact zeros).
Tensor magnitude_spectrogram(const ComplexTensor& stfts);

// ---- mel features ---------------------------------------------------------------------------------

/// HTK mel scale used by tf.signal: 1127 ln(1 + hz / 700).
double hertz_to_mel(double hertz);
/// Inverse of hertz_to_mel.
double mel_to_hertz(double mel);
/// The [num_spectrogram_bins, num_mel_bins] matrix mapping a linear
/// spectrogram to mel bands (TF signal.linear_to_mel_weight_matrix, same
/// defaults): triangular filters with unit peak, centered on num_mel_bins
/// points evenly spaced in mel between the edges; the DC bin row is zero.
/// Computed in double precision, then converted to `dtype`.
Tensor linear_to_mel_weight_matrix(std::int64_t num_mel_bins = 20, std::int64_t num_spectrogram_bins = 129,
                                   double sample_rate = 8000.0, double lower_edge_hertz = 125.0,
                                   double upper_edge_hertz = 3800.0, DType dtype = DType::Float32);
/// spectrogram [..., bins] x weights [bins, mels] -> [..., mels] (TF
/// tensordot(spectrogram, weights, 1)).
Tensor mel_spectrogram(const Tensor& spectrogram, const Tensor& mel_weights);
/// log(max(spectrogram x weights, floor)): natural log with an explicit
/// floor, as in tflow's common/data/audio.h LogMel.
Tensor log_mel_spectrogram(const Tensor& spectrogram, const Tensor& mel_weights, double floor = 1e-6);
/// DCT along the last axis (TF signal.dct). type 2: y_k = 2 sum_n x_n
/// cos(pi k (2n + 1) / 2N); type 3 is its unnormalized inverse up to 2N.
/// `ortho` applies TF's norm="ortho" scaling (orthonormal for type 2 and 3).
Tensor dct(const Tensor& x, int type = 2, bool ortho = false);
/// TF signal.mfccs_from_log_mel_spectrograms: dct type 2 (unnormalized)
/// times rsqrt(2 * num_mel_bins). Keep the leading coefficients by slicing.
Tensor mfccs_from_log_mel_spectrograms(const Tensor& log_mel_spectrograms);

// ---- mu-law companding ---------------------------------------------------------------------------

/// WaveNet / TF mu-law quantization with mu = quantization_channels - 1:
/// x clamped to [-1, 1], y = sign(x) ln(1 + mu |x|) / ln(1 + mu), code =
/// floor((y + 1) mu / 2 + 0.5) in [0, mu]. Returns int32 codes, same shape.
/// Not differentiable (it quantizes). quantization_channels must be >= 2.
Tensor mu_law_encode(const Tensor& audio, std::int64_t quantization_channels = 256);
/// Inverse of mu_law_encode: y = 2 code / mu - 1, x = sign(y) ((1 + mu)^|y| - 1) / mu.
/// Integer codes give `dtype` samples in [-1, 1]; floating input is
/// differentiable (and keeps its own dtype).
Tensor mu_law_decode(const Tensor& codes, std::int64_t quantization_channels = 256, DType dtype = DType::Float32);

// ---- WAV ----------------------------------------------------------------------------------------

/// Decoded audio: float32 samples [frames, channels] in [-1, 1].
struct Wav {
  Tensor samples;
  std::int32_t sample_rate{0};
};

/// Sample encodings `encode_wav` can write.
enum class WavFormat { Pcm16, Pcm24, Pcm32, Pcm8, Float32 };

/// Parses a RIFF/WAVE byte buffer (TF DecodeWav, plus 8/24/32-bit PCM and
/// float). `desired_channels` > 0 keeps the first channels or repeats a mono
/// channel; `desired_samples` > 0 crops or zero-pads the frames. Throws
/// std::runtime_error on malformed input.
Wav decode_wav(const std::vector<unsigned char>& bytes, std::int64_t desired_channels = -1,
               std::int64_t desired_samples = -1);
/// Reads and decodes a WAV file.
Wav read_wav(const std::string& path, std::int64_t desired_channels = -1, std::int64_t desired_samples = -1);
/// Encodes samples [frames, channels] (or [frames] for mono; values clamped
/// to [-1, 1] for PCM) as a WAV byte buffer (TF EncodeWav for Pcm16).
std::vector<unsigned char> encode_wav(const Tensor& samples, std::int32_t sample_rate,
                                      WavFormat format = WavFormat::Pcm16);
/// Encodes and writes a WAV file.
void write_wav(const std::string& path, const Tensor& samples, std::int32_t sample_rate,
               WavFormat format = WavFormat::Pcm16);

}  // namespace datamunge::nn
