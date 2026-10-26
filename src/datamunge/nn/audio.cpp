// Audio features with tf.signal semantics (framing, windows, STFT, mel,
// MFCC) composed from differentiable nn ops, plus WAV encode/decode.

#include "datamunge/nn/audio.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "detail.hpp"

namespace datamunge::nn {

using I64 = std::int64_t;

namespace {

constexpr double kPi = 3.14159265358979323846;

Tensor default_window(const Tensor& window, I64 frame_length, DType dtype) {
  return window.defined() ? window : hann_window(frame_length, true, dtype);
}

Tensor cosine_window(I64 length, bool periodic, double a, double b, DType dtype) {
  if (length < 0) throw std::invalid_argument("window: length must be >= 0");
  if (length == 1) return ones({1}, dtype);
  std::vector<double> w(static_cast<std::size_t>(length));
  const double        denom = periodic ? double(length) : double(length - 1);
  for (I64 n = 0; n < length; ++n) w[n] = a - b * std::cos(2.0 * kPi * double(n) / denom);
  return from_vector(w).to(dtype);
}

}  // namespace

// ---- framing ------------------------------------------------------------------------------------

Tensor frame(const Tensor& signal, std::int64_t frame_length, std::int64_t frame_step, bool pad_end,
             double pad_value, std::int64_t axis) {
  if (frame_length < 1 || frame_step < 1) throw std::invalid_argument("frame: frame_length and frame_step must be >= 1");
  if (signal.dim() < 1) throw std::invalid_argument("frame: needs rank >= 1");
  const I64 a = detail::normalize_axis(axis, signal.dim(), "frame");
  const I64 n = signal.size(a);
  I64       frames;
  Tensor    x = signal;
  if (pad_end) {
    frames             = (n + frame_step - 1) / frame_step;
    const I64 required = frame_length + frame_step * (frames - 1);
    if (required > n) {
      std::vector<std::pair<I64, I64>> p(static_cast<std::size_t>(signal.dim()), {0, 0});
      p[static_cast<std::size_t>(a)] = {0, required - n};
      x                              = pad(signal, p, PadMode::Constant, pad_value);
    }
  } else {
    frames = n < frame_length ? 0 : 1 + (n - frame_length) / frame_step;
  }
  std::vector<std::int64_t> idx(static_cast<std::size_t>(frames * frame_length));
  for (I64 f = 0; f < frames; ++f)
    for (I64 j = 0; j < frame_length; ++j) idx[f * frame_length + j] = f * frame_step + j;
  return gather(x, from_vector(idx, {frames, frame_length}), a);
}

Tensor overlap_and_add(const Tensor& frames, std::int64_t frame_step) {
  if (frames.dim() < 2) throw std::invalid_argument("overlap_and_add: needs [..., frames, frame_length]");
  if (frame_step < 1) throw std::invalid_argument("overlap_and_add: frame_step must be >= 1");
  const I64 f = frames.size(-2), l = frames.size(-1);
  const I64 out = f == 0 ? 0 : (f - 1) * frame_step + l;
  Shape     lead(frames.shape().begin(), frames.shape().end() - 2);
  Shape     flat = lead, result = lead;
  flat.push_back(f * l);
  result.push_back(out);
  std::vector<std::int64_t> idx(static_cast<std::size_t>(f * l));
  for (I64 i = 0; i < f; ++i)
    for (I64 j = 0; j < l; ++j) idx[i * l + j] = i * frame_step + j;
  const Tensor base = zeros(result, to_floating(frames.dtype()));
  if (f * l == 0) return base;
  return index_add(base, static_cast<I64>(lead.size()), from_vector(idx), reshape(frames, flat));
}

// ---- windows ------------------------------------------------------------------------------------

Tensor hann_window(std::int64_t length, bool periodic, DType dtype) {
  return cosine_window(length, periodic, 0.5, 0.5, dtype);
}

Tensor hamming_window(std::int64_t length, bool periodic, DType dtype) {
  return cosine_window(length, periodic, 0.54, 0.46, dtype);
}

// ---- STFT ---------------------------------------------------------------------------------------

ComplexTensor stft(const Tensor& signals, std::int64_t frame_length, std::int64_t frame_step,
                   std::int64_t fft_length, const Tensor& window, bool pad_end) {
  const Tensor x = is_floating(signals.dtype()) ? signals : cast(signals, DType::Float32);
  const I64    n = fft_length < 0 ? next_power_of_two(frame_length) : fft_length;
  if (n < frame_length) throw std::invalid_argument("stft: fft_length must be >= frame_length");
  const Tensor framed = frame(x, frame_length, frame_step, pad_end);
  const Tensor w      = default_window(window, frame_length, x.dtype());
  return rfft(framed * w.to(x.dtype()), n);
}

Tensor inverse_stft(const ComplexTensor& stfts, std::int64_t frame_length, std::int64_t frame_step,
                    std::int64_t fft_length, const Tensor& window) {
  if (stfts.dim() < 2) throw std::invalid_argument("inverse_stft: needs [..., frames, bins]");
  const I64 n      = fft_length < 0 ? next_power_of_two(frame_length) : fft_length;
  Tensor    frames = irfft(stfts, n);
  if (frame_length < n) {
    frames = slice_axis(frames, -1, 0, frame_length);
  } else if (frame_length > n) {
    std::vector<std::pair<I64, I64>> p(static_cast<std::size_t>(frames.dim()), {0, 0});
    p.back() = {0, frame_length - n};
    frames   = pad(frames, p);
  }
  const Tensor w = default_window(window, frame_length, frames.dtype());
  return overlap_and_add(frames * w.to(frames.dtype()), frame_step);
}

Tensor inverse_stft_window(std::int64_t frame_length, std::int64_t frame_step, const Tensor& forward_window) {
  if (frame_length < 1 || frame_step < 1) throw std::invalid_argument("inverse_stft_window: bad lengths");
  const Tensor              fw = default_window(forward_window, frame_length, DType::Float32);
  const std::vector<double> w  = fw.to_vector<double>();
  if (static_cast<I64>(w.size()) != frame_length) throw std::invalid_argument("inverse_stft_window: window length");
  std::vector<double> denom(static_cast<std::size_t>(frame_step), 0.0);
  for (I64 i = 0; i < frame_length; ++i) denom[i % frame_step] += w[i] * w[i];
  std::vector<double> out(w.size());
  for (I64 i = 0; i < frame_length; ++i) out[i] = w[i] / denom[i % frame_step];
  return from_vector(out).to(fw.dtype());
}

Tensor mu_law_encode(const Tensor& audio, std::int64_t quantization_channels) {
  if (quantization_channels < 2) throw std::invalid_argument("mu_law_encode: quantization_channels must be >= 2");
  const double              mu = static_cast<double>(quantization_channels - 1);
  const std::vector<double> x  = audio.to_vector<double>();
  std::vector<std::int32_t> codes(x.size());
  const double              scale = 1.0 / std::log1p(mu);
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (std::isnan(x[i])) throw std::invalid_argument("mu_law_encode: NaN sample");
    const double v = std::clamp(x[i], -1.0, 1.0);
    const double y = std::copysign(std::log1p(mu * std::abs(v)) * scale, v);
    codes[i] = static_cast<std::int32_t>(std::clamp(std::floor((y + 1) * mu / 2 + 0.5), 0.0, mu));
  }
  return from_vector(codes, audio.shape());
}

Tensor mu_law_decode(const Tensor& codes, std::int64_t quantization_channels, DType dtype) {
  if (quantization_channels < 2) throw std::invalid_argument("mu_law_decode: quantization_channels must be >= 2");
  const double mu  = static_cast<double>(quantization_channels - 1);
  const DType  out = is_floating(codes.dtype()) ? codes.dtype() : dtype;
  if (!is_floating(out)) throw std::invalid_argument("mu_law_decode: dtype must be floating point");
  const Tensor y = sub(mul(cast(codes, DType::Float64), 2.0 / mu), 1.0);
  const Tensor x = mul(mul(sign(y), expm1(mul(abs(y), std::log1p(mu)))), 1.0 / mu);
  return cast(x, out);
}

Tensor power_spectrogram(const ComplexTensor& stfts) { return complex_abs_squared(stfts); }
Tensor magnitude_spectrogram(const ComplexTensor& stfts) { return complex_abs(stfts); }

// ---- mel features ---------------------------------------------------------------------------------

double hertz_to_mel(double hertz) { return 1127.0 * std::log1p(hertz / 700.0); }
double mel_to_hertz(double mel) { return 700.0 * std::expm1(mel / 1127.0); }

Tensor linear_to_mel_weight_matrix(std::int64_t num_mel_bins, std::int64_t num_spectrogram_bins, double sample_rate,
                                   double lower_edge_hertz, double upper_edge_hertz, DType dtype) {
  if (num_mel_bins < 1) throw std::invalid_argument("linear_to_mel_weight_matrix: num_mel_bins must be positive");
  if (num_spectrogram_bins < 1)
    throw std::invalid_argument("linear_to_mel_weight_matrix: num_spectrogram_bins must be positive");
  if (!(sample_rate > 0)) throw std::invalid_argument("linear_to_mel_weight_matrix: sample_rate must be positive");
  if (!(lower_edge_hertz >= 0)) throw std::invalid_argument("linear_to_mel_weight_matrix: lower_edge_hertz must be >= 0");
  if (!(lower_edge_hertz < upper_edge_hertz))
    throw std::invalid_argument("linear_to_mel_weight_matrix: lower_edge_hertz must be < upper_edge_hertz");
  if (upper_edge_hertz > sample_rate / 2)
    throw std::invalid_argument("linear_to_mel_weight_matrix: upper_edge_hertz must not exceed the Nyquist frequency");
  const double        nyquist = sample_rate / 2.0;
  const double        lo = hertz_to_mel(lower_edge_hertz), hi = hertz_to_mel(upper_edge_hertz);
  const I64           edges = num_mel_bins + 2;
  std::vector<double> band(static_cast<std::size_t>(edges));
  for (I64 i = 0; i < edges; ++i) band[i] = lo + (hi - lo) * double(i) / double(edges - 1);
  std::vector<double> w(static_cast<std::size_t>(num_spectrogram_bins * num_mel_bins), 0.0);
  // Row 0 (DC) stays zero, as in TF (bands_to_zero = 1).
  for (I64 bin = 1; bin < num_spectrogram_bins; ++bin) {
    const double hz  = num_spectrogram_bins == 1 ? 0.0 : nyquist * double(bin) / double(num_spectrogram_bins - 1);
    const double mel = hertz_to_mel(hz);
    for (I64 m = 0; m < num_mel_bins; ++m) {
      const double lower = (mel - band[m]) / (band[m + 1] - band[m]);
      const double upper = (band[m + 2] - mel) / (band[m + 2] - band[m + 1]);
      w[bin * num_mel_bins + m] = std::max(0.0, std::min(lower, upper));
    }
  }
  return from_vector(w, {num_spectrogram_bins, num_mel_bins}).to(dtype);
}

Tensor mel_spectrogram(const Tensor& spectrogram, const Tensor& mel_weights) {
  if (spectrogram.dim() < 1 || mel_weights.dim() != 2 || spectrogram.size(-1) != mel_weights.size(0))
    throw std::invalid_argument("mel_spectrogram: need spectrogram [..., bins] and weights [bins, mels]");
  const I64 bins = mel_weights.size(0), mels = mel_weights.size(1);
  const I64 rows = spectrogram.numel() / std::max<I64>(bins, 1);
  Shape     out  = spectrogram.shape();
  out.back()     = mels;
  const Tensor w = mel_weights.to(to_floating(spectrogram.dtype()));
  if (rows == 0) return zeros(out, w.dtype());
  return reshape(matmul(reshape(spectrogram, {rows, bins}), w), out);
}

Tensor log_mel_spectrogram(const Tensor& spectrogram, const Tensor& mel_weights, double floor) {
  if (!(floor > 0) || !std::isfinite(floor)) throw std::invalid_argument("log_mel_spectrogram: floor must be positive");
  const Tensor mel = mel_spectrogram(spectrogram, mel_weights);
  return log(maximum(mel, scalar_like(floor, mel)));
}

Tensor dct(const Tensor& x_in, int type, bool ortho) {
  if (type != 2 && type != 3) throw std::invalid_argument("dct: only types 2 and 3 are supported");
  if (x_in.dim() < 1) throw std::invalid_argument("dct: needs rank >= 1");
  const Tensor x = is_floating(x_in.dtype()) ? x_in : cast(x_in, DType::Float32);
  const I64    n = x.size(-1);
  if (n < 1) throw std::invalid_argument("dct: empty last axis");
  // y = x @ D with D[i, k] the contribution of input i to output k.
  std::vector<double> d(static_cast<std::size_t>(n * n));
  for (I64 i = 0; i < n; ++i)
    for (I64 k = 0; k < n; ++k) {
      if (type == 2) {
        // input sample i, output coefficient k
        double s = 2.0 * std::cos(kPi * double(k) * double(2 * i + 1) / double(2 * n));
        if (ortho) s *= k == 0 ? 0.5 / std::sqrt(double(n)) : 0.5 * std::sqrt(2.0 / double(n));
        d[i * n + k] = s;
      } else {
        // input coefficient i, output sample k
        double c = std::cos(kPi * double(i) * double(2 * k + 1) / double(2 * n));
        if (ortho) c *= i == 0 ? 1.0 / std::sqrt(double(n)) : std::sqrt(2.0 / double(n));
        else c *= i == 0 ? 1.0 : 2.0;
        d[i * n + k] = c;
      }
    }
  const Tensor dm = from_vector(d, {n, n}).to(x.dtype());
  return mel_spectrogram(x, dm);
}

Tensor mfccs_from_log_mel_spectrograms(const Tensor& log_mel_spectrograms) {
  const I64 n = log_mel_spectrograms.size(-1);
  return dct(log_mel_spectrograms, 2, false) * (1.0 / std::sqrt(2.0 * double(n)));
}

// ---- WAV ----------------------------------------------------------------------------------------

namespace {

std::uint32_t le32(const unsigned char* p) {
  return std::uint32_t(p[0]) | std::uint32_t(p[1]) << 8 | std::uint32_t(p[2]) << 16 | std::uint32_t(p[3]) << 24;
}
std::uint16_t le16(const unsigned char* p) { return std::uint16_t(p[0] | p[1] << 8); }

void put32(std::vector<unsigned char>& b, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) b.push_back(static_cast<unsigned char>((v >> (8 * i)) & 255));
}
void put16(std::vector<unsigned char>& b, std::uint16_t v) {
  b.push_back(static_cast<unsigned char>(v & 255));
  b.push_back(static_cast<unsigned char>(v >> 8));
}

[[noreturn]] void wav_error(const std::string& what) { throw std::runtime_error("WAV decode failed: " + what); }

}  // namespace

Wav decode_wav(const std::vector<unsigned char>& bytes, std::int64_t desired_channels, std::int64_t desired_samples) {
  const std::size_t size = bytes.size();
  if (size < 12) wav_error("data too short for a RIFF header");
  const unsigned char* b = bytes.data();
  if (std::memcmp(b, "RIFF", 4) != 0 || std::memcmp(b + 8, "WAVE", 4) != 0) wav_error("not a RIFF/WAVE file");
  int           format = -1, channels = 0, bits = 0;
  std::uint32_t rate = 0, block_align = 0;
  const unsigned char* data = nullptr;
  std::size_t          data_size = 0;
  std::size_t          pos = 12;
  while (pos + 8 <= size) {
    const std::uint32_t chunk = le32(b + pos + 4);
    const unsigned char* body = b + pos + 8;
    const std::size_t    avail = size - (pos + 8);
    if (std::memcmp(b + pos, "fmt ", 4) == 0) {
      if (chunk < 16 || chunk > avail) wav_error("bad fmt chunk");
      format      = le16(body);
      channels    = le16(body + 2);
      rate        = le32(body + 4);
      block_align = le16(body + 12);
      bits        = le16(body + 14);
      if (format == 0xFFFE) {
        if (chunk < 40) wav_error("bad WAVE_FORMAT_EXTENSIBLE chunk");
        format = le16(body + 24);  // first two bytes of the subformat GUID
      }
    } else if (std::memcmp(b + pos, "data", 4) == 0) {
      data      = body;
      data_size = std::min<std::size_t>(chunk, avail);  // tolerate truncated streaming headers
      break;
    }
    if (chunk > avail) wav_error("chunk exceeds file size");
    pos += 8 + chunk + (chunk & 1);
  }
  if (format < 0) wav_error("missing fmt chunk");
  if (!data) wav_error("missing data chunk");
  if (channels < 1) wav_error("no channels");
  if (rate == 0) wav_error("zero sample rate");
  const bool pcm = format == 1, fl = format == 3;
  if (!(pcm && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) && !(fl && (bits == 32 || bits == 64)))
    wav_error("unsupported format " + std::to_string(format) + " with " + std::to_string(bits) + " bits");
  const std::size_t bps = static_cast<std::size_t>(bits / 8);
  if (block_align != bps * static_cast<std::size_t>(channels)) wav_error("inconsistent block alignment");
  const I64 frames = static_cast<I64>(data_size / block_align);
  Tensor    out    = empty({frames, channels}, DType::Float32);
  float*    dst    = out.data_ptr<float>();
  for (I64 i = 0; i < frames * channels; ++i) {
    const unsigned char* p = data + static_cast<std::size_t>(i) * bps;
    float                v = 0;
    if (pcm) {
      switch (bits) {
        case 8: v = float(int(p[0]) - 128) / 128.0f; break;
        case 16: v = float(std::int16_t(le16(p))) * (1.0f / 32768.0f); break;
        case 24: {
          std::int32_t s = std::int32_t(p[0] | p[1] << 8 | p[2] << 16);
          if (s & 0x800000) s -= 0x1000000;
          v = float(double(s) / 8388608.0);
          break;
        }
        default: v = float(double(std::int32_t(le32(p))) / 2147483648.0);
      }
    } else if (bits == 32) {
      const std::uint32_t u = le32(p);
      std::memcpy(&v, &u, 4);
    } else {
      std::uint64_t u = std::uint64_t(le32(p)) | std::uint64_t(le32(p + 4)) << 32;
      double        d;
      std::memcpy(&d, &u, 8);
      v = float(d);
    }
    dst[i] = v;
  }
  if (desired_channels > 0 && desired_channels != channels) {
    if (channels == 1) out = tile(out, {1, desired_channels}).contiguous();
    else if (desired_channels < channels) out = slice_axis(out, 1, 0, desired_channels).contiguous();
    else wav_error("cannot expand " + std::to_string(channels) + " channels to " + std::to_string(desired_channels));
  }
  if (desired_samples > 0 && desired_samples != frames) {
    if (desired_samples < frames) out = slice_axis(out, 0, 0, desired_samples).contiguous();
    else out = pad(out, {{0, desired_samples - frames}, {0, 0}});
  }
  return {out, static_cast<std::int32_t>(rate)};
}

Wav read_wav(const std::string& path, std::int64_t desired_channels, std::int64_t desired_samples) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("read_wav: cannot open " + path);
  std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return decode_wav(bytes, desired_channels, desired_samples);
}

std::vector<unsigned char> encode_wav(const Tensor& samples, std::int32_t sample_rate, WavFormat format) {
  if (sample_rate <= 0) throw std::invalid_argument("encode_wav: sample_rate must be positive");
  if (samples.dim() != 1 && samples.dim() != 2) throw std::invalid_argument("encode_wav: samples must be [frames, channels]");
  const I64 frames   = samples.size(0);
  const I64 channels = samples.dim() == 2 ? samples.size(1) : 1;
  if (channels < 1 || channels > 65535) throw std::invalid_argument("encode_wav: bad channel count");
  int bits = 16, tag = 1;
  switch (format) {
    case WavFormat::Pcm8: bits = 8; break;
    case WavFormat::Pcm16: bits = 16; break;
    case WavFormat::Pcm24: bits = 24; break;
    case WavFormat::Pcm32: bits = 32; break;
    case WavFormat::Float32: bits = 32, tag = 3; break;
  }
  const std::uint64_t bps = std::uint64_t(bits / 8), align = bps * std::uint64_t(channels);
  const std::uint64_t data_size = align * std::uint64_t(frames);
  if (data_size + 36 > 0xFFFFFFFFull) throw std::invalid_argument("encode_wav: data too large for a WAV file");
  std::vector<unsigned char> b;
  b.reserve(static_cast<std::size_t>(44 + data_size));
  b.insert(b.end(), {'R', 'I', 'F', 'F'});
  put32(b, static_cast<std::uint32_t>(36 + data_size));
  b.insert(b.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
  put32(b, 16);
  put16(b, static_cast<std::uint16_t>(tag));
  put16(b, static_cast<std::uint16_t>(channels));
  put32(b, static_cast<std::uint32_t>(sample_rate));
  put32(b, static_cast<std::uint32_t>(std::uint64_t(sample_rate) * align));
  put16(b, static_cast<std::uint16_t>(align));
  put16(b, static_cast<std::uint16_t>(bits));
  b.insert(b.end(), {'d', 'a', 't', 'a'});
  put32(b, static_cast<std::uint32_t>(data_size));
  const std::vector<float> v = samples.to_vector<float>();
  for (float x : v) {
    if (format == WavFormat::Float32) {
      std::uint32_t u;
      std::memcpy(&u, &x, 4);
      put32(b, u);
      continue;
    }
    // TF EncodeWav: round(x * 2^(bits-1)) clamped to the integer range.
    const double scale = std::ldexp(1.0, bits - 1);
    double       s     = std::round(double(x) * scale);
    if (bits == 16) s = std::round(x * 32768.0f);  // float arithmetic, as TF does
    s                 = std::min(std::max(s, -scale), scale - 1);
    const auto i      = static_cast<std::int64_t>(s);
    switch (bits) {
      case 8: b.push_back(static_cast<unsigned char>(i + 128)); break;
      case 16: put16(b, static_cast<std::uint16_t>(static_cast<std::int16_t>(i))); break;
      case 24:
        for (int k = 0; k < 3; ++k) b.push_back(static_cast<unsigned char>((std::uint32_t(i) >> (8 * k)) & 255));
        break;
      default: put32(b, static_cast<std::uint32_t>(static_cast<std::int32_t>(i)));
    }
  }
  return b;
}

void write_wav(const std::string& path, const Tensor& samples, std::int32_t sample_rate, WavFormat format) {
  const auto    bytes = encode_wav(samples, sample_rate, format);
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("write_wav: cannot open " + path);
  out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!out) throw std::runtime_error("write_wav: write failed for " + path);
}

}  // namespace datamunge::nn
