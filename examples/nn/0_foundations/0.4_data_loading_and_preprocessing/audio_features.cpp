// CMD: just run-nn audio_features
// PCM16 WAV decoding, STFT power and log-mel spectrograms with datamunge::nn.
//
// Port of tflow's audio_features. Encode a stereo 1 kHz signal at 8 kHz, load
// it back through the WAV decoder, verify PCM quantization, and average
// channels. Compute a periodic-Hann STFT (256-sample frames, 128-sample hop),
// compare every first-frame complex bin to a direct DFT, verify the 1 kHz peak
// and Parseval energy, and test FFT zero-padding.
//
// Apply 20 HTK mel triangles and a natural-log floor. Check an analytic mel
// triangle, manual filter-bank multiplication, silence, short signals, and
// corrupt WAV rejection. There is no resampling. Optional file=PATH processes
// another WAV at its own rate.
// Reference: https://www.tensorflow.org/api_docs/python/tf/signal/linear_to_mel_weight_matrix
//
// Differences from tflow: TF's EncodeWav/DecodeWav ops become nn::encode_wav /
// nn::read_wav (same int16 conversion, so the PCM error is identical); the
// hand-built Gather/Pad/RFFT graph of tflow's Stft becomes nn::stft with
// fft_length and a periodic Hann window (identical framing: valid frames, no
// centering or end padding); MelWeights becomes nn::linear_to_mel_weight_matrix
// (the same tf.signal formula). The fixture is written under output/nn.

#include <algorithm>
#include <cmath>
#include <complex>
#include <datamunge/nn/audio.hpp>
#include <datamunge/nn/fft.hpp>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/paths.hpp"

namespace nn = datamunge::nn;
namespace fs = std::filesystem;
using nn::Tensor;

namespace {

constexpr double Pi = 3.14159265358979323846;

struct Spectrogram {
  nn::ComplexTensor stft;
  Tensor            power;
};

/// tflow's Stft: valid frames, periodic Hann, unnormalized real FFT, |X|^2.
Spectrogram Stft(const Tensor& mono, int frame_length, int frame_step, int fft_length) {
  if (frame_length < 2 || frame_step <= 0 || fft_length < frame_length) throw std::invalid_argument("invalid STFT lengths");
  auto transform = nn::stft(mono, frame_length, frame_step, fft_length);
  return {transform, nn::power_spectrogram(transform)};
}

/// tflow's MelWeights: bands spaced over [low_hz, high_hz]; even FFT required.
Tensor MelWeights(int sample_rate, int fft_length, int bands, double low_hz, double high_hz) {
  if (fft_length < 2 || fft_length % 2) throw std::invalid_argument("invalid mel filter configuration (even FFT required)");
  return nn::linear_to_mel_weight_matrix(bands, fft_length / 2 + 1, sample_rate, low_hz, high_hz);
}

Tensor Mono(const Tensor& samples) { return nn::mean(samples, {1}); }

nn::Wav Fixture() {
  Tensor original = nn::empty({2048, 2});
  float* o        = original.data_ptr<float>();
  for (int i = 0; i < 2048; ++i) {
    const double sine = std::sin(2 * Pi * 1000 * i / 8000);
    o[2 * i]          = float(.5 * sine);
    o[2 * i + 1]      = float(.25 * sine);
  }
  const fs::path dir  = paths::OutputDir("fixtures/audio");
  const fs::path path = dir / "tone.wav";
  nn::write_wav(path.string(), original, 8000);
  const auto loaded = nn::read_wav(path.string());
  check::Require(loaded.sample_rate == 8000 && loaded.samples.shape() == original.shape(), "WAV rate/channel preservation");
  const auto a = original.to_vector<float>(), b = loaded.samples.to_vector<float>();
  double     worst = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double error = std::abs(double(a[i]) - double(b[i]));
    worst              = std::max(worst, error);
    check::Near(error, 0, 1. / 32768, "PCM16 quantization bound");
  }
  const auto corrupt = dir / "bad.wav";
  {
    std::ofstream out(corrupt, std::ios::binary | std::ios::trunc);
    out << "bad";
  }
  bool rejected = false;
  try {
    nn::read_wav(corrupt.string());
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  check::Require(rejected, "corrupt WAV must fail");
  std::cout << "  WAV: 2048 samples, 2 channels, 8000 Hz; max PCM error=" << worst << '\n';
  return loaded;
}

void SpectralChecks(const nn::Wav& wave) {
  const Tensor mono     = Mono(wave.samples);
  const auto   spectrum = Stft(mono, 256, 128, 256), padded = Stft(mono, 256, 128, 512);
  const Tensor weights  = MelWeights(wave.sample_rate, 256, 20, 0, wave.sample_rate / 2.);
  const Tensor log_mel  = nn::log_mel_spectrogram(spectrum.power, weights);
  check::Require(spectrum.stft.shape() == nn::Shape{15, 129} && log_mel.shape() == nn::Shape{15, 20},
                 "valid-frame STFT/mel dimensions");
  const auto window = nn::hann_window(256).to_vector<float>();
  const auto m = mono.to_vector<float>(), s = wave.samples.to_vector<float>();
  const auto re = spectrum.stft.real.to_vector<float>(), im = spectrum.stft.imag.to_vector<float>();
  const auto power = spectrum.power.to_vector<float>(), padded_power = padded.power.to_vector<float>();
  const auto mel = log_mel.to_vector<float>(), w = weights.to_vector<float>();
  double     worst = 0, window_energy = 0, spectral_energy = 0;
  int        peak = 0, padded_peak = 0;
  for (int i = 0; i < 256; ++i) {
    check::Near(m[i], .5f * (s[2 * i] + s[2 * i + 1]), 0, "channel mean");
    const double value = m[i] * window[i];
    window_energy += value * value;
  }
  for (int bin = 0; bin <= 128; ++bin) {
    std::complex<double> reference{0, 0};
    for (int i = 0; i < 256; ++i) {
      const double value = m[i] * window[i], angle = -2 * Pi * bin * i / 256;
      reference += value * std::complex<double>(std::cos(angle), std::sin(angle));
    }
    const double error = std::abs(std::complex<double>(re[bin], im[bin]) - reference);
    worst              = std::max(worst, error);
    check::Near(error, 0, 5e-6, "FFT vs independent direct DFT");
    spectral_energy += power[bin] * (bin == 0 || bin == 128 ? 1 : 2) / 256;
    if (power[bin] > power[peak]) peak = bin;
  }
  for (int bin = 0; bin <= 256; ++bin)
    if (padded_power[bin] > padded_power[padded_peak]) padded_peak = bin;
  check::Require(peak == 32 && padded_peak == 64, "FFT zero-padding preserves 1000 Hz peak");
  check::Near(std::sqrt(power[peak]), 24, .01, "periodic Hann amplitude A*N/4");
  check::Near(spectral_energy, window_energy, 5e-6, "one-sided Parseval weighting");
  for (int frame = 0; frame < 15; ++frame)
    for (int band = 0; band < 20; ++band) {
      double p = 0;
      for (int bin = 0; bin <= 128; ++bin) p += power[frame * 129 + bin] * w[bin * 20 + band];
      check::Near(mel[frame * 20 + band], std::log(std::max(p, 1e-6)), 2e-6, "manual mel multiplication and log floor");
    }
  // Center of a one-band filter from 0 to 2100 Hz is exactly 700 Hz in mel.
  const Tensor analytic = MelWeights(4200, 6, 1, 0, 2100);
  check::Near(analytic.at({1, 0}), 1, 1e-7, "analytic mel center");
  check::Near(analytic.at({0, 0}), 0, 0, "mel DC excluded");
  check::Near(analytic.at({3, 0}), 0, 1e-7, "mel high edge");
  const Tensor silence       = nn::zeros({256, 2});
  const auto   silent        = Stft(Mono(silence), 256, 128, 256);
  const auto   silent_power  = silent.power.to_vector<float>();
  const auto   silent_logmel = nn::log_mel_spectrogram(silent.power, weights).to_vector<float>();
  for (float v : silent_power) check::Near(v, 0, 0, "silence power");
  for (float v : silent_logmel) check::Near(v, std::log(1e-6), 1e-6, "finite silence log floor");
  const auto short_power = Stft(Mono(nn::zeros({10, 2})), 256, 128, 256).power;
  check::Require(short_power.shape() == nn::Shape{0, 129}, "short signals yield no valid frames");
  std::cout << "  STFT [15,129], log-mel [15,20]; direct DFT max error=" << worst << '\n'
            << "  peak=1000 Hz (bin 32, zero-padded bin 64), Hann amplitude=24\n"
            << "  window energy=" << window_energy << ", Parseval energy=" << spectral_energy << '\n'
            << "  mel triangles, silence floor and short-signal checks passed\n";
}

void Inspect(const fs::path& path) {
  const auto   wave     = nn::read_wav(path.string());
  const auto   spectrum = Stft(Mono(wave.samples), 256, 128, 256);
  const Tensor mel = nn::log_mel_spectrogram(spectrum.power, MelWeights(wave.sample_rate, 256, 20, 0, wave.sample_rate / 2.));
  std::cout << "  requested WAV: samples=" << nn::shape_str(wave.samples.shape()) << ", rate=" << wave.sample_rate
            << ", log-mel=" << nn::shape_str(mel.shape()) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    fs::path path;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("file=", 0) != 0 || arg.size() == 5) throw std::invalid_argument("expected file=PATH");
      path = arg.substr(5);
    }
    std::cout << "== WAV loading and spectral preprocessing\n";
    const auto wave = Fixture();
    SpectralChecks(wave);
    if (!path.empty()) Inspect(path);
    std::cout << "All audio feature checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
