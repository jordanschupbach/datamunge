# Native Python NN example. PCM16 WAV decoding, STFT power and log-mel spectrograms with datamunge::nn.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/0_foundations/0.4_data_loading_and_preprocessing/audio_features.py
"""PCM16 WAV decoding, STFT power and log-mel spectrograms with datamunge::nn.

Port of tflow's audio_features. Encode a stereo 1 kHz signal at 8 kHz, load
it back through the WAV decoder, verify PCM quantization, and average
channels. Compute a periodic-Hann STFT (256-sample frames, 128-sample hop),
compare every first-frame complex bin to a direct DFT, verify the 1 kHz peak
and Parseval energy, and test FFT zero-padding.

Apply 20 HTK mel triangles and a natural-log floor. Check an analytic mel
triangle, manual filter-bank multiplication, silence, short signals, and
corrupt WAV rejection. There is no resampling. Optional file=PATH processes
another WAV at its own rate.
Reference: https://www.tensorflow.org/api_docs/python/tf/signal/linear_to_mel_weight_matrix

Differences from tflow: TF's EncodeWav/DecodeWav ops become nn.write_wav /
nn.read_wav (same int16 conversion, so the PCM error is identical); the
hand-built Gather/Pad/RFFT graph of tflow's Stft becomes nn.stft with
fft_length and a periodic Hann window (identical framing: valid frames, no
centering or end padding); MelWeights becomes nn.linear_to_mel_weight_matrix
(the same tf.signal formula). The fixture is written under
output/nn/python/<mode>/fixtures/audio. Host-side checks round float
products to float32 exactly where the C++ multiplies two floats.
"""
import math
import struct
import sys

from pydatamunge import nn
from pydatamunge.nn_workflows import helper

checks = helper("checks")

PI = 3.14159265358979323846


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def stft(mono, frame_length, frame_step, fft_length):
    """tflow's Stft: valid frames, periodic Hann, unnormalized real FFT, |X|^2."""
    if frame_length < 2 or frame_step <= 0 or fft_length < frame_length:
        raise ValueError("invalid STFT lengths")
    transform = nn.stft(mono, frame_length, frame_step, fft_length)
    return transform, nn.power_spectrogram(transform)


def mel_weights(sample_rate, fft_length, bands, low_hz, high_hz):
    """tflow's MelWeights: bands spaced over [low_hz, high_hz]; even FFT required."""
    if fft_length < 2 or fft_length % 2:
        raise ValueError("invalid mel filter configuration (even FFT required)")
    return nn.linear_to_mel_weight_matrix(bands, fft_length // 2 + 1, sample_rate, low_hz, high_hz)


def mono_of(samples):
    return nn.mean(samples, [1])


def fixture():
    values = []
    for i in range(2048):
        sine = math.sin(2 * PI * 1000 * i / 8000)
        values += [0.5 * sine, 0.25 * sine]
    original = nn.from_vector(values, [2048, 2])
    directory = checks.output_dir("fixtures/audio")
    path = directory / "tone.wav"
    nn.write_wav(str(path), original, 8000)
    loaded = nn.read_wav(str(path))
    checks.require(
        loaded.sample_rate == 8000 and tuple(loaded.samples.shape()) == tuple(original.shape()),
        "WAV rate/channel preservation",
    )
    a, b = original.values(), loaded.samples.values()
    worst = 0.0
    for x, y in zip(a, b):
        error = abs(x - y)
        worst = max(worst, error)
        checks.near(error, 0, 1.0 / 32768, "PCM16 quantization bound")
    corrupt = directory / "bad.wav"
    corrupt.write_bytes(b"bad")
    rejected = False
    try:
        nn.read_wav(str(corrupt))
    except RuntimeError:
        rejected = True
    checks.require(rejected, "corrupt WAV must fail")
    print(f"  WAV: 2048 samples, 2 channels, 8000 Hz; max PCM error={'%g' % worst}")
    return loaded


def spectral_checks(wave):
    mono = mono_of(wave.samples)
    spectrum, power_t = stft(mono, 256, 128, 256)
    _, padded_t = stft(mono, 256, 128, 512)
    weights = mel_weights(wave.sample_rate, 256, 20, 0, wave.sample_rate / 2.0)
    log_mel = nn.log_mel_spectrogram(power_t, weights)
    checks.require(
        tuple(spectrum.shape()) == (15, 129) and tuple(log_mel.shape()) == (15, 20), "valid-frame STFT/mel dimensions"
    )
    window = nn.hann_window(256).values()
    m, s = mono.values(), wave.samples.values()
    re, im = spectrum.real.values(), spectrum.imag.values()
    power, padded_power = power_t.values(), padded_t.values()
    mel, w = log_mel.values(), weights.values()
    worst = window_energy = spectral_energy = 0.0
    peak = padded_peak = 0
    for i in range(256):
        checks.near(m[i], f32(0.5 * f32(s[2 * i] + s[2 * i + 1])), 0, "channel mean")
        value = f32(m[i] * window[i])
        window_energy += value * value
    for frequency in range(129):
        reference = complex(0, 0)
        for i in range(256):
            value, angle = f32(m[i] * window[i]), -2 * PI * frequency * i / 256
            reference += value * complex(math.cos(angle), math.sin(angle))
        error = abs(complex(re[frequency], im[frequency]) - reference)
        worst = max(worst, error)
        checks.near(error, 0, 5e-6, "FFT vs independent direct DFT")
        spectral_energy += power[frequency] * (1 if frequency in (0, 128) else 2) / 256
        if power[frequency] > power[peak]:
            peak = frequency
    for frequency in range(257):
        if padded_power[frequency] > padded_power[padded_peak]:
            padded_peak = frequency
    checks.require(peak == 32 and padded_peak == 64, "FFT zero-padding preserves 1000 Hz peak")
    checks.near(math.sqrt(power[peak]), 24, 0.01, "periodic Hann amplitude A*N/4")
    checks.near(spectral_energy, window_energy, 5e-6, "one-sided Parseval weighting")
    for frame in range(15):
        for band in range(20):
            p = 0.0
            for frequency in range(129):
                p += f32(power[frame * 129 + frequency] * w[frequency * 20 + band])
            checks.near(mel[frame * 20 + band], math.log(max(p, 1e-6)), 2e-6, "manual mel multiplication and log floor")
    # Center of a one-band filter from 0 to 2100 Hz is exactly 700 Hz in mel.
    analytic = mel_weights(4200, 6, 1, 0, 2100)
    checks.near(analytic.at([1, 0]), 1, 1e-7, "analytic mel center")
    checks.near(analytic.at([0, 0]), 0, 0, "mel DC excluded")
    checks.near(analytic.at([3, 0]), 0, 1e-7, "mel high edge")
    silence = nn.zeros([256, 2])
    _, silent_power = stft(mono_of(silence), 256, 128, 256)
    silent_logmel = nn.log_mel_spectrogram(silent_power, weights).values()
    for v in silent_power.values():
        checks.near(v, 0, 0, "silence power")
    for v in silent_logmel:
        checks.near(v, math.log(1e-6), 1e-6, "finite silence log floor")
    _, short_power = stft(mono_of(nn.zeros([10, 2])), 256, 128, 256)
    checks.require(tuple(short_power.shape()) == (0, 129), "short signals yield no valid frames")
    print(f"  STFT [15,129], log-mel [15,20]; direct DFT max error={'%g' % worst}")
    print("  peak=1000 Hz (bin 32, zero-padded bin 64), Hann amplitude=24")
    print(f"  window energy={'%g' % window_energy}, Parseval energy={'%g' % spectral_energy}")
    print("  mel triangles, silence floor and short-signal checks passed")


def inspect(path):
    wave = nn.read_wav(str(path))
    _, power = stft(mono_of(wave.samples), 256, 128, 256)
    mel = nn.log_mel_spectrogram(power, mel_weights(wave.sample_rate, 256, 20, 0, wave.sample_rate / 2.0))
    print(
        f"  requested WAV: samples={nn.shape_str(wave.samples.shape())}, rate={wave.sample_rate}, "
        f"log-mel={nn.shape_str(mel.shape())}"
    )


def main(arguments=None):
    try:
        path = None
        for arg in sys.argv[1:] if arguments is None else arguments:
            if arg in ("--quick", "--full"):
                continue
            if not arg.startswith("file=") or len(arg) == 5:
                raise ValueError("expected file=PATH")
            path = arg[5:]
        print("== WAV loading and spectral preprocessing")
        wave = fixture()
        spectral_checks(wave)
        if path is not None:
            inspect(path)
        print("All audio feature checks passed.")
    except Exception as error:  # noqa: BLE001 - mirrors the C++ catch-all
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
