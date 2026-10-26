# Native R NN example. PCM16 WAV decoding, STFT power and log-mel spectrograms with datamunge::nn.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)
# CMD: just r-example audio_features
# Port of tflow's audio_features. Encode a stereo 1 kHz signal at 8 kHz, load
# it back through the WAV decoder, verify PCM quantization, and average
# channels. Compute a periodic-Hann STFT (256-sample frames, 128-sample hop),
# compare every first-frame complex bin to a direct DFT, verify the 1 kHz peak
# and Parseval energy, and test FFT zero-padding. Apply 20 HTK mel triangles
# and a natural-log floor. Check an analytic mel triangle, manual filter-bank
# multiplication, silence, short signals, and corrupt WAV rejection. There is
# no resampling. Optional file=PATH processes another WAV at its own rate.
# Reference: https://www.tensorflow.org/api_docs/python/tf/signal/linear_to_mel_weight_matrix
#
# nn$write_wav / nn$read_wav, nn$stft (periodic Hann), nn$power_spectrogram,
# nn$linear_to_mel_weight_matrix and nn$log_mel_spectrogram do the signal
# processing; the independent checks run in vectorized R. Host-side checks
# round float products to float32 where the C++ multiplies two floats.
nn_helper("checks")

audio_f32 <- function(values) readBin(writeBin(as.numeric(values), raw(), size = 4L), "double",
    n = length(values), size = 4L)

# tflow's Stft: valid frames, periodic Hann, unnormalized real FFT, |X|^2.
audio_stft <- function(mono, frame_length, frame_step, fft_length) {
    if (frame_length < 2 || frame_step <= 0 || fft_length < frame_length)
        stop("invalid STFT lengths", call. = FALSE)
    transform <- nn$stft(mono, frame_length, frame_step, fft_length)
    list(stft = transform, power = nn$power_spectrogram(transform))
}

# tflow's MelWeights: bands spaced over [low_hz, high_hz]; even FFT required.
audio_mel_weights <- function(sample_rate, fft_length, bands, low_hz, high_hz) {
    if (fft_length < 2 || fft_length %% 2)
        stop("invalid mel filter configuration (even FFT required)", call. = FALSE)
    nn$linear_to_mel_weight_matrix(bands, fft_length %/% 2 + 1, sample_rate, low_hz, high_hz)
}

audio_mono <- function(samples) nn$mean(samples, 1L)

audio_fixture <- function() {
    sine <- sin(2 * pi * 1000 * 0:2047 / 8000)
    original <- nn$from_vector(as.numeric(rbind(0.5 * sine, 0.25 * sine)), c(2048, 2))
    directory <- nn_output_dir("fixtures/audio")
    path <- file.path(directory, "tone.wav")
    nn$write_wav(path, original, 8000L)
    loaded <- nn$read_wav(path)
    nn_require(loaded$sample_rate == 8000 && identical(as.numeric(loaded$samples$shape()), as.numeric(original$shape())),
        "WAV rate/channel preservation")
    errors <- abs(original$values() - loaded$samples$values())
    for (error in errors) nn_near(error, 0, 1 / 32768, "PCM16 quantization bound")
    worst <- max(errors)
    corrupt <- file.path(directory, "bad.wav")
    writeBin(charToRaw("bad"), corrupt)
    rejected <- tryCatch({
        nn$read_wav(corrupt)
        FALSE
    }, error = function(e) TRUE)
    nn_require(rejected, "corrupt WAV must fail")
    cat(sprintf("  WAV: 2048 samples, 2 channels, 8000 Hz; max PCM error=%g\n", worst))
    loaded
}

audio_spectral_checks <- function(wave) {
    mono <- audio_mono(wave$samples)
    spectrum <- audio_stft(mono, 256, 128, 256)
    padded <- audio_stft(mono, 256, 128, 512)
    weights <- audio_mel_weights(wave$sample_rate, 256, 20, 0, wave$sample_rate / 2)
    log_mel <- nn$log_mel_spectrogram(spectrum$power, weights)
    nn_require(identical(as.numeric(spectrum$stft$shape()), c(15, 129)) &&
        identical(as.numeric(log_mel$shape()), c(15, 20)), "valid-frame STFT/mel dimensions")
    window <- nn$hann_window(256)$values()
    m <- mono$values()
    s <- wave$samples$values()
    re <- spectrum$stft$real$values()
    im <- spectrum$stft$imag$values()
    power <- spectrum$power$values()
    padded_power <- padded$power$values()
    mel <- matrix(log_mel$values(), nrow = 15L, byrow = TRUE)
    w <- matrix(weights$values(), nrow = 129L, byrow = TRUE)
    first <- 1:256
    channel_mean <- audio_f32(0.5 * audio_f32(s[2 * first - 1] + s[2 * first]))
    for (i in first) nn_near(m[i], channel_mean[i], 0, "channel mean")
    value <- audio_f32(m[first] * window[first])
    window_energy <- sum(value * value)
    bins <- 0:128
    errors <- vapply(bins, function(frequency) {
        angle <- -2 * pi * frequency * (first - 1) / 256
        reference <- sum(value * complex(real = cos(angle), imaginary = sin(angle)))
        Mod(complex(real = re[frequency + 1], imaginary = im[frequency + 1]) - reference)
    }, 0)
    for (error in errors) nn_near(error, 0, 5e-6, "FFT vs independent direct DFT")
    worst <- max(errors)
    first_power <- power[bins + 1]
    spectral_energy <- sum(first_power * ifelse(bins == 0 | bins == 128, 1, 2) / 256)
    peak <- which.max(first_power) - 1L
    padded_peak <- which.max(padded_power[1:257]) - 1L
    nn_require(peak == 32 && padded_peak == 64, "FFT zero-padding preserves 1000 Hz peak")
    nn_near(sqrt(first_power[peak + 1]), 24, 0.01, "periodic Hann amplitude A*N/4")
    nn_near(spectral_energy, window_energy, 5e-6, "one-sided Parseval weighting")
    frames <- matrix(power, nrow = 15L, byrow = TRUE)
    for (frame in 1:15) for (band in 1:20) {
        p <- sum(audio_f32(frames[frame, ] * w[, band]))
        nn_near(mel[frame, band], log(max(p, 1e-6)), 2e-6, "manual mel multiplication and log floor")
    }
    # Center of a one-band filter from 0 to 2100 Hz is exactly 700 Hz in mel.
    analytic <- audio_mel_weights(4200, 6, 1, 0, 2100)
    nn_near(analytic$at(c(1, 0)), 1, 1e-7, "analytic mel center")
    nn_near(analytic$at(c(0, 0)), 0, 0, "mel DC excluded")
    nn_near(analytic$at(c(3, 0)), 0, 1e-7, "mel high edge")
    silent <- audio_stft(audio_mono(nn$zeros(c(256, 2))), 256, 128, 256)
    for (v in silent$power$values()) nn_near(v, 0, 0, "silence power")
    for (v in nn$log_mel_spectrogram(silent$power, weights)$values())
        nn_near(v, log(1e-6), 1e-6, "finite silence log floor")
    short <- audio_stft(audio_mono(nn$zeros(c(10, 2))), 256, 128, 256)
    nn_require(identical(as.numeric(short$power$shape()), c(0, 129)), "short signals yield no valid frames")
    cat(sprintf("  STFT [15,129], log-mel [15,20]; direct DFT max error=%g\n", worst))
    cat("  peak=1000 Hz (bin 32, zero-padded bin 64), Hann amplitude=24\n")
    cat(sprintf("  window energy=%g, Parseval energy=%g\n", window_energy, spectral_energy))
    cat("  mel triangles, silence floor and short-signal checks passed\n")
}

audio_inspect <- function(path) {
    wave <- nn$read_wav(path)
    spectrum <- audio_stft(audio_mono(wave$samples), 256, 128, 256)
    mel <- nn$log_mel_spectrogram(spectrum$power, audio_mel_weights(wave$sample_rate, 256, 20, 0, wave$sample_rate / 2))
    cat(sprintf("  requested WAV: samples=%s, rate=%d, log-mel=%s\n", nn$shape_str(wave$samples$shape()),
        as.integer(wave$sample_rate), nn$shape_str(mel$shape())))
}

main <- function(arguments = commandArgs(TRUE)) {
    tryCatch({
        path <- ""
        for (arg in arguments) {
            if (!startsWith(arg, "file=") || nchar(arg) == 5L)
                stop("expected file=PATH", call. = FALSE)
            path <- substring(arg, 6L)
        }
        cat("== WAV loading and spectral preprocessing\n")
        wave <- audio_fixture()
        audio_spectral_checks(wave)
        if (nzchar(path)) audio_inspect(path)
        cat("All audio feature checks passed.\n")
        0L
    }, error = function(e) {
        message(conditionMessage(e))
        1L
    })
}

if (sys.nframe() == 0L) quit(status = main())
