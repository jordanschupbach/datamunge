#pragma once

/// @file inference_benchmark.hpp
/// @brief Warmed CPU timing for inference calls over prepared batches.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

namespace datamunge::nn {

/// @brief Call latency and throughput from one fixed-size batch benchmark.
/// @par Example
/// @code
/// const dm::nn::InferenceTiming stats;
/// std::println("{}", stats.median_us); // 0
/// @endcode
/// @see benchmark_inference()
struct InferenceTiming {
  double median_us{}; ///< Median elapsed time per call in microseconds.
  double p95_us{}; ///< Nearest-rank 95th percentile elapsed time in microseconds.
  double examples_per_second{}; ///< Total rows divided by total elapsed seconds.
};

/// @brief Measure warmed synchronous inference on cyclic, prebuilt batches.
/// @tparam Predict Callable taking a const batch reference. It must finish inference before returning.
/// @tparam Batch Prepared batch type.
/// @param predict Function to measure, including lifetime of each returned result.
/// @param batches Nonempty collection cycled in order during warm-up and measurement.
/// @param batch_size Positive row count shared by the prepared batches.
/// @param warmups Number of untimed calls, default 12.
/// @param samples Positive number of timed calls, default 100.
/// @return Median, nearest-rank p95, and aggregate throughput.
/// @throws std::invalid_argument Empty batches or zero rows or samples.
/// @throws std::runtime_error The measured aggregate duration is zero.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const std::vector batches{n::ones({4, 2})};
/// const auto timing = n::benchmark_inference([](const n::Tensor& x) { return n::relu(x); }, batches, 4);
/// @endcode
template<class Predict, class Batch>
[[nodiscard]] InferenceTiming benchmark_inference(Predict&& predict, const std::vector<Batch>& batches,
    std::size_t batch_size, std::size_t warmups = 12, std::size_t samples = 100) {
  if (batches.empty() || batch_size == 0 || samples == 0)
    throw std::invalid_argument("benchmark_inference: expected batches and positive rows and samples");
  for (std::size_t i = 0; i < warmups; ++i)
    (void)std::invoke(predict, batches[i % batches.size()]);

  using Clock = std::chrono::steady_clock;
  std::vector<double> durations(samples);
  double elapsed_us = 0.;
  for (std::size_t i = 0; i < samples; ++i) {
    const auto& batch = batches[i % batches.size()];
    const auto start = Clock::now();
    (void)std::invoke(predict, batch);
    durations[i] = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
    elapsed_us += durations[i];
  }
  if (elapsed_us <= 0.)
    throw std::runtime_error("benchmark_inference: duration below clock resolution");
  std::ranges::sort(durations);
  const auto p95 = samples - samples / 20 - 1;
  return {(durations[(samples - 1) / 2] + durations[samples / 2]) * .5,
          durations[p95], static_cast<double>(batch_size) * samples * 1e6 / elapsed_us};
}

} // namespace datamunge::nn
