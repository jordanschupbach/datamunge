/// @file pcnn.cpp
/// @brief Canonical producing implementation and actual-site diagnostics; public contracts are in the paired header.
#include <cmath>
#include <datamunge/algorithms/pcnn.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief File-local validated kernels shared by ordinary and observed entry points.
namespace {
/// @brief Original synchronous producer with complete admission and optional actual cell observer.
PCNNTrace run(
    const std::vector<std::vector<double>>& image,
    const PCNNParameters&                   params,
    PCNNOptions                             options,
    bool                                    capture) {
  PCNNTrace trace;
  auto&     result = trace.result;
  if (params.iterations > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      image.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("PCNN iteration/index range");
  if (!std::isfinite(params.beta) || params.beta < 0 || !std::isfinite(params.threshold_gain) ||
      params.threshold_gain < 0 || !std::isfinite(params.initial_threshold) ||
      params.initial_threshold < 0 || !std::isfinite(params.threshold_decay) ||
      params.threshold_decay < 0 || params.threshold_decay > 1)
    throw std::invalid_argument("PCNN parameters");
  const auto width = image.empty() ? 0 : image.front().size();
  if (width > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("PCNN column index");
  for (const auto& row : image) {
    if (row.size() != width)
      throw std::invalid_argument("PCNN rectangular image required");
    for (double v : row)
      if (!std::isfinite(v) || v < 0)
        throw std::invalid_argument("PCNN finite nonnegative input required");
  }
  /// @brief Multiply logical counts without wraparound.
  auto mul = [](std::size_t x, std::size_t y) {
    if (y && x > std::numeric_limits<std::size_t>::max() / y)
      throw std::length_error("PCNN count overflow");
    return x * y;
  };
  const auto pixels = mul(image.size(), width);
  if (pixels > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("PCNN signature count range");
  const auto output = mul(pixels, params.iterations + 1);
  if (output > std::numeric_limits<std::size_t>::max() - params.iterations)
    throw std::length_error("PCNN output count");
  if (pixels && options.max_output_entries &&
      output + params.iterations > options.max_output_entries)
    throw std::length_error("PCNN output policy");
  if (capture && options.max_trace_records &&
      mul(pixels, params.iterations) > options.max_trace_records)
    throw std::length_error("PCNN trace policy");
  const std::size_t rows = image.size();
  const std::size_t cols = rows ? image.front().size() : 0;
  if (rows == 0 || cols == 0)
    return trace;

  std::vector<std::vector<double>> theta(rows, std::vector<double>(cols, params.initial_threshold));
  std::vector<std::vector<int>>    Y(rows, std::vector<int>(cols, 0));
  result.first_fire.assign(rows, std::vector<int>(cols, 0));

  for (std::size_t n = 1; n <= params.iterations; ++n) {
    std::vector<std::vector<int>> Ynext(rows, std::vector<int>(cols, 0));
    int                           fired = 0;
    for (std::size_t i = 0; i < rows; ++i)
      for (std::size_t j = 0; j < cols; ++j) {
        // Linking: sum of previous pulses over 8-connected neighbors.
        double L = 0.0;
        for (int di = -1; di <= 1; ++di)
          for (int dj = -1; dj <= 1; ++dj) {
            if (di == 0 && dj == 0)
              continue;
            const int ni = static_cast<int>(i) + di, nj = static_cast<int>(j) + dj;
            if (ni >= 0 && ni < static_cast<int>(rows) && nj >= 0 && nj < static_cast<int>(cols))
              L += Y[static_cast<std::size_t>(ni)][static_cast<std::size_t>(nj)];
          }
        const double U      = image[i][j] * (1.0 + params.beta * L);
        const auto   before = theta[i][j];
        theta[i][j] *= params.threshold_decay;
        const auto decayed = theta[i][j];
        if (!std::isfinite(U) || !std::isfinite(decayed))
          throw std::overflow_error("PCNN activity/threshold overflow");
        const int y = (U > theta[i][j]) ? 1 : 0;
        if (y) {
          theta[i][j] += params.threshold_gain;
          ++fired;
          if (result.first_fire[i][j] == 0)
            result.first_fire[i][j] = static_cast<int>(n);
        }
        if (!std::isfinite(theta[i][j]))
          throw std::overflow_error("PCNN threshold jump overflow");
        if (capture)
          trace.cells.push_back({n, i, j, L, U, before, decayed, y, theta[i][j]});
        Ynext[i][j] = y;
      }
    Y = Ynext;
    result.pulses.push_back(Y);
    result.time_signature.push_back(fired);
  }
  return trace;
}

} // namespace
PCNNResult pcnn_run(const std::vector<std::vector<double>>& image, const PCNNParameters& params) {
  return run(image, params, {}, false).result;
}
PCNNTrace pcnn_trace(
    const std::vector<std::vector<double>>& image,
    const PCNNParameters&                   params,
    PCNNOptions                             options) {
  return run(image, params, options, true);
}
} // namespace datamunge::algorithms
