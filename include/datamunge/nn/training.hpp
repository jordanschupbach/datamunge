#pragma once

/// @file training.hpp
/// @brief Training-loop helpers: mini-batch order, early stopping, running means.

#include <cstdint>
#include <ostream>
#include <random>
#include <string>
#include <vector>

#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

/// Mini-batch index order over `samples` rows, reshuffled every epoch with
/// a seeded std::mt19937_64 (the same stream as tflow's BatchOrder, so ports
/// see identical batches).
class BatchOrder {
 public:
  BatchOrder(std::size_t samples, std::size_t batch_size, std::uint64_t seed, bool shuffle = true,
             bool drop_last = false);
  /// Starts a new epoch (reshuffling if enabled).
  void begin_epoch();
  /// Next batch of row indices; empty at the end of the epoch.
  std::vector<std::int64_t> next();
  /// Number of batches per epoch.
  [[nodiscard]] std::size_t   batches_per_epoch() const;
  [[nodiscard]] std::int64_t  epoch() const noexcept { return epoch_; }
  [[nodiscard]] std::size_t   cursor() const noexcept { return cursor_; }
  /// Text snapshot of the order, cursor and RNG (for resumable training).
  [[nodiscard]] std::string   save_state() const;
  void                        restore_state(const std::string& state);

 private:
  std::vector<std::int64_t> order_;
  std::size_t               batch_size_;
  std::size_t               cursor_{0};
  std::mt19937_64           rng_;
  bool                      shuffle_, drop_last_;
  std::int64_t              epoch_{-1};
};

/// The rows `indices` of x (along axis 0), as an int64 gather.
Tensor take_rows(const Tensor& x, const std::vector<std::int64_t>& indices);

/// Tracks a validation metric and decides when to stop (tflow nn::EarlyStopping).
class EarlyStopping {
 public:
  struct Decision {
    bool improved;
    bool stop;
  };
  /// Stop after `patience` epochs without an improvement larger than `min_delta`.
  explicit EarlyStopping(int patience, double min_delta = 0.0, bool maximize = false);
  /// Records the metric of `epoch` (epochs must increase).
  Decision observe(double metric, std::int64_t epoch);
  [[nodiscard]] double       best() const;
  [[nodiscard]] std::int64_t best_epoch() const noexcept { return best_epoch_; }
  [[nodiscard]] int          misses() const noexcept { return misses_; }

 private:
  int          patience_;
  double       delta_;
  bool         maximize_;
  double       best_{0.0};
  int          misses_{0};
  std::int64_t best_epoch_{-1}, last_epoch_{-1};
};

/// Mean of batch means weighted by batch size.
class WeightedMean {
 public:
  void                      add(double mean, std::size_t samples);
  [[nodiscard]] double      value() const;
  [[nodiscard]] std::size_t count() const noexcept { return count_; }
  void                      reset() { sum_ = 0, count_ = 0; }

 private:
  double      sum_{0.0};
  std::size_t count_{0};
};

/// "  epoch E: batches=B, samples=N, mean loss=L"
void log_epoch(std::ostream& out, std::int64_t epoch, std::size_t batches, const WeightedMean& loss);

}  // namespace datamunge::nn
