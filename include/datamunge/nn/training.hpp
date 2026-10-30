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
  /// @brief Creates a seeded row-order stream; call begin_epoch() before requesting batches.
  /// @param samples Positive number of dataset rows indexed from zero.
  /// @param batch_size Positive maximum rows per returned batch.
  /// @param seed Seed for the store's std::mt19937_64 shuffle engine.
  /// @param shuffle Reshuffle a fresh ascending row order on every begin_epoch().
  /// @param drop_last Omit a final batch smaller than batch_size when true.
  /// @throws std::invalid_argument If the dataset or batch size is zero.
  BatchOrder(std::size_t samples, std::size_t batch_size, std::uint64_t seed, bool shuffle = true,
             bool drop_last = false);
  /// Starts a new epoch (reshuffling if enabled).
  void begin_epoch();
  /// Next batch of row indices; empty at the end of the epoch.
  std::vector<std::int64_t> next();
  /// Number of batches per epoch.
  [[nodiscard]] std::size_t   batches_per_epoch() const;
  /// @brief Returns the zero-based epoch index, or -1 before the first begin_epoch().
  [[nodiscard]] std::int64_t  epoch() const noexcept { return epoch_; }
  /// @brief Returns the number of row positions consumed in the current order.
  [[nodiscard]] std::size_t   cursor() const noexcept { return cursor_; }
  /// Text snapshot of the order, cursor and RNG (for resumable training).
  [[nodiscard]] std::string   save_state() const;
  /// @brief Restores order, cursor, epoch, and RNG from save_state() output.
  /// @throws std::invalid_argument For malformed state or different sample/batch/shuffle/drop settings.
  void                        restore_state(const std::string& state);

 private:
  /// @brief Current permutation of all dataset row indices, rebuilt each epoch.
  std::vector<std::int64_t> order_;
  /// @brief Maximum rows emitted by one next() call.
  std::size_t               batch_size_;
  /// @brief Position of the next unread row in order_, or its size after exhaustion.
  std::size_t               cursor_{0};
  /// @brief Seeded engine advanced by begin_epoch() when shuffling is enabled.
  std::mt19937_64           rng_;
  bool                      shuffle_; ///< Whether each epoch reshuffles ascending row indices.
  bool                      drop_last_; ///< Whether incomplete final batches are discarded.

  /// @brief Zero-based epoch count; -1 enforces begin_epoch() before the first read.
  std::int64_t              epoch_{-1};
};

/// The rows `indices` of x (along axis 0), as an int64 gather.
Tensor take_rows(const Tensor& x, const std::vector<std::int64_t>& indices);

/// Tracks a validation metric and decides when to stop (tflow nn::EarlyStopping).
class EarlyStopping {
 public:
  /// @brief Outcome of recording one validation metric.
  struct Decision {
    bool improved; ///< Whether this observation became the new best by more than min_delta.
    bool stop; ///< Whether a non-improving observation exhausted the configured patience.
  };
  /// Stop after `patience` epochs without an improvement larger than `min_delta`.
  explicit EarlyStopping(int patience, double min_delta = 0.0, bool maximize = false);
  /// Records the metric of `epoch` (epochs must increase).
  Decision observe(double metric, std::int64_t epoch);
  /// @brief Returns the best accepted validation metric.
  /// @throws std::logic_error If no metric has been observed yet.
  [[nodiscard]] double       best() const;
  /// @brief Returns the epoch of the best accepted metric, or -1 before any observation.
  [[nodiscard]] std::int64_t best_epoch() const noexcept { return best_epoch_; }
  /// @brief Returns consecutive non-improving observations since the latest improvement.
  [[nodiscard]] int          misses() const noexcept { return misses_; }

 private:
  /// @brief Miss threshold for returning stop; zero still waits for the first non-improving observation.
  int          patience_;
  /// @brief Minimum strict change required to accept a metric as an improvement.
  double       delta_;
  /// @brief Whether larger metrics are better; false selects minimization.
  bool         maximize_;
  /// @brief Best accepted metric; meaningful only when best_epoch_ is nonnegative.
  double       best_{0.0};
  /// @brief Consecutive observations that failed to improve by more than delta_.
  int          misses_{0};
  std::int64_t best_epoch_{-1}; ///< Epoch containing the best accepted metric, or -1 before observation.
  std::int64_t last_epoch_{-1}; ///< Most recent observed epoch, enforcing strictly increasing epochs.
};

/// Mean of batch means weighted by batch size.
class WeightedMean {
 public:
  /// @brief Adds a finite batch mean with a positive sample weight.
  /// @throws std::invalid_argument If mean is non-finite or samples is zero.
  void                      add(double mean, std::size_t samples);
  /// @brief Returns the sample-weighted mean of all added batch means.
  /// @throws std::logic_error If no samples have been added.
  [[nodiscard]] double      value() const;
  /// @brief Returns the accumulated sample weight.
  [[nodiscard]] std::size_t count() const noexcept { return count_; }
  /// @brief Clears the weighted sum and sample count to start a new aggregation.
  void                      reset() { sum_ = 0, count_ = 0; }

 private:
  /// @brief Sum of batch means multiplied by their respective sample counts.
  double      sum_{0.0};
  /// @brief Total sample count contributing to sum_.
  std::size_t count_{0};
};

/// "  epoch E: batches=B, samples=N, mean loss=L"
void log_epoch(std::ostream& out, std::int64_t epoch, std::size_t batches, const WeightedMean& loss);

}  // namespace datamunge::nn
