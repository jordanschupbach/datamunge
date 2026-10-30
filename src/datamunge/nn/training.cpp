/// @file training.cpp
/// @brief Resumable mini-batch ordering, validation stopping rules, and weighted metric summaries.

#include "datamunge/nn/training.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

// ---- BatchOrder ----------------------------------------------------------------------

BatchOrder::BatchOrder(std::size_t samples, std::size_t batch_size, std::uint64_t seed, bool shuffle, bool drop_last)
    : order_(samples), batch_size_(batch_size), rng_(seed), shuffle_(shuffle), drop_last_(drop_last) {
  if (!samples || !batch_size) throw std::invalid_argument("BatchOrder: dataset and batch size must be positive");
  std::iota(order_.begin(), order_.end(), std::int64_t{0});
}

void BatchOrder::begin_epoch() {
  std::iota(order_.begin(), order_.end(), std::int64_t{0});
  if (shuffle_) std::shuffle(order_.begin(), order_.end(), rng_);
  cursor_ = 0;
  ++epoch_;
}

std::vector<std::int64_t> BatchOrder::next() {
  if (epoch_ < 0) throw std::logic_error("BatchOrder: call begin_epoch before reading batches");
  const std::size_t size = std::min(batch_size_, order_.size() - cursor_);
  if (!size || (drop_last_ && size < batch_size_)) {
    cursor_ = order_.size();
    return {};
  }
  std::vector<std::int64_t> batch(order_.begin() + static_cast<std::ptrdiff_t>(cursor_),
                                  order_.begin() + static_cast<std::ptrdiff_t>(cursor_ + size));
  cursor_ += size;
  return batch;
}

std::size_t BatchOrder::batches_per_epoch() const {
  return drop_last_ ? order_.size() / batch_size_ : (order_.size() + batch_size_ - 1) / batch_size_;
}

std::string BatchOrder::save_state() const {
  std::ostringstream out;
  out << "DATAMUNGE_BATCH_ORDER_V1 " << order_.size() << ' ' << batch_size_ << ' ' << shuffle_ << ' ' << drop_last_
      << ' ' << epoch_ << ' ' << cursor_ << '\n'
      << rng_ << '\n';
  for (auto index : order_) out << index << ' ';
  return out.str();
}

void BatchOrder::restore_state(const std::string& state) {
  std::istringstream in(state);
  std::string        magic;
  std::size_t        size = 0, batch = 0, cursor = 0;
  int                shuffle = 0, drop = 0;
  std::int64_t       epoch = -1;
  std::mt19937_64    rng;
  if (!(in >> magic >> size >> batch >> shuffle >> drop >> epoch >> cursor >> rng) ||
      magic != "DATAMUNGE_BATCH_ORDER_V1" || size != order_.size() || batch != batch_size_ ||
      (shuffle != 0) != shuffle_ || (drop != 0) != drop_last_ || epoch < -1 || cursor > size ||
      (epoch == -1 && cursor != 0))
    throw std::invalid_argument("BatchOrder: invalid or incompatible batching state");
  std::vector<std::int64_t> order(size);
  std::vector<bool>         seen(size, false);
  for (auto& index : order) {
    if (!(in >> index) || index < 0 || static_cast<std::size_t>(index) >= size ||
        seen[static_cast<std::size_t>(index)])
      throw std::invalid_argument("BatchOrder: batch order must be a permutation");
    seen[static_cast<std::size_t>(index)] = true;
  }
  in >> std::ws;
  if (!in.eof()) throw std::invalid_argument("BatchOrder: trailing batching state data");
  order_  = std::move(order);
  rng_    = rng;
  epoch_  = epoch;
  cursor_ = cursor;
}

Tensor take_rows(const Tensor& x, const std::vector<std::int64_t>& indices) {
  return gather(x, from_vector(indices), 0);
}

// ---- EarlyStopping ---------------------------------------------------------------------

EarlyStopping::EarlyStopping(int patience, double min_delta, bool maximize)
    : patience_(patience), delta_(min_delta), maximize_(maximize) {
  if (patience < 0 || !std::isfinite(min_delta) || min_delta < 0)
    throw std::invalid_argument("EarlyStopping: needs nonnegative patience and min_delta");
}

EarlyStopping::Decision EarlyStopping::observe(double metric, std::int64_t epoch) {
  if (!std::isfinite(metric) || epoch < 0 || epoch <= last_epoch_)
    throw std::invalid_argument("EarlyStopping: validation metric must be finite and epochs must increase");
  last_epoch_         = epoch;
  const bool improved = best_epoch_ < 0 || (maximize_ ? metric > best_ + delta_ : metric < best_ - delta_);
  if (improved) {
    best_       = metric;
    best_epoch_ = epoch;
    misses_     = 0;
  } else {
    ++misses_;
  }
  return {improved, !improved && misses_ >= patience_};
}

double EarlyStopping::best() const {
  if (best_epoch_ < 0) throw std::logic_error("EarlyStopping: no validation observation yet");
  return best_;
}

// ---- WeightedMean ----------------------------------------------------------------------

void WeightedMean::add(double mean, std::size_t samples) {
  if (!std::isfinite(mean) || !samples)
    throw std::invalid_argument("WeightedMean: needs a finite batch mean and positive weight");
  sum_ += mean * static_cast<double>(samples);
  count_ += samples;
}

double WeightedMean::value() const {
  if (!count_) throw std::logic_error("WeightedMean: no observations");
  return sum_ / static_cast<double>(count_);
}

void log_epoch(std::ostream& out, std::int64_t epoch, std::size_t batches, const WeightedMean& loss) {
  out << "  epoch " << epoch << ": batches=" << batches << ", samples=" << loss.count()
      << ", mean loss=" << loss.value() << '\n';
}

}  // namespace datamunge::nn
