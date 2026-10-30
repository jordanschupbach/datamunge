/// @file memory.cpp
/// @brief Process-wide relaxed atomic accounting for tensor-buffer allocation lifetimes.

#include "datamunge/nn/memory.hpp"

#include <atomic>

namespace datamunge::nn {

/// @brief Atomic allocation counters shared by every tensor allocator in this process.
namespace {
/// @brief Sum of logical byte sizes of live tensor buffers; allocator overhead is excluded.
std::atomic<std::int64_t> live_bytes{0};
/// @brief Largest observed live-byte count since reset_peak_memory().
std::atomic<std::int64_t> peak_bytes{0};
/// @brief Number of live buffers, including those with zero logical bytes.
std::atomic<std::int64_t> live_count{0};
}  // namespace

namespace detail {

void note_storage_alloc(std::int64_t nbytes) {
  live_count.fetch_add(1, std::memory_order_relaxed);
  const std::int64_t now = live_bytes.fetch_add(nbytes, std::memory_order_relaxed) + nbytes;
  // Raise the high-water mark; usually a single relaxed load.
  std::int64_t peak = peak_bytes.load(std::memory_order_relaxed);
  while (now > peak && !peak_bytes.compare_exchange_weak(peak, now, std::memory_order_relaxed)) {
  }
}

void note_storage_free(std::int64_t nbytes) {
  live_count.fetch_sub(1, std::memory_order_relaxed);
  live_bytes.fetch_sub(nbytes, std::memory_order_relaxed);
}

}  // namespace detail

std::int64_t live_memory_bytes() { return live_bytes.load(std::memory_order_relaxed); }
std::int64_t peak_memory_bytes() { return peak_bytes.load(std::memory_order_relaxed); }
void         reset_peak_memory() { peak_bytes.store(live_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed); }
std::int64_t live_storage_count() { return live_count.load(std::memory_order_relaxed); }

PeakMemoryScope::PeakMemoryScope() {
  reset_peak_memory();
  baseline_ = live_memory_bytes();
}
std::int64_t PeakMemoryScope::peak_above_baseline() const { return peak_memory_bytes() - baseline_; }
std::int64_t PeakMemoryScope::live_above_baseline() const { return live_memory_bytes() - baseline_; }

}  // namespace datamunge::nn
