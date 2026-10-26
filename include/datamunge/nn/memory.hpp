#pragma once

/// @file memory.hpp
/// @brief Byte counts of live tensor storage, for memory tests and reports.
///
/// Every `nn::Tensor` buffer is a `Storage` allocated by `empty()`. The
/// library counts the bytes of the buffers that are alive and the high-water
/// mark of that count, process wide, with two atomics updated when a buffer is
/// created or freed. Unlike the process resident set size, these numbers are
/// exact, portable and unaffected by the allocator keeping freed pages.
///
/// Typical use, measuring the peak of one training step above its start:
///
///   nn::reset_peak_memory();
///   const auto before = nn::live_memory_bytes();
///   step();
///   const auto extra = nn::peak_memory_bytes() - before;

#include <cstdint>

namespace datamunge::nn {

/// Bytes held by tensor buffers that are alive now.
std::int64_t live_memory_bytes();
/// Highest `live_memory_bytes()` since the last `reset_peak_memory()`.
std::int64_t peak_memory_bytes();
/// Restarts the high-water mark at the current live byte count.
void reset_peak_memory();
/// Number of tensor buffers alive now.
std::int64_t live_storage_count();

/// Records the peak of live bytes over its lifetime, relative to the live
/// bytes when it was created. Resets the process-wide peak when constructed.
class PeakMemoryScope {
 public:
  PeakMemoryScope();
  /// Live bytes when the scope began.
  [[nodiscard]] std::int64_t baseline() const noexcept { return baseline_; }
  /// Peak live bytes since the scope began, minus the baseline.
  [[nodiscard]] std::int64_t peak_above_baseline() const;
  /// Live bytes now, minus the baseline.
  [[nodiscard]] std::int64_t live_above_baseline() const;

 private:
  std::int64_t baseline_;
};

namespace detail {
/// Allocation hooks called by the Storage allocator in tensor.cpp.
void note_storage_alloc(std::int64_t nbytes);
void note_storage_free(std::int64_t nbytes);
}  // namespace detail

}  // namespace datamunge::nn
