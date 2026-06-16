#pragma once

#include <initializer_list>
#include <algorithm>
#include <atomic>
#include <functional>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace datamunge::dstruct {

template <typename T>
class ThreadSafeVector {
 public:
  using value_type = T;
  using size_type = typename std::vector<T>::size_type;

  ThreadSafeVector() = default;

  explicit ThreadSafeVector(std::vector<T> values) : data_(std::move(values)) {}

  ThreadSafeVector(std::initializer_list<T> init) : data_(init) {}

  ThreadSafeVector(const ThreadSafeVector& other) {
    std::shared_lock lock(other.mutex_);
    data_ = other.data_;
  }

  ThreadSafeVector(ThreadSafeVector&& other) noexcept {
    std::unique_lock lock(other.mutex_);
    data_ = std::move(other.data_);
  }

  ThreadSafeVector& operator=(const ThreadSafeVector& other) {
    if (this == &other) {
      return *this;
    }

    std::vector<T> copy;
    {
      std::shared_lock lock(other.mutex_);
      copy = other.data_;
    }
    {
      std::unique_lock lock(mutex_);
      data_ = std::move(copy);
    }
    return *this;
  }

  ThreadSafeVector& operator=(ThreadSafeVector&& other) noexcept {
    if (this == &other) {
      return *this;
    }

    std::scoped_lock lock(mutex_, other.mutex_);
    data_ = std::move(other.data_);
    return *this;
  }

  [[nodiscard]] bool empty() const {
    std::shared_lock lock(mutex_);
    return data_.empty();
  }

  [[nodiscard]] size_type size() const {
    std::shared_lock lock(mutex_);
    return data_.size();
  }

  void reserve(size_type capacity) {
    std::unique_lock lock(mutex_);
    data_.reserve(capacity);
  }

  void clear() {
    std::unique_lock lock(mutex_);
    data_.clear();
  }

  void push_back(const T& value) {
    std::unique_lock lock(mutex_);
    data_.push_back(value);
  }

  void push_back(T&& value) {
    std::unique_lock lock(mutex_);
    data_.push_back(std::move(value));
  }

  template <typename... Args>
  void emplace_back(Args&&... args) {
    std::unique_lock lock(mutex_);
    data_.emplace_back(std::forward<Args>(args)...);
  }

  void append(const std::vector<T>& values) {
    std::unique_lock lock(mutex_);
    data_.insert(data_.end(), values.begin(), values.end());
  }

  [[nodiscard]] T at(size_type index) const {
    std::shared_lock lock(mutex_);
    return data_.at(index);
  }

  void set(size_type index, const T& value) {
    std::unique_lock lock(mutex_);
    if (index >= data_.size()) {
      throw std::out_of_range("ThreadSafeVector::set index out of range");
    }
    data_[index] = value;
  }

  void set(size_type index, T&& value) {
    std::unique_lock lock(mutex_);
    if (index >= data_.size()) {
      throw std::out_of_range("ThreadSafeVector::set index out of range");
    }
    data_[index] = std::move(value);
  }

  [[nodiscard]] std::optional<T> pop_back() {
    std::unique_lock lock(mutex_);
    if (data_.empty()) {
      return std::nullopt;
    }

    T value = std::move(data_.back());
    data_.pop_back();
    return value;
  }

  [[nodiscard]] std::vector<T> snapshot() const {
    std::shared_lock lock(mutex_);
    return data_;
  }

 private:
  mutable std::shared_mutex mutex_;
  std::vector<T> data_;
};

template <typename T, typename Func>
auto apply(const ThreadSafeVector<T>& values, Func&& fun, std::size_t ncores, std::size_t blocksize)
    -> std::vector<std::decay_t<std::invoke_result_t<Func&, const T&>>> {
  using result_type = std::decay_t<std::invoke_result_t<Func&, const T&>>;

  const auto snapshot = values.snapshot();
  std::vector<result_type> result(snapshot.size());
  if (snapshot.empty()) {
    return result;
  }

  const std::size_t chunk_size = std::max<std::size_t>(1, blocksize);
  const std::size_t block_count = (snapshot.size() + chunk_size - 1) / chunk_size;
  const std::size_t worker_count = std::min(std::max<std::size_t>(1, ncores), block_count);
  auto fn = std::forward<Func>(fun);
  std::atomic<std::size_t> next_index{0};
  std::vector<std::thread> workers;
  workers.reserve(worker_count);

  for (std::size_t worker = 0; worker < worker_count; ++worker) {
    workers.emplace_back([&snapshot, &result, &fn, &next_index, chunk_size] {
      while (true) {
        const std::size_t start = next_index.fetch_add(chunk_size, std::memory_order_relaxed);
        if (start >= snapshot.size()) {
          break;
        }

        const std::size_t end = std::min(start + chunk_size, snapshot.size());
        for (std::size_t index = start; index < end; ++index) {
          result[index] = std::invoke(fn, snapshot[index]);
        }
      }
    });
  }

  for (auto& worker : workers) {
    worker.join();
  }

  return result;
}

} // namespace datamunge::dstruct
