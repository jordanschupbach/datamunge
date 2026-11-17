/// @file leaky_bucket.cpp
/// @brief Shared actual lazy meter transitions and complete checked event capture.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/leaky_bucket.hpp>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private complete event contract validation.
namespace leaky_detail {
/// @brief Validate one proposed event before any mutation.
/// @param now Proposed finite nonnegative timestamp.
/// @param amount Proposed finite nonnegative amount.
/// @param last Last committed finite timestamp.
/// @throws std::invalid_argument On backward/negative/nonfinite values.
void event(double now, double amount, double last) {
  if (!std::isfinite(now) || now < last || !std::isfinite(amount) || amount < 0)
    throw std::invalid_argument("leaky bucket requires ordered finite nonnegative events");
}
} // namespace leaky_detail
LeakyBucket::LeakyBucket(double capacity, double leak_rate)
    : capacity_(capacity), leak_rate_(leak_rate) {
  if (!std::isfinite(capacity) || capacity <= 0 || !std::isfinite(leak_rate) || leak_rate < 0)
    throw std::invalid_argument("leaky bucket capacity/rate");
}
double LeakyBucket::leak_value(double now, bool& widened) const {
  const double elapsed = now - last_, drain = elapsed * leak_rate_, value = level_ - drain;
  if (std::isfinite(drain) && std::isfinite(value))
    return std::max(0., value);
  widened = true;
  return static_cast<double>(std::max(
      0.L,
      static_cast<long double>(level_) - (static_cast<long double>(now) - last_) * leak_rate_));
}
LeakyBucketEvent LeakyBucket::add_detail(double now, double amount) {
  leaky_detail::event(now, amount, last_);
  LeakyBucketEvent e;
  e.time                 = now;
  e.amount               = amount;
  e.previous_time        = last_;
  e.before               = level_;
  e.leaked_level         = leak_value(now, e.widened);
  const double candidate = e.leaked_level + amount;
  e.accepted             = std::isfinite(candidate) && candidate <= capacity_;
  e.after                = e.accepted ? candidate : e.leaked_level;
  level_                 = e.after;
  last_                  = now;
  return e;
}
bool LeakyBucket::add(double now, double amount) {
  return add_detail(now, amount).accepted;
}
double LeakyBucket::level(double now) {
  leaky_detail::event(now, 0, last_);
  bool         widened = false;
  const double value   = leak_value(now, widened);
  level_               = value;
  last_                = now;
  return level_;
}
LeakyBucketTrace leaky_bucket_trace(
    double                                 capacity,
    double                                 rate,
    const std::vector<LeakyBucketArrival>& arrivals,
    std::size_t                            max_entries) {
  LeakyBucket bucket(capacity, rate);
  if ((max_entries && arrivals.size() > max_entries) ||
      arrivals.size() > std::vector<LeakyBucketEvent>{}.max_size())
    throw std::length_error("leaky bucket complete capture quota");
  double last = 0;
  for (auto& a : arrivals) {
    leaky_detail::event(a.time, a.amount, last);
    last = a.time;
  }
  LeakyBucketTrace out;
  out.capacity = capacity;
  out.rate     = rate;
  out.events.reserve(arrivals.size());
  for (auto& a : arrivals) {
    auto e = bucket.add_detail(a.time, a.amount);
    e.accepted ? ++out.accepted : ++out.rejected;
    out.events.push_back(e);
  }
  out.captured_entries = out.events.size();
  return out;
}
} // namespace datamunge::algorithms
