/// @file token_bucket.cpp
/// @brief Actual checked lazy refill and admission using caller-monotone time.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/token_bucket.hpp>
#include <stdexcept>
namespace datamunge::algorithms {
TokenBucket::TokenBucket(double capacity, double rate)
    : capacity_(capacity), rate_(rate), tokens_(capacity) {
  if (!std::isfinite(capacity) || !std::isfinite(rate) || capacity < 0 || rate < 0)
    throw std::invalid_argument("token bucket finite nonnegative configuration");
}
TokenBucketStep TokenBucket::refill(double now) {
  if (!std::isfinite(now) || now < last_)
    throw std::invalid_argument("token bucket monotone finite time");
  TokenBucketStep s;
  s.time          = now;
  s.previous_time = last_;
  s.before        = tokens_;
  double elapsed  = now - last_;
  if (rate_ > 0 && elapsed > 0) {
    long double credit = (long double)elapsed * rate_;
    if (credit >= (long double)capacity_ - tokens_)
      tokens_ = capacity_;
    else
      tokens_ = std::min(capacity_, tokens_ + elapsed * rate_);
  }
  last_       = now;
  s.refilled  = tokens_ - s.before;
  s.available = s.after = tokens_;
  return s;
}
TokenBucketStep TokenBucket::consume(double now, double cost) {
  if (!std::isfinite(cost) || cost < 0)
    throw std::invalid_argument("token bucket finite nonnegative cost");
  auto s = refill(now);
  s.cost = cost;
  if (tokens_ >= cost) {
    tokens_ -= cost;
    s.consumed = true;
  }
  s.after = tokens_;
  return s;
}
bool TokenBucket::allow(double now, double cost) {
  return consume(now, cost).consumed;
}
TokenBucketStep TokenBucket::peek(double now) {
  return refill(now);
}
double TokenBucket::available(double now) {
  return peek(now).after;
}
double TokenBucket::capacity() const noexcept {
  return capacity_;
}
double TokenBucket::rate() const noexcept {
  return rate_;
}
double TokenBucket::stored_tokens() const noexcept {
  return tokens_;
}
double TokenBucket::last_time() const noexcept {
  return last_;
}
} // namespace datamunge::algorithms
