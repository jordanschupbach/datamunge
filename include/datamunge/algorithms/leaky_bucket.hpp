#pragma once

/// \file leaky_bucket.hpp
/// \brief Leaky-bucket rate limiting (the "leaky bucket as a meter").
///
/// A *leaky bucket* enforces a smooth output rate. Picture a bucket with a hole: admitted
/// units raise the water *level*, which drains ("leaks") at a constant rate; a unit is accepted
/// only if it fits under the bucket's *capacity*, otherwise it overflows and is dropped. Unlike
/// the token bucket -- which permits an initial burst -- the leaky bucket paces traffic to a
/// steady rate, tolerating only the slack the current level leaves. This module implements the
/// meter with lazy leaking.

#include <algorithm>

namespace datamunge::algorithms {

/// A leaky-bucket rate limiter (meter form).
class LeakyBucket {
  public:
    /// \param capacity   bucket size (how much backlog it tolerates).
    /// \param leak_rate  units drained per unit time.
    LeakyBucket(double capacity, double leak_rate)
        : capacity_(capacity), leak_rate_(leak_rate) {}

    /// \brief Try to admit \c amount at time \c now; returns true if accepted, false if dropped.
    bool add(double now, double amount = 1.0) {
        leak(now);
        if (level_ + amount <= capacity_) {
            level_ += amount;
            return true;
        }
        return false;   // overflow -> dropped
    }

    /// \brief Current fill level at time \c now (peek, with leaking applied).
    double level(double now) {
        leak(now);
        return level_;
    }

  private:
    void leak(double now) {
        if (now > last_) {
            level_ = std::max(0.0, level_ - (now - last_) * leak_rate_);
            last_  = now;
        }
    }

    double capacity_, leak_rate_;
    double level_ = 0.0;
    double last_  = 0.0;
};

}  // namespace datamunge::algorithms
