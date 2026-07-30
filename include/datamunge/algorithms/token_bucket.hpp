#pragma once

/// \file token_bucket.hpp
/// \brief Token-bucket rate limiting.
///
/// A *token bucket* meters a stream while permitting controlled bursts. Tokens are added to a
/// bucket at a fixed *rate* up to a *capacity*; each request consumes tokens, and is allowed
/// only if enough are available. A full bucket lets a burst through immediately (up to the
/// capacity), after which the long-run rate is capped at the refill rate -- the standard model
/// for API rate limits and traffic shaping. This module implements the meter with lazy refill.

#include <algorithm>

namespace datamunge::algorithms {

/// A token-bucket rate limiter.
class TokenBucket {
  public:
    /// \param capacity  maximum tokens (the largest allowed burst).
    /// \param rate      tokens added per unit time.
    TokenBucket(double capacity, double rate)
        : capacity_(capacity), rate_(rate), tokens_(capacity) {}

    /// \brief Try to consume \c cost tokens at time \c now; returns true if allowed.
    bool allow(double now, double cost = 1.0) {
        refill(now);
        if (tokens_ >= cost) {
            tokens_ -= cost;
            return true;
        }
        return false;
    }

    /// \brief Tokens available at time \c now (peek, with refill applied).
    double available(double now) {
        refill(now);
        return tokens_;
    }

  private:
    void refill(double now) {
        if (now > last_) {
            tokens_ = std::min(capacity_, tokens_ + (now - last_) * rate_);
            last_   = now;
        }
    }

    double capacity_, rate_, tokens_;
    double last_ = 0.0;
};

}  // namespace datamunge::algorithms
