#pragma once
/// @file token_bucket.hpp
/// @brief Checked caller-clock token bucket and actual refill/consume decisions.
namespace datamunge::algorithms {
/// @brief Own TokenBucketStep values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TokenBucketStep r{};
/// (void)r;
/// @endcode
struct TokenBucketStep {
  double time{};          ///< Actual supplied nondecreasing finite time.
  double previous_time{}; ///< Actual stored time before this operation.
  double before{};        ///< Actual stored tokens before lazy refill.
  double refilled{};      ///< Actual tokens added, after capacity saturation.
  double available{};     ///< Actual tokens after refill and before optional consumption.
  double cost{};          ///< Actual supplied nonnegative cost; zero for peek.
  bool
      consumed{}; ///< Actual allow operation found enough tokens and consumed cost; false for peek.
  double after{}; ///< Actual stored tokens after the operation.
};
/// @brief Own one mutable caller-clock token bucket.
/// Configured finite capacity/rate and actual stored tokens/time. Construction starts full at time0; operations require caller serialization and never read a wall clock.
/// @par Example
/// @code
/// dm::algorithms::TokenBucket b(2,1);
/// std::print("{}\n",b.allow(0)); // true
/// @endcode
/// @see consume()
class TokenBucket {
 public:
  /// @brief Create a full checked token bucket on a caller-supplied monotone clock.
  /// Capacity and rate must be finite nonnegative. Starts with capacity tokens at time0. Times use caller-selected common units and must be finite, nondecreasing and at least0. No wall clock is read. Mutable instance has no internal locking; caller serializes operations. Token quantities are double with rounding; this is not an exact rational admission certificate.
  /// @param capacity Finite nonnegative maximum banked tokens.
  /// @param rate Finite nonnegative added tokens per time unit.
  /// @throws std::invalid_argument Nonfinite or negative capacity/rate.
  /// @par Example
  /// @code
  /// dm::algorithms::TokenBucket b(2,1);
  /// std::print("{}\n",b.stored_tokens()); // 2
  /// @endcode
  /// @see consume()
  TokenBucket(double capacity, double rate);
  /// @brief Consume when the actual lazily refilled bucket covers the cost.
  /// Validate time/cost before mutation. Refill to capacity using elapsed time, then consume if enough. Costs above capacity are valid and denied; zero cost is admitted without spending tokens. Overflowing refill credit saturates rather than producing infinity. Failed admission still advances the refill clock and retains refilled tokens. Constant work; mutable caller-serialized state.
  /// @param now Finite time at least last_time(), initially0.
  /// @param cost Finite nonnegative requested tokens, default1.
  /// @return Owned actual before/refill/admission/after state.
  /// @throws std::invalid_argument Nonfinite/negative cost or nonfinite/backward time.
  /// @par Example
  /// @code
  /// dm::algorithms::TokenBucket b(2,1);
  /// std::print("{}\n",b.consume(0,1).after); // 1
  /// @endcode
  /// @see peek()
  TokenBucketStep consume(double now, double cost = 1);
  /// @brief Return the actual admission decision from the shared consume operation.
  /// Validate time/cost before mutation. Refill to capacity using elapsed time, then consume if enough. Costs above capacity are valid and denied; zero cost is admitted without spending tokens. Overflowing refill credit saturates rather than producing infinity. Failed admission still advances the refill clock and retains refilled tokens. Constant work; mutable caller-serialized state.
  /// @param now Finite time at least last_time(), initially0.
  /// @param cost Finite nonnegative requested tokens, default1.
  /// @return Actual true if admitted, false if insufficient tokens.
  /// @throws std::invalid_argument Nonfinite/negative cost or nonfinite/backward time.
  /// @par Example
  /// @code
  /// dm::algorithms::TokenBucket b(2,1);
  /// std::print("{}\n",b.allow(0,1)); // true
  /// @endcode
  /// @see peek()
  bool allow(double now, double cost = 1);
  /// @brief Inspect actual lazy refill without consuming tokens.
  /// This is a mutable operation: it advances the stored clock even when no credit can be added. Finite nondecreasing time required. Capacity saturation prevents overflow; constant work.
  /// @param now Finite time at least last_time().
  /// @return Owned actual refill state, with consumedfalse.
  /// @throws std::invalid_argument Nonfinite or backward time.
  /// @par Example
  /// @code
  /// dm::algorithms::TokenBucket b(2,1);
  /// b.allow(0,2);
  /// std::print("{}\n",b.peek(1).after); // 1
  /// @endcode
  /// @see consume()
  TokenBucketStep peek(double now);
  /// @brief Return tokens after actual lazy refill without consumption.
  /// This is a mutable operation: it advances the stored clock even when no credit can be added. Finite nondecreasing time required. Capacity saturation prevents overflow; constant work.
  /// @param now Finite time at least last_time().
  /// @return Actual available tokens after refill.
  /// @throws std::invalid_argument Nonfinite or backward time.
  /// @par Example
  /// @code
  /// dm::algorithms::TokenBucket b(2,1);
  /// b.allow(0,2);
  /// std::print("{}\n",b.available(1)); // 1
  /// @endcode
  /// @see consume()
  double available(double now);
  /// @brief Inspect configured token capacity.
  /// Const constant-time read with no mutation or clock access.
  /// @return Actual configured token capacity.
  /// @par Example
  /// @code
  /// dm::algorithms::TokenBucket b(2,1);
  /// std::print("{}\n",b.capacity()); // 2
  /// @endcode
  /// @see available()
  double capacity() const noexcept;
  /// @brief Inspect configured tokens per time unit.
  /// Const constant-time read with no mutation or clock access.
  /// @return Actual configured tokens per time unit.
  /// @par Example
  /// @code
  /// dm::algorithms::TokenBucket b(2,1);
  /// std::print("{}\n",b.rate()); // 1
  /// @endcode
  /// @see available()
  double rate() const noexcept;
  /// @brief Inspect actual current tokens without refill.
  /// Const constant-time read with no mutation or clock access.
  /// @return Actual actual current tokens without refill.
  /// @par Example
  /// @code
  /// dm::algorithms::TokenBucket b(2,1);
  /// std::print("{}\n",b.stored_tokens()); // 2
  /// @endcode
  /// @see available()
  double stored_tokens() const noexcept;
  /// @brief Inspect actual last refill time, initially0.
  /// Const constant-time read with no mutation or clock access.
  /// @return Actual actual last refill time, initially0.
  /// @par Example
  /// @code
  /// dm::algorithms::TokenBucket b(2,1);
  /// std::print("{}\n",b.last_time()); // 0
  /// @endcode
  /// @see available()
  double last_time() const noexcept;

 private:
  double capacity_{}; ///< Finite immutable configured capacity.
  double rate_{};     ///< Finite immutable configured refill rate.
  double tokens_{};   ///< Actual stored tokens in[0,capacity].
  double last_{};     ///< Actual most recent supplied time, initially0.

  /// @brief Validate and apply actual lazy credit with saturation.
  /// @param now Finite nondecreasing caller time.
  /// @return Actual operation state before any consumption.
  /// @throws std::invalid_argument Nonfinite or backward time.
  TokenBucketStep refill(double now);
};
} // namespace datamunge::algorithms
