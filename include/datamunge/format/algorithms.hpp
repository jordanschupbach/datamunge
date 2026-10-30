#pragma once

/// @file algorithms.hpp
/// @brief std::format / std::print support for datamunge::algorithms.
/// Included by <datamunge/format.hpp>; requires C++20 <format>.
///
/// Result, option and data structs print on one line as `TypeName(field=value, ...)`
/// (see format/algorithms/records.hpp); enums print their enumerator name; stateful
/// classes (counters, clocks, allocators, parsers, online statistics) print their
/// public state. Formatting never mutates or advances the object.
///
/// @par Example
/// @code
/// const auto g = dm::algorithms::extended_gcd(240, 46);
/// std::print("{}\n", g);   // ExtendedGcd(gcd=2, x=-9, y=47)
/// @endcode

#include <datamunge/algorithms/algorithms.hpp>
#include <datamunge/format/algorithms/common.hpp>
#include <datamunge/format/text_formatter.hpp>

#include <format>
#include <string>
#include <string_view>

// ---- enums -------------------------------------------------------------------------------------

/// @brief Prints an algorithms::PegOp as its enumerator name (`Literal`, `Seq`, `Choice`, ...).
template <>
struct std::formatter<datamunge::algorithms::PegOp> : datamunge::format_detail::text_formatter<datamunge::algorithms::PegOp> {
  /// @brief Maps the enumerator to its name.
  /// @param op The PEG operator.
  /// @return The enumerator name, or `PegOp(<int>)` for an out-of-range value.
  static std::string text(const datamunge::algorithms::PegOp& op) {
    using enum datamunge::algorithms::PegOp;
    switch (op) {
      case Literal: return "Literal";
      case Ref: return "Ref";
      case Seq: return "Seq";
      case Choice: return "Choice";
      case Star: return "Star";
      case Plus: return "Plus";
      case Opt: return "Opt";
      case And: return "And";
      case Not: return "Not";
    }
    return std::format("PegOp({})", static_cast<int>(op));
  }
};

/// @brief Prints an algorithms::RoundMode as its enumerator name (`HalfToEven`, `Ceil`, ...).
template <>
struct std::formatter<datamunge::algorithms::RoundMode>
    : datamunge::format_detail::text_formatter<datamunge::algorithms::RoundMode> {
  /// @brief Maps the enumerator to its name.
  /// @param mode The rounding mode.
  /// @return The enumerator name, or `RoundMode(<int>)` for an out-of-range value.
  static std::string text(const datamunge::algorithms::RoundMode& mode) {
    using enum datamunge::algorithms::RoundMode;
    switch (mode) {
      case HalfToEven: return "HalfToEven";
      case HalfAwayFromZero: return "HalfAwayFromZero";
      case HalfUp: return "HalfUp";
      case HalfDown: return "HalfDown";
      case TowardZero: return "TowardZero";
      case Ceil: return "Ceil";
      case Floor: return "Floor";
    }
    return std::format("RoundMode({})", static_cast<int>(mode));
  }
};

/// @brief Prints an algorithms::SimplexStatus as `Optimal` or `Unbounded`.
template <>
struct std::formatter<datamunge::algorithms::SimplexStatus>
    : datamunge::format_detail::text_formatter<datamunge::algorithms::SimplexStatus> {
  /// @brief Maps the enumerator to its name.
  /// @param status The simplex termination status.
  /// @return The enumerator name, or `SimplexStatus(<int>)` for an out-of-range value.
  static std::string text(const datamunge::algorithms::SimplexStatus& status) {
    switch (status) {
      case datamunge::algorithms::SimplexStatus::Optimal: return "Optimal";
      case datamunge::algorithms::SimplexStatus::Unbounded: return "Unbounded";
    }
    return std::format("SimplexStatus({})", static_cast<int>(status));
  }
};

/// @brief Prints an algorithms::NagleEvent::Type as `Write` or `Ack`.
template <>
struct std::formatter<datamunge::algorithms::NagleEvent::Type>
    : datamunge::format_detail::text_formatter<datamunge::algorithms::NagleEvent::Type> {
  /// @brief Maps the enumerator to its name.
  /// @param type The event kind.
  /// @return The enumerator name, or `NagleEvent::Type(<int>)` for an out-of-range value.
  static std::string text(const datamunge::algorithms::NagleEvent::Type& type) {
    switch (type) {
      case datamunge::algorithms::NagleEvent::Write: return "Write";
      case datamunge::algorithms::NagleEvent::Ack: return "Ack";
    }
    return std::format("NagleEvent::Type({})", static_cast<int>(type));
  }
};

// ---- small value types -------------------------------------------------------------------------

/// @brief Prints an algorithms::Point2D as "Point2D(x, y)".
template <>
struct std::formatter<datamunge::algorithms::Point2D>
    : datamunge::format_detail::text_formatter<datamunge::algorithms::Point2D> {
  /// @brief The point's coordinates.
  /// @param p The point.
  /// @return "Point2D(x, y)".
  static std::string text(const datamunge::algorithms::Point2D& p) { return std::format("Point2D({}, {})", p.x, p.y); }
};

/// @brief Prints an algorithms::PlanarPoint as "PlanarPoint(x, y)".
template <>
struct std::formatter<datamunge::algorithms::PlanarPoint>
    : datamunge::format_detail::text_formatter<datamunge::algorithms::PlanarPoint> {
  /// @brief The point's coordinates.
  /// @param p The point.
  /// @return "PlanarPoint(x, y)".
  static std::string text(const datamunge::algorithms::PlanarPoint& p) {
    return std::format("PlanarPoint({}, {})", p.x, p.y);
  }
};

/// @brief Prints an algorithms::Pixel (scan-line fill output) as "Pixel(x, y)".
template <>
struct std::formatter<datamunge::algorithms::Pixel> : datamunge::format_detail::text_formatter<datamunge::algorithms::Pixel> {
  /// @brief The pixel's integer coordinates.
  /// @param p The pixel.
  /// @return "Pixel(x, y)".
  static std::string text(const datamunge::algorithms::Pixel& p) { return std::format("Pixel({}, {})", p.x, p.y); }
};

/// @brief Prints an algorithms::NagleEvent as "NagleEvent(Write, bytes=100)".
template <>
struct std::formatter<datamunge::algorithms::NagleEvent>
    : datamunge::format_detail::text_formatter<datamunge::algorithms::NagleEvent> {
  /// @brief The event kind and byte count.
  /// @param e The event.
  /// @return "NagleEvent(<Write|Ack>, bytes=<n>)".
  static std::string text(const datamunge::algorithms::NagleEvent& e) {
    return std::format("NagleEvent({}, bytes={})", e.type, e.bytes);
  }
};

/// @brief Prints a grammar algorithms::Production in BNF style, "Production(S -> a S b)" (empty right side: "ε").
template <>
struct std::formatter<datamunge::algorithms::Production>
    : datamunge::format_detail::text_formatter<datamunge::algorithms::Production> {
  /// @brief Renders "lhs -> rhs..." without the type name; shared with the ContextFreeGrammar listing.
  /// @param p The production.
  /// @return "lhs -> sym sym ..." (or "lhs -> ε").
  static std::string rule(const datamunge::algorithms::Production& p) {
    std::string out = p.lhs + " ->";
    if (p.rhs.empty()) return out + " ε";
    for (const auto& sym : p.rhs) out += " " + sym;
    return out;
  }
  /// @brief Wraps the rule in the type name.
  /// @param p The production.
  /// @return "Production(lhs -> rhs...)".
  static std::string text(const datamunge::algorithms::Production& p) { return "Production(" + rule(p) + ")"; }
};

/// @brief Prints an algorithms::ContextFreeGrammar as a header line plus one indented rule per line (at most 20).
template <>
struct std::formatter<datamunge::algorithms::ContextFreeGrammar>
    : datamunge::format_detail::text_formatter<datamunge::algorithms::ContextFreeGrammar> {
  /// @brief Lists the start symbol and productions.
  /// @param g The grammar.
  /// @return "ContextFreeGrammar(start=S, productions=N)" followed by "  S -> ..." lines.
  static std::string text(const datamunge::algorithms::ContextFreeGrammar& g) {
    constexpr std::size_t max_rules = 20;
    std::string out = std::format("ContextFreeGrammar(start={}, productions={})", g.start, g.productions.size());
    for (std::size_t i = 0; i < g.productions.size() && i < max_rules; ++i)
      out += "\n  " + std::formatter<datamunge::algorithms::Production>::rule(g.productions[i]);
    if (g.productions.size() > max_rules) out += std::format("\n  ... ({} more)", g.productions.size() - max_rules);
    return out;
  }
};

// ---- stateful classes --------------------------------------------------------------------------

/// @brief Prints an algorithms::MorrisCounter as "MorrisCounter(exponent=e, estimate=n)".
template <>
struct std::formatter<datamunge::algorithms::MorrisCounter>
    : datamunge::format_detail::text_formatter<datamunge::algorithms::MorrisCounter> {
  /// @brief The stored exponent and the count it estimates.
  /// @param c The counter.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::MorrisCounter& c) {
    return std::format("MorrisCounter(exponent={}, estimate={})", c.exponent(), c.estimate());
  }
};

/// @brief Prints an algorithms::BloomFilter as "BloomFilter(bits=m, hashes=k, fill_ratio=r)".
template <>
struct std::formatter<datamunge::algorithms::BloomFilter>
    : datamunge::format_detail::text_formatter<datamunge::algorithms::BloomFilter> {
  /// @brief The bit count, hash count and fraction of set bits.
  /// @param f The filter.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::BloomFilter& f) {
    return std::format("BloomFilter(bits={}, hashes={}, fill_ratio={})", f.num_bits(), f.num_hashes(), f.fill_ratio());
  }
};

/// @brief Prints an algorithms::BuddyAllocator's size, usage and free-list counts per order.
template <>
struct std::formatter<datamunge::algorithms::BuddyAllocator>
    : datamunge::format_detail::text_formatter<datamunge::algorithms::BuddyAllocator> {
  /// @brief Arena size, allocated/requested bytes and free blocks per order.
  /// @param a The allocator.
  /// @return "BuddyAllocator(size=..., min_block=..., allocated=..., requested=..., free_blocks=[...])".
  static std::string text(const datamunge::algorithms::BuddyAllocator& a) {
    using datamunge::format_detail::algorithms_fmt::value;
    return std::format("BuddyAllocator(size={}, min_block={}, allocated={}, requested={}, free_blocks={})",
                       a.block_size(a.max_order()), a.block_size(0), a.allocated_bytes(), a.requested_bytes(),
                       value(a.free_blocks_per_order()));
  }
};

/// @brief Prints an algorithms::KarnEstimator's retransmission-timeout state.
template <>
struct std::formatter<datamunge::algorithms::KarnEstimator>
    : datamunge::format_detail::text_formatter<datamunge::algorithms::KarnEstimator> {
  /// @brief The RTO, smoothed RTT and RTT variance.
  /// @param k The estimator.
  /// @return "KarnEstimator(rto=..., srtt=..., rttvar=..., initialized=...)".
  static std::string text(const datamunge::algorithms::KarnEstimator& k) {
    return std::format("KarnEstimator(rto={}, srtt={}, rttvar={}, initialized={})", k.rto(), k.srtt(), k.rttvar(),
                       k.initialized());
  }
};

/// @brief Prints an algorithms::LamportClock as "LamportClock(time=t)".
template <>
struct std::formatter<datamunge::algorithms::LamportClock>
    : datamunge::format_detail::text_formatter<datamunge::algorithms::LamportClock> {
  /// @brief The current logical time.
  /// @param c The clock.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::LamportClock& c) {
    return std::format("LamportClock(time={})", c.time());
  }
};

/// @brief Prints an algorithms::OnlineVariance's running count, mean and variances.
template <>
struct std::formatter<datamunge::algorithms::OnlineVariance>
    : datamunge::format_detail::text_formatter<datamunge::algorithms::OnlineVariance> {
  /// @brief Count, mean, population and sample variance (0 while too few values).
  /// @param s The accumulator.
  /// @return "OnlineVariance(n=..., mean=..., variance=..., sample_variance=...)".
  static std::string text(const datamunge::algorithms::OnlineVariance& s) {
    return std::format("OnlineVariance(n={}, mean={}, variance={}, sample_variance={})", s.count(), s.mean(),
                       s.variance(), s.sample_variance());
  }
};

/// @brief Prints an algorithms::OnlineCovariance's running count, means, covariance and correlation.
template <>
struct std::formatter<datamunge::algorithms::OnlineCovariance>
    : datamunge::format_detail::text_formatter<datamunge::algorithms::OnlineCovariance> {
  /// @brief Count, means, population covariance and correlation (0 while too few values).
  /// @param s The accumulator.
  /// @return "OnlineCovariance(n=..., mean_x=..., mean_y=..., covariance=..., correlation=...)".
  static std::string text(const datamunge::algorithms::OnlineCovariance& s) {
    return std::format("OnlineCovariance(n={}, mean_x={}, mean_y={}, covariance={}, correlation={})", s.count(),
                       s.mean_x(), s.mean_y(), s.covariance(), s.correlation());
  }
};

/// @brief Prints an algorithms::LL1Parser as "LL1Parser(ll1=true)" (whether the grammar's table is conflict-free).
template <>
struct std::formatter<datamunge::algorithms::LL1Parser>
    : datamunge::format_detail::text_formatter<datamunge::algorithms::LL1Parser> {
  /// @brief Whether the grammar is LL(1).
  /// @param p The parser.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::LL1Parser& p) {
    return std::format("LL1Parser(ll1={})", p.is_ll1());
  }
};

/// @brief Prints an algorithms::SLRParser as "SLRParser(slr=true, states=N)".
template <>
struct std::formatter<datamunge::algorithms::SLRParser>
    : datamunge::format_detail::text_formatter<datamunge::algorithms::SLRParser> {
  /// @brief Whether the grammar is SLR(1) and the LR(0) automaton's state count.
  /// @param p The parser.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::SLRParser& p) {
    return std::format("SLRParser(slr={}, states={})", p.is_slr(), p.state_count());
  }
};

/// @brief Prints an algorithms::PackratParser's memoisation counters from its last parse.
template <>
struct std::formatter<datamunge::algorithms::PackratParser>
    : datamunge::format_detail::text_formatter<datamunge::algorithms::PackratParser> {
  /// @brief Rule calls and memo hits.
  /// @param p The parser.
  /// @return "PackratParser(rule_calls=..., memo_hits=...)".
  static std::string text(const datamunge::algorithms::PackratParser& p) {
    return std::format("PackratParser(rule_calls={}, memo_hits={})", p.rule_calls(), p.memo_hits());
  }
};

/// @brief Prints an algorithms::SummedAreaTable as "SummedAreaTable(rows=r, cols=c, total=sum)".
template <>
struct std::formatter<datamunge::algorithms::SummedAreaTable>
    : datamunge::format_detail::text_formatter<datamunge::algorithms::SummedAreaTable> {
  /// @brief Grid shape and the sum over the whole grid (0 when empty).
  /// @param t The table.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::SummedAreaTable& t) {
    const double total = (t.rows() > 0 && t.cols() > 0) ? t.rect_sum(0, 0, t.rows() - 1, t.cols() - 1) : 0.0;
    return std::format("SummedAreaTable(rows={}, cols={}, total={})", t.rows(), t.cols(), total);
  }
};

/// @brief Prints an algorithms::LeakyBucket's configuration and stored (not re-leaked) level.
template <>
struct std::formatter<datamunge::algorithms::LeakyBucket>
    : datamunge::format_detail::text_formatter<datamunge::algorithms::LeakyBucket> {
  /// @brief Capacity, leak rate, and the level as of the last update (printing does not leak the bucket).
  /// @param b The bucket.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::LeakyBucket& b) {
    return std::format("LeakyBucket(capacity={}, leak_rate={}, level={}, last_time={})", b.capacity(), b.leak_rate(),
                       b.stored_level(), b.last_time());
  }
};

/// @brief Prints an algorithms::TokenBucket's configuration and stored (not re-filled) token count.
template <>
struct std::formatter<datamunge::algorithms::TokenBucket>
    : datamunge::format_detail::text_formatter<datamunge::algorithms::TokenBucket> {
  /// @brief Capacity, rate, and the tokens as of the last update (printing does not refill the bucket).
  /// @param b The bucket.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::TokenBucket& b) {
    return std::format("TokenBucket(capacity={}, rate={}, tokens={}, last_time={})", b.capacity(), b.rate(),
                       b.stored_tokens(), b.last_time());
  }
};

/// @brief Prints an algorithms::CosineLSH index's dimension, table count and bits per table.
template <>
struct std::formatter<datamunge::algorithms::CosineLSH>
    : datamunge::format_detail::text_formatter<datamunge::algorithms::CosineLSH> {
  /// @brief The index configuration.
  /// @param x The index.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::CosineLSH& x) {
    return std::format("CosineLSH(dim={}, tables={}, bits={})", x.dimension(), x.num_tables(), x.bits_per_table());
  }
};

/// @brief Prints an algorithms::Rc4 keystream generator as just "Rc4" (its state is secret and the key length is
/// not stored).
template <>
struct std::formatter<datamunge::algorithms::Rc4> : datamunge::format_detail::text_formatter<datamunge::algorithms::Rc4> {
  /// @brief The type name (no state is public).
  /// @param x The object (unused).
  /// @return "Rc4".
  static std::string text([[maybe_unused]] const datamunge::algorithms::Rc4& x) { return "Rc4"; }
};

/// @brief Prints an algorithms::ZigguratNormal sampler's layer count and tail boundary.
template <>
struct std::formatter<datamunge::algorithms::ZigguratNormal>
    : datamunge::format_detail::text_formatter<datamunge::algorithms::ZigguratNormal> {
  /// @brief The strip count and tail start r.
  /// @param x The sampler.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::ZigguratNormal& x) {
    return std::format("ZigguratNormal(layers={}, tail={})", x.layers(), x.tail_boundary());
  }
};

// ---- plain result / option / data structs ------------------------------------------------------

#include <datamunge/format/algorithms/records.hpp>
