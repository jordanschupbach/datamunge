/// @file geohash.cpp
/// @brief Shared ordered double encoder, actual state capture and strict geohash cell utilities.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/geohash.hpp>
#include <limits>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Implementation details for deterministic longitude-first geohashes.
namespace {
/// @brief Legacy lowercase alphabet indexed by a completed five-bit accumulator.
constexpr std::string_view alphabet = "0123456789bcdefghjkmnpqrstuvwxyz";
/// @brief Validate finite inclusive coordinate domains before any numeric work.
/// @param lat Latitude in degrees.
/// @param lon Longitude in degrees, with both end-meridians allowed distinctly.
/// @throws std::invalid_argument Nonfinite or out-of-range coordinates.
void coordinates(double lat, double lon) {
  if (!std::isfinite(lat) || !std::isfinite(lon) || lat < -90 || lat > 90 || lon < -180 ||
      lon > 180)
    throw std::invalid_argument(
        "geohash: finite coordinates in latitude/longitude ranges required");
}
/// @brief Check a nonnegative analysis count before casting.
/// @param precision Requested output characters.
/// @throws std::invalid_argument Negative analysis count.
void count(int precision) {
  if (precision < 0)
    throw std::invalid_argument("geohash: negative analysis precision");
}
/// @brief Multiply complete history dimensions without wrapping.
/// @param a First dimension.
/// @param b Second dimension.
/// @return Representable product.
/// @throws std::length_error Product cannot fit size_t.
std::size_t multiply(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("geohash: count overflow");
  return a * b;
}
/// @brief Increment an actual executed-work counter.
/// @param value Counter updated in the current call.
/// @throws std::length_error Next actual count cannot fit size_t.
void increment(std::size_t& value) {
  if (value == std::numeric_limits<std::size_t>::max())
    throw std::length_error("geohash: work overflow");
  ++value;
}
/// @brief Map a strict lowercase alphabet byte to its symbol value.
/// @param c Byte to inspect, including possible embedded NUL/high bytes.
/// @return Value0..31.
/// @throws std::invalid_argument Byte is outside the complete alphabet.
int symbol(char c) {
  const auto index = alphabet.find(c);
  if (index == std::string_view::npos)
    throw std::invalid_argument("geohash: invalid alphabet byte");
  return static_cast<int>(index);
}
/// @brief Validate an entire key before any decoding/comparison arithmetic.
/// @param key Strict alphabet key, possibly empty.
/// @throws std::invalid_argument Any invalid byte.
void key_input(std::string_view key) {
  for (char c : key)
    (void)symbol(c);
}
/// @brief Apply one already-decided bit to the actual selected interval.
/// @param bounds Current endpoints, mutated only along the selected axis.
/// @param longitude Select longitude rather than latitude.
/// @param midpoint Actual double midpoint used by this bit.
/// @param upper Actual comparison decision or decoded bit.
/// @return Owned primitive endpoints and decision; caller attaches step/group metadata.
GeohashBit update(GeohashBounds& bounds, bool longitude, double midpoint, bool upper) {
  auto&      lo = longitude ? bounds.lon_lo : bounds.lat_lo;
  auto&      hi = longitude ? bounds.lon_hi : bounds.lat_hi;
  GeohashBit bit;
  bit.longitude    = longitude;
  bit.lower_before = lo;
  bit.upper_before = hi;
  bit.midpoint     = midpoint;
  bit.upper        = upper;
  if (upper)
    lo = midpoint;
  else
    hi = midpoint;
  bit.lower_after = lo;
  bit.upper_after = hi;
  return bit;
}
/// @brief Execute the one legacy-ordered double recurrence with optional actual capture.
/// @param lat Validated latitude.
/// @param lon Validated longitude.
/// @param precision Nonnegative character count.
/// @param trace Optional whole-admitted trace, already initialized with one initial record.
/// @return Owned string, actual bounds/work and first precision-limit flags.
/// @throws std::length_error Actual counters or output/container capacities overflow.
GeohashRun encode(double lat, double lon, int precision, GeohashTrace* trace = nullptr) {
  GeohashRun out;
  if (static_cast<std::size_t>(precision) > out.hash.max_size())
    throw std::length_error("geohash: output string capacity");
  bool longitude = true;
  int  bit = 0, ch = 0;
  while (static_cast<int>(out.hash.size()) < precision) {
    const double lo       = longitude ? out.bounds.lon_lo : out.bounds.lat_lo,
                 hi       = longitude ? out.bounds.lon_hi : out.bounds.lat_hi;
    const double midpoint = (lo + hi) / 2.0;
    const bool   upper    = (longitude ? lon : lat) >= midpoint;
    if (upper)
      ch = (ch << 1) | 1;
    else
      ch = (ch << 1);
    auto actual = update(out.bounds, longitude, midpoint, upper);
    increment(out.work.bisections);
    if (longitude)
      increment(out.work.longitude_steps);
    else
      increment(out.work.latitude_steps);
    actual.step          = out.work.bisections;
    actual.symbol_value  = ch;
    actual.bit_in_symbol = bit;
    if (actual.lower_before == actual.lower_after && actual.upper_before == actual.upper_after) {
      increment(out.work.unchanged_intervals);
      if (!out.first_unchanged_bit)
        out.first_unchanged_bit = actual.step;
    }
    if (actual.lower_after == actual.upper_after) {
      increment(out.work.collapsed_intervals);
      if (!out.first_collapsed_bit)
        out.first_collapsed_bit = actual.step;
    }
    if (trace) {
      trace->bits.push_back(actual);
      ++trace->captured_records;
    }
    longitude = !longitude;
    if (++bit == 5) {
      out.hash += alphabet[static_cast<std::size_t>(ch)];
      increment(out.work.emissions);
      if (trace) {
        trace->emissions.push_back(
            {out.work.bisections, out.hash.size(), ch, out.hash.back(), out.bounds});
        ++trace->captured_records;
      }
      bit = 0;
      ch  = 0;
    }
  }
  return out;
}
/// @brief Round one ideal dyadic span, without narrowing a huge exponent unsafely.
/// @param initial Whole-world angular span in degrees.
/// @param bits Count of ideal halvings.
/// @return Rounded nonnegative double span, zero after exponent underflow.
double span(double initial, std::size_t bits) {
  if (bits > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    return 0;
  return std::ldexp(initial, -static_cast<int>(bits));
}
} // namespace

std::string geohash_encode(double lat, double lon, int precision) {
  coordinates(lat, lon);
  return encode(lat, lon, std::max(0, precision)).hash;
}
GeohashRun geohash_encode_run(double lat, double lon, int precision) {
  coordinates(lat, lon);
  count(precision);
  return encode(lat, lon, precision);
}
std::size_t geohash_trace_records(int precision) {
  count(precision);
  const auto records = multiply(static_cast<std::size_t>(precision), 6);
  if (records == std::numeric_limits<std::size_t>::max())
    throw std::length_error("geohash: initial record overflow");
  return records + 1;
}
GeohashTrace geohash_encode_trace(double lat, double lon, int precision, std::size_t max_records) {
  coordinates(lat, lon);
  count(precision);
  const auto records = geohash_trace_records(precision);
  if (max_records && records > max_records)
    throw std::length_error("geohash: whole record cap");
  if (multiply(static_cast<std::size_t>(precision), 5) > std::vector<GeohashBit>{}.max_size() ||
      static_cast<std::size_t>(precision) > std::vector<GeohashEmission>{}.max_size())
    throw std::length_error("geohash: trace vector capacity");
  GeohashTrace out;
  out.latitude         = lat;
  out.longitude        = lon;
  out.captured_records = 1;
  out.run              = encode(lat, lon, precision, &out);
  return out;
}
GeohashCell geohash_decode(std::string_view hash) {
  key_input(hash);
  (void)multiply(hash.size(), 5);
  GeohashCell out;
  bool        longitude = true;
  for (char c : hash) {
    const auto value = symbol(c);
    for (int position = 4; position >= 0; --position) {
      const double lo = longitude ? out.bounds.lon_lo : out.bounds.lat_lo,
                   hi = longitude ? out.bounds.lon_hi : out.bounds.lat_hi;
      const auto actual =
          update(out.bounds, longitude, (lo + hi) / 2.0, ((value >> position) & 1) != 0);
      if ((actual.lower_after == actual.lower_before &&
           actual.upper_after == actual.upper_before) ||
          actual.lower_after == actual.upper_after)
        out.limited = true;
      if (longitude)
        ++out.longitude_bits;
      else
        ++out.latitude_bits;
      longitude = !longitude;
    }
  }
  out.latitude       = (out.bounds.lat_lo + out.bounds.lat_hi) / 2.0;
  out.longitude      = (out.bounds.lon_lo + out.bounds.lon_hi) / 2.0;
  out.latitude_span  = out.bounds.lat_hi - out.bounds.lat_lo;
  out.longitude_span = out.bounds.lon_hi - out.bounds.lon_lo;
  return out;
}
GeohashResolution geohash_resolution(int precision) {
  count(precision);
  const auto        total = multiply(static_cast<std::size_t>(precision), 5);
  GeohashResolution out;
  out.latitude_bits     = total / 2;
  out.longitude_bits    = total / 2 + total % 2;
  out.latitude_degrees  = span(180, out.latitude_bits);
  out.longitude_degrees = span(360, out.longitude_bits);
  out.spans_positive    = out.latitude_degrees > 0 && out.longitude_degrees > 0;
  return out;
}
std::size_t geohash_common_prefix(std::string_view a, std::string_view b) {
  key_input(a);
  key_input(b);
  std::size_t n = 0;
  while (n < std::min(a.size(), b.size()) && a[n] == b[n])
    ++n;
  return n;
}
} // namespace datamunge::algorithms
