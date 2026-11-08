#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <datamunge/algorithms/geohash.hpp>
#include <limits>
#include <random>
#include <stdexcept>

#include "../oracles/geohash_fraction.hpp"

namespace {
using namespace datamunge::algorithms;
constexpr char alphabet[] = "0123456789bcdefghjkmnpqrstuvwxyz";
TEST(GeohashTrace, HandCheckedFirstSymbolShowsActualMidpointsAndAccumulator) {
  const auto t = geohash_encode_trace(42.6, -5.6, 1, 7);
  ASSERT_EQ(t.bits.size(), 5);
  ASSERT_EQ(t.emissions.size(), 1);
  const std::array<double, 5> mid{0, 0, -90, 45, -45};
  const std::array<bool, 5>   upper{false, true, true, false, true};
  const std::array<int, 5>    acc{0, 1, 3, 6, 13};
  for (std::size_t i = 0; i < 5; ++i) {
    const auto& e = t.bits[i];
    EXPECT_EQ(e.step, i + 1);
    EXPECT_EQ(e.longitude, i % 2 == 0);
    EXPECT_DOUBLE_EQ(e.midpoint, mid[i]);
    EXPECT_EQ(e.upper, upper[i]);
    EXPECT_EQ(e.symbol_value, acc[i]);
    EXPECT_EQ(e.bit_in_symbol, i);
  }
  const auto& e = t.emissions[0];
  EXPECT_EQ(e.bit_step, 5);
  EXPECT_EQ(e.characters, 1);
  EXPECT_EQ(e.symbol_value, 13);
  EXPECT_EQ(e.symbol, 'e');
  EXPECT_EQ(t.run.hash, "e");
  EXPECT_EQ(t.captured_records, 7);
  EXPECT_DOUBLE_EQ(e.bounds.lon_lo, -45);
  EXPECT_DOUBLE_EQ(e.bounds.lon_hi, 0);
  EXPECT_DOUBLE_EQ(e.bounds.lat_lo, 0);
  EXPECT_DOUBLE_EQ(e.bounds.lat_hi, 45);
  EXPECT_EQ(t.run.work.longitude_steps, 3);
  EXPECT_EQ(t.run.work.latitude_steps, 2);
}
TEST(GeohashTrace, IndependentExactRationalFixturesCheckEncodingAndDecodedCell) {
  for (const auto& d : geohash_oracle::data) {
    const auto r = geohash_encode_run(d.latitude, d.longitude, d.precision);
    EXPECT_EQ(r.hash, d.hash);
    const auto c = geohash_decode(d.hash);
    EXPECT_DOUBLE_EQ(c.bounds.lat_lo, d.lat_lo);
    EXPECT_DOUBLE_EQ(c.bounds.lat_hi, d.lat_hi);
    EXPECT_DOUBLE_EQ(c.bounds.lon_lo, d.lon_lo);
    EXPECT_DOUBLE_EQ(c.bounds.lon_hi, d.lon_hi);
    EXPECT_FALSE(c.limited);
    EXPECT_DOUBLE_EQ(r.bounds.lat_lo, d.lat_lo);
    EXPECT_DOUBLE_EQ(r.bounds.lat_hi, d.lat_hi);
    EXPECT_DOUBLE_EQ(r.bounds.lon_lo, d.lon_lo);
    EXPECT_DOUBLE_EQ(r.bounds.lon_hi, d.lon_hi);
    EXPECT_EQ(geohash_encode(c.latitude, c.longitude, d.precision), d.hash);
    EXPECT_LE(c.bounds.lat_lo, d.latitude);
    EXPECT_GE(c.bounds.lat_hi, d.latitude);
    EXPECT_LE(c.bounds.lon_lo, d.longitude);
    EXPECT_GE(c.bounds.lon_hi, d.longitude);
  }
}
TEST(GeohashTrace, ExhaustiveThreeCharacterMortonIndicesGiveIndependentTiling) {
  for (unsigned packed = 0; packed < 32768; ++packed) {
    const std::string key{
        alphabet[(packed >> 10) & 31], alphabet[(packed >> 5) & 31], alphabet[packed & 31]};
    unsigned x = 0, y = 0;
    for (int i = 14; i >= 0; --i) {
      if (i % 2 == 0)
        x = (x << 1) | ((packed >> i) & 1);
      else
        y = (y << 1) | ((packed >> i) & 1);
    }
    const auto c = geohash_decode(key);
    EXPECT_DOUBLE_EQ(c.bounds.lon_lo, -180 + x * (360. / 256));
    EXPECT_DOUBLE_EQ(c.bounds.lon_hi, -180 + (x + 1) * (360. / 256));
    EXPECT_DOUBLE_EQ(c.bounds.lat_lo, -90 + y * (180. / 128));
    EXPECT_DOUBLE_EQ(c.bounds.lat_hi, -90 + (y + 1) * (180. / 128));
    EXPECT_DOUBLE_EQ(c.longitude_span, 360. / 256);
    EXPECT_DOUBLE_EQ(c.latitude_span, 180. / 128);
    EXPECT_EQ(c.longitude_bits, 8);
    EXPECT_EQ(c.latitude_bits, 7);
    EXPECT_EQ(geohash_encode(c.latitude, c.longitude, 3), key);
  }
}
TEST(GeohashTrace, CoordinatesCountsAndWholeQuotaAreCheckedInDocumentedOrder) {
  const double inf = std::numeric_limits<double>::infinity(),
               nan = std::numeric_limits<double>::quiet_NaN();
  for (const auto p : std::array<std::array<double, 2>, 8>{
           {{91, 0}, {-91, 0}, {0, 181}, {0, -181}, {nan, 0}, {0, nan}, {inf, 0}, {0, -inf}}}) {
    EXPECT_THROW(geohash_encode(p[0], p[1], 0), std::invalid_argument);
    EXPECT_THROW(geohash_encode(p[0], p[1], -1), std::invalid_argument);
    EXPECT_THROW(geohash_encode_trace(p[0], p[1], 1, 1), std::invalid_argument);
  }
  EXPECT_EQ(geohash_encode(0, 0, std::numeric_limits<int>::min()), "");
  EXPECT_EQ(geohash_encode(0, 0, -1), "");
  EXPECT_EQ(geohash_encode(0, 0, 0), "");
  EXPECT_THROW(geohash_encode_run(0, 0, -1), std::invalid_argument);
  EXPECT_THROW(geohash_encode_trace(0, 0, -1, 1), std::invalid_argument);
  EXPECT_THROW(geohash_trace_records(-1), std::invalid_argument);
  EXPECT_THROW(geohash_resolution(-1), std::invalid_argument);
  EXPECT_THROW(geohash_encode_trace(0, 0, std::numeric_limits<int>::max(), 1), std::length_error);
  if constexpr (sizeof(std::size_t) >= 8)
    EXPECT_EQ(
        geohash_trace_records(std::numeric_limits<int>::max()),
        1 + 6 * static_cast<std::size_t>(std::numeric_limits<int>::max()));
}
TEST(GeohashTrace, StrictAlphabetFullValidationAndUnequalPrefixLengths) {
  for (const std::string bad :
       {std::string("EZ"),
        std::string("a"),
        std::string("i"),
        std::string("l"),
        std::string("o"),
        std::string("0!"),
        std::string("0\0z", 3),
        std::string(1, char(255))}) {
    EXPECT_THROW(geohash_decode(bad), std::invalid_argument);
    EXPECT_THROW(geohash_common_prefix("z", bad), std::invalid_argument);
    EXPECT_THROW(geohash_common_prefix(bad, ""), std::invalid_argument);
  }
  EXPECT_EQ(geohash_common_prefix("", ""), 0);
  EXPECT_EQ(geohash_common_prefix("ez", "ezs42"), 2);
  EXPECT_EQ(geohash_common_prefix("gcpvj0d", "gcpuzgq"), 3);
  EXPECT_EQ(geohash_common_prefix("ezs42", "ezs42"), 5);
  const auto world = geohash_decode("");
  EXPECT_DOUBLE_EQ(world.latitude, 0);
  EXPECT_DOUBLE_EQ(world.longitude, 0);
  EXPECT_DOUBLE_EQ(world.latitude_span, 180);
  EXPECT_DOUBLE_EQ(world.longitude_span, 360);
  EXPECT_EQ(world.longitude_bits, 0);
  EXPECT_FALSE(world.limited);
}
TEST(GeohashTrace, BoundaryProximityDoesNotImplyPrefixAndEndMeridiansStayDistinct) {
  const auto tiny = std::numeric_limits<double>::denorm_min();
  EXPECT_EQ(geohash_encode(0, 0, 1), "s");
  EXPECT_EQ(geohash_encode(0, -tiny, 1), "e");
  EXPECT_EQ(geohash_common_prefix(geohash_encode(0, tiny, 12), geohash_encode(0, -tiny, 12)), 0);
  EXPECT_EQ(geohash_common_prefix(geohash_encode(0, -180, 12), geohash_encode(0, 180, 12)), 0);
  EXPECT_EQ(geohash_encode(-90, -180, 12), std::string(12, '0'));
  EXPECT_EQ(geohash_encode(90, 180, 12), std::string(12, 'z'));
  EXPECT_EQ(geohash_encode(-0., -0., 12), geohash_encode(0., 0., 12));
  for (int p = 0; p <= 64; ++p) {
    const auto key = geohash_encode(42.6, -5.6, p);
    EXPECT_EQ(geohash_common_prefix(key, geohash_encode(42.6, -5.6, p + 1)), key.size());
  }
}
TEST(GeohashTrace, IdealAngularCountsAlternateAxesAndRemainDistinctFromActualDoubleWidths) {
  double prior_lon = 360, prior_lat = 180;
  for (int p = 0; p <= 24; ++p) {
    const auto r = geohash_resolution(p);
    EXPECT_EQ(r.longitude_bits, (5 * p + 1) / 2);
    EXPECT_EQ(r.latitude_bits, 5 * p / 2);
    EXPECT_DOUBLE_EQ(r.longitude_degrees, std::ldexp(360., -int(r.longitude_bits)));
    EXPECT_DOUBLE_EQ(r.latitude_degrees, std::ldexp(180., -int(r.latitude_bits)));
    EXPECT_TRUE(r.spans_positive);
    if (p) {
      EXPECT_DOUBLE_EQ(prior_lon / r.longitude_degrees, p % 2 ? 8 : 4);
      EXPECT_DOUBLE_EQ(prior_lat / r.latitude_degrees, p % 2 ? 4 : 8);
      EXPECT_DOUBLE_EQ((prior_lon * prior_lat) / (r.longitude_degrees * r.latitude_degrees), 32);
    }
    if (p % 2)
      EXPECT_DOUBLE_EQ(r.longitude_degrees, r.latitude_degrees);
    else
      EXPECT_DOUBLE_EQ(r.longitude_degrees, 2 * r.latitude_degrees);
    prior_lon = r.longitude_degrees;
    prior_lat = r.latitude_degrees;
  }
  EXPECT_DOUBLE_EQ(geohash_resolution(5).longitude_degrees, .0439453125);
  const auto ideal = geohash_resolution(1024);
  EXPECT_FALSE(ideal.spans_positive);
  EXPECT_DOUBLE_EQ(ideal.longitude_degrees, 0);
  EXPECT_DOUBLE_EQ(ideal.latitude_degrees, 0);
  const auto actual = geohash_decode(geohash_encode(42.6, -5.6, 1024));
  EXPECT_TRUE(actual.limited);
  EXPECT_GT(actual.longitude_span, 0);
  EXPECT_GT(actual.latitude_span, 0);
}
TEST(GeohashTrace, ActualLongSuffixKeepsAllRequestedWorkAndFlagsStagnationHonestly) {
  const auto t = geohash_encode_trace(42.6, -5.6, 1024, 6145);
  EXPECT_EQ(t.run.hash.size(), 1024);
  EXPECT_EQ(t.bits.size(), 5120);
  EXPECT_EQ(t.emissions.size(), 1024);
  EXPECT_EQ(t.captured_records, 6145);
  EXPECT_EQ(t.run.work.bisections, 5120);
  EXPECT_EQ(t.run.work.longitude_steps, 2560);
  EXPECT_EQ(t.run.work.latitude_steps, 2560);
  EXPECT_EQ(t.run.first_unchanged_bit, 110);
  EXPECT_EQ(t.run.first_collapsed_bit, 0);
  EXPECT_GT(t.run.work.unchanged_intervals, 0);
  const auto& first = t.bits[109];
  EXPECT_FALSE(first.longitude);
  EXPECT_DOUBLE_EQ(first.lower_before, first.lower_after);
  EXPECT_DOUBLE_EQ(first.upper_before, first.upper_after);
  EXPECT_NE(first.lower_after, first.upper_after);
  EXPECT_THROW(geohash_encode_trace(42.6, -5.6, 1024, 6144), std::length_error);
  const auto poles = geohash_encode_run(90, 180, 1024);
  EXPECT_GT(poles.first_collapsed_bit, 0);
  EXPECT_DOUBLE_EQ(poles.bounds.lat_lo, poles.bounds.lat_hi);
  EXPECT_DOUBLE_EQ(poles.bounds.lon_lo, poles.bounds.lon_hi);
  EXPECT_EQ(poles.hash, std::string(1024, 'z'));
  const auto longrun = geohash_encode_run(42.6, -5.6, 4096);
  EXPECT_EQ(longrun.hash.substr(0, 1024), t.run.hash);
  EXPECT_EQ(longrun.work.bisections, 20480);
  EXPECT_EQ(longrun.work.emissions, 4096);
}
TEST(GeohashTrace, WholeRecordCapsZeroCountAndOwnedPrimitiveRecords) {
  EXPECT_EQ(geohash_trace_records(0), 1);
  EXPECT_EQ(geohash_trace_records(1), 7);
  EXPECT_THROW(geohash_encode_trace(42.6, -5.6, 1, 6), std::length_error);
  EXPECT_NO_THROW(geohash_encode_trace(42.6, -5.6, 1, 7));
  EXPECT_NO_THROW(geohash_encode_trace(42.6, -5.6, 1, 0));
  auto       t                 = geohash_encode_trace(42.6, -5.6, 2, 13);
  const auto final             = t.run.bounds;
  const auto key               = t.run.hash;
  t.bits[0].lower_after        = 999;
  t.emissions[0].bounds.lat_hi = 999;
  t.initial_bounds.lon_lo      = 999;
  EXPECT_EQ(t.run.hash, key);
  EXPECT_DOUBLE_EQ(t.run.bounds.lat_hi, final.lat_hi);
  EXPECT_NE(t.emissions[1].bounds.lat_hi, 999);
  EXPECT_NE(t.bits[1].lower_before, 999);
  const auto zero = geohash_encode_trace(42.6, -5.6, 0, 1);
  EXPECT_EQ(zero.captured_records, 1);
  EXPECT_TRUE(zero.bits.empty());
  EXPECT_TRUE(zero.emissions.empty());
  EXPECT_TRUE(zero.run.hash.empty());
  EXPECT_DOUBLE_EQ(zero.initial_bounds.lon_lo, -180);
  EXPECT_EQ(zero.run.work.bisections, 0);
}
TEST(GeohashTrace, ActualBitAndEmissionCertificatesAgreeOnBoundsCountersAndGroups) {
  std::mt19937_64                        rng(123);
  std::uniform_real_distribution<double> u(-1, 1);
  for (int trial = 0; trial < 1000; ++trial) {
    const double lat = 90 * u(rng), lon = 180 * u(rng);
    const int    p = 1 + trial % 24;
    const auto   t = geohash_encode_trace(lat, lon, p, 0);
    ASSERT_EQ(t.bits.size(), 5 * p);
    ASSERT_EQ(t.emissions.size(), p);
    EXPECT_EQ(t.captured_records, 1 + 6 * p);
    GeohashBounds previous;
    std::size_t   emissions = 0;
    for (const auto& e : t.bits) {
      const double coordinate = e.longitude ? lon : lat;
      EXPECT_LE(e.lower_before, coordinate);
      EXPECT_GE(e.upper_before, coordinate);
      EXPECT_LE(e.lower_after, coordinate);
      EXPECT_GE(e.upper_after, coordinate);
      EXPECT_EQ(e.upper, coordinate >= e.midpoint);
      if (e.longitude) {
        EXPECT_DOUBLE_EQ(e.lower_before, previous.lon_lo);
        EXPECT_DOUBLE_EQ(e.upper_before, previous.lon_hi);
        previous.lon_lo = e.lower_after;
        previous.lon_hi = e.upper_after;
      } else {
        EXPECT_DOUBLE_EQ(e.lower_before, previous.lat_lo);
        EXPECT_DOUBLE_EQ(e.upper_before, previous.lat_hi);
        previous.lat_lo = e.lower_after;
        previous.lat_hi = e.upper_after;
      }
      if (e.bit_in_symbol == 4) {
        const auto& c = t.emissions[emissions++];
        EXPECT_EQ(c.bit_step, e.step);
        EXPECT_EQ(c.symbol_value, e.symbol_value);
        EXPECT_EQ(c.symbol, alphabet[e.symbol_value]);
        EXPECT_EQ(c.symbol, t.run.hash[c.characters - 1]);
        EXPECT_DOUBLE_EQ(c.bounds.lon_lo, previous.lon_lo);
        EXPECT_DOUBLE_EQ(c.bounds.lat_hi, previous.lat_hi);
      }
    }
    EXPECT_EQ(t.run.work.bisections, 5 * p);
    EXPECT_EQ(t.run.work.emissions, p);
    EXPECT_EQ(t.run.work.longitude_steps, (5 * p + 1) / 2);
    EXPECT_EQ(t.run.work.latitude_steps, 5 * p / 2);
    EXPECT_DOUBLE_EQ(t.run.bounds.lon_lo, previous.lon_lo);
    EXPECT_DOUBLE_EQ(t.run.bounds.lat_hi, previous.lat_hi);
    EXPECT_EQ(t.run.hash, geohash_encode(lat, lon, p));
    const auto decoded = geohash_decode(t.run.hash);
    EXPECT_DOUBLE_EQ(decoded.bounds.lon_lo, previous.lon_lo);
    EXPECT_DOUBLE_EQ(decoded.bounds.lat_hi, previous.lat_hi);
  }
}
} // namespace
