#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <datamunge/geometry/geometric_hashing.hpp>
#include <limits>
#include <map>
#include <numbers>
#include <random>
#include <tuple>

#include "../oracles/geometric_hash_fraction.hpp"

namespace geo = datamunge::geometry;
namespace {
using Entry = std::tuple<std::uint64_t, int, int>;
std::vector<Entry> entries(const geo::GeoHashTable& t) {
  std::vector<Entry> v;
  for (const auto& [key, ids] : t.bins)
    for (auto [i, j] : ids)
      v.emplace_back(key, i, j);
  std::sort(v.begin(), v.end());
  return v;
}
std::int64_t floor_div(std::int64_t n, std::int64_t d) {
  const auto q = n / d, r = n % d;
  return q - (r < 0);
}
std::uint64_t pack(std::int64_t u, std::int64_t v) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(u)) << 32) |
         static_cast<std::uint32_t>(v);
}
std::vector<Entry> integer_table(const std::vector<std::array<int, 2>>& pts, int denominator) {
  std::vector<Entry> out;
  for (int i = 0; i < int(pts.size()); ++i)
    for (int j = 0; j < int(pts.size()); ++j) {
      if (i == j)
        continue;
      const std::int64_t dx = pts[j][0] - pts[i][0], dy = pts[j][1] - pts[i][1],
                         den = dx * dx + dy * dy;
      if (!den)
        continue;
      for (int k = 0; k < int(pts.size()); ++k) {
        if (k == i || k == j)
          continue;
        const std::int64_t ax = pts[k][0] - pts[i][0], ay = pts[k][1] - pts[i][1];
        out.emplace_back(
            pack(
                floor_div((ax * dx + ay * dy) * denominator, den),
                floor_div((ay * dx - ax * dy) * denominator, den)),
            i,
            j);
      }
    }
  std::sort(out.begin(), out.end());
  return out;
}
const std::vector<geo::Point2D> tiny{{0, 0}, {2, 0}, {0, 1}};
} // namespace
TEST(GeometricHashChecked, HandFrameTransformAndReflection) {
  const auto f = geo::geometric_hash_frame({1, 1}, {0, 0}, {2, 0});
  ASSERT_TRUE(f.valid);
  EXPECT_EQ(f.u, .5);
  EXPECT_EQ(f.v, .5);
  EXPECT_EQ(f.length_squared, 4);
  const auto s =
      geo::geometric_transform({{0, 0}, {2, 0}, {1, 1}}, 2, std::numbers::pi / 2, {4, -3});
  const auto sf = geo::geometric_hash_frame(s[2], s[0], s[1]);
  EXPECT_NEAR(sf.u, f.u, 1e-15);
  EXPECT_NEAR(sf.v, f.v, 1e-15);
  EXPECT_EQ(geo::geometric_hash_frame({1, -1}, {0, 0}, {2, 0}).v, -.5);
  EXPECT_FALSE(geo::geometric_hash_frame({1e-10, 1e-10}, {0, 0}, {2e-10, 0}).valid);
  const auto copied = geo::geometric_transform(tiny, 2, 0, {4, -3});
  EXPECT_EQ(copied[1].x, 8);
  EXPECT_EQ(tiny[1].x, 2);
  double u = 7, v = 8;
  EXPECT_FALSE(geo::detail::gh_frame_coord({1, 1}, {0, 0}, {0, 0}, u, v));
  EXPECT_EQ(u, 7);
  EXPECT_EQ(v, 8);
}
TEST(GeometricHashChecked, IndependentExactRationalFixtures) {
  ASSERT_GT(geometric_hash_fraction::cases.size(), 1000u);
  for (const auto& c : geometric_hash_fraction::cases) {
    const auto f = geo::geometric_hash_frame({c.px, c.py}, {c.ax, c.ay}, {c.bx, c.by});
    ASSERT_TRUE(f.valid);
    EXPECT_EQ(f.length_squared, c.length_squared);
    EXPECT_EQ(f.u, c.u);
    EXPECT_EQ(f.v, c.v);
    const auto b = geo::geometric_hash_bin(f.u, f.v, c.quant);
    EXPECT_EQ(b.u, c.cell_u);
    EXPECT_EQ(b.v, c.cell_v);
    EXPECT_EQ(b.key, pack(c.cell_u, c.cell_v));
  }
}
TEST(GeometricHashChecked, IndependentIntegerTablesAndMultiplicity) {
  std::mt19937 rng(133);
  for (int trial = 0; trial < 500; ++trial) {
    const int                       n = trial % 9;
    std::vector<std::array<int, 2>> raw;
    std::vector<geo::Point2D>       m;
    for (int i = 0; i < n; ++i) {
      raw.push_back({int(rng() % 17) - 8, int(rng() % 17) - 8});
      m.push_back({raw.back()[0] / 4., raw.back()[1] / 4.});
    }
    for (int denominator : {1, 2, 4, 8}) {
      const auto r = geo::build_geometric_hash_run(m, 1. / denominator);
      EXPECT_EQ(entries(r.table), integer_table(raw, denominator));
      EXPECT_EQ(r.work.frame_attempts, n < 3 ? 0u : std::size_t(n) * (n - 1) * (n - 2));
      EXPECT_EQ(r.work.insertions + r.work.degenerate_frames, r.work.frame_attempts);
      EXPECT_EQ(r.work.vote_updates, 0u);
    }
  }
}
TEST(GeometricHashChecked, ExactBuildHistoryQuotaAndOwnedState) {
  const auto t = geo::build_geometric_hash_trace(tiny, .25, 12);
  ASSERT_EQ(t.events.size(), 12u);
  EXPECT_EQ(t.run.work.frame_attempts, 6u);
  EXPECT_EQ(t.run.work.insertions, 6u);
  EXPECT_THROW(geo::build_geometric_hash_trace(tiny, .25, 11), std::length_error);
  EXPECT_EQ(geo::build_geometric_hash_trace(tiny, .25, 0).events.size(), 12u);
  std::map<std::uint64_t, std::size_t> sizes;
  for (std::size_t i = 0; i < t.events.size(); i += 2) {
    const auto& f = t.events[i];
    const auto& e = t.events[i + 1];
    EXPECT_EQ(f.kind, geo::GeoHashEventKind::build_frame);
    EXPECT_TRUE(f.accepted);
    EXPECT_EQ(e.kind, geo::GeoHashEventKind::insert);
    EXPECT_EQ(e.first, f.first);
    EXPECT_EQ(e.second, f.second);
    EXPECT_EQ(e.point, f.point);
    EXPECT_EQ(e.key, geo::geometric_hash_bin(f.u, f.v, .25).key);
    EXPECT_EQ(e.count, ++sizes[e.key]);
  }
  auto copy        = t;
  copy.events[0].u = 999;
  copy.run.table.bins.clear();
  EXPECT_NE(t.events[0].u, 999);
  EXPECT_EQ(t.run.table.bins.size(), 6u);
  std::vector<geo::Point2D> dup{{0, 0}, {0, 0}, {1, 0}};
  const auto                d = geo::build_geometric_hash_trace(dup, .25);
  EXPECT_EQ(d.run.work.degenerate_frames, 2u);
  EXPECT_EQ(d.events.size(), 10u);
}
TEST(GeometricHashChecked, ActualRecognitionWorkAndHistoryCertificates) {
  std::mt19937 rng(1337);
  for (int trial = 0; trial < 120; ++trial) {
    std::vector<geo::Point2D> model = tiny;
    for (int j = 0; j < trial % 4; ++j)
      model.push_back({double(int(rng() % 17) - 8) / 4, double(int(rng() % 17) - 8) / 4});
    auto scene = geo::geometric_transform(model, 2, 0, {4, -3});
    for (int j = 0; j < trial % 3; ++j)
      scene.push_back({double(20 + j), double(-30 - j)});
    const auto table = geo::build_geometric_hash(model, .25);
    const auto t     = geo::recognize_trace(table, model, scene, 1e-10, 1, 0);
    const auto plain = geo::recognize_run(table, model, scene, 1e-10, 1);
    EXPECT_EQ(t.run.match.votes, plain.match.votes);
    EXPECT_EQ(t.run.match.inliers, plain.match.inliers);
    std::array<std::size_t, 8>                         kinds{};
    std::map<std::tuple<int, int, std::uint64_t>, int> votes;
    int                                                best = 0, inliers = 0;
    std::size_t                                        improvements = 0;
    std::vector<int>                                   checks(model.size()), hits(model.size());
    for (const auto& e : t.events) {
      ++kinds[static_cast<std::size_t>(e.kind)];
      switch (e.kind) {
        case geo::GeoHashEventKind::scene_frame: {
          const auto f = geo::geometric_hash_frame(scene[e.point], scene[e.first], scene[e.second]);
          EXPECT_EQ(f.valid, e.accepted);
          EXPECT_EQ(f.u, e.u);
          EXPECT_EQ(f.v, e.v);
          EXPECT_EQ(f.length_squared, e.squared_distance);
          break;
        }
        case geo::GeoHashEventKind::lookup:
          EXPECT_EQ(table.bins.contains(e.key), e.accepted);
          break;
        case geo::GeoHashEventKind::vote:
          EXPECT_EQ(e.count, ++votes[std::make_tuple(e.first, e.second, e.key)]);
          EXPECT_EQ(e.key, pack(e.model_i, e.model_j));
          break;
        case geo::GeoHashEventKind::candidate:
          EXPECT_EQ(e.count, votes[std::make_tuple(e.first, e.second, e.key)]);
          EXPECT_EQ(e.accepted, int(e.count) > best);
          if (e.accepted) {
            best = int(e.count);
            ++improvements;
          }
          break;
        case geo::GeoHashEventKind::recover:
          EXPECT_EQ(e.scale, t.run.match.scale);
          EXPECT_EQ(e.translation, t.run.match.translation);
          break;
        case geo::GeoHashEventKind::distance: {
          ASSERT_GE(e.first, 0);
          ASSERT_LT(e.first, int(model.size()));
          EXPECT_EQ(e.second, checks[e.first]++);
          const double dx = e.x - scene[e.second].x, dy = e.y - scene[e.second].y;
          EXPECT_EQ(dx * dx + dy * dy, e.squared_distance);
          EXPECT_EQ(e.accepted, e.squared_distance <= 1e-20);
          if (e.accepted) {
            ++inliers;
            ++hits[e.first];
          }
          break;
        }
        default:
          FAIL();
      }
    }
    const auto& w = t.run.work;
    EXPECT_EQ(w.frame_attempts, kinds[2]);
    EXPECT_EQ(w.bin_lookups, kinds[3]);
    EXPECT_EQ(w.vote_updates, kinds[4]);
    EXPECT_EQ(w.candidate_comparisons, kinds[5]);
    EXPECT_EQ(w.recoveries, kinds[6]);
    EXPECT_EQ(w.distance_checks, kinds[7]);
    EXPECT_EQ(w.best_updates, improvements);
    EXPECT_EQ(t.run.match.votes, best);
    EXPECT_EQ(t.run.match.inliers, inliers);
    for (auto h : hits)
      EXPECT_LE(h, 1);
    if (!t.events.empty()) {
      EXPECT_THROW(
          geo::recognize_trace(table, model, scene, 1e-10, 1, t.events.size() - 1),
          std::length_error);
      EXPECT_EQ(
          geo::recognize_trace(table, model, scene, 1e-10, 1, t.events.size()).events.size(),
          t.events.size());
    }
  }
}
TEST(GeometricHashChecked, CandidateFoundIsNotInlierAcceptanceAndDuplicatesAreManyToOne) {
  const std::vector<geo::Point2D> model{{0, 0}, {2, 0}, {1, 1.5}, {2.2, 1.1}, {.5, -.7}, {-.4, .9}};
  auto                            scene = geo::geometric_transform(model, 1.7, .9, {12, -8});
  for (auto p : std::vector<geo::Point2D>{{-30, -20}, {-25, -34}, {-38, -12}, {-20, -28}})
    scene.push_back(p);
  const auto fine = geo::recognize(geo::build_geometric_hash(model, .05), model, scene, 1e-4, 2);
  ASSERT_TRUE(fine.found);
  EXPECT_EQ(fine.inliers, 6);
  EXPECT_EQ(fine.votes, 4);
  EXPECT_NEAR(fine.scale, 1.7, 1e-14);
  const auto coarse = geo::recognize(geo::build_geometric_hash(model, .5), model, scene, 1e-4, 2);
  EXPECT_TRUE(coarse.found);
  EXPECT_EQ(coarse.votes, 8);
  EXPECT_LT(coarse.inliers, 6);
  const std::vector<geo::Point2D> duplicate{{0, 0}, {2, 0}, {0, 1}, {0, 1}};
  const auto                      d =
      geo::recognize(geo::build_geometric_hash(duplicate, .25), duplicate, tiny, 1e-10, 1);
  EXPECT_TRUE(d.found);
  EXPECT_EQ(d.inliers, 4);
  EXPECT_GT(d.inliers, int(tiny.size()));
}
TEST(GeometricHashChecked, BinFloorBoundariesAndSignedCapacity) {
  EXPECT_EQ(geo::geometric_hash_bin(-.1, .6, .5).u, -1);
  EXPECT_EQ(geo::geometric_hash_bin(-.1, .6, .5).v, 1);
  EXPECT_EQ(geo::geometric_hash_bin(std::nextafter(.5, 0.), 0, .5).u, 0);
  EXPECT_EQ(geo::geometric_hash_bin(.5, 0, .5).u, 1);
  EXPECT_EQ(geo::geometric_hash_bin(std::nextafter(-.5, -1.), 0, .5).u, -2);
  EXPECT_EQ(geo::geometric_hash_bin(-.5, 0, .5).u, -1);
  const double low  = std::numeric_limits<std::int32_t>::min(),
               high = std::numeric_limits<std::int32_t>::max();
  EXPECT_EQ(geo::geometric_hash_bin(low, high, 1).key, pack(std::int64_t(low), std::int64_t(high)));
  EXPECT_THROW(geo::geometric_hash_bin(low - 1, 0, 1), std::overflow_error);
  EXPECT_THROW(geo::geometric_hash_bin(high + 1, 0, 1), std::overflow_error);
  EXPECT_THROW(geo::geometric_hash_bin(1, 0, 1e-300), std::overflow_error);
  EXPECT_EQ(geo::detail::gh_bin_key(-.1, .6, .5), geo::geometric_hash_bin(-.1, .6, .5).key);
}
TEST(GeometricHashChecked, CompleteDomainValidationBeforeQuotaAndOldUndefinedInputs) {
  const double nan  = std::numeric_limits<double>::quiet_NaN(),
               inf  = std::numeric_limits<double>::infinity(),
               huge = std::numeric_limits<double>::max();
  for (double q : {0., -1., nan, inf}) {
    EXPECT_THROW(geo::build_geometric_hash({}, q), std::invalid_argument);
    EXPECT_THROW(geo::build_geometric_hash_trace(tiny, q, 1), std::invalid_argument);
    EXPECT_THROW(geo::geometric_hash_bin(0, 0, q), std::invalid_argument);
  }
  auto bad     = tiny;
  bad.back().x = nan;
  EXPECT_THROW(geo::build_geometric_hash_trace(bad, .25, 1), std::invalid_argument);
  EXPECT_THROW(geo::recognize_trace({}, tiny, bad, 1e-6, 2, 1), std::invalid_argument);
  auto t = geo::build_geometric_hash(tiny, .25);
  for (double tol : {-1., nan, inf, huge})
    EXPECT_THROW(geo::recognize_trace(t, tiny, tiny, tol, 2, 1), std::invalid_argument);
  for (int threshold : {0, -1, std::numeric_limits<int>::min()})
    EXPECT_THROW(geo::recognize({}, tiny, tiny, 1e-6, threshold), std::invalid_argument);
  for (auto ids : std::vector<std::pair<int, int>>{{-1, 1}, {0, 90}, {1, 1}, {90, 91}}) {
    auto corrupt = t;
    corrupt.bins[0].push_back(ids);
    EXPECT_THROW(geo::recognize_trace(corrupt, tiny, tiny, 1e-6, 1, 1), std::invalid_argument);
  }
  EXPECT_THROW(geo::build_geometric_hash(tiny, 1e-300), std::overflow_error);
  EXPECT_THROW(geo::geometric_hash_frame({1, 1}, {-huge, 0}, {huge, 0}), std::overflow_error);
  EXPECT_THROW(geo::geometric_hash_bin(nan, 0), std::invalid_argument);
  EXPECT_THROW(geo::geometric_transform(tiny, 0), std::invalid_argument);
  EXPECT_THROW(geo::geometric_transform({}, 1, nan), std::invalid_argument);
  EXPECT_THROW(geo::geometric_transform({{huge, huge}}, 2), std::overflow_error);
  EXPECT_EQ(tiny[2].x, 0);
  EXPECT_EQ(t.bins.size(), 6u);
}
TEST(GeometricHashChecked, EmptySmallDegenerateThresholdAndOwnedRecognition) {
  EXPECT_TRUE(geo::build_geometric_hash_trace({}, .25, 1).events.empty());
  EXPECT_TRUE(geo::build_geometric_hash({{0, 0}, {1, 0}}).bins.empty());
  EXPECT_TRUE(geo::recognize_trace({}, {}, {}, 1e-6, 2, 1).events.empty());
  EXPECT_FALSE(geo::recognize({}, tiny, tiny).found);
  const auto table = geo::build_geometric_hash(tiny, .25);
  const auto high  = geo::recognize_run(table, tiny, tiny, 1e-6, 10);
  EXPECT_FALSE(high.match.found);
  EXPECT_EQ(high.match.votes, 1);
  EXPECT_EQ(high.work.recoveries, 0u);
  EXPECT_EQ(high.work.distance_checks, 0u);
  const auto t = geo::recognize_trace(table, tiny, tiny, 0, 1);
  ASSERT_TRUE(t.run.match.found);
  EXPECT_EQ(t.run.match.inliers, 3);
  EXPECT_EQ(t.run.work.distance_checks, 6u);
  EXPECT_EQ(t.events.size(), 31u);
  auto copy                    = t;
  copy.events[0].u             = 123;
  copy.run.match.translation.x = 999;
  EXPECT_NE(t.events[0].u, 123);
  EXPECT_NE(t.run.match.translation.x, 999);
  EXPECT_THROW(geo::recognize_trace(table, tiny, tiny, 0, 1, 30), std::length_error);
  EXPECT_EQ(geo::recognize_trace(table, tiny, tiny, 0, 1, 31).events.size(), 31u);
  std::vector<geo::Point2D> collinear{{0, 0}, {0, 0}, {0, 0}};
  const auto                d = geo::recognize_trace({}, collinear, collinear, 0, 1, 6);
  EXPECT_EQ(d.events.size(), 6u);
  EXPECT_EQ(d.run.work.degenerate_frames, 6u);
  EXPECT_FALSE(d.run.match.found);
}
