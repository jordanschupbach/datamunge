#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/selection_sort.hpp>
#include <datamunge/algorithms/sha1.hpp>
#include <datamunge/algorithms/hmac.hpp>
#include <datamunge/algorithms/sha256.hpp>
#include <datamunge/algorithms/shamir_secret_sharing.hpp>
#include <datamunge/algorithms/shannon_fano.hpp>
#include <datamunge/algorithms/shannon_fano_elias.hpp>
#include <datamunge/geometry/segment_intersection.hpp>
#include <datamunge/geometry/shamos_hoey.hpp>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/ops.hpp>
#include <datamunge/nn/optim.hpp>
#include <datamunge/nn/self_organizing_map.hpp>
#include <datamunge/random/sfc64_analysis.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <set>
namespace a = datamunge::algorithms;
namespace z = datamunge::geometry;
namespace r = datamunge::random;
namespace n = datamunge::nn;
TEST(Batch344Selection, ActualStatesAndInvariants) {
  auto t = a::selection_sort_trace({5, 3, 8, 1, 9, 2, 7});
  EXPECT_TRUE(std::is_sorted(t.values.begin(), t.values.end()));
  EXPECT_EQ(t.comparisons, 21);
  ASSERT_EQ(t.steps.size(), 6);
  std::size_t swaps = 0;
  for (const auto& s : t.steps) {
    EXPECT_TRUE(std::is_sorted(s.after.begin(), s.after.begin() + s.position + 1));
    EXPECT_EQ(s.after[s.position], s.minimum);
    EXPECT_EQ(s.comparisons, 6 - s.position);
    swaps += s.swapped;
  }
  EXPECT_EQ(t.swaps, swaps);
  EXPECT_EQ(a::selection_sort_run({1, 2, 3}).swaps, 0);
  EXPECT_TRUE(a::selection_sort({}).empty());
}
TEST(Batch344Selection, CompletePreflightAndOriginalSizes) {
  a::SelectionSortOptions o;
  o.max_comparisons = 2;
  EXPECT_THROW(a::selection_sort_run({3, 2, 1}, o), std::length_error);
  o                   = {};
  o.max_trace_entries = 7;
  EXPECT_THROW(a::selection_sort_trace({3, 2, 1}, o), std::length_error);
  auto v = a::selection_sort_example_arrays();
  ASSERT_EQ(v.size(), 40);
  EXPECT_EQ(v.back().size(), 200);
}
TEST(Batch344SHA1, FipsVectorsAndActualRoundChain) {
  EXPECT_EQ(a::sha1_hex(""), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
  EXPECT_EQ(a::sha1_hex("abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
  auto t = a::sha1_trace("abc");
  ASSERT_EQ(t.blocks.size(), 1);
  ASSERT_EQ(t.blocks[0].rounds.size(), 80);
  EXPECT_EQ(t.blocks[0].schedule[0], 0x61626380U);
  EXPECT_EQ(t.blocks[0].schedule[15], 24);
  auto state = t.blocks[0].before;
  for (const auto& s : t.blocks[0].rounds) {
    EXPECT_EQ(state, s.before);
    state = s.after;
    EXPECT_EQ(s.word, t.blocks[0].schedule[s.index]);
  }
  for (int i = 0; i < 5; ++i)
    EXPECT_EQ(t.blocks[0].after[i], std::uint32_t(state[i] + t.blocks[0].before[i]));
}
TEST(Batch344SHA256, FipsVectorsAndActualRoundChain) {
  EXPECT_EQ(a::sha256_hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(
      a::sha256_hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  auto t = a::sha256_trace("abc");
  ASSERT_EQ(t.blocks.size(), 1);
  ASSERT_EQ(t.blocks[0].rounds.size(), 64);
  auto state = t.blocks[0].before;
  for (const auto& s : t.blocks[0].rounds) {
    EXPECT_EQ(state, s.before);
    state = s.after;
  }
  for (int i = 0; i < 8; ++i)
    EXPECT_EQ(t.blocks[0].after[i], std::uint32_t(state[i] + t.blocks[0].before[i]));
}
TEST(Batch344SHA, PaddingCaptureAndFullAvalancheSupport) {
  for (std::size_t size : {0, 1, 55, 56, 63, 64, 65, 119, 120}) {
    auto       one    = a::sha1_run(std::string(size, char{0}));
    auto       two    = a::sha256_run(std::string(size, char{0}));
    const auto blocks = size / 64 + 1 + (size % 64 >= 56);
    EXPECT_EQ(one.compression_blocks, blocks);
    EXPECT_EQ(two.compression_blocks, blocks);
    EXPECT_EQ(one.rounds, 80 * blocks);
    EXPECT_EQ(two.rounds, 64 * blocks);
    EXPECT_EQ(one.schedule_expansions, 64 * blocks);
    EXPECT_EQ(two.schedule_expansions, 48 * blocks);
  }
  auto one = a::sha1_avalanche("avalanche"), two = a::sha256_avalanche("avalanche");
  EXPECT_EQ(one.histogram.size(), 161);
  EXPECT_EQ(two.histogram.size(), 257);
  EXPECT_EQ(one.changed_bits.size(), 72);
  EXPECT_EQ(std::accumulate(one.histogram.begin(), one.histogram.end(), std::size_t{}), 72);
  EXPECT_EQ(one.hash_calls, 73);
  EXPECT_EQ(a::sha256_avalanche("").hash_calls, 1);
  a::SHAOptions o;
  o.max_trace_entries = 1;
  EXPECT_THROW(a::sha1_trace("abc", o), std::length_error);
  EXPECT_THROW(a::sha256_trace("abc", o), std::length_error);
}
TEST(Batch344Shamir, OriginalTermsAndEverySufficientSubset) {
  auto t = a::shamir_split_trace(1234, 6, {166, 94}, 2087);
  ASSERT_EQ(t.steps.size(), 12);
  for (const auto& s : t.steps)
    EXPECT_EQ(s.after, (s.before + s.term) % 2087);
  for (std::size_t i = 0; i < 6; ++i)
    for (std::size_t j = i + 1; j < 6; ++j)
      for (std::size_t k = j + 1; k < 6; ++k) {
        auto c = a::shamir_combine_trace({t.shares[i], t.shares[j], t.shares[k]}, 2087);
        EXPECT_EQ(c.secret, 1234);
        ASSERT_EQ(c.terms.size(), 3);
        EXPECT_EQ(c.terms.back().partial, 1234);
      }
}
TEST(Batch344Shamir, LargePrimeAndAnyCandidateCompletion) {
  constexpr std::uint64_t prime = 18446744073709551557ULL;
  auto                    s     = a::shamir_split(prime - 2, 5, {prime - 3, prime - 4}, prime);
  EXPECT_EQ(a::shamir_combine({s[0], s[2], s[4]}, prime), prime - 2);
  for (std::uint64_t secret = 0; secret < 17; ++secret) {
    auto c = a::shamir_quadratic_completion({{1, 5}, {2, 9}}, secret, 17);
    auto v = a::shamir_split(secret, 3, c, 17);
    EXPECT_EQ(v[0].y, 5);
    EXPECT_EQ(v[1].y, 9);
  }
  EXPECT_THROW(a::shamir_split(1, 3, {2}, 15), std::invalid_argument);
  EXPECT_THROW(a::shamir_combine({{1, 2}, {1, 3}}, 17), std::invalid_argument);
  EXPECT_THROW(a::shamir_combine({}, 17), std::invalid_argument);
}
TEST(Batch344SFE, ExactWordsInactiveAndRoundTrip) {
  auto b = a::shannon_fano_elias_trace({1, 1});
  EXPECT_EQ(b.codebook, (std::vector<std::vector<bool>>{{false, true}, {true, true}}));
  EXPECT_EQ(b.emitted_bits, 4);
  EXPECT_EQ(b.average_bits, 2);
  EXPECT_EQ(b.entropy, 1);
  auto sparse = a::shannon_fano_elias_codebook({0, 1, 0, 1});
  EXPECT_TRUE(sparse[0].empty());
  EXPECT_TRUE(sparse[2].empty());
  std::vector<int> message{0, 0, 1, 0, 2, 0, 1, 3, 0, 2};
  auto             e = a::shannon_fano_elias_encode_trace(message, {10, 5, 3, 2});
  auto             d = a::shannon_fano_elias_decode_trace(e.bits, {10, 5, 3, 2}, message.size());
  EXPECT_EQ(d.symbols, message);
  ASSERT_EQ(e.words.size(), message.size());
  EXPECT_EQ(d.words.size(), message.size());
  for (std::size_t i = 0; i < message.size(); ++i) {
    EXPECT_EQ(e.words[i].begin, d.words[i].begin);
    EXPECT_EQ(e.words[i].end, d.words[i].end);
  }
}
TEST(Batch344SFE, RejectIncompleteExtraAndInvalidModels) {
  EXPECT_THROW(a::shannon_fano_elias_codebook({0, 0}), std::invalid_argument);
  EXPECT_THROW(a::shannon_fano_elias_encode({-1}, {1, 1}), std::invalid_argument);
  EXPECT_THROW(a::shannon_fano_elias_encode({0}, {0, 1}), std::invalid_argument);
  EXPECT_THROW(a::shannon_fano_elias_decode({false}, {1, 1}, 1), std::invalid_argument);
  EXPECT_THROW(
      a::shannon_fano_elias_decode({false, true, true, true}, {1, 1}, 1), std::invalid_argument);
  EXPECT_THROW(a::shannon_fano_elias_decode({}, {1, 1}, 1), std::invalid_argument);
  EXPECT_TRUE(a::shannon_fano_elias_decode({}, {}, 0).empty());
}
TEST(Batch344Fano, ActualPartitionsAndRawByteRoundTrip) {
  std::string text;
  for (int i = 0; i < 256; ++i)
    text += char(i);
  text += char{0};
  auto b = a::shannon_fano_build_trace(text);
  auto e = a::shannon_fano_encode_trace(b.code, text);
  auto d = a::shannon_fano_decode_trace(b.code, e.bits, text.size());
  EXPECT_EQ(d.text, text);
  EXPECT_EQ(b.weighted_bits, e.bits.size());
  EXPECT_EQ(b.frequencies[0], 2);
  EXPECT_EQ(b.splits.size(), 255);
  for (const auto& s : b.splits) {
    EXPECT_LT(s.begin, s.middle);
    EXPECT_LT(s.middle, s.end);
    EXPECT_EQ(s.imbalance, s.left > s.right ? s.left - s.right : s.right - s.left);
  }
  EXPECT_EQ(a::shannon_fano_build_run("aaaa").average_bits, 1);
}
TEST(Batch344Fano, CompleteModelAndPayloadValidation) {
  auto b = a::shannon_fano_build("aab");
  EXPECT_THROW(a::shannon_fano_encode(b, "c"), std::invalid_argument);
  EXPECT_THROW(a::shannon_fano_decode_run(b, "2", 1), std::invalid_argument);
  EXPECT_THROW(
      a::shannon_fano_decode_run(b, a::shannon_fano_encode(b, "aab"), 2), std::invalid_argument);
  a::ShannonFanoCode bad{{{'a', "0"}, {'b', "01"}}};
  EXPECT_THROW(a::shannon_fano_encode(bad, "a"), std::invalid_argument);
  EXPECT_EQ(a::shannon_fano_average_bits(a::ShannonFanoCode{}, ""), 0);
}
TEST(Batch344SFC, ActualTransitionSnapshotAndWrap) {
  r::Sfc64 g(std::array<std::uint64_t, 4>{1, 2, 3, 4});
  auto     s = g.next_trace();
  EXPECT_EQ(s.value, 7);
  EXPECT_EQ(s.after, (std::array<std::uint64_t, 4>{2, 27, 50331655, 5}));
  EXPECT_EQ(g.state(), s.after);
  r::Sfc64 restored(g.state());
  EXPECT_EQ(g.next_u64(), restored.next_u64());
  r::Sfc64 wrap(std::array<std::uint64_t, 4>{0, 0, 0, UINT64_MAX});
  EXPECT_EQ(wrap.next_trace().after[3], 0);
  r::Sfc64 seeded(0xC0FFEEULL);
  EXPECT_EQ(seeded.next_u64(), 13215376215447164276ULL);
}
TEST(Batch344SFC, ActualCountsAndCapturePolicies) {
  auto t = r::sfc64_trace(7, 1000);
  ASSERT_EQ(t.steps.size(), 1000);
  EXPECT_EQ(std::accumulate(t.low_bytes.begin(), t.low_bytes.end(), std::size_t{}), 1000);
  for (std::size_t i = 0; i < 1000; ++i) {
    EXPECT_EQ(t.outputs[i], t.steps[i].value);
    if (i)
      EXPECT_EQ(t.steps[i].before, t.steps[i - 1].after);
  }
  auto empty = r::sfc64_trace(7, 0);
  EXPECT_EQ(empty.initial, empty.final);
  r::Sfc64Options o;
  o.max_trace_entries = 1;
  EXPECT_THROW(r::sfc64_trace(7, 2, o), std::length_error);
}
TEST(Batch344Segment, ClosedDegeneraciesAndUnchangedOutputs) {
  auto cross = z::segment_intersection_trace({0, 0}, {2, 2}, {0, 2}, {2, 0});
  EXPECT_EQ(cross.contact, z::SegmentContact::Point);
  EXPECT_EQ(cross.first, (z::Point2D{1, 1}));
  EXPECT_EQ(cross.orientation_tests, 4);
  auto overlap = z::segment_intersection_trace({0, 0}, {4, 0}, {2, 0}, {6, 0});
  EXPECT_EQ(overlap.contact, z::SegmentContact::Overlap);
  EXPECT_EQ(overlap.first, (z::Point2D{2, 0}));
  EXPECT_EQ(overlap.second, (z::Point2D{4, 0}));
  z::Point2D out{9, 9};
  EXPECT_FALSE(z::segment_intersection_point({0, 0}, {4, 0}, {2, 0}, {6, 0}, out));
  EXPECT_EQ(out, (z::Point2D{9, 9}));
  EXPECT_TRUE(z::segment_intersection_point({0, 0}, {2, 0}, {2, 0}, {3, 0}, out));
  EXPECT_EQ(out, (z::Point2D{2, 0}));
  EXPECT_TRUE(z::segments_intersect({1, 0}, {1, 0}, {0, 0}, {2, 0}));
  EXPECT_FALSE(z::segments_intersect({0, 0}, {0, 0}, {1, 1}, {1, 1}));
}
TEST(Batch344Segment, ExtremeDyadicsAndExactQuotients) {
  for (double scale : {1e-300, 1e-150, 1., 1e150, 1e300}) {
    auto t = z::segment_intersection_trace(
        {-scale, -scale}, {scale, scale}, {-scale, scale}, {scale, -scale});
    EXPECT_EQ(t.contact, z::SegmentContact::Point);
    EXPECT_EQ(t.first, (z::Point2D{0, 0}));
    EXPECT_EQ(t.parameter, .5L);
  }
  EXPECT_EQ(
      z::cross_difference_ratio({0, 0}, {1, 0}, {0, 1}, {0, 0}, {0, 0}, {2, 0}, {0, 1}, {0, 0}),
      .5L);
  EXPECT_THROW(z::segments_intersect({0, 0}, {NAN, 0}, {1, 1}, {2, 2}), std::invalid_argument);
}
TEST(Batch344Shamos, VerticalClosedContactsAndAVLGrowth) {
  EXPECT_TRUE(z::any_segments_intersect({{{2, 0}, {2, 4}}, {{0, 2}, {4, 2}}}));
  EXPECT_FALSE(z::any_segments_intersect({{{2, 0}, {2, 1}}, {{2, 2}, {2, 3}}}));
  EXPECT_TRUE(z::any_segments_intersect({{{2, 0}, {2, 4}}, {{2, 3}, {2, 5}}}));
  EXPECT_TRUE(z::any_segments_intersect({{{1, 1}, {1, 1}}, {{0, 1}, {2, 1}}}));
  auto t = z::shamos_hoey_trace(z::shamos_horizontal_segments(500));
  EXPECT_FALSE(t.found);
  EXPECT_EQ(t.events_processed, 1000);
  EXPECT_GT(t.rotations, 0);
  for (const auto& e : t.events)
    EXPECT_LE(e.tree_height_after, 20);
}
TEST(Batch344Shamos, RandomSetsAgainstAllPairs) {
  std::mt19937 rng(344);
  for (int trial = 0; trial < 3000; ++trial) {
    std::vector<std::pair<z::Point2D, z::Point2D>> segments;
    const int                                      count = trial % 15;
    for (int i = 0; i < count; ++i) {
      auto coord = [&] { return double(int(rng() % 33) - 16) / 4; };
      segments.push_back({{coord(), coord()}, {coord(), coord()}});
    }
    bool expected = false;
    for (std::size_t i = 0; i < segments.size(); ++i)
      for (std::size_t j = 0; j < i; ++j)
        expected |= z::segments_intersect(
            segments[i].first, segments[i].second, segments[j].first, segments[j].second);
    auto t = z::shamos_hoey_trace(segments);
    ASSERT_EQ(t.found, expected) << trial;
    if (t.found) {
      ASSERT_LT(t.first, segments.size());
      ASSERT_LT(t.second, segments.size());
      EXPECT_TRUE(
          z::segments_intersect(
              segments[t.first].first,
              segments[t.first].second,
              segments[t.second].first,
              segments[t.second].second));
    }
  }
}
TEST(Batch344SOM, PlanAndActualScalarReferenceUpdates) {
  a::SOMParameters p;
  p.rows       = 1;
  p.cols       = 2;
  p.iterations = 3;
  p.seed       = 1;
  auto plan    = a::som_training_plan({{0}, {2}}, p);
  auto t       = a::som_train_trace({{0}, {2}}, p);
  EXPECT_EQ(plan.sample_indices.size(), 3);
  ASSERT_EQ(t.steps.size(), 3);
  for (std::size_t i = 0; i < 3; ++i) {
    const auto& s = t.steps[i];
    EXPECT_EQ(s.sample, plan.sample_indices[i]);
    for (std::size_t k = 0; k < 2; ++k) {
      double input = s.sample ? 2 : 0;
      EXPECT_NEAR(
          s.after[k][0],
          s.before[k][0] + s.learning_rate * s.influence[k] * (input - s.before[k][0]),
          1e-14);
    }
  }
  p.iterations = 0;
  EXPECT_EQ(a::som_train_run({{2}}, p).updates, 0);
  EXPECT_THROW(a::som_train({}, p), std::invalid_argument);
  p.rows = 0;
  EXPECT_THROW(a::som_train({{2}}, p), std::invalid_argument);
}
TEST(Batch344TensorSOM, FrozenWinnerGradientAndExplicitSGD) {
  auto weights = n::from_vector(std::vector<double>{0, 2}, {2, 1}).requires_grad_();
  auto input   = n::from_vector(std::vector<double>{1});
  auto grid    = n::from_vector(std::vector<double>{0, 0, 0, 1}, {2, 2});
  auto step    = n::som_neighborhood_objective(weights, input, grid, 1);
  EXPECT_EQ(step.winner, 0);
  auto gradients = n::grad(step.objective, std::vector<n::Tensor>{weights});
  auto g         = gradients[0].to_vector<double>();
  EXPECT_NEAR(g[0], -1, 1e-14);
  EXPECT_NEAR(g[1], std::exp(-.5), 1e-14);
  n::OptimizerConfig config;
  config.kind = n::OptimizerKind::Sgd;
  n::Optimizer optimizer(std::vector<n::Tensor>{weights}, .5, config);
  optimizer.step(gradients);
  auto actual = weights.to_vector<double>();
  EXPECT_NEAR(actual[0], .5, 1e-14);
  EXPECT_NEAR(actual[1], 2 - .5 * std::exp(-.5), 1e-14);
}
TEST(Batch344TensorSOM, ExplicitTrajectoryMatchesScalarAndFiniteControls) {
  std::vector<std::vector<double>> data{{0, 0}, {1, 0}, {0, 1}, {1, 1}};
  a::SOMParameters                 p;
  p.rows = p.cols            = 2;
  p.iterations               = 30;
  p.seed                     = 7;
  auto               plan    = n::som_tensor_plan(data, p);
  auto               weights = plan.initial_weights.clone().requires_grad_();
  n::OptimizerConfig config;
  config.kind = n::OptimizerKind::Sgd;
  n::Optimizer optimizer(std::vector<n::Tensor>{weights}, p.learning_rate, config);
  for (std::size_t t = 0; t < plan.sample_indices.size(); ++t) {
    auto id = n::from_vector(
        std::vector<std::int64_t>{static_cast<std::int64_t>(plan.sample_indices[t])});
    auto sample = n::gather(plan.inputs, id, 0).reshape({2});
    auto step   = n::som_neighborhood_objective(weights, sample, plan.grid, plan.radii[t]);
    optimizer.set_learning_rate(plan.learning_rates[t]);
    optimizer.minimize(step.objective);
  }
  auto reference = a::som_train(data, p);
  auto actual    = weights.to_vector<double>();
  for (std::size_t k = 0; k < 4; ++k)
    for (std::size_t d = 0; d < 2; ++d)
      EXPECT_NEAR(actual[k * 2 + d], reference.weights[k][d], 1e-12);
  EXPECT_NEAR(
      n::som_tensor_quantization_error(weights, plan.inputs).item(),
      a::som_quantization_error(reference, data),
      1e-12);
  EXPECT_THROW(
      n::som_neighborhood_objective(weights, n::zeros({2}, n::DType::Float64), plan.grid, 0),
      std::invalid_argument);
}

TEST(Batch344SHA, LegacyLargeMessagesAndHmacComposition){
 const std::string large(1000064,'a');
 EXPECT_THROW(a::sha256_run(large),std::length_error);
 EXPECT_EQ(a::sha1_hex(large),"cd33aeba10c1d3935b476eaa4f47dfaf5d1756ff");
 EXPECT_EQ(a::sha256_hex(large),"00c653f07896c2bb74de3db928250366da8220e5ffc67cb6bc1f485325757a06");
 EXPECT_EQ(a::hmac_sha256_hex("key",std::string(1000000,'a')),"6e7005164aec3b1035635787fbdd6b729031b2eb39915ec3bd249d52731cc7a5");
}
