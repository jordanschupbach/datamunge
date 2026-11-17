#pragma once
/// @file ransac.hpp
/// @brief Checked planar line RANSAC with actual sample decisions and separately rescored TLS refits.
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
/// @brief Canonical robust estimation and owned producing records.
namespace datamunge::algorithms {
/// @brief Legacy normalized implicit-line and consensus layout.
/// @par Example
/// @code
/// const dm::algorithms::RansacLine value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct RansacLine {
  double            a{0}; ///< Actual normalized x coefficient; meaningful only for a found model.
  double            b{0}; ///< Actual normalized y coefficient.
  double            c{0}; ///< Actual finite intercept in a*x+b*y+c=0.
  std::vector<char> inliers;         ///< Actual threshold flags for this represented line.
  std::size_t       inlier_count{0}; ///< Actual number of nonzero flags.
  std::size_t       trials{0};       ///< Actual attempted random pairs, including skipped pairs.
};
/// @brief Actual fate of one independently sampled index pair.
/// @par Example
/// @code
/// const auto value=dm::algorithms::RansacTrialKind::DuplicateIndex;
/// std::print("{}\n",value==dm::algorithms::RansacTrialKind::DuplicateIndex); // true
/// @endcode
enum class RansacTrialKind {
  DuplicateIndex,   ///< Both actual draws selected the same stored point.
  CoincidentPoints, ///< Distinct ids have equal represented coordinates.
  Scored            ///< A real normalized hypothesis was scored on every input point.
};
/// @brief One actual sampled pair and its real pre-refit consensus.
/// @par Example
/// @code
/// const dm::algorithms::RansacTrial value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct RansacTrial {
  std::size_t       iteration{}; ///< Actual zero-based attempted pair.
  std::size_t       first{};     ///< Actual first sampled input id.
  std::size_t       second{};    ///< Actual second sampled input id.
  RansacTrialKind   kind{};      ///< Actual skip/score branch.
  double            a{};         ///< Actual normalized scored coefficient; zero for skipped trials.
  double            b{};         ///< Actual normalized scored coefficient; zero for skipped trials.
  double            c{};         ///< Actual normalized scored coefficient; zero for skipped trials.
  std::vector<char> inliers;     ///< Actual full scored mask, empty for skipped trials.
  std::size_t       count{};     ///< Actual scored consensus cardinality.
  bool improved{}; ///< Whether this real trial replaced the strictly smaller best consensus.
};
/// @brief Complete input, actual scoring and retained-history bounds; zero disables a cap.
/// @par Example
/// @code
/// const dm::algorithms::RansacOptions value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct RansacOptions {
  std::size_t max_points{1000000};          ///< Complete coordinate-pair allocation policy.
  std::size_t max_trials{1000000};          ///< Requested attempted random pairs.
  std::size_t max_distance_tests{10000000}; ///< Actual hypothesis/final-refit distance tests.
  std::size_t max_trace_entries{
      1000000}; ///< Actual mask scalars plus one header per attempted pair.
};
/// @brief Owned sampled winner and separately rescored actual TLS result.
/// @par Example
/// @code
/// const dm::algorithms::RansacRun value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct RansacRun {
  RansacLine  hypothesis; ///< Actual winning minimal model and its pre-refit consensus.
  RansacLine  line;       ///< Actual TLS-refit model with a newly scored post-refit mask/count.
  bool        found{};    ///< At least one positive-consensus valid hypothesis was found.
  bool        refitted{}; ///< The actual winning consensus had at least two points and TLS ran.
  std::size_t winning_trial{
      std::numeric_limits<std::size_t>::max()}; ///< Actual winning attempt, SIZE_MAX if absent.
  std::size_t draws{};              ///< Actual independent index draws, two per attempted pair.
  std::size_t distance_tests{};     ///< Actual scored perpendicular-distance evaluations.
  std::size_t skipped_duplicates{}; ///< Actual equal-index skipped pairs.
  std::size_t skipped_coincident{}; ///< Actual distinct-index coincident-coordinate pairs.
  std::vector<RansacTrial> trials;  ///< Actual optional attempted-pair records.
};
/// @brief Fit a planar line using one checked native random-sample-consensus producer.
/// Validates complete finite equal-length coordinates, nonnegative finite threshold and execution policies before fitting. Draws two independent uniform indices per attempt from mt19937_64(seed); duplicate ids and exactly coincident coordinates skip scoring. Scores normalized implicit hypotheses with the represented-coefficient perpendicular residual, retains the first strictly largest consensus, performs orthogonal TLS on it, then rescans the actual refitted line. Pre-refit and post-refit masks remain separate. Uses extended arithmetic/hypot for geometry but does not promise exact geometric decisions. Fewer than two input points or zero attempts returns found=false. O(attempts*points) scoring plus O(points) working storage, inputs unchanged and outputs owned.
/// @param xs Finite x coordinates.
/// @param ys Finite matching y coordinates.
/// @param threshold Finite nonnegative represented-line perpendicular threshold.
/// @param iterations Actual attempted independent index pairs.
/// @param seed MT19937-64 seed, default0.
/// @param options Input, trial, actual-distance and retained-history policies.
/// @return Actual winning minimal model, actual TLS refit with rescored mask, and selected work.
/// @throws std::invalid_argument Shape, coordinate or threshold invalid.
/// @throws std::length_error Any input/trial/scoring/history capacity exceeded.
/// @throws std::overflow_error Normalized geometry or residual is outside finite double range.
/// @par Example
/// @code
/// auto r=dm::algorithms::ransac_line_run({0,1,2},{1,3,5},.01,20,1);
/// std::print("{} {}\n",r.found,r.line.inlier_count); // true 3
/// @endcode
/// @par References
/// - Fischler and Bolles (1981), Random Sample Consensus, https://doi.org/10.1145/358669.358692
/// @see ransac_line_trace(), ransac_required_iterations()
RansacRun ransac_line_run(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    double                     threshold,
    std::size_t                iterations,
    std::uint64_t              seed    = 0,
    const RansacOptions&       options = {});
/// @brief Capture actual index draws, skipped pairs and scored pre-refit masks.
/// Same single producer as ransac_line_run(); scored records cost points+1 history entries and skipped records cost1. Admission occurs before retaining the actual record. No second fitting run populates history. Returned final line is rescored separately, so its mask may differ from the winning hypothesis.
/// @param xs Finite x coordinates.
/// @param ys Matching finite y coordinates.
/// @param threshold Nonnegative finite distance threshold.
/// @param iterations Actual attempted pairs.
/// @param seed Seed, default0.
/// @param options Complete execution/history policies.
/// @return Actual attempted-pair history, both masks and selected work.
/// @throws std::invalid_argument Invalid complete input/threshold.
/// @throws std::length_error Any execution/history policy exceeded.
/// @throws std::overflow_error Unrepresentable geometry/residual.
/// @par Example
/// @code
/// auto t=dm::algorithms::ransac_line_trace({0,1},{0,1},.01,5,1);
/// std::print("{}\n",t.trials.size()); // 5
/// @endcode
/// @see ransac_line_run()
RansacRun ransac_line_trace(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    double                     threshold,
    std::size_t                iterations,
    std::uint64_t              seed    = 0,
    const RansacOptions&       options = {});
/// @brief Return the actual rescored TLS-refit line through the preserved signature.
/// Uses the shared native producer and default policies. Unlike the legacy implementation, the returned inliers/count are rescored against the returned refitted coefficients. Distinct arbitrarily close finite points are not rejected merely by a fixed1e-12 scale cutoff. No-found result has zero coefficients and empty mask; do not interpret it as a normalized line.
/// @param xs Finite x coordinates.
/// @param ys Matching finite y coordinates.
/// @param threshold Nonnegative finite threshold.
/// @param iterations Requested actual pairs.
/// @param seed Native seed, default0.
/// @return Actual refitted model and matching threshold mask, or no-model sentinel.
/// @throws std::invalid_argument Invalid input/threshold.
/// @throws std::length_error Default policy exceeded.
/// @throws std::overflow_error Unrepresentable geometry/residual.
/// @par Example
/// @code
/// auto r=dm::algorithms::ransac_line({0,1,2},{1,3,5},.01,20,1);
/// std::print("{}\n",r.inlier_count); // 3
/// @endcode
/// @see ransac_line_run()
RansacLine ransac_line(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    double                     threshold,
    std::size_t                iterations,
    std::uint64_t              seed = 0);
/// @brief Compute the ideal independent-sample RANSAC trial bound.
/// Returns ceil(log1p(-p)/log1p(-w^s)). This ideal model assumes an all-inlier minimal sample always gives a valid model; it is not the exact finite index-pair probability of the native sampler, which can draw duplicate ids. p0 needs zero attempts; w1 with p>0 needs one. w0 or p1 with w<1 has no finite bound and throws. No mutation/allocation; constant arithmetic work.
/// @param inlier_fraction Finite ideal probability w in[0,1].
/// @param sample_size Positive minimal sample size s.
/// @param success_prob Finite target p in[0,1], default0.99.
/// @return Finite ideal trial ceiling.
/// @throws std::invalid_argument Invalid probability or zero sample size.
/// @throws std::domain_error No finite success bound or underflowed positive sample probability.
/// @throws std::length_error Required ceiling exceeds size_t.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::ransac_required_iterations(.67,2,.99)); // 8
/// @endcode
/// @see ransac_pair_iterations()
std::size_t ransac_required_iterations(
    double      inlier_fraction,
    std::size_t sample_size,
    double      success_prob = .99);
/// @brief Compute the exact finite distinct-inlier-pair bound for the native draw/skip mechanism.
/// Two independent uniform input indices are drawn per attempt. With I inliers among N points, a distinct all-inlier pair has probability I*(I-1)/N^2. This adjusts duplicate-index skips but still assumes distinct inlier points define a valid target model; coincident coordinates/threshold failures can reduce success further. Probability zero has no finite positive-success bound.
/// @param inliers Known inlier count I, no greater than total.
/// @param total Positive input count N.
/// @param success_prob Finite target success probability, default0.99.
/// @return Finite attempted-pair ceiling under the stated geometric assumption.
/// @throws std::invalid_argument Invalid counts/probability.
/// @throws std::domain_error No finite positive-success bound.
/// @throws std::length_error Ceiling outside size_t.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::ransac_pair_iterations(2,2,.99)); // 7
/// @endcode
/// @see ransac_required_iterations()
std::size_t
ransac_pair_iterations(std::size_t inliers, std::size_t total, double success_prob = .99);
/// @brief Owned original synthetic line-plus-outliers fixture and generation identities.
/// @par Example
/// @code
/// const dm::algorithms::RansacPoints value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct RansacPoints {
  std::vector<double> x; ///< Actual generated x coordinates.
  std::vector<double> y; ///< Actual generated y coordinates.
  std::vector<char>
      generated_inlier; ///< Fixture generation label, not an estimated consensus flag.
};
/// @brief Generate the original noisy y=2x+1 fixture and uniform gross outliers.
/// Uses one mt19937_64(seed) stream: inliers consume uniform x0..10 then normal noise sd0.05; outliers consume uniform x0..10 then uniform y-20..40. Generation labels identify construction, not guaranteed fitted threshold membership. O(count) owned output and deterministic native stream, subject to standard-library distribution implementation.
/// @param inliers Noisy line observations, default80.
/// @param outliers Uniform gross-outlier observations, default40.
/// @param seed Native generator seed, default7.
/// @param max_points Complete allocation policy, default1000000; zero disables.
/// @return Owned original synthetic coordinates and generation identities.
/// @throws std::length_error Complete input size/capacity policy exceeded.
/// @par Example
/// @code
/// auto p=dm::algorithms::ransac_example_points();
/// std::print("{}\n",p.x.size()); // 120
/// @endcode
/// @see ransac_line_run()
RansacPoints ransac_example_points(
    std::size_t   inliers    = 80,
    std::size_t   outliers   = 40,
    std::uint64_t seed       = 7,
    std::size_t   max_points = 1000000);
} // namespace datamunge::algorithms
