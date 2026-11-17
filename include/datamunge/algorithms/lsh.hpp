#pragma once
/// @file lsh.hpp
/// @brief Owned checked random-hyperplane cosine index and actual producing query evidence.
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual selected query result/work; dense seen initialization remains linear in indexed row count.
/// @par Example
/// @code
/// dm::algorithms::CosineLSH index({{1,0},{0,1}},1,0,7);
/// auto a=index.query_analysis({1,0});
/// std::print("{} {}\n",a.best,a.examined); // 0 2
/// @endcode
/// @par References
/// - Charikar(2002), Similarity Estimation Techniques from Rounding Algorithms, https://doi.org/10.1145/509907.509965; checked owned finite Gaussian-plane implementation here.
/// @see CosineLSH, LSHQueryTrace, lsh_collision_probability()
struct LSHQueryAnalysis {
  std::size_t best{
      std::numeric_limits<std::size_t>::
          max()}; ///< Actual first strict maximum candidate id, SIZE_MAX when none collide.
  double      similarity{-2};   ///< Actual best cosine, or sentinel minus2 for no candidate.
  std::size_t examined{0};      ///< Actual distinct candidate cosine evaluations.
  std::size_t projections{0};   ///< Actual query hyperplane dot evaluations, tables*bits.
  std::size_t bucket_visits{0}; ///< Actual member visits, including duplicates across tables.
  std::size_t seen_entries{0};  ///< Actual dense visited marks initialized, one per indexed row.
};
/// @brief One actual hyperplane sign decision from the producing query.
/// @par Example
/// @code
/// dm::algorithms::CosineLSH index({{1,0}},1,1,7);
/// auto t=index.query_trace({1,0});
/// std::print("{}\n",t.projections.size()); // 1
/// @endcode
/// @par References
/// - Charikar(2002), Similarity Estimation Techniques from Rounding Algorithms, https://doi.org/10.1145/509907.509965; checked owned finite Gaussian-plane implementation here.
/// @see CosineLSH, LSHQueryTrace, lsh_collision_probability()
struct LSHProjection {
  std::size_t table; ///< Actual table index.
  std::size_t bit;   ///< Actual signature bit position, at most63.
  double
      dot; ///< Actual finite deciding dot, after positive query scaling only if raw arithmetic failed.
  bool nonnegative; ///< Actual dot>=0 decision, with exact zero mapped to bit1.
  bool scaled; ///< Whether the raw nonfinite dot required positive max-magnitude query scaling.
};
/// @brief Actual query table lookup and owned members of its real bucket.
/// @par Example
/// @code
/// dm::algorithms::CosineLSH index({{1,0}},1,0,7);
/// auto t=index.query_trace({1,0});
/// std::print("{}\n",t.buckets.front().members.size()); // 1
/// @endcode
/// @par References
/// - Charikar(2002), Similarity Estimation Techniques from Rounding Algorithms, https://doi.org/10.1145/509907.509965; checked owned finite Gaussian-plane implementation here.
/// @see CosineLSH, LSHQueryTrace, lsh_collision_probability()
struct LSHBucketProbe {
  std::size_t   table;     ///< Actual queried table index.
  std::uint64_t signature; ///< Actual assembled unsigned signature.
  std::vector<std::size_t>
      members; ///< Actual stored insertion-order bucket ids, empty for a missing key.
};
/// @brief Actual ordered candidate visit with duplicate and strict winner decisions.
/// @par Example
/// @code
/// dm::algorithms::CosineLSH index({{1,0}},2,0,7);
/// auto t=index.query_trace({1,0});
/// std::print("{}\n",t.candidates.back().duplicate); // true
/// @endcode
/// @par References
/// - Charikar(2002), Similarity Estimation Techniques from Rounding Algorithms, https://doi.org/10.1145/509907.509965; checked owned finite Gaussian-plane implementation here.
/// @see CosineLSH, LSHQueryTrace, lsh_collision_probability()
struct LSHCandidate {
  std::size_t table; ///< Actual visiting table.
  std::size_t id;    ///< Actual stored input identity.
  bool duplicate; ///< True when already examined in an earlier table; no cosine is evaluated then.
  double similarity; ///< Actual newly evaluated cosine, or placeholder0 for a duplicate.
  bool   new_best;   ///< Actual strict running winner update, false for duplicates/ties.
};
/// @brief Owned actual query projection, bucket and candidate records from one producer execution.
/// @par Example
/// @code
/// dm::algorithms::CosineLSH index({{1,0}},1,0,7);
/// auto t=index.query_trace({1,0});
/// std::print("{}\n",t.analysis.best); // 0
/// @endcode
/// @par References
/// - Charikar(2002), Similarity Estimation Techniques from Rounding Algorithms, https://doi.org/10.1145/509907.509965; checked owned finite Gaussian-plane implementation here.
/// @see CosineLSH, LSHQueryTrace, lsh_collision_probability()
struct LSHQueryTrace {
  std::vector<double> query; ///< Original query, independently owned.
  std::vector<LSHProjection>
      projections;                     ///< Actual table/bit dot decisions, not regenerated planes.
  std::vector<LSHBucketProbe> buckets; ///< Actual table lookup keys and stored members.
  std::vector<LSHCandidate>
                   candidates; ///< Actual ordered bucket visits, with duplicate suppression.
  LSHQueryAnalysis analysis;   ///< Actual same-run result and work.
  std::size_t      captured_entries{
      0}; ///< Query coordinates plus real projection/probe/member/candidate records.
};
/// @brief Own an immutable finite index with seeded Gaussian hyperplanes and insertion-order buckets.
/// Input data are copied, fixing the historical borrowed-pointer lifetime hazard and stale buckets after caller mutation.
/// Bit agreement has probability1-theta/pi for nonzero vectors and an isotropic continuous plane;
/// finite sampled Gaussian arithmetic is the implementation. Zero vectors use cosine0 and nonnegative sign bits.
/// Query still initializes N dense marks: O(N+L*k*d+V+C*d), with V actual member visits and C distinct candidates.
/// There is no worst-case sublinear guarantee. Copies own all data/planes/buckets; const queries share no mutable state.
/// @par Example
/// @code
/// dm::algorithms::CosineLSH index({{1,0},{0,1}},1,0,7);
/// std::print("{}\n",index.query({0,1})); // 1
/// @endcode
/// @par References
/// M. Charikar(2002), Similarity Estimation Techniques from Rounding Algorithms, https://doi.org/10.1145/509907.509965.
/// @see CosineLSH, LSHQueryTrace, lsh_collision_probability()
class CosineLSH {
 public:
  /// @brief Validate complete finite rows before owning a seeded Gaussian-plane index.
  /// @param data Rectangular finite rows; nonempty data require positive dimension. Zero vectors are allowed operationally.
  /// @param num_tables Number of tables, including zero for a deliberately empty candidate index.
  /// @param bits_per_table Bits in0..64; zero puts every row in key0 of each table.
  /// @param seed Seed for mt19937_64, consumed table/bit/coordinate order by normal_distribution.
  /// @throws std::invalid_argument On malformed/nonfinite data or a signature width above64.
  /// @throws std::length_error For unrepresentable allocation geometry.
  /// @note Seeded normal samples depend on the standard-library distribution implementation.
  /// @par Example
  /// @code
  /// dm::algorithms::CosineLSH index({{1,0},{0,1}},1,0,7);
  /// std::print("{}\n",index.dimension()); // 2
  /// @endcode
  /// @par References
  /// - Charikar(2002), Similarity Estimation Techniques from Rounding Algorithms, https://doi.org/10.1145/509907.509965; checked owned finite Gaussian-plane implementation here.
  /// @see CosineLSH, LSHQueryTrace, lsh_collision_probability()
  CosineLSH(
      const std::vector<std::vector<double>>& data,
      std::size_t                             num_tables,
      std::size_t                             bits_per_table,
      std::uint64_t                           seed = 0);
  /// @brief Inspect the owned row dimension, zero for empty input.
  /// @return Immutable indexed coordinate count.
  /// @par Example
  /// @code
  /// dm::algorithms::CosineLSH index({{1,0},{0,1}},1,0,7);
  /// std::print("{}\n",index.dimension()); // 2
  /// @endcode
  /// @par References
  /// - Charikar(2002), Similarity Estimation Techniques from Rounding Algorithms, https://doi.org/10.1145/509907.509965; checked owned finite Gaussian-plane implementation here.
  /// @see CosineLSH, LSHQueryTrace, lsh_collision_probability()
  std::size_t dimension() const noexcept { return dim_; }
  /// @brief Inspect the configured number of tables.
  /// @return Immutable table count, possibly zero.
  /// @par Example
  /// @code
  /// dm::algorithms::CosineLSH index({{1,0},{0,1}},1,0,7);
  /// std::print("{}\n",index.num_tables()); // 1
  /// @endcode
  /// @par References
  /// - Charikar(2002), Similarity Estimation Techniques from Rounding Algorithms, https://doi.org/10.1145/509907.509965; checked owned finite Gaussian-plane implementation here.
  /// @see CosineLSH, LSHQueryTrace, lsh_collision_probability()
  std::size_t num_tables() const noexcept { return tables_; }
  /// @brief Inspect the checked signature width.
  /// @return Immutable bit count in0..64.
  /// @par Example
  /// @code
  /// dm::algorithms::CosineLSH index({{1,0},{0,1}},1,0,7);
  /// std::print("{}\n",index.bits_per_table()); // 0
  /// @endcode
  /// @par References
  /// - Charikar(2002), Similarity Estimation Techniques from Rounding Algorithms, https://doi.org/10.1145/509907.509965; checked owned finite Gaussian-plane implementation here.
  /// @see CosineLSH, LSHQueryTrace, lsh_collision_probability()
  std::size_t bits_per_table() const noexcept { return bits_; }
  /// @brief Borrow immutable owned input rows; valid for the lifetime of this index.
  /// @return Const reference to copied original data in insertion order.
  /// @par Example
  /// @code
  /// dm::algorithms::CosineLSH index({{1,0},{0,1}},1,0,7);
  /// std::print("{}\n",index.data().size()); // 2
  /// @endcode
  /// @par References
  /// - Charikar(2002), Similarity Estimation Techniques from Rounding Algorithms, https://doi.org/10.1145/509907.509965; checked owned finite Gaussian-plane implementation here.
  /// @see CosineLSH, LSHQueryTrace, lsh_collision_probability()
  const std::vector<std::vector<double>>& data() const noexcept { return data_; }
  /// @brief Borrow actual seeded Gaussian hyperplanes; valid for the lifetime of this index.
  /// @return Actual coefficients indexed table/bit/coordinate, without regeneration.
  /// @par Example
  /// @code
  /// dm::algorithms::CosineLSH index({{1,0},{0,1}},1,0,7);
  /// std::print("{}\n",index.hyperplanes().size()); // 1
  /// @endcode
  /// @par References
  /// - Charikar(2002), Similarity Estimation Techniques from Rounding Algorithms, https://doi.org/10.1145/509907.509965; checked owned finite Gaussian-plane implementation here.
  /// @see CosineLSH, LSHQueryTrace, lsh_collision_probability()
  const std::vector<std::vector<std::vector<double>>>& hyperplanes() const noexcept {
    return hyperplanes_;
  }
  /// @brief Return the actual first strict best collided candidate, preserving ordered-table/bucket tie behavior.
  /// @param q Complete finite row with exactly dimension() entries, including empty for an empty-data index.
  /// @param out_examined Optional successful-call output count of distinct cosine evaluations; unchanged on guards.
  /// @return Actual best input identity, or SIZE_MAX if no bucket members collide.
  /// @throws std::invalid_argument For invalid complete query shape/numbers.
  /// @par Example
  /// @code
  /// dm::algorithms::CosineLSH index({{1,0},{0,1}},1,0,7);
  /// std::print("{}\n",index.query({0,1})); // 1
  /// @endcode
  /// @par References
  /// - Charikar(2002), Similarity Estimation Techniques from Rounding Algorithms, https://doi.org/10.1145/509907.509965; checked owned finite Gaussian-plane implementation here.
  /// @see CosineLSH, LSHQueryTrace, lsh_collision_probability()
  std::size_t query(const std::vector<double>& q, std::size_t* out_examined = nullptr) const;
  /// @brief Run the same producing query and return its actual work/result without history.
  /// @param q Complete finite dimension-matched query.
  /// @return Same-run selected candidate, cosine and actual query work.
  /// @throws std::invalid_argument For invalid complete query inputs.
  /// @par Example
  /// @code
  /// dm::algorithms::CosineLSH index({{1,0},{0,1}},1,0,7);
  /// std::print("{}\n",index.query_analysis({1,0}).examined); // 2
  /// @endcode
  /// @par References
  /// - Charikar(2002), Similarity Estimation Techniques from Rounding Algorithms, https://doi.org/10.1145/509907.509965; checked owned finite Gaussian-plane implementation here.
  /// @see CosineLSH, LSHQueryTrace, lsh_collision_probability()
  LSHQueryAnalysis query_analysis(const std::vector<double>& q) const;
  /// @brief Capture actual projections, bucket lookups and candidate visits without query replay.
  /// @param q Complete finite dimension-matched query.
  /// @param max_entries Optional logical capture limit, one million by default; zero disables it.
  /// @return Independent actual records; planes/data stay in the immutable index, not copied into the trace.
  /// @throws std::invalid_argument For malformed complete query.
  /// @throws std::length_error Before each record/copy if capture admission would exceed its cap.
  /// @note Counts d query coordinates, each projection, each table probe plus bucket member, and each candidate visit.
  /// Excludes scalar fields, capacities, analysis, index/runtime copies, seen marks and bytes. Complete trace or exception.
  /// @par Example
  /// @code
  /// dm::algorithms::CosineLSH index({{1,0},{0,1}},1,0,7);
  /// std::print("{}\n",index.query_trace({1,0}).analysis.examined); // 2
  /// @endcode
  /// @par References
  /// - Charikar(2002), Similarity Estimation Techniques from Rounding Algorithms, https://doi.org/10.1145/509907.509965; checked owned finite Gaussian-plane implementation here.
  /// @see CosineLSH, LSHQueryTrace, lsh_collision_probability()
  LSHQueryTrace query_trace(const std::vector<double>& q, std::size_t max_entries = 1000000) const;
  /// @brief Compute checked cosine similarity with finite-range scaling fallback; either zero vector yields0.
  /// @param a Complete finite first vector.
  /// @param b Complete finite second vector of identical size; empty pairs return0.
  /// @return Finite cosine, preserving historical finite ordinary arithmetic where representable.
  /// @throws std::invalid_argument For dimension mismatch or nonfinite coordinates.
  /// @par Example
  /// @code
  /// std::print("{}\n",dm::algorithms::CosineLSH::cosine({1,0},{0,1})); // 0
  /// @endcode
  /// @par References
  /// - Charikar(2002), Similarity Estimation Techniques from Rounding Algorithms, https://doi.org/10.1145/509907.509965; checked owned finite Gaussian-plane implementation here.
  /// @see CosineLSH, LSHQueryTrace, lsh_collision_probability()
  static double cosine(const std::vector<double>& a, const std::vector<double>& b);
  /// @brief Assemble a real signature using the stored planes and nonnegative-dot bit rule.
  /// @param x Complete finite dimension-matched vector.
  /// @param t Valid table index.
  /// @return Actual unsigned signature, supporting bit63 without undefined shifts.
  /// @throws std::invalid_argument For malformed complete vector.
  /// @throws std::out_of_range For invalid table index.
  /// @par Example
  /// @code
  /// dm::algorithms::CosineLSH index({{1,0}},1,1,7);
  /// std::print("{}\n",index.signature({0,0},0)); // 1
  /// @endcode
  /// @par References
  /// - Charikar(2002), Similarity Estimation Techniques from Rounding Algorithms, https://doi.org/10.1145/509907.509965; checked owned finite Gaussian-plane implementation here.
  /// @see CosineLSH, LSHQueryTrace, lsh_collision_probability()
  std::uint64_t signature(const std::vector<double>& x, std::size_t t) const;

 private:
  std::vector<std::vector<double>> data_; ///< Owned immutable insertion-order finite data.
  std::size_t                      dim_; ///< Checked positive row dimension, or zero on empty data.
  std::size_t                      tables_; ///< Immutable possibly-zero table count.
  std::size_t                      bits_;   ///< Immutable checked signature width in0..64.
  std::vector<std::vector<std::vector<double>>>
      hyperplanes_; ///< Owned actual seeded Gaussian coefficients, table/bit/coordinate.
  std::vector<std::unordered_map<std::uint64_t, std::vector<std::size_t>>>
      buckets_; ///< Actual table/key insertion-order ids.

  /// @brief Shared actual query producer with optional admitted capture.
  /// @param q Original complete query.
  /// @param trace Optional actual history output.
  /// @param quota Optional selected capture cap.
  /// @return Actual query result/work.
  LSHQueryAnalysis produce(const std::vector<double>& q, LSHQueryTrace* trace, std::size_t quota)
      const;
  /// @brief Shared actual signature producer with optional query work/capture.
  /// @param x Original complete checked query.
  /// @param t Actual checked table id.
  /// @param work Optional selected query operation counts.
  /// @param trace Optional actual query capture.
  /// @param quota Optional logical capture cap.
  /// @return Actual stored-plane signature.
  std::uint64_t hash(
      const std::vector<double>& x,
      std::size_t                t,
      LSHQueryAnalysis*          work,
      LSHQueryTrace*             trace,
      std::size_t                quota) const;
};
/// @brief Ideal single isotropic hyperplane-bit collision probability for a finite angle in[0,pi].
/// @param theta Angle in radians, inclusive endpoints.
/// @return1-theta/pi, in[0,1]; does not apply to zero-vector angles or prove finite-index recall.
/// @throws std::invalid_argument For nonfinite or out-of-range angles.
/// @code
/// std::print("{}\n",dm::algorithms::lsh_collision_probability(0)); // 1
/// @endcode
double lsh_collision_probability(double theta);
} // namespace datamunge::algorithms
