#pragma once
/// @file lsh_study.hpp
/// @brief Original seeded hyperplane-collision and near-duplicate ANN study with actual search outcomes.
#include <datamunge/algorithms/lsh.hpp>
namespace datamunge::algorithms {
/// @brief Actual independent Gaussian-plane agreement count at a prescribed two-dimensional angle.
/// @par Example
/// @code
/// auto s=dm::algorithms::lsh_original_study();
/// std::print("{}\n",s.angles.front().trials); // 20000
/// @endcode
/// @par References
/// - Charikar(2002), https://doi.org/10.1145/509907.509965; the original seeded synthetic workload is an illustrative control, not a recall guarantee.
/// @see CosineLSH, lsh_original_study()
struct LSHCollisionCount {
  int         degrees;    ///< Prescribed angle0,20,...,180.
  std::size_t agreements; ///< Actual sign agreements in20000 independently drawn planes.
  std::size_t trials;     ///< Actual independent plane draws.
  double      empirical;  ///< Actual agreements divided by trials.
  double      theory; ///< Ideal isotropic law1-theta/pi, not a deterministic agreement guarantee.
};
/// @brief One actual original near-duplicate query, exact baseline and approximate result.
/// @par Example
/// @code
/// auto s=dm::algorithms::lsh_original_study();
/// std::print("{}\n",s.queries.front().query.size()); // 30
/// @endcode
/// @par References
/// - Charikar(2002), https://doi.org/10.1145/509907.509965; the original seeded synthetic workload is an illustrative control, not a recall guarantee.
/// @see CosineLSH, lsh_original_study()
struct LSHDemoQuery {
  std::vector<double> query;       ///< Actual original noisy30-dimensional query.
  std::size_t         source;      ///< Original sampled source identity before independent noise.
  std::size_t         exact;       ///< First strict maximum from a real full cosine scan.
  std::size_t         approximate; ///< Actual collided-candidate winner, or SIZE_MAX if absent.
  std::size_t         examined;    ///< Actual distinct approximate cosine evaluations.
  double              exact_similarity; ///< Actual exact-scan winning cosine.
  double approximate_similarity;        ///< Actual approximate cosine, sentinel minus2 if absent.
};
/// @brief Owned original collision experiment, synthetic index data and actual independent search outcomes.
/// @par Example
/// @code
/// auto s=dm::algorithms::lsh_original_study();
/// std::print("{}\n",s.queries.size()); // 200
/// @endcode
/// @par References
/// - Charikar(2002), https://doi.org/10.1145/509907.509965; the original seeded synthetic workload is an illustrative control, not a recall guarantee.
/// @see CosineLSH, lsh_original_study()
struct LSHDemoStudy {
  std::vector<LSHCollisionCount>
      angles; ///< Actual agreement studies, generated before the indexed data.
  std::vector<std::vector<double>> data; ///< Actual2000 owned Gaussian30-dimensional rows.
  std::vector<LSHDemoQuery>
      queries; ///< Actual200 noisy near-duplicate queries with exact and approximate results.
};
/// @brief Run the original complete seed3 fixture, including its exact generator consumption order.
/// @return Owned angle counts, actual data and exact/approximate query results for15 tables/14 bits/seed99.
/// @note Draws20000 Gaussian planes per angle0..180 by20, then the original2000-by30 data and200 queries.
/// Seeded distribution bit patterns are standard-library dependent. Measured recall/candidate counts describe
/// this synthetic workload, not a universal guarantee, elapsed-time speedup or sublinear total query complexity.
/// @code
/// auto study=dm::algorithms::lsh_original_study();
/// std::print("{} {}\n",study.data.size(),study.queries.size()); // 2000 200
/// @endcode
LSHDemoStudy lsh_original_study();
} // namespace datamunge::algorithms
