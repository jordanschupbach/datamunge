#pragma once
/// @file nearest_neighbor_interpolation.hpp
/// @brief Prepared checked nearest-sample interpolation with actual binary-search decisions.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual inclusive binary-search bounds and abscissa comparison.
/// @par Example
/// @code
/// dm::algorithms::NearestNeighborInterpolant m({0,2,5},{10,20,30});
/// std::print("{}\n",m.trace(1).steps.front().mid); // 1
/// @endcode
/// @see NearestNeighborTrace
struct NearestNeighborStep {
  std::size_t lo{0};           ///< Actual inclusive lower index before this comparison.
  std::size_t hi{0};           ///< Actual inclusive upper index before this comparison.
  std::size_t mid{0};          ///< Actual floor midpoint, computed without index-sum overflow.
  double      abscissa{0};     ///< Actual xs[mid] compared with the query.
  bool        go_right{false}; ///< True exactly when xs[mid]<query.
};
/// @brief Owned nodes/query and actual binary-search/nearest-choice observations.
/// Capture units are both node arrays, query/result scalar records and actual
/// search-step records:2n+2+steps. Empty result index0 is a placeholder.
/// Route0=empty,1=lower endpoint,2=upper endpoint,3=interior nearest choice.
/// Distances are meaningful only for route3 and use widened arithmetic.
/// @par Example
/// @code
/// dm::algorithms::NearestNeighborInterpolant m({0,2},{10,20});auto t=m.trace(1);
/// std::print("{} {}\n",t.selected,t.value); // 0 10
/// @endcode
/// @see NearestNeighborInterpolant::trace()
struct NearestNeighborTrace {
  std::vector<double> xs;                ///< Owned original validated abscissae.
  std::vector<double> ys;                ///< Owned original validated ordinates.
  double              query{0};          ///< Original finite query.
  double              value{0};          ///< Actual selected ordinate, zero for empty input.
  std::size_t         selected{0};       ///< Actual selected sample ID, or0 empty placeholder.
  unsigned            route{0};          ///< Actual early/nearest route0..3.
  std::size_t         endpoint_tests{0}; ///< Actual query comparisons with first/last abscissae.
  long double         left_distance{0}; ///< Actual widened distance to the lower bracketing sample.
  long double right_distance{0};        ///< Actual widened distance to the upper bracketing sample.
  std::vector<NearestNeighborStep> steps; ///< Actual binary-search iterations, no replay.
  std::size_t                      captured_entries{0}; ///< Actual admitted history units.
};
/// @brief Owned prepared one-dimensional nearest-sample interpolant.
/// Equal-size finite node arrays, with strictly increasing xs. Empty arrays are
/// allowed and produce zero by legacy convention; this is not a data estimate.
/// Finite out-of-range queries select the nearest endpoint; exact midpoint ties
/// select the lower index. Ordinates may be nonmonotone. No smoothing or new
/// ordinate values are introduced. O(n) validation/setup/storage, then O(log(n+1))
/// query work and constant query-driver storage. Const queries do not mutate.
/// @par Example
/// @code
/// dm::algorithms::NearestNeighborInterpolant m({0,2},{10,20});
/// std::print("{} {}\n",m.evaluate(1),m.evaluate(1.5)); // 10 20
/// @endcode
/// @par References
/// - https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html (nearest lower-neighbor midpoint convention; native validation/endpoint policy here).
/// @see nearest_neighbor_interpolate()
class NearestNeighborInterpolant {
 public:
  /// @brief Validate and own complete node arrays once for repeated queries.
  /// @param xs Finite strictly increasing abscissae, empty allowed.
  /// @param ys Same-size finite ordinates.
  /// @param max_nodes Complete node cap, default1000000; zero disables.
  /// @throws std::invalid_argument If shape, finiteness or ordering is invalid.
  /// @throws std::length_error If complete node count exceeds a nonzero cap.
  /// @par Example
  /// @code
  /// dm::algorithms::NearestNeighborInterpolant m({0,2},{10,20});std::print("{}\n",m.evaluate(3)); // 20
  /// @endcode
  /// @see NearestNeighborInterpolant
  NearestNeighborInterpolant(
      std::vector<double> xs,
      std::vector<double> ys,
      std::size_t         max_nodes = 1000000);
  /// @brief Select the actual nearest ordinate with lower-index midpoint ties.
  /// @param x Finite query, including values outside the node range.
  /// @return Actual nearest stored ordinate; zero for an empty model.
  /// @throws std::invalid_argument If x is nonfinite, including on an empty model.
  /// @par Example
  /// @code
  /// dm::algorithms::NearestNeighborInterpolant m({0,2},{10,20});std::print("{}\n",m.evaluate(1)); // 10
  /// @endcode
  /// @see trace()
  double evaluate(double x) const;
  /// @brief Capture actual search bounds and nearest-sample decision from evaluate's producer.
  /// No replay. Copies validated model nodes into the owned trace after admitting
  /// their complete2n+2 base units, then admits actual search records online.
  /// No partial result escapes failure; driver work follows evaluate().
  /// @param x Finite query, retained unreduced.
  /// @param max_entries History quota, default1000000; zero disables.
  /// @return Owned actual query history and selected ordinate.
  /// @throws std::invalid_argument If x is nonfinite, before capture.
  /// @throws std::length_error If complete base geometry or actual quota is exceeded.
  /// @throws std::bad_alloc If capture allocation fails.
  /// @par Example
  /// @code
  /// dm::algorithms::NearestNeighborInterpolant m({0,2},{10,20});std::print("{}\n",m.trace(1).value); // 10
  /// @endcode
  /// @see NearestNeighborTrace
  NearestNeighborTrace trace(double x, std::size_t max_entries = 1000000) const;
  /// @brief Borrow immutable validated sample coordinates.
  /// @return Model-lifetime const abscissa reference.
  const std::vector<double>& xs() const noexcept { return xs_; }
  /// @brief Borrow immutable validated ordinates.
  /// @return Model-lifetime const ordinate reference.
  const std::vector<double>& ys() const noexcept { return ys_; }

 private:
  std::vector<double> xs_; ///< Owned complete validated abscissae.
  std::vector<double> ys_; ///< Owned complete validated ordinates.

  /// @brief Shared actual binary-search/selection producer.
  /// @param x Validated finite query.
  /// @param out Optional actual-history sink.
  /// @param cap Actual capture quota, zero disables.
  /// @return Actual selected ordinate.
  /// @throws std::length_error If capture admission exceeds quota.
  double run(double x, NearestNeighborTrace* out, std::size_t cap) const;
};
/// @brief Validate, prepare and evaluate a single nearest-neighbor query.
/// Uses the same producer as NearestNeighborInterpolant, with its optional node
/// cap disabled. O(n) setup plus O(log(n+1)) query work, O(n) copied driver storage.
/// Reuse a prepared model for many queries. Empty matching arrays return zero;
/// complete shape/finiteness/order validation now precedes endpoint indexing.
/// @param xs Finite strictly increasing abscissae, empty allowed.
/// @param ys Same-size finite ordinates.
/// @param x Finite query, nearest endpoint outside the node range.
/// @return Actual nearest ordinate with lower-index exact ties, or zero if empty.
/// @throws std::invalid_argument If node geometry or query is invalid.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::nearest_neighbor_interpolate({0,2},{10,20},1)); // 10
/// @endcode
/// @see NearestNeighborInterpolant
double nearest_neighbor_interpolate(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    double                     x);
/// @brief Owned original smooth-function demonstration samples.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::nearest_neighbor_demo().xs.size()); // 9
/// @endcode
/// @see nearest_neighbor_demo()
struct NearestNeighborDemo {
  std::vector<double> xs; ///< Original abscissae0,.8,...,6.4.
  std::vector<double> ys; ///< Actual sin(x)+.3*x samples at these abscissae.
};
/// @brief Generate the original nine coarse smooth-function samples natively.
/// @return Owned original nodes, generated by i*.8 then sin(x)+.3*x.
/// @par Example
/// @code
/// auto d=dm::algorithms::nearest_neighbor_demo();std::print("{}\n",d.ys.front()); // 0
/// @endcode
/// @see NearestNeighborInterpolant
NearestNeighborDemo nearest_neighbor_demo();
/// @brief Owned native smooth-function interpolation and finite-grid error study.
/// The reference bound1.3*h/2 uses |cos(x)+.3|<=1.3 on the sampled domain.
/// It bounds exact-arithmetic nearest-sample error on[0,6.4], not extrapolation
/// or floating-point roundoff; the reported maximum is only over its query grid.
/// @par Example
/// @code
/// auto s=dm::algorithms::nearest_neighbor_demo_study();std::print("{}\n",s.queries.size()); // 321
/// @endcode
/// @see nearest_neighbor_demo_study()
struct NearestNeighborStudy {
  NearestNeighborDemo samples; ///< Actual uniformly spaced sample nodes/values.
  std::vector<double> queries; ///< Actual integer-generated grid in[0,6.4], endpoints included.
  std::vector<double> truth;   ///< Actual native sin(x)+.3*x reference values.
  std::vector<double> values;  ///< Actual prepared interpolant values.
  std::vector<double> absolute_error;     ///< Actual absolute reference differences.
  double              max_grid_error{0};  ///< Observed finite-grid maximum absolute error.
  double              error_reference{0}; ///< Exact-arithmetic Lipschitz reference1.3*h/2.
};
/// @brief Run the original smooth-function example using one prepared native model.
/// O(nodes+grid_intervals*log(nodes+1)) work and O(nodes+grid_intervals) owned
/// storage. Grid points use6.4*i/grid_intervals to avoid accumulated step drift.
/// @param nodes At least two nodes, default9; at most1000000.
/// @param grid_intervals Positive grid intervals, default320; at most1000000.
/// @return Owned actual samples, queries, interpolation/errors and theoretical reference.
/// @throws std::invalid_argument If nodes<2 or grid_intervals=0.
/// @throws std::length_error If either count exceeds1000000.
/// @par Example
/// @code
/// auto s=dm::algorithms::nearest_neighbor_demo_study();std::print("{}\n",s.error_reference); // 0.52
/// @endcode
/// @see NearestNeighborStudy, NearestNeighborInterpolant
NearestNeighborStudy nearest_neighbor_demo_study(
    std::size_t nodes          = 9,
    std::size_t grid_intervals = 320);
} // namespace datamunge::algorithms
