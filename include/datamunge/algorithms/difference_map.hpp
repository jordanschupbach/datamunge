#pragma once
/// @file difference_map.hpp
/// @brief Checked difference-map iterations, actual cross-projection decoding and histories.
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Original result layout with a decoded B-side cross-projection estimate.
/// After positive iterations solution is P_B(f_A(x_before)), not P_B(x_after).
/// At a true fixed point it equals P_A(f_B(x_before)) and lies in both sets if
/// callbacks are actual projections. converged retains only the strict numerical
/// step<tol rule: small beta can hide a large projection gap. It is not a feasibility
/// certificate. With zero iterations, solution=P_B(x0), residual=0 is an unevaluated
/// historical sentinel, and converged=false. Containers own data; defaults are empty/zero.
/// @par Example
/// @code
/// const auto id=[](const std::vector<double>& x){return x;};
/// const auto r=dm::algorithms::difference_map(id,id,{2.0});
/// std::print("{} {} {}\n",r.solution,r.iterations,r.converged); // [2] 1 true
/// @endcode
/// @see difference_map(), DifferenceMapTrace
struct DifferenceMapResult {
  std::vector<double> solution; ///< Last B-side cross estimate, or P_B(x0) with zero budget.
  int iterations{0}; ///< Number of completed updates.
  double residual{0}; ///< Last Euclidean step norm; zero sentinel when no step was evaluated.
  bool converged{false}; ///< Strict step<tol numerical stop flag, not certified feasibility.
};
/// @brief Type-erased projection from a borrowed vector to an owned vector of the same size.
/// Outputs must be finite. Callbacks may hold external state; the algorithm validates
/// shape/finiteness but cannot establish nearest-point or set-membership properties.
/// @par Example
/// @code
/// const dm::algorithms::DifferenceProjection id=[](const std::vector<double>& x){return x;};
/// std::print("{}\n",id({3.0})); // [3]
/// @endcode
/// @see difference_map_trace(), project_affine_hyperplane(), project_finite_set()
using DifferenceProjection=std::function<std::vector<double>(const std::vector<double>&)>;
/// @brief Reason iteration stopped; neither value certifies constraint feasibility.
/// Zero tolerance disables the strict step test, even at an exact fixed point.
/// @par Example
/// @code
/// const auto id=[](const std::vector<double>& x){return x;};
/// const auto t=dm::algorithms::difference_map_trace(id,id,{1.0},1,1,0);
/// std::print("{}\n",t.stop==dm::algorithms::DifferenceMapStop::iteration_limit); // true
/// @endcode
/// @see DifferenceMapTrace
 enum class DifferenceMapStop {
  step_tolerance, ///< The computed step norm was strictly below tol.
  iteration_limit ///< Completed max_iter updates, possibly zero; no successful strict test.
};
/// @brief Owned vectors from one actual four-projection difference-map update.
/// Calls occur in order A(x), B(x), A(f_B), B(f_A). f_A=(1-1/beta)A(x)+x/beta;
/// f_B=(1+1/beta)B(x)-x/beta. estimate_a and estimate_b are the cross projections;
/// x_after=x_before+beta*(estimate_a-estimate_b). Eight n-vectors are retained.
/// @par Example
/// @code
/// const auto id=[](const std::vector<double>& x){return x;};
/// const auto t=dm::algorithms::difference_map_trace(id,id,{2.0});
/// std::print("{} {} {}\n",t.steps[0].x_before,t.steps[0].estimate_b,t.steps[0].x_after); // [2] [2] [2]
/// @endcode
/// @see difference_map_trace()
struct DifferenceMapStep {
  int iteration{0}; ///< One-based completed update number in native traces.
  std::vector<double> x_before; ///< Actual input to the two initial projections.
  std::vector<double> projection_a; ///< Actual P_A(x_before).
  std::vector<double> projection_b; ///< Actual P_B(x_before).
  std::vector<double> f_a; ///< Actual relaxed A vector.
  std::vector<double> f_b; ///< Actual relaxed B vector.
  std::vector<double> estimate_a; ///< Actual P_A(f_b), in A for a genuine projector.
  std::vector<double> estimate_b; ///< Actual P_B(f_a), the returned B-side estimate.
  std::vector<double> x_after; ///< Actual updated iterate; it need not be feasible.
  double step_norm{0}; ///< Stable hypot norm of the applied displacement.
  double gap_norm{0}; ///< Stable hypot norm of estimate_a-estimate_b, before beta scaling.
};
/// @brief Actual callback/update counts and retained-history entry count.
/// Each completed update calls A and B twice and updates n coordinates. A zero
/// budget calls only B once. Counts exclude callback internals, validation, norm
/// arithmetic, copying and allocation. History entries count one per record plus
/// its eight n-vectors, excluding initial/final/result vectors and transient buffers.
/// @par Example
/// @code
/// const auto id=[](const std::vector<double>& x){return x;};
/// const auto t=dm::algorithms::difference_map_trace(id,id,{2.0});
/// std::print("{} {} {} {}\n",t.work.projection_a_calls,t.work.projection_b_calls,t.work.coordinate_updates,t.work.history_entries); // 2 2 1 9
/// @endcode
/// @see difference_map_history_cost(), DifferenceMapTrace
struct DifferenceMapWork {
  std::size_t projection_a_calls{0}; ///< Actual A callback invocations.
  std::size_t projection_b_calls{0}; ///< Actual B callback invocations, including the zero-budget fallback.
  std::size_t coordinate_updates{0}; ///< Actual coordinate additions in completed updates.
  std::size_t history_entries{0}; ///< Retained records plus their coordinate values.
};
/// @brief Owned initial/final iterates, actual steps, decoded result and stop metadata.
/// No replay or independent reconstruction. Callback side effects occur only during
/// the recorded execution; a thrown call returns no partial trace but prior external
/// effects remain. The iterate, B-side estimate and cross-projection gap are distinct.
/// No shared mutable state except what callbacks capture. Empty vectors are valid
/// zero-dimensional states if both callbacks return empty vectors.
/// @par Example
/// @code
/// const auto id=[](const std::vector<double>& x){return x;};
/// const auto t=dm::algorithms::difference_map_trace(id,id,{2.0});
/// std::print("{} {} {}\n",t.initial,t.final_iterate,t.steps.size()); // [2] [2] 1
/// @endcode
/// @see difference_map_trace(), DifferenceMapResult
struct DifferenceMapTrace {
  DifferenceMapResult result; ///< Legacy-layout result with corrected cross-projection decoding.
  std::vector<double> initial; ///< Owned original x0.
  std::vector<double> final_iterate; ///< Actual x after all completed updates, or x0 for zero budget.
  std::vector<DifferenceMapStep> steps; ///< Actual eight-vector update records, in order.
  DifferenceMapWork work; ///< Actual calls, updates and retained entries.
  DifferenceMapStop stop{DifferenceMapStop::iteration_limit}; ///< Actual numerical stopping reason.
  double beta{1}; ///< Requested nonzero finite relaxation parameter.
  double tolerance{1e-12}; ///< Requested nonnegative finite strict step tolerance.
  int max_iterations{10000}; ///< Requested nonnegative update budget.
};
/// @brief Checked entry cost of one actual retained n-dimensional update: 1+8n.
/// One entry counts the record, eight n entries count its vectors. Does not include
/// initial/final/result vectors, vector metadata or transient work. O(1), no allocation.
/// @param dimension State vector length, including zero.
/// @return Representable record-plus-coordinate cost.
/// @throws std::length_error If 1+8*dimension would overflow size_t.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::difference_map_history_cost(2)); // 17
/// @endcode
/// @see difference_map_trace()
std::size_t difference_map_history_cost(std::size_t dimension);
/// @brief Implementation bridge shared by the callable template and public trace.
namespace difference_map_detail {
/// @brief Borrow a live callback for a synchronous compiled-kernel invocation.
/// @tparam Projector Callable returning a vector from a const vector reference; move-only types are supported.
/// @param projector Live callable; the returned adapter must not outlive this object.
/// @return Type-erased adapter referring to projector without copying it.
/// @throws std::invalid_argument For null function pointers or empty DifferenceProjection objects.
/// @throws std::bad_alloc If type erasure allocates and fails.
template<class Projector> DifferenceProjection borrow(Projector& projector) {
  if constexpr(std::is_pointer_v<Projector>)if(projector==nullptr)throw std::invalid_argument("difference_map: null projection");
  if constexpr(std::is_same_v<Projector,DifferenceProjection>)if(!projector)throw std::invalid_argument("difference_map: empty projection");
  return [&projector](const std::vector<double>& x)->std::vector<double>{return projector(x);};
}
/// @brief Execute validated projections, update and optional actual history.
/// @param a Nonempty A callback, borrowed for this invocation.
/// @param b Nonempty B callback, borrowed for this invocation.
/// @param x Owned initial vector, moved through updates.
/// @param beta Nonzero finite parameter with finite reciprocal.
/// @param max_iter Nonnegative iteration budget.
/// @param tol Finite nonnegative strict step tolerance.
/// @param trace Null for ordinary calls; otherwise receives owned actual state.
/// @param max_history_entries Online record/coordinate cap, ignored without a trace; zero disables.
/// @return Correct decoded result with the preserved strict numerical stop flag.
/// @throws std::invalid_argument For invalid controls/inputs or callback shape/nonfinite output.
/// @throws std::overflow_error For a nonrepresentable reciprocal, relaxed state, gap, step or iterate.
/// @throws std::length_error For history/counter/capacity overflow or the online cap.
/// @throws std::bad_alloc For allocation failure. Callback exceptions propagate.
DifferenceMapResult run(const DifferenceProjection& a,const DifferenceProjection& b,std::vector<double> x,double beta,int max_iter,double tol,DifferenceMapTrace* trace,std::size_t max_history_entries);
} // namespace difference_map_detail
/// @brief Run the difference map with validated vectors and corrected solution decoding.
/// Preserves callable-by-value parameters, defaults and the original result layout.
/// Inputs/outputs must be finite; each output must keep x0's dimension. beta must
/// have a finite reciprocal. Updates preserve A(x),B(x),A(f_B),B(f_A) order; the
/// last B cross-projection is returned without an extra final projection call.
/// With max_iter=0 only B(x0) is evaluated, and the historical zero residual is
/// unevaluated. tol=0 disables strict stopping. O(n) driver storage and O(kn)
/// vector work for k updates, plus four projection costs per update. No general
/// nonconvex convergence or arbitrary-beta guarantee is made. Genuine projectors
/// are a caller responsibility; step<tol is not a membership certificate.
/// @tparam ProjA A projection callable; may be move-only and is held by value.
/// @tparam ProjB B projection callable; may be move-only and is held by value.
/// @param projA A callback, called twice per completed update, never for a zero budget.
/// @param projB B callback, called twice per completed update or once for a zero budget.
/// @param x0 Owned finite initial vector; empty dimension is allowed.
/// @param beta Nonzero finite parameter, default 1 (Douglas–Rachford order A after B).
/// @param max_iter Nonnegative update budget, default 10000.
/// @param tol Finite nonnegative strict step tolerance, default 1e-12.
/// @return Correct B-side cross estimate; see DifferenceMapResult for zero-budget behavior.
/// @throws std::invalid_argument For invalid inputs/controls, null/empty known callbacks or malformed outputs.
/// @throws std::overflow_error For nonfinite arithmetic, including an unrepresentable reciprocal.
/// @throws std::length_error For an actual counter/capacity overflow.
/// @throws std::bad_alloc For allocation failure. Callback exceptions propagate without rollback.
/// @par Example
/// @code
/// const auto a=[](const std::vector<double>& x){return dm::algorithms::project_finite_set(x,{{0.0},{3.0}});};
/// const auto b=[](const std::vector<double>& x){return dm::algorithms::project_finite_set(x,{{0.0},{2.0}});};
/// const auto r=dm::algorithms::difference_map(a,b,{2.25},.25);
/// std::print("{} {} {}\n",r.solution,r.iterations,r.converged); // [0] 1 true
/// @endcode
/// @par References
/// - V. Elser (2003), Phase retrieval by iterated projections, JOSA A 20, 40–55; https://arxiv.org/abs/math/0111080. Solution is the common cross-projection, not the fixed iterate or an arbitrary projection of it.
/// @see difference_map_trace(), DifferenceMapResult
 template<class ProjA,class ProjB>
DifferenceMapResult difference_map(ProjA projA,ProjB projB,std::vector<double> x0,double beta=1,int max_iter=10000,double tol=1e-12) {
  return difference_map_detail::run(difference_map_detail::borrow(projA),difference_map_detail::borrow(projB),std::move(x0),beta,max_iter,tol,nullptr,0);
}
/// @brief Record actual eight-vector steps, stable norms and projection calls without replay.
/// Same compiled kernel and decoder as difference_map(). Online entry caps are
/// checked before each new iteration's callbacks/retained snapshots; early stopping
/// therefore needs space only for actual records. One record costs 1+8n entries.
/// Initial/final/result vectors and temporary work are excluded from this history
/// cap. Zero disables only the optional cap; checked arithmetic remains. O(kn)
/// history in addition to driver storage/callback costs. Prior callback side effects
/// remain if a later iteration or cap throws; no partial trace is returned.
/// @param projA Nonempty A projection with finite dimension-preserving output.
/// @param projB Nonempty B projection with finite dimension-preserving output.
/// @param x0 Owned finite initial vector, including empty vectors.
/// @param beta Finite nonzero parameter with representable reciprocal, default 1.
/// @param max_iter Nonnegative update budget, default 10000.
/// @param tol Nonnegative finite strict step tolerance, default 1e-12; zero disables stopping.
/// @param max_history_entries Online record/coordinate cap, default 1000000; zero disables.
/// @return Owned actual steps, initial/final iterates, corrected result and numerical stop reason.
/// @throws std::invalid_argument For invalid controls/inputs or empty/malformed/nonfinite projections.
/// @throws std::overflow_error For nonrepresentable arithmetic.
/// @throws std::length_error For history/counter/capacity overflow or the actual history cap.
/// @throws std::bad_alloc For allocation failure. Callback exceptions propagate.
/// @par Example
/// @code
/// const auto id=[](const std::vector<double>& x){return x;};
/// const auto t=dm::algorithms::difference_map_trace(id,id,{2.0},1,10000,1e-12,9);
/// std::print("{} {}\n",t.steps.size(),t.work.history_entries); // 1 9
/// @endcode
/// @par References
/// - V. Elser (2003), Phase retrieval by iterated projections, https://arxiv.org/abs/math/0111080.
/// @see difference_map(), difference_map_history_cost(), DifferenceMapTrace
DifferenceMapTrace difference_map_trace(DifferenceProjection projA,DifferenceProjection projB,std::vector<double> x0,double beta=1,int max_iter=10000,double tol=1e-12,std::size_t max_history_entries=1000000);
/// @brief Euclidean projection onto the affine hyperplane normal dot x = offset.
/// Scales the normal by its largest component, accumulates in long double, and
/// checks intermediates/results; wider arithmetic depends on the platform. No exact
/// or universal full-range guarantee. O(n) work/storage; inputs borrowed unchanged,
/// output owned, no shared state. Zero-dimensional/zero normals are invalid.
/// @param point Finite vector to project.
/// @param normal Finite nonzero normal, of the same positive dimension as point.
/// @param offset Finite affine right-hand side.
/// @return Owned projected vector, with ordinary floating rounding.
/// @throws std::invalid_argument For empty/mismatched dimensions, nonfinite inputs or zero normal.
/// @throws std::overflow_error For nonfinite intermediate/output arithmetic.
/// @throws std::bad_alloc For allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::project_affine_hyperplane({7,-3},{1,-1},0)); // [2, 2]
/// @endcode
/// @par References
/// - V. Elser (2003), Phase retrieval by iterated projections, https://arxiv.org/abs/math/0111080 (Euclidean nearest-point projections).
/// @see project_finite_set(), difference_map()
std::vector<double> project_affine_hyperplane(const std::vector<double>& point,const std::vector<double>& normal,double offset);
/// @brief Return the closest listed finite point, using earliest input on a numerical tie.
/// Validates every point before searching, including unreachable/nonwinning entries.
/// Empty-dimensional points are allowed, but the point set must be nonempty. Uses
/// long-double hypot distances with checked intermediates; no exact tie predicate.
/// O(M*n) validation/search work and O(n) owned output, ignoring borrowed set storage.
/// No input mutation or shared state. Represents a generally nonconvex constraint.
/// @param point Finite n-vector to project.
/// @param points Nonempty list of finite n-vectors; duplicates allowed, order breaks ties.
/// @return Owned copy of the earliest minimum-distance point.
/// @throws std::invalid_argument For empty set, inconsistent shapes or any nonfinite input.
/// @throws std::overflow_error If a distance intermediate is nonfinite.
/// @throws std::bad_alloc For output allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::project_finite_set({1},{{0},{2}})); // [0]
/// @endcode
/// @par References
/// - V. Elser (2003), Phase retrieval by iterated projections, https://arxiv.org/abs/math/0111080 (nearest-point constraints; numerical tie rule specified here).
/// @see project_affine_hyperplane(), difference_map_trace()
std::vector<double> project_finite_set(const std::vector<double>& point,const std::vector<std::vector<double>>& points);
} // namespace datamunge::algorithms
