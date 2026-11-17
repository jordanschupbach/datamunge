#pragma once
/// @file moead.hpp
/// @brief Decomposition optimizer with actual scalar replacements and owned execution traces.
#include <cstddef>
#include <cstdint>
#include <datamunge/optim/multi_objective.hpp>
#include <string>
#include <vector>
namespace datamunge::optim {
/// @brief Owned configuration for the native MOEA/D variant.
/// Validation occurs on optimize()/trace(), before callbacks except objective
/// dimension, which requires one actual discarded probe. Same legacy field order.
/// @par Example
/// @code
/// dm::optim::MOEADOptions o;o.population_size=5;o.max_generations=0;
/// std::print("{}\n",o.population_size); // 5
/// @endcode
/// @see MOEAD
struct MOEADOptions {
  std::size_t population_size{
      100}; ///< Requested lattice size,2..1000000; actual largest lattice<=max(requested,objectives).
  std::size_t max_generations{
      250}; ///< Complete sequential sweeps, zero permits initialization-only runs.
  std::size_t neighborhood_size{
      20}; ///< Effective size min(max(requested,2),actual population), includes self.
  std::string decomposition{"tchebycheff"}; ///< Exactly tchebycheff or weighted_sum.
  double      crossover_rate{.9};           ///< Finite SBX probability in[0,1].
  double      eta_crossover{20};            ///< Finite nonnegative SBX distribution index.
  double mutation_rate{-1}; ///< Any finite negative selects1/nvars; otherwise probability in[0,1].
  double eta_mutation{20};  ///< Finite nonnegative polynomial mutation index.
  std::uint64_t seed{42};   ///< MT19937-64 seed, including zero.
};
/// @brief Actual two scalar values and update decision for one neighboring slot.
/// Both scores use the same ideal point after the child evaluation. Equality
/// replaces, so duplicate individuals can propagate. Values are observed, not replayed.
/// @par Example
/// @code
/// dm::optim::MOEADReplacement r;
/// std::print("{}\n",r.replaced); // false
/// @endcode
/// @see MOEADStep
struct MOEADReplacement {
  std::size_t neighbor{0};        ///< Actually visited neighbor slot.
  double      incumbent_score{0}; ///< Actual incumbent scalarization before this update.
  double      child_score{0};     ///< Actual child scalarization at this neighbor's weight.
  bool        replaced{false};    ///< Actual child_score<=incumbent_score decision.
};
/// @brief Actual offspring and all neighbor updates from one subproblem visit.
/// Parents are drawn with replacement and can coincide. Only the first SBX
/// child is mutated/evaluated. These are sequential in-place population updates.
/// @par Example
/// @code
/// dm::optim::MOEADStep s;
/// std::print("{}\n",s.replacements.size()); // 0
/// @endcode
/// @see MOEADTrace
struct MOEADStep {
  std::size_t         generation{0}; ///< Zero-based sweep index.
  std::size_t         subproblem{0}; ///< Actually visited scalar subproblem slot.
  std::size_t         parent_a{0};   ///< Actual first parent slot.
  std::size_t         parent_b{0};   ///< Actual second parent slot, possibly equal to first.
  ParetoPoint         child;         ///< Owned evaluated first offspring.
  std::vector<double> ideal;         ///< Actual componentwise ideal after this child.
  std::vector<MOEADReplacement> replacements; ///< Actual neighbor visitation order and decisions.
};
/// @brief Owned actual setup, offspring, scalar decisions and final population/front.
/// The probe determines objective count but is discarded from the population and
/// ideal point. Capture units count bound scalars, each stored vector row plus its
/// elements, each ParetoPoint record plus coordinates/objectives, ideal scalars,
/// each offspring record plus child coordinates/objectives/ideal, and each
/// replacement record. Options and other scalar metadata are not additional units.
/// @par Example
/// @code
/// dm::optim::MOEADTrace t;
/// std::print("{}\n",t.evaluations); // 0
/// @endcode
/// @see MOEAD::trace()
struct MOEADTrace {
  MOEADOptions                          options;   ///< Actual copied configuration.
  std::vector<double>                   lower;     ///< Owned original finite lower bounds.
  std::vector<double>                   upper;     ///< Owned original finite upper bounds.
  ParetoPoint                           probe;     ///< Actual discarded objective-dimension probe.
  std::vector<std::vector<double>>      weights;   ///< Actual Das-Dennis lattice rows.
  std::vector<std::vector<std::size_t>> neighbors; ///< Actual distance-sorted neighbor slots.
  std::vector<ParetoPoint> initial_population;     ///< Actual initialized slots, in weight order.
  std::vector<double>      initial_ideal; ///< Componentwise minimum of initial population only.
  std::vector<MOEADStep>   steps;         ///< Actual sequential offspring/update history.
  std::vector<ParetoPoint>
      final_population; ///< Actual final slots, including duplicates/dominated points.
  std::vector<double>
      final_ideal; ///< Actual final componentwise minimum seen after initialization.
  std::vector<ParetoPoint>
              front; ///< Actual non-dominated subset of final population, duplicates retained.
  std::size_t evaluations{0};      ///< Actual callbacks,1+N+generations*N on success.
  std::size_t captured_entries{0}; ///< Actual admitted capture units.
};
/// @brief Approximate a multi-objective minimum using neighboring scalar subproblems.
/// Weights use the largest Das-Dennis lattice fitting max(requested,m); exactly
/// requested for two objectives. Neighbors use Euclidean weight distance; equal
/// distances retain the standard sort's unspecified tie arrangement. Tchebycheff
/// uses max(w,1e-6)*abs(f-ideal); weighted_sum uses sum(w*(f-ideal)). No global
/// archive or convergence guarantee. Final non-dominated extraction is quadratic
/// in population size, so the complete optimizer is not merely O(N*T).
/// Fixed seed repeats on the same standard library; callbacks and shared variation
/// operators remain observable. Const calls share no optimizer state, but the
/// callback is mutable and responsible for its own concurrency behavior.
/// @par Example
/// @code
/// dm::optim::MOEAD solver;
/// std::print("{}\n",solver.options().population_size); // 100
/// @endcode
/// @par References
/// - Q. Zhang and H. Li (2007), MOEA/D: A multiobjective evolutionary algorithm based on decomposition, IEEE Transactions on Evolutionary Computation11,712–731, doi:10.1109/TEVC.2007.892759.
/// @see MOEADOptions
class MOEAD {
 public:
  /// @brief Own a configuration; defer full validation until execution.
  /// @param options Configuration, default MOEADOptions{}.
  /// @par Example
  /// @code
  /// dm::optim::MOEAD s;std::print("{}\n",s.options().max_generations); // 250
  /// @endcode
  /// @see MOEAD
  explicit MOEAD(MOEADOptions options = {});
  /// @brief Run the canonical solver and return its actual final non-dominated subset.
  /// At least two finite objectives of fixed shape are required. First callback is
  /// a discarded dimension probe, then N initial evaluations and N per sweep.
  /// No partial front on failure; earlier callback effects remain. Bounds/options
  /// validate before callbacks, objective shape afterward. Setup uses O(N^2*m)
  /// distance work with O(N^2 log N) comparator overhead, stored neighborhoods
  /// O(N*T). Each sweep costs O(N*(n+T*m)) excluding callbacks and shared variation
  /// details. Final dominance extraction O(m*N^2). No universal timing guarantee.
  /// @param function Mutable objective, all components minimized, exceptions propagate.
  /// @param lower_bound Nonempty finite lower bounds.
  /// @param upper_bound Matching finite strictly higher bounds with finite widths.
  /// @return Owned actual final front, including identical non-dominated duplicates.
  /// @throws std::invalid_argument If bounds/options or fixed objective shape is invalid.
  /// @throws std::domain_error If an objective result is nonfinite.
  /// @throws std::overflow_error If variation/scalarization arithmetic is unrepresentable.
  /// @throws std::length_error If population/objective count exceeds1000000 or evaluation count overflows.
  /// @par Example
  /// @code
  /// dm::optim::ZDT1Objective f(2);dm::optim::MOEADOptions o;o.population_size=3;o.max_generations=0;
  /// auto front=dm::optim::MOEAD(o).optimize(f,{0,0},{1,1});
  /// std::print("{}\n",!front.empty()); // true
  /// @endcode
  /// @see trace()
  [[nodiscard]] std::vector<ParetoPoint> optimize(
      MultiObjectiveFunction&    function,
      const std::vector<double>& lower_bound,
      const std::vector<double>& upper_bound) const;
  /// @brief Capture actual setup, offspring and scalar decisions of the same producer.
  /// No replay or callback/RNG changes. Online quota admission returns no partial
  /// trace; earlier callbacks can already have occurred when a quota fails. Storage
  /// is proportional to admitted entries, in addition to ordinary driver storage.
  /// @param function Mutable fixed-shape finite objective, exceptions propagate.
  /// @param lower_bound Nonempty finite lower bounds, retained by value.
  /// @param upper_bound Matching finite upper bounds, retained by value.
  /// @param max_entries Actual capture-unit cap, default1000000; zero disables.
  /// @return Owned complete execution and final front.
  /// @throws std::invalid_argument If bounds/options/objective shape is invalid.
  /// @throws std::domain_error If objective values are nonfinite.
  /// @throws std::overflow_error If arithmetic is unrepresentable.
  /// @throws std::length_error If geometry/count or actual capture quota is exceeded.
  /// @throws std::bad_alloc If allocation fails.
  /// @par Example
  /// @code
  /// dm::optim::ZDT1Objective f(2);dm::optim::MOEADOptions o;o.population_size=3;o.max_generations=0;
  /// auto t=dm::optim::MOEAD(o).trace(f,{0,0},{1,1});
  /// std::print("{}\n",t.evaluations); // 4
  /// @endcode
  /// @see MOEADTrace, optimize()
  [[nodiscard]] MOEADTrace trace(
      MultiObjectiveFunction&    function,
      const std::vector<double>& lower_bound,
      const std::vector<double>& upper_bound,
      std::size_t                max_entries = 1000000) const;
  /// @brief Borrow the stored immutable options.
  /// @return Configuration reference valid for this solver's lifetime.
  [[nodiscard]] const MOEADOptions& options() const noexcept { return options_; }

 private:
  MOEADOptions options_; ///< Owned solver configuration.
  /// @brief Shared ordinary/capture producer, preserving callback and RNG scheduling.
  /// @param function Mutable objective.
  /// @param lower_bound Validated-in-body lower bounds.
  /// @param upper_bound Validated-in-body upper bounds.
  /// @param out Optional actual-history sink.
  /// @param cap Actual capture quota, zero disables.
  /// @return Actual final non-dominated front.
  std::vector<ParetoPoint> run(
      MultiObjectiveFunction&    function,
      const std::vector<double>& lower_bound,
      const std::vector<double>& upper_bound,
      MOEADTrace*                out,
      std::size_t                cap) const;
};
/// @brief Native original ZDT1 benchmark, minimizing two objectives on[0,1]^n.
/// f1=x0, g=1+9*mean(x1..), f2=g*(1-sqrt(f1/g)). Optimum g=1 gives
/// f2=1-sqrt(f1). No optimizer or sampling is hidden in the objective.
/// @par Example
/// @code
/// dm::optim::ZDT1Objective f(2);auto y=f.evaluate({.25,0});
/// std::print("{} {}\n",y[0],y[1]); // 0.25 0.5
/// @endcode
/// @par References
/// - E. Zitzler, K. Deb and L. Thiele (2000), Comparison of multiobjective evolutionary algorithms: Empirical results, Evolutionary Computation8,173–195, doi:10.1162/106365600568202.
/// @see MOEAD
class ZDT1Objective final : public MultiObjectiveFunction {
 public:
  /// @brief Store a validated benchmark dimension.
  /// @param variables At least two variables, default30.
  /// @throws std::invalid_argument If variables is below two.
  /// @par Example
  /// @code
  /// dm::optim::ZDT1Objective f(2);std::print("{}\n",f.evaluate({0,0})[1]); // 1
  /// @endcode
  /// @see ZDT1Objective
  explicit ZDT1Objective(std::size_t variables = 30);
  /// @brief Evaluate the original benchmark at one bounded decision vector.
  /// @param coordinates Exactly the configured number of finite values in[0,1].
  /// @return Owned two-objective vector in fixed order.
  /// @throws std::invalid_argument If shape, finiteness or bounds are invalid.
  /// @par Example
  /// @code
  /// dm::optim::ZDT1Objective f(2);std::print("{}\n",f.evaluate({1,0})[1]); // 0
  /// @endcode
  /// @see ZDT1Objective
  std::vector<double> evaluate(const std::vector<double>& coordinates) override;

 private:
  std::size_t variables_; ///< Validated fixed decision dimension.
};
/// @brief Observed coverage and vertical gaps of an approximate ZDT1 front.
/// Empty fronts are rejected by the producer. Gaps are objective2 minus
/// 1-sqrt(objective1), not hypervolume, distance to a set or a convergence proof.
/// @par Example
/// @code
/// auto s=dm::optim::zdt1_front_summary({{{.25,0},{.25,.5}}});
/// std::print("{}\n",s.mean_gap); // 0
/// @endcode
/// @see zdt1_front_summary()
struct ZDT1FrontSummary {
  std::size_t points{0};   ///< Number of supplied points, duplicates included.
  double      min_f1{0};   ///< Observed minimum first objective.
  double      max_f1{0};   ///< Observed maximum first objective.
  double      mean_gap{0}; ///< Mean signed vertical gap.
  double      max_gap{0};  ///< Maximum signed vertical gap.
};
/// @brief Summarize actual two-objective values against the analytic ZDT1 curve.
/// O(N) work, O(1) driver storage; coordinates are ignored and no refitting occurs.
/// @param front Nonempty points with two finite objectives and f1 in[0,1].
/// @return Actual observed coverage and signed vertical-gap summary.
/// @throws std::invalid_argument If objective shape/domain is invalid.
/// @par Example
/// @code
/// auto s=dm::optim::zdt1_front_summary({{{.25,0},{.25,.5}}});
/// std::print("{}\n",s.points); // 1
/// @endcode
/// @see ZDT1FrontSummary
ZDT1FrontSummary zdt1_front_summary(const std::vector<ParetoPoint>& front);
/// @brief Sample the analytic ZDT1 front with equally spaced f1 coordinates.
/// @param points At least two and at most1000000 curve samples, default101.
/// @return Owned ParetoPoints with empty decision coordinates and two objectives.
/// @throws std::invalid_argument If points is below two.
/// @throws std::length_error If points exceeds1000000.
/// @par Example
/// @code
/// std::print("{}\n",dm::optim::zdt1_front(3)[1].objectives[0]); // 0.5
/// @endcode
/// @see ZDT1Objective
std::vector<ParetoPoint> zdt1_front(std::size_t points = 101);
/// @brief Actual discrete scalarization choice for a supplied weight.
/// @par Example
/// @code
/// std::print("{}\n",dm::optim::moead_original_choices().size()); // 5
/// @endcode
/// @see moead_original_choices()
struct MOEADDiscreteChoice {
  std::vector<double> weight;     ///< Owned original two-component weight.
  std::vector<double> objectives; ///< Chosen original rounded objective pair.
  double              score{0};   ///< Actual Tchebycheff score, ideal(0,0), weight floor1e-6.
};
/// @brief Retain the original five-rounded-points/five-weights scalarization illustration.
/// Candidates(0,1),(.25,.5),(.5,.293),(.75,.134),(1,0), weights(1,0),(.7,.3),
/// (.5,.5),(.3,.7),(0,1). First strict minimum wins. This is a discrete toy
/// search, not a continuous front optimum or actual solver population.
/// @return Owned native discrete choices and scores, in original weight order.
/// @par Example
/// @code
/// auto r=dm::optim::moead_original_choices();std::print("{}\n",r.front().objectives[0]); // 0
/// @endcode
/// @see MOEADDiscreteChoice
std::vector<MOEADDiscreteChoice> moead_original_choices();
} // namespace datamunge::optim
