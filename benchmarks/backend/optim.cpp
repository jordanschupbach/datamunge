#include <datamunge/optim/adam.hpp>
#include <datamunge/optim/differential_evolution.hpp>
#include <datamunge/optim/gradient_descent.hpp>
#include <datamunge/optim/lbfgs.hpp>
#include <datamunge/optim/nelder_mead.hpp>
#include <datamunge/optim/newton.hpp>
#include <datamunge/optim/nsga2.hpp>
#include <datamunge/optim/pso.hpp>
#include <datamunge/optim/sqp.hpp>

#include <cstdint>
#include <cstring>

#include "common.hpp"
namespace {
// Untimed deterministic signatures keep seeded solver behavior reviewable.
std::uint64_t result_signature(const std::vector<double>& values,
                               std::uint64_t hash = 1469598103934665603ULL) {
  for (double value : values) {
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    hash = (hash ^ bits) * 1099511628211ULL;
  }
  return hash;
}
void publish_signature(benchmark::State& s, const char* prefix, std::uint64_t hash) {
  s.counters[std::string(prefix) + "_lo"] = double(static_cast<std::uint32_t>(hash));
  s.counters[std::string(prefix) + "_hi"] = double(static_cast<std::uint32_t>(hash >> 32));
}

namespace opt = datamunge::optim;
struct Objective : opt::HessianFunction {
  int                 kind;
  std::int64_t        values = 0, grads = 0, hessians = 0;
  std::vector<double> weights;
  explicit Objective(int k, std::size_t n = 16) : kind(k), weights(n, 1.0) {
    if (kind == 1)
      for (std::size_t i = 0; i < n; ++i)
        weights[i] = std::pow(100.0, double(i) / double(n - 1));
  }
  double weight(std::size_t i, std::size_t) const { return weights[i]; }
  double evaluate(const std::vector<double>& x) override {
    ++values;
    double v = 0;
    if (kind == 2) {
      for (std::size_t i = 0; i + 1 < x.size(); ++i)
        v += 100 * std::pow(x[i + 1] - x[i] * x[i], 2) + std::pow(1 - x[i], 2);
    } else
      for (std::size_t i = 0; i < x.size(); ++i)
        v += kind == 3 ? x[i] * x[i] + 10 * (1 - std::cos(6.283185307179586 * x[i]))
                       : 0.5 * weight(i, x.size()) * x[i] * x[i];
    return v;
  }
  std::vector<double> gradient(const std::vector<double>& x) override {
    ++grads;
    std::vector<double> g(x.size());
    if (kind == 2) {
      for (std::size_t i = 0; i + 1 < x.size(); ++i) {
        g[i] += -400 * x[i] * (x[i + 1] - x[i] * x[i]) + 2 * (x[i] - 1);
        g[i + 1] += 200 * (x[i + 1] - x[i] * x[i]);
      }
    } else
      for (std::size_t i = 0; i < x.size(); ++i)
        g[i] = kind == 3 ? 2 * x[i] + 62.83185307179586 * std::sin(6.283185307179586 * x[i])
                         : weight(i, x.size()) * x[i];
    return g;
  }
  std::vector<std::vector<double>> hessian(const std::vector<double>& x) override {
    ++hessians;
    std::vector<std::vector<double>> h(x.size(), std::vector<double>(x.size()));
    if (kind == 2) {
      for (std::size_t i = 0; i + 1 < x.size(); ++i) {
        h[i][i] += 1200 * x[i] * x[i] - 400 * x[i + 1] + 2;
        h[i][i + 1] = h[i + 1][i] = -400 * x[i];
        h[i + 1][i + 1] += 200;
      }
    } else
      for (std::size_t i = 0; i < x.size(); ++i)
        h[i][i] = kind == 3 ? 2 + 394.7841760435743 * std::cos(6.283185307179586 * x[i])
                            : weight(i, x.size());
    return h;
  }
};
// Bounded solves include algorithm allocations and objective evaluations. Reset is untimed.
void BM_GeneralOptimizer(benchmark::State& s, int algorithm) {
  const auto          n = s.range(0);
  Objective           f(static_cast<int>(s.range(1)), static_cast<std::size_t>(n));
  std::vector<double> initial(n, f.kind == 2 ? -1.2 : 2.0), lower(n, -5), upper(n, 5), x;
  auto                solve = [&]() {
    switch (algorithm) {
      case 0: {
        opt::GradientDescentOptions o;
        o.step_size      = f.kind == 2 ? 0.001 : 0.01;
        o.max_iterations = 100;
        return opt::GradientDescent(o).optimize(f, x);
      }
      case 1: {
        opt::AdamOptions o;
        o.step_size      = 0.05;
        o.max_iterations = 100;
        return opt::Adam(o).optimize(f, x);
      }
      case 2: {
        opt::LBFGSOptions o;
        o.max_iterations = 100;
        return opt::LBFGS(o).optimize(f, x);
      }
      case 3: {
        opt::NewtonOptions o;
        o.max_iterations = 100;
        return opt::Newton(o).optimize(f, x);
      }
      case 4: {
        opt::NelderMeadOptions o;
        o.max_iterations = 100;
        return opt::NelderMead(o).optimize(f, x);
      }
      case 5: {
        opt::DEOptions o;
        o.population_size = 24;
        o.max_generations = 30;
        o.seed            = s.range(2);
        return opt::DifferentialEvolution(o).optimize(f, x, lower, upper);
      }
      default: {
        opt::PSOOptions o;
        o.population_size = 24;
        o.max_iterations  = 30;
        o.seed            = s.range(2);
        if (algorithm == 7) o.topology = "ring";
        return opt::PSO(o).optimize(f, x, lower, upper);
      }
    }
  };
  x             = initial;
  double before = f.evaluate(x), value = solve();
  nb::check(
      std::isfinite(value) && value <= before + 1e-6, "optimizer failed to improve objective");
  nb::check(std::abs(f.evaluate(x) - value) < 1e-6, "optimizer value/point mismatch");
  f.values = f.grads = f.hessians = 0;
  for (auto _ : s) {
    s.PauseTiming();
    x = initial;
    s.ResumeTiming();
    value = solve();
    benchmark::DoNotOptimize(value);
    benchmark::DoNotOptimize(x);
  }
  auto iterations                     = double(s.iterations());
  s.counters["objective_calls/solve"] = f.values / iterations;
  s.counters["gradient_calls/solve"]  = f.grads / iterations;
  s.counters["hessian_calls/solve"]   = f.hessians / iterations;
  s.counters["final_objective"]       = value;
  publish_signature(s, "coordinate_signature", result_signature(x));
  for (std::size_t j = 0; j < x.size(); ++j)
    s.counters["coordinate_" + std::to_string(j)] = x[j];
  s.counters["target_met"]            = value < 1e-6 ? 1 : 0;
  s.SetLabel("bounded solve; target=1e-6; budget=100 iterations (population:24x30)");
}
void optim_args(benchmark::internal::Benchmark* b) {
  b->ArgsProduct({{2, 16}, {0, 1, 2}, {42}})
      ->ArgNames({"dimension", "problem", "seed"})
      ->UseRealTime();
}
#define O(name, id) BENCHMARK_CAPTURE(BM_GeneralOptimizer, name, id)->Apply(optim_args)
O(GradientDescent, 0);
O(Adam, 1);
O(LBFGS, 2);
O(Newton, 3);
O(NelderMead, 4);
BENCHMARK_CAPTURE(BM_GeneralOptimizer, DifferentialEvolution, 5)
    ->ArgsProduct({{2, 16}, {0, 1, 2, 3}, {42, 43, 44}})
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_GeneralOptimizer, PSO, 6)
    ->ArgsProduct({{2, 16}, {0, 1, 2, 3}, {42, 43, 44}})
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_GeneralOptimizer, PSORing, 7)
    ->ArgsProduct({{2, 16}, {0, 1, 2, 3}, {42, 43, 44}})
    ->UseRealTime();
struct Equality : opt::EqualityConstrainedFunction {
  double evaluate(const std::vector<double>& x) override { return x[0] * x[0] + x[1] * x[1]; }
  std::vector<double> gradient(const std::vector<double>& x) override {
    return {2 * x[0], 2 * x[1]};
  }
  std::vector<double> constraints(const std::vector<double>& x) override {
    return {x[0] + x[1] - 1};
  }
  std::vector<std::vector<double>> constraint_jacobian(const std::vector<double>&) override {
    return {{1, 1}};
  }
};
void BM_ConstrainedSQP(benchmark::State& s) {
  Equality            f;
  opt::SQP            solver;
  std::vector<double> x{2, 2};
  solver.optimize(f, x);
  nb::check(std::abs(x[0] - 0.5) < 1e-5 && std::abs(x[1] - 0.5) < 1e-5, "SQP reference failed");
  for (auto _ : s) {
    s.PauseTiming();
    x = {2, 2};
    s.ResumeTiming();
    benchmark::DoNotOptimize(solver.optimize(f, x));
  }
}
BENCHMARK(BM_ConstrainedSQP)->UseRealTime();
struct Tradeoff : opt::MultiObjectiveFunction {
  std::int64_t        calls = 0;
  std::vector<double> evaluate(const std::vector<double>& x) override {
    ++calls;
    return {x[0] * x[0], (x[0] - 2) * (x[0] - 2)};
  }
};
void BM_MultiObjectiveNSGA2(benchmark::State& s) {
  Tradeoff          f;
  opt::NSGA2Options o;
  o.population_size = 32;
  o.max_generations = 20;
  o.seed            = s.range(0);
  opt::NSGA2 solver(o);
  auto       check = solver.optimize(f, {0}, {2});
  nb::check(!check.empty(), "empty Pareto front");
  for (const auto& p : check)
    nb::check(
        p.coordinates[0] >= 0 && p.coordinates[0] <= 2 &&
            std::abs(p.objectives[0] - p.coordinates[0] * p.coordinates[0]) < 1e-10,
        "invalid Pareto point");
  std::uint64_t signature = 1469598103934665603ULL;
  for (const auto& point : check) {
    signature = result_signature(point.coordinates, signature);
    signature = result_signature(point.objectives, signature);
  }
  publish_signature(s, "pareto_signature", signature);
  f.calls = 0;
  for (auto _ : s)
    benchmark::DoNotOptimize(solver.optimize(f, {0}, {2}));
  s.counters["objective_calls/solve"] = double(f.calls) / s.iterations();
}
BENCHMARK(BM_MultiObjectiveNSGA2)->Arg(42)->Arg(43)->Arg(44)->UseRealTime();

// An explicit target adapter counts its additional objective checks. The timer
// includes target detection and exception unwinding, as well as solver work.
struct TargetReached {};
struct TargetObjective : Objective {
  double achieved = 0;
  explicit TargetObjective(int kind) : Objective(kind) {}
  double evaluate(const std::vector<double>& x) override {
    achieved = Objective::evaluate(x);
    if (achieved <= 1e-6)
      throw TargetReached{};
    return achieved;
  }
  std::vector<double> gradient(const std::vector<double>& x) override {
    evaluate(x);
    return Objective::gradient(x);
  }
};
void BM_OptimizerTimeToTarget(benchmark::State& s, int algorithm) {
  TargetObjective     f(static_cast<int>(s.range(0)));
  std::vector<double> x;
  auto                run = [&]() {
    try {
      if (algorithm == 0) {
        opt::GradientDescentOptions o;
        o.step_size      = 0.01;
        o.tolerance      = 1e-14;
        o.max_iterations = 10000;
        opt::GradientDescent(o).optimize(f, x);
      } else if (algorithm == 1) {
        opt::AdamOptions o;
        o.step_size      = 0.05;
        o.tolerance      = 1e-14;
        o.max_iterations = 10000;
        opt::Adam(o).optimize(f, x);
      } else if (algorithm == 2) {
        opt::LBFGSOptions o;
        o.tolerance      = 1e-14;
        o.max_iterations = 10000;
        opt::LBFGS(o).optimize(f, x);
      } else {
        opt::NewtonOptions o;
        o.tolerance      = 1e-14;
        o.max_iterations = 10000;
        opt::Newton(o).optimize(f, x);
      }
    } catch (const TargetReached&) {
      return true;
    }
    return false;
  };
  x.assign(16, 2);
  nb::check(run(), "target not reached within iteration budget");
  f.values = f.grads = f.hessians = 0;
  for (auto _ : s) {
    s.PauseTiming();
    x.assign(16, 2);
    s.ResumeTiming();
    nb::check(run(), "target not reached");
  }
  s.counters["objective_calls/solve"] = double(f.values) / s.iterations();
  s.counters["gradient_calls/solve"]  = double(f.grads) / s.iterations();
  s.counters["hessian_calls/solve"]   = double(f.hessians) / s.iterations();
  s.counters["final_objective"]       = f.achieved;
  s.SetLabel("quadratic; dimension=16; target=1e-6; cap=10000; includes target-check adapter");
}
BENCHMARK_CAPTURE(BM_OptimizerTimeToTarget, GradientDescent, 0)->Arg(0)->Arg(1)->UseRealTime();
BENCHMARK_CAPTURE(BM_OptimizerTimeToTarget, Adam, 1)->Arg(0)->Arg(1)->UseRealTime();
BENCHMARK_CAPTURE(BM_OptimizerTimeToTarget, LBFGS, 2)->Arg(0)->Arg(1)->UseRealTime();
BENCHMARK_CAPTURE(BM_OptimizerTimeToTarget, Newton, 3)->Arg(0)->Arg(1)->UseRealTime();
} // namespace
