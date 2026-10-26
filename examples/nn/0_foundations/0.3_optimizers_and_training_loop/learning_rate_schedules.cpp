// CMD: just run-nn learning_rate_schedules
// Step, exponential, cosine, warmup, one-cycle and cyclical learning rates.
//
// Port of tflow's learning_rate_schedules. Schedules take the number of
// COMPLETED optimizer updates, beginning at 0. Check boundary values,
// clamping, continuity, triangular2 peak decay and invalid arguments, print a
// table, then feed every schedule to nn::Optimizer SGD on a scalar
// quadratic. The closed-form product of (1-lr_t) independently predicts the
// final parameter and catches off-by-one schedule application. Warmup starts
// at zero; a zero-rate call still advances the optimizer step. One-cycle here
// schedules learning rate only, with cosine rise and fall.
//
// Differences from TF: the schedules are nn::schedule functions (same
// formulas as tflow's nn::schedule); the rate is set on the optimizer with
// set_learning_rate() instead of being fed through a placeholder, and the
// completed-update index is Optimizer::iterations() instead of the step variable.

#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "common/checks.hpp"

namespace nn  = datamunge::nn;
namespace sch = datamunge::nn::schedule;
using nn::Tensor;

namespace {

using Schedule = std::pair<const char*, std::function<double(std::int64_t)>>;

void Boundaries() {
  check::Near(sch::step_decay(1, .5, 10, 9), 1, 0, "step before boundary");
  check::Near(sch::step_decay(1, .5, 10, 10), .5, 0, "step at boundary");
  check::Near(sch::exponential(1, .25, 10, 5), .5, 1e-12, "continuous exponential");
  check::Near(sch::cosine(1, .1, 100, 0), 1, 1e-12, "cosine start");
  check::Near(sch::cosine(1, .1, 100, 50), .55, 1e-12, "cosine midpoint");
  check::Near(sch::cosine(1, .1, 100, 200), .1, 1e-12, "cosine clamped tail");
  check::Near(sch::warmup(1, 10, 0), 0, 0, "warmup start");
  check::Near(sch::warmup(1, 0, 0), 1, 0, "disabled warmup");
  check::Near(sch::warmup_cosine(1, .1, 10, 100, 10), 1, 1e-12, "warmup transition");
  check::Near(sch::one_cycle(.1, 1, .01, 30, 100, 30), 1, 1e-12, "one-cycle peak");
  check::Near(sch::one_cycle(.1, 1, .01, 30, 100, 200), .01, 1e-12, "one-cycle tail");
  check::Near(sch::cyclical(.1, 1, 10, 10), 1, 1e-12, "cyclical peak");
  check::Near(sch::cyclical(.1, 1, 10, 20), .1, 1e-12, "cyclical restart");
  check::Near(sch::cyclical(.1, 1, 10, 30, .5), .55, 1e-12, "triangular2 next peak");
  int rejected = 0;
  for (const std::function<double()>& invalid : std::vector<std::function<double()>>{
           [] { return sch::step_decay(1, .5, 0, 1); }, [] { return sch::cosine(1, 0, 0, 1); },
           [] { return sch::warmup_cosine(1, 0, 10, 10, 1); }, [] { return sch::one_cycle(1, .1, 0, 5, 10, 0); },
           [] { return sch::cyclical(.1, 1, 0, 0); }, [] { return sch::warmup(1, 10, -1); }}) {
    try {
      invalid();
    } catch (const std::invalid_argument&) {
      ++rejected;
    }
  }
  check::Require(rejected == 6, "all invalid schedules must be rejected");
  std::cout << "  endpoint, transition, clamping, cycle and invalid-input checks passed\n";
}

void Train(const Schedule& schedule, int steps) {
  Tensor        w = nn::scalar(2).requires_grad_();
  nn::Optimizer opt({w}, 0.0);
  double        expected = 2;
  for (int i = 0; i < steps; ++i) {
    const std::int64_t step = opt.iterations();
    check::Require(step == i, "schedule uses optimizer update index");
    const float rate = static_cast<float>(schedule.second(step));
    expected *= 1 - static_cast<double>(rate);
    opt.set_learning_rate(rate);
    opt.minimize(.5 * nn::square(w));
  }
  check::Near(w.item(), expected, 2e-6, "closed-form scheduled SGD trajectory");
  check::Require(std::abs(expected) <= 2, "scheduled fit cannot increase parameter error");
  if (steps >= 100) check::Require(std::abs(expected) < .5, "scheduled fit must reduce parameter error");
  std::printf("  %-15s w=%11.5g, closed form=%11.5g\n", schedule.first, w.item<float>(), expected);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int steps = 100;
    for (int i = 1; i < argc; ++i) {
      const std::string argument(argv[i]);
      if (argument.rfind("epochs=", 0) == 0) steps = std::stoi(argument.substr(7));
      if (argument.rfind("steps=", 0) == 0) steps = std::stoi(argument.substr(6));
    }
    check::Require(steps > 0, "positive training budget required");
    const std::vector<Schedule> schedules{
        {"step", [](std::int64_t t) { return sch::step_decay(.1, .5, 25, t); }},
        {"exponential", [](std::int64_t t) { return sch::exponential(.1, .5, 25, t); }},
        {"cosine", [](std::int64_t t) { return sch::cosine(.1, .001, 100, t); }},
        {"warmup+cosine", [](std::int64_t t) { return sch::warmup_cosine(.1, .001, 10, 100, t); }},
        {"one-cycle", [](std::int64_t t) { return sch::one_cycle(.01, .1, .0001, 30, 100, t); }},
        {"cyclical", [](std::int64_t t) { return sch::cyclical(.01, .1, 20, t); }},
        {"triangular2", [](std::int64_t t) { return sch::cyclical(.01, .1, 20, t, .5); }}};
    std::cout << "== Schedule boundaries\n";
    Boundaries();
    std::cout << "\n== Learning rates by completed update\n  schedule       ";
    for (int step : {0, 10, 20, 30, 50, 80, 100, 120}) std::printf(" %8d", step);
    std::cout << '\n';
    for (const auto& schedule : schedules) {
      std::printf("  %-15s", schedule.first);
      for (int step : {0, 10, 20, 30, 50, 80, 100, 120}) std::printf(" %8.4f", schedule.second(step));
      std::cout << '\n';
    }
    std::cout << "\n== nn::Optimizer SGD vs closed-form trajectory\n";
    for (const auto& schedule : schedules) Train(schedule, steps);
    std::cout << "All learning-rate schedule checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
