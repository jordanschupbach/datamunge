// CMD: just run-nn gradient_clipping_accumulation
// Gradient clipping and accumulation across unequal microbatches.
//
// Port of tflow's gradient_clipping_accumulation. Global-norm clipping
// scales the complete gradient vector together, keeping its direction. Value
// clipping clamps components separately and can change that direction. Check
// both on a 3-4-5 example and on zero gradients.
//
// Accumulate mean-loss gradients from microbatches of 2, 3 and 2 samples,
// weighted by their sample counts (nn::GradientAccumulator::add(grads,
// weight)). Compare clipped Adam updates, all moments and the integer step
// with a true batch of 7 over ten updates. Clipping is done AFTER averaging;
// applying it to individual microbatches is different. Empty accumulation
// and nonpositive weights must not mutate training state.
//
// Differences from TF: there is no graph, so the accumulate / apply-and-reset
// ops become plain calls, failed TF Asserts become exceptions, and the clip
// scale (which nn::clip_by_global_norm does not return) is recomputed from the
// returned norm as max_norm / max(norm, max_norm).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "common/checks.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

void Clipping() {
  const Tensor g              = nn::tensor({3, 4});
  const auto [clipped, norm]  = nn::clip_by_global_norm({g}, 2);
  const double scale          = 2 / std::max(norm, 2.0);
  const auto   values         = nn::clip_by_value({g}, -2, 2);
  // The complete norm includes all tensors, not each tensor separately.
  const auto split = nn::clip_by_global_norm({nn::scalar(3), nn::scalar(4)}, 2).first;
  const auto cv    = clipped[0].to_vector<double>();
  check::Near(cv[0], 1.2, 1e-6, "global clipping x");
  check::Near(cv[1], 1.6, 1e-6, "global clipping y");
  check::Near(norm, 5, 0, "global norm");
  check::Near(scale, .4, 1e-6, "global scale");
  for (double v : values[0].to_vector<double>()) check::Near(v, 2, 0, "component clipping");
  check::Near(split[0].item(), 1.2, 1e-6, "global norm spans tensors");
  check::Near(split[1].item(), 1.6, 1e-6, "global norm spans tensors");
  const auto [zero, zero_norm] = nn::clip_by_global_norm({nn::tensor({0, 0})}, 2);
  check::Near(zero[0].to_vector<double>()[0], 0, 0, "zero gradient stays zero");
  check::Near(2 / std::max(zero_norm, 2.0), 1, 0, "zero gradient does not divide by zero");
  std::cout << "  (3,4), limit 2: global norm -> (1.2,1.6); by value -> (2,2)\n"
               "  split tensors share one norm; zero gradients remain finite\n";
}

/// Optimizer slots (and the step) by name, for comparing two optimizers.
std::vector<std::pair<std::string, Tensor>> Slots(const nn::Optimizer& opt) {
  std::vector<std::pair<std::string, Tensor>> out;
  for (const auto& entry : opt.state())
    if (entry.first != "step" && entry.first != "nadam_product") out.push_back(entry);
  return out;
}

void Accumulation(int steps) {
  Tensor w = nn::scalar(3).requires_grad_(), b = nn::scalar(-1).requires_grad_();
  Tensor w_ref = nn::scalar(3).requires_grad_(), b_ref = nn::scalar(-1).requires_grad_();
  auto   loss  = [](const Tensor& weight, const Tensor& bias, const Tensor& x, const Tensor& y) {
    return nn::mean(nn::square(x * weight + bias - y));
  };
  nn::OptimizerConfig cfg;
  cfg.kind = nn::OptimizerKind::Adam;
  nn::Optimizer           micro({w, b}, .01, cfg), batch({w_ref, b_ref}, .01, cfg);
  nn::GradientAccumulator accumulator;
  // Apply the accumulated mean (clipped) and reset; throws when empty.
  auto apply = [&] {
    const auto mean = accumulator.mean();
    micro.step(nn::clip_by_global_norm(mean, 1).first);
    accumulator.reset();
  };

  check::Throws(apply, "an empty accumulator must reject an optimizer step");
  check::Require(micro.iterations() == 0, "empty update cannot advance Adam");
  check::Near(w.item(), 3, 0, "empty update cannot change weights");

  const std::vector<float> xs{-3, -2, -1, 0, 1, 2, 3}, ys{-5, -3, -1, 1, 3, 5, 7};
  const Tensor             x_all = nn::from_vector(xs), y_all = nn::from_vector(ys);
  check::Throws([&] { accumulator.add(nn::grad(loss(w, b, x_all, y_all), {w, b}), 0); },
                "zero microbatch weight must fail");
  check::Near(accumulator.total_weight(), 0, 0, "invalid weight cannot mutate accumulation");

  double worst = 0;
  for (int update = 0; update < steps; ++update) {
    int start = 0;
    for (int size : {2, 3, 2}) {
      const Tensor bx = nn::from_vector(std::vector<float>(xs.begin() + start, xs.begin() + start + size));
      const Tensor by = nn::from_vector(std::vector<float>(ys.begin() + start, ys.begin() + start + size));
      accumulator.add(nn::grad(loss(w, b, bx, by), {w, b}), size);
      start += size;
    }
    if (update == 0) {
      const auto mean = accumulator.mean();
      check::Near(mean[0].item(), 8, 1e-6, "sample-weighted slope gradient");
      check::Near(mean[1].item(), -4, 1e-6, "sample-weighted bias gradient");
    }
    apply();
    batch.step(nn::clip_by_global_norm(nn::grad(loss(w_ref, b_ref, x_all, y_all), {w_ref, b_ref}), 1).first);

    std::vector<std::pair<double, double>> pairs{{w.item(), w_ref.item()}, {b.item(), b_ref.item()}};
    const auto a = Slots(micro), r = Slots(batch);
    check::Require(a.size() == r.size() && a.size() == 4, "same Adam slots in both optimizers");
    for (std::size_t i = 0; i < a.size(); ++i) {
      check::Require(a[i].first == r[i].first, "slot names line up");
      pairs.emplace_back(a[i].second.item(), r[i].second.item());
    }
    for (const auto& [got, want] : pairs) {
      worst = std::max(worst, std::abs(got - want));
      check::Near(got, want, 2e-6, "microbatch vs full-batch optimizer state");
    }
    check::Require(micro.iterations() == update + 1, "one optimizer step per accumulated batch");
    check::Near(accumulator.total_weight(), 0, 0, "accumulator reset after apply");
  }
  std::printf("  microbatches 2+3+2 vs batch 7: max weight/moment error %.3g over %d Adam updates\n", worst, steps);
  std::cout << "  empty updates and zero weights rejected without changing optimizer state\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int steps = 10;
    for (int i = 1; i < argc; ++i) {
      const std::string argument(argv[i]);
      if (argument.rfind("epochs=", 0) == 0) steps = std::stoi(argument.substr(7));
      if (argument.rfind("steps=", 0) == 0) steps = std::stoi(argument.substr(6));
    }
    check::Require(steps > 0, "positive training budget required");
    std::cout << "== Gradient clipping\n";
    Clipping();
    std::cout << "\n== Sample-weighted accumulation, then global-norm clipping\n";
    Accumulation(steps);
    std::cout << "All clipping and accumulation checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
