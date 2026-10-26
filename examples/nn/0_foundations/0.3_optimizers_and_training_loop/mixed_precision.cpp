// CMD: just run-nn mixed_precision
// Mixed float16/bfloat16 compute with float32 master weights and loss scaling.
//
// Port of tflow's mixed_precision. Show float16 gradient underflow and
// recovery by scaling the loss, train a dense regression model in all three
// precisions, and force an overflowing half gradient to verify that weights,
// Adam moments and the step are unchanged. Finite gradients are unscaled into
// independent float32 buffers before Adam. A host-driven loss scaler halves on
// overflow and doubles after good steps.
// Reference: https://www.tensorflow.org/guide/mixed_precision.
//
// Differences from TF: nn has no float16/bfloat16 dtype and no GPU (both out
// of scope in the plan). Low precision is EMULATED: common/mixed_precision.hpp
// rounds every tensor TF would hold in half precision (casts, matmul and bias
// outputs, and their gradients) to the nearest float16 / bfloat16 value, so
// underflow, overflow and rounding behave like TF's, but the arithmetic runs
// in float32 on the CPU and says nothing about accelerator throughput. The
// device options (device=, require_gpu=, bf16_gpu=) are accepted for
// compatibility: require_gpu=1 / bf16_gpu=1 fail as no GPU exists; threads=N
// sets the nn thread count. The dense weights are initialised from nn's
// generator (seed 11), so the final MSEs match TF only statistically.

#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/mixed_precision.hpp"

namespace nn = datamunge::nn;
using mixed::Precision;
using nn::Tensor;

namespace {

void ScalingChecks() {
  Tensor     w       = nn::scalar(1).requires_grad_();
  auto       grad_at = [&](float scale) {
    const Tensor half = mixed::Quantize(w, Precision::Float16);
    const Tensor loss = nn::mul(half, 1e-8);
    return nn::grad(nn::mul(loss, scale), {w});
  };
  std::vector<Tensor> unscaled;
  check::Near(grad_at(1)[0].item(), 0, 0, "float16 tiny gradient underflows");
  check::Require(mixed::UnscaleGradients(grad_at(65536), 65536, &unscaled), "scaled gradient must be finite");
  check::Near(unscaled[0].item(), 1e-8, 1e-11, "loss scaling recovers small derivative");
  std::printf("  float16 derivative: scale 1 -> 0; scale 65536 -> %.9g after unscaling\n", unscaled[0].item<float>());
  mixed::DynamicLossScale scaler(128, 2, 1, 256);
  scaler.Applied();
  check::Near(scaler.value(), 128, 0, "growth interval");
  scaler.Applied();
  check::Near(scaler.value(), 256, 0, "scale growth");
  scaler.Overflow();
  check::Near(scaler.value(), 128, 0, "scale backoff");
  std::cout << "  dynamic scale growth and overflow backoff passed\n";
}

void Fit(Precision precision) {
  std::vector<float> xv(32), yv(16);
  for (int i = 0; i < 16; ++i) {
    const float a = std::sin(i * 0.7f), b = std::cos(i * 0.4f);
    xv[2 * i]     = a;
    xv[2 * i + 1] = b;
    yv[i]         = .73f * a - 1.27f * b + .13f;
  }
  const Tensor       inputs = nn::from_vector(xv, {16, 2}), targets = nn::from_vector(yv, {16, 1});
  nn::ParameterStore p(11);
  auto               loss_of = [&] {
    const Tensor compute = mixed::MixedDense(p, "dense", inputs, 2, 1, precision);
    return nn::mean(nn::square(nn::sub(compute, targets)));
  };
  (void)loss_of();  // creates the float32 master parameters
  nn::OptimizerConfig cfg;
  cfg.kind = nn::OptimizerKind::Adam;
  nn::Optimizer optimizer(p, 0.03, cfg);
  auto          gradients = [&](float scale) { return nn::grad(nn::mul(loss_of(), scale), p.parameters()); };
  auto          state     = [&] {
    std::vector<std::vector<double>> out;  // exact values (float32 and int64 widen to double losslessly)
    for (const auto& t : p.parameters()) out.push_back(t.to_vector<double>());
    for (const auto& [name, t] : optimizer.state()) out.push_back(t.to_vector<double>());
    return out;
  };
  mixed::DynamicLossScale scaler(precision == Precision::Float16 ? 128 : 1, 100, 1,
                                 precision == Precision::Float16 ? 65536 : 1);
  int                     applied = 0, skipped = 0;
  if (precision == Precision::Float16) {
    // Run one good update first so Adam's moments and step exist (nn creates
    // slots lazily; TF's are initialised zeros) and the comparison is nontrivial.
    std::vector<Tensor> unscaled;
    check::Require(mixed::UnscaleGradients(gradients(scaler.value()), scaler.value(), &unscaled), "warm-up gradients");
    optimizer.step(unscaled);
    scaler.Applied();
    ++applied;
    const auto before = state();
    check::Require(!mixed::UnscaleGradients(gradients(1e12f), 1e12f, &unscaled), "half overflow must be detected");
    // Deliberately do not step the optimizer when gradients are nonfinite.
    check::Require(state() == before, "overflow must not change any training state");
    std::cout << "  float16 forced overflow skipped; weights, moments and step unchanged\n";
  }
  const int budget = 400;  // including the float16 warm-up update
  for (int attempt = 0; applied < budget && attempt < 450; ++attempt) {
    std::vector<Tensor> unscaled;
    if (!mixed::UnscaleGradients(gradients(scaler.value()), scaler.value(), &unscaled)) {
      scaler.Overflow();
      ++skipped;
      continue;
    }
    optimizer.step(unscaled);
    scaler.Applied();
    ++applied;
  }
  check::Require(applied == budget, "mixed training must finish its update budget");
  const double final_loss = loss_of().item();
  check::Require(final_loss < 3e-4, "mixed regression must converge");
  check::Require(optimizer.iterations() == 400 && p.at("dense/W").dtype() == nn::DType::Float32,
                 "master weights/counter precision");
  std::printf("  %-8s %-15s final MSE %.6g, master dtype float32, scale %.0f, skipped %d\n",
              mixed::PrecisionName(precision).c_str(), precision == Precision::Float32 ? "on CPU" : "emulated on CPU", final_loss, scaler.value(), skipped);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    bool require_gpu = false, bf16_gpu = false;
    int  threads     = 2;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto        pos = arg.find('=');
      if (pos == std::string::npos) throw std::invalid_argument("expected key=value");
      const auto key = arg.substr(0, pos), value = arg.substr(pos + 1);
      if (key == "device" && (value == "cpu" || value == "gpu")) {
        check::Require(value == "cpu", "device=gpu: nn has no GPU backend");
      } else if (key == "threads") {
        threads = std::stoi(value);
      } else if ((key == "require_gpu" || key == "bf16_gpu") && (value == "0" || value == "1")) {
        (key == "require_gpu" ? require_gpu : bf16_gpu) = value == "1";
      } else {
        throw std::invalid_argument("unknown mixed precision option: " + arg);
      }
    }
    nn::set_num_threads(threads);
    check::Require(!(require_gpu || bf16_gpu), "GPU validation requested but nn has no GPU backend");
    std::cout << "== Mixed precision (no GPU in nn; float16/bfloat16 emulated by rounding on CPU)\n";
    ScalingChecks();
    Fit(Precision::Float32);
    Fit(Precision::Float16);
    Fit(Precision::BFloat16);
    std::cout << "All mixed-precision numerical checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
