#include "datamunge/nn/gradcheck.hpp"

#include <cmath>
#include <sstream>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/random.hpp"

namespace datamunge::nn {

namespace {

/// Fresh leaves with the same values and requires_grad flags.
std::vector<Tensor> fresh_leaves(const std::vector<Tensor>& inputs) {
  std::vector<Tensor> out;
  for (const auto& t : inputs) {
    if (!t.defined()) {
      out.push_back(t);
      continue;
    }
    Tensor leaf = t.detach().clone();
    if (t.requires_grad()) leaf.requires_grad_(true);
    out.push_back(leaf);
  }
  return out;
}

std::vector<double> evaluate_flat(const TensorFunction& f, const std::vector<Tensor>& inputs) {
  return cast(f(inputs).detach(), DType::Float64).to_vector<double>();
}

}  // namespace

Tensor numerical_jacobian(const TensorFunction& f, const std::vector<Tensor>& inputs, std::size_t input_index,
                          double eps) {
  EnableGradGuard enable;  // f may itself call grad() (gradgradcheck)
  std::vector<Tensor> base   = fresh_leaves(inputs);
  const Tensor        x      = base.at(input_index);
  const auto          values = x.to_vector<double>();
  const std::int64_t  cols   = static_cast<std::int64_t>(values.size());
  const std::int64_t  rows   = static_cast<std::int64_t>(evaluate_flat(f, base).size());
  Tensor              jac    = zeros({rows, cols}, DType::Float64);
  double*             J      = jac.data_ptr<double>();
  for (std::int64_t j = 0; j < cols; ++j) {
    auto perturbed = [&](double delta) {
      std::vector<double> v = values;
      v[static_cast<std::size_t>(j)] += delta;
      std::vector<Tensor> ins = base;
      Tensor              p   = cast(from_vector(v, x.shape().empty() ? Shape{1} : x.shape()).reshape(x.shape()), x.dtype());
      if (x.requires_grad()) p.requires_grad_(true);
      ins[input_index] = p;
      return evaluate_flat(f, ins);
    };
    const auto plus  = perturbed(eps);
    const auto minus = perturbed(-eps);
    for (std::int64_t r = 0; r < rows; ++r)
      J[r * cols + j] = (plus[static_cast<std::size_t>(r)] - minus[static_cast<std::size_t>(r)]) / (2 * eps);
  }
  return jac;
}

Tensor analytical_jacobian(const TensorFunction& f, const std::vector<Tensor>& inputs, std::size_t input_index) {
  EnableGradGuard     enable;  // works even when called under NoGradGuard
  std::vector<Tensor> leaves = fresh_leaves(inputs);
  Tensor              x      = leaves.at(input_index);
  if (!x.requires_grad()) x.requires_grad_(true);
  leaves[input_index] = x;
  const Tensor       out  = f(leaves);
  const std::int64_t rows = out.numel(), cols = x.numel();
  Tensor             jac  = zeros({rows, cols}, DType::Float64);
  if (!out.requires_grad()) return jac;
  double* J = jac.data_ptr<double>();
  for (std::int64_t r = 0; r < rows; ++r) {
    Tensor onehot = one_hot(scalar(static_cast<double>(r), DType::Int64), rows, 1.0, 0.0, out.dtype()).reshape(out.shape());
    GradOptions opts;
    opts.retain_graph = true;
    opts.allow_unused = true;
    const auto g      = grad({out}, {x}, {onehot}, opts);
    if (!g[0].defined()) continue;
    const auto gv = cast(g[0].detach(), DType::Float64).to_vector<double>();
    for (std::int64_t c = 0; c < cols; ++c) J[r * cols + c] = gv[static_cast<std::size_t>(c)];
  }
  return jac;
}

GradcheckResult gradcheck(const TensorFunction& f, const std::vector<Tensor>& inputs, const GradcheckOptions& options) {
  GradcheckResult result;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    if (!inputs[i].defined() || !inputs[i].requires_grad()) continue;
    const auto numeric  = numerical_jacobian(f, inputs, i, options.eps).to_vector<double>();
    const auto analytic = analytical_jacobian(f, inputs, i).to_vector<double>();
    for (std::size_t k = 0; k < numeric.size(); ++k) {
      const double err = std::abs(analytic[k] - numeric[k]);
      const bool   bad = !(err <= options.atol + options.rtol * std::abs(numeric[k]));
      if (err > result.max_abs_error || (bad && result.ok)) result.max_abs_error = std::max(result.max_abs_error, err);
      if (bad && result.ok) {
        result.ok = false;
        std::ostringstream os;
        os << "input " << i << ", jacobian entry " << k << ": analytic " << analytic[k] << " vs numeric "
           << numeric[k];
        result.message = os.str();
      }
    }
  }
  return result;
}

GradcheckResult gradgradcheck(const TensorFunction& f, const std::vector<Tensor>& inputs,
                              const GradcheckOptions& options) {
  // Fixed cotangent so the first-order map is deterministic.
  const Tensor probe = [&] {
    NoGradGuard guard;
    return f(fresh_leaves(inputs)).detach();
  }();
  Generator    gen(1234);
  const Tensor v = cast(normal(probe.shape(), 0.0, 1.0, gen, DType::Float64), probe.dtype());

  std::vector<std::size_t> checked;
  for (std::size_t i = 0; i < inputs.size(); ++i)
    if (inputs[i].defined() && inputs[i].requires_grad()) checked.push_back(i);

  TensorFunction first_order = [f, v, checked](const std::vector<Tensor>& ins) {
    const Tensor        out = f(ins);
    std::vector<Tensor> wrt;
    for (auto i : checked) wrt.push_back(ins[i]);
    std::vector<Tensor> pieces;
    if (out.requires_grad()) {
      GradOptions opts;
      opts.create_graph = true;
      opts.allow_unused = true;
      const auto g      = grad({out}, wrt, {v}, opts);
      for (std::size_t k = 0; k < g.size(); ++k)
        pieces.push_back(reshape(g[k].defined() ? g[k] : zeros_like(wrt[k]), {-1}));
    } else {
      for (const auto& w : wrt) pieces.push_back(zeros({w.numel()}, w.dtype()));
    }
    return concat(pieces, 0);
  };
  return gradcheck(first_order, inputs, options);
}

}  // namespace datamunge::nn
