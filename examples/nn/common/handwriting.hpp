// Graves-style monotonic text window and bivariate-mixture stroke likelihood
// (port of tflow's common/nn/handwriting.h). Each step feeds the previous pen
// offset (dx, dy, pen-up) and the previous window over the 3-character prompt
// (one-hot over 21 symbols) to an 8-unit LSTM "lstm"; a dense "window" 8 -> 6
// gives two Gaussian window components (alpha, beta, monotonic kappa
// increments of .05 exp(.)); a dense "mdn" over [h, window] gives 19 mixture
// parameters: 3 component logits, 3+3 means, 3+3 log-scales, 3 correlations
// and one pen logit. 1680 parameters in 7 tensors.
#pragma once

#include <vector>

#include <datamunge/nn/nn.hpp>

namespace handwriting {

namespace nn = datamunge::nn;
using nn::Tensor;

constexpr int kSteps = 33, kSymbols = 21, kRaw = 19;

struct Step {
  nn::LstmState state;
  Tensor        kappa, window, phi, raw;
};

/// stroke [B, 3], text [B, 3, 21], kappa [B, 2], window [B, 21].
inline Step HandwritingCell(nn::ParameterStore& p, const Tensor& stroke, const Tensor& text, const nn::LstmState& previous,
                            const Tensor& kappa, const Tensor& window) {
  Step out;
  out.state             = nn::lstm_step(p, "lstm", nn::concat({stroke, window}, 1), previous);
  const Tensor params   = nn::dense(p, "window", out.state.h, 6);
  const Tensor alpha    = nn::exp(nn::slice_axis(params, 1, 0, 2)), beta = nn::exp(nn::slice_axis(params, 1, 2, 4));
  out.kappa             = nn::add(kappa, nn::mul(nn::exp(nn::slice_axis(params, 1, 4, 6)), .05));
  const Tensor position = nn::reshape(nn::arange(0, 3, 1, kappa.dtype()), {1, 1, 3});
  const Tensor delta    = nn::sub(nn::expand_dims(out.kappa, 2), position);
  out.phi = nn::sum(nn::mul(nn::expand_dims(alpha, 2), nn::exp(nn::neg(nn::mul(nn::expand_dims(beta, 2), nn::square(delta))))), {1});
  out.window = nn::sum(nn::mul(text, nn::expand_dims(out.phi, 2)), {1});
  out.raw    = nn::dense(p, "mdn", nn::concat({out.state.h, out.window}, 1), kRaw);
  return out;
}

struct Path {
  Tensor raw, h, c, kappa, window, phi;
};

/// Teacher-forced unroll over stroke [B, 33, 3] from zero states.
inline Path Handwriting(nn::ParameterStore& p, const Tensor& stroke, const Tensor& text) {
  const auto    B = stroke.shape()[0];
  nn::LstmState s = nn::lstm_zero_state(B, 8, stroke.dtype());
  Tensor        kappa = nn::zeros({B, 2}, stroke.dtype()), window = nn::zeros({B, kSymbols}, stroke.dtype());
  std::vector<Tensor> hs{s.h}, cs{s.c}, ks{kappa}, ws{window}, raw, phi;
  for (const auto& input : nn::time_steps(stroke)) {
    const Step next = HandwritingCell(p, input, text, s, kappa, window);
    s               = next.state;
    kappa           = next.kappa;
    window          = next.window;
    hs.push_back(s.h);
    cs.push_back(s.c);
    ks.push_back(kappa);
    ws.push_back(window);
    raw.push_back(next.raw);
    phi.push_back(next.phi);
  }
  return {nn::stack_time(raw), nn::stack_time(hs), nn::stack_time(cs), nn::stack_time(ks), nn::stack_time(ws), nn::stack_time(phi)};
}

/// Mean per-step negative log-likelihood of target strokes (dx, dy, pen) under
/// the 3-component bivariate Gaussian mixture plus a Bernoulli pen.
inline Tensor HandwritingNll(const Tensor& raw, const Tensor& target) {
  const Tensor r = nn::reshape(raw, {-1, kRaw}), y = nn::reshape(target, {-1, 3});
  auto         part = [&](std::int64_t begin, std::int64_t n = 3) { return nn::slice_axis(r, 1, begin, begin + n); };
  const Tensor lp   = nn::log_softmax(part(0), -1);
  const Tensor sx = nn::add(nn::exp(part(9)), .001), sy = nn::add(nn::exp(part(12)), .001), rho = nn::mul(nn::tanh(part(15)), .95);
  const Tensor dx = nn::div(nn::sub(nn::slice_axis(y, 1, 0, 1), part(3)), sx);
  const Tensor dy = nn::div(nn::sub(nn::slice_axis(y, 1, 1, 2), part(6)), sy);
  const Tensor one = nn::sub(1.0, nn::square(rho));
  const Tensor z   = nn::sub(nn::add(nn::square(dx), nn::square(dy)), nn::mul(nn::mul(rho, nn::mul(dx, dy)), 2.0));
  const Tensor normalizer = nn::add(nn::add(nn::log(sx), nn::log(sy)), nn::add(nn::mul(nn::log(one), .5), 1.8378770664));
  const Tensor scores     = nn::sub(nn::sub(lp, normalizer), nn::div(z, nn::mul(one, 2.0)));
  const Tensor maximum    = nn::stop_gradient(nn::max(scores, {1}, true));
  const Tensor logdensity = nn::add(maximum, nn::log(nn::sum(nn::exp(nn::sub(scores, maximum)), {1}, true)));
  const Tensor pen        = part(18, 1);
  const Tensor bce        = nn::sub(nn::softplus(pen), nn::mul(nn::slice_axis(y, 1, 2, 3), pen));
  return nn::mean(nn::sub(bce, logdensity));
}

}  // namespace handwriting
