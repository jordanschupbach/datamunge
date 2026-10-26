// CMD: just run-nn world_models
// World Models: VAE + MDN-RNN + controller.
//
// Port of tflow's world_models.
// Ha & Schmidhuber 2018 split the agent into three parts trained in turn:
//   V: a variational autoencoder compresses each frame to a latent z,
//   M: a mixture-density LSTM predicts the next z given (z, a), so it
//      learns the dynamics in latent space and can "dream" sequences,
//   C: a tiny linear controller on [z, h] is trained with an evolution
//      strategy, since it has only a few hundred parameters.
// The frames come from a pixel rendering of CartPole (12 x 24 binary
// image). C is evolved against the real environment here; the paper also
// evolves it inside M's dreams, which the final section illustrates.
//
// The TF graphs become eager nn functions on two ParameterStores ("vae" seed
// 0, "rnn" seed 1) with tflow's layer names; the M optimiser clips the global
// norm at 1 as tflow's nn::Adam. Data collection, batches, ES noise and every
// random draw use the same std::mt19937_64 and shared normal_distribution as
// TF; weights use nn's generator, so results match TF statistically. The vsim
// video is replaced by playing the same episode and printing its return.
// Options: vae_steps=2000 rnn_steps=1500 generations=60 threads=N seed=0 (the
// random stream and weight seeds; TF fixes 0). The port
// adds checks (exit 1 on failure).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <datamunge/nn/nn.hpp>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/classic_control.hpp"
#include "common/rl/dqn_common.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

constexpr int kH = 12, kW = 24, kPixels = kH * kW;

// CartPole observed as a small binary image: the cart is a block on the
// bottom row, the pole a line of pixels above it.
class PixelCartPole : public rl::Env {
 public:
  std::string name() const override { return "PixelCartPole"; }
  int obs_size() const override { return kPixels; }
  rl::ActionSpace action_space() const override { return inner_.action_space(); }
  void Seed(uint64_t seed) override { inner_.Seed(seed); }
  int max_steps() const override { return inner_.max_steps(); }
  rl::StepResult step(const rl::Action& a) override {
    rl::StepResult r = inner_.step(a);
    r.obs = Render(r.obs);
    return r;
  }
  const rl::CartPole& inner() const { return inner_; }
  std::vector<std::string> action_names() const override { return inner_.action_names(); }
  static rl::Obs Render(const rl::Obs& state) {
    rl::Obs img(kPixels, 0.f);
    const float x = state[0], theta = state[2];
    const int cx = std::max(1, std::min(kW - 2, static_cast<int>(std::round((x + 2.4f) / 4.8f * (kW - 1)))));
    for (int c = cx - 1; c <= cx + 1; ++c) img[(kH - 1) * kW + c] = 1.f;
    for (int k = 1; k <= kH - 2; ++k) {
      const int r = kH - 1 - k, c = static_cast<int>(std::round(cx + std::sin(theta) * k * 2.f));
      if (c >= 0 && c < kW) img[r * kW + c] = 1.f;
    }
    return img;
  }
  static std::string Ascii(const std::vector<float>& img) {
    std::string s;
    for (int r = 0; r < kH; ++r) {
      s += "      ";
      for (int c = 0; c < kW; ++c) s += img[r * kW + c] > 0.5f ? '#' : '.';
      s += '\n';
    }
    return s;
  }

 protected:
  rl::Obs Reset() override { return Render(inner_.reset()); }

 private:
  rl::CartPole inner_;
};

}  // namespace

int main(int argc, char** argv) {
  try {
  const int Z = 8, Hs = 64, K = 3, T = 16, A = 2;
  int vae_steps = 2000, rnn_steps = 1500, generations = 60, seed = 0;
  const int population = 24;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const size_t eq = arg.find('=');
    if (eq == std::string::npos) continue;
    const std::string key = arg.substr(0, eq);
    const int v = std::atoi(arg.c_str() + eq + 1);
    if (key == "threads") nn::set_num_threads(v);
    else if (key == "vae_steps") vae_steps = v;
    else if (key == "rnn_steps") rnn_steps = v;
    else if (key == "generations") generations = v;
    else if (key == "seed") seed = v;
    else if (!rl::IsHardwareOption(key)) std::fprintf(stderr, "unknown option %s\n", arg.c_str());
  }
  std::printf("== World Models on pixel CartPole (%dx%d): VAE (z=%d) -> MDN-RNN (%d units, %d mixtures) -> linear controller by ES\n", kH, kW, Z, Hs, K);

  // ---- Data: random-policy episodes.
  PixelCartPole env;
  std::mt19937_64 rng(seed);
  struct Episode {
    std::vector<rl::Obs> frames;
    std::vector<int> actions;
  };
  std::vector<Episode> episodes;
  int total_frames = 0;
  for (int e = 0; e < 300; ++e) {
    Episode ep;
    rl::Obs o = env.reset(e);
    while (true) {
      const int a = static_cast<int>(rng() % 2);
      ep.frames.push_back(o), ep.actions.push_back(a);
      rl::StepResult r = env.step(a);
      o = r.obs;
      if (r.done()) break;
    }
    total_frames += static_cast<int>(ep.frames.size());
    episodes.push_back(std::move(ep));
  }
  std::printf("  collected %d frames from 300 random episodes\n", total_frames);

  // ---- V: VAE with Bernoulli pixels.
  nn::ParameterStore vae_p(seed);
  struct VaeOut {
    Tensor z_mean, z_logvar;
  };
  auto encoder = [&](const Tensor& frame) {
    const Tensor enc = nn::dense(vae_p, "enc", frame, 128, nn::Activation::Relu);
    return VaeOut{nn::dense(vae_p, "z_mean", enc, Z), nn::dense(vae_p, "z_logvar", enc, Z)};
  };
  auto decode = [&](const Tensor& latent) {
    return nn::dense(vae_p, "dec_out", nn::dense(vae_p, "dec", latent, 128, nn::Activation::Relu), kPixels);
  };
  auto vae_loss_fn = [&](const Tensor& frame, const Tensor& z_eps) {
    const VaeOut q = encoder(frame);
    const Tensor z = q.z_mean + nn::exp(0.5 * q.z_logvar) * z_eps;
    const Tensor logits = decode(z);
    // Bernoulli NLL from logits (TF's stable form) and the Gaussian KL.
    const Tensor bce = nn::sum((nn::relu(logits) + nn::log(1.0 + nn::exp(-nn::abs(logits)))) - logits * frame, {1});
    const Tensor kl = -0.5 * nn::sum((1.0 + q.z_logvar) - (nn::square(q.z_mean) + nn::exp(q.z_logvar)), {1});
    return nn::mean(bce + kl, {0});
  };

  // ---- M: MDN-RNN over (z_t, a_t) predicting z_{t+1}.
  nn::ParameterStore rnn_p(seed + 1);
  auto mdn_head = [&](const Tensor& h) { return nn::dense(rnn_p, "mdn", h, K * (2 * Z + 1)); };  // K means, K log-stds, K logits
  auto mdn_nll = [&](const Tensor& o, const Tensor& target) {  // target [B, Z]
    const Tensor means = nn::reshape(nn::slice_axis(o, 1, 0, K * Z), {-1, K, Z});
    const Tensor log_std = nn::clip(nn::reshape(nn::slice_axis(o, 1, K * Z, 2 * K * Z), {-1, K, Z}), -4.0, 2.0);
    const Tensor log_pi = nn::log_softmax(nn::slice_axis(o, 1, 2 * K * Z, 2 * K * Z + K), -1);
    const Tensor diff = (nn::expand_dims(target, 1) - means) / nn::exp(log_std);
    const Tensor comp_logp = nn::sum(-0.5 * nn::square(diff) - (log_std + 0.5 * std::log(2 * 3.14159265f)), {2});  // [B, K]
    return -nn::logsumexp(log_pi + comp_logp, {1});
  };
  auto rnn_step = [&](const Tensor& z, const Tensor& a, const nn::LstmState& st) {
    return nn::lstm_step(rnn_p, "lstm", nn::concat({z, a}, 1), st);
  };
  {  // build every weight once so the optimisers see them
    nn::NoGradGuard no_grad;
    vae_loss_fn(nn::zeros({1, kPixels}), nn::zeros({1, Z}));
    mdn_head(rnn_step(nn::zeros({1, Z}), nn::zeros({1, A}), nn::lstm_zero_state(1, Hs)).h);
  }
  rl::Adam vae_adam(vae_p, 1e-3), rnn_adam(rnn_p, 1e-3, 1.0);
  std::normal_distribution<float> normal(0.f, 1.f);

  // ---- Train V.
  std::printf("\n-- V: training the VAE\n");
  std::vector<const rl::Obs*> all_frames;
  for (auto& ep : episodes)
    for (auto& f : ep.frames) all_frames.push_back(&f);
  float vae_first = -1, vae_last = 0;
  for (int step = 1; step <= vae_steps; ++step) {
    const int B = 64;
    std::vector<float> F(static_cast<size_t>(B) * kPixels), E(static_cast<size_t>(B) * Z);
    for (int i = 0; i < B; ++i) {
      const rl::Obs& f = *all_frames[rng() % all_frames.size()];
      std::copy(f.begin(), f.end(), F.begin() + static_cast<size_t>(i) * kPixels);
      for (int j = 0; j < Z; ++j) E[i * Z + j] = normal(rng);
    }
    const Tensor loss = vae_loss_fn(nn::from_vector(F, {B, kPixels}), nn::from_vector(E, {B, Z}));
    vae_last = loss.item<float>();
    if (vae_first < 0) vae_first = vae_last;
    vae_adam.Minimize(loss);
    if (step % 500 == 0) std::printf("  step %5d  loss %.2f nats/frame\n", step, vae_last), std::fflush(stdout);
  }
  // Encode every frame (posterior mean) for M.
  auto encode = [&](const rl::Obs& f) {
    nn::NoGradGuard no_grad;
    return encoder(rl::Row(f)).z_mean.to_vector<float>();
  };
  std::vector<std::vector<std::vector<float>>> latents(episodes.size());
  for (size_t e = 0; e < episodes.size(); ++e)
    for (auto& f : episodes[e].frames) latents[e].push_back(encode(f));
  int recon_errors = 0;
  {
    nn::NoGradGuard no_grad;
    const rl::Obs& f = episodes[0].frames[5];
    // recon with z_eps = 0: the decoded posterior mean
    const std::vector<float> rec = nn::sigmoid(decode(encoder(rl::Row(f)).z_mean)).to_vector<float>();
    for (int i = 0; i < kPixels; ++i) recon_errors += (rec[i] > 0.5f) != (f[i] > 0.5f);
    std::printf("  a frame and its reconstruction:\n%s      ---\n%s", PixelCartPole::Ascii(f).c_str(), PixelCartPole::Ascii(rec).c_str());
  }

  // ---- Train M on windows of T transitions (episodes shorter than T+1 are skipped).
  std::printf("\n-- M: training the MDN-RNN\n");
  std::vector<int> long_eps;
  for (size_t e = 0; e < episodes.size(); ++e)
    if (static_cast<int>(episodes[e].frames.size()) > T) long_eps.push_back(static_cast<int>(e));
  float rnn_first = 1e30f, rnn_last = 0;
  for (int step = 1; step <= rnn_steps; ++step) {
    const int B = 32;
    std::vector<float> ZS(static_cast<size_t>(B) * (T + 1) * Z), AS(static_cast<size_t>(B) * T * A, 0.f);
    for (int b = 0; b < B; ++b) {
      const int e = long_eps[rng() % long_eps.size()];
      const int start = static_cast<int>(rng() % (episodes[e].frames.size() - T));
      for (int t = 0; t <= T; ++t)
        for (int j = 0; j < Z; ++j) ZS[(b * (T + 1) + t) * Z + j] = latents[e][start + t][j];
      for (int t = 0; t < T; ++t) AS[(b * T + t) * A + episodes[e].actions[start + t]] = 1.f;
    }
    const Tensor zs = nn::from_vector(ZS, {B, T + 1, Z}), as = nn::from_vector(AS, {B, T, A});
    nn::LstmState st = nn::lstm_zero_state(B, Hs);
    Tensor nll_sum;
    for (int t = 0; t < T; ++t) {
      st = rnn_step(nn::squeeze(nn::slice_axis(zs, 1, t, t + 1), {1}), nn::squeeze(nn::slice_axis(as, 1, t, t + 1), {1}), st);
      const Tensor nll = nn::mean(mdn_nll(mdn_head(st.h), nn::squeeze(nn::slice_axis(zs, 1, t + 1, t + 2), {1})), {0});
      nll_sum = t ? nll_sum + nll : nll;
    }
    const Tensor rnn_loss = nll_sum / static_cast<double>(T);
    rnn_last = rnn_loss.item<float>();
    if (step == 1) rnn_first = rnn_last;
    rnn_adam.Minimize(rnn_loss);
    if (step % 500 == 0) std::printf("  step %5d  NLL %.3f per step\n", step, rnn_last), std::fflush(stdout);
  }

  // ---- C: linear controller on [z, h] evolved with a simple ES against the real environment.
  std::printf("\n-- C: evolving a linear controller (%d parameters) with ES, population %d\n", (Z + Hs + 1) * A, population);
  const int P = (Z + Hs + 1) * A;
  // `on_step(logits, r)` is called after every step with the controller's logit per action and the step's result.
  auto rollout = [&](const std::vector<float>& w, uint64_t seed, int max_steps, const std::function<void(const std::vector<float>&, const rl::StepResult&)>& on_step = {}) {
    nn::NoGradGuard no_grad;
    rl::Obs o = env.reset(seed);
    nn::LstmState st = nn::lstm_zero_state(1, Hs);
    std::vector<float> h(Hs, 0.f);
    float ret = 0;
    for (int t = 0; t < max_steps; ++t) {
      std::vector<float> zt = encode(o);
      int best = 0;
      float best_logit = -1e9f;
      std::vector<float> logits(A);
      for (int a = 0; a < A; ++a) {
        float logit = w[a * (Z + Hs + 1) + Z + Hs];
        for (int j = 0; j < Z; ++j) logit += w[a * (Z + Hs + 1) + j] * zt[j];
        for (int j = 0; j < Hs; ++j) logit += w[a * (Z + Hs + 1) + Z + j] * h[j];
        logits[a] = logit;
        if (logit > best_logit) best_logit = logit, best = a;
      }
      rl::StepResult r = env.step(best);
      ret += r.reward;
      if (on_step) on_step(logits, r);
      std::vector<float> at(A, 0.f);
      at[best] = 1.f;
      st = rnn_step(nn::from_vector(zt, {1, Z}), nn::from_vector(at, {1, A}), st);
      h = st.h.to_vector<float>();
      o = r.obs;
      if (r.done()) break;
    }
    return ret;
  };
  std::vector<float> w(P, 0.f);
  const float sigma = 0.1f, lr = 0.15f;
  float last_current = 0, best_fitness = 0;
  for (int gen = 1; gen <= generations; ++gen) {
    std::vector<std::vector<float>> noise(population, std::vector<float>(P));
    std::vector<float> fitness(population);
    for (int i = 0; i < population; ++i) {
      std::vector<float> cand = w;
      for (int j = 0; j < P; ++j) cand[j] += sigma * (noise[i][j] = normal(rng));
      fitness[i] = rollout(cand, 1000 * gen + i, 200);
    }
    best_fitness = std::max(best_fitness, *std::max_element(fitness.begin(), fitness.end()));
    // Rank-normalised ES update.
    std::vector<int> order(population);
    for (int i = 0; i < population; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int x, int y) { return fitness[x] < fitness[y]; });
    std::vector<float> util(population);
    for (int r = 0; r < population; ++r) util[order[r]] = static_cast<float>(r) / (population - 1) - 0.5f;
    for (int j = 0; j < P; ++j) {
      float g = 0;
      for (int i = 0; i < population; ++i) g += util[i] * noise[i][j];
      w[j] += lr / (population * sigma) * g;
    }
    if (gen % 5 == 0 || gen == 1) {
      last_current = rollout(w, 99, 500);
      std::printf("  generation %3d  population best %.0f  mean %.1f  current controller %.0f\n", gen, *std::max_element(fitness.begin(), fitness.end()),
                  std::accumulate(fitness.begin(), fitness.end(), 0.f) / population, last_current);
      std::fflush(stdout);
    }
  }
  const float final_eval = [&] {
    float s = 0;
    for (int i = 0; i < 10; ++i) s += rollout(w, 5000 + i, 500) / 10;
    return s;
  }();
  std::printf("  final controller over 10 episodes (up to 500 steps): %.1f\n", final_eval);
  // On film in TF (CartPole's 3D world, the controller's logit per action as the overlay).
  const float filmed = rollout(w, 5000, 500);
  std::printf("  played world_models_cartpole (video omitted): return %.0f\n", filmed);

  // ---- Dreaming: roll M forward on its own predictions and decode frames.
  std::printf("\n-- Dreaming: 3 imagined frames after pushing right from a real start\n");
  {
    nn::NoGradGuard no_grad;
    rl::Obs o = env.reset(3);
    std::vector<float> zt = encode(o);
    nn::LstmState st = nn::lstm_zero_state(1, Hs);
    for (int t = 0; t < 3; ++t) {
      st = rnn_step(nn::from_vector(zt, {1, Z}), nn::from_vector(std::vector<float>{0.f, 1.f}, {1, A}), st);
      const std::vector<float> m = mdn_head(st.h).to_vector<float>();
      // Pick the most likely mixture component and take its mean.
      int best = 0;
      for (int k = 1; k < K; ++k) if (m[2 * K * Z + k] > m[2 * K * Z + best]) best = k;
      for (int j = 0; j < Z; ++j) zt[j] = m[best * Z + j];
      const std::vector<float> frame = nn::sigmoid(decode(nn::from_vector(zt, {1, Z}))).to_vector<float>();
      std::printf("  t+%d:\n%s", t + 1, PixelCartPole::Ascii(frame).c_str());
    }
  }

  // Checks added by the port (tflow prints the tables only).
  check::Require(vae_last < 0.5f * vae_first, "the VAE loss falls by more than half");
  check::Require(recon_errors <= 6, "the VAE reconstructs the sample frame (at most 6 wrong pixels)");
  check::Require(rnn_last < rnn_first, "the MDN-RNN's NLL falls");
  // ES on 24 rollouts is noisy: the final controller scored 71/91/92/386/31/117 for seeds 0-5 of the
  // port and 192 in TF's (seed 0) run, so the checks ask for learning, not for a balanced pole.
  if (generations >= 60) {
    check::Require(best_fitness >= 150, "some evolved candidate balances the pole for 150+ steps from pixels");
    check::Require(final_eval >= 25, "the evolved controller beats a random policy (about 22 steps)");
  }
  std::printf("\nchecks passed (vsim video omitted)\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "CHECK FAILED: %s\n", e.what());
    return 1;
  }
  return 0;
}
