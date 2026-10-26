// CMD: just run-nn dqn_atari
// DQN on Atari through ALE.
//
// Port of tflow's dqn_atari. The Double DQN from the earlier examples with the
// original DQN Nature network (three convolutions on the 84x84x4 frame stack)
// and reward clipping, on an ALE game: Tetris by default (the only ROM
// bundled; put other ROMs in data/roms/ and pass game=<name>). Learning Atari
// takes millions of frames; the default budget here is far smaller and exists
// to exercise the full pipeline end to end on a CPU. Scale up with e.g.
// steps=500000. Note the replay buffer stores whole float frame stacks, as in
// TF: buffer=30000 needs about 7 GB.
//
// Deviations: ALE is not a datamunge dependency (the plan keeps emulators out
// of scope), so the Atari part only runs when the example is compiled with
// ALE on the include path (common/rl/ale_env.hpp, DATAMUNGE_EXAMPLES_HAVE_ALE)
// and linked with libale + zlib; `just run-nn` builds without it. Without ALE
// the program checks the same network and Double DQN update on a synthetic
// batch of frame stacks (rl::AtariFrameStack applied to a 160x210 test screen
// with a moving sprite): the parameter count, the output shape, and that a few
// hundred updates fit the synthetic targets. Nothing in that mode is Atari
// output. In both modes the video of the greedy agent (raw screen, HUD,
// reward bar) is omitted; with ALE the greedy episode is still played and its
// score printed. The networks are eager nn ops over two ParameterStores; the
// weights use nn's initialiser, so scores match TF statistically.
// Options: game=<name> and the rl::DqnConfig keys.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/ale_env.hpp"
#include "common/rl/dqn_common.hpp"
#include "common/rl/replay.hpp"
#include "common/rl/utils.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

constexpr int S = rl::AtariFrameStack::kSize, F = rl::AtariFrameStack::kStack;

// Nature DQN: [B, 4*84*84] -> NHWC [B, 84, 84, 4] -> conv 8x8/4 (32) -> 4x4/2 (64) -> 3x3/1 (64) -> fc 512 -> A
Tensor QNet(nn::ParameterStore& p, const Tensor& x, int A) {
  const Tensor img = nn::transpose(nn::reshape(x, {-1, F, S, S}), {0, 2, 3, 1});
  auto conv = [](std::int64_t stride) {
    nn::ConvLayerOptions o;
    o.stride = stride;
    o.padding = nn::Padding::Valid;
    o.activation = nn::Activation::Relu;
    return o;
  };
  const Tensor c1 = nn::conv2d(p, "conv1", img, 32, 8, conv(4));
  const Tensor c2 = nn::conv2d(p, "conv2", c1, 64, 4, conv(2));
  const Tensor c3 = nn::conv2d(p, "conv3", c2, 64, 3, conv(1));
  const Tensor flat = nn::reshape(c3, {-1, 7 * 7 * 64});
  const Tensor h = nn::dense(p, "fc", flat, 512, nn::Activation::Relu);
  return nn::dense(p, "q", h, A);
}

std::int64_t ExpectedParams(int A) { return 8224 + 32832 + 36928 + (3136 * 512 + 512) + (512 * A + A); }

// The Double DQN update (y from the online argmax and the target's value).
struct DoubleDqn {
  nn::ParameterStore online, target;
  int A;
  rl::DqnConfig cfg;
  std::unique_ptr<rl::Adam> adam;
  DoubleDqn(const rl::DqnConfig& c, int n_actions) : online(c.seed), target(c.seed + 1), A(n_actions), cfg(c) {
    nn::NoGradGuard no_grad;
    QNet(online, nn::zeros({1, F * S * S}), A);
    QNet(target, nn::zeros({1, F * S * S}), A);
    adam = std::make_unique<rl::Adam>(online, cfg.lr, cfg.clip_norm);
    Sync();
  }
  void Sync() { target.copy_from(online, cfg.tau); }
  float Train(const rl::Batch& b) {
    Tensor y;
    {
      nn::NoGradGuard no_grad;
      const Tensor a_star = nn::argmax(QNet(online, b.s2, A), 1);
      const Tensor next_value = nn::sum(QNet(target, b.s2, A) * nn::one_hot(a_star, A), {1});
      y = b.r + (cfg.gamma * (1.0 - b.done)) * next_value;
    }
    const Tensor q_a = nn::sum(QNet(online, b.s, A) * nn::one_hot(b.a, A), {1});
    const Tensor loss = nn::mean(nn::huber(y - q_a), {0});
    adam->Minimize(loss);
    return loss.item<float>();
  }
  std::vector<float> QValues(const rl::Obs& o) {
    nn::NoGradGuard no_grad;
    return QNet(online, rl::Row(o), A).to_vector<float>();
  }
};

#if !DATAMUNGE_EXAMPLES_HAVE_ALE
// A 160x210 grayscale test screen: dark background, an 8x8 sprite at column x.
std::vector<unsigned char> TestScreen(int x) {
  std::vector<unsigned char> s(160 * 210, 20);
  for (int y = 100; y < 108; ++y)
    for (int i = x; i < x + 8 && i < 160; ++i) s[y * 160 + i] = 230;
  return s;
}

void SyntheticPipeline(rl::DqnConfig cfg) {
  std::printf("== ALE is not built in: Atari training is skipped (see the header).\n"
              "   Checking the Nature network and the Double DQN update on synthetic frame stacks instead.\n");
  const int A = 5, B = 16;
  cfg.lr = 1e-4f;
  DoubleDqn agent(cfg, A);
  std::printf("  network: %lld parameters\n", static_cast<long long>(agent.online.count()));
  check::Require(agent.online.count() == ExpectedParams(A), "Nature DQN parameter count");
  // B transitions: a sprite moving right (s) then one step further (s2), action a = i % A,
  // reward +1 when the action matches the sprite's position bucket, all terminal so y = r.
  rl::ReplayBuffer buffer(B, F * S * S);
  for (int i = 0; i < B; ++i) {
    rl::AtariFrameStack stack(160, 210);
    const int x0 = 8 * i;
    for (int f = 0; f < F; ++f) stack.PushFrame(TestScreen(x0 + 2 * f));
    const rl::Obs s = stack.StackedObs();
    stack.PushFrame(TestScreen(x0 + 2 * F));
    const int a = i % A;
    buffer.Add({s, a, a == (i / 4) % A ? 1.f : 0.f, stack.StackedObs(), 1.f, {}});
  }
  std::mt19937_64 rng(cfg.seed);
  float first = 0, last = 0;
  for (int it = 0; it < 300; ++it) {
    rl::Batch b = buffer.Sample(B, rng);
    const float l = agent.Train(b);
    check::Require(std::isfinite(l), "finite TD loss");
    if (it == 0) first = l;
    last = l;
    agent.Sync();
  }
  const std::vector<float> q = agent.QValues(buffer.Sample(1, rng).s.to_vector<float>());
  std::printf("  Huber TD loss on the synthetic batch: %.4f -> %.4f after 300 updates; Q(s, .) has %zu entries\n", first,
              last, q.size());
  check::Require(static_cast<int>(q.size()) == A, "Q(s, .) has one value per action");
  check::Require(last < 0.25f * first, "the update fits the synthetic targets");
}
#endif

}  // namespace

int main(int argc, char** argv) {
  try {
    rl::DqnConfig cfg;
    cfg.steps = 30000;
    cfg.warmup = 2000;
    cfg.batch = 32;
    cfg.buffer = 30000;
    cfg.train_every = 4;
    cfg.lr = 2.5e-4f;
    cfg.tau = 1.f;
    cfg.target_sync = 1000;
    cfg.eps_decay_steps = 20000;
    cfg.eps_end = 0.05f;
    cfg.log_every = 5000;
    cfg.eval_every = 15000;
    cfg.eval_episodes = 1;
    std::string game = "tetris";
    for (int i = 1; i < argc; ++i)
      if (std::string(argv[i]).rfind("game=", 0) == 0) game = std::string(argv[i]).substr(5);
    rl::ParseConfig(argc, argv, &cfg);

#if DATAMUNGE_EXAMPLES_HAVE_ALE
    rl::AtariEnv::Options opt;
    opt.clip_rewards = true;
    rl::AtariEnv env(game, opt), eval_env(game);
    const int A = env.action_space().n;
    std::printf("== Double DQN on %s: %d agent steps (%d frames), %d actions, train every %d steps, target sync %d\n",
                env.name().c_str(), cfg.steps, 4 * cfg.steps, A, cfg.train_every, cfg.target_sync);
    DoubleDqn agent(cfg, A);
    std::printf("  network: %lld parameters\n", static_cast<long long>(agent.online.count()));
    check::Require(agent.online.count() == ExpectedParams(A), "Nature DQN parameter count");

    rl::ReplayBuffer buffer(cfg.buffer, env.obs_size());
    rl::Monitor monitor(cfg);
    std::mt19937_64 rng(cfg.seed);
    auto greedy = [&](const rl::Obs& o) { return rl::ArgMax(agent.QValues(o)); };
    rl::Obs obs = env.reset(cfg.seed);
    float ret = 0;
    for (int step = 1; step <= cfg.steps; ++step) {
      const float eps = rl::Epsilon(cfg, step);
      const int act = std::uniform_real_distribution<float>(0, 1)(rng) < eps ? static_cast<int>(rng() % A) : greedy(obs);
      rl::StepResult sr = env.step(act);
      buffer.Add({obs, act, sr.reward, sr.obs, sr.terminated ? 1.f : 0.f, {}});
      ret += env.last_raw_reward();
      obs = sr.obs;
      if (sr.done()) monitor.EndEpisode(ret), ret = 0, obs = env.reset();
      if (step >= cfg.warmup && step % cfg.train_every == 0) {
        const float loss = agent.Train(buffer.Sample(cfg.batch, rng));
        check::Require(std::isfinite(loss), "finite TD loss");
        monitor.AddLoss(loss);
      }
      if (cfg.tau < 1.f ? step >= cfg.warmup : step % cfg.target_sync == 0) agent.Sync();
      if (step % cfg.log_every == 0)
        monitor.Log(step, eps, step % cfg.eval_every == 0 ? rl::Evaluate(eval_env, greedy, cfg.eval_episodes, 1000 + step) : NAN);
    }
    std::printf("  (returns are unclipped game scores; training uses clipped rewards)\n");

    // tflow films this greedy episode (900 video frames, every 2nd agent step); the video is omitted.
    rl::Obs o = eval_env.reset(4242);
    float score = 0;
    int steps = 0;
    while (steps < 1800) {
      rl::StepResult sr = eval_env.step(greedy(o));
      score += sr.reward;
      o = sr.obs;
      ++steps;
      if (sr.done()) break;
    }
    std::printf("  greedy episode (video omitted): %d agent steps, score %.0f\n", steps, score);
#else
    (void)game;
    SyntheticPipeline(cfg);
#endif
    std::cout << "\nchecks passed\n";
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
