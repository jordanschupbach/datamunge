// datamunge::nn port of tflow_examples src/common/rl/ale_env.h.
//
// ALE is not a datamunge dependency (the plan keeps emulators in the examples
// repo), so this header splits tflow's AtariEnv in two:
//   - AtariFrameStack: the DQN preprocessing (max-pool of the last two frames,
//     box-filter resize to 84x84, a stack of the last 4 frames, scaling to
//     [0, 1]), plain C++ with no emulator; same code as tflow's private helpers.
//   - AtariEnv: the ALE-backed environment, identical to tflow's, compiled only
//     when <ale/ale_interface.hpp> is on the include path (then link libale).
//     DATAMUNGE_EXAMPLES_HAVE_ALE says which.
// FindRom also looks in ~/projects/tflow_examples/data/roms.
//
// Original comment:
// Atari 2600 games through the Arcade Learning Environment, with the
// standard DQN preprocessing (Mnih et al. 2015): frame skip with max-pooling
// over the last two frames, grayscale, resize to 84x84, a stack of the last
// 4 frames, random no-ops at reset, and optional reward clipping. The ROM
// is looked up in data/roms/ and then $ALE_ROM_DIR (the flake points that
// at the free Tetris ROM shipped with ALE).
#pragma once

#if defined(DATAMUNGE_EXAMPLES_DISABLE_ALE)
// Build systems can select the dependency-free preprocessing path even when
// an unrelated development shell exposes ALE headers.
#define DATAMUNGE_EXAMPLES_HAVE_ALE 0
#else
#if __has_include(<ale/ale_interface.hpp>)
#define DATAMUNGE_EXAMPLES_HAVE_ALE 1
#include <ale/ale_interface.hpp>
#include <ale/common/Log.hpp>
#else
#define DATAMUNGE_EXAMPLES_HAVE_ALE 0
#endif
#endif

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/rl/env.hpp"

namespace rl {

// Directories searched for <game>.bin, in order.
inline std::vector<std::string> RomDirs() {
  std::vector<std::string> dirs = {"data/roms"};
  if (const char* d = std::getenv("TFLOW_DATA_DIR")) dirs.push_back(std::string(d) + "/roms");
  if (const char* d = std::getenv("ALE_ROM_DIR")) dirs.push_back(d);
  if (const char* h = std::getenv("HOME")) dirs.push_back(std::string(h) + "/projects/tflow_examples/data/roms");
  return dirs;
}

inline std::string FindRom(const std::string& game) {
  for (const std::string& dir : RomDirs()) {
    const std::string p = dir + "/" + game + ".bin";
    if (std::filesystem::exists(p)) return p;
  }
  throw std::runtime_error("ROM " + game + ".bin not found in data/roms/ or $ALE_ROM_DIR");
}

struct AtariOptions {
  int frame_skip = 4;
  bool clip_rewards = false;
  int noop_max = 30;
  int max_frames = 27000;  // 108k emulator frames at frame skip 4: Gymnasium's limit
  float repeat_action_probability = 0.25f;  // sticky actions
};

// The DQN frame pipeline over raw grayscale screens of raw_w x raw_h.
class AtariFrameStack {
 public:
  static constexpr int kSize = 84, kStack = 4;
  AtariFrameStack(int raw_w = 160, int raw_h = 210) : raw_w_(raw_w), raw_h_(raw_h) {}
  void Clear() { frames_.clear(); }
  // Max over the last two frames removes Atari flicker (`prev` may be empty).
  static void MaxPool(std::vector<unsigned char>& cur, const std::vector<unsigned char>& prev) {
    if (!prev.empty())
      for (size_t i = 0; i < cur.size(); ++i) cur[i] = std::max(cur[i], prev[i]);
  }
  // Box-filter resize of the grayscale screen to 84x84.
  void PushFrame(const std::vector<unsigned char>& gray) {
    std::vector<unsigned char> small(kSize * kSize);
    for (int y = 0; y < kSize; ++y)
      for (int x = 0; x < kSize; ++x) {
        const int y0 = y * raw_h_ / kSize, y1 = std::max(y0 + 1, (y + 1) * raw_h_ / kSize);
        const int x0 = x * raw_w_ / kSize, x1 = std::max(x0 + 1, (x + 1) * raw_w_ / kSize);
        int sum = 0, n = 0;
        for (int yy = y0; yy < y1; ++yy)
          for (int xx = x0; xx < x1; ++xx) sum += gray[yy * raw_w_ + xx], ++n;
        small[y * kSize + x] = static_cast<unsigned char>(sum / n);
      }
    frames_.push_back(std::move(small));
    while (static_cast<int>(frames_.size()) > kStack) frames_.pop_front();
  }
  int obs_size() const { return kStack * kSize * kSize; }
  Obs StackedObs() const {
    Obs o(obs_size());
    for (int f = 0; f < kStack; ++f)
      for (int i = 0; i < kSize * kSize; ++i) o[f * kSize * kSize + i] = frames_[f][i] / 255.f;
    return o;
  }
  const std::vector<unsigned char>& last_frame() const { return frames_.back(); }

 private:
  int raw_w_, raw_h_;
  std::deque<std::vector<unsigned char>> frames_;
};

#if DATAMUNGE_EXAMPLES_HAVE_ALE
class AtariEnv : public Env {
 public:
  static constexpr int kSize = AtariFrameStack::kSize, kStack = AtariFrameStack::kStack;
  using Options = AtariOptions;
  AtariEnv(const std::string& game, Options opt = AtariOptions()) : game_(game), opt_(opt) {
    ale::Logger::setMode(ale::Logger::Error);  // no ROM banner / "Sending Reset..." chatter
    ale_.setInt("random_seed", 0);
    ale_.setFloat("repeat_action_probability", opt.repeat_action_probability);
    ale_.setInt("frame_skip", 1);  // skipping is done here so frames can be max-pooled
    ale_.loadROM(FindRom(game));
    actions_ = ale_.getMinimalActionSet();
    raw_w_ = static_cast<int>(ale_.getScreen().width());
    raw_h_ = static_cast<int>(ale_.getScreen().height());
    frames_ = AtariFrameStack(raw_w_, raw_h_);
  }
  std::string name() const override { return "Atari(" + game_ + ")"; }
  int obs_size() const override { return kStack * kSize * kSize; }
  ActionSpace action_space() const override { return ActionSpace::Discrete(static_cast<int>(actions_.size())); }
  void Seed(uint64_t seed) override {
    rng_.seed(seed);
    ale_.setInt("random_seed", static_cast<int>(seed));
    ale_.loadROM(FindRom(game_));
  }
  int max_steps() const override { return opt_.max_frames; }
  int lives() { return ale_.lives(); }
  std::vector<std::string> action_names() const override {
    static const char* names[] = {"NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT",
                                  "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"};
    std::vector<std::string> out;
    for (auto a : actions_) out.push_back(names[static_cast<int>(a)]);
    return out;
  }
  // The current raw RGB screen (raw_w x raw_h), for videos.
  const std::vector<unsigned char>& rgb() {
    ale_.getScreenRGB(rgb_);
    return rgb_;
  }
  int raw_width() const { return raw_w_; }
  int raw_height() const { return raw_h_; }
  // The most recent preprocessed 84x84 frame (0..255), for inspection.
  const std::vector<unsigned char>& last_frame() const { return frames_.last_frame(); }

  StepResult step(const Action& action) override {
    StepResult r;
    float reward = 0;
    std::vector<unsigned char> prev, cur;
    for (int k = 0; k < opt_.frame_skip; ++k) {
      reward += static_cast<float>(ale_.act(actions_[action.discrete]));
      if (k == opt_.frame_skip - 2) ale_.getScreenGrayscale(prev);
      if (ale_.game_over()) break;
    }
    ale_.getScreenGrayscale(cur);
    AtariFrameStack::MaxPool(cur, prev);
    frames_.PushFrame(cur);
    r.obs = frames_.StackedObs();
    r.reward = opt_.clip_rewards ? (reward > 0 ? 1.f : (reward < 0 ? -1.f : 0.f)) : reward;
    raw_reward_ = reward;
    r.terminated = ale_.game_over();
    r.truncated = !r.terminated && ++steps_ >= opt_.max_frames;
    return r;
  }
  float last_raw_reward() const { return raw_reward_; }

 protected:
  Obs Reset() override {
    ale_.reset_game();
    steps_ = 0;
    frames_.Clear();
    const int noops = opt_.noop_max > 0 ? static_cast<int>(rng_() % (opt_.noop_max + 1)) : 0;
    for (int i = 0; i < noops; ++i) ale_.act(actions_[0]);
    std::vector<unsigned char> g;
    ale_.getScreenGrayscale(g);
    for (int i = 0; i < kStack; ++i) frames_.PushFrame(g);
    return frames_.StackedObs();
  }

 private:
  std::string game_;
  Options opt_;
  ale::ALEInterface ale_;
  ale::ActionVect actions_;
  AtariFrameStack frames_;
  std::vector<unsigned char> rgb_;
  std::mt19937_64 rng_{0};
  int raw_w_ = 0, raw_h_ = 0, steps_ = 0;
  float raw_reward_ = 0;
};
#endif  // DATAMUNGE_EXAMPLES_HAVE_ALE

}  // namespace rl
