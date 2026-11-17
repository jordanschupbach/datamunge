// datamunge::nn port: copied from tflow_examples src/common/rl/snake.h (TF-free dynamics; checked
// native public inputs). Snake on a small grid, with a pixel-style observation: three planes (head,
// body, food) of height x width, flattened to a float vector. Eating food gives +1, dying gives -1,
// every other step 0. The episode is truncated if the snake goes too long without eating.
#pragma once
/// @file snake.hpp
/// @brief Native snake simulation components for explicit caller-owned rollouts.

#include <datamunge/rl/environments/interface.hpp>
#include <deque>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::rl::environments {

/// @brief Local snake teaching game with three binary observation planes and starvation truncation.
///
/// Instances own mutable state and a local RNG; concurrent access to one instance requires external
/// synchronization. Returned observations own their values. Reset before stepping and after done.
/// VecEnv owns its children; client calls remain serialized, while workers access distinct
/// children.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;e::Snake obj;obj.reset(1);
/// @endcode
/// @par References
/// - Repository original local simulation protocol, preserved in
/// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see Env, VecEnv
class Snake : public EnvBase {
 public:
  /// @brief Source action-direction or reward-family enumeration.
  /// @par Example
  /// @code
  /// auto choice=dm::rl::environments::Snake::kUp;
  /// @endcode
  enum Dir {
    kUp    = 0, ///< Move up (decrease row).
    kRight = 1, ///< Move right (increase column).
    kDown  = 2, ///< Move down (increase row).
    kLeft  = 3, ///< Move left (decrease column).
  };
  /// @brief Construct Snake with independent owned simulation state.
  /// @param height Board row count; source geometry must be positive and is bounded1024.
  /// @param width Board column count; source geometry must be positive and is bounded1024.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::invalid_argument Invalid checked bounds, geometry or action.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Snake obj;obj.reset(1);
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  Snake(int height = 8, int width = 8) : EnvBase(0), h_(height), w_(width) {
    if (height < 2 || width < 3 || height > 1024 || width > 1024)
      throw std::invalid_argument("snake geometry2x3..1024x1024 required");
  }
  /// @brief Return the stable human-readable simulation name.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Snake obj;obj.reset(1);auto value=obj.name();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::string name() const override { return "Snake"; }
  /// @brief Return the flattened observation width in Float32 values.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Snake obj;obj.reset(1);auto value=obj.obs_size();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int obs_size() const override { return 3 * h_ * w_; }
  // Observation shape as [planes, height, width], for reshaping into an image.
  /// @brief Return owning three-plane observation geometry[head/body/food,height,width].
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Snake obj;obj.reset(1);auto value=obj.obs_shape();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::vector<int> obs_shape() const { return {3, h_, w_}; }
  /// @brief Return an owning copy of the discrete or continuous action space.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Snake obj;obj.reset(1);auto value=obj.action_space();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  ActionSpace action_space() const override { return ActionSpace::Discrete(4); }
  /// @brief Return owning labels for discrete actions or continuous dimensions.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Snake obj;obj.reset(1);auto value=obj.action_names();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::vector<std::string> action_names() const override { return {"up", "right", "down", "left"}; }
  /// @brief Return the snake body length including its head.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Snake obj;obj.reset(1);auto value=obj.length();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int length() const { return static_cast<int>(body_.size()); }
  /// @brief Return the current head row/column; reset before querying.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Snake obj;obj.reset(1);auto value=obj.head();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::pair<int, int> head() const { return body_.front(); }
  // Body cells from the head to the tail.
  /// @brief Borrow head-to-tail body cells until the next mutation; reset before querying.
  /// @return Const borrowed ordered body deque, valid until mutation.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Snake obj;obj.reset(1);auto value=obj.body();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  const std::deque<std::pair<int, int>>& body() const { return body_; }
  /// @brief Return the current food row/column.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Snake obj;obj.reset(1);auto value=obj.food();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::pair<int, int> food() const { return food_; }
  /// @brief Return the number of board rows.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Snake obj;obj.reset(1);auto value=obj.height();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int height() const { return h_; }
  /// @brief Return the number of board columns.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Snake obj;obj.reset(1);auto value=obj.width();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int width() const { return w_; }
  /// @brief Test whether a board coordinate belongs to the snake body.
  /// @param r Board row within the owning grid.
  /// @param c Board column within the owning grid.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Snake obj;obj.reset(1);auto value=obj.Occupied(0,0);
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  bool Occupied(int r, int c) const {
    for (const auto& seg : body_)
      if (seg == std::make_pair(r, c))
        return true;
    return false;
  }

  /// @brief Reset the simulation state and per-episode counter, retaining the current local RNG
  /// stream.
  /// @return Owning initial observation for environments; board Reset mutates in place.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Snake obj;obj.reset(1);obj.Reset();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  Obs Reset() override {
    BeginEpisode();
    body_.clear();
    body_.push_back({h_ / 2, w_ / 2});
    body_.push_back({h_ / 2, w_ / 2 - 1});
    dir_        = kRight;
    since_food_ = 0;
    PlaceFood();
    return ObsNow();
  }

  /// @brief Advance the mutable simulation once, returning observation, reward and distinct
  /// terminal/time-limit flags.
  /// @param action Action in this instance space; board actions must appear in LegalActions().
  /// Continuous physics clips finite torque/throttle to bounds.
  /// @return Owning StepResult; vector stepping returns one owning row per environment with
  /// separate final_obs on auto-reset.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::invalid_argument Invalid checked bounds, geometry or action.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Snake obj;obj.reset(1);auto result=obj.step(e::Action(0));
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  StepResult step(const Action& action) override {
    ValidateAction(action_space(), action);
    // Reversing into the body is ignored; the snake keeps its direction.
    const int a = action.discrete;
    if ((a + 2) % 4 != dir_)
      dir_ = static_cast<Dir>(a);
    static const int    dr[4] = {-1, 0, 1, 0}, dc[4] = {0, 1, 0, -1};
    std::pair<int, int> next = {body_.front().first + dr[dir_], body_.front().second + dc[dir_]};
    StepResult          r;
    const bool          ate = next == food_;
    if (!ate)
      body_.pop_back(); // the tail moves away unless we grow
    const bool hit_wall =
        next.first < 0 || next.first >= h_ || next.second < 0 || next.second >= w_;
    bool hit_self = false;
    for (const auto& seg : body_)
      hit_self |= seg == next;
    if (hit_wall || hit_self) {
      r.reward     = -1.f;
      r.terminated = true;
      r.obs        = ObsNow();
      return r;
    }
    body_.push_front(next);
    if (ate) {
      r.reward    = 1.f;
      since_food_ = 0;
      if (length() == h_ * w_)
        r.terminated = true; // the board is full: won
      else
        PlaceFood();
    } else if (++since_food_ > 4 * h_ * w_) {
      r.truncated = true;
    }
    r.obs = ObsNow();
    return r;
  }

  /// @brief Return an owning ASCII picture of the current simulation state.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Snake obj;obj.reset(1);auto value=obj.render();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::string render() const override {
    std::vector<std::string> g(h_, std::string(w_, '.'));
    for (const auto& seg : body_)
      g[seg.first][seg.second] = 'o';
    g[body_.front().first][body_.front().second] = '@';
    g[food_.first][food_.second]                 = '*';
    std::string s;
    for (auto& row : g)
      s += row + "\n";
    return s;
  }

 private:
  /// @brief Return owning plane-major binary head/body/food observations.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  Obs ObsNow() const {
    Obs       o(3 * h_ * w_, 0.f);
    const int plane                                                = h_ * w_;
    o[0 * plane + body_.front().first * w_ + body_.front().second] = 1.f;
    for (size_t i = 1; i < body_.size(); ++i)
      o[1 * plane + body_[i].first * w_ + body_[i].second] = 1.f;
    o[2 * plane + food_.first * w_ + food_.second] = 1.f;
    return o;
  }
  /// @brief Draw a free food cell using the local generator, preserving source row-major free-cell
  /// enumeration and modulo selection.
  void PlaceFood() {
    std::vector<std::pair<int, int>> free;
    for (int r = 0; r < h_; ++r)
      for (int c = 0; c < w_; ++c) {
        bool occupied = false;
        for (const auto& seg : body_)
          occupied |= seg == std::make_pair(r, c);
        if (!occupied)
          free.push_back({r, c});
      }
    food_ = free[rng()() % free.size()];
  }
  int h_; ///< Owned snake/filmstrip board height.
  int w_; ///< Owned snake/filmstrip board width or contextual weight matrix, according to its

  ///< declared type.
  std::deque<std::pair<int, int>> body_; ///< Owned head-to-tail snake cells.
  Dir dir_ = kRight;               ///< Current snake direction; reversal into the body is ignored.
  std::pair<int, int> food_{0, 0}; ///< Current food row/column.
  int since_food_ = 0; ///< Steps since last food; source starvation limit uses this counter.
};

} // namespace datamunge::rl::environments
