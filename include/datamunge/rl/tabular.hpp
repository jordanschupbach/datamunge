#pragma once
/// @file tabular.hpp
/// @brief Caller-updated action-value tables, source tie breaking and grid views.
#include <cstdio>
#include <datamunge/nn/tensor.hpp>
#include <datamunge/rl/environments/grid_world.hpp>
#include <limits>
#include <numeric>
/// @brief Native one-step RL mathematical components; callers own policies, objectives and all
/// training loops.
namespace datamunge::rl::foundations {

/// @brief Select the first exact maximizer of finite Float32 action values.
///
/// @param v Nonempty finite action values.
/// @return Lowest maximizing index.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// auto action=f::ArgMax({1.f,2.f,2.f});
/// @endcode
/// @par References
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see ArgMaxRandomTie()
inline int ArgMax(const std::vector<float>& v) {
  if (v.empty())
    throw std::invalid_argument("nonempty action values required");
  for (float x : v)
    if (!std::isfinite(x))
      throw std::invalid_argument("finite action values required");
  return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
}

// Argmax with ties broken uniformly at random.

/// @brief Select an exact maximizing action with the original modulo tie draw.
///
/// @param q Borrowed contiguous finite row of at least n Float32 values.
/// @param n Positive action count.
/// @param rng Caller-owned MT19937-64, advanced by exactly one raw word even for a singleton tie.
/// @return One maximizing index; raw-word modulo tie count preserves the source stream.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// std::vector<float> q{1,2,2};std::mt19937_64 rng(1);auto
/// action=f::ArgMaxRandomTie(q.data(),3,rng);
/// @endcode
/// @par References
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see ArgMax()
inline int ArgMaxRandomTie(const float* q, int n, std::mt19937_64& rng) {
  if (!q || n < 1)
    throw std::invalid_argument("nonempty action row required");
  for (int i = 0; i < n; ++i)
    if (!std::isfinite(q[i]))
      throw std::invalid_argument("finite action values required");
  float best = q[0];
  for (int i = 1; i < n; ++i)
    best = std::max(best, q[i]);
  std::vector<int> ties;
  for (int i = 0; i < n; ++i)
    if (q[i] == best)
      ties.push_back(i);
  return ties[rng() % ties.size()];
}

/// @brief Own a caller-updated action-value table in native Float32 CPU tensor storage.
///
/// Copies clone storage; moves transfer it and invalidate checked access on the source. No
/// optimizer, learning rule or rollout is owned. Scalar/row references borrow storage until
/// assignment, move or destruction. tensor() returns an independent owning copy. Concurrent
/// mutation requires a caller lock.
///
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// f::QTable q(3,2);q(0,1)=.5f;auto snapshot=q.tensor();
/// @endcode
/// @par References
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see QTable::Greedy(), QTable::EpsilonGreedy()
class QTable {
 public:
  /// @brief Allocate a finite native action-value table.
  ///
  /// @param num_states Positive state count.
  /// @param num_actions Positive action count; product at most16000000.
  /// @param init Finite Float32 initial value, default0.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// f::QTable q(3,2,.5f);
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see tensor()
  QTable(int num_states, int num_actions, float init = 0.f)
      : S(num_states),
        A(num_actions),
        q_(datamunge::nn::from_vector(
            std::vector<float>(Elements(num_states, num_actions, init), init),
            {num_states, num_actions})) {}

  /// @brief Deep-copy a valid action-value table.
  ///
  /// @param other Source table; self-assignment is harmless.
  /// @throws std::logic_error Copy source has invalid/mutated geometry.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// f::QTable a(3,2);f::QTable b(a);
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see tensor()
  QTable(const QTable& other) : S(other.S), A(other.A), q_(CloneStorage(other)) {}

  /// @brief Replace this table with an independent clone of another.
  ///
  /// @param other Source table; self-assignment is harmless.
  /// @return This table by reference; clone completes before mutation.
  /// @throws std::logic_error Copy source has invalid/mutated geometry.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// f::QTable a(3,2),b(4,3);b=a;
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see tensor()
  QTable& operator=(const QTable& other) {
    if (this != &other) {
      other.Geometry();
      auto copy = other.q_.clone();
      q_        = std::move(copy);
      S         = other.S;
      A         = other.A;
    }
    return *this;
  }

  /// @brief Transfer native storage and invalidate checked access on the source.
  ///
  /// @param other Source table; self-assignment is harmless.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// f::QTable a(3,2);f::QTable b(std::move(a));
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see tensor()
  QTable(QTable&& other) : S(other.S), A(other.A), q_(std::move(other.q_)) {
    other.q_ = datamunge::nn::Tensor{};
    other.S = other.A = 0;
  }

  /// @brief Transfer another table storage into this object.
  ///
  /// @param other Source table; self-assignment is harmless.
  /// @return This table by reference; moved-from descriptors become0.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// f::QTable a(3,2),b(4,3);b=std::move(a);
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see tensor()
  QTable& operator=(QTable&& other) {
    if (this != &other) {
      q_       = std::move(other.q_);
      S        = other.S;
      A        = other.A;
      other.q_ = datamunge::nn::Tensor{};
      other.S = other.A = 0;
    }
    return *this;
  }

  /// @brief Clone table values into an independent detached native tensor.
  ///
  /// @return Owning Float32 CPU[S,A] snapshot, unaffected by later table writes.
  /// @throws std::logic_error Geometry descriptors were changed or this object was moved from.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// f::QTable q(3,2);auto snapshot=q.tensor();
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Row()
  datamunge::nn::Tensor tensor() const {
    Geometry();
    return q_.detach().clone();
  }

  /// @brief Borrow one mutable table value for an explicit caller update.
  ///
  /// @param s State index in0..S-1.
  /// @param a Action index in0..A-1.
  /// @return Mutable Float32 reference, invalidated by assignment, move or destruction.
  /// @throws std::out_of_range State/action is outside the table.
  /// @throws std::logic_error Geometry descriptors changed or storage was moved.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// f::QTable q(3,2);q(0,1)+=.1f;
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Row()
  float& operator()(int s, int a) {
    Check(s, a);
    return q_.data_ptr<float>()[static_cast<size_t>(s) * A + a];
  }

  /// @brief Read one table value without mutating it.
  ///
  /// @param s State index in0..S-1.
  /// @param a Action index in0..A-1.
  /// @return Float32 value copy.
  /// @throws std::out_of_range State/action is outside the table.
  /// @throws std::logic_error Geometry descriptors changed or storage was moved.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// const f::QTable q(3,2);auto value=q(0,1);
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Row()
  float operator()(int s, int a) const {
    Check(s, a);
    return q_.data_ptr<float>()[static_cast<size_t>(s) * A + a];
  }

  /// @brief Borrow one contiguous action row.
  ///
  /// @param s State index in0..S-1.
  /// @return Const pointer to A values, valid until assignment, move or destruction.
  /// @throws std::out_of_range State is outside the table.
  /// @throws std::logic_error Table geometry is invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// f::QTable q(3,2);const float* row=q.Row(0);
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see tensor()
  const float* Row(int s) const {
    Check(s, 0);
    return q_.data_ptr<float>() + static_cast<size_t>(s) * A;
  }

  /// @brief Read the largest finite action value in one row.
  ///
  /// @param s Valid state index.
  /// @return Float32 maximum.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @throws std::out_of_range State is outside the table.
  /// @throws std::logic_error Table geometry is invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// f::QTable q(3,2);auto value=q.Max(0);
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Greedy()
  float Max(int s) const {
    const float* r = Row(s);
    for (int a = 0; a < A; ++a)
      if (!std::isfinite(r[a]))
        throw std::invalid_argument("finite action values required");
    float m = r[0];
    for (int a = 1; a < A; ++a)
      m = std::max(m, r[a]);
    return m;
  }

  /// @brief Choose a maximizing action with source random tie breaking.
  ///
  /// @param s Valid state index.
  /// @param rng Caller MT19937-64; one modulo draw is consumed even with one maximizer.
  /// @return Maximizing action index.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @throws std::out_of_range State is outside the table.
  /// @throws std::logic_error Table geometry is invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// f::QTable q(3,2);std::mt19937_64 rng(1);auto action=q.Greedy(0,rng);
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see ArgMaxRandomTie()
  int Greedy(int s, std::mt19937_64& rng) const { return ArgMaxRandomTie(Row(s), A, rng); }

  /// @brief Choose the first maximizing action without a random draw.
  ///
  /// @param s Valid state index.
  /// @return Lowest maximizing action index.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @throws std::out_of_range State is outside the table.
  /// @throws std::logic_error Table geometry is invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// f::QTable q(3,2);auto action=q.Greedy(0);
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see ArgMax()
  int Greedy(int s) const {
    std::vector<float> r(Row(s), Row(s) + A);
    return ArgMax(r);
  }

  /// @brief Apply source Float32 epsilon exploration and modulo random ties.
  ///
  /// @param s Valid state index.
  /// @param eps Finite exploration probability in[0,1].
  /// @param rng Caller RNG; one uniform draw then one modulo exploration/tie draw.
  /// @return Action index; eps0 still consumes the uniform and tie draws.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @throws std::out_of_range State is outside the table.
  /// @throws std::logic_error Table geometry is invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// f::QTable q(3,2);std::mt19937_64 rng(1);auto action=q.EpsilonGreedy(0,.1f,rng);
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Greedy()
  int EpsilonGreedy(int s, float eps, std::mt19937_64& rng) const {
    Check(s, 0);
    if (!std::isfinite(eps) || eps < 0 || eps > 1)
      throw std::invalid_argument("epsilon outside[0,1]");
    if (std::uniform_real_distribution<float>(0, 1)(rng) < eps)
      return static_cast<int>(rng() % A);
    return Greedy(s, rng);
  }
  int S; ///< Read-only compatibility descriptor: number of states. Changing it invalidates checked
         ///< access.
  int A; ///< Read-only compatibility descriptor: number of actions. Changing it invalidates checked
         ///< access.

 private:
  /// @brief Clone a geometry-checked table backing tensor.
  ///
  /// @param other Valid source table.
  /// @return Independent native storage.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  static datamunge::nn::Tensor CloneStorage(const QTable& other) {
    other.Geometry();
    return other.q_.clone();
  }

  /// @brief Check finite table initialization before allocating.
  ///
  /// @param states Positive row count.
  /// @param actions Positive column count.
  /// @param init Finite initial value.
  /// @return Product bounded16000000.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  static size_t Elements(int states, int actions, float init) {
    if (states < 1 || actions < 1 || !std::isfinite(init) || uint64_t(states) * actions > 16000000)
      throw std::invalid_argument("bounded finite Q-table geometry required");
    return size_t(states) * actions;
  }

  /// @brief Reject mutated descriptors and moved-from storage.
  ///
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  void Geometry() const {
    if (!q_.defined() || q_.shape() != datamunge::nn::Shape{S, A})
      throw std::logic_error("Q-table geometry descriptors changed");
  }

  /// @brief Check geometry and scalar table indices.
  ///
  /// @param state State row.
  /// @param action Action column.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  void Check(int state, int action) const {
    Geometry();
    if (state < 0 || state >= S || action < 0 || action >= A)
      throw std::out_of_range("Q-table index");
  }
  datamunge::nn::Tensor q_; ///< Exclusively owned Float32 CPU[S,A] values; never an autograd leaf.
};

// Greedy policy arrows on the grid.

/// @brief Render an owning ASCII grid view of caller-supplied values or actions.
///
/// @param env Native four-action grid, read only.
/// @param q Aligned[S,4] finite action-value table.
/// @return Owning text; no table, model or environment mutation.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// e::GridWorld env;env.reset(1);f::QTable q(env.num_states(),4);auto text=f::PolicyGrid(env,q);
/// @endcode
/// @par References
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see environments::GridBase::Render()
inline std::string PolicyGrid(const environments::GridBase& env, const QTable& q) {
  if (q.S != env.num_states() || q.A != 4)
    throw std::invalid_argument("grid Q-table shape");
  return env.Render([&](int r, int c) -> std::string {
    if (env.IsWall(r, c))
      return "#";
    if (env.IsGoal(r, c))
      return "G";
    return environments::GridBase::kArrows[q.Greedy(env.Index(r, c))];
  });
}

// Greedy policy arrows from an explicit policy vector.

/// @brief Render an owning ASCII grid view of caller-supplied values or actions.
///
/// @param env Native four-action grid, read only.
/// @param policy Exactly num_states() actions, each in0..3.
/// @return Owning text; no table, model or environment mutation.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// e::GridWorld env;env.reset(1);auto text=f::PolicyGrid(env,std::vector<int>(env.num_states(),1));
/// @endcode
/// @par References
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see environments::GridBase::Render()
inline std::string PolicyGrid(const environments::GridBase& env, const std::vector<int>& policy) {
  if (policy.size() != size_t(env.num_states()))
    throw std::invalid_argument("grid policy shape");
  for (int a : policy)
    if (a < 0 || a > 3)
      throw std::invalid_argument("grid policy action");
  return env.Render([&](int r, int c) -> std::string {
    if (env.IsWall(r, c))
      return "#";
    if (env.IsGoal(r, c))
      return "G";
    return environments::GridBase::kArrows[policy[env.Index(r, c)]];
  });
}

// State values printed as a table.

/// @brief Render an owning ASCII grid view of caller-supplied values or actions.
///
/// @param env Native grid, read only.
/// @param v Exactly num_states() Float32 values; standard snprintf formatting handles NaNs.
/// @return Owning text; no table, model or environment mutation.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// e::GridWorld env;env.reset(1);auto
/// text=f::ValueGrid(env,std::vector<float>(env.num_states(),0));
/// @endcode
/// @par References
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see environments::GridBase::Render()
inline std::string ValueGrid(const environments::GridBase& env, const std::vector<float>& v) {
  if (v.size() != size_t(env.num_states()))
    throw std::invalid_argument("grid value shape");
  std::string s;
  for (int r = 0; r < env.height(); ++r) {
    for (int c = 0; c < env.width(); ++c) {
      char buf[16];
      if (env.IsWall(r, c))
        std::snprintf(buf, sizeof buf, "%7s", "#");
      else
        std::snprintf(buf, sizeof buf, "%7.2f", v[env.Index(r, c)]);
      s += buf;
    }
    s += "\n";
  }
  return s;
}

} // namespace datamunge::rl::foundations
