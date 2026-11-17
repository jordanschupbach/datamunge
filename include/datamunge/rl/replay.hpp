#pragma once
/// @file replay.hpp
/// @brief Owning source-order uniform/prioritized replay and explicit n-step transition
/// accumulation.
#include <algorithm>
#include <cmath>
#include <datamunge/nn/tensor.hpp>
#include <datamunge/rl/environments/interface.hpp>
#include <limits>
#include <random>
/// @brief Owning replay data structures; caller owns sampling engines, objectives and every
/// rollout/update loop.
namespace datamunge::rl::replay {

/// @brief Own one discrete or continuous transition with terminal-only bootstrap suppression.
///
/// Copies own independent host vectors. done is1 only for an actual terminal successor;
/// time-limit truncation uses0. NStepBuffer() keeps the same representation for reward sums.
///
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// dm::rl::replay::Experience t{{0,1},0,1,{1,2},0,{}};
/// @endcode
/// @par References
/// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952; Sutton
/// and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
/// @see Batch, NStepBuffer
struct Experience {
  environments::Obs  s;  ///< Owning finite source observation, nonempty.
  int                a;  ///< Nonnegative discrete action; ignored by a continuous-action consumer.
  float              r;  ///< Finite task reward or accumulated n-step reward.
  environments::Obs  s2; ///< Owning successor observation, same width as s.
  float              done;  ///< Binary terminal flag; time-limit truncation uses0.
  std::vector<float> act_c; ///< Owning optional finite continuous action; then a is unused.
};

/// @brief Internal validation shared by native replay structures.
namespace detail {

/// @brief Validate bounded positive ring capacity and observation width.
///
/// @param capacity Positive at most1million.
/// @param width Positive at most1million; product at most64million.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par References
/// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952; Sutton
/// and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
inline void Geometry(int capacity, int width) {
  if (capacity < 1 || width < 1 || capacity > 1000000 || width > 1000000 ||
      int64_t(capacity) * width > 64000000)
    throw std::invalid_argument("bounded positive replay geometry required");
}

/// @brief Validate finite owning transition values before storage or batch packing.
///
/// @param t Transition with nonempty matching observations, nonnegative action, finite reward,
/// binary terminality.
/// @param width If positive, require this observation width;0 infers it.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par References
/// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952; Sutton
/// and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
inline void ExperienceValues(const Experience& t, int width = 0) {
  if (t.s.empty() || t.s.size() != t.s2.size() || (width && t.s.size() != size_t(width)) ||
      t.a < 0 || !std::isfinite(t.r) || (t.done != 0 && t.done != 1))
    throw std::invalid_argument("valid finite experience geometry/reward/terminality required");
  for (const auto& row : {t.s, t.s2, t.act_c})
    for (float v : row)
      if (!std::isfinite(v))
        throw std::invalid_argument("finite replay observations/actions required");
}

/// @brief Validate positive batch size and nonempty source replay.
///
/// @param count Requested sample count at most1million.
/// @param size Current stored transition count, positive.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par References
/// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952; Sutton
/// and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
inline void SampleSize(int count, int size) {
  if (count < 1 || count > 1000000 || size < 1)
    throw std::invalid_argument("positive bounded batch and nonempty replay required");
}
} // namespace detail
// A minibatch of transitions as tensors.
/// @brief Owning native CPU minibatch copied from replay experiences.
/// @details Tensor assignment shares native tensor storage; samples themselves are fresh owning
/// copies.
struct Batch {
  datamunge::nn::Tensor s;       ///< Float32 source observations[N,D].
  datamunge::nn::Tensor a;       ///< Int64 discrete actions[N].
  datamunge::nn::Tensor r;       ///< Float32 rewards[N].
  datamunge::nn::Tensor s2;      ///< Float32 successor observations[N,D].
  datamunge::nn::Tensor done;    ///< Float32 binary terminal flags[N].
  datamunge::nn::Tensor weights; ///< Float32 positive sampling weights[N].
  datamunge::nn::Tensor act_c;   ///< Float32 continuous actions[N,C], undefined when C0.
  std::vector<int>      indices; ///< Owning sampled buffer positions, for later priority updates.
};

/// @brief Copy borrowed transitions into independent native minibatch tensors.
///
/// Inputs are validated before tensor construction; input pointers are never retained.
/// An empty input yields empty tensors with the supplied observation width.
///
/// @param ts Nonnull experience pointers, valid during call; up to1million rows, same
/// observation/continuous-action geometry.
/// @param obs_dim Positive observation width; row count*width at most64million.
/// @param weights One finite positive Float32 weight per row.
/// @param indices One nonnegative buffer position per row; copied/moved into result.
/// @return Owning Float32/Int64 CPU Batch; no autograd path to host input vectors.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// dm::rl::replay::Experience t{{0},0,1,{1},0,{}};auto
/// batch=dm::rl::replay::MakeBatch({&t},1,{1},{0});
/// @endcode
/// @par References
/// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952; Sutton
/// and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
/// @see ReplayBuffer::Sample(), PrioritizedReplay::Sample()
inline Batch MakeBatch(
    const std::vector<const Experience*>& ts,
    int                                   obs_dim,
    const std::vector<float>&             weights,
    std::vector<int>                      indices) {
  namespace nn = datamunge::nn;
  if (obs_dim < 1 || ts.size() > 1000000 || weights.size() != ts.size() ||
      indices.size() != ts.size() || int64_t(ts.size()) * obs_dim > 64000000)
    throw std::invalid_argument("aligned bounded replay minibatch required");
  for (size_t i = 0; i < ts.size(); ++i) {
    if (!ts[i])
      throw std::invalid_argument("nonnull experience required");
    detail::ExperienceValues(*ts[i], obs_dim);
    if (ts[i]->act_c.size() != ts[0]->act_c.size() || !std::isfinite(weights[i]) ||
        weights[i] <= 0 || indices[i] < 0)
      throw std::invalid_argument("aligned action geometry and positive finite weights required");
  }
  const int          n       = static_cast<int>(ts.size());
  const int          act_dim = n ? static_cast<int>(ts[0]->act_c.size()) : 0;
  std::vector<float> s(static_cast<std::size_t>(n) * obs_dim), s2(s.size()), r(n), done(n),
      act_c(static_cast<std::size_t>(n) * act_dim);
  std::vector<std::int64_t> a(n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < act_dim; ++j)
      act_c[i * act_dim + j] = ts[i]->act_c[j];
    for (int j = 0; j < obs_dim; ++j) {
      s[i * obs_dim + j]  = ts[i]->s[j];
      s2[i * obs_dim + j] = ts[i]->s2[j];
    }
    a[i]    = ts[i]->a;
    r[i]    = ts[i]->r;
    done[i] = ts[i]->done;
  }
  Batch b;
  b.s       = nn::from_vector(s, {n, obs_dim});
  b.s2      = nn::from_vector(s2, {n, obs_dim});
  b.a       = nn::from_vector(a, {n});
  b.r       = nn::from_vector(r, {n});
  b.done    = nn::from_vector(done, {n});
  b.weights = nn::from_vector(weights, {n});
  if (act_dim)
    b.act_c = nn::from_vector(act_c, {n, act_dim});
  b.indices = std::move(indices);
  return b;
}

// Uniform replay: a ring buffer sampled uniformly at random.

/// @brief Uniform ring replay with source raw-word modulo sampling.
///
/// Copies own independent experiences. Add overwrites circularly; Sample samples with replacement
/// and advances exactly n raw caller-engine words. No learner, optimizer or RNG is retained.
///
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// dm::rl::replay::ReplayBuffer buffer(8,2);
/// @endcode
/// @par References
/// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952; Sutton
/// and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
/// @see Experience, Batch
class ReplayBuffer {
 public:
  /// @brief Construct empty owning replay storage.
  ///
  /// @param capacity Positive ring capacity at most1million.
  /// @param obs_dim Positive width at most1million; capacity*width at most64million.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// dm::rl::replay::ReplayBuffer buffer(8,2);
  /// @endcode
  /// @par References
  /// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952;
  /// Sutton and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Add(), Sample()
  ReplayBuffer(int capacity, int obs_dim) : capacity_(capacity), obs_dim_(obs_dim) {
    detail::Geometry(capacity, obs_dim);
  }

  /// @brief Copy/move one validated transition into the next ring slot.
  ///
  /// The oldest slot is overwritten once full. Sampled batches remain independent of later writes.
  ///
  /// @param t Owning finite Experience with fixed observation and continuous-action widths.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// dm::rl::replay::ReplayBuffer buffer(8,2);buffer.Add({{0,1},0,1,{1,2},0,{}});
  /// @endcode
  /// @par References
  /// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952;
  /// Sutton and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Sample()
  void Add(Experience t) {
    detail::ExperienceValues(t, obs_dim_);
    if (!data_.empty() && t.act_c.size() != data_[0].act_c.size())
      throw std::invalid_argument("consistent continuous action width required");
    if (static_cast<int>(data_.size()) < capacity_) {
      data_.push_back(std::move(t));
    } else {
      data_[next_] = std::move(t);
    }
    next_ = (next_ + 1) % capacity_;
  }

  /// @brief Return the current stored transition count.
  ///
  /// @return Integer in[0,capacity].
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// dm::rl::replay::ReplayBuffer buffer(8,2);auto size=buffer.size();
  /// @endcode
  /// @par References
  /// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952;
  /// Sutton and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Add()
  int size() const { return static_cast<int>(data_.size()); }

  /// @brief Sample uniformly with replacement into fresh native tensors.
  ///
  /// @param n Positive batch count at most1million; replay must be nonempty.
  /// @param rng Caller engine advanced by exactly n raw words, source modulo order.
  /// @return Owning Batch, all weights1; duplicate indices are possible.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// dm::rl::replay::ReplayBuffer buffer(8,2);buffer.Add({{0,1},0,1,{1,2},0,{}});std::mt19937_64
  /// rng(1);auto sample=buffer.Sample(2,rng);
  /// @endcode
  /// @par References
  /// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952;
  /// Sutton and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
  /// @see MakeBatch()
  Batch Sample(int n, std::mt19937_64& rng) const {
    detail::SampleSize(n, size());
    std::vector<const Experience*> ts(n);
    std::vector<int>               idx(n);
    for (int i = 0; i < n; ++i) {
      idx[i] = static_cast<int>(rng() % data_.size());
      ts[i]  = &data_[idx[i]];
    }
    return MakeBatch(ts, obs_dim_, std::vector<float>(n, 1.f), idx);
  }

 private:
  int                     capacity_; ///< Fixed owning ring capacity.
  int                     obs_dim_;  ///< Fixed source/successor observation width.
  std::vector<Experience> data_;     ///< Owning experience ring, independent on copies.
  int                     next_ = 0; ///< Next overwritten ring slot.
};

// Prioritised replay (Schaul et al. 2016): sampling probability proportional
// to priority^alpha, with importance weights (N p)^-beta normalised by their
// maximum. A sum tree makes sampling O(log N).

/// @brief Source sum-tree proportional replay with stratified sampling and importance weights.
///
/// Copies own independent tree and experiences. New rows receive the maximum historical transformed
/// priority. Sampling uses one Float32 uniform per stratum. Weight normalization uses the sampled
/// batch maximum, not a full-buffer maximum. Concurrent writes or sample/write races require caller
/// synchronization.
///
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// dm::rl::replay::PrioritizedReplay buffer(8,2);
/// @endcode
/// @par References
/// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952; Sutton
/// and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
/// @see Experience, Batch
class PrioritizedReplay {
 public:
  /// @brief Construct empty owning replay storage.
  ///
  /// @param capacity Positive ring capacity at most1million.
  /// @param obs_dim Positive width at most1million; capacity*width at most64million.
  /// @param alpha Finite priority exponent in[0,1], default.6.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// dm::rl::replay::PrioritizedReplay buffer(8,2);
  /// @endcode
  /// @par References
  /// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952;
  /// Sutton and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Add(), Sample()
  PrioritizedReplay(int capacity, int obs_dim, float alpha = 0.6f)
      : capacity_(capacity),
        obs_dim_(obs_dim),
        alpha_(alpha),
        tree_(TreeSize(capacity, obs_dim, alpha), 0.f) {}

  /// @brief Copy/move one validated transition into the next ring slot.
  ///
  /// The oldest slot is overwritten once full. Sampled batches remain independent of later writes.
  ///
  /// @param t Owning finite Experience with fixed observation and continuous-action widths.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// dm::rl::replay::PrioritizedReplay buffer(8,2);buffer.Add({{0,1},0,1,{1,2},0,{}});
  /// @endcode
  /// @par References
  /// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952;
  /// Sutton and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Sample()
  void Add(Experience t) {
    detail::ExperienceValues(t, obs_dim_);
    if (!data_.empty() && t.act_c.size() != data_[0].act_c.size())
      throw std::invalid_argument("consistent continuous action width required");
    const int pos = next_;
    if (static_cast<int>(data_.size()) < capacity_)
      data_.push_back(std::move(t));
    else
      data_[pos] = std::move(t);
    SetPriority(pos, max_priority_); // new samples are tried at least once
    next_ = (next_ + 1) % capacity_;
  }

  /// @brief Return the current stored transition count.
  ///
  /// @return Integer in[0,capacity].
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// dm::rl::replay::PrioritizedReplay buffer(8,2);auto size=buffer.size();
  /// @endcode
  /// @par References
  /// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952;
  /// Sutton and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Add()
  int size() const { return static_cast<int>(data_.size()); }

  /// @brief Read the Float32 sum-tree root mass.
  ///
  /// @return Sum of transformed priorities;0 when empty.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// dm::rl::replay::PrioritizedReplay buffer(8,2);auto total=buffer.total();
  /// @endcode
  /// @par References
  /// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952;
  /// Sutton and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Sample()
  float total() const { return tree_[1]; }

  /// @brief Read the maximum historical transformed priority.
  ///
  /// @return Initial1 or the largest pow(abs_td+1e-6,alpha) seen so far.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// dm::rl::replay::PrioritizedReplay buffer(8,2);auto maximum=buffer.max_priority();
  /// @endcode
  /// @par References
  /// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952;
  /// Sutton and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
  /// @see UpdatePriorities()
  float max_priority() const { return max_priority_; }

  /// @brief Sample stratified proportional replay with source Float32 importance weights.
  ///
  /// Each stratum spans total_mass/n. Exact-zero draws cannot select a zero-mass empty leaf.
  /// Caller refreshes priorities after computing its own learning errors.
  ///
  /// @param n Positive batch count at most1million; replay must be nonempty.
  /// @param beta Finite correction exponent in[0,1].
  /// @param rng Caller engine advanced by one Float32 uniform draw per sampled row.
  /// @return Owning native Batch with weights(N*p)^(-beta), normalized by sampled maximum.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// dm::rl::replay::PrioritizedReplay
  /// buffer(8,2);buffer.Add({{0,1},0,1,{1,2},0,{}});std::mt19937_64 rng(1);auto
  /// sample=buffer.Sample(2,.4f,rng);
  /// @endcode
  /// @par References
  /// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952;
  /// Sutton and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
  /// @see SampleUniform(), UpdatePriorities()
  Batch Sample(int n, float beta, std::mt19937_64& rng) const {
    detail::SampleSize(n, size());
    if (!std::isfinite(beta) || beta < 0 || beta > 1 || !std::isfinite(total()) || total() <= 0)
      throw std::invalid_argument("finite positive priority mass and beta in[0,1] required");
    std::vector<const Experience*> ts(n);
    std::vector<int>               idx(n);
    std::vector<float>             w(n);
    const float                    segment = total() / n; // stratified sampling
    float                          max_w   = 0;
    for (int i = 0; i < n; ++i) {
      const float u = segment * (i + std::uniform_real_distribution<float>(0, 1)(rng));
      idx[i]        = Find(u);
      ts[i]         = &data_[idx[i]];
      const float p = tree_[idx[i] + capacity_] / total();
      w[i]          = std::pow(size() * p, -beta);
      max_w         = std::max(max_w, w[i]);
    }
    for (float& x : w)
      x /= max_w;
    return MakeBatch(ts, obs_dim_, w, idx);
  }
  // A uniformly sampled batch (weights 1), for comparisons.

  /// @brief Sample the same stored rows uniformly for an explicit control comparison.
  ///
  /// @param n Positive bounded batch count; replay must be nonempty.
  /// @param rng Caller engine advanced by n raw modulo words.
  /// @return Owning native Batch with all weights1.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// dm::rl::replay::PrioritizedReplay
  /// buffer(8,2);buffer.Add({{0,1},0,1,{1,2},0,{}});std::mt19937_64 rng(1);auto
  /// sample=buffer.SampleUniform(2,rng);
  /// @endcode
  /// @par References
  /// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952;
  /// Sutton and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Sample()
  Batch SampleUniform(int n, std::mt19937_64& rng) const {
    detail::SampleSize(n, size());
    std::vector<const Experience*> ts(n);
    std::vector<int>               idx(n);
    for (int i = 0; i < n; ++i) {
      idx[i] = static_cast<int>(rng() % data_.size());
      ts[i]  = &data_[idx[i]];
    }
    return MakeBatch(ts, obs_dim_, std::vector<float>(n, 1.f), idx);
  }
  // New priorities from the absolute TD errors of the last batch.

  /// @brief Refresh caller-selected priorities in the supplied update order.
  ///
  /// All indices/errors are validated before mutation. Priorities are(abs_td+1e-6)^alpha.
  /// Maximum historical transformed priority never decreases, including after overwritten slots.
  ///
  /// @param indices Aligned valid stored positions; repeated indices are allowed, last update wins.
  /// @param td_abs Aligned finite nonnegative errors; transformed masses must leave total Float32
  /// mass representable.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// dm::rl::replay::PrioritizedReplay
  /// buffer(8,2);buffer.Add({{0,1},0,1,{1,2},0,{}});buffer.UpdatePriorities({0},{2});
  /// @endcode
  /// @par References
  /// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952;
  /// Sutton and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Sample()
  void UpdatePriorities(const std::vector<int>& indices, const std::vector<float>& td_abs) {
    if (indices.size() != td_abs.size())
      throw std::invalid_argument("aligned priority updates required");
    for (size_t j = 0; j < indices.size(); ++j)
      if (indices[j] < 0 || indices[j] >= size() || !std::isfinite(td_abs[j]) || td_abs[j] < 0 ||
          !std::isfinite(std::pow(td_abs[j] + 1e-6f, alpha_)) ||
          std::pow(td_abs[j] + 1e-6f, alpha_) > std::numeric_limits<float>::max() / capacity_)
        throw std::invalid_argument(
            "valid priority indices and bounded nonnegative errors required");
    for (size_t i = 0; i < indices.size(); ++i) {
      const float p = std::pow(td_abs[i] + 1e-6f, alpha_);
      max_priority_ = std::max(max_priority_, p);
      SetPriority(indices[i], p);
    }
  }

 private:
  /// @brief Validate replay geometry/exponent before allocating the sum tree.
  ///
  /// @param capacity Requested ring size.
  /// @param width Observation width.
  /// @param alpha Finite exponent in[0,1].
  /// @return Twice the capacity.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par References
  /// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952;
  /// Sutton and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
  static size_t TreeSize(int capacity, int width, float alpha) {
    detail::Geometry(capacity, width);
    if (!std::isfinite(alpha) || alpha < 0 || alpha > 1)
      throw std::invalid_argument("priority alpha in[0,1] required");
    return size_t(capacity) * 2;
  }

  /// @brief Replace one valid leaf mass and recompute its ancestors.
  ///
  /// @param pos Stored ring position.
  /// @param p Validated positive transformed mass.
  /// @par References
  /// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952;
  /// Sutton and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
  void SetPriority(int pos, float p) {
    int i    = pos + capacity_;
    tree_[i] = p;
    for (i /= 2; i >= 1; i /= 2)
      tree_[i] = tree_[2 * i] + tree_[2 * i + 1];
  }
  // Index of the leaf whose cumulative range contains u.

  /// @brief Find a source closed-boundary cumulative leaf, skipping empty zero-mass leaves.
  ///
  /// @param u Cumulative query in[0,total].
  /// @return Stored index with positive mass.
  /// @par References
  /// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952;
  /// Sutton and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
  int Find(float u) const {
    int i = 1;
    while (i < capacity_) {
      if (u <= tree_[2 * i])
        i = 2 * i;
      else
        u -= tree_[2 * i], i = 2 * i + 1;
    }
    int pos = std::min(i - capacity_, size() - 1);
    // A uniform draw can be exactly0; never select an empty zero-mass leaf.
    if (pos < 0 || tree_[pos + capacity_] <= 0) {
      for (int j = 0; j < size(); ++j)
        if (tree_[j + capacity_] > 0)
          return j;
      throw std::logic_error("priority tree has no positive leaf");
    }
    return pos;
  }
  int                     capacity_;         ///< Fixed owning ring capacity.
  int                     obs_dim_;          ///< Fixed source/successor observation width.
  float                   alpha_;            ///< Priority exponent in[0,1].
  std::vector<float>      tree_;             ///< One-indexed sum tree; leaves[capacity,2*capacity).
  std::vector<Experience> data_;             ///< Owning experience ring, independent on copies.
  int                     next_         = 0; ///< Next overwritten ring slot.
  float                   max_priority_ = 1.f; ///< Largest historical transformed priority.
};

// Turns one-step transitions into n-step ones: reward r_t + g r_{t+1} + ... +
// g^{n-1} r_{t+n-1}, next state s_{t+n}. Push() returns the completed
// transition once n steps are available; Flush() drains the rest when an
// episode ends (shorter horizons, bootstrapping off the terminal state).

/// @brief Explicit transition accumulation for caller-controlled n-step replay.
///
/// Push returns full-horizon rows; an actual terminal flushes all pending prefixes. Caller must
/// Flush at a time-limit boundary. Short flushed prefixes retain terminality but do not encode
/// horizon length: the caller is responsible for any shortened-horizon bootstrap discount. No
/// episode or fitting loop is owned.
///
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// dm::rl::replay::NStepBuffer buffer(3,.9f);
/// @endcode
/// @par References
/// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952; Sutton
/// and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
/// @see Experience, Batch
class NStepBuffer {
 public:
  /// @brief Construct an empty explicit n-step accumulator.
  ///
  /// @param n Horizon in[1,4096].
  /// @param gamma Finite discount in[0,1].
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// dm::rl::replay::NStepBuffer buffer(3,.9f);
  /// @endcode
  /// @par References
  /// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952;
  /// Sutton and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Push(), Flush()
  NStepBuffer(int n, float gamma) : n_(n), gamma_(gamma) {
    if (n < 1 || n > 4096 || !std::isfinite(gamma) || gamma < 0 || gamma > 1)
      throw std::invalid_argument("n in[1,4096] and discount in[0,1] required");
  }

  /// @brief Append one transition and return any completed source-order prefixes.
  ///
  /// Rewards accumulate in Float32 as r0+gamma*r1+...; successor and terminality come from
  /// the final included transition. Input is copied. No reset or episode boundary is inferred from
  /// observations.
  ///
  /// @param t Finite owning transition; observation widths must match pending rows.
  /// @return Independent experiences; empty until n rows, or all pending prefixes on a terminal.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// dm::rl::replay::NStepBuffer buffer(3,.9f);auto ready=buffer.Push({{0},0,1,{1},0,{}});
  /// @endcode
  /// @par References
  /// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952;
  /// Sutton and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Flush()
  std::vector<Experience> Push(const Experience& t) {
    detail::ExperienceValues(t);
    if (!pending_.empty())
      detail::ExperienceValues(t, int(pending_[0].s.size()));
    pending_.push_back(t);
    std::vector<Experience> out;
    if (t.done > 0)
      return Flush();
    if (static_cast<int>(pending_.size()) >= n_) {
      out.push_back(Combine(0));
      pending_.erase(pending_.begin());
    }
    return out;
  }

  /// @brief Drain pending shorter prefixes at an explicit caller-supplied episode boundary.
  ///
  /// Nonterminal time-limit prefixes keep done0. Their bootstrap horizon is shorter than n;
  /// the caller must preserve that distinction if using a horizon-dependent discount.
  ///
  /// @return Owning oldest-to-newest combined transitions; pending rows are cleared.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// dm::rl::replay::NStepBuffer buffer(3,.9f);buffer.Push({{0},0,1,{1},0,{}});auto
  /// ready=buffer.Flush();
  /// @endcode
  /// @par References
  /// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952;
  /// Sutton and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Push()
  std::vector<Experience> Flush() {
    std::vector<Experience> out;
    for (size_t i = 0; i < pending_.size(); ++i)
      out.push_back(Combine(i));
    pending_.clear();
    return out;
  }

 private:
  /// @brief Combine one pending prefix without mutation.
  ///
  /// @param start Valid pending row index.
  /// @return Owning discounted prefix, ending at min(start+n,pending size).
  /// @par References
  /// - Schaul et al. (2016), Prioritized Experience Replay, https://arxiv.org/abs/1511.05952;
  /// Sutton and Barto (2018), Chapters7 and11, http://incompleteideas.net/book/the-book-2nd.html .
  Experience Combine(size_t start) const {
    Experience t        = pending_[start];
    float      discount = gamma_;
    for (size_t i = start + 1; i < pending_.size() && i < start + n_; ++i) {
      t.r += discount * pending_[i].r;
      discount *= gamma_;
      t.s2   = pending_[i].s2;
      t.done = pending_[i].done;
    }
    return t;
  }
  int                     n_;       ///< Maximum reward accumulation horizon.
  float                   gamma_;   ///< Finite per-step discount.
  std::vector<Experience> pending_; ///< Owning pending prefix transitions.
};

} // namespace datamunge::rl::replay
