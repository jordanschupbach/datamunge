#pragma once
/// @file sequence_replay.hpp
/// @brief Source-overlapping, owning recurrent replay windows with native padded minibatches.
#include <datamunge/rl/replay.hpp>
namespace datamunge::rl::replay {
/// @brief Fresh owning native tensors for one recurrent replay sample.
struct SeqBatch {
  datamunge::nn::Tensor obs;  ///< Float32 CPU[B,L+1,D] observations, last padded by repetition.
  datamunge::nn::Tensor a;    ///< Int64 CPU[B,L] actions, zero beyond episode end.
  datamunge::nn::Tensor r;    ///< Float32 CPU[B,L] rewards, zero beyond episode end.
  datamunge::nn::Tensor done; ///< Float32 CPU[B,L] terminal flags, one beyond episode end.
  datamunge::nn::Tensor mask; ///< Float32 CPU[B,L], one for actual rows and zero for padding.
};

/// @brief Own source-overlapping episode windows for explicitly sampled recurrent learning.
///
/// AddEpisode builds windows beginning every max(1,L/2) transitions, retaining L+1 observations.
/// Tail observations repeat the last state; tail actions/rewards are0, done1 and mask0.
/// Copies own independent windows. Sampling uses raw caller-engine modulo words with replacement.
/// There is no episode collection, hidden recurrent state, burn-in, priority or training runner.
///
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// dm::rl::replay::SequenceReplay replay(8,4,2);
/// @endcode
/// @par References
/// - Kapturowski et al. (2019), Recurrent Experience Replay,
/// https://openreview.net/forum?id=r1lyTjAqYX; Sutton and Barto (2018),
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see AddEpisode(), Sample()
class SequenceReplay {
 public:
  /// @brief Construct an empty bounded recurrent-window ring.
  ///
  /// @param capacity Positive capacity at most1million windows.
  /// @param length L in[1,4096].
  /// @param obs_dim Positive width at most1million; capacity*(L+1)*width at most64million.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// dm::rl::replay::SequenceReplay replay(8,4,2);
  /// @endcode
  /// @par References
  /// - Kapturowski et al. (2019), Recurrent Experience Replay,
  /// https://openreview.net/forum?id=r1lyTjAqYX; Sutton and Barto (2018),
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see AddEpisode()
  SequenceReplay(int capacity, int length, int obs_dim)
      : capacity_(capacity), L_(length), obs_dim_(obs_dim) {
    detail::Geometry(capacity, obs_dim);
    if (length < 1 || length > 4096 || int64_t(capacity) * (length + 1) * obs_dim > 64000000)
      throw std::invalid_argument("bounded positive recurrent replay geometry required");
  }
  // obs has T + 1 entries, the others T.

  /// @brief Copy one caller-completed episode into overlapping ring windows.
  ///
  /// Validates all episode fields before mutation. Windows overwrite circularly once capacity is
  /// reached. Each window has independent owning host arrays; previous sampled tensors remain
  /// valid.
  ///
  /// @param obs T+1 finite equal-width observations, T at most1million; T0 is allowed.
  /// @param a T nonnegative discrete actions.
  /// @param r T finite Float32 rewards.
  /// @param done T binary terminal flags; source time-limit terminality convention remains
  /// caller-owned.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// dm::rl::replay::SequenceReplay
  /// replay(8,4,2);replay.AddEpisode({{0,0},{1,0},{2,0}},{0,1},{1,1},{0,1});
  /// @endcode
  /// @par References
  /// - Kapturowski et al. (2019), Recurrent Experience Replay,
  /// https://openreview.net/forum?id=r1lyTjAqYX; Sutton and Barto (2018),
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Sample()
  void AddEpisode(
      const std::vector<environments::Obs>& obs,
      const std::vector<int>&               a,
      const std::vector<float>&             r,
      const std::vector<float>&             done) {
    if (a.size() > 1000000 || obs.size() != a.size() + 1 || r.size() != a.size() ||
        done.size() != a.size())
      throw std::invalid_argument("aligned bounded episode arrays required");
    for (const auto& row : obs) {
      if (row.size() != size_t(obs_dim_))
        throw std::invalid_argument("fixed replay observation width required");
      for (float v : row)
        if (!std::isfinite(v))
          throw std::invalid_argument("finite replay observations required");
    }
    for (size_t t = 0; t < a.size(); ++t)
      if (a[t] < 0 || !std::isfinite(r[t]) || (done[t] != 0 && done[t] != 1))
        throw std::invalid_argument(
            "valid actions, finite rewards and binary terminality required");
    const int T = static_cast<int>(a.size());
    for (int start = 0; start < T; start += std::max(1, L_ / 2)) {
      Window w;
      for (int t = 0; t <= L_; ++t) {
        const environments::Obs& o = obs[std::min(start + t, T)];
        w.obs.insert(w.obs.end(), o.begin(), o.end());
      }
      for (int t = 0; t < L_; ++t) {
        const bool valid = start + t < T;
        w.a.push_back(valid ? a[start + t] : 0);
        w.r.push_back(valid ? r[start + t] : 0.f);
        w.done.push_back(valid ? done[start + t] : 1.f);
        w.mask.push_back(valid ? 1.f : 0.f);
      }
      if (static_cast<int>(windows_.size()) < capacity_)
        windows_.push_back(std::move(w));
      else
        windows_[next_] = std::move(w);
      next_ = (next_ + 1) % capacity_;
    }
  }

  /// @brief Read the current stored window count.
  ///
  /// @return Integer in[0,capacity].
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// dm::rl::replay::SequenceReplay replay(8,4,2);auto size=replay.size();
  /// @endcode
  /// @par References
  /// - Kapturowski et al. (2019), Recurrent Experience Replay,
  /// https://openreview.net/forum?id=r1lyTjAqYX; Sutton and Barto (2018),
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see AddEpisode()
  int size() const { return static_cast<int>(windows_.size()); }

  /// @brief Sample stored windows uniformly into independent native padded tensors.
  ///
  /// @param B Positive batch count at most1million; B*(L+1)*D at most64million; replay must be
  /// nonempty.
  /// @param rng Caller engine advanced by exactly B raw modulo words.
  /// @return Fresh owning Float32/Int64 CPU SeqBatch, with padding mask preserving source layout.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// dm::rl::replay::SequenceReplay
  /// replay(8,4,2);replay.AddEpisode({{0,0},{1,0}},{0},{1},{1});std::mt19937_64 rng(0);auto
  /// batch=replay.Sample(2,rng);
  /// @endcode
  /// @par References
  /// - Kapturowski et al. (2019), Recurrent Experience Replay,
  /// https://openreview.net/forum?id=r1lyTjAqYX; Sutton and Barto (2018),
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see SeqBatch
  SeqBatch Sample(int B, std::mt19937_64& rng) const {
    detail::SampleSize(B, size());
    if (int64_t(B) * (L_ + 1) * obs_dim_ > 64000000)
      throw std::invalid_argument("bounded recurrent batch geometry required");
    namespace nn = datamunge::nn;
    std::vector<float>        obs, r, done, mask;
    std::vector<std::int64_t> a; ///< Actions[L], zero on padding.
    for (int i = 0; i < B; ++i) {
      const Window& w = windows_[rng() % windows_.size()];
      obs.insert(obs.end(), w.obs.begin(), w.obs.end());
      a.insert(a.end(), w.a.begin(), w.a.end());
      r.insert(r.end(), w.r.begin(), w.r.end());
      done.insert(done.end(), w.done.begin(), w.done.end());
      mask.insert(mask.end(), w.mask.begin(), w.mask.end());
    }
    SeqBatch b;
    b.obs  = nn::from_vector(obs, {B, L_ + 1, obs_dim_});
    b.a    = nn::from_vector(a, {B, L_});
    b.r    = nn::from_vector(r, {B, L_});
    b.done = nn::from_vector(done, {B, L_});
    b.mask = nn::from_vector(mask, {B, L_});
    return b;
  }

 private:
  /// @brief Owning flattened host storage for one padded recurrent window.
  struct Window {
    std::vector<float>        obs;  ///< Flattened[L+1,D] observations.
    std::vector<std::int64_t> a;    ///< Actions[L], zero on padding.
    std::vector<float>        r;    ///< Rewards[L], zero on padding.
    std::vector<float>        done; ///< Terminal flags[L], one on padding.
    std::vector<float>        mask; ///< Valid-row indicators[L].
  };
  int                 capacity_; ///< Fixed owning window capacity.
  int                 L_;        ///< Per-window transition horizon.
  int                 obs_dim_;  ///< Fixed observation width.
  std::vector<Window> windows_;  ///< Owning circular window storage, independent on copies.
  int                 next_ = 0; ///< Next overwritten window slot.
};

} // namespace datamunge::rl::replay
