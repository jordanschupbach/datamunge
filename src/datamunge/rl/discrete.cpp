/// @file discrete.cpp
/// @brief Native masked tabular/Double-DQN/dueling/PPO training and inference checkpoints.
#include <cstring>
#include <datamunge/nn/io.hpp>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/losses.hpp>
#include <datamunge/nn/optim.hpp>
#include <datamunge/nn/training.hpp>
#include <datamunge/rl/discrete.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <set>

#include "detail.hpp"

namespace datamunge::rl {
/// @brief Private validation, network and batching helpers.
namespace {
namespace n = datamunge::nn; ///< Native tensor/autograd backend.
/// @brief Independent deterministic reset-seed mixing, never consuming action RNGs.
/// @param v Window seed.
/// @return Full-width mixed seed.
std::uint64_t mix(std::uint64_t v) {
  v += 0x9e3779b97f4a7c15ULL;
  v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
  v = (v ^ (v >> 27)) * 0x94d049bb133111ebULL;
  return v ^ (v >> 31);
}
/// @brief Validate finite scalar settings in an inclusive range.
/// @param v Scalar.
/// @param lo Lower bound.
/// @param hi Upper bound.
/// @return Whether valid.
bool range(double v, double lo, double hi) {
  return std::isfinite(v) && v >= lo && v <= hi;
}
/// @brief Validate model dimensions/method before allocating weights.
/// @param m Proposed architecture.
/// @throws std::invalid_argument Invalid architecture.
void model_ok(const DiscreteModel& m) {
  if (int(m.method) < 0 || int(m.method) > 3 || !m.features || m.features > 4096 || m.actions < 2 ||
      m.actions > 1024 || !m.hidden || m.hidden > 1024 || !m.states || m.states > 1000000 ||
      (m.method == DiscreteMethod::QLambda && m.states > 10000000 / m.actions))
    throw std::invalid_argument("invalid discrete RL architecture");
}
/// @brief Validate finite features, binary mask and a valid optional table row.
/// @param o Observation.
/// @param m Architecture.
/// @param terminal Allow empty feasibility only for a true terminal successor.
/// @throws std::invalid_argument Invalid observation.
void observation_ok(const DiscreteObservation& o, const DiscreteModel& m, bool terminal = false) {
  if (o.features.size() != m.features || o.allowed.size() != m.actions ||
      (m.method == DiscreteMethod::QLambda && o.state >= m.states))
    throw std::invalid_argument("invalid RL observation shape/row");
  for (float v : o.features)
    if (!std::isfinite(v))
      throw std::invalid_argument("nonfinite RL observation");
  std::size_t count = 0;
  for (auto a : o.allowed) {
    if (a > 1)
      throw std::invalid_argument("nonbinary RL mask");
    count += a;
  }
  if (!terminal && !count)
    throw std::invalid_argument("empty RL action mask");
}
/// @brief Validate all training controls before invoking callbacks.
/// @param o Proposed options.
/// @throws std::invalid_argument Invalid settings.
void options_ok(const DiscreteTrainingOptions& o) {
  model_ok(o.model);
  if (!o.windows || !std::isfinite(o.window_seconds) || o.window_seconds <= 0 ||
      !range(o.discount_per_second, 1e-12, 1) || !range(o.learning_rate, 1e-12, 1) ||
      !range(o.table_learning_rate, 1e-12, 1) || !range(o.epsilon_start, 0, 1) ||
      !range(o.epsilon_end, 0, 1) || !range(o.trace_decay, 0, 1) || !o.batch ||
      o.replay_capacity < o.batch || o.warmup < o.batch || o.warmup > o.replay_capacity ||
      !o.train_every || !o.target_every || !o.rollout || !o.epochs ||
      !range(o.clip, 1e-8, 1 - 1e-8) || !range(o.gae_lambda, 0, 1) ||
      !range(o.entropy_cost, 0, 100) || !range(o.value_cost, 0, 100) ||
      !range(o.gradient_clip, 1e-12, 1e6) || !o.forced_chain || o.forced_chain > 1024)
    throw std::invalid_argument("invalid discrete RL training options");
}
/// @brief Neural actor/value or Q output.
struct Output {
  n::Tensor scores; ///< Q values or actor logits, [B,A].
  n::Tensor value;  ///< PPO state value, [B], undefined for Q networks.
};
/// @brief Build a complete two-layer native network, or reuse named parameters.
/// @param p Owned parameter store.
/// @param x Float32 [B,features] input.
/// @param m Valid model definition.
/// @return Scores and optional value head, preserving current gradient mode.
Output network(n::ParameterStore& p, const n::Tensor& x, const DiscreteModel& m) {
  auto h = n::mlp(
      p,
      "body",
      x,
      {int(m.hidden), int(m.hidden)},
      m.method == DiscreteMethod::Ppo ? n::Activation::Tanh : n::Activation::Relu);
  if (m.method == DiscreteMethod::Ppo)
    return {n::dense(p, "actor", h, m.actions), n::squeeze(n::dense(p, "critic", h, 1), {1})};
  if (m.method == DiscreteMethod::DuelingDoubleDqn) {
    auto v = n::dense(p, "value", h, 1), a = n::dense(p, "advantage", h, m.actions);
    return {v + a - n::mean(a, {1}, true), {}};
  }
  return {n::dense(p, "q", h, m.actions), {}};
}
/// @brief Validate inference output before exposing it or constructing a distribution.
/// @param values Owned scores.
/// @throws std::runtime_error Nonfinite network values.
void finite(const std::vector<float>& values) {
  for (float x : values)
    if (!std::isfinite(x))
      throw std::runtime_error("nonfinite discrete RL network output");
}
/// @brief Stable host categorical distribution with exactly zero mass on invalid actions.
/// @param scores Finite logits.
/// @param mask Nonempty support.
/// @return Owned probabilities.
std::vector<double> distribution(
    const std::vector<float>&         scores,
    const std::vector<unsigned char>& mask) {
  double              high = scores[detail::argmax(scores, mask)], sum = 0;
  std::vector<double> p(scores.size());
  for (std::size_t a = 0; a < p.size(); ++a)
    if (mask[a])
      sum += p[a] = std::exp(double(scores[a]) - high);
  for (auto& v : p)
    v /= sum;
  return p;
}
/// @brief Owned transition for uniform replay, sparse traces or on-policy rollout updates.
struct Experience {
  DiscreteObservation before;    ///< Actual initial observation.
  DiscreteObservation after;     ///< Actual final observation.
  std::size_t         action{0}; ///< Initial learned choice.
  double              reward{0}; ///< Correctly discounted composition of supplied option rewards.
  double              raw_reward{0};   ///< Original undiscounted rewards, for history only.
  double              seconds{0};      ///< Total elapsed time.
  double              discount{1};     ///< Product of elapsed-time discounts.
  bool                terminal{false}; ///< Zero bootstrap.
  bool                boundary{false}; ///< Cut GAE/traces before reset.
  std::size_t         steps{0};        ///< Underlying environment callbacks.
  float               value{0};        ///< Frozen pre-update value.
  float               next_value{0};   ///< Frozen pre-update successor value.
  float               logp{0};         ///< Frozen behavior log-probability of action.
};
/// @brief Pack a chosen side of an experience minibatch.
/// @param rows Borrowed entries.
/// @param next Select successor rather than initial features.
/// @param features Fixed input width.
/// @return Owned float32 [B,features] tensor.
n::Tensor
observations(const std::vector<const Experience*>& rows, bool next, std::size_t features) {
  std::vector<float> data;
  data.reserve(rows.size() * features);
  for (auto r : rows) {
    const auto& x = next ? r->after.features : r->before.features;
    data.insert(data.end(), x.begin(), x.end());
  }
  return n::from_vector(data, {std::int64_t(rows.size()), std::int64_t(features)});
}
/// @brief Add finite negative logits to invalid actions before log-softmax.
/// @param rows On-policy initial observations.
/// @param actions Output width.
/// @return Float32 [B,actions] zero/-1e9 tensor, avoiding zero-times-infinity entropy.
n::Tensor mask_tensor(const std::vector<const Experience*>& rows, std::size_t actions) {
  std::vector<float> data;
  for (auto r : rows)
    for (auto a : r->before.allowed)
      data.push_back(a ? 0.f : -1e9f);
  return n::from_vector(data, {std::int64_t(rows.size()), std::int64_t(actions)});
}
} // namespace

/// @brief Exclusive inference parameters or table values.
struct DiscretePolicy::Impl {
  DiscreteModel       model;      ///< Validated architecture.
  n::ParameterStore   parameters; ///< Owned neural leaves, empty for QLambda.
  std::vector<double> table;      ///< Owned double table, empty for neural models.

  /// @brief Allocate exactly one architecture with private RNG state.
  /// @param m Validated model.
  explicit Impl(DiscreteModel m) : model(m), parameters(m.seed) {
    if (m.method == DiscreteMethod::QLambda)
      table.resize(m.states * m.actions);
    else {
      n::NoGradGuard guard;
      network(parameters, n::zeros({1, std::int64_t(m.features)}), m);
    }
  }
  /// @brief Evaluate one unmasked observation under NoGradGuard.
  /// @param o Valid observation.
  /// @return Owned scores and optional scalar state value.
  std::pair<std::vector<float>, float> predict(const DiscreteObservation& o) {
    if (model.method == DiscreteMethod::QLambda) {
      std::vector<float> q(model.actions);
      for (std::size_t a = 0; a < q.size(); ++a)
        q[a] = float(table[o.state * model.actions + a]);
      finite(q);
      return {std::move(q), 0};
    }
    n::NoGradGuard guard;
    auto           out =
        network(parameters, n::from_vector(o.features, {1, std::int64_t(model.features)}), model);
    auto scores = out.scores.to_vector<float>();
    finite(scores);
    float v = out.value.defined() ? out.value.item<float>() : 0;
    if (!std::isfinite(v))
      throw std::runtime_error("nonfinite RL critic");
    return {std::move(scores), v};
  }
};
DiscretePolicy::DiscretePolicy(DiscreteModel model) {
  model_ok(model);
  impl_ = std::make_unique<Impl>(model);
}
DiscretePolicy::~DiscretePolicy()                                       = default;
DiscretePolicy::DiscretePolicy(DiscretePolicy&&) noexcept               = default;
DiscretePolicy&    DiscretePolicy::operator=(DiscretePolicy&&) noexcept = default;
std::vector<float> DiscretePolicy::scores(const DiscreteObservation& o) const {
  observation_ok(o, impl_->model);
  return impl_->predict(o).first;
}
std::size_t DiscretePolicy::action(const DiscreteObservation& o) const {
  observation_ok(o, impl_->model);
  if (std::count(o.allowed.begin(), o.allowed.end(), 1) == 1)
    return std::find(o.allowed.begin(), o.allowed.end(), 1) - o.allowed.begin();
  if (impl_->model.method == DiscreteMethod::QLambda) {
    std::size_t best = std::find(o.allowed.begin(), o.allowed.end(), 1) - o.allowed.begin();
    for (std::size_t a = best + 1; a < o.allowed.size(); ++a)
      if (o.allowed[a] && impl_->table[o.state * o.allowed.size() + a] >
                              impl_->table[o.state * o.allowed.size() + best])
        best = a;
    return best;
  }
  return detail::argmax(impl_->predict(o).first, o.allowed);
}
DiscreteModel DiscretePolicy::model() const {
  return impl_->model;
}
DiscretePolicy DiscretePolicy::clone() const {
  DiscretePolicy result(impl_->model);
  result.impl_->table = impl_->table;
  if (impl_->model.method != DiscreteMethod::QLambda)
    result.impl_->parameters.copy_from(impl_->parameters);
  return result;
}
void DiscretePolicy::save(const std::string& path) const {
  auto         m = impl_->model;
  std::int64_t seed;
  std::memcpy(&seed, &m.seed, sizeof(seed));
  n::NamedTensors values{
      {"architecture",
       n::from_vector(
           std::vector<std::int64_t>{
               1,
               int(m.method),
               std::int64_t(m.features),
               std::int64_t(m.actions),
               std::int64_t(m.hidden),
               std::int64_t(m.states),
               seed})}};
  if (m.method == DiscreteMethod::QLambda)
    values.emplace_back(
        "table", n::from_vector(impl_->table, {std::int64_t(m.states), std::int64_t(m.actions)}));
  else
    for (const auto& name : impl_->parameters.names())
      values.emplace_back("parameters/" + name, impl_->parameters.at(name).detach().clone());
  n::save_npz(path, values);
}
DiscretePolicy DiscretePolicy::load(const std::string& path) {
  auto                      values = n::load_npz(path);
  std::vector<std::int64_t> header;
  for (const auto& [key, value] : values)
    if (key == "architecture") {
      if (!header.empty() || value.dtype() != n::DType::Int64 || value.shape() != n::Shape{7})
        throw std::runtime_error("invalid RL architecture tensor");
      header = value.to_vector<std::int64_t>();
    }
  if (header.size() != 7 || header[0] != 1 || header[1] < 0 || header[1] > 3 || header[2] < 1 ||
      header[3] < 2 || header[4] < 1 || header[5] < 1)
    throw std::runtime_error("invalid RL checkpoint header");
  DiscreteModel m;
  m.method   = DiscreteMethod(header[1]);
  m.features = header[2];
  m.actions  = header[3];
  m.hidden   = header[4];
  m.states   = header[5];
  std::memcpy(&m.seed, &header[6], sizeof(m.seed));
  try {
    model_ok(m);
  } catch (const std::invalid_argument&) {
    throw std::runtime_error("invalid RL checkpoint dimensions");
  }
  DiscretePolicy        result(m);
  std::set<std::string> seen;
  for (const auto& [key, value] : values) {
    if (!seen.insert(key).second)
      throw std::runtime_error("duplicate RL checkpoint tensor");
    if (key == "architecture")
      continue;
    if (m.method == DiscreteMethod::QLambda) {
      if (key != "table" || value.dtype() != n::DType::Float64 ||
          value.shape() != n::Shape{std::int64_t(m.states), std::int64_t(m.actions)})
        throw std::runtime_error("invalid RL table tensor");
      result.impl_->table = value.to_vector<double>();
      for (double x : result.impl_->table)
        if (!std::isfinite(x))
          throw std::runtime_error("nonfinite RL table");
    } else {
      if (key.rfind("parameters/", 0) != 0)
        throw std::runtime_error("unknown RL checkpoint tensor");
      auto name = key.substr(11);
      if (!result.impl_->parameters.contains(name) || value.dtype() != n::DType::Float32 ||
          value.shape() != result.impl_->parameters.at(name).shape())
        throw std::runtime_error("RL parameter shape/name/dtype mismatch");
      finite(value.to_vector<float>());
      result.impl_->parameters.set(name, value);
    }
  }
  if (seen.size() !=
      (m.method == DiscreteMethod::QLambda ? 2 : 1 + result.impl_->parameters.names().size()))
    throw std::runtime_error("incomplete RL checkpoint");
  return result;
}

/// @brief Library-owned complete training state, never retained by inference snapshots.
struct DiscreteTrainer {
  DiscreteEnvironment           environment; ///< Borrowed callback owners.
  DiscreteTrainingOptions       options;     ///< Validated fixed controls.
  DiscreteTrainingResult        result;      ///< Final model and growing evidence.
  n::ParameterStore             target;      ///< Independent hard-synchronized target network.
  std::unique_ptr<n::Optimizer> optimizer;   ///< Native Adam and moment state.
  std::mt19937_64               behavior;    ///< Action/exploration draws.
  std::mt19937_64 sampling;        ///< Replay/minibatch draws, independent of reset/action streams.
  std::vector<Experience>  replay; ///< Bounded uniform replay ring.
  std::size_t              replay_next{0}; ///< Next overwritten replay slot after filling.
  std::vector<Experience>  rollout;        ///< Bounded on-policy samples.
  std::size_t              decisions{0};   ///< Cumulative composite decisions.
  std::size_t              updates{0};     ///< Cumulative gradient/TD updates.
  double                   loss{0};        ///< Last finite neural objective.
  std::vector<double>      eligibility;    ///< Table-only replacing state-action traces.
  std::vector<std::size_t> active;         ///< Nonzero sparse trace indices.

  /// @brief Initialize learner state before any environment mutation.
  /// @param e Valid callbacks.
  /// @param o Validated options.
  DiscreteTrainer(DiscreteEnvironment e, DiscreteTrainingOptions o)
      : environment(std::move(e)),
        options(std::move(o)),
        target(options.model.seed + 1),
        behavior(options.seed),
        sampling(mix(options.seed)) {
    result.policy = DiscretePolicy(options.model);
    if (options.model.method == DiscreteMethod::QLambda)
      eligibility.resize(options.model.states * options.model.actions);
    else {
      auto& p   = *result.policy.impl_;
      optimizer = std::make_unique<n::Optimizer>(
          p.parameters, options.learning_rate, n::OptimizerConfig{n::OptimizerKind::Adam});
      if (options.model.method != DiscreteMethod::Ppo) {
        n::NoGradGuard guard;
        network(target, n::zeros({1, std::int64_t(options.model.features)}), options.model);
        target.copy_from(p.parameters);
      }
    }
  }
  /// @brief Evaluate current epsilon-greedy behavior, respecting even a singleton mask.
  /// @param observation Valid current state.
  /// @param epsilon Exploration probability.
  /// @return Allowed action.
  std::size_t epsilon_action(const DiscreteObservation& observation, double epsilon) {
    std::vector<std::size_t> legal;
    for (std::size_t a = 0; a < observation.allowed.size(); ++a)
      if (observation.allowed[a])
        legal.push_back(a);
    if (legal.size() == 1)
      return legal[0];
    if (std::uniform_real_distribution<double>(0, 1)(behavior) < epsilon)
      return legal[std::uniform_int_distribution<std::size_t>(0, legal.size() - 1)(behavior)];
    return result.policy.action(observation);
  }
  /// @brief Execute the learned choice and any bounded trailing choices with singleton support.
  /// @param before Current observation.
  /// @param action Allowed initial choice.
  /// @param remaining Minimum remaining time before the window boundary.
  /// @return Composed SMDP transition with original step/time accounting.
  Experience advance(const DiscreteObservation& before, std::size_t action, double remaining) {
    Experience r;
    r.before      = before;
    r.action      = action;
    std::size_t a = action;
    for (std::size_t k = 0; k < options.forced_chain; ++k) {
      auto step = environment.step(a);
      if (!std::isfinite(step.reward) || !std::isfinite(step.seconds) || step.seconds <= 0)
        throw std::invalid_argument("invalid RL transition reward/time");
      observation_ok(step.next, options.model, step.terminated);
      r.reward += r.discount * step.reward;
      r.raw_reward += step.reward;
      r.seconds += step.seconds;
      r.discount *= std::pow(options.discount_per_second, step.seconds);
      ++r.steps;
      r.after    = std::move(step.next);
      r.terminal = step.terminated;
      r.boundary = step.terminated || step.truncated || r.seconds >= remaining;
      if (r.boundary || std::count(r.after.allowed.begin(), r.after.allowed.end(), 1) != 1)
        break;
      a = std::find(r.after.allowed.begin(), r.after.allowed.end(), 1) - r.after.allowed.begin();
    }
    return r;
  }
  /// @brief Apply checked, globally clipped native Adam gradients.
  /// @param objective Scalar finite objective.
  /// @throws std::runtime_error Nonfinite objective/gradient norm.
  void minimize(const n::Tensor& objective) {
    loss = objective.item<float>();
    if (!std::isfinite(loss))
      throw std::runtime_error("nonfinite RL loss");
    auto grads   = n::grad(objective, optimizer->parameters());
    auto clipped = n::clip_by_global_norm(grads, options.gradient_clip);
    if (!std::isfinite(clipped.second))
      throw std::runtime_error("nonfinite RL gradients");
    optimizer->step(clipped.first);
    ++updates;
  }
  /// @brief Replay one uniform Double-DQN minibatch with exact stored successor masks/discounts.
  void dqn_update() {
    std::vector<const Experience*> rows;
    for (std::size_t i = 0; i < options.batch; ++i)
      rows.push_back(
          &replay[std::uniform_int_distribution<std::size_t>(0, replay.size() - 1)(sampling)]);
    std::vector<float>        targets;
    std::vector<std::int64_t> actions;
    {
      n::NoGradGuard guard;
      auto           x = observations(rows, true, options.model.features);
      auto           online =
               network(result.policy.impl_->parameters, x, options.model).scores.to_vector<float>(),
           frozen = network(target, x, options.model).scores.to_vector<float>();
      finite(online);
      finite(frozen);
      for (std::size_t i = 0; i < rows.size(); ++i) {
        auto               r = rows[i];
        std::vector<float> a(
            online.begin() + i * options.model.actions,
            online.begin() + (i + 1) * options.model.actions),
            b(frozen.begin() + i * options.model.actions,
              frozen.begin() + (i + 1) * options.model.actions);
        targets.push_back(
            float(detail::double_target(
                r->reward, r->discount, r->terminal, a, b, r->after.allowed)));
        actions.push_back(r->action);
      }
    }
    auto q = network(
                 result.policy.impl_->parameters,
                 observations(rows, false, options.model.features),
                 options.model)
                 .scores;
    auto chosen = n::sum(q * n::one_hot(n::from_vector(actions), options.model.actions), {1});
    minimize(n::mean(n::huber(n::from_vector(targets) - chosen)));
    if (updates % options.target_every == 0)
      target.copy_from(result.policy.impl_->parameters);
  }
  /// @brief Optimize a complete on-policy rollout, including its final partial minibatch.
  void ppo_update() {
    if (rollout.empty())
      return;
    std::vector<detail::ValueSample> samples;
    for (const auto& r : rollout)
      samples.push_back({r.reward, r.discount, r.value, r.next_value, r.boundary});
    auto               adv = detail::gae(samples, options.gae_lambda);
    std::vector<float> returns(adv.size());
    double mean = std::accumulate(adv.begin(), adv.end(), 0.) / adv.size(), variance = 0;
    for (std::size_t i = 0; i < adv.size(); ++i) {
      returns[i] = adv[i] + rollout[i].value;
      variance += (adv[i] - mean) * (adv[i] - mean);
    }
    double sd = std::sqrt(variance / adv.size() + 1e-8);
    if (adv.size() > 1 && variance > 1e-8)
      for (auto& a : adv)
        a = float((a - mean) / sd);
    std::vector<std::size_t> order(rollout.size());
    std::iota(order.begin(), order.end(), 0);
    for (std::size_t epoch = 0; epoch < options.epochs; ++epoch) {
      std::shuffle(order.begin(), order.end(), sampling);
      for (std::size_t start = 0; start < order.size(); start += options.batch) {
        std::vector<const Experience*> rows;
        std::vector<float>             a, ret, old;
        std::vector<std::int64_t>      act;
        for (std::size_t j = start; j < std::min(start + options.batch, order.size()); ++j) {
          auto i = order[j];
          rows.push_back(&rollout[i]);
          a.push_back(adv[i]);
          ret.push_back(returns[i]);
          old.push_back(rollout[i].logp);
          act.push_back(rollout[i].action);
        }
        auto out = network(
            result.policy.impl_->parameters,
            observations(rows, false, options.model.features),
            options.model);
        auto logp     = n::log_softmax(out.scores + mask_tensor(rows, options.model.actions));
        auto selected = n::sum(logp * n::one_hot(n::from_vector(act), options.model.actions), {1});
        auto entropy  = -n::mean(n::sum(n::exp(logp) * logp, {1}));
        minimize(
            detail::ppo_loss(selected, n::from_vector(old), n::from_vector(a), options.clip) +
            options.value_cost * n::mean(n::square(n::from_vector(ret) - out.value)) -
            options.entropy_cost * entropy);
      }
    }
    rollout.clear();
  }
  /// @brief Run independent windows with shared time budgets and no reset RNG coupling.
  /// @return Owned completed policies, snapshots and statistics.
  DiscreteTrainingResult run() {
    for (std::size_t window = 0; window < options.windows; ++window) {
      auto observation = environment.reset(mix(options.seed + window));
      observation_ok(observation, options.model);
      DiscreteWindow stats;
      auto           before_updates = updates;
      int            carried        = -1;
      for (auto i : active)
        eligibility[i] = 0;
      active.clear();
      double epsilon = options.epsilon_start + (options.epsilon_end - options.epsilon_start) *
                                                   double(window) /
                                                   std::max<std::size_t>(1, options.windows - 1);
      while (stats.seconds < options.window_seconds) {
        std::size_t a;
        float       value = 0, logp = 0;
        if (options.model.method == DiscreteMethod::Ppo) {
          auto   prediction = result.policy.impl_->predict(observation);
          auto   p          = distribution(prediction.first, observation.allowed);
          double u = std::uniform_real_distribution<double>(0, 1)(behavior), sum = 0;
          a = detail::argmax(prediction.first, observation.allowed);
          for (std::size_t j = 0; j < p.size(); ++j) {
            sum += p[j];
            if (u < sum) {
              a = j;
              break;
            }
          }
          value = prediction.second;
          logp  = float(std::log(std::max(1e-30, p[a])));
        } else
          a = carried < 0 ? epsilon_action(observation, epsilon) : std::size_t(carried);
        auto r = advance(observation, a, options.window_seconds - stats.seconds);
        stats.seconds += r.seconds;
        stats.reward += r.raw_reward;
        stats.environment_steps += r.steps;
        ++stats.decisions;
        ++decisions;
        if (options.model.method == DiscreteMethod::QLambda) {
          auto&  table = result.policy.impl_->table;
          double next  = 0;
          bool   keep  = false;
          carried      = -1;
          if (!r.terminal) {
            auto best = result.policy.action(r.after);
            next      = table[r.after.state * options.model.actions + best];
            if (!r.boundary) {
              carried = int(epsilon_action(r.after, epsilon));
              keep    = table[r.after.state * options.model.actions + carried] >= next;
            }
          }
          const auto current = observation.state * options.model.actions + a;
          double     delta =
              options.table_learning_rate * (r.reward + r.discount * next - table[current]);
          if (eligibility[current] == 0)
            active.push_back(current);
          eligibility[current] = 1;
          for (auto it = active.begin(); it != active.end();) {
            auto i = *it;
            table[i] += delta * eligibility[i];
            if (!std::isfinite(table[i]))
              throw std::runtime_error("nonfinite RL Q table");
            eligibility[i] *= keep ? r.discount * options.trace_decay : 0;
            if (eligibility[i] < 1e-8) {
              eligibility[i] = 0;
              *it            = active.back();
              active.pop_back();
            } else
              ++it;
          }
          ++updates;
        } else if (options.model.method == DiscreteMethod::Ppo) {
          r.value      = value;
          r.logp       = logp;
          r.next_value = r.terminal ? 0 : result.policy.impl_->predict(r.after).second;
          rollout.push_back(r);
          if (rollout.size() >= options.rollout || r.boundary)
            ppo_update();
        } else {
          if (replay.size() < options.replay_capacity)
            replay.push_back(r);
          else {
            replay[replay_next] = r;
            replay_next         = (replay_next + 1) % options.replay_capacity;
          }
          if (replay.size() >= options.warmup && decisions % options.train_every == 0)
            dqn_update();
        }
        observation = std::move(r.after);
        if (r.boundary)
          break;
      }
      stats.updates = updates - before_updates;
      stats.loss    = loss;
      result.windows.push_back(stats);
      auto completed = window + 1;
      if ((completed >= 128 && (completed & (completed - 1)) == 0) ||
          completed == options.windows) {
        result.checkpoints.push_back(result.policy.clone());
        result.checkpoint_windows.push_back(completed);
      }
      if (options.progress)
        options.progress(completed, stats, result.policy);
      if (options.stop && options.stop(completed, stats, result.policy)) {
        if (result.checkpoint_windows.empty() || result.checkpoint_windows.back() != completed) {
          result.checkpoints.push_back(result.policy.clone());
          result.checkpoint_windows.push_back(completed);
        }
        break;
      }
    }
    return std::move(result);
  }
};
DiscreteTrainingResult train_discrete(
    DiscreteEnvironment     environment,
    DiscreteTrainingOptions options) {
  options_ok(options);
  if (!environment.reset || !environment.step)
    throw std::invalid_argument("missing RL environment callbacks");
  return DiscreteTrainer(std::move(environment), std::move(options)).run();
}
} // namespace datamunge::rl
