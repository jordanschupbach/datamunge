// Port of tflow's common/ml/autorec.h + common/nn/autorec.h: small MovieLens warm-user
// U-AutoRec (256 ratings -> 32 sigmoid -> 256 linear) with an observed-only masked MSE
// (missing ratings are never zero targets) plus .00005 * (||W_enc||^2 + ||W_dec||^2).
// Prediction feeds train-only ratings; no target/mask is required.
// Parameter names match tflow's ("encoder/W", ...), so its initial_arrays.npz loads directly.
#pragma once

#include <fstream>
#include <map>

#include "common/autoencoder.hpp"

namespace ae {

/// sum(mask * (prediction - target)^2) / sum(mask).
inline nn::Tensor ObservedRatingMse(const nn::Tensor& prediction, const nn::Tensor& target, const nn::Tensor& mask) {
  return nn::sum(mask * nn::square(prediction - target)) / nn::sum(mask);
}

struct AutoRecOutput {
  nn::Tensor code, prediction, observed_mse, regularizer, loss;
};

/// With mask undefined only code/prediction are computed (input-only inference).
inline AutoRecOutput AutoRec(nn::ParameterStore& p, const nn::Tensor& input, const nn::Tensor& mask = {}) {
  AutoRecOutput o;
  o.code       = nn::dense(p, "encoder", input, 32, nn::Activation::Sigmoid);
  o.prediction = nn::dense(p, "decoder", o.code, 256);
  if (mask.defined()) {
    o.observed_mse = ObservedRatingMse(o.prediction, input, mask);
    o.regularizer  = .00005 * (nn::sum(nn::square(p.at("encoder/W"))) + nn::sum(nn::square(p.at("decoder/W"))));
    o.loss         = o.observed_mse + o.regularizer;
  }
  return o;
}

struct AutoRecData {
  nn::Tensor input, ratings, train, validation, test, users, items;  // float [199, 256] x5; int32 [199], [256]
};

/// ratings.npz with every tflow integrity check against the original u.data.
inline AutoRecData LoadAutoRec(const std::filesystem::path& root) {
  const auto file = root / "ratings.npz";
  check::Require(std::filesystem::exists(file), file.string() + " missing; run `just autorec-assets` in tflow_examples or pass data=");
  const auto  a = nn::load_npz(file.string());
  AutoRecData d{Find(a, "input").contiguous(),           Find(a, "ratings").contiguous(), Find(a, "train_mask").contiguous(),
                Find(a, "validation_mask").contiguous(), Find(a, "test_mask").contiguous(), Find(a, "user_ids").contiguous(),
                Find(a, "item_ids").contiguous()};
  for (const auto& t : {d.input, d.ratings, d.train, d.validation, d.test})
    check::Require(t.dtype() == nn::DType::Float32 && t.shape() == nn::Shape({199, 256}), "exact199users256itemsratingsschema");
  check::Require(d.users.dtype() == nn::DType::Int32 && d.users.shape() == nn::Shape({199}) && d.items.dtype() == nn::DType::Int32 &&
                     d.items.shape() == nn::Shape({256}),
                 "originaluser/itemidentityvectors");
  std::ifstream                      raw(root / "u.data");
  std::map<std::pair<int, int>, int> source;
  int                                user, item, rating, timestamp, total = 0, window[257] = {};
  while (raw >> user >> item >> rating >> timestamp) {
    check::Require(user >= 1 && user <= 943 && item >= 1 && item <= 1682 && rating >= 1 && rating <= 5 &&
                       source.emplace(std::make_pair(user, item), rating).second,
                   "all100Kuniquevalidoriginalratings");
    ++total;
    if (user <= 256 && item <= 256) ++window[user];
  }
  check::Require(total == 100000, "all100000originalMovieLensratings");
  std::vector<int> expected;
  for (int u = 1; u <= 256; ++u)
    if (window[u] >= 10) expected.push_back(u);
  check::Require(expected.size() == 199, "deterministicuserwindowfilter");
  const auto users = d.users.to_vector<int>(), items = d.items.to_vector<int>();
  const auto in = d.input.to_vector<float>(), ratings = d.ratings.to_vector<float>(), tr = d.train.to_vector<float>(),
             va = d.validation.to_vector<float>(), te = d.test.to_vector<float>();
  int train = 0, val = 0, test = 0, item_fit[256] = {};
  for (int u = 0; u < 199; ++u) {
    check::Require(users[u] == expected[u], "originaluserIDsortedwindow");
    int observed = 0, v = 0, t = 0, f = 0;
    for (int i = 0; i < 256; ++i) {
      check::Require(items[i] == i + 1, "originalitemIDs1..256");
      auto      found = source.find({expected[u], i + 1});
      const int value = found == source.end() ? 0 : found->second;
      const int k     = u * 256 + i;
      check::Near(ratings[k], value, 0, "everyselectedratingagainstoriginalsource");
      const float a1 = tr[k], b1 = va[k], c1 = te[k];
      check::Require((a1 == 0 || a1 == 1) && (b1 == 0 || b1 == 1) && (c1 == 0 || c1 == 1) && a1 + b1 + c1 == (value ? 1 : 0),
                     "disjointobserved-onlymasks,unknownneverzero-negatives");
      check::Near(in[k], a1 ? float(value) / 5.f : 0, 0, "traininginputEXCLUDESallheldout/unknownratings");
      observed += value != 0;
      f += int(a1);
      v += int(b1);
      t += int(c1);
      item_fit[i] += int(a1);
    }
    check::Require(v == std::max(1, observed / 10) && t == v && f > 0, "peruserholdoutcountsandwarmtraining");
    train += f;
    val += v;
    test += t;
  }
  for (int c : item_fit) check::Require(c > 0, "everyitemhastrainingobservations");
  check::Require(train == 8079 && val == 898 && test == 898, "exactsourcepinnedtraining/validation/testcounts");
  return d;
}

class AutoRecModel {
 public:
  nn::ParameterStore             parameters{4603};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit AutoRecModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      AutoRec(parameters, nn::zeros({1, 256}));
    }
    if (training) optimizer = MakeAdam(parameters);
  }
  static void CheckFeeds(const nn::Tensor& x, const nn::Tensor& m) {
    check::Require(x.shape() == m.shape(), "alignedtrainingratingsandobservedmask");
    double count = 0;
    for (float v : m.contiguous().to_vector<float>()) count += v;
    check::Require(count > 0, "nonemptyobserved-onlytrainingbatch");
  }
  /// {code, prediction}: inference uses ONLY training observations, no held-out ratings/mask.
  std::vector<nn::Tensor> Predict(const nn::Tensor& x) {
    nn::NoGradGuard         guard;
    auto                    o   = AutoRec(parameters, x);
    std::vector<nn::Tensor> out{o.code.contiguous(), o.prediction.contiguous()};
    for (const auto& t : out)
      for (float v : t.to_vector<float>()) check::Require(std::isfinite(v), "finiteAutoRecpredictions");
    return out;
  }
  std::vector<nn::Tensor> Gradient(const nn::Tensor& x, const nn::Tensor& m) {
    CheckFeeds(x, m);
    return nn::grad(AutoRec(parameters, x, m).loss, parameters.parameters());
  }
  /// {observed MSE, regularizer, loss}
  std::vector<nn::Tensor> Objective(const nn::Tensor& x, const nn::Tensor& m) {
    CheckFeeds(x, m);
    nn::NoGradGuard guard;
    auto            o = AutoRec(parameters, x, m);
    return {o.observed_mse, o.regularizer, o.loss};
  }
  void Update(const nn::Tensor& x, const nn::Tensor& m) {
    CheckFeeds(x, m);
    optimizer->minimize(AutoRec(parameters, x, m).loss);
  }
};

/// {RMSE, MAE} in star units over mask entries. Float32 predictions are normalized (x5),
/// float64 ones (baselines) are already stars; clip clamps to 1..5.
inline std::vector<double> RatingErrors(const nn::Tensor& prediction, const nn::Tensor& ratings, const nn::Tensor& mask,
                                        bool clip = false) {
  const bool normalized = prediction.dtype() == nn::DType::Float32;
  const auto p = nn::cast(prediction, nn::DType::Float64).contiguous().to_vector<double>();
  const auto r = ratings.contiguous().to_vector<float>(), m = mask.contiguous().to_vector<float>();
  double     sum = 0, absolute = 0, count = 0;
  for (std::size_t i = 0; i < m.size(); ++i)
    if (m[i]) {
      double value = normalized ? p[i] * 5 : p[i];
      if (clip) value = std::clamp(value, 1., 5.);
      const double delta = value - r[i];
      sum += delta * delta;
      absolute += std::abs(delta);
      ++count;
    }
  check::Require(count > 0, "nonemptyheldoutobservedratings");
  return {std::sqrt(sum / count), absolute / count};
}

}  // namespace ae
