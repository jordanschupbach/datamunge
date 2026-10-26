// CMD: just run-nn consistency_model
// Consistency distillation with an analytic Gaussian score teacher; three cheap epochs.
//
// Port of tflow's consistency_model (common/{nn,ml}/consistency_model.h ->
// examples/nn/common/consistency_model.hpp). Explicit toy adaptation of
// Consistency Models (Song et al. 2023): known 2-D Gaussian, mean (.3, -.2),
// variance (.49, .25), 512 fit / 128 validation / 128 test points (no dataset
// needed). The exact Gaussian VE score gives the teacher drift t (x - mu) /
// (variance + t^2); one backwards Euler step between 16 fixed linear levels
// .01..2. Online 1474-parameter SiLU MLP with input preconditioning and eight
// log-time features; f(x, t) = cskip(t) x + cout(t) F(x, t). Loss: MSE between
// the online map at the noisy higher state and the STOPPED EMA target at the
// Euler lower state. Three epochs / 48 Adam(.003) updates in tflow's
// BatchOrder(512, 32, 19019), then a separate post-Adam EMA target = .9 target +
// .1 online. One-, two- and four-step sampling from N(0, 4 I) and from the exact
// p_T.
//
// Checks (as tflow): census; target hard-synced at start; Adam never changes the
// target; the stop-gradient leaves the target without gradients; restore
// bit-exact; boundary f(x, .01) = x and Jacobian = I bit-exactly for EMA and
// online; fixed-seed sampling repeats bit-exactly; innovation control; weights
// immutable. TF parity: init=auto replays TF's initial_arrays.npz (online; the
// target is its hard copy) and compares the records with TF's; TF's final
// online/target arrays must reproduce TF's test outputs. Deviations: final.npz
// (online + "target/..." arrays) instead of a SaveV2 bundle; no
// independent_audit.json. Options: epochs=3 out=DIR init=auto|tf|nn|PATH tf=DIR
// threads=4. Output: output/nn/models/consistency_model.
#include "common/consistency_model.hpp"

namespace {

namespace nn = datamunge::nn;
namespace fs = std::filesystem;
using nn::Tensor;
using namespace diffusion;

Tensor Gaussian2(std::int64_t n, std::uint32_t seed, bool wide = false) {
  const Tensor raw = nn::reshape(ae::GaussianNoise({n, 2}, 1, seed), {n, 2});
  return wide ? raw : nn::cast(raw, nn::DType::Float32);
}

Tensor Time(std::int64_t n, float value) { return nn::full({n}, value, nn::DType::Float32); }

std::pair<Tensor, Tensor> Times(std::int64_t n, std::uint32_t seed, bool cyclic = false) {
  std::vector<float> t(static_cast<std::size_t>(n)), p(static_cast<std::size_t>(n));
  std::mt19937       rng(seed);
  for (std::int64_t i = 0; i < n; ++i) {
    const int index = cyclic ? int(i % 15) : int(15 * (double(rng()) + .5) / 4294967296.);
    p[i]            = float(.01 + 1.99 * index / 15);
    t[i]            = float(.01 + 1.99 * (index + 1) / 15);
  }
  return {nn::from_vector(t, {n}), nn::from_vector(p, {n})};
}

Map Sample(ConsistencyModel& model, int steps, bool exact_initial = false, std::uint32_t seed = 19201, int innovation_offset = 0) {
  const int    n    = 1024;
  Map          result;
  const Tensor base = Gaussian2(n, seed);
  const auto   bv   = base.to_vector<float>();
  std::vector<float> sv(bv.size());
  for (std::size_t i = 0; i < sv.size(); ++i)
    sv[i] = exact_initial ? float((i % 2 ? -.2 : .3) + std::sqrt(4 + (i % 2 ? .25 : .49)) * bv[i]) : float(2 * bv[i]);
  Tensor state     = nn::from_vector(sv, {n, 2});
  result["base"]   = base;
  result["initial"] = state;
  std::vector<Tensor> path{state};
  std::vector<float>  levels(steps);
  for (int k = 0; k < steps; ++k) {
    const float t = k == 0 ? 2.f : steps == 2 ? 1.f : k == 1 ? 1.5f : k == 2 ? 1.f : .5f;
    levels[k]     = t;
    if (k) {
      const Tensor z = Gaussian2(n, seed + k + innovation_offset);
      result["innovation" + std::to_string(k)] = Gaussian2(n, seed + k + innovation_offset, true);
      const auto         s = state.to_vector<float>(), zv = z.to_vector<float>();
      std::vector<float> next(s.size());
      const float        amount = float(std::sqrt(double(t) * t - .0001));
      for (std::size_t i = 0; i < s.size(); ++i) next[i] = s[i] + amount * zv[i];
      state = nn::from_vector(next, {n, 2});
    }
    result["input" + std::to_string(k)] = state;
    const auto out = model.Predict(state, Time(n, t));
    Finite(out);
    for (const auto& [key, value] : out) result["step" + std::to_string(k) + "_" + key] = value;
    state = out.at("value");
    path.push_back(state);
  }
  result["samples"] = state;
  result["path"]    = nn::stack(path, 0);
  result["levels"]  = nn::from_vector(levels, {steps});
  return result;
}

/// Blue samples and orange held-out points on axes [-3, 3], one 384x384 panel per sampler.
void Plot(const fs::path& file, const Tensor& heldout, const std::vector<Tensor>& samples) {
  const std::int64_t W = std::int64_t(samples.size()) * 384;
  std::vector<float> rgb(static_cast<std::size_t>(384 * W * 3), 245.f);
  auto set = [&](std::int64_t y, std::int64_t x, const std::array<int, 3>& c) {
    for (int k = 0; k < 3; ++k) rgb[std::size_t((y * W + x) * 3 + k)] = float(c[k]);
  };
  for (std::size_t p = 0; p < samples.size(); ++p) {
    for (int j = 0; j < 384; ++j) set(192, std::int64_t(p) * 384 + j, {190, 190, 190}), set(j, std::int64_t(p) * 384 + 192, {190, 190, 190});
    auto draw = [&](const Tensor& x, int radius, std::array<int, 3> color) {
      const auto v = x.to_vector<float>();
      for (std::size_t b = 0; b < v.size() / 2; ++b) {
        const int px = int(std::floor((double(v[2 * b]) + 3) * 64)), py = int(std::floor((3 - double(v[2 * b + 1])) * 64));
        if (px < 0 || px >= 384 || py < 0 || py >= 384) continue;
        for (int dy = -radius; dy <= radius; ++dy)
          for (int dx = -radius; dx <= radius; ++dx)
            if (px + dx >= 0 && px + dx < 384 && py + dy >= 0 && py + dy < 384) set(py + dy, std::int64_t(p) * 384 + px + dx, color);
      }
    };
    draw(samples[p], 1, {30, 90, 190});
    draw(heldout, 2, {230, 100, 20});
  }
  nn::save_png(nn::from_vector(rgb, {384, W, 3}), file.string(), 0, 255);
}

double Scalar(const Map& m, const std::string& k) { return m.at(k).item<float>(); }

/// Online arrays plus "target/<name>" arrays (our final.npz).
void SaveBoth(const fs::path& file, const ConsistencyModel& m) {
  Map all = m.Arrays();
  for (const auto& [k, v] : m.Arrays(true)) all["target/" + k] = v;
  Save(file, all);
}
void LoadBoth(const Map& online, const Map& target, ConsistencyModel& m) {
  for (const auto& [k, v] : online) m.online.set(k, v);
  for (const auto& [k, v] : target) m.target.set(k, v);
}
void LoadBoth(const fs::path& file, ConsistencyModel& m) {
  Map online, target;
  for (const auto& [k, v] : Load(file)) (k.rfind("target/", 0) == 0 ? target[k.substr(7)] : online[k]) = v;
  check::Require(online.size() == 6 && target.size() == 6, "final.npz holds online and target arrays");
  LoadBoth(online, target, m);
}

void Run(const Options& o) {
  const fs::path root = o.out, tf = o.tf;
  Map            data;
  std::vector<Tensor> pools;
  for (int k = 0; k < 3; ++k) {
    const int         n     = k ? 128 : 512;
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    const Tensor      base  = Gaussian2(n, 19001 + 2 * k, true);
    const auto        bv    = base.to_vector<double>();
    std::vector<float>        x(bv.size());
    std::vector<std::int32_t> ids(n);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = float((i % 2 ? -.2 : .3) + (i % 2 ? .5 : .7) * bv[i]);
    for (int i = 0; i < n; ++i) ids[i] = k * 10000 + i;
    data[split + "_base"]       = base;
    data[split + "_points"]     = nn::from_vector(x, {n, 2});
    data[split + "_source_ids"] = Ints(ids);
    pools.push_back(data[split + "_points"]);
  }
  const Tensor vn = Gaussian2(128, 19101), tn = Gaussian2(128, 19103);
  const auto [vt, vp] = Times(128, 0, true);
  const auto [tt, tp] = Times(128, 0, true);
  data["validation_noise"]    = vn;
  data["test_noise"]          = tn;
  data["validation_time"]     = vt;
  data["validation_previous"] = vp;
  data["test_time"]           = tt;
  data["test_previous"]       = tp;
  Save(root / "data.npz", data);
  if (fs::exists(tf / "data.npz")) {
    const auto theirs = Load(tf / "data.npz");
    for (const auto& [k, v] : data) check::Require(Bits(theirs.at(k)) == Bits(v), "toy Gaussian data bit-identical to TF's data.npz: " + k);
    std::cout << "data: all " << data.size() << " arrays bit-identical to TF's data.npz\n";
  }
  ConsistencyModel model;
  check::Require(model.online.count() == 1474 && model.target.count() == 1474, "1474 online plus 1474 EMA");
  const fs::path init      = bpix::InitFile(o.init, tf);
  const bool     replaying = !init.empty();
  if (replaying) {
    replay::LoadInto(model.online, init);
    model.HardSync();
    std::cout << "replaying TF's initial arrays " << init.string() << "\n";
  }
  check::Require(model.Snapshot() == model.Snapshot(true), "initial target hard synced");
  Save(root / "initial_arrays.npz", model.Arrays());
  Save(root / "initial_target_arrays.npz", model.Arrays(true));
  Save(root / "initial_outputs.npz", model.Evaluate(pools[2], tn, tt, tp));
  nn::BatchOrder order(512, 32, 19019);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tvalidation_consistency_mse\n";
  batches << "epoch\tstep\trow\n";
  Map randomness, ema_trace;
  int updates = 0;
  for (int epoch = 1; epoch <= o.epochs; ++epoch) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const auto   n = std::int64_t(rows.size());
      const Tensor x = Rows(pools[0], rows), e = Gaussian2(n, 1900051 + updates);
      const auto [t, p]   = Times(n, 2900021 + updates);
      const std::string s = "step" + std::to_string(updates);
      randomness[s + "_noise"]    = Gaussian2(n, 1900051 + updates, true);
      randomness[s + "_time"]     = t;
      randomness[s + "_previous"] = p;
      if (!updates) {
        Save(root / "first_batch.npz", {{"points", x}, {"noise", e}, {"time", t}, {"previous", p}});
        Save(root / "first_forward.npz", model.Evaluate(x, e, t, p));
        Save(root / "initial_gradients.npz", model.Gradient(x, e, t, p));
        check::Require(model.TargetGradientsAbsent(x, e, t, p), "ALL target parameters have NO gradient from stop_gradient");
      }
      const auto target_before = model.Snapshot(true);
      model.Update(x, e, t, p);
      check::Require(target_before == model.Snapshot(true), "Adam does not change stopped target");
      if (!updates) {
        Save(root / "first_after.npz", model.Arrays());
        Save(root / "first_target_before_ema.npz", model.Arrays(true));
      }
      model.Ema();
      for (const auto& [key, value] : model.Arrays()) ema_trace[s + "_online_" + key] = value;
      for (const auto& [key, value] : model.Arrays(true)) ema_trace[s + "_target_" + key] = value;
      if (!updates) Save(root / "first_target_after_ema.npz", model.Arrays(true));
      for (auto row : rows) batches << epoch << '\t' << updates << '\t' << row << '\n';
      ++updates;
    }
    const auto eval = model.Evaluate(pools[1], vn, vt, vp);
    Finite(eval);
    history << epoch << '\t' << Scalar(eval, "loss") << '\n';
    Save(root / ("epoch" + std::to_string(epoch) + "_arrays.npz"), model.Arrays());
    Save(root / ("epoch" + std::to_string(epoch) + "_target_arrays.npz"), model.Arrays(true));
    Save(root / ("epoch" + std::to_string(epoch) + "_validation.npz"), eval);
    std::cout << "epoch" << epoch << " consistencyMSE=" << Scalar(eval, "loss") << std::endl;
  }
  history.flush();
  Save(root / "training_randomness.npz", randomness);
  Save(root / "ema_trace.npz", ema_trace);
  const auto batch = Load(root / "first_batch.npz");
  Save(root / "final_gradients.npz", model.Gradient(batch.at("points"), batch.at("noise"), batch.at("time"), batch.at("previous")));
  SaveBoth(root / "final.npz", model);
  ConsistencyModel deployed(false);
  LoadBoth(root / "final.npz", deployed);
  check::Require(model.Snapshot() == deployed.Snapshot() && model.Snapshot(true) == deployed.Snapshot(true), "all 2948 weights bitexact restore");
  const auto online_before = deployed.Snapshot(), target_before = deployed.Snapshot(true);
  Save(root / "model_arrays.npz", deployed.Arrays());
  Save(root / "target_arrays.npz", deployed.Arrays(true));
  const auto output = deployed.Evaluate(pools[2], tn, tt, tp);
  Finite(output);
  Save(root / "outputs.npz", output);
  const auto probe = deployed.Predict(output.at("noised"), tt);
  Save(root / "probe.npz", probe);
  const auto boundary = deployed.Predict(pools[2], Time(128, .01f));
  check::Require(Bits(boundary.at("value")) == Bits(pools[2]), "boundary identity bitexact for EMA");
  check::Require(Bits(boundary.at("online")) == Bits(pools[2]), "boundary identity bitexact for online");
  const auto j0 = boundary.at("jacobian_row0").to_vector<float>(), j1 = boundary.at("jacobian_row1").to_vector<float>();
  for (int b = 0; b < 128; ++b)
    for (int c = 0; c < 2; ++c) {
      check::Require(j0[b * 2 + c] == (c == 0 ? 1.f : 0.f), "boundary identity Jacobian row0");
      check::Require(j1[b * 2 + c] == (c == 1 ? 1.f : 0.f), "boundary identity Jacobian row1");
    }
  Save(root / "boundary.npz", boundary);
  std::vector<Tensor> draws, exact_draws;
  std::map<int, Map>  sample_maps;
  for (int steps : {1, 2, 4}) {
    const auto sample = Sample(deployed, steps), again = Sample(deployed, steps), exact = Sample(deployed, steps, true);
    for (const auto& [key, value] : sample) check::Require(Bits(value) == Bits(again.at(key)), "fixed seed consistency sampling bitexact");
    Save(root / ("samples" + std::to_string(steps) + ".npz"), sample);
    Save(root / ("exact_prior_samples" + std::to_string(steps) + ".npz"), exact);
    draws.push_back(sample.at("samples"));
    exact_draws.push_back(exact.at("samples"));
    sample_maps[steps] = sample;
    if (steps > 1) {
      const auto changed = Sample(deployed, steps, false, 19201, 1000);
      check::Require(Bits(sample.at("initial")) == Bits(changed.at("initial")), "innovation control keeps initial state fixed");
      check::Require(Bits(sample.at("samples")) != Bits(changed.at("samples")), "changed innovations change few-step samples");
      Save(root / ("innovation_control" + std::to_string(steps) + ".npz"), changed);
    }
  }
  Plot(root / "samples.png", pools[2], draws);
  Plot(root / "exact_prior_samples.png", pools[2], exact_draws);
  double              oracle_error = 0, identity_error = 0, teacher_error = 0;
  std::vector<double> oracle(256), teacher_exact(256);
  const auto ttv = tt.to_vector<float>(), tpv = tp.to_vector<float>(), noised = output.at("noised").to_vector<float>(),
             pv = probe.at("value").to_vector<float>(), teacher = output.at("teacher").to_vector<float>();
  for (int b = 0; b < 128; ++b)
    for (int c = 0; c < 2; ++c) {
      const int    i = b * 2 + c;
      const double m = c ? -.2 : .3, v = c ? .25 : .49, t = ttv[b], p = tpv[b], state = noised[i];
      const double truth = m + std::sqrt((v + .0001) / (v + t * t)) * (state - m);
      oracle[i]          = truth;
      teacher_exact[i]   = m + std::sqrt((v + p * p) / (v + t * t)) * (state - m);
      oracle_error += std::pow(pv[i] - truth, 2) / 256;
      identity_error += std::pow(state - truth, 2) / 256;
      teacher_error = std::max(teacher_error, std::abs(double(teacher[i]) - teacher_exact[i]));
    }
  Save(root / "oracle.npz", {{"boundary_endpoint", nn::from_vector(oracle, {128, 2})}, {"adjacent_exact", nn::from_vector(teacher_exact, {128, 2})}});
  check::Require(online_before == deployed.Snapshot() && target_before == deployed.Snapshot(true), "all consistency inference weights immutable");
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"online_parameters\":1474,\"target_parameters\":1474,\"epochs\":" << o.epochs << ",\"optimizer_updates\":" << updates
       << ",\"ema_new_online_fraction\":0.1,\"levels\":16,\"epsilon\":0.01,\"maximum_time\":2,\"test_consistency_mse\":"
       << Scalar(output, "loss") << ",\"test_oracle_endpoint_mse\":" << oracle_error << ",\"identity_oracle_endpoint_mse\":" << identity_error
       << ",\"euler_adjacent_exact_max_error\":" << teacher_error << ",\"samples_per_sampler\":1024,\"replayed_tf_initial_arrays\":"
       << (replaying ? "true" : "false") << "}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Consistency model</title><style>body{font:16px system-ui;margin:30px;"
          "max-width:1250px}img{max-width:100%}</style><h1>Consistency distillation with an analytic Gaussian teacher</h1><p>Explicit "
          "toy adaptation of <a href='https://arxiv.org/abs/2303.01469'>Consistency Models</a>: known 2-D Gaussian, exact VE-score "
          "Euler teacher between 16 levels .01..2, 1474-parameter online map, stopped EMA target. Datamunge port of tflow's "
          "consistency_model.</p><h2>Approximate noise prior: one, two, four evaluations</h2><p>Blue generated, orange 128 "
          "held-out, axes [-3, 3].</p><img src='samples.png'><h2>Exact p_T initial control</h2><img src='exact_prior_samples.png'>"
          "<p>Small consistency loss alone does not certify a correct transport; see the oracle endpoint error in metrics.json.</p>";
  check::Require(bool(meta) && bool(html) && bool(history) && bool(batches), "complete consistency records");
  std::cout << "testConsistencyMSE=" << Scalar(output, "loss") << " EMAoracleEndpointMSE=" << oracle_error << " identity=" << identity_error
            << " EulerAdjacentError=" << teacher_error << ";boundary/AdamTargetFrozen/EMA/restore/repeat/immutable PASS\n";

  if (fs::exists(tf / "model_arrays.npz")) {
    std::cout << "TF parity (" << tf.string() << "):\n";
    if (replaying) {
      const double fwd = bpix::Compare(tf, "first_forward.npz", Load(root / "first_forward.npz"));
      const double grd = bpix::Compare(tf, "initial_gradients.npz", Load(root / "initial_gradients.npz"));
      const double aft = bpix::Compare(tf, "first_after.npz", Load(root / "first_after.npz"));
      const double ema = bpix::Compare(tf, "first_target_after_ema.npz", Load(root / "first_target_after_ema.npz"));
      check::Require(fwd < 1e-5 && grd < 1e-5 && aft < 1e-5 && ema < 1e-5, "replayed first forward/gradients/update/EMA match TF");
      bpix::Compare(tf, "model_arrays.npz", deployed.Arrays());
      bpix::Compare(tf, "target_arrays.npz", deployed.Arrays(true));
      const auto ours = replay::History(root / "history.tsv"), theirs = replay::History(tf / "history.tsv");
      for (std::size_t i = 0; i < std::min(ours.size(), theirs.size()); ++i)
        std::printf("  epoch %zu validation consistency MSE ours %.6g TF %.6g\n", i + 1, ours[i], theirs[i]);
    }
    ConsistencyModel tf_model(false);
    LoadBoth(Load(tf / "model_arrays.npz"), Load(tf / "target_arrays.npz"), tf_model);
    const auto   tf_out = tf_model.Evaluate(pools[2], tn, tt, tp);
    const double d1     = bpix::Compare(tf, "outputs.npz", tf_out);
    const double d2     = bpix::Compare(tf, "probe.npz", tf_model.Predict(tf_out.at("noised"), tt));
    const double d3     = bpix::Compare(tf, "samples4.npz", Sample(tf_model, 4));
    check::Require(d1 < 1e-5 && d2 < 1e-4 && d3 < 1e-3, "TF's final weights reproduce TF's test outputs, probe and samples");
  }
  std::cout << "Artifacts: " << root.string() << "\nAll consistency_model checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto o = diffusion::ParseOptions(argc, argv, "consistency_model", {}, "epochs=N out=PATH init=auto|tf|nn|PATH tf=DIR threads=N");
    Run(o);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
