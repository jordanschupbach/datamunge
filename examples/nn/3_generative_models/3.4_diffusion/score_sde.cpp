// CMD: just run-nn score_sde
// VE reverse SDE versus probability-flow ODE on a frozen NCSN, with Gaussian
// oracle convergence.
//
// Port of tflow's score_sde (common/{nn,ml}/score_sde.h ->
// examples/nn/common/score_sde.hpp). No training: the 5504 frozen weights of
// noise_conditional_score are restored. ncsn=auto (default) uses our port's run
// (output/nn/models/noise_conditional_score/final.npz) when present, else tflow's
// recorded run (output/models/noise_conditional_score/model_arrays.npz);
// ncsn=ours|tf|DIR selects explicitly. Continuous sigma(t) = .05 20^t, t 1 -> 0,
// g^2 = 2 log 20 sigma^2, network level 15 (1 - t) (continuous sinusoidal
// conditioning interpolates the 16 trained levels). Reverse Euler-Maruyama, 64
// steps: x + h g^2 score + sqrt(h g^2) z; probability-flow ODE drift .5 g^2 score
// with RK4 8/16/32 steps. Initial N(0, 1) is an approximate terminal prior;
// optional final Tweedie correction saved separately. Gaussian oracle score
// (.3 - x) / (.49 + sigma^2) runs through the same integrators (8/16/32/64 steps).
//
// Checks (as tflow): restore bit-exact vs the frozen run's model_arrays.npz;
// continuous score matches the discrete NCSN at all 16 grid levels (< 3e-5); ODE
// ignores ALL innovations bit-exactly; SDE innovations change samples; oracle
// probability flow reproduces the analytic scale (< 8e-6 at 64 steps); weights
// immutable. TF parity: every record is compared with tflow's
// output/models/score_sde (exact when ncsn=tf). Deviations: no
// independent_audit.json. Options: ncsn=auto|ours|tf|DIR out=DIR tf=DIR
// threads=4. Output: output/nn/models/score_sde.
#include "common/score_sde.hpp"

namespace {

namespace nn = datamunge::nn;
namespace fs = std::filesystem;
using nn::Tensor;
using namespace diffusion;
using scoresde::ScoreSdeModel;

/// tflow's Png: clamp [-1, 1], floor((v + 1) .5 255 + .5).
void Png(const fs::path& path, const Tensor& images, int columns) {
  const auto         n = images.size(0);
  const auto         v = images.contiguous().to_vector<float>();
  const std::int64_t H = (n + columns - 1) / columns * 64, W = columns * 64;
  std::vector<float> out(static_cast<std::size_t>(H * W * 3), 220.f);
  for (std::int64_t b = 0; b < n; ++b)
    for (int y = 0; y < 64; ++y)
      for (int x = 0; x < 64; ++x) {
        const double value = std::clamp(double(v[std::size_t(b * 64 + (y / 8) * 8 + x / 8)]), -1., 1.);
        for (int c = 0; c < 3; ++c) out[std::size_t(((b / columns * 64 + y) * W + b % columns * 64 + x) * 3 + c)] = float(std::floor((value + 1) * .5 * 255 + .5));
      }
  bpix::PngBytes(path, out, H, W);
}

Map Sample(const ScoreSdeModel& model, int steps, bool ode, std::int64_t n = 32, std::uint32_t initial_seed = 13201,
           std::uint32_t innovation_seed = 13202, bool oracle = false) {
  Tensor initial = Gaussian8(n, initial_seed);
  if (oracle) {
    auto v = initial.to_vector<float>();
    for (auto& x : v) x = float(.3 + std::sqrt(double(.49f) + 1) * x);
    initial = nn::from_vector(v, {n, 8, 8, 1});
  }
  const float         h = 1.f / steps;
  Tensor              current = initial;
  std::vector<Tensor> trajectory{initial}, innovations;
  std::vector<float>  times;
  std::vector<double> variance_curve{double(.49f) + 1};
  StepStack           saved;
  for (int step = 0; step < steps; ++step) {
    const float t = 1.f - step * h;
    times.push_back(t);
    const Tensor z = Gaussian8(n, innovation_seed + step);
    innovations.push_back(Gaussian8(n, innovation_seed + step, true));
    const auto out = model.Evaluate(current, t, h, z, ode ? 1 : 0);
    Finite(out);
    saved.add(out);
    current = out.at(ode ? "ode_next" : "em_next");
    trajectory.push_back(current);
    if (oracle) {
      const double a  = out.at(ode ? "ode_input_gradient" : "em_input_gradient").contiguous().to_vector<float>()[0];
      const double g2 = out.at("g_squared").contiguous().to_vector<float>()[0];
      variance_curve.push_back(a * a * variance_curve.back() + (ode ? 0 : double(h) * g2));
    }
  }
  const auto final = model.Evaluate(current, 0, 0, nn::zeros({n, 8, 8, 1}), 0);
  auto       tw = current.to_vector<float>();
  const auto fsc = final.at("score").contiguous().to_vector<float>();
  for (std::size_t i = 0; i < tw.size(); ++i) tw[i] = tw[i] + (.05f * .05f) * fsc[i];
  Map result = saved.stacked();
  result["initial"]        = initial;
  result["images"]         = current;
  result["tweedie_images"] = nn::from_vector(tw, current.shape());
  result["final_score"]    = final.at("score");
  result["trajectory"]     = nn::stack(trajectory, 0);
  result["innovations"]    = nn::stack(innovations, 0);
  result["physical_time"]  = nn::from_vector(times, {std::int64_t(steps)});
  result["h"]              = nn::full({}, h);
  if (oracle) result["oracle_variance_recurrence"] = nn::from_vector(variance_curve, {std::int64_t(steps) + 1});
  return result;
}

struct Args {
  std::string ncsn = "auto";
  fs::path    out, tf = replay::TfModels("score_sde");
  int         threads = 4;
};

void Run(const Args& o) {
  const fs::path root = o.out, tf = o.tf;
  const fs::path trained = FrozenDir(o.ncsn, paths::OutputRoot() / "models" / "noise_conditional_score",
                                     replay::TfModels("noise_conditional_score"));
  const fs::path frozen  = FrozenFile(trained);
  check::Require(!frozen.empty() && fs::exists(trained / "model_arrays.npz") && fs::exists(trained / "data.npz"),
                 "score SDE requires an NCSN run: run noise_conditional_score first, or ncsn=tf|DIR");
  std::cout << "frozen NCSN weights: " << frozen.string() << "\n";
  fs::create_directories(root);
  ScoreSdeModel model;
  replay::LoadInto(model.store, frozen);
  const auto expected = Load(trained / "model_arrays.npz");
  const auto arrays   = model.Arrays();
  check::Require(model.store.count() == 5504 && arrays.size() == 6, "frozen score SDE 5504 weights");
  for (const auto& [key, value] : arrays) check::Require(Bits(value) == Bits(expected.at(key)), "score SDE restore BITEXACT matches NCSN");
  Save(root / "model_arrays.npz", arrays);
  if (fs::exists(trained / "metrics.json")) fs::copy_file(trained / "metrics.json", root / "training_metrics.json", fs::copy_options::overwrite_existing);
  const auto before    = model.Snapshot();
  const auto ncsn_data = Load(trained / "data.npz");
  const Tensor input = Gaussian8(4, 13401), zero = nn::zeros({4, 8, 8, 1});
  ncsn::NoiseConditionalScoreModel discrete(false);
  replay::LoadInto(discrete.store, frozen);
  Map    grid{{"input", input}};
  double grid_error = 0;
  for (int i = 0; i < 16; ++i) {
    const float t          = 1.f - i / 15.f;
    const auto  old        = discrete.Reverse(input, FullInt(4, i), zero);
    const auto  continuous = model.Evaluate(input, t, 0, zero, 0);
    grid["discrete_score" + std::to_string(i)]     = old.at("score");
    grid["continuous_score" + std::to_string(i)]   = continuous.at("score");
    grid["continuous_epsilon" + std::to_string(i)] = continuous.at("epsilon");
    grid_error = std::max(grid_error, MaxAbs(old.at("score"), continuous.at("score")));
  }
  check::Require(grid_error < 3e-5, "continuous score conditioning matches ALL 16 training grid levels");
  Save(root / "conditioning_grid.npz", grid);

  std::vector<std::string> names;
  std::vector<std::vector<Tensor>> comparison(8, std::vector<Tensor>(4));
  double noisy_effect = 0;
  for (int config = 0; config < 4; ++config) {
    const bool        ode   = config > 0;
    const int         steps = ode ? (config == 1 ? 8 : config == 2 ? 16 : 32) : 64;
    const std::string name  = ode ? "ode_rk4_" + std::to_string(steps) : "reverse_sde_em64";
    const auto        sample  = Sample(model, steps, ode);
    const auto        changed = Sample(model, steps, ode, 32, 13201, 14202);
    if (ode) {
      check::Require(Bits(sample.at("trajectory")) == Bits(changed.at("trajectory")), "probability flow ODE ignores ALL innovations BITEXACT");
    } else {
      noisy_effect = MaxAbs(sample.at("images"), changed.at("images"));
      check::Require(noisy_effect > .01, "reverse SDE innovation changes actual samples");
      Save(root / "sde_changed_innovations.npz", changed);
    }
    Save(root / (name + ".npz"), sample);
    Png(root / (name + ".png"), sample.at("images"), 8);
    if (!ode) Png(root / "sde_tweedie.png", sample.at("tweedie_images"), 8);
    std::vector<std::vector<Tensor>> cells(8);
    for (int b = 0; b < 8; ++b) {
      comparison[b][config] = Row(sample.at("images"), b);
      for (int j = 0; j < 5; ++j) cells[b].push_back(Row(At(sample.at("trajectory"), j * steps / 4), b));
    }
    const Tensor gallery = Gallery(cells);
    Save(root / (name + "_gallery.npz"), {{"images", gallery}});
    Png(root / (name + "_trajectory.png"), gallery, 5);
    names.push_back(name);
    std::cout << name << ":integratorScoreEvaluations=" << (ode ? steps * 4 : steps) << std::endl;
  }
  const Tensor comparison_images = Gallery(comparison);
  Save(root / "comparison.npz", {{"images", comparison_images}});
  Png(root / "comparison.png", comparison_images, 4);

  ScoreSdeModel oracle(true);
  std::ofstream convergence(root / "oracle_convergence.tsv");
  convergence.precision(17);
  convergence << "steps\tode_pointwise_max_error\tem_expected_variance\tode_expected_variance\ttarget_variance\n";
  const double variance = double(.49f), target_variance = variance + .05 * .05;
  double       oracle_error64 = 0;
  for (int steps : {8, 16, 32, 64}) {
    const auto em = Sample(oracle, steps, false, 4, 13301, 13302, true), ode = Sample(oracle, steps, true, 4, 13301, 13302, true);
    Save(root / ("oracle_em" + std::to_string(steps) + ".npz"), em);
    Save(root / ("oracle_ode" + std::to_string(steps) + ".npz"), ode);
    const double scale = std::sqrt(target_variance / (variance + 1));
    double       error = 0;
    const auto   init = ode.at("initial").to_vector<float>(), img = ode.at("images").to_vector<float>();
    for (int j = 0; j < 4 * 64; ++j) error = std::max(error, std::abs(.3 + scale * (double(init[j]) - .3) - img[j]));
    const auto ev = em.at("oracle_variance_recurrence").to_vector<double>(), ov = ode.at("oracle_variance_recurrence").to_vector<double>();
    convergence << steps << '\t' << error << '\t' << ev[steps] << '\t' << ov[steps] << '\t' << target_variance << '\n';
    std::printf("oracle %2d steps: ODE max error %.3g, EM variance %.6f, ODE variance %.6f (target %.6f)\n", steps, error, ev[steps],
                ov[steps], target_variance);
    if (steps == 64) {
      oracle_error64 = error;
      check::Require(error < 8e-6, "Gaussian oracle probability flow analytic scale");
    }
  }
  double     prior_bound = 0;
  const auto fit         = ncsn_data.at("fit_images").to_vector<float>();
  for (float x : fit) prior_bound += .5 * std::pow(double(x), 2) / 1000;
  check::Require(before == model.Snapshot(), "ALL score SDE/ODE controls leave 6 weights immutable");
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters_frozen\":5504,\"parameter_tensors\":6,\"additional_training_updates\":0,\"sigma_min\":0.05,\"sigma_max\":1,"
          "\"sde_steps\":64,\"ode_steps\":[8,16,32],\"samples_each\":32,\"continuous_training_grid_score_max_error\":"
       << grid_error << ",\"sde_changed_innovation_max_effect\":" << noisy_effect
       << ",\"ode_innovation_canary_bitexact\":true,\"fit_average_conditional_terminal_prior_kl_bound_per_image\":" << prior_bound
       << ",\"frozen_ncsn\":\"" << frozen.string() << "\"}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Score-based SDEs</title><style>body{font:16px system-ui;margin:30px;"
          "max-width:1100px}img{max-width:100%;image-rendering:pixelated}</style><h1>VE reverse SDE and probability-flow ODE</h1><p><a "
          "href='https://arxiv.org/abs/2011.13456'>Score-based SDEs</a>, frozen three-epoch NCSN 5504 weights / 6 tensors, NO additional "
          "training. Continuous sigma(t)=.05*20^t, t 0 to 1, g&sup2;=2 log 20 sigma&sup2;, forward drift 0. Network level 15*(1-t); "
          "continuous sinusoidal conditioning interpolates the 16 trained levels. All 16 grid outputs checked against the discrete "
          "frozen model. Datamunge port of tflow's score_sde.</p><p>Reverse Euler-Maruyama from t 1 to 0: "
          "x_next=x+h*g&sup2;*score+sqrt(h*g&sup2;)*z, 64 steps. Probability-flow ODE drift .5g&sup2;*score, RK4 steps 8/16/32. ODE "
          "ignores ALL innovations bit-exactly; SDE changes with them. Initial standard Gaussian is an APPROXIMATE terminal prior. "
          "Gaussian oracle score=(.3-x)/(.49+sigma&sup2;) runs through the same integrators.</p><h2>Paired first 8 initial latents</h2>"
          "<p>Columns reverse SDE 64, ODE-RK4 steps 8, 16, 32. Raw states clipped ONLY for display.</p><img src='comparison.png'>";
  for (const auto& name : names)
    html << "<h2>" << name << "</h2><img src='" << name << ".png'><p>First 8 trajectories at physical times 1, .75, .5, .25, 0.</p>"
         << "<img src='" << name << "_trajectory.png'>";
  html << "<h2>SDE samples after optional final Tweedie correction</h2><img src='sde_tweedie.png'><p>See metrics.json, "
          "oracle_convergence.tsv. Weak inherited score and prior mismatch limit generation.</p>";
  check::Require(bool(meta) && bool(html) && bool(convergence), "complete score SDE records");
  std::cout << "gridError=" << grid_error << " sdeInnovationEffect=" << noisy_effect << " oracleODE64Error=" << oracle_error64
            << " priorKLBound=" << prior_bound
            << "\ncontinuousGrid/reverseSDE/probabilityRK4/oracleConvergence/noiseControls/frozenRestore/immutable PASS\n";

  // ---- TF parity ----
  if (!fs::exists(tf / "metrics.json")) {
    std::cout << "no TF record at " << tf.string() << "; parity checks skipped\nArtifacts: " << root.string() << "\n";
    return;
  }
  std::cout << "TF parity (" << tf.string() << "):\n";
  double oracle_diff = 0;
  for (int steps : {8, 16, 32, 64})
    for (const std::string kind : {"em", "ode"}) {
      const std::string f = "oracle_" + kind + std::to_string(steps) + ".npz";
      oracle_diff         = std::max(oracle_diff, bpix::Compare(tf, f, Load(root / f)));
    }
  check::Require(oracle_diff < 1e-4, "Gaussian oracle integrators match TF's records");
  const bool tf_weights = bpix::Compare(tf, "model_arrays.npz", arrays) == 0;
  double     learned    = bpix::Compare(tf, "conditioning_grid.npz", grid);
  for (const auto& name : names) learned = std::max(learned, bpix::Compare(tf, name + ".npz", Load(root / (name + ".npz"))));
  if (tf_weights) check::Require(learned < 1e-3, "TF's frozen NCSN reproduces TF's grid, SDE and ODE records");
  std::cout << (tf_weights ? "frozen weights are TF's: learned records checked (< 1e-3)\n"
                           : "frozen weights are our NCSN port's (replayed TF run, not bit-identical): learned records compared only\n")
            << "Artifacts: " << root.string() << "\nAll score_sde checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args o;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      const auto        e = a.find('=');
      if (e == std::string::npos) throw std::invalid_argument("ncsn=auto|ours|tf|PATH out=PATH tf=DIR threads=N");
      const std::string k = a.substr(0, e), v = a.substr(e + 1);
      if (k == "ncsn") o.ncsn = v;
      else if (k == "out") o.out = v;
      else if (k == "tf") o.tf = v;
      else if (k == "threads") o.threads = std::stoi(v);
      else throw std::invalid_argument("ncsn=auto|ours|tf|PATH out=PATH tf=DIR threads=N");
    }
    if (o.out.empty()) o.out = paths::OutputRoot() / "models" / "score_sde";
    if (o.threads > 0) nn::set_num_threads(o.threads);
    Run(o);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
