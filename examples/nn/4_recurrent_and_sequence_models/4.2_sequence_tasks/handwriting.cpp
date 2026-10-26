// CMD: just run-nn handwriting
// Text-conditioned stroke synthesis with Graves monotonic windows and mixture density.
//
// Port of tflow's handwriting. Data: tflow's output/datasets/handwriting
// (data.npz: short real pen trajectories of 3-symbol prompts resampled to 33
// offset steps (dx, dy, pen-up), input = previous offset, target = next one;
// 140 fit / 40 validation / 40 test; ATTRIBUTION.txt).
// An 8-unit LSTM reads the previous offset and the previous soft window over
// the prompt; two Gaussian window components with monotonic centres pick the
// characters, and a 3-component bivariate Gaussian mixture plus a Bernoulli
// pen give the next-offset likelihood (1680 parameters). Adam .005, batch 8,
// three epochs (54 updates) on the mean per-step NLL.
// Checks: a mixture-density fixture (loss and its raw/target derivatives,
// compared with TF's density_fixture.npz when present); when tflow's recorded
// run is present, a replay from TF's initial weights (initial outputs with the
// stroke/text partials, first-batch gradients, history, final weights, and the
// seeded sample from the replayed weights); then the own-initialisation run
// audits restore, the streaming one-step cell against the teacher-forced unroll
// (all 33 steps), seeded repeat sampling and immutability. The path figure
// (paths.png) is not reproduced; samples are saved as .npz like TF's.
// Options: epochs=3 data=PATH tf=PATH.
// Output: output/nn/models/handwriting/.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

#include "common/checks.hpp"
#include "common/handwriting.hpp"
#include "common/paths.hpp"
#include "common/tf_replay.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;
using replay::Map;
using namespace handwriting;

namespace {

struct Model {
  nn::ParameterStore p;
  explicit Model(std::uint64_t seed) : p(seed) {
    nn::NoGradGuard guard;
    Handwriting(p, nn::zeros({1, kSteps, 3}), Text(nn::zeros({1, 3}, nn::DType::Int32)));
    check::Require(p.count() == 1680 && p.size() == 7, "1680 parameters seven tensors");
  }
  static Tensor Text(const Tensor& labels) { return nn::one_hot(labels, kSymbols); }
  Map Evaluate(const Map& b) {
    Tensor x = b.at("input").detach().clone(), text = Text(b.at("labels"));
    x.requires_grad_(true);
    text.requires_grad_(true);
    const Path   path = Handwriting(p, x, text);
    const Tensor loss = HandwritingNll(path.raw, b.at("target"));
    const auto   g    = nn::grad(loss, {x, text});
    return {{"raw", path.raw.detach()},     {"loss", loss.detach()},     {"h", path.h.detach()},
            {"c", path.c.detach()},         {"kappa", path.kappa.detach()}, {"window", path.window.detach()},
            {"phi", path.phi.detach()},     {"input_gradient", g[0]},    {"text_gradient", g[1]}};
  }
  std::vector<Tensor> Gradients(const Map& b) {
    return nn::grad(HandwritingNll(Handwriting(p, b.at("input"), Text(b.at("labels"))).raw, b.at("target")), p.parameters());
  }
  /// One streaming step from owned state: {raw, h, c, kappa, window, phi}.
  std::vector<Tensor> StepOnce(const Tensor& stroke, const Tensor& labels, const Tensor& h, const Tensor& c,
                               const Tensor& k, const Tensor& w) {
    nn::NoGradGuard guard;
    const auto      s = HandwritingCell(p, stroke, Text(labels), {h, c}, k, w);
    return {s.raw, s.state.h, s.state.c, s.kappa, s.window, s.phi};
  }
  /// Samples up to 66 offsets for the prompt (first, second, end) with tflow's
  /// mt19937 uniforms: component by inverse CDF, Box-Muller Gaussians, Bernoulli pen.
  Map Sample(int first, int second, std::uint32_t seed) {
    Tensor stroke = nn::zeros({1, 3}), h = nn::zeros({1, 8}), c = nn::zeros({1, 8}), k = nn::zeros({1, 2}),
           w = nn::zeros({1, kSymbols});
    const Tensor label = nn::from_vector(std::vector<std::int32_t>{first, second, 20}, {1, 3});
    const std::vector<std::pair<std::string, int>> fields{{"raw", 19}, {"h", 8}, {"c", 8}, {"kappa", 2}, {"window", 21}, {"phi", 3}};
    std::map<std::string, std::vector<float>> a;
    for (const auto& [name, width] : fields) a[name].assign(66 * width, 0.f);
    a["strokes"].assign(66 * 3, 0.f);
    std::vector<double> uniform(66 * 4, 0.0);
    std::mt19937        rng(seed);
    int                 used = 0;
    for (int t = 0; t < 66; ++t) {
      const auto out = StepOnce(stroke, label, h, c, k, w);
      h = out[1], c = out[2], k = out[3], w = out[4];
      for (std::size_t f = 0; f < fields.size(); ++f) {
        const auto v = out[f].to_vector<float>();
        std::copy(v.begin(), v.end(), a[fields[f].first].begin() + t * fields[f].second);
      }
      double u[4];
      for (int j = 0; j < 4; ++j) uniform[t * 4 + j] = u[j] = (double(rng()) + .5) / 4294967296.;
      const auto r       = out[0].to_vector<float>();
      double     maximum = std::max({r[0], r[1], r[2]}), mass = 0, cumulative = 0;
      for (int j = 0; j < 3; ++j) mass += std::exp(r[j] - maximum);
      int component = 2;
      for (int j = 0; j < 3; ++j) {
        cumulative += std::exp(r[j] - maximum) / mass;
        if (u[0] < cumulative) {
          component = j;
          break;
        }
      }
      const double radius = std::sqrt(-2 * std::log(u[1])), z1 = radius * std::cos(6.283185307179586 * u[2]),
                   z2 = radius * std::sin(6.283185307179586 * u[2]);
      const double sx = std::exp(r[9 + component]) + .001, sy = std::exp(r[12 + component]) + .001,
                   rho = .95 * std::tanh(r[15 + component]);
      const float s0 = static_cast<float>(r[3 + component] + sx * z1);
      const float s1 = static_cast<float>(r[6 + component] + sy * (rho * z1 + std::sqrt(1 - rho * rho) * z2));
      const float s2 = u[3] < 1 / (1 + std::exp(-double(r[18])));
      stroke         = nn::from_vector(std::vector<float>{s0, s1, s2}, {1, 3});
      const float sampled[3] = {s0, s1, s2};
      for (int j = 0; j < 3; ++j) {
        a["strokes"][t * 3 + j] = sampled[j];
        check::Require(std::isfinite(sampled[j]), "finite sampled handwriting");
      }
      used           = t + 1;
      const auto phi = out[5].to_vector<float>();
      if (s2 == 1 && phi[2] > phi[0] && phi[2] > phi[1]) break;
    }
    Map result;
    for (const auto& [name, width] : fields) result[name] = nn::from_vector(a[name], {66, width});
    result["strokes"] = nn::from_vector(a["strokes"], {66, 3});
    result["uniform"] = nn::from_vector(uniform, {66, 4});
    result["prompt"]  = label;
    result["length"]  = nn::from_vector(std::vector<std::int32_t>{used}, {});
    return result;
  }
};

std::vector<double> Train(Model& m, const Map& fit, const Map& validation, int epochs, std::ofstream* batches,
                          const std::filesystem::path* root) {
  nn::Optimizer       opt(m.p, .005, replay::Adam());
  nn::BatchOrder      order(140, 8, 45007);
  std::vector<double> history;
  int                 updates = 0;
  for (int e = 1; e <= epochs; ++e) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const auto b = replay::Rows(fit, rows);
      if (!updates && root) {
        nn::save_npz((*root / "first_batch.npz").string(), replay::ToNamed(b));
        nn::save_npz((*root / "initial_gradients.npz").string(), replay::ToNamed(replay::Named(m.p, m.Gradients(b))));
      }
      opt.step(m.Gradients(b));
      if (!updates && root) m.p.save((*root / "first_after.npz").string());
      if (batches)
        for (auto r : rows) *batches << e << '\t' << updates << '\t' << r << '\n';
      ++updates;
    }
    nn::NoGradGuard guard;
    const double    nll = HandwritingNll(Handwriting(m.p, validation.at("input"), Model::Text(validation.at("labels"))).raw,
                                         validation.at("target")).item<double>();
    check::Require(std::isfinite(nll), "finite NLL");
    history.push_back(nll);
  }
  return history;
}

void DensityFixture(const std::filesystem::path& root, const std::filesystem::path& tf) {
  std::vector<float> rv(4 * 19);
  for (std::size_t i = 0; i < rv.size(); ++i) rv[i] = static_cast<float>(.7 * std::sin(i * .43));
  Tensor raw    = nn::from_vector(rv, {4, 19});
  Tensor target = nn::from_vector(std::vector<float>{.2f, -.8f, 0.f, 1.f, .3f, 1.f, -.5f, .7f, 0.f, 2.f, -1.f, 1.f}, {4, 3});
  raw.requires_grad_(true);
  target.requires_grad_(true);
  const Tensor loss = HandwritingNll(raw, target);
  const auto   g    = nn::grad(loss, {raw, target});
  // Reference: direct double-precision mixture density of each row.
  double nll = 0;
  for (int n = 0; n < 4; ++n) {
    const float* r = rv.data() + n * 19;
    const double y0 = target.to_vector<float>()[n * 3], y1 = target.to_vector<float>()[n * 3 + 1], pen = target.to_vector<float>()[n * 3 + 2];
    double       zsum = 0, m = std::max({r[0], r[1], r[2]});
    for (int j = 0; j < 3; ++j) zsum += std::exp(r[j] - m);
    double density = 0;
    for (int j = 0; j < 3; ++j) {
      const double pi = std::exp(r[j] - m) / zsum, sx = std::exp(r[9 + j]) + .001, sy = std::exp(r[12 + j]) + .001,
                   rho = .95 * std::tanh(r[15 + j]), dx = (y0 - r[3 + j]) / sx, dy = (y1 - r[6 + j]) / sy;
      const double z = dx * dx + dy * dy - 2 * rho * dx * dy;
      density += pi * std::exp(-z / (2 * (1 - rho * rho))) / (2 * M_PI * sx * sy * std::sqrt(1 - rho * rho));
    }
    nll += (std::log1p(std::exp(r[18])) - pen * r[18]) - std::log(density);
  }
  check::Near(loss.item<double>(), nll / 4, 1e-5, "mixture NLL matches a direct bivariate-Gaussian evaluation");
  nn::save_npz((root / "density_fixture.npz").string(), {{"raw", raw.detach()}, {"target", target.detach()}, {"loss", loss.detach()},
                                                         {"raw_gradient", g[0]}, {"target_gradient", g[1]}});
  std::cout << " MDN fixture: loss matches direct density";
  if (std::filesystem::exists(tf / "density_fixture.npz")) {
    const auto   ref = nn::load_npz((tf / "density_fixture.npz").string());
    const double d   = std::max({replay::MaxAbsDiff(loss, replay::Get(ref, "loss")), replay::MaxAbsDiff(g[0], replay::Get(ref, "raw_gradient")),
                                 replay::MaxAbsDiff(g[1], replay::Get(ref, "target_gradient"))});
    check::Near(d, 0, 2e-6, "MDN fixture loss/raw/target derivatives match TF");
    std::cout << "; vs TF loss/derivatives max diff=" << d;
  }
  std::cout << "\n";
}

void TfParity(const std::filesystem::path& tf, const Map& fit, const Map& validation, const Map& test, int epochs) {
  if (!std::filesystem::exists(tf / "initial_arrays.npz")) {
    std::cout << " (tflow run not found at " << tf << "; TF parity skipped)\n";
    return;
  }
  Model m(0);
  replay::LoadInto(m.p, tf / "initial_arrays.npz");
  const double out = replay::MaxDiff(m.Evaluate(test), tf / "initial_outputs.npz");
  const auto   fb  = replay::ToMap(nn::load_npz((tf / "first_batch.npz").string()));
  const double gd  = replay::MaxDiff(replay::Named(m.p, m.Gradients(fb)), tf / "initial_gradients.npz");
  check::Near(out, 0, 2e-5, "initial outputs and stroke/text partials match TF");
  check::Near(gd, 0, 2e-5, "first-batch parameter gradients match TF");
  const auto hist = Train(m, fit, validation, epochs, nullptr, nullptr);
  const auto tfh  = replay::History(tf / "history.tsv");
  double     worst = 0;
  for (std::size_t e = 0; e < hist.size() && e < tfh.size(); ++e) {
    std::cout << "  replay epoch=" << e + 1 << " validationNLL=" << hist[e] << " (TF " << tfh[e] << ")\n";
    worst = std::max(worst, std::abs(hist[e] - tfh[e]));
  }
  std::cout << " TF parity from TF's initial weights: initial outputs max diff=" << out << ", first-batch gradients max diff=" << gd;
  if (epochs == 3) {
    const double w = replay::MaxDiff(replay::Arrays(m.p), tf / "model_arrays.npz");
    std::cout << ", final weights max diff=" << w;
    check::Near(worst, 0, 1e-3, "validation NLL history tracks TF's recorded run");
    check::Near(w, 0, 1e-3, "final weights track TF's after 54 Adam updates");
    if (std::filesystem::exists(tf / "sample_ab.npz")) {
      const auto ours = m.Sample(0, 1, 45011);
      const auto ref  = replay::ToMap(nn::load_npz((tf / "sample_ab.npz").string()));
      const int  n = ours.at("length").item<int>(), tn = ref.at("length").item<int>();
      std::cout << "; seeded sample 'ab' from the replayed weights: length " << n << " (TF " << tn
                << "), strokes max diff=" << replay::MaxAbsDiff(ours.at("strokes"), ref.at("strokes"));
    }
  }
  std::cout << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto            start = std::chrono::steady_clock::now();
    int                   epochs = 3;
    std::filesystem::path data = paths::Dataset("handwriting"), tf = replay::TfModels("handwriting");
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("epochs=", 0) == 0) epochs = std::stoi(a.substr(7));
      else if (a.rfind("data=", 0) == 0) data = a.substr(5);
      else if (a.rfind("tf=", 0) == 0) tf = a.substr(3);
      else throw std::invalid_argument("epochs=N data=PATH tf=PATH");
    }
    check::Require(epochs > 0 && epochs <= 100, "positive epochs");
    const auto root = paths::OutputDir("models/handwriting");
    replay::CopyAssets(data, root, {"data.npz", "manifest.json", "ATTRIBUTION.txt"});
    DensityFixture(root, tf);
    const auto all = nn::load_npz((data / "data.npz").string());
    const std::vector<std::string> keys{"input", "target", "labels", "source_ids"};
    const Map fit = replay::Split(all, "fit", keys), validation = replay::Split(all, "validation", keys),
              test = replay::Split(all, "test", keys);
    TfParity(tf, fit, validation, test, epochs);

    Model m(45003);
    m.p.save((root / "initial_arrays.npz").string());
    nn::save_npz((root / "initial_outputs.npz").string(), replay::ToNamed(m.Evaluate(test)));
    std::ofstream history(root / "history.tsv"), batches(root / "batches.tsv");
    history << "epoch\tvalidation_nll\n";
    batches << "epoch\tstep\trow\n";
    const auto hist = Train(m, fit, validation, epochs, &batches, &root);
    for (std::size_t e = 0; e < hist.size(); ++e) {
      history << e + 1 << '\t' << hist[e] << '\n';
      std::cout << "epoch=" << e + 1 << " validationNLL=" << hist[e] << std::endl;
    }
    const Map first = replay::ToMap(nn::load_npz((root / "first_batch.npz").string()));
    nn::save_npz((root / "final_gradients.npz").string(), replay::ToNamed(replay::Named(m.p, m.Gradients(first))));
    m.p.save((root / "final.npz").string());
    Model deployed(0);
    deployed.p.load((root / "final.npz").string());
    const Map weights = replay::Arrays(deployed.p);
    check::Require(replay::SameBits(replay::Arrays(m.p), weights), "restore bitexact");
    const Map out = deployed.Evaluate(test);
    // The streaming cell from teacher-forced states reproduces every unrolled step.
    const char* names[6] = {"raw", "h", "c", "kappa", "window", "phi"};
    double      stream   = 0;
    for (int t = 0; t < kSteps; ++t) {
      auto at = [&](const std::string& name, int step) { return nn::select(out.at(name), 1, step); };
      const auto step = deployed.StepOnce(nn::select(test.at("input"), 1, t), test.at("labels"), at("h", t), at("c", t),
                                          at("kappa", t), at("window", t));
      for (int f = 0; f < 6; ++f) {
        const int next = (f >= 1 && f <= 4) ? 1 : 0;
        stream         = std::max(stream, replay::MaxAbsDiff(step[f], at(names[f], t + next)));
      }
    }
    check::Near(stream, 0, 2e-5, "native shared teacher/streaming path");
    const auto sample = deployed.Sample(0, 1, 45011);
    check::Require(replay::SameBits(sample.at("strokes"), deployed.Sample(0, 1, 45011).at("strokes")), "seeded repeat sample");
    nn::save_npz((root / "sample_ab.npz").string(), replay::ToNamed(sample));
    nn::save_npz((root / "sample_ba.npz").string(), replay::ToNamed(deployed.Sample(1, 0, 45011)));
    nn::save_npz((root / "sample_op.npz").string(), replay::ToNamed(deployed.Sample(10, 11, 45013)));
    nn::save_npz((root / "outputs.npz").string(), replay::ToNamed(out));
    nn::save_npz((root / "model_arrays.npz").string(), replay::ToNamed(weights));
    check::Require(replay::SameBits(replay::Arrays(deployed.p), weights), "immutable generation");
    const double  test_nll = out.at("loss").item<double>();
    std::ofstream metrics(root / "metrics.json");
    metrics << "{\"parameters\":1680,\"epochs\":" << epochs << ",\"updates\":" << epochs * 18 << ",\"test_nll\":" << test_nll << "}\n";
    std::cout << "testNLL=" << test_nll << " (TF recorded 2.84756); streaming max diff=" << stream << ", sample 'ab' length "
              << sample.at("length").item<int>() << ";restore/sharedStreaming/seeded/immutable PASS\n";
    std::cout << "wall " << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
