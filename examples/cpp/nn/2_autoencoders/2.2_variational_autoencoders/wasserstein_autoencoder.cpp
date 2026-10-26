// CMD: just run-nn wasserstein_autoencoder
// Prepare: just beta-vae-assets. THREE default epochs; epochs=N/data=PATH/out=PATH/init=PATH.
//
// Port of tflow's wasserstein_autoencoder (WAE-MMD). Preparation (in ~/projects/tflow_examples):
// just beta-vae-assets; data are read from $DATAMUNGE_DATA_DIR/beta_vae (the 294-image
// known-factor lattice). Outputs go to output/nn/models/wasserstein_autoencoder. Checkpoints are
// best.npz instead of SaveV2 bundles; metrics.json gains training_seconds. init=PATH loads initial
// parameters (e.g. tflow's initial_arrays.npz) to follow its trajectory exactly.

#include <chrono>
#include <iostream>
#include <limits>

#include "common/wasserstein_autoencoder.hpp"

namespace {

namespace nn = datamunge::nn;

double Difference(const nn::Tensor& a, const nn::Tensor& b) {
  check::Require(a.shape() == b.shape(), "matching outputs");
  const auto x = a.contiguous().to_vector<float>(), y = b.contiguous().to_vector<float>();
  double     d = 0;
  for (size_t i = 0; i < x.size(); ++i) d = std::max(d, std::abs(double(x[i]) - y[i]));
  return d;
}
float Scalar(const nn::NamedTensors& f, const std::string& name) { return ae::Find(f, name).item<float>(); }

void Algebra(const std::filesystem::path& root) {
  std::vector<float> a(12), b(16);
  for (int i = 0; i < 12; ++i) a[i] = .2f * (i - 5);
  for (int i = 0; i < 16; ++i) b[i] = .15f * ((i * 7) % 17 - 8);
  auto x = nn::from_vector(a, {3, 4}).requires_grad_(), y = nn::from_vector(b, {4, 4}).requires_grad_();
  auto m = ae::ImqMmd(x, y);
  auto g = nn::grad(m.value, {x, y});
  nn::NoGradGuard guard;
  auto            same = ae::ImqMmd(x, x), zero = ae::ImqMmd(nn::zeros({3, 4}), nn::zeros({4, 4}));
  check::Near(zero.value.item<float>(), 0, 0, "all-identical latent kernel yields exactzero MMD");
  check::Require(same.value.item<float>() < 0, "same finite empirical sample unbiased estimator CAN be negative,do not clip");
  const auto gx = g[0].to_vector<float>(), gy = g[1].to_vector<float>();
  for (int j = 0; j < 4; ++j) {
    double sum = 0;
    for (int i = 0; i < 3; ++i) sum += gx[i * 4 + j];
    for (int i = 0; i < 4; ++i) sum += gy[i * 4 + j];
    check::Near(sum, 0, 1e-6, "joint translation invariance gradient");
  }
  nn::save_npz((root / "mmd_algebra.npz").string(), {{"x", x.detach()},
                                                     {"y", y.detach()},
                                                     {"xx", m.xx.detach()},
                                                     {"yy", m.yy.detach()},
                                                     {"xy", m.xy.detach()},
                                                     {"mmd", m.value.detach()},
                                                     {"gradient_x", g[0]},
                                                     {"gradient_y", g[1]},
                                                     {"same_sample_unbiased_mmd", same.value},
                                                     {"all_identical_mmd", zero.value}});
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  auto pools = ae::LoadBetaFactors(data);
  std::filesystem::create_directories(root);
  Algebra(root);
  std::vector<ae::VaeNoiseBank> priors;
  nn::NamedTensors              bank;
  for (int k = 0; k < 3; ++k) {
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    priors.push_back(ae::BetaVaeNoise(pools[k].images.size(0), 1, 3719 + k * 6));
    bank.emplace_back(split + "_standard", priors[k].standard);
    bank.emplace_back(split + "_epsilon", priors[k].epsilon);
  }
  auto generation = ae::BetaVaeNoise(20, 1, 3749);
  bank.emplace_back("generation_standard", generation.standard);
  bank.emplace_back("generation_epsilon", generation.epsilon);
  ae::SaveFactorData(root, pools);
  nn::save_npz((root / "fixed_noise.npz").string(), bank);
  ae::WassersteinAutoencoderModel model;
  if (!init.empty()) model.parameters.load(init);
  check::Require(model.parameters.count() == 26036 && model.parameters.size() == 8, "compact deterministic WAE census");
  nn::save_npz((root / "initial_arrays.npz").string(), ae::Arrays(model.parameters));
  nn::BatchOrder order(210, 32, 3761);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tvalidation_squared_cost\tvalidation_mmd\tvalidation_objective\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto fit_source = pools[0].source.to_vector<int>();
  const auto best_path  = (root / "best.npz").string();
  double     best       = std::numeric_limits<double>::infinity();
  int        selected = 0, steps = 0;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    auto noise = ae::BetaVaeNoise(210, 1, 3779 + epoch);
    nn::save_npz((root / ("train_noise_epoch" + std::to_string(epoch) + ".npz")).string(),
                 {{"standard", noise.standard}, {"epsilon", noise.epsilon}});
    auto prior = ae::BetaVaeNoiseAt(noise, 0);
    order.begin_epoch();
    int seen = 0;
    for (;;) {
      auto rows = order.next();
      if (rows.empty()) break;
      auto x = nn::take_rows(pools[0].images, rows), p = nn::take_rows(prior, rows);
      if (steps == 0) {
        nn::save_npz((root / "first_batch.npz").string(), {{"images", x}, {"prior", p}});
        nn::save_npz((root / "first_forward.npz").string(), model.Evaluate(x, p));
        nn::save_npz((root / "first_gradients.npz").string(),
                     ae::CheckGradients(model.parameters, model.Gradient(x, p), "all WAE derivatives finite",
                                        "all8tensors receive WAE gradients"));
      }
      model.Update(x, p);
      if (steps == 0) nn::save_npz((root / "first_update_arrays.npz").string(), ae::Arrays(model.parameters));
      for (auto row : rows) batches << epoch << '\t' << steps << '\t' << row << '\t' << fit_source[row] << '\n';
      ++steps;
      seen += int(rows.size());
    }
    check::Require(seen == 210, "all fit images once per epoch");
    auto         v     = model.Evaluate(pools[1].images, ae::BetaVaeNoiseAt(priors[1], 0));
    const double score = Scalar(v, "reconstruction_loss");
    history << epoch << '\t' << score << '\t' << Scalar(v, "mmd") << '\t' << Scalar(v, "loss") << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), ae::Arrays(model.parameters));
    if (score < best) {
      best     = score;
      selected = epoch;
      model.parameters.save(best_path);  // validation squared transport cost selects WAE
    }
    std::cout << "epoch" << epoch << ":validation squaredcost=" << score << ",MMD=" << Scalar(v, "mmd")
              << ",objective=" << Scalar(v, "loss") << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  ae::WassersteinAutoencoderModel deployed(false);
  deployed.parameters.load(best_path);  // fresh WAE model-only restore
  check::Near(Scalar(deployed.Evaluate(pools[1].images, ae::BetaVaeNoiseAt(priors[1], 0)), "reconstruction_loss"), best,
              1e-8, "restored validation cost");
  const auto       before = ae::Snapshot(deployed.parameters);
  nn::NamedTensors outputs;
  for (int k = 0; k < 3; ++k) {
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    for (const auto& [key, value] : deployed.Evaluate(pools[k].images, ae::BetaVaeNoiseAt(priors[k], 0)))
      outputs.emplace_back(split + "_" + key, value);
  }
  auto p    = ae::BetaVaeNoiseAt(priors[2], 0);
  auto test = deployed.Evaluate(pools[2].images, p), repeat = deployed.Evaluate(pools[2].images, p),
       other = deployed.Evaluate(pools[2].images, ae::BetaVaeNoiseAt(ae::BetaVaeNoise(42, 1, 3793), 0));
  const auto probability = ae::Find(test, "probability");
  check::Require(ae::SameBytes(probability, ae::Find(repeat, "probability")) &&
                     ae::SameBytes(probability, deployed.Decode(ae::Find(test, "code"))),
                 "repeat/code-only decoder BITEXACT");
  check::Require(ae::SameBytes(ae::Find(test, "code"), ae::Find(other, "code")) &&
                     !ae::SameBytes(ae::Find(test, "mmd"), ae::Find(other, "mmd")),
                 "prior changes aggregate regularizer,not deterministic encoding");
  auto small = deployed.Evaluate(nn::take_rows(pools[2].images, {0, 1}), nn::take_rows(p, {0, 1, 2}));
  const double parity = Difference(ae::Find(small, "probability"), nn::take_rows(probability, {0, 1}));
  check::Require(parity < 2e-6, "reconstruction small/batch parity with UNEQUAL prior count");
  bool rejected = false;
  try {
    deployed.Evaluate(nn::take_rows(pools[2].images, {0}), p);
  } catch (const std::exception&) {
    rejected = true;
  }
  check::Require(rejected, "singleton unbiased MMD sample rejected before division");
  auto prior_z = ae::BetaVaeNoiseAt(generation, 0), prior = deployed.Decode(prior_z);
  nn::save_npz((root / "prior.npz").string(), {{"z", prior_z}, {"probability", prior}});
  nn::save_npz((root / "model_arrays.npz").string(), ae::Arrays(deployed.parameters));
  nn::save_npz((root / "outputs.npz").string(), outputs);
  check::Require(before == ae::Snapshot(deployed.parameters), "all WAE inference immutable");
  auto         mean = ae::Constant(ae::PixelMean(pools[0].images), 42);
  const double mse = ae::Mse(probability, pools[2].images), baseline = ae::Mse(mean, pools[2].images);
  const double cost = Scalar(test, "reconstruction_loss"), mmd = Scalar(test, "mmd"), objective = Scalar(test, "loss");
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":26036,\"tensors\":8,\"epochs\":" << epochs << ",\"updates\":" << steps
       << ",\"selected_epoch\":" << selected << ",\"mmd_weight\":10,\"test_squared_cost\":" << cost
       << ",\"test_mmd\":" << mmd << ",\"test_objective\":" << objective << ",\"test_mse\":" << mse
       << ",\"fit_mean_mse\":" << baseline << ",\"small_batch_max_error\":" << parity
       << ",\"training_seconds\":" << seconds << "}\n";
  ae::Mosaic(root / "prior_means.png", prior, 5, 3);
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>WAE-MMD</title><style>body{font:16px "
          "system-ui;margin:30px;max-width:1100px}img{max-width:100%;image-rendering:pixelated}</style><h1>Wasserstein "
          "autoencoder with MMD</h1><p><a href='https://arxiv.org/abs/1711.01558'>Wasserstein Auto-Encoders</a>,compact "
          "WAE-MMD variant. Deterministic784-16tanh-code4,decoder4-16tanh-784sigmoid,26036parameters8tensors. Train "
          "mean SUM784squared pixel transport cost+10unbiased aggregate latent MMD to N0I. No per-case KL/"
          "discriminator/Bernoulli training loss. IMQ kernel AVERAGES c/(c+||z-z'||^2) for c=1,4,16. Within-sample "
          "sums exclude diagonal,divide n(n-1)/m(m-1);cross sum includes ALLpairs/divide nm. Unequal sample counts "
          "supported,at least2each. Unbiased MMD CAN be negative and is not clipped;finite empirical estimator is not "
          "exact Wasserstein distance.</p><p>THREEdefault epochs21Adam updates,validation reconstruction cost selects "
          "epoch"
       << selected
       << ". Reuse294generated shapes210fit42validation42test;factors are provenance only. Fresh model-only restore,"
          "explicit independent prior,code-only decode,unequal-count batch parity,singleton rejection and immutable "
          "inference pass. Kernel normalization changes its scale;weight10 stated explicitly.</p><p>Test squaredcost="
       << cost << ",MMD=" << mmd << ",objective=" << objective << ",MSE=" << mse << ",fit-onlymeanMSE=" << baseline
       << ". This penalized short run does not establish prior matching,optimal transport equality or generation "
          "quality.</p><p>Ten evenly spaced unfiltered heldout cases:original/reconstruction/fit-onlymean,fixed0..1"
          "gray. Outputs are sigmoid probabilities,not pixel draws.</p>";
  const auto                    source = pools[2].source.to_vector<int>();
  const std::vector<nn::Tensor> values{ae::Flatten(pools[2].images), probability, mean};
  for (int k = 0; k < 10; ++k) {
    const int               b = k * 41 / 9;
    std::vector<nn::Tensor> panels;
    for (const auto& v : values) panels.push_back(nn::take_rows(v, {b}));
    const std::string name = "case_" + std::to_string(k) + ".png";
    ae::Mosaic(root / name, nn::concat(panels, 0), 3, 3);
    html << "<p>Source" << source[b] << "<br><img src='" << name << "'></p>";
  }
  html << "<h2>20unfiltered N0I prior decoder probabilities</h2><img src='prior_means.png'><p>Poor short-run outputs "
          "retained;no semantic/quality improvement claim.</p>";
  check::Require(bool(meta) && bool(html) && bool(history) && bool(batches), "complete WAE artifacts");
  std::cout << "selected" << selected << ",testMSE=" << mse << ",fitMean=" << baseline << ",MMD=" << mmd
            << ";kernelAlgebra/restore/priorIndependent/unequalcounts/singleton/codeOnly/immutable PASS\ntraining time "
            << seconds << " s\nArtifacts:" << (root / "index.html").string() << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::string           init;
    std::filesystem::path data = paths::Dataset("beta_vae"), out = paths::OutputRoot() / "models/wasserstein_autoencoder";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("epochs=", 0) == 0)
        epochs = std::stoi(a.substr(7));
      else if (a.rfind("data=", 0) == 0)
        data = a.substr(5);
      else if (a.rfind("out=", 0) == 0)
        out = a.substr(4);
      else if (a.rfind("init=", 0) == 0)
        init = a.substr(5);
      else
        throw std::invalid_argument("expected epochs=N data=PATH out=PATH init=PATH");
    }
    check::Require(epochs > 0 && epochs <= 100, "three default epochs or explicit1..100");
    Run(data, out, epochs, init);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
