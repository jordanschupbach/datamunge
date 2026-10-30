// CMD: just run-nn flow_variational_autoencoder
// Amortized conditional affine-flow posterior with explicit change-of-variables and standard-normal prior.
// THREE default epochs.
//
// Port of tflow's flow_variational_autoencoder. Preparation (in ~/projects/tflow_examples):
// just beta-vae-assets; data are read from $DATAMUNGE_DATA_DIR/beta_vae. Outputs go to
// output/nn/models/flow_variational_autoencoder. Checkpoints are best.npz instead of SaveV2
// bundles; the full 4x4 coupling Jacobian of the unit check comes from four reverse passes.
// Options: epochs=N data=PATH out=PATH init=PATH (start from saved parameters, e.g. tflow's
// initial_arrays.npz, to follow its trajectory exactly).

#include <chrono>
#include <iostream>
#include <limits>

#include "common/flow_variational_autoencoder.hpp"

namespace {

namespace nn = datamunge::nn;

double Difference(const nn::Tensor& a, const nn::Tensor& b) {
  check::Require(a.shape() == b.shape(), "same comparison shape");
  return ae::MaxAbsDiff(a, b, a.numel());
}

double Determinant4(const std::vector<float>& jac, int batch) {
  double a[4][4], out = 1;
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) a[r][c] = jac[batch * 16 + r * 4 + c];
  for (int c = 0; c < 4; ++c) {
    int pivot = c;
    for (int r = c + 1; r < 4; ++r)
      if (std::abs(a[r][c]) > std::abs(a[pivot][c])) pivot = r;
    check::Require(std::abs(a[pivot][c]) > 1e-10, "nonsingular coupling Jacobian");
    if (pivot != c) {
      for (int k = 0; k < 4; ++k) std::swap(a[pivot][k], a[c][k]);
      out = -out;
    }
    out *= a[c][c];
    for (int r = c + 1; r < 4; ++r) {
      const double ratio = a[r][c] / a[c][c];
      for (int k = c + 1; k < 4; ++k) a[r][k] -= ratio * a[c][k];
    }
  }
  return out;
}

void Checks(const std::filesystem::path& root) {
  nn::ParameterStore p(3049);
  const std::vector<float> zv{.2f, -.4f, .6f, -.8f, 1.1f, .3f, -.2f, .5f, -.7f, .9f, .4f, -.1f},
      cv{.1f, .2f, -.3f, .5f, .8f, -.6f};
  auto z = nn::from_vector(zv, {3, 4}).requires_grad_(), context = nn::from_vector(cv, {3, 2});
  auto a = ae::ConditionalAffinePosteriorFlow(p, "flow0", z, context),
       b = ae::ConditionalAffinePosteriorFlow(p, "flow1", a.value, context, true);
  auto inv1 = ae::ConditionalAffinePosteriorFlow(p, "flow1", b.value, context, true, true),
       inv0 = ae::ConditionalAffinePosteriorFlow(p, "flow0", inv1.value, context, false, true);
  auto ld = a.log_determinant + b.log_determinant, ild = inv1.log_determinant + inv0.log_determinant;
  p.set("identity/heads/W", nn::zeros({16, 4}));
  p.set("identity/heads/b", nn::zeros({4}));
  auto identity = ae::ConditionalAffinePosteriorFlow(p, "identity", z, context);
  std::vector<float> jac(3 * 16);
  for (int j = 0; j < 4; ++j) {
    auto row = nn::grad(nn::sum(nn::select(b.value, 1, j)), {z}, nn::GradOptions{true, false, false})[0].to_vector<float>();
    for (int batch = 0; batch < 3; ++batch)
      for (int k = 0; k < 4; ++k) jac[batch * 16 + j * 4 + k] = row[batch * 4 + k];  // full 4x4 incl. conditioners
  }
  const auto zd = z.detach();
  check::Require(Difference(inv0.value, zd) < 2e-6, "conditional flow inverse recovers all inputs");
  check::Require(ae::Bytes(identity.value) == ae::Bytes(zd), "zero scale/shift is BITEXACT identity");
  const auto lds = ld.to_vector<float>(), ilds = ild.to_vector<float>(), ids = identity.log_determinant.to_vector<float>();
  for (int batch = 0; batch < 3; ++batch) {
    const double det = Determinant4(jac, batch);
    check::Require(det > 0, "positive affine coupling determinant");
    check::Near(std::log(det), lds[batch], 2e-6, "fullJacobian determinant equals summed triangular logscales");
    check::Near(lds[batch] + ilds[batch], 0, 2e-6, "inverse logdet cancels forward");
    check::Near(ids[batch], 0, 0, "identity logdet exactly0");
  }
  nn::save_npz((root / "flow_algebra.npz").string(), {{"z", zd},
                                                      {"context", context},
                                                      {"mapped", b.value.detach()},
                                                      {"logdet", ld.detach()},
                                                      {"inverse", inv0.value.detach()},
                                                      {"inverse_logdet", ild.detach()},
                                                      {"identity", identity.value.detach()},
                                                      {"identity_logdet", identity.log_determinant.detach()},
                                                      {"jacobian", nn::from_vector(jac, {3, 4, 4})}});
  nn::save_npz((root / "unit_flow_arrays.npz").string(), ae::Arrays(p));
  std::cout << "conditional flow fullJacobian/inverse/logdensity-sign/identity checks passed" << std::endl;
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  auto pools = ae::LoadBetaFactors(data);
  std::filesystem::create_directories(root);
  Checks(root);
  ae::SaveFactorData(root, pools);
  std::vector<ae::VaeNoiseBank> noise{ae::BetaVaeNoise(210, 4, 3011), ae::BetaVaeNoise(42, 4, 3019),
                                      ae::BetaVaeNoise(42, 4, 3023)},
      training;
  auto             prior_noise = ae::BetaVaeNoise(20, 1, 3037);
  nn::NamedTensors fixed{{"prior_standard", prior_noise.standard}, {"prior_epsilon", prior_noise.epsilon}};
  for (int split = 0; split < 3; ++split) {
    const std::string name = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    fixed.emplace_back(name + "_standard", noise[split].standard);
    fixed.emplace_back(name + "_epsilon", noise[split].epsilon);
  }
  nn::save_npz((root / "fixed_noise.npz").string(), fixed);
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    training.push_back(ae::BetaVaeNoise(210, 1, 3041 + epoch));
    nn::save_npz((root / ("train_noise_epoch" + std::to_string(epoch) + ".npz")).string(),
                 {{"standard", training.back().standard}, {"epsilon", training.back().epsilon}});
  }
  ae::FlowVariationalModel model;
  if (!init.empty()) model.parameters.load(init);
  check::Require(model.parameters.count() == 52672 && model.parameters.size() == 18,
                 "two conditional affine posterior layers census");
  nn::save_npz((root / "initial_arrays.npz").string(), ae::Arrays(model.parameters));
  nn::BatchOrder order(210, 32, 3007);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\ttrain_preupdate_negative_elbo\tvalidation_nll\tvalidation_logq_minus_logp\tvalidation_logdet\t"
             "validation_negative_elbo\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto fit_source = pools[0].source.to_vector<int>();
  const auto best_path  = (root / "best.npz").string();
  double     best       = std::numeric_limits<double>::infinity();
  int        selected = 0, step = 0;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    auto   epsilon = ae::BetaVaeNoiseAt(training[epoch - 1], 0);
    double sum     = 0;
    int    seen    = 0;
    for (;;) {
      auto rows = order.next();
      if (rows.empty()) break;
      auto images = nn::take_rows(pools[0].images, rows), e = nn::take_rows(epsilon, rows);
      if (step == 0) {
        nn::save_npz((root / "first_batch.npz").string(), {{"images", images}, {"epsilon", e}});
        auto g = ae::CheckGradients(model.parameters, model.Gradient(images, e), "all flow gradients finite",
                                    "all18posterior/conditioner/decoder tensors receive derivatives");
        nn::save_npz((root / "first_gradients.npz").string(), g);
        nn::save_npz((root / "first_forward.npz").string(), model.Evaluate(images, e));
      }
      sum += double(rows.size()) * model.Update(images, e);
      seen += int(rows.size());
      for (auto row : rows) batches << epoch << '\t' << step << '\t' << row << '\t' << fit_source[row] << '\n';
      if (step == 0) nn::save_npz((root / "first_update_arrays.npz").string(), ae::Arrays(model.parameters));
      ++step;
    }
    check::Require(seen == 210, "all fit images once per epoch");
    auto score = ae::FlowVaeScore(model, pools[1].images, noise[1]);
    history << epoch << '\t' << sum / seen;
    for (double value : score) history << '\t' << value;
    history << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), ae::Arrays(model.parameters));
    if (score[3] < best) {
      best     = score[3];
      selected = epoch;
      model.parameters.save(best_path);  // four-draw validation Monte Carlo negative ELBO selection
    }
    std::cout << "epoch" << epoch << ":validation NLL=" << score[0] << ",logratio=" << score[1]
              << ",logdet=" << score[2] << ",negativeELBO=" << score[3] << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  ae::FlowVariationalModel deployed(false);
  deployed.parameters.load(best_path);
  check::Near(ae::FlowVaeScore(deployed, pools[1].images, noise[1])[3], best, 1e-8, "fresh selected negativeELBO");
  const auto       before = ae::Snapshot(deployed.parameters);
  nn::NamedTensors output;
  double           inverse_error = 0, inverse_logdet_error = 0, min_ratio = std::numeric_limits<double>::infinity();
  for (int split = 0; split < 3; ++split) {
    const std::int64_t n    = pools[split].images.size(0);
    const std::string  name = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    auto               deterministic = deployed.Evaluate(pools[split].images, nn::zeros({n, 4}));
    check::Require(ae::Bytes(ae::Find(deterministic, "z0")) == ae::Bytes(ae::Find(deterministic, "mu")),
                   "zero base noise gives z0=mu,not flowed posterior mean");
    for (const auto& [key, value] : deterministic) output.emplace_back(name + "_zero_" + key, value);
    for (int k = 0; k < 4; ++k) {
      auto f = deployed.Evaluate(pools[split].images, ae::BetaVaeNoiseAt(noise[split], k));
      check::Require(ae::Bytes(ae::Find(f, "context")) == ae::Bytes(ae::Find(deterministic, "context")) &&
                         ae::Bytes(ae::Find(f, "mu")) == ae::Bytes(ae::Find(deterministic, "mu")),
                     "base posterior/context independent of Gaussian draw");
      inverse_error = std::max(inverse_error, Difference(ae::Find(f, "z0"), ae::Find(f, "inverse_z0")));
      const auto ld = ae::Find(f, "logdet").to_vector<float>(), ild = ae::Find(f, "inverse_logdet").to_vector<float>(),
                 ratio = ae::Find(f, "logratio").to_vector<float>(), logq = ae::Find(f, "logq").to_vector<float>(),
                 logq0 = ae::Find(f, "logq0").to_vector<float>();
      for (std::int64_t b = 0; b < n; ++b) {
        check::Require(std::abs(ld[b]) <= 2.00001, "two-layer bounded logdet within[-2,2]");
        inverse_logdet_error = std::max(inverse_logdet_error, std::abs(double(ld[b]) + ild[b]));
        min_ratio            = std::min(min_ratio, double(ratio[b]));
        check::Near(logq[b], double(logq0[b]) - ld[b], 2e-6, "change-of-variables MINUS forward logdet");
      }
      for (const auto& [key, value] : f) output.emplace_back(name + "_draw" + std::to_string(k) + "_" + key, value);
    }
  }
  check::Require(inverse_error < 3e-5 && inverse_logdet_error < 2e-5,
                 "trained conditional flow roundtrip and inverse logdet cancellation");
  auto test   = deployed.Evaluate(pools[2].images, ae::BetaVaeNoiseAt(noise[2], 0)),
       repeat = deployed.Evaluate(pools[2].images, ae::BetaVaeNoiseAt(noise[2], 0)),
       other  = deployed.Evaluate(pools[2].images, ae::BetaVaeNoiseAt(noise[2], 1));
  const auto test_p = ae::Find(test, "probability");
  check::Require(ae::Bytes(test_p) == ae::Bytes(ae::Find(repeat, "probability")) &&
                     ae::Bytes(test_p) == ae::Bytes(deployed.Decode(ae::Find(test, "z"))),
                 "fixednoise repeat and flowed-z-only decode BITEXACT");
  check::Require(ae::Bytes(ae::Find(test, "z")) != ae::Bytes(ae::Find(other, "z")) &&
                     ae::Bytes(test_p) != ae::Bytes(ae::Find(other, "probability")),
                 "changed base noise changes flowed latent and image probabilities");
  auto single = deployed.Evaluate(nn::take_rows(pools[2].images, {0}), nn::take_rows(ae::BetaVaeNoiseAt(noise[2], 0), {0}));
  const double parity = Difference(ae::Find(single, "probability"), nn::take_rows(test_p, {0}));
  check::Require(parity < 2e-6, "flow VAE single/batch parity");
  auto prior_z = ae::BetaVaeNoiseAt(prior_noise, 0);
  auto prior   = deployed.Decode(prior_z);
  nn::save_npz((root / "prior.npz").string(), {{"z", prior_z}, {"probability", prior}});
  ae::Mosaic(root / "prior_means.png", prior, 5, 3);
  check::Require(before == ae::Snapshot(deployed.parameters), "all posterior/inverse/PRIOR decoding calls immutable");
  nn::save_npz((root / "outputs.npz").string(), output);
  nn::save_npz((root / "model_arrays.npz").string(), ae::Arrays(deployed.parameters));
  auto         score    = ae::FlowVaeScore(deployed, pools[2].images, noise[2]);
  auto         mean     = ae::Constant(ae::PixelMean(pools[0].images), 42);
  const auto   zero_p   = ae::Find(output, "test_zero_probability");
  const double mse      = ae::Mse(zero_p, pools[2].images), mean_mse = ae::Mse(mean, pools[2].images);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":52672,\"tensors\":18,\"flow_layers\":2,\"latent_dimensions\":4,\"epochs\":" << epochs
       << ",\"updates\":" << step << ",\"selected_epoch\":" << selected << ",\"mc_samples\":4,\"test_nll\":" << score[0]
       << ",\"test_mc_logq_minus_logp\":" << score[1] << ",\"test_mean_forward_logdet\":" << score[2]
       << ",\"test_negative_elbo\":" << score[3] << ",\"decode_f_mu_test_mse\":" << mse
       << ",\"fit_mean_test_mse\":" << mean_mse << ",\"maximum_inverse_error\":" << inverse_error
       << ",\"maximum_inverse_logdet_error\":" << inverse_logdet_error << ",\"minimum_case_logratio\":" << min_ratio
       << ",\"single_batch_max_error\":" << parity << ",\"training_seconds\":" << seconds << "}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>VAE with conditional-flow posterior</title><style>body{font:16px "
          "system-ui;max-width:1000px;margin:30px}img{max-width:100%;image-rendering:pixelated}</style><h1>Normalizing-flow "
          "posterior VAE</h1><p>Base4Ddiagonal Gaussian from784->32ReLUencoder. Two alternating2+2affine coupling layers "
          "are conditioned on image encoder context32;conditioner34->16ReLU->4scale/shift,scale=.5tanh(rawscale). "
          "Analytic inverses and logJacobians. Decoder4->32ReLU->784Bernoulli sees ONLY flowedz.52672parameters/18tensors,"
          "THREEepochs21Adamupdates. <a href='https://proceedings.mlr.press/v37/rezende15.html'>Flow variational "
          "inference</a>,<a href='https://arxiv.org/abs/1605.08803'>Affine coupling</a>.</p><p>logqK=logq0-SUMforwardlogdet,"
          "logp=standardnormal density at flowedz. Train NLL+logqK-logp,not base analyticGaussianKL. Conditional flow "
          "context fixed when computing Jacobian/inverse. Four Gaussian validation draws select MCnegativeELBO. Case "
          "logratio can be negative and is not clipped. No image-to-decoder skip. This adds posterior flexibility,not an "
          "image-space flow or learned generative prior.</p><p>Original,decode(f(mu)),one posterior decoder-probability "
          "draw,fit-only mean. Ten evenly spaced heldout cases,unfiltered,fixed0..1gray. f(mu)is not the mean of nonlinear "
          "flowed posterior,its decode is not posterior predictive mean. TestNLL="
       << score[0] << ",MClogratio=" << score[1] << ",negativeELBO=" << score[3] << ",decode(fmu)MSE=" << mse
       << ",fitmeanMSE=" << mean_mse
       << ". Single seed,short-run quality retained;no tighter-than-baseline or semantic success claim.</p>";
  const auto source = pools[2].source.to_vector<int>();
  const std::vector<nn::Tensor> values{ae::Flatten(pools[2].images), zero_p, test_p, mean};
  for (int k = 0; k < 10; ++k) {
    const int               b = k * 41 / 9;
    std::vector<nn::Tensor> panels;
    for (const auto& v : values) panels.push_back(ae::Row(v, b));
    const std::string file = "case" + std::to_string(k) + ".png";
    ae::Mosaic(root / file, nn::stack(panels, 0), 4, 4);
    html << "<h2>Case" << k << ",source" << source[b] << "</h2><img src='" << file << "'>";
  }
  html << "<h2>20standard-normal PRIOR decoder means</h2><p>Sample finalz~N0I and decode DIRECTLY without "
          "image/context/posterior flow. Applying the posterior flow to prior samples would change the generative "
          "model.</p><img src='prior_means.png'>";
  check::Require(bool(meta) && bool(html) && bool(history) && bool(batches), "complete flow VAE evidence");
  std::cout << "selected" << selected << ",test NLL=" << score[0] << ",MClogratio=" << score[1]
            << ",meanlogdet=" << score[2] << ",negativeELBO=" << score[3] << ",decode(fmu)MSE=" << mse
            << ",fitmeanMSE=" << mean_mse << ",maxinverse=" << inverse_error
            << ";restore/densitysign/inverse/decode/noise/batch/immutable passed\ntraining time " << seconds
            << " s\nArtifacts:" << (root / "index.html").string() << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::string           init;
    std::filesystem::path data = paths::Dataset("beta_vae"),
                          out  = paths::OutputRoot() / "models/flow_variational_autoencoder";
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
