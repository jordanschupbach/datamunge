// CMD: just run-nn gumbel_categorical_autoencoder
// Four 8-way categorical VAE with explicit Gumbels, soft training and hard inference diagnostics.
// THREE default epochs.
//
// Port of tflow's gumbel_categorical_autoencoder. Preparation (in ~/projects/tflow_examples):
// just beta-vae-assets; data are read from $DATAMUNGE_DATA_DIR/beta_vae. Outputs go to
// output/nn/models/gumbel_categorical_autoencoder. Uniform/Gumbel noise banks are bit-identical
// to tflow's; checkpoints are best.npz instead of SaveV2 bundles. Options: temperature=.7
// epochs=N data=PATH out=PATH init=PATH (initial parameters, e.g. tflow's initial_arrays.npz,
// to follow its trajectory exactly).

#include <chrono>
#include <iostream>
#include <limits>

#include "common/gumbel_categorical_autoencoder.hpp"

namespace {

namespace nn = datamunge::nn;

void Checks(const std::filesystem::path& root) {
  const std::vector<float> lv{.1f, -.4f, .7f, .2f, .6f, -.3f}, nv{.3f, -.2f, .6f, -.1f, .4f, .2f};
  auto L = nn::from_vector(lv, {2, 3}).requires_grad_(), G = nn::from_vector(nv, {2, 3}).requires_grad_();
  auto a = ae::GumbelCategorical(L, G, .7f), cold = ae::GumbelCategorical(L, G, .25f),
       warm = ae::GumbelCategorical(L, G, 2.f);
  auto kl        = ae::CategoricalUniformKl(a.probability, a.log_probability, 3);
  auto objective = 0.5 * nn::sum(nn::square(a.relaxed)) + 0.3 * nn::sum(kl, {0});
  auto gradients = nn::grad(objective, {L, G});
  nn::Tensor extreme_kl;
  {
    auto extreme = ae::GumbelCategorical(nn::tensor({{100.f, -100.f, 0.f}, {-100.f, 0.f, 100.f}}), nn::zeros({2, 3}), .7f);
    extreme_kl   = ae::CategoricalUniformKl(extreme.probability, extreme.log_probability, 3);
  }
  auto calibration = ae::GumbelNoise(4096, 1, 2999, 1, 3);
  auto calibration_noise = nn::reshape(calibration.noise, {4096, 3}), uniform = nn::reshape(calibration.uniform, {4096, 3}),
       gumbel = nn::reshape(calibration.gumbel, {4096, 3});
  auto sampled = ae::GumbelCategorical(
      nn::tensor({{float(std::log(.2)), float(std::log(.3)), float(std::log(.5))}}), calibration_noise, .7f);
  auto ia = nn::cast(a.indices, nn::DType::Int32), ic = nn::cast(cold.indices, nn::DType::Int32),
       iw = nn::cast(warm.indices, nn::DType::Int32);
  check::Require(ae::Bytes(ic) == ae::Bytes(iw) && ae::Bytes(ia) == ae::Bytes(ic),
                 "hard Gumbel-max samples independent of positive temperature");
  const auto q_out = a.probability.to_vector<float>(), y_out = a.relaxed.to_vector<float>(),
             k_out = kl.to_vector<float>(), dl = gradients[0].to_vector<float>(), dg = gradients[1].to_vector<float>(),
             ek = extreme_kl.to_vector<float>();
  for (int b = 0; b < 2; ++b) {
    double q[3], y[3], zq = 0, zy = 0;
    for (int j = 0; j < 3; ++j) {
      q[j] = std::exp(lv[b * 3 + j]);
      y[j] = std::exp((lv[b * 3 + j] + nv[b * 3 + j]) / .7);
      zq += q[j];
      zy += y[j];
    }
    double k = 0, yy = 0;
    for (int j = 0; j < 3; ++j) {
      q[j] /= zq;
      y[j] /= zy;
      k += q[j] * std::log(q[j] * 3);
      yy += y[j] * y[j];
      check::Near(q_out[b * 3 + j], q[j], 2e-7, "normalized categorical q");
      check::Near(y_out[b * 3 + j], y[j], 2e-7, "known relaxed draw");
    }
    check::Near(k_out[b], k, 2e-7, "known KL to categorical uniform");
    for (int j = 0; j < 3; ++j) {
      const double rec = y[j] * (y[j] - yy) / .7, posterior = .3 * q[j] * (std::log(q[j] * 3) - k);
      check::Near(dl[b * 3 + j], rec + posterior, 3e-7, "all six score derivatives include sampled reconstruction and KL");
      check::Near(dg[b * 3 + j], rec, 3e-7, "all six Gumbel derivatives exclude posterior KL");
    }
    check::Require(ek[b] <= std::log(3.) + 1e-6, "extreme categorical KL bounded and finite");
  }
  const auto       sampled_ids = nn::cast(sampled.indices, nn::DType::Int32).contiguous();
  std::vector<int> counts(3, 0);
  for (int id : sampled_ids.to_vector<int>()) ++counts[id];
  for (int j = 0; j < 3; ++j) {
    const double target = j == 0 ? .2 : j == 1 ? .3 : .5;
    check::Require(std::abs(counts[j] - 4096 * target) < 6 * std::sqrt(4096 * target * (1 - target)) + 1,
                   "4096Gumbel-max draws follow known categorical distribution within6sigma");
  }
  nn::save_npz((root / "categorical_algebra.npz").string(),
               {{"logits", L.detach()},
                {"noise", G.detach()},
                {"probability", a.probability},
                {"relaxed", a.relaxed},
                {"indices", ia},
                {"kl", kl},
                {"dlogits", gradients[0]},
                {"dnoise", gradients[1]},
                {"cold", cold.relaxed},
                {"warm", warm.relaxed},
                {"extreme_kl", extreme_kl},
                {"calibration_uniform", uniform},
                {"calibration_gumbel", gumbel},
                {"calibration_noise", calibration_noise},
                {"calibration_indices", sampled_ids},
                {"calibration_counts", nn::from_vector(counts, {3})}});
  std::cout << "categorical probabilities/KL/ALL12score+Gumbel partials/temperature invariance/extremes/4096calibration "
               "draws passed"
            << std::endl;
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, float temperature,
         const std::string& init) {
  auto pools = ae::LoadBetaFactors(data);
  std::filesystem::create_directories(root);
  Checks(root);
  ae::SaveFactorData(root, pools);
  std::vector<ae::GumbelNoiseBank> noise{ae::GumbelNoise(210, 4, 2927), ae::GumbelNoise(42, 4, 2939),
                                         ae::GumbelNoise(42, 4, 2953)},
      training;
  nn::NamedTensors fixed;
  for (int split = 0; split < 3; ++split) {
    const std::string name = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    fixed.emplace_back(name + "_uniform", noise[split].uniform);
    fixed.emplace_back(name + "_gumbel", noise[split].gumbel);
    fixed.emplace_back(name + "_noise", noise[split].noise);
  }
  nn::save_npz((root / "fixed_noise.npz").string(), fixed);
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    training.push_back(ae::GumbelNoise(210, 1, 2969 + epoch));
    nn::save_npz((root / ("train_noise_epoch" + std::to_string(epoch) + ".npz")).string(),
                 {{"uniform", training.back().uniform},
                  {"gumbel", training.back().gumbel},
                  {"noise", training.back().noise}});
  }
  ae::GumbelCategoricalModel model(temperature);
  if (!init.empty()) model.parameters.load(init);
  check::Require(model.parameters.count() == 53104 && model.parameters.size() == 8,
                 "four8-way compact categorical model census");
  nn::save_npz((root / "initial_arrays.npz").string(), ae::Arrays(model.parameters));
  nn::BatchOrder order(210, 32, 2917);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\ttrain_preupdate_relaxed_objective\tvalidation_soft_nll\tvalidation_hard_nll\tvalidation_"
             "categorical_kl\tvalidation_relaxed_objective\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto fit_source = pools[0].source.to_vector<int>();
  const auto best_path  = (root / "best.npz").string();
  double     best       = std::numeric_limits<double>::infinity();
  int        selected = 0, step = 0;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    auto   bank = ae::GumbelNoiseAt(training[epoch - 1], 0);
    double sum  = 0;
    int    seen = 0;
    for (;;) {
      auto rows = order.next();
      if (rows.empty()) break;
      auto images = nn::take_rows(pools[0].images, rows), gumbel = nn::take_rows(bank, rows);
      if (step == 0) {
        nn::save_npz((root / "first_batch.npz").string(), {{"images", images}, {"noise", gumbel}});
        auto g = ae::CheckGradients(model.parameters, model.Gradient(images, gumbel), "all categorical gradients finite",
                                    "all eight tensors receive pathwise+categoricalKL gradients");
        nn::save_npz((root / "first_gradients.npz").string(), g);
        nn::save_npz((root / "first_forward.npz").string(), model.Evaluate(images, gumbel));
      }
      sum += double(rows.size()) * model.Update(images, gumbel);
      seen += int(rows.size());
      for (auto row : rows) batches << epoch << '\t' << step << '\t' << row << '\t' << fit_source[row] << '\n';
      if (step == 0) nn::save_npz((root / "first_update_arrays.npz").string(), ae::Arrays(model.parameters));
      ++step;
    }
    check::Require(seen == 210, "all fitting cases once per epoch");
    auto score = ae::GumbelScore(model, pools[1].images, noise[1]);
    history << epoch << '\t' << sum / seen;
    for (double value : score) history << '\t' << value;
    history << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), ae::Arrays(model.parameters));
    if (score[3] < best) {
      best     = score[3];
      selected = epoch;
      model.parameters.save(best_path);  // fixed 4-draw validation RELAXED objective selects
    }
    std::cout << "epoch" << epoch << ":validation softNLL=" << score[0] << ",hardNLL=" << score[1]
              << ",categoricalKL=" << score[2] << ",relaxedObjective=" << score[3] << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  ae::GumbelCategoricalModel deployed(temperature, false);
  deployed.parameters.load(best_path);
  check::Near(ae::GumbelScore(deployed, pools[1].images, noise[1])[3], best, 1e-8,
              "fresh selected relaxed validation objective");
  const auto       before = ae::Snapshot(deployed.parameters);
  nn::NamedTensors output;
  for (int split = 0; split < 3; ++split) {
    const std::int64_t n             = pools[split].images.size(0);
    const std::string  name          = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    auto               deterministic = deployed.Evaluate(pools[split].images, nn::zeros({n, 4, 8}));
    output.emplace_back(name + "_posterior_probability_decode",
                        deployed.DecodeSoft(ae::Find(deterministic, "probability")));
    output.emplace_back(name + "_map_ids", ae::Find(deterministic, "indices"));
    output.emplace_back(name + "_map_prediction", deployed.Decode(ae::Find(deterministic, "indices")));
    for (int k = 0; k < 4; ++k) {
      auto f = deployed.Evaluate(pools[split].images, ae::GumbelNoiseAt(noise[split], k));
      check::Require(ae::Bytes(ae::Find(f, "probability")) == ae::Bytes(ae::Find(deterministic, "probability")),
                     "Gumbel noise changes samples,not categorical q");
      check::Require(ae::Bytes(ae::Find(f, "soft_prediction")) == ae::Bytes(deployed.DecodeSoft(ae::Find(f, "relaxed"))) &&
                         ae::Bytes(ae::Find(f, "hard_prediction")) == ae::Bytes(deployed.Decode(ae::Find(f, "indices"))),
                     "soft and hard code-only decoders bitexact");
      const auto kl = ae::Find(f, "kl").to_vector<float>(), q = ae::Find(f, "probability").to_vector<float>(),
                 y = ae::Find(f, "relaxed").to_vector<float>(), hd = ae::Find(f, "hard").to_vector<float>();
      for (std::int64_t b = 0; b < n; ++b) {
        check::Require(kl[b] >= -1e-5 && kl[b] <= 4 * std::log(8.) + 1e-5, "categoricalKL bounds0..4log8");
        for (int group = 0; group < 4; ++group) {
          double qsum = 0, ysum = 0, hsum = 0;
          for (int c = 0; c < 8; ++c) {
            const auto i = (b * 4 + group) * 8 + c;
            qsum += q[i];
            ysum += y[i];
            hsum += hd[i];
          }
          check::Near(qsum, 1, 2e-6, "posterior simplex normalized");
          check::Near(ysum, 1, 2e-6, "relaxed simplex normalized");
          check::Near(hsum, 1, 0, "hard codes exactly one-hot");
        }
      }
      for (const auto& [key, value] : f) output.emplace_back(name + "_draw" + std::to_string(k) + "_" + key, value);
    }
  }
  auto test   = deployed.Evaluate(pools[2].images, ae::GumbelNoiseAt(noise[2], 0)),
       repeat = deployed.Evaluate(pools[2].images, ae::GumbelNoiseAt(noise[2], 0)),
       other  = deployed.Evaluate(pools[2].images, ae::GumbelNoiseAt(noise[2], 1));
  check::Require(ae::Bytes(ae::Find(test, "soft_prediction")) == ae::Bytes(ae::Find(repeat, "soft_prediction")) &&
                     ae::Bytes(ae::Find(test, "indices")) == ae::Bytes(ae::Find(repeat, "indices")),
                 "fixedGumbel repeat bitexact");
  check::Require(ae::Bytes(ae::Find(test, "relaxed")) != ae::Bytes(ae::Find(other, "relaxed")) &&
                     ae::Bytes(ae::Find(test, "indices")) != ae::Bytes(ae::Find(other, "indices")),
                 "different Gumbels change relaxed AND hard samples");
  auto single = deployed.Evaluate(nn::take_rows(pools[2].images, {0}), nn::take_rows(ae::GumbelNoiseAt(noise[2], 0), {0}));
  const double parity = ae::MaxAbsDiff(ae::Find(single, "soft_prediction"), ae::Find(test, "soft_prediction"), 784);
  check::Require(parity < 2e-6, "categorical single/batch parity");
  for (int bad : {-1, 8}) {
    bool rejected = false;
    try {
      deployed.Decode(nn::from_vector(std::vector<int>(4, bad), {1, 4}));
    } catch (const std::exception&) {
      rejected = true;
    }
    check::Require(rejected, "invalid category IDs rejected");
  }
  bool simplex_rejected = false;
  try {
    deployed.DecodeSoft(nn::zeros({1, 4, 8}));
  } catch (const std::exception&) {
    simplex_rejected = true;
  }
  check::Require(simplex_rejected, "non-simplex decoder code rejected");
  std::mt19937        random(2963);
  std::vector<int>    ids(80);
  std::vector<double> us(80);
  for (int i = 0; i < 80; ++i) {
    const double u = (double(random()) + .5) / 4294967296.;
    us[i]          = u;
    ids[i]         = int(8 * u);
  }
  auto prior_ids = nn::from_vector(ids, {20, 4});
  auto prior     = deployed.Decode(prior_ids);
  nn::save_npz((root / "prior.npz").string(),
               {{"indices", prior_ids}, {"uniform", nn::from_vector(us, {20, 4})}, {"probability", prior}});
  ae::Mosaic(root / "prior_means.png", prior, 5, 3);
  check::Require(before == ae::Snapshot(deployed.parameters),
                 "all categorical inference/decoding/prior diagnostics immutable");
  nn::save_npz((root / "model_arrays.npz").string(), ae::Arrays(deployed.parameters));
  nn::save_npz((root / "outputs.npz").string(), output);
  auto         score           = ae::GumbelScore(deployed, pools[2].images, noise[2]);
  auto         mean            = ae::Constant(ae::PixelMean(pools[0].images), 42);
  const double mean_mse        = ae::Mse(mean, pools[2].images),
               probability_mse = ae::Mse(ae::Find(output, "test_posterior_probability_decode"), pools[2].images),
               map_mse         = ae::Mse(ae::Find(output, "test_map_prediction"), pools[2].images);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":53104,\"tensors\":8,\"groups\":4,\"classes\":8,\"temperature\":" << temperature
       << ",\"epochs\":" << epochs << ",\"updates\":" << step << ",\"selected_epoch\":" << selected
       << ",\"mc_samples\":4,\"test_soft_nll\":" << score[0] << ",\"test_hard_nll\":" << score[1]
       << ",\"test_categorical_kl\":" << score[2] << ",\"test_relaxed_objective\":" << score[3]
       << ",\"test_hard_mc_negative_elbo\":" << score[1] + score[2] << ",\"probability_decode_test_mse\":" << probability_mse
       << ",\"map_decode_test_mse\":" << map_mse << ",\"fit_mean_test_mse\":" << mean_mse
       << ",\"single_batch_max_error\":" << parity << ",\"training_seconds\":" << seconds << "}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Gumbel-softmax categorical VAE</title><style>body{font:16px "
          "system-ui;max-width:1100px;margin:30px}img{max-width:100%;image-rendering:pixelated}</style><h1>Gumbel-softmax "
          "categorical VAE</h1><p>Encoder784->32ReLU->4x8logits,four independent8-way categorical posterior variables. "
          "q=softmax(logits),explicitg=-log(-logU),y=softmax((logits+g)/temperature). Decoder32->32ReLU->784Bernoulli."
          "53104parameters/8tensors,THREEepochs21Adamupdates,temperature="
       << temperature
       << "fixed(no annealing). Train relaxed reconstructionNLL+analytic categoricalKL(q||Uniform8),4fixed validation "
          "draws select relaxed objective. <a href='https://arxiv.org/abs/1611.01144'>Gumbel-softmax paper</a>,<a "
          "href='https://arxiv.org/abs/1611.00712'>Concrete paper</a>.</p><p>This soft-likelihood/discrete-KL "
          "combination is a BIASED CONTINUOUS SURROGATE,not an exact discrete ELBO or Concrete-distribution KL. Hard "
          "argmax(logits+g) gives actual categorical samples;hardNLL+discreteKL is reported separately as a Monte Carlo "
          "discrete negativeELBO diagnostic. No hard straight-through estimator is used in fitting. Hard samples are "
          "independent of positive temperature,soft values/gradients are not.</p><p>Tiny294generatedbinaryimages210fit"
          "42validation42test;factors do not feed fitting/selection. Test softNLL="
       << score[0] << ",hardNLL=" << score[1] << ",categoricalKL=" << score[2] << ",relaxedobjective=" << score[3]
       << ". decode(q)MSE=" << probability_mse << ",MAPMSE=" << map_mse << ",fitmeanMSE=" << mean_mse
       << ". Neither decode(q)nor MAP decoding is a posterior predictive mean. Soft/hard mismatch and weak quality are "
          "retained.</p><p>Original,decode(q),MAPhard,one relaxed posterior draw,the MATCHED hard draw,fit-only mean. "
          "Ten evenly spaced unfiltered cases,fixed0..1gray. Probabilities,not binary draws. Four categories "
          "require12rawsymbolbits only for hardcodes,excluding network/framing;no actual codec claim.</p>";
  const std::vector<nn::Tensor> values{ae::Flatten(pools[2].images),
                                       ae::Find(output, "test_posterior_probability_decode"),
                                       ae::Find(output, "test_map_prediction"),
                                       ae::Find(test, "soft_prediction"),
                                       ae::Find(test, "hard_prediction"),
                                       mean};
  const auto source = pools[2].source.to_vector<int>();
  for (int k = 0; k < 10; ++k) {
    const int         b    = k * 41 / 9;
    const std::string file = "case" + std::to_string(k) + ".png";
    ae::CaseStrip(root / file, values, b);
    html << "<h2>Case" << k << ",source" << source[b] << "</h2><img src='" << file << "'>";
  }
  html << "<h2>20uniform categorical prior decoder means</h2><p>Four independent uniform0..7IDs;target-free decoder,not "
          "aggregate-posterior sampling.</p><img src='prior_means.png'>";
  check::Require(bool(meta) && bool(html) && bool(history) && bool(batches), "complete categorical VAE evidence");
  std::cout << "selected" << selected << ",softNLL=" << score[0] << ",hardNLL=" << score[1]
            << ",categoricalKL=" << score[2] << ",relaxedobjective=" << score[3]
            << ",hardMCnegativeELBO=" << score[1] + score[2] << ",decode(q)MSE=" << probability_mse
            << ",MAPMSE=" << map_mse << ",fitmeanMSE=" << mean_mse
            << ";restore/soft+hard decode/noise/simplex/batch/invalid/immutable passed\ntraining time " << seconds
            << " s\nArtifacts:" << (root / "index.html").string() << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs      = 3;
    float                 temperature = .7f;
    std::string           init;
    std::filesystem::path data = paths::Dataset("beta_vae"),
                          out  = paths::OutputRoot() / "models/gumbel_categorical_autoencoder";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("epochs=", 0) == 0)
        epochs = std::stoi(a.substr(7));
      else if (a.rfind("temperature=", 0) == 0)
        temperature = std::stof(a.substr(12));
      else if (a.rfind("data=", 0) == 0)
        data = a.substr(5);
      else if (a.rfind("out=", 0) == 0)
        out = a.substr(4);
      else if (a.rfind("init=", 0) == 0)
        init = a.substr(5);
      else
        throw std::invalid_argument("expected epochs=N temperature=F data=PATH out=PATH init=PATH");
    }
    check::Require(epochs > 0 && epochs <= 100 && std::isfinite(temperature) && temperature >= .2f && temperature <= 2,
                   "three default epochs,optional1..100,and temperature.2..2");
    Run(data, out, epochs, temperature, init);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
