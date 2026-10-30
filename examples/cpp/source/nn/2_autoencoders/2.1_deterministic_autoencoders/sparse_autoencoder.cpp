// CMD: just run-nn sparse_autoencoder
// BOTH activation-L1 and mean-unit Bernoulli-KL sparse autoencoders with matched controls.
// Four matched 64-sigmoid variants, three epochs/default, no label targets.
//
// Port of tflow's sparse_autoencoder (model and penalties: examples/nn/common/sparse_autoencoder.hpp).
// Preparation (in ~/projects/tflow_examples): just autoencoder-assets; data are read from
// $DATAMUNGE_DATA_DIR/saliency. Outputs (plain/, l1/, kl/, combined/ training records, data.npz,
// index.html and digit<k>.png galleries) go to output/nn/models/sparse_autoencoder. Checkpoints
// are best.npz files instead of TF SaveV2 bundles; metrics.json gains "training_seconds".
// Options: epochs=N data=PATH out=PATH init=PATH (start all four variants from saved
// parameters, e.g. tflow's plain/initial_arrays.npz, to follow its trajectories exactly; by
// default nn's own initialiser is used, identically for every variant).

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>

#include "common/sparse_autoencoder.hpp"

namespace {

namespace nn = datamunge::nn;

void Checks() {
  const std::vector<float> values{0, .05f, .2f, 1, 0, .05f, .6f, 1, 0, .05f, 1, 1};
  auto                     h        = nn::from_vector(values, {3, 4}).requires_grad_();
  auto                     p        = ae::ActivationSparsity(h, .1f, .2f);
  const auto               gradient = nn::grad(p.total, {h})[0].to_vector<float>();
  double                   l1 = 0, kl = 0;
  for (int i = 0; i < 12; ++i) l1 += std::abs(double(values[i])) / 12;
  for (int j = 0; j < 4; ++j) {
    double q = 0;
    for (int b = 0; b < 3; ++b) q += values[b * 4 + j] / 3.;
    const double clipped = std::clamp(q, double(1e-5f), double(1.f - 1e-5f)), rho = .05f;
    kl += (rho * std::log(rho / clipped) + (1.f - .05f) * std::log(double(1.f - .05f) / (1 - clipped))) / 4;
    for (int b = 0; b < 3; ++b) {
      double expected = values[b * 4 + j] == 0 ? 0 : double(.1f) / 12;
      if (q > 1e-5f && q < 1.f - 1e-5f) expected += double(.2f) * (-rho / q + double(1.f - .05f) / (1 - q)) / 12;
      check::Near(gradient[b * 4 + j], expected, 3e-7, "all12L1+KLpartials including low/high clamp and targetrho");
    }
  }
  check::Near(p.l1.item<float>(), l1, 4e-8, "activation L1 mean not parameter penalty");
  check::Near(p.kl.item<float>(), kl, 4e-6, "clamped perunit Bernoulli KL mean including extreme means");
  check::Near(p.total.item<float>(), double(.1f) * l1 + double(.2f) * kl, 1e-6, "weighted sum of distinct penalties");
  check::Throws([&] { ae::ActivationSparsity(h, .1f, .2f, 1.f); }, "invalid targetrho rejected");
  std::cout << "analytic L1/KL means/ALL12derivatives/zero/targetrho/clamped0+1means/invalidrho fixtures passed\n";
}

struct Variant {
  std::string name;
  float       l1, kl;
};

double Train(const Variant& config, const std::vector<ae::SaliencyData>& pools, int epochs,
             const std::filesystem::path& root, const std::string& init) {
  const auto directory = root / config.name;
  std::filesystem::create_directories(directory);
  ae::SparseAutoencoderModel model(config.l1, config.kl);
  if (!init.empty()) model.parameters.load(init);
  check::Require(model.parameters.count() == 101200 && model.parameters.size() == 4,
                 "independent784x64+64+64x784+784 inventory");
  nn::save_npz((directory / "initial_arrays.npz").string(), ae::Arrays(model.parameters));
  nn::BatchOrder order(1000, 32, 1607);
  std::ofstream  history(directory / "history.tsv"), batches(directory / "batches.tsv");
  history.precision(17);
  history << "epoch\tfit_preupdate_loss\tvalidation_loss\tvalidation_mse\tvalidation_l1\tvalidation_kl\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto fit_source = pools[0].source.to_vector<int>();
  const auto best_path  = (directory / "best.npz").string();
  double     best       = std::numeric_limits<double>::infinity();
  int        selected = 0, step = 0;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    double sum  = 0;
    int    seen = 0;
    for (;;) {
      auto rows = order.next();
      if (rows.empty()) break;
      auto images = nn::take_rows(pools[0].images, rows);
      // Forward: reconstruction MSE + L1/KL activation penalties of this batch.
      auto loss = ae::SparseAutoencoder(model.parameters, ae::Flatten(images), model.l1_weight, model.kl_weight).loss;
      if (step == 0) {
        nn::GradOptions keep;
        keep.retain_graph = true;
        nn::save_npz((directory / "first_gradients.npz").string(),
                     ae::CheckGradients(model.parameters, nn::grad(loss, model.parameters.parameters(), keep),
                                        "finite all penalized parameter gradients",
                                        "all4tensors receive complete loss gradients"));
        nn::save_npz((directory / "first_batch.npz").string(), {{"images", images}});
      }
      const double value = loss.item<float>();
      check::Require(std::isfinite(value) && value >= 0, "finite pre-update regularized loss");
      model.optimizer->minimize(loss);
      sum += double(rows.size()) * value;
      seen += int(rows.size());
      for (auto row : rows) batches << epoch << '\t' << step << '\t' << row << '\t' << fit_source[row] << '\n';
      if (step == 0) nn::save_npz((directory / "first_update_arrays.npz").string(), ae::Arrays(model.parameters));
      ++step;
    }
    check::Require(seen == 1000, "each fitting image once per variant epoch");
    auto         validation = model.Evaluate(pools[1].images);
    const double loss       = validation[2].item<float>();
    history << epoch << '\t' << sum / seen;
    for (int i = 2; i < 6; ++i) history << '\t' << validation[i].item<float>();
    history << '\n';
    nn::save_npz((directory / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(),
                 ae::Arrays(model.parameters));
    if (loss < best) {
      best     = loss;
      selected = epoch;
      model.parameters.save(best_path);  // validation own full regularized objective selects variant checkpoint
    }
    std::cout << config.name << " epoch" << epoch << ":validation total/MSE/L1/KL=" << loss << '/'
              << validation[3].item<float>() << '/' << validation[4].item<float>() << '/'
              << validation[5].item<float>() << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  ae::SparseAutoencoderModel deployed(config.l1, config.kl, false);
  deployed.parameters.load(best_path);  // fresh restore matched sparse variant
  const auto       before = ae::Snapshot(deployed.parameters);
  nn::NamedTensors arrays;
  for (int split = 0; split < 3; ++split) {
    const std::string name   = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    auto              result = deployed.Evaluate(pools[split].images);
    arrays.emplace_back(name + "_code", result[0]);
    arrays.emplace_back(name + "_reconstruction", result[1]);
    arrays.emplace_back(name + "_unit_mean", result[6]);
    std::vector<float> components;
    for (int i = 0; i < 4; ++i) components.push_back(result[i + 2].item<float>());
    arrays.emplace_back(name + "_components", nn::from_vector(components, {4}));
    if (split == 1) check::Near(result[2].item<float>(), best, 1e-7, "selected regularized validation loss restored");
  }
  auto test = deployed.Evaluate(pools[2].images);
  check::Require(ae::Bytes(test[1]) == ae::Bytes(ae::Find(arrays, "test_reconstruction")),
                 "deterministic sparse reconstruction");
  check::Require(ae::Bytes(deployed.Decode(test[0])) == ae::Bytes(test[1]), "explicit sparse encode-decode exact");
  auto         one   = deployed.Evaluate(ae::Rows(pools[2], {0}).images);
  const double error = ae::MaxAbsDiff(one[1], test[1], 784);
  check::Require(error < 1e-6, "inference single/batch parity despite batch-dependent KL objective");
  check::Require(before == ae::Snapshot(deployed.parameters), "all4sparse parameters immutable during evaluation");
  nn::save_npz((directory / "outputs.npz").string(), arrays);
  nn::save_npz((directory / "model_arrays.npz").string(), ae::Arrays(deployed.parameters));
  const auto code = ae::Find(arrays, "test_code").contiguous().to_vector<float>();
  double     mean = 0;
  int        low  = 0;
  for (float v : code) {
    mean += v;
    low += v < .1f;
  }
  mean /= double(code.size());
  const double  mse = ae::Mse(test[1], pools[2].images);
  std::ofstream metrics(directory / "metrics.json");
  metrics.precision(17);
  metrics << "{\"l1_weight\":" << config.l1 << ",\"kl_weight\":" << config.kl << ",\"rho\":" << double(.05f)
          << ",\"epsilon\":" << double(1e-5f) << ",\"epochs\":" << epochs << ",\"selected_epoch\":" << selected
          << ",\"updates\":" << step << ",\"test_mse\":" << mse << ",\"test_mean_activation\":" << mean
          << ",\"test_fraction_below_point1\":" << double(low) / double(code.size())
          << ",\"single_batch_max_error\":" << error << ",\"training_seconds\":" << seconds << "}\n";
  check::Require(bool(metrics) && bool(history) && bool(batches), "write pervariant metrics and training provenance");
  std::cout << config.name << " selected" << selected << ":testMSE=" << mse << ",meanactivation=" << mean
            << ",fractionbelow.1=" << double(low) / double(code.size())
            << ";freshrestore/decode/repeat/batch/immutable4weights passed\n";
  return seconds;
}

void Run(const std::filesystem::path& data, const std::filesystem::path& directory, int epochs,
         const std::string& init) {
  auto pools = ae::LoadSaliency(data);
  const std::vector<Variant> variants{{"plain", 0, 0}, {"l1", .05f, 0}, {"kl", 0, .05f}, {"combined", .05f, .05f}};
  double                     seconds = 0;
  for (const auto& v : variants) seconds += Train(v, pools, epochs, directory, init);
  const auto original = nn::load_npz((directory / "plain" / "initial_arrays.npz").string());
  for (const auto& v : variants) {
    const auto initial = nn::load_npz((directory / v.name / "initial_arrays.npz").string());
    for (const auto& [key, value] : original)
      check::Require(ae::Bytes(ae::Find(initial, key)) == ae::Bytes(value), "ALL4variants identical initial parameters");
  }
  ae::SaveData(directory / "data.npz", pools);
  ae::CopyManifests(data, directory);
  const auto    mean = ae::Constant(ae::PixelMean(pools[0].images), 200);
  std::ofstream html(directory / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>L1 and KL sparse autoencoders</title><style>body{font:16px "
          "system-ui;max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:"
          "25px 0}</style><h1>Sparse autoencoders:activation L1 and KL</h1><p>Original,plain,L1,KL,combined,fit-only "
          "mean. ALL use784->64sigmoid->784sigmoid/101200parameters,identical initialization and batch order,"
       << epochs
       << "epochs PER variant. Fixed[0,1]intensity/nearest4,no quality-filtering. Labels do not enter training or "
          "selection.</p><p>Total=MSE+lambda*mean(abs(code))+beta*mean_unit KL(rho||mean_batch(code)). rho=.05,"
          "lambda=.05 for L1,beta=.05 for KL. Unit means clamp to[1e-5,1-1e-5]before logs;clamp derivative is flat "
          "outside. KL is a penalty on each unit's BATCH mean,not each image's activation,not a softmax/class KL. "
          "Minibatch KL and full-pool KL differ. Each variant selects on its OWN full validation objective,not "
          "comparable objective numbers across penalties.</p><p>Sigmoid codes are generally positive:small activation "
          "is not exact zero sparsity. Mean activation and fraction below.1 are descriptive,threshold not used in "
          "optimization. Stronger sparsity can worsen reconstruction or collapse features;three short epochs do not "
          "guarantee rho is reached. MSE alone can favor blurry averages. This is not a proof of semantic features,"
          "denoising,generative modeling or bit-rate compression.</p>";
  std::vector<nn::Tensor> panels{pools[2].images};
  for (const auto& v : variants) {
    const auto raw        = nn::load_npz((directory / v.name / "outputs.npz").string());
    const auto components = ae::Find(raw, "test_components").to_vector<float>();
    html << "<p>" << v.name << ":test total/MSE/L1/KL=" << components[0] << '/' << components[1] << '/'
         << components[2] << '/' << components[3]
         << ". Mean activation equals L1 for these nonnegative sigmoid codes. rho=.05 remains a target rather than an "
            "attained guarantee;per-unit means/codes and threshold metrics are saved in variant artifacts.</p>";
    panels.push_back(ae::Find(raw, "test_reconstruction"));
  }
  panels.push_back(mean);
  ae::DigitGallery(directory, pools[2], panels, html);
  check::Require(bool(html), "write complete sparse gallery");
  std::cout << "ALL4variants identicalinitial/sourceorder,96updates each at default3epochs. L1/KL tradeoffs saved "
               "without superiority claim.\ntraining time "
            << seconds << " s\nArtifacts:" << directory.string() << "/index.html\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::string           init;
    std::filesystem::path data = paths::Dataset("saliency"), out = paths::OutputRoot() / "models/sparse_autoencoder";
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
    Checks();
    Run(data, out, epochs, init);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
