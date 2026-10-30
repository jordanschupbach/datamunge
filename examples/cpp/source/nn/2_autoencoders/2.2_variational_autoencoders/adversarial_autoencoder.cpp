// CMD: just run-nn adversarial_autoencoder
// Compact deterministic adversarial autoencoder on the tiny known-factor lattice. Three cheap
// default epochs.
//
// Port of tflow's adversarial_autoencoder. Preparation (in ~/projects/tflow_examples):
// just beta-vae-assets; data are read from $DATAMUNGE_DATA_DIR/beta_vae. Outputs go to
// output/nn/models/adversarial_autoencoder. Checkpoints are best.npz (all three groups, "<k>/<name>")
// instead of SaveV2 bundles. Options: epochs=N data=PATH out=PATH init=PATH (tflow's
// initial_arrays.npz, to follow its trajectory exactly).

#include <chrono>
#include <iostream>
#include <limits>

#include "common/adversarial_autoencoder.hpp"

namespace {

namespace nn = datamunge::nn;

double Difference(const nn::Tensor& a, const nn::Tensor& b) {
  check::Require(a.shape() == b.shape(), "matching outputs");
  return ae::MaxAbsDiff(a, b, a.numel());
}

nn::NamedTensors Prefixed(const nn::NamedTensors& in, const std::string& prefix) {
  nn::NamedTensors out;
  for (const auto& [key, value] : in) out.emplace_back(prefix + key, value);
  return out;
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  auto pools = ae::LoadBetaFactors(data);
  std::filesystem::create_directories(root);
  nn::NamedTensors              bank;
  std::vector<ae::VaeNoiseBank> priors;
  for (int k = 0; k < 3; ++k) {
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    priors.push_back(ae::BetaVaeNoise(pools[k].images.size(0), 1, 3541 + k * 6));
    bank.emplace_back(split + "_standard", priors[k].standard);
    bank.emplace_back(split + "_epsilon", priors[k].epsilon);
  }
  auto generation = ae::BetaVaeNoise(20, 1, 3571);
  bank.emplace_back("generation_standard", generation.standard);
  bank.emplace_back("generation_epsilon", generation.epsilon);
  ae::SaveFactorData(root, pools);
  nn::save_npz((root / "fixed_noise.npz").string(), bank);
  ae::AdversarialAutoencoderModel model;
  if (!init.empty()) model.Load(nn::load_npz(init));
  const std::int64_t parameters = model.Count();
  const std::size_t  tensors    = model.Tensors();
  check::Require(parameters == 26133 && tensors == 12, "AAE encoder/decoder/latent-discriminator census");
  nn::save_npz((root / "initial_arrays.npz").string(), model.Arrays());
  nn::BatchOrder order(210, 32, 3581);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tvalidation_nll\tvalidation_latent_discriminator_loss\tvalidation_encoder_adversarial_loss\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto fit_source = pools[0].source.to_vector<int>();
  const auto best_path  = (root / "best.npz").string();
  double     best       = std::numeric_limits<double>::infinity();
  int        selected = 0, steps = 0;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    auto noise = ae::BetaVaeNoise(210, 1, 3593 + epoch);
    nn::save_npz((root / ("train_noise_epoch" + std::to_string(epoch) + ".npz")).string(),
                 {{"standard", noise.standard}, {"epsilon", noise.epsilon}});
    auto prior = ae::BetaVaeNoiseAt(noise, 0);
    order.begin_epoch();
    int seen = 0;
    for (;;) {
      auto rows = order.next();
      if (rows.empty()) break;
      auto x = nn::take_rows(pools[0].images, rows), p = nn::take_rows(prior, rows);
      if (!steps) {
        nn::save_npz((root / "first_batch.npz").string(), {{"images", x}, {"prior", p}});
        nn::save_npz((root / "first_forward.npz").string(), model.Evaluate(x, p));
        for (int k = 0; k < 4; ++k)
          nn::save_npz((root / ("initial_gradients_group" + std::to_string(k) + ".npz")).string(), model.Gradient(k, x, p));
      }
      for (int phase = 0; phase < 3; ++phase) {
        const auto before = steps ? nn::NamedTensors{} : model.Arrays();
        if (!steps) {
          if (!phase)
            for (int k = 0; k < 2; ++k)
              nn::save_npz((root / ("first_actual_gradients_group" + std::to_string(k) + ".npz")).string(),
                           model.Gradient(k, x, p));
          else {
            const int k = phase == 1 ? 2 : 3;
            nn::save_npz((root / ("first_actual_gradients_group" + std::to_string(k) + ".npz")).string(),
                         model.Gradient(k, x, p));
          }
        }
        model.Update(phase, x, p);
        if (!steps) {
          auto after   = model.Arrays();
          bool changed = false;
          for (const auto& [key, value] : before) {
            const bool intended = phase == 0 ? key[0] != '2' : phase == 1 ? key[0] == '2' : key[0] == '0';
            if (intended)
              changed |= ae::Bytes(value) != ae::Bytes(ae::Find(after, key));
            else
              check::Require(ae::Bytes(value) == ae::Bytes(ae::Find(after, key)),
                             "AAE phase leaves unrelated groups BITEXACT");
          }
          check::Require(changed, "AAE phase changes intended weights");
          nn::save_npz((root / ("first_after_phase" + std::to_string(phase) + ".npz")).string(), after);
        }
      }
      for (auto row : rows) batches << epoch << '\t' << steps << '\t' << row << '\t' << fit_source[row] << '\n';
      ++steps;
      seen += int(rows.size());
    }
    check::Require(seen == 210, "all fit images once per epoch");
    auto         v     = model.Evaluate(pools[1].images, ae::BetaVaeNoiseAt(priors[1], 0));
    const double score = ae::Find(v, "reconstruction_loss").item<float>();
    const double dl = ae::Find(v, "discriminator_loss").item<float>(), al = ae::Find(v, "adversarial_loss").item<float>();
    history << epoch << '\t' << score << '\t' << dl << '\t' << al << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), model.Arrays());
    if (score < best) {
      best     = score;
      selected = epoch;
      nn::save_npz(best_path, model.Arrays());  // validation reconstruction NLL selects all AAE groups
    }
    std::cout << "epoch" << epoch << ":validationNLL=" << score << ",latentD=" << dl << ",encoderAdv=" << al << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  ae::AdversarialAutoencoderModel deployed(false);
  deployed.Load(nn::load_npz(best_path));  // fresh AAE all-group model-only restore
  check::Near(ae::Find(deployed.Evaluate(pools[1].images, ae::BetaVaeNoiseAt(priors[1], 0)), "reconstruction_loss")
                  .item<float>(),
              best, 1e-8, "restored validation score");
  const auto       before = deployed.Snapshot();
  nn::NamedTensors outputs;
  for (int k = 0; k < 3; ++k) {
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    auto              f     = deployed.Evaluate(pools[k].images, ae::BetaVaeNoiseAt(priors[k], 0));
    auto              more  = Prefixed(f, split + "_");
    outputs.insert(outputs.end(), more.begin(), more.end());
  }
  auto p    = ae::BetaVaeNoiseAt(priors[2], 0);
  auto test = deployed.Evaluate(pools[2].images, p), repeat = deployed.Evaluate(pools[2].images, p),
       other = deployed.Evaluate(pools[2].images, ae::BetaVaeNoiseAt(ae::BetaVaeNoise(42, 1, 3613), 0));
  auto B     = [](const nn::NamedTensors& m, const std::string& k) { return ae::Bytes(ae::Find(m, k)); };
  check::Require(B(test, "probability") == B(repeat, "probability") &&
                     B(test, "probability") == ae::Bytes(deployed.Decode(ae::Find(test, "code"))),
                 "repeat and code-only decoder BITEXACT");
  check::Require(B(test, "code") == B(other, "code") && B(test, "probability") == B(other, "probability") &&
                     B(test, "prior_logit") != B(other, "prior_logit"),
                 "prior samples affect discriminator ONLY,deterministic reconstruction");
  auto single = deployed.Evaluate(nn::take_rows(pools[2].images, {0}), nn::take_rows(p, {0}));
  const double parity =
      Difference(ae::Find(single, "probability"), nn::take_rows(ae::Find(test, "probability"), {0}));
  check::Require(parity < 2e-6, "AAE single/batch parity");
  auto prior_z = ae::BetaVaeNoiseAt(generation, 0), prior = deployed.Decode(prior_z);
  nn::save_npz((root / "prior.npz").string(), {{"z", prior_z}, {"probability", prior}});
  nn::save_npz((root / "model_arrays.npz").string(), deployed.Arrays());
  nn::save_npz((root / "outputs.npz").string(), outputs);
  check::Require(before == deployed.Snapshot(), "AAE inference preserves all weights");
  auto         mean     = ae::Constant(ae::PixelMean(pools[0].images), 42);
  const auto   test_p   = ae::Find(test, "probability");
  const double mse      = ae::Mse(test_p, pools[2].images), baseline = ae::Mse(mean, pools[2].images);
  const double test_nll = ae::Find(test, "reconstruction_loss").item<float>();
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":" << parameters << ",\"tensors\":" << tensors << ",\"epochs\":" << epochs
       << ",\"steps_per_optimizer\":" << steps << ",\"optimizer_updates\":" << 4 * steps
       << ",\"selected_epoch\":" << selected << ",\"test_nll\":" << test_nll << ",\"test_mse\":" << mse
       << ",\"fit_mean_mse\":" << baseline
       << ",\"test_discriminator_loss\":" << ae::Find(test, "discriminator_loss").item<float>()
       << ",\"test_encoder_adversarial_loss\":" << ae::Find(test, "adversarial_loss").item<float>()
       << ",\"single_batch_max_error\":" << parity << ",\"training_seconds\":" << seconds << "}\n";
  ae::Mosaic(root / "prior_means.png", prior, 5, 3);
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Adversarial autoencoder</title><style>body{font:16px "
          "system-ui;margin:30px;max-width:1100px}img{max-width:100%;image-rendering:pixelated}</style><h1>Adversarial "
          "autoencoder</h1><p><a href='https://arxiv.org/abs/1511.05644'>Makhzani et al.</a> compact deterministic "
          "AAE:encoder784-16tanh-4,decoder4-16tanh-784Bernoulli,latentdiscriminator4-16tanh-1. Match aggregate encoded "
          "distribution to N0I using discriminator real=prior/fake=encoded. No per-example Gaussian KL or posterior "
          "epsilon. Joint reconstruction BCE updates E+G from SAME gradient snapshot;then D update;then encoder "
          "non-saturating adversarial update. Four independent Adam states (E-reconstruction,G-reconstruction,D,"
          "E-adversarial);common reconstruction gradient barrier ensures no cross-group write race. Three default "
          "epochs21steps each,84optimizer updates.</p><p>294generated binary circle/square lattice "
          "images,210fit42validation42test,validation reconstructionNLL selects epoch"
       << selected
       << ". Fresh all-group restore,phase isolation,prior-independent deterministic encoding,repeated/code-only "
          "decode,batch/immutable checks pass. Discriminator labels do not prove aggregate matching;finite capacity and "
          "tiny training leave mismatch. No exact likelihood/ELBO or disentanglement claim.</p><p>TestNLL="
       << test_nll << ",MSE=" << mse << ",fit-onlymeanMSE=" << baseline
       << ". Ten evenly spaced heldout cases,unfiltered:original/reconstruction/fit-onlymean,fixed0..1gray. Sigmoid "
          "outputs are probabilities,not sampled pixels.</p>";
  const auto source = pools[2].source.to_vector<int>();
  const std::vector<nn::Tensor> values{ae::Flatten(pools[2].images), test_p, mean};
  for (int k = 0; k < 10; ++k) {
    const int               b = k * 41 / 9;
    std::vector<nn::Tensor> panels;
    for (const auto& v : values) panels.push_back(ae::Row(v, b));
    const std::string name = "case_" + std::to_string(k) + ".png";
    ae::Mosaic(root / name, nn::stack(panels, 0), 3, 3);
    html << "<p>Source" << source[b] << "<br><img src='" << name << "'></p>";
  }
  html << "<h2>20unfiltered N0I prior decoder probabilities</h2><img src='prior_means.png'><p>Short-run weaknesses "
          "retained,no quality improvement or aggregate-prior matching certification.</p>";
  check::Require(bool(meta) && bool(html) && bool(history) && bool(batches), "complete AAE artifacts");
  std::cout << "params=" << parameters << ",tensors=" << tensors << ",selected" << selected << ",testNLL=" << test_nll
            << ",MSE=" << mse << ",mean=" << baseline
            << ";reconstructionBarrier/phaseIsolation/restore/priorIndependent/codeOnly/batch/immutable PASS\n"
            << "training time " << seconds << " s\nArtifacts:" << root.string() << "/index.html" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::string           init;
    std::filesystem::path data = paths::Dataset("beta_vae"), out = paths::OutputRoot() / "models/adversarial_autoencoder";
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
