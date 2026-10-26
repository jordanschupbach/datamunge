// CMD: just run-nn variational_autoencoder
// Diagonal-Gaussian VAE with explicit reparameterized noise and Bernoulli negative ELBO.
// Static MNIST threshold 128, three epochs.
//
// Port of tflow's variational_autoencoder. Preparation (in ~/projects/tflow_examples):
// just autoencoder-assets; data are read from $DATAMUNGE_DATA_DIR/saliency. Outputs
// (history, npz arrays, metrics.json, index.html, digit<k>.png, prior_means.png and
// latent_grid.png) go to output/nn/models/variational_autoencoder. Options: epochs=N
// data=PATH out=PATH init=PATH (start from saved parameters, e.g. tflow's
// initial_arrays.npz, to follow its trajectory exactly).

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>

#include "common/autoencoder.hpp"

namespace {

namespace nn = datamunge::nn;

void Checks() {
  const std::vector<float> m{.2f, -.4f, .1f, .3f}, l{0.f, float(std::log(4.)), float(std::log(.25)), .2f},
      e{1.f, -.5f, 2.f, -1.f};
  auto mu = nn::from_vector(m, {2, 2}).requires_grad_(), lv = nn::from_vector(l, {2, 2}).requires_grad_(),
       eps = nn::from_vector(e, {2, 2}).requires_grad_();
  auto z = ae::VaeReparameterize(mu, lv, eps), kl = ae::VaeKl(mu, lv);
  auto loss = nn::mean(0.5 * nn::sum(nn::square(z), {1}) + kl, {0});
  auto grad = nn::grad(loss, {mu, lv, eps});
  auto extreme = ae::VaePixelNll(nn::tensor({{100., -100., 100., -100.}}), nn::tensor({{1., 0., 0., 1.}}));
  auto zero_kl = ae::VaeKl(nn::zeros({1, 2}), nn::zeros({1, 2}));
  const auto zs = z.to_vector<float>(), kls = kl.to_vector<float>(), g0 = grad[0].to_vector<float>(),
             g1 = grad[1].to_vector<float>(), g2 = grad[2].to_vector<float>();
  for (int b = 0; b < 2; ++b) {
    double k = 0;
    for (int j = 0; j < 2; ++j) {
      const int    i     = b * 2 + j;
      const double mean = m[i], logv = l[i], noise = e[i], sigma = std::exp(.5 * logv), value = mean + sigma * noise;
      k += .5 * (mean * mean + std::exp(logv) - 1 - logv);
      check::Near(zs[i], value, 2e-7, "all4reparameterized values");
      check::Near(g0[i], (value + mean) / 2, 2e-7, "all4mu derivatives include pathwise PLUS KL");
      check::Near(g1[i], (.5 * value * sigma * noise + .5 * (std::exp(logv) - 1)) / 2, 3e-7,
                  "all4logvariance derivatives include pathwise PLUS KL");
      check::Near(g2[i], value * sigma / 2, 3e-7, "all4epsilon derivatives follow sigma");
    }
    check::Near(kls[b], k, 2e-7, "all2analytic Gaussian KLs");
  }
  check::Near(extreme.at({0}), 200, 1e-6, "stable BCE for logits+-100");
  check::Near(zero_kl.at({0}), 0, 0, "q=N(0,I)KL exactly0");
  std::cout << "known4pathwise values/2KL/ALL12mu-logvariance-epsilon derivatives/zeroKL/extremeBCE passed\n";
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  auto pools = ae::LoadSaliency(data);
  std::filesystem::create_directories(root);
  nn::NamedTensors saved_data;
  for (int split = 0; split < 3; ++split) {
    const std::string name = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    saved_data.emplace_back(name + "_original_images", pools[split].images);
    pools[split].images = nn::cast(nn::greater_equal(pools[split].images, .5), nn::DType::Float32).contiguous();
    saved_data.emplace_back(name + "_images", pools[split].images);
    saved_data.emplace_back(name + "_source_ids", pools[split].source);
  }
  saved_data.emplace_back("test_labels", pools[2].labels);
  nn::save_npz((root / "data.npz").string(), saved_data);
  ae::CopyManifests(data, root);
  auto val_noise = ae::VaeNoise(200, 4, 1987), test_noise = ae::VaeNoise(200, 4, 1989),
       fit_noise = ae::VaeNoise(1000, 4, 1991);
  nn::save_npz((root / "fixed_noise.npz").string(), {{"validation_standard", val_noise.standard},
                                                     {"validation_epsilon", val_noise.epsilon},
                                                     {"test_standard", test_noise.standard},
                                                     {"test_epsilon", test_noise.epsilon},
                                                     {"fit_standard", fit_noise.standard},
                                                     {"fit_epsilon", fit_noise.epsilon}});
  ae::VariationalAutoencoderModel model;
  if (!init.empty()) model.parameters.load(init);
  check::Require(model.parameters.size() == 10 && model.parameters.count() == 101652,
                 "twohead diagonalGaussian parameter census");
  nn::save_npz((root / "initial_arrays.npz").string(), ae::Arrays(model.parameters));
  auto           initial = ae::VaeScore(model, pools[2].images, test_noise);
  nn::BatchOrder order(1000, 32, 1957);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tfit_preupdate_negative_elbo\tvalidation_reconstruction_nll\tvalidation_kl\tvalidation_"
             "negative_elbo\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto fit_source = pools[0].source.to_vector<int>();
  const auto best_path  = (root / "best.npz").string();
  double     best       = std::numeric_limits<double>::infinity();
  int        selected = 0, step = 0;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    auto noise = ae::VaeNoise(1000, 1, 1973 + epoch);  // fresh explicit epsilon every epoch
    nn::save_npz((root / ("train_noise_epoch" + std::to_string(epoch) + ".npz")).string(),
                 {{"standard", noise.standard}, {"epsilon", noise.epsilon}});
    auto epsilon = ae::VaeNoiseAt(noise, 0);
    order.begin_epoch();
    double sum  = 0;
    int    seen = 0;
    for (;;) {
      auto rows = order.next();
      if (rows.empty()) break;
      auto input = nn::take_rows(pools[0].images, rows), e = nn::take_rows(epsilon, rows);
      if (step == 0) {
        auto g = ae::CheckGradients(model.parameters, model.Gradient(input, e),
                                    "all posterior/decoder negativeELBO derivatives finite",
                                    "ALL10posterior/decoder tensors receive gradients");
        nn::save_npz((root / "first_gradients.npz").string(), g);
        nn::save_npz((root / "first_batch.npz").string(), {{"images", input}, {"epsilon", e}});
      }
      sum += double(rows.size()) * model.Update(input, e);
      seen += int(rows.size());
      for (auto row : rows) batches << epoch << '\t' << step << '\t' << row << '\t' << fit_source[row] << '\n';
      if (step == 0) nn::save_npz((root / "first_update_arrays.npz").string(), ae::Arrays(model.parameters));
      ++step;
    }
    check::Require(seen == 1000, "allfittingrecords once per epoch");
    auto score = ae::VaeScore(model, pools[1].images, val_noise);
    history << epoch << '\t' << sum / seen << '\t' << score[0] << '\t' << score[1] << '\t' << score[2] << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), ae::Arrays(model.parameters));
    if (score[2] < best) {
      best     = score[2];
      selected = epoch;
      model.parameters.save(best_path);  // fixed four-sample validation negative ELBO selects the epoch
    }
    std::cout << "VAE epoch" << epoch << ":validation NLL=" << score[0] << ",KL=" << score[1]
              << ",negativeELBO=" << score[2] << std::endl;
  }
  const double train_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  ae::VariationalAutoencoderModel deployed(false);
  deployed.parameters.load(best_path);  // fresh posterior/prior inference restore
  const auto before = ae::Snapshot(deployed.parameters);
  check::Near(ae::VaeScore(deployed, pools[1].images, val_noise)[2], best, 1e-8,
              "fresh selected MC validation objective");
  nn::NamedTensors                output;
  std::vector<ae::VaeNoiseBank>   banks{fit_noise, val_noise, test_noise};
  for (int split = 0; split < 3; ++split) {
    const std::string       name = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    const std::int64_t      n    = pools[split].images.size(0);
    auto                    mean = deployed.Evaluate(pools[split].images, nn::zeros({n, 2}));
    check::Require(ae::Bytes(mean[2]) == ae::Bytes(mean[0]), "epsilon0 means z=mu EXACT");
    std::vector<nn::Tensor> sample_z, prediction, nll;
    for (int k = 0; k < 4; ++k) {
      auto draw = deployed.Evaluate(pools[split].images, ae::VaeNoiseAt(banks[split], k));
      check::Require(ae::Bytes(draw[0]) == ae::Bytes(mean[0]) && ae::Bytes(draw[1]) == ae::Bytes(mean[1]),
                     "epsilon changes samples not posterior parameters");
      sample_z.push_back(draw[2]);
      prediction.push_back(draw[3]);
      nll.push_back(draw[4]);
    }
    output.emplace_back(name + "_mu", mean[0]);
    output.emplace_back(name + "_log_variance", mean[1]);
    output.emplace_back(name + "_mean_reconstruction", mean[3]);
    output.emplace_back(name + "_kl", mean[5]);
    output.emplace_back(name + "_sample_z", nn::stack(sample_z, 0));
    output.emplace_back(name + "_sample_prediction", nn::stack(prediction, 0));
    output.emplace_back(name + "_sample_nll", nn::stack(nll, 0));
  }
  auto test   = deployed.Evaluate(pools[2].images, ae::VaeNoiseAt(test_noise, 0)),
       repeat = deployed.Evaluate(pools[2].images, ae::VaeNoiseAt(test_noise, 0));
  check::Require(ae::Bytes(test[2]) == ae::Bytes(repeat[2]) && ae::Bytes(test[3]) == ae::Bytes(repeat[3]) &&
                     ae::Bytes(deployed.Decode(test[2])) == ae::Bytes(test[3]),
                 "fixed epsilon repeatability and target-free exact sampled decode");
  auto other = deployed.Evaluate(pools[2].images, ae::VaeNoiseAt(test_noise, 1));
  check::Require(ae::Bytes(test[2]) != ae::Bytes(other[2]) && ae::Bytes(test[3]) != ae::Bytes(other[3]),
                 "second epsilon changes latent samples AND decoder probabilities");
  auto single = deployed.Evaluate(ae::Rows(pools[2], {0}).images,
                                  nn::take_rows(ae::VaeNoiseAt(test_noise, 0), {0}));
  const double parity = ae::MaxAbsDiff(single[3], test[3], 784);
  check::Require(parity < 2e-6, "posterior single/batch parity");
  auto zero_latent = deployed.Decode(nn::zeros({200, 2}));
  output.emplace_back("zero_latent_prediction", zero_latent);
  auto prior_noise = ae::VaeNoise(20, 1, 1997);
  auto prior_z = ae::VaeNoiseAt(prior_noise, 0), prior = deployed.Decode(prior_z);
  std::vector<float> grid_values;
  for (int row = 0; row < 9; ++row)
    for (int col = 0; col < 9; ++col) {
      grid_values.push_back(-2.f + .5f * col);
      grid_values.push_back(2.f - .5f * row);
    }
  auto grid_z = nn::from_vector(grid_values, {81, 2});
  auto grid   = deployed.Decode(grid_z);
  nn::save_npz((root / "prior.npz").string(), {{"standard", prior_noise.standard},
                                               {"z", prior_z},
                                               {"probability", prior},
                                               {"grid_z", grid_z},
                                               {"grid_probability", grid}});
  const double gray = 220.0 / 255.0;
  nn::save_image_grid(nn::reshape(prior, {20, 28, 28, 1}), (root / "prior_means.png").string(), 5, 0, 1, 4, gray, 3);
  nn::save_image_grid(nn::reshape(grid, {81, 28, 28, 1}), (root / "latent_grid.png").string(), 9, 0, 1, 4, gray, 2);
  check::Require(before == ae::Snapshot(deployed.parameters),
                 "posterior evaluation/prior generation leaves ALL10parameters immutable");
  nn::save_npz((root / "outputs.npz").string(), output);
  nn::save_npz((root / "model_arrays.npz").string(), ae::Arrays(deployed.parameters));
  auto         score    = ae::VaeScore(deployed, pools[2].images, test_noise);
  const auto   mean_rec = ae::Find(output, "test_mean_reconstruction");
  const double mean_mse = ae::Mse(mean_rec, pools[2].images);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":101652,\"epochs\":" << epochs << ",\"updates\":" << step
       << ",\"selected_epoch\":" << selected << ",\"mc_samples\":4,\"initial_test_negative_elbo\":" << initial[2]
       << ",\"test_reconstruction_nll\":" << score[0] << ",\"test_kl\":" << score[1]
       << ",\"test_negative_elbo\":" << score[2] << ",\"mean_decode_test_mse\":" << mean_mse
       << ",\"single_batch_max_error\":" << parity << "}\n";
  auto       fit_mean   = ae::Constant(ae::PixelMean(pools[0].images), 200);
  const auto predictions = ae::Find(output, "test_sample_prediction");
  const auto binary      = ae::VariationalAutoencoderModel::Flatten(pools[2].images);
  const std::vector<nn::Tensor> panels{binary,
                                       mean_rec,
                                       nn::select(predictions, 0, 0),
                                       nn::select(predictions, 0, 1),
                                       fit_mean,
                                       nn::abs(mean_rec - binary),
                                       zero_latent};
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Diagonal Gaussian VAE</title><style>body{font:16px "
          "system-ui;max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}</style><h1>VAE: "
          "explicit pathwise Gaussian samples</h1><p>Binary input, decode(mu), posterior draw 1 probabilities, draw 2 "
          "probabilities, fit-only mean, absolute decode(mu) error, zero-latent probabilities. decode(mu) is not the "
          "posterior predictive mean. First test record per digit, unfiltered; fixed [0,1] gray/nearest 4x/no error "
          "amplification.</p><p>Original MNIST statically thresholded at uint8>=128, not a claim to the official "
          "binarized MNIST benchmark. Encoder 784->64 ReLU, mu/log-variance 2 heads, unclipped "
          "z=mu+exp(.5 logvariance)*epsilon, decoder 2->64 ReLU->784 logits. Analytic KL to N(0,I); sum of 784 pixel "
          "NLL + sum of 2 latent KL, mean over the batch, beta 1, no KL annealing. Explicit standard Gaussian float32 "
          "eps changes each training epoch; validation selects a four-sample MC negative ELBO with fixed noise. Not "
          "an exact log likelihood or importance-weighted bound.</p><p>"
       << epochs << " epochs/101652 parameters/Adam .003/batch 32. Test NLL=" << score[0] << ", KL=" << score[1]
       << ", negative ELBO=" << score[2] << " nats/image. decode(mu) MSE=" << mean_mse
       << ", fit-only binary mean MSE=" << ae::Mse(fit_mean, pools[2].images)
       << ". Short training/two latents blur or lose identity. <a href='https://arxiv.org/abs/1312.6114'>"
          "Kingma/Welling VAE</a>.</p>";
  const auto labels = pools[2].labels.to_vector<int>(), source = pools[2].source.to_vector<int>();
  for (int digit = 0; digit < 10; ++digit) {
    int b = 0;
    while (labels[b] != digit) ++b;
    std::vector<nn::Tensor> row;
    for (const auto& p : panels) row.push_back(ae::Row(p, b));
    const std::string file = "digit" + std::to_string(digit) + ".png";
    ae::PanelRow(root / file, row);
    html << "<h2>Digit " << digit << ", source " << source[b] << "</h2><img src='" << file << "'>";
  }
  html << "<h2>20 unfiltered standard-normal prior decoder means</h2><p>Probabilities, not Bernoulli pixel draws; no "
          "input or label feed.</p><img src='prior_means.png'><h2>Fixed 2D latent grid</h2><p>x from -2 to 2 "
          "left-right, y from 2 to -2 top-bottom, step .5. Fixed [0,1] gray, not normalized per image.</p><img "
          "src='latent_grid.png'>";
  check::Require(bool(meta) && bool(html) && bool(history) && bool(batches), "write complete VAE evidence");
  std::cout << "VAE:selected" << selected << ",testNLL=" << score[0] << ",KL=" << score[1]
            << ",negativeELBO=" << score[2] << ",decode(mu)MSE=" << mean_mse
            << ";freshrestore/reparameterization/decode/batch/prior/immutable passed\ntraining time " << train_seconds
            << " s\nArtifacts:" << root.string() << "/index.html\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::string           init;
    std::filesystem::path data = paths::Dataset("saliency"), out = paths::OutputRoot() / "models/variational_autoencoder";
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
