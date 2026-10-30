// CMD: just run-nn conditional_variational_autoencoder
// Class-conditioned Gaussian VAE with explicit reparameterized noise and Bernoulli negative ELBO.
// Static MNIST threshold 128, three epochs.
//
// Port of tflow's conditional_variational_autoencoder. Preparation (in ~/projects/tflow_examples):
// just autoencoder-assets; data are read from $DATAMUNGE_DATA_DIR/saliency. Outputs go to
// output/nn/models/conditional_variational_autoencoder. Checkpoints are best.npz instead of
// SaveV2 bundles. Options: epochs=N data=PATH out=PATH init=PATH (start from saved parameters,
// e.g. tflow's initial_arrays.npz, to follow its trajectory exactly).

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>

#include "common/beta_variational_autoencoder.hpp"
#include "common/conditional_variational_autoencoder.hpp"

namespace {

namespace nn = datamunge::nn;

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
    saved_data.emplace_back(name + "_labels", pools[split].labels);
  }
  nn::save_npz((root / "data.npz").string(), saved_data);
  ae::CopyManifests(data, root);
  auto val_noise = ae::VaeNoise(200, 4, 2131), test_noise = ae::VaeNoise(200, 4, 2137),
       fit_noise = ae::VaeNoise(1000, 4, 2141);
  nn::save_npz((root / "fixed_noise.npz").string(), {{"validation_standard", val_noise.standard},
                                                     {"validation_epsilon", val_noise.epsilon},
                                                     {"test_standard", test_noise.standard},
                                                     {"test_epsilon", test_noise.epsilon},
                                                     {"fit_standard", fit_noise.standard},
                                                     {"fit_epsilon", fit_noise.epsilon}});
  ae::ConditionalVariationalAutoencoderModel model;
  if (!init.empty()) model.parameters.load(init);
  check::Require(model.parameters.size() == 10 && model.parameters.count() == 102932,
                 "twohead diagonalGaussian parameter census");
  nn::save_npz((root / "initial_arrays.npz").string(), ae::Arrays(model.parameters));
  auto           initial = ae::ConditionalVaeScore(model, pools[2].images, pools[2].labels, test_noise);
  nn::BatchOrder order(1000, 32, 2113);
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
    auto noise = ae::VaeNoise(1000, 1, 2129 + epoch);
    nn::save_npz((root / ("train_noise_epoch" + std::to_string(epoch) + ".npz")).string(),
                 {{"standard", noise.standard}, {"epsilon", noise.epsilon}});
    auto epsilon = ae::VaeNoiseAt(noise, 0);
    order.begin_epoch();
    double sum  = 0;
    int    seen = 0;
    for (;;) {
      auto rows = order.next();
      if (rows.empty()) break;
      auto input = nn::take_rows(pools[0].images, rows), e = nn::take_rows(epsilon, rows),
           classes = nn::take_rows(pools[0].labels, rows);
      if (step == 0) {
        auto g = ae::CheckGradients(model.parameters, model.Gradient(input, classes, e),
                                    "all posterior/decoder negativeELBO derivatives finite",
                                    "ALL10posterior/decoder tensors receive gradients");
        nn::save_npz((root / "first_gradients.npz").string(), g);
        nn::save_npz((root / "first_batch.npz").string(), {{"images", input}, {"labels", classes}, {"epsilon", e}});
      }
      sum += double(rows.size()) * model.Update(input, classes, e);
      seen += int(rows.size());
      for (auto row : rows) batches << epoch << '\t' << step << '\t' << row << '\t' << fit_source[row] << '\n';
      if (step == 0) nn::save_npz((root / "first_update_arrays.npz").string(), ae::Arrays(model.parameters));
      ++step;
    }
    check::Require(seen == 1000, "allfittingrecords once per epoch");
    auto score = ae::ConditionalVaeScore(model, pools[1].images, pools[1].labels, val_noise);
    history << epoch << '\t' << sum / seen << '\t' << score[0] << '\t' << score[1] << '\t' << score[2] << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), ae::Arrays(model.parameters));
    if (score[2] < best) {
      best     = score[2];
      selected = epoch;
      model.parameters.save(best_path);  // fixed FOUR-sample validation negative ELBO selects epoch
    }
    std::cout << "VAE epoch" << epoch << ":validation NLL=" << score[0] << ",KL=" << score[1]
              << ",negativeELBO=" << score[2] << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  ae::ConditionalVariationalAutoencoderModel deployed(false);
  deployed.parameters.load(best_path);
  const auto before = ae::Snapshot(deployed.parameters);
  check::Near(ae::ConditionalVaeScore(deployed, pools[1].images, pools[1].labels, val_noise)[2], best, 1e-8,
              "fresh selected MC validation objective");
  nn::NamedTensors              output;
  std::vector<ae::VaeNoiseBank> banks{fit_noise, val_noise, test_noise};
  for (int split = 0; split < 3; ++split) {
    const std::string  name = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    const std::int64_t n    = pools[split].images.size(0);
    auto               mean = deployed.Evaluate(pools[split].images, pools[split].labels, nn::zeros({n, 2}));
    check::Require(ae::Bytes(mean[2]) == ae::Bytes(mean[0]), "epsilon0 means z=mu EXACT");
    std::vector<nn::Tensor> sample_z, prediction, nll;
    for (int k = 0; k < 4; ++k) {
      auto draw = deployed.Evaluate(pools[split].images, pools[split].labels, ae::VaeNoiseAt(banks[split], k));
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
  const auto& test_images = pools[2].images;
  const auto& test_labels = pools[2].labels;
  auto        test   = deployed.Evaluate(test_images, test_labels, ae::VaeNoiseAt(test_noise, 0)),
       repeat = deployed.Evaluate(test_images, test_labels, ae::VaeNoiseAt(test_noise, 0));
  check::Require(ae::Bytes(test[2]) == ae::Bytes(repeat[2]) && ae::Bytes(test[3]) == ae::Bytes(repeat[3]) &&
                     ae::Bytes(deployed.Decode(test[2], test_labels)) == ae::Bytes(test[3]),
                 "fixed epsilon repeatability and target-free exact sampled decode");
  auto other = deployed.Evaluate(test_images, test_labels, ae::VaeNoiseAt(test_noise, 1));
  check::Require(ae::Bytes(test[2]) != ae::Bytes(other[2]) && ae::Bytes(test[3]) != ae::Bytes(other[3]),
                 "second epsilon changes latent samples AND decoder probabilities");
  auto first  = ae::Rows(pools[2], {0});
  auto single = deployed.Evaluate(first.images, first.labels, nn::take_rows(ae::VaeNoiseAt(test_noise, 0), {0}));
  const double parity = ae::MaxAbsDiff(single[3], test[3], 784);
  check::Require(parity < 2e-6, "posterior single/batch parity");
  output.emplace_back("zero_latent_prediction", deployed.Decode(nn::zeros({200, 2}), test_labels));
  std::vector<int> shifted;
  for (int c : test_labels.to_vector<int>()) shifted.push_back((c + 1) % 10);
  auto shifted_labels = nn::from_vector(shifted, {200});
  auto wrong_mean     = deployed.Decode(ae::Find(output, "test_mu"), shifted_labels);
  auto wrong_sample   = deployed.Decode(test[2], shifted_labels);
  output.emplace_back("wrong_label_mean_prediction", wrong_mean);
  output.emplace_back("wrong_label_sample_prediction", wrong_sample);
  output.emplace_back("wrong_labels", shifted_labels);
  auto         changed_posterior = deployed.Evaluate(test_images, shifted_labels, ae::VaeNoiseAt(test_noise, 0));
  const double posterior_change  = ae::MaxAbsDiff(test[0], changed_posterior[0], 400);
  check::Require(posterior_change > 1e-5, "supplied class also affects posterior,not decoder-only conditioning");
  output.emplace_back("wrong_label_mu", changed_posterior[0]);
  output.emplace_back("wrong_label_log_variance", changed_posterior[1]);
  auto               prior_noise = ae::VaeNoise(5, 1, 2143);
  auto               base_z      = ae::VaeNoiseAt(prior_noise, 0);
  const auto         base        = base_z.to_vector<float>();
  std::vector<float> pz(100);
  std::vector<int>   pl(50);
  for (int c = 0; c < 10; ++c)
    for (int k = 0; k < 5; ++k) {
      pl[c * 5 + k] = c;
      for (int d = 0; d < 2; ++d) pz[(c * 5 + k) * 2 + d] = base[k * 2 + d];
    }
  auto prior_z = nn::from_vector(pz, {50, 2}), prior_labels = nn::from_vector(pl, {50});
  auto prior   = deployed.Decode(prior_z, prior_labels);
  const auto pv = prior.to_vector<float>();
  double condition_change = 0;
  for (int c = 1; c < 10; ++c)
    for (int k = 0; k < 5 * 784; ++k)
      condition_change = std::max(condition_change, std::abs(double(pv[c * 5 * 784 + k]) - pv[k]));
  check::Require(condition_change > 1e-5, "same prior z changes probabilities when ONLY supplied class changes");
  std::vector<float> gz(162);
  for (int row = 0; row < 9; ++row)
    for (int col = 0; col < 9; ++col) {
      gz[(row * 9 + col) * 2]     = -2.f + .5f * col;
      gz[(row * 9 + col) * 2 + 1] = 2.f - .5f * row;
    }
  auto grid_z = nn::from_vector(gz, {81, 2}), grid_labels = nn::from_vector(std::vector<int>(81, 3), {81});
  auto grid   = deployed.Decode(grid_z, grid_labels);
  nn::save_npz((root / "prior.npz").string(), {{"standard", prior_noise.standard},
                                               {"base_z", base_z},
                                               {"z", prior_z},
                                               {"labels", prior_labels},
                                               {"probability", prior},
                                               {"grid_z", grid_z},
                                               {"grid_labels", grid_labels},
                                               {"grid_probability", grid}});
  ae::Mosaic(root / "prior_means.png", prior, 5, 3);
  ae::Mosaic(root / "latent_grid.png", grid, 9, 2);
  auto probe_rows    = ae::Rows(pools[2], {0, 1, 2, 3, 4, 5, 6, 7});
  auto probe_epsilon = nn::take_rows(ae::VaeNoiseAt(test_noise, 0), {0, 1, 2, 3, 4, 5, 6, 7});
  auto probe         = deployed.Probe(probe_rows.images, probe_rows.labels, probe_epsilon);
  {
    const auto lab = probe_rows.labels.to_vector<int>();
    const auto img = probe_rows.images.contiguous().to_vector<float>();
    const auto oh = probe[0].to_vector<float>(), enc = probe[1].to_vector<float>(), dec = probe[2].to_vector<float>(),
               zz = probe[3].to_vector<float>();
    for (int b = 0; b < 8; ++b) {
      for (int c = 0; c < 10; ++c) {
        const float expected = lab[b] == c ? 1.f : 0.f;
        check::Near(oh[b * 10 + c], expected, 0, "actual onehot");
        check::Near(enc[b * 794 + 784 + c], expected, 0, "encoder receives exact same condition");
        check::Near(dec[b * 12 + 2 + c], expected, 0, "decoder receives exact same condition");
      }
      for (int i = 0; i < 784; ++i) check::Near(enc[b * 794 + i], img[b * 784 + i], 0, "encoder concat image prefix exact");
      for (int i = 0; i < 2; ++i) check::Near(dec[b * 12 + i], zz[b * 2 + i], 0, "decoder concat sampled z prefix exact");
    }
  }
  nn::save_npz((root / "condition_probe.npz").string(), {{"images", probe_rows.images},
                                                         {"labels", probe_rows.labels},
                                                         {"epsilon", probe_epsilon},
                                                         {"one_hot", probe[0]},
                                                         {"encoder_input", probe[1]},
                                                         {"decoder_input", probe[2]},
                                                         {"z", probe[3]}});
  for (int invalid : {-1, 10}) {
    bool rejected = false;
    try {
      deployed.Decode(nn::zeros({1, 2}), nn::from_vector(std::vector<int>{invalid}, {1}));
    } catch (const std::exception&) {
      rejected = true;
    }
    check::Require(rejected, "invalid supplied class rejected before graph execution");
  }
  check::Require(before == ae::Snapshot(deployed.parameters),
                 "conditional posterior/prior/controls leave ALL10parameters immutable");
  nn::save_npz((root / "outputs.npz").string(), output);
  nn::save_npz((root / "model_arrays.npz").string(), ae::Arrays(deployed.parameters));
  auto          score    = ae::ConditionalVaeScore(deployed, test_images, test_labels, test_noise);
  const auto    mean_rec = ae::Find(output, "test_mean_reconstruction");
  const double  mean_mse = ae::Mse(mean_rec, test_images);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":102932,\"epochs\":" << epochs << ",\"updates\":" << step << ",\"selected_epoch\":" << selected
       << ",\"mc_samples\":4,\"initial_test_negative_elbo\":" << initial[2] << ",\"test_reconstruction_nll\":" << score[0]
       << ",\"test_kl\":" << score[1] << ",\"test_negative_elbo\":" << score[2] << ",\"mean_decode_test_mse\":" << mean_mse
       << ",\"single_batch_max_error\":" << parity << ",\"posterior_label_change_max\":" << posterior_change
       << ",\"fixed_z_label_change_max\":" << condition_change
       << ",\"wrong_label_mean_mse\":" << ae::Mse(wrong_mean, test_images) << ",\"training_seconds\":" << seconds
       << "}\n";
  auto mean = ae::Constant(ae::PixelMean(pools[0].images), 200);
  // fit-only class mean (100 fitting records per class)
  std::vector<double> class_sums(10 * 784, 0.0);
  int                 class_counts[10] = {};
  {
    const auto fl = pools[0].labels.to_vector<int>();
    const auto fx = pools[0].images.contiguous().to_vector<float>();
    for (int b = 0; b < 1000; ++b) {
      ++class_counts[fl[b]];
      for (int i = 0; i < 784; ++i) class_sums[fl[b] * 784 + i] += fx[b * 784 + i];
    }
  }
  for (int c = 0; c < 10; ++c) check::Require(class_counts[c] == 100, "classmean uses ONLY100fitting records per supplied class");
  std::vector<float> cm(200 * 784);
  const auto         tl = test_labels.to_vector<int>();
  for (int b = 0; b < 200; ++b)
    for (int i = 0; i < 784; ++i) cm[b * 784 + i] = float(class_sums[tl[b] * 784 + i] / class_counts[tl[b]]);
  auto class_mean = nn::from_vector(cm, {200, 784});
  nn::save_npz((root / "class_mean.npz").string(), {{"test_class_mean", class_mean}});
  const auto predictions = ae::Find(output, "test_sample_prediction");
  const auto binary      = ae::Flatten(test_images);
  const std::vector<nn::Tensor> panels{binary,     mean_rec, nn::select(predictions, 0, 0), nn::select(predictions, 0, 1),
                                       class_mean, nn::abs(mean_rec - binary), wrong_mean};
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Class-conditioned VAE</title><style>body{font:16px "
          "system-ui;max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}</style><h1>Conditional "
          "VAE:the supplied class enters both paths</h1><p>Binary input,decode(mu),posterior draw1 probabilities,draw2 "
          "probabilities,fit-only CLASS mean,absolute decode(mu) error,wrong-label decode(SAME mu). decode(mu) is NOT the "
          "posterior predictive mean. First test record per digit,unfiltered;fixed[0,1]gray/nearest4/no error "
          "amplification.</p><p>OriginalMNIST statically thresholded at uint8>=128,not a claim to the official "
          "binarizedMNIST benchmark. Encoder concat(image784,onehot10)->64ReLU/mu+logvariance2heads. q(z|image,class),"
          "unclipped z=mu+exp(.5logvariance)*epsilon,decoder concat(z2,onehot10)->64ReLU->784logits models "
          "p(image|z,class). PriorN(0,I) is shared across classes,not learned. Supplied class is required at "
          "inference;this is not a label predictor. Analytic KL toN(0,I);SUM784pixel NLL+SUM2latent KL,MEANbatch,beta1,"
          "noKLannealing. Explicit standardGaussian float32eps changes each training epoch;validation selects a "
          "FOURsample MC negativeELBO with fixed noise. Not exact loglikelihood or importance-weighted bound.</p><p>"
       << epochs << "epochs/102932parameters/Adam.003/batch32. Test NLL=" << score[0] << ",KL=" << score[1]
       << ",negativeELBO=" << score[2] << " nats/image. decode(mu) MSE=" << mean_mse
       << ",fit-only binary mean MSE=" << ae::Mse(mean, test_images)
       << ",fit-only CLASSmean MSE=" << ae::Mse(class_mean, test_images)
       << ". Short training/two latents blur or lose identity. No semantic,disentanglement,likelihood superiority or "
          "bitrate claim. <a href='https://arxiv.org/abs/1312.6114'>Kingma/Welling VAE</a>.</p>";
  const auto source = pools[2].source.to_vector<int>();
  for (int digit = 0; digit < 10; ++digit) {
    const int         b    = ae::FirstOfDigit(test_labels, digit);
    const std::string file = "digit" + std::to_string(digit) + ".png";
    ae::CaseStrip(root / file, panels, b);
    html << "<h2>Digit" << digit << ",source" << source[b] << "</h2><img src='" << file << "'>";
  }
  html << "<h2>Conditional prior decoder means:rows0..9</h2><p>Same5prior z values in every class row,conditions0..9 "
          "top-bottom. Probabilities,not Bernoulli pixel draws;no image input,explicit supplied class.</p><img "
          "src='prior_means.png'><h2>Fixed2Dlatent grid:condition3</h2><p>x from-2to2 left-right,y from2to-2 "
          "top-bottom,step.5. Fixed[0,1]gray,not normalized per image.</p><img src='latent_grid.png'>";
  check::Require(bool(meta) && bool(html) && bool(history) && bool(batches), "write complete VAE evidence");
  std::cout << "VAE:selected" << selected << ",testNLL=" << score[0] << ",KL=" << score[1] << ",negativeELBO=" << score[2]
            << ",decode(mu)MSE=" << mean_mse
            << ";freshrestore/reparameterization/decode/batch/prior/immutable passed\ntraining time " << seconds
            << " s\nArtifacts:" << root.string() << "/index.html\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::string           init;
    std::filesystem::path data = paths::Dataset("saliency"),
                          out  = paths::OutputRoot() / "models/conditional_variational_autoencoder";
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
    ae::VaeKnownChecks(1, "");
    Run(data, out, epochs, init);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
