// CMD: just run-nn convolutional_variational_autoencoder
// Convolutional Gaussian VAE with explicit reparameterized noise and Bernoulli negative ELBO.
// Static MNIST threshold 128, three epochs.
//
// Port of tflow's convolutional_variational_autoencoder. Preparation (in ~/projects/tflow_examples):
// just autoencoder-assets; data are read from $DATAMUNGE_DATA_DIR/saliency. Outputs go to
// output/nn/models/convolutional_variational_autoencoder. Checkpoints are best.npz instead of
// SaveV2 bundles. Options: epochs=N data=PATH out=PATH init=PATH (start from saved parameters,
// e.g. tflow's initial_arrays.npz, to follow its trajectory exactly).

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>

#include "common/beta_variational_autoencoder.hpp"
#include "common/convolutional_variational_autoencoder.hpp"

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
  }
  saved_data.emplace_back("test_labels", pools[2].labels);
  nn::save_npz((root / "data.npz").string(), saved_data);
  ae::CopyManifests(data, root);
  auto val_noise = ae::VaeNoise(200, 4, 2039), test_noise = ae::VaeNoise(200, 4, 2053),
       fit_noise = ae::VaeNoise(1000, 4, 2063);
  nn::save_npz((root / "fixed_noise.npz").string(), {{"validation_standard", val_noise.standard},
                                                     {"validation_epsilon", val_noise.epsilon},
                                                     {"test_standard", test_noise.standard},
                                                     {"test_epsilon", test_noise.epsilon},
                                                     {"fit_standard", fit_noise.standard},
                                                     {"fit_epsilon", fit_noise.epsilon}});
  ae::ConvolutionalVariationalAutoencoderModel model;
  if (!init.empty()) model.parameters.load(init);
  check::Require(model.parameters.size() == 14 && model.parameters.count() == 2117,
                 "twohead diagonalGaussian parameter census");
  nn::save_npz((root / "initial_arrays.npz").string(), ae::Arrays(model.parameters));
  auto           initial = ae::ConvolutionalVaeScore(model, pools[2].images, test_noise);
  nn::BatchOrder order(1000, 32, 2017);
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
    auto noise = ae::VaeNoise(1000, 1, 2027 + epoch);
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
                                    "ALL14posterior/decoder tensors receive gradients");
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
    auto score = ae::ConvolutionalVaeScore(model, pools[1].images, val_noise);
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
  ae::ConvolutionalVariationalAutoencoderModel deployed(false);
  deployed.parameters.load(best_path);
  const auto before = ae::Snapshot(deployed.parameters);
  check::Near(ae::ConvolutionalVaeScore(deployed, pools[1].images, val_noise)[2], best, 1e-8,
              "fresh selected MC validation objective");
  nn::NamedTensors              output;
  std::vector<ae::VaeNoiseBank> banks{fit_noise, val_noise, test_noise};
  for (int split = 0; split < 3; ++split) {
    const std::string  name = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    const std::int64_t n    = pools[split].images.size(0);
    auto               mean = deployed.Evaluate(pools[split].images, nn::zeros({n, 2}));
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
  const std::vector<std::int64_t> eight{0, 1, 2, 3, 4, 5, 6, 7};
  auto probe_images = nn::take_rows(pools[2].images, eight),
       probe_epsilon = nn::take_rows(ae::VaeNoiseAt(test_noise, 0), eight);
  auto stages = deployed.Stages(probe_images, probe_epsilon);
  const std::vector<nn::Shape> shapes{{8, 28, 28, 8}, {8, 14, 14, 8}, {8, 14, 14, 4}, {8, 7, 7, 4},
                                      {8, 7, 7, 4},   {8, 14, 14, 8}, {8, 28, 28, 1}};
  nn::NamedTensors stage_arrays{{"images", probe_images}, {"epsilon", probe_epsilon}};
  for (std::size_t i = 0; i < stages.size(); ++i) {
    check::Require(stages[i].shape() == shapes[i], "all seven actual convolution/pool/repeat decoder stage shapes");
    stage_arrays.emplace_back("stage" + std::to_string(i), stages[i]);
  }
  nn::save_npz((root / "stages.npz").string(), stage_arrays);
  ae::ConvolutionalVariationalAutoencoderModel gradient_probe;
  gradient_probe.parameters.load(best_path);  // independent selected-gradient review restore
  auto gradient_images = nn::take_rows(pools[2].images, {0, 1}),
       gradient_epsilon = nn::take_rows(ae::VaeNoiseAt(test_noise, 0), {0, 1});
  auto             selected_gradient = gradient_probe.Gradient(gradient_images, gradient_epsilon);
  nn::NamedTensors gradient_arrays;
  for (std::size_t i = 0; i < selected_gradient.size(); ++i)
    gradient_arrays.emplace_back(gradient_probe.parameters.names()[i], selected_gradient[i]);
  nn::save_npz((root / "selected_gradients.npz").string(), gradient_arrays);
  nn::save_npz((root / "selected_gradient_batch.npz").string(),
               {{"images", gradient_images}, {"epsilon", gradient_epsilon}});
  auto test   = deployed.Evaluate(pools[2].images, ae::VaeNoiseAt(test_noise, 0)),
       repeat = deployed.Evaluate(pools[2].images, ae::VaeNoiseAt(test_noise, 0));
  check::Require(ae::Bytes(test[2]) == ae::Bytes(repeat[2]) && ae::Bytes(test[3]) == ae::Bytes(repeat[3]) &&
                     ae::Bytes(deployed.Decode(test[2])) == ae::Bytes(test[3]),
                 "fixed epsilon repeatability and target-free exact sampled decode");
  auto other = deployed.Evaluate(pools[2].images, ae::VaeNoiseAt(test_noise, 1));
  check::Require(ae::Bytes(test[2]) != ae::Bytes(other[2]) && ae::Bytes(test[3]) != ae::Bytes(other[3]),
                 "second epsilon changes latent samples AND decoder probabilities");
  auto single = deployed.Evaluate(nn::take_rows(pools[2].images, {0}),
                                  nn::take_rows(ae::VaeNoiseAt(test_noise, 0), {0}));
  const double parity = ae::MaxAbsDiff(single[3], test[3], 784);
  check::Require(parity < 2e-6, "posterior single/batch parity");
  auto zero_latent = deployed.Decode(nn::zeros({200, 2}));
  output.emplace_back("zero_latent_prediction", zero_latent);
  auto               prior_noise = ae::VaeNoise(20, 1, 2069);
  auto               prior_z = ae::VaeNoiseAt(prior_noise, 0), prior = deployed.Decode(prior_z);
  std::vector<float> gz(162);
  for (int row = 0; row < 9; ++row)
    for (int col = 0; col < 9; ++col) {
      gz[(row * 9 + col) * 2]     = -2.f + .5f * col;
      gz[(row * 9 + col) * 2 + 1] = 2.f - .5f * row;
    }
  auto grid_z = nn::from_vector(gz, {81, 2});
  auto grid   = deployed.Decode(grid_z);
  nn::save_npz((root / "prior.npz").string(), {{"standard", prior_noise.standard},
                                               {"z", prior_z},
                                               {"probability", prior},
                                               {"grid_z", grid_z},
                                               {"grid_probability", grid}});
  ae::Mosaic(root / "prior_means.png", prior, 5, 3);
  ae::Mosaic(root / "latent_grid.png", grid, 9, 2);
  check::Require(before == ae::Snapshot(deployed.parameters),
                 "posterior evaluation/prior generation leaves ALL14parameters immutable");
  nn::save_npz((root / "outputs.npz").string(), output);
  nn::save_npz((root / "model_arrays.npz").string(), ae::Arrays(deployed.parameters));
  auto          score    = ae::ConvolutionalVaeScore(deployed, pools[2].images, test_noise);
  const auto    mean_rec = ae::Find(output, "test_mean_reconstruction");
  const double  mean_mse = ae::Mse(mean_rec, pools[2].images);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":2117,\"epochs\":" << epochs << ",\"updates\":" << step << ",\"selected_epoch\":" << selected
       << ",\"mc_samples\":4,\"initial_test_negative_elbo\":" << initial[2] << ",\"test_reconstruction_nll\":" << score[0]
       << ",\"test_kl\":" << score[1] << ",\"test_negative_elbo\":" << score[2] << ",\"mean_decode_test_mse\":" << mean_mse
       << ",\"single_batch_max_error\":" << parity << ",\"training_seconds\":" << seconds << "}\n";
  auto       mean        = ae::Constant(ae::PixelMean(pools[0].images), 200);
  const auto predictions = ae::Find(output, "test_sample_prediction");
  const auto binary      = ae::Flatten(pools[2].images);
  const std::vector<nn::Tensor> panels{binary, mean_rec, nn::select(predictions, 0, 0), nn::select(predictions, 0, 1),
                                       mean,   nn::abs(mean_rec - binary), zero_latent};
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Convolutional Gaussian VAE</title><style>body{font:16px "
          "system-ui;max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}</style><h1>Convolutional "
          "VAE:spatial inference and prior decoding</h1><p>Binary input,decode(mu),posterior draw1 probabilities,draw2 "
          "probabilities,fit-only mean,absolute decode(mu) error,zero-latent probabilities. decode(mu) is NOT the "
          "posterior predictive mean. First test record per digit,unfiltered;fixed[0,1]gray/nearest4/no error "
          "amplification.</p><p>OriginalMNIST statically thresholded at uint8>=128,not a claim to the official "
          "binarizedMNIST benchmark. This makes the observation model Bernoulli rather than treating fractional "
          "grayscale as binary data. Encoder Conv3same1->8ReLU/avg2/Conv3same8->4ReLU/avg2=7x7x4,mu/logvariance196->2heads. "
          "Unclipped z=mu+exp(.5logvariance)*epsilon is the ONLY bridge. Decoder2->196ReLU/reshape7x7x4/nearest-repeat2/"
          "Conv4->8ReLU/nearest-repeat2/Conv8->1logits. NO skips or deterministic spatial-code bypass. Analytic KL "
          "toN(0,I);SUM784pixel NLL+SUM2latent KL,MEANbatch,beta1,noKLannealing. Explicit standardGaussian float32eps "
          "changes each training epoch;validation selects a FOURsample MC negativeELBO with fixed noise. Not exact "
          "loglikelihood or importance-weighted bound.</p><p>"
       << epochs << "epochs/2117parameters/Adam.003/batch32. Test NLL=" << score[0] << ",KL=" << score[1]
       << ",negativeELBO=" << score[2] << " nats/image. decode(mu) MSE=" << mean_mse
       << ",fit-only binary mean MSE=" << ae::Mse(mean, pools[2].images)
       << ". Short training/two latents blur or lose identity. No semantic,disentanglement,likelihood superiority or "
          "bitrate claim. <a href='https://arxiv.org/abs/1312.6114'>Kingma/Welling VAE</a>.</p>";
  const auto source = pools[2].source.to_vector<int>();
  for (int digit = 0; digit < 10; ++digit) {
    const int         b    = ae::FirstOfDigit(pools[2].labels, digit);
    const std::string file = "digit" + std::to_string(digit) + ".png";
    ae::CaseStrip(root / file, panels, b);
    html << "<h2>Digit" << digit << ",source" << source[b] << "</h2><img src='" << file << "'>";
  }
  html << "<h2>20 unfiltered standard-normal prior decoder means</h2><p>Probabilities,not Bernoulli pixel draws;no "
          "input or label feed.</p><img src='prior_means.png'><h2>Fixed2Dlatent grid</h2><p>x from-2to2 left-right,y "
          "from2to-2 top-bottom,step.5. Fixed[0,1]gray,not normalized per image.</p><img src='latent_grid.png'>";
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
                          out  = paths::OutputRoot() / "models/convolutional_variational_autoencoder";
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
