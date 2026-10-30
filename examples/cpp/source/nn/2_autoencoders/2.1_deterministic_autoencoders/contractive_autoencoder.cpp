// CMD: just run-nn contractive_autoencoder
// Exact sigmoid-encoder Jacobian-Frobenius penalty, plain and strong matched controls.
// Three epochs/default; differentiate through activation slopes AND encoder weights.
//
// Port of tflow's contractive_autoencoder. Preparation (in ~/projects/tflow_examples):
// just autoencoder-assets; data are read from $DATAMUNGE_DATA_DIR/saliency. Outputs
// (per-variant plain/contractive/strong histories, npz arrays and metrics.json, data.npz,
// index.html and digit<k>.png galleries) go to output/nn/models/contractive_autoencoder.
// Checkpoints are best.npz files instead of TF SaveV2 bundles. Options: epochs=N
// data=PATH out=PATH init=PATH (start all three variants from saved parameters, e.g.
// tflow's plain/initial_arrays.npz, to follow its trajectories exactly; by default nn's
// own initialiser is used, identically for the three variants).

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>

#include "common/contractive_autoencoder.hpp"

namespace {

namespace nn = datamunge::nn;

void Checks() {
  auto x = nn::from_vector(std::vector<float>{.1f, .2f, .7f, .8f, .3f, .4f}, {2, 3}).requires_grad_();
  auto w = nn::from_vector(std::vector<float>{.3f, -.4f, .7f, .2f, -.2f, .5f}, {3, 2}).requires_grad_();
  auto b = nn::from_vector(std::vector<float>{.1f, -.2f}, {2}).requires_grad_();
  auto h = nn::sigmoid(nn::bias_add(nn::matmul(x, w), b));
  auto p = ae::SigmoidEncoderContraction(h, w);
  nn::GradOptions keep;
  keep.retain_graph = true;
  auto grad = nn::grad(p.mean, {w, b}, keep);
  std::vector<nn::Tensor> rows;
  for (int j = 0; j < 2; ++j) rows.push_back(nn::grad(nn::sum(nn::slice_axis(h, 1, j, j + 1)), {x}, keep)[0]);
  auto   jac     = nn::stack(rows, 1);  // [2, 2, 3]
  double penalty = 0, gw[3][2] = {}, gb[2] = {};
  for (int n = 0; n < 2; ++n)
    for (int j = 0; j < 2; ++j) {
      double z = b.at({j}), norm = 0;
      for (int i = 0; i < 3; ++i) {
        z += x.at({n, i}) * w.at({i, j});
        norm += std::pow(w.at({i, j}), 2);
      }
      const double activation = 1 / (1 + std::exp(-z)), slope = activation * (1 - activation),
                   dz         = slope * slope * (1 - 2 * activation) * norm;
      penalty += slope * slope * norm / 2;
      gb[j] += dz;
      for (int i = 0; i < 3; ++i) {
        check::Near(jac.at({n, j, i}), slope * w.at({i, j}), 3e-8, "ALL12true encoder-Jacobian entries");
        gw[i][j] += x.at({n, i}) * dz + w.at({i, j}) * slope * slope;
      }
    }
  check::Near(p.mean.item<float>(), penalty, 3e-8, "SUMallJacobian coordinates,MEANbatch");
  for (int j = 0; j < 2; ++j) {
    check::Near(grad[1].at({j}), gb[j], 3e-8, "ALL2penalty bias derivatives");
    for (int i = 0; i < 3; ++i)
      check::Near(grad[0].at({i, j}), gw[i][j], 3e-8, "ALL6penalty weight derivatives include direct2W term");
  }
  std::cout << "known encoder:ALL12Jacobian coordinates/Frobeniusnorm/ALL8penalty parameter derivatives passed\n";
}

struct Variant {
  std::string name;
  float       weight;
};

double Train(const Variant& config, const std::vector<ae::SaliencyData>& pools, int epochs,
             const std::filesystem::path& root, const std::string& init) {
  const auto directory = root / config.name;
  std::filesystem::create_directories(directory);
  ae::ContractiveAutoencoderModel model(config.weight);
  if (!init.empty()) model.parameters.load(init);
  check::Require(model.parameters.count() == 50992 && model.parameters.size() == 4,
                 "independent784->32->784untied inventory");
  nn::save_npz((directory / "initial_arrays.npz").string(), ae::Arrays(model.parameters));
  nn::BatchOrder order(1000, 32, 1759);
  std::ofstream  history(directory / "history.tsv"), batches(directory / "batches.tsv");
  history.precision(17);
  history << "epoch\tfit_preupdate_loss\tvalidation_loss\tvalidation_mse\tvalidation_jacobian_squared\n";
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
      auto input = nn::take_rows(pools[0].images, rows);
      // Forward: MSE + weight * exact Jacobian penalty, differentiated through h AND W.
      auto loss = model.Forward(ae::Flatten(input)).loss;
      if (step == 0) {
        nn::GradOptions keep;
        keep.retain_graph = true;
        auto gradients    = ae::CheckGradients(model.parameters, nn::grad(loss, model.parameters.parameters(), keep),
                                               "every full contractive gradient finite",
                                               "all4parameters receive complete gradients");
        nn::save_npz((directory / "first_gradients.npz").string(), gradients);
        nn::save_npz((directory / "first_batch.npz").string(), {{"images", input}});
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
    check::Require(seen == 1000, "every fit image once per variant epoch");
    auto         result = model.Evaluate(pools[1].images);
    const double value  = result[2].item<float>();
    history << epoch << '\t' << sum / seen;
    for (int i = 2; i < 5; ++i) history << '\t' << result[i].item<float>();
    history << '\n';
    nn::save_npz((directory / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(),
                 ae::Arrays(model.parameters));
    if (value < best) {
      best     = value;
      selected = epoch;
      model.parameters.save(best_path);  // own full contractive validation objective selects checkpoint
    }
    std::cout << config.name << " epoch" << epoch << ":validation total/MSE/JacobianSquared=" << value << '/'
              << result[3].item<float>() << '/' << result[4].item<float>() << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  ae::ContractiveAutoencoderModel deployed(config.weight, false);
  deployed.parameters.load(best_path);  // fresh restore all contractive tensors
  const auto       before = ae::Snapshot(deployed.parameters);
  nn::NamedTensors arrays;
  for (int split = 0; split < 3; ++split) {
    const std::string name   = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    auto              result = deployed.Evaluate(pools[split].images);
    arrays.emplace_back(name + "_code", result[0]);
    arrays.emplace_back(name + "_reconstruction", result[1]);
    arrays.emplace_back(name + "_jacobian_squared", result[5]);
    auto components = nn::stack({result[2], result[3], result[4]}, 0).contiguous();
    arrays.emplace_back(name + "_components", components);
    if (split == 1) check::Near(result[2].item<float>(), best, 2e-7, "fresh selected-loss restore");
  }
  auto         probe    = ae::Rows(pools[2], {0, 1, 2, 3, 4, 5, 6, 7});
  auto         actual_j = deployed.Jacobian(probe.images);
  auto         eval     = deployed.Evaluate(probe.images);
  const auto   jv = actual_j.to_vector<float>(), per = eval[5].to_vector<float>();
  double       norm_error = 0;
  for (int n = 0; n < 8; ++n) {
    double sum = 0;
    for (int k = 0; k < 32 * 784; ++k) sum += std::pow(double(jv[n * 32 * 784 + k]), 2);
    norm_error = std::max(norm_error, std::abs(sum - per[n]));
  }
  check::Require(norm_error < 2e-5, "analytic contraction equals independent ALLautodiff Jacobian Frobenius sum");
  arrays.emplace_back("probe_images", probe.images);
  arrays.emplace_back("probe_jacobian", actual_j);
  auto repeated = deployed.Evaluate(pools[2].images);
  check::Require(ae::Bytes(repeated[1]) == ae::Bytes(ae::Find(arrays, "test_reconstruction")),
                 "deterministic contractive reconstruction");
  check::Require(ae::Bytes(deployed.Decode(repeated[0])) == ae::Bytes(repeated[1]),
                 "explicit contractive encode-decode exact");
  check::Require(before == ae::Snapshot(deployed.parameters), "all4contractive parameters immutable");
  nn::save_npz((directory / "outputs.npz").string(), arrays);
  nn::save_npz((directory / "model_arrays.npz").string(), ae::Arrays(deployed.parameters));
  std::ofstream meta(directory / "metrics.json");
  meta.precision(17);
  meta << "{\"weight\":" << config.weight << ",\"epochs\":" << epochs << ",\"updates\":" << step
       << ",\"selected_epoch\":" << selected << ",\"test_mse\":" << ae::Mse(repeated[1], pools[2].images)
       << ",\"test_jacobian_squared\":" << repeated[4].item<float>() << ",\"autodiff_norm_max_error\":" << norm_error
       << ",\"training_seconds\":" << seconds << "}\n";
  check::Require(bool(meta) && bool(history) && bool(batches), "write full contractive evidence");
  std::cout << config.name << ":fullautodiff Jacobian-norm maxerror=" << norm_error
            << ",freshrestore/decode/repeat/immutable4weights passed (training " << seconds << " s)\n";
  return seconds;
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  auto pools = ae::LoadSaliency(data);
  std::filesystem::create_directories(root);
  const std::vector<Variant> variants{{"plain", 0}, {"contractive", .01f}, {"strong", .1f}};
  double                     seconds = 0;
  for (const auto& v : variants) seconds += Train(v, pools, epochs, root, init);
  const auto base = nn::load_npz((root / "plain" / "initial_arrays.npz").string());
  for (const auto& v : variants) {
    const auto initial = nn::load_npz((root / v.name / "initial_arrays.npz").string());
    for (const auto& [key, value] : base)
      check::Require(ae::Bytes(ae::Find(initial, key)) == ae::Bytes(value), "all3initializations byteidentical");
  }
  ae::SaveData(root / "data.npz", pools);
  ae::CopyManifests(data, root);
  const auto mean = ae::Constant(ae::PixelMean(pools[0].images), 200);
  std::vector<nn::NamedTensors> outputs;
  for (const auto& v : variants) outputs.push_back(nn::load_npz((root / v.name / "outputs.npz").string()));
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Contractive autoencoder</title><style>body{font:16px "
          "system-ui;max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:"
          "25px 0}</style><h1>Contractive autoencoder:exact encoder Jacobian</h1><p>Original,plain,lambda.01,"
          "lambda.1,fit-only mean. Fixed[0,1]grayscale/nearest4,no quality filtering. 784->32sigmoid->784sigmoid/"
          "50992parameters,"
       << epochs
       << "epochs each,matched initialization/order. Clean inputs reconstruct themselves;no corruption or class "
          "targets.</p><p>Loss=MSE+lambda*MEANexamples SUMunits,inputs (dh_j/dx_i)^2. For sigmoid h:Jacobian "
          "diag(h*(1-h))*W^T. Analytic exact norm differentiates through BOTH h and W,including direct weight "
          "dependence;it is not weight decay or a noisy-input proxy. Coordinates are original[0,1]pixels. "
          "Independent reverse-mode Jacobian rows validate the actual norm.</p><p>Each variant selects its OWN full "
          "validation objective;objective numbers across different lambdas are not a common-MSE ranking. A small "
          "encoder Jacobian can result from saturation or collapsed representations,not meaningful invariance. "
          "Reconstruction detail,latent variance and saturation accompany the metric. Local contraction does not "
          "prove robustness for finite changes or preservation of digit identity.</p><table><tr><th>Variant</th>"
          "<th>Test MSE</th><th>Mean Jacobian squared norm</th><th>Fraction saturated outside[.01,.99]</th><th>Mean "
          "code coordinate std</th></tr>";
  for (std::size_t k = 0; k < variants.size(); ++k) {
    const auto codes      = ae::Find(outputs[k], "test_code").to_vector<float>();
    const auto components = ae::Find(outputs[k], "test_components").to_vector<float>();
    int        saturated  = 0;
    double     standard_deviation = 0;
    for (float value : codes) saturated += value < .01 || value > .99;
    for (int j = 0; j < 32; ++j) {
      double mean_j = 0, sum = 0;
      for (int n = 0; n < 200; ++n) mean_j += codes[n * 32 + j] / 200.;
      for (int n = 0; n < 200; ++n) sum += std::pow(double(codes[n * 32 + j]) - mean_j, 2) / 200.;
      standard_deviation += std::sqrt(sum) / 32;
    }
    html << "<tr><td>" << variants[k].name << "</td><td>" << components[1] << "</td><td>" << components[2]
         << "</td><td>" << double(saturated) / codes.size() << "</td><td>" << standard_deviation << "</td></tr>";
    std::cout << variants[k].name << ": test MSE=" << components[1] << ", Jacobian squared=" << components[2]
              << ", saturated=" << double(saturated) / codes.size() << ", code std=" << standard_deviation << '\n';
  }
  html << "</table><p>Short sigmoid runs may produce blurry average-looking outputs despite very small derivatives. "
          "Full-code variance/saturation and numerical evidence are saved. These are continuous32float coordinates,"
          "not a bit-rate compression or generative claim.</p>";
  ae::DigitGallery(root, pools[2],
                   {pools[2].images, ae::Find(outputs[0], "test_reconstruction"),
                    ae::Find(outputs[1], "test_reconstruction"), ae::Find(outputs[2], "test_reconstruction"), mean},
                   html);
  check::Require(bool(html), "write all contractive gallery panels");
  std::cout << "Three matched variants/" << epochs
            << " epochs,actual squared encoder Jacobian penalty;no semantic/finite-robustness superiority claim.\n"
               "training time "
            << seconds << " s\nArtifacts:" << root.string() << "/index.html\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::string           init;
    std::filesystem::path data = paths::Dataset("saliency"),
                          out  = paths::OutputRoot() / "models/contractive_autoencoder";
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
