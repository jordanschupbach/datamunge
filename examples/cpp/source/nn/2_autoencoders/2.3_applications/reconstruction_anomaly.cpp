// CMD: just run-nn reconstruction_anomaly
// Prepare:just autoencoder-assets;just run undercomplete_autoencoder(THREEdefault epochs).
//
// Port of tflow's reconstruction_anomaly (helpers: examples/nn/common/reconstruction_anomaly.hpp).
// Data: $DATAMUNGE_DATA_DIR/saliency (tflow's `just autoencoder-assets`). The frozen model is
// read from encoder=DIR: best.npz written by our undercomplete_autoencoder port (default
// output/nn/models/undercomplete_autoencoder) or, when pointed at tflow's
// output/models/undercomplete_autoencoder, its model_arrays.npz (the restored best TF
// checkpoint), which reproduces tflow's scores exactly. Outputs (analysis.npz,
// model_arrays.npz, rankings.tsv, metrics.json, index.html, case_<k>.png) go to
// output/nn/models/reconstruction_anomaly. Options: data=PATH encoder=PATH out=PATH.

#include <fstream>
#include <iostream>
#include <map>

#include "common/reconstruction_anomaly.hpp"

namespace {

namespace nn = datamunge::nn;

void Controls() {
  auto normal = nn::from_vector(std::vector<double>{1, 2}, {2}), anomaly = nn::from_vector(std::vector<double>{2, 3}, {2});
  std::vector<double> c(20);
  for (int i = 0; i < 20; ++i) c[i] = i;
  auto r = ae::RankAnomalies(normal, anomaly, 2);
  check::Near(r.auroc, .875, 0, "tied score AUROC half credit");
  check::Near(r.average_precision, 5. / 6, 1e-15, "group-tied average precision");
  check::Near(r.normal_fpr, 0, 0, "threshold comparison strict greater");
  check::Near(r.anomaly_tpr, .5, 0, "threshold ties not anomalous");
  check::Near(ae::NominalThreshold(nn::from_vector(c, {20})), 18, 0, "nearest-rank95percentile");
}

nn::Tensor InkScores(const nn::Tensor& images, double fit_mean) {
  const auto          x = images.contiguous().to_vector<float>();
  const auto          n = images.size(0);
  std::vector<double> out(static_cast<std::size_t>(n));
  for (std::int64_t b = 0; b < n; ++b) {
    double sum = 0;
    for (int j = 0; j < 784; ++j) sum += x[b * 784 + j];
    out[b] = std::abs(sum / 784 - fit_mean);
  }
  return nn::from_vector(out, {n});
}

void Run(const std::filesystem::path& data, const std::filesystem::path& encoder, const std::filesystem::path& root) {
  Controls();
  auto pools = ae::LoadSaliency(data);
  std::filesystem::create_directories(root);
  ae::UndercompleteModel model(false);
  const auto restored = ae::LoadFrozen(model.parameters, encoder, "prepare verified3epoch undercomplete_autoencoder first");
  check::Require(model.parameters.count() == 50992, "frozen nominal clean-MNIST reconstruction model");
  const auto before = ae::Snapshot(model.parameters);
  const auto mean   = ae::PixelMean(pools[0].images);
  double     fit_ink = 0;
  for (float v : mean.to_vector<float>()) fit_ink += v;
  fit_ink /= 784;
  std::map<std::string, nn::Tensor> arrays;
  for (int k = 0; k < 3; ++k) {
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    auto              f     = model.Evaluate(pools[k].images);
    arrays.emplace(split + "_images", pools[k].images);
    arrays.emplace(split + "_labels", pools[k].labels);
    arrays.emplace(split + "_source_ids", pools[k].source);
    arrays.emplace(split + "_reconstruction", f[1]);
    arrays.emplace(split + "_ae_score", ae::ReconstructionScores(pools[k].images, f[1]));
    arrays.emplace(split + "_mean_score",
                   ae::ReconstructionScores(pools[k].images, ae::Constant(mean, pools[k].images.size(0))));
    arrays.emplace(split + "_ink_score", InkScores(pools[k].images, fit_ink));
  }
  arrays.emplace("fit_pixel_mean", mean);
  const std::vector<std::string> names{"patch", "darkening", "shift"};
  for (int kind = 0; kind < 3; ++kind) {
    auto images = ae::AnomalyCorruption(pools[2].images, kind);
    auto f      = model.Evaluate(images);
    arrays.emplace(names[kind] + "_images", images);
    arrays.emplace(names[kind] + "_reconstruction", f[1]);
    arrays.emplace(names[kind] + "_ae_score", ae::ReconstructionScores(images, f[1]));
    arrays.emplace(names[kind] + "_mean_score", ae::ReconstructionScores(images, ae::Constant(mean, 200)));
    arrays.emplace(names[kind] + "_ink_score", InkScores(images, fit_ink));
  }
  auto blank        = nn::zeros({1, 28, 28, 1});
  auto blank_output = model.Evaluate(blank);
  arrays.emplace("blank_image", blank);
  arrays.emplace("blank_reconstruction", blank_output[1]);
  arrays.emplace("blank_ae_score", ae::ReconstructionScores(blank, blank_output[1]));
  check::Require(before == ae::Snapshot(model.parameters), "all clean/corrupted/blank inference immutable");
  nn::save_npz((root / "analysis.npz").string(), nn::NamedTensors(arrays.begin(), arrays.end()));
  nn::save_npz((root / "model_arrays.npz").string(), ae::Arrays(model.parameters));
  if (std::filesystem::exists(data / "manifest.json"))
    std::filesystem::copy_file(data / "manifest.json", root / "data_manifest.json",
                               std::filesystem::copy_options::overwrite_existing);
  const double  blank_score = arrays.at("blank_ae_score").to_vector<double>()[0];
  std::ofstream table(root / "rankings.tsv"), meta(root / "metrics.json");
  table.precision(17);
  meta.precision(17);
  table << "score\tcorruption\tvalidation_nominal_threshold\tvalidation_nominal_fpr\ttest_nominal_fpr\tcorruption_tpr\tauroc"
           "\taverage_precision\n";
  meta << "{\"additional_training_updates\":0,\"fit_records\":1000,\"validation_nominal_records\":200,\"test_nominal_"
          "records\":200,\"each_corruption_records\":200,\"quantile\":0.95,\"fit_mean_ink\":"
       << fit_ink << ",\"blank_ae_score\":" << blank_score << ",\"thresholds\":{";
  bool first = true;
  for (const std::string score : {"ae", "mean", "ink"}) {
    const auto   validation = arrays.at("validation_" + score + "_score");
    const double threshold  = ae::NominalThreshold(validation);
    int          fp         = 0;
    for (double v : validation.to_vector<double>()) fp += v > threshold;
    if (!first) meta << ',';
    first = false;
    meta << '"' << score << "\":" << threshold;
    for (const auto& name : names) {
      auto r = ae::RankAnomalies(arrays.at("test_" + score + "_score"), arrays.at(name + "_" + score + "_score"), threshold);
      table << score << '\t' << name << '\t' << threshold << '\t' << fp / 200. << '\t' << r.normal_fpr << '\t'
            << r.anomaly_tpr << '\t' << r.auroc << '\t' << r.average_precision << '\n';
      std::cout << score << '/' << name << ":nominalFPR=" << r.normal_fpr << ",TPR=" << r.anomaly_tpr
                << ",AUROC=" << r.auroc << ",AP=" << r.average_precision << std::endl;
    }
  }
  meta << "}}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Reconstruction anomaly stress tests</title><style>body{font:16px "
          "system-ui;margin:30px;max-width:1300px}img{max-width:100%;image-rendering:pixelated}</style><h1>Anomaly scoring by "
          "reconstruction error</h1><p>Frozen THREEepoch MNIST784-32ReLU-784sigmoid autoencoder,0additionaltraining. "
          "Cleanfit1000nominaldigits,cleanvalidation200for threshold only,cleanheldouttest200. Every digit class is nominal;this "
          "is NOT one-class digit rejection. Labels do not train detector. Threshold nearest-rank95th percentile of nominal "
          "VALIDATION scores,strict score>threshold. No corrupted labels or test outcomes select threshold.</p><p>Predefined "
          "paired heldout stress transformations:8x8WHITEtop-leftpatch;globaldarkening*.35;horizontalshift6rightwithzeropadding. "
          "All200originaltestimages mutated EACHway,600syntheticcases. These labels describe artificial transformations,not "
          "disease/fraud or real-world outliers;shift/darkening may be legitimate domain variation. Shared originalsources make "
          "them paired,not independent extra source examples. One blankimage is a separate degenerate control,excluded from "
          "ranking pools.</p><p>Scores:AE per-pixel squared reconstruction residual;fit-onlypixelmean residual;absolute "
          "mean-ink deviation from fit mean. EACH calibrated separately on SAMEcleanvalidation only. rankings.tsv gives "
          "cleanheldoutFPR,eachcorruptionTPR,AUROC(halfcredit ties),averageprecision(group ties). Equal200nominal/200corruptcases "
          "pertype,APbaseprevalence.5;prevalence changes AP. Single small split,no significance/real deployment claim. "
          "BlankAEscore="
       << blank_score
       << " is a separate degenerate-input control,excluded from ranking pools. Residuals can miss low-ink corruptions;"
          "allmisses retained.</p><p>Ten evenly spaced originaltestcases,unfiltered. Columns original/reconstruction/WHITEpatch+"
          "reconstruction/darkening+reconstruction/shift+reconstruction,fixed0..1gray. Pixel outputs are probabilities,not draws. "
          "See <a href='rankings.tsv'>allranking results</a> and <a href='metrics.json'>validation thresholds</a>. Frozen "
          "weights: "
       << restored.string() << ".</p>";
  const auto source = pools[2].source.to_vector<int>(), labels = pools[2].labels.to_vector<int>();
  for (int k = 0; k < 10; ++k) {
    const int b = k * 199 / 9;
    ae::CaseStrip3(root / ("case_" + std::to_string(k) + ".png"),
                   {arrays.at("test_images"), arrays.at("test_reconstruction"), arrays.at("patch_images"),
                    arrays.at("patch_reconstruction"), arrays.at("darkening_images"), arrays.at("darkening_reconstruction"),
                    arrays.at("shift_images"), arrays.at("shift_reconstruction")},
                   b);
    html << "<p>Source" << source[b] << ",digit" << labels[b] << "<br><img src='case_" << k << ".png'></p>";
  }
  check::Require(bool(table) && bool(meta) && bool(html), "complete anomaly score artifacts");
  std::cout << "nominalValidationOnly95percentile/strictThreshold/tiedAUROC+AP/frozenWeights/600pairedCorruptions PASS\n"
               "Artifacts:"
            << (root / "index.html").string() << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path data    = paths::Dataset("saliency"),
                          encoder = paths::OutputRoot() / "models/undercomplete_autoencoder",
                          out     = paths::OutputRoot() / "models/reconstruction_anomaly";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("data=", 0) == 0)
        data = a.substr(5);
      else if (a.rfind("encoder=", 0) == 0)
        encoder = a.substr(8);
      else if (a.rfind("out=", 0) == 0)
        out = a.substr(4);
      else
        throw std::invalid_argument("expected data=PATH encoder=PATH out=PATH");
    }
    Run(data, encoder, out);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
