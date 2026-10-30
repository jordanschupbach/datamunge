// CMD: just run-nn gan_diagnostics
// GAN diagnostics: FID / Inception score algebra, collapse/copy controls and
// small-sample caveats on 32-image CIFAR-10 variants.
//
// Port of tflow's gan_diagnostics — PARTIAL. tflow runs the ORIGINAL frozen
// ImageNet Inception GraphDef (classify_image_graph_def.pb) to extract pool3
// features and 1008-way probabilities, and generates the "singan" variant by
// restoring the three tflow SinGAN scale checkpoints. Neither has a datamunge
// counterpart (loading GraphDef/TF checkpoint files is out of scope, and the
// SinGAN example is ported separately), so this port reads tflow's RECORDED
// extractor outputs and SinGAN images from a tflow gan_diagnostics run
// (features=DIR, default ~/projects/tflow_examples/output/models/gan_diagnostics:
// images.npz, {real_reference,independent_real,singan}_inception.npz). Every
// metric is recomputed here with datamunge (float64 low-rank Frechet distance
// via SVD of the [32, 32] cross matrix, Inception score, 4 splits of 8).
//
// What is ported in full: the metric algebra fixtures (balanced one-hot IS 4,
// uniform and collapsed IS 1, FID 0 for different distributions with equal
// moments, FID symmetric for unequal N, rejection of non-normalized
// probabilities and of N = 1 covariances); the real CIFAR-10 variants built
// from $DATAMUNGE_DATA_DIR/cifar10 (test rows 64..95 as reference, train
// batch 1 rows 128..159 as an independent real sample), which must equal the
// recorded tflow images BIT EXACTLY (so the recorded features belong to them);
// the controls derived by row selection (one image collapsed 32x, four images
// repeated, the reference copied in reverse order: FID 0 without learning)
// and the checks IS(collapsed) = 1, FID(copy) = FID(reference) = 0. Our FIDs
// are compared with tflow's recorded *_frechet.npz. The frozen-extractor
// repeat check is not applicable. Outputs (algebra.npz, scores.tsv,
// <variant>_frechet.npz, <variant>.png, metrics.json) go to
// output/nn/models/gan_diagnostics. Options: data=PATH features=PATH out=PATH.

#include "common/dataset_assets.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>

#include "common/gan.hpp"
#include "common/generative_metrics.hpp"
#include "common/paths.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

Tensor Rows(const Tensor& x, const std::vector<std::int64_t>& rows) { return nn::take_rows(x, rows); }

/// tflow Pixels: [0, 1] -> clamp(v * 255, 0, 255).
Tensor Pixels(const Tensor& x) {
  auto v = x.contiguous().to_vector<float>();
  for (auto& p : v) p = float(std::clamp(double(p) * 255., 0., 255.));
  return nn::from_vector(v, x.shape());
}

double Scalar(const std::map<std::string, Tensor>& m, const std::string& key) { return m.at(key).item(); }

void Algebra(const std::filesystem::path& root) {
  const auto   f64 = nn::DType::Float64;
  const double r2  = std::sqrt(2.);
  const Tensor a = nn::tensor({-1, -1, -1, 1, 1, -1, 1, 1}, {4, 2}, f64);
  const Tensor b = nn::tensor({r2, 0, -r2, 0, 0, r2, 0, -r2}, {4, 2}, f64);
  const Tensor unequal = nn::tensor({0, 0, 1, 2, -1, 3, 2, 1, -2, -1}, {5, 2}, f64);
  std::vector<double> onehot(16, 0.), collapsed(16, 0.), uniform(16, .25);
  for (int i = 0; i < 4; ++i) onehot[i * 4 + i] = 1, collapsed[i * 4] = 1;
  const Tensor oh = nn::from_vector(onehot, {4, 4}), co = nn::from_vector(collapsed, {4, 4}), un = nn::from_vector(uniform, {4, 4});
  check::Near(gan::InceptionScore(oh), 4, 1e-12, "balanced confident IS classes");
  check::Near(gan::InceptionScore(un), 1, 1e-12, "uniform conditional IS 1");
  check::Near(gan::InceptionScore(co), 1, 1e-12, "collapsed class IS 1");
  check::Near(Scalar(gan::Frechet(a, b), "fid"), 0, 1e-12,
              "FID cannot distinguish different distributions with same mean and covariance");
  const auto unequal_score = gan::Frechet(a, unequal);
  check::Near(Scalar(unequal_score, "fid"), Scalar(gan::Frechet(unequal, a), "fid"), 1e-12, "Frechet symmetric unequal N");
  gan::Arrays fixture{{"a", a}, {"b_same_moments", b}, {"unequal", unequal}, {"onehot", oh}, {"uniform", un}, {"collapsed", co}};
  for (const auto& [k, v] : unequal_score) fixture.emplace("unequal_" + k, v);
  gan::SaveArrays((root / "algebra.npz").string(), fixture);
  bool rejected = false;
  try {
    gan::InceptionScore(nn::tensor({.5, .8}, {1, 2}, f64));
  } catch (const std::exception&) {
    rejected = true;
  }
  check::Require(rejected, "reject non-normalized probabilities");
  rejected = false;
  try {
    gan::Frechet(nn::zeros({1, 2}, f64), a);
  } catch (const std::exception&) {
    rejected = true;
  }
  check::Require(rejected, "reject undefined N=1 sample covariance");
  std::cout << "metric algebra: IS(one-hot)=4, IS(uniform)=IS(collapsed)=1, FID(same moments)=0, unequal-N FID="
            << Scalar(unequal_score, "fid") << " (symmetric) PASS\n";
}

gan::Arrays LoadNpz(const std::filesystem::path& path) {
  check::Require(std::filesystem::exists(path), path.string() +
                                                    " missing: run tflow's gan_diagnostics (needs its Inception assets "
                                                    "and SinGAN) or pass features=DIR");
  gan::Arrays out;
  for (auto& [k, v] : nn::load_npz(path.string())) out.emplace(k, v);
  return out;
}

void Run(const std::filesystem::path& data, const std::filesystem::path& recorded, const std::filesystem::path& root) {
  std::filesystem::create_directories(root);
  Algebra(root);
  const auto test = dataset_assets::LoadCifar10(data.string(), "test"), train = dataset_assets::LoadCifar10(data.string(), "train");
  std::vector<std::int64_t> rows;
  for (int i = 64; i < 96; ++i) rows.push_back(i);
  const Tensor ref = Pixels(Rows(test.images, rows));
  rows.clear();
  for (int i = 128; i < 160; ++i) rows.push_back(i);
  const Tensor independent = Pixels(Rows(train.images, rows));
  const auto   tf_images   = LoadNpz(recorded / "images.npz");
  check::Require(gan::BitEqual(ref, tf_images.at("real_reference")) && gan::BitEqual(independent, tf_images.at("independent_real")),
                 "CIFAR reference/independent images equal tflow's recorded extractor inputs BIT EXACT");
  const Tensor singan = tf_images.at("singan");
  check::Require(singan.shape() == nn::Shape({32, 32, 32, 3}), "recorded SinGAN chain output 32 x 32x32 RGB");
  std::vector<std::int64_t> reverse, collapse, limited;
  for (int i = 0; i < 32; ++i) reverse.push_back(31 - i), collapse.push_back(0), limited.push_back(i % 4);
  const gan::Arrays images{{"real_reference", ref},
                           {"independent_real", independent},
                           {"singan", singan},
                           {"collapsed_real", Rows(ref, collapse)},
                           {"four_repeated_real", Rows(ref, limited)},
                           {"copied_heldout", Rows(ref, reverse)}};
  gan::SaveArrays((root / "images.npz").string(), images);

  std::map<std::string, gan::Arrays> fields;
  for (std::string name : {"real_reference", "independent_real", "singan"}) {
    fields[name] = LoadNpz(recorded / (name + "_inception.npz"));
    check::Require(fields[name].at("features").shape() == nn::Shape({32, 2048}) &&
                       fields[name].at("probabilities").shape() == nn::Shape({32, 1008}),
                   "full Inception features and 1008 class probabilities");
    gan::RequireFinite(fields[name], "finite frozen features/probabilities/logits");
  }
  for (const auto& [name, index] : std::vector<std::pair<std::string, std::vector<std::int64_t>>>{
           {"collapsed_real", collapse}, {"four_repeated_real", limited}, {"copied_heldout", reverse}})
    for (const auto& [key, value] : fields.at("real_reference")) fields[name].emplace(key, Rows(value, index));

  std::ofstream table(root / "scores.tsv"), json(root / "metrics.json");
  table.precision(17);
  json.precision(17);
  table << "variant\tfid\tinception_score\tsplit_mean\tsplit_population_std\texact_unique_images\ttflow_fid\n";
  json << "{\"samples_per_variant\":32,\"feature_dimension\":2048,\"classes\":1008,\"split_count\":4,\"split_size\":8,"
          "\"train_new_epochs\":0,\"features_from_tflow_record\":true,\"results\":{";
  bool   comma          = false;
  double worst_fid_diff = 0;
  for (const auto& [name, rgb] : images) {
    const auto& f     = fields.at(name);
    const auto  score = gan::Frechet(fields.at("real_reference").at("features"), f.at("features"));
    gan::SaveArrays((root / (name + "_frechet.npz")).string(), {score.begin(), score.end()});
    const double is   = gan::InceptionScore(f.at("probabilities"));
    double       mean = 0, var = 0;
    std::vector<double> split;
    for (int k = 0; k < 4; ++k) {
      std::vector<std::int64_t> indices;
      for (int j = 0; j < 8; ++j) indices.push_back(k * 8 + j);
      split.push_back(gan::InceptionScore(Rows(f.at("probabilities"), indices)));
      mean += split.back() / 4;
    }
    for (auto x : split) var += (x - mean) * (x - mean) / 4;
    std::set<std::vector<float>> unique;
    const auto                   pixels = rgb.contiguous().to_vector<float>();
    for (int i = 0; i < 32; ++i) unique.emplace(pixels.begin() + i * 3072, pixels.begin() + (i + 1) * 3072);
    const double fid    = Scalar(score, "fid");
    const double tf_fid = LoadNpz(recorded / (name + "_frechet.npz")).at("fid").item();
    worst_fid_diff      = std::max(worst_fid_diff, std::abs(fid - tf_fid) / std::max(1., std::abs(tf_fid)));
    table << name << '\t' << fid << '\t' << is << '\t' << mean << '\t' << std::sqrt(var) << '\t' << unique.size() << '\t'
          << tf_fid << '\n';
    if (comma) json << ',';
    comma = true;
    json << '"' << name << "\":{\"fid\":" << fid << ",\"inception_score\":" << is << ",\"split_mean\":" << mean
         << ",\"split_population_std\":" << std::sqrt(var) << ",\"unique_images\":" << unique.size()
         << ",\"tflow_fid\":" << tf_fid << '}';
    if (name == "collapsed_real") check::Near(is, 1, 3e-6, "one image collapse IS 1");
    if (name == "copied_heldout" || name == "real_reference") check::Near(fid, 0, 1e-7, "held-out copy can obtain FID 0 without learning");
    nn::save_image_grid(rgb, (root / (name + ".png")).string(), 8, 0, 255, 0, 220, 3);
    std::cout << name << ": FID=" << fid << " (tflow " << tf_fid << "), IS=" << is << ", exact unique=" << unique.size()
              << std::endl;
  }
  check::Require(worst_fid_diff < 1e-6, "datamunge FID reproduces tflow's recorded FID on the same features");
  json << "},\"max_relative_fid_difference_vs_tflow\":" << worst_fid_diff << "}\n";
  check::Require(bool(json) && bool(table), "complete GAN metric reports");
  std::cout << "FID/IS/low rank/unequal N/same moments/collapse/copy PASS (Inception features: tflow record; max relative "
               "FID difference vs tflow "
            << worst_fid_diff << ")\nArtifacts: " << root.string() << "\nAll gan_diagnostics checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path data = paths::Dataset("cifar10"), out = paths::OutputRoot() / "models" / "gan_diagnostics";
    std::filesystem::path recorded = paths::DataRoot().parent_path() / "models" / "gan_diagnostics";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("data=", 0) == 0)
        data = a.substr(5);
      else if (a.rfind("features=", 0) == 0)
        recorded = a.substr(9);
      else if (a.rfind("out=", 0) == 0)
        out = a.substr(4);
      else
        throw std::invalid_argument("expected data=PATH features=PATH out=PATH");
    }
    Run(data, recorded, out);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
