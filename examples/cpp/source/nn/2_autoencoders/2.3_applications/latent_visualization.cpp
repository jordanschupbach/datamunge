// CMD: just run-nn latent_visualization
// Reuse frozen three-epoch MNIST undercomplete autoencoder;0additionaltraining.
// Prepare:just autoencoder-assets;just run undercomplete_autoencoder.
//
// Port of tflow's latent_visualization (helpers: examples/nn/common/latent_visualization.hpp,
// PCA from examples/nn/common/pca.hpp). Data: $DATAMUNGE_DATA_DIR/saliency. The frozen model is
// read from encoder=DIR: best.npz of our undercomplete_autoencoder port (default
// output/nn/models/undercomplete_autoencoder) or tflow's model_arrays.npz when pointed at its
// output/models/undercomplete_autoencoder. PCA uses nn::svd (LAPACK); component signs are
// arbitrary, so projections/scatter plots may be mirrored relative to tflow while every
// distance, reconstruction and neighbour list is sign-invariant. Outputs (analysis.npz,
// model_arrays.npz, ae_latent_pca2.png, pixel_pca2.png, metrics.json, encoder_metrics.json,
// index.html) go to output/nn/models/latent_visualization. Options: data=PATH encoder=PATH out=PATH.

#include <fstream>
#include <iostream>
#include <map>

#include "common/latent_visualization.hpp"

namespace {

namespace nn = datamunge::nn;

double Mse(const nn::Tensor& raw, const nn::Tensor& target, bool clip = false) {
  const auto x = ae::DoubleMatrix(raw), y = ae::DoubleMatrix(target);
  check::Require(x.shape() == y.shape(), "aligned PCA reconstruction");
  const auto a = x.to_vector<double>(), b = y.to_vector<double>();
  double     sum = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    double v = a[i];
    if (clip) v = std::max(0., std::min(1., v));
    sum += (v - b[i]) * (v - b[i]);
  }
  return sum / double(a.size());
}

int Scatter(const std::filesystem::path& path, const nn::Tensor& fit, const nn::Tensor& test, const nn::Tensor& labels) {
  const int          width = 600, height = 600, left = 40, top = 40, span = 520;
  std::vector<float> pixels(height * width * 3, 255.f);
  auto set = [&](int y, int x, int c, float v) { pixels[(y * width + x) * 3 + c] = v; };
  const auto f = fit.to_vector<double>(), t = test.to_vector<double>();
  const auto l = labels.contiguous().to_vector<int>();
  double     low[2] = {INFINITY, INFINITY}, high[2] = {-INFINITY, -INFINITY};
  for (std::int64_t i = 0; i < fit.size(0); ++i)
    for (int j = 0; j < 2; ++j) {
      low[j]  = std::min(low[j], f[i * 2 + j]);
      high[j] = std::max(high[j], f[i * 2 + j]);
    }
  for (int j = 0; j < 2; ++j) {
    const double pad = std::max(1e-6, .05 * (high[j] - low[j]));
    low[j] -= pad;
    high[j] += pad;
  }
  const int color[10][3] = {{31, 119, 180}, {255, 127, 14},  {44, 160, 44},   {214, 39, 40},  {148, 103, 189},
                            {140, 86, 75},  {227, 119, 194}, {127, 127, 127}, {188, 189, 34}, {23, 190, 207}};
  for (int pos = left; pos <= left + span; ++pos)
    for (int c = 0; c < 3; ++c) {
      set(top + span, pos, c, 0);
      set(pos, left, c, 0);
    }
  int clipped = 0;
  for (int split = 0; split < 2; ++split) {
    const auto& values = split ? t : f;
    const auto  n      = std::int64_t(values.size() / 2);
    for (std::int64_t b = 0; b < n; ++b) {
      const double u = (values[b * 2] - low[0]) / (high[0] - low[0]), v = (values[b * 2 + 1] - low[1]) / (high[1] - low[1]);
      if (split) clipped += u < 0 || u > 1 || v < 0 || v > 1;
      const int x = left + int(std::lround(span * std::max(0., std::min(1., u)))),
                y = top + span - int(std::lround(span * std::max(0., std::min(1., v)))), radius = split ? 3 : 1;
      for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx)
          if (dx * dx + dy * dy <= radius * radius)
            for (int c = 0; c < 3; ++c) set(y + dy, x + dx, c, split ? float(color[l[b]][c]) : 210.f);
    }
  }
  nn::save_png(nn::from_vector(pixels, {height, width, 3}), path.string(), 0.0, 255.0);
  return clipped;
}

void Run(const std::filesystem::path& data, const std::filesystem::path& encoder, const std::filesystem::path& root) {
  auto pools = ae::LoadSaliency(data);
  std::filesystem::create_directories(root);
  ae::UndercompleteModel model(false);
  ae::LoadFrozen(model.parameters, encoder, "run undercomplete_autoencoder first (three default epochs)");
  const auto                           before = ae::Snapshot(model.parameters);
  std::vector<std::vector<nn::Tensor>> f;
  std::map<std::string, nn::Tensor>    arrays;
  for (int k = 0; k < 3; ++k) {
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    f.push_back(model.Evaluate(pools[k].images));
    arrays.emplace(split + "_images", pools[k].images);
    arrays.emplace(split + "_labels", pools[k].labels);
    arrays.emplace(split + "_source_ids", pools[k].source);
    arrays.emplace(split + "_code", f[k][0]);
    arrays.emplace(split + "_ae_reconstruction", f[k][1]);
  }
  auto pixels = ae::FitLatentPca(pools[0].images, pools[1].images, pools[2].images, 32),
       codes  = ae::FitLatentPca(f[0][0], f[1][0], f[2][0], 2);
  for (const auto& [name, p] : std::vector<std::pair<std::string, ae::LatentProjection>>{{"pixel_pca", pixels},
                                                                                          {"code_pca", codes}}) {
    arrays.emplace(name + "_mean", p.mean);
    arrays.emplace(name + "_components", p.components);
    arrays.emplace(name + "_variances", p.variances);
    arrays.emplace(name + "_fit", p.fit);
    arrays.emplace(name + "_validation", p.validation);
    arrays.emplace(name + "_test", p.test);
    arrays.emplace(name + "_fit_reconstruction", p.fit_reconstruction);
    arrays.emplace(name + "_validation_reconstruction", p.validation_reconstruction);
    arrays.emplace(name + "_test_reconstruction", p.test_reconstruction);
  }
  auto ae2decoded = model.Decode(nn::cast(codes.test_reconstruction, nn::DType::Float32).contiguous());
  arrays.emplace("ae_projected2_decoder_reconstruction", ae2decoded);
  auto pixel_fit2 = ae::FirstColumns(pixels.fit, 2), pixel_test2 = ae::FirstColumns(pixels.test, 2);
  // pixel 2-D PCA reconstruction using FIT-only mean/components
  auto pixel2 = (nn::matmul(pixel_test2, ae::FirstColumns(pixels.components, 2), false, true) + pixels.mean).contiguous();
  arrays.emplace("pixel_pca2_test_reconstruction", pixel2);
  auto neighbors = ae::NearestFitRows(pools[0].images, pools[2].images, 10), ae32 = ae::NearestFitRows(f[0][0], f[2][0], 10),
       ae2 = ae::NearestFitRows(codes.fit, codes.test, 10), pca32 = ae::NearestFitRows(pixels.fit, pixels.test, 10),
       pca2 = ae::NearestFitRows(pixel_fit2, pixel_test2, 10);
  const std::map<std::string, nn::Tensor> lists{
      {"pixel", neighbors}, {"ae32", ae32}, {"ae2", ae2}, {"pca32", pca32}, {"pca2", pca2}};
  for (const auto& [key, value] : lists) arrays.emplace(key + "_neighbors", value);
  const double ae_mse = Mse(f[2][1], pools[2].images), ae2_mse = Mse(ae2decoded, pools[2].images),
               pca_mse = Mse(pixels.test_reconstruction, pools[2].images),
               pca_clipped = Mse(pixels.test_reconstruction, pools[2].images, true), pca2_mse = Mse(pixel2, pools[2].images);
  const int ac = Scatter(root / "ae_latent_pca2.png", codes.fit, codes.test, pools[2].labels),
            pc = Scatter(root / "pixel_pca2.png", pixel_fit2, pixel_test2, pools[2].labels);
  check::Require(before == ae::Snapshot(model.parameters), "projection/retrieval/decoder calls preserve all pretrained weights");
  nn::save_npz((root / "analysis.npz").string(), nn::NamedTensors(arrays.begin(), arrays.end()));
  nn::save_npz((root / "model_arrays.npz").string(), ae::Arrays(model.parameters));
  const auto opts = std::filesystem::copy_options::overwrite_existing;
  if (std::filesystem::exists(encoder / "metrics.json"))
    std::filesystem::copy_file(encoder / "metrics.json", root / "encoder_metrics.json", opts);
  if (std::filesystem::exists(data / "manifest.json"))
    std::filesystem::copy_file(data / "manifest.json", root / "data_manifest.json", opts);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"additional_training_updates\":0,\"encoder_parameters\":50992,\"code_dimensions\":32,\"display_dimensions\":2,"
          "\"fit_records\":1000,\"validation_records\":200,\"test_records\":200,\"ae_test_mse\":"
       << ae_mse << ",\"ae_projected2_decoder_test_mse\":" << ae2_mse << ",\"pixel_pca32_test_mse\":" << pca_mse
       << ",\"pixel_pca32_clipped_test_mse\":" << pca_clipped << ",\"pixel_pca2_test_mse\":" << pca2_mse
       << ",\"ae_display_clipped_test_points\":" << ac << ",\"pixel_display_clipped_test_points\":" << pc;
  for (const auto& [key, value] : lists)
    meta << ",\"" << key << "_test_1nn_label_accuracy\":" << ae::NeighborLabelAccuracy(value, pools[0].labels, pools[2].labels)
         << ",\"" << key << "_top10_pixel_neighbor_overlap\":" << ae::NeighborOverlap(neighbors, value);
  meta << "}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Autoencoder dimensionality reduction</title><style>body{font:16px "
          "system-ui;margin:30px;max-width:1250px}.plots{display:flex;gap:20px}img{max-width:48%}</style><h1>Dimensionality "
          "reduction and latent visualization</h1><p>Frozen verified THREEepoch MNIST784->32ReLU->784sigmoid "
          "autoencoder,50992params;0newtraining.1000fit200validation200test original balanced digit images. Code is32D;its "
          "TWOdimensional display is FIT-only PCA of codes,not a trained2Dautoencoder. Side-by-side pixelPCA2 baseline. PCA "
          "centering/components/variance and plot limits use FIT only;heldout labels color points AFTER fitting,no supervised "
          "representation fitting or label-selected checkpoints.</p><div class='plots'><img src='ae_latent_pca2.png' "
          "alt='Two-dimensional PCA of32Dautoencoder codes'><img src='pixel_pca2.png' alt='Two-dimensional PCA of784Dpixels'>"
          "</div><p>Left:AEcode PCA2,right:pixelPCA2. Axis1 increases right,axis2 up. Graydots1000fit,largecoloredpoints200"
          "heldouttest;digits0..9 colors blue/orange/green/red/purple/brown/pink/gray/olive/cyan. Axes are independent fit "
          "min/max+5%padding;they do not support comparing absolute scale between panels. Heldout out-of-range points clipped to "
          "fit plot boundary,counts AE="
       << ac << ",pixel=" << pc << ". Every heldout point retained. PCA signs are arbitrary.</p><p>Test per-pixelMSE AE32="
       << ae_mse << ",AEcodePCA2 inverse-projected then decoded=" << ae2_mse << ",pixelPCA32 raw=" << pca_mse
       << ",pixelPCA32clipped0..1=" << pca_clipped << ",pixelPCA2raw=" << pca2_mse
       << ". Raw PCA can leave0..1;clipping is reported separately. Same32coordinate budget does not mean same optimized "
          "loss,training time or compression bits. PCA2 baselines clarify lost display information.</p><p>analysis.npz exports "
          "allfit/validation/testcode+PCA projections/reconstructions and deterministic Euclidean top10fitting-neighbor indices. "
          "metrics.json reports overlap with pixel-space top10 and posthoc1NNdigit-label accuracy for pixels/AE32/AE2/pixelPCA32/"
          "pixelPCA2. Tie-break smaller fitting row. Labels are diagnostic only;distance geometry differs,projection can lose "
          "neighbors,and colored clusters do not certify semantic separation. No tSNE/UMAP,nonlinear2Dmanifold or learnedcodec "
          "claim.</p>";
  check::Require(bool(meta) && bool(html), "complete latent visualization artifacts");
  std::cout << "0newupdates;AE32MSE=" << ae_mse << ",AEdisplay2decodedMSE=" << ae2_mse << ",PCA32MSE=" << pca_mse
            << ",PCA2MSE=" << pca2_mse << ",neighboroverlapAE32=" << ae::NeighborOverlap(neighbors, ae32)
            << ",AE2=" << ae::NeighborOverlap(neighbors, ae2)
            << ";frozenEncoder/fitOnlyPCA/retrieval/immutable PASS\nArtifacts:" << (root / "index.html").string() << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path data    = paths::Dataset("saliency"),
                          encoder = paths::OutputRoot() / "models/undercomplete_autoencoder",
                          out     = paths::OutputRoot() / "models/latent_visualization";
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
