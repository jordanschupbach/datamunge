// CMD: just run-nn big_gan
// Compact BigGAN (Brock et al. 2019) on 8x8 digits: conditional residual block,
// hierarchical latent, spectral norm, class-conditional BN, projection D,
// orthogonal regularization and rejection truncation.
//
// Port of tflow's big_gan. Preparation (in ~/projects/tflow_examples): just
// autoencoder-assets; the data are read from $DATAMUNGE_DATA_DIR/saliency
// (saliency_data.npz) and bilinearly resized (half-pixel centres) to 8x8.
// G (789 parameters): z 12 = three chunks of 4; z0 -> SN dense 4x4x4 stem;
// [class embedding 4, z1] and [class embedding 4, z2] drive two conditional BNs
// (gain 1 + SN dense, bias SN dense) of a residual nearest-up block
// (BN-ReLU-up-conv3-BN-ReLU-conv3 + up(conv1) shortcut), then output BN
// (gamma/beta), ReLU, SN conv3 sigmoid. D (241): residual down block (SN conv3,
// ReLU, SN conv3, avg-pool + pooled SN conv1 shortcut), ReLU sum-pool
// features, SN linear score + projection <SN embed(y), phi(x)>. Every kernel
// is spectrally normalized by v^T W u with stopped vectors; each phase first
// advances its OWN vectors by one power iteration (16 at initialisation).
// Hinge losses; G adds the modified orthogonal penalty .5e-4 ||W^T W o
// (1 - I)||^2. Adam (.0002, beta1 0, beta2 .999) per network; the G phase also
// applies BN moving averages (momentum .9, unbiased variance) from the moments
// captured with its gradient. Per batch D then G; three default epochs of
// 32-row batches (96 steps each), fixed final epoch. Inference uses moving BN
// statistics; truncation redraws |z| > limit (inf, 2, 1, .5).
//
// Priors and batch order are tflow's exactly; weights use datamunge's
// initializer unless init=PATH names a tflow initial_arrays.npz (all five
// groups, including SN vectors and BN statistics). tflow's per-step .npz dumps
// and index.html are not written. Outputs (history, batches, checkpoints,
// outputs.npz, class/hierarchical/truncation controls, class_grid.png,
// truncation.png, generated.png, metrics.json) go to output/nn/models/big_gan.
// Options: epochs=3 data=PATH out=PATH init=PATH seed=N
// (seed=N > 0 replaces the default weight-initializer seeds; for seed-spread runs).

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>

#include "common/big_gan.hpp"
#include "common/paths.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

double Value(const gan::Arrays& a, const std::string& key) { return a.at(key).item(); }

/// tflow Resize8: half-pixel-centre bilinear 28x28 -> 8x8 (scale 3.5).
Tensor Resize8(const Tensor& images) {
  const auto         in = images.contiguous().to_vector<float>();
  const std::int64_t n  = images.size(0);
  std::vector<float> out(n * 64);
  for (std::int64_t b = 0; b < n; ++b)
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x) {
        const double py = (y + .5) * 3.5 - .5, px = (x + .5) * 3.5 - .5;
        const int    iy = int(std::floor(py)), ix = int(std::floor(px));
        double       v = 0;
        for (int dy = 0; dy < 2; ++dy)
          for (int dx = 0; dx < 2; ++dx)
            v += (dy ? py - iy : 1 - (py - iy)) * (dx ? px - ix : 1 - (px - ix)) * in[b * 784 + (iy + dy) * 28 + ix + dx];
        out[b * 64 + y * 8 + x] = float(v);
      }
  return nn::from_vector(out, {n, 8, 8, 1});
}

/// 8x8 digits at 8x with 4-pixel gray-220 gaps.
void Mosaic(const std::filesystem::path& path, const Tensor& images, std::int64_t columns) {
  nn::save_image_grid(nn::reshape(images, {-1, 8, 8, 1}), path.string(), columns, 0, 1, 4, 220 / 255., 8);
}

struct Prior {
  Tensor z, proposals;
};
/// tflow Draw: Box-Muller pairs from mt19937(seed); proposals with |x| > limit are rejected.
Prior Draw(std::int64_t n, std::uint32_t seed, double limit = std::numeric_limits<double>::infinity()) {
  check::Require(n > 0 && n <= 10000 && limit > 0 && !std::isnan(limit), "valid Gaussian rejection limit");
  std::mt19937        rng(seed);
  std::vector<double> proposals;
  std::vector<float>  z(n * 12);
  bool                spare  = false;
  double              cached = 0;
  for (std::int64_t i = 0; i < n * 12;) {
    double candidate;
    if (spare) {
      candidate = cached;
      spare     = false;
    } else {
      const double u = (double(rng()) + .5) / 4294967296., v = (double(rng()) + .5) / 4294967296.;
      const double radius = std::sqrt(-2 * std::log(u)), angle = 6.283185307179586476925286766559 * v;
      candidate = radius * std::cos(angle);
      cached    = radius * std::sin(angle);
      spare     = true;
    }
    proposals.push_back(candidate);
    if (std::abs(candidate) <= limit) z[i++] = float(candidate);
    check::Require(proposals.size() < std::size_t(n * 12) * 1000, "bounded teaching rejection sampler work");
  }
  return {nn::from_vector(z, {n, 12}), nn::from_vector(proposals, {std::int64_t(proposals.size())})};
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init, std::uint64_t seed) {
  const auto pools = gan::LoadSaliency(data);
  std::filesystem::create_directories(root);
  std::vector<Tensor> resized;
  gan::Arrays         records;
  for (int k = 0; k < 3; ++k) {
    resized.push_back(Resize8(pools[k].images));
    records.emplace(std::string(k == 0 ? "fit" : k == 1 ? "validation" : "test") + "_images", resized.back());
  }
  gan::SaveArrays((root / "data.npz").string(), records);
  const auto validation = Draw(200, 6841), test = Draw(200, 6857);
  gan::SaveArrays((root / "validation_prior.npz").string(), {{"z", validation.z}, {"proposals", validation.proposals}});
  gan::SaveArrays((root / "test_prior.npz").string(), {{"z", test.z}, {"proposals", test.proposals}});
  gan::BigGanModel model(true, seed ? seed : 6803, seed ? seed + 1 : 6823);
  if (!init.empty()) model.Load(init);
  std::size_t tensors = 0;
  for (int k = 0; k < 5; ++k) tensors += model.Group(k).size();
  check::Require(model.Count(0) == 789 && model.Count(1) == 241 && model.Count(2) == 241 && model.Count(3) == 77 &&
                     model.Count(4) == 24 && tensors == 60,
                 "1030 trainable + 342 state values, 60 tensors");
  model.Save((root / "initial_arrays.npz").string());
  gan::SaveArrays((root / "initial_outputs.npz").string(), model.Evaluate(resized[2], test.z, pools[2].labels));
  nn::BatchOrder order(1000, 32, 6863);
  int            steps = 0;
  Tensor         first_x, first_z, first_labels;
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\td_loss\tg_source\tg_orthogonal\tg_loss\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto source  = nn::cast(pools[0].source, nn::DType::Int64).to_vector<std::int64_t>();
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    const auto noise = Draw(1000, 6871 + epoch);
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const Tensor x = gan::Rows(resized[0], rows), z = gan::Rows(noise.z, rows),
                   condition = gan::Rows(pools[0].labels, rows);
      if (!steps) {
        first_x = x, first_z = z, first_labels = condition;
        gan::SaveArrays((root / "first_forward.npz").string(), model.Evaluate(x, z, condition, true));
        const auto adv = model.Component(false, x, z, condition), ortho = model.Component(true, x, z, condition);
        const auto full = model.Capture(0, x, z, condition);
        for (const auto& [key, value] : adv) {
          const auto a = value.to_vector<float>(), o = ortho.at(key).to_vector<float>(), f = full.at(key).to_vector<float>();
          for (std::size_t i = 0; i < a.size(); ++i)
            check::Near(double(a[i]) + o[i], f[i], 3e-5, "G partials = adversarial PLUS modified orthogonal");
        }
        gan::SaveArrays((root / "first_adversarial_gradients.npz").string(), adv);
        gan::SaveArrays((root / "first_orthogonal_gradients.npz").string(), ortho);
      }
      for (int group : {1, 0}) {
        const auto before = model.AllArrays();
        model.Advance(group);
        const auto after_sn = model.AllArrays();
        gan::RequireOnlyOwnersChanged(before, after_sn, {std::to_string(group + 2) + "/"},
                                      "SN iteration changes ONLY own phase vectors", false);
        model.ApplyCaptured(group, model.Capture(group, x, z, condition));
        std::vector<std::string> owners{std::to_string(group) + "/"};
        if (group == 0) owners.push_back("4/");
        gan::RequireOnlyOwnersChanged(after_sn, model.AllArrays(), owners,
                                      "D freezes ALL G weights/SN/BN; G freezes ALL D weights/SN", false);
      }
      for (auto row : rows) batches << epoch << '\t' << steps << '\t' << row << '\t' << source[row] << '\n';
      ++steps;
    }
    const auto v = model.Evaluate(resized[1], validation.z, pools[1].labels);
    history << epoch;
    for (auto key : {"d_loss", "g_source", "g_orthogonal", "g_loss"}) history << '\t' << Value(v, key);
    history << '\n';
    std::cout << "epoch " << epoch << ": D/G=" << Value(v, "d_loss") << '/' << Value(v, "g_loss")
              << ", orthogonal=" << Value(v, "g_orthogonal") << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

  const auto checkpoint = (root / "final.npz").string();
  model.Save(checkpoint);
  for (int group = 0; group < 2; ++group)
    gan::SaveArrays((root / ("final_gradients_group" + std::to_string(group) + ".npz")).string(),
                    model.Capture(group, first_x, first_z, first_labels));
  gan::BigGanModel deployed(false, 1, 2);
  deployed.Load(checkpoint);
  check::Require(gan::BitEqual(model.AllArrays(), deployed.AllArrays()), "ALL trainable/SN/conditional BN vars restore exact");
  const auto   before    = deployed.AllArrays();
  const auto   output    = deployed.Evaluate(resized[2], test.z, pools[2].labels);
  const Tensor generated = deployed.Generate(test.z, pools[2].labels);
  check::Require(gan::BitEqual(generated, output.at("fake_inference")), "class + hierarchical z ONLY generation with FROZEN stats");
  const double parity =
      gan::MaxAbsDiff(deployed.Generate(gan::Rows(test.z, {0}), gan::Rows(pools[2].labels, {0})), gan::Rows(generated, {0}));
  check::Require(parity < 3e-6, "inference BN single batch parity");
  std::vector<std::int64_t> grid_rows;
  std::vector<std::int32_t> grid_classes;
  for (int c = 0; c < 10; ++c)
    for (int b = 0; b < 8; ++b) grid_rows.push_back(b), grid_classes.push_back(c);
  const Tensor grid_z = gan::Rows(test.z, grid_rows), grid_labels = nn::from_vector(grid_classes, {80});
  const Tensor grid   = deployed.Generate(grid_z, grid_labels);
  const double class_effect =
      gan::MaxAbsDiff(nn::slice_axis(grid, 0, 8, 80), nn::tile(nn::slice_axis(grid, 0, 0, 8), {9, 1, 1, 1}));
  auto changed_classes = pools[2].labels.contiguous().to_vector<std::int32_t>();
  for (auto& c : changed_classes) c = (c + 1) % 10;
  const Tensor original_scores = deployed.Score(resized[2], pools[2].labels);
  const Tensor changed_scores  = deployed.Score(resized[2], nn::from_vector(changed_classes, {200}));
  check::Require(gan::BitEqual(original_scores, output.at("real_score")), "projection D image + class ONLY");
  const double projection_effect = gan::MaxAbsDiff(original_scores, changed_scores);
  check::Require(class_effect > 1e-5 && projection_effect > 1e-5, "G class conditioning AND D projection class paths active");
  gan::SaveArrays((root / "class_controls.npz").string(),
                  {{"z", grid_z}, {"labels", grid_labels}, {"fake", grid}, {"changed_real_scores", changed_scores}});
  Mosaic(root / "class_grid.png", grid, 8);
  gan::Arrays hierarchy;
  for (int chunk = 0; chunk < 3; ++chunk) {
    auto values = test.z.contiguous().to_vector<float>();
    for (int b = 0; b < 200; ++b)
      for (int j = 0; j < 4; ++j) values[b * 12 + chunk * 4 + j] = 0;
    const Tensor z    = nn::from_vector(values, {200, 12});
    const Tensor fake = deployed.Generate(z, pools[2].labels);
    check::Require(gan::MaxAbsDiff(fake, generated) > 1e-5, "ALL THREE hierarchical latent chunks affect G");
    hierarchy.emplace("zero_chunk" + std::to_string(chunk) + "_z", z);
    hierarchy.emplace("zero_chunk" + std::to_string(chunk) + "_fake", fake);
  }
  gan::SaveArrays((root / "hierarchical_controls.npz").string(), hierarchy);
  std::vector<std::int32_t> trunc_classes(200);
  for (int i = 0; i < 200; ++i) trunc_classes[i] = i % 10;
  const Tensor        trunc_labels = nn::from_vector(trunc_classes, {200});
  std::vector<Tensor> trunc_tiles(32);
  for (int k = 0; k < 4; ++k) {
    const double limit = k == 0 ? std::numeric_limits<double>::infinity() : k == 1 ? 2 : k == 2 ? 1 : .5;
    const auto   prior = Draw(200, 6907, limit);
    gan::SaveArrays((root / ("truncated_prior" + std::to_string(k) + ".npz")).string(),
                    {{"z", prior.z}, {"proposals", prior.proposals}});
    const Tensor fake = deployed.Generate(prior.z, trunc_labels);
    gan::SaveArrays((root / ("truncated_output" + std::to_string(k) + ".npz")).string(),
                    {{"fake", fake}, {"labels", trunc_labels}});
    for (std::int64_t b = 0; b < 8; ++b) trunc_tiles[b * 4 + k] = gan::Rows(fake, {b});
  }
  const Tensor trunc_gallery = nn::concat(trunc_tiles, 0);
  gan::SaveArrays((root / "truncation_gallery.npz").string(), {{"images", trunc_gallery}});
  Mosaic(root / "truncation.png", trunc_gallery, 4);
  int  invalid = 0;
  auto bad     = grid_classes;
  for (int c : {-1, 10}) {
    bad[0] = c;
    try {
      deployed.Generate(grid_z, nn::from_vector(bad, {80}));
    } catch (const std::exception&) {
      ++invalid;
    }
  }
  try {
    deployed.Generate(test.z, gan::Rows(pools[2].labels, {0}));
  } catch (const std::exception&) {
    ++invalid;
  }
  check::Require(invalid == 3, "invalid class conditions rejected before OneHot");
  check::Require(gan::BitEqual(before, deployed.AllArrays()), "all truncation/conditional inference keeps SN/BN weights immutable");
  gan::SaveArrays((root / "outputs.npz").string(), output);
  Mosaic(root / "generated.png", nn::slice_axis(generated, 0, 0, 40), 8);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"trainable_parameters\":1030,\"nontrainable_state_values\":342,\"epochs\":" << epochs
       << ",\"steps_per_group\":" << steps << ",\"optimizer_updates\":" << 2 * steps
       << ",\"selected_epoch_fixed_final\":" << epochs << ",\"generator_class_pixel_max_effect\":" << class_effect
       << ",\"discriminator_projection_score_max_effect\":" << projection_effect
       << ",\"single_batch_max_error\":" << parity << ",\"invalid_class_rejected\":" << invalid;
  for (auto key : {"d_loss", "g_source", "g_orthogonal", "g_loss"}) meta << ",\"test_" << key << "\":" << Value(output, key);
  meta << ",\"training_seconds\":" << seconds << "}\n";
  check::Require(bool(meta) && bool(history) && bool(batches), "complete BigGAN records");
  std::cout << "fixed final=" << epochs << ", test D/G=" << Value(output, "d_loss") << "/" << Value(output, "g_loss")
            << ", class/projection effects=" << class_effect << '/' << projection_effect
            << "; hierarchical/CCBN/residual/SN/projection/orthogonal/rejection/restore/immutable PASS (training "
            << seconds << " s)\nArtifacts: " << root.string() << "\nAll big_gan checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::filesystem::path data = paths::Dataset("saliency"), out = paths::OutputRoot() / "models" / "big_gan";
    std::string           init;
    std::uint64_t         seed = 0;
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
      else if (a.rfind("seed=", 0) == 0)
        seed = std::stoull(a.substr(5));
      else
        throw std::invalid_argument("expected epochs=N data=PATH out=PATH init=PATH seed=N");
    }
    check::Require(epochs > 0 && epochs <= 100, "three default or explicit 1..100 epochs");
    Run(data, out, epochs, init, seed);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
