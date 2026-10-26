// CMD: just run-nn progressive_gan
// Progressive growing of GANs (Karras et al. 2018), 4x4 -> 8x8 MNIST:
// stable4, fade8 and stable8 in three default epochs.
//
// Port of tflow's progressive_gan. Preparation (in ~/projects/tflow_examples):
// just autoencoder-assets; the data are read from $DATAMUNGE_DATA_DIR/saliency
// (saliency_data.npz: 1000 fit / 200 validation / 200 test digits), resampled
// 28 -> 8 bilinear (half-pixel). G: 16 -> 64 tanh = 4x4x4 pixel-normed ->
// 3x3 conv (pixel norm) -> old 1x1 sigmoid RGB; new block: nearest x2, two 3x3
// convs (pixel norm) -> new 1x1 sigmoid RGB; image = alpha new8 + (1 - alpha)
// up(old4). D mirrors it: old path avg-pools 8 -> 4 into its 1x1 RGB, new path
// RGB + two convs + avg pool, faded, core conv + 64 -> 1 score. LSGAN
// objectives. Eight parameter groups with their own Adam (.0002, beta1 .5);
// active groups: epoch 1 stable4 {G core, old RGB}, epoch 2 fade8 (alpha =
// (step + 1) / 32) all, epoch 3+ stable8 without the old RGB paths. Per batch
// the D phase captures ALL active D gradients before any D write, then the G
// phase captures ALL active G gradients through the updated D. 32 batches of
// 32 per epoch; fixed final checkpoint; labels unused. No WGAN-GP, equalized
// learning rate or minibatch std in this small demonstration (as tflow).
//
// Data, noise banks and batch order are tflow's exactly; weights use
// datamunge's initializer unless init=PATH names a tflow initial_arrays.npz
// (then the run follows TF's trajectory up to float rounding). tflow's
// per-stage/per-epoch .npz dumps, optimizer-state dumps and index.html are not
// written. Outputs (data.npz, fixed_noise.npz, history, batches,
// initial/final checkpoint .npz, smooth_gradients.npz, outputs.npz,
// fade_controls.npz, epoch*.png, fade_grid.png, metrics.json) go to
// output/nn/models/progressive_gan.
// Options: epochs=3 data=PATH out=PATH init=PATH.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include "common/paths.hpp"
#include "common/progressive_gan.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

double Value(const gan::Arrays& a, const std::string& key) { return a.at(key).item(); }

/// 28x28 -> 8x8 bilinear with half-pixel centres (scale 3.5).
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

/// 8x8 images at 8x with 4-pixel gray-220 gaps.
void Mosaic(const std::filesystem::path& path, const Tensor& images, std::int64_t columns) {
  nn::save_image_grid(nn::reshape(images, {-1, 8, 8, 1}), path.string(), columns, 0, 1, 4, 220 / 255., 8);
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  const auto pools = gan::LoadSaliency(data);
  std::filesystem::create_directories(root);
  std::vector<Tensor> resized;
  gan::Arrays         records;
  for (int k = 0; k < 3; ++k) {
    resized.push_back(Resize8(pools[k].images));
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    records.emplace(split + "_original", pools[k].images);
    records.emplace(split + "_images", resized.back());
    records.emplace(split + "_labels", pools[k].labels);
    records.emplace(split + "_source_ids", pools[k].source);
  }
  gan::SaveArrays((root / "data.npz").string(), records);
  const Tensor vz = gan::Gaussian(200, 16, 6481), tz = gan::Gaussian(200, 16, 6491);
  gan::SaveArrays((root / "fixed_noise.npz").string(), {{"validation_z", vz}, {"test_z", tz}});
  gan::ProgressiveGanModel model;
  if (!init.empty()) model.Load(init);
  const int count[] = {1236, 5, 296, 5, 213, 8, 296, 8};
  for (int i = 0; i < 8; ++i) check::Require(model.Count(i) == count[i], "progressive parameter-group counts");
  const auto initial = model.AllArrays();
  gan::SaveArrays((root / "initial_arrays.npz").string(), initial);
  gan::SaveArrays((root / "initial_outputs.npz").string(), model.Evaluate(resized[2], tz, 0));
  nn::BatchOrder order(1000, 32, 6497);
  int            batches_count = 0, group_updates = 0;
  gan::Arrays    old_at_stable;
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  batches.precision(17);
  history << "epoch\tstage\talpha\td_loss\tg_loss\n";
  batches << "epoch\tstep\tstage\talpha\trow\tsource_id\n";
  const auto source = nn::cast(pools[0].source, nn::DType::Int64).to_vector<std::int64_t>();
  Tensor     fade_x, fade_z;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    const int         stage = epoch == 1 ? 0 : epoch == 2 ? 1 : 2;
    const std::string stage_name = stage == 0 ? "stable4" : stage == 1 ? "fade8" : "stable8";
    const Tensor      noise = gan::Gaussian(1000, 16, 6503 + epoch);
    order.begin_epoch();
    int step = 0;
    if (epoch == 3) old_at_stable = model.AllArrays();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const float  alpha = stage == 0 ? 0 : stage == 1 ? float(step + 1) / 32 : 1;
      const Tensor x = gan::Rows(resized[0], rows), z = gan::Rows(noise, rows);
      if (stage == 1 && step == 0) fade_x = x, fade_z = z;
      for (bool generator : {false, true}) {
        const auto active   = gan::ProgressiveGanModel::Active(stage, generator);
        const auto captured = model.Gradient(active, x, z, alpha);
        const auto before   = model.AllArrays();
        for (int group : active) {
          model.Apply(group, captured);
          ++group_updates;
        }
        const auto after = model.AllArrays();
        for (const auto& [key, value] : before)
          if (std::find(active.begin(), active.end(), key[0] - '0') == active.end())
            check::Require(gan::BitEqual(value, after.at(key)), "inactive/opposite group frozen including old/new RGB paths");
      }
      for (auto row : rows)
        batches << epoch << '\t' << batches_count << '\t' << stage << '\t' << alpha << '\t' << row << '\t' << source[row]
                << '\n';
      ++batches_count;
      ++step;
    }
    check::Require(step == 32, "32 compact batches / epoch");
    const auto after_epoch = model.AllArrays();
    if (stage == 0)
      for (const auto& [key, value] : initial)
        if (key[0] == '2' || key[0] == '3' || key[0] == '6' || key[0] == '7')
          check::Require(gan::BitEqual(value, after_epoch.at(key)), "new blocks UNTRAINED before fade");
    if (stage == 2)
      for (const auto& [key, value] : old_at_stable)
        if (key[0] == '1' || key[0] == '5')
          check::Require(gan::BitEqual(value, after_epoch.at(key)), "old RGB paths FROZEN after fade");
    const float alpha = stage == 0 ? 0 : 1;
    const auto  v     = model.Evaluate(resized[1], vz, alpha);
    history << epoch << '\t' << stage << '\t' << alpha << '\t' << Value(v, "d_loss") << '\t' << Value(v, "g_loss") << '\n';
    Mosaic(root / ("epoch" + std::to_string(epoch) + ".png"), nn::slice_axis(model.Sample(tz, alpha), 0, 0, 40), 8);
    std::cout << "epoch " << epoch << ": " << stage_name << ", alpha=" << alpha << ", D/G=" << Value(v, "d_loss") << '/'
              << Value(v, "g_loss") << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  gan::SaveArrays((root / "smooth_gradients.npz").string(), model.Gradient({0, 1, 2, 3, 4, 5, 6, 7}, fade_x, fade_z, .5f));
  const auto checkpoint = (root / "final.npz").string();
  model.Save(checkpoint);
  gan::ProgressiveGanModel deployed(false, {1, 2, 3, 4, 5, 6, 7, 8});
  deployed.Load(checkpoint);
  check::Require(gan::BitEqual(model.AllArrays(), deployed.AllArrays()), "all progressive groups restore exact");
  const auto   before    = deployed.AllArrays();
  const auto   output    = deployed.Evaluate(resized[2], tz, 1), old = deployed.Evaluate(resized[2], tz, 0);
  const Tensor generated = deployed.Sample(tz, 1);
  check::Require(gan::BitEqual(generated, output.at("new8")), "alpha 1 EXACT new8 endpoint");
  check::Require(gan::BitEqual(old.at("fake"), nn::upsample_nearest(old.at("old4"), 2)), "alpha 0 EXACT upsampled old4 endpoint");
  const double parity = gan::MaxAbsDiff(deployed.Sample(gan::Rows(tz, {0}), 1), gan::Rows(generated, {0}));
  check::Require(parity < 2e-6, "noise only single batch parity");
  gan::Arrays         controls;
  std::vector<Tensor> fade_rows(40);
  for (int k = 0; k < 5; ++k) {
    const Tensor image = deployed.Sample(tz, k * .25f);
    controls.emplace("alpha" + std::to_string(k), image);
    for (int b = 0; b < 8; ++b) fade_rows[b * 5 + k] = gan::Rows(image, {b});
  }
  const Tensor fade = nn::concat(fade_rows, 0);
  int          invalid = 0;
  for (float value : {-.1f, 1.1f}) try {
      deployed.Sample(tz, value);
    } catch (const std::exception&) {
      ++invalid;
    }
  check::Require(invalid == 2, "invalid fade alpha rejected");
  check::Require(gan::BitEqual(before, deployed.AllArrays()), "immutable all progressive stages");
  controls.emplace("old4", old.at("old4"));
  controls.emplace("new8", output.at("new8"));
  controls.emplace("gallery", fade);
  gan::SaveArrays((root / "fade_controls.npz").string(), controls);
  Mosaic(root / "fade_grid.png", fade, 5);
  gan::SaveArrays((root / "outputs.npz").string(), output);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":2067,\"active_parameters_stable4\":1462,\"active_parameters_fade\":2067,"
          "\"active_parameters_stable8\":2054,\"epochs\":"
       << epochs << ",\"batches\":" << batches_count << ",\"parameter_group_optimizer_updates\":" << group_updates
       << ",\"selected_epoch_fixed_final\":" << epochs << ",\"test_d_loss\":" << Value(output, "d_loss")
       << ",\"test_g_loss\":" << Value(output, "g_loss") << ",\"single_batch_max_error\":" << parity
       << ",\"invalid_alpha_rejected\":2,\"training_seconds\":" << seconds << "}\n";
  check::Require(bool(meta) && bool(history) && bool(batches), "complete progressive artifacts");
  std::cout << "fixed final=" << epochs << ", test D/G=" << Value(output, "d_loss") << '/' << Value(output, "g_loss")
            << ", active params 1462 -> 2067 -> 2054, group updates=" << group_updates
            << "; actual growth/alpha endpoints/inactive freeze/captured phase grads/restore/immutable PASS (training "
            << seconds << " s)\nArtifacts: " << root.string() << "\nAll progressive_gan checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::filesystem::path data = paths::Dataset("saliency"), out = paths::OutputRoot() / "models" / "progressive_gan";
    std::string           init;
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
    check::Require(epochs >= 3 && epochs <= 100, "three curriculum epochs minimum, extras stable8");
    Run(data, out, epochs, init);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
