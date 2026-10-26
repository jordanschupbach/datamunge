// CMD: just run-nn cycle_gan
// CycleGAN (Zhu et al. 2017) on unpaired, disjoint MNIST domains, with the
// coupled generator gradients captured together before either write.
//
// Port of tflow's cycle_gan. Preparation (in ~/projects/tflow_examples): just
// autoencoder-assets; the data are read from $DATAMUNGE_DATA_DIR/saliency
// (saliency_data.npz). Each pool is split per class: the first half of every
// digit class is domain A (original digits), the second half becomes domain B
// after a fixed 90-degree rotation + inversion (B = 1 - rot(x)), so the two
// domains are DISJOINT images and batches are unpaired (independent orders).
// Generators a_to_b and b_to_a: two-level U-Net (pix2pix's PixGenerate, 1390
// parameters each); critics: 4x4/2 conv 4 -> 4x4/2 conv 8 -> 3x3 patch score
// (661 each). G loss = LS adversarial (both directions) + 10 L1 cycle + 5 L1
// identity; critic loss .5 (mean (real - 1)^2 + mean fake^2). Separate Adam
// (.0002, beta1 .5) per group. Per batch: critic A, critic B, then ONE joint
// capture of both generator gradients applied to a_to_b and b_to_a (the
// first step shows a sequential recomputation would differ). Three default
// epochs of 32-row batches (16 steps each), fixed final epoch. The known
// rotation/inversion is used ONLY for diagnostic mapping MSEs.
//
// Data split and batch orders are tflow's exactly; weights use datamunge's
// initializer unless init=PATH names a tflow initial_arrays.npz. tflow's
// per-step .npz dumps and index.html are not written. Outputs (history,
// batches, checkpoints, outputs.npz, final_gradients.npz,
// diagnostic_known_mapping.npz, a_to_b.png / b_to_a.png galleries,
// metrics.json) go to output/nn/models/cycle_gan.
// Options: epochs=3 data=PATH out=PATH init=PATH.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include "common/cycle_gan.hpp"
#include "common/paths.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

double Value(const gan::Arrays& a, const std::string& key) { return a.at(key).item(); }

/// B = 1 - (image rotated): out(y, x) = 1 - in(27 - x, y) (inverse: 1 - in(x, 27 - y)).
Tensor Transform(const Tensor& image, bool inverse = false) {
  const auto         in = image.contiguous().to_vector<float>();
  std::vector<float> out(in.size());
  const std::int64_t n  = image.size(0);
  for (std::int64_t b = 0; b < n; ++b)
    for (int y = 0; y < 28; ++y)
      for (int x = 0; x < 28; ++x)
        out[b * 784 + y * 28 + x] = 1.f - (inverse ? in[b * 784 + x * 28 + 27 - y] : in[b * 784 + (27 - x) * 28 + y]);
  return nn::from_vector(out, image.shape());
}

struct Domains {
  Tensor a, b, ids_a, ids_b, original_b;
};
Domains Split(const gan::DigitPool& pool) {
  std::vector<std::int64_t> a, b;
  int                       seen[10] = {};
  const std::int64_t        half     = pool.images.size(0) / 20;
  const auto                labels   = nn::cast(pool.labels, nn::DType::Int64).to_vector<std::int64_t>();
  for (std::int64_t i = 0; i < pool.images.size(0); ++i) (seen[labels[i]]++ < half ? a : b).push_back(i);
  check::Require(a.size() == b.size(), "balanced DISJOINT unpaired domains");
  const Tensor original_b = gan::Rows(pool.images, b);
  return {gan::Rows(pool.images, a), Transform(original_b), gan::Rows(pool.source, a), gan::Rows(pool.source, b),
          original_b};
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  std::filesystem::create_directories(root);
  const auto           pools = gan::LoadSaliency(data);
  std::vector<Domains> domains;
  gan::Arrays          records;
  for (int k = 0; k < 3; ++k) {
    domains.push_back(Split(pools[k]));
    const std::string name = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    records.emplace(name + "_a", domains.back().a);
    records.emplace(name + "_b", domains.back().b);
    records.emplace(name + "_ids_a", domains.back().ids_a);
    records.emplace(name + "_ids_b", domains.back().ids_b);
  }
  gan::SaveArrays((root / "data.npz").string(), records);
  gan::CycleGanModel model;
  if (!init.empty()) model.Load(init);
  check::Require(model.Count(0) == 1390 && model.Count(1) == 1390 && model.Count(2) == 661 && model.Count(3) == 661,
                 "4102 params");
  std::size_t tensors = 0;
  for (int k = 0; k < 4; ++k) tensors += model.Group(k).size();
  check::Require(tensors == 32, "32 tensors");
  model.Save((root / "initial_arrays.npz").string());
  gan::SaveArrays((root / "initial_outputs.npz").string(), model.Evaluate(domains[2].a, domains[2].b));
  nn::BatchOrder order_a(500, 32, 6233), order_b(500, 32, 6247);
  int            steps                   = 0;
  double         wrong_sequential_effect = 0;
  Tensor         first_a, first_b;
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tg_loss\td_a_loss\td_b_loss\tcycle_loss_a\tcycle_loss_b\tidentity_loss_a\tidentity_loss_b\n";
  batches << "epoch\tstep\ta_row\tb_row\ta_source_id\tb_source_id\n";
  const auto ids_a   = nn::cast(domains[0].ids_a, nn::DType::Int64).to_vector<std::int64_t>();
  const auto ids_b   = nn::cast(domains[0].ids_b, nn::DType::Int64).to_vector<std::int64_t>();
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order_a.begin_epoch();
    order_b.begin_epoch();
    for (;;) {
      const auto rows_a = order_a.next(), rows_b = order_b.next();
      check::Require(rows_a.size() == rows_b.size(), "equal domain batch sizes but independent orders");
      if (rows_a.empty()) break;
      const Tensor a = gan::Rows(domains[0].a, rows_a), b = gan::Rows(domains[0].b, rows_b);
      if (!steps) {
        first_a = a, first_b = b;
        gan::SaveArrays((root / "first_forward.npz").string(), model.Evaluate(a, b));
      }
      for (int group : {2, 3}) {
        const auto captured = model.Gradient({group}, a, b);
        const auto before   = model.AllArrays();
        model.Apply(group, captured);
        gan::RequireOnlyOwnersChanged(before, model.AllArrays(), {std::to_string(group) + "/"},
                                      "D step changes ONLY its own critic", false);
      }
      const auto joint = model.Gradient({0, 1}, a, b);
      for (int group : {0, 1}) {
        const auto before = model.AllArrays();
        model.Apply(group, joint);
        gan::RequireOnlyOwnersChanged(before, model.AllArrays(), {std::to_string(group) + "/"},
                                      "captured G step changes ONLY own group", false);
        if (!steps && group == 0)
          for (const auto& [key, value] : model.Gradient({1}, a, b))
            wrong_sequential_effect = std::max(wrong_sequential_effect, gan::MaxAbsDiff(value, joint.at(key)));
      }
      for (std::size_t i = 0; i < rows_a.size(); ++i)
        batches << epoch << '\t' << steps << '\t' << rows_a[i] << '\t' << rows_b[i] << '\t' << ids_a[rows_a[i]]
                << '\t' << ids_b[rows_b[i]] << '\n';
      ++steps;
    }
    const auto v = model.Evaluate(domains[1].a, domains[1].b);
    history << epoch;
    for (auto key : {"g_loss", "d_a_loss", "d_b_loss", "cycle_loss_a", "cycle_loss_b", "identity_loss_a", "identity_loss_b"})
      history << '\t' << Value(v, key);
    history << '\n';
    std::cout << "epoch " << epoch << ": G=" << Value(v, "g_loss") << ", cycle A/B=" << Value(v, "cycle_loss_a") << "/"
              << Value(v, "cycle_loss_b") << ", D A/B=" << Value(v, "d_a_loss") << "/" << Value(v, "d_b_loss")
              << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  check::Require(wrong_sequential_effect > 1e-8, "coupled G gradients MUST be captured before either write");

  const auto checkpoint = (root / "final.npz").string();
  model.Save(checkpoint);
  gan::SaveArrays((root / "final_gradients.npz").string(), model.Gradient({0, 1, 2, 3}, first_a, first_b));
  gan::CycleGanModel deployed(false, {1, 2, 3, 4});
  deployed.Load(checkpoint);
  check::Require(gan::BitEqual(model.AllArrays(), deployed.AllArrays()), "all four groups restore exact");
  const auto   before = deployed.AllArrays();
  const auto   output = deployed.Evaluate(domains[2].a, domains[2].b);
  const Tensor ab = deployed.Translate(true, domains[2].a), ba = deployed.Translate(false, domains[2].b);
  check::Require(gan::BitEqual(ab, output.at("fake_b")) && gan::BitEqual(ba, output.at("fake_a")),
                 "SOURCE ONLY both translations without opposite domain images");
  double parity = 0;
  for (bool direction : {true, false})
    parity = std::max(parity, gan::MaxAbsDiff(deployed.Translate(direction, gan::Rows(direction ? domains[2].a : domains[2].b, {0})),
                                              gan::Rows(direction ? ab : ba, {0})));
  check::Require(parity < 2e-6, "both G single batch parity");
  check::Require(gan::BitEqual(before, deployed.AllArrays()), "immutable all four groups");
  gan::SaveArrays((root / "outputs.npz").string(), output);
  const Tensor true_b = Transform(domains[2].a), true_a = domains[2].original_b;
  gan::SaveArrays((root / "diagnostic_known_mapping.npz").string(), {{"a_to_b_target", true_b}, {"b_to_a_target", true_a}});
  auto mse = [](const Tensor& x, const Tensor& y) {
    return nn::mean(nn::square(nn::cast(x, nn::DType::Float64) - nn::cast(y, nn::DType::Float64))).item();
  };
  const double mse_ab = mse(ab, true_b), mse_ba = mse(ba, true_a);
  for (bool direction : {true, false}) {
    std::vector<Tensor> rows;
    for (std::int64_t i = 0; i < 10; ++i) {
      rows.push_back(gan::Rows(direction ? domains[2].a : domains[2].b, {i}));
      rows.push_back(gan::Rows(direction ? ab : ba, {i}));
      rows.push_back(gan::Rows(output.at(direction ? "cycle_a" : "cycle_b"), {i}));
      rows.push_back(gan::Rows(direction ? true_b : true_a, {i}));
    }
    const Tensor      gallery = nn::reshape(nn::concat(rows, 0), {40, 784});
    const std::string name    = direction ? "a_to_b" : "b_to_a";
    gan::SaveArrays((root / (name + "_gallery.npz")).string(), {{"images", gallery}});
    gan::Mosaic(root / (name + ".png"), gallery, 4);
  }
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":4102,\"epochs\":" << epochs << ",\"steps_per_group\":" << steps
       << ",\"optimizer_updates\":" << 4 * steps << ",\"selected_epoch_fixed_final\":" << epochs
       << ",\"wrong_sequential_generator_gradient_max_difference\":" << wrong_sequential_effect
       << ",\"diagnostic_mapping_a_to_b_mse\":" << mse_ab << ",\"diagnostic_mapping_b_to_a_mse\":" << mse_ba
       << ",\"single_batch_max_error\":" << parity;
  for (auto key : {"g_loss", "d_a_loss", "d_b_loss", "cycle_loss_a", "cycle_loss_b", "identity_loss_a", "identity_loss_b"})
    meta << ",\"test_" << key << "\":" << Value(output, key);
  meta << ",\"training_seconds\":" << seconds << "}\n";
  check::Require(bool(meta) && bool(history) && bool(batches), "complete CycleGAN records");
  std::cout << "fixed final=" << epochs << ", joint G snapshot wrong sequential difference=" << wrong_sequential_effect
            << ", mapping MSE=" << mse_ab << "/" << mse_ba
            << "; unpaired/joint grad/restore/source only/immutable PASS (training " << seconds << " s)\nArtifacts: "
            << root.string() << "\nAll cycle_gan checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::filesystem::path data = paths::Dataset("saliency"), out = paths::OutputRoot() / "models" / "cycle_gan";
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
    check::Require(epochs > 0 && epochs <= 100, "three default or explicit 1..100 epochs");
    Run(data, out, epochs, init);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
