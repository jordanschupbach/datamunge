// CMD: just run-nn star_gan
// StarGAN (Choi et al. 2018) on three disjoint synthetic MNIST domains with a
// single domain-conditioned generator and a source/domain discriminator.
//
// Port of tflow's star_gan. Preparation (in ~/projects/tflow_examples): just
// autoencoder-assets; the data are read from $DATAMUNGE_DATA_DIR/saliency
// (saliency_data.npz). Per pool, the first 3 x 30 (fit) or 3 x 6
// (validation/test) examples of every digit are assigned to domains 0
// (original), 1 (inverted 1 - x) and 2 (rotated 90 degrees), giving 900 / 180 /
// 180 disjoint images; the rest are recorded as unused. G (1609 parameters):
// U-Net on source ++ tiled one-hot target domain. D (688): shared 4/8-feature
// trunk with a 7x7 raw LSGAN source head and pooled 3 domain logits; no domain
// enters D's input. G = LS source + fake-domain CE + 10 L1 cycle
// reconstruction to the original domain; D = .5 LS source + real-domain CE.
// Separate Adam (.0002, beta1 .5), per batch D then G. Training targets are
// uniform domains from std::mt19937(6337 + epoch) (rejecting 2^32 - 1); fixed
// validation/test targets are the next domain. Three default epochs of 32-row
// batches (29 steps each), fixed final epoch. The known domain mapping is used
// ONLY for a diagnostic MSE.
//
// Data, targets and batch order are tflow's exactly; weights use datamunge's
// initializer unless init=PATH names a tflow initial_arrays.npz (then the run
// follows TF's trajectory up to float rounding). tflow's per-step .npz dumps
// and index.html are not written. Outputs (data.npz, fixed_targets.npz,
// history, batches, initial/final checkpoint .npz, final_gradients_group*.npz,
// outputs.npz, domain_controls.npz, diagnostic_known_mapping.npz,
// gallery.npz, all_domains.png, metrics.json) go to output/nn/models/star_gan.
// Options: epochs=3 data=PATH out=PATH init=PATH.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include "common/paths.hpp"
#include "common/star_gan.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

double Value(const gan::Arrays& a, const std::string& key) { return a.at(key).item(); }
std::vector<std::int32_t> Ints(const Tensor& t) { return nn::cast(t, nn::DType::Int32).contiguous().to_vector<std::int32_t>(); }
Tensor IntTensor(const std::vector<std::int32_t>& v) { return nn::from_vector(v, {std::int64_t(v.size())}); }

/// Domain 0 = original, 1 = 1 - x, 2 = rotated (out(y, x) = in(27 - x, y)).
Tensor DomainImages(const Tensor& original, const Tensor& domain) {
  const auto in = original.contiguous().to_vector<float>();
  const auto d  = Ints(domain);
  std::vector<float> out(in.size());
  for (std::size_t b = 0; b < d.size(); ++b)
    for (int y = 0; y < 28; ++y)
      for (int x = 0; x < 28; ++x)
        out[b * 784 + y * 28 + x] = d[b] == 0   ? in[b * 784 + y * 28 + x]
                                    : d[b] == 1 ? 1.f - in[b * 784 + y * 28 + x]
                                                : in[b * 784 + (27 - x) * 28 + y];
  return nn::from_vector(out, original.shape());
}

struct Pool {
  Tensor image, original, domain, digit, source, unused;
};
Pool Select(const gan::DigitPool& data) {
  const int                 per = data.images.size(0) == 1000 ? 30 : 6;
  std::vector<std::int64_t> rows;
  std::vector<std::int32_t> domains, unused;
  int                       seen[10] = {};
  const auto                labels = Ints(data.labels), source = Ints(data.source);
  for (std::int64_t i = 0; i < data.images.size(0); ++i) {
    const int index = seen[labels[i]]++;
    if (index < per * 3) {
      rows.push_back(i);
      domains.push_back(index / per);
    } else
      unused.push_back(source[i]);
  }
  check::Require(rows.size() == std::size_t(per * 30) && unused.size() == std::size_t(data.images.size(0) - per * 30),
                 "explicit 900 fit / 180 heldout with unused IDs declared");
  const Tensor original = gan::Rows(data.images, rows), domain = IntTensor(domains);
  return {DomainImages(original, domain), original, domain, nn::cast(gan::Rows(data.labels, rows), nn::DType::Int32),
          gan::Rows(data.source, rows), IntTensor(unused)};
}
/// Uniform domains v % 3 from std::mt19937(seed), rejecting 2^32 - 1; also the raw draws.
std::pair<Tensor, Tensor> Targets(int n, std::uint32_t seed) {
  std::mt19937              rng(seed);
  std::vector<std::int32_t> labels;
  std::vector<double>       raw;
  for (int i = 0; i < n; ++i) {
    std::uint32_t v;
    do v = rng();
    while (v == 4294967295u);
    labels.push_back(int(v % 3));
    raw.push_back(v);
  }
  return {IntTensor(labels), nn::from_vector(raw, {n})};
}
Tensor NextDomain(const Tensor& original) {
  auto d = Ints(original);
  for (auto& v : d) v = (v + 1) % 3;
  return IntTensor(d);
}
std::int64_t Argmax(const std::vector<float>& p, std::int64_t row) {
  std::int64_t best = 0;
  for (int c = 1; c < 3; ++c)
    if (p[row * 3 + c] > p[row * 3 + best]) best = c;
  return best;
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  std::filesystem::create_directories(root);
  const auto        originals = gan::LoadSaliency(data);
  std::vector<Pool> pools;
  gan::Arrays       records;
  for (int k = 0; k < 3; ++k) {
    pools.push_back(Select(originals[k]));
    const std::string name = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    records.emplace(name + "_all_original", originals[k].images);
    records.emplace(name + "_all_digit", originals[k].labels);
    records.emplace(name + "_all_source_ids", originals[k].source);
    records.emplace(name + "_images", pools[k].image);
    records.emplace(name + "_original", pools[k].original);
    records.emplace(name + "_domains", pools[k].domain);
    records.emplace(name + "_digits", pools[k].digit);
    records.emplace(name + "_source_ids", pools[k].source);
    records.emplace(name + "_unused_source_ids", pools[k].unused);
  }
  gan::SaveArrays((root / "data.npz").string(), records);
  const Tensor vt = NextDomain(pools[1].domain), tt = NextDomain(pools[2].domain);
  gan::SaveArrays((root / "fixed_targets.npz").string(), {{"validation", vt}, {"test", tt}});
  gan::StarGanModel model;
  if (!init.empty()) model.Load(init);
  check::Require(model.Count(0) == 1609 && model.Count(1) == 688 && model.Group(0).size() + model.Group(1).size() == 18,
                 "2297 params 18 tensors");
  model.Save((root / "initial_arrays.npz").string());
  gan::SaveArrays((root / "initial_outputs.npz").string(), model.Evaluate(pools[2].image, pools[2].domain, tt));
  nn::BatchOrder order(900, 32, 6323);
  int            steps = 0;
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tdiscriminator_loss\tgenerator_loss\treal_domain_loss\tfake_domain_loss\treconstruction_loss\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto source = Ints(pools[0].source);
  Tensor     first_x, first_original, first_target;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    const auto targets = Targets(900, 6337 + epoch);
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const Tensor x = gan::Rows(pools[0].image, rows), original = gan::Rows(pools[0].domain, rows),
                   target = gan::Rows(targets.first, rows);
      if (!steps) first_x = x, first_original = original, first_target = target;
      for (int group : {1, 0}) {
        const auto before = model.AllArrays();
        model.Update(group, x, original, target);
        gan::RequireOnlyOwnersChanged(before, model.AllArrays(), {std::to_string(group) + "/"},
                                      "each StarGAN step updates ONLY own group", !steps);
      }
      for (auto row : rows) batches << epoch << '\t' << steps << '\t' << row << '\t' << source[row] << '\n';
      ++steps;
    }
    const auto v = model.Evaluate(pools[1].image, pools[1].domain, vt);
    history << epoch;
    for (auto key : {"discriminator_loss", "generator_loss", "real_domain_loss", "fake_domain_loss", "reconstruction_loss"})
      history << '\t' << Value(v, key);
    history << '\n';
    std::cout << "epoch " << epoch << ": D=" << Value(v, "discriminator_loss") << ", G=" << Value(v, "generator_loss")
              << ", reconstruction L1=" << Value(v, "reconstruction_loss") << std::endl;
  }
  const double seconds    = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  const auto   checkpoint = (root / "final.npz").string();
  model.Save(checkpoint);
  for (int group = 0; group < 2; ++group)
    gan::SaveArrays((root / ("final_gradients_group" + std::to_string(group) + ".npz")).string(),
                    model.Gradient(group, first_x, first_original, first_target));
  gan::StarGanModel deployed(false, {1, 2});
  deployed.Load(checkpoint);
  check::Require(gan::BitEqual(model.AllArrays(), deployed.AllArrays()), "all G/D source-domain heads restore exact");
  const auto   before     = deployed.AllArrays();
  const auto   output     = deployed.Evaluate(pools[2].image, pools[2].domain, tt);
  const Tensor prediction = deployed.Generate(pools[2].image, tt);
  check::Require(gan::BitEqual(prediction, output.at("fake")), "source + target domain ONLY generation exact");
  const auto classify = deployed.Classify(pools[2].image);
  check::Require(gan::BitEqual(classify.at("source"), output.at("real_source")) &&
                     gan::BitEqual(classify.at("domain_probability"), output.at("real_domain_probability")),
                 "IMAGE ONLY D without domain/noise feed exact");
  const double parity =
      gan::MaxAbsDiff(deployed.Generate(gan::Rows(pools[2].image, {0}), gan::Rows(tt, {0})), gan::Rows(prediction, {0}));
  check::Require(parity < 2e-6, "single batch parity");
  std::vector<std::int32_t> all(180, 0);
  const Tensor p0 = deployed.Generate(pools[2].image, IntTensor(all));
  std::fill(all.begin(), all.end(), 1);
  const Tensor p1 = deployed.Generate(pools[2].image, IntTensor(all));
  std::fill(all.begin(), all.end(), 2);
  const Tensor p2           = deployed.Generate(pools[2].image, IntTensor(all));
  const double domain_effect = std::max(gan::MaxAbsDiff(p0, p1), gan::MaxAbsDiff(p0, p2));
  check::Require(domain_effect > 1e-5, "target domain changes G pixels");
  int rejected = 0;
  for (int invalid : {-1, 3}) {
    std::fill(all.begin(), all.end(), 0);
    all[0] = invalid;
    try {
      deployed.Generate(pools[2].image, IntTensor(all));
    } catch (const std::exception&) {
      ++rejected;
    }
  }
  try {
    deployed.Generate(pools[2].image, gan::Rows(tt, {0}));
  } catch (const std::exception&) {
    ++rejected;
  }
  check::Require(rejected == 3, "invalid domains rejected before one hot");
  check::Require(gan::BitEqual(before, deployed.AllArrays()), "all inference immutable");
  gan::SaveArrays((root / "outputs.npz").string(), output);
  gan::SaveArrays((root / "domain_controls.npz").string(), {{"target0_prediction", p0},
                                                            {"target1_prediction", p1},
                                                            {"target2_prediction", p2},
                                                            {"image_only_source", classify.at("source")},
                                                            {"image_only_domain_probability", classify.at("domain_probability")}});
  const Tensor known = DomainImages(pools[2].original, tt);
  gan::SaveArrays((root / "diagnostic_known_mapping.npz").string(), {{"target", known}});
  const double mse =
      nn::mean(nn::square(nn::cast(prediction, nn::DType::Float64) - nn::cast(known, nn::DType::Float64))).item();
  double     real_accuracy = 0, fake_accuracy = 0;
  const auto rp = output.at("real_domain_probability").contiguous().to_vector<float>(),
             fp = output.at("fake_domain_probability").contiguous().to_vector<float>();
  const auto domain = Ints(pools[2].domain), targets = Ints(tt), digits = Ints(pools[2].digit);
  for (int b = 0; b < 180; ++b) real_accuracy += Argmax(rp, b) == domain[b], fake_accuracy += Argmax(fp, b) == targets[b];
  std::vector<std::int32_t> gallery_ids;
  std::vector<Tensor>       gallery_rows;
  for (int d = 0; d < 3; ++d)
    for (int digit = 0; digit < 3; ++digit) {
      int row = -1;
      for (int b = 0; b < 180 && row < 0; ++b)
        if (domain[b] == d && digits[b] == digit) row = b;
      check::Require(row >= 0, "first sample per 3x3 source-domain/digit grid");
      gallery_ids.push_back(row);
      for (const Tensor* t : std::vector<const Tensor*>{&pools[2].image, &p0, &p1, &p2, &output.at("reconstruction"), &known})
        gallery_rows.push_back(gan::Flat784(gan::Rows(*t, {row})));
    }
  const Tensor gallery = nn::concat(gallery_rows, 0);
  gan::SaveArrays((root / "gallery.npz").string(), {{"images", gallery}, {"rows", IntTensor(gallery_ids)}});
  gan::Mosaic(root / "all_domains.png", gallery, 6);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":2297,\"epochs\":" << epochs << ",\"steps_per_group\":" << steps
       << ",\"optimizer_updates\":" << 2 * steps << ",\"selected_epoch_fixed_final\":" << epochs
       << ",\"diagnostic_known_mapping_mse\":" << mse << ",\"test_real_domain_accuracy\":" << real_accuracy / 180
       << ",\"test_generated_requested_domain_accuracy\":" << fake_accuracy / 180
       << ",\"target_domain_pixel_max_effect\":" << domain_effect << ",\"single_batch_max_error\":" << parity
       << ",\"invalid_domain_controls_rejected\":" << rejected;
  for (auto key : {"discriminator_loss", "generator_loss", "source_loss_g", "source_loss_d", "real_domain_loss",
                   "fake_domain_loss", "reconstruction_loss"})
    meta << ",\"test_" << key << "\":" << Value(output, key);
  meta << ",\"training_seconds\":" << seconds << "}\n";
  check::Require(bool(meta) && bool(history) && bool(batches), "complete StarGAN records");
  std::cout << "fixed final=" << epochs << ", real/generated domain accuracy=" << real_accuracy / 180 << "/"
            << fake_accuracy / 180 << ", mapping MSE=" << mse << ", test D/G=" << Value(output, "discriminator_loss") << "/"
            << Value(output, "generator_loss")
            << "; single G/3 domains/classifier/roundtrip/restore/immutable PASS (training " << seconds
            << " s)\nArtifacts: " << root.string() << "\nAll star_gan checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::filesystem::path data = paths::Dataset("saliency"), out = paths::OutputRoot() / "models" / "star_gan";
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
