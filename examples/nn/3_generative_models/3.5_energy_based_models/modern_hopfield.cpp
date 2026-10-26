// CMD: just run-nn modern_hopfield
// Training-free continuous Hopfield retrieval using FIT-only MNIST memories and
// actual energy derivatives.
//
// Port of tflow's modern_hopfield (common/{nn,ml}/modern_hopfield.h ->
// examples/nn/common/modern_hopfield.hpp). Data: tflow's saliency MNIST pools
// (paths::Dataset("saliency"), `just autoencoder-assets` in tflow_examples; 1000 fit
// / 200 validation / 200 test), half-pixel bilinear 28 -> 8 gray. FIT-only 64-pixel
// mean; every vector centered and L2-normalized. The first fit image of each class
// gives the 10 stored unit-norm patterns. Queries: 100 noisy copies (10 per memory,
// + .15 N(0, 1) per coordinate from tflow's Gaussian bank 24001, no renormalization)
// and 200 heldout test images. Retrieval x' = softmax(beta x M^T) M is
// nn::scaled_dot_product_attention(x, M, M, scale = beta); 8 updates at beta 1, 8, 32.
// No trainable parameters, no training.
//
// Checks (as tflow): energy gradient == x - x' (actual autograd derivative) at every
// update; energy never increases (f32); zero query -> uniform attention; memory
// permutation preserves retrieval; fixed inputs repeat bit-exactly; single-row batch
// parity; memory/query bytes immutable; complete records. TF parity: data (gray
// pools, memories, noise, queries) checked bit-identical to tflow's
// output/models/modern_hopfield/data.npz; trajectories, probes (full Jacobians) and
// metrics.tsv compared with TF's record. Deviations: no independent_audit.json.
// Options: data=DIR out=DIR tf=DIR threads=4. Output: output/nn/models/modern_hopfield.
#include "common/modern_hopfield.hpp"

namespace {

using namespace ebm;

std::pair<Tensor, Tensor> Encode(const Tensor& x, const std::vector<double>& center) {
  const auto          v = x.to_vector<float>();
  const auto          n = x.size(0);
  std::vector<float>  values(v.size());
  std::vector<double> norms(n);
  for (std::int64_t b = 0; b < n; ++b) {
    double norm = 0;
    for (int j = 0; j < 64; ++j) norm += std::pow(double(v[b * 64 + j]) - center[j], 2);
    norm = std::sqrt(norm);
    check::Require(norm > 1e-6, "nonzero centered image norm");
    norms[b] = norm;
    for (int j = 0; j < 64; ++j) values[b * 64 + j] = float((double(v[b * 64 + j]) - center[j]) / norm);
  }
  return {nn::from_vector(values, {n, 64}), nn::from_vector(norms, {n})};
}

void Png(const fs::path& path, const Tensor& images, int columns) { bpix::Png(path, nn::reshape(images, {-1, 8, 8, 1}), columns, 2); }

void Heatmap(const fs::path& path, const Tensor& weights) {
  const auto         n = weights.size(0);
  const auto         w = weights.to_vector<float>();
  std::vector<float> rgb(static_cast<std::size_t>(n * 20 * 300 * 3), 220.f);
  for (std::int64_t b = 0; b < n; ++b)
    for (int m = 0; m < 10; ++m) {
      const double a = w[b * 10 + m], color[] = {255 * (1 - a), 255 * (1 - .65 * a), 255};
      for (int y = 0; y < 19; ++y)
        for (int x = 0; x < 29; ++x)
          for (int c = 0; c < 3; ++c) rgb[static_cast<std::size_t>(((b * 20 + y) * 300 + m * 30 + x) * 3 + c)] = float(std::floor(color[c] + .5));
    }
  bpix::PngBytes(path, rgb, n * 20, 300);
}

Tensor WeightedImages(const Tensor& weights, const Tensor& gray) {
  const auto         n = weights.size(0);
  const auto         w = weights.to_vector<float>(), g = gray.to_vector<float>();
  std::vector<float> out(static_cast<std::size_t>(n * 64));
  for (std::int64_t b = 0; b < n; ++b)
    for (int j = 0; j < 64; ++j) {
      double sum = 0;
      for (int m = 0; m < 10; ++m) sum += double(w[b * 10 + m]) * g[m * 64 + j];
      out[b * 64 + j] = float(sum);
    }
  return nn::from_vector(out, {n, 64});
}

int Argmax(const std::vector<float>& a, std::int64_t b) {
  int best = 0;
  for (int c = 1; c < 10; ++c)
    if (a[b * 10 + c] > a[b * 10 + best]) best = c;
  return best;
}

void Run(const Options& o) {
  const fs::path root = o.out, tf = o.tf, data_root = o.extra.at("data");
  const auto     pools = ae::LoadSaliency(data_root);
  Map            data;
  std::vector<Tensor> gray;
  for (int k = 0; k < 3; ++k) {
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    gray.push_back(nn::reshape(bpix::Resize8(pools[k].images), {-1, 64}));
    data[split + "_original"]   = pools[k].images;
    data[split + "_gray"]       = gray.back();
    data[split + "_labels"]     = pools[k].labels;
    data[split + "_source_ids"] = pools[k].source;
  }
  ae::CopyManifests(data_root, root);
  const auto          fit = gray[0].to_vector<float>();
  std::vector<double> center(64);
  for (int j = 0; j < 64; ++j) {
    double mean = 0;
    for (int b = 0; b < 1000; ++b) mean += double(fit[b * 64 + j]) / 1000;
    center[j] = mean;
  }
  const auto                fit_labels = pools[0].labels.to_vector<std::int32_t>(), fit_source = pools[0].source.to_vector<std::int32_t>();
  const auto                test_labels = pools[2].labels.to_vector<std::int32_t>();
  std::vector<std::int64_t> selected;
  for (int c = 0; c < 10; ++c)
    for (int b = 0; b < 1000; ++b)
      if (fit_labels[b] == c) {
        selected.push_back(b);
        break;
      }
  check::Require(selected.size() == 10, "one fit memory per class");
  const Tensor memory_gray        = Rows(gray[0], selected);
  const auto [memory, norms]      = Encode(memory_gray, center);
  const auto [heldout, heldout_n] = Encode(gray[2], center);
  const auto raw_noise = ae::GaussianNoise({100, 64}, 1, 24001).to_vector<double>();
  const auto mv = memory.to_vector<float>(), hv = heldout.to_vector<float>();
  const auto nv = norms.to_vector<double>();
  std::vector<float>        queries(300 * 64), noisy_gray(100 * 64);
  std::vector<std::int32_t> targets(300), memory_indices(10), memory_sources(10);
  for (int c = 0; c < 10; ++c) memory_indices[c] = int(selected[c]), memory_sources[c] = fit_source[selected[c]];
  for (int b = 0; b < 100; ++b) {
    targets[b] = b / 10;
    for (int j = 0; j < 64; ++j) {
      queries[b * 64 + j]    = float(double(mv[(b / 10) * 64 + j]) + .15 * raw_noise[b * 64 + j]);
      noisy_gray[b * 64 + j] = float(center[j] + nv[b / 10] * queries[b * 64 + j]);
    }
  }
  for (int b = 0; b < 200; ++b) {
    targets[100 + b] = test_labels[b];
    std::copy_n(hv.data() + b * 64, 64, queries.data() + (100 + b) * 64);
  }
  const Tensor queries_t = nn::from_vector(queries, {300, 64}), noisy_gray_t = nn::from_vector(noisy_gray, {100, 64});
  data["fit_center"]          = nn::from_vector(center, {64});
  data["memory_gray"]         = memory_gray;
  data["memory"]              = memory;
  data["memory_norms"]        = norms;
  data["memory_fit_indices"]  = Ints(memory_indices);
  data["memory_source_ids"]   = Ints(memory_sources);
  data["heldout"]             = heldout;
  data["heldout_norms"]       = heldout_n;
  data["noise"]               = nn::from_vector(raw_noise, {100, 64});
  data["queries"]             = queries_t;
  data["noisy_gray"]          = noisy_gray_t;
  data["target_memory_class"] = Ints(targets);
  std::vector<std::int64_t> gallery_indices, probe_indices;
  for (int c = 0; c < 10; ++c) gallery_indices.push_back(c * 10), probe_indices.push_back(c * 10);
  for (int c = 0; c < 10; ++c)
    for (int b = 0; b < 200; ++b)
      if (test_labels[b] == c) {
        gallery_indices.push_back(100 + b);
        if (c < 6) probe_indices.push_back(100 + b);
        break;
      }
  std::vector<std::int32_t> gi(gallery_indices.begin(), gallery_indices.end()), pi(probe_indices.begin(), probe_indices.end());
  data["gallery_indices"] = Ints(gi);
  data["probe_indices"]   = Ints(pi);
  Save(root / "data.npz", data);
  CheckData(tf, data,
            {"fit_gray", "test_gray", "fit_center", "memory_gray", "memory", "memory_norms", "memory_fit_indices", "noise", "queries",
             "noisy_gray", "heldout", "target_memory_class", "gallery_indices", "probe_indices"});
  Png(root / "memories.png", memory_gray, 10);
  const ModernHopfieldModel model;
  const std::string         memory_bytes = Bits(memory), query_bytes = Bits(queries_t);
  std::ofstream             metrics(root / "metrics.tsv");
  metrics.precision(17);
  metrics << "beta\tnoisy_memory_initial_accuracy\tnoisy_memory_final_accuracy\theldout_initial_nearest_class_accuracy\theldout_final_"
             "class_accuracy\tmax_energy_increase\tfinal_update_max_residual\n";
  for (float beta : {1.f, 8.f, 32.f}) {
    const std::string   name = "beta" + std::to_string(int(beta));
    Tensor              state = queries_t;
    std::vector<Tensor> states{state}, energies, attentions, gradients, memory_gradients;
    Map                 first, last;
    double              energy_increase = 0, residual = 0;
    std::vector<float>  previous;
    for (int k = 0; k <= 8; ++k) {
      const auto out = model.Evaluate(state, memory, beta);
      if (!k) first = out;
      last              = out;
      const auto energy = out.at("energy").to_vector<float>();
      if (k)
        for (int b = 0; b < 300; ++b) energy_increase = std::max(energy_increase, double(energy[b]) - previous[b]);
      previous = energy;
      energies.push_back(out.at("energy"));
      attentions.push_back(out.at("attention"));
      gradients.push_back(out.at("state_gradient"));
      memory_gradients.push_back(out.at("memory_gradient"));
      const auto sg = out.at("state_gradient").to_vector<float>(), sv = state.to_vector<float>(), xv = out.at("next").to_vector<float>();
      for (std::size_t i = 0; i < sg.size(); ++i) {
        check::Near(sg[i], double(sv[i]) - xv[i], 2e-6, "energy gradient equals state minus attention retrieval");
        if (k == 8) residual = std::max(residual, std::abs(double(sv[i]) - xv[i]));
      }
      if (k < 8) {
        state = out.at("next");
        states.push_back(state);
      }
    }
    check::Require(energy_increase < 1e-5, "Hopfield retrieval energy does not increase within f32");
    Save(root / (name + "_trajectory.npz"), {{"states", nn::stack(states, 0)},
                                              {"energies", nn::stack(energies, 0)},
                                              {"attentions", nn::stack(attentions, 0)},
                                              {"state_gradients", nn::stack(gradients, 0)},
                                              {"memory_gradients", nn::stack(memory_gradients, 0)}});
    const Tensor probes = Rows(queries_t, probe_indices);
    Save(root / (name + "_probe.npz"), model.Evaluate(probes, memory, beta, true));
    const auto uniform = model.Evaluate(nn::zeros({16, 64}), memory, beta);
    for (float a : uniform.at("attention").to_vector<float>()) check::Near(a, .1, 1e-7, "zero query uniform attention");
    Save(root / (name + "_zero_control.npz"), uniform);
    std::vector<std::int64_t> reversed;
    for (int c = 0; c < 10; ++c) reversed.push_back(9 - c);
    const auto permutation = model.Evaluate(probes, Rows(memory, reversed), beta);
    const auto probe       = model.Evaluate(probes, memory, beta);
    check::Require(replay::MaxAbsDiff(probe.at("next"), permutation.at("next")) <= 2e-6,
                   "memory permutation preserves retrieval within reduction roundoff");
    Save(root / (name + "_permutation_control.npz"), permutation);
    const auto repeat = model.Evaluate(probes, memory, beta);
    for (const auto& [key, value] : probe) check::Require(Bits(value) == Bits(repeat.at(key)), "fixed input Hopfield repeat bitexact");
    const auto single = model.Evaluate(Rows(probes, {0}), memory, beta);
    check::Require(replay::MaxAbsDiff(single.at("next"), Rows(probe.at("next"), {0})) <= 2e-6, "single batch Hopfield parity");
    const Tensor first_images = WeightedImages(first.at("attention"), memory_gray), final_images = WeightedImages(last.at("attention"), memory_gray);
    const auto   la = last.at("attention").to_vector<float>(), fa = first.at("attention").to_vector<float>();
    const auto   mg = memory_gray.to_vector<float>(), tg = gray[2].to_vector<float>(), ng = noisy_gray, fi = first_images.to_vector<float>(),
               li = final_images.to_vector<float>();
    std::vector<float> gallery(100 * 64), heat(20 * 10);
    for (int row = 0; row < 20; ++row) {
      const auto b      = gallery_indices[row];
      const int  chosen = Argmax(la, b);
      for (int j = 0; j < 64; ++j) {
        gallery[(row * 5) * 64 + j]     = b < 100 ? mg[(b / 10) * 64 + j] : tg[(b - 100) * 64 + j];
        gallery[(row * 5 + 1) * 64 + j] = b < 100 ? ng[b * 64 + j] : tg[(b - 100) * 64 + j];
        gallery[(row * 5 + 2) * 64 + j] = fi[b * 64 + j];
        gallery[(row * 5 + 3) * 64 + j] = li[b * 64 + j];
        gallery[(row * 5 + 4) * 64 + j] = mg[chosen * 64 + j];
      }
      std::copy_n(la.data() + b * 10, 10, heat.data() + row * 10);
    }
    const Tensor gallery_t = nn::from_vector(gallery, {100, 64}), heat_t = nn::from_vector(heat, {20, 10});
    Save(root / (name + "_gallery.npz"),
         {{"images", gallery_t}, {"final_attention", heat_t}, {"first_weighted_images", first_images}, {"final_weighted_images", final_images}});
    Png(root / (name + "_retrieval.png"), gallery_t, 5);
    Heatmap(root / (name + "_attention.png"), heat_t);
    int initial_mem = 0, final_mem = 0, initial_test = 0, final_test = 0;
    for (int b = 0; b < 300; ++b) {
      const int a = Argmax(fa, b), c = Argmax(la, b);
      if (b < 100) initial_mem += a == targets[b], final_mem += c == targets[b];
      else initial_test += a == targets[b], final_test += c == targets[b];
    }
    metrics << beta << '\t' << initial_mem / 100. << '\t' << final_mem / 100. << '\t' << initial_test / 200. << '\t' << final_test / 200.
            << '\t' << energy_increase << '\t' << residual << '\n';
    std::cout << name << ":noisyMemoryAccuracy=" << initial_mem << "->" << final_mem << "/100,heldoutClass=" << initial_test << "->"
              << final_test << "/200,maxEnergyIncrease=" << energy_increase << ",residual=" << residual << std::endl;
  }
  metrics.flush();
  check::Require(memory_bytes == Bits(memory) && query_bytes == Bits(queries_t), "all memory and input bytes immutable");
  std::ofstream meta(root / "metrics.json");
  meta << "{\"trainable_parameters\":0,\"memory_patterns\":10,\"memory_coordinates\":640,\"noisy_memory_queries\":100,\"heldout_"
          "queries\":200,\"retrieval_updates\":8,\"betas\":[1,8,32],\"training_epochs\":0}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Modern Hopfield retrieval</title><style>body{font:16px system-ui;margin:30px;"
          "max-width:1100px}img{max-width:100%;image-rendering:pixelated}</style><h1>Continuous modern Hopfield associative memory</h1>"
          "<p><a href='https://arxiv.org/abs/2008.02217'>Modern Hopfield</a> energy E(x) = .5||x||^2 - logsumexp(beta M x)/beta. "
          "Retrieval x_next = softmax(beta x M^T) M (tied keys and values, no learned projection or optimizer); the actual energy "
          "gradient equals x - x_next. Real MNIST, half-pixel 28->8 gray, fit-only centering and unit-norm vectors; the first fit "
          "image per class gives 10 stored patterns. 100 noisy-memory queries (+.15 Gaussian) and 200 heldout queries. Datamunge "
          "port of tflow's modern_hopfield.</p><h2>Stored fit memories, classes 0..9</h2><img src='memories.png'>";
  for (int b : {1, 8, 32})
    html << "<h2>beta=" << b << "</h2><p>Columns truth / query / one-update weighted stored image / eight-update weighted stored image "
                                "/ hard-selected stored image; heatmap rows same cases, columns memories 0..9.</p><img src='beta"
         << b << "_retrieval.png'><img src='beta" << b << "_attention.png'>";
  html << "<p>See metrics.tsv and metrics.json.</p>";
  check::Require(bool(metrics) && bool(meta) && bool(html), "complete modern Hopfield records");
  std::cout << "fixedMemory/actualEnergyGrad/energyDescent/zeroQuery/permutation/repeat/singleBatch/immutable PASS\n";

  // ---- TF parity ----
  if (!fs::exists(tf / "metrics.tsv")) {
    std::cout << "no TF record at " << tf.string() << "; parity checks skipped\nArtifacts: " << root.string() << "\n";
    return;
  }
  std::cout << "TF parity (" << tf.string() << "):\n";
  double worst = 0;
  for (const std::string name : {"beta1", "beta8", "beta32"}) {
    worst = std::max(worst, bpix::Compare(tf, name + "_trajectory.npz", Load(root / (name + "_trajectory.npz"))));
    worst = std::max(worst, bpix::Compare(tf, name + "_probe.npz", Load(root / (name + "_probe.npz"))));
    bpix::Compare(tf, name + "_gallery.npz", Load(root / (name + "_gallery.npz")));
  }
  for (int column = 1; column <= 4; ++column) {
    const auto a = replay::History(root / "metrics.tsv", column), b = replay::History(tf / "metrics.tsv", column);
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i)
      check::Require(std::abs(a[i] - b[i]) < 1e-12, "retrieval accuracies identical to TF's metrics.tsv");
  }
  std::printf("  metrics.tsv accuracies (4 columns x 3 betas) identical to TF's\n");
  check::Require(worst < 1e-4, "trajectories and Jacobian probes match TF's record");
  std::cout << "Artifacts: " << root.string() << "\nAll modern_hopfield checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Run(ParseOptions(argc, argv, "modern_hopfield", {{"data", paths::Dataset("saliency").string()}}, "data=PATH out=PATH tf=DIR threads=N"));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
