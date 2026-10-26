// CMD: just run-nn slowfast
// SlowFast-R50 alpha8/beta1/8,two full3/4/6/3 bottleneck paths,four lateral
// temporal5/stride8 convolutions,zero-final BN gates,concatenated pooled head.
// Full32x224 graph checked without full allocation;compact16x32/base8 trained
// on real KTH clips for3 epochs default. Options:epochs=N,data=PATH,checks_only=1.
// Eager adaptation: reference geometry/counts checked analytically without full
// weight allocation; compact tensors and true gradients are executed.
#include "common/slowfast.hpp"

// Experiment workflow from examples/nn/common/slowfast.hpp.
namespace slowfast {
using namespace ::slowfast;
inline void Train(const std::filesystem::path& root, int epochs) {
  using namespace c3d;
  VideoTrainSpec spec;
  spec.directory = "slowfast";
  spec.title     = "SlowFast-R50 compact";
  spec.tsv       = true;
  spec.note =
      "Tiny3-action/6-person KTH subset,one test person,six clips;random initialization,estimated "
      "subset quality reported without threshold.";
  check::Require(epochs > 0, "positive short video epochs");
  const auto  pools      = LoadVideo(root);
  const auto& fit        = pools[0];
  const auto& validation = pools[1];
  const auto& test       = pools[2];
  Model       model;
  int         gates = 0;
  for (auto name : model.parameters.names())
    if (name.find("/expand/bn/gamma") != std::string::npos) {
      ++gates;
      check::Near(
          nn::max(nn::abs(model.parameters.at(name))).item(),
          0,
          0,
          "all32 final BN gates start zero");
    }
  check::Require(gates == 32, "both16-bottleneck pathways zero-final initialization");
  const auto directory        = paths::OutputDir("models/" + spec.directory);
  const auto checkpoint       = (directory / "best.npz").string(),
             stats_checkpoint = (directory / "best_state.npz").string();
  const auto   probe          = VideoRows(fit, {0, 1});
  const double initial        = ScoreLogits(VideoModelPredict(model, fit), fit.labels, 3).nll;
  const Tensor untrained      = VideoModelPredict(model, test);
  {
    const auto params = model.parameters.parameters();
    const auto grads  = nn::grad(
        nn::sparse_softmax_cross_entropy(model.Logits(probe.rgb, true, false), probe.labels),
        params);
    check::Require(grads.size() == params.size(), "all compact video parameter derivatives");
    int blocked = 0, nonzero = 0;
    for (size_t gi = 0; gi < grads.size(); ++gi) {
      const Tensor& g    = grads[gi];
      double        norm = 0;
      for (float x : g.to_vector<float>()) {
        check::Require(std::isfinite(x), "finite every video kernel/offset gradient coordinate");
        norm += static_cast<double>(x) * x;
      }
      auto name = model.parameters.names()[gi];
      if (norm > 0)
        ++nonzero;
      if (name.find("/reduce/W") != std::string::npos ||
          name.find("/spatial/W") != std::string::npos ||
          name.find("/expand/W") != std::string::npos) {
        ++blocked;
        check::Near(norm, 0, 0, "all96 main-branch kernels blocked by zero-final BN");
      }
    }
    check::Require(
        blocked == 96 && nonzero > 0 && nonzero < int(grads.size()),
        "96 gated kernels,some other paths active");
  }
  std::cout << spec.title << ":" << model.parameters.count() << " parameters/"
            << model.parameters.size() << " learned tensors+" << model.state.size()
            << " moving-stat tensors,16x32x32RGB;" << epochs
            << " epochs,batch2,Adam.001,all learned gradients finite;96 kernels gated "
               "initially,initial fitCE="
            << initial << std::endl;
  nn::OptimizerConfig oc;
  oc.kind = nn::OptimizerKind::Adam;
  nn::Optimizer  optimizer(model.parameters, .001, oc);
  nn::BatchOrder batch_order(static_cast<std::size_t>(fit.size()), 2, 647);
  double         best     = INFINITY;
  int            selected = 0;
  std::ofstream  history(directory / "history.tsv");
  history << "epoch\ttrain_dropout_ce\tvalidation_ce\n";
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    batch_order.begin_epoch();
    double      sum  = 0;
    std::size_t seen = 0;
    for (;;) {
      const auto rows = batch_order.next();
      if (rows.empty())
        break;
      const auto   batch = VideoRows(fit, rows);
      const Tensor loss =
          nn::sparse_softmax_cross_entropy(model.Logits(batch.rgb, true, true), batch.labels);
      const double value = loss.item<double>();
      check::Require(std::isfinite(value) && value >= 0, "finite video CE");
      optimizer.minimize(loss);
      sum += static_cast<double>(rows.size()) * value;
      seen += rows.size();
    }
    check::Require(
        seen == static_cast<std::size_t>(fit.size()), "all fitting real clips once per epoch");
    const double value =
        ScoreLogits(VideoModelPredict(model, validation), validation.labels, 3).nll;
    if (value < best) {
      best     = value;
      selected = epoch;
      model.parameters.save(checkpoint);
      model.state.save(stats_checkpoint);
    }
    history << epoch << '\t' << sum / seen << '\t' << value << '\n';
    history.flush();
    std::cout << " epoch " << epoch << ":trainCE=" << sum / seen << ",validationCE=" << value
              << std::endl;
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  model.parameters.load(checkpoint);
  model.state.load(stats_checkpoint);
  check::Near(
      ScoreLogits(VideoModelPredict(model, validation), validation.labels, 3).nll,
      best,
      1e-7,
      "restored video validation CE");
  {
    auto gradients = nn::grad(
        nn::sparse_softmax_cross_entropy(model.Logits(probe.rgb, true, false), probe.labels),
        model.parameters.parameters());
    int opened = 0;
    for (size_t i = 0; i < gradients.size(); ++i) {
      auto   name = model.parameters.names()[i];
      double norm = 0;
      for (float v : gradients[i].to_vector<float>()) {
        check::Require(std::isfinite(v), "finite every selected-model derivative");
        norm += v * v;
      }
      if (norm > 0 && (name.find("/reduce/W") != std::string::npos ||
                       name.find("/spatial/W") != std::string::npos ||
                       name.find("/expand/W") != std::string::npos))
        ++opened;
    }
    check::Require(opened > 0, "training opens formerly gated residual kernels");
    std::cout << "opened residual kernels=" << opened << "/96" << std::endl;
  }
  const double learned = ScoreLogits(VideoModelPredict(model, fit), fit.labels, 3).nll;
  if (spec.require_improvement)
    check::Require(learned < initial, "short-run fixed fitting inference CE improves");
  else
    check::Require(
        std::isfinite(learned),
        "finite restored fitting inference CE;quality is reported without threshold");

  // A fresh model restored from the checkpoint sees only RGB clips.
  Model deployed;
  deployed.parameters.load(checkpoint);
  deployed.state.load(stats_checkpoint);
  const auto   before    = VideoModelSnapshot(deployed);
  const Tensor logits    = VideoModelPredict(deployed, test),
               reference = VideoModelPredict(model, test),
               repeat    = VideoModelPredict(deployed, test);
  const auto lv = logits.to_vector<float>(), rv = reference.to_vector<float>(),
             pv = repeat.to_vector<float>();
  for (std::size_t i = 0; i < lv.size(); ++i) {
    check::Near(lv[i], rv[i], 1e-6, "independent video checkpoint parity");
    check::Near(lv[i], pv[i], 0, "dropout-free deterministic video inference");
  }
  const auto single = VideoModelPredict(deployed, VideoRows(test, {0})).template to_vector<float>();
  for (std::size_t c = 0; c < 3; ++c)
    check::Near(single[c], lv[c], 1e-6, "single/batch clip inference");

  const std::vector<std::int64_t> all{0, 1, 2, 3, 4, 5};
  VideoData                       reverse = VideoRows(test, all), still = VideoRows(test, all);
  reverse.rgb = nn::reverse(test.rgb, {1}).contiguous();
  still.rgb   = nn::tile(nn::slice_axis(test.rgb, 1, 0, 1), {1, 16, 1, 1, 1}).contiguous();
  const Tensor reversed       = VideoModelPredict(deployed, reverse),
               repeated_frame = VideoModelPredict(deployed, still);
  double     order = 0, motion = 0;
  const auto rev = reversed.to_vector<float>(), rep = repeated_frame.to_vector<float>();
  for (std::size_t i = 0; i < lv.size(); ++i) {
    order += std::abs(static_cast<double>(lv[i]) - rev[i]);
    motion += std::abs(static_cast<double>(lv[i]) - rep[i]);
  }
  check::Require(
      order > 1e-7 && motion > 1e-7,
      "video model depends on temporal order and genuine multi-frame input");
  check::Require(
      before == VideoModelSnapshot(deployed), "all learned and moving tensors immutable");
  VideoData           changed = VideoRows(test, all);
  std::vector<Tensor> frames;
  for (int t = 0; t < 16; ++t)
    frames.push_back(
        nn::slice_axis(test.rgb, 1, t > 0 && t < 15 ? 0 : t, t > 0 && t < 15 ? 1 : t + 1));
  changed.rgb = nn::concat(frames, 1);
  check::Near(
      replay::MaxAbsDiff(Sample(test.rgb, 16), Sample(changed.rgb, 16)),
      0,
      0,
      "Fast-only control preserves both raw Slow samples");
  Tensor slow_body, changed_body, no_fusion_logits, changed_logits;
  {
    nn::NoGradGuard guard;
    auto            original = Forward(deployed.parameters, deployed.state, test.rgb, {8, 3, 16});
    auto            altered = Forward(deployed.parameters, deployed.state, changed.rgb, {8, 3, 16});
    slow_body               = original.slow_final;
    changed_body            = altered.slow_final;
    changed_logits          = altered.logits;
    no_fusion_logits = Forward(deployed.parameters, deployed.state, test.rgb, {8, 3, 16, 0}).logits;
  }
  double transfer       = nn::sum(nn::abs(slow_body - changed_body)).item(),
         lateral_effect = nn::sum(nn::abs(logits - no_fusion_logits)).item();
  check::Require(
      transfer > 1e-7 && lateral_effect > 1e-7,
      "Fast-only frames affect Slow body;disabling allfour fusions changes logits");
  check::Require(
      before == VideoModelSnapshot(deployed), "fusion controls preserve learned/moving state");
  nn::save_npz(
      (directory / "pathway_controls.npz").string(),
      {{"fast_changed_rgb", changed.rgb},
       {"fast_changed_logits", changed_logits},
       {"no_fusion_logits", no_fusion_logits},
       {"slow_body", slow_body},
       {"fast_changed_slow_body", changed_body}});
  std::cout << "Fast-to-Slow transfer=" << transfer
            << " zero-lateral logit effect=" << lateral_effect << std::endl;
  {
    nn::NamedTensors arrays;
    const auto       ps = deployed.parameters.parameters(), ss = deployed.state.parameters();
    std::ofstream    mapping;
    if (spec.tsv) {
      mapping.open(directory / "model_arrays.tsv");
      mapping << "key\tname\tkind\n";
    }
    for (std::size_t i = 0; i < ps.size() + ss.size(); ++i) {
      const bool        learned_tensor = i < ps.size();
      const std::string key            = "p" + std::to_string(i);
      arrays.emplace_back(key, learned_tensor ? ps[i] : ss[i - ps.size()]);
      if (spec.tsv)
        mapping << key << '\t'
                << (learned_tensor ? deployed.parameters.names()[i]
                                   : deployed.state.names()[i - ps.size()])
                << '\t' << (learned_tensor ? "learned" : "moving") << '\n';
    }
    nn::save_npz((directory / "model_arrays.npz").string(), arrays);
    if (spec.tsv)
      check::Require(bool(mapping), "export named learned/moving inventory");
  }
  const Tensor appearance = VideoNearest(fit, test, false), energy = VideoNearest(fit, test, true);
  const Tensor prior = nn::zeros({6, 3});
  PrintVideo("trained " + spec.title, test, logits);
  PrintVideo("untrained", test, untrained);
  PrintVideo("fit static mean-frame1NN", test, appearance);
  PrintVideo("fit absolute-motion1NN", test, energy);
  PrintVideo("balanced prior tie class0", test, prior);
  PrintVideo("reversed-frame control", test, reversed);
  PrintVideo("repeated-first-frame control", test, repeated_frame);

  std::ofstream predictions(directory / "predictions.tsv");
  predictions << "video_id\tsegment\tperson\tlabel\tprediction\n";
  const auto vid = test.video.to_vector<int>(), seg = test.segment.to_vector<int>(),
             per = test.person.to_vector<int>(), lab = test.labels.to_vector<int>();
  for (std::size_t n = 0; n < 6; ++n) {
    std::size_t best_c = 0;
    for (std::size_t c = 1; c < 3; ++c)
      if (lv[n * 3 + c] > lv[n * 3 + best_c])
        best_c = c;
    predictions << vid[n] << '\t' << seg[n] << '\t' << per[n] << '\t' << lab[n] << '\t' << best_c
                << '\n';
  }
  nn::save_npz(
      (directory / "test_outputs.npz").string(),
      {{"rgb", test.rgb},
       {"labels", test.labels},
       {"person", test.person},
       {"video_id", test.video},
       {"segment", test.segment},
       {"logits", logits},
       {"untrained", untrained},
       {"appearance", appearance},
       {"motion", energy},
       {"reverse", reversed},
       {"repeat_first", repeated_frame},
       {"fit_rgb", fit.rgb},
       {"fit_labels", fit.labels},
       {"fit_person", fit.person},
       {"fit_video_id", fit.video}});
  for (const char* name : {"manifest.json", "ATTRIBUTION.txt", "test_contact.png"})
    if (std::filesystem::exists(root / name))
      std::filesystem::copy_file(
          root / name,
          directory / (std::string(name) == "manifest.json" ? "data_manifest.json" : name),
          std::filesystem::copy_options::overwrite_existing);
  check::Require(bool(history) && bool(predictions), "save real-video predictions/histories");
  std::cout << " selected=" << selected << ",fixed fitCE " << initial << " -> " << learned
            << ",reverse/repeat-first total logit changes=" << order << "/" << motion
            << ";standalone RGB-only restore/single-batch/determinism/immutable model passed\n";
  std::printf(" training time %.1f s (%.2f s/epoch)\n", seconds, seconds / epochs);
  std::cout << "Artifacts:" << directory.string() << "\n" << spec.note << "\n";
}
}  // namespace slowfast


namespace nn = datamunge::nn;
namespace {
void SamplingChecks() {
  for (int frames : {16, 32}) {
    std::vector<float> values(frames);
    std::iota(values.begin(), values.end(), 0.f);
    auto x        = nn::from_vector(values, {1, frames, 1, 1, 1}).requires_grad_();
    auto y        = slowfast::Sample(x, frames);
    auto gradient = nn::grad(nn::sum(y), {x})[0].to_vector<float>();
    auto expected = frames == 16 ? std::vector<int>{0, 15} : std::vector<int>{0, 10, 20, 31};
    check::Require(
        slowfast::Indices(frames) == expected, "authors linspace endpoints/truncated indices");
    auto out = y.to_vector<float>();
    for (size_t i = 0; i < expected.size(); ++i)
      check::Near(out[i], expected[i], 0, "sample actual frame index");
    for (int i = 0; i < frames; ++i)
      check::Near(
          gradient[i],
          std::find(expected.begin(), expected.end(), i) != expected.end() ? 1 : 0,
          0,
          "sampled-only input gradient routing");
  }
}
void FusionChecks() {
  std::vector<std::vector<double>> values(3);
  int                              sizes[] = {40, 128, 24};
  for (int a = 0; a < 3; ++a) {
    values[a].resize(sizes[a]);
    for (int i = 0; i < sizes[a]; ++i)
      values[a][i] = .2 * std::sin(.27 * i + .3 * a);
  }
  std::vector<double> coefficient(56);
  for (int i = 0; i < 56; ++i)
    coefficient[i] = .15 * std::cos(.39 * i);
  auto kernel    = nn::from_vector(values[0], {5, 1, 1, 2, 4}).requires_grad_(),
       fast      = nn::from_vector(values[1], {1, 16, 2, 2, 2}).requires_grad_(),
       slow      = nn::from_vector(values[2], {1, 2, 2, 2, 3}).requires_grad_();
  auto lateral   = factorized_video::FactorConv(fast, kernel, 8),
       joined    = slowfast::Join(slow, lateral);
  auto gradients = nn::grad(
      nn::sum(joined * nn::from_vector(coefficient, {1, 2, 2, 2, 7})), {kernel, fast, slow});
  auto host = [&](const std::vector<std::vector<double>>& a,
                  std::vector<double>*                    output = nullptr) {
    double total = 0;
    for (int t = 0; t < 2; ++t)
      for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x)
          for (int c = 0; c < 7; ++c) {
            double value = 0;
            if (c < 3)
              value = a[2][((t * 2 + y) * 2 + x) * 3 + c];
            else
              for (int kt = 0; kt < 5; ++kt)
                for (int ci = 0; ci < 2; ++ci) {
                  int it = t * 8 + kt - 2;
                  if (it >= 0 && it < 16)
                    value +=
                        a[1][((it * 2 + y) * 2 + x) * 2 + ci] * a[0][(kt * 2 + ci) * 4 + c - 3];
                }
            int i = ((t * 2 + y) * 2 + x) * 7 + c;
            total += value * coefficient[i];
            if (output)
              output->push_back(value);
          }
    return total;
  };
  std::vector<double> expected;
  host(values, &expected);
  auto   output = joined.to_vector<double>();
  double error = 0, derivative = 0;
  int    count = 0;
  for (int i = 0; i < 56; ++i)
    error = std::max(error, std::abs(expected[i] - output[i]));
  for (int a = 0; a < 3; ++a) {
    auto g = gradients[a].to_vector<double>();
    for (size_t i = 0; i < values[a].size(); ++i) {
      auto plus = values, minus = values;
      plus[a][i] += 1e-6;
      minus[a][i] -= 1e-6;
      derivative = std::max(derivative, std::abs((host(plus) - host(minus)) / 2e-6 - g[i]));
      ++count;
    }
  }
  check::Require(count == 192, "all40lateral+128Fast+24Slow coordinates");
  check::Near(error, 0, 1e-7, "all56 joined outputs host convolution");
  check::Near(derivative, 0, 1e-7, "ALL192 fusion/pathway derivatives host FD");
  std::cout << "lateral5/stride8 all56 values=" << error << " all192 FD=" << derivative
            << std::endl;
}
void ZeroGateChecks() {
  nn::ParameterStore p(761), state;
  auto               input =
      nn::from_vector(std::vector<float>{.1f, .3f, .8f, .2f, .4f, .9f, .6f, .5f}, {2, 2, 1, 1, 2})
          .requires_grad_();
  auto weight = nn::from_vector(std::vector<float>{.4f, -.2f, .3f, .7f}, {1, 1, 1, 2, 2});
  p.get("expand/W", weight.shape(), nn::init::zeros());
  p.set("expand/W", weight);
  std::vector<float> coefficients(8);
  for (int i = 0; i < 8; ++i)
    coefficients[i] = .13 * std::cos(.61 * i);
  auto coefficient = nn::from_vector(coefficients, {2, 2, 1, 1, 2});
  auto run         = [&]() {
    return nn::relu(
        input +
        slowfast::Unit(p, state, "expand", input, 2, 1, 1, 1, 1, 1, 1, false, true, true, false));
  };
  auto y      = run();
  auto leaves = p.parameters();
  leaves.push_back(input);
  auto grad = nn::grad(nn::sum(y * coefficient), leaves);
  check::Near(replay::MaxAbsDiff(y, input), 0, 0, "zero-final exact residual identity");
  check::Near(replay::MaxAbsDiff(grad.back(), coefficient), 0, 1e-7, "identity input gradients");
  auto                              names = p.names();
  std::map<std::string, nn::Tensor> g;
  for (size_t i = 0; i < names.size(); ++i)
    g[names[i]] = grad[i];
  check::Near(
      nn::max(nn::abs(g.at("expand/W"))).item(),
      0,
      0,
      "zero gate blocks every branch kernel derivative");
  auto x = input.to_vector<float>(), w = weight.to_vector<float>();
  for (int c = 0; c < 2; ++c) {
    double z[4], mean = 0, var = 0, expected_gamma = 0, expected_beta = 0;
    for (int n = 0; n < 4; ++n) {
      z[n] = double(x[2 * n]) * w[c] + double(x[2 * n + 1]) * w[2 + c];
      mean += z[n] / 4;
    }
    for (double v : z)
      var += std::pow(v - mean, 2) / 4;
    for (int n = 0; n < 4; ++n) {
      expected_gamma += coefficients[2 * n + c] * (z[n] - mean) / std::sqrt(var + 1e-5);
      expected_beta += coefficients[2 * n + c];
    }
    check::Near(
        g.at("expand/bn/gamma").to_vector<float>()[c],
        expected_gamma,
        1e-6,
        "host initial gamma derivative");
    check::Near(
        g.at("expand/bn/beta").to_vector<float>()[c],
        expected_beta,
        1e-7,
        "host initial beta derivative");
  }
  p.set("expand/bn/gamma", nn::full({2}, .2));
  auto opened = run();
  auto gw     = nn::grad(nn::sum(opened * coefficient), {p.at("expand/W")})[0];
  check::Require(
      nn::sum(nn::square(gw)).item() > 1e-8, "opened final gate unlocks branch kernel derivatives");
}
void GraphChecks() {
  for (bool compact : {false, true}) {
    slowfast::Config cfg      = compact ? slowfast::Config{8, 3, 16} : slowfast::Config{};
    int              image    = compact ? 32 : 224;
    std::int64_t     expected = 0;
    int              units    = 0;
    auto             count    = [&](int in, int out, int volume) {
      expected += std::int64_t(in) * out * volume + 2 * out;
      ++units;
    };
    int b = cfg.base, f = b / 8;
    count(3, b, 49);
    count(3, f, 245);
    int si = b, fi = f;
    count(fi, 2 * fi, 5);
    si += 2 * fi;
    const int depths[] = {3, 4, 6, 3};
    for (int stage = 0; stage < 4; ++stage) {
      for (int path = 0; path < 2; ++path) {
        auto& incoming = path ? fi : si;
        int   width = (b << stage) / (path ? 8 : 1), channels = 4 * width;
        for (int block = 0; block < depths[stage]; ++block) {
          count(incoming, width, path || stage >= 2 ? 3 : 1);
          count(width, width, 9);
          count(width, channels, 1);
          if (incoming != channels || (stage && block == 0))
            count(incoming, channels, 1);
          incoming = channels;
        }
      }
      if (stage < 3) {
        count(fi, 2 * fi, 5);
        si += 2 * fi;
      }
    }
    expected += std::int64_t(36 * b) * cfg.classes + cfg.classes;
    check::Require(units == 110, "independent two-path/lateral/BN/head inventory110 units");
    if (compact) {
      nn::ParameterStore p(757), state;
      nn::NoGradGuard    guard;
      auto net = slowfast::Forward(p, state, nn::zeros({1, cfg.frames, image, image, 3}), cfg);
      check::Require(
          p.count() == expected && p.size() == 332 && state.size() == 220,
          "332learned/220moving inventory");
      check::Require(
          net.slow_input.shape() == nn::Shape({1, cfg.frames / 8, image, image, 3}),
          "actual slow sampled endpoint");
      check::Require(
          net.slow_stages.size() == 5 && net.fast_stages.size() == 5 && net.lateral.size() == 4 &&
              net.slow_blocks.size() == 16 && net.fast_blocks.size() == 16,
          "all32 residual blocks/four fusions retained");
      for (int i = 0; i < 5; ++i) {
        int channels = i ? cfg.base * (4 << (i - 1)) : cfg.base,
            space    = image / (i ? 4 << (i - 1) : 4);
        check::Require(
            net.slow_stages[i].shape() == nn::Shape({1, cfg.frames / 8, space, space, channels}) &&
                net.fast_stages[i].shape() ==
                    nn::Shape({1, cfg.frames, space, space, channels / 8}),
            "every actual paired stage endpoint");
        if (i < 4)
          check::Require(
              net.lateral[i].shape() == nn::Shape({1, cfg.frames / 8, space, space, channels / 4}),
              "every actual fusion endpoint");
      }
      int index = 0;
      for (int stage = 0; stage < 4; ++stage)
        for (int block = 0; block < depths[stage]; ++block, ++index) {
          int channels = cfg.base * (4 << stage), space = image / (4 << stage);
          check::Require(
              net.slow_blocks[index].shape() ==
                      nn::Shape({1, cfg.frames / 8, space, space, channels}) &&
                  net.fast_blocks[index].shape() ==
                      nn::Shape({1, cfg.frames, space, space, channels / 8}),
              "all32 actual residual block endpoints");
        }
      check::Require(
          net.pooled.shape() == nn::Shape({1, 36 * cfg.base}) &&
              net.logits.shape() == nn::Shape({1, cfg.classes}),
          "pooled/head endpoints");
    }
    std::cout << (compact ? "compact" : "full reference") << " SlowFast:" << expected
              << " parameters/332learned+220moving;two16-block paths/four fusions PASS\n";
  }
}
} // namespace
int main(int argc, char** argv) {
  try {
    nn::set_num_threads(2);
    int  epochs = 3;
    bool checks = false;
    auto root   = paths::Dataset("action_video");
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a.rfind("epochs=", 0) == 0)
        epochs = std::stoi(a.substr(7));
      else if (a.rfind("data=", 0) == 0)
        root = a.substr(5);
      else if (a == "checks_only=1")
        checks = true;
      else if (a != "checks_only=0")
        throw std::invalid_argument("epochs=N data=PATH checks_only=0|1");
    }
    SamplingChecks();
    FusionChecks();
    ZeroGateChecks();
    GraphChecks();
    if (!checks) {
      slowfast::Train(root, epochs);
      auto tf = replay::TfModels("slowfast");
      if (std::filesystem::exists(tf / "model_arrays.npz")) {
        slowfast::Model restored;
        check::Near(c3d::ReplayTfVideo(tf, restored), 0, 2e-5, "TF SlowFast trained-weight replay");
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
