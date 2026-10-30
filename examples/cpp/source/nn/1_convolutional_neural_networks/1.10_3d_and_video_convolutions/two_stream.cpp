// CMD: just run-nn two_stream
// RGB single-frame/ten-field dense optical-flow CNN-M-2048 streams,
// independent CE losses and average SOFTMAX probability late fusion.
// Full224px reference graphs only;actual96px/divisor16/FC64/32 training.
// Default3 epochs perstream. Prepare:just two-stream-assets with pinned Python deps.
// Options:epochs=N,data=PATH,checks_only=1.
// Eager adaptation: full reference geometry/parameter inventory is computed
// without allocating its large FC weights; compact endpoints and every learned
// derivative are exercised by the real KTH run. Four independent dropout streams.
#include "common/two_stream.hpp"

// Experiment workflow from examples/nn/common/two_stream.hpp.
namespace two_stream {
using namespace ::two_stream;
inline void Train(const std::filesystem::path& assets, int epochs) {
  check::Require(epochs > 0, "positive short epochs");
  auto   pools = Load(assets);
  auto   fit = pools[0], val = pools[1], test = pools[2];
  Model  model;
  auto   root      = paths::OutputDir("models/two_stream");
  auto   prefix    = root / "best.npz";
  double initial   = c3d::ScoreLogits(Predict(model, fit), fit.at("label"), 3).nll;
  auto   untrained = Predict(model, test);
  auto   batch     = replay::Rows(fit, {0, 1});
  auto   loss      = model.Loss(batch);
  auto   gradients = nn::grad(loss, model.parameters.parameters());
  for (const auto& g : gradients) {
    double norm = 0;
    for (float v : g.to_vector<float>()) {
      check::Require(std::isfinite(v), "finite every RGB/flow gradient coordinate");
      norm += v * v;
    }
    check::Require(norm > 0, "nonzero each learned tensor both streams");
  }
  check::Require(gradients.size() == 32, "all32 true stream gradients");
  std::cout << "two-stream compact:" << model.parameters.count()
            << " parameters/32 tensors;all32 finite/nonzero;" << epochs
            << " epochs,batch2,Adam.001,independent CE sum;initial fused fitNLL=" << initial
            << std::endl;
  nn::BatchOrder order(24, 2, 857);
  double         best     = INFINITY;
  int            selected = 0;
  std::ofstream  history(root / "history.tsv");
  history << "epoch\ttrain_stream_ce_sum\tvalidation_fused_nll\n";
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    double sum  = 0;
    int    seen = 0;
    for (;;) {
      auto rows = order.next();
      if (rows.empty())
        break;
      sum += rows.size() * model.Update(replay::Rows(fit, rows));
      seen += rows.size();
    }
    check::Require(seen == 24, "all24 fitting pairs once/epoch");
    double nll = c3d::ScoreLogits(Predict(model, val), val.at("label"), 3).nll;
    if (nll < best) {
      best     = nll;
      selected = epoch;
      model.parameters.save(prefix.string());
    }
    history << epoch << '\t' << sum / seen << '\t' << nll << '\n';
    history.flush();
    std::cout << "epoch " << epoch << ":trainCEsum=" << sum / seen << ",validation fusedNLL=" << nll
              << std::endl;
  }
  model.parameters.load(prefix.string());
  check::Near(
      c3d::ScoreLogits(Predict(model, val), val.at("label"), 3).nll,
      best,
      1e-7,
      "restored validation fusedNLL");
  double learned = c3d::ScoreLogits(Predict(model, fit), fit.at("label"), 3).nll;
  Model  deployed(false);
  deployed.parameters.load(prefix.string());
  auto before = deployed.Arrays();
  auto score = Predict(deployed, test), reference = Predict(model, test);
  check::Near(replay::MaxAbsDiff(score, reference), 0, 1e-6, "fresh restore fusion parity");
  check::Near(
      replay::MaxAbsDiff(score, Predict(deployed, test)),
      0,
      0,
      "dropout-free deterministic fusion");
  auto one = deployed.Predict(replay::Rows(test, {0}));
  check::Near(
      replay::MaxAbsDiff(one, nn::slice(score, {0, 0}, {1, 3})), 0, 1e-6, "single/batch parity");
  auto path      = deployed.Run(test);
  auto rgb_score = path.rgb.logits, flow_score = path.flow.logits;
  {
    nn::NoGradGuard guard;
    auto rgb = CnnM(deployed.parameters, "rgb", test.at("rgb") * 2. - 1., deployed.config, false)
                   .logits,
         flow =
             CnnM(
                 deployed.parameters, "flow", FlowStack(test.at("flow"), 96), deployed.config, true)
                 .logits;
    check::Near(replay::MaxAbsDiff(rgb, rgb_score), 0, 0, "RGB stream independent of flow");
    check::Near(replay::MaxAbsDiff(flow, flow_score), 0, 0, "flow stream independent of RGB");
  }
  auto zero = test, neutral = test, reverse = test;
  zero["flow"]   = nn::zeros(test.at("flow").shape());
  neutral["rgb"] = nn::full(test.at("rgb").shape(), .5);
  std::vector<Tensor> frames;
  for (int t = 9; t >= 0; --t)
    frames.push_back(nn::slice(test.at("flow"), {0, t, 0, 0, 0}, {6, 1, 96, 96, 2}));
  reverse["flow"] = nn::concat(frames, 1);
  auto z = Predict(deployed, zero), gray = Predict(deployed, neutral),
       rev  = Predict(deployed, reverse);
  double dz = nn::sum(nn::abs(score - z)).item(), drgb = nn::sum(nn::abs(score - gray)).item(),
         dorder = nn::sum(nn::abs(score - rev)).item();
  check::Require(
      dz > 1e-7 && drgb > 1e-7 && dorder > 1e-7, "fusion depends on RGB,real flow and order");
  check::Require(replay::SameBits(before, deployed.Arrays()), "all32 parameters immutable");
  nn::save_npz((root / "feature_arrays.npz").string(), replay::ToNamed(deployed.Features(test)));
  nn::NamedTensors exported;
  std::ofstream    mapping(root / "model_arrays.tsv");
  mapping << "key\tname\n";
  for (size_t i = 0; i < deployed.parameters.size(); ++i) {
    auto key = "p" + std::to_string(i), name = deployed.parameters.names()[i];
    exported.emplace_back(key, deployed.parameters.at(name));
    mapping << key << '\t' << name << '\n';
  }
  nn::save_npz((root / "model_arrays.npz").string(), exported);
  auto appearance = Nearest(fit, test, false), motion = Nearest(fit, test, true);
  auto print = [&](const std::string& name, const Tensor& scores) {
    auto s = c3d::ScoreLogits(scores, test.at("label"), 3);
    std::cout << name << ":accuracy=" << 100 * s.accuracy << "%,macroF1=" << s.macro_f1
              << ",NLL=" << s.nll << ",confusion=";
    for (int x : s.confusion)
      std::cout << x << ',';
    std::cout << '\n';
  };
  print("fused two-stream", score);
  print("RGB stream", rgb_score);
  print("flow stream", flow_score);
  print("untrained fused", untrained);
  print("fit RGB-grid1NN", appearance);
  print("fit abs-flow-grid1NN", motion);
  print("zero-flow control", z);
  print("neutral-RGB control", gray);
  print("reversed-stack-order control", rev);
  nn::save_npz(
      (root / "test_outputs.npz").string(),
      {{"rgb", test.at("rgb")},
       {"flow", test.at("flow")},
       {"labels", test.at("label")},
       {"person", test.at("person")},
       {"video_id", test.at("video_id")},
       {"segment", test.at("segment")},
       {"fit_rgb", fit.at("rgb")},
       {"fit_flow", fit.at("flow")},
       {"fit_labels", fit.at("label")},
       {"fit_person", fit.at("person")},
       {"fit_video_id", fit.at("video_id")},
       {"logits", score},
       {"rgb_logits", rgb_score},
       {"flow_logits", flow_score},
       {"zero_flow", z},
       {"neutral_rgb", gray},
       {"reverse_order", rev},
       {"untrained", untrained},
       {"appearance", appearance},
       {"motion", motion}});
  std::ofstream predictions(root / "predictions.tsv");
  predictions << "video_id\tsegment\tperson\tlabel\tprediction\n";
  auto labels = test.at("label").to_vector<int>(), videos = test.at("video_id").to_vector<int>(),
       segments = test.at("segment").to_vector<int>(), people = test.at("person").to_vector<int>();
  auto values = score.to_vector<float>();
  for (int n = 0; n < 6; ++n) {
    int best = 0;
    for (int c = 1; c < 3; ++c)
      if (values[n * 3 + c] > values[n * 3 + best])
        best = c;
    predictions << videos[n] << '\t' << segments[n] << '\t' << people[n] << '\t' << labels[n]
                << '\t' << best << '\n';
  }
  std::ofstream meta(root / "metrics.json");
  meta << "{\"epochs\":" << epochs << ",\"selected_epoch\":" << selected
       << ",\"initial_fit_nll\":" << initial << ",\"learned_fit_nll\":" << learned
       << ",\"validation_nll\":" << best << ",\"zero_flow_effect\":" << dz
       << ",\"neutral_rgb_effect\":" << drgb << ",\"reversed_order_effect\":" << dorder << "}\n";
  for (auto file : {"manifest.json", "ATTRIBUTION.txt", "test_contact.png"})
    std::filesystem::copy_file(
        assets / file,
        root / (std::string(file) == "manifest.json" ? "data_manifest.json" : file),
        std::filesystem::copy_options::overwrite_existing);
  check::Require(bool(history) && bool(predictions), "saved histories/predictions");
  std::cout << "two-stream independent loss/fusion/restore/modality controls PASS\n";
}
}  // namespace two_stream


namespace nn = datamunge::nn;
namespace {
void FlowStackChecks() {
  std::vector<double> values(16), coeff(16);
  for (int i = 0; i < 16; ++i) {
    values[i] = i == 0 ? 40. : i == 9 ? -35. : 3 * std::sin(.7 * i);
    coeff[i]  = .2 * std::cos(.33 * i);
  }
  auto x = nn::from_vector(values, {1, 2, 2, 2, 2}).requires_grad_(),
       y = two_stream::FlowStack(x, 2, 2);
  auto g = nn::grad(nn::sum(y * nn::from_vector(coeff, {1, 2, 2, 4})), {x})[0].to_vector<double>();
  auto host = [&](const std::vector<double>& v, std::vector<double>* output = nullptr) {
    std::vector<double> z(16);
    double              loss = 0;
    for (int t = 0; t < 2; ++t)
      for (int c = 0; c < 2; ++c) {
        double mean = 0;
        for (int p = 0; p < 4; ++p)
          mean += v[t * 8 + p * 2 + c] / 4;
        for (int p = 0; p < 4; ++p) {
          int j = p * 4 + 2 * t + c;
          z[j]  = std::clamp((v[t * 8 + p * 2 + c] - mean) / 10., -1., 1.);
          loss += z[j] * coeff[j];
        }
      }
    if (output)
      *output = z;
    return loss;
  };
  std::vector<double> expected;
  host(values, &expected);
  auto   output  = y.to_vector<double>();
  double forward = 0, error = 0;
  for (int i = 0; i < 16; ++i) {
    forward   = std::max(forward, std::abs(expected[i] - output[i]));
    auto plus = values, minus = values;
    plus[i] += 1e-5;
    minus[i] -= 1e-5;
    error = std::max(error, std::abs((host(plus) - host(minus)) / 2e-5 - g[i]));
  }
  check::Near(forward, 0, 1e-7, "signed dx0/dy0/dx1/dy1 order/mean/clipping");
  check::Near(error, 0, 1e-7, "ALL16 centered/clipped/raw flow FD coordinates");
  check::Near(
      nn::max(nn::abs(two_stream::FlowStack(nn::full({1, 2, 2, 2, 2}, 7), 2, 2))).item(),
      0,
      0,
      "constant camera displacement zero");
  std::cout << "flow stack ALL16 host FD=" << error << ";constant displacement passed\n";
}
void FusionChecks() {
  std::vector<double> av{4, -2, 1, -1, 2, .3}, bv{-1, 2, 0, 3, -2, 1};
  auto                a = nn::from_vector(av, {2, 3}).requires_grad_(),
       b                = nn::from_vector(bv, {2, 3}).requires_grad_();
  auto fused            = two_stream::Fusion(a, b);
  auto target           = nn::from_vector(std::vector<double>{1, 0, 0, 0, 0, 1}, {2, 3});
  auto g                = nn::grad(-nn::sum(nn::log(fused) * target) * .5, {a, b});
  auto host             = [&](const std::vector<double>& x, const std::vector<double>& y) {
    double loss = 0;
    for (int row = 0; row < 2; ++row) {
      double sumx = 0, sumy = 0;
      for (int c = 0; c < 3; ++c) {
        sumx += std::exp(x[row * 3 + c]);
        sumy += std::exp(y[row * 3 + c]);
      }
      int c = row ? 2 : 0;
      loss -=
          .5 * std::log(.5 * (std::exp(x[row * 3 + c]) / sumx + std::exp(y[row * 3 + c]) / sumy));
    }
    return loss;
  };
  double error = 0;
  for (int stream = 0; stream < 2; ++stream) {
    auto derivative = g[stream].to_vector<double>();
    for (int i = 0; i < 6; ++i) {
      auto plus = stream ? bv : av, minus = plus;
      plus[i] += 1e-5;
      minus[i] -= 1e-5;
      double fd = stream ? (host(av, plus) - host(av, minus)) / 2e-5
                         : (host(plus, bv) - host(minus, bv)) / 2e-5;
      error     = std::max(error, std::abs(fd - derivative[i]));
    }
  }
  check::Near(error, 0, 1e-7, "ALL12 late probability fusion NLL derivatives");
  auto alternate = nn::softmax((a + b) * .5, -1);
  check::Require(
      nn::sum(nn::abs(alternate - fused)).item() > .01,
      "probability average differs from logit average");
  auto actual = fused.to_vector<double>();
  for (int row = 0; row < 2; ++row) {
    double ax = 0, bx = 0;
    for (int c = 0; c < 3; ++c) {
      ax += std::exp(av[row * 3 + c]);
      bx += std::exp(bv[row * 3 + c]);
    }
    for (int c = 0; c < 3; ++c)
      check::Near(
          actual[row * 3 + c],
          .5 * (std::exp(av[row * 3 + c]) / ax + std::exp(bv[row * 3 + c]) / bx),
          1e-7,
          "independent fusion values");
  }
  std::cout << "fusion ALL12 host FD=" << error << " PASS\n";
}
void PoolChecks() {
  std::vector<float> values(16);
  for (int i = 0; i < 16; ++i)
    values[i] = -10 - i;
  auto x      = nn::from_vector(values, {1, 4, 4, 1}).requires_grad_();
  auto padded = nn::pad(x, {{0, 0}, {0, 1}, {0, 1}, {0, 0}}, nn::PadMode::Constant, -INFINITY);
  auto y      = nn::max_pool2d(padded, {3, 3}, {2, 2}, nn::Padding::Valid);
  auto g      = nn::grad(nn::sum(y), {x})[0].to_vector<float>();
  auto z      = y.to_vector<float>();
  const float expected[] = {-10, -12, -18, -20};
  for (int i = 0; i < 4; ++i)
    check::Near(z[i], expected[i], 0, "pool2 negative partial border");
  for (int i = 0; i < 16; ++i)
    check::Near(g[i], i == 0 || i == 2 || i == 8 || i == 10 ? 1 : 0, 0, "all16 max derivatives");
}
void GraphChecks() {
  for (bool compact : {false, true}) {
    two_stream::Config cfg = compact ? two_stream::Config{96, 16, 64, 32, 3} : two_stream::Config{};
    const int sides[] = {109, 26, 13, 13, 13}, small[] = {45, 10, 5, 5, 5}, pool[] = {54, 13, 6},
              sp[] = {22, 5, 2}, widths[] = {96, 256, 512, 512, 512}, kernel[] = {7, 5, 3, 3, 3};
    std::int64_t total = 0;
    for (bool temporal : {false, true}) {
      int          in = temporal ? 20 : 3, side = cfg.image;
      std::int64_t count = 0;
      for (int i = 0; i < 5; ++i) {
        int out = widths[i] / cfg.divisor;
        count += std::int64_t(kernel[i]) * kernel[i] * in * out + out;
        in   = out;
        side = (side + (i ? 2 : 0) - kernel[i]) / (i < 2 ? 2 : 1) + 1;
        check::Require(
            side == (compact ? small[i] : sides[i]), "full/compact independent convolution side");
        if (i == 0 || i == 1 || i == 4) {
          side  = (side + (i == 1 ? 1 : 0) - 3) / 2 + 1;
          int j = i == 0 ? 0 : i == 1 ? 1 : 2;
          check::Require(
              side == (compact ? sp[j] : pool[j]), "full/compact independent pooling side");
        }
      }
      int flat = side * side * in;
      check::Require(flat == (compact ? 128 : 18432), "CNN-M flattened side");
      count += std::int64_t(flat) * cfg.fc6 + cfg.fc6 + std::int64_t(cfg.fc6) * cfg.fc7 + cfg.fc7 +
               std::int64_t(cfg.fc7) * cfg.classes + cfg.classes;
      total += count;
      std::cout << (temporal ? "flow" : "RGB") << (compact ? " compact" : " full CNN-M-2048") << ":"
                << count << " learned parameters\n";
    }
    if (compact) {
      nn::ParameterStore p(863);
      nn::NoGradGuard    guard;
      auto               path =
          two_stream::Forward(p, nn::zeros({1, 96, 96, 3}), nn::zeros({1, 10, 96, 96, 2}), cfg);
      check::Require(p.count() == total && p.size() == 32, "independent32 tensor inventory");
      check::Require(
          path.stack.shape() == nn::Shape({1, 96, 96, 20}) &&
              path.probabilities.shape() == nn::Shape({1, 3}),
          "flow stack and fusion shapes");
      for (const auto* stream : {&path.rgb, &path.flow}) {
        for (int i = 0; i < 5; ++i)
          check::Require(
              stream->conv[i].shape() == nn::Shape({1, small[i], small[i], widths[i] / 16}),
              "all five compact convolution endpoints");
        for (int i = 0; i < 3; ++i)
          check::Require(stream->pool[i].size(1) == sp[i], "three actual pool endpoints");
        check::Require(
            stream->flat.shape() == nn::Shape({1, 128}) &&
                stream->fc6.shape() == nn::Shape({1, 64}) &&
                stream->fc7.shape() == nn::Shape({1, 32}),
            "compact dense endpoints");
      }
    }
    std::cout << (compact ? "compact" : "full") << " two-stream inventory=" << total
              << "/32 tensors PASS\n";
  }
}
} // namespace
int main(int argc, char** argv) {
  try {
    nn::set_num_threads(2);
    int  epochs = 3;
    bool checks = false;
    auto root   = paths::Dataset("two_stream_video");
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
    FlowStackChecks();
    FusionChecks();
    PoolChecks();
    GraphChecks();
    if (!checks) {
      two_stream::Train(root, epochs);
      auto tf = replay::TfModels("two_stream");
      if (std::filesystem::exists(tf / "model_arrays.npz"))
        two_stream::ReplayTf(tf);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
