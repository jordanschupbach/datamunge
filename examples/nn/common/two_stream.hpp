// Two-stream CNN-M-2048: RGB frame, ten signed dense dx/dy fields, and
// independent stream losses with average-softmax late fusion.
#pragma once
#include "common/c3d.hpp"
#include "common/tf_replay.hpp"
namespace two_stream {
namespace nn = datamunge::nn;
using nn::Tensor;
using A = replay::Map;
struct Config {
  int image = 224, divisor = 1, fc6 = 4096, fc7 = 2048, classes = 101;
};
inline int FinalSide(int image) {
  int side = (image - 7) / 2 + 1;
  side     = (side - 3) / 2 + 1;
  side     = (side + 2 - 5) / 2 + 1;
  side     = (side + 1 - 3) / 2 + 1;
  side     = (side - 3) / 2 + 1;
  return side;
}
inline Tensor FlowStack(const Tensor& flow, int image, int fields = 10) {
  auto centered = flow - nn::mean(flow, {2, 3}, true);
  auto scaled   = nn::clip(centered, -10, 10) / 10.;
  return nn::reshape(
      nn::transpose(scaled, {0, 2, 3, 1, 4}), {flow.size(0), image, image, 2 * fields});
}
inline Tensor Fusion(const Tensor& a, const Tensor& b) {
  return (nn::softmax(a, -1) + nn::softmax(b, -1)) * .5;
}
struct Stream {
  Tensor              logits, flat, fc6, fc7;
  std::vector<Tensor> conv, pool;
};
inline Stream CnnM(
    nn::ParameterStore& p,
    const std::string&  name,
    Tensor              h,
    const Config&       cfg,
    bool                temporal,
    bool                training = false,
    nn::Generator*      drop6    = nullptr,
    nn::Generator*      drop7    = nullptr) {
  check::Require(
      cfg.divisor > 0 && cfg.fc6 > 0 && cfg.fc7 > 0 && cfg.classes >= 2 && FinalSide(cfg.image) > 0,
      "CNN-M geometry/width");
  Stream    out;
  const int widths[] = {96, 256, 512, 512, 512}, kernel[] = {7, 5, 3, 3, 3};
  for (int l = 0; l < 5; ++l) {
    int width = widths[l] / cfg.divisor;
    check::Require(width > 0, "positive CNN-M widths");
    if (l)
      h = nn::pad(h, {{0, 0}, {1, 1}, {1, 1}, {0, 0}});
    auto prefix = name + "/conv" + std::to_string(l + 1);
    p.get(prefix + "/b", {width}, nn::init::constant(.01));
    nn::ConvLayerOptions opt;
    opt.padding    = nn::Padding::Valid;
    opt.stride     = l < 2 ? 2 : 1;
    opt.init       = nn::init::he_uniform();
    opt.activation = nn::Activation::Relu;
    h              = nn::conv2d(p, prefix, h, width, kernel[l], opt);
    out.conv.push_back(h);
    if (l == 0 || (l == 1 && !temporal))
      h = nn::local_response_normalization(h, 2, 1, .0001 / 5, .75);
    if (l == 0 || l == 1 || l == 4) {
      if (l == 1)
        h = nn::pad(h, {{0, 0}, {0, 1}, {0, 1}, {0, 0}}, nn::PadMode::Constant, -INFINITY);
      h = nn::max_pool2d(h, {3, 3}, {2, 2}, nn::Padding::Valid);
      out.pool.push_back(h);
    }
  }
  out.flat = nn::reshape(
      h, {h.size(0), FinalSide(cfg.image) * FinalSide(cfg.image) * (512 / cfg.divisor)});
  nn::DenseOptions dense;
  dense.init       = nn::init::he_uniform();
  dense.activation = nn::Activation::Relu;
  dense.bias_value = .01;
  out.fc6          = nn::dense(p, name + "/fc6", out.flat, cfg.fc6, dense);
  h                = out.fc6;
  if (training)
    h = nn::dropout(h, .5, true, *drop6);
  out.fc7 = nn::dense(p, name + "/fc7", h, cfg.fc7, dense);
  h       = out.fc7;
  if (training)
    h = nn::dropout(h, .5, true, *drop7);
  out.logits = nn::dense(p, name + "/fc8", h, cfg.classes);
  return out;
}
struct Path {
  Stream rgb, flow;
  Tensor stack, probabilities, log_scores;
};
inline Path Forward(
    nn::ParameterStore&         p,
    const Tensor&               rgb,
    const Tensor&               displacement,
    const Config&               cfg      = {},
    bool                        training = false,
    std::vector<nn::Generator>* drops    = nullptr) {
  Path out;
  out.stack = FlowStack(displacement, cfg.image);
  out.rgb   = CnnM(
      p,
      "rgb",
      rgb * 2. - 1.,
      cfg,
      false,
      training,
      drops ? &(*drops)[0] : nullptr,
      drops ? &(*drops)[1] : nullptr);
  out.flow = CnnM(
      p,
      "flow",
      out.stack,
      cfg,
      true,
      training,
      drops ? &(*drops)[2] : nullptr,
      drops ? &(*drops)[3] : nullptr);
  out.probabilities = Fusion(out.rgb.logits, out.flow.logits);
  out.log_scores    = nn::log(nn::maximum(out.probabilities, nn::scalar(1e-8)));
  return out;
}
inline std::vector<A> Load(const std::filesystem::path& root) {
  auto                          a = replay::ToMap(nn::load_npz((root / "two_stream.npz").string()));
  std::vector<A>                pools;
  std::set<int>                 prior_people, prior_videos;
  std::set<std::pair<int, int>> clips;
  for (const std::string split : {"fit", "validation", "test"}) {
    A d;
    for (auto key : {"rgb", "flow", "label", "person", "video_id", "segment"})
      d[key] = a.at(split + "_" + key);
    int n = d.at("rgb").size(0);
    check::Require(
        n == (split == "fit" ? 24 : 6) && d.at("rgb").shape() == nn::Shape({n, 96, 96, 3}) &&
            d.at("flow").shape() == nn::Shape({n, 10, 96, 96, 2}),
        "actual aligned RGB/ten displacement fields");
    for (auto key : {"rgb", "flow"}) {
      auto x = d.at(key);
      check::Require(x.dtype() == nn::DType::Float32, "float observed modality");
      for (float v : x.to_vector<float>())
        check::Require(
            std::isfinite(v) && ((std::string(key) != "rgb") || (v >= 0 && v <= 1)),
            "finite modality/RGB[0,1]");
    }
    for (auto key : {"label", "person", "video_id", "segment"})
      check::Require(
          d.at(key).dtype() == nn::DType::Int32 && d.at(key).shape() == nn::Shape({n}),
          "annotation alignment");
    auto labels = d.at("label").to_vector<int>(), people = d.at("person").to_vector<int>(),
         videos = d.at("video_id").to_vector<int>(), segments = d.at("segment").to_vector<int>();
    std::set<int> persons, vids;
    for (int i = 0; i < n; ++i) {
      check::Require(
          labels[i] >= 0 && labels[i] < 3 && !prior_people.count(people[i]) &&
              !prior_videos.count(videos[i]) && clips.emplace(videos[i], segments[i]).second,
          "unique clips/person/source-video heldout pools");
      persons.insert(people[i]);
      vids.insert(videos[i]);
    }
    prior_people.insert(persons.begin(), persons.end());
    prior_videos.insert(vids.begin(), vids.end());
    std::cout << "KTH " << split << ":" << n << " RGB+flow pairs," << persons.size() << " people,"
              << vids.size() << " source AVIs\n";
    pools.push_back(d);
  }
  return pools;
}
class Model {
 public:
  nn::ParameterStore         parameters{853};
  Config                     config{96, 16, 64, 32, 3};
  std::vector<nn::Generator> drops{
      nn::Generator(823),
      nn::Generator(827),
      nn::Generator(829),
      nn::Generator(839)};
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit Model(bool training = true) {
    {
      nn::NoGradGuard guard;
      Forward(parameters, nn::zeros({1, 96, 96, 3}), nn::zeros({1, 10, 96, 96, 2}), config);
    }
    if (training) {
      nn::OptimizerConfig opt;
      opt.kind  = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(parameters, .001, opt);
    }
  }
  static Tensor CE(const Tensor& logits, const Tensor& labels) {
    return -nn::mean(nn::sum(nn::log_softmax(logits, -1) * nn::one_hot(labels, 3), {1}));
  }
  Path Run(const A& b) {
    nn::NoGradGuard guard;
    return Forward(parameters, b.at("rgb"), b.at("flow"), config);
  }
  Tensor Predict(const A& b) { return Run(b).log_scores; }
  Tensor Loss(const A& b) {
    auto path = Forward(parameters, b.at("rgb"), b.at("flow"), config, true, &drops);
    return CE(path.rgb.logits, b.at("label")) + CE(path.flow.logits, b.at("label"));
  }
  double Update(const A& b) {
    auto   loss  = Loss(b);
    double value = loss.item();
    check::Require(std::isfinite(value), "finite separate stream loss sum");
    optimizer->minimize(loss);
    return value;
  }
  A Arrays() const { return replay::Arrays(parameters); }
  A Features(const A& b) {
    auto path = Run(b);
    A    result;
    for (bool flow : {false, true}) {
      auto& stream = flow ? path.flow : path.rgb;
      auto  name   = flow ? "flow" : "rgb";
      for (int i = 0; i < 5; ++i)
        result[std::string(name) + "_conv" + std::to_string(i + 1)] = stream.conv[i];
      for (int i = 0; i < 3; ++i)
        result[std::string(name) + "_pool" + std::to_string(i + 1)] = stream.pool[i];
      result[std::string(name) + "_fc6"] = stream.fc6;
      result[std::string(name) + "_fc7"] = stream.fc7;
    }
    return result;
  }
};
inline Tensor Predict(Model& m, const A& d) {
  std::vector<Tensor> values;
  for (int start = 0; start < d.at("rgb").size(0); start += 2) {
    std::vector<std::int64_t> rows;
    for (int i = start; i < std::min<int>(start + 2, d.at("rgb").size(0)); ++i)
      rows.push_back(i);
    values.push_back(m.Predict(replay::Rows(d, rows)));
  }
  return nn::concat(values, 0);
}
inline Tensor GridFeatures(const A& d, bool motion) {
  int                channels = motion ? 2 : 3, n = d.at("rgb").size(0);
  std::vector<float> out(n * 16 * channels);
  auto               input = d.at(motion ? "flow" : "rgb").to_vector<float>();
  for (int b = 0; b < n; ++b)
    for (int gy = 0; gy < 4; ++gy)
      for (int gx = 0; gx < 4; ++gx)
        for (int c = 0; c < channels; ++c) {
          double sum = 0;
          for (int t = 0; t < (motion ? 10 : 1); ++t)
            for (int y = gy * 24; y < (gy + 1) * 24; ++y)
              for (int x = gx * 24; x < (gx + 1) * 24; ++x) {
                auto i = motion ? ((((b * 10 + t) * 96 + y) * 96 + x) * 2 + c)
                                : ((b * 96 + y) * 96 + x) * 3 + c;
                sum += motion ? std::abs(input[i]) : input[i];
              }
          out[(b * 16 + gy * 4 + gx) * channels + c] = sum / (576 * (motion ? 10 : 1));
        }
  return nn::from_vector(out, {n, 16 * channels});
}
inline Tensor Nearest(const A& fit, const A& test, bool motion) {
  auto                a = GridFeatures(fit, motion), b = GridFeatures(test, motion);
  int                 dims = a.size(1);
  auto                av = a.to_vector<float>(), bv = b.to_vector<float>();
  auto                labels = fit.at("label").to_vector<int>();
  std::vector<double> scale(dims);
  for (int c = 0; c < dims; ++c) {
    double mean = 0, var = 0;
    for (int i = 0; i < a.size(0); ++i)
      mean += av[i * dims + c] / double(a.size(0));
    for (int i = 0; i < a.size(0); ++i)
      var += std::pow(av[i * dims + c] - mean, 2) / a.size(0);
    scale[c] = std::max(1e-6, std::sqrt(var));
  }
  std::vector<float> out(b.size(0) * 3);
  for (int n = 0; n < b.size(0); ++n) {
    double best  = INFINITY;
    int    label = 0;
    for (int i = 0; i < a.size(0); ++i) {
      double distance = 0;
      for (int c = 0; c < dims; ++c)
        distance += std::pow((av[i * dims + c] - bv[n * dims + c]) / scale[c], 2);
      if (distance < best) {
        best  = distance;
        label = labels[i];
      }
    }
    out[n * 3 + label] = 1;
  }
  return nn::from_vector(out, {b.size(0), 3});
}

inline double ReplayTf(const std::filesystem::path& root) {
  Model         model(false);
  auto          arrays = replay::ToMap(nn::load_npz((root / "model_arrays.npz").string()));
  std::ifstream mapping(root / "model_arrays.tsv");
  check::Require(bool(mapping) && arrays.size() == 32, "TF32 tensor mapping");
  std::string line, key, name;
  std::getline(mapping, line);
  int count = 0;
  while (mapping >> key >> name) {
    check::Require(
        model.parameters.contains(name) &&
            model.parameters.at(name).shape() == arrays.at(key).shape(),
        "TF/native CNN-M parameter shape " + name);
    model.parameters.set(name, arrays.at(key));
    ++count;
  }
  check::Require(count == 32, "all32 TF tensors imported");
  auto   outputs = replay::ToMap(nn::load_npz((root / "test_outputs.npz").string()));
  A      b{{"rgb", outputs.at("rgb")}, {"flow", outputs.at("flow")}};
  double worst   = 0;
  auto   compare = [&](const Tensor& value, const std::string& key) {
    double error = replay::MaxAbsDiff(value, outputs.at(key));
    check::Near(error, 0, 2e-5, "TF two-stream trained replay " + key);
    worst = std::max(worst, error);
  };
  auto path = model.Run(b);
  compare(Predict(model, b), "logits");
  compare(path.rgb.logits, "rgb_logits");
  compare(path.flow.logits, "flow_logits");
  auto zero    = b;
  zero["flow"] = nn::zeros(b.at("flow").shape());
  compare(Predict(model, zero), "zero_flow");
  auto neutral   = b;
  neutral["rgb"] = nn::full(b.at("rgb").shape(), .5);
  compare(Predict(model, neutral), "neutral_rgb");
  auto reversed    = b;
  reversed["flow"] = nn::reverse(b.at("flow"), {1});
  compare(Predict(model, reversed), "reverse_order");
  auto features = root / "feature_arrays.npz";
  if (std::filesystem::exists(features)) {
    double error = replay::MaxDiff(model.Features(b), features);
    check::Near(error, 0, 2e-5, "TF all CNN-M feature endpoints");
    worst = std::max(worst, error);
  }
  std::cout << "TF two-stream trained-weight/all controls/features max=" << worst << std::endl;
  return worst;
}

} // namespace two_stream
