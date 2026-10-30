#pragma once
#include <datamunge/nn/two_stream_models.hpp>
#include "common/c3d.hpp"
#include "common/tf_replay.hpp"
namespace two_stream{namespace nn=datamunge::nn;using nn::Tensor;using A=nn::models::TwoStreamData;using Config=nn::models::TwoStreamConfig;using Stream=nn::models::TwoStreamBranch;using Path=nn::models::TwoStreamOutput;
inline auto FinalSide(int image){return nn::models::TwoStreamFinalSide(image);}
inline auto FlowStack(const Tensor&x,int image,int fields=10){return nn::models::TwoStreamFlowStack(x,image,fields);}
inline auto Fusion(const Tensor&a,const Tensor&b){return nn::models::TwoStreamFusion(a,b);}
inline auto CnnM(nn::ParameterStore&p,const std::string&name,Tensor h,const Config&c,bool temporal,bool training=false,nn::Generator*d6=nullptr,nn::Generator*d7=nullptr){return nn::models::TwoStreamCnnM(p,name,h,c,temporal,training,d6,d7);}
inline auto Forward(nn::ParameterStore&p,const Tensor&rgb,const Tensor&flow,const Config&c={},bool training=false,std::vector<nn::Generator>*drops=nullptr){return nn::models::TwoStream(p,rgb,flow,c,training,drops);}
inline auto Load(const std::filesystem::path&root){return nn::models::LoadTwoStream(root);}
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
inline auto GridFeatures(const A&d,bool motion){return nn::models::TwoStreamGridFeatures(d,motion);}
inline auto Nearest(const A&a,const A&b,bool motion){return nn::models::TwoStreamNearest(a,b,motion);}
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
