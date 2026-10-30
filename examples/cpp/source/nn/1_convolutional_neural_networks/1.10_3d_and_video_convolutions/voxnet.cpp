// CMD: just run-nn voxnet
// Original-width VoxNet32cube,32filters5/stride2,32filters3,pool2,FC128/head.
// LeakyReLU.1,drop rates.2/.3/.4,momentum.9,lr.001,L2.001 weights only.
// Actual canonical40-class fullwidth forward/all8 gradients on real CAD;
// default3epochs3-class ModelNet10 subset,exactC4 rotation training/voting.
// Prepare:just voxel-shape-assets. Options:epochs=N,data=PATH,checks_only=1.
// Eager shapes/gradients replace graph refinement; NPZ replaces checkpoints.
// Native initialization/dropout is statistical; trained TF arrays check inference.
#include "common/voxnet.hpp"

// Experiment workflow from examples/nn/common/voxnet.hpp.
namespace vox {
using namespace ::vox;
inline void FullRuntime(const Data& sample, const std::filesystem::path& root) {
  nn::ParameterStore p(911);
  auto               net = Forward(p, sample.voxels, 40);
  auto               g   = nn::grad(nn::mean(nn::square(net.logits)), p.parameters());
  for (const auto& v : g) {
    check::Require(
        nn::all(nn::is_finite(v)).item<bool>() && nn::sum(nn::square(v)).item() > 0,
        "all8 full40 gradients finite/nonzero");
  }
  check::Require(
      p.count() == 921736 && net.logits.shape() == nn::Shape({1, 40}),
      "canonical full40 count/logits");
  nn::NamedTensors out{{"voxels", sample.voxels}, {"logits", net.logits}};
  for (std::size_t i = 0; i < p.size(); ++i)
    out.emplace_back("p" + std::to_string(i), p.at(p.names()[i]));
  nn::save_npz((root / "full_runtime_fixture.npz").string(), out);
  std::cout << "canonical full40 VoxNet921736 parameters: actual real-occupancy forward/all8 "
               "gradients pass\n";
}
}  // namespace vox



// Experiment workflow from examples/nn/common/voxnet.hpp.
namespace vox {
using namespace ::vox;
inline void Train(const std::filesystem::path& root, int epochs) {
  check::Require(epochs > 0, "positive short epochs");
  auto        pools     = Load(root);
  const auto& fit       = pools[0];
  const auto& val       = pools[1];
  const auto& test      = pools[2];
  const auto  directory = paths::OutputDir("models/voxnet"), checkpoint = directory / "best.npz";
  FullRuntime(Rows(fit, {0}), directory);
  Model      model;
  const auto initial   = sensor::ScoreSensor(Predict(model, fit), fit.labels, 3).nll;
  auto       untrained = Predict(model, test);
  auto ce = nn::grad(model.Classification(Rows(fit, {0, 8, 16, 1})), model.parameters.parameters());
  auto regularized = Regularized(model.parameters, ce);
  for (std::size_t i = 0; i < ce.size(); ++i) {
    check::Require(
        nn::all(nn::is_finite(ce[i])).item<bool>() && nn::sum(nn::square(ce[i])).item() > 0,
        "all8 classification-only gradients finite/nonzero");
    auto expected = ce[i];
    if (model.parameters.names()[i].find("/W") != std::string::npos)
      expected = expected + model.parameters.at(model.parameters.names()[i]) * .002;
    check::Near(
        replay::MaxAbsDiff(regularized[i], expected),
        0,
        1e-7,
        "every coordinate CE+2lambdaW,all biases unchanged");
  }
  std::cout
      << "VoxNet original widths:" << model.parameters.count() << " parameters/8 tensors;" << epochs
      << " epochs,batch4,momentum.9/lr.001,L2.001W,drop rates.2/.3/.4,random C4;initialfitNLL="
      << initial << std::endl;
  nn::BatchOrder order(24, 4, 919);
  std::mt19937   rng(929);
  int            selected = 0;
  double         best     = INFINITY;
  std::ofstream  history(directory / "history.tsv");
  history << "epoch\ttrain_dropout_ce_plus_l2\tvalidation_c4_nll\n";
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    double      sum  = 0;
    std::size_t seen = 0;
    for (;;) {
      auto rows = order.next();
      if (rows.empty())
        break;
      auto batch   = Rows(fit, rows);
      batch.voxels = Rotate(batch.voxels, std::uniform_int_distribution<int>(0, 3)(rng));
      sum += rows.size() * model.Update(batch);
      seen += rows.size();
    }
    check::Require(seen == 24, "all24 fitting objects once per epoch");
    double score = sensor::ScoreSensor(Predict(model, val), val.labels, 3).nll;
    if (score < best) {
      best     = score;
      selected = epoch;
      model.parameters.save(checkpoint.string());
    }
    history << epoch << '\t' << sum / seen << '\t' << score << '\n';
    history.flush();
    std::cout << "epoch " << epoch << ":trainCE+L2=" << sum / seen
              << ",validation votedNLL=" << score << std::endl;
  }
  model.parameters.load(checkpoint.string());
  check::Near(
      sensor::ScoreSensor(Predict(model, val), val.labels, 3).nll,
      best,
      1e-7,
      "restored validation score");
  double learned = sensor::ScoreSensor(Predict(model, fit), fit.labels, 3).nll;
  Model  deployed(false);
  deployed.parameters.load(checkpoint.string());
  auto before = deployed.Arrays();
  auto logits = Predict(deployed, test), canonical = Predict(deployed, test, false),
       reference = Predict(model, test), repeated = Predict(deployed, test);
  check::Near(replay::MaxAbsDiff(logits, reference), 0, 1e-6, "fresh8-tensor restore");
  check::Near(
      replay::MaxAbsDiff(logits, repeated), 0, 0, "deterministic dropout-free C4 inference");
  check::Near(
      replay::MaxAbsDiff(deployed.Predict(Rows(test, {0})), nn::slice(logits, {0, 0}, {1, 3})),
      0,
      1e-6,
      "single/batch occupancy inference");
  auto rotated           = test;
  rotated.voxels         = Rotate(test.voxels, 1);
  auto rotated_vote      = Predict(deployed, rotated),
       rotated_canonical = Predict(deployed, rotated, false);
  check::Near(
      replay::MaxAbsDiff(rotated_vote, logits), 0, 1e-6, "C4 vote invariant to orbit permutation");
  double change = nn::sum(nn::abs(canonical - rotated_canonical)).item();
  check::Require(change > 1e-7, "single-view orientation sensitivity");
  check::Require(replay::SameBits(before, deployed.Arrays()), "immutable8 tensors at inference");
  deployed.Export(directory / "model_arrays.npz");
  auto nearest = Nearest(fit, test);
  Metrics("C4-voted VoxNet", test, logits);
  Metrics("canonical single view", test, canonical);
  Metrics("untrained C4-vote", test, untrained);
  Metrics("fit C4-coarse occupancy1NN", test, nearest);
  Metrics("rotated canonical control", test, rotated_canonical);
  std::ofstream predictions(directory / "predictions.tsv");
  predictions << "source_id\tlabel\tprediction\n";
  const auto z      = logits.to_vector<float>();
  const auto labels = test.labels.to_vector<int>(), ids = test.source.to_vector<int>();
  for (int n = 0; n < 12; ++n) {
    int best = 0;
    for (int c = 1; c < 3; ++c)
      if (z[n * 3 + c] > z[n * 3 + best])
        best = c;
    predictions << ids[n] << '\t' << labels[n] << '\t' << best << '\n';
  }
  nn::save_npz(
      (directory / "test_outputs.npz").string(),
      {{"voxels", test.voxels},
       {"labels", test.labels},
       {"source_id", test.source},
       {"logits", logits},
       {"canonical", canonical},
       {"untrained", untrained},
       {"nearest", nearest},
       {"rotated_vote", rotated_vote},
       {"rotated_canonical", rotated_canonical},
       {"fit_voxels", fit.voxels},
       {"fit_labels", fit.labels},
       {"fit_source_id", fit.source}});
  for (const auto name : {"manifest.json", "ATTRIBUTION.txt", "test_contact.png"})
    std::filesystem::copy_file(
        root / name,
        directory / (std::string(name) == "manifest.json" ? "data_manifest.json" : name),
        std::filesystem::copy_options::overwrite_existing);
  check::Require(bool(predictions) && bool(history), "saved voxel histories/predictions");
  std::cout
      << "selected=" << selected << ",fixed C4fitNLL " << initial << " -> " << learned
      << ",canonical quarterturn change=" << change
      << ";restore/occupancy-only/single-batch/C4invariance/immutable PASS\nTiny3-class42-object "
         "ModelNet10 subset;12 test objects;four quarterturn votes,not fullbenchmark accuracy.\n";
  const auto tf = replay::TfModels("voxnet");
  if (std::filesystem::exists(tf / "model_arrays.npz") &&
      std::filesystem::exists(tf / "test_outputs.npz")) {
    Model recorded(false);
    recorded.LoadExport(tf / "model_arrays.npz");
    auto   arrays = replay::ToMap(nn::load_npz((tf / "test_outputs.npz").string()));
    Data   d{arrays.at("voxels"), arrays.at("labels"), arrays.at("source_id")};
    double error = replay::MaxAbsDiff(Predict(recorded, d), arrays.at("logits"));
    check::Near(error, 0, 1e-5, "TF-trained C4 inference parity");
    std::cout << "TF trained-weight C4 max difference=" << error << std::endl;
  }
}
}  // namespace vox


namespace nn = datamunge::nn;
namespace {
void ConvChecks() {
  std::vector<float> input(686), kernel(500), coefficient(16);
  for (int i = 0; i < 686; ++i)
    input[i] = .2 * std::sin(.31 * i);
  for (int i = 0; i < 500; ++i)
    kernel[i] = .03 * std::cos(.27 * i);
  for (int i = 0; i < 16; ++i)
    coefficient[i] = .13 * std::cos(.41 * i);
  auto              x = nn::from_vector(input, {1, 7, 7, 7, 2}).requires_grad_();
  auto              w = nn::from_vector(kernel, {5, 5, 5, 2, 2}).requires_grad_();
  auto              b = nn::from_vector(std::vector<float>{.3f, -.3f}, {2}).requires_grad_();
  nn::Conv3DOptions opt;
  opt.strides    = {2, 2, 2};
  opt.padding    = nn::Padding::Valid;
  auto y         = nn::leaky_relu(nn::conv3d(x, w, opt) + b, .1);
  auto gradients = nn::grad(nn::sum(y * nn::from_vector(coefficient, {1, 2, 2, 2, 2})), {w, b, x});
  std::vector<std::vector<double>> values{
      w.to_vector<double>(), b.to_vector<double>(), x.to_vector<double>()};
  auto host = [&](const std::vector<std::vector<double>>& values,
                  std::vector<double>*                    outputs = nullptr) {
    double total = 0;
    for (int x = 0; x < 2; ++x)
      for (int y = 0; y < 2; ++y)
        for (int z = 0; z < 2; ++z)
          for (int c = 0; c < 2; ++c) {
            double value = values[1][c];
            for (int a = 0; a < 5; ++a)
              for (int b = 0; b < 5; ++b)
                for (int d = 0; d < 5; ++d)
                  for (int ci = 0; ci < 2; ++ci)
                    value += values[2][(((2 * x + a) * 7 + 2 * y + b) * 7 + 2 * z + d) * 2 + ci] *
                             values[0][(((a * 5 + b) * 5 + d) * 2 + ci) * 2 + c];
            value = value >= 0 ? value : .1 * value;
            int i = ((x * 2 + y) * 2 + z) * 2 + c;
            total += value * coefficient[i];
            if (outputs)
              outputs->push_back(value);
          }
    return total;
  };
  std::vector<double> expected;
  host(values, &expected);
  const auto actual = y.to_vector<float>();
  double     error = 0, derivative = 0;
  int        coordinates = 0;
  for (int i = 0; i < 16; ++i)
    error = std::max(error, std::abs(expected[i] - actual[i]));
  for (int variable = 0; variable < 3; ++variable) {
    const auto g = gradients[variable].to_vector<float>();
    for (std::size_t i = 0; i < values[variable].size(); ++i) {
      auto plus = values, minus = values;
      plus[variable][i] += 1e-5;
      minus[variable][i] -= 1e-5;
      derivative = std::max(derivative, std::abs((host(plus) - host(minus)) / 2e-5 - g[i]));
      ++coordinates;
    }
  }
  check::Require(coordinates == 1188, "all500kernel+2bias+686input coordinates");
  check::Near(error, 0, 3e-7, "all16 stride2 valid5cube leaky outputs");
  check::Near(derivative, 0, 2e-7, "all1188 independent host finite differences");
  std::cout << "VoxNet5cube/stride2/leaky.1:forward=" << error << ",all1188 FD error=" << derivative
            << std::endl;
}
void RotationPoolChecks() {
  std::vector<float> input(27);
  for (int x = 0; x < 3; ++x)
    for (int y = 0; y < 3; ++y)
      for (int z = 0; z < 3; ++z)
        input[(x * 3 + y) * 3 + z] = 100 * x + 10 * y + z;
  auto tensor = nn::from_vector(input, {1, 3, 3, 3, 1}).requires_grad_();
  for (int k = 0; k < 4; ++k) {
    auto v = vox::Rotate(tensor, k).to_vector<float>();
    for (int x = 0; x < 3; ++x)
      for (int y = 0; y < 3; ++y)
        for (int z = 0; z < 3; ++z) {
          int a = x, b = y;
          for (int step = 0; step < k; ++step) {
            int previous = a;
            a            = b;
            b            = 2 - previous;
          }
          check::Near(
              v[(x * 3 + y) * 3 + z],
              100 * a + 10 * b + z,
              0,
              "every C4 coordinate,preserved third axis");
        }
  }
  check::Near(
      replay::MaxAbsDiff(vox::Rotate(vox::Rotate(tensor, 3), 1), tensor),
      0,
      0,
      "four turns compose to identity");
  auto rotation_gradient = nn::grad(nn::sum(vox::Rotate(tensor, 1)), {tensor})[0];
  for (float x : rotation_gradient.to_vector<float>())
    check::Near(x, 1, 0, "all27 rotation input gradients");
  std::vector<float> negatives(64);
  for (int i = 0; i < 64; ++i)
    negatives[i] = -1 - i;
  auto x      = nn::from_vector(negatives, {1, 4, 4, 4, 1}).requires_grad_();
  auto pooled = nn::max_pool3d(x, 2, 2, nn::Padding::Valid);
  auto grad = nn::grad(nn::sum(pooled), {x})[0].to_vector<float>(), v = pooled.to_vector<float>();
  for (int x = 0; x < 2; ++x)
    for (int y = 0; y < 2; ++y)
      for (int z = 0; z < 2; ++z)
        check::Near(
            v[(x * 2 + y) * 2 + z],
            -1 - ((2 * x * 4 + 2 * y) * 4 + 2 * z),
            0,
            "unique real-sample maximum");
  for (int x = 0; x < 4; ++x)
    for (int y = 0; y < 4; ++y)
      for (int z = 0; z < 4; ++z)
        check::Near(
            grad[(x * 4 + y) * 4 + z],
            x % 2 == 0 && y % 2 == 0 && z % 2 == 0 ? 1 : 0,
            0,
            "all64 pool input derivatives");
  std::cout
      << "C4 all108 coordinates/closure/27 input gradients,3Dpool8 values/all64 derivatives pass\n";
}
void RegularizerChecks() {
  nn::ParameterStore p;
  p.set("layer/W", nn::from_vector(std::vector<float>{2, -1, .5}, {3}));
  p.set("layer/b", nn::from_vector(std::vector<float>{.4f}, {1}));
  const auto coefficient =
      std::vector<double>{double(.3f), double(-.2f), double(.7f), double(1.2f)};
  auto linear =
      nn::sum(p.at("layer/W") * nn::from_vector(std::vector<float>{.3f, -.2f, .7f}, {3})) +
      nn::sum(p.at("layer/b") * 1.2);
  auto                g = vox::Regularized(p, nn::grad(linear, p.parameters()));
  std::vector<double> values{2, -1, .5, double(.4f)}, actual = g[0].to_vector<double>();
  actual.push_back(g[1].item());
  auto host = [&](const std::vector<double>& x) {
    double total = 0;
    for (int i = 0; i < 4; ++i)
      total += coefficient[i] * x[i];
    for (int i = 0; i < 3; ++i)
      total += .001 * x[i] * x[i];
    return total;
  };
  double error = 0;
  for (int i = 0; i < 4; ++i) {
    auto plus = values, minus = values;
    plus[i] += 1e-6;
    minus[i] -= 1e-6;
    error = std::max(error, std::abs((host(plus) - host(minus)) / 2e-6 - actual[i]));
  }
  check::Near(
      error, 0, 1e-7, "all4 host finite differences,exact weight-only L2,unregularized bias");
  std::cout << "weight-only L2 hostFD error=" << error << std::endl;
}
void GraphChecks() {
  for (int classes : {40, 3}) {
    nn::ParameterStore p(941);
    auto               net = vox::Forward(p, nn::zeros({1, 32, 32, 32, 1}), classes);
    check::Require(
        net.conv1.shape() == nn::Shape({1, 14, 14, 14, 32}) &&
            net.conv2.shape() == nn::Shape({1, 12, 12, 12, 32}) &&
            net.pool.shape() == nn::Shape({1, 6, 6, 6, 32}) &&
            net.flat.shape() == nn::Shape({1, 6912}) && net.fc1.shape() == nn::Shape({1, 128}) &&
            net.logits.shape() == nn::Shape({1, classes}),
        "canonical 32->14->12->6/FC128/head geometry");
    const auto expected =
        5 * 5 * 5 * 32 + 32 + 3 * 3 * 3 * 32 * 32 + 32 + 6912 * 128 + 128 + 128 * classes + classes;
    check::Require(
        p.count() == expected && p.size() == 8 && expected == (classes == 40 ? 921736 : 916963),
        "independent all4-layer count");
    const auto g = nn::grad(nn::mean(nn::square(net.logits)), p.parameters());
    for (const auto& v : g)
      check::Require(
          v.defined() && nn::all(nn::is_finite(v)).item<bool>(),
          "all8 canonical gradients exist and are finite");
    std::cout << "VoxNet " << classes << " classes:" << expected
              << " parameters/8 tensors;all shapes/counts/gradients pass\n";
  }
}
} // namespace
int main(int argc, char** argv) {
  try {
    nn::set_num_threads(2);
    int  epochs = 3;
    bool checks = false;
    auto root   = paths::Dataset("voxnet");
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a.rfind("epochs=", 0) == 0)
        epochs = std::stoi(a.substr(7));
      else if (a.rfind("data=", 0) == 0)
        root = a.substr(5);
      else if (a == "checks_only=1")
        checks = true;
      else if (a != "checks_only=0")
        throw std::invalid_argument("expected epochs=N data=PATH checks_only=0|1");
    }
    ConvChecks();
    RotationPoolChecks();
    RegularizerChecks();
    GraphChecks();
    if (!checks)
      vox::Train(root, epochs);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
