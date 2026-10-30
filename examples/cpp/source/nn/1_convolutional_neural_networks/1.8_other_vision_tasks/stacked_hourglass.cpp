// CMD: just run-nn stacked_hourglass
// Two stacked hourglasses with depth4,preactivation residuals,BN and feedback.
// https://arxiv.org/abs/1603.06937
// https://github.com/princeton-vl/pose-hg-train/blob/master/src/models/hg.lua
// Full width256 and compact32 graphs;compact128RGB->32px COCO17 heatmaps.
// Intermediate masked Gaussian supervision at BOTH stacks;three default epochs.
// Real COCO8-Pose source-image-disjoint oracle person crops,fit-only mirror.
// Native target/loss/decoder/metric/residual/BN/feedback gradient checks;
// restored independent RGB-only inference includes moving BN state.
//
// Port of tflow's stacked_hourglass. The crops are tflow's
// $DATAMUNGE_DATA_DIR/human_pose/poses.npz (tflow: just human-pose-assets);
// targets and the batch order (nn::BatchOrder seed 181) are tflow's exactly;
// weights come from datamunge's generator, so trained numbers match TF
// statistically. TF builds the full width-256 model as a graph only; here it
// also runs forward/backward once on a 2x256x256 batch. BN moving averages
// update during the training-mode forward (TF: update ops in the same run).
// The checkpoint is best.npz (weights + BN state) instead of a SaveV2 bundle;
// the "deployment graph has only an RGB placeholder" check becomes a model
// built without optimizer whose Predict() takes RGB alone.
// Outputs go to output/nn/models/stacked_hourglass.
// Options: epochs=3 data=PATH checks_only=0|1 threads=N.

#include <datamunge/datasets/assets.hpp>
#include <cmath>
#include <iostream>

#include "common/hourglass_training.hpp"
#include "common/paths.hpp"

// Experiment workflow from examples/nn/common/hourglass_training.hpp.
namespace pose {
using namespace ::pose;
inline void TrainHourglass(const std::filesystem::path& root, int epochs, const std::filesystem::path& directory) {
  check::Require(epochs >= 2, "at least two short pose epochs");
  if (!std::filesystem::exists(root / "poses.npz"))
    datamunge::datasets::prepare_example_data("pose", (root).string());
  const auto     data = LoadPoseCrops(root);
  HourglassModel model;
  const auto     probe   = PoseBatch(data.fit, {0, 1});
  const double   initial = model.Loss(probe, false, true)[0];
  std::filesystem::create_directories(directory);
  const auto    prefix = (directory / "best.npz").string();
  std::ofstream history(directory / "history.tsv");
  history << "epoch\ttotal\tstack0\tstack1\tvalidation\n";
  nn::BatchOrder      order(12, 2, 181);
  double              best     = INFINITY;
  int                 selected = 0;
  std::vector<double> totals;
  std::cout << "stacked_hourglass:" << model.parameters.count() << " parameters/" << model.parameters.size() << " tensors,BNstate=" << model.state.count()
            << "/" << model.state.size() << ";2stacks/depth4/width32,input128/heat32;" << epochs << " epochs,batch2,Adam.001\n"
            << " true COCO17:12fit crops from6people/3images(includes mirror),1val person/1image,14test people/4images;oracle bbox crops,not detector accuracy\n"
            << " fixed training-mode fit probe initial=" << initial << "\n";
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    std::array<double, 3> average{};
    std::size_t           seen = 0;
    for (;;) {
      const auto ids = order.next();
      if (ids.empty()) break;
      const std::vector<std::size_t> rows(ids.begin(), ids.end());
      const auto                     loss = model.Loss(PoseBatch(data.fit, rows), true);
      for (std::size_t i = 0; i < 3; ++i) average[i] += double(rows.size()) * loss[i];
      seen += rows.size();
    }
    check::Require(seen == 12, "every real-person/flip crop used once per epoch");
    for (auto& value : average) value /= double(seen);
    totals.push_back(average[0]);
    const double validation = HourglassValidation(model, data.validation);
    if (validation < best) {
      best = validation, selected = epoch;
      model.Save(prefix);
    }
    history << epoch;
    for (double value : average) history << '\t' << value;
    history << '\t' << validation << '\n';
    history.flush();
    std::cout << " epoch " << epoch << ":total/stack0/stack1=" << average[0] << "/" << average[1] << "/" << average[2] << ",validation=" << validation
              << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  model.Restore(prefix);
  check::Near(HourglassValidation(model, data.validation), best, 1e-7, "restored pose validation agreement");
  const double learned = model.Loss(probe, false, true)[0];
  check::Require(learned < initial && totals.back() < totals.front(), "fixed training-mode and mean epoch pose objectives improve");
  HourglassModel deployed(false);
  deployed.Restore(prefix);
  check::Require(!deployed.optimizer, "only observed RGB crop plus checkpoint path (no targets/optimizer in deployment)");
  const auto before = deployed.Snapshot();
  const auto heat = HourglassPredict(deployed, data.test), reference = HourglassPredict(model, data.test), repeat = HourglassPredict(deployed, data.test);
  const auto hv = heat.to_vector<float>(), rv = reference.to_vector<float>(), pv = repeat.to_vector<float>();
  double     parity = 0;
  for (std::size_t i = 0; i < hv.size(); ++i) {
    parity = std::max(parity, std::abs(double(hv[i]) - rv[i]));
    check::Near(hv[i], pv[i], 0, "deterministic hourglass inference");
  }
  check::Near(parity, 0, 1e-6, "standalone restored pose parity");
  const auto single = deployed.Predict(PoseBatch(data.test.rgb, {3})).to_vector<float>();
  for (std::size_t i = 0; i < single.size(); ++i) check::Near(single[i], hv[3 * single.size() + i], 2e-5, "single/batch BN inference agreement");
  const auto changed = deployed.Predict(PoseBatch(data.test.rgb, {1})).to_vector<float>(),
             original = deployed.Predict(PoseBatch(data.test.rgb, {0})).to_vector<float>();
  double sensitivity = 0;
  for (std::size_t i = 0; i < changed.size(); ++i) sensitivity += std::abs(double(changed[i]) - original[i]);
  sensitivity /= double(changed.size());
  check::Require(sensitivity > 1e-6, "different observed crops change predicted heatmaps");
  check::Require(before == deployed.Snapshot(), "inference leaves ALL weights and BN state unchanged");
  const auto decoded = DecodePoseHeatmaps(heat);
  const auto mean    = FitPoseMean(data.fit);
  const auto prior   = ConstantPose(data.test, mean);
  const auto center  = ConstantPose(data.test, nn::full({17, 3}, 63.5));
  std::cout << " selected=" << selected << ",restored validation=" << best << ",fixed training probe " << initial << " -> " << learned
            << ";validation uses moving BN,probe uses batch BN\n";
  PrintPose("learned", ScorePose(decoded, data.test));
  PrintPose("fit-only mean pose", ScorePose(prior, data.test));
  PrintPose("crop center", ScorePose(center, data.test));
  PosePreview(data.test, decoded, heat, directory / "comparison.png");
  nn::save_npz((directory / "test_outputs.npz").string(), {{"rgb", data.test.rgb},
                                                           {"truth", data.test.joints},
                                                           {"box_size", data.test.box_size},
                                                           {"heatmaps", heat},
                                                           {"prediction", decoded},
                                                           {"fit_mean", prior},
                                                           {"center", center},
                                                           {"image_ids", data.test.image_ids},
                                                           {"person_ids", data.test.person_ids}});
  std::filesystem::copy_file(root / "manifest.json", directory / "data_manifest.json", std::filesystem::copy_options::overwrite_existing);
  check::Require(bool(history), "save pose history");
  std::cout << " independent RGB-only checkpoint parity=" << parity << ",crop sensitivity=" << sensitivity
            << ",immutable weights/BN and single/batch agreement;artifacts:" << directory.string() << " (training " << seconds
            << " s)\nComplete short stacked-hourglass demonstration passed;8-source-image custom smoke test,not MPII/COCO benchmark accuracy.\n";
}
}  // namespace pose



namespace nn = datamunge::nn;
using nn::Tensor;
using namespace pose;

namespace {

void TargetLossDecoderChecks() {
  std::vector<float> jv(2 * 17 * 3, 0.f);
  auto               J = [&](int n, int j, int c) -> float& { return jv[std::size_t((n * 17 + j) * 3 + c)]; };
  J(0, 0, 0) = 41.5, J(0, 0, 1) = 81.5, J(0, 0, 2) = 2;
  J(0, 1, 0) = 0, J(0, 1, 1) = 127, J(0, 1, 2) = 1;
  J(0, 2, 0) = NAN, J(0, 2, 1) = NAN;
  const auto target = PoseHeatmaps(nn::from_vector(jv, {2, 17, 3}));
  const auto heat = target.heatmap.to_vector<float>(), mask = target.mask.to_vector<float>();
  for (int n = 0; n < 2; ++n)
    for (int y = 0; y < 32; ++y)
      for (int x = 0; x < 32; ++x)
        for (int j = 0; j < 17; ++j) {
          double       expected = 0;
          const double cx = j == 0 ? 10 : -.375, cy = j == 0 ? 20 : 31.375;
          if (n == 0 && j < 2 && std::abs(x - cx) <= 3 && std::abs(y - cy) <= 3) expected = std::exp(-((x - cx) * (x - cx) + (y - cy) * (y - cy)) / 2);
          check::Near(heat[std::size_t(((n * 32 + y) * 32 + x) * 17 + j)], expected, 3e-8,
                      "every exact Gaussian including pixel-center coordinates,border truncation and unlabeled NaNs");
        }
  check::Require(mask[0] == 1 && mask[1] == 1 && mask[2] == 0, "visible AND occluded labels supervise;missing joint ignored");
  const auto decoded = DecodePoseHeatmaps(target.heatmap).to_vector<float>();
  check::Near(decoded[0], 41.5, 0, "exact centered x recovery");
  check::Near(decoded[1], 81.5, 0, "exact centered y recovery");
  std::vector<float> prediction(24, NAN), truth(24, NAN), m(6, 0.f);
  m[0] = 1, m[1] = 1, m[2] = 1;  // (0,0,0,0), (0,0,0,1), (1,0,0,0)
  for (int n = 0; n < 2; ++n)
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 2; ++x)
        for (int j = 0; j < (n == 0 ? 2 : 1); ++j) {
          const std::size_t i = std::size_t(((n * 2 + y) * 2 + x) * 2 + j);
          prediction[i] = n ? 3.f : 2.f, truth[i] = 0;
        }
  Tensor     p    = nn::from_vector(prediction, {3, 2, 2, 2}).requires_grad_();
  const auto loss = PoseHeatmapMSE(p, nn::from_vector(truth, {3, 2, 2, 2}), nn::from_vector(m, {3, 1, 1, 2}), 2);
  const auto g    = nn::grad(loss, {p})[0].to_vector<float>();
  check::Near(loss.item(), 6.5, 0, "MSE equal image weighting and exclude empty images");
  for (int n = 0; n < 3; ++n)
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 2; ++x)
        for (int j = 0; j < 2; ++j)
          check::Near(g[std::size_t(((n * 2 + y) * 2 + x) * 2 + j)], n == 0 ? .25 : n == 1 && j == 0 ? .75 : 0, 0,
                      "every masked MSE derivative;missing NaNs never contribute");
  const auto ties = DecodePoseHeatmaps(nn::zeros_like(target.heatmap)).to_vector<float>();
  check::Near(ties[0], 1.5, 0, "first maximum wins deterministic heatmap tie");
  PoseData data;
  std::vector<float> dj(51, 0.f);
  dj[2]         = 1;
  data.joints   = nn::from_vector(dj, {1, 17, 3});
  data.box_size = nn::tensor({25, 20}, {1, 2});
  std::vector<float> pj(51, 0.f);
  pj[0]       = 3, pj[1] = 4;
  auto score  = ScorePose(nn::from_vector(pj, {1, 17, 3}), data);
  check::Near(score.pixels, 5, 0, "known3/4/5 keypoint distance");
  check::Near(score.normalized_error, .2, 1e-12, "max-box-side normalization");
  check::Near(score.pck, 1, 0, "inclusive PCK threshold");
  pj[1] = 4.001f;
  check::Near(ScorePose(nn::from_vector(pj, {1, 17, 3}), data).pck, 0, 0, "PCK outside boundary");
  std::cout << " every Gaussian/visibility/half-pixel/border target,empty/NaN-safe MSE and ALL derivatives,decoder/tie,known inclusive PCK fixtures passed\n";
}

void ResidualChecks() {
  nn::ParameterStore p(191), state(193);
  {
    nn::NoGradGuard guard;
    PoseResidual(p, state, "block", nn::zeros({1, 2, 2, 2}), 4, false);
  }
  for (const auto& name : p.names()) {
    if (name.find("/W") == std::string::npos) continue;
    const auto         shape = p.at(name).shape();
    std::vector<float> value(std::size_t(p.at(name).numel()), 0.f);
    if (name == "block/conv1/W") value[0] = 1, value[3] = 1;
    if (name == "block/conv2/W") value[std::size_t(((1 * 3 + 1) * 2 + 0) * 2 + 0)] = .5f, value[std::size_t(((1 * 3 + 1) * 2 + 1) * 2 + 1)] = -1;
    if (name == "block/conv3/W") value = {1, -2, .5f, 0, 0, 1, -1, .25f};
    if (name == "block/skip/W") value = {1, 0, -1, .5f, 0, 2, .5f, -1};
    p.set(name, nn::from_vector(value, shape));
  }
  const std::vector<float> in = {-2, 1, 1, -3, 2, 4, -1, -2};
  Tensor                   x  = nn::from_vector(in, {1, 2, 2, 2}).requires_grad_();
  const auto               output   = PoseResidual(p, state, "block", x, 4, false);
  const auto               gradient = nn::grad(nn::sum(output), {x})[0].to_vector<double>();
  const auto               out      = output.to_vector<double>();
  const double             z        = std::sqrt(1 + 1e-5);
  const double             weights[2][4] = {{1, -2, .5, 0}, {0, 1, -1, .25}}, skip[2][4] = {{1, 0, -1, .5}, {0, 2, .5, -1}};
  double                   error         = 0;
  for (int i = 0; i < 4; ++i)
    for (int c = 0; c < 4; ++c) {
      const double a = in[std::size_t(2 * i)], b = in[std::size_t(2 * i + 1)], h0 = std::max(0., std::max(0., a / z) / z * .5) / z,
                   h1       = std::max(0., -std::max(0., b / z) / z) / z;
      const double expected = h0 * weights[0][c] + h1 * weights[1][c] + a * skip[0][c] + b * skip[1][c];
      error                 = std::max(error, std::abs(expected - out[std::size_t(4 * i + c)]));
    }
  check::Near(error, 0, 1e-6, "preactivation BN/ReLU 1-3-1 and raw skip projection,NO terminal ReLU");
  for (int i = 0; i < 8; ++i) {
    const double expected = (i % 2 ? 1.5 : .5) + (i % 2 || in[std::size_t(i)] < 0 ? 0 : -.25 / std::pow(z, 3));
    check::Near(gradient[std::size_t(i)], expected, 1e-6, "analytic signed residual input derivative through positive/negative gates and skip");
  }
  bool unchanged = state.size() == 6;
  for (std::size_t i = 0; i < state.size(); ++i) {
    const bool var = state.names()[i].find("moving_var") != std::string::npos;
    for (float v : state.parameters()[i].to_vector<float>()) unchanged = unchanged && v == (var ? 1.f : 0.f);
  }
  check::Require(unchanged, "inference residual has no BN updates");
  std::cout << " signed preactivation residual/projection/every input derivative/no terminal ReLU max value error=" << error << " passed\n";
}

std::int64_t Census(int stem, int width) {
  std::int64_t count    = 0;
  auto         conv     = [&](std::int64_t in, std::int64_t out, std::int64_t kernel = 1) { count += kernel * kernel * in * out + out; };
  auto         bn       = [&](std::int64_t channels) { count += 2 * channels; };
  auto         residual = [&](int in, int out) {
    bn(in);
    conv(in, out / 2);
    bn(out / 2);
    conv(out / 2, out / 2, 3);
    bn(out / 2);
    conv(out / 2, out);
    if (in != out) conv(in, out);
  };
  conv(3, stem, 7);
  bn(stem);
  residual(stem, stem * 2);
  residual(stem * 2, stem * 2);
  residual(stem * 2, width);
  for (int stack = 0; stack < 2; ++stack) {
    for (int i = 0; i < 13; ++i) residual(width, width);
    residual(width, width);
    conv(width, width);
    bn(width);
    conv(width, 17);
    if (stack == 0) {
      conv(width, width);
      conv(17, width);
    }
  }
  return count;
}

void ModelChecks() {
  for (bool compact : {false, true}) {
    nn::ParameterStore p(173), state(179);
    HourglassConfig    config;
    if (compact) config = CompactHourglassConfig();
    std::vector<float> rgb(std::size_t(2 * config.size * config.size * 3));
    for (std::size_t i = 0; i < rgb.size(); ++i) rgb[i] = float(.5 + .4 * std::sin(double(i) * .11));
    Tensor     image = nn::from_vector(rgb, {2, config.size, config.size, 3}).requires_grad_();
    const auto net   = StackedHourglass(p, state, image, config, true);
    const auto objective = nn::mean(nn::square(net.heatmaps[0])) + nn::mean(nn::square(net.heatmaps[1]));
    auto       inputs    = p.parameters();
    inputs.push_back(image);
    nn::GradOptions keep;
    keep.retain_graph     = true;
    const auto gradients  = nn::grad(objective, inputs, keep);
    const auto feedback   = nn::grad(nn::mean(nn::square(net.heatmaps[1])), {net.heatmaps[0]})[0];
    check::Require(p.count() == Census(config.stem, config.width) && net.stages.size() == 14 && net.heatmaps.size() == 2,
                   "independent two-stack/depth4 parameter census and all exposed stages");
    std::cout << " " << (compact ? "compact" : "full") << " hourglass:" << p.count() << " parameters/" << p.size()
              << " derivatives (executed),BN=" << state.count() << "/" << state.size() << ",14stages,feedback connected\n";
    if (!compact) continue;
    std::vector<nn::Shape> shapes{{2, 32, 32, 32}};
    for (int stack = 0; stack < 2; ++stack) {
      for (std::int64_t size : {4, 8, 16, 32}) shapes.push_back({2, size, size, 32});
      shapes.push_back({2, 32, 32, 32});
      shapes.push_back({2, 32, 32, 17});
      if (stack == 0) shapes.push_back({2, 32, 32, 32});
    }
    for (std::size_t i = 0; i < 14; ++i) check::Require(net.stages[i].shape() == shapes[i], "all recursive resolution/merge/stack shape checks");
    std::vector<Tensor> all = gradients;
    all.push_back(feedback);
    int    nonzero = 0;
    double smallest = INFINITY, largest = 0;
    for (std::size_t i = 0; i < all.size(); ++i) {
      double norm = 0;
      for (float v : all[i].to_vector<float>()) {
        check::Require(std::isfinite(v), "all hourglass tensor gradients finite");
        norm += double(v) * v;
      }
      norm = std::sqrt(norm);
      if (norm > 0) ++nonzero, smallest = std::min(smallest, norm), largest = std::max(largest, norm);
      if (i >= p.size() || p.names()[i].find("/W") != std::string::npos)
        check::Require(norm > 0, "every convolution kernel,interstack feedback and RGB receive gradients");
    }
    std::cout << " compact all" << p.size() << " parameter derivatives finite;" << nonzero << " nonzero including feedback/RGB,norm range[" << smallest
              << "," << largest << "];all kernels nonzero,all14 recursive shapes passed\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs      = 3;
    bool                  checks_only = false;
    std::filesystem::path root        = paths::Dataset("human_pose");
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("epochs=", 0) == 0)
        epochs = std::stoi(arg.substr(7));
      else if (arg.rfind("data=", 0) == 0)
        root = arg.substr(5);
      else if (arg == "checks_only=0" || arg == "checks_only=1")
        checks_only = arg == "checks_only=1";
      else if (arg.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(arg.substr(8)));
      else
        throw std::invalid_argument("expected epochs=N data=PATH checks_only=0|1 threads=N");
    }
    TargetLossDecoderChecks();
    ResidualChecks();
    ModelChecks();
    if (checks_only) return 0;
    TrainHourglass(root, epochs, paths::OutputDir("models/stacked_hourglass"));
    std::cout << "All stacked-hourglass checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
