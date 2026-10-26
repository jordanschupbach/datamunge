// CMD: just run-nn flownets
// FlowNetS optical flow with all six encoder scales and learned decoding.
// https://arxiv.org/abs/1504.06852
// https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetS.py
// Concatenate two jointly centered RGB frames;7/5/5/3/3/3 stride2 encoder,
// extra conv3_1..6_1,learned4x4 feature and flow transposes,skip fusion.
// Fullbase64 graph and compactbase8 numerical/training model,no BN,
// LeakyReLU.1,Glorot kernels,zero biases. Outputs atlevels2..6.
// EVERY internal flow is input-image pixels/20;finest bilinear resize*20.
// Masked multi-scale smooth endpoint objective,3epochs default,batch2,Adam.001.
// True Middlebury frame10->11 flows,family-disjoint32fit/8val/24test crops.
// Independent correlation/warp/transpose/units/loss/metric Jacobian checks,
// all compact parameter gradients,validation restore and pair-only inference.
// Export full predictions/warped second image;report zero/fit-mean baselines.
// This implements the S variant;PWC-Net is the other required model in the
// combined catalogue item. No pretrained/FlyingChairs/benchmark claim.
//
// Port of tflow's flownets. Preparation (in ~/projects/tflow_examples): just
// optical-flow-assets; pairs.npz is read from $DATAMUNGE_DATA_DIR/optical_flow.
// The crops, pyramid targets and batch order (nn::BatchOrder seed 167) are
// tflow's; weights come from datamunge's generator, so trained numbers match
// TF statistically. The backward warp uses nn::sample_pixels (TF: explicit
// four-tap gathers); TF builds the full base-64 model as a graph only, here it
// also runs forward/backward once on the 2x64x64 fixture. The checkpoint is
// best.npz. Outputs go to output/nn/models/flownets.
// Options: epochs=3 data=PATH checks_only=0|1 threads=N.

#include <cmath>
#include <iostream>

#include "common/flownet.hpp"
#include "common/optical_flow.hpp"
#include "common/paths.hpp"

// Experiment workflow from examples/nn/common/optical_flow.hpp.
namespace flow {
using namespace ::flow;
inline void TrainOpticalFlow(const std::filesystem::path& root, int epochs, const std::string& name, const FlowBuilder& builder,
                             const std::filesystem::path& directory) {
  check::Require(epochs >= 2, "two or more flow epochs for learning demonstration");
  const auto         split = LoadFlowPairs(root);
  const auto         mean  = FitMeanFlow(split.fit);
  const FlowPrepared fitting{split.fit, PrepareFlowSupervision(split.fit)}, validation{split.validation, PrepareFlowSupervision(split.validation)};
  const auto         probe = FlowBatch(fitting, {0, 1});
  FlowModel          model(builder);
  const double       initial = model.Loss(probe)[0];
  std::filesystem::create_directories(directory);
  const auto    prefix = (directory / "best.npz").string();
  std::ofstream history(directory / "history.tsv");
  history << "epoch\ttotal\tlevel2\tlevel3\tlevel4\tlevel5\tlevel6\tvalidation\n";
  nn::BatchOrder      order(32, 2, 167);
  double              best     = INFINITY;
  int                 selected = -1;
  std::vector<double> totals;
  std::cout << name << ":" << model.parameters.count() << " parameters/" << model.parameters.size() << " tensors;" << epochs
            << " epochs,32fit/8val/24test Middlebury64px,batch2,Adam.001\n"
            << " disjoint scene families;32 fit patches from4 pairs,8val from1,24test from3;NOT independent source pairs per patch\n"
            << " GT(frame10->11):dx/right,dy/down,input pixels;all pyramid fields INPUT units/20;loss weights .005/.01/.02/.08/.32\n"
            << " fixed fitting initial=" << initial << ",fit-only mean vector=" << mean[0] << "," << mean[1]
            << "px;no pretraining/decay/online augmentation\n";
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    std::array<double, 6> average{};
    std::size_t           seen = 0;
    for (;;) {
      const auto ids = order.next();
      if (ids.empty()) break;
      const auto loss = model.Loss(FlowBatch(fitting, std::vector<std::size_t>(ids.begin(), ids.end())), true);
      for (std::size_t i = 0; i < 6; ++i) average[i] += double(ids.size()) * loss[i];
      seen += ids.size();
    }
    check::Require(seen == 32, "every true-flow crop used once per epoch");
    for (auto& value : average) value /= double(seen);
    totals.push_back(average[0]);
    const double value = FlowValidation(model, validation);
    if (value < best) {
      best = value, selected = epoch;
      model.parameters.save(prefix);
    }
    history << epoch;
    for (double loss : average) history << '\t' << loss;
    history << '\t' << value << '\n';
    history.flush();
    std::cout << " epoch " << epoch << ": total/L2/L3/L4/L5/L6=";
    for (double loss : average) std::cout << loss << ",";
    std::cout << "validation=" << value << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  model.parameters.load(prefix);
  check::Near(FlowValidation(model, validation), best, 1e-8, "restored multiscale validation agreement");
  const double learned = model.Loss(probe)[0];
  check::Require(learned < initial * (1 - 1e-4) && totals.back() < totals.front() * (1 - 1e-4), "fixed fitting and mean epoch flow objectives improve");
  FlowModel deployed(builder, false);
  deployed.parameters.load(prefix);
  check::Require(!deployed.optimizer, "only two observed images are inference inputs (no GT/validity/optimizer)");
  const auto before    = deployed.Snapshot();
  const auto predicted = FlowPredict(deployed, split.test), reference = FlowPredict(model, split.test), repeat = FlowPredict(deployed, split.test);
  const auto pv = predicted.to_vector<float>(), rv = reference.to_vector<float>(), qv = repeat.to_vector<float>();
  double     parity = 0;
  for (std::size_t i = 0; i < pv.size(); ++i) {
    parity = std::max(parity, std::abs(double(pv[i]) - rv[i]));
    check::Near(qv[i], pv[i], 0, "repeat deterministic flow inference");
  }
  check::Near(parity, 0, 1e-7, "standalone flow checkpoint parity");
  const auto single = deployed.Predict(FlowBatch(split.test.first, {3}), FlowBatch(split.test.second, {3})).to_vector<float>();
  for (int i = 0; i < 8192; ++i) check::Near(single[std::size_t(i)], pv[3 * 8192 + std::size_t(i)], 2e-6, "single/batch image-pair flow agreement");
  const auto same_frame  = deployed.Predict(split.test.first, split.test.first).to_vector<float>();
  double     pair_change = 0;
  for (std::size_t i = 0; i < pv.size(); ++i) pair_change += std::abs(double(pv[i]) - same_frame[i]);
  pair_change /= double(pv.size());
  check::Require(pair_change > 1e-6, "changing the second image changes predicted motion");
  check::Require(before == deployed.Snapshot(), "pair inference never updates model values");
  const auto zero = ConstantFlow(split.test, {0, 0}), constant = ConstantFlow(split.test, mean);
  std::cout << " selected=" << selected << ",restored validation=" << best << " verified;fixed fitting " << initial << " -> " << learned << "\n";
  PrintFlowMetrics("learned", ScoreFlow(predicted, split.test.flow, split.test.valid));
  PrintFlowMetrics("zero flow", ScoreFlow(zero, split.test.flow, split.test.valid));
  PrintFlowMetrics("fit-only constant", ScoreFlow(constant, split.test.flow, split.test.valid));
  const auto    scene_ids = split.test.scene_ids.to_vector<int>();
  std::set<int> scenes(scene_ids.begin(), scene_ids.end());
  for (int scene : scenes) {
    std::vector<std::size_t> rows;
    for (std::size_t i = 0; i < scene_ids.size(); ++i)
      if (scene_ids[i] == scene) rows.push_back(i);
    const auto data = FlowBatch(split.test, rows);
    std::cout << " test scene_id=" << scene;
    PrintFlowMetrics("learned", ScoreFlow(FlowBatch(predicted, rows), data.flow, data.valid));
  }
  const auto warp = FlowWarp(split.test.second, predicted);
  FlowPreview(split.test, predicted, warp.value, directory / "comparison.png");
  nn::save_npz((directory / "test_outputs.npz").string(),
               {{"first", split.test.first}, {"second", split.test.second}, {"truth", split.test.flow}, {"valid", split.test.valid},
                {"prediction", predicted}, {"zero", zero}, {"fit_mean", constant}, {"warped_second", warp.value},
                {"warp_valid", warp.valid}, {"scene_ids", split.test.scene_ids}, {"family_ids", split.test.family_ids}});
  if (std::filesystem::exists(root / "manifest.json"))
    std::filesystem::copy_file(root / "manifest.json", directory / "data_manifest.json", std::filesystem::copy_options::overwrite_existing);
  check::Require(bool(history), "write flow history");
  std::cout << " restored pair-only parity=" << parity << ",second-image sensitivity=" << pair_change
            << "px,immutable parameters,single/batch agreement;checkpoint/history/all24 arrays/preview:" << directory.string() << " (training "
            << seconds << " s)\nComplete short " << name << " supervised flow demonstration passed;custom tiny split,not a flow benchmark result.\n";
}
}  // namespace flow



namespace nn = datamunge::nn;
using nn::Tensor;
using namespace flow;

namespace {

void CorrelationChecks() {
  for (int radius : {1, 2}) {
    constexpr int b = 2, h = 3, w = 4, c = 2;
    const int     costs = (2 * radius + 1) * (2 * radius + 1);
    std::vector<float> a(b * h * w * c), z(a.size()), up(std::size_t(b * h * w * costs));
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = float((int(i) % 13 - 6) * .17), z[i] = float((int(i) % 11 - 5) * .23);
    for (std::size_t i = 0; i < up.size(); ++i) up[i] = float((int(i) % 7 - 3) * .11);
    Tensor     first = nn::from_vector(a, {b, h, w, c}).requires_grad_(), second = nn::from_vector(z, {b, h, w, c}).requires_grad_();
    const auto output   = FlowCorrelation(first, second, radius);
    const auto gradient = nn::grad(nn::sum(output * nn::from_vector(up, {b, h, w, costs})), {first, second});
    const auto out = output.to_vector<double>(), ga = gradient[0].to_vector<double>(), gz = gradient[1].to_vector<double>();
    std::vector<double> da(a.size()), dz(z.size());
    double              value_error = 0, gradient_error = 0;
    for (int n = 0; n < b; ++n)
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
          for (int dy = -radius; dy <= radius; ++dy)
            for (int dx = -radius; dx <= radius; ++dx) {
              const int channel = (dy + radius) * (2 * radius + 1) + dx + radius;
              const int o       = ((n * h + y) * w + x) * costs + channel;
              double    value   = 0;
              if (y + dy >= 0 && y + dy < h && x + dx >= 0 && x + dx < w)
                for (int k = 0; k < c; ++k) {
                  const int i = ((n * h + y) * w + x) * c + k, j = ((n * h + y + dy) * w + x + dx) * c + k;
                  value += double(a[i]) * z[j] / c;
                  da[i] += up[o] * double(z[j]) / c;
                  dz[j] += up[o] * double(a[i]) / c;
                }
              value_error = std::max(value_error, std::abs(value - out[o]));
            }
    for (std::size_t i = 0; i < da.size(); ++i) {
      gradient_error = std::max(gradient_error, std::abs(da[i] - ga[i]));
      gradient_error = std::max(gradient_error, std::abs(dz[i] - gz[i]));
    }
    check::Near(value_error, 0, 3e-7, "every signed normalized correlation coordinate");
    check::Near(gradient_error, 0, 5e-7, "every correlation derivative in BOTH images");
    std::vector<float> ia(a.size(), 0.f), iz(z.size(), 0.f);
    ia[((0 * h + 1) * w + 1) * c] = 1;
    iz[((0 * h + 0) * w + 2) * c] = 1;
    const auto impulse = FlowCorrelation(nn::from_vector(ia, {b, h, w, c}), nn::from_vector(iz, {b, h, w, c}), radius).to_vector<double>();
    for (int k = 0; k < costs; ++k)
      check::Near(impulse[((0 * h + 1) * w + 1) * costs + k], k == (radius - 1) * (2 * radius + 1) + radius + 1 ? .5 : 0, 0,
                  "dy-major/dx-minor displaced impulse direction");
    std::cout << " radius" << radius << " cost volume every signed multi-batch/channel value/input gradient max errors=" << value_error << "/"
              << gradient_error << ",border zeros and displacement order passed\n";
  }
}

void WarpChecks() {
  for (bool strict : {false, true}) {
    constexpr int      b = 2, h = 3, w = 4, c = 2;
    std::vector<float> input(b * h * w * c), field(b * h * w * 2), up(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = float((int(i) % 17 - 8) * .13), up[i] = float((int(i) % 11 - 5) * .19);
    for (std::size_t i = 0; i < field.size() / 2; ++i)
      field[2 * i] = float((int(i) % 5 - 2) * .27 + .13), field[2 * i + 1] = float((int(i) % 7 - 3) * .19 + .07);
    Tensor     image = nn::from_vector(input, {b, h, w, c}).requires_grad_(), flow = nn::from_vector(field, {b, h, w, 2}).requires_grad_();
    const auto warp     = FlowWarp(image, flow, strict);
    const auto gradient = nn::grad(nn::sum(warp.value * nn::from_vector(up, {b, h, w, c})), {image, flow});
    const auto value = warp.value.to_vector<double>(), validv = warp.valid.to_vector<double>(), gi = gradient[0].to_vector<double>(),
               gf = gradient[1].to_vector<double>();
    std::vector<double> di(input.size()), df(field.size());
    double              value_error = 0, gradient_error = 0;
    for (int n = 0; n < b; ++n)
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          const int    pixel = (n * h + y) * w + x;
          const double yy = y + double(field[2 * pixel + 1]), xx = x + double(field[2 * pixel]);
          const int    y0 = int(std::floor(yy)), x0 = int(std::floor(xx));
          const double wy = yy - y0, wx = xx - x0;
          double       coverage = 0;
          for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx)
              if (y0 + dy >= 0 && y0 + dy < h && x0 + dx >= 0 && x0 + dx < w) coverage += (dy ? wy : 1 - wy) * (dx ? wx : 1 - wx);
          const bool valid = coverage >= .9999;
          check::Near(validv[pixel], valid, 0, "whole-window validity includes bilinear border coverage");
          for (int k = 0; k < c; ++k) {
            double v = 0;
            if (!strict || valid)
              for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx) {
                  if (y0 + dy < 0 || y0 + dy >= h || x0 + dx < 0 || x0 + dx >= w) continue;
                  const int    j = ((n * h + y0 + dy) * w + x0 + dx) * c + k;
                  const double a = dy ? wy : 1 - wy, z = dx ? wx : 1 - wx, g = up[pixel * c + k], sample = input[j];
                  v += a * z * sample;
                  di[j] += g * a * z;
                  df[2 * pixel] += g * a * (dx ? 1 : -1) * sample;
                  df[2 * pixel + 1] += g * z * (dy ? 1 : -1) * sample;
                }
            value_error = std::max(value_error, std::abs(v - value[pixel * c + k]));
          }
        }
    for (std::size_t i = 0; i < di.size(); ++i) gradient_error = std::max(gradient_error, std::abs(di[i] - gi[i]));
    for (std::size_t i = 0; i < df.size(); ++i) gradient_error = std::max(gradient_error, std::abs(df[i] - gf[i]));
    check::Near(value_error, 0, 5e-7, "every backward-warp value including partial borders");
    check::Near(gradient_error, 0, 8e-7, "every image/flow bilinear derivative and dx/dy convention");
    std::cout << " backward warp strict=" << strict << ":every multi-batch/channel image/dx/dy value/Jacobian max errors=" << value_error << "/"
              << gradient_error << ",fractional coordinates/border masks passed\n";
  }
}

void TransposeChecks() {
  std::vector<float> input(24), filter(64), up(96);
  for (int i = 0; i < 24; ++i) input[i] = float((i % 9 - 4) * .17);
  for (int i = 0; i < 64; ++i) filter[i] = float((i % 13 - 6) * .11);
  for (int i = 0; i < 96; ++i) up[i] = float((i % 7 - 3) * .07);
  const std::vector<float> offset = {-.2f, .3f};
  Tensor     x = nn::from_vector(input, {2, 2, 3, 2}).requires_grad_(), weights = nn::from_vector(filter, {4, 4, 2, 2}).requires_grad_(),
         bias  = nn::from_vector(offset).requires_grad_();
  const auto output    = nn::bias_add(FlowTranspose(x, weights, 2), bias);
  const auto gradients = nn::grad(nn::sum(output * nn::from_vector(up, {2, 4, 6, 2})), {x, weights, bias});
  const auto out       = output.to_vector<double>();
  std::vector<double> expected(96), dx(24), dw(64), db(2);
  for (int n = 0; n < 2; ++n)
    for (int y = 0; y < 4; ++y)
      for (int col = 0; col < 6; ++col)
        for (int c = 0; c < 2; ++c) {
          const int j = ((n * 4 + y) * 6 + col) * 2 + c;
          expected[j] = offset[c];
          db[c] += up[j];
        }
  for (int n = 0; n < 2; ++n)
    for (int y = 0; y < 2; ++y)
      for (int col = 0; col < 3; ++col)
        for (int i = 0; i < 2; ++i)
          for (int ky = 0; ky < 4; ++ky)
            for (int kx = 0; kx < 4; ++kx)
              for (int o = 0; o < 2; ++o) {
                const int oy = 2 * y + ky - 1, ox = 2 * col + kx - 1;
                if (oy < 0 || oy >= 4 || ox < 0 || ox >= 6) continue;
                const int j = ((n * 4 + oy) * 6 + ox) * 2 + o, a = ((n * 2 + y) * 3 + col) * 2 + i, z = ((ky * 4 + kx) * 2 + o) * 2 + i;
                expected[j] += double(input[a]) * filter[z];
                dx[a] += double(up[j]) * filter[z];
                dw[z] += double(up[j]) * input[a];
              }
  double value_error = 0, gradient_error = 0;
  for (std::size_t i = 0; i < expected.size(); ++i) value_error = std::max(value_error, std::abs(expected[i] - out[i]));
  const std::vector<double>* values[] = {&dx, &dw, &db};
  for (int k = 0; k < 3; ++k) {
    const auto g = gradients[std::size_t(k)].to_vector<double>();
    for (std::size_t i = 0; i < values[k]->size(); ++i) gradient_error = std::max(gradient_error, std::abs(values[k]->at(i) - g[i]));
  }
  check::Near(value_error, 0, 5e-7, "signed learned transpose matches symmetric pad1 coordinates");
  check::Near(gradient_error, 0, 8e-7, "every transpose input/filter/bias derivative");
  std::cout << " learned4x4 stride2/pad1 transpose values/input-kernel-bias Jacobian max errors=" << value_error << "/" << gradient_error << " passed\n";
}

void UnitsLossMetricChecks() {
  std::vector<float> input(24);
  for (int i = 0; i < 24; ++i) input[i] = i % 2 ? -3.f : 2.f;
  Tensor     x       = nn::from_vector(input, {2, 2, 3, 2}).requires_grad_();
  const auto values  = FlowResizeValues(x, 4, 9).to_vector<double>();
  const auto vectors = FlowResizeVectors(x, 2, 3, 4, 9);
  const auto pyramid = FlowPyramidPixels(x, 2).to_vector<double>();
  const auto derivative = nn::grad(nn::sum(vectors), {x})[0].to_vector<double>();
  const auto vv         = vectors.to_vector<double>();
  for (std::size_t i = 0; i < values.size(); ++i) {
    check::Near(values[i], i % 2 ? -3 : 2, 1e-6, "resizing input-unit fields does NOT multiply vectors");
    check::Near(vv[i], i % 2 ? -6 : 6, 1e-6, "grid-relative vectors scale width and height separately");
  }
  for (std::size_t i = 0; i < input.size(); ++i) {
    check::Near(pyramid[i], 5 * input[i], 0, "PWC level2 uses20/4 conversion");
    check::Near(derivative[i], i % 2 ? 12 : 18, 5e-6, "every anisotropic flow resize input derivative");
  }
  FlowData           data;
  std::vector<float> f(2 * 64 * 64 * 2), v(2 * 64 * 64, 1.f);
  for (std::size_t i = 0; i < f.size(); ++i) f[i] = i % 2 ? -8.f : 4.f;
  for (int y = 0; y < 16; ++y)
    for (int col = 0; col < 16; ++col) v[std::size_t((1 * 64 + y) * 64 + col)] = 0;
  data.flow          = nn::from_vector(f, {2, 64, 64, 2});
  data.valid         = nn::from_vector(v, {2, 64, 64, 1});
  const auto target  = PrepareFlowSupervision(data);
  for (std::size_t level = 0; level < 5; ++level) {
    const auto tv = target.valid[level].to_vector<float>(), tf = target.flow[level].to_vector<float>();
    for (std::size_t i = 0; i < tv.size(); ++i)
      for (int c = 0; c < 2; ++c)
        check::Near(tf[2 * i + std::size_t(c)], tv[i] ? (c ? -.4 : .2) : 0, 2e-8, "ALL pyramid targets remain input-pixel/20 units,invalid bins zero");
  }
  std::vector<float> truth(18, NAN), prediction(18, NAN), mask(9, 1.f);
  truth[0] = 0, truth[1] = 0, prediction[0] = 3, prediction[1] = 4, truth[2] = 0, truth[3] = 0, prediction[2] = -4, prediction[3] = 3;
  truth[6] = 0, truth[7] = 0, prediction[6] = 0, prediction[7] = 0;
  Tensor     p    = nn::from_vector(prediction, {3, 1, 3, 2}).requires_grad_();
  const auto loss = FlowEndpointLoss(p, nn::from_vector(truth, {3, 1, 3, 2}), nn::from_vector(mask, {3, 1, 3, 1}));
  const auto g    = nn::grad(loss, {p})[0].to_vector<double>();
  check::Near(loss.item(), (std::sqrt(25 + 1e-6) - .001) / 2, 2e-7, "smooth endpoint distance per-image mean,empty images excluded");
  for (int i = 0; i < 18; ++i) {
    const double expected = i < 4 ? prediction[i] / (4 * std::sqrt(25 + 1e-6)) : 0;
    check::Near(g[i], expected, 2e-8, "every smooth EPE gradient including NaN-invalid and perfect pixels");
  }
  const Tensor one = nn::ones({1, 1, 1, 1});
  auto         score = ScoreFlow(nn::zeros({1, 1, 1, 2}), nn::tensor({3, 4}, {1, 1, 1, 2}), one);
  check::Near(score.epe, 5, 0, "known3/4/5 endpoint error in input pixels");
  check::Near(score.aae, std::acos(1 / std::sqrt(26.)) * 180 / 3.14159265358979323846, 1e-12, "augmented-vector angular error degrees");
  check::Require(score.bad1 == 1 && score.bad3 == 1 && score.fl_all == 1, "known bad-pixel criteria");
  score = ScoreFlow(nn::tensor({3, 4}, {1, 1, 1, 2}), nn::tensor({3, 4}, {1, 1, 1, 2}), one);
  check::Near(score.epe, 0, 0, "perfect optical flow");
  check::Near(score.aae, 0, 1e-6, "perfect angular error");
  score = ScoreFlow(nn::tensor({3, 0}, {1, 1, 1, 2}), nn::tensor({0, 0}, {1, 1, 1, 2}), one);
  check::Require(score.bad3 == 0 && score.fl_all == 0, "strict3px outlier boundary");
  score = ScoreFlow(nn::tensor({105, 0}, {1, 1, 1, 2}), nn::tensor({100, 0}, {1, 1, 1, 2}), one);
  check::Require(score.bad3 == 1 && score.fl_all == 0, "strict5% relative outlier boundary");
  std::cout << " anisotropic grid/vector/input-unit scaling,all5 masked GT scales,safe smooth EPE/every-gradient,perfect and strict outlier/angular metric fixtures passed\n";
}

void ModelChecks() {
  for (int base : {64, 8}) {
    nn::ParameterStore p(163);
    std::vector<float> a(2 * 64 * 64 * 3), bv(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = float(.5 + .4 * std::sin(i * .17)), bv[i] = float(.5 + .4 * std::cos(i * .13));
    Tensor     first = nn::from_vector(a, {2, 64, 64, 3}).requires_grad_(), second = nn::from_vector(bv, {2, 64, 64, 3}).requires_grad_();
    const auto net = FlowNetS(p, first, second, 64, 64, base);
    Tensor     objective;
    for (const auto& level : net.pyramid) objective = objective.defined() ? objective + nn::mean(nn::square(level)) : nn::mean(nn::square(level));
    auto vars = p.parameters();
    vars.push_back(first);
    vars.push_back(second);
    const auto gradients = nn::grad(objective, vars);
    check::Require(p.count() == (base == 64 ? 38676506 : 615658) && p.size() == 42, "exact no-BN6-input FlowNetS parameter census");
    check::Require(net.stages.size() == 27, "all encoder/flow/decode/skip stages exposed");
    std::cout << " " << (base == 64 ? "full" : "compact") << " FlowNetS:" << p.count() << " parameters/all42 derivatives (executed),27 stages\n";
    if (base == 64) continue;
    std::vector<nn::Shape> expected;
    const int              widths[] = {8, 16, 32, 64, 64, 128};
    for (int level = 1; level <= 6; ++level) {
      const int side = 64 / (1 << level);
      expected.push_back({2, side, side, widths[level - 1]});
      if (level >= 3) expected.push_back({2, side, side, widths[level - 1]});
    }
    expected.push_back({2, 1, 1, 2});
    for (int level = 5; level >= 2; --level) {
      const int side = 64 / (1 << level), channels = 8 * (1 << (level - 2));
      expected.push_back({2, side, side, channels});
      expected.push_back({2, side, side, 2});
      expected.push_back({2, side, side, widths[level - 1] + channels + 2});
      expected.push_back({2, side, side, 2});
    }
    for (std::size_t i = 0; i < expected.size(); ++i)
      check::Require(net.stages[i].shape() == expected[i], "every exact encoder/decode/flow/skip batch-spatial-channel geometry");
    check::Require(net.flow.shape() == nn::Shape({2, 64, 64, 2}), "full input-pixel field output shape");
    double minimum = INFINITY, maximum = 0;
    for (const auto& g : gradients) {
      double norm = 0;
      for (float v : g.to_vector<float>()) {
        check::Require(std::isfinite(v), "finite FlowNetS parameter/image derivative");
        norm += double(v) * v;
      }
      norm = std::sqrt(norm);
      check::Require(norm > 0, "every compact model tensor and BOTH image inputs receive gradients");
      minimum = std::min(minimum, norm), maximum = std::max(maximum, norm);
    }
    std::cout << " compact all42 parameter and both-image gradients finite/nonzero,norm range[" << minimum << "," << maximum
              << "],all27 exact stages and input-pixel output passed\n";
  }
}

FlowResult Build(nn::ParameterStore& p, const Tensor& first, const Tensor& second) { return FlowNetS(p, first, second, 64, 64, 8); }

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs      = 3;
    bool                  checks_only = false;
    std::filesystem::path data        = paths::Dataset("optical_flow");
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("epochs=", 0) == 0)
        epochs = std::stoi(arg.substr(7));
      else if (arg.rfind("data=", 0) == 0)
        data = arg.substr(5);
      else if (arg == "checks_only=0" || arg == "checks_only=1")
        checks_only = arg == "checks_only=1";
      else if (arg.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(arg.substr(8)));
      else
        throw std::invalid_argument("expected epochs=N data=PATH checks_only=0|1 threads=N");
    }
    CorrelationChecks();
    WarpChecks();
    TransposeChecks();
    UnitsLossMetricChecks();
    ModelChecks();
    if (checks_only) {
      std::cout << "FlowNetS/motion-operator fixtures passed (training not requested).\n";
      return 0;
    }
    check::Require(std::filesystem::exists(data / "pairs.npz"), "true optical-flow pairs missing;run just optical-flow-assets first");
    TrainOpticalFlow(data, epochs, "flownets", Build, paths::OutputDir("models/flownets"));
    std::cout << "All FlowNetS checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
