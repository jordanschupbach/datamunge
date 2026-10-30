// CMD: just run-nn image_colorization
// Lightness-only Lab chroma regression with an encoder-decoder/global context.
// https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py
// D65/2degree Lab via inverse sRGB gamma and XYZ; L/50-1 input, tanh ab/128.
// Two stride2 encoders,three8px context convolutions including fusion,
// global-average context,spatial skips and half-pixel bilinear decoder.
// Predict only chroma,preserve given L,clip only decoded RGB for display/PSNR.
// Default3epochs,256fit/32validation/32official-test CIFAR32px,batch8,Adam.001.
// Select smallest validation chroma MSE; independently restore inference-only
// model,check output parity,batch independence and immutable parameters.
// Report ab MSE,DeltaE76,RGB PSNR,chroma and out-of-gamut fraction against
// grayscale and fit-only mean-chroma baselines; inspect8-row color preview.
// Regression is ambiguous and can desaturate; no Zhang classification,
// pretrained model,semantic-color recovery or benchmark accuracy claim.
// Optional accuracy gate requires5% lower chroma MSE than both baselines.
//
// Port of tflow's image_colorization. CIFAR-10 binaries are read from
// $DATAMUNGE_DATA_DIR/cifar10 (data_batch_1.bin, test_batch.bin); the Lab
// pairs and the batch order (nn::BatchOrder seed 37) are tflow's exactly;
// weights come from datamunge's generator, so trained numbers match TF
// statistically. TF's "deployment graph inspection" becomes a structural
// check on a second model built without an optimizer or target. The
// checkpoint is best.npz. Outputs go to output/nn/models/image_colorization.
// Options: epochs=3 data=PATH checks_only=0|1 accuracy=0|1 threads=N.

#include <cmath>
#include <iostream>

#include "common/colorization.hpp"
#include "common/paths.hpp"

// Experiment workflow from examples/nn/common/colorization.hpp.
namespace color {
using namespace ::color;
inline void TrainColorization(const std::filesystem::path& root, int epochs, bool accuracy, const std::filesystem::path& dir) {
  check::Require(epochs >= 2, "at least two epochs for colorization learning check");
  const auto training = data::LoadCifar({root / "data_batch_1.bin"}, data::Cifar::k10),
             testing  = data::LoadCifar({root / "test_batch.bin"}, data::Cifar::k10);
  check::Require(training.size() == 10000 && testing.size() == 10000, "original CIFAR binary splits");
  const auto fit = ColorPairs(training.images, ColorRows(0, 256)), val = ColorPairs(training.images, ColorRows(256, 288)),
             test = ColorPairs(testing.images, ColorRows(96, 128));
  const auto mean = ColorMeanChroma(fit);
  ColorModel model;
  std::filesystem::create_directories(dir);
  const auto    prefix = (dir / "best.npz").string();
  std::ofstream history(dir / "history.tsv");
  history << "epoch\ttraining_ab_mse\tvalidation_ab_mse\n";
  const auto   probe_l = ColorBatch(fit.lightness, ColorRows(0, 8)), probe_ab = ColorBatch(fit.chroma, ColorRows(0, 8));
  const double initial = model.Loss(probe_l, probe_ab);
  double       best = INFINITY;
  int          selected = -1;
  std::vector<double> losses;
  nn::BatchOrder      order(256, 8, 37);
  std::cout << "Colorization: " << model.parameters.count() << " parameters/" << model.parameters.size() << " tensors; " << epochs
            << " epochs,Adam.001,batch8,no augmentation/decay\n"
            << " fit first-train rows0..255,validation256..287,official-test96..127;original32px;D65/2deg Lab L only -> ab/128 MSE\n"
            << " fixed fitting initial=" << initial << ", fit-only mean ab=" << 128 * mean[0] << "," << 128 * mean[1] << std::endl;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    double      sum  = 0;
    std::size_t seen = 0;
    for (;;) {
      const auto ids = order.next();
      if (ids.empty()) break;
      const std::vector<std::size_t> rows(ids.begin(), ids.end());
      sum += double(rows.size()) * model.Loss(ColorBatch(fit.lightness, rows), ColorBatch(fit.chroma, rows), true);
      seen += rows.size();
    }
    check::Require(seen == 256, "all fitting images used once per colorization epoch");
    losses.push_back(sum / double(seen));
    const double value = ColorLoss(model, val);
    if (value < best) {
      best = value, selected = epoch;
      model.Save(prefix);
    }
    history << epoch << '\t' << losses.back() << '\t' << value << '\n';
    history.flush();
    std::cout << " epoch " << epoch << ": ab MSE=" << losses.back() << ", validation=" << value << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  model.Restore(prefix);
  check::Near(ColorLoss(model, val), best, 1e-8, "restored colorization validation agreement");
  const double learned = model.Loss(probe_l, probe_ab);
  check::Require(learned < initial * (1 - 1e-4) && losses.back() < losses.front() * (1 - 1e-4), "fixed-fitting and epoch-average colorization learning");
  const auto prediction = ColorPredict(model, test);
  // Deployment model: one lightness input, model parameters and checkpoint IO only.
  ColorModel deployed(false);
  deployed.Restore(prefix);
  check::Require(!deployed.optimizer && deployed.parameters.size() == 20, "one grayscale input plus checkpoint filename only; no optimizer");
  const auto snapshot    = deployed.Snapshot();
  const auto independent = ColorPredict(deployed, test), repeat = ColorPredict(deployed, test);
  const auto pv = prediction.to_vector<float>(), iv = independent.to_vector<float>(), rv = repeat.to_vector<float>();
  double     error = 0;
  for (std::size_t i = 0; i < pv.size(); ++i) {
    error = std::max(error, std::abs(double(pv[i]) - iv[i]));
    check::Near(rv[i], iv[i], 0, "repeat deterministic inference");
  }
  check::Near(error, 0, 1e-7, "standalone checkpoint output agreement");
  check::Require(snapshot == deployed.Snapshot(), "inference changes no model values");
  const auto single = deployed.Predict(ColorBatch(test.lightness, {3})).to_vector<float>();
  for (int i = 0; i < 2048; ++i) check::Near(single[std::size_t(i)], pv[3 * 2048 + std::size_t(i)], 1e-6, "batch-independent grayscale inference");
  double variation = 0;
  for (int i = 0; i < 2048; ++i) variation += std::abs(double(pv[std::size_t(i)]) - pv[2048 + std::size_t(i)]);
  variation /= 2048;
  check::Require(variation > 1e-6, "learned chroma responds to distinct luminance inputs");
  const auto metrics = ColorScore(test, prediction), gray = ColorScore(test, ColorConstantChroma(test, {0, 0})),
             constant = ColorScore(test, ColorConstantChroma(test, mean));
  auto print = [&](const char* name, const ColorMetrics& x) {
    std::cout << " " << name << ": normalized ab MSE=" << x.ab_mse << ",mean DeltaE76=" << x.delta_e76 << ",mean RGB PSNR=" << x.rgb_psnr
              << "dB,mean chroma=" << x.chroma << ",out-of-gamut pixels=" << 100 * x.clipped_pixels << "%\n";
  };
  std::cout << " selected=" << selected << ",restored validation=" << best << " verified;fixed fitting " << initial << " -> " << learned << "\n";
  print("learned", metrics);
  print("grayscale", gray);
  print("fit-only mean chroma", constant);
  std::cout << " target-free restore max error=" << error << ",cross-input chroma difference=" << variation
            << ";single/batch parity and immutable model passed\n";
  ColorPreview(test, metrics.rgb, gray.rgb, constant.rgb, dir / "comparison.png");
  nn::save_npz((dir / "test_outputs.npz").string(), {{"lightness_normalized", test.lightness},
                                                       {"target_ab_normalized", test.chroma},
                                                       {"prediction_ab_normalized", prediction},
                                                       {"rgb_target", test.rgb},
                                                       {"rgb_prediction", metrics.rgb},
                                                       {"rgb_grayscale", gray.rgb},
                                                       {"rgb_fit_mean", constant.rgb}});
  check::Require(bool(history), "write colorization history");
  if (accuracy) check::Require(metrics.ab_mse < .95 * std::min(gray.ab_mse, constant.ab_mse), "optional 5% chroma-MSE gain over both baselines");
  std::cout << " checkpoint/history/all32 predictions/8-row grayscale-mean-learned-truth preview: " << dir.string() << " (training " << seconds
            << " s)\nShort image-colorization demonstration passed; no plausible-color or paper-accuracy claim.\n";
}
}  // namespace color



namespace nn = datamunge::nn;
using nn::Tensor;
using namespace color;

namespace {

void ColorScienceChecks() {
  const std::array<ColorTriple, 6> rgb = {{{0, 0, 0}, {1, 1, 1}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {.5, .5, .5}}};
  // Independent known D65/2deg fixtures for the published scikit-image matrix.
  const std::array<ColorTriple, 6> expected = {{{0, 0, 0},
                                                {100, -.00245494, .00465342},
                                                {53.24058794, 80.09230823, 67.20275104},
                                                {87.73509949, -86.18302974, 83.17970318},
                                                {32.29567257, 79.18559091, -107.85730021},
                                                {53.38896474, -.00146850, .00278359}}};
  for (std::size_t i = 0; i < rgb.size(); ++i) {
    const auto lab = ColorRGBToLab(rgb[i]), back = ColorLabToRGB(lab);
    for (int c = 0; c < 3; ++c) {
      check::Near(lab[c], expected[i][c], 1e-5, "known independent primary/gray/white/black Lab coordinate");
      check::Near(back[c], rgb[i][c], 1e-12, "primary RGB roundtrip");
    }
  }
  double error = 0;
  for (int r = 0; r <= 12; ++r)
    for (int g = 0; g <= 12; ++g)
      for (int b = 0; b <= 12; ++b) {
        const ColorTriple value = {r / 12., g / 12., b / 12.};
        const auto        back  = ColorLabToRGB(ColorRGBToLab(value));
        for (int c = 0; c < 3; ++c) error = std::max(error, std::abs(value[c] - back[c]));
      }
  check::Near(error, 0, 1e-12, "2197-color gamma/matrix/Lab roundtrip");
  for (double value : {0., .001, .0031307, .0031309, .02, .2, 1.}) check::Near(ColorLinear(ColorGamma(value)), value, 1e-12, "both sRGB transfer branches");
  constexpr double delta = 6. / 29;
  for (double value : {0., .0001, delta * delta * delta - 1e-8, delta * delta * delta + 1e-8, .5, 1.})
    check::Near(ColorLabInverse(ColorLabF(value)), value, 1e-14, "both exact Lab branches and knee");
  const auto outside = ColorLabToRGB({50, 128, -128});
  bool       clipped = false;
  for (double c : outside) clipped |= c < 0 || c > 1;
  check::Require(clipped, "out-of-gamut chroma remains visible before display clipping");
  check::Throws([] { ColorRGBToLab({NAN, 0, 0}); }, "reject nonfinite input color");
  std::cout << " known D65 Lab primaries/gray/white/black,2197-color roundtrip max error=" << error
            << ",piecewise knees and raw gamut checks passed\n";
}

void DataMetricChecks() {
  std::vector<std::int32_t> raw(4 * 3072);
  for (int i = 0; i < int(raw.size()); ++i) raw[std::size_t(i)] = (i * 13 + i / 3072 * 37) % 256;
  const Tensor source = nn::from_vector(raw, {4, 32, 32, 3});
  const auto   all = ColorPairs(source, {0, 1, 2, 3}), pair = ColorPairs(source, {3, 1});
  check::Require(pair.lightness.shape() == nn::Shape({2, 32, 32, 1}) && pair.chroma.shape() == nn::Shape({2, 32, 32, 2}),
                 "true one-channel input and two-channel target");
  const auto pl = pair.lightness.to_vector<float>(), al = all.lightness.to_vector<float>(), pc = pair.chroma.to_vector<float>(),
             ac = all.chroma.to_vector<float>();
  for (int n = 0; n < 2; ++n) {
    const int row = n ? 1 : 3;
    for (int i = 0; i < 1024; ++i) {
      check::Near(pl[std::size_t(n * 1024 + i)], al[std::size_t(row * 1024 + i)], 0, "reordered luminance pairing");
      for (int c = 0; c < 2; ++c)
        check::Near(pc[std::size_t((n * 1024 + i) * 2 + c)], ac[std::size_t((row * 1024 + i) * 2 + c)], 0, "reordered signed chroma pairing");
    }
  }
  const auto perfect = ColorScore(all, all.chroma);
  check::Near(perfect.ab_mse, 0, 0, "perfect normalized ab MSE");
  check::Near(perfect.delta_e76, 0, 0, "perfect DeltaE76");
  double     rgb_error = 0;
  const auto ar = all.rgb.to_vector<float>(), prgb = perfect.rgb.to_vector<float>();
  for (std::size_t i = 0; i < ar.size(); ++i) rgb_error = std::max(rgb_error, std::abs(double(ar[i]) - prgb[i]));
  check::Near(rgb_error, 0, 3e-6, "float32 paired RGB reconstruction");
  // Two independent one-pixel images isolate metric normalizations and mean PSNR.
  const auto         neutral = ColorLabToRGB({50, 0, 0});
  std::vector<float> trgb(6);
  for (int n = 0; n < 2; ++n)
    for (int c = 0; c < 3; ++c) trgb[std::size_t(3 * n + c)] = float(neutral[c] + (n ? .2 : .1));
  ColorData tiny{nn::zeros({2, 1, 1, 1}), nn::from_vector(std::vector<float>{.125f, 0, 0, .25f}, {2, 1, 1, 2}), nn::from_vector(trgb, {2, 1, 1, 3})};
  const auto gray = ColorScore(tiny, ColorConstantChroma(tiny, {0, 0}));
  check::Near(gray.ab_mse, 1280. / (4 * 128 * 128), 0, "mean over pixels AND two ab channels");
  check::Near(gray.delta_e76, 24, 0, "mean Euclidean Lab distance with preserved L");
  check::Near(gray.rgb_psnr, (20 - 20 * std::log10(.2)) / 2, 2e-6, "mean per-image RGB PSNR versus pooled MSE");
  const auto mean = ColorMeanChroma(tiny);
  check::Near(mean[0], .0625, 0, "fit-only a average");
  check::Near(mean[1], .125, 0, "fit-only b average");
  const auto rgb = ColorLabToRGB({50, 0, 0}), lab = ColorRGBToLab({std::clamp(rgb[0], 0., 1.), std::clamp(rgb[1], 0., 1.), std::clamp(rgb[2], 0., 1.)});
  check::Near(lab[0], 50, 1e-12, "neutral reconstruction preserves supplied lightness");
  check::Near(lab[1], 0, 1e-12, "neutral a");
  check::Near(lab[2], 0, 1e-12, "neutral b");
  std::cout << " signed L/ab pairing/reordered batches,perfect reconstruction,fit-only mean,DeltaE76/MSE/mean-PSNR and lightness-preservation fixtures passed\n";
}

void LossAndResizeChecks() {
  std::vector<float> values(12), target(12);
  for (int i = 0; i < 12; ++i) values[std::size_t(i)] = float((i - 5) * .13), target[std::size_t(i)] = float((i % 3 - 1) * .2);
  Tensor       raw  = nn::from_vector(values, {2, 1, 3, 2}).requires_grad_();
  const Tensor pred = nn::tanh(raw);
  const Tensor loss = ColorizationLoss(pred, nn::from_vector(target, {2, 1, 3, 2}));
  const auto   gradient = nn::grad(loss, {raw})[0].to_vector<float>();
  const auto   pv       = pred.to_vector<float>();
  double       host = 0, error = 0;
  for (int i = 0; i < 12; ++i) {
    const double t = std::tanh(values[std::size_t(i)]), difference = t - target[std::size_t(i)];
    host += difference * difference / 12;
    const double derivative = 2 * difference * (1 - t * t) / 12;
    error                   = std::max(error, std::abs(gradient[std::size_t(i)] - derivative));
    check::Near(pv[std::size_t(i)], t, 1e-7, "bounded signed tanh chroma");
  }
  check::Near(loss.item(), host, 1e-7, "independent two-channel MSE");
  check::Near(error, 0, 1e-7, "every signed MSE/tanh coordinate derivative");
  Tensor     small = nn::from_vector(std::vector<float>{-.3f, .8f, .2f, -.7f}, {2, 1, 1, 2}).requires_grad_();
  const auto up    = ColorResize(small, nn::zeros({2, 3, 5, 1}));
  const auto dup   = nn::grad(nn::sum(up), {small})[0].to_vector<float>();
  const auto uv = up.to_vector<float>(), sv = small.to_vector<float>();
  for (int n = 0; n < 2; ++n)
    for (int y = 0; y < 3; ++y)
      for (int x = 0; x < 5; ++x)
        for (int c = 0; c < 2; ++c)
          check::Near(uv[std::size_t(((n * 3 + y) * 5 + x) * 2 + c)], sv[std::size_t(n * 2 + c)], 1e-7,
                      "global context broadcasts without batch/channel mixing");
  for (int i = 0; i < 4; ++i) check::Near(dup[std::size_t(i)], 15, 5e-6, "every global-context derivative equals spatial mass");
  // Identical grayscale observations with opposite colors cannot have a unique inverse.
  Tensor       neutral  = nn::zeros({1, 1, 1, 2}).requires_grad_();
  const Tensor amb_loss = ColorizationLoss(nn::tile(neutral, {2, 1, 1, 1}), nn::from_vector(std::vector<float>{.25f, -.5f, -.25f, .5f}, {2, 1, 1, 2}));
  const auto   amb      = nn::grad(amb_loss, {neutral})[0].to_vector<float>();
  check::Near(amb_loss.item(), .15625, 0, "irreducible opposite-color error at conditional mean");
  for (int i = 0; i < 2; ++i) check::Near(amb[std::size_t(i)], 0, 0, "opposite colors cancel regression gradient");
  std::cout << " independent tanh/MSE every-coordinate derivative max error=" << error
            << ",global context values/Jacobian and ambiguous-color conditional mean passed\n";
}

void SpatialResizeChecks() {
  std::vector<float> values(24), upstream(60);
  for (int i = 0; i < 24; ++i) values[std::size_t(i)] = float((i - 10) * .13);
  for (int i = 0; i < 60; ++i) upstream[std::size_t(i)] = float((i % 11 - 5) * .17);
  Tensor     input    = nn::from_vector(values, {2, 2, 3, 2}).requires_grad_();
  const auto output   = ColorResize(input, nn::zeros({2, 3, 5, 1}));
  const auto gradient = nn::grad(nn::sum(output * nn::from_vector(upstream, {2, 3, 5, 2})), {input})[0].to_vector<float>();
  const auto ov       = output.to_vector<float>();
  std::vector<double> expected_gradient(24);
  double              value_error = 0, gradient_error = 0;
  for (int n = 0; n < 2; ++n)
    for (int y = 0; y < 3; ++y)
      for (int x = 0; x < 5; ++x)
        for (int c = 0; c < 2; ++c) {
          const double iy = std::clamp((y + .5) * 2 / 3 - .5, 0., 1.), ix = std::clamp((x + .5) * 3 / 5 - .5, 0., 2.);
          const int    y0 = int(std::floor(iy)), x0 = int(std::floor(ix));
          double       expected = 0;
          const int    q        = ((n * 3 + y) * 5 + x) * 2 + c;
          for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) {
              const double weight = (dy ? iy - y0 : 1 - iy + y0) * (dx ? ix - x0 : 1 - ix + x0);
              const int    index  = ((n * 2 + std::min(y0 + dy, 1)) * 3 + std::min(x0 + dx, 2)) * 2 + c;
              expected += weight * values[std::size_t(index)];
              expected_gradient[std::size_t(index)] += weight * upstream[std::size_t(q)];
            }
          value_error = std::max(value_error, std::abs(expected - ov[std::size_t(q)]));
        }
  for (int i = 0; i < 24; ++i) gradient_error = std::max(gradient_error, std::abs(expected_gradient[std::size_t(i)] - gradient[std::size_t(i)]));
  check::Near(value_error, 0, 1e-6, "every signed half-pixel interpolation value");
  check::Near(gradient_error, 0, 1e-6, "every independently scattered bilinear input derivative");
  std::cout << " half-pixel bilinear signed multi-batch/channel spatial values/Jacobians max errors=" << value_error << "/" << gradient_error
            << " passed\n";
}

void ModelChecks() {
  nn::ParameterStore p(31);
  std::vector<float> input(2 * 33 * 35), target(2 * 33 * 35 * 2);
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(.8 * std::sin(double(i) * .17));
  for (std::size_t i = 0; i < target.size(); ++i) target[i] = float(.3 * std::cos(double(i) * .13));
  const Tensor l       = nn::from_vector(input, {2, 33, 35, 1});
  const auto   network = Colorization(p, l);
  const Tensor loss    = ColorizationLoss(network.chroma, nn::from_vector(target, {2, 33, 35, 2}));
  const auto   gradient = nn::grad(loss, p.parameters());
  check::Require(p.count() == 140082 && p.size() == 20, "exact colorization parameter census");
  const std::vector<nn::Shape> expected = {{2, 33, 35, 16}, {2, 17, 18, 32}, {2, 9, 9, 64},   {2, 9, 9, 64},   {2, 9, 9, 64},
                                           {2, 1, 1, 32},   {2, 9, 9, 64},   {2, 17, 18, 32}, {2, 33, 35, 16}, {2, 33, 35, 2}};
  check::Require(network.stages.size() == expected.size(), "all encoder/decoder stages exposed");
  for (std::size_t i = 0; i < expected.size(); ++i) check::Require(network.stages[i].shape() == expected[i], "exact odd-rectangular batch/spatial/channel shape");
  double minimum = INFINITY, maximum = 0;
  for (const auto& g : gradient) {
    double norm = 0;
    for (float v : g.to_vector<float>()) {
      check::Require(std::isfinite(v), "finite colorization parameter derivative");
      norm += double(v) * v;
    }
    norm = std::sqrt(norm);
    check::Require(norm > 0, "every colorization tensor receives gradient");
    minimum = std::min(minimum, norm), maximum = std::max(maximum, norm);
  }
  {
    nn::NoGradGuard guard;
    for (std::size_t i = 0; i < p.size(); ++i) p.set(p.names()[i], nn::zeros(p.parameters()[i].shape()));
    for (float v : Colorization(p, l).chroma.to_vector<float>()) check::Near(v, 0, 0, "zero-model has no hidden color shortcut");
  }
  std::cout << " colorizer140082 parameters/all20 finite nonzero derivative norms[" << minimum << "," << maximum
            << "],10 exact odd-rectangular stage shapes and neutral zero-model passed\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    bool                  checks_only = false, accuracy = false;
    std::filesystem::path data = paths::Dataset("cifar10");
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("epochs=", 0) == 0)
        epochs = std::stoi(arg.substr(7));
      else if (arg.rfind("data=", 0) == 0)
        data = arg.substr(5);
      else if (arg == "checks_only=0" || arg == "checks_only=1")
        checks_only = arg == "checks_only=1";
      else if (arg == "accuracy=0" || arg == "accuracy=1")
        accuracy = arg == "accuracy=1";
      else if (arg.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(arg.substr(8)));
      else
        throw std::invalid_argument("expected epochs=N data=PATH checks_only=0|1 accuracy=0|1 threads=N");
    }
    ColorScienceChecks();
    DataMetricChecks();
    LossAndResizeChecks();
    SpatialResizeChecks();
    ModelChecks();
    if (checks_only) {
      std::cout << "Colorization fixtures passed (training not requested).\n";
      return 0;
    }
    TrainColorization(data, epochs, accuracy, paths::OutputDir("models/image_colorization"));
    std::cout << "All image-colorization checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
