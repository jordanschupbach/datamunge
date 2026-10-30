// CMD: just run-nn partial_inpainting
// Irregular-hole inpainting with a genuine partial-convolution U-Net/full loss.
// https://arxiv.org/abs/1804.07723 https://github.com/NVIDIA/partialconv
// Full8enc/8dec model,compact5enc/5dec width8 training at original32px.
// Every layer masks,renormalizes and updates validity;nearest decoding
// concatenates BOTH features and masks,including original image at finalskip.
// BN exceptfirst/final,ReLU encoder,LeakyReLU decoder,linear raw RGB.
// TensorFlow SAME stride alignment is an explicit padding adaptation.
// Loss valid+6hole+.05percept+120(style_raw+style_comp)+.1TV,full-image
// pixel normalization,raw+composite genuine frozen VGG16pool1/2/3 features.
// Keras ImageNet BGR255 preprocessing,feature activations divided by255;
// document these feature units rather than claim Torch/pretrained parity.
// Default3epochs,32fit/8val/16test,batch2,Adam.0002,fixedirregular masks.
// Independent pixel/TV/Gram/Jacobians,Keras every-image-gradient parity,
// full/compact all gradients/mask and skip geometry,empty/allvalid tests.
// Restore standalone image+mask-only model/BN,immutable inference/hidden-fill
// independence;report raw/clipped hole PSNR versus visible-only mean fill.
//
// Port of tflow's partial_inpainting. Assets come from tflow's
// `just style-assets; just johnson-assets; just partial-inpainting-assets` in
// $DATAMUNGE_DATA_DIR/style_transfer (vgg16.npz, partial_reference.npz Keras
// fixture) and CIFAR-10 binaries from $DATAMUNGE_DATA_DIR/cifar10. Images,
// irregular masks (tflow's mt19937 draws) and the batch order (nn::BatchOrder
// seed 127) are tflow's exactly; U-Net weights come from datamunge's
// generator, so trained numbers match TF statistically. TF builds the full
// 256 px model as a graph only; here it also runs forward/backward once on a
// 1x256x256 batch (TF's graph uses batch 4; the census is batch-independent).
// The "deployment graph inspection" becomes a structural check of a model
// built without VGG or optimizer. The checkpoint is best.npz (weights + moving
// BN). Outputs go to output/nn/models/partial_inpainting.
// Options: epochs=3 data=PATH assets=PATH checks_only=0|1 threads=N.

#include <datamunge/datasets/assets.hpp>
#include <cmath>
#include <iostream>

#include "common/partial_inpainting.hpp"
#include "common/paths.hpp"

// Experiment workflow from examples/nn/common/partial_inpainting.hpp.
namespace inpaint {
using namespace ::inpaint;
inline void TrainPartialInpainting(const std::filesystem::path& root, const std::filesystem::path& assets, const style::TensorMap& weights, int epochs,
                                   const std::filesystem::path& directory) {
  check::Require(epochs >= 2, "at least two partial epochs for learning check");
  const auto source = data::LoadCifar({root / "data_batch_1.bin"}, data::Cifar::k10), testing = data::LoadCifar({root / "test_batch.bin"}, data::Cifar::k10);
  check::Require(source.size() == 10000 && testing.size() == 10000, "original CIFAR inpainting splits");
  PartialModel model(&weights);
  const auto   fit = model.Prepare(PartialPairs(source.images, PartialRows(0, 32), 107)),
             validation = model.Prepare(PartialPairs(source.images, PartialRows(32, 40), 109));
  const auto test  = PartialPairs(testing.images, PartialRows(192, 208), 113);
  const auto probe = PartialBatch(fit, {0, 1});
  const auto initial = model.Loss(probe, true);
  std::filesystem::create_directories(directory);
  const auto    prefix = (directory / "best.npz").string();
  std::ofstream history(directory / "history.tsv");
  history << "epoch\ttotal\tvalid\thole\tperceptual\tstyle_raw\tstyle_comp\ttv\tvalidation\n";
  double              best     = INFINITY;
  int                 selected = -1;
  std::vector<double> totals;
  nn::BatchOrder      order(32, 2, 127);
  std::cout << "Partial UNet: " << model.parameters.count() << " parameters/" << model.parameters.size() << " tensors," << model.state.count()
            << " BN values;5enc/5dec,width8..64\n"
            << " " << epochs << " epochs,32fit/8validation/16test CIFAR32,fit0..31,val32..39,test192..207;mask seeds107/109/113,12..55% holes\n"
            << " Adam.0002,batch2,no augmentation/decay;genuine frozen Keras VGG16 pool1/2/3 features /255\n"
            << " full objective valid+6hole+.05percept+120(style_raw+style_comp)+.1TV;fixed fitting initial=" << initial[0] << std::endl;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    std::array<double, 7> mean{};
    std::size_t           seen = 0;
    for (;;) {
      const auto ids = order.next();
      if (ids.empty()) break;
      const std::vector<std::size_t> rows(ids.begin(), ids.end());
      const auto                     values = model.Loss(PartialBatch(fit, rows), true, true);
      for (std::size_t i = 0; i < 7; ++i) mean[i] += double(rows.size()) * values[i];
      seen += rows.size();
    }
    check::Require(seen == 32, "all partial fitting images/masks visited each epoch");
    for (auto& value : mean) value /= double(seen);
    totals.push_back(mean[0]);
    const double value = PartialValidation(model, validation);
    if (value < best) {
      best = value, selected = epoch;
      model.Save(prefix);
    }
    history << epoch;
    for (double component : mean) history << '\t' << component;
    history << '\t' << value << '\n';
    history.flush();
    std::cout << " epoch " << epoch << ": components[total,valid,hole,percept,style_raw,style_comp,TV]=";
    for (double component : mean) std::cout << component << ",";
    std::cout << "validation=" << value << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  model.Restore(prefix);
  check::Near(PartialValidation(model, validation), best, 1e-5, "restored full partial validation agreement");
  const auto learned = model.Loss(probe, true);
  check::Require(learned[0] < initial[0] * (1 - 1e-4) && totals.back() < totals.front() * (1 - 1e-4), "fixed and mean full-objective partial training learning");
  model.vgg->CheckFrozen();
  PartialModel deployed;
  deployed.Restore(prefix);
  check::Require(!deployed.vgg && !deployed.optimizer, "deployment contains no perceptual network/optimizer; only masked image and mask inputs");
  const auto before = deployed.Snapshot();
  const auto output = PartialPredict(deployed, test), reference = PartialPredict(model, test), repeat = PartialPredict(deployed, test);
  const double parity=nn::max(nn::abs(output.first-reference.first)).item();check::Near(parity,0,1e-6,"restored standalone partial output parity");check::Require(nn::all(nn::equal(output.first,repeat.first)).item<bool>(),"deterministic partial inference");
  const auto single=deployed.Predict(PartialBatch(test,{3})).first;const double batch_error=nn::max(nn::abs(single-nn::slice_axis(output.first,0,3,4))).item();std::cout<<" partial single/batch maximum absolute difference="<<batch_error<<" (Float32 tolerance1e-5)\n";check::Near(batch_error,0,1e-5,"partial single/batch parity");
  auto hidden=PartialBatch(test,{0,1});const auto positions=nn::arange(0,hidden.input.numel(),1,nn::DType::Float64).reshape(hidden.input.shape());const auto fill=(1000.+positions-7.*nn::floor(positions/7.)).to(hidden.input.dtype());hidden.input=nn::where(nn::greater(hidden.mask,0.),hidden.input,fill);const auto invariant=deployed.Predict(hidden).first;check::Near(nn::max(nn::abs(invariant-nn::slice_axis(output.first,0,0,2))).item(),0,0,"whole UNet independent of finite hidden fill");
  check::Require(before == deployed.Snapshot(), "inference leaves all model/BN bytes unchanged");
  const auto mean  = PartialMean(test);
  const auto score = PartialScore(test, output.first), clipped = PartialScore(test, output.first, true), baseline = PartialScore(test, mean);
  check::Near(score.known_error, 0, 0, "composite preserves every observed pixel exactly");
  std::cout << " selected=" << selected << ",restored validation=" << best << " verified;fixed fitting total " << initial[0] << " -> " << learned[0] << "\n"
            << " all16 test:raw hole MSE=" << score.hole_mse << ",mean hole PSNR=" << score.hole_psnr << "dB,full PSNR=" << score.full_psnr
            << "dB;clipped hole PSNR=" << clipped.hole_psnr << "dB;visible-mean hole PSNR=" << baseline.hole_psnr << "dB\n"
            << " known pixel max error=" << score.known_error << ",standalone parity=" << parity
            << ";all26 VGG tensors unchanged,immutable BN,single/batch and hidden-fill independence passed\n";
  PartialPreview(test, output.first, mean, directory / "comparison.png");
  nn::save_npz((directory / "test_outputs.npz").string(), {{"truth", test.truth},
                                                             {"input", test.input},
                                                             {"mask", test.mask},
                                                             {"composite", output.first},
                                                             {"raw", output.second},
                                                             {"visible_mean", mean}});
  if (std::filesystem::exists(assets / "partial_manifest.json"))
    std::filesystem::copy_file(assets / "partial_manifest.json", directory / "assets_manifest.json", std::filesystem::copy_options::overwrite_existing);
  check::Require(bool(history), "write partial inpainting history");
  std::cout << " model/BN checkpoint,history,all16 arrays,preview:" << directory.string() << " (training " << seconds
            << " s)\nComplete short full-objective partial-convolution inpainting demonstration passed.\n";
}
}  // namespace inpaint



namespace nn = datamunge::nn;
using nn::Tensor;
using namespace inpaint;

namespace {

double Sign(double x) { return (x > 0) - (x < 0); }

struct ToyResult {
  std::array<double, 7> loss{};
  std::vector<double>   pixel, raw_feature, comp_feature;
};

ToyResult HostObjective(const std::vector<float>& raw, const std::vector<float>& truth, const std::vector<float>& mask, const nn::Shape& image_shape,
                        const std::vector<float>& rf, const std::vector<float>& cf, const std::vector<float>& target, const nn::Shape& feature_shape) {
  const int batch = int(image_shape[0]), height = int(image_shape[1]), width = int(image_shape[2]), channels = int(image_shape[3]),
            total = int(raw.size());
  ToyResult out;
  out.pixel.resize(raw.size());
  out.raw_feature.resize(rf.size());
  out.comp_feature.resize(cf.size());
  std::vector<double> composite(raw.size());
  std::vector<int>    region(raw.size());
  for (int i = 0; i < total; ++i) {
    const double valid = mask[std::size_t(i)], difference = double(raw[std::size_t(i)]) - truth[std::size_t(i)];
    out.loss[1] += valid * std::abs(difference) / total;
    out.loss[2] += (1 - valid) * std::abs(difference) / total;
    out.pixel[std::size_t(i)] = (valid + 6 * (1 - valid)) * Sign(difference) / total;
    composite[std::size_t(i)] = valid != 0 ? truth[std::size_t(i)] : raw[std::size_t(i)];
  }
  auto index = [&](int n, int y, int x, int c) { return std::size_t(((n * height + y) * width + x) * channels + c); };
  for (int n = 0; n < batch; ++n)
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x)
        for (int c = 0; c < channels; ++c)
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
              if (y + dy >= 0 && y + dy < height && x + dx >= 0 && x + dx < width && mask[index(n, y + dy, x + dx, c)] == 0) region[index(n, y, x, c)] = 1;
  auto edge = [&](std::size_t a, std::size_t b) {
    if (!region[a] || !region[b]) return;
    const double difference = composite[a] - composite[b], derivative = .1 * Sign(difference) / total;
    out.loss[6] += std::abs(difference) / total;
    if (mask[a] == 0) out.pixel[a] += derivative;
    if (mask[b] == 0) out.pixel[b] -= derivative;
  };
  for (int n = 0; n < batch; ++n)
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x)
        for (int c = 0; c < channels; ++c) {
          if (x + 1 < width) edge(index(n, y, x, c), index(n, y, x + 1, c));
          if (y + 1 < height) edge(index(n, y, x, c), index(n, y + 1, x, c));
        }
  const int b = int(feature_shape[0]), pixels = int(feature_shape[1] * feature_shape[2]), ch = int(feature_shape[3]), elements = int(rf.size());
  for (int level = 1; level <= 3; ++level) {
    for (int i = 0; i < elements; ++i) {
      const double dr = double(rf[std::size_t(i)]) - target[std::size_t(i)], dc = double(cf[std::size_t(i)]) - target[std::size_t(i)];
      out.loss[3] += level * (std::abs(dr) + std::abs(dc)) / elements;
      out.raw_feature[std::size_t(i)] += .05 * level * Sign(dr) / elements;
      out.comp_feature[std::size_t(i)] += .05 * level * Sign(dc) / elements;
    }
    for (int which = 0; which < 2; ++which) {
      const auto& f        = which ? cf : rf;
      auto&       gradient = which ? out.comp_feature : out.raw_feature;
      for (int n = 0; n < b; ++n) {
        std::vector<double> signs(std::size_t(ch * ch));
        for (int i = 0; i < ch; ++i)
          for (int j = 0; j < ch; ++j) {
            double actual = 0, wanted = 0;
            for (int k = 0; k < pixels; ++k) {
              actual += double(f[std::size_t((n * pixels + k) * ch + i)]) * f[std::size_t((n * pixels + k) * ch + j)];
              wanted += double(target[std::size_t((n * pixels + k) * ch + i)]) * target[std::size_t((n * pixels + k) * ch + j)];
            }
            const double difference = level * level * (actual - wanted) / (pixels * ch);
            out.loss[std::size_t(4 + which)] += std::abs(difference) / (b * ch * ch);
            signs[std::size_t(i * ch + j)] = Sign(difference);
          }
        for (int k = 0; k < pixels; ++k)
          for (int i = 0; i < ch; ++i)
            for (int j = 0; j < ch; ++j)
              gradient[std::size_t((n * pixels + k) * ch + i)] += 120. * level * level * (signs[std::size_t(i * ch + j)] + signs[std::size_t(j * ch + i)]) *
                                                                  f[std::size_t((n * pixels + k) * ch + j)] / (b * ch * ch * pixels * ch);
      }
    }
  }
  out.loss[0] = out.loss[1] + 6 * out.loss[2] + .05 * out.loss[3] + 120 * (out.loss[4] + out.loss[5]) + .1 * out.loss[6];
  return out;
}

void LossChecks() {
  std::vector<float> image(80), truth(80), mask(80), raw_feature(24), comp_feature(24), target(24);
  for (int i = 0; i < 80; ++i) {
    image[std::size_t(i)] = float(.11 + .017 * i);
    truth[std::size_t(i)] = float(.03 * (i % 11 - 5));
    mask[std::size_t(i)]  = float(((i / 2) % 9 != 0) && ((i / 2) % 13 != 4));
  }
  for (int i = 0; i < 24; ++i) {
    raw_feature[std::size_t(i)]  = float(.13 * (i % 7 - 3) + .021 * i);
    comp_feature[std::size_t(i)] = float(.2 + .07 * (i % 5 - 2) - .013 * i);
    target[std::size_t(i)]       = float(.09 * (i % 5 - 2) - .017 * i);
  }
  const nn::Shape is{2, 4, 5, 2}, fs{2, 2, 3, 2};
  Tensor          prediction = nn::from_vector(image, is).requires_grad_(), rf = nn::from_vector(raw_feature, fs).requires_grad_(),
         cf = nn::from_vector(comp_feature, fs).requires_grad_();
  const Tensor    t = nn::from_vector(target, fs);
  PartialFeatures r, c, tt;
  for (std::size_t i = 0; i < 3; ++i) r[i] = double(i + 1) * rf, c[i] = double(i + 1) * cf, tt[i] = double(i + 1) * t;
  const auto losses   = PartialObjective(prediction, nn::from_vector(truth, is), nn::from_vector(mask, is), r, c, tt);
  const auto gradient = nn::grad(losses[0], {prediction, rf, cf});
  const auto host     = HostObjective(image, truth, mask, is, raw_feature, comp_feature, target, fs);
  double     scalar_error = 0, gradient_error = 0;
  for (std::size_t i = 0; i < 7; ++i) scalar_error = std::max(scalar_error, std::abs(losses[i].item() - host.loss[i]));
  const std::vector<double>* expected[] = {&host.pixel, &host.raw_feature, &host.comp_feature};
  for (std::size_t i = 0; i < 3; ++i) {
    const auto g = gradient[i].to_vector<float>();
    for (std::size_t j = 0; j < expected[i]->size(); ++j) gradient_error = std::max(gradient_error, std::abs(g[j] - expected[i]->at(j)));
  }
  check::Near(scalar_error, 0, 2e-5, "independent all seven partial objective scalars");
  check::Near(gradient_error, 0, 2e-5, "every image/raw-feature/composite-feature partial derivative");
  // Exact-perfect loss includes nonzero TV if the image varies, so test TV independently.
  check::Near(PartialTV(nn::full(is, .25), nn::from_vector(mask, is)).item(), 0, 0, "constant image has zero hole-TV");
  std::cout << " independent seven pixel/perceptual/Gram/TV values and every image/feature derivative max errors=" << scalar_error << "/" << gradient_error
            << ";both endpoints/dilated-hole/full-image normalization passed\n";
}

void KerasParity(const style::TensorMap& weights, const style::TensorMap& fixture) {
  const style::FrozenVgg16 vgg(weights);
  Tensor                   x     = fixture.at("prediction").clone().requires_grad_();
  const Tensor             truth = fixture.at("truth"), mask = fixture.at("mask");
  const auto rf = PartialVgg(vgg, x), cf = PartialVgg(vgg, PartialComposite(truth, x, mask)), tf_features = PartialVgg(vgg, truth);
  const auto loss     = PartialObjective(x, truth, mask, rf, cf, tf_features);
  const auto gradient = nn::grad(loss[0], {x})[0].to_vector<float>();
  double     feature_error = 0, feature_peak = 0, scalar_error = 0, gradient_error = 0, gradient_peak = 0;
  for (int i = 0; i < 6; ++i) {
    const Tensor& actual   = i < 3 ? rf[std::size_t(i)] : tf_features[std::size_t(i % 3)];
    const Tensor& expected = fixture.at(std::string(i < 3 ? "raw_pool" : "target_pool") + std::to_string(i % 3 + 1));
    check::Require(actual.shape() == expected.shape(), "independent VGG pool layer shapes");
    const auto a = actual.to_vector<float>(), e = expected.to_vector<float>();
    for (std::size_t j = 0; j < e.size(); ++j) {
      feature_error = std::max(feature_error, std::abs(double(a[j]) - e[j]));
      feature_peak  = std::max(feature_peak, std::abs(double(e[j])));
    }
  }
  const auto components = fixture.at("components").to_vector<float>();
  for (std::size_t i = 0; i < 7; ++i) scalar_error = std::max(scalar_error, std::abs(loss[i].item() - components[i]));
  const auto expected = fixture.at("gradient").to_vector<float>();
  check::Require(gradient.size() == expected.size(), "every batched image gradient coordinate");
  for (std::size_t i = 0; i < expected.size(); ++i) {
    gradient_error = std::max(gradient_error, std::abs(double(gradient[i]) - expected[i]));
    gradient_peak  = std::max(gradient_peak, std::abs(double(expected[i])));
  }
  // TF: absolute 2e-5 (TF's Eigen kernels match Keras to 3.8e-6). The fixture is
  // float32 Keras output (a float64 evaluation differs from it by 1.1e-5 at
  // |pool3| <= 42.5), so the check is relative to the peak feature: 2e-6.
  check::Near(feature_error / feature_peak, 0, 2e-6, "genuine Keras/native pool1/2/3 features (relative to peak)");
  check::Near(scalar_error, 0, 2e-5, "all seven genuine Keras/native loss components");
  check::Near(gradient_error / (gradient_peak + 1e-12), 0, 5e-5, "genuine Keras/native every-image gradient relative max error");
  vgg.CheckFrozen();
  std::cout << " genuine Keras pool1/2/3 feature/loss errors=" << feature_error << "(relative " << feature_error / feature_peak << ")/" << scalar_error
            << ",every batched image gradient absolute/relative max error=" << gradient_error << "/" << gradient_error / (gradient_peak + 1e-12)
            << ";all26 weights unchanged\n";
}

void DataChecks() {
  std::vector<std::int32_t> raw(4 * 3072);
  for (int i = 0; i < int(raw.size()); ++i) raw[std::size_t(i)] = (i * 13 + i / 3072 * 37) % 256;
  const Tensor source = nn::from_vector(raw, {4, 32, 32, 3});
  const auto   all = PartialPairs(source, {0, 1, 2, 3}, 107), repeat = PartialPairs(source, {0, 1, 2, 3}, 107), reordered = PartialPairs(source, {3, 1}, 107),
             fresh = PartialPairs(source, {0, 1, 2, 3}, 109);
  const auto am = all.mask.to_vector<float>(), ai = all.input.to_vector<float>(), at = all.truth.to_vector<float>();
  check::Require(am == repeat.mask.to_vector<float>() && am != fresh.mask.to_vector<float>(), "fixed irregular mask seed/fresh split masks");
  for (int n = 0; n < 4; ++n) {
    int holes = 0;
    for (int i = 0; i < 1024; ++i) {
      const float mask = am[std::size_t((n * 1024 + i) * 3)];
      check::Require(mask == 0 || mask == 1, "binary irregular validity");
      holes += mask == 0;
      for (int c = 0; c < 3; ++c) {
        const std::size_t j = std::size_t((n * 1024 + i) * 3 + c);
        check::Near(am[j], mask, 0, "input RGB masks agree");
        check::Near(ai[j], mask != 0 ? at[j] : 0, 0, "hidden targets excluded from observed input");
      }
    }
    check::Require(holes >= 123 && holes <= 563, "irregular mask region within configured range");
  }
  const auto rm = reordered.mask.to_vector<float>(), ri = reordered.input.to_vector<float>();
  for (int n = 0; n < 2; ++n)
    for (int i = 0; i < 3072; ++i) {
      check::Near(rm[std::size_t(n * 3072 + i)], am[std::size_t((n ? 1 : 3) * 3072 + i)], 0, "per-source-row mask order independence");
      check::Near(ri[std::size_t(n * 3072 + i)], ai[std::size_t((n ? 1 : 3) * 3072 + i)], 0, "reordered observed/target pairing");
    }
  auto changed = raw;
  for (std::size_t i = 0; i < am.size(); ++i)
    if (am[i] == 0) changed[i] ^= 255;
  const auto hidden = PartialPairs(nn::from_vector(changed, {4, 32, 32, 3}), {0, 1, 2, 3}, 107);
  check::Require(hidden.input.to_vector<float>() == ai && hidden.truth.to_vector<float>() != at, "hidden ground truth cannot leak through irregular input");
  check::Require(PartialMean(all).to_vector<float>() == PartialMean(hidden).to_vector<float>(), "baseline uses observed pixels only");
  std::cout << " irregular seed-separated masks,12..55% coverage,reordering/pairing and hidden-target/visible-mean independence passed\n";
}

void ModelChecks() {
  for (bool compact : {false, true}) {
    nn::ParameterStore p(83), state(84);
    const int          batch = compact ? 4 : 1, side = compact ? 33 : 256, width = compact ? 35 : 256;
    const auto         config = compact ? PartialConfig::Compact() : PartialConfig{};
    std::vector<float> image(std::size_t(batch * side * width * 3)), valid(image.size());
    for (std::size_t i = 0; i < image.size(); ++i) {
      image[i]       = float(.5 + .4 * std::sin(double(i) * .17));
      const int pixel = int(i / 3), y = pixel / width % side, col = pixel % width;
      valid[i]       = float(compact ? !(y > 7 && y < 25 && col > 8 && col < 27) : !(y > 60 && y < 190 && col > 70 && col < 200));
    }
    const Tensor x = nn::from_vector(image, {batch, side, width, 3}), mask = nn::from_vector(valid, {batch, side, width, 3});
    const auto   net      = PartialUNet(p, state, x, mask, config, true, false);
    const auto   gradient = nn::grad(nn::mean(nn::square(net.image)), p.parameters());
    check::Require(p.count() == (compact ? 183684 : 32865236) && state.count() == (compact ? 592 : 10880) && p.size() == std::size_t(compact ? 36 : 60),
                   "exact full/compact partial UNet model/BN census");
    std::cout << " " << (compact ? "compact" : "full") << " partial UNet:" << p.count() << " parameters/" << p.size() << " derivatives," << state.count()
              << " BN values," << net.stages.size() << " mask/feature stages" << (compact ? "" : " (executed, batch1)") << "\n";
    if (!compact) continue;
    const int h[] = {17, 9, 5, 3, 2, 3, 5, 9, 17, 33}, w[] = {18, 9, 5, 3, 2, 3, 5, 9, 18, 35}, c[] = {8, 16, 32, 64, 64, 64, 32, 16, 8, 3};
    check::Require(net.stages.size() == 10 && net.masks.size() == 10, "all10 compact stages exposed");
    for (int i = 0; i < 10; ++i) {
      check::Require(net.stages[std::size_t(i)].shape() == nn::Shape({4, h[i], w[i], c[i]}) && net.masks[std::size_t(i)].shape() == net.stages[std::size_t(i)].shape(),
                     "every exact odd-rectangle feature and multi-channel mask shape");
      for (float v : net.masks[std::size_t(i)].to_vector<float>()) check::Require(v == 0 || v == 1, "every propagated mask is binary");
    }
    for (float v : net.masks[9].to_vector<float>()) check::Near(v, 1, 0, "valid context propagates through all decoder holes");
    double minimum = INFINITY, maximum = 0;
    int    zeros = 0;
    for (std::size_t i = 0; i < gradient.size(); ++i) {
      double norm = 0;
      for (float v : gradient[i].to_vector<float>()) {
        check::Require(std::isfinite(v), "finite every partial UNet parameter derivative");
        norm += double(v) * v;
      }
      norm = std::sqrt(norm);
      if (norm == 0) {
        ++zeros;
        check::Require(p.names()[i].find("/W") == std::string::npos, "every partial kernel receives nonzero gradient");
      } else {
        minimum = std::min(minimum, norm);
      }
      maximum = std::max(maximum, norm);
    }
    {
      nn::NoGradGuard guard;
      const auto      empty = PartialUNet(p, state, x, nn::zeros(mask.shape()), config, false);
      for (const Tensor& value : {empty.image, empty.mask})
        for (float v : value.to_vector<float>()) check::Near(v, 0, 0, "empty mask produces zero raw output/updated mask");
      const Tensor ones      = nn::ones(mask.shape());
      const auto   known     = PartialUNet(p, state, x, ones, config, false);
      check::Require(PartialComposite(x, known.image, ones).to_vector<float>() == image, "all-known composite is exact input identity");
    }
    std::cout << " compact all36 gradients finite,nonzero norm range[" << minimum << "," << maximum << "],zero affine tensors=" << zeros
              << ";all10 feature/mask shapes,mask propagation,empty and known identities passed\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs      = 3;
    bool                  checks_only = false;
    std::filesystem::path root = paths::Dataset("cifar10"), assets = paths::Dataset("style_transfer");
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("epochs=", 0) == 0)
        epochs = std::stoi(arg.substr(7));
      else if (arg.rfind("data=", 0) == 0)
        root = arg.substr(5);
      else if (arg.rfind("assets=", 0) == 0)
        assets = arg.substr(7);
      else if (arg == "checks_only=0" || arg == "checks_only=1")
        checks_only = arg == "checks_only=1";
      else if (arg.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(arg.substr(8)));
      else
        throw std::invalid_argument("expected epochs=N data=PATH assets=PATH checks_only=0|1 threads=N");
    }
    LossChecks();
    DataChecks();
    ModelChecks();
    if (!std::filesystem::exists(assets / "vgg16.npz"))
      datamunge::datasets::prepare_example_data("style", assets.string());
    const auto weights = style::LoadTensorMap((assets / "vgg16.npz").string());
    if (std::filesystem::exists(assets / "partial_reference.npz"))
      KerasParity(weights, style::LoadTensorMap((assets / "partial_reference.npz").string()));
    else
      std::cout << "Independent Keras parity fixture unavailable; native checks run, external parity not run.\n";
    if (checks_only) {
      std::cout << "All partial-inpainting fixtures passed (training not requested).\n";
      return 0;
    }
    TrainPartialInpainting(root, assets, weights, epochs, paths::OutputDir("models/partial_inpainting"));
    std::cout << "All partial-convolution inpainting checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
