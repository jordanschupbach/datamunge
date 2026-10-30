// CMD: just run-nn grad_cam
// BOTH Grad-CAM and paper-equation Grad-CAM++,verified classifier,no retraining.
// Last post-ReLU conv14x14x16;fixed target/runner/initial/head controls.
//
// Port of tflow's grad_cam. Prepare: run-nn vanilla_saliency (writes
// output/nn/models/vanilla_saliency), or pass source=tflow to use tflow's recorded
// classifier (then every map is comparable with tflow's maps.npz). Data:
// saliency_data.npz (paths::Dataset("saliency"), data=PATH). Map formulas:
// common/grad_cam.hpp. The feature derivative is nn::grad of the selected logit
// with respect to the (non-leaf) conv2 activation. Outputs: output/nn/models/grad_cam/.
// Options:source=PATH|tflow,data=PATH,out=PATH,threads=N.

#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "common/checks.hpp"
#include "common/grad_cam.hpp"
#include "common/paths.hpp"
#include "common/saliency.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

void FormulaChecks() {
  std::vector<float> av(16), gv(16);
  for (int i = 0; i < 16; ++i) {
    av[i] = .1f + .03f * i;
    gv[i] = std::array<float, 8>{.3f, -.2f, 0.f, .4f, -.1f, .2f, .5f, 0.f}[i % 8];
  }
  const auto maps = cam::ClassActivationMaps(sal::FromF(av, {2, 2, 2, 2}), sal::FromF(gv, {2, 2, 2, 2}));
  const auto w = sal::F(maps.grad_weights), sgn = sal::F(maps.grad_signed), gm = sal::F(maps.grad_map),
             alpha_t = sal::F(maps.positive_alpha), pw_t = sal::F(maps.plus_weights), pm = sal::F(maps.plus_map),
             zero = sal::F(cam::NormalizeCam(nn::zeros({2, 2, 2})));
  auto A = [&](int b, int y, int x, int c) { return double(av[((b * 2 + y) * 2 + x) * 2 + c]); };
  auto G = [&](int b, int y, int x, int c) { return double(gv[((b * 2 + y) * 2 + x) * 2 + c]); };
  double error = 0;
  for (int b = 0; b < 2; ++b) {
    double sums[2] = {}, cw[2] = {}, pw[2] = {};
    for (int c = 0; c < 2; ++c)
      for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x) {
          sums[c] += A(b, y, x, c);
          cw[c] += G(b, y, x, c) / 4.;
        }
    for (int c = 0; c < 2; ++c)
      for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x) {
          const double v = G(b, y, x, c);
          const double alpha = v > 0 ? v * v / (2 * v * v + sums[c] * v * v * v) : 0;
          error = std::max(error, std::abs(alpha - alpha_t[((b * 2 + y) * 2 + x) * 2 + c]));
          pw[c] += alpha * std::max(v, 0.);
        }
    for (int c = 0; c < 2; ++c) {
      check::Near(w[b * 2 + c], cw[c], 1e-7, "CAM averages signed gradients");
      check::Near(pw_t[b * 2 + c], pw[c], 1e-7, "CAM++ positive derivative-weight sum");
    }
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 2; ++x) {
        double classic = 0, plus = 0;
        for (int c = 0; c < 2; ++c) {
          classic += cw[c] * A(b, y, x, c);
          plus += pw[c] * A(b, y, x, c);
        }
        const int i = (b * 2 + y) * 2 + x;
        check::Near(sgn[i], classic, 1e-7, "all signed CAM pixels");
        check::Near(gm[i], std::max(classic, 0.), 1e-7, "all rectified CAM pixels");
        check::Near(pm[i], std::max(plus, 0.), 1e-7, "all CAM++ pixels");
      }
  }
  check::Near(error, 0, 1e-7, "all16positivealpha paper unreduced g2/g3 ratios");
  for (float v : zero) check::Near(v, 0, 0, "zeroCAM normalizes tozero,notNaN");
  // True higher derivatives of exp(affine score),not higher derivatives of rawlogit.
  double          higher_error = 0;
  nn::GradOptions create;
  create.create_graph = true;
  for (int i = 0; i < 8; ++i) {
    const float coefficient = .2f * std::sin(.7f * i + .2f);
    Tensor      x = nn::scalar(.3f);
    x.requires_grad_(true);
    const Tensor exp_score = nn::exp(nn::add(nn::mul(x, double(coefficient)), double(.1f)));
    const Tensor d1 = nn::grad(exp_score, {x}, create)[0], d2 = nn::grad(d1, {x}, create)[0], d3 = nn::grad(d2, {x}, create)[0];
    const double e = std::exp(double(coefficient) * .3f + .1f);
    const Tensor derivatives[] = {exp_score, d1, d2, d3};
    for (int order = 1; order <= 3; ++order)
      higher_error = std::max(higher_error, std::abs(e * std::pow(coefficient, order) - derivatives[order].item<double>()));
  }
  check::Near(higher_error, 0, 2e-7, "ALL24 higher-order identities expS*g^order");
  std::cout << "two-batch CAM/CAM++ allweights/alpha/maps/zero normalization passed;alphaerror=" << error
            << ",ALL24 actual1st/2nd/3rd exp-affine derivatives error=" << higher_error
            << ";raw-logit g2/g3 areNOT mislabeled as raw higher derivatives\n";
}

/// tflow's CamModel: the saliency classifier with exact source weights, the selected-logit
/// derivative with respect to the last post-ReLU conv features, both CAMs and the tail probe.
struct CamModel {
  sal::SaliencyModel model{false};
  explicit CamModel(const sal::Arrays& weights) { model.Assign(weights); }
  Tensor Predict(const sal::SaliencyData& d) { return model.Predict(d); }
  /// activation,gradient,cam_weights,cam_signed,cam,positive_alpha,plus_weights,plus,
  /// cam_normalized,plus_normalized,cam_resized,plus_resized,logits
  std::vector<Tensor> Maps(const sal::SaliencyData& d, const Tensor& classes) {
    sal::CheckTargets(classes, d.images.size(0));
    const auto   net = model.Forward(d.images);
    const Tensor gradient = nn::grad(nn::sum(sal::SaliencyScore(net.logits, classes)), {net.conv2})[0].detach();
    nn::NoGradGuard guard;
    const Tensor    activation = net.conv2.detach();
    const auto      cams = cam::ClassActivationMaps(activation, gradient);
    const Tensor    normalized = cam::NormalizeCam(cams.grad_map), plus_normalized = cam::NormalizeCam(cams.plus_map);
    auto            resize = [](const Tensor& map) {
      return nn::reshape(nn::resize_bilinear(nn::expand_dims(map, 3), {28, 28}, false, true), {map.size(0), 28, 28});
    };
    std::vector<Tensor> out = {activation,       gradient,          cams.grad_weights, cams.grad_signed, cams.grad_map,
                               cams.positive_alpha, cams.plus_weights, cams.plus_map,   normalized,       plus_normalized,
                               resize(normalized),  resize(plus_normalized), net.logits.detach()};
    for (auto& t : out) {
      t = t.contiguous();
      for (float v : sal::F(t)) check::Require(std::isfinite(v), "finite every CAM intermediate");
    }
    for (float v : sal::F(out[0])) check::Require(v >= 0, "CAM++ stable denominator assumes post-ReLU nonnegative features");
    return out;
  }
  /// The same trained classifier tail on explicit feature coordinates: {logits, hidden pre-activation}.
  std::vector<Tensor> Probe(const Tensor& a) {
    nn::NoGradGuard guard;
    const Tensor    pooled = nn::avg_pool2d(a, 2, 2, nn::Padding::Valid);
    const Tensor    pre = nn::dense(model.parameters, "hidden", nn::reshape(pooled, {a.size(0), 784}), 32);
    return {nn::dense(model.parameters, "head", nn::relu(pre), 10).contiguous(), pre.contiguous()};
  }
  std::vector<std::string> Snapshot() const { return model.Snapshot(); }
};

const std::vector<std::string> Keys = {"activation",     "gradient",     "cam_weights", "cam_signed",     "cam",
                                       "positive_alpha", "plus_weights", "plus",        "cam_normalized", "plus_normalized",
                                       "cam_resized",    "plus_resized", "logits"};

double Cosine(const std::vector<float>& a, const std::vector<float>& b, int row) {
  double dot = 0, aa = 0, bb = 0;
  for (int i = 0; i < 196; ++i) {
    const double x = a[row * 196 + i], y = b[row * 196 + i];
    dot += x * y;
    aa += x * x;
    bb += y * y;
  }
  return aa > 0 && bb > 0 ? dot / std::sqrt(aa * bb) : 0;
}

void Run(const std::filesystem::path& source, const std::filesystem::path& data, const std::filesystem::path& directory) {
  const auto  pools = ae::LoadSaliency(data);
  const auto& test = pools[2];
  sal::SaliencyModel ordinary(false), fresh(false);
  const auto         w = sal::RestoreSource(source, ordinary, fresh);  // restore same verified3epochclassifier
  auto               head = w.trained;
  head.at("head/W") = w.initial.at("head/W");
  head.at("head/b") = w.initial.at("head/b");
  CamModel   model(w.trained), random(w.initial), head_random(head);
  const auto before = model.Snapshot(), initial_before = random.Snapshot(), head_before = head_random.Snapshot(),
             ordinary_before = ordinary.Snapshot();
  check::Require(before == ordinary_before, "all8CAMtensors identical to source checkpoint");
  const auto logits = model.Predict(test), reference = ordinary.Predict(test);
  {
    const auto l = sal::F(logits), r = sal::F(reference);
    for (std::size_t i = 0; i < l.size(); ++i) check::Near(l[i], r[i], 1e-6, "all2000unchanged forward logits");
  }
  const auto examples = sal::Pick(test, sal::FirstPerDigit(test));
  const auto scores = model.Predict(examples);
  const auto [targets, runners] = sal::TopTwo(scores);
  const auto trained = model.Maps(examples, targets), runner = model.Maps(examples, runners),
             untrained = random.Maps(examples, targets), head_maps = head_random.Maps(examples, targets),
             repeat = model.Maps(examples, targets);
  for (std::size_t k = 0; k < trained.size(); ++k)
    check::Require(sal::Bytes(trained[k]) == sal::Bytes(repeat[k]), "every CAM intermediate repeat deterministic");
  const auto probe = model.Probe(trained[0]);
  double     margin = INFINITY;
  for (float v : sal::F(probe[1])) margin = std::min(margin, std::abs(double(v)));
  check::Require(margin > 1e-8, "selected tail inputs are strictly away from hidden ReLU boundaries");
  {
    const auto p = sal::F(probe[0]), s = sal::F(scores);
    for (std::size_t i = 0; i < s.size(); ++i) check::Near(p[i], s[i], 1e-6, "explicit feature tail matches actual model");
  }
  const auto one = sal::Pick(examples, {0});
  const int  tc = sal::I(targets)[0];
  const auto single = model.Maps(one, sal::FromI({tc}));
  for (int k : {1, 2, 4, 5, 6, 7, 8, 9, 10, 11}) {
    const auto s = sal::F(single[k]), t = sal::F(trained[k]);
    for (std::size_t i = 0; i < s.size(); ++i) check::Near(s[i], t[i], 2e-6, "single/batch CAMweights/maps/derivative agreement");
  }
  const auto sg = sal::F(single[1]);
  int        coordinate = 0;
  for (int i = 1; i < 3136; ++i)
    if (std::abs(sg[i]) > std::abs(sg[coordinate])) coordinate = i;
  auto plus = sal::F(single[0]), minus = plus;
  plus[coordinate] += .001f;
  minus[coordinate] -= .001f;
  const auto   high = sal::F(model.Probe(sal::FromF(plus, {1, 14, 14, 16}))[0]),
             low = sal::F(model.Probe(sal::FromF(minus, {1, 14, 14, 16}))[0]);
  const double fd = (double(high[tc]) - low[tc]) / .002;
  const double fd_error = std::abs(fd - sg[coordinate]);
  check::Require(fd_error < .01 + .01 * std::abs(sg[coordinate]), "actual selected-logit/feature central difference");
  check::Require(before == model.Snapshot() && initial_before == random.Snapshot() && head_before == head_random.Snapshot() &&
                     ordinary_before == ordinary.Snapshot(),
                 "all4x8model tensors immutable across CAM/control/probe calls");
  std::filesystem::create_directories(directory);
  sal::Arrays exported = {{"images", examples.images},   {"labels", examples.labels},       {"source_ids", examples.source},
                          {"targets", targets},          {"runners", runners},              {"test_logits", logits},
                          {"test_labels", test.labels},  {"test_source_ids", test.source}, {"trained_hidden_pre", probe[1]}};
  for (const auto& [name, maps] : std::vector<std::pair<std::string, const std::vector<Tensor>*>>{
           {"trained", &trained}, {"runner", &runner}, {"initial", &untrained}, {"head_initial", &head_maps}})
    for (std::size_t k = 0; k < Keys.size(); ++k) exported[name + "_" + Keys[k]] = maps->at(k);
  sal::SaveArrays(directory / "maps.npz", exported);
  sal::SaveArrays(directory / "model_arrays.npz", w.trained);
  sal::SaveArrays(directory / "initial_arrays.npz", w.initial);
  sal::SaveArrays(directory / "head_initial_arrays.npz", head);
  sal::CopyManifests(directory, {{source / "data_manifest.json", "data_manifest.json"},
                                 {paths::Dataset("grad_cam") / "source_manifest.json", "source_manifest.json"}});
  const auto tr8 = sal::F(trained[8]), tr9 = sal::F(trained[9]), ru8 = sal::F(runner[8]), ru9 = sal::F(runner[9]),
             un8 = sal::F(untrained[8]), un9 = sal::F(untrained[9]), hd8 = sal::F(head_maps[8]), hd9 = sal::F(head_maps[9]),
             tr4 = sal::F(trained[4]), tr7 = sal::F(trained[7]);
  std::ofstream stats(directory / "controls.tsv");
  stats << "case\tcam_runner_cosine\tplus_runner_cosine\tcam_initial_cosine\tplus_initial_cosine\tcam_head_cosine\tplus_head_cosine\n";
  double cam_head = 0, plus_head = 0, difference = 0;
  int    zero_cam = 0;
  for (int b = 0; b < 10; ++b) {
    const double ch = Cosine(tr8, hd8, b), ph = Cosine(tr9, hd9, b);
    cam_head += ch;
    plus_head += ph;
    stats << b << '\t' << Cosine(tr8, ru8, b) << '\t' << Cosine(tr9, ru9, b) << '\t' << Cosine(tr8, un8, b) << '\t' << Cosine(tr9, un9, b)
          << '\t' << ch << '\t' << ph << '\n';
    double maximum = 0;
    for (int i = 0; i < 196; ++i) {
      maximum = std::max(maximum, double(tr4[b * 196 + i]));
      difference += std::abs(double(tr8[b * 196 + i]) - tr9[b * 196 + i]);
    }
    zero_cam += maximum == 0;
  }
  check::Require(difference > 1e-4, "both distinct method maps genuinely computed");
  std::ofstream html(directory / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Grad-CAM and Grad-CAM++</title><style>body{font:16px "
          "system-ui;max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:30px "
          "0}</style><h1>Grad-CAM and paper-equation Grad-CAM++</h1><p>Lastpost-ReLU convolution:14x14x16. Grad-CAM averages rawlogit "
          "gradients per channel. Grad-CAM++ uses positive pixel weights from exponential-score diagonal2nd/3rd derivatives,with a stopped "
          "constant shift so exponential=1 at the evaluation point. Rawlogit higher derivatives are not g-squared/g-cubed. PaperEq11/19;no "
          "extra legacy spatial alpha renormalization.</p><p>Each map is normalized separately to its own maximum for "
          "SHAPEcomparison;brightness is NOT comparable across maps/methods/targets. Black=0,orange=1. Raw coarse "
          "maps/weights/gradients/peaks remain in maps.npz. Bilinear14-to28 uses half-pixel centers,then4x nearest display. Zero maps "
          "remain black.</p><p>Top row:input,Grad-CAMheat,Grad-CAMoverlay,Grad-CAM++heat,Grad-CAM++overlay(blank lastcell). "
          "Bottom:runnerCAM,runner++,initialCAM,initial++,head-initialCAM,head-initial++. Controls keep trained predictedclass FIXED. "
          "Targets/truth/sourceIDs shown,including classifier errors. Overlays mix60%grayscale+40%heat.</p><p>Classifier reused from "
          "verifiedthree-epoch run;0newtraining. Coarse maps are not segmentation masks or causal proofs. No claim that++is better on "
          "these digits.</p>";
  const auto image = sal::F(examples.images);
  const auto src = sal::I(examples.source), labels = sal::I(examples.labels), t = sal::I(targets), r = sal::I(runners);
  const auto tr10 = sal::F(trained[10]), tr11 = sal::F(trained[11]), ru10 = sal::F(runner[10]), ru11 = sal::F(runner[11]),
             un10 = sal::F(untrained[10]), un11 = sal::F(untrained[11]), hd10 = sal::F(head_maps[10]), hd11 = sal::F(head_maps[11]);
  for (int b = 0; b < 10; ++b) {
    sal::Canvas pixels(232, 692);
    auto        draw = [&](int oy, int ox, const std::vector<float>* map, bool overlay, bool input) {
      for (int y = 0; y < 112; ++y)
        for (int x = 0; x < 112; ++x) {
          const int    i = b * 784 + (y / 4) * 28 + x / 4;
          const double intensity = image[i], v = map ? (*map)[i] : 0;
          const std::array<double, 3> color = {255 * v, 128 * v, 0};
          for (int c = 0; c < 3; ++c)
            pixels.Set(oy + y, ox + x, c, input ? 255 * intensity : overlay ? .6 * 255 * intensity + .4 * color[c] : color[c]);
        }
    };
    draw(0, 0, nullptr, false, true);
    draw(0, 116, &tr10, false, false);
    draw(0, 232, &tr10, true, false);
    draw(0, 348, &tr11, false, false);
    draw(0, 464, &tr11, true, false);
    draw(120, 0, &ru10, false, false);
    draw(120, 116, &ru11, false, false);
    draw(120, 232, &un10, false, false);
    draw(120, 348, &un11, false, false);
    draw(120, 464, &hd10, false, false);
    draw(120, 580, &hd11, false, false);
    const std::string name = "case" + std::to_string(b) + ".png";
    sal::SaliencyPng(directory / name, pixels);
    double cm = 0, pm = 0;
    for (int i = 0; i < 196; ++i) {
      cm = std::max(cm, double(tr4[b * 196 + i]));
      pm = std::max(pm, double(tr7[b * 196 + i]));
    }
    html << "<section><h2>Source" << src[b] << ",truth" << labels[b] << ",pred" << t[b] << ",runner" << r[b]
         << "</h2><p>Raw CAM/shifted++peaks=" << cm << '/' << pm << "</p><img src='" << name << "'></section>";
  }
  check::Require(bool(html) && bool(stats), "save bothCAM galleries/controlstats");
  sal::SaliencyMetrics("unchanged3epochclassifier", test, logits);
  std::cout << "Grad-CAM AND Grad-CAM++ actual lastfeature14x14x16,10realcases,ordinaryclass "
               "gradients;bothmethods/runner/fullinitial/headinitial controls;0newtraining\nminhiddenReLUboundarymargin="
            << margin << ",realfeatureFDcoordinate=" << coordinate << ",error=" << fd_error << ",normalizedmethodL1difference=" << difference
            << ",blankpredictedCAMs=" << zero_cam << "\nmeanhead-initialcosine CAM/++=" << cam_head / 10 << '/' << plus_head / 10
            << ";all4x8weightsimmutable/repeat/single-batch/probetail/all2000logits passed\nGrad-CAM++ uses exp(S-c),c fixed at "
               "currentS,analyticdiagonal derivatives valid locally awayReLUkinks;no false raw-logit high-order derivative claim or "
               "superiority/segmentation/causal claim.\nArtifacts:"
            << directory.string() << "/index.html\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    nn::set_num_threads(4);
    std::filesystem::path source = sal::DefaultSource(), data = paths::Dataset("saliency"), out = paths::OutputRoot() / "models/grad_cam";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("source=", 0) == 0)
        source = sal::SourceDir(a.substr(7));
      else if (a.rfind("data=", 0) == 0)
        data = a.substr(5);
      else if (a.rfind("out=", 0) == 0)
        out = a.substr(4);
      else if (a.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(a.substr(8)));
      else
        throw std::invalid_argument("expected source=PATH|tflow data=PATH out=PATH threads=N");
    }
    FormulaChecks();
    Run(source, data, out);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
