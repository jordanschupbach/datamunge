// CMD: just run-nn occlusion_sensitivity
// Fixed raw-logit patch removal scores,edge-complete grids and overlap means.
// Replacement/target/patch/stride/initial/null/full controls;0newtraining.
//
// Port of tflow's occlusion_sensitivity. Prepare: run-nn vanilla_saliency (writes
// output/nn/models/vanilla_saliency), or pass source=tflow to use tflow's recorded
// classifier (then every drop is comparable with tflow's occlusion.npz). Data:
// saliency_data.npz (paths::Dataset("saliency"), data=PATH). Patch scoring:
// common/occlusion.hpp. Outputs: output/nn/models/occlusion_sensitivity/.
// Options:patch=N,stride=N,source=PATH|tflow,data=PATH,out=PATH,threads=N.

#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "common/checks.hpp"
#include "common/occlusion.hpp"
#include "common/paths.hpp"
#include "common/saliency.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;
using occlusion::OcclusionResult;
using occlusion::OcclusionSpec;

namespace {

void Checks() {
  std::vector<float> kernel(20), input(40);
  for (int i = 0; i < 20; ++i) kernel[i] = .1f * std::sin(.7 * i);
  for (int i = 0; i < 40; ++i) input[i] = .1f + .02f * i;
  const Tensor k = sal::FromF(kernel, {1, 4, 5, 1}), x = sal::FromF(input, {2, 4, 5, 1}), replace = nn::full({2, 4, 5, 1}, double(.2f)),
               target = sal::FromI({0, 1});
  auto predict = [&](const Tensor& points) {
    const Tensor score = nn::add(nn::sum(nn::mul(points, k), {1, 2, 3}), double(.3f));
    return nn::stack({score, nn::neg(score)}, 1).contiguous();
  };
  const auto result = occlusion::OcclusionSensitivity(x, replace, target, {2, 2, 2, 2}, 3, predict);
  check::Require(result.patches == 6 && result.rectangles[5 * 4] == 2 && result.rectangles[5 * 4 + 1] == 3,
                 "rectangular odd-width edge anchor appended");
  int    coverage[4][5] = {};
  double expected[2][4][5] = {};
  for (int p = 0; p < 6; ++p) {
    const int top = result.rectangles[p * 4], left = result.rectangles[p * 4 + 1];
    for (int y = top; y < top + 2; ++y)
      for (int a = left; a < left + 2; ++a) ++coverage[y][a];
    for (int b = 0; b < 2; ++b) {
      double drop = 0;
      for (int y = 0; y < 4; ++y)
        for (int a = 0; a < 5; ++a) {
          const bool  covered = y >= top && y < top + 2 && a >= left && a < left + 2;
          const float value = covered ? .2f : input[b * 20 + y * 5 + a];
          check::Near(result.input(p, b, y, a, 0), value, 0, "ALL240actual occluded pixels exact");
          if (covered) drop += (input[b * 20 + y * 5 + a] - double(.2f)) * kernel[y * 5 + a] * (b == 0 ? 1 : -1);
        }
      check::Near(result.drop(p, b), drop, 2e-7, "ALL12 signed raw target score drops");
      for (int y = top; y < top + 2; ++y)
        for (int a = left; a < left + 2; ++a) expected[b][y][a] += drop;
    }
  }
  for (int y = 0; y < 4; ++y)
    for (int a = 0; a < 5; ++a) {
      check::Near(result.coverage[y * 5 + a], coverage[y][a], 0, "ALL20coverage counts");
      for (int b = 0; b < 2; ++b) check::Near(result.mean(b, y, a), expected[b][y][a] / coverage[y][a], 2e-7, "ALL40overlap averages");
    }
  const auto identity = occlusion::OcclusionSensitivity(x, x, target, {2, 2, 2, 2}, 1, predict);
  for (double d : identity.drops) check::Near(d, 0, 2e-7, "identityreplacement givesnullscorechange");
  const auto full = occlusion::OcclusionSensitivity(x, replace, target, {4, 5, 4, 5}, 2, predict);
  check::Require(full.patches == 1, "full image replacement one patch");
  const auto replacement_score = sal::F(predict(replace));
  for (int b = 0; b < 2; ++b)
    check::Near(full.score(0, b), replacement_score[b * 2 + b], 0, "full image score is replacement image score");
  check::Throws([] { occlusion::OcclusionAnchors(5, 2, 3); }, "stride>patch rejected instead of silently leaving holes");
  std::cout << "occlusion ALL240mutatedpixels/12signeddrops/20coverage/40overlapmeans and rectangular-edge/identity/full/invalidstride "
               "fixtures passed\n";
}

OcclusionResult Compute(sal::SaliencyModel& model, const sal::SaliencyData& examples, const Tensor& replacement, const Tensor& targets,
                        OcclusionSpec spec, int batch = 32) {
  return occlusion::OcclusionSensitivity(examples.images, replacement, targets, spec, batch,
                                         [&](const Tensor& points) { return model.Predict(points); });
}

void Run(const std::filesystem::path& source, const std::filesystem::path& data, const std::filesystem::path& directory, int patch,
         int stride) {
  occlusion::OcclusionAnchors(28, patch, stride);
  const auto  pools = ae::LoadSaliency(data);
  const auto& test = pools[2];
  sal::SaliencyModel model(false), initial(false);
  sal::RestoreSource(source, model, initial);  // same8trained/originalinitial weights
  const auto before = model.Snapshot(), initial_before = initial.Snapshot();
  const auto test_logits = model.Predict(test);
  const auto examples = sal::Pick(test, sal::FirstPerDigit(test));
  const auto logits = model.Predict(examples);
  const auto [targets, runners] = sal::TopTwo(logits);
  const Tensor black = nn::zeros(examples.images.shape()), gray = nn::full(examples.images.shape(), .5);
  const OcclusionSpec spec{patch, patch, stride, stride};
  std::map<std::string, OcclusionResult> groups;
  groups.emplace("main", Compute(model, examples, black, targets, spec));
  groups.emplace("gray", Compute(model, examples, gray, targets, spec));
  groups.emplace("runner", Compute(model, examples, black, runners, spec));
  groups.emplace("initial", Compute(initial, examples, black, targets, spec));
  groups.emplace("small", Compute(model, examples, black, targets, {5, 5, 3, 3}));
  groups.emplace("stride", Compute(model, examples, black, targets, {7, 7, 7, 7}));
  groups.emplace("identity", Compute(model, examples, examples.images, targets, spec));
  groups.emplace("full", Compute(model, examples, black, targets, {28, 28, 28, 28}));
  const auto& main = groups.at("main");
  const auto  repeat = Compute(model, examples, black, targets, spec);
  check::Require(repeat.occluded_logits == main.occluded_logits && repeat.pixel_mean == main.pixel_mean,
                 "exact repeated occlusion deterministic");
  double null_error = 0;
  for (double d : groups.at("identity").drops) null_error = std::max(null_error, std::abs(d));
  check::Near(null_error, 0, 1e-5, "all identity replacement score drops nearzero");
  const auto bz = sal::F(model.Predict(black));
  const auto t = sal::I(targets);
  for (int b = 0; b < 10; ++b)
    check::Near(groups.at("full").score(0, b), bz[b * 10 + t[b]], 1e-6, "actualfullreplacement equalsbaseline image prediction");
  const auto one = sal::Pick(examples, {0});
  const auto single = Compute(model, one, nn::zeros(one.images.shape()), sal::FromI({t[0]}), spec, 1);
  double     batch_error = 0;
  for (int p = 0; p < main.patches; ++p) batch_error = std::max(batch_error, std::abs(single.drop(p, 0) - main.drop(p, 0)));
  check::Near(batch_error, 0, 1e-5, "single/batched float32 counterfactual score parity");
  std::cout << "maximum single/batched float32 drop difference=" << batch_error << "\n";
  check::Require(before == model.Snapshot() && initial_before == initial.Snapshot(),
                 "all2x8model tensors immutable through all interventions");
  double peak = 0;
  int    positive = 0, negative = 0;
  for (const std::string name : {"main", "gray", "runner", "small", "stride", "initial"})
    for (double v : groups.at(name).pixel_mean) peak = std::max(peak, std::abs(v));
  for (double d : main.drops) {
    positive += d > 0;
    negative += d < 0;
  }
  check::Require(peak > 0 && positive > 0 && negative > 0, "both score-decreasing and score-increasing occlusions retained");
  std::filesystem::create_directories(directory);
  sal::Arrays arrays = {{"images", examples.images},  {"labels", examples.labels},  {"source_ids", examples.source},
                        {"targets", targets},         {"runners", runners},         {"logits", logits},
                        {"test_logits", test_logits}, {"test_labels", test.labels}, {"test_source_ids", test.source},
                        {"black_replacement", black}, {"gray_replacement", gray}};
  for (const auto& [name, g] : groups) occlusion::Export(g, name, arrays);
  arrays["shared_peak"] = sal::Scalar(peak);
  sal::SaveArrays(directory / "occlusion.npz", arrays);
  model.Export(directory / "model_arrays.npz");
  initial.Export(directory / "initial_arrays.npz");
  sal::CopyManifests(directory, {{source / "data_manifest.json", "data_manifest.json"},
                                 {paths::Dataset("occlusion") / "source_manifest.json", "source_manifest.json"}});
  std::ofstream html(directory / "index.html"), stats(directory / "patches.tsv");
  stats << "variant\tpatch\tcase\ttop\tleft\theight\twidth\toriginal\toccluded\tdrop\n";
  for (const auto& [name, g] : groups)
    for (int p = 0; p < g.patches; ++p)
      for (int b = 0; b < 10; ++b)
        stats << name << '\t' << p << '\t' << b << '\t' << g.rectangles[p * 4] << '\t' << g.rectangles[p * 4 + 1] << '\t'
              << g.rectangles[p * 4 + 2] << '\t' << g.rectangles[p * 4 + 3] << '\t' << g.original_score[b] << '\t' << g.score(p, b)
              << '\t' << g.drop(p, b) << '\n';
  html << "<!doctype html><meta charset='utf-8'><title>Occlusion sensitivity</title><style>body{font:16px "
          "system-ui;max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:30px "
          "0}</style><h1>Finite patch occlusion sensitivity</h1><p>Drop=original FIXEDraw-class logit minus logit after replacing one "
          "patch. Redpositive means scoredecreased;blue negative means scoreINCREASED. Negative changes are retained. Grid appends final "
          "anchors to cover every boundary;all patches remain inside theimage. Pixelmaps are AVERAGES of overlapping patch drops,not "
          "additive pixelattributions or segmentation masks.</p><p>Top row:input,signed overlapmean,positive heat,positive overlay,actual "
          "maximum-drop input,actual minimum-drop input. Bottom:signed midgray-replacement,runner-target,smaller-patch,nonoverlap-stride,"
          "initial-model maps,and coverage count(grayscale). One shared absolute scale for signed/controlmaps;whitezero,redpositive,"
          "bluenegative. Orange heat shows positive drops only;overlay60%image/40%heat. Coverage has its own countscale. All input "
          "examples/counterfactuals are exact classifier pixels enlarged4xnearest.</p><p>Defaultblack patch="
       << patch << ",stride=" << stride
       << ". Controls midgray0.5,patch5stride3,patch7stride7,runner-up and actual initialweights. Targets fixed from "
          "unperturbedtrainedpredictions. Identity and full-replacement scores checked separately. No newtraining. Replacement creates a "
          "model counterfactual,not proof of real-world causality.</p>";
  int min_coverage = 100000, max_coverage = 0;
  for (int v : main.coverage) {
    min_coverage = std::min(min_coverage, v);
    max_coverage = std::max(max_coverage, v);
  }
  const auto image = sal::F(examples.images);
  const auto src = sal::I(examples.source), labels = sal::I(examples.labels), r = sal::I(runners);
  for (int b = 0; b < 10; ++b) {
    int hi = 0, lo = 0;
    for (int p = 1; p < main.patches; ++p) {
      if (main.drop(p, b) > main.drop(hi, b)) hi = p;
      if (main.drop(p, b) < main.drop(lo, b)) lo = p;
    }
    sal::Canvas pixels(232, 692);
    auto        draw = [&](int oy, int ox, const OcclusionResult* map, int input_kind, int patch_index) {
      for (int y = 0; y < 112; ++y)
        for (int x = 0; x < 112; ++x) {
          const int    iy = y / 4, ix = x / 4;
          const double v = map ? map->mean(b, iy, ix) / peak : 0;
          const double intensity = image[b * 784 + iy * 28 + ix];
          const auto   signed_color = sal::SignedColor(v);
          const double pos = std::max(v, 0.);
          const std::array<double, 3> heat{255 * pos, 128 * pos, 0};
          for (int c = 0; c < 3; ++c) {
            double value = signed_color[c];
            if (input_kind == 1)
              value = 255 * intensity;
            else if (input_kind == 2)
              value = heat[c];
            else if (input_kind == 3)
              value = .6 * 255 * intensity + .4 * heat[c];
            else if (input_kind == 4)
              value = 255 * main.input(patch_index, b, iy, ix, 0);
            else if (input_kind == 5)
              value = 255. * main.coverage[iy * 28 + ix] / max_coverage;
            pixels.Set(oy + y, ox + x, c, value);
          }
        }
    };
    draw(0, 0, nullptr, 1, 0);
    draw(0, 116, &main, 0, 0);
    draw(0, 232, &main, 2, 0);
    draw(0, 348, &main, 3, 0);
    draw(0, 464, nullptr, 4, hi);
    draw(0, 580, nullptr, 4, lo);
    int col = 0;
    for (const std::string name : {"gray", "runner", "small", "stride", "initial"}) draw(120, 116 * col++, &groups.at(name), 0, 0);
    draw(120, 580, nullptr, 5, 0);
    const std::string filename = "case" + std::to_string(b) + ".png";
    sal::SaliencyPng(directory / filename, pixels);
    html << "<section><h2>Source" << src[b] << ",truth" << labels[b] << ",pred" << t[b] << ",runner" << r[b]
         << "</h2><p>Maximum/minimum patch drop=" << main.drop(hi, b) << '/' << main.drop(lo, b) << ",patchindices=" << hi << '/' << lo
         << ";coordinates in patches.tsv. Exact finite drops are patch/replacement dependent,not a completeness decomposition.</p><img "
            "src='"
         << filename << "'></section>";
  }
  check::Require(bool(html) && bool(stats), "save occlusiongallery/rawpatchtable");
  sal::SaliencyMetrics("unchanged3epochclassifier", test, test_logits);
  std::cout << "occlusion sensitivity actualfinitepatchpredictions:patch=" << patch << ",stride=" << stride
            << ",windows=" << main.patches << ",coverage min/max=" << min_coverage << '/' << max_coverage
            << ",positive/negative drops=" << positive << '/' << negative << "\nidentitymaxerror=" << null_error
            << ";fullreplacement/boundarycoverage/repeat/single-batch/immutable2x8weights passed;0newtraining\nAll8variants "
               "actualscores/occludedpixels saved. Overlapmean is NOT additive causal "
               "pixelimportance;patch/replacement/stride/target/model limits explicit.\nArtifacts:"
            << directory.string() << "/index.html\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    nn::set_num_threads(4);
    int                   patch = 7, stride = 4;
    std::filesystem::path source = sal::DefaultSource(), data = paths::Dataset("saliency"),
                          out = paths::OutputRoot() / "models/occlusion_sensitivity";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("source=", 0) == 0)
        source = sal::SourceDir(a.substr(7));
      else if (a.rfind("data=", 0) == 0)
        data = a.substr(5);
      else if (a.rfind("out=", 0) == 0)
        out = a.substr(4);
      else if (a.rfind("patch=", 0) == 0)
        patch = std::stoi(a.substr(6));
      else if (a.rfind("stride=", 0) == 0)
        stride = std::stoi(a.substr(7));
      else if (a.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(a.substr(8)));
      else
        throw std::invalid_argument("expected source=PATH|tflow data=PATH out=PATH patch=N stride=N threads=N");
    }
    Checks();
    Run(source, data, out, patch, stride);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
