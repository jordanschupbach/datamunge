// CMD: just run-nn image_retrieval
// CNN descriptor retrieval with learned GeM,hard triplets and an exact gallery index.
// https://arxiv.org/abs/1711.02512
// https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/layers/functional.py
// VGG-style13-conv encoder,learned p>1,linear projection and unit descriptors.
// Three default epochs on true CIFAR-10,class-disjoint fit/validation/test,
// disjoint query/gallery IDs,validation-mAP-selected checkpoint and image-only restore.
// Independent GeM/normalization/triplet gradients,mining,ranking/AP/tie/self tests.
// Baselines:untrained same CNN,pooled RGB,mean RGB. No instance-retrieval claim.
//
// Port of tflow's image_retrieval. CIFAR-10 binaries are read from
// $DATAMUNGE_DATA_DIR/cifar10 (data_batch_1..5.bin, test_batch.bin); pools and
// balanced batches are tflow's exactly (same std::mt19937 draws). Weights come
// from datamunge's generator, so trained numbers match TF statistically. TF
// builds the full model (14.8M parameters) as a graph only; here it also runs
// forward/backward once. The checkpoint is best.npz; the deployment check is
// structural (a second model built without an optimizer). Outputs go to
// output/nn/models/image_retrieval.
// Options: epochs=3 data=PATH checks_only=0|1 threads=N.

#include <cmath>
#include <iostream>

#include "common/paths.hpp"
#include "common/retrieval.hpp"

// Experiment workflow from examples/nn/common/retrieval.hpp.
namespace metric {
using namespace ::metric;
inline void TrainRetrieval(const std::filesystem::path& root, int epochs, const std::filesystem::path& directory) {
  check::Require(epochs >= 2, "at least two short retrieval epochs");
  const auto     data = LoadRetrievalData(root);
  RetrievalModel model;
  const double   initial_p = model.Exponent();
  const auto     probe     = RetrievalRows(data.fit, RetrievalBatches(data.fit, 0)[0]);
  const auto     fixed     = RetrievalHardTriplets(model.Encode(probe.images), probe.labels);
  const double   initial   = model.Loss(probe.images, fixed);
  const auto     untrained_query = EncodeRetrieval(model, data.test_query), untrained_gallery = EncodeRetrieval(model, data.test_gallery);
  std::filesystem::create_directories(directory);
  const auto    prefix = (directory / "best.npz").string();
  std::ofstream history(directory / "history.tsv");
  history << "epoch\ttriplet\tvalidation_map\tvalidation_hit1\tp\n";
  double best     = -1;
  int    selected = 0;
  std::cout << "image_retrieval:" << model.parameters.count() << " parameters/" << model.parameters.size() << " tensors,13conv+learnedGeM+bias-free32Dprojection;"
            << epochs << " epochs,8 balanced20image batches/epoch,Adam.0005\n"
            << " fit160 images/classes0..4,val16query/32gallery classes5..6,test48query/96gallery unseen classes7..9;allsourceIDs disjoint\n"
            << " semantic category relevance,not particular-object retrieval;no augmentation/pretraining/whitening;initial fixed fit=" << initial
            << ",p=" << initial_p << std::endl;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    double                 average = 0;
    std::set<std::int64_t> seen;
    for (const auto& rows : RetrievalBatches(data.fit, epoch)) {
      const auto batch = RetrievalRows(data.fit, rows);
      for (std::int64_t id : batch.ids) check::Require(seen.insert(id).second, "fit source used once per epoch");
      const auto triples = RetrievalHardTriplets(model.Encode(batch.images), batch.labels);
      average += model.Loss(batch.images, triples, true) / 8;
    }
    check::Require(seen.size() == 160, "all fitting images used");
    const auto score = RetrievalScore(EncodeRetrieval(model, data.validation_query), data.validation_query,
                                      EncodeRetrieval(model, data.validation_gallery), data.validation_gallery);
    if (score.map > best) {
      best = score.map, selected = epoch;
      model.Save(prefix);
    }
    history << epoch << '\t' << average << '\t' << score.map << '\t' << score.hit1 << '\t' << model.Exponent() << '\n';
    history.flush();
    std::cout << " epoch " << epoch << ":batch-hard loss=" << average << ",validation mAP=" << score.map << ",hit1=" << score.hit1
              << ",p=" << model.Exponent() << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  model.Restore(prefix);
  const auto validation = RetrievalScore(EncodeRetrieval(model, data.validation_query), data.validation_query,
                                         EncodeRetrieval(model, data.validation_gallery), data.validation_gallery);
  check::Near(validation.map, best, 1e-12, "restored validation rankings/mAP agreement");
  const double learned = model.Loss(probe.images, fixed);
  check::Require(learned < initial, "fixed fitting triplet objective improves");
  RetrievalModel deployed(false);
  deployed.Restore(prefix);
  check::Require(!deployed.optimizer && deployed.parameters.size() == model.parameters.size(),
                 "only observed RGB and checkpoint filename: no triplet/label/optimizer state in the descriptor model");
  const auto before  = deployed.Snapshot();
  const auto queries = EncodeRetrieval(deployed, data.test_query), gallery = EncodeRetrieval(deployed, data.test_gallery),
             reference = EncodeRetrieval(model, data.test_query), repeat = EncodeRetrieval(deployed, data.test_query);
  double parity = 0;
  for (std::size_t i = 0; i < queries.values.size(); ++i) {
    parity = std::max(parity, std::abs(double(queries.values[i]) - reference.values[i]));
    check::Near(queries.values[i], repeat.values[i], 0, "deterministic descriptor extraction");
  }
  check::Near(parity, 0, 1e-7, "independent restored descriptor parity");
  const auto one = deployed.Encode(RetrievalRows(data.test_query, {3}).images);
  for (int c = 0; c < 32; ++c) check::Near(one.at(0, c), queries.at(3, c), 2e-6, "single/batch descriptor agreement");
  check::Require(before == deployed.Snapshot(), "query/gallery extraction leaves all model values unchanged");
  DescriptorIndex index(gallery, data.test_gallery.ids);
  std::cout << " selected=" << selected << ",validation mAP=" << best << ",fixed fitting " << initial << " -> " << learned
            << ",p=" << deployed.Exponent() << "\n";
  PrintRetrieval("learned", ScoreRetrieval(queries, data.test_query, index, data.test_gallery.labels));
  PrintRetrieval("untrained same CNN", RetrievalScore(untrained_query, data.test_query, untrained_gallery, data.test_gallery));
  const auto pixels_q = RetrievalPixels(data.test_query), pixels_g = RetrievalPixels(data.test_gallery),
             color_q = RetrievalPixels(data.test_query, 32), color_g = RetrievalPixels(data.test_gallery, 32);
  PrintRetrieval("8x8 pooled RGB", RetrievalScore(pixels_q, data.test_query, pixels_g, data.test_gallery));
  PrintRetrieval("mean RGB", RetrievalScore(color_q, data.test_query, color_g, data.test_gallery));
  const auto index_file = (directory / "gallery_index.npz").string();
  nn::save_npz(index_file, {{"descriptors", DescriptorTensor(gallery)},
                            {"source_ids", RetrievalIds(data.test_gallery.ids)},
                            {"labels", RetrievalLabels(data.test_gallery.labels)}});
  nn::Tensor saved_descriptors, saved_ids;
  for (const auto& [name, value] : nn::load_npz(index_file)) {
    if (name == "descriptors") saved_descriptors = value;
    if (name == "source_ids") saved_ids = value;
  }
  check::Require(saved_ids.defined() && saved_ids.dtype() == nn::DType::Int64 && saved_ids.numel() == 96, "serialized index source IDs");
  const auto recovered_ids = saved_ids.to_vector<std::int64_t>();
  check::Require(recovered_ids == data.test_gallery.ids, "serialized IDs round-trip");
  DescriptorIndex reloaded(saved_descriptors, recovered_ids);
  for (int n = 0; n < 48; ++n) {
    const auto a = index.Rank(queries, n, data.test_query.ids[std::size_t(n)]), b = reloaded.Rank(queries, n, data.test_query.ids[std::size_t(n)]);
    for (std::size_t rank = 0; rank < 96; ++rank)
      check::Require(a[rank].id == b[rank].id && a[rank].distance == b[rank].distance, "serialized gallery index preserves ALL exact rankings/distances");
  }
  std::vector<std::int64_t> rankings(48 * 96);
  std::ofstream             ranking_file(directory / "rankings.tsv");
  ranking_file << "query_id\trank\tgallery_id\tdistance\trelevant\n";
  for (int n = 0; n < 48; ++n) {
    const auto ranking = index.Rank(queries, n, data.test_query.ids[std::size_t(n)]);
    for (int rank = 0; rank < 96; ++rank) {
      const auto& item               = ranking[std::size_t(rank)];
      rankings[std::size_t(n * 96 + rank)] = item.id;
      ranking_file << data.test_query.ids[std::size_t(n)] << '\t' << rank + 1 << '\t' << item.id << '\t' << item.distance << '\t'
                   << (data.test_query.labels[std::size_t(n)] == data.test_gallery.labels[item.row]) << '\n';
    }
  }
  nn::save_npz((directory / "test_outputs.npz").string(),
               {{"query_rgb", data.test_query.images},
                {"gallery_rgb", data.test_gallery.images},
                {"query_descriptors", DescriptorTensor(queries)},
                {"gallery_descriptors", DescriptorTensor(gallery)},
                {"query_ids", RetrievalIds(data.test_query.ids)},
                {"gallery_ids", RetrievalIds(data.test_gallery.ids)},
                {"query_labels", RetrievalLabels(data.test_query.labels)},
                {"gallery_labels", RetrievalLabels(data.test_gallery.labels)},
                {"rankings", nn::from_vector(rankings, {48, 96})},
                {"untrained_query", DescriptorTensor(untrained_query)},
                {"untrained_gallery", DescriptorTensor(untrained_gallery)},
                {"rgb_query", DescriptorTensor(pixels_q)},
                {"rgb_gallery", DescriptorTensor(pixels_g)},
                {"mean_rgb_query", DescriptorTensor(color_q)},
                {"mean_rgb_gallery", DescriptorTensor(color_g)},
                {"fit_ids", RetrievalIds(data.fit.ids)},
                {"validation_query_ids", RetrievalIds(data.validation_query.ids)},
                {"validation_gallery_ids", RetrievalIds(data.validation_gallery.ids)}});
  RetrievalPreview(data.test_query, data.test_gallery, queries, index, directory / "comparison.png");
  check::Require(bool(history) && bool(ranking_file), "write retrieval history and complete rankings");
  std::cout << " image-only parity=" << parity << ",immutable parameters,single/batch and serialized-index agreement;artifacts:" << directory
            << " (training " << seconds << " s)\nComplete short CNN descriptor retrieval passed;custom class-disjoint semantic subset,not an instance-retrieval benchmark.\n";
}
}  // namespace metric



namespace nn = datamunge::nn;
using nn::Tensor;
using namespace metric;

namespace {

void PoolingChecks() {
  std::vector<float> input(24), up = {.3f, -.7f, .2f, .5f};
  for (int i = 0; i < 24; ++i) input[std::size_t(i)] = i % 7 == 0 ? -.2f : i % 11 == 0 ? 1e-7f : float(.1 + (i % 9) * .17);
  const float power = 2.3f;
  Tensor      x = nn::from_vector(input, {2, 2, 3, 2}).requires_grad_(), exponent = nn::from_vector(std::vector<float>{power}, {}).requires_grad_();
  const Tensor pooled = GeneralizedMean(x, exponent);
  const auto   g      = nn::grad(nn::sum(pooled * nn::from_vector(up, {2, 2})), {x, exponent});
  const auto   pv = pooled.to_vector<float>(), gx = g[0].to_vector<float>();
  double       value_error = 0, gradient_error = 0, dp = 0;
  for (int n = 0; n < 2; ++n)
    for (int c = 0; c < 2; ++c) {
      double sum = 0, logsum = 0;
      for (int i = 0; i < 6; ++i) {
        const double a = std::max(double(input[std::size_t((n * 6 + i) * 2 + c)]), double(1e-6f));
        sum += std::pow(a, double(power));
        logsum += std::pow(a, double(power)) * std::log(a);
      }
      const double mean = sum / 6, gm = std::pow(mean, 1 / double(power)), weight = up[std::size_t(n * 2 + c)];
      value_error = std::max(value_error, std::abs(gm - pv[std::size_t(n * 2 + c)]));
      dp += weight * gm * (-std::log(mean) / (double(power) * power) + logsum / (double(power) * sum));
      for (int i = 0; i < 6; ++i) {
        const int    j = (n * 6 + i) * 2 + c;
        const double a = input[std::size_t(j)], expected = a > 1e-6 ? weight * std::pow(a, power - 1) * std::pow(gm, 1 - power) / 6 : 0;
        gradient_error = std::max(gradient_error, std::abs(expected - gx[std::size_t(j)]));
      }
    }
  check::Near(value_error, 0, 2e-7, "every GeM value");
  check::Near(gradient_error, 0, 2e-7, "every signed/clamped GeM activation derivative");
  check::Near(g[1].item(), dp, 2e-7, "independent analytic learned exponent derivative");
  const auto mean1 = GeneralizedMean(x.detach(), nn::from_vector(std::vector<float>{1.f}, {})).to_vector<float>();
  for (int n = 0; n < 2; ++n)
    for (int c = 0; c < 2; ++c) {
      double sum = 0;
      for (int i = 0; i < 6; ++i) sum += std::max(input[std::size_t((n * 6 + i) * 2 + c)], 1e-6f);
      check::Near(mean1[std::size_t(n * 2 + c)], sum / 6, 1e-7, "GeM p1 mean-pooling limit");
    }
  const std::vector<float> values = {3, 4, 0, -2, 1, 2, 0, 0, 0}, weights = {.3f, -.7f, .2f, .3f, -.7f, .2f, .3f, -.7f, .2f};
  Tensor       v = nn::from_vector(values, {3, 3}).requires_grad_();
  const Tensor normalized = DescriptorUnit(v);
  const auto   nv = normalized.to_vector<float>(), ng = nn::grad(nn::sum(normalized * nn::from_vector(weights, {3, 3})), {v})[0].to_vector<float>();
  for (int n = 0; n < 3; ++n) {
    double square = 0, dot = 0;
    for (int c = 0; c < 3; ++c) {
      const double a = values[std::size_t(n * 3 + c)];
      square += a * a;
      dot += a * weights[std::size_t(n * 3 + c)];
    }
    const double norm = std::sqrt(std::max(square, 1e-12));
    for (int c = 0; c < 3; ++c) {
      const double a = values[std::size_t(n * 3 + c)], expected = weights[std::size_t(n * 3 + c)] / norm - (square > 1e-12 ? a * dot / std::pow(norm, 3) : 0);
      check::Near(nv[std::size_t(n * 3 + c)], a / norm, 1e-7, "signed unit or zero descriptor values");
      check::Near(ng[std::size_t(n * 3 + c)], expected, n == 2 ? .1 : 1e-7, "all unit-normalization input derivatives including zero-vector epsilon");
    }
  }
  std::cout << " GeM every value/input derivative max errors=" << value_error << "/" << gradient_error
            << ",analytic p gradient,p1 mean,signed/zero normalization/every derivative passed\n";
}

void MiningRankingChecks() {
  const std::vector<float> f = {0, .3f, .1f, .9f};
  const Tensor             features = nn::from_vector(f, {4, 1});
  const std::vector<int>   labels   = {0, 0, 1, 1};
  const auto               triples  = RetrievalHardTriplets(features, labels);
  const std::vector<TripletIndex> expected = {{0, 1, 2}, {1, 0, 2}, {2, 3, 0}, {3, 2, 1}};
  check::Require(triples == expected, "independent hardest-positive/nearest-negative selection and self exclusion");
  Tensor       x    = features.clone().requires_grad_();
  const Tensor loss = MeanTriplet(x, triples, .2f);
  const auto   g    = nn::grad(loss, {x})[0].to_vector<float>();
  double       value = 0;
  std::array<double, 4> gradient{};
  for (const auto& t : expected) {
    const double a = f[std::size_t(t[0])], p = f[std::size_t(t[1])], n = f[std::size_t(t[2])], hinge = (a - p) * (a - p) - (a - n) * (a - n) + .2f;
    value += std::max(hinge, 0.) / 4;
    if (hinge > 0) {
      gradient[std::size_t(t[0])] += 2 * (n - p) / 4;
      gradient[std::size_t(t[1])] += 2 * (p - a) / 4;
      gradient[std::size_t(t[2])] += 2 * (a - n) / 4;
    }
  }
  check::Near(loss.item(), value, 1e-7, "known frozen-hard-triplet hinge mean");
  for (int i = 0; i < 4; ++i) check::Near(g[std::size_t(i)], gradient[std::size_t(i)], 2e-7, "every mined triplet descriptor derivative");
  const Tensor    gallery = nn::from_vector(std::vector<float>{1, 0, 0, 1, -1, 0, 1, 0}, {4, 2});
  const HostEmbeddings query(nn::from_vector(std::vector<float>{1, 0, 1, 0}, {2, 2}));
  DescriptorIndex index(gallery, {9, 3, 7, 2});
  const auto      ranking = index.Rank(query, 0, 9);
  check::Require(ranking.size() == 3 && ranking[0].id == 2 && ranking[1].id == 3 && ranking[2].id == 7,
                 "exact squared distances,deterministic source-ID ties and self exclusion");
  check::Near(ranking[0].distance, 0, 0, "zero distance");
  check::Near(ranking[1].distance, 2, 0, "orthogonal unit squared distance");
  check::Near(ranking[2].distance, 4, 0, "opposite unit squared distance");
  RetrievalPool queries;
  queries.labels    = {1, 99};
  queries.ids       = {9, 90};
  const auto score = ScoreRetrieval(query, queries, index, {1, 0, 1, 1});
  check::Near(score.map, 5. / 6, 1e-12, "known AP positives at ranks1 and3,exclude self from denominator");
  check::Require(score.eligible == 1 && score.without_positive == 1 && score.hit1 == 1 && score.hit5 == 1,
                 "queries without positives excluded with explicit count");
  const auto tied = index.Rank(query, 0, -1);
  check::Require(tied[0].id == 2 && tied[1].id == 9, "equal distances ordered by source ID,not gallery insertion");
  check::Throws([&] { DescriptorIndex invalid(gallery, {1, 1, 2, 3}); }, "duplicate source IDs rejected");
  std::cout << " exact hard mining/every triplet gradient,known AP5/6,unit cosine-distance relation,ties,self exclusion,no-positive queries and duplicate-ID rejection passed\n";
}

std::int64_t Census(int divisor, int dimensions) {
  const int    widths[] = {64, 128, 256, 512, 512}, counts[] = {2, 2, 3, 3, 3};
  int          in    = 3;
  std::int64_t total = 1;
  for (int block = 0; block < 5; ++block)
    for (int layer = 0; layer < counts[block]; ++layer) {
      const int out = widths[block] / divisor;
      total += 9LL * in * out + out;
      in = out;
    }
  return total + std::int64_t(in) * dimensions;
}

void ModelChecks() {
  for (int divisor : {1, 8}) {
    const int          dimensions = divisor == 1 ? 128 : 32;
    nn::ParameterStore p(227);
    std::vector<float> input(2 * 3072);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(.5 + .4 * std::sin(i * .17));
    Tensor             rgb = nn::from_vector(input, {2, 32, 32, 3}).requires_grad_();
    const auto         net = RetrievalEncoder(p, rgb, divisor, dimensions);
    std::vector<float> up(std::size_t(2 * dimensions));
    for (std::size_t i = 0; i < up.size(); ++i) up[i] = float((int(i) % 13 - 6) * .07f);
    const Tensor objective = nn::sum(net.descriptor * nn::from_vector(up, {2, dimensions}));
    auto         wrt       = p.parameters();
    wrt.push_back(rgb);
    const auto gradients = nn::grad(objective, wrt);
    check::Require(p.count() == Census(divisor, dimensions) && p.size() == 28 && net.stages.size() == 19,
                   "independent full/compact parameter and stage census");
    std::cout << " " << (divisor == 1 ? "full" : "compact") << " retrieval:" << p.count() << " parameters/28 derivatives,19 stages (executed)\n";
    if (divisor == 1) continue;
    std::vector<nn::Shape> shapes;
    const int              widths[] = {8, 16, 32, 64, 64}, counts[] = {2, 2, 3, 3, 3};
    std::int64_t           side = 32;
    for (int block = 0; block < 5; ++block) {
      for (int layer = 0; layer < counts[block]; ++layer) shapes.push_back({2, side, side, widths[block]});
      if (block < 4) {
        side /= 2;
        shapes.push_back({2, side, side, widths[block]});
      }
    }
    shapes.push_back({2, 64});
    shapes.push_back({2, 32});
    for (int i = 0; i < 19; ++i) check::Require(net.stages[std::size_t(i)].shape() == shapes[std::size_t(i)], "all convolution/pool/GeM/projection shapes");
    check::Near(net.exponent.item(), 3, 3e-7, "learnable exponent initialized3");
    double minimum = INFINITY, maximum = 0;
    for (const auto& g : gradients) {
      double norm = 0;
      for (float value : g.to_vector<float>()) {
        check::Require(std::isfinite(value), "finite descriptor model/image gradient");
        norm += double(value) * value;
      }
      norm = std::sqrt(norm);
      check::Require(norm > 0, "all28 model tensors including rho/projection and RGB receive gradients");
      minimum = std::min(minimum, norm), maximum = std::max(maximum, norm);
    }
    std::cout << " compact all28 parameter and RGB gradients finite/nonzero,norm range[" << minimum << "," << maximum << "],all19 shapes passed\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs      = 3;
    bool                  checks_only = false;
    std::filesystem::path root        = paths::Dataset("cifar10");
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
    PoolingChecks();
    MiningRankingChecks();
    ModelChecks();
    if (checks_only) return 0;
    TrainRetrieval(root, epochs, paths::OutputDir("models/image_retrieval"));
    std::cout << "All CNN descriptor retrieval checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
