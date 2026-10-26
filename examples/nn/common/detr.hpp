// DETR: ResNet encoder, 2D sine positions and learned object-query decoder,
// with the bipartite set loss (port of tflow's common/nn/detr.h and
// common/ml/detr.h).
// https://arxiv.org/abs/2005.12872
// https://github.com/facebookresearch/detr
//
// Model: original post-norm ordering: positional vectors affect attention
// Q/K only; values remain unmodified. The decoder starts from zero contents
// and injects learned query positions into Q/K. Shared class and three-layer
// box heads supervise each normalized decoder output. Full defaults:
// ResNet-50, six encoder/six decoder layers, 256 dimensions, eight heads,
// FF2048, 100 queries. Backbone BN uses the shared train/infer library rather
// than pretrained frozen BN. Dropout masks draw from the parameter store's
// generator (tflow: seeded TF dropout ops).
//
// Loss: detached Hungarian assignments per decoder layer with cost
// -class_probability + 5*L1 - 2*GIoU, weighted class cross entropy (no-object
// .1), 5*L1 and 2*GIoU, summed over decoder layers. Boxes use normalized
// (cy,cx,h,w); loss normalization uses total objects, including empty images
// safely. Each query predicts one class and box; inference requires neither
// anchors nor NMS.
#pragma once

#include "common/detection_scenes.hpp"
#include "common/hungarian.hpp"
#include "common/resnet.hpp"

namespace detr {

namespace nn = datamunge::nn;
using detector::Box;
using detector::DetectionScenes;

/// Dense layer over the last axis of [..., in] ("<name>/W" glorot, "<name>/b").
inline nn::Tensor DetrLinear(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x, std::int64_t out) {
  return nn::dense(p, name, x, out);
}
/// Normalized 2D sine positions [1, height*width, dimension]: the first half
/// encodes rows, the second columns.
inline nn::Tensor DetrPositions(int height, int width, int dimension) {
  if (height < 1 || width < 1 || dimension < 4 || dimension % 4)
    throw std::invalid_argument("DETR positions need positive grids and dimension divisible by four");
  std::vector<float> result(std::size_t(height * width * dimension));
  const int          half = dimension / 2;
  const double       tau  = 2 * std::acos(-1.);
  for (int r = 0; r < height; ++r)
    for (int c = 0; c < width; ++c)
      for (int k = 0; k < half; ++k) {
        const double scale = std::pow(10000., 2 * (k / 2) / double(half));
        const double y = (r + 1) * tau / (height + 1e-6) / scale, x = (c + 1) * tau / (width + 1e-6) / scale;
        const auto   at = std::size_t((r * width + c) * dimension + k);
        result[at]      = float(k % 2 ? std::cos(y) : std::sin(y));
        result[at + std::size_t(half)] = float(k % 2 ? std::cos(x) : std::sin(x));
      }
  return nn::from_vector(result, {1, height * width, dimension});
}

inline nn::Tensor DetrAttention(nn::ParameterStore& p, const std::string& name, const nn::Tensor& query,
                                const nn::Tensor& key, const nn::Tensor& value, int dimension, int heads, double dropout,
                                bool training) {
  const std::int64_t head = dimension / heads;
  auto project = [&](const std::string& part, const nn::Tensor& x) {
    const auto flat = DetrLinear(p, name + "/" + part, x, dimension);
    return nn::transpose(nn::reshape(flat, {x.shape()[0], x.shape()[1], heads, head}), {0, 2, 1, 3});
  };
  const auto q = project("q", query), k = project("k", key), v = project("v", value);
  auto       weights = nn::softmax(nn::matmul(q, k, false, true) * (1. / std::sqrt(double(head))), -1);
  if (training && dropout > 0) weights = nn::dropout(weights, dropout, true, p.generator());
  const auto merged = nn::transpose(nn::matmul(weights, v), {0, 2, 1, 3});
  return DetrLinear(p, name + "/output", nn::reshape(merged, query.shape()), dimension);
}

struct DetrConfig {
  int    classes = 80, image_size = 640, backbone_depth = 50, width_divisor = 1, dimension = 256, heads = 8,
      feedforward = 2048, encoder_layers = 6, decoder_layers = 6, queries = 100;
  double dropout = .1;
  bool   grayscale = false, small_stem = false;
  bool   update_stats = true;
};
struct DetrOutputs {
  std::vector<nn::Tensor> logits, boxes;  // per decoder layer: [N, Q, classes + 1], [N, Q, 4]
  nn::Tensor              memory;
};
inline DetrOutputs Detr(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& images, const DetrConfig& cfg = {},
                        bool training = false) {
  if (cfg.classes < 1 || cfg.image_size < 32 || cfg.image_size % 32 || cfg.dimension < 4 || cfg.dimension % 4 ||
      cfg.heads < 1 || cfg.dimension % cfg.heads || cfg.encoder_layers < 1 || cfg.decoder_layers < 1 ||
      cfg.feedforward < 1 || cfg.queries < 1 || !(cfg.dropout >= 0 && cfg.dropout < 1))
    throw std::invalid_argument("invalid DETR configuration");
  resnet::ResNetConfig backbone;
  backbone.depth              = cfg.backbone_depth;
  backbone.width_divisor      = cfg.width_divisor;
  backbone.include_classifier = false;
  backbone.small_stem         = cfg.small_stem;
  backbone.stride_on_3x3      = true;
  backbone.update_stats       = cfg.update_stats;
  const auto features = resnet::ResNet(p, state, cfg.grayscale ? nn::tile(images, {1, 1, 1, 3}) : images, backbone, training);
  const int  side     = cfg.image_size / (cfg.small_stem ? 8 : 32);
  if (features.stages[3].shape()[1] != side) throw std::invalid_argument("DETR image_size does not match the input images");
  DetrOutputs result;
  nn::Tensor  memory = nn::reshape(nn::conv2d(p, "input_projection", features.stages[3], cfg.dimension, 1), {-1, side * side, cfg.dimension});
  const auto  position = DetrPositions(side, side, cfg.dimension);
  auto drop = [&](const nn::Tensor& x) { return training && cfg.dropout > 0 ? nn::dropout(x, cfg.dropout, true, p.generator()) : x; };
  auto ffn  = [&](const std::string& name, const nn::Tensor& x) {
    const auto h = nn::relu(DetrLinear(p, name + "/linear1", x, cfg.feedforward));
    return DetrLinear(p, name + "/linear2", drop(h), cfg.dimension);
  };
  for (int layer = 0; layer < cfg.encoder_layers; ++layer) {
    const auto name       = "encoder" + std::to_string(layer);
    const auto positioned = memory + position;
    const auto attention  = DetrAttention(p, name + "/attention", positioned, positioned, memory, cfg.dimension, cfg.heads, cfg.dropout, training);
    memory = nn::layer_norm(p, name + "/norm1", memory + drop(attention));
    memory = nn::layer_norm(p, name + "/norm2", memory + drop(ffn(name + "/ffn", memory)));
  }
  result.memory     = memory;
  const auto query  = p.get("query_positions", {1, cfg.queries, cfg.dimension}, nn::init::normal(1.0));
  nn::Tensor contents = nn::zeros({images.shape()[0], cfg.queries, cfg.dimension});
  for (int layer = 0; layer < cfg.decoder_layers; ++layer) {
    const auto name = "decoder" + std::to_string(layer);
    const auto q    = contents + query;
    const auto self = DetrAttention(p, name + "/self", q, q, contents, cfg.dimension, cfg.heads, cfg.dropout, training);
    contents        = nn::layer_norm(p, name + "/norm1", contents + drop(self));
    const auto cross = DetrAttention(p, name + "/cross", contents + query, memory + position, memory, cfg.dimension, cfg.heads,
                                     cfg.dropout, training);
    contents = nn::layer_norm(p, name + "/norm2", contents + drop(cross));
    contents = nn::layer_norm(p, name + "/norm3", contents + drop(ffn(name + "/ffn", contents)));
    const auto normalized = nn::layer_norm(p, "decoder_norm", contents);
    result.logits.push_back(DetrLinear(p, "class_head", normalized, cfg.classes + 1));
    nn::Tensor box = nn::relu(DetrLinear(p, "box_head/layer1", normalized, cfg.dimension));
    box            = nn::relu(DetrLinear(p, "box_head/layer2", box, cfg.dimension));
    result.boxes.push_back(nn::sigmoid(DetrLinear(p, "box_head/layer3", box, 4)));
  }
  return result;
}

// ---- set loss, matching and decoding ----------------------------------------------

inline double GeneralizedIou(const Box& a, const Box& b) {
  const double intersection = std::max(0.f, std::min(a[2], b[2]) - std::max(a[0], b[0])) *
                              std::max(0.f, std::min(a[3], b[3]) - std::max(a[1], b[1]));
  const double combined  = detector::BoxArea(a) + detector::BoxArea(b) - intersection;
  const double enclosing = (std::max(a[2], b[2]) - std::min(a[0], b[0])) * (std::max(a[3], b[3]) - std::min(a[1], b[1]));
  return intersection / std::max(combined, 1e-7) - (enclosing - combined) / std::max(enclosing, 1e-7);
}
/// Elementwise GIoU of centre-format (cy, cx, h, w) boxes [..., 4] -> [...].
inline nn::Tensor GeneralizedIou(const nn::Tensor& prediction, const nn::Tensor& target) {
  const auto p = nn::unstack(prediction, -1), t = nn::unstack(target, -1);
  const auto py0 = p[0] - p[2] * .5, px0 = p[1] - p[3] * .5, py1 = p[0] + p[2] * .5, px1 = p[1] + p[3] * .5;
  const auto ty0 = t[0] - t[2] * .5, tx0 = t[1] - t[3] * .5, ty1 = t[0] + t[2] * .5, tx1 = t[1] + t[3] * .5;
  const auto zero         = nn::scalar(0.0);
  const auto ih           = nn::maximum(nn::minimum(py1, ty1) - nn::maximum(py0, ty0), zero);
  const auto iw           = nn::maximum(nn::minimum(px1, tx1) - nn::maximum(px0, tx0), zero);
  const auto intersection = ih * iw;
  const auto combined     = p[2] * p[3] + t[2] * t[3] - intersection;
  const auto enclosing    = (nn::maximum(py1, ty1) - nn::minimum(py0, ty0)) * (nn::maximum(px1, tx1) - nn::minimum(px0, tx0));
  const auto eps          = nn::scalar(1e-7);
  return intersection / nn::maximum(combined, eps) - (enclosing - combined) / nn::maximum(enclosing, eps);
}

/// Targets of one decoder layer: one-hot labels [N, Q, C+1] (no-object
/// last), normalized boxes [N, Q, 4] and matched-query mask [N, Q].
struct DetrTargets {
  std::vector<float> labels, boxes, positive;
  std::int64_t       batch = 0, queries = 0, classes = 0;
};
inline nn::Tensor DetrLayerLoss(const nn::Tensor& logits, const nn::Tensor& boxes, const DetrTargets& t) {
  const auto labels   = nn::from_vector(t.labels, {t.batch, t.queries, t.classes + 1});
  const auto target   = nn::from_vector(t.boxes, {t.batch, t.queries, 4});
  const auto positive = nn::from_vector(t.positive, {t.batch, t.queries});
  const auto log_prob = nn::log_softmax(logits, -1);
  const auto weight   = positive * .9 + .1;
  const auto ce       = -nn::sum(labels * log_prob, {2});
  const auto classification = nn::sum(ce * weight) / nn::maximum(nn::sum(weight), nn::scalar(1e-7));
  const auto l1       = nn::sum(nn::abs(boxes - target), {2});
  const auto giou     = GeneralizedIou(boxes, target);
  const auto regression = nn::sum(positive * (l1 * 5.0 + (1.0 - giou) * 2.0)) / nn::maximum(nn::sum(positive), nn::scalar(1.0));
  return classification + regression;
}

inline Box DetrCorners(const float* b) { return {b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2}; }

/// Hungarian targets of one layer from its (detached) logits [N, Q, C+1] and boxes [N, Q, 4].
inline DetrTargets MatchDetrTargets(const std::vector<float>& logits, const std::vector<float>& predictions,
                                    std::int64_t queries, const DetectionScenes& scenes, const std::vector<int>& rows,
                                    int classes) {
  DetrTargets t;
  t.batch   = std::int64_t(rows.size());
  t.queries = queries;
  t.classes = classes;
  const auto Q = std::size_t(queries), C1 = std::size_t(classes + 1);
  t.labels.assign(rows.size() * Q * C1, 0.f);
  t.boxes.assign(rows.size() * Q * 4, 0.f);
  t.positive.assign(rows.size() * Q, 0.f);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto&                       truth = scenes.boxes[std::size_t(rows[i])];
    std::vector<std::vector<double>>  cost(truth.size(), std::vector<double>(Q));
    std::vector<std::array<float, 4>> normalized;
    for (const auto& b : truth)
      normalized.push_back({(b[0] + b[2]) / float(2 * scenes.height), (b[1] + b[3]) / float(2 * scenes.width),
                            (b[2] - b[0]) / float(scenes.height), (b[3] - b[1]) / float(scenes.width)});
    for (std::size_t q = 0; q < Q; ++q) {
      t.labels[(i * Q + q) * C1 + std::size_t(classes)] = 1;
      const float* raw     = logits.data() + (i * Q + q) * C1;
      const double maximum = *std::max_element(raw, raw + C1);
      double       denominator = 0;
      for (std::size_t c = 0; c < C1; ++c) denominator += std::exp(raw[c] - maximum);
      const float* pred = predictions.data() + (i * Q + q) * 4;
      for (std::size_t g = 0; g < truth.size(); ++g) {
        const int label = scenes.labels[std::size_t(rows[i])][g];
        if (label < 0 || label >= classes) throw std::invalid_argument("DETR target outside class range");
        double l1 = 0;
        for (std::size_t k = 0; k < 4; ++k) l1 += std::abs(pred[k] - normalized[g][k]);
        cost[g][q] = -std::exp(raw[label] - maximum) / denominator + 5 * l1 -
                     2 * GeneralizedIou(DetrCorners(pred), DetrCorners(normalized[g].data()));
      }
    }
    const auto assignment = hungarian::HungarianAssignment(cost);
    for (std::size_t g = 0; g < truth.size(); ++g) {
      const auto q     = std::size_t(assignment[g]);
      const int  label = scenes.labels[std::size_t(rows[i])][g];
      t.labels[(i * Q + q) * C1 + std::size_t(classes)] = 0;
      t.labels[(i * Q + q) * C1 + std::size_t(label)] = t.positive[i * Q + q] = 1;
      for (std::size_t k = 0; k < 4; ++k) t.boxes[(i * Q + q) * 4 + k] = normalized[g][k];
    }
  }
  return t;
}

/// Sum over decoder layers of the layer losses, each layer matched independently on detached outputs.
inline nn::Tensor DetrLoss(const DetrOutputs& net, const DetectionScenes& scenes, const std::vector<int>& rows, int classes) {
  nn::Tensor total;
  for (std::size_t layer = 0; layer < net.logits.size(); ++layer) {
    const auto targets = MatchDetrTargets(net.logits[layer].to_vector<float>(), net.boxes[layer].to_vector<float>(),
                                          net.logits[layer].shape()[1], scenes, rows, classes);
    const auto loss = DetrLayerLoss(net.logits[layer], net.boxes[layer], targets);
    total           = layer ? total + loss : loss;
  }
  return total;
}

/// One best foreground class per query (score = its softmax probability, kept above .01).
inline std::vector<detector::Detection> DecodeDetr(const nn::Tensor& logits, const nn::Tensor& boxes,
                                                   const DetectionScenes& scenes, const std::vector<int>& rows, int classes) {
  const auto                       L = logits.to_vector<float>(), B = boxes.to_vector<float>();
  const auto                       Q = std::size_t(logits.shape()[1]), C1 = std::size_t(classes + 1);
  std::vector<detector::Detection> result;
  for (std::size_t i = 0; i < rows.size(); ++i)
    for (std::size_t q = 0; q < Q; ++q) {
      const float* raw     = L.data() + (i * Q + q) * C1;
      const int    label   = int(std::max_element(raw, raw + classes) - raw);
      const float  maximum = *std::max_element(raw, raw + C1);
      double       denominator = 0;
      for (std::size_t c = 0; c < C1; ++c) denominator += std::exp(raw[c] - maximum);
      const float score = float(std::exp(raw[label] - maximum) / denominator);
      if (score < .01f) continue;
      auto box = DetrCorners(B.data() + (i * Q + q) * 4);
      box      = detector::ClipBox({box[0] * float(scenes.height), box[1] * float(scenes.width), box[2] * float(scenes.height),
                                    box[3] * float(scenes.width)},
                                   float(scenes.height), float(scenes.width));
      if (detector::BoxArea(box) > 0) result.push_back({rows[i], label, score, box});
    }
  return result;
}
inline std::vector<detector::Detection> DetectDetr(nn::ParameterStore& p, nn::ParameterStore& state, const DetrConfig& cfg,
                                                   const DetectionScenes& scenes, const std::vector<int>& rows, int classes) {
  nn::NoGradGuard                  guard;
  std::vector<detector::Detection> result;
  for (std::size_t start = 0; start < rows.size(); start += 32) {
    const std::vector<int> chunk(rows.begin() + std::ptrdiff_t(start), rows.begin() + std::ptrdiff_t(std::min(rows.size(), start + 32)));
    const auto             net   = Detr(p, state, detector::SceneImages(scenes, chunk), cfg, false);
    const auto             found = DecodeDetr(net.logits.back(), net.boxes.back(), scenes, chunk, classes);
    result.insert(result.end(), found.begin(), found.end());
  }
  return result;
}

}  // namespace detr
