/// @file pose.cpp
/// @brief Native pose tensor preparation from verified source metadata.
#ifdef DATAMUNGE_ENABLE_DATA_ASSETS
#include <datamunge/image/jpeg.hpp>
#include <dm/dm.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

#include "preparation.hpp"

/// @brief Private source ingress and tensor preparation helpers.
namespace {
namespace n = dm::nn;
/// @brief Own original decoded RGB and unmodified YOLO person annotation ingress.
struct SourceFrame {
  n::Tensor rgb;         ///< Original Float32[1,H,W,3] byte-valued JPEG decode.
  n::Tensor annotations; ///< Double[P,56] class,normalized box and17 triples.
  int       image_id;    ///< Original COCO identity.
};
/// @brief Decode original JPEG bytes and parse official text annotation values.
/// @param image Original JPEG path.
/// @param labels Original YOLO pose annotation path.
/// @param id Original COCO image ID.
/// @return Owning image/annotation ingress without resize or coordinate arithmetic.
SourceFrame
ReadFrame(const std::filesystem::path& image, const std::filesystem::path& labels, int id) {
  dm::image::JpegDecodeOptions options;
  options.channels = 3;
  const auto rgb = n::expand_dims(n::from_image(dm::image::read_jpeg(image.string(), options)), 0);
  std::ifstream       stream(labels);
  std::vector<double> values;
  std::string         line;
  int                 count = 0;
  while (std::getline(stream, line)) {
    std::istringstream  row(line);
    std::vector<double> fields;
    double              value;
    while (row >> value)
      fields.push_back(value);
    if (fields.size() != 56)
      throw std::invalid_argument("COCO8-Pose expects56 annotation fields");
    values.insert(values.end(), fields.begin(), fields.end());
    ++count;
  }
  if (count < 1 || count > 8)
    throw std::invalid_argument("one to eight original annotated people required");
  const auto annotations = n::from_vector(values, {count, 56});
  if (!n::all(n::is_finite(annotations)).item<bool>() ||
      !n::all(n::equal(n::slice_axis(annotations, 1, 0, 1), 0.)).item<bool>())
    throw std::invalid_argument("finite class0 original pose annotations required");
  return {rgb, annotations, id};
}
/// @brief Convert normalized official17-joint rows to an explicit128px observation frame.
/// @param raw Double[P,17,3] original normalized x/y and visibility0/1/2.
/// @param xy Transformed Double[P,17,2] coordinates.
/// @return Float32[P,17,3],unlabeled coordinates0,labeled coordinates clipped0..127.
n::Tensor Joints(const n::Tensor& raw, const n::Tensor& xy) {
  const auto visibility = n::slice_axis(raw, 2, 2, 3);
  return n::concat({n::where(n::greater(visibility, 0.), n::clip(xy, 0., 127.), 0.), visibility}, 2)
      .to(n::DType::Float32);
}
/// @brief Flip COCO17 annotations and exchange anatomical left/right identities.
/// @param joints Float32[...,17,3] bounded coordinates and original visibility.
/// @return Same geometry with x=127-x at labeled joints and correct left/right permutation.
n::Tensor MirrorJoints(const n::Tensor& joints) {
  const auto axis    = joints.dim() - 2;
  const auto indices = n::from_vector(
      std::vector<std::int64_t>{0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15}, {17});
  const auto swapped = n::gather(joints, indices, axis);
  const auto x       = n::slice_axis(swapped, swapped.dim() - 1, 0, 1),
             y       = n::slice_axis(swapped, swapped.dim() - 1, 1, 2),
             v       = n::slice_axis(swapped, swapped.dim() - 1, 2, 3);
  return n::concat({n::where(n::greater(v, 0.), 127. - x, 0.), y, v}, swapped.dim() - 1);
}
/// @brief Append one tensor row to its future named archive field.
/// @param parts Mutable tensor-row registry.
/// @param name Exact named archive member.
/// @param value Tensor row retaining its leading observation axis.
void Append(
    std::map<std::string, std::vector<n::Tensor>>& parts,
    const std::string&                             name,
    const n::Tensor&                               value) {
  parts[name].push_back(value);
}
/// @brief Prepare one original full image and its optional fitting-only mirror.
/// @param frame Original image and annotation ingress.
/// @param split Archive split prefix.
/// @param parts Mutable named tensor rows.
/// @param mirror Append the anatomical horizontal mirror if true.
void Full(
    const SourceFrame&                             frame,
    const std::string&                             split,
    std::map<std::string, std::vector<n::Tensor>>& parts,
    bool                                           mirror) {
  const auto count  = frame.annotations.size(0);
  const auto raw    = n::slice_axis(frame.annotations, 1, 5, 56).reshape({count, 17, 3});
  const auto joints = Joints(raw, n::slice_axis(raw, 2, 0, 2) * 128.);
  const auto rgb    = n::resize_bilinear(frame.rgb, {128, 128}) / 255.;
  const auto boxes  = (n::slice_axis(frame.annotations, 1, 1, 5) * 128.).to(n::DType::Float32);
  const auto padded_joints =
                 n::pad(n::expand_dims(joints, 0), {{0, 0}, {0, 8 - count}, {0, 0}, {0, 0}}),
             padded_boxes = n::pad(n::expand_dims(boxes, 0), {{0, 0}, {0, 8 - count}, {0, 0}});
  for (int flip = 0; flip <= (mirror ? 1 : 0); ++flip) {
    const auto joint = flip ? MirrorJoints(padded_joints) : padded_joints;
    auto       box   = padded_boxes;
    if (flip) {
      const auto present = n::arange(0, 8).reshape({1, 8, 1});
      const auto x       = n::slice_axis(box, 2, 0, 1);
      box                = n::concat(
          {n::where(n::less(present, double(count)), 127. - x, 0.), n::slice_axis(box, 2, 1, 4)},
          2);
    }
    Append(parts, split + "_full_rgb", flip ? n::reverse(rgb, {2}) : rgb);
    Append(parts, split + "_full_joints", joint);
    Append(parts, split + "_full_boxes", box);
    Append(
        parts,
        split + "_full_counts",
        n::from_vector(std::vector<std::int32_t>{std::int32_t(count)}, {1}));
    Append(
        parts,
        split + "_full_image_ids",
        n::from_vector(std::vector<std::int32_t>{frame.image_id}, {1}));
  }
}
/// @brief Prepare original oracle person crops and optional fitting-only mirrors.
/// @param frame Original RGB and normalized person annotations.
/// @param split Named archive split.
/// @param parts Mutable named tensor rows.
/// @param mirror Append anatomical mirrors only for fitting.
/// @param provenance Record integer crop geometry and source-person identity.
void Crops(
    const SourceFrame&                             frame,
    const std::string&                             split,
    std::map<std::string, std::vector<n::Tensor>>& parts,
    bool                                           mirror,
    std::ostream&                                  provenance) {
  const auto h = frame.rgb.size(1), w = frame.rgb.size(2);
  for (std::int64_t person = 0; person < frame.annotations.size(0); ++person) {
    const auto annotation = n::select(frame.annotations, 0, person),
               box        = n::slice_axis(annotation, 0, 1, 5);
    const auto cx = n::select(box, 0, 0) * double(w), cy = n::select(box, 0, 1) * double(h),
               bw = n::select(box, 0, 2) * double(w), bh = n::select(box, 0, 3) * double(h);
    const auto
        left   = n::clip(n::floor(cx - .625 * bw), 0., double(w - 1)).item<std::int64_t>(),
        top    = n::clip(n::floor(cy - .625 * bh), 0., double(h - 1)).item<std::int64_t>(),
        right  = n::clip(n::ceil(cx + .625 * bw), double(left + 1), double(w)).item<std::int64_t>(),
        bottom = n::clip(n::ceil(cy + .625 * bh), double(top + 1), double(h)).item<std::int64_t>();
    const auto raw    = n::slice_axis(annotation, 0, 5, 56).reshape({1, 17, 3}),
               xy     = n::slice_axis(raw, 2, 0, 2);
    const auto scale  = n::from_vector(std::vector<double>{double(w), double(h)}, {2}),
               offset = n::from_vector(std::vector<double>{double(left), double(top)}, {2}),
               extent = n::from_vector(
                   std::vector<double>{double(right - left), double(bottom - top)}, {2});
    const auto joints = Joints(raw, (xy * scale - offset) * 128. / extent),
               rgb    = n::resize_bilinear(
                         n::slice(frame.rgb, {0, top, left, 0}, {1, bottom - top, right - left, 3}),
                         {128, 128}) /
                     255.,
               size =
                   n::stack({bw * 128. / double(right - left), bh * 128. / double(bottom - top)}, 0)
                       .reshape({1, 2})
                       .to(n::DType::Float32);
    const int pid = frame.image_id * 10 + int(person);
    for (int flip = 0; flip <= (mirror ? 1 : 0); ++flip) {
      Append(parts, split + "_rgb", flip ? n::reverse(rgb, {2}) : rgb);
      Append(parts, split + "_joints", flip ? MirrorJoints(joints) : joints);
      Append(parts, split + "_box_size", size);
      Append(
          parts,
          split + "_image_ids",
          n::from_vector(std::vector<std::int32_t>{frame.image_id}, {1}));
      Append(parts, split + "_person_ids", n::from_vector(std::vector<std::int32_t>{pid}, {1}));
      provenance << split << '\t' << frame.image_id << '\t' << pid << '\t' << left << '\t' << top
                 << '\t' << right << '\t' << bottom << '\t' << flip << '\n';
    }
  }
}
} // namespace
/// @brief Prepare tensor128px original full-image and oracle-person COCO8-Pose arrays.
/// @param argc Program name and source/cache root.
/// @param argv Root containing extracted sources and source_files.tsv.
/// @return Zero after named Float32/Int32 archive and exact crop provenance are saved.
int datamunge::datasets::detail::prepare_pose(int argc, char** argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("expected pose source/cache root");
    n::set_num_threads(2);
    n::NoGradGuard                                guard;
    const std::filesystem::path                   root = argv[1];
    std::ifstream                                 files(root / "source_files.tsv");
    std::ofstream                                 provenance(root / "crops.tsv");
    std::map<std::string, std::vector<n::Tensor>> parts;
    std::string                                   split, image, label;
    int                                           id;
    int                                           count = 0;
    while (files >> split >> id >> image >> label) {
      const auto source = ReadFrame(root / image, root / label, id);
      Full(source, split, parts, split == "fit");
      Crops(source, split, parts, split == "fit", provenance);
      ++count;
    }
    if (count != 8)
      throw std::invalid_argument("expected eight original pose source images");
    n::NamedTensors arrays;
    for (const auto& [name, rows] : parts)
      arrays.push_back({name, n::concat(rows, 0)});
    n::save_npz((root / "poses.npz").string(), arrays);
    if (!provenance)
      throw std::runtime_error("pose provenance write failed");
    std::cout << "Prepared genuine COCO8-Pose:12/1/14 oracle person crops and6/1/4 full-image "
                 "observations,source-image-disjoint;fitting-only anatomical mirrors.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

#endif // DATAMUNGE_ENABLE_DATA_ASSETS
