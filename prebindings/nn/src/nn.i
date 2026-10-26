%module nn_native
%naturalvar;
%include <stdint.i>
namespace std {
  typedef ::int64_t int64_t;
  typedef ::uint64_t uint64_t;
  typedef ::int32_t int32_t;
  typedef ::uint32_t uint32_t;
}
%include <std_string.i>
%include <std_vector.i>
%include <std_pair.i>
#ifdef SWIGPYTHON
%include <std_array.i>
#else
// Scalar integer literals are valid C++ double arguments. R's default
// SWIG typemap uses REAL() directly and otherwise rejects values like 2L.
%typemap(in) double {
  if ((!Rf_isReal($input) && !Rf_isInteger($input)) || Rf_length($input) != 1)
    SWIG_exception_fail(SWIG_TypeError, "expected one numeric scalar");
  $1 = Rf_asReal($input);
}
%typemap(scoercein) double "$input = as.numeric($input);";
// R has no std_array.i: use ordinary numeric vectors for fixed-size options.
%typemap(scoerceout) const std::vector<std::string>& "";
%typemap(rtype) const std::vector<std::string>& "character";
%define NN_R_ARRAY(N)
%typemap(rtype) std::array<std::int64_t, N> "numeric";
%typemap(scoercein) std::array<std::int64_t, N> "$input = as.numeric($input);";
%typemap(scoerceout) std::array<std::int64_t, N> "";
%typemap(rtypecheck) std::array<std::int64_t, N> "is.numeric($arg) && length($arg) == N";
%typemap(in) std::array<std::int64_t, N> {
  if ((!Rf_isInteger($input) && !Rf_isReal($input)) || Rf_length($input) != N)
    SWIG_exception_fail(SWIG_TypeError, "expected numeric vector of length " #N);
  for (int i = 0; i < N; ++i)
    $1[i] = static_cast<std::int64_t>(Rf_isInteger($input) ? INTEGER($input)[i] : REAL($input)[i]);
}
%typemap(out) std::array<std::int64_t, N> {
  $result = PROTECT(Rf_allocVector(REALSXP, N));
  for (int i = 0; i < N; ++i) REAL($result)[i] = $1[i];
  UNPROTECT(1);
}
%enddef
NN_R_ARRAY(2)
NN_R_ARRAY(3)
NN_R_ARRAY(4)
NN_R_ARRAY(6)
%ignore datamunge::nn::NoGradGuard;
%ignore datamunge::nn::EnableGradGuard;
%rename(CoreTensor) datamunge::nn::Tensor;
#endif
%include <exception.i>
%exception {
  try { $action }
  catch (const std::exception& e) { SWIG_exception(SWIG_RuntimeError, e.what()); }
}
%{
#include "datamunge/nn/nn.hpp"
#include "datamunge/nn/mt19937.hpp"
namespace datamunge::nn {
using EarlyStoppingDecision = EarlyStopping::Decision;
}
%}
%ignore datamunge::nn::Storage;
%ignore datamunge::nn::TensorImpl;
%ignore datamunge::nn::detail;
%ignore datamunge::nn::Tensor::Tensor(std::shared_ptr<datamunge::nn::TensorImpl>);
%ignore datamunge::nn::Tensor::impl;
%ignore datamunge::nn::Tensor::impl_ptr;
%ignore datamunge::nn::operator<<;
%ignore datamunge::nn::Node;
%ignore datamunge::nn::AccumulateGrad;
%ignore datamunge::nn::Edge;
%ignore datamunge::nn::TensorMeta;
%ignore datamunge::nn::gradient_edge;
%ignore datamunge::nn::record;
%ignore datamunge::nn::apply_function;
%ignore datamunge::nn::Function;
%ignore datamunge::nn::checkpoint;
%ignore datamunge::nn::checkpoint_sequential;
%ignore datamunge::nn::ReversibleBlock;
%ignore datamunge::nn::reversible_sequence;
%ignore datamunge::nn::log_epoch;
%template(ShapeVector) std::vector<std::int64_t>;
%template(DoubleVector) std::vector<double>;
%template(FloatVector) std::vector<float>;
%template(StringVector) std::vector<std::string>;
#ifdef SWIGPYTHON
%template(Int64Pair) std::array<std::int64_t, 2>;
%template(Int64Triple) std::array<std::int64_t, 3>;
%template(Int64Quad) std::array<std::int64_t, 4>;
%template(Int64Six) std::array<std::int64_t, 6>;
#endif
%include "datamunge/nn/dtype.hpp"
// Tensor is a reference-counted value handle. Returning a borrowed C++ Tensor&
// from a fluent mutation can outlive a temporary Python/R receiver. Return an
// owned handle sharing the same TensorImpl, preserving aliasing and gradients.
%typemap(out) datamunge::nn::Tensor & {
  $result = SWIG_NewPointerObj(new datamunge::nn::Tensor(*$1),
      $descriptor(datamunge::nn::Tensor *), SWIG_POINTER_OWN);
}
%include "datamunge/nn/tensor.hpp"
%template(TensorVector) std::vector<datamunge::nn::Tensor>;
%extend datamunge::nn::Tensor {
  double item() const { return $self->item<double>(); }
  std::vector<double> values() const { return $self->to_vector<double>(); }
  std::string __repr__() const { return $self->str(); }
}
%template(from_vector) datamunge::nn::from_vector<float>;
%template(from_vector_double) datamunge::nn::from_vector<double>;
%template(from_vector_int64) datamunge::nn::from_vector<std::int64_t>;
%include "datamunge/nn/ops.hpp"
%include "datamunge/nn/autograd.hpp"
%include "datamunge/nn/random.hpp"
%rename(init_constant) datamunge::nn::init::constant;
%rename(init_glorot_normal) datamunge::nn::init::glorot_normal;
%rename(init_glorot_uniform) datamunge::nn::init::glorot_uniform;
%rename(init_he_normal) datamunge::nn::init::he_normal;
%rename(init_he_uniform) datamunge::nn::init::he_uniform;
%rename(init_lecun_normal) datamunge::nn::init::lecun_normal;
%rename(init_lecun_uniform) datamunge::nn::init::lecun_uniform;
%rename(init_normal) datamunge::nn::init::normal;
%rename(init_ones) datamunge::nn::init::ones;
%rename(init_orthogonal) datamunge::nn::init::orthogonal;
%rename(init_truncated_normal) datamunge::nn::init::truncated_normal;
%rename(init_uniform) datamunge::nn::init::uniform;
%rename(init_zeros) datamunge::nn::init::zeros;
namespace std {
  template<typename Signature> class function {
  public:
    ~function();
  };
}
%template(Initializer) std::function<datamunge::nn::Tensor(const datamunge::nn::Shape&, datamunge::nn::DType, datamunge::nn::Generator&)>;
%include "datamunge/nn/params.hpp"
%include "datamunge/nn/conv.hpp"
%include "datamunge/nn/layers.hpp"
%template(TensorPair) std::pair<datamunge::nn::Tensor, datamunge::nn::Tensor>;
%include "datamunge/nn/attention.hpp"
%include "datamunge/nn/losses.hpp"
%template(ClippedGradients) std::pair<std::vector<datamunge::nn::Tensor>, double>;
%include "datamunge/nn/optim.hpp"
%rename(schedule_step_decay) datamunge::nn::schedule::step_decay;
%rename(schedule_exponential) datamunge::nn::schedule::exponential;
%rename(schedule_cosine) datamunge::nn::schedule::cosine;
%rename(schedule_warmup) datamunge::nn::schedule::warmup;
%rename(schedule_warmup_cosine) datamunge::nn::schedule::warmup_cosine;
%rename(schedule_one_cycle) datamunge::nn::schedule::one_cycle;
%rename(schedule_cyclical) datamunge::nn::schedule::cyclical;
%include "datamunge/nn/schedules.hpp"
// BatchOrder's private std::mt19937_64 state is intentionally opaque to SWIG.
namespace datamunge::nn {
class BatchOrder {
public:
  BatchOrder(std::size_t samples, std::size_t batch_size, std::uint64_t seed,
             bool shuffle = true, bool drop_last = false);
  void begin_epoch();
  std::vector<std::int64_t> next();
  std::size_t batches_per_epoch() const;
  std::int64_t epoch() const;
  std::size_t cursor() const;
  std::string save_state() const;
  void restore_state(const std::string& state);
};
Tensor take_rows(const Tensor& x, const std::vector<std::int64_t>& indices);
class WeightedMean {
public:
  void add(double mean, std::size_t samples);
  double value() const;
  std::size_t count() const;
  void reset();
};
// Alias the existing nested value type: SWIG cannot expose nested structs.
struct EarlyStoppingDecision {
  bool improved;
  bool stop;
};
class EarlyStopping {
public:
  explicit EarlyStopping(int patience, double min_delta = 0.0, bool maximize = false);
  EarlyStoppingDecision observe(double metric, std::int64_t epoch);
  double best() const;
  std::int64_t best_epoch() const;
  int misses() const;
};
}
%include "datamunge/nn/fft.hpp"
%include "datamunge/nn/linalg.hpp"
%template(NamedTensor) std::pair<std::string, datamunge::nn::Tensor>;
%template(NamedTensors) std::vector<std::pair<std::string, datamunge::nn::Tensor> >;
%include "datamunge/nn/io.hpp"
%include "datamunge/nn/datasets.hpp"


%include "datamunge/nn/parallel.hpp"
%ignore datamunge::nn::to_image;
%ignore datamunge::nn::from_image;
%include "datamunge/nn/image_io.hpp"

// ---- extra functional surface: einsum, CTC/CRF, boxes, image ops, augmentation, audio ----
%template(ShapeVectorVector) std::vector<std::vector<std::int64_t> >;
%template(IntVector) std::vector<int>;
%template(ByteVector) std::vector<unsigned char>;
%include "datamunge/nn/einsum.hpp"
%include "datamunge/nn/sequence.hpp"
// Host-side std::array box helpers stay C++-only; tensor box ops are bound.
%ignore datamunge::nn::kDefaultBoxVariances;
%ignore datamunge::nn::BoxT;
%ignore datamunge::nn::box_area;
%ignore datamunge::nn::box_iou;
%ignore datamunge::nn::encode_box;
%ignore datamunge::nn::decode_box;
%ignore datamunge::nn::non_max_suppression(const std::vector<Box>&, const std::vector<float>&, std::int64_t, float, float, float, std::vector<float>*);
%ignore datamunge::nn::non_max_suppression(const std::vector<Box>&, const std::vector<float>&, std::int64_t, float, float, float);
%ignore datamunge::nn::non_max_suppression(const std::vector<Box>&, const std::vector<float>&, std::int64_t, float, float);
%ignore datamunge::nn::non_max_suppression(const std::vector<Box>&, const std::vector<float>&, std::int64_t, float);
%ignore datamunge::nn::non_max_suppression(const std::vector<Box>&, const std::vector<float>&, std::int64_t);
%ignore datamunge::nn::match_boxes(const std::vector<Box>&, const std::vector<Box>&, float, float, bool);
%ignore datamunge::nn::match_boxes(const std::vector<Box>&, const std::vector<Box>&, float, float);
%ignore datamunge::nn::boxes_to_vector;
%ignore datamunge::nn::boxes_from_vector;
%include "datamunge/nn/boxes.hpp"
// image_ops.hpp declared here: SWIG cannot parse its multi-declarator brace initializers.
namespace datamunge::nn {
struct ImageSize {
  std::int64_t height{0};
  std::int64_t width{0};
};
Tensor resize_bilinear(const Tensor& images, ImageSize size, bool align_corners = false,
                       bool half_pixel_centers = false);
Tensor resize_nearest(const Tensor& images, ImageSize size, bool align_corners = false,
                      bool half_pixel_centers = false);
enum class ResizeMethod { Bilinear, Nearest };
Tensor crop_and_resize(const Tensor& images, const Tensor& boxes, const Tensor& box_indices, ImageSize crop_size,
                       ResizeMethod method = ResizeMethod::Bilinear, double extrapolation_value = 0.0);
enum class DepthToSpaceMode { DCR, CRD };
Tensor space_to_depth(const Tensor& x, std::int64_t block_size);
Tensor depth_to_space(const Tensor& x, std::int64_t block_size);
Tensor space_to_depth(const Tensor& x, std::int64_t block_size, DepthToSpaceMode mode);
Tensor depth_to_space(const Tensor& x, std::int64_t block_size, DepthToSpaceMode mode);
Tensor upsample_nearest(const Tensor& x, std::int64_t factor);
Tensor mirror_pad(const Tensor& x, std::int64_t pad_h, std::int64_t pad_w, PadMode mode = PadMode::Reflect);
enum class GridSampleMode { Bilinear, Nearest };
enum class GridSamplePadding { Zeros, Border };
struct GridSampleOptions {
  GridSampleMode    mode{GridSampleMode::Bilinear};
  GridSamplePadding padding{GridSamplePadding::Zeros};
  bool align_corners{false};
};
Tensor grid_sample(const Tensor& images, const Tensor& grid, const GridSampleOptions& options = {});
Tensor sample_pixels(const Tensor& images, const Tensor& coords, GridSampleMode mode = GridSampleMode::Bilinear,
                     GridSamplePadding padding = GridSamplePadding::Zeros);
Tensor affine_grid(const Tensor& theta, ImageSize size, bool align_corners = false);
struct DeformableOptions {
  std::int64_t kernel_h{3};
  std::int64_t kernel_w{3};
  std::int64_t stride{1};
  std::int64_t pad_h{1};
  std::int64_t pad_w{1};
  std::int64_t dilation{1};
};
ImageSize deformable_output_size(std::int64_t height, std::int64_t width, const DeformableOptions& options);
Tensor deformable_sample(const Tensor& images, const Tensor& offsets, const DeformableOptions& options = {});
Tensor deformable_conv2d(const Tensor& images, const Tensor& offsets, const Tensor& filter,
                         const DeformableOptions& options = {}, const Tensor& mask = Tensor());
}  // namespace datamunge::nn
%rename(augment_flip_left_right) datamunge::nn::augment::flip_left_right;
%rename(augment_flip_up_down) datamunge::nn::augment::flip_up_down;
%rename(augment_crop) datamunge::nn::augment::crop;
%rename(augment_cutout) datamunge::nn::augment::cutout;
%rename(augment_mixup) datamunge::nn::augment::mixup;
%rename(augment_cutmix) datamunge::nn::augment::cutmix;
%rename(augment_adjust_brightness) datamunge::nn::augment::adjust_brightness;
%rename(augment_adjust_contrast) datamunge::nn::augment::adjust_contrast;
%rename(augment_adjust_saturation) datamunge::nn::augment::adjust_saturation;
%rename(augment_adjust_hue) datamunge::nn::augment::adjust_hue;
%rename(augment_color_jitter) datamunge::nn::augment::color_jitter;
%rename(augment_per_image_standardization) datamunge::nn::augment::per_image_standardization;
%rename(augment_random_flip_left_right) datamunge::nn::augment::random_flip_left_right;
%rename(augment_random_flip_up_down) datamunge::nn::augment::random_flip_up_down;
%rename(augment_random_pad_crop) datamunge::nn::augment::random_pad_crop;
%rename(augment_random_crop) datamunge::nn::augment::random_crop;
%rename(augment_random_cutout) datamunge::nn::augment::random_cutout;
%rename(augment_random_brightness) datamunge::nn::augment::random_brightness;
%rename(augment_random_contrast) datamunge::nn::augment::random_contrast;
%rename(augment_random_saturation) datamunge::nn::augment::random_saturation;
%rename(augment_random_hue) datamunge::nn::augment::random_hue;
%rename(augment_beta_mix_weight) datamunge::nn::augment::beta_mix_weight;
%rename(augment_random_mixup) datamunge::nn::augment::random_mixup;
%rename(augment_random_cutmix) datamunge::nn::augment::random_cutmix;
%include "datamunge/nn/augment.hpp"
%include "datamunge/nn/audio.hpp"
namespace datamunge::nn {
struct ClassMetrics {
  double precision, recall, f1, iou;
  std::uint64_t support;
};
double roc_auc(const std::vector<int>& truth, const std::vector<double>& scores);
double average_precision(const std::vector<int>& truth, const std::vector<double>& scores);
}

#ifdef DATAMUNGE_NN_REFERENCE_EXAMPLES
// Transitional, explicitly named C++ reference bridge. Language ports never call it.
%{
#include "nn_examples.hpp"
%}
#ifdef SWIGR
%rename(native_nn_example_names) datamunge::examples::nn_example_names;
#endif
%include "nn_examples.hpp"
#endif

// Bind the reusable metric without exposing its private nested STL storage.
namespace datamunge::nn {
class ConfusionMatrix {
public:
  explicit ConfusionMatrix(std::size_t classes);
  void add(std::size_t truth, std::size_t prediction);
  void add(const Tensor& truth, const Tensor& prediction);
  std::size_t classes() const;
  double accuracy() const;
  ClassMetrics per_class(std::size_t c) const;
  ClassMetrics macro() const;
  Tensor to_tensor() const;
};
double accuracy(const Tensor& scores_or_predictions, const Tensor& labels);
double top_k_accuracy(const Tensor& scores, const Tensor& labels, std::int64_t k);
}

%include "datamunge/nn/mt19937.hpp"
