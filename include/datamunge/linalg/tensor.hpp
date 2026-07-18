#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <variant>
#include <vector>

namespace datamunge::linalg {

/// @brief Element type stored by a Tensor. Determined by which alternative of
///        the internal variant is active -- never set independently of it.
enum class TensorDType { Float64, Bool, String };

/// @brief A dense, row-major, N-dimensional array that can hold double,
///        bool, or string elements (chosen at construction, not compile time).
///
/// Structural operations (reshape, transpose, slicing, concatenation, ...)
/// work for every dtype. Arithmetic, comparisons, reductions, and linear
/// algebra require a numeric-like dtype (Float64 or Bool); String tensors
/// support only structural ops and element access via string_at/set_string.
class Tensor {
public:
    Tensor();
    explicit Tensor(std::vector<std::size_t> shape, TensorDType dtype = TensorDType::Float64);

    static Tensor zeros(std::vector<std::size_t> shape);
    static Tensor ones(std::vector<std::size_t> shape);
    static Tensor full(std::vector<std::size_t> shape, double value);
    static Tensor from_values(std::vector<std::size_t> shape, std::vector<double> values);
    static Tensor from_bool_values(std::vector<std::size_t> shape, std::vector<int> values);
    static Tensor from_string_values(std::vector<std::size_t> shape, std::vector<std::string> values);
    static Tensor arange(double start, double stop, double step = 1.0);
    static Tensor eye(std::size_t n);

    // ---- metadata ----
    [[nodiscard]] std::size_t ndim() const noexcept;
    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] TensorDType dtype() const noexcept;
    [[nodiscard]] std::string dtype_name() const;

    // ---- element access ----
    [[nodiscard]] double at(const std::vector<std::size_t>& index) const;
    void set(const std::vector<std::size_t>& index, double value);
    [[nodiscard]] std::string string_at(const std::vector<std::size_t>& index) const;
    void set_string(const std::vector<std::size_t>& index, const std::string& value);

    [[nodiscard]] double at_flat(std::size_t i) const;
    void set_flat(std::size_t i, double value);
    [[nodiscard]] std::string string_at_flat(std::size_t i) const;
    void set_string_flat(std::size_t i, const std::string& value);

    // ---- shape ops (all dtypes) ----
    [[nodiscard]] Tensor reshape(std::vector<std::size_t> new_shape) const;
    [[nodiscard]] Tensor flatten() const;
    [[nodiscard]] Tensor transpose(std::vector<std::size_t> permutation = {}) const;
    [[nodiscard]] Tensor squeeze() const;
    [[nodiscard]] Tensor squeeze_axis(std::size_t axis) const;
    [[nodiscard]] Tensor expand_dims(std::size_t axis) const;
    [[nodiscard]] Tensor slice(std::size_t axis, std::size_t start, std::size_t stop, std::size_t step = 1) const;
    [[nodiscard]] Tensor index_select(std::size_t axis, const std::vector<std::size_t>& indices) const;

    [[nodiscard]] static Tensor concatenate(const std::vector<Tensor>& tensors, std::size_t axis);
    [[nodiscard]] static Tensor stack(const std::vector<Tensor>& tensors, std::size_t axis);

    // ---- elementwise arithmetic (numeric dtypes only; broadcasting; result is Float64) ----
    [[nodiscard]] Tensor add(const Tensor& other) const;
    [[nodiscard]] Tensor subtract(const Tensor& other) const;
    [[nodiscard]] Tensor multiply(const Tensor& other) const;
    [[nodiscard]] Tensor divide(const Tensor& other) const;
    [[nodiscard]] Tensor power(const Tensor& other) const;

    [[nodiscard]] Tensor add_scalar(double scalar) const;
    [[nodiscard]] Tensor subtract_scalar(double scalar) const;
    [[nodiscard]] Tensor multiply_scalar(double scalar) const;
    [[nodiscard]] Tensor divide_scalar(double scalar) const;
    [[nodiscard]] Tensor power_scalar(double exponent) const;

    [[nodiscard]] Tensor negate() const;
    [[nodiscard]] Tensor abs() const;
    [[nodiscard]] Tensor sqrt() const;
    [[nodiscard]] Tensor exp() const;
    [[nodiscard]] Tensor log() const;

    /// @brief Applies an arbitrary elementwise transform. Requires a numeric dtype; result is Float64.
    [[nodiscard]] Tensor apply(const std::function<double(double)>& fn) const;

    // ---- comparisons (broadcasting; operands must share dtype; result is Bool) ----
    [[nodiscard]] Tensor equal(const Tensor& other) const;
    [[nodiscard]] Tensor not_equal(const Tensor& other) const;
    [[nodiscard]] Tensor less(const Tensor& other) const;
    [[nodiscard]] Tensor less_equal(const Tensor& other) const;
    [[nodiscard]] Tensor greater(const Tensor& other) const;
    [[nodiscard]] Tensor greater_equal(const Tensor& other) const;

    // ---- global reductions (numeric dtypes only) ----
    [[nodiscard]] double sum() const;
    [[nodiscard]] double mean() const;
    [[nodiscard]] double max() const;
    [[nodiscard]] double min() const;
    [[nodiscard]] double prod() const;
    [[nodiscard]] std::size_t argmax() const;
    [[nodiscard]] std::size_t argmin() const;
    [[nodiscard]] bool all() const;
    [[nodiscard]] bool any() const;

    // ---- per-axis reductions (numeric dtypes only) ----
    [[nodiscard]] Tensor sum_axis(std::size_t axis, bool keepdims = false) const;
    [[nodiscard]] Tensor mean_axis(std::size_t axis, bool keepdims = false) const;
    [[nodiscard]] Tensor max_axis(std::size_t axis, bool keepdims = false) const;
    [[nodiscard]] Tensor min_axis(std::size_t axis, bool keepdims = false) const;
    [[nodiscard]] Tensor prod_axis(std::size_t axis, bool keepdims = false) const;
    [[nodiscard]] Tensor argmax_axis(std::size_t axis, bool keepdims = false) const;
    [[nodiscard]] Tensor argmin_axis(std::size_t axis, bool keepdims = false) const;

    // ---- linear algebra (numeric dtypes only) ----
    [[nodiscard]] Tensor matmul(const Tensor& other) const;
    [[nodiscard]] double dot(const Tensor& other) const;
    [[nodiscard]] Tensor outer(const Tensor& other) const;

    // ---- printing ----
    [[nodiscard]] std::string to_string(std::size_t max_elements = 100) const;

private:
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::variant<std::vector<double>, std::vector<std::uint8_t>, std::vector<std::string>> data_;

    void compute_strides();
    [[nodiscard]] std::size_t flat_index(const std::vector<std::size_t>& index) const;
};

} // namespace datamunge::linalg
