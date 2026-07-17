#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/random/random.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace datamunge {

/// @brief Prints a hello message to standard output.
void hello();

/// @brief Base class for user-defined callbacks invoked by datamunge functions.
///
/// Derive from this class and override @c call() to supply custom behavior
/// wherever a @c Callback* is accepted.
class Callback {
 public:
  /// @brief Virtual destructor — ensures proper cleanup of derived objects.
  virtual ~Callback() = default;

  /// @brief Applies the callback to a single value.
  /// @param x Input value.
  /// @return Transformed value; the default implementation returns @p x unchanged.
  virtual double call(double x) { return x; }
};

/// @brief Invokes a callback with the given value.
/// @param x    Input value passed to the callback.
/// @param cb   Pointer to a @c Callback instance; must not be null.
/// @return     Result of @c cb->call(x).
double call_with_callback(double x, Callback* cb);

/// @brief Applies a callback to every element of a vector.
/// @param values Source vector of doubles.
/// @param cb     Pointer to a @c Callback instance; must not be null.
/// @return       New vector where each element is the result of @c cb->call(v)
///               for the corresponding element @c v in @p values.
std::vector<double> map_dvector_with_callback(const std::vector<double>& values, Callback* cb);

/// @brief Constructs a three-element vector from individual values.
/// @param a First element.
/// @param b Second element.
/// @param c Third element.
/// @return  @c std::vector<double>{a, b, c}.
std::vector<double> make_dvector(double a, double b, double c);

/// @brief Computes the sum of all elements in a vector.
/// @param values Vector of doubles to sum.
/// @return       Sum of all elements, or 0.0 if the vector is empty.
double sum_dvector(const std::vector<double>& values);

/// @brief Constructs a pair of doubles.
/// @param a First element.
/// @param b Second element.
/// @return  @c std::pair<double, double>{a, b}.
std::pair<double, double> make_dpair(double a, double b);

/// @brief Computes the sum of both elements in a pair.
/// @param values Pair of doubles.
/// @return       @c values.first + values.second.
double sum_dpair(const std::pair<double, double>& values);

/// @brief SWIG-friendly facade for the C++ dataframe API exposed to bindings.
class DataFrame {
 public:
  DataFrame() = default;

  [[nodiscard]] std::size_t nrows() const;
  [[nodiscard]] std::size_t ncols() const;
  [[nodiscard]] std::vector<std::size_t> shape() const;
  [[nodiscard]] std::vector<std::string> columns() const;

  void add_numeric_column(const std::string& column_name, const std::vector<double>& values,
                          const std::vector<int>& valid_mask = {});
  void add_string_column(const std::string& column_name, const std::vector<std::string>& values,
                         const std::vector<int>& valid_mask = {});
  void add_string_column_encoded(const std::string& column_name, const std::string& encoded_values,
                                 const std::vector<int>& valid_mask = {});
  void fill_null_numeric(const std::string& column_name, double value);
  void fill_null_string(const std::string& column_name, const std::string& value);

  [[nodiscard]] DataFrame* select(const std::vector<std::string>& selected_columns) const;
  [[nodiscard]] DataFrame* select_encoded(const std::string& encoded_columns) const;
  [[nodiscard]] DataFrame* sort_by(const std::string& column_name, bool ascending = true) const;
  [[nodiscard]] DataFrame* drop_duplicates(const std::vector<std::string>& subset = {}) const;
  [[nodiscard]] DataFrame* drop_duplicates_encoded(const std::string& encoded_subset) const;
  [[nodiscard]] DataFrame* group_by_sum(const std::vector<std::string>& key_columns,
                                        const std::vector<std::string>& value_columns) const;
  [[nodiscard]] DataFrame* group_by_sum_encoded(const std::string& encoded_key_columns,
                                                const std::string& encoded_value_columns) const;
  [[nodiscard]] DataFrame* join(const DataFrame& right, const std::string& left_key,
                                const std::string& right_key, bool left_join = false) const;

  [[nodiscard]] std::size_t numeric_count(const std::string& column_name) const;
  [[nodiscard]] std::size_t numeric_null_count(const std::string& column_name) const;
  [[nodiscard]] double numeric_sum(const std::string& column_name) const;
  [[nodiscard]] double numeric_mean(const std::string& column_name) const;
  [[nodiscard]] double numeric_min(const std::string& column_name) const;
  [[nodiscard]] double numeric_max(const std::string& column_name) const;
  [[nodiscard]] std::string to_string(std::size_t max_rows = 10) const;

 private:
  explicit DataFrame(dstruct::DataFrame frame);

  template <typename T>
  static std::vector<std::optional<T>> apply_valid_mask(const std::vector<T>& values, const std::vector<int>& valid_mask);
  static std::vector<std::string> split_encoded_strings(const std::string& encoded_values);

  dstruct::DataFrame frame_;
};

} // namespace datamunge
