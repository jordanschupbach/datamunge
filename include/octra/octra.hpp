#include <utility>
#include <vector>

namespace octra {

/// @brief Prints a hello message to standard output.
void hello();

/// @brief Base class for user-defined callbacks invoked by octra functions.
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

} // namespace octra
