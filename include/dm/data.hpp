#pragma once

/// @file data.hpp
/// @brief R-style named dataset lookup and the legacy C++ dataset facade.
#include <datamunge/datamunge.hpp>

/// @brief Short public namespace for the canonical datamunge C++ API.
namespace dm {
using namespace datamunge;

/// @brief Legacy C++ spelling for the existing binding-facing dataset functions.
///
/// Language bindings retain the original datamunge::data namespace. In C++, the
/// same functions are available through these static members, alongside the named
/// dm::data() table lookup. The named lookup returns an owning core frame by value.
///
/// @par Example
/// @code
/// const std::unique_ptr<dm::DataFrame> iris(dm::data::iris());
/// std::print("{}\n", iris->nrows());  // 150
/// @endcode
struct data {
  /// @copydoc datamunge::data::iris()
  static DataFrame* iris() { return datamunge::data::iris(); }
  /// @copydoc datamunge::data::penguins()
  static DataFrame* penguins() { return datamunge::data::penguins(); }
  /// @copydoc datamunge::data::tooth_growth()
  static DataFrame* tooth_growth() { return datamunge::data::tooth_growth(); }
  /// @copydoc datamunge::data::wine()
  static DataFrame* wine() { return datamunge::data::wine(); }
  /// @copydoc datamunge::data::titanic()
  static DataFrame* titanic() { return datamunge::data::titanic(); }
  /// @copydoc datamunge::data::eight_schools()
  static DataFrame* eight_schools() { return datamunge::data::eight_schools(); }
  /// @copydoc datamunge::data::air_passengers()
  static DataFrame* air_passengers() { return datamunge::data::air_passengers(); }
  /// @copydoc datamunge::data::volcano()
  static DataFrame* volcano() { return datamunge::data::volcano(); }
  /// @copydoc datamunge::data::faithful()
  static DataFrame* faithful() { return datamunge::data::faithful(); }
  /// @copydoc datamunge::data::mtcars()
  static DataFrame* mtcars() { return datamunge::data::mtcars(); }
  /// @copydoc datamunge::data::airquality()
  static DataFrame* airquality() { return datamunge::data::airquality(); }
  /// @copydoc datamunge::data::breast_cancer()
  static DataFrame* breast_cancer() { return datamunge::data::breast_cancer(); }
  /// @copydoc datamunge::data::diabetes()
  static DataFrame* diabetes() { return datamunge::data::diabetes(); }
  /// @copydoc datamunge::data::digits()
  static DataFrame* digits() { return datamunge::data::digits(); }
  /// @copydoc datamunge::data::diamonds()
  static DataFrame* diamonds() { return datamunge::data::diamonds(); }
  /// @copydoc datamunge::data::california_housing()
  static DataFrame* california_housing() { return datamunge::data::california_housing(); }
  /// @copydoc datamunge::data::load()
  static DataFrame* load(const std::string& name) { return datamunge::data::load(name); }
  /// @copydoc datamunge::data::catalog()
  static DataFrame* catalog() { return datamunge::data::catalog(); }
};

/// @brief Loads a registered table by name, like R's data(), returning it by value.
///
/// Bundled tables work offline. Downloaded tables use the existing verified
/// local cache. Use datasets::catalog() to discover entries and storage types.
/// @param name Canonical id or recognized source spelling, such as "iris" or "ToothGrowth".
/// @return A new owning datamunge::dstruct::DataFrame.
/// @throws std::invalid_argument If the dataset is unknown or requires an image or graph loader.
/// @throws std::runtime_error If a downloaded table cannot be fetched or parsed.
///
/// @par Example
/// @code
/// const auto iris = dm::data("iris");
/// std::print("{} x {}\n", iris.nrows(), iris.ncols()); // 150 x 5
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/utils/html/data.html
/// @see datamunge::datasets::load(), datamunge::datasets::catalog()
[[nodiscard]] inline dstruct::DataFrame data(const std::string& name) {
  return datasets::load(name);
}

/// @brief Lists the canonical names in the dataset registry in lexical order.
///
/// Includes datasets with dedicated image or graph loaders. Reads only registry
/// metadata and works offline without a populated cache.
/// @return An owned vector of names, one per row in datasets::catalog().
///
/// @par Example
/// @code
/// const dm::Vector<std::string> names = dm::dataset_names();
/// std::print("{}\n", names[0]);  // air_passengers
/// @endcode
/// @see datamunge::datasets::catalog(), dm::data()
[[nodiscard]] inline Vector<std::string> dataset_names() {
  const auto catalog = datamunge::datasets::catalog();
  return catalog.get_column<std::string>("name").values();
}

} // namespace dm
