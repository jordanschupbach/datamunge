#pragma once

#include <datamunge/dstruct/dataframe.hpp>

namespace datamunge::datasets {

// Fisher/Anderson's iris dataset (150 rows): Sepal.Length, Sepal.Width,
// Petal.Length, Petal.Width (numeric, cm) and Species (setosa, versicolor,
// virginica). Matches R's built-in `datasets::iris` exactly.
dstruct::DataFrame iris();

// The Palmer Archipelago penguins dataset (344 rows): species (Adelie,
// Chinstrap, Gentoo), island (Biscoe, Dream, Torgersen), bill_length_mm,
// bill_depth_mm, flipper_length_mm, body_mass_g, sex (male/female, some
// missing), year. A handful of rows have missing measurements/sex, matching
// the well-known `palmerpenguins::penguins` dataset.
dstruct::DataFrame penguins();

} // namespace datamunge::datasets
