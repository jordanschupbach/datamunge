# CMD: just TARGET=glm_iris_ex run-ruby
# Generalized linear models on iris: logistic regression (binomial) and Poisson regression.
require "datamunge"

iris = Datamunge::Data.iris
rows = (0...iris.nrows).to_a

# Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
# the other two on these predictors, which sends logistic regression's coefficients toward
# +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
# binary split for a demo.
kept = rows.select { |i| %w[versicolor virginica].include?(iris.string_at("Species", i)) }
is_virginica = kept.map { |i| iris.string_at("Species", i) == "virginica" ? 1.0 : 0.0 }
petal_length = kept.map { |i| iris.numeric_at("Petal.Length", i) }
petal_width = kept.map { |i| iris.numeric_at("Petal.Width", i) }

sub = Datamunge::DataFrame.new
sub.add_numeric_column("Petal.Length", petal_length)
sub.add_numeric_column("Petal.Width", petal_width)
sub.add_numeric_column("is_virginica", is_virginica)

puts "=================== Logistic regression (binomial, logit link) ==================="
logit = Datamunge::GLM.new(sub, "is_virginica ~ Petal.Length + Petal.Width", "binomial")
logit.print_summary

fitted = logit.fitted_values
n = is_virginica.size
correct = fitted.zip(is_virginica).count { |f, y| (f >= 0.5) == (y >= 0.5) }
puts "\nResubstitution accuracy at 0.5 threshold: #{100.0 * correct / n}%"

logit.save_diagnostic_plots("glm_logistic_iris")
puts "\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg"

# Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
# classic sigmoid shape of a fitted logistic regression, with a 95% confidence band.
grid_n = 100
width_mean = petal_width.sum / petal_width.size
pl_min, pl_max = petal_length.min - 0.3, petal_length.max + 0.3
grid = Datamunge::DataFrame.new
grid.add_numeric_column("Petal.Length", Array.new(grid_n) { |i| pl_min + (pl_max - pl_min) * i / (grid_n - 1) })
grid.add_numeric_column("Petal.Width", [width_mean] * grid_n)
curve_frame = logit.predict_frame(grid, "confidence")
puts "\nPredicted-probability curve (first 5 rows):"
puts curve_frame.to_string(5)

# Poisson regression, for contrast: same IRLS engine, different family/link.
puts "\n=================== Poisson regression (log link) ==================="
count_data = Datamunge::DataFrame.new
count_data.add_numeric_column("Sepal.Width", rows.map { |i| iris.numeric_at("Sepal.Width", i) })
count_data.add_numeric_column("Petal.Length", rows.map { |i| iris.numeric_at("Petal.Length", i) })
count_data.add_numeric_column("count", rows.map { |i| (iris.numeric_at("Sepal.Length", i) + 0.5).floor.to_f })
Datamunge::GLM.new(count_data, "count ~ Sepal.Width + Petal.Length", "poisson").print_summary
