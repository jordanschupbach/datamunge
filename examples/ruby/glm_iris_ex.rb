require "octruby"

iris = Datamunge::DataFrame.iris

# Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
# the other two on these predictors, which sends logistic regression's coefficients toward
# +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
# binary split for a demo.
is_virginica = []
petal_length = []
petal_width = []
iris.nrows.times do |i|
  species = iris.string_at("Species", i)
  next unless ["versicolor", "virginica"].include?(species)
  is_virginica << (species == "virginica" ? 1.0 : 0.0)
  petal_length << iris.numeric_at("Petal.Length", i)
  petal_width << iris.numeric_at("Petal.Width", i)
end

sub = Datamunge::DataFrame.new
sub.add_numeric_column("Petal.Length", petal_length)
sub.add_numeric_column("Petal.Width", petal_width)
sub.add_numeric_column("is_virginica", is_virginica)

puts "=================== Logistic regression (binomial, logit link) ==================="
logit = Datamunge::GLM.new(sub, "is_virginica ~ Petal.Length + Petal.Width", "binomial")
logit.print_summary

fitted = logit.fitted_values
correct = (0...is_virginica.length).count { |i| (fitted[i] >= 0.5) == (is_virginica[i] >= 0.5) }
puts "\nResubstitution accuracy at 0.5 threshold: #{100.0 * correct / is_virginica.length}%"

logit.save_diagnostic_plots("glm_logistic_iris")
puts "\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg"

# Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
# classic sigmoid shape of a fitted logistic regression, with a 95% confidence band.
width_mean = petal_width.sum / petal_width.length
grid_n = 100
pl_min = petal_length.min - 0.3
pl_max = petal_length.max + 0.3
grid_x = (0...grid_n).map { |i| pl_min + (pl_max - pl_min) * i / (grid_n - 1) }
grid = Datamunge::DataFrame.new
grid.add_numeric_column("Petal.Length", grid_x)
grid.add_numeric_column("Petal.Width", [width_mean] * grid_n)
curve_frame = logit.predict_frame(grid, "confidence")
puts "\nPredicted-probability curve (first 5 rows):"
puts curve_frame.to_string(5)

# Poisson regression, for contrast: same IRLS engine, different family/link.
puts "\n=================== Poisson regression (log link) ==================="
count = (0...iris.nrows).map { |i| iris.numeric_at("Sepal.Length", i).round }
count_data = Datamunge::DataFrame.new
sepal_width = (0...iris.nrows).map { |i| iris.numeric_at("Sepal.Width", i) }
all_petal_length = (0...iris.nrows).map { |i| iris.numeric_at("Petal.Length", i) }
count_data.add_numeric_column("Sepal.Width", sepal_width)
count_data.add_numeric_column("Petal.Length", all_petal_length)
count_data.add_numeric_column("count", count.map(&:to_f))

poisson = Datamunge::GLM.new(count_data, "count ~ Sepal.Width + Petal.Length", "poisson")
poisson.print_summary
