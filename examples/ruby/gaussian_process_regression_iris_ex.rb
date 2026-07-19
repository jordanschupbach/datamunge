require "octruby"

FORMULA = "Petal.Length ~ Petal.Width"

iris = Datamunge::DataFrame.iris
puts "iris: #{iris.nrows} rows x #{iris.ncols} cols"
puts "formula: #{FORMULA}\n\n"

# Both the length scale and the noise ratio are auto-selected by maximizing the exact log
# marginal likelihood.
model = Datamunge::GaussianProcessRegression.new(iris, FORMULA)
model.print_summary

model.plot_fit(iris).save("gpr_iris_fit.svg")
model.plot_length_scale_profile.save("gpr_iris_length_scale_profile.svg")
puts "\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg"

# Unlike every other regressor in this suite, a GP gives a genuine posterior confidence
# interval at every point -- including far outside the training data, where it should widen
# substantially as the model's uncertainty grows.
query = Datamunge::DataFrame.new
query.add_numeric_column("Petal.Width", [0.2, 1.3, 2.5, 10.0])
detail = model.predict_frame(query, "confidence")
puts "\nPredictions with 95% confidence intervals:"
puts detail.to_string
puts "(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its\n" \
     " interval is than the in-range predictions.)"
