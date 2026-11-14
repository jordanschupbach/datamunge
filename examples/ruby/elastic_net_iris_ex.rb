# CMD: just TARGET=elastic_net_iris_ex run-ruby
# Ridge, Lasso and Elastic Net on iris, with coefficient-path and CV-curve figures.
require "datamunge"

formula = "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width"

iris = Datamunge::Data.iris
puts "iris: #{iris.nrows} rows x #{iris.ncols} cols"
puts "formula: #{formula}\n\n"

puts "=================== Ridge ==================="
ridge = Datamunge::Ridge.new(iris, formula)
ridge.print_summary

puts "\n=================== Lasso ==================="
lasso = Datamunge::Lasso.new(iris, formula)
lasso.print_summary

puts "\n================= Elastic Net ================="
# Datamunge::ElasticNet.new(data, formula, alpha = 0.5, lambda, n_lambda, cv_folds, standardize, seed)
elastic = Datamunge::ElasticNet.new(iris, formula, 0.5)
elastic.print_summary

puts "\nSaved figures showing how each model's coefficients respond to the regularization strength, " \
     "and the cross-validation curve used to pick it:"

{ "ridge" => [ridge, "ridge:      "], "lasso" => [lasso, "lasso:      "],
  "elasticnet" => [elastic, "elastic net:"] }.each do |name, (model, label)|
  model.plot_coefficient_path.save("elastic_net_#{name}_path.svg")
  model.plot_cv_curve.save("elastic_net_#{name}_cv.svg")
  puts "  #{label} elastic_net_#{name}_path.svg, elastic_net_#{name}_cv.svg"
end
