require "octruby"

FORMULA = "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width"

iris = Datamunge::DataFrame.iris
puts "iris: #{iris.nrows} rows x #{iris.ncols} cols"
puts "formula: #{FORMULA}\n\n"

puts "=================== Ridge ==================="
ridge = Datamunge::Ridge.new(iris, FORMULA)
ridge.print_summary

puts "\n=================== Lasso ==================="
lasso = Datamunge::Lasso.new(iris, FORMULA)
lasso.print_summary

puts "\n================= Elastic Net ================="
# ElasticNet(data, formula, alpha, lambda, n_lambda, cv_folds, standardize, seed)
elastic = Datamunge::ElasticNet.new(iris, FORMULA, 0.5)
elastic.print_summary

puts "\nSaved figures showing how each model's coefficients respond to the " \
     "regularization strength, and the cross-validation curve used to pick it:"

ridge.plot_coefficient_path.save("elastic_net_ridge_path.svg")
ridge.plot_cv_curve.save("elastic_net_ridge_cv.svg")
puts "  ridge:       elastic_net_ridge_path.svg, elastic_net_ridge_cv.svg"

lasso.plot_coefficient_path.save("elastic_net_lasso_path.svg")
lasso.plot_cv_curve.save("elastic_net_lasso_cv.svg")
puts "  lasso:       elastic_net_lasso_path.svg, elastic_net_lasso_cv.svg"

elastic.plot_coefficient_path.save("elastic_net_elasticnet_path.svg")
elastic.plot_cv_curve.save("elastic_net_elasticnet_cv.svg")
puts "  elastic net: elastic_net_elasticnet_path.svg, elastic_net_elasticnet_cv.svg"
