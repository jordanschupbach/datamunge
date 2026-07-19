package require Datamunge 0.0.1

set FORMULA "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width"

set iris [datamunge::DataFrame_iris]
puts "iris: [datamunge::DataFrame_nrows $iris] rows x [datamunge::DataFrame_ncols $iris] cols"
puts "formula: $FORMULA\n"

puts "=================== Ridge ==================="
set ridge [datamunge::new_Ridge $iris $FORMULA]
datamunge::Ridge_print_summary $ridge

puts "\n=================== Lasso ==================="
set lasso [datamunge::new_Lasso $iris $FORMULA]
datamunge::Lasso_print_summary $lasso

puts "\n================= Elastic Net ================="
set elastic [datamunge::new_ElasticNet $iris $FORMULA 0.5]
datamunge::ElasticNet_print_summary $elastic

puts "\nSaved figures showing how each model's coefficients respond to the regularization strength, and the cross-validation curve used to pick it:"

datamunge::Plot_save [datamunge::Ridge_plot_coefficient_path $ridge] "elastic_net_ridge_path.svg"
datamunge::Plot_save [datamunge::Ridge_plot_cv_curve $ridge] "elastic_net_ridge_cv.svg"
puts "  ridge:       elastic_net_ridge_path.svg, elastic_net_ridge_cv.svg"

datamunge::Plot_save [datamunge::Lasso_plot_coefficient_path $lasso] "elastic_net_lasso_path.svg"
datamunge::Plot_save [datamunge::Lasso_plot_cv_curve $lasso] "elastic_net_lasso_cv.svg"
puts "  lasso:       elastic_net_lasso_path.svg, elastic_net_lasso_cv.svg"

datamunge::Plot_save [datamunge::ElasticNet_plot_coefficient_path $elastic] "elastic_net_elasticnet_path.svg"
datamunge::Plot_save [datamunge::ElasticNet_plot_cv_curve $elastic] "elastic_net_elasticnet_cv.svg"
puts "  elastic net: elastic_net_elasticnet_path.svg, elastic_net_elasticnet_cv.svg"
