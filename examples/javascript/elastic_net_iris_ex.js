const datamunge = require("../../index.js");

const FORMULA = "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width";

const iris = datamunge.DataFrame.iris();
console.log(`iris: ${iris.nrows()} rows x ${iris.ncols()} cols`);
console.log(`formula: ${FORMULA}\n`);

console.log("=================== Ridge ===================");
const ridge = new datamunge.Ridge(iris, FORMULA);
ridge.print_summary();

console.log("\n=================== Lasso ===================");
const lasso = new datamunge.Lasso(iris, FORMULA);
lasso.print_summary();

console.log("\n================= Elastic Net =================");
const elastic = new datamunge.ElasticNet(iris, FORMULA, 0.5);
elastic.print_summary();

console.log("\nSaved figures showing how each model's coefficients respond to the regularization strength, and the cross-validation curve used to pick it:");

ridge.plot_coefficient_path().save("elastic_net_ridge_path.svg");
ridge.plot_cv_curve().save("elastic_net_ridge_cv.svg");
console.log("  ridge:       elastic_net_ridge_path.svg, elastic_net_ridge_cv.svg");

lasso.plot_coefficient_path().save("elastic_net_lasso_path.svg");
lasso.plot_cv_curve().save("elastic_net_lasso_cv.svg");
console.log("  lasso:       elastic_net_lasso_path.svg, elastic_net_lasso_cv.svg");

elastic.plot_coefficient_path().save("elastic_net_elasticnet_path.svg");
elastic.plot_cv_curve().save("elastic_net_elasticnet_cv.svg");
console.log("  elastic net: elastic_net_elasticnet_path.svg, elastic_net_elasticnet_cv.svg");
