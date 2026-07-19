const datamunge = require("../../index.js");

const iris = datamunge.DataFrame.iris();
console.log(`iris: ${iris.nrows()} rows x ${iris.ncols()} cols\n`);

const model = new datamunge.XGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width");
model.print_summary();

console.log("\nConfusion matrix (rows = actual, cols = predicted):");
console.log(model.confusion_matrix().to_string());

console.log("\nMisclassified rows:");
const predictions = model.predict(iris);
let misclassified = 0;
const n = iris.nrows();
for (let i = 0; i < n; i++) {
  const actual = iris.string_at("Species", i);
  const pred = predictions.get(i);
  if (pred !== actual) {
    misclassified += 1;
    console.log(`  row ${i}: Petal.Length=${iris.numeric_at("Petal.Length", i)} Petal.Width=${iris.numeric_at("Petal.Width", i)}  actual=${actual}  predicted=${pred}`);
  }
}
console.log(`${misclassified} of ${n} misclassified (${(100.0 * misclassified) / n}%)`);

model.plot_training_deviance().save("xgboost_iris_training_deviance.svg");
model.plot_decision_regions("Petal.Length", "Petal.Width").save("xgboost_iris_decision_regions.svg");
console.log("\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg");

const heavy = new datamunge.XGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width", 100, 0.3, 6, 50.0);
const modelDev = model.training_deviance();
const heavyDev = heavy.training_deviance();
console.log(`\nlambda=1 (default):   training accuracy=${model.training_accuracy() * 100.0}%  deviance=${modelDev.get(modelDev.size() - 1)}`);
console.log(`lambda=50 (heavy L2): training accuracy=${heavy.training_accuracy() * 100.0}%  deviance=${heavyDev.get(heavyDev.size() - 1)}`);
heavy.plot_decision_regions("Petal.Length", "Petal.Width").save("xgboost_iris_decision_regions_heavy_lambda.svg");
console.log("Saved xgboost_iris_decision_regions_heavy_lambda.svg");
