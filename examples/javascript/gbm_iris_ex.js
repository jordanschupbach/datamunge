const datamunge = require("../../index.js");

const iris = datamunge.DataFrame.iris();
console.log(`iris: ${iris.nrows()} rows x ${iris.ncols()} cols\n`);

const model = new datamunge.GBMClassifier(iris, "Species ~ Petal.Length + Petal.Width");
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

model.plot_training_deviance().save("gbm_iris_training_deviance.svg");
model.plot_decision_regions("Petal.Length", "Petal.Width").save("gbm_iris_decision_regions.svg");
console.log("\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg");

const few = new datamunge.GBMClassifier(iris, "Species ~ Petal.Length + Petal.Width", 5);
const fewDev = few.training_deviance();
const modelDev = model.training_deviance();
console.log(`\n5-round ensemble:   training accuracy=${few.training_accuracy() * 100.0}%  deviance=${fewDev.get(fewDev.size() - 1)}`);
console.log(`100-round ensemble: training accuracy=${model.training_accuracy() * 100.0}%  deviance=${modelDev.get(modelDev.size() - 1)}`);
few.plot_decision_regions("Petal.Length", "Petal.Width").save("gbm_iris_decision_regions_5rounds.svg");
console.log("Saved gbm_iris_decision_regions_5rounds.svg");
