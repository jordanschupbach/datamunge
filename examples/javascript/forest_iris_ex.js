const datamunge = require("../../index.js");

const iris = datamunge.DataFrame.iris();
console.log(`iris: ${iris.nrows()} rows x ${iris.ncols()} cols\n`);

const model = new datamunge.RandomForestClassifier(iris, "Species ~ Petal.Length + Petal.Width");
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

model.plot_classification(iris, "Petal.Length", "Petal.Width").save("forest_iris_classification.svg");
model.plot_decision_regions("Petal.Length", "Petal.Width").save("forest_iris_decision_regions.svg");
console.log("\nSaved forest_iris_classification.svg and forest_iris_decision_regions.svg");

const smallForest = new datamunge.RandomForestClassifier(iris, "Species ~ Petal.Length + Petal.Width", 5);
console.log(`\n5-tree forest:   training accuracy=${smallForest.training_accuracy() * 100.0}%  OOB accuracy=${smallForest.oob_accuracy() * 100.0}%`);
console.log(`100-tree forest: training accuracy=${model.training_accuracy() * 100.0}%  OOB accuracy=${model.oob_accuracy() * 100.0}%`);
smallForest.plot_decision_regions("Petal.Length", "Petal.Width").save("forest_iris_decision_regions_5trees.svg");
console.log("Saved forest_iris_decision_regions_5trees.svg");
