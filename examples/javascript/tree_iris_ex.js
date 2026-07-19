const datamunge = require("../../index.js");

const iris = datamunge.DataFrame.iris();
console.log(`iris: ${iris.nrows()} rows x ${iris.ncols()} cols\n`);

const model = new datamunge.DecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width");
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

model.plot_classification(iris, "Petal.Length", "Petal.Width").save("tree_iris_classification.svg");
model.plot_decision_regions("Petal.Length", "Petal.Width").save("tree_iris_decision_regions.svg");
console.log("\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg");

const shallow = new datamunge.DecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width", 2);
console.log(`\nDepth-2 tree training accuracy: ${shallow.training_accuracy() * 100.0}% (${shallow.leaf_count()} leaves)`);
shallow.plot_decision_regions("Petal.Length", "Petal.Width").save("tree_iris_decision_regions_depth2.svg");
console.log("Saved tree_iris_decision_regions_depth2.svg");
