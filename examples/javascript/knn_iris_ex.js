const datamunge = require("../../index.js");

const iris = datamunge.DataFrame.iris();
console.log(`iris: ${iris.nrows()} rows x ${iris.ncols()} cols\n`);

const model = new datamunge.KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width");
model.print_summary();

console.log("\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):");
console.log(model.confusion_matrix().to_string());

console.log("\nLeave-one-out misclassified rows:");
const fitted = model.predict(iris);
let misclassified = 0;
const n = iris.nrows();
for (let i = 0; i < n; i++) {
  const actual = iris.string_at("Species", i);
  const pred = fitted.get(i);
  if (pred !== actual) {
    misclassified += 1;
    console.log(`  row ${i}: Petal.Length=${iris.numeric_at("Petal.Length", i)} Petal.Width=${iris.numeric_at("Petal.Width", i)}  actual=${actual}  predicted=${pred}`);
  }
}
console.log(`${misclassified} of ${n} misclassified (${(100.0 * misclassified) / n}%)`);

model.plot_decision_regions("Petal.Length", "Petal.Width").save("knn_iris_decision_regions_k5.svg");
console.log("\nSaved knn_iris_decision_regions_k5.svg");

const k1 = new datamunge.KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width", 1);
console.log(`\nk=1  leave-one-out accuracy: ${k1.training_accuracy() * 100.0}%`);
k1.plot_decision_regions("Petal.Length", "Petal.Width").save("knn_iris_decision_regions_k1.svg");
console.log("Saved knn_iris_decision_regions_k1.svg");

const k25 = new datamunge.KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width", 25);
console.log(`\nk=25 leave-one-out accuracy: ${k25.training_accuracy() * 100.0}%`);
k25.plot_decision_regions("Petal.Length", "Petal.Width").save("knn_iris_decision_regions_k25.svg");
console.log("Saved knn_iris_decision_regions_k25.svg");
