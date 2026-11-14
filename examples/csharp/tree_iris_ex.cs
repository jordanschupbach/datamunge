// CMD: just TARGET=tree_iris_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// Decision tree classifier on iris: summary, confusion matrix, misclassified rows and plots.
using Datamunge;

var iris = Data.Iris()!;
Console.WriteLine($"iris: {iris.Nrows()} rows x {iris.Ncols()} cols\n");

var model = new DecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width");
model.PrintSummary();

Console.WriteLine("\nConfusion matrix (rows = actual, cols = predicted):");
Console.WriteLine(model.ConfusionMatrix());

Console.WriteLine("\nMisclassified rows:");
var predictions = model.Predict(iris);
var n = iris.Nrows();
var misclassified = 0;
for (ulong i = 0; i < n; i++)
{
    var (actual, predicted) = (iris.StringAt("Species", i), predictions[(int)i]);
    if (predicted == actual) continue;
    misclassified++;
    Console.WriteLine($"  row {i}: Petal.Length={iris.NumericAt("Petal.Length", i)} Petal.Width={iris.NumericAt("Petal.Width", i)}  actual={actual}  predicted={predicted}");
}
Console.WriteLine($"{misclassified} of {n} misclassified ({100.0 * misclassified / n}%)");

model.PlotClassification(iris, "Petal.Length", "Petal.Width").Save("tree_iris_classification.svg");
model.PlotDecisionRegions("Petal.Length", "Petal.Width").Save("tree_iris_decision_regions.svg");
Console.WriteLine("\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg");

// A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision boundary.
var shallow = new DecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width", 2);
Console.WriteLine($"\nDepth-2 tree training accuracy: {shallow.TrainingAccuracy() * 100}% ({shallow.LeafCount()} leaves)");
shallow.PlotDecisionRegions("Petal.Length", "Petal.Width").Save("tree_iris_decision_regions_depth2.svg");
Console.WriteLine("Saved tree_iris_decision_regions_depth2.svg");
