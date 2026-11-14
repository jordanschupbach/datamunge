// CMD: just TARGET=knn_iris_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// k-nearest-neighbours classifier on iris: leave-one-out accuracy and decision regions for several k.
using Datamunge;

var iris = Data.Iris()!;
Console.WriteLine($"iris: {iris.Nrows()} rows x {iris.Ncols()} cols\n");

var model = new KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width");
model.PrintSummary();

Console.WriteLine("\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):");
Console.WriteLine(model.ConfusionMatrix());

Console.WriteLine("\nLeave-one-out misclassified rows:");
var fitted = model.Predict(iris);
var n = iris.Nrows();
var misclassified = 0;
for (ulong i = 0; i < n; i++)
{
    var (actual, predicted) = (iris.StringAt("Species", i), fitted[(int)i]);
    if (predicted == actual) continue;
    misclassified++;
    Console.WriteLine($"  row {i}: Petal.Length={iris.NumericAt("Petal.Length", i)} Petal.Width={iris.NumericAt("Petal.Width", i)}  actual={actual}  predicted={predicted}");
}
Console.WriteLine($"{misclassified} of {n} misclassified ({100.0 * misclassified / n}%)");

model.PlotDecisionRegions("Petal.Length", "Petal.Width").Save("knn_iris_decision_regions_k5.svg");
Console.WriteLine("\nSaved knn_iris_decision_regions_k5.svg");

// k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
// every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
// underfit boundary). new KNNClassifier(data, formula, k, metric, weighted, standardize)
foreach (var k in new ulong[] { 1, 25 })
{
    var knn = new KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width", k);
    var file = $"knn_iris_decision_regions_k{k}.svg";
    Console.WriteLine($"\nk={k,-2} leave-one-out accuracy: {knn.TrainingAccuracy() * 100}%");
    knn.PlotDecisionRegions("Petal.Length", "Petal.Width").Save(file);
    Console.WriteLine($"Saved {file}");
}
