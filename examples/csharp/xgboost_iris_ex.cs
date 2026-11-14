// CMD: just TARGET=xgboost_iris_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// XGBoost classifier on iris: summary, confusion matrix, misclassified rows, L2-regularization comparison.
using Datamunge;

var iris = Data.Iris()!;
Console.WriteLine($"iris: {iris.Nrows()} rows x {iris.Ncols()} cols\n");

var model = new XGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width");
model.PrintSummary();

Console.WriteLine($"\nConfusion matrix (rows = actual, cols = predicted):\n{model.ConfusionMatrix()}");

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

model.PlotTrainingDeviance().Save("xgboost_iris_training_deviance.svg");
model.PlotDecisionRegions("Petal.Length", "Petal.Width").Save("xgboost_iris_decision_regions.svg");
Console.WriteLine("\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg");

// XGBoost's defining lever is regularization: a heavily L2-regularized model (large lambda)
// grows the same deep trees but keeps every leaf weight small, producing a much smoother
// decision boundary than the lightly-regularized default.
// new XGBoostClassifier(data, formula, nTrees, learningRate, maxDepth, lambda, alpha, gamma,
//                       minChildWeight, minSamplesLeaf, subsample, colsampleBytree, seed)
var heavy = new XGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width", 100, 0.3, 6, 50.0);
Console.WriteLine($"\nlambda=1 (default):   training accuracy={100 * model.TrainingAccuracy()}%  deviance={model.TrainingDeviance()[^1]}");
Console.WriteLine($"lambda=50 (heavy L2): training accuracy={100 * heavy.TrainingAccuracy()}%  deviance={heavy.TrainingDeviance()[^1]}");
heavy.PlotDecisionRegions("Petal.Length", "Petal.Width").Save("xgboost_iris_decision_regions_heavy_lambda.svg");
Console.WriteLine("Saved xgboost_iris_decision_regions_heavy_lambda.svg");
