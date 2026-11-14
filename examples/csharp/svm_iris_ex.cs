// CMD: just TARGET=svm_iris_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// Support vector machine on iris: RBF vs. linear kernels, confusion matrix and predictions.
using Datamunge;

const string formula = "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width";
var iris = Data.Iris()!;
Console.WriteLine($"iris: {iris.Nrows()} rows x {iris.Ncols()} cols\n");

Console.WriteLine("=== RBF kernel (default) ===");
var rbfModel = new SVM(iris, formula);
rbfModel.PrintSummary();
Console.WriteLine($"\nConfusion matrix (rows = actual, cols = predicted):\n{rbfModel.ConfusionMatrix()}");

Console.WriteLine("\n=== Linear kernel, for comparison ===");
var linearModel = new SVM(iris, formula, "linear");
Console.WriteLine($"Training accuracy: {linearModel.TrainingAccuracy() * 100}%");
Console.WriteLine($"Support vectors: {linearModel.NumSupportVectors()}");

var newdata = new DataFrame();
newdata.AddNumericColumn("Sepal.Length", [5.1, 6.0, 6.5, 6.2]);
newdata.AddNumericColumn("Sepal.Width", [3.5, 2.7, 3.0, 2.8]);
newdata.AddNumericColumn("Petal.Length", [1.4, 4.5, 5.5, 4.8]);
newdata.AddNumericColumn("Petal.Width", [0.2, 1.5, 2.0, 1.8]);
Console.WriteLine($"\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):\n{rbfModel.PredictFrame(newdata)}");
