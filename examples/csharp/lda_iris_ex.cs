// CMD: just TARGET=lda_iris_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// Linear discriminant analysis on iris: summary, confusion matrix, predictions and discriminant plot.
using Datamunge;

var iris = Data.Iris()!;
Console.WriteLine($"iris: {iris.Nrows()} rows x {iris.Ncols()} cols\n");

var model = new LDA(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
model.PrintSummary();

Console.WriteLine($"\nConfusion matrix (rows = actual, cols = predicted):\n{model.ConfusionMatrix()}");

var newdata = new DataFrame();
newdata.AddNumericColumn("Sepal.Length", [5.1, 6.0, 6.5, 6.2]);
newdata.AddNumericColumn("Sepal.Width", [3.5, 2.7, 3.0, 2.8]);
newdata.AddNumericColumn("Petal.Length", [1.4, 4.5, 5.5, 4.8]);
newdata.AddNumericColumn("Petal.Width", [0.2, 1.5, 2.0, 1.8]);
Console.WriteLine($"\nPredictions for new flowers:\n{model.PredictFrame(newdata)}");

model.SaveDiscriminantPlot("lda_iris_discriminants.svg");
Console.WriteLine("\nSaved discriminant plot as lda_iris_discriminants.svg");
