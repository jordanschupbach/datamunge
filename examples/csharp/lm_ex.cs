// CMD: just TARGET=lm_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// Linear model: mpg ~ hp + wt + transmission, its ANOVA table and predictions.
using Datamunge;

var cars = new DataFrame();
cars.AddNumericColumn("hp", [110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0]);
cars.AddNumericColumn("wt", [2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44]);
cars.AddStringColumn("transmission", ["manual", "manual", "manual", "automatic", "automatic",
                                      "automatic", "automatic", "automatic", "automatic", "automatic"]);
cars.AddNumericColumn("mpg", [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2]);

Console.WriteLine("Fitting: mpg ~ hp + wt + transmission\n");
var model = new LM(cars, "mpg ~ hp + wt + transmission");
model.PrintSummary();

Console.WriteLine($"\nSequential ANOVA:\n{model.Anova()}");

var newcars = new DataFrame();
newcars.AddNumericColumn("hp", [150.0, 90.0]);
newcars.AddNumericColumn("wt", [3.0, 2.5]);
newcars.AddStringColumn("transmission", ["manual", "automatic"]);
Console.WriteLine($"\nPredictions with 95% confidence intervals:\n{model.PredictFrame(newcars, "confidence")}");

model.SaveDiagnosticPlots("lm_ex_diagnostics");
Console.WriteLine("\nSaved diagnostic plots as lm_ex_diagnostics_*.svg");
