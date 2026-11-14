// CMD: just TARGET=bspline_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// Tensor-product B-spline regression: fit z ~ bs(x, y) to a smooth surface and predict.
using Datamunge;

// Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2.
List<double> x = [], y = [], z = [];
for (var iy = 0; iy < 8; iy++)
    for (var ix = 0; ix < 8; ix++)
    {
        double xv = ix / 7.0, yv = iy / 7.0;
        x.Add(xv);
        y.Add(yv);
        z.Add(Math.Sin(xv) + yv * yv);
    }
var data = new DataFrame();
data.AddNumericColumn("x", x);
data.AddNumericColumn("y", y);
data.AddNumericColumn("z", z);

var surface = new LM(data, "z ~ bs(x, y)");
Console.WriteLine("Fitted z ~ bs(x, y)");
surface.PrintSummary();

var newPoints = new DataFrame();
newPoints.AddNumericColumn("x", [0.25, 0.75]);
newPoints.AddNumericColumn("y", [0.50, 0.25]);
var pred = surface.Predict(newPoints);
Console.WriteLine($"Predictions: [{pred[0]:F6}, {pred[1]:F6}]");
