// CMD: just TARGET=tensor_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// Tensors: construction, shape ops, broadcasting, reductions, linear algebra, masks, iris.
using Datamunge;

static string ShapeString(Tensor tensor) => $"[{string.Join(", ", tensor.Shape())}]";

Console.WriteLine("=================== Construction ===================");
Console.WriteLine($"zeros([2,3]): {Tensor.Zeros([2, 3])}");
Console.WriteLine($"eye(3): {Tensor.Eye(3)}");

var r = Tensor.Arange(0.0, 12.0, 1.0)!.Reshape([3, 4])!;
Console.WriteLine($"arange(0,12).reshape([3,4]): {r}");

Console.WriteLine("\n=================== Shape ops ===================");
Console.WriteLine($"transpose -> shape {ShapeString(r.Transpose()!)}");
Console.WriteLine($"slice(axis=1, start=1, stop=3): {r.Slice(1, 1, 3)}");

Console.WriteLine("\n=================== Broadcasting arithmetic ===================");
{
    var col = Tensor.FromValues([3, 1], [1.0, 2.0, 3.0])!;
    var row = Tensor.FromValues([1, 4], [10.0, 20.0, 30.0, 40.0])!;
    Console.WriteLine($"(3,1) + (1,4) -> {col.Add(row)}");
}

Console.WriteLine("\n=================== Reductions ===================");
Console.WriteLine($"r.sum() = {r.Sum()}, r.mean() = {r.Mean()}");
Console.WriteLine($"column means (axis=0): {r.MeanAxis(0)}");

Console.WriteLine("\n=================== Linear algebra ===================");
{
    var a = Tensor.FromValues([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])!;
    var b = Tensor.FromValues([3, 2], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])!;
    Console.WriteLine($"matmul(2x3, 3x2) -> {a.Matmul(b)}");
}
{
    var v1 = Tensor.FromValues([3], [1.0, 2.0, 3.0])!;
    var v2 = Tensor.FromValues([3], [4.0, 5.0, 6.0])!;
    Console.WriteLine($"dot([1,2,3], [4,5,6]) = {v1.Dot(v2)}");
    Console.WriteLine($"outer(v1, v2) -> {v1.Outer(v2)}");
}

Console.WriteLine("\n=================== Comparisons & masks ===================");
{
    var mask = r.GreaterEqual(Tensor.Full([3, 4], 6.0)!)!;
    Console.WriteLine($"r >= 6 -> {mask}");
    Console.WriteLine($"count(r >= 6) = {mask.Sum()}");
}

Console.WriteLine("\n=================== A real dataset as a Tensor ===================");
{
    var iris = Data.Iris()!;
    var n = iris.Nrows();
    string[] columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"];
    List<double> flat = [];
    for (ulong i = 0; i < n; i++)
        foreach (var column in columns)
            flat.Add(iris.NumericAt(column, i));
    var x = Tensor.FromValues([n, 4], flat)!;
    Console.WriteLine($"iris feature tensor shape: {ShapeString(x)}");

    var featureMeans = x.MeanAxis(0)!;
    var centered = x.Subtract(featureMeans.Reshape([1, 4])!)!;
    var scatter = centered.Transpose()!.Matmul(centered)!;
    Console.WriteLine($"feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): {featureMeans}");
    Console.WriteLine($"(X-mean)^T (X-mean) [4x4 scatter matrix]: {scatter.ToString(16)}");

    var species = new List<string>();
    for (ulong i = 0; i < n; i++)
        species.Add(iris.StringAt("Species", i));
    var speciesTensor = Tensor.FromStringValues([n], species)!;
    var setosaMask = speciesTensor.Equal(Tensor.FromStringValues([1], ["setosa"])!)!;
    Console.WriteLine($"setosa count = {setosaMask.Sum()} (of {n} rows)");
}
