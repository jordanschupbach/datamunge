using System;
using System.Collections.Generic;
using System.Linq;

class Program {
  static string ShapeStr(SizeVector shp) {
    var parts = new List<string>();
    for (int i = 0; i < shp.Count; i++) parts.Add(shp[i].ToString());
    return string.Join(", ", parts);
  }

  static void Main() {
    Console.WriteLine("=================== Construction ===================");
    var z = Tensor.zeros(new SizeVector(new uint[] { 2, 3 }));
    Console.WriteLine($"zeros([2,3]): {z.to_string()}");

    var eyet = Tensor.eye(3);
    Console.WriteLine($"eye(3): {eyet.to_string()}");

    var r = Tensor.arange(0.0, 12.0, 1.0).reshape(new SizeVector(new uint[] { 3, 4 }));
    Console.WriteLine($"arange(0,12).reshape([3,4]): {r.to_string()}");

    Console.WriteLine("\n=================== Shape ops ===================");
    var rt = r.transpose();
    Console.WriteLine($"transpose -> shape [{ShapeStr(rt.shape())}]");
    var sliced = r.slice(1, 1, 3);
    Console.WriteLine($"slice(axis=1, start=1, stop=3): {sliced.to_string()}");

    Console.WriteLine("\n=================== Broadcasting arithmetic ===================");
    var col = Tensor.from_values(new SizeVector(new uint[] { 3, 1 }), new DVector(new double[] { 1, 2, 3 }));
    var row = Tensor.from_values(new SizeVector(new uint[] { 1, 4 }), new DVector(new double[] { 10, 20, 30, 40 }));
    var broadcastSum = col.add(row);
    Console.WriteLine($"(3,1) + (1,4) -> {broadcastSum.to_string()}");

    Console.WriteLine("\n=================== Reductions ===================");
    Console.WriteLine($"r.sum() = {r.sum()}, r.mean() = {r.mean()}");
    var colMeans = r.mean_axis(0);
    Console.WriteLine($"column means (axis=0): {colMeans.to_string()}");

    Console.WriteLine("\n=================== Linear algebra ===================");
    var a = Tensor.from_values(new SizeVector(new uint[] { 2, 3 }), new DVector(new double[] { 1, 2, 3, 4, 5, 6 }));
    var b = Tensor.from_values(new SizeVector(new uint[] { 3, 2 }), new DVector(new double[] { 7, 8, 9, 10, 11, 12 }));
    Console.WriteLine($"matmul(2x3, 3x2) -> {a.matmul(b).to_string()}");

    var v1 = Tensor.from_values(new SizeVector(new uint[] { 3 }), new DVector(new double[] { 1, 2, 3 }));
    var v2 = Tensor.from_values(new SizeVector(new uint[] { 3 }), new DVector(new double[] { 4, 5, 6 }));
    Console.WriteLine($"dot([1,2,3], [4,5,6]) = {v1.dot(v2)}");
    Console.WriteLine($"outer(v1, v2) -> {v1.outer(v2).to_string()}");

    Console.WriteLine("\n=================== Comparisons & masks ===================");
    var mask = r.greater_equal(Tensor.full(new SizeVector(new uint[] { 3, 4 }), 6.0));
    Console.WriteLine($"r >= 6 -> {mask.to_string()}");
    Console.WriteLine($"count(r >= 6) = {mask.sum()}");

    Console.WriteLine("\n=================== A real dataset as a Tensor ===================");
    var iris = DataFrame.iris();
    uint n = iris.nrows();
    var flat = new List<double>();
    for (uint i = 0; i < n; i++) {
      flat.Add(iris.numeric_at("Sepal.Length", i));
      flat.Add(iris.numeric_at("Sepal.Width", i));
      flat.Add(iris.numeric_at("Petal.Length", i));
      flat.Add(iris.numeric_at("Petal.Width", i));
    }
    var x = Tensor.from_values(new SizeVector(new uint[] { n, 4 }), new DVector(flat));
    Console.WriteLine($"iris feature tensor shape: [{ShapeStr(x.shape())}]");

    var featureMeans = x.mean_axis(0);
    var centered = x.subtract(featureMeans.reshape(new SizeVector(new uint[] { 1, 4 })));
    var scatter = centered.transpose().matmul(centered);
    Console.WriteLine($"feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): {featureMeans.to_string()}");
    Console.WriteLine($"(X-mean)^T (X-mean) [4x4 scatter matrix]: {scatter.to_string(16)}");

    var species = new List<string>();
    for (uint i = 0; i < n; i++) species.Add(iris.string_at("Species", i));
    var speciesTensor = Tensor.from_string_values(new SizeVector(new uint[] { n }), new SVector(species));
    var setosaMask = speciesTensor.equal(Tensor.from_string_values(new SizeVector(new uint[] { 1 }), new SVector(new string[] { "setosa" })));
    Console.WriteLine($"setosa count = {setosaMask.sum()} (of {n} rows)");
  }
}
