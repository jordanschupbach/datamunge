module app;

import std.stdio : writeln;
import std.conv : to;
import std.array : join;
import std.algorithm : map;
import datamunge;

SizeVector sv(size_t[] t) {
  auto v = new SizeVector();
  foreach (x; t) v.push_back(x);
  return v;
}

DVector dv(double[] t) {
  auto v = new DVector();
  foreach (x; t) v.push_back(x);
  return v;
}

SVector strv(string[] t) {
  auto v = new SVector();
  foreach (x; t) v.push_back(x);
  return v;
}

string shapeStr(SizeVector s) {
  string[] parts;
  for (size_t i = 0; i < s.size(); i++) parts ~= to!string(s[i]);
  return parts.join(", ");
}

void main() {
  writeln("=================== Construction ===================");
  auto z = Tensor.zeros(sv([2, 3]));
  writeln("zeros([2,3]): ", z.to_string());

  auto eyet = Tensor.eye(3);
  writeln("eye(3): ", eyet.to_string());

  auto r = Tensor.arange(0.0, 12.0, 1.0).reshape(sv([3, 4]));
  writeln("arange(0,12).reshape([3,4]): ", r.to_string());

  writeln("\n=================== Shape ops ===================");
  auto rt = r.transpose();
  writeln("transpose -> shape [", shapeStr(rt.shape()), "]");
  auto sliced = r.slice(1, 1, 3);
  writeln("slice(axis=1, start=1, stop=3): ", sliced.to_string());

  writeln("\n=================== Broadcasting arithmetic ===================");
  auto col = Tensor.from_values(sv([3, 1]), dv([1, 2, 3]));
  auto row = Tensor.from_values(sv([1, 4]), dv([10, 20, 30, 40]));
  auto broadcast_sum = col.add(row);
  writeln("(3,1) + (1,4) -> ", broadcast_sum.to_string());

  writeln("\n=================== Reductions ===================");
  writeln("r.sum() = ", r.sum(), ", r.mean() = ", r.mean());
  auto col_means = r.mean_axis(0);
  writeln("column means (axis=0): ", col_means.to_string());

  writeln("\n=================== Linear algebra ===================");
  auto a = Tensor.from_values(sv([2, 3]), dv([1, 2, 3, 4, 5, 6]));
  auto b = Tensor.from_values(sv([3, 2]), dv([7, 8, 9, 10, 11, 12]));
  writeln("matmul(2x3, 3x2) -> ", a.matmul(b).to_string());

  auto v1 = Tensor.from_values(sv([3]), dv([1, 2, 3]));
  auto v2 = Tensor.from_values(sv([3]), dv([4, 5, 6]));
  writeln("dot([1,2,3], [4,5,6]) = ", v1.dot(v2));
  writeln("outer(v1, v2) -> ", v1.outer(v2).to_string());

  writeln("\n=================== Comparisons & masks ===================");
  auto mask = r.greater_equal(Tensor.full(sv([3, 4]), 6.0));
  writeln("r >= 6 -> ", mask.to_string());
  writeln("count(r >= 6) = ", mask.sum());

  writeln("\n=================== A real dataset as a Tensor ===================");
  auto iris = DataFrame.iris();
  auto n = iris.nrows();
  double[] flat;
  for (size_t i = 0; i < n; i++) {
    flat ~= iris.numeric_at("Sepal.Length", i);
    flat ~= iris.numeric_at("Sepal.Width", i);
    flat ~= iris.numeric_at("Petal.Length", i);
    flat ~= iris.numeric_at("Petal.Width", i);
  }
  auto x = Tensor.from_values(sv([n, 4]), dv(flat));
  writeln("iris feature tensor shape: [", shapeStr(x.shape()), "]");

  auto feature_means = x.mean_axis(0);
  auto centered = x.subtract(feature_means.reshape(sv([1, 4])));
  auto scatter = centered.transpose().matmul(centered);
  writeln("feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): ", feature_means.to_string());
  writeln("(X-mean)^T (X-mean) [4x4 scatter matrix]: ", scatter.to_string(16));

  string[] species;
  for (size_t i = 0; i < n; i++) species ~= iris.string_at("Species", i);
  auto species_tensor = Tensor.from_string_values(sv([n]), strv(species));
  auto setosa_mask = species_tensor.equal(Tensor.from_string_values(sv([1]), strv(["setosa"])));
  writeln("setosa count = ", setosa_mask.sum(), " (of ", n, " rows)");
}
