package js.datamunge.jdatamunge.examples;

import js.datamunge.jdatamunge.DVector;
import js.datamunge.jdatamunge.SVector;
import js.datamunge.jdatamunge.SizeVector;
import js.datamunge.jdatamunge.Tensor;
import js.datamunge.jdatamunge.DataFrame;

import java.util.ArrayList;
import java.util.List;

public class TensorEx {
  static {
    System.loadLibrary("datamunge_jni");
  }

  static String shapeStr(SizeVector shp) {
    List<String> parts = new ArrayList<>();
    for (int i = 0; i < shp.size(); i++) parts.add(String.valueOf(shp.get(i)));
    return String.join(", ", parts);
  }

  public static void run() {
    System.out.println("=================== Construction ===================");
    var z = Tensor.zeros(new SizeVector(new long[] {2, 3}));
    System.out.println("zeros([2,3]): " + z.to_string());

    var eyet = Tensor.eye(3);
    System.out.println("eye(3): " + eyet.to_string());

    var r = Tensor.arange(0.0, 12.0, 1.0).reshape(new SizeVector(new long[] {3, 4}));
    System.out.println("arange(0,12).reshape([3,4]): " + r.to_string());

    System.out.println("\n=================== Shape ops ===================");
    var rt = r.transpose();
    System.out.println("transpose -> shape [" + shapeStr(rt.shape()) + "]");
    var sliced = r.slice(1, 1, 3);
    System.out.println("slice(axis=1, start=1, stop=3): " + sliced.to_string());

    System.out.println("\n=================== Broadcasting arithmetic ===================");
    var col = Tensor.from_values(new SizeVector(new long[] {3, 1}), new DVector(new double[] {1, 2, 3}));
    var row = Tensor.from_values(new SizeVector(new long[] {1, 4}), new DVector(new double[] {10, 20, 30, 40}));
    var broadcastSum = col.add(row);
    System.out.println("(3,1) + (1,4) -> " + broadcastSum.to_string());

    System.out.println("\n=================== Reductions ===================");
    System.out.println("r.sum() = " + r.sum() + ", r.mean() = " + r.mean());
    var colMeans = r.mean_axis(0);
    System.out.println("column means (axis=0): " + colMeans.to_string());

    System.out.println("\n=================== Linear algebra ===================");
    var a = Tensor.from_values(new SizeVector(new long[] {2, 3}), new DVector(new double[] {1, 2, 3, 4, 5, 6}));
    var b = Tensor.from_values(new SizeVector(new long[] {3, 2}), new DVector(new double[] {7, 8, 9, 10, 11, 12}));
    System.out.println("matmul(2x3, 3x2) -> " + a.matmul(b).to_string());

    var v1 = Tensor.from_values(new SizeVector(new long[] {3}), new DVector(new double[] {1, 2, 3}));
    var v2 = Tensor.from_values(new SizeVector(new long[] {3}), new DVector(new double[] {4, 5, 6}));
    System.out.println("dot([1,2,3], [4,5,6]) = " + v1.dot(v2));
    System.out.println("outer(v1, v2) -> " + v1.outer(v2).to_string());

    System.out.println("\n=================== Comparisons & masks ===================");
    var mask = r.greater_equal(Tensor.full(new SizeVector(new long[] {3, 4}), 6.0));
    System.out.println("r >= 6 -> " + mask.to_string());
    System.out.println("count(r >= 6) = " + mask.sum());

    System.out.println("\n=================== A real dataset as a Tensor ===================");
    var iris = DataFrame.iris();
    long n = iris.nrows();
    List<Double> flat = new ArrayList<>();
    for (long i = 0; i < n; i++) {
      flat.add(iris.numeric_at("Sepal.Length", i));
      flat.add(iris.numeric_at("Sepal.Width", i));
      flat.add(iris.numeric_at("Petal.Length", i));
      flat.add(iris.numeric_at("Petal.Width", i));
    }
    var x = Tensor.from_values(new SizeVector(new long[] {n, 4}), new DVector(flat));
    System.out.println("iris feature tensor shape: [" + shapeStr(x.shape()) + "]");

    var featureMeans = x.mean_axis(0);
    var centered = x.subtract(featureMeans.reshape(new SizeVector(new long[] {1, 4})));
    var scatter = centered.transpose().matmul(centered);
    System.out.println("feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): " + featureMeans.to_string());
    System.out.println("(X-mean)^T (X-mean) [4x4 scatter matrix]: " + scatter.to_string(16));

    List<String> species = new ArrayList<>();
    for (long i = 0; i < n; i++) species.add(iris.string_at("Species", i));
    var speciesTensor = Tensor.from_string_values(new SizeVector(new long[] {n}), new SVector(species));
    var setosaMask = speciesTensor.equal(Tensor.from_string_values(new SizeVector(new long[] {1}), new SVector(new String[] {"setosa"})));
    System.out.println("setosa count = " + setosaMask.sum() + " (of " + n + " rows)");
  }
}
