// CMD: just TARGET=tensor_ex run-java
// Tensors: construction, shape ops, broadcasting, reductions, linear algebra, masks, iris.
import datamunge.Tensor;
import datamunge.data.Data;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.LongStream;

public class TensorEx {
  static String shapeString(Tensor tensor) {
    return LongStream.of(tensor.shape())
        .mapToObj(Long::toString)
        .collect(Collectors.joining(", ", "[", "]"));
  }

  public static void main(String[] args) {
    System.out.println("=================== Construction ===================");
    System.out.println("zeros([2,3]): " + Tensor.zeros(new long[] {2, 3}));
    System.out.println("eye(3): " + Tensor.eye(3));

    var r = Tensor.arange(0.0, 12.0, 1.0).reshape(new long[] {3, 4});
    System.out.println("arange(0,12).reshape([3,4]): " + r);

    System.out.println("\n=================== Shape ops ===================");
    System.out.println("transpose -> shape " + shapeString(r.transpose()));
    System.out.println("slice(axis=1, start=1, stop=3): " + r.slice(1, 1, 3));

    System.out.println("\n=================== Broadcasting arithmetic ===================");
    var col = Tensor.fromValues(new long[] {3, 1}, new double[] {1.0, 2.0, 3.0});
    var row = Tensor.fromValues(new long[] {1, 4}, new double[] {10.0, 20.0, 30.0, 40.0});
    System.out.println("(3,1) + (1,4) -> " + col.add(row));

    System.out.println("\n=================== Reductions ===================");
    System.out.println("r.sum() = " + r.sum() + ", r.mean() = " + r.mean());
    System.out.println("column means (axis=0): " + r.meanAxis(0));

    System.out.println("\n=================== Linear algebra ===================");
    var a = Tensor.fromValues(new long[] {2, 3}, new double[] {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    var b = Tensor.fromValues(new long[] {3, 2}, new double[] {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    System.out.println("matmul(2x3, 3x2) -> " + a.matmul(b));
    var v1 = Tensor.fromValues(new long[] {3}, new double[] {1.0, 2.0, 3.0});
    var v2 = Tensor.fromValues(new long[] {3}, new double[] {4.0, 5.0, 6.0});
    System.out.println("dot([1,2,3], [4,5,6]) = " + v1.dot(v2));
    System.out.println("outer(v1, v2) -> " + v1.outer(v2));

    System.out.println("\n=================== Comparisons & masks ===================");
    var mask = r.greaterEqual(Tensor.full(new long[] {3, 4}, 6.0));
    System.out.println("r >= 6 -> " + mask);
    System.out.println("count(r >= 6) = " + mask.sum());

    System.out.println("\n=================== A real dataset as a Tensor ===================");
    var iris = Data.iris();
    long n = iris.nrows();
    List<String> columns = List.of("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width");
    double[] flat = new double[(int) n * columns.size()];
    int k = 0;
    for (long i = 0; i < n; i++) {
      for (String column : columns) {
        flat[k++] = iris.numericAt(column, i);
      }
    }
    var x = Tensor.fromValues(new long[] {n, 4}, flat);
    System.out.println("iris feature tensor shape: " + shapeString(x));

    var featureMeans = x.meanAxis(0);
    var centered = x.subtract(featureMeans.reshape(new long[] {1, 4}));
    var scatter = centered.transpose().matmul(centered);
    System.out.println(
        "feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): " + featureMeans);
    System.out.println("(X-mean)^T (X-mean) [4x4 scatter matrix]: " + scatter.toString(16));

    List<String> species = new ArrayList<>();
    for (long i = 0; i < n; i++) {
      species.add(iris.stringAt("Species", i));
    }
    var speciesTensor = Tensor.fromStringValues(new long[] {n}, species);
    var setosaMask = speciesTensor.equal(Tensor.fromStringValues(new long[] {1}, List.of("setosa")));
    System.out.println("setosa count = " + setosaMask.sum() + " (of " + n + " rows)");
  }
}
