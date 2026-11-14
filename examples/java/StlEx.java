// CMD: just TARGET=stl_ex run-java
// C++ std::vector and std::pair values in Java: vectors are arrays, pairs are datamunge.Pair records.
import datamunge.Datamunge;
import datamunge.Pair;
import java.util.stream.IntStream;

public class StlEx {
  public static void main(String[] args) {
    int n = 100;
    double[] v = IntStream.range(0, n).mapToDouble(i -> i * 1.5).toArray();
    for (double x : v) {
      System.out.println(x);
    }
    int[] v2 = IntStream.range(0, n).map(i -> (int) Math.floor(i * 1.5)).toArray();
    for (int x : v2) {
      System.out.println(x);
    }

    var p = new Pair<>(3, 4);
    System.out.println("p: (" + p.first() + ", " + p.second() + ")");

    var p2 = new Pair<>(10, 20);
    System.out.println("p2: (" + p2.first() + ", " + p2.second() + ")");

    // A C++ std::pair<double,double> comes back as a Pair<Double, Double>.
    Pair<Double, Double> p3 = Datamunge.makeDpair(1.25, 2.75);
    System.out.println(
        "make_dpair(1.25, 2.75): ("
            + p3.first()
            + ", "
            + p3.second()
            + "), sum_dpair = "
            + Datamunge.sumDpair(p3));
  }
}
