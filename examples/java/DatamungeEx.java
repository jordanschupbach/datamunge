// CMD: just TARGET=datamunge_ex run-java
// Smoke test of the basic bindings: vectors, pairs and a native Callback object.
import datamunge.Callback;
import datamunge.Datamunge;

public class DatamungeEx {
  public static void main(String[] args) {
    Datamunge.hello();

    double[] v = Datamunge.makeDvector(1.0, 2.0, 3.0);
    System.out.println("sum_dvector:\t" + Datamunge.sumDvector(v));

    var p = Datamunge.makeDpair(1.25, 2.75);
    System.out.println("sum_dpair:\t" + Datamunge.sumDpair(p));

    try (var cb = new Callback()) {
      System.out.println("call_with_callback(3.0):\t" + Datamunge.callWithCallback(3.0, cb));
      double[] v2 = Datamunge.mapDvectorWithCallback(Datamunge.makeDvector(1.0, 2.0, 3.0), cb);
      System.out.println("sum_dvector(map_dvector_with_callback(1,2,3)):\t" + Datamunge.sumDvector(v2));
    }
  }
}
