package js.datamunge.jdatamunge.examples;

import js.datamunge.jdatamunge.Callback;
import js.datamunge.jdatamunge.datamunge;

public class DatamungeEx {
  static {
    System.loadLibrary("datamunge_jni");
  }

  public static void run() {
    datamunge.hello();

    var v = datamunge.make_dvector(1.0, 2.0, 3.0);
    System.out.println("sum_dvector: " + datamunge.sum_dvector(v));

    var p = datamunge.make_dpair(1.25, 2.75);
    System.out.println("sum_dpair: " + datamunge.sum_dpair(p));

    var cb = new Callback();
    System.out.println("call_with_callback(3.0): " + datamunge.call_with_callback(3.0, cb));
    var v2 = datamunge.map_dvector_with_callback(datamunge.make_dvector(1.0, 2.0, 3.0), cb);
    System.out.println("sum_dvector(map_dvector_with_callback(1,2,3)): " + datamunge.sum_dvector(v2));
  }
}
