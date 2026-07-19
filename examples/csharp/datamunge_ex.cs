using System;

class Program {
  static void Main() {
    datamunge.hello();

    var v = datamunge.make_dvector(1.0, 2.0, 3.0);
    Console.WriteLine("sum_dvector: " + datamunge.sum_dvector(v));

    var p = datamunge.make_dpair(1.25, 2.75);
    Console.WriteLine("sum_dpair: " + datamunge.sum_dpair(p));

    var cb = new Callback();
    Console.WriteLine("call_with_callback(3.0): " + datamunge.call_with_callback(3.0, cb));
    var v2 = datamunge.map_dvector_with_callback(datamunge.make_dvector(1.0, 2.0, 3.0), cb);
    Console.WriteLine("sum_dvector(map_dvector_with_callback(1,2,3)): " + datamunge.sum_dvector(v2));
  }
}
