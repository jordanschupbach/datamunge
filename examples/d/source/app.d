module app;

import std.stdio : writeln;
import octra;

class TimesTwo : Callback {
  override double call(double x) {
    return x * 2.0;
  }
}

void main() {
  hello();

  auto vec = make_dvector(1.0, 2.0, 3.0);
  writeln("sum_dvector(1,2,3) = ", sum_dvector(vec));

  auto pair = make_dpair(4.0, 5.0);
  writeln("sum_dpair(4,5) = ", sum_dpair(pair));

  auto cb = new TimesTwo();
  writeln("call_with_callback(3.0) = ", call_with_callback(3.0, cb));
  auto vec2 = map_dvector_with_callback(vec, cb);
  writeln("sum_dvector(map_dvector_with_callback(1,2,3)) = ", sum_dvector(vec2));
}
