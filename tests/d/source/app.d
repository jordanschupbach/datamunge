module app;

import octra;

class TimesTwo : octra.Callback {
  override double call(double x) {
    return x * 2.0;
  }
}

unittest {
  hello();

  auto vec = make_dvector(1.0, 2.0, 3.0);
  assert(sum_dvector(vec) == 6.0);

  auto pair = make_dpair(1.0, 2.0);
  assert(sum_dpair(pair) == 3.0);

  auto cb = new TimesTwo();
  assert(call_with_callback(3.0, cb) == 6.0);
  auto out = map_dvector_with_callback(vec, cb);
  assert(sum_dvector(out) == 12.0);
}

void main() {}
