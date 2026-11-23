// CMD: just TARGET=datamunge_ex run-dart
// Smoke test of the basic bindings: vectors, pairs and a native Callback object.
import 'package:datamunge/datamunge.dart';

/// A double as C#'s default ToString() prints it: the shortest round-trip digits, in E notation
/// when the decimal exponent is below -4 or at least 15 (6, 0.1, 1E-05, 1E+15).
String fmt(double x) {
  if (x.isNaN) return 'NaN';
  if (x.isInfinite) return x < 0 ? '-∞' : '∞';
  var s = x.toString();
  final negative = s.startsWith('-');
  if (negative) s = s.substring(1);
  var mantissa = s;
  var exponent = 0;
  final e = s.indexOf('e');
  if (e >= 0) {
    mantissa = s.substring(0, e);
    exponent = int.parse(s.substring(e + 1));
  }
  final parts = mantissa.split('.');
  var digits = parts[0] + (parts.length > 1 ? parts[1] : '');
  var point = parts[0].length + exponent;
  while (digits.startsWith('0')) {
    digits = digits.substring(1);
    point--;
  }
  while (digits.endsWith('0')) {
    digits = digits.substring(0, digits.length - 1);
  }
  final sign = negative ? '-' : '';
  if (digits.isEmpty) return '${sign}0';
  final exp10 = point - 1;
  if (exp10 < -4 || exp10 >= 15) {
    final rest = digits.substring(1);
    final magnitude = exp10.abs().toString().padLeft(2, '0');
    return '$sign${digits[0]}${rest.isEmpty ? '' : '.$rest'}E${exp10 < 0 ? '-' : '+'}$magnitude';
  }
  if (point <= 0) return '${sign}0.${'0' * -point}$digits';
  if (point >= digits.length) return '$sign$digits${'0' * (point - digits.length)}';
  return '$sign${digits.substring(0, point)}.${digits.substring(point)}';
}

void main() {
  hello();

  final v = makeDvector(1.0, 2.0, 3.0);
  print('sum_dvector:\t${fmt(sumDvector(v))}');

  final p = makeDpair(1.25, 2.75);
  print('sum_dpair:\t${fmt(sumDpair(p))}');

  final cb = Callback();
  print('call_with_callback(3.0):\t${fmt(callWithCallback(3.0, cb))}');
  final v2 = mapDvectorWithCallback(makeDvector(1.0, 2.0, 3.0), cb);
  print('sum_dvector(map_dvector_with_callback(1,2,3)):\t${fmt(sumDvector(v2))}');
  cb.close();
}
