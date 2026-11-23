// Smoke tests of the Rosetta-generated Dart binding (just test-dart). The native library is the file
// DATAMUNGE_DART_LIBRARY names (build/dart/prefix/lib/libdatamunge_dart.so).
import 'dart:io' show exit;

import 'package:datamunge/datamunge.dart';

var checks = 0;

void check(bool ok, String message) {
  checks++;
  if (!ok) {
    print('check failed: $message');
    exit(1);
  }
}

class Doubler extends Callback {
  @override
  double call(double x) => 2 * x;
}

class FromDart implements Exception {}

class Failing extends Callback {
  @override
  double call(double x) => throw FromDart();
}

void main() {
  // Smoke test: call into the native library.
  hello();

  // std::vector<double> and std::pair<double,double> round trips.
  final v = makeDvector(1, 2, 3);
  check(v.length == 3 && v[0] == 1 && v[2] == 3, 'std::vector<double> is a List<double>');
  check(sumDvector(v) == 6, 'sumDvector');
  final p = makeDpair(1.25, 2.75);
  check(p.$1 == 1.25 && p.$2 == 2.75, 'std::pair is a record');
  check(sumDpair(p) == 4, 'sumDpair');

  // The C++ Callback class, and a Dart subclass overriding its virtual call (a director).
  final native = Callback();
  check(callWithCallback(3, native) == 3, 'native Callback');
  check(sumDvector(mapDvectorWithCallback([1, 2, 3], native)) == 6, 'native map');
  final doubler = Doubler();
  check(callWithCallback(3, doubler) == 6, 'the Dart override runs');
  check(sumDvector(mapDvectorWithCallback([1, 2, 3], doubler)) == 12, 'override from C++ loops');
  native.close();

  // A Dart exception inside an override is thrown again after C++ returns.
  var thrown = false;
  try {
    callWithCallback(1, Failing());
  } on FromDart {
    thrown = true;
  }
  check(thrown, 'the Dart exception propagates');

  // C++ exceptions throw NativeError with their kind.
  thrown = false;
  try {
    runif(1, 0, 3);
  } on NativeError catch (e) {
    thrown = e.kind == NativeErrorKind.invalidArgument;
  }
  check(thrown, 'C++ exceptions throw NativeError');

  // Classes, methods and the policy namespace Data.
  final frame = DataFrame();
  frame.addNumericColumn('x', [1, 2, 3]);
  check(frame.nrows() == 3, 'DataFrame');
  frame.close();
  final seq = seql(0, 1, 5);
  check(seq.length == 5 && seq[1] == 0.25 && seq[4] == 1, 'seql');
  check(Data.iris()!.nrows() == 150, 'the Data namespace');

  print('test_datamunge.dart: all $checks checks passed');
}
