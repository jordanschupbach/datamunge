// Smoke tests of the Rosetta-generated Swift binding (just test-swift).
import Datamunge
#if canImport(Glibc)
import Glibc
#else
import Darwin
#endif

var checks = 0
func check(_ ok: Bool, _ message: String) {
    checks += 1
    if !ok {
        print("check failed: \(message)")
        exit(1)
    }
}

final class Doubler: Callback {
    override func call(_ x: Double) throws -> Double { 2 * x }
}

struct FromSwift: Error {}

final class Failing: Callback {
    override func call(_ x: Double) throws -> Double { throw FromSwift() }
}

do {
    // Smoke test: call into the native library.
    try hello()

    // std::vector<double> and std::pair<double,double> round trips.
    let v = try makeDvector(1.0, 2.0, 3.0)
    check(v == [1.0, 2.0, 3.0], "std::vector<double> is [Double]")
    check(try sumDvector(v) == 6.0, "sumDvector")
    check(try sumDvector([0.5, 1.5]) == 2.0, "arrays convert to std::vector")
    let p = try makeDpair(1.25, 2.75)
    check(p == (1.25, 2.75), "std::pair is a tuple")
    check(try sumDpair(p) == 4.0, "sumDpair")

    // The C++ Callback class, and a Swift subclass overriding its virtual call() (a director).
    let native = try Callback()
    check(try callWithCallback(3.0, native) == 3.0, "native Callback")
    check(try sumDvector(try mapDvectorWithCallback([1.0, 2.0, 3.0], native)) == 6.0, "native map")
    let doubler = try Doubler()
    check(try callWithCallback(3.0, doubler) == 6.0, "the Swift override runs")
    check(try mapDvectorWithCallback([1.0, 2.0, 3.0], doubler) == [2.0, 4.0, 6.0], "override from C++ loops")

    // A Swift error inside an override crosses C++ and is rethrown unchanged.
    do {
        _ = try callWithCallback(1.0, try Failing())
        check(false, "the Swift error propagates")
    } catch is FromSwift {
        check(true, "the Swift error propagates")
    }

    // C++ exceptions throw NativeError with their kind.
    do {
        _ = try runif(1.0, 0.0, 3)
        check(false, "C++ exceptions throw")
    } catch let error as NativeError {
        check(error.kind == .invalidArgument, "C++ exceptions throw NativeError")
    }

    // Classes, methods and the policy namespace Data.
    let frame = try DataFrame()
    try frame.addNumericColumn("x", [1.0, 2.0, 3.0])
    check(try frame.nrows() == 3, "DataFrame")
    check(try seql(0.0, 1.0, 5) == [0.0, 0.25, 0.5, 0.75, 1.0], "seql")
    check(try Data.iris()?.nrows() == 150, "the Data namespace")
} catch {
    print("unexpected error: \(error)")
    exit(1)
}
print("test_datamunge.swift: all \(checks) checks passed")
