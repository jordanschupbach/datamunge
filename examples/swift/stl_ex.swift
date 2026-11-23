// CMD: just TARGET=stl_ex run-swift
// C++ std::vector and std::pair values in Swift: vectors are arrays, pairs are tuples.
import Datamunge
#if canImport(Glibc)
import Glibc
#else
import Darwin
#endif

/// A Double as C#'s default ToString() prints it: the shortest round-trip digits, in E notation
/// when the decimal exponent is below -4 or at least 15 (6, 0.1, 1E-05, 1E+15).
func fmt(_ x: Double) -> String {
    if x.isNaN { return "NaN" }
    if x.isInfinite { return x < 0 ? "-∞" : "∞" }
    var s = "\(x)"
    let negative = s.hasPrefix("-")
    if negative { s.removeFirst() }
    var mantissa = Substring(s)
    var exponent = 0
    if let e = s.firstIndex(of: "e") {
        mantissa = s[..<e]
        exponent = Int(s[s.index(after: e)...])!
    }
    let parts = mantissa.split(separator: ".", omittingEmptySubsequences: false)
    var digits = String(parts[0]) + (parts.count > 1 ? String(parts[1]) : "")
    var point = parts[0].count + exponent
    while digits.hasPrefix("0") { digits.removeFirst(); point -= 1 }
    while digits.hasSuffix("0") { digits.removeLast() }
    let sign = negative ? "-" : ""
    if digits.isEmpty { return sign + "0" }
    let e = point - 1
    if e < -4 || e >= 15 {
        let rest = digits.dropFirst()
        let magnitude = abs(e) < 10 ? "0\(abs(e))" : "\(abs(e))"
        return sign + String(digits.first!) + (rest.isEmpty ? "" : "." + rest) + "E" + (e < 0 ? "-" : "+") + magnitude
    }
    if point <= 0 { return sign + "0." + String(repeating: "0", count: -point) + digits }
    if point >= digits.count { return sign + digits + String(repeating: "0", count: point - digits.count) }
    return sign + digits.prefix(point) + "." + digits.dropFirst(point)
}


do {
    let n = 100
    let v = (0..<n).map { Double($0) * 1.5 }
    v.forEach { print(fmt($0)) }
    let v2 = (0..<n).map { Int((Double($0) * 1.5).rounded(.down)) }
    v2.forEach { print($0) }

    let p = (3, 4)
    print("p: (\(p.0), \(p.1))")

    let p2 = (10, 20)
    print("p2: (\(p2.0), \(p2.1))")

    // A C++ std::pair<double,double> comes back as a tuple.
    let p3 = try makeDpair(1.25, 2.75)
    print("make_dpair(1.25, 2.75): (\(fmt(p3.0)), \(fmt(p3.1))), sum_dpair = \(fmt(try sumDpair(p3)))")
} catch {
    print(error)
    exit(1)
}
