// CMD: just TARGET=datamunge_ex run-swift
// Smoke test of the basic bindings: vectors, pairs and a native Callback object.
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
    try hello()

    let v = try makeDvector(1.0, 2.0, 3.0)
    print("sum_dvector:\t\(fmt(try sumDvector(v)))")

    let p = try makeDpair(1.25, 2.75)
    print("sum_dpair:\t\(fmt(try sumDpair(p)))")

    let cb = try Callback()
    defer { cb.close() }
    print("call_with_callback(3.0):\t\(fmt(try callWithCallback(3.0, cb)))")
    let v2 = try mapDvectorWithCallback(try makeDvector(1.0, 2.0, 3.0), cb)
    print("sum_dvector(map_dvector_with_callback(1,2,3)):\t\(fmt(try sumDvector(v2)))")
} catch {
    print(error)
    exit(1)
}
