// CMD: just TARGET=bspline_ex run-swift
// Tensor-product B-spline regression: fit z ~ bs(x, y) to a smooth surface and predict.
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

/// A Double through a C printf format, e.g. cfmt("%.6f", x) for C#'s {x:F6}.
func cfmt(_ format: String, _ x: Double) -> String {
    var buffer = [CChar](repeating: 0, count: 64)
    _ = withVaList([x]) { vsnprintf(&buffer, buffer.count, format, $0) }
    return String(cString: buffer)
}

do {
    // Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2.
    var x: [Double] = [], y: [Double] = [], z: [Double] = []
    for iy in 0..<8 {
        for ix in 0..<8 {
            let xv = Double(ix) / 7.0, yv = Double(iy) / 7.0
            x.append(xv)
            y.append(yv)
            z.append(sin(xv) + yv * yv)
        }
    }
    let data = try DataFrame()
    try data.addNumericColumn("x", x)
    try data.addNumericColumn("y", y)
    try data.addNumericColumn("z", z)

    let surface = try Lm(data, "z ~ bs(x, y)")
    print("Fitted z ~ bs(x, y)")
    try surface.printSummary()

    let newPoints = try DataFrame()
    try newPoints.addNumericColumn("x", [0.25, 0.75])
    try newPoints.addNumericColumn("y", [0.50, 0.25])
    let pred = try surface.predict(newPoints)
    print("Predictions: [\(cfmt("%.6f", pred[0])), \(cfmt("%.6f", pred[1]))]")
} catch {
    print(error)
    exit(1)
}
