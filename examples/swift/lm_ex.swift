// CMD: just TARGET=lm_ex run-swift
// Linear model: mpg ~ hp + wt + transmission, its ANOVA table and predictions.
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
    let cars = try DataFrame()
    try cars.addNumericColumn("hp", [110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0])
    try cars.addNumericColumn("wt", [2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44])
    try cars.addStringColumn("transmission", ["manual", "manual", "manual", "automatic", "automatic",
                                              "automatic", "automatic", "automatic", "automatic", "automatic"])
    try cars.addNumericColumn("mpg", [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2])

    print("Fitting: mpg ~ hp + wt + transmission\n")
    let model = try Lm(cars, "mpg ~ hp + wt + transmission")
    try model.printSummary()

    print("\nSequential ANOVA:\n\(try model.anova()!)")

    let newcars = try DataFrame()
    try newcars.addNumericColumn("hp", [150.0, 90.0])
    try newcars.addNumericColumn("wt", [3.0, 2.5])
    try newcars.addStringColumn("transmission", ["manual", "automatic"])
    print("\nPredictions with 95% confidence intervals:\n\(try model.predictFrame(newcars, "confidence")!)")

    try model.saveDiagnosticPlots("lm_ex_diagnostics")
    print("\nSaved diagnostic plots as lm_ex_diagnostics_*.svg")
} catch {
    print(error)
    exit(1)
}
