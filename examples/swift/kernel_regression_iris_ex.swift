// CMD: just TARGET=kernel_regression_iris_ex run-swift
// Nadaraya-Watson kernel regression on iris: CV-selected bandwidth vs. too-small and too-large ones.
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

let formula = "Petal.Length ~ Petal.Width"

do {
    let iris = try Data.iris()!
    print("iris: \(try iris.nrows()) rows x \(try iris.ncols()) cols")
    print("formula: \(formula)\n")

    let model = try KernelRegression(iris, formula)  // bandwidth auto-selected via leave-one-out CV
    try model.printSummary()

    try model.plotFit(iris).save("kernel_regression_iris_fit.svg")
    try model.plotCvCurve().save("kernel_regression_iris_cv.svg")
    print("\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg")

    // A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
    // bias-variance story. KernelRegression(data, formula, kernel, bandwidth, nBandwidth, standardize)
    let small = try KernelRegression(iris, formula, "gaussian", 0.05)
    print("\nbandwidth=0.05 (too small): LOO R-squared=\(fmt(try small.rSquared()))  LOO RMSE=\(fmt(try small.rmse()))")
    try small.plotFit(iris).save("kernel_regression_iris_fit_small_bandwidth.svg")

    let large = try KernelRegression(iris, formula, "gaussian", 5.0)
    print("bandwidth=5.0 (too large):  LOO R-squared=\(fmt(try large.rSquared()))  LOO RMSE=\(fmt(try large.rmse()))")
    try large.plotFit(iris).save("kernel_regression_iris_fit_large_bandwidth.svg")

    print("bandwidth=\(fmt(try model.bandwidth())) (CV-selected): LOO R-squared=\(fmt(try model.rSquared()))  LOO RMSE=\(fmt(try model.rmse()))")
    print("\nSaved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg")
} catch {
    print(error)
    exit(1)
}
