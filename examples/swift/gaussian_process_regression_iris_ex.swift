// CMD: just TARGET=gaussian_process_regression_iris_ex run-swift
// Gaussian process regression of Petal.Length on Petal.Width (iris), with confidence intervals.
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
    let formula = "Petal.Length ~ Petal.Width"

    let iris = try Data.iris()!
    print("iris: \(try iris.nrows()) rows x \(try iris.ncols()) cols")
    print("formula: \(formula)\n")

    // Both the length scale and the noise ratio are auto-selected by maximizing the exact log
    // marginal likelihood.
    let model = try GaussianProcessRegression(iris, formula)
    try model.printSummary()

    try model.plotFit(iris).save("gpr_iris_fit.svg")
    try model.plotLengthScaleProfile().save("gpr_iris_length_scale_profile.svg")
    print("\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg")

    // Unlike every other regressor in this suite, a GP gives a genuine posterior confidence
    // interval at every point -- including far outside the training data, where it should widen
    // substantially as the model's uncertainty grows.
    let query = try DataFrame()
    try query.addNumericColumn("Petal.Width", [0.2, 1.3, 2.5, 10.0])
    let detail = try model.predictFrame(query, "confidence")!
    print("\nPredictions with 95% confidence intervals:")
    print(detail)
    print("(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its\n interval is than the in-range predictions.)")
} catch {
    print(error)
    exit(1)
}
