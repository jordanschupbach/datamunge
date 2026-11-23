// CMD: just TARGET=elastic_net_iris_ex run-swift
// Ridge, Lasso and Elastic Net on iris, with coefficient-path and CV-curve figures.
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
    let formula = "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width"

    let iris = try Data.iris()!
    print("iris: \(try iris.nrows()) rows x \(try iris.ncols()) cols")
    print("formula: \(formula)\n")

    print("=================== Ridge ===================")
    let ridge = try Ridge(iris, formula)
    try ridge.printSummary()

    print("\n=================== Lasso ===================")
    let lasso = try Lasso(iris, formula)
    try lasso.printSummary()

    print("\n================= Elastic Net =================")
    // try ElasticNet(data, formula, alpha=0.5, lambda, nLambda, cvFolds, standardize, seed)
    let elastic = try ElasticNet(iris, formula, 0.5)
    try elastic.printSummary()

    print("\nSaved figures showing how each model's coefficients respond to the regularization strength, " +
          "and the cross-validation curve used to pick it:")

    let figures: [(path: RPlot, cv: RPlot, name: String, label: String)] = [
        (try ridge.plotCoefficientPath(), try ridge.plotCvCurve(), "ridge", "ridge:      "),
        (try lasso.plotCoefficientPath(), try lasso.plotCvCurve(), "lasso", "lasso:      "),
        (try elastic.plotCoefficientPath(), try elastic.plotCvCurve(), "elasticnet", "elastic net:"),
    ]
    for (path, cv, name, label) in figures {
        try path.save("elastic_net_\(name)_path.svg")
        try cv.save("elastic_net_\(name)_cv.svg")
        print("  \(label) elastic_net_\(name)_path.svg, elastic_net_\(name)_cv.svg")
    }
} catch {
    print(error)
    exit(1)
}
