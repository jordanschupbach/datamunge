// CMD: just TARGET=glm_iris_ex run-swift
// Generalized linear models on iris: logistic regression (binomial) and Poisson regression.
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
    let iris = try Data.iris()!
    let rows = Array(0..<(try iris.nrows()))

    // Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
    // the other two on these predictors, which sends logistic regression's coefficients toward
    // +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
    // binary split for a demo.
    let kept = try rows.filter { let s = try iris.stringAt("Species", $0); return s == "versicolor" || s == "virginica" }
    let isVirginica = try kept.map { try iris.stringAt("Species", $0) == "virginica" ? 1.0 : 0.0 }
    let petalLength = try kept.map { try iris.numericAt("Petal.Length", $0) }
    let petalWidth = try kept.map { try iris.numericAt("Petal.Width", $0) }

    let sub = try DataFrame()
    try sub.addNumericColumn("Petal.Length", petalLength)
    try sub.addNumericColumn("Petal.Width", petalWidth)
    try sub.addNumericColumn("is_virginica", isVirginica)

    print("=================== Logistic regression (binomial, logit link) ===================")
    let logit = try Glm(sub, "is_virginica ~ Petal.Length + Petal.Width", "binomial")
    try logit.printSummary()

    let fitted = try logit.fittedValues()
    let n = isVirginica.count
    let correct = (0..<n).filter { (fitted[$0] >= 0.5) == (isVirginica[$0] >= 0.5) }.count
    print("\nResubstitution accuracy at 0.5 threshold: \(fmt(100.0 * Double(correct) / Double(n)))%")

    try logit.saveDiagnosticPlots("glm_logistic_iris")
    print("\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg")

    // Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
    // classic sigmoid shape of a fitted logistic regression, with a 95% confidence band.
    do {
        let gridN = 100
        let widthMean = petalWidth.reduce(0, +) / Double(petalWidth.count)
        let plMin = petalLength.min()! - 0.3, plMax = petalLength.max()! + 0.3
        let grid = try DataFrame()
        try grid.addNumericColumn("Petal.Length", (0..<gridN).map { plMin + (plMax - plMin) * Double($0) / Double(gridN - 1) })
        try grid.addNumericColumn("Petal.Width", Array(repeating: widthMean, count: gridN))
        let curveFrame = try logit.predictFrame(grid, "confidence")!
        print("\nPredicted-probability curve (first 5 rows):")
        print(try curveFrame.toString(5))
    }

    // Poisson regression, for contrast: same IRLS engine, different family/link.
    print("\n=================== Poisson regression (log link) ===================")
    let countData = try DataFrame()
    try countData.addNumericColumn("Sepal.Width", try rows.map { try iris.numericAt("Sepal.Width", $0) })
    try countData.addNumericColumn("Petal.Length", try rows.map { try iris.numericAt("Petal.Length", $0) })
    try countData.addNumericColumn("count", try rows.map { (try iris.numericAt("Sepal.Length", $0) + 0.5).rounded(.down) })
    try Glm(countData, "count ~ Sepal.Width + Petal.Length", "poisson").printSummary()
} catch {
    print(error)
    exit(1)
}
