// CMD: just TARGET=knn_iris_ex run-swift
// k-nearest-neighbours classifier on iris: leave-one-out accuracy and decision regions for several k.
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
    print("iris: \(try iris.nrows()) rows x \(try iris.ncols()) cols\n")

    let model = try KnnClassifier(iris, "Species ~ Petal.Length + Petal.Width")
    try model.printSummary()

    print("\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):")
    print(try model.confusionMatrix()!)

    print("\nLeave-one-out misclassified rows:")
    let fitted = try model.predict(iris)
    let n = try iris.nrows()
    var misclassified = 0
    for i in 0..<n {
        let (actual, predicted) = (try iris.stringAt("Species", i), fitted[i])
        if predicted == actual { continue }
        misclassified += 1
        print("  row \(i): Petal.Length=\(fmt(try iris.numericAt("Petal.Length", i))) Petal.Width=\(fmt(try iris.numericAt("Petal.Width", i)))  actual=\(actual)  predicted=\(predicted)")
    }
    print("\(misclassified) of \(n) misclassified (\(fmt(100.0 * Double(misclassified) / Double(n)))%)")

    try model.plotDecisionRegions("Petal.Length", "Petal.Width").save("knn_iris_decision_regions_k5.svg")
    print("\nSaved knn_iris_decision_regions_k5.svg")

    // k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
    // every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
    // underfit boundary). KnnClassifier(data, formula, k, metric, weighted, standardize)
    for k in [1, 25] {
        let knn = try KnnClassifier(iris, "Species ~ Petal.Length + Petal.Width", k)
        let file = "knn_iris_decision_regions_k\(k).svg"
        let label = "\(k)".padding(2)
        print("\nk=\(label) leave-one-out accuracy: \(fmt(try knn.trainingAccuracy() * 100))%")
        try knn.plotDecisionRegions("Petal.Length", "Petal.Width").save(file)
        print("Saved \(file)")
    }
} catch {
    print(error)
    exit(1)
}

extension String {
    /// The string left-aligned in `width` columns, like C#'s {value,-width}.
    func padding(_ width: Int) -> String {
        count >= width ? self : self + String(repeating: " ", count: width - count)
    }
}
