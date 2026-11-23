// CMD: just TARGET=tree_iris_ex run-swift
// Decision tree classifier on iris: summary, confusion matrix, misclassified rows and plots.
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
    let iris = try Data.iris()!
    print("iris: \(try iris.nrows()) rows x \(try iris.ncols()) cols\n")

    let model = try DecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width")
    try model.printSummary()

    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print(try model.confusionMatrix()!)

    print("\nMisclassified rows:")
    let predictions = try model.predict(iris)
    let n = try iris.nrows()
    var misclassified = 0
    for i in 0..<n {
        let (actual, predicted) = (try iris.stringAt("Species", i), predictions[i])
        if predicted == actual { continue }
        misclassified += 1
        print("  row \(i): Petal.Length=\(fmt(try iris.numericAt("Petal.Length", i))) Petal.Width=\(fmt(try iris.numericAt("Petal.Width", i)))  actual=\(actual)  predicted=\(predicted)")
    }
    print("\(misclassified) of \(n) misclassified (\(fmt(100.0 * Double(misclassified) / Double(n)))%)")

    try model.plotClassification(iris, "Petal.Length", "Petal.Width").save("tree_iris_classification.svg")
    try model.plotDecisionRegions("Petal.Length", "Petal.Width").save("tree_iris_decision_regions.svg")
    print("\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg")

    // A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision boundary.
    let shallow = try DecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width", 2)
    print("\nDepth-2 tree training accuracy: \(fmt(try shallow.trainingAccuracy() * 100))% (\(try shallow.leafCount()) leaves)")
    try shallow.plotDecisionRegions("Petal.Length", "Petal.Width").save("tree_iris_decision_regions_depth2.svg")
    print("Saved tree_iris_decision_regions_depth2.svg")
} catch {
    print(error)
    exit(1)
}
