// CMD: just TARGET=gbm_iris_ex run-swift
// Gradient-boosted trees (GBM) on iris: summary, confusion matrix, misclassified rows, 5 vs 100 rounds.
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

    let model = try GbmClassifier(iris, "Species ~ Petal.Length + Petal.Width")
    try model.printSummary()

    print("\nConfusion matrix (rows = actual, cols = predicted):\n\(try model.confusionMatrix()!)")

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

    try model.plotTrainingDeviance().save("gbm_iris_training_deviance.svg")
    try model.plotDecisionRegions("Petal.Length", "Petal.Width").save("gbm_iris_decision_regions.svg")
    print("\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg")

    // A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away
    // at the training loss, gradually sharpening the decision boundary.
    // try GbmClassifier(data, formula, nTrees, learningRate, maxDepth, minSamplesSplit,
    //                   minSamplesLeaf, subsample, seed)
    let few = try GbmClassifier(iris, "Species ~ Petal.Length + Petal.Width", 5)
    print("\n5-round ensemble:   training accuracy=\(fmt(100 * (try few.trainingAccuracy())))%  deviance=\(fmt(try few.trainingDeviance().last!))")
    print("100-round ensemble: training accuracy=\(fmt(100 * (try model.trainingAccuracy())))%  deviance=\(fmt(try model.trainingDeviance().last!))")
    try few.plotDecisionRegions("Petal.Length", "Petal.Width").save("gbm_iris_decision_regions_5rounds.svg")
    print("Saved gbm_iris_decision_regions_5rounds.svg")
} catch {
    print(error)
    exit(1)
}
