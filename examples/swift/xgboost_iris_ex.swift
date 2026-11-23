// CMD: just TARGET=xgboost_iris_ex run-swift
// XGBoost classifier on iris: summary, confusion matrix, misclassified rows, L2-regularization comparison.
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

    let model = try XgBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width")
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

    try model.plotTrainingDeviance().save("xgboost_iris_training_deviance.svg")
    try model.plotDecisionRegions("Petal.Length", "Petal.Width").save("xgboost_iris_decision_regions.svg")
    print("\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg")

    // XGBoost's defining lever is regularization: a heavily L2-regularized model (large lambda)
    // grows the same deep trees but keeps every leaf weight small, producing a much smoother
    // decision boundary than the lightly-regularized default.
    // XgBoostClassifier(data, formula, nTrees, learningRate, maxDepth, lambda, alpha, gamma,
    //                   minChildWeight, minSamplesLeaf, subsample, colsampleBytree, seed)
    let heavy = try XgBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width", 100, 0.3, 6, 50.0)
    print("\nlambda=1 (default):   training accuracy=\(fmt(100 * (try model.trainingAccuracy())))%  deviance=\(fmt(try model.trainingDeviance().last!))")
    print("lambda=50 (heavy L2): training accuracy=\(fmt(100 * (try heavy.trainingAccuracy())))%  deviance=\(fmt(try heavy.trainingDeviance().last!))")
    try heavy.plotDecisionRegions("Petal.Length", "Petal.Width").save("xgboost_iris_decision_regions_heavy_lambda.svg")
    print("Saved xgboost_iris_decision_regions_heavy_lambda.svg")
} catch {
    print(error)
    exit(1)
}
