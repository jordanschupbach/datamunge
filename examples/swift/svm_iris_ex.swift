// CMD: just TARGET=svm_iris_ex run-swift
// Support vector machine on iris: RBF vs. linear kernels, confusion matrix and predictions.
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
    let formula = "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width"
    let iris = try Data.iris()!
    print("iris: \(try iris.nrows()) rows x \(try iris.ncols()) cols\n")

    print("=== RBF kernel (default) ===")
    let rbfModel = try Svm(iris, formula)
    try rbfModel.printSummary()
    print("\nConfusion matrix (rows = actual, cols = predicted):\n\(try rbfModel.confusionMatrix()!)")

    print("\n=== Linear kernel, for comparison ===")
    let linearModel = try Svm(iris, formula, "linear")
    print("Training accuracy: \(fmt(try linearModel.trainingAccuracy() * 100))%")
    print("Support vectors: \(try linearModel.numSupportVectors())")

    let newdata = try DataFrame()
    try newdata.addNumericColumn("Sepal.Length", [5.1, 6.0, 6.5, 6.2])
    try newdata.addNumericColumn("Sepal.Width", [3.5, 2.7, 3.0, 2.8])
    try newdata.addNumericColumn("Petal.Length", [1.4, 4.5, 5.5, 4.8])
    try newdata.addNumericColumn("Petal.Width", [0.2, 1.5, 2.0, 1.8])
    print("\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):\n\(try rbfModel.predictFrame(newdata)!)")
} catch {
    print(error)
    exit(1)
}
