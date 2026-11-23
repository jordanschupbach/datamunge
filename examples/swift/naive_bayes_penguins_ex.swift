// CMD: just TARGET=naive_bayes_penguins_ex run-swift
// Naive Bayes on penguins: mixed numeric/categorical predictors, confusion matrix, misclassified rows, plots.
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
    // Note: NaiveBayesClassifier drops incomplete rows for its own formula columns internally
    // when fitting, matching every other formula-based model in this library.
    let penguins = try Data.penguins()!
    print("penguins: \(try penguins.nrows()) rows x \(try penguins.ncols()) cols\n")

    // A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
    // predictors in one formula -- each modeled independently given the class, per the naive
    // Bayes assumption.
    let model = try NaiveBayesClassifier(penguins, "species ~ bill_length_mm + bill_depth_mm + island + sex")
    try model.printSummary()

    print("\nConfusion matrix (rows = actual, cols = predicted):\n\(try model.confusionMatrix()!)")

    print("\nMisclassified rows:")
    let predictions = try model.predict(penguins)
    var misclassified = 0
    let columns = ["species", "bill_length_mm", "bill_depth_mm", "island", "sex"]
    for i in 0..<(try penguins.nrows()) {
        if try columns.contains(where: { try penguins.isNull($0, i) }) { continue }
        let (actual, predicted) = (try penguins.stringAt("species", i), predictions[i])
        if predicted == actual { continue }
        misclassified += 1
        print("  row \(i): bill_length=\(fmt(try penguins.numericAt("bill_length_mm", i))) bill_depth=\(fmt(try penguins.numericAt("bill_depth_mm", i))) "
              + "island=\(try penguins.stringAt("island", i)) sex=\(try penguins.stringAt("sex", i))  actual=\(actual)  predicted=\(predicted)")
    }
    print("\(misclassified) misclassified (of \(try penguins.nrows()) rows, some incomplete)")

    // PlotDecisionRegions requires exactly two NUMERIC predictors, so build a separate
    // two-predictor model (bill measurements alone) just for visualization.
    let billOnly = try NaiveBayesClassifier(penguins, "species ~ bill_length_mm + bill_depth_mm")
    print("\nbill-measurements-only model training accuracy: \(fmt(100 * (try billOnly.trainingAccuracy())))%")
    try billOnly.plotClassification(penguins, "bill_length_mm", "bill_depth_mm").save("naive_bayes_penguins_classification.svg")
    try billOnly.plotDecisionRegions("bill_length_mm", "bill_depth_mm").save("naive_bayes_penguins_decision_regions.svg")
    print("Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg")
} catch {
    print(error)
    exit(1)
}
