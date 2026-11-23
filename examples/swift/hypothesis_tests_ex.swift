// CMD: just TARGET=hypothesis_tests_ex run-swift
// Classical hypothesis tests (t, Wilcoxon, ANOVA, correlation, F, normality, chi-squared, binomial) on iris.
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

func printResult(_ label: String, _ r: HypothesisTestResult) {
    let alternative: String
    switch r.alternative {
    case .less: alternative = "less"
    case .greater: alternative = "greater"
    default: alternative = "two.sided"
    }
    var line = "\(label): statistic=\(fmt(r.statistic))"
    if r.parameter1 != 0 { line += ", df1=\(fmt(r.parameter1))" }
    if r.parameter2 != 0 { line += ", df2=\(fmt(r.parameter2))" }
    line += ", p=\(fmt(r.pValue)) (\(alternative))"
    if r.hasConfInt { line += ", CI=[\(fmt(r.confIntLower)), \(fmt(r.confIntUpper))]" }
    print("\(line) -- \(r.method)")
}

func columnForSpecies(_ df: DataFrame, _ column: String, _ species: String) throws -> [Double] {
    var values: [Double] = []
    for i in 0..<(try df.nrows()) where try df.stringAt("Species", i) == species {
        values.append(try df.numericAt(column, i))
    }
    return values
}

func column(_ df: DataFrame, _ column: String) throws -> [Double] {
    try (0..<(try df.nrows())).map { try df.numericAt(column, $0) }
}

do {
    let iris = try Data.iris()!

    let setosaPetal = try columnForSpecies(iris, "Petal.Length", "setosa")
    let versicolorPetal = try columnForSpecies(iris, "Petal.Length", "versicolor")
    let virginicaPetal = try columnForSpecies(iris, "Petal.Length", "virginica")

    print("=================== t-tests: petal length, setosa vs. versicolor ===================")
    printResult("Welch two-sample t-test", try tTestTwoSample(setosaPetal, versicolorPetal))
    printResult("Wilcoxon rank-sum test", try wilcoxonRankSumTest(setosaPetal, versicolorPetal))

    print("\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================")
    let allPetal = setosaPetal + versicolorPetal + virginicaPetal
    let sizes = [setosaPetal.count, versicolorPetal.count, virginicaPetal.count]
    printResult("One-way ANOVA", try oneWayAnova(allPetal, sizes))
    printResult("Kruskal-Wallis", try kruskalWallisTest(allPetal, sizes))

    print("\n=================== Correlation: sepal length vs. petal length ===================")
    let sepalLength = try column(iris, "Sepal.Length")
    let petalLength = try column(iris, "Petal.Length")
    printResult("Pearson correlation", try pearsonCorrelationTest(sepalLength, petalLength))
    printResult("Spearman correlation", try spearmanCorrelationTest(sepalLength, petalLength))

    print("\n=================== F-test: petal length variance, setosa vs. virginica ===================")
    printResult("F test", try fTestVariance(setosaPetal, virginicaPetal))

    print("\n=================== Normality: is sepal length normally distributed within setosa? ===================")
    let setosaSepal = try columnForSpecies(iris, "Sepal.Length", "setosa")
    printResult("Shapiro-Francia", try shapiroFranciaTest(setosaSepal))
    printResult("KS vs. fitted normal", try ksTestOneSampleNormal(setosaSepal, 5.006, 0.3525))

    print("\n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? ===================")
    let medianAll = allPetal.sorted()[allPetal.count / 2]
    func countLong(_ v: [Double]) -> Int { v.filter { $0 > medianAll }.count }
    do {
        let setosaLong = countLong(setosaPetal)
        let setosaShort = setosaPetal.count - setosaLong
        let versicolorLong = countLong(versicolorPetal)
        let versicolorShort = versicolorPetal.count - versicolorLong
        print("table: setosa=[\(setosaLong),\(setosaShort)] versicolor=[\(versicolorLong),\(versicolorShort)]")
        let table = [setosaLong, setosaShort, versicolorLong, versicolorShort].map(Double.init)
        printResult("Chi-squared independence", try chiSquaredTestIndependence(table, 2, 2))
        printResult("Fisher's exact test", try fisherExactTest2x2(setosaLong, setosaShort, versicolorLong, versicolorShort))
    }

    print("\n=================== Proportion / binomial: fraction of \"long\" petals overall ===================")
    do {
        let longCount = countLong(allPetal)
        let n = allPetal.count
        printResult("One-sample proportion test (vs 0.5)", try proportionTestOneSample(longCount, n, 0.5))
        printResult("Exact binomial test (vs 0.5)", try binomialTest(longCount, n, 0.5))
    }
} catch {
    print(error)
    exit(1)
}
