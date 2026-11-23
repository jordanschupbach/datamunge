// CMD: just TARGET=ggplot_ex run-swift
// Demonstrates GGPlot -- the ggplot2-style grammar-of-graphics library.
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

    // ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
    try GgPlot(iris, "Sepal.Length", "Sepal.Width", "Species")
        .geomPoint()
        .labs("Iris Sepal Dimensions", "Sepal Length", "Sepal Width")
        .themeMinimal()
        .saveSvg("swift_ggplot_point.svg")

    // + geom_smooth(): an lm() fit line, reusing stats::LM internally.
    try GgPlot(iris, "Sepal.Length", "Petal.Length")
        .geomPoint(try Rgb(156, 163, 175), 2.5)
        .geomSmooth()
        .labs("Petal Length vs Sepal Length With a Linear Fit", "Sepal Length", "Petal Length")
        .saveSvg("swift_ggplot_smooth.svg")

    // geom_bar(): counts a discrete column (stat = "count").
    try GgPlot(iris, "Species")
        .geomBar()
        .labs("Observations per Species", "Species", "Count")
        .themeBw()
        .saveSvg("swift_ggplot_bar.svg")

    // geom_boxplot(): grouped by a discrete x column.
    try GgPlot(iris, "Species", "Petal.Width")
        .geomBoxplot()
        .labs("Petal Width by Species", "Species", "Petal Width")
        .saveSvg("swift_ggplot_boxplot.svg")

    // geom_histogram() + geom_density(): distribution of a single numeric column.
    try GgPlot(iris, "Sepal.Length")
        .geomHistogram(20)
        .labs("Distribution of Sepal Length", "Sepal Length", "Count")
        .saveSvg("swift_ggplot_histogram.svg")

    try GgPlot(iris, "Sepal.Length")
        .geomDensity()
        .labs("Density of Sepal Length", "Sepal Length", "Density")
        .saveSvg("swift_ggplot_density.svg")

    // facet_wrap(): one panel per Species, composed via RLayout under the hood.
    try GgPlot(iris, "Petal.Length", "Petal.Width")
        .geomPoint()
        .facetWrap("Species")
        .labs("Petal Dimensions", "Petal Length", "Petal Width")
        .saveSvg("swift_ggplot_facet.svg")

    // theme_classic(): another built-in theme.
    try GgPlot(iris, "Sepal.Length", "Sepal.Width", "Species")
        .geomPoint()
        .themeClassic()
        .saveSvg("swift_ggplot_classic.svg")

    print("Wrote 7 SVGs to swift_ggplot_*.svg")
} catch {
    print(error)
    exit(1)
}
