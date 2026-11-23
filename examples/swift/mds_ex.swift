// CMD: just TARGET=mds_ex run-swift
// Classical multidimensional scaling of iris: euclidean embedding, plot, and a manhattan comparison.
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
    let featureColumns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
    let species = try (0..<(try iris.nrows())).map { try iris.stringAt("Species", $0) }

    print("=================== Classical MDS on iris (euclidean) ===================")
    let mds = try Mds(iris, featureColumns, 2, "euclidean")
    try mds.printSummary()

    print("\nEmbedding as a DataFrame:\n\(try mds.embeddingFrame()!)")

    try mds.plotEmbeddingGrouped(species).saveSvg("mds_iris_embedding.svg")
    print("\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg")

    print("\n=================== Classical MDS on iris (manhattan) ===================")
    let manhattanMds = try Mds(iris, featureColumns, 2, "manhattan")
    print("Goodness of fit: euclidean=\(cfmt("%.4f", try mds.goodnessOfFit())), manhattan=\(cfmt("%.4f", try manhattanMds.goodnessOfFit()))")
} catch {
    print(error)
    exit(1)
}
