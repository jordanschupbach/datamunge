// CMD: just TARGET=clustering_ex run-swift
// Clustering iris: k-means, agglomerative (several linkages, dendrogram re-cut) and DBSCAN.
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
    let species = try (0..<iris.nrows()).map { try iris.stringAt("Species", $0) }
    let featureColumns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]

    // For each cluster, print its size and its majority species (negative labels are noise).
    func reportPurity(_ clusterLabels: [Int], _ nClusters: Int) {
        var votes = [[String: Int]](repeating: [:], count: nClusters)
        for (label, sp) in zip(clusterLabels, species) where label >= 0 {
            votes[label][sp, default: 0] += 1
        }
        for c in 0..<nClusters {
            let total = votes[c].values.reduce(0, +)
            if total == 0 { continue }
            // The first species to reach the maximum count, in order of first appearance in the
            // cluster (C#'s MaxBy over a Dictionary's insertion order).
            var bestSpecies = "", best = 0
            for (label, sp) in zip(clusterLabels, species) where label == c && votes[c][sp]! > best {
                (bestSpecies, best) = (sp, votes[c][sp]!)
            }
            print("  cluster \(c): \(total) points, majority \(bestSpecies) (\(best)/\(total))")
        }
    }

    print("=================== K-means (k=3) on iris ===================")
    let kmeans = try KMeans(iris, featureColumns, 3)
    print("Inertia: \(fmt(try kmeans.inertia())), iterations (best run): \(try kmeans.iterationsUsed())")
    reportPurity(try kmeans.labels(), 3)

    print("\n=================== Agglomerative clustering on iris ===================")
    for linkage in ["ward", "complete", "average"] {
        let model = try AgglomerativeClustering(iris, featureColumns, 3, linkage)
        print("--- linkage=\(linkage) ---")
        reportPurity(try model.labels(), 3)
    }

    print("\n--- Re-cutting the ward dendrogram at k=2 without refitting ---")
    let wardModel = try AgglomerativeClustering(iris, featureColumns, 3, "ward")
    reportPurity(try wardModel.cut(2), 2)

    print("\n=================== DBSCAN on iris ===================")
    let dbscan = try Dbscan(iris, featureColumns, 0.6, 5)
    print("Clusters found: \(try dbscan.nClusters()), noise points: \(try dbscan.nNoise()) (of \(try dbscan.observations()))")
    reportPurity(try dbscan.labels().map { Int($0) }, try dbscan.nClusters())
} catch {
    print(error)
    exit(1)
}
