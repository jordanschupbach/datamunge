// CMD: just TARGET=dataframe_ex run-swift
// DataFrame basics: nulls, drop_duplicates, fill_null, select/sort, group-by, join and summaries.
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
    let sales = try DataFrame()
    try sales.addStringColumn("region", ["west", "west", "east", "south", "south", "south"])
    try sales.addStringColumn("product", ["widget", "widget", "widget", "gizmo", "gizmo", "gizmo"])
    try sales.addNumericColumn("sales", [10.0, 10.0, 14.0, 8.0, 0.0, 11.0], [1, 1, 1, 1, 0, 1])
    try sales.addStringColumn("quarter", ["Q1", "Q1", "Q1", "Q2", "Q2", ""], [1, 1, 1, 1, 1, 0])

    print("raw data\n\(sales)\n")

    let cleaned = try sales.dropDuplicates(["region", "product", "sales", "quarter"])!
    try cleaned.fillNullString("quarter", "unknown")
    try cleaned.fillNullNumeric("sales", 0.0)
    print("after drop_duplicates + fill_null\n\(cleaned)\n")

    let selected = try cleaned.select(["region", "sales", "quarter"])!.sortBy("sales", false)!
    print("selected + sorted\n\(selected)\n")

    let grouped = try cleaned.groupBySum(["region"], ["sales"])!.sortBy("sales", false)!
    print("group_by_sum(region)\n\(grouped)\n")

    let targets = try DataFrame()
    try targets.addStringColumn("region", ["west", "east", "south"])
    try targets.addNumericColumn("target", [18.0, 12.0, 25.0])
    print("joined with targets\n\(try grouped.join(targets, "region", "region", "left")!)\n")

    let shape = try cleaned.shape()
    print("shape = (\(shape[0]), \(shape[1]))")
    print("sales count = \(try cleaned.numericCount("sales"))")
    print("sales nulls = \(try cleaned.numericNullCount("sales"))")
    print("sales sum = \(fmt(try cleaned.numericSum("sales")))")
    print("sales mean = \(fmt(try cleaned.numericMean("sales")))")
} catch {
    print(error)
    exit(1)
}
