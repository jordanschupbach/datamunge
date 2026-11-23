// CMD: just TARGET=lmm_ex run-swift
// Linear mixed models: random intercept (penguins) and random intercept + slope (simulated schools).
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

/// System.Random(seed) of .NET (the seeded, Knuth subtractive generator), so the simulated data
/// match the C# example's exactly.
struct DotNetRandom {
    private var seeds = [Int](repeating: 0, count: 56)
    private var inext = 0, inextp = 21
    private static let mbig = Int(Int32.max)

    init(_ seed: Int32) {
        var mj = 161803398 - (seed == Int32.min ? Int(Int32.max) : abs(Int(seed)))
        seeds[55] = mj
        var mk = 1
        for i in 1..<55 {
            let ii = (21 * i) % 55
            seeds[ii] = mk
            mk = mj - mk
            if mk < 0 { mk += DotNetRandom.mbig }
            mj = seeds[ii]
        }
        for _ in 1..<5 {
            for i in 1..<56 {
                seeds[i] -= seeds[1 + (i + 30) % 55]
                if seeds[i] < 0 { seeds[i] += DotNetRandom.mbig }
            }
        }
    }

    private mutating func sample() -> Double {
        inext = inext + 1 >= 56 ? 1 : inext + 1
        inextp = inextp + 1 >= 56 ? 1 : inextp + 1
        var value = seeds[inext] - seeds[inextp]
        if value == DotNetRandom.mbig { value -= 1 }
        if value < 0 { value += DotNetRandom.mbig }
        seeds[inext] = value
        return Double(value) * (1.0 / Double(DotNetRandom.mbig))
    }

    /// Random.NextDouble(): [0, 1).
    mutating func nextDouble() -> Double { sample() }

    /// Random.Next(minValue, maxValue): [minValue, maxValue).
    mutating func next(_ minValue: Int, _ maxValue: Int) -> Int {
        Int(sample() * Double(maxValue - minValue)) + minValue
    }
}

var rng = DotNetRandom(2024)
func uniform() -> Double { 1 - rng.nextDouble() }  // (0, 1]: avoids log(0) in the Box-Muller transform.
func gauss() -> Double {
    let radius = sqrt(-2 * log(uniform()))
    return radius * cos(2 * Double.pi * uniform())
}

do {
    print("=================== Random intercept on a real dataset (penguins) ===================")
    try Lmm(try Data.penguins()!, "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)").printSummary()

    print("\n=================== Random intercept + slope on a simulated multi-school dataset ===================")
    // 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
    // and the return on an extra hour of study vary by school -- the textbook case for a
    // random-intercept-and-slope model.
    let nSchools = 30
    let schoolIntercept = (0..<nSchools).map { _ in gauss() * 6 }
    let schoolSlope = (0..<nSchools).map { _ in gauss() * 1.2 }
    let trueIntercept = 60.0
    let trueSlope = 3.0

    var school: [Double] = [], studyHours: [Double] = [], score: [Double] = []
    for s in 0..<nSchools {
        let nStudents = 15 + rng.next(0, 21)
        for _ in 0..<nStudents {
            let hours = rng.nextDouble() * 10
            school.append(Double(s))
            studyHours.append(hours)
            score.append(trueIntercept + schoolIntercept[s] + (trueSlope + schoolSlope[s]) * hours + gauss() * 4)
        }
    }
    let df = try DataFrame()
    try df.addNumericColumn("school", school)
    try df.addNumericColumn("study_hours", studyHours)
    try df.addNumericColumn("score", score)

    let model = try Lmm(df, "score ~ study_hours + (1 + study_hours | school)")
    try model.printSummary()

    print("\nTrue generating values: intercept=\(fmt(trueIntercept)), slope=\(fmt(trueSlope)), random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0")

    print("\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---")
    let labels = try model.groupLabels()
    for idx in 0..<3 {
        let re = try model.randomEffectsForGroup(idx)
        print("school \(labels[idx]): intercept shift=\(fmt(re[0])), slope shift=\(fmt(re[1]))")
    }

    print("\n--- Prediction: population-level vs. school-adjusted ---")
    let population = try DataFrame()
    let school0 = try DataFrame()
    try population.addNumericColumn("study_hours", [5.0])
    try school0.addNumericColumn("study_hours", [5.0])
    try school0.addNumericColumn("school", [0.0])
    print("5 study hours, unseen school:      \(fmt(try model.predict(population)[0])) (fixed effects only)")
    print("5 study hours, school 0 (known):    \(fmt(try model.predict(school0)[0])) (fixed effects + school 0's BLUP)")
} catch {
    print(error)
    exit(1)
}
