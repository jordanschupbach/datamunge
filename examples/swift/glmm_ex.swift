// CMD: just TARGET=glmm_ex run-swift
// Generalized linear mixed models: a binomial GLMM on penguins and a Poisson GLMM on simulated store data.
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

var rng = DotNetRandom(4242)
func uniform() -> Double { rng.nextDouble() }
func gauss() -> Double {
    let u1 = 1 - uniform(), u2 = uniform()  // 1 - u keeps log() away from 0
    return sqrt(-2 * log(u1)) * cos(2 * Double.pi * u2)
}

do {
    print("=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================")
    // Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
    // confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
    // is not tied to location, so this is a well-behaved fit.
    do {
        let penguins = try Data.penguins()!
        var isMale: [Double] = [], bodyMass: [Double] = []
        var island: [String] = []
        for i in 0..<(try penguins.nrows()) {
            if try ["sex", "body_mass_g", "island"].contains(where: { try penguins.isNull($0, i) }) { continue }
            isMale.append(try penguins.stringAt("sex", i) == "male" ? 1.0 : 0.0)
            bodyMass.append(try penguins.numericAt("body_mass_g", i))
            island.append(try penguins.stringAt("island", i))
        }
        let sexDf = try DataFrame()
        try sexDf.addNumericColumn("is_male", isMale)
        try sexDf.addNumericColumn("body_mass_g", bodyMass)
        try sexDf.addStringColumn("island", island)
        try Glmm(sexDf, "is_male ~ body_mass_g + (1 | island)", "binomial").printSummary()
    }

    print("\n=================== Poisson mixed model on simulated multi-site count data ===================")
    // 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
    // the baseline traffic and the promo's effectiveness vary by store.
    let nStores = 25
    let storeEffect = (0..<nStores).map { _ in gauss() * 0.4 }
    let trueIntercept = 2.0
    let trueSlope = 0.3

    var store: [Double] = [], promo: [Double] = [], visits: [Double] = []
    for s in 0..<nStores {
        let nDays = 15 + rng.next(0, 11)
        for _ in 0..<nDays {
            let promoIntensity = uniform() * 3
            let lam = exp(trueIntercept + storeEffect[s] + trueSlope * promoIntensity)
            // Knuth's Poisson sampler.
            var (threshold, k, p) = (exp(-lam), 0, 1.0)
            repeat {
                k += 1
                p *= uniform()
            } while p > threshold
            store.append(Double(s))
            promo.append(promoIntensity)
            visits.append(Double(k - 1))
        }
    }
    let df = try DataFrame()
    try df.addNumericColumn("store", store)
    try df.addNumericColumn("promo", promo)
    try df.addNumericColumn("visits", visits)
    let storeModel = try Glmm(df, "visits ~ promo + (1 | store)", "poisson")

    try storeModel.printSummary()

    print("\nTrue generating values: intercept=\(fmt(trueIntercept)), slope=\(fmt(trueSlope)), random-intercept SD (log scale)=0.4")

    print("\n--- BLUPs for a few stores ---")
    let labels = try storeModel.groupLabels()
    for idx in 0..<3 {
        print("store \(labels[idx]): intercept shift=\(fmt(try storeModel.randomEffectsForGroup(idx)[0]))")
    }

    print("\n--- Prediction: population-level vs. store-adjusted ---")
    let population = try DataFrame()
    let store0 = try DataFrame()
    try population.addNumericColumn("promo", [1.5])
    try store0.addNumericColumn("promo", [1.5])
    try store0.addNumericColumn("store", [0.0])
    print("promo=1.5, unseen store:   \(fmt(try storeModel.predict(population)[0])) expected visits (fixed effects only)")
    print("promo=1.5, store 0 (known): \(fmt(try storeModel.predict(store0)[0])) expected visits (fixed effects + store 0's BLUP)")
} catch {
    print(error)
    exit(1)
}
