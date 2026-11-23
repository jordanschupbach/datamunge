// CMD: just TARGET=timeseries_ex run-swift
// Time series: ARIMA, seasonal ARIMA and exponential smoothing (SES, Holt, Holt-Winters).
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


/// .NET's seeded System.Random (Knuth's subtractive generator), so the simulated series -- and
/// so every fitted number -- match the C# example exactly.
struct DotNetRandom {
    private var seedArray = [Int32](repeating: 0, count: 56)
    private var inext = 0
    private var inextp = 21

    init(_ seed: Int32) {
        let big = Int32.max
        let subtraction = seed == Int32.min ? Int32.max : abs(seed)
        var mj = 161_803_398 - subtraction
        seedArray[55] = mj
        var mk: Int32 = 1
        for i in 1..<55 {
            let ii = (21 * i) % 55
            seedArray[ii] = mk
            mk = mj - mk
            if mk < 0 { mk += big }
            mj = seedArray[ii]
        }
        for _ in 1..<5 {
            for i in 1..<56 {
                seedArray[i] = seedArray[i] &- seedArray[1 + (i + 30) % 55]
                if seedArray[i] < 0 { seedArray[i] += big }
            }
        }
    }

    /// NextDouble(): [0, 1).
    mutating func nextDouble() -> Double {
        inext = inext + 1 >= 56 ? 1 : inext + 1
        inextp = inextp + 1 >= 56 ? 1 : inextp + 1
        var value = seedArray[inext] - seedArray[inextp]
        if value == Int32.max { value -= 1 }
        if value < 0 { value += Int32.max }
        seedArray[inext] = value
        return Double(value) * (1.0 / Double(Int32.max))
    }
}

var rng = DotNetRandom(42)
func seed(_ n: Int32) { rng = DotNetRandom(n) }
func uniform() -> Double { 1 - rng.nextDouble() }  // (0, 1]: avoids log(0) in the Box-Muller transform.
func gauss() -> Double { sqrt(-2 * log(uniform())) * cos(2 * Double.pi * uniform()) }
func joined(_ v: [Double]) -> String { v.map(fmt).joined(separator: ", ") }

do {
    print("=================== ARIMA(1,1,1) on a simulated random walk with drift ===================")
    seed(42)
    var y: [Double] = []
    do {
        var level = 100.0, prevShock = 0.0
        for _ in 0..<150 {
            let shock = gauss()
            level += 0.3 + 0.4 * prevShock + shock  // integrated ARMA(0,1) increments with drift
            y.append(level)
            prevShock = shock
        }
    }

    let options = try ArimaOptions()
    options.p = 1
    options.d = 1
    options.q = 1
    options.dePopulationSize = 80
    options.deMaxGenerations = 400
    let model = try Arima(y, options)

    print("AR coefficient: \(fmt(try model.arCoefficients()[0]))")
    print("MA coefficient: \(fmt(try model.maCoefficients()[0]))")
    print("sigma^2: \(fmt(try model.sigma2())), AIC: \(fmt(try model.aic())), BIC: \(fmt(try model.bic()))")

    let (forecastValues, stdErrors) = try model.forecastWithIntervals(6)
    print("6-step forecast: \(joined(forecastValues))")
    print("forecast std. errors: \(joined(stdErrors))")

    print("\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================")
    seed(7)
    var s: [Double] = []
    do {
        var prev = 0.0
        for i in 0..<120 {
            prev = 0.5 * prev + gauss()
            s.append(20 + 0.2 * Double(i) + 5 * sin(2 * Double.pi * Double(i) / 12) + prev)
        }
    }

    let options2 = try ArimaOptions()
    options2.p = 1
    options2.seasonalP = 1
    options2.seasonalD = 1
    options2.seasonalPeriod = 12
    options2.dePopulationSize = 100
    options2.deMaxGenerations = 500
    let model2 = try Arima(s, options2)
    print("AR coefficient: \(fmt(try model2.arCoefficients()[0])), seasonal AR coefficient: \(fmt(try model2.seasonalArCoefficients()[0]))")
    print("12-step forecast: \(joined(try model2.forecast(12)))")

    print("\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================")
    let seasonalShape = [0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6]
    seed(11)
    let y2 = (0..<48).map { (i: Int) -> Double in
        let trend: Double = 100 + 2 * Double(i)
        return trend * seasonalShape[i % 12] + gauss() * 3
    }

    let esOptions = try ExponentialSmoothingOptions()
    esOptions.trend = .additive
    esOptions.seasonal = .multiplicative
    esOptions.seasonalPeriod = 12
    esOptions.dePopulationSize = 60
    esOptions.deMaxGenerations = 300
    let esModel = try ExponentialSmoothing(y2, esOptions)
    print("alpha=\(fmt(try esModel.alpha())) beta=\(fmt(try esModel.beta())) gamma=\(fmt(try esModel.gamma()))")
    print("sigma^2: \(fmt(try esModel.sigma2())), AIC: \(fmt(try esModel.aic()))")
    print("12-month forecast: \(joined(try esModel.forecast(12)))")

    print("\n=================== Simple exponential smoothing vs. Holt's linear trend ===================")
    let flat = [50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7]

    let ses = try ExponentialSmoothing(flat, try ExponentialSmoothingOptions())
    print("SES alpha= \(fmt(try ses.alpha()))")
    print("SES 5-step forecast: \(joined(try ses.forecast(5)))")

    let holtOptions = try ExponentialSmoothingOptions()
    holtOptions.trend = .additive
    let holt = try ExponentialSmoothing(flat, holtOptions)
    print("Holt alpha=\(fmt(try holt.alpha())) beta=\(fmt(try holt.beta()))")
    print("Holt 5-step forecast: \(joined(try holt.forecast(5)))")
} catch {
    print(error)
    exit(1)
}
