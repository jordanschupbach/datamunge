// CMD: just TARGET=optim_advanced_ex run-swift
// Advanced datamunge optimizers (ProximalGradient, FISTA, LevenbergMarquardt, Newton, TrustRegionNewton,
// AugmentedLagrangian, SQP, InteriorPoint, BayesianOptimization, RBFGaussianProcessSurrogate) from Swift.
//
// Every one of these optimizers takes an interface (ProximalFunction / HessianFunction /
// EqualityConstrainedFunction / InequalityConstrainedFunction / ResidualFunction, or
// ArbitraryFunction for BayesianOptimization) that the C++ library leaves abstract: callers
// implement the objective. From Swift, subclass the interface class and override its virtual
// methods; the optimizer then calls the Swift overrides. Coordinates are updated in place through
// an inout parameter (the C++ parameter is a non-const std::vector<double>&).
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

/// C#'s {x:G<digits>}: printf %g with C#'s "E" exponent.
func fmtG(_ x: Double, _ digits: Int) -> String {
    String(cfmt("%.\(digits)g", x).map { $0 == "e" ? "E" : $0 })
}

final class Quadratic: HessianFunction {
    override func evaluate(_ c: [Double]) throws -> Double {
        (c[0] - 1) * (c[0] - 1) + 10 * (c[1] + 2) * (c[1] + 2)
    }

    override func gradient(_ c: [Double]) throws -> [Double] {
        [2 * (c[0] - 1), 20 * (c[1] + 2)]
    }

    override func hessian(_ coordinates: [Double]) throws -> [[Double]] { [[2.0, 0.0], [0.0, 20.0]] }
}

final class Exponential: ResidualFunction {
    let ts: [Double]

    init(_ ts: [Double]) throws {
        self.ts = ts
        try super.init()
    }

    override func residuals(_ p: [Double]) throws -> [Double] {
        ts.map { t in p[0] * exp(p[1] * t) - 2 * exp(0.5 * t) }
    }

    override func jacobian(_ p: [Double]) throws -> [[Double]] {
        ts.map { t in [exp(p[1] * t), p[0] * t * exp(p[1] * t)] }
    }
}

final class Incomplete: HessianFunction {
    override func evaluate(_ coordinates: [Double]) throws -> Double { 0.0 }
}

do {
    print("=================== RBFGaussianProcessSurrogate (concrete BayesianSurrogate) ===================")
    // Fit a 1D surrogate to a simple bowl y = (x-1)^2 sampled at a few points,
    // then check the acquisition function favors an unexplored region.
    do {
        let xs = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
        let ys = xs.map { ($0 - 1) * ($0 - 1) }
        let surrogate = try RbfGaussianProcessSurrogate(1.0, 1e-6)
        try surrogate.fit(xs.map { [$0] }, ys)
        let incumbent = min(0.0, ys.min()!)  // best observed value so far
        print("incumbent (best observed y): \(fmt(incumbent))")
        for x in [0.25, 1.0, 2.5] {
            print("acquisition(x=\(cfmt("%.2f", x))) = \(cfmt("%.6f", try surrogate.acquisition([x], incumbent)))")
        }
    }

    print("\n=================== Options structs + optimizer construction (all 10 new classes) ===================")
    // Every new Options struct and optimizer class constructs cleanly, proving
    // the binding for the whole batch is wired correctly.
    let pgOptions = try ProximalGradientOptions()
    pgOptions.stepSize = 0.05
    let pg = try ProximalGradient(pgOptions)
    print("ProximalGradient: \(pg)")
    let fista = try Fista(try FistaOptions())
    print("FISTA: \(fista)")
    let lmOptions = try LevenbergMarquardtOptions()
    lmOptions.maxIterations = 200
    let lm = try LevenbergMarquardt(lmOptions)
    print("LevenbergMarquardt: \(lm)")
    let newton = try Newton(try NewtonOptions())
    print("Newton: \(newton)")
    let trn = try TrustRegionNewton(try TrustRegionNewtonOptions())
    print("TrustRegionNewton: \(trn)")
    let al = try AugmentedLagrangian(try AugmentedLagrangianOptions())
    print("AugmentedLagrangian: \(al)")
    let sqp = try Sqp(try SqpOptions())
    print("SQP: \(sqp)")
    let ip = try InteriorPoint(try InteriorPointOptions())
    print("InteriorPoint: \(ip)")
    let boOptions = try BayesianOptimizationOptions()
    boOptions.initialSamples = 8
    let bo = try BayesianOptimization(boOptions)
    print("BayesianOptimization: \(bo)")

    print("\n=================== Swift objectives (subclassing the C++ interfaces) ===================")
    // Newton's method on f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
    do {
        var start = [5.0, 5.0]
        let value = try newton.optimize(try Quadratic(), &start)
        print("Newton: f = \(fmtG(value, 6)) at (\(cfmt("%.6f", start[0])), \(cfmt("%.6f", start[1]))) (exact minimum 0 at (1, -2))")
    }

    // Levenberg-Marquardt fitting y = a * exp(b t) to exact data from a = 2, b = 0.5.
    do {
        var parameters = [1.0, 0.1]
        let sse = try lm.optimize(try Exponential([0.0, 0.5, 1.0, 1.5, 2.0]), &parameters)
        print("LevenbergMarquardt: residual = \(fmtG(sse, 3)), a = \(cfmt("%.6f", parameters[0])), b = \(cfmt("%.6f", parameters[1])) (exact: 2, 0.5)")
    }

    // An interface method a subclass leaves out throws when the optimizer calls it.
    do {
        var err: Error? = nil
        do {
            var origin = [0.0, 0.0]
            _ = try newton.optimize(try Incomplete(), &origin)
        } catch {
            err = error
        }
        let mentions = err.map { "\($0)".contains("gradient") } ?? false
        print("missing gradient(): ok=\(err == nil ? "True" : "False"), error mentions it: \(mentions ? "True" : "False")")
    }
} catch {
    print(error)
    exit(1)
}
