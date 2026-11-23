// CMD: just TARGET=autodiff_ex run-swift
// Automatic differentiation: forward mode (Dual), reverse mode (Tape/Var), Hessians (HyperDual).
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

// Note: the C++ autodiff_ex.cpp additionally demonstrates the generic template drivers
// (derivative/gradient_forward/gradient_reverse/jacobian_forward/hessian), which can't be
// bound. This port demonstrates the same ideas with the bound named-method API.
// Var.index() is the Var's position in the adjoint vector Tape.backward returns.

func rosenbrockDual(_ x0: Dual, _ x1: Dual) throws -> Dual {
    let a = try Dual(1.0, 0.0).subtract(x0)
    let b = try x1.subtract(x0.multiply(x0))
    return try a.multiply(a).add(b.multiply(b).multiplyScalar(100.0))
}

func rosenbrockValue(_ x0: Double, _ x1: Double) -> Double { pow(1.0 - x0, 2) + 100.0 * pow(x1 - x0 * x0, 2) }

// Build the Rosenbrock graph on tape; returns the output Var.
func rosenbrockVar(_ tape: Tape, _ x0: Var, _ x1: Var) throws -> Var {
    let a = try Var(tape, 1.0).subtract(x0)
    let b = try x1.subtract(x0.multiply(x0))
    return try a.multiply(a).add(b.multiply(b).multiplyScalar(100.0))
}

func rosenbrockHyperDual(_ x0: HyperDual, _ x1: HyperDual) throws -> HyperDual {
    let a = try HyperDual(1.0, 0.0, 0.0, 0.0).subtract(x0)
    let b = try x1.subtract(x0.multiply(x0))
    return try a.multiply(a).add(b.multiply(b).multiplyScalar(100.0))
}

func adjoint(_ adjoints: [Double], _ v: Var) throws -> Double { adjoints[try v.index()] }

do {
    print("=================== Forward mode: scalar derivative ===================")
    do {
        let x0 = try Dual(2.0, 1.0)  // seed derivative = 1 to read df/dx directly
        let f = try x0.multiply(x0).multiply(x0).subtract(x0.multiplyScalar(2.0))  // x^3 - 2x
        print("f(x) = x^3 - 2x, f'(2) = \(fmt(try f.derivative())) (exact: 10)")
    }

    print("\n=================== Reverse mode: build a graph by hand ===================")
    do {
        let tape = try Tape()
        let a = try Var(tape, 2.0)
        let b = try Var(tape, 3.0)
        let y = try a.multiply(b).add(a.sin())
        print("y = a*b + sin(a) at a=2, b=3 -> y = \(fmt(try y.value()))")
        let adjoints = try tape.backward(y)
        print("dy/da = \(fmt(try adjoint(adjoints, a))) (exact: b + cos(a))")
        print("dy/db = \(fmt(try adjoint(adjoints, b))) (exact: a)")
    }

    print("\n=================== Forward vs reverse mode agree on the Rosenbrock function ===================")
    do {
        let px = 0.0, py = 0.0
        // Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
        let gx = try rosenbrockDual(Dual(px, 1.0), Dual(py, 0.0)).derivative()
        let gy = try rosenbrockDual(Dual(px, 0.0), Dual(py, 1.0)).derivative()
        // Reverse mode: one Tape/Var pass computes every partial at once.
        let tape = try Tape()
        let vx = try Var(tape, px)
        let vy = try Var(tape, py)
        let grad = try tape.backward(rosenbrockVar(tape, vx, vy))
        print("f(0,0) = \(fmt(rosenbrockValue(px, py)))")
        print("gradient (forward mode): [\(fmt(gx)), \(fmt(gy))]")
        print("gradient (reverse mode): [\(fmt(try adjoint(grad, vx))), \(fmt(try adjoint(grad, vy)))]")
    }

    print("\n=================== Jacobian of a vector-valued function ===================")
    // f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
    do {
        func vectorFn(_ x: Dual, _ y: Dual) throws -> [Dual] { [try x.multiply(x), try x.multiply(y), try y.multiply(y).multiply(y)] }
        var jac = [[Double]](repeating: [0.0, 0.0], count: 3)
        for (col, sx, sy) in [(0, 1.0, 0.0), (1, 0.0, 1.0)] {
            let outputs = try vectorFn(Dual(2.0, sx), Dual(3.0, sy))
            for row in 0..<outputs.count {
                jac[row][col] = try outputs[row].derivative()
            }
        }
        print("f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:")
        for row in 0..<3 {
            print("  [\(fmt(jac[row][0])), \(fmt(jac[row][1]))]")
        }
    }

    print("\n=================== Hessian via second-order forward mode (HyperDual) ===================")
    // One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
    do {
        let mx = 1.0, my = 1.0  // the Rosenbrock function's minimum
        let seeds: [(x: Double, y: Double)] = [(1.0, 0.0), (0.0, 1.0)]
        var h = [[Double]](repeating: [0.0, 0.0], count: 2)
        for i in 0..<2 {
            for j in 0..<2 {
                let (e1, e2) = (seeds[i], seeds[j])
                h[i][j] = try rosenbrockHyperDual(HyperDual(mx, e1.x, e2.x, 0.0), HyperDual(my, e1.y, e2.y, 0.0)).eps1eps2()
            }
        }
        print("Hessian of the Rosenbrock function at its minimum (1,1):")
        for i in 0..<2 {
            print("  [\(fmt(h[i][0])), \(fmt(h[i][1]))]")
        }
    }

    print("\n=================== Gradient descent driven by reverse-mode gradients ===================")
    do {
        var point = [-1.2, 1.0]  // the classic Rosenbrock start
        let learningRate = 0.001
        let nSteps = 2000
        for step in 0..<nSteps {
            let tape = try Tape()
            let vx = try Var(tape, point[0])
            let vy = try Var(tape, point[1])
            let vf = try rosenbrockVar(tape, vx, vy)
            let grad = try tape.backward(vf)
            let loss = try vf.value()
            point[0] -= learningRate * (try adjoint(grad, vx))
            point[1] -= learningRate * (try adjoint(grad, vy))
            if step == 0 || step == nSteps - 1 {
                print("step \(step): loss = \(fmt(loss)), x = [\(fmt(point[0])), \(fmt(point[1]))]")
            }
        }
        print("(true minimum is at [1, 1] with loss 0)")
    }
} catch {
    print(error)
    exit(1)
}
