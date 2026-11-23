// CMD: just TARGET=ode_ex run-swift
// ODEs: built-in named systems, harmonic-oscillator energy check and a Lorenz phase plot.
import Datamunge
#if canImport(Glibc)
import Glibc
#else
import Darwin
#endif

/// A Double through a C printf format, e.g. cfmt("%.6f", x) for C#'s {x:F6}.
func cfmt(_ format: String, _ x: Double) -> String {
    var buffer = [CChar](repeating: 0, count: 64)
    _ = withVaList([x]) { vsnprintf(&buffer, buffer.count, format, $0) }
    return String(cString: buffer)
}

/// C#'s {x:G6}: six significant digits, no padding (printf %g with C#'s "E+XX" exponent).
func fmtG(_ x: Double) -> String {
    String(cfmt("%.6g", x).map { $0 == "e" ? "E" : $0 })
}

/// C#'s {s,-width}: left-aligned in a field of `width` characters.
func padRight(_ s: String, _ width: Int) -> String {
    s.count >= width ? s : s + String(repeating: " ", count: width - s.count)
}

func lastState(_ sol: OdeSolution) throws -> [Double] { try sol.stateAt(sol.size() - 1) }

// The time vector followed by the requested state components (0-based), one value per step.
func solutionSeries(_ sol: OdeSolution, _ components: Int...) throws -> (times: [Double], components: [[Double]]) {
    let steps = 0..<sol.size()
    let states = try steps.map { try sol.stateAt($0) }
    return (try steps.map { try sol.timeAt($0) }, components.map { c in states.map { $0[c] } })
}

func rk4Solver(_ step: Double) throws -> OdeSolver {
    let options = try OdeOptions()
    options.method = .rk4
    options.stepSize = step
    return try OdeSolver(options)
}

// NOTE: like the Lua version this example uses the built-in named systems via solveBuiltin.

do {
    print("=================== Every named built-in system (no director needed) ===================")
    do {
        let solver = try OdeSolver()
        let systems: [(name: String, params: [Double], y0: [Double])] = [
            ("exponential_decay", [1.0], [1.0]),
            ("logistic_growth", [1.0, 1.0], [0.5]),
            ("harmonic_oscillator", [1.0], [1.0, 0.0]),
            ("van_der_pol", [1.0], [2.0, 0.0]),
            ("lorenz", [10.0, 28.0, 8.0 / 3], [1.0, 1.0, 1.0]),
        ]
        for (name, parameters, y0) in systems {
            let sol = try solver.solveBuiltin(name, parameters, y0, 0.0, 1.0)
            print("\(padRight(name, 22)) steps=\(padRight(String(sol.stepsTaken), 6)) final state=[\(try lastState(sol).map(fmtG).joined(separator: ", "))]")
        }
    }

    print("\n=================== Harmonic oscillator (energy conservation) ===================")
    do {
        let sol = try rk4Solver(0.01).solveBuiltin("harmonic_oscillator", [1.0], [1.0, 0.0], 0.0, 20.0)
        let last = try lastState(sol)
        let (x, v) = (last[0], last[1])
        print("x(20) = \(cfmt("%.6f", x)) (cos(20) = \(cfmt("%.6f", cos(20.0))))")
        print("energy x^2+v^2 = \(cfmt("%.6f", x * x + v * v)) (should stay near 1.0)")
        let (ts, series) = try solutionSeries(sol, 0, 1)
        let p = try RPlot.plot(ts, series[0], "l", "x(t)")
        _ = try p.lines(ts, series[1], "v(t)")
        _ = try p.title("Harmonic Oscillator").xLabel("t").yLabel("state")
        try p.saveSvg("ode_harmonic_oscillator_swift.svg")
        print("wrote ode_harmonic_oscillator_swift.svg")
    }

    print("\n=================== Lorenz attractor (phase plane) ===================")
    do {
        let sol = try rk4Solver(0.005).solveBuiltin("lorenz", [10.0, 28.0, 8.0 / 3], [1.0, 1.0, 1.0], 0.0, 25.0)
        print("steps_taken = \(sol.stepsTaken)")
        let (_, series) = try solutionSeries(sol, 0, 2)
        let p = try RPlot.plot(series[0], series[1], "l", "trajectory")
        _ = try p.title("Lorenz Attractor (x-z phase plane)").xLabel("x").yLabel("z")
        try p.saveSvg("ode_lorenz_phase_plane_swift.svg")
        print("wrote ode_lorenz_phase_plane_swift.svg")
    }
} catch {
    print(error)
    exit(1)
}
