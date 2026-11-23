// CMD: just TARGET=ode_ex run-rust
// ODEs: built-in named systems, harmonic-oscillator energy check and a Lorenz phase plot.
use datamunge::*;
use datamunge_rust_examples::net_g;

/// Approximates C's %g: six significant digits, no padding.
fn fmt_g(x: &f64) -> String {
    net_g(*x, 6)
}

fn last_state(sol: &ODESolution) -> Result<Vec<f64>> {
    sol.state_at(sol.size() - 1)
}

/// The time vector, and each requested state component (0-based), one value per step.
fn solution_series(sol: &ODESolution, components: &[usize]) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let states = (0..sol.size()).map(|i| sol.state_at(i)).collect::<Result<Vec<_>>>()?;
    let times = (0..sol.size()).map(|i| sol.time_at(i)).collect::<Result<Vec<_>>>()?;
    Ok((times, components.iter().map(|&c| states.iter().map(|s| s[c]).collect()).collect()))
}

fn rk4_solver(step: f64) -> Result<ODESolver> {
    let options = ODEOptions::new()?;
    options.set_method(StepMethod::RK4)?;
    options.set_step_size(step)?;
    ODESolver::new_with_options(&options)
}

// NOTE: like the Lua version this example uses the built-in named systems via solve_builtin.

fn main() -> datamunge::Result<()> {
    println!("=================== Every named built-in system (no director needed) ===================");
    {
        let solver = ODESolver::new()?;
        let systems: [(&str, &[f64], &[f64]); 5] = [
            ("exponential_decay", &[1.0], &[1.0]),
            ("logistic_growth", &[1.0, 1.0], &[0.5]),
            ("harmonic_oscillator", &[1.0], &[1.0, 0.0]),
            ("van_der_pol", &[1.0], &[2.0, 0.0]),
            ("lorenz", &[10.0, 28.0, 8.0 / 3.0], &[1.0, 1.0, 1.0]),
        ];
        for (name, parameters, y0) in systems {
            let sol = solver.solve_builtin(name, parameters, y0, 0.0, 1.0)?;
            let state: Vec<String> = last_state(&sol)?.iter().map(fmt_g).collect();
            println!("{name:<22} steps={:<6} final state=[{}]", sol.steps_taken(), state.join(", "));
        }
    }

    println!("\n=================== Harmonic oscillator (energy conservation) ===================");
    {
        let sol = rk4_solver(0.01)?.solve_builtin("harmonic_oscillator", &[1.0], &[1.0, 0.0], 0.0, 20.0)?;
        let last = last_state(&sol)?;
        let (x, v) = (last[0], last[1]);
        println!("x(20) = {x:.6} (cos(20) = {:.6})", 20.0f64.cos());
        println!("energy x^2+v^2 = {:.6} (should stay near 1.0)", x * x + v * v);
        let (ts, series) = solution_series(&sol, &[0, 1])?;
        let p = RPlot::plot_with_label(&ts, &series[0], "l", "x(t)")?;
        p.lines_with_label(&ts, &series[1], "v(t)")?;
        p.title("Harmonic Oscillator")?.x_label("t")?.y_label("state")?;
        p.save_svg("ode_harmonic_oscillator_rust.svg")?;
        println!("wrote ode_harmonic_oscillator_rust.svg");
    }

    println!("\n=================== Lorenz attractor (phase plane) ===================");
    {
        let sol = rk4_solver(0.005)?.solve_builtin("lorenz", &[10.0, 28.0, 8.0 / 3.0], &[1.0, 1.0, 1.0], 0.0, 25.0)?;
        println!("steps_taken = {}", sol.steps_taken());
        let (_, series) = solution_series(&sol, &[0, 2])?;
        let p = RPlot::plot_with_label(&series[0], &series[1], "l", "trajectory")?;
        p.title("Lorenz Attractor (x-z phase plane)")?.x_label("x")?.y_label("z")?;
        p.save_svg("ode_lorenz_phase_plane_rust.svg")?;
        println!("wrote ode_lorenz_phase_plane_rust.svg");
    }
    Ok(())
}
