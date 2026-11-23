# CMD: just TARGET=ode_ex run-perl
# ODEs: built-in named systems, harmonic-oscillator energy check and a Lorenz phase plot.
use strict;
use warnings;

use Datamunge qw(:all);

$| = 1;    # flush each print so it interleaves with the C++ library's output

# C's %g: six significant digits, no padding.
sub fmt_g { return sprintf('%.6g', $_[0]) }

sub last_state { my ($sol) = @_; return $sol->state_at($sol->size() - 1) }

# The time vector followed by the requested state components (0-based), one value per step.
sub solution_series {
    my ($sol, @components) = @_;
    my @steps = 0 .. $sol->size() - 1;
    my @states = map { $sol->state_at($_) } @steps;
    return ([map { $sol->time_at($_) } @steps], map { my $c = $_; [map { $_->[$c] } @states] } @components);
}

sub rk4_solver {
    my ($step) = @_;
    my $options = Datamunge::OdeOptions->new();
    $options->method(Datamunge::StepMethod::RK4);
    $options->step_size($step);
    return Datamunge::OdeSolver->new($options);
}

# NOTE: like the Lua version this example uses the built-in named systems via solve_builtin.

print "=================== Every named built-in system (no director needed) ===================\n";
{
    my $solver = Datamunge::OdeSolver->new();
    my @systems = (
        ["exponential_decay", [1.0], [1.0]],
        ["logistic_growth", [1.0, 1.0], [0.5]],
        ["harmonic_oscillator", [1.0], [1.0, 0.0]],
        ["van_der_pol", [1.0], [2.0, 0.0]],
        ["lorenz", [10.0, 28.0, 8.0 / 3], [1.0, 1.0, 1.0]],
    );
    for my $system (@systems) {
        my ($name, $parameters, $y0) = @$system;
        my $sol = $solver->solve_builtin($name, $parameters, $y0, 0.0, 1.0);
        printf "%-22s steps=%-6s final state=[%s]\n", $name, $sol->steps_taken, join(", ", map { fmt_g($_) } @{ last_state($sol) });
    }
}

print "\n=================== Harmonic oscillator (energy conservation) ===================\n";
{
    my $sol = rk4_solver(0.01)->solve_builtin("harmonic_oscillator", [1.0], [1.0, 0.0], 0.0, 20.0);
    my ($x, $v) = @{ last_state($sol) };
    printf "x(20) = %.6f (cos(20) = %.6f)\n", $x, cos(20.0);
    printf "energy x^2+v^2 = %.6f (should stay near 1.0)\n", $x * $x + $v * $v;
    my ($ts, $xs, $vs) = solution_series($sol, 0, 1);
    my $p = Datamunge::RPlot->plot($ts, $xs, "l", "x(t)");
    $p->lines($ts, $vs, "v(t)");
    $p->title("Harmonic Oscillator")->x_label("t")->y_label("state");
    $p->save_svg("ode_harmonic_oscillator_perl.svg");
    print "wrote ode_harmonic_oscillator_perl.svg\n";
}

print "\n=================== Lorenz attractor (phase plane) ===================\n";
{
    my $sol = rk4_solver(0.005)->solve_builtin("lorenz", [10.0, 28.0, 8.0 / 3], [1.0, 1.0, 1.0], 0.0, 25.0);
    print "steps_taken = ", $sol->steps_taken, "\n";
    my (undef, $xs, $zs) = solution_series($sol, 0, 2);
    my $p = Datamunge::RPlot->plot($xs, $zs, "l", "trajectory");
    $p->title("Lorenz Attractor (x-z phase plane)")->x_label("x")->y_label("z");
    $p->save_svg("ode_lorenz_phase_plane_perl.svg");
    print "wrote ode_lorenz_phase_plane_perl.svg\n";
}
