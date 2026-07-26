use strict;
use warnings;

use Datamunge;

# NOTE: the Python ode example also demonstrates a live, user-supplied RHS by subclassing
# datamunge.RHS (a SWIG director). RHS is director-enabled ONLY in the Python binding (a
# virtual method returning std::vector<double> cannot be wrapped as a working director in the
# other SWIG backends -- see rhs.hpp), so this Perl port uses the built-in named systems via
# solve_builtin(), exactly as the Lua template does.

# state_at() returns std::vector<double> by const reference, which the Perl binding wraps as a
# blessed DVector object (indexed with ->get($i)/->size()) -- unlike a by-value vector<double>
# return, which comes back as a plain Perl arrayref.
sub state_to_array {
  my ($st) = @_;
  my @out;
  for (my $i = 0; $i < $st->size(); $i++) { push @out, $st->get($i); }
  return \@out;
}

sub last_state {
  my ($sol) = @_;
  return $sol->state_at($sol->size() - 1);
}

print "=================== Every named built-in system (no director needed) ===================\n";
my $solver = Datamunge::ODESolver->new();
my @systems = (
  ["exponential_decay", [1.0], [1.0]],
  ["logistic_growth", [1.0, 1.0], [0.5]],
  ["harmonic_oscillator", [1.0], [1.0, 0.0]],
  ["van_der_pol", [1.0], [2.0, 0.0]],
  ["lorenz", [10.0, 28.0, 8.0 / 3.0], [1.0, 1.0, 1.0]],
);
for my $s (@systems) {
  my ($name, $params, $y0) = @$s;
  my $sol = $solver->solve_builtin($name, $params, $y0, 0.0, 1.0);
  my $final = state_to_array(last_state($sol));
  my $parts = join(", ", map { sprintf("%g", $_) } @$final);
  printf("%-22s steps=%-6d final state=[%s]\n", $name, $sol->swig_steps_taken_get(), $parts);
}

print "\n=================== Harmonic oscillator (energy conservation) ===================\n";
my $options = Datamunge::ODEOptions->new();
$options->swig_method_set($Datamunge::StepMethod_RK4);
$options->swig_step_size_set(0.01);
$solver = Datamunge::ODESolver->new($options);
my $sol = $solver->solve_builtin("harmonic_oscillator", [1.0], [1.0, 0.0], 0.0, 20.0);
my $final = last_state($sol);
my ($x, $v) = ($final->get(0), $final->get(1));
printf("x(20) = %.6f (cos(20) = %.6f)\n", $x, cos(20));
printf("energy x^2+v^2 = %.6f (should stay near 1.0)\n", $x * $x + $v * $v);

my (@t_series, @x_series, @v_series);
for (my $i = 0; $i < $sol->size(); $i++) {
  push @t_series, $sol->time_at($i);
  my $st = $sol->state_at($i);
  push @x_series, $st->get(0);
  push @v_series, $st->get(1);
}
my $plot = Datamunge::RPlot::plot(\@t_series, \@x_series, "l", "x(t)");
$plot->lines(\@t_series, \@v_series, "v(t)");
$plot->title("Harmonic Oscillator");
$plot->x_label("t");
$plot->y_label("state");
$plot->save_svg("ode_harmonic_oscillator_perl.svg");
print "wrote ode_harmonic_oscillator_perl.svg\n";

print "\n=================== Lorenz attractor (phase plane) ===================\n";
$options = Datamunge::ODEOptions->new();
$options->swig_method_set($Datamunge::StepMethod_RK4);
$options->swig_step_size_set(0.005);
$solver = Datamunge::ODESolver->new($options);
$sol = $solver->solve_builtin("lorenz", [10.0, 28.0, 8.0 / 3.0], [1.0, 1.0, 1.0], 0.0, 25.0);
printf("steps_taken = %d\n", $sol->swig_steps_taken_get());

my (@xs, @zs);
for (my $i = 0; $i < $sol->size(); $i++) {
  my $st = $sol->state_at($i);
  push @xs, $st->get(0);
  push @zs, $st->get(2);
}
$plot = Datamunge::RPlot::plot(\@xs, \@zs, "l", "trajectory");
$plot->title("Lorenz Attractor (x-z phase plane)");
$plot->x_label("x");
$plot->y_label("z");
$plot->save_svg("ode_lorenz_phase_plane_perl.svg");
print "wrote ode_lorenz_phase_plane_perl.svg\n";
