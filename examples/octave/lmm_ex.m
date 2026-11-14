% CMD: just TARGET=lmm_ex run-octave
% Linear mixed models: random intercept (penguins) and random intercept + slope (simulated schools).
1;

% (0, 1]: avoids log(0) in the Box-Muller transform.
function u = uniform ()
  u = 1 - rand ();
end

function g = gauss ()
  g = sqrt (-2 * log (uniform ())) * cos (2 * pi * uniform ());
end

% Shortest round-trip decimal text for x (like C#/Julia default double printing).
function s = num (x)
  if x == 0
    s = '0.0';
    return;
  end
  for d = 1:17
    s = sprintf ('%.*e', d - 1, x);
    if str2double (s) == x
      break;
    end
  end
  epos = strfind (s, 'e');
  e = str2double (s(epos+1:end));
  if e >= -5 && e < 16
    s = sprintf ('%.*f', max (1, d - 1 - e), x);
  else
    s = sprintf ('%se%d', s(1:epos-1), e);
  end
end

rand ('state', 2024);

printf ('=================== Random intercept on a real dataset (penguins) ===================\n');
datamunge.LMM (datamunge.Data.penguins (), 'body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)').print_summary ();

printf ('\n=================== Random intercept + slope on a simulated multi-school dataset ===================\n');
% 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
% and the return on an extra hour of study vary by school -- the textbook case for a
% random-intercept-and-slope model.
n_schools = 30;
school_intercept = arrayfun (@(~) gauss () * 6, 1:n_schools);
school_slope = arrayfun (@(~) gauss () * 1.2, 1:n_schools);
true_intercept = 60.0;
true_slope = 3.0;

school = [];
study_hours = [];
score = [];
for s = 0:n_schools-1
  n_students = 15 + floor (rand () * 21);
  for student = 1:n_students
    hours = rand () * 10;
    school(end+1) = s;
    study_hours(end+1) = hours;
    score(end+1) = true_intercept + school_intercept(s+1) + (true_slope + school_slope(s+1)) * hours + gauss () * 4;
  end
end
df = datamunge.DataFrame ();
df.add_numeric_column ('school', school);
df.add_numeric_column ('study_hours', study_hours);
df.add_numeric_column ('score', score);

model = datamunge.LMM (df, 'score ~ study_hours + (1 + study_hours | school)');
model.print_summary ();

printf ('\nTrue generating values: intercept=%.1f, slope=%.1f, random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0\n', ...
        true_intercept, true_slope);

printf ('\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---\n');
labels = model.group_labels ();
for idx = 0:2
  re = model.random_effects_for_group (idx);
  printf ('school %s: intercept shift=%s, slope shift=%s\n', labels{idx+1}, num (re(1)), num (re(2)));
end

printf ('\n--- Prediction: population-level vs. school-adjusted ---\n');
population = datamunge.DataFrame ();
school0 = datamunge.DataFrame ();
population.add_numeric_column ('study_hours', 5.0);
school0.add_numeric_column ('study_hours', 5.0);
school0.add_numeric_column ('school', 0.0);
printf ('5 study hours, unseen school:      %s (fixed effects only)\n', num (model.predict (population)(1)));
printf ('5 study hours, school 0 (known):    %s (fixed effects + school 0''s BLUP)\n', num (model.predict (school0)(1)));
