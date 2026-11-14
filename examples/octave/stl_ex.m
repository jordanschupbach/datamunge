% CMD: just TARGET=stl_ex run-octave
% C++ std::vector and std::pair values in Octave: vectors are row vectors, pairs are cells.

n = 100;
v = (0:n - 1) * 1.5;
printf ('%g\n', v);
v2 = floor ((0:n - 1) * 1.5);
printf ('%d\n', v2);

p = [3 4];
printf ('p: (%d, %d)\n', p(1), p(2));

p2 = [10 20];
printf ('p2: (%d, %d)\n', p2(1), p2(2));

% A C++ std::pair<double,double> comes back as a 1x2 cell.
p3 = datamunge.make_dpair (1.25, 2.75);
printf ('make_dpair(1.25, 2.75): (%g, %g), sum_dpair = %g\n', p3{1}, p3{2}, datamunge.sum_dpair (p3));
