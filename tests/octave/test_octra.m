1; % Octave test script

%!test
%! octra; % load SWIG module and register symbols
%! assert (exist ("octra", "file") == 3); % 3 = dynamically loaded function (.oct)
%! hello ();

%!test
%! octra;
%! v = DVector ();
%! v.append (3.0);
%! v.append (4.5);
%! assert (v.size () == 2);
%! assert (v.__paren__ (0) == 3.0);
%! assert (v.__paren__ (1) == 4.5);

%!test
%! octra;
%! p = DPair (1.25, 2.5);
%! assert (p.first == 1.25);
%! assert (p.second == 2.5);
