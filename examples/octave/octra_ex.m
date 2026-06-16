1;

octra;

hello ();

v = DVector ();
v.append (1.0);
v.append (2.0);
v.append (3.5);
printf ("vector size: %d\n", v.size ());
printf ("vector[0]=%f vector[1]=%f\n", v.__paren__ (0), v.__paren__ (1));

p = DPair (10.0, 25.5);
printf ("pair sum: %f\n", p.first + p.second);
