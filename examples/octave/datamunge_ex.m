% CMD: just TARGET=datamunge_ex run-octave
% Smoke test of the basic bindings: vectors, pairs and a native Callback object.

datamunge.hello ();

v = datamunge.make_dvector (1.0, 2.0, 3.0);
printf ('sum_dvector:\t%.1f\n', datamunge.sum_dvector (v));

p = datamunge.make_dpair (1.25, 2.75);
printf ('sum_dpair:\t%.1f\n', datamunge.sum_dpair (p));

cb = datamunge.Callback ();
printf ('call_with_callback(3.0):\t%.1f\n', datamunge.call_with_callback (3.0, cb));
v2 = datamunge.map_dvector_with_callback (datamunge.make_dvector (1.0, 2.0, 3.0), cb);
printf ('sum_dvector(map_dvector_with_callback(1,2,3)):\t%.1f\n', datamunge.sum_dvector (v2));
