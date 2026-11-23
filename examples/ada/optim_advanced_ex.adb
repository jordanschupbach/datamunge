--  CMD: just TARGET=optim_advanced_ex run-ada
--  Advanced datamunge optimizers (ProximalGradient, FISTA, LevenbergMarquardt, Newton, TrustRegionNewton,
--  AugmentedLagrangian, SQP, InteriorPoint, BayesianOptimization, RBFGaussianProcessSurrogate) from Ada.
--
--  Every one of these optimizers takes an interface (ProximalFunction / HessianFunction /
--  EqualityConstrainedFunction / InequalityConstrainedFunction / ResidualFunction, or
--  ArbitraryFunction for BayesianOptimization) that the C++ library leaves abstract: callers
--  implement the objective. From Ada, extend the interface's tagged type (in a library-level
--  package, Optim_Advanced_Objectives) and override its primitive operations; the optimizer then
--  calls the Ada overrides. Coordinates are updated in place through an in out parameter (the C++
--  parameter is a non-const std::vector<double>&).
with Ada.Exceptions; use Ada.Exceptions;
with Ada.Strings.Fixed;
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;
with Optim_Advanced_Objectives; use Optim_Advanced_Objectives;

procedure Optim_Advanced_Ex is
   Pg_Options     : Proximal_Gradient_Options := New_Proximal_Gradient_Options;
   Fista_Options  : Datamunge.Fista_Options := New_Fista_Options;
   Lm_Options     : Levenberg_Marquardt_Options := New_Levenberg_Marquardt_Options;
   Newton_Options : Datamunge.Newton_Options := New_Newton_Options;
   Trn_Options    : Trust_Region_Newton_Options := New_Trust_Region_Newton_Options;
   Al_Options     : Augmented_Lagrangian_Options := New_Augmented_Lagrangian_Options;
   Sqp_Options    : Datamunge.Sqp_Options := New_Sqp_Options;
   Ip_Options     : Interior_Point_Options := New_Interior_Point_Options;
   Bo_Options     : Bayesian_Optimization_Options := New_Bayesian_Optimization_Options;
   Pg             : Proximal_Gradient;
   Fista_Opt      : Fista;
   Lm             : Levenberg_Marquardt;
   Newton_Opt     : Newton;
   Trn            : Trust_Region_Newton;
   Al             : Augmented_Lagrangian;
   Sqp_Opt        : Sqp;
   Ip             : Interior_Point;
   Bo             : Bayesian_Optimization;
begin
   Put_Line ("=================== RBFGaussianProcessSurrogate (concrete BayesianSurrogate) ===================");
   --  Fit a 1D surrogate to a simple bowl y = (x-1)^2 sampled at a few points,
   --  then check the acquisition function favors an unexplored region.
   declare
      Xs        : constant Long_Float_Vector := [0.0, 0.5, 1.0, 1.5, 2.0, 3.0];
      Ys        : Long_Float_Vector;
      Points    : Long_Float_Vector_Vector;
      Surrogate : Rbf_Gaussian_Process_Surrogate := New_Rbf_Gaussian_Process_Surrogate (1.0, 1.0E-6);
      Incumbent : Long_Float := 0.0;
   begin
      for X of Xs loop
         Ys.Append ((X - 1.0) * (X - 1.0));
         Points.Append (Long_Float_Vector'[X]);
      end loop;
      Surrogate.Fit (Points, Ys);
      for Y of Ys loop
         Incumbent := Long_Float'Min (Incumbent, Y);  -- best observed value so far
      end loop;
      Put_Line ("incumbent (best observed y): " & Fmt (Incumbent));
      for X of Long_Float_Vector'[0.25, 1.0, 2.5] loop
         Put_Line ("acquisition(x=" & F (X, 2) & ") = " & F (Surrogate.Acquisition_2 ([X], Incumbent), 6));
      end loop;
      Free (Surrogate);
   end;

   Put_Line (ASCII.LF
             & "=================== Options structs + optimizer construction (all 10 new classes) ===================");
   --  Every new Options struct and optimizer class constructs cleanly, proving
   --  the binding for the whole batch is wired correctly. (The optimizers copy their options.)
   Pg_Options.Set_Step_Size (0.05);
   Lm_Options.Set_Max_Iterations (200);
   Bo_Options.Set_Initial_Samples (8);
   Pg := New_Proximal_Gradient (Pg_Options);
   Fista_Opt := New_Fista (Fista_Options);
   Lm := New_Levenberg_Marquardt (Lm_Options);
   Newton_Opt := New_Newton (Newton_Options);
   Trn := New_Trust_Region_Newton (Trn_Options);
   Al := New_Augmented_Lagrangian (Al_Options);
   Sqp_Opt := New_Sqp (Sqp_Options);
   Ip := New_Interior_Point (Ip_Options);
   Bo := New_Bayesian_Optimization (Bo_Options);
   Put_Line ("ProximalGradient: " & Pg.Image);
   Put_Line ("FISTA: " & Fista_Opt.Image);
   Put_Line ("LevenbergMarquardt: " & Lm.Image);
   Put_Line ("Newton: " & Newton_Opt.Image);
   Put_Line ("TrustRegionNewton: " & Trn.Image);
   Put_Line ("AugmentedLagrangian: " & Al.Image);
   Put_Line ("SQP: " & Sqp_Opt.Image);
   Put_Line ("InteriorPoint: " & Ip.Image);
   Put_Line ("BayesianOptimization: " & Bo.Image);

   Put_Line (ASCII.LF & "=================== Ada objectives (subclassing the C++ interfaces) ===================");
   --  Newton's method on f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
   declare
      Start     : Long_Float_Vector := [5.0, 5.0];
      Objective : Hessian_Function := new Quadratic_Record;
      Value     : Long_Float;
   begin
      Objective.Construct_Hessian_Function;
      Value := Newton_Opt.Optimize (Objective, Start);
      Put_Line ("Newton: f = " & G (Value, 6) & " at (" & F (Start (1), 6) & ", " & F (Start (2), 6)
                & ") (exact minimum 0 at (1, -2))");
      Free (Objective);
   end;

   --  Levenberg-Marquardt fitting y = a * exp(b t) to exact data from a = 2, b = 0.5.
   declare
      Parameters : Long_Float_Vector := [1.0, 0.1];
      Objective  : Residual_Function := new Exponential_Record'(Residual_Function_Record with
                                                                     Ts => [0.0, 0.5, 1.0, 1.5, 2.0]);
      Sse        : Long_Float;
   begin
      Objective.Construct_Residual_Function;
      Sse := Lm.Optimize (Objective, Parameters);
      Put_Line ("LevenbergMarquardt: residual = " & G (Sse, 3) & ", a = " & F (Parameters (1), 6)
                & ", b = " & F (Parameters (2), 6) & " (exact: 2, 0.5)");
      Free (Objective);
   end;

   --  An interface method a subclass leaves out raises an exception when the optimizer calls it.
   declare
      Origin    : Long_Float_Vector := [0.0, 0.0];
      Objective : Hessian_Function := new Incomplete_Record;
      Value     : Long_Float;
      Raised    : Boolean := False;
      Mentions  : Boolean := False;
   begin
      Objective.Construct_Hessian_Function;
      begin
         Value := Newton_Opt.Optimize (Objective, Origin);
         Put_Line ("unexpected result" & Value'Image);
      exception
         when E : others =>
            Raised := True;
            Mentions := Ada.Strings.Fixed.Index (Exception_Message (E), "gradient") > 0;
      end;
      Put_Line ("missing gradient(): ok=" & (if Raised then "False" else "True")
                & ", error mentions it: " & (if Mentions then "True" else "False"));
      Free (Objective);
   end;

   Free (Bo);
   Free (Ip);
   Free (Sqp_Opt);
   Free (Al);
   Free (Trn);
   Free (Newton_Opt);
   Free (Lm);
   Free (Fista_Opt);
   Free (Pg);
   Free (Bo_Options);
   Free (Ip_Options);
   Free (Sqp_Options);
   Free (Al_Options);
   Free (Trn_Options);
   Free (Newton_Options);
   Free (Lm_Options);
   Free (Fista_Options);
   Free (Pg_Options);
end Optim_Advanced_Ex;
