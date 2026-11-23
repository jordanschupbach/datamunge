--  CMD: just TARGET=ode_ex run-ada
--  ODEs: built-in named systems, harmonic-oscillator energy check and a Lorenz phase plot.
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;

procedure Ode_Ex is
   function Last_State (Sol : Ode_Solution) return Long_Float_Vector is (Sol.State_At (Sol.Size - 1));

   --  The time vector, one value per step.
   function Solution_Times (Sol : Ode_Solution) return Long_Float_Vector is
      Result : Long_Float_Vector;
   begin
      for I in 0 .. Sol.Size - 1 loop
         Result.Append (Sol.Time_At (I));
      end loop;
      return Result;
   end Solution_Times;

   --  The requested state component (0-based), one value per step.
   function Solution_Component (Sol : Ode_Solution; Component : Natural) return Long_Float_Vector is
      Result : Long_Float_Vector;
   begin
      for I in 0 .. Sol.Size - 1 loop
         Result.Append (Sol.State_At (I) (Component + 1));
      end loop;
      return Result;
   end Solution_Component;

   function Rk4_Solver (Step : Long_Float) return Ode_Solver is
      Options : Ode_Options := New_Ode_Options;
      Result  : Ode_Solver;
   begin
      Options.Set_Method (Rk4);
      Options.Set_Step_Size (Step);
      Result := New_Ode_Solver (Options);
      Free (Options);
      return Result;
   end Rk4_Solver;

   --  Prints one built-in system's step count and final state.
   procedure Run_Builtin (Solver : Ode_Solver; Name : String; Params, Y0 : Long_Float_Vector) is
      Sol  : Ode_Solution := Solver.Solve_Builtin (Name, Params, Y0, 0.0, 1.0);
      Text : Unbounded_String;
      State : constant Long_Float_Vector := Last_State (Sol);
   begin
      for I in State.First_Index .. State.Last_Index loop
         if I > State.First_Index then
            Append (Text, ", ");
         end if;
         Append (Text, G (State (I), 6));
      end loop;
      Put_Line (Pad_Right (Name, 22) & " steps=" & Pad_Right (Img (Sol.Steps_Taken), 6)
                & " final state=[" & To_String (Text) & "]");
      Free (Sol);
   end Run_Builtin;

   Solver : Ode_Solver;
   Sol    : Ode_Solution;
   Last, Ts, Xs, Vs, Zs : Long_Float_Vector;
   X, V   : Long_Float;
   P, Chained : R_Plot;
begin
   --  NOTE: like the Lua version this example uses the built-in named systems via Solve_Builtin.

   Put_Line ("=================== Every named built-in system (no director needed) ===================");
   Solver := New_Ode_Solver;
   Run_Builtin (Solver, "exponential_decay", [1.0], [1.0]);
   Run_Builtin (Solver, "logistic_growth", [1.0, 1.0], [0.5]);
   Run_Builtin (Solver, "harmonic_oscillator", [1.0], [1.0, 0.0]);
   Run_Builtin (Solver, "van_der_pol", [1.0], [2.0, 0.0]);
   Run_Builtin (Solver, "lorenz", [10.0, 28.0, 8.0 / 3.0], [1.0, 1.0, 1.0]);
   Free (Solver);

   Put_Line (ASCII.LF & "=================== Harmonic oscillator (energy conservation) ===================");
   Solver := Rk4_Solver (0.01);
   Sol := Solver.Solve_Builtin ("harmonic_oscillator", [1.0], [1.0, 0.0], 0.0, 20.0);
   Last := Last_State (Sol);
   X := Last (1);
   V := Last (2);
   Put_Line ("x(20) = " & F (X, 6) & " (cos(20) = " & F (C_Cos (20.0), 6) & ")");
   Put_Line ("energy x^2+v^2 = " & F (X * X + V * V, 6) & " (should stay near 1.0)");
   Ts := Solution_Times (Sol);
   Xs := Solution_Component (Sol, 0);
   Vs := Solution_Component (Sol, 1);
   P := R_Plot_Plot (Ts, Xs, "l", "x(t)");
   --  The layer and label functions return the plot itself (RPlot&), so Chained is P; RPlot's
   --  own title/xlabel/ylabel are Title_2/X_Label_2/Y_Label_2 (Title etc. are Plot's, returning Plot).
   Chained := P.Lines (Ts, Vs, "v(t)");
   Chained := Chained.Title_2 ("Harmonic Oscillator").X_Label_2 ("t").Y_Label_2 ("state");
   Chained.Save_Svg ("ode_harmonic_oscillator_ada.svg");
   Free (P);
   Put_Line ("wrote ode_harmonic_oscillator_ada.svg");
   Free (Sol);
   Free (Solver);

   Put_Line (ASCII.LF & "=================== Lorenz attractor (phase plane) ===================");
   Solver := Rk4_Solver (0.005);
   Sol := Solver.Solve_Builtin ("lorenz", [10.0, 28.0, 8.0 / 3.0], [1.0, 1.0, 1.0], 0.0, 25.0);
   Put_Line ("steps_taken = " & Img (Sol.Steps_Taken));
   Xs := Solution_Component (Sol, 0);
   Zs := Solution_Component (Sol, 2);
   P := R_Plot_Plot (Xs, Zs, "l", "trajectory");
   Chained := P.Title_2 ("Lorenz Attractor (x-z phase plane)").X_Label_2 ("x").Y_Label_2 ("z");
   Chained.Save_Svg ("ode_lorenz_phase_plane_ada.svg");
   Free (P);
   Put_Line ("wrote ode_lorenz_phase_plane_ada.svg");
   Free (Sol);
   Free (Solver);
end Ode_Ex;
