--  CMD: just TARGET=autodiff_ex run-ada
--  Automatic differentiation: forward mode (Dual), reverse mode (Tape/Var), Hessians (HyperDual).
with Ada.Containers.Vectors;
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;

procedure Autodiff_Ex is
   --  Note: the C++ autodiff_ex.cpp additionally demonstrates the generic template drivers
   --  (derivative/gradient_forward/gradient_reverse/jacobian_forward/hessian), which can't be
   --  bound. This port demonstrates the same ideas with the bound named-method API.
   --  Index (V) is the Var's position in the adjoint vector Backward returns.

   --  Every Dual/Var/Hyper_Dual operation returns a new object that must be freed. The example
   --  keeps them in an arena and frees it at the end of each section.
   package Object_Vectors is new Ada.Containers.Vectors (Positive, Native_Object);
   Arena : Object_Vectors.Vector;

   function Keep (D : Dual) return Dual is
   begin
      Arena.Append (Native_Object (D));
      return D;
   end Keep;

   function Keep (V : Var) return Var is
   begin
      Arena.Append (Native_Object (V));
      return V;
   end Keep;

   function Keep (H : Hyper_Dual) return Hyper_Dual is
   begin
      Arena.Append (Native_Object (H));
      return H;
   end Keep;

   function Keep (T : Tape) return Tape is
   begin
      Arena.Append (Native_Object (T));
      return T;
   end Keep;

   --  Frees the arena's objects, newest first (so each Var goes before its Tape).
   procedure Free_Arena is
   begin
      for I in reverse Arena.First_Index .. Arena.Last_Index loop
         declare
            Object : Native_Object := Arena (I);
         begin
            Free (Object);
         end;
      end loop;
      Arena.Clear;
   end Free_Arena;

   function Rosenbrock_Dual (X0, X1 : Dual) return Dual is
      A : constant Dual := Keep (Keep (New_Dual (1.0, 0.0)).Subtract (X0));
      B : constant Dual := Keep (X1.Subtract (Keep (X0.Multiply (X0))));
   begin
      return Keep (Keep (A.Multiply (A)).Add (Keep (Keep (B.Multiply (B)).Multiply_Scalar (100.0))));
   end Rosenbrock_Dual;

   function Rosenbrock_Value (X0, X1 : Long_Float) return Long_Float is
     (C_Pow (1.0 - X0, 2.0) + 100.0 * C_Pow (X1 - X0 * X0, 2.0));

   --  Build the Rosenbrock graph on tape; returns the output Var.
   function Rosenbrock_Var (T : Tape; X0, X1 : Var) return Var is
      A : constant Var := Keep (Keep (New_Var (T, 1.0)).Subtract (X0));
      B : constant Var := Keep (X1.Subtract (Keep (X0.Multiply (X0))));
   begin
      return Keep (Keep (A.Multiply (A)).Add (Keep (Keep (B.Multiply (B)).Multiply_Scalar (100.0))));
   end Rosenbrock_Var;

   function Rosenbrock_Hyper_Dual (X0, X1 : Hyper_Dual) return Hyper_Dual is
      A : constant Hyper_Dual := Keep (Keep (New_Hyper_Dual (1.0, 0.0, 0.0, 0.0)).Subtract (X0));
      B : constant Hyper_Dual := Keep (X1.Subtract (Keep (X0.Multiply (X0))));
   begin
      return Keep (Keep (A.Multiply (A)).Add (Keep (Keep (B.Multiply (B)).Multiply_Scalar (100.0))));
   end Rosenbrock_Hyper_Dual;

   function Adjoint (Adjoints : Long_Float_Vector; V : Var) return Long_Float is
     (Adjoints (Positive (V.Index + 1)));

   type Seed is record
      X, Y : Long_Float;
   end record;
   Seeds : constant array (0 .. 1) of Seed := [(1.0, 0.0), (0.0, 1.0)];
begin
   Put_Line ("=================== Forward mode: scalar derivative ===================");
   declare
      X0 : constant Dual := Keep (New_Dual (2.0, 1.0));  --  seed derivative = 1 to read df/dx directly
      F  : constant Dual :=
        Keep (Keep (Keep (X0.Multiply (X0)).Multiply (X0)).Subtract (Keep (X0.Multiply_Scalar (2.0))));  --  x^3 - 2x
   begin
      Put_Line ("f(x) = x^3 - 2x, f'(2) = " & Fmt (F.Derivative) & " (exact: 10)");
   end;
   Free_Arena;

   Put_Line (ASCII.LF & "=================== Reverse mode: build a graph by hand ===================");
   declare
      T       : constant Tape := Keep (New_Tape);
      A       : constant Var := Keep (New_Var (T, 2.0));
      B       : constant Var := Keep (New_Var (T, 3.0));
      Y       : constant Var := Keep (Keep (A.Multiply (B)).Add (Keep (A.Sin)));
      Adjoints : Long_Float_Vector;
   begin
      Put_Line ("y = a*b + sin(a) at a=2, b=3 -> y = " & Fmt (Y.Value));
      Adjoints := T.Backward (Y);
      Put_Line ("dy/da = " & Fmt (Adjoint (Adjoints, A)) & " (exact: b + cos(a))");
      Put_Line ("dy/db = " & Fmt (Adjoint (Adjoints, B)) & " (exact: a)");
   end;
   Free_Arena;

   Put_Line (ASCII.LF & "=================== Forward vs reverse mode agree on the Rosenbrock function ===================");
   declare
      Px : constant Long_Float := 0.0;
      Py : constant Long_Float := 0.0;
      --  Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
      Gx : constant Long_Float := Rosenbrock_Dual (Keep (New_Dual (Px, 1.0)), Keep (New_Dual (Py, 0.0))).Derivative;
      Gy : constant Long_Float := Rosenbrock_Dual (Keep (New_Dual (Px, 0.0)), Keep (New_Dual (Py, 1.0))).Derivative;
      --  Reverse mode: one Tape/Var pass computes every partial at once.
      T    : constant Tape := Keep (New_Tape);
      Vx   : constant Var := Keep (New_Var (T, Px));
      Vy   : constant Var := Keep (New_Var (T, Py));
      Grad : constant Long_Float_Vector := T.Backward (Rosenbrock_Var (T, Vx, Vy));
   begin
      Put_Line ("f(0,0) = " & Fmt (Rosenbrock_Value (Px, Py)));
      Put_Line ("gradient (forward mode): [" & Fmt (Gx) & ", " & Fmt (Gy) & "]");
      Put_Line ("gradient (reverse mode): [" & Fmt (Adjoint (Grad, Vx)) & ", " & Fmt (Adjoint (Grad, Vy)) & "]");
   end;
   Free_Arena;

   Put_Line (ASCII.LF & "=================== Jacobian of a vector-valued function ===================");
   --  f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
   declare
      type Dual_Array is array (0 .. 2) of Dual;
      function Vector_Fn (X, Y : Dual) return Dual_Array is
        ([Keep (X.Multiply (X)), Keep (X.Multiply (Y)), Keep (Keep (Y.Multiply (Y)).Multiply (Y))]);
      Jac : array (0 .. 2, 0 .. 1) of Long_Float;
   begin
      for Col in 0 .. 1 loop
         declare
            Outputs : constant Dual_Array :=
              Vector_Fn (Keep (New_Dual (2.0, Seeds (Col).X)), Keep (New_Dual (3.0, Seeds (Col).Y)));
         begin
            for Row in Outputs'Range loop
               Jac (Row, Col) := Outputs (Row).Derivative;
            end loop;
         end;
      end loop;
      Put_Line ("f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:");
      for Row in 0 .. 2 loop
         Put_Line ("  [" & Fmt (Jac (Row, 0)) & ", " & Fmt (Jac (Row, 1)) & "]");
      end loop;
   end;
   Free_Arena;

   Put_Line (ASCII.LF & "=================== Hessian via second-order forward mode (HyperDual) ===================");
   --  One Hyper_Dual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
   declare
      Mx : constant Long_Float := 1.0;  --  the Rosenbrock function's minimum
      My : constant Long_Float := 1.0;
      H  : array (0 .. 1, 0 .. 1) of Long_Float;
   begin
      for I in 0 .. 1 loop
         for J in 0 .. 1 loop
            H (I, J) := Rosenbrock_Hyper_Dual
              (Keep (New_Hyper_Dual (Mx, Seeds (I).X, Seeds (J).X, 0.0)),
               Keep (New_Hyper_Dual (My, Seeds (I).Y, Seeds (J).Y, 0.0))).Eps1eps2;
         end loop;
      end loop;
      Put_Line ("Hessian of the Rosenbrock function at its minimum (1,1):");
      for I in 0 .. 1 loop
         Put_Line ("  [" & Fmt (H (I, 0)) & ", " & Fmt (H (I, 1)) & "]");
      end loop;
   end;
   Free_Arena;

   Put_Line (ASCII.LF & "=================== Gradient descent driven by reverse-mode gradients ===================");
   declare
      Point         : array (0 .. 1) of Long_Float := [-1.2, 1.0];  --  the classic Rosenbrock start
      Learning_Rate : constant Long_Float := 0.001;
      N_Steps       : constant := 2000;
   begin
      for Step in 0 .. N_Steps - 1 loop
         declare
            T    : constant Tape := Keep (New_Tape);
            Vx   : constant Var := Keep (New_Var (T, Point (0)));
            Vy   : constant Var := Keep (New_Var (T, Point (1)));
            Vf   : constant Var := Rosenbrock_Var (T, Vx, Vy);
            Grad : constant Long_Float_Vector := T.Backward (Vf);
            Loss : constant Long_Float := Vf.Value;
         begin
            Point (0) := Point (0) - Learning_Rate * Adjoint (Grad, Vx);
            Point (1) := Point (1) - Learning_Rate * Adjoint (Grad, Vy);
            if Step = 0 or else Step = N_Steps - 1 then
               Put_Line ("step " & Img (Long_Long_Integer (Step)) & ": loss = " & Fmt (Loss) & ", x = ["
                         & Fmt (Point (0)) & ", " & Fmt (Point (1)) & "]");
            end if;
         end;
         Free_Arena;
      end loop;
      Put_Line ("(true minimum is at [1, 1] with loss 0)");
   end;
end Autodiff_Ex;
