with Dm_Examples; use Dm_Examples;

package body Optim_Advanced_Objectives is

   overriding function Evaluate (Self : in out Quadratic_Record; Coordinates : Long_Float_Vector) return Long_Float is
     (C_Pow (Coordinates (1) - 1.0, 2.0) + 10.0 * C_Pow (Coordinates (2) + 2.0, 2.0));

   overriding function Gradient (Self : in out Quadratic_Record; Coordinates : Long_Float_Vector)
     return Long_Float_Vector is
     ([2.0 * (Coordinates (1) - 1.0), 20.0 * (Coordinates (2) + 2.0)]);

   overriding function Hessian (Self : in out Quadratic_Record; Coordinates : Long_Float_Vector)
     return Long_Float_Vector_Vector is
     ([[2.0, 0.0], [0.0, 20.0]]);

   overriding function Residuals (Self : in out Exponential_Record; Coordinates : Long_Float_Vector)
     return Long_Float_Vector
   is
      Result : Long_Float_Vector;
   begin
      for T of Self.Ts loop
         Result.Append (Coordinates (1) * C_Exp (Coordinates (2) * T) - 2.0 * C_Exp (0.5 * T));
      end loop;
      return Result;
   end Residuals;

   overriding function Jacobian (Self : in out Exponential_Record; Coordinates : Long_Float_Vector)
     return Long_Float_Vector_Vector
   is
      Result : Long_Float_Vector_Vector;
   begin
      for T of Self.Ts loop
         Result.Append (Long_Float_Vector'[C_Exp (Coordinates (2) * T), Coordinates (1) * T * C_Exp (Coordinates (2) * T)]);
      end loop;
      return Result;
   end Jacobian;

   overriding function Evaluate (Self : in out Incomplete_Record; Coordinates : Long_Float_Vector) return Long_Float is
     (0.0);

end Optim_Advanced_Objectives;
