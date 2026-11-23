--  Ada type extensions of the datamunge objective interfaces for optim_advanced_ex.adb (a
--  library-level package, so the objects can be allocated for the binding's access types).
with Datamunge; use Datamunge;

package Optim_Advanced_Objectives is

   --  f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with its exact gradient and Hessian.
   type Quadratic_Record is new Hessian_Function_Record with null record;
   overriding function Evaluate (Self : in out Quadratic_Record; Coordinates : Long_Float_Vector) return Long_Float;
   overriding function Gradient (Self : in out Quadratic_Record; Coordinates : Long_Float_Vector)
     return Long_Float_Vector;
   overriding function Hessian (Self : in out Quadratic_Record; Coordinates : Long_Float_Vector)
     return Long_Float_Vector_Vector;

   --  Residuals of y = a * exp(b t) against exact data from a = 2, b = 0.5 at the times Ts.
   type Exponential_Record is new Residual_Function_Record with record
      Ts : Long_Float_Vector;
   end record;
   overriding function Residuals (Self : in out Exponential_Record; Coordinates : Long_Float_Vector)
     return Long_Float_Vector;
   overriding function Jacobian (Self : in out Exponential_Record; Coordinates : Long_Float_Vector)
     return Long_Float_Vector_Vector;

   --  A Hessian_Function that leaves out Gradient and Hessian.
   type Incomplete_Record is new Hessian_Function_Record with null record;
   overriding function Evaluate (Self : in out Incomplete_Record; Coordinates : Long_Float_Vector) return Long_Float;

end Optim_Advanced_Objectives;
