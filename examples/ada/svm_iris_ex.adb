--  CMD: just TARGET=svm_iris_ex run-ada
--  Support vector machine on iris: RBF vs. linear kernels, confusion matrix and predictions.
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;

procedure Svm_Iris_Ex is
   Formula      : constant String := "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width";
   Iris         : Data_Frame := Data.Iris;
   Newdata      : Data_Frame := New_Data_Frame;
   Rbf_Model    : Svm;
   Linear_Model : Svm;
   Table        : Data_Frame;
begin
   Put_Line ("iris: " & Img (Iris.Nrows) & " rows x " & Img (Iris.Ncols) & " cols" & ASCII.LF);

   Put_Line ("=== RBF kernel (default) ===");
   Rbf_Model := New_Svm (Iris, Formula);
   Rbf_Model.Print_Summary;
   Table := Rbf_Model.Confusion_Matrix;
   Put_Line (ASCII.LF & "Confusion matrix (rows = actual, cols = predicted):" & ASCII.LF & Table.Image);
   Free (Table);

   Put_Line (ASCII.LF & "=== Linear kernel, for comparison ===");
   Linear_Model := New_Svm (Iris, Formula, "linear");
   Put_Line ("Training accuracy: " & Fmt (Linear_Model.Training_Accuracy * 100.0) & "%");
   Put_Line ("Support vectors: " & Img (Linear_Model.Num_Support_Vectors));

   Newdata.Add_Numeric_Column ("Sepal.Length", [5.1, 6.0, 6.5, 6.2]);
   Newdata.Add_Numeric_Column ("Sepal.Width", [3.5, 2.7, 3.0, 2.8]);
   Newdata.Add_Numeric_Column ("Petal.Length", [1.4, 4.5, 5.5, 4.8]);
   Newdata.Add_Numeric_Column ("Petal.Width", [0.2, 1.5, 2.0, 1.8]);
   Table := Rbf_Model.Predict_Frame (Newdata);
   Put_Line (ASCII.LF & "RBF predictions for new flowers (votes out of 3 one-vs-one pairs):" & ASCII.LF
             & Table.Image);
   Free (Table);

   Free (Linear_Model);
   Free (Rbf_Model);
   Free (Newdata);
   Free (Iris);
end Svm_Iris_Ex;
