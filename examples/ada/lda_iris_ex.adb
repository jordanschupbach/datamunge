--  CMD: just TARGET=lda_iris_ex run-ada
--  Linear discriminant analysis on iris: summary, confusion matrix, predictions and discriminant plot.
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;

procedure Lda_Iris_Ex is
   Iris    : Data_Frame := Data.Iris;
   Newdata : Data_Frame := New_Data_Frame;
   Model   : Lda;
   Table   : Data_Frame;
begin
   Put_Line ("iris: " & Img (Iris.Nrows) & " rows x " & Img (Iris.Ncols) & " cols" & ASCII.LF);

   Model := New_Lda (Iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
   Model.Print_Summary;

   Table := Model.Confusion_Matrix;
   Put_Line (ASCII.LF & "Confusion matrix (rows = actual, cols = predicted):" & ASCII.LF & Table.Image);
   Free (Table);

   Newdata.Add_Numeric_Column ("Sepal.Length", [5.1, 6.0, 6.5, 6.2]);
   Newdata.Add_Numeric_Column ("Sepal.Width", [3.5, 2.7, 3.0, 2.8]);
   Newdata.Add_Numeric_Column ("Petal.Length", [1.4, 4.5, 5.5, 4.8]);
   Newdata.Add_Numeric_Column ("Petal.Width", [0.2, 1.5, 2.0, 1.8]);
   Table := Model.Predict_Frame (Newdata);
   Put_Line (ASCII.LF & "Predictions for new flowers:" & ASCII.LF & Table.Image);
   Free (Table);

   Model.Save_Discriminant_Plot ("lda_iris_discriminants.svg");
   Put_Line (ASCII.LF & "Saved discriminant plot as lda_iris_discriminants.svg");

   Free (Model);
   Free (Newdata);
   Free (Iris);
end Lda_Iris_Ex;
