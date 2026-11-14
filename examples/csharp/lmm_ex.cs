// CMD: just TARGET=lmm_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// Linear mixed models: random intercept (penguins) and random intercept + slope (simulated schools).
using Datamunge;

var rng = new Random(2024);
double Uniform() => 1 - rng.NextDouble();  // (0, 1]: avoids log(0) in the Box-Muller transform.
double Gauss() => Math.Sqrt(-2 * Math.Log(Uniform())) * Math.Cos(2 * Math.PI * Uniform());

Console.WriteLine("=================== Random intercept on a real dataset (penguins) ===================");
new LMM(Data.Penguins()!, "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)").PrintSummary();

Console.WriteLine("\n=================== Random intercept + slope on a simulated multi-school dataset ===================");
// 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
// and the return on an extra hour of study vary by school -- the textbook case for a
// random-intercept-and-slope model.
const int nSchools = 30;
var schoolIntercept = Enumerable.Range(0, nSchools).Select(_ => Gauss() * 6).ToArray();
var schoolSlope = Enumerable.Range(0, nSchools).Select(_ => Gauss() * 1.2).ToArray();
const double trueIntercept = 60.0;
const double trueSlope = 3.0;

List<double> school = [], studyHours = [], score = [];
for (var s = 0; s < nSchools; s++)
{
    var nStudents = 15 + rng.Next(0, 21);
    for (var student = 0; student < nStudents; student++)
    {
        var hours = rng.NextDouble() * 10;
        school.Add(s);
        studyHours.Add(hours);
        score.Add(trueIntercept + schoolIntercept[s] + (trueSlope + schoolSlope[s]) * hours + Gauss() * 4);
    }
}
var df = new DataFrame();
df.AddNumericColumn("school", school);
df.AddNumericColumn("study_hours", studyHours);
df.AddNumericColumn("score", score);

var model = new LMM(df, "score ~ study_hours + (1 + study_hours | school)");
model.PrintSummary();

Console.WriteLine($"\nTrue generating values: intercept={trueIntercept}, slope={trueSlope}, random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0");

Console.WriteLine("\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---");
var labels = model.GroupLabels();
for (var idx = 0; idx < 3; idx++)
{
    var re = model.RandomEffectsForGroup((ulong)idx);
    Console.WriteLine($"school {labels[idx]}: intercept shift={re[0]}, slope shift={re[1]}");
}

Console.WriteLine("\n--- Prediction: population-level vs. school-adjusted ---");
var population = new DataFrame();
var school0 = new DataFrame();
population.AddNumericColumn("study_hours", [5.0]);
school0.AddNumericColumn("study_hours", [5.0]);
school0.AddNumericColumn("school", [0.0]);
Console.WriteLine($"5 study hours, unseen school:      {model.Predict(population)[0]} (fixed effects only)");
Console.WriteLine($"5 study hours, school 0 (known):    {model.Predict(school0)[0]} (fixed effects + school 0's BLUP)");
