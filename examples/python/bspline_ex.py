"""Fits a two-dimensional B-spline surface with the bs(x, y) formula term."""
import math

from pydatamunge import datamunge

x, y, z = [], [], []
for iy in range(8):
    for ix in range(8):
        xv, yv = ix / 7.0, iy / 7.0
        x.append(xv)
        y.append(yv)
        z.append(math.sin(xv) + yv * yv)

data = datamunge.DataFrame()
data.add_numeric_column("x", datamunge.DVector(x))
data.add_numeric_column("y", datamunge.DVector(y))
data.add_numeric_column("z", datamunge.DVector(z))

surface = datamunge.LM(data, "z ~ bs(x, y)")
print("Fitted z ~ bs(x, y)")
surface.print_summary()

new_points = datamunge.DataFrame()
new_points.add_numeric_column("x", datamunge.DVector([0.25, 0.75]))
new_points.add_numeric_column("y", datamunge.DVector([0.50, 0.25]))
print("Predictions:", list(surface.predict(new_points)))
