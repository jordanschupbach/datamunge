# Short names matching the C++ API (dm::BSR, dm::PSR, dm::RGBA); see include/datamunge/vector.i.
BSR <- function(...) BSplineRegression(...)
PSR <- function(...) PSplineRegression(...)
RGBA <- function(...) RGB(...)

# RPlot's setters return *this for chaining, but SWIG hands back a second, non-owning wrapper of
# the same C++ object. In `RPlot_create()$points(...)$title(...)` the owning wrapper is then
# unreferenced, and once it is garbage collected its finalizer frees the plot the chain still
# points at. Return the receiver itself instead -- invisibly, like R's own modify-and-return
# helpers, so a bare chain at top level doesn't print `An object of class ...`.
.dm_chainable <- function(f) {
  force(f)
  function(self, ...) {
    f(self, ...)
    invisible(self)
  }
}
local({
  ns <- topenv()
  methods <- c(
    "size", "title", "x_label", "y_label", "xlab", "ylab", "background", "axis_color",
    "grid_color", "show_grid", "x_limits", "y_limits", "hide_axes", "x_tick_labels",
    "points", "line", "lines", "bars", "box", "abline", "abline_h", "abline_v", "qqline",
    "legend", "text", "polygon", "segments"
  )
  for (m in methods) {
    name <- paste0("RPlot_", m)
    assign(name, .dm_chainable(get(name, envir = ns)), envir = ns)
  }
  for (m in c("add", "size")) {
    name <- paste0("RLayout_", m)
    assign(name, .dm_chainable(get(name, envir = ns)), envir = ns)
  }
})
