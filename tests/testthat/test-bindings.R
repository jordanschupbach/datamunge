test_that("hello() is callable", {
  expect_true(exists("hello", where = asNamespace("datamunger")))
  expect_silent(datamunger::hello())
})

test_that("STL templates are usable", {
  expect_true(exists("DVector", where = asNamespace("datamunger")))
  expect_true(exists("DPair", where = asNamespace("datamunger")))
})

test_that("R dataframe wrapper supports the example workflow", {
  sales <- DataFrame()
  sales$add_column("region", c("west", "west", "east", "south", "south", "south"))
  sales$add_column("product", c("widget", "widget", "widget", "gizmo", "gizmo", "gizmo"))
  sales$add_column("sales", c(10, 10, 14, 8, NA, 11))
  sales$add_column("quarter", c("Q1", "Q1", "Q1", "Q2", "Q2", NA))

  cleaned <- sales$drop_duplicates(c("region", "product", "sales", "quarter"))
  cleaned$fill_null("quarter", "unknown")
  cleaned$fill_null("sales", 0)

  grouped <- cleaned$group_by_sum("region", "sales")$sort_by("sales", FALSE)

  targets <- DataFrame()
  targets$add_column("region", c("west", "east", "south"))
  targets$add_column("target", c(18, 12, 25))
  joined <- grouped$join(targets, "region", "region", JoinType$Left)

  summary <- cleaned$describe_numeric("sales")
  expect_equal(cleaned$shape(), c(5L, 4L))
  expect_equal(summary$count, 5L)
  expect_equal(summary$null_count, 0L)
  expect_equal(summary$sum, 43)
  expect_equal(summary$mean, 8.6)

  printed <- joined$to_string()
  expect_match(printed, "DataFrame\\[3 x 3\\]")
  expect_match(printed, "south\\s+\\| 19\\s+\\| 25")
})

test_that("R dataframe wrapper supports tibble-style piping via $", {
  sales <- DataFrame()
  sales$add_column("region", c("west", "west", "east", "south", "south"))
  sales$add_column("sales", c(10, 20, 14, 8, 11))

  piped <- sales$mutate("tax", sales$pull("sales") * 0.1)$arrange("sales", FALSE)$select(c("region", "sales", "tax"))
  expect_equal(piped$columns(), c("region", "sales", "tax"))
  expect_equal(piped$nrows(), 5L)

  expect_true("area" %in% sales$rename("region", "area")$columns())
  expect_equal(sales$distinct(c("region"))$nrows(), 3L)
  expect_equal(sales$n_distinct("region"), 3L)
  expect_equal(sales$relocate(c("sales"))$columns()[1], "sales")

  with_na <- DataFrame()
  with_na$add_column("x", c(1, NA, 3))
  pulled <- with_na$pull("x")
  expect_true(is.na(pulled[2]))
  expect_equal(pulled[c(1, 3)], c(1, 3))

  grouped <- sales$group_by(c("region"))$summarise(total = c("sales", "sum"), n = c("", "count"))
  expect_true(all(c("region", "total", "n") %in% grouped$columns()))
  expect_equal(sum(grouped$pull("total")), sum(sales$pull("sales")))

  wide <- DataFrame()
  wide$add_column("id", c("p1", "p2"))
  wide$add_column("q1", c(1, 3))
  wide$add_column("q2", c(2, 4))
  longer <- wide$pivot_longer(c("q1", "q2"), "quarter", "value")
  expect_equal(longer$nrows(), 4L)
  wider <- longer$pivot_wider("quarter", "value")
  expect_equal(wider$nrows(), 2L)
  expect_true(all(c("q1", "q2") %in% wider$columns()))

  extra <- DataFrame()
  extra$add_column("region", c("north"))
  extra$add_column("sales", c(5))
  expect_equal(sales$bind_rows(extra)$nrows(), 6L)

  targets <- DataFrame()
  targets$add_column("region", c("west", "east"))
  targets$add_column("target", c(100, 200))
  expect_equal(sales$join(targets, "region", "region", JoinType$Right)$nrows(), 3L)
  expect_equal(sales$join(targets, "region", "region", JoinType$Anti)$nrows(), 2L)
})
