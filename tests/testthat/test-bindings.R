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
  expect_match(printed, "south \\| 19 \\| 25")
})
