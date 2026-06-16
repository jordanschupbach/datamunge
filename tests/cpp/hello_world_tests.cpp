#include <gtest/gtest.h>

#include <octra/octra.hpp>

TEST(octra, hello_is_callable) {
  ASSERT_NO_THROW(octra::hello());
}
