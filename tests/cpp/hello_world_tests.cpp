#include <gtest/gtest.h>

#include <datamunge/datamunge.hpp>

TEST(datamunge, hello_is_callable) {
  ASSERT_NO_THROW(datamunge::hello());
}
