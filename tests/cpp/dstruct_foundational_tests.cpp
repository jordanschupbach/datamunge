#include <gtest/gtest.h>

#include <datamunge/dstruct/dstruct.hpp>

using namespace datamunge::dstruct;

TEST(FoundationalStructures, ArrayAndDynamicArrayStoreValues) {
  Array<int, 2> fixed;
  fixed[0] = 3;
  fixed[1] = 5;
  EXPECT_EQ(fixed.at(1), 5);
  EXPECT_THROW(static_cast<void>(fixed.at(2)), std::out_of_range);

  DynamicArray<int> dynamic{1, 2};
  dynamic.push_back(3);
  EXPECT_EQ(dynamic.size(), 3U);
  EXPECT_EQ(dynamic.pop_back(), 3);
}

TEST(FoundationalStructures, LinkedListsSupportBothEnds) {
  LinkedList<int> singly;
  singly.push_front(2);
  singly.push_back(3);
  singly.push_front(1);
  EXPECT_EQ(singly.pop_front(), 1);
  EXPECT_EQ(singly.pop_front(), 2);

  DoublyLinkedList<int> doubly;
  doubly.push_back(2);
  doubly.push_front(1);
  doubly.push_back(3);
  EXPECT_EQ(doubly.pop_back(), 3);
  EXPECT_EQ(doubly.pop_front(), 1);
}

TEST(FoundationalStructures, StackQueueDequeAndCircularBufferHaveExpectedOrder) {
  Stack<int> stack;
  stack.push(1); stack.push(2);
  EXPECT_EQ(stack.pop(), 2);

  Queue<int> queue;
  queue.push(1); queue.push(2);
  EXPECT_EQ(queue.pop(), 1);

  DoubleEndedQueue<int> deque;
  deque.push_front(2); deque.push_front(1); deque.push_back(3);
  EXPECT_EQ(deque.pop_front(), 1);
  EXPECT_EQ(deque.pop_back(), 3);

  CircularBuffer<int> buffer(2);
  EXPECT_TRUE(buffer.push(1)); EXPECT_TRUE(buffer.push(2)); EXPECT_FALSE(buffer.push(3));
  EXPECT_EQ(buffer.pop(), 1); EXPECT_TRUE(buffer.push(3)); EXPECT_EQ(buffer.pop(), 2); EXPECT_EQ(buffer.pop(), 3);
}

TEST(FoundationalStructures, BitArrayManipulatesIndividualBits) {
  BitArray bits(10);
  bits.set(0); bits.set(9);
  EXPECT_TRUE(bits.at(0)); EXPECT_TRUE(bits.at(9)); EXPECT_FALSE(bits.at(1));
  bits.flip(0);
  EXPECT_FALSE(bits.at(0));
  EXPECT_THROW(static_cast<void>(bits.at(10)), std::out_of_range);
}
