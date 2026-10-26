#pragma once
#include <optional>
#include <utility>
#include <datamunge/dstruct/foundational.hpp>
namespace datamunge::dstruct { template <typename T> class SelfOrganizingList { public: void push_front(T value) { values_.push_front(std::move(value)); } std::optional<T> find(const T& value) { LinkedList<T> prior; while (auto item=values_.pop_front()) { if (*item==value) { values_.push_front(*item); while(auto p=prior.pop_front()) values_.push_back(*p); return item; } prior.push_front(*item); } while(auto p=prior.pop_front()) values_.push_back(*p); return std::nullopt; } T& front() { return values_.front(); } private: LinkedList<T> values_; }; }
