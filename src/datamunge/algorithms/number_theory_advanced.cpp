/// @file number_theory_advanced.cpp
/// @brief Compatibility AKS forwarding implementation; ECM has its own checked canonical producer.
#include <datamunge/algorithms/number_theory_advanced.hpp>
#include <datamunge/algorithms/aks.hpp>
namespace datamunge::algorithms {
bool aks_is_prime(std::uint64_t n) { return aks_trace(n).is_prime; }
}
