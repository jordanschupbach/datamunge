/// @file build_info.cpp
/// @brief build_info(), BuildInfo::to_markdown() and matmul_flops().

#include "datamunge/nn/build_info.hpp"

#include "datamunge/nn/linalg.hpp"
#include "datamunge/nn/parallel.hpp"

#include <sstream>
#include <thread>

#if __has_include("datamunge/version.h")
#include "datamunge/version.h"
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#if defined(__linux__) && defined(__aarch64__)
#include <sys/auxv.h>
#endif

namespace datamunge::nn {

/// @brief Helpers private to this file.
namespace {

/// @brief The SIMD and arithmetic features of the running CPU, by the platform's own probe.
/// @return Feature names; empty where no probe is available.
std::vector<std::string> probe_cpu_features() {
    std::vector<std::string> out;
#if defined(__APPLE__)
    const std::pair<const char*, const char*> keys[] = {
        {"NEON", "hw.optional.neon"},       {"FP16", "hw.optional.arm.FEAT_FP16"}, {"DotProd", "hw.optional.arm.FEAT_DotProd"},
        {"BF16", "hw.optional.arm.FEAT_BF16"}, {"I8MM", "hw.optional.arm.FEAT_I8MM"}, {"SME", "hw.optional.arm.FEAT_SME"},
        {"AVX", "hw.optional.avx1_0"},      {"AVX2", "hw.optional.avx2_0"},        {"FMA", "hw.optional.fma"},
        {"AVX512F", "hw.optional.avx512f"}};
    for (const auto& [name, key] : keys) {
        long long   v    = 0;
        std::size_t size = sizeof v;
        if (sysctlbyname(key, &v, &size, nullptr, 0) == 0 && (size == 4 ? *reinterpret_cast<int*>(&v) : v) == 1)
            out.emplace_back(name);
    }
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("sse4.2")) out.emplace_back("SSE4.2");
    if (__builtin_cpu_supports("avx")) out.emplace_back("AVX");
    if (__builtin_cpu_supports("avx2")) out.emplace_back("AVX2");
    if (__builtin_cpu_supports("fma")) out.emplace_back("FMA");
    if (__builtin_cpu_supports("avx512f")) out.emplace_back("AVX512F");
#elif defined(__linux__) && defined(__aarch64__)
    const unsigned long hw = getauxval(AT_HWCAP);
    out.emplace_back("NEON");  // mandatory on AArch64
    if (hw & (1UL << 9)) out.emplace_back("FP16");     // HWCAP_ASIMDHP
    if (hw & (1UL << 20)) out.emplace_back("DotProd");  // HWCAP_ASIMDDP
#endif
    return out;
}

/// @brief The compiler that built this file.
/// @return Name and version.
std::string compiler_name() {
#if defined(__clang__)
    return "clang " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__) + "." +
           std::to_string(__clang_patchlevel__);
#elif defined(__GNUC__)
    return "gcc " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." + std::to_string(__GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
    return "msvc " + std::to_string(_MSC_VER);
#else
    return "unknown";
#endif
}

/// @brief The architecture this file was compiled for.
/// @return A short name.
std::string architecture_name() {
#if defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "aarch64";
#elif defined(__i386__)
    return "x86";
#else
    return "other";
#endif
}

}  // namespace

BuildInfo build_info() {
    BuildInfo info;
#ifdef DATAMUNGE_VERSION
    info.version = DATAMUNGE_VERSION;
#else
    info.version = "unknown";
#endif
    info.dtypes           = {DType::Bool, DType::Int32, DType::Int64, DType::Float32, DType::Float64};
    info.gemm_backend     = gemm_backend();
    info.lapack           = linalg_has_lapack();
    info.threads          = get_num_threads();
    info.hardware_threads = std::thread::hardware_concurrency();
    info.architecture     = architecture_name();
    info.compiler         = compiler_name();
    info.cpu_features     = probe_cpu_features();
    return info;
}

std::string BuildInfo::to_markdown() const {
    std::ostringstream out;
    auto               join = [](const auto& items, auto&& show) {
        std::string s;
        for (const auto& item : items) s += (s.empty() ? "" : ", ") + show(item);
        return s.empty() ? std::string("none detected") : s;
    };
    out << "| property | value |\n| :--- | :--- |\n";
    out << "| version | " << version << " |\n";
    out << "| execution | eager, define-by-run autograd |\n";
    out << "| device | CPU (one host device) |\n";
    out << "| dtypes | "
        << join(dtypes, [](DType d) { return dtype_name(d) + " (" + std::to_string(dtype_size(d)) + " B)"; }) << " |\n";
    out << "| GEMM backend | " << gemm_backend << " |\n";
    out << "| LAPACK | " << (lapack ? "yes" : "no (built-in kernels)") << " |\n";
    out << "| nn threads | " << threads << " |\n";
    out << "| hardware threads | " << hardware_threads << " |\n";
    out << "| architecture | " << architecture << " |\n";
    out << "| compiler | " << compiler << " |\n";
    out << "| CPU features | " << join(cpu_features, [](const std::string& s) { return s; }) << " |\n";
    return out.str();
}

double matmul_flops(std::int64_t m, std::int64_t k, std::int64_t n) {
    return 2.0 * static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(n);
}

}  // namespace datamunge::nn
