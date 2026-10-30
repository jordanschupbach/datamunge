/// @file profiler.cpp
/// @brief Profiler, ProfileScope, summaries, Chrome trace export and plot_timeline().

#include "datamunge/nn/profiler.hpp"

#include <algorithm>
#include <cstdio>
#include <atomic>
#include <cmath>
#include <functional>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace datamunge::nn {

/// @brief Helpers private to this file.
namespace {

/// The active profiler, or null.
std::atomic<Profiler*> g_active{nullptr};
/// Recording session number; bumped by every start() and stop() so stale scopes are dropped.
std::atomic<std::uint64_t> g_generation{0};
/// Guards g_active changes and every profiler's event list.
std::mutex g_mutex;
/// Nesting depth of live ProfileScopes on this thread.
thread_local int t_depth = 0;

/// @brief Microseconds from @p a to @p b.
/// @param a  Earlier time.
/// @param b  Later time.
/// @return The difference.
double micros(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

/// @brief Escapes a string for a JSON string literal.
/// @param s  The text.
/// @return The escaped text (without quotes).
std::string json_escape(const std::string& s) {
    std::string out;
    for (const char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

/// @brief Formats a duration in microseconds with a readable unit.
/// @param us  Microseconds.
/// @return E.g. "12.3 us", "4.56 ms", "1.20 s".
std::string format_time(double us) {
    char buf[32];
    if (us < 1000.0) std::snprintf(buf, sizeof buf, "%.1f us", us);
    else if (us < 1e6) std::snprintf(buf, sizeof buf, "%.2f ms", us / 1e3);
    else std::snprintf(buf, sizeof buf, "%.2f s", us / 1e6);
    return buf;
}

/// @brief Self time of each event: its duration minus those of the events nested directly inside it.
/// @param events  Events of one profiler.
/// @return Self times, parallel to @p events.
std::vector<double> self_times(const std::vector<ProfileEvent>& events) {
    std::vector<std::size_t> order(events.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        const auto& x = events[a];
        const auto& y = events[b];
        if (x.thread != y.thread) return x.thread < y.thread;
        if (x.start_us != y.start_us) return x.start_us < y.start_us;
        return x.depth < y.depth;
    });
    std::vector<double>      self(events.size());
    std::vector<std::size_t> stack;
    int                      thread = -1;
    for (const std::size_t i : order) {
        const auto& e = events[i];
        self[i]       = e.duration_us;
        if (e.thread != thread) stack.clear(), thread = e.thread;
        while (!stack.empty() && events[stack.back()].depth >= e.depth) stack.pop_back();
        if (!stack.empty() && events[stack.back()].depth == e.depth - 1) self[stack.back()] -= e.duration_us;
        stack.push_back(i);
    }
    return self;
}

}  // namespace

Profiler::~Profiler() { stop(); }

void Profiler::start() {
    std::lock_guard lock(g_mutex);
    events_.clear();
    threads_.clear();
    origin_ = std::chrono::steady_clock::now();
    g_generation.fetch_add(1);
    g_active.store(this);
}

void Profiler::stop() {
    std::lock_guard lock(g_mutex);
    if (g_active.load() != this) return;
    stopped_ = std::chrono::steady_clock::now();
    g_generation.fetch_add(1);
    g_active.store(nullptr);
}

bool Profiler::active() const { return g_active.load() == this; }

void Profiler::clear() {
    std::lock_guard lock(g_mutex);
    events_.clear();
}

std::vector<ProfileEvent> Profiler::events() const {
    std::lock_guard lock(g_mutex);
    return events_;
}

double Profiler::wall_us() const {
    const auto end = active() ? std::chrono::steady_clock::now() : stopped_;
    return std::max(0.0, micros(origin_, end));
}

void Profiler::record(std::string name, std::string category, int depth, std::chrono::steady_clock::time_point begin,
                      std::chrono::steady_clock::time_point end, std::uint64_t generation) {
    std::lock_guard lock(g_mutex);
    if (g_active.load() != this || g_generation.load() != generation) return;
    append_locked(std::move(name), std::move(category), depth, begin, end);
}

void Profiler::append_locked(std::string name, std::string category, int depth, std::chrono::steady_clock::time_point begin,
                             std::chrono::steady_clock::time_point end) {
    const std::size_t id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    auto              it = std::find(threads_.begin(), threads_.end(), id);
    if (it == threads_.end()) it = threads_.insert(threads_.end(), id);
    events_.push_back({std::move(name), std::move(category), static_cast<int>(it - threads_.begin()), depth,
                       micros(origin_, begin), micros(begin, end)});
}

std::vector<ProfileRow> Profiler::summary() const {
    const auto                                               ev   = events();
    const auto                                               self = self_times(ev);
    std::map<std::pair<std::string, std::string>, ProfileRow> rows;
    for (std::size_t i = 0; i < ev.size(); ++i) {
        auto& r = rows[{ev[i].category, ev[i].name}];
        if (r.calls == 0) {
            r.name     = ev[i].name;
            r.category = ev[i].category;
            r.min_us   = ev[i].duration_us;
            r.max_us   = ev[i].duration_us;
        }
        ++r.calls;
        r.total_us += ev[i].duration_us;
        r.self_us += self[i];
        r.min_us = std::min(r.min_us, ev[i].duration_us);
        r.max_us = std::max(r.max_us, ev[i].duration_us);
    }
    std::vector<ProfileRow> out;
    for (auto& [key, r] : rows) {
        r.mean_us = r.total_us / static_cast<double>(r.calls);
        out.push_back(r);
    }
    std::stable_sort(out.begin(), out.end(), [](const ProfileRow& a, const ProfileRow& b) { return a.total_us > b.total_us; });
    return out;
}

std::string Profiler::to_markdown(std::size_t max_rows) const {
    auto         rows = summary();
    const double wall = wall_us();
    if (max_rows > 0 && rows.size() > max_rows) {
        ProfileRow other{"(other)", "", 0, 0, 0, 0, 0, 0};
        for (std::size_t i = max_rows; i < rows.size(); ++i) {
            other.calls += rows[i].calls;
            other.total_us += rows[i].total_us;
            other.self_us += rows[i].self_us;
        }
        other.mean_us = other.calls ? other.total_us / static_cast<double>(other.calls) : 0;
        rows.resize(max_rows);
        rows.push_back(other);
    }
    std::ostringstream out;
    out << "| name | category | calls | total | self | mean | self share |\n";
    out << "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |\n";
    for (const auto& r : rows) {
        char share[16];
        std::snprintf(share, sizeof share, "%.1f%%", wall > 0 ? 100.0 * r.self_us / wall : 0.0);
        out << "| " << r.name << " | " << r.category << " | " << r.calls << " | " << format_time(r.total_us) << " | "
            << format_time(r.self_us) << " | " << format_time(r.mean_us) << " | " << share << " |\n";
    }
    return out.str();
}

std::string Profiler::to_chrome_trace() const {
    const auto         ev = events();
    std::ostringstream out;
    out.precision(17);
    out << "{\"displayTimeUnit\":\"ms\",\"traceEvents\":[";
    for (std::size_t i = 0; i < ev.size(); ++i) {
        out << (i ? "," : "") << "\n{\"name\":\"" << json_escape(ev[i].name) << "\",\"cat\":\"" << json_escape(ev[i].category)
            << "\",\"ph\":\"X\",\"pid\":0,\"tid\":" << ev[i].thread << ",\"ts\":" << ev[i].start_us
            << ",\"dur\":" << ev[i].duration_us << "}";
    }
    out << "\n]}\n";
    return out.str();
}

bool profiling() { return g_active.load(std::memory_order_relaxed) != nullptr; }

ProfileScope::ProfileScope(std::string_view name, std::string_view category) {
    Profiler* p = g_active.load(std::memory_order_acquire);
    if (p == nullptr) return;
    profiler_   = p;
    generation_ = g_generation.load();
    name_.assign(name);
    category_.assign(category);
    depth_ = t_depth++;
    begin_ = std::chrono::steady_clock::now();
}

ProfileScope::~ProfileScope() {
    if (profiler_ == nullptr) return;
    const auto end = std::chrono::steady_clock::now();
    --t_depth;
    // Only dereference the profiler if it is still the active one in the same session; stop()
    // and the destructor take g_mutex, so the check below cannot race with destruction.
    std::lock_guard lock(g_mutex);
    if (g_active.load() != profiler_ || g_generation.load() != generation_) return;
    profiler_->append_locked(std::move(name_), std::move(category_), depth_, begin_, end);
}

graphics::RPlot plot_timeline(const Profiler& profiler, const std::string& title, double from_us, double to_us) {
    if (!std::isfinite(to_us)) to_us = profiler.wall_us();
    if (!(to_us > from_us)) throw std::invalid_argument("plot_timeline: empty time window");
    std::vector<ProfileEvent> ev;
    for (auto& e : profiler.events())
        if (e.start_us < to_us && e.start_us + e.duration_us > from_us) ev.push_back(e);
    if (ev.empty()) throw std::invalid_argument("plot_timeline: no events in the time window");

    // Lanes: one per (thread, depth), thread-major, depth 0 on top.
    std::map<std::pair<int, int>, int> lane_of;
    int                                threads = 0;
    for (const auto& e : ev) lane_of[{e.thread, e.depth}] = 0, threads = std::max(threads, e.thread + 1);
    int lanes = 0;
    for (auto& [key, lane] : lane_of) lane = lanes++;

    // Colours by name, most total time first.
    std::map<std::string, double> by_name;
    for (const auto& e : ev) by_name[e.name] += std::min(e.start_us + e.duration_us, to_us) - std::max(e.start_us, from_us);
    std::vector<std::pair<double, std::string>> ranked;
    for (const auto& [n, t] : by_name) ranked.emplace_back(-t, n);
    std::sort(ranked.begin(), ranked.end());
    static const std::vector<graphics::RGB> palette = {
        {37, 99, 235},  {234, 88, 12},  {22, 163, 74},  {220, 38, 38},  {147, 51, 234}, {8, 145, 178},
        {202, 138, 4},  {219, 39, 119}, {79, 70, 229},  {101, 163, 13}, {120, 113, 108}, {14, 116, 144}};
    std::map<std::string, graphics::RGB> colour;
    for (std::size_t i = 0; i < ranked.size(); ++i) colour[ranked[i].second] = palette[i % palette.size()];

    const std::size_t legend_rows = std::min<std::size_t>(ranked.size(), palette.size());
    const double      lane_px     = 44.0;
    const double      headroom    = static_cast<double>(legend_rows) * 20.0 / lane_px + 0.3;
    auto              plot        = graphics::RPlot::create();
    const double      span        = to_us - from_us;
    for (const auto& e : ev) {
        const double x0 = std::max(e.start_us, from_us);
        const double x1 = std::min(e.start_us + e.duration_us, to_us);
        const double y  = static_cast<double>(lanes - 1 - lane_of[{e.thread, e.depth}]);
        plot.polygon({x0, x1, x1, x0}, {y - 0.38, y - 0.38, y + 0.38, y + 0.38}, colour[e.name]);
        if ((x1 - x0) / span * 900.0 > 8.0 * static_cast<double>(e.name.size()) + 6.0)
            plot.text(0.5 * (x0 + x1), y - 0.08, e.name, graphics::RGB{255, 255, 255}, 11.0);
    }
    std::vector<std::string> labels(static_cast<std::size_t>(lanes));
    for (const auto& [key, lane] : lane_of)
        labels[static_cast<std::size_t>(lanes - 1 - lane)] =
            (threads > 1 ? "thread " + std::to_string(key.first) + ", " : std::string()) + "depth " + std::to_string(key.second);
    std::vector<std::string> legend_labels;
    std::vector<graphics::RGB> legend_colours;
    for (std::size_t i = 0; i < legend_rows; ++i) {
        legend_labels.push_back(ranked[i].second);
        legend_colours.push_back(colour[ranked[i].second]);
    }
    plot.legend(legend_labels, legend_colours);
    plot.y_tick_labels(labels);
    plot.x_limits(from_us, to_us).y_limits(-0.5, static_cast<double>(lanes) - 0.5 + headroom);
    plot.xlab("time since start (us)").ylab("").title(title);
    plot.size(1100, static_cast<std::size_t>(150.0 + lane_px * (static_cast<double>(lanes) + headroom)));
    return plot;
}

}  // namespace datamunge::nn
