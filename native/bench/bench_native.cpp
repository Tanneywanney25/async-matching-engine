// Native latency harness: replays an order stream through ee::OrderBook with no Python in the
// loop and reports add and cancel latency percentiles.
//
//   bench_native <stream.bin> <out.json> [--passes N] [--window-log2 K]
//
// The stream file is written by scripts/benchmark_native.py. Two formats:
//   "EESTREAM" v1: the 100,000-order, 10,000-cancel stream of scripts/benchmark.py in ticks and
//                  lots: a 32-byte header (magic, version, n_adds, n_cancels), n_adds records
//                  {id, price ticks, qty lots, side} of 32 bytes, then n_cancels order ids;
//   "EEOPSV02":    a mixed stream in recorded order (the one derived from the archive): a 24-byte
//                  header (magic, n_ops), then n_ops records {kind, side, id, price, qty} of 32 bytes,
//                  kind 0 limit, 1 market, 2 cancel; each kind's latencies are reported separately. Each operation is timed with two steady_clock reads; the
// clock's own cost (median of back-to-back reads) is measured and reported, not subtracted.
// Every pass replays the whole stream on a fresh book; the first pass is reported as the
// headline (a cold book, like scripts/benchmark.py's single pass) and every pass's statistics
// are kept so the variation between passes is visible.

#include "ee/order_book.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct AddRec {
    std::int64_t id, price, qty;
    std::uint8_t side;
    std::uint8_t pad[7];
};
static_assert(sizeof(AddRec) == 32);

struct OpRec {
    std::uint8_t kind, side;
    std::uint8_t pad[6];
    std::int64_t id, price, qty;
};
static_assert(sizeof(OpRec) == 32);

using Clock = std::chrono::steady_clock;

double percentile(const std::vector<double>& sorted, double pct) {  // nearest rank, as scripts/benchmark.py
    if (sorted.empty()) return 0.0;
    long rank = static_cast<long>(pct / 100.0 * static_cast<double>(sorted.size()) + 0.5) - 1;
    rank = std::max(0L, std::min(static_cast<long>(sorted.size()) - 1, rank));
    return sorted[static_cast<std::size_t>(rank)];
}

std::string stats_json(std::vector<double> us) {
    std::sort(us.begin(), us.end());
    double mean = 0;
    for (double x : us) mean += x;
    mean = us.empty() ? 0.0 : mean / static_cast<double>(us.size());
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "{\"n\":%zu,\"mean_us\":%.6f,\"p50_us\":%.6f,\"p95_us\":%.6f,\"p99_us\":%.6f,\"p999_us\":%.6f,\"max_us\":%.6f}",
                  us.size(), mean, percentile(us, 50), percentile(us, 95), percentile(us, 99), percentile(us, 99.9),
                  us.empty() ? 0.0 : us.back());
    return buf;
}

std::string array_json(const std::vector<double>& xs) {
    std::string s = "[";
    char buf[32];
    for (std::size_t i = 0; i < xs.size(); ++i) {
        std::snprintf(buf, sizeof buf, i ? ",%.4f" : "%.4f", xs[i]);
        s += buf;
    }
    return s + "]";
}

// The clock's own cost: back-to-back reads, sorted.
std::vector<double> timer_overhead() {
    std::vector<double> overhead(100000);
    for (auto& o : overhead) {
        const auto a = Clock::now();
        const auto b = Clock::now();
        o = std::chrono::duration<double, std::micro>(b - a).count();
    }
    std::sort(overhead.begin(), overhead.end());
    return overhead;
}

// Mixed stream: every operation timed, latencies kept per kind.
int run_ops(std::ifstream& in, const char* out_path, int passes, int window_log2) {
    std::uint64_t n_ops = 0, pad = 0;
    in.read(reinterpret_cast<char*>(&n_ops), 8);
    in.read(reinterpret_cast<char*>(&pad), 8);
    std::vector<OpRec> ops(n_ops);
    in.read(reinterpret_cast<char*>(ops.data()), static_cast<std::streamsize>(n_ops * sizeof(OpRec)));
    if (!in) {
        std::fprintf(stderr, "truncated ops file\n");
        return 2;
    }
    static const char* kNames[3] = {"limit", "market", "cancel"};
    std::string per_pass = "[";
    std::vector<double> first[3];
    std::size_t fills_first = 0, resting_first = 0, found_first = 0, far_first = 0;
    for (int pass = 0; pass < passes; ++pass) {
        ee::Config cfg;
        cfg.order_capacity = std::max<std::size_t>(n_ops / 4, 1024);
        cfg.level_capacity = cfg.order_capacity;
        cfg.far_capacity = std::max<std::size_t>(n_ops / 16, 1024);
        cfg.window_ticks = 1u << window_log2;
        cfg.prefault = true;
        ee::OrderBook book(cfg);
        std::size_t fills = 0, found = 0;
        auto on_fill = [&](const ee::FillEvent&) { ++fills; };
        std::vector<double> us[3];
        for (auto& v : us) v.reserve(n_ops);
        for (const OpRec& r : ops) {
            const ee::Side side = r.side ? ee::Side::Ask : ee::Side::Bid;
            const auto t0 = Clock::now();
            if (r.kind == 0) book.add_limit(r.id, side, r.price, r.qty, on_fill);
            else if (r.kind == 1) book.match_market(side, r.qty, on_fill);
            else found += book.cancel(r.id).found;
            const auto t1 = Clock::now();
            us[r.kind < 3 ? r.kind : 2].push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        per_pass += pass ? "," : "";
        per_pass += "{";
        for (int k = 0; k < 3; ++k) per_pass += (k ? ",\"" : "\"") + std::string(kNames[k]) + "\":" + stats_json(us[k]);
        per_pass += "}";
        if (pass == 0) {
            for (int k = 0; k < 3; ++k) first[k] = us[k];
            fills_first = fills;
            resting_first = book.size();
            found_first = found;
            far_first = book.far_levels(ee::Side::Bid) + book.far_levels(ee::Side::Ask);
        }
    }
    per_pass += "]";
    const std::vector<double> overhead = timer_overhead();
    std::FILE* out = std::fopen(out_path, "w");
    if (!out) return 2;
    std::fprintf(out, "{\"ops\":%llu,\"passes\":%d,\"window_ticks\":%u,\"fills\":%zu,\"resting_at_end\":%zu,"
                      "\"cancels_found\":%zu,\"far_levels_at_end\":%zu,\"timer\":\"std::chrono::steady_clock\","
                      "\"timer_overhead_us\":{\"p50\":%.6f,\"p99\":%.6f},"
                      "\"limit\":%s,\"market\":%s,\"cancel\":%s,\"per_pass\":%s,"
                      "\"limit_us\":%s,\"market_us\":%s,\"cancel_us\":%s}\n",
                 static_cast<unsigned long long>(n_ops), passes, 1u << window_log2, fills_first, resting_first, found_first,
                 far_first, percentile(overhead, 50), percentile(overhead, 99),
                 stats_json(first[0]).c_str(), stats_json(first[1]).c_str(), stats_json(first[2]).c_str(),
                 per_pass.c_str(), array_json(first[0]).c_str(), array_json(first[1]).c_str(), array_json(first[2]).c_str());
    std::fclose(out);
    std::printf("bench_native: %llu mixed operations, %d passes, %zu fills -> %s\n", static_cast<unsigned long long>(n_ops),
                passes, fills_first, out_path);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <stream.bin> <out.json> [--passes N] [--window-log2 K]\n", argv[0]);
        return 2;
    }
    int passes = 5;
    int window_log2 = 18;
    for (int i = 3; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--passes")) passes = std::max(1, std::atoi(argv[i + 1]));
        else if (!std::strcmp(argv[i], "--window-log2")) window_log2 = std::atoi(argv[i + 1]);
    }
    std::ifstream in(argv[1], std::ios::binary);
    char magic[8];
    std::uint32_t version = 0, pad = 0;
    std::uint64_t n_adds = 0, n_cancels = 0;
    in.read(magic, 8);
    if (in && std::memcmp(magic, "EEOPSV02", 8) == 0) return run_ops(in, argv[2], passes, window_log2);
    in.read(reinterpret_cast<char*>(&version), 4);
    in.read(reinterpret_cast<char*>(&pad), 4);
    in.read(reinterpret_cast<char*>(&n_adds), 8);
    in.read(reinterpret_cast<char*>(&n_cancels), 8);
    if (!in || std::memcmp(magic, "EESTREAM", 8) != 0 || version != 1) {
        std::fprintf(stderr, "not an EESTREAM v1 file: %s\n", argv[1]);
        return 2;
    }
    std::vector<AddRec> adds(n_adds);
    std::vector<std::int64_t> cancels(n_cancels);
    in.read(reinterpret_cast<char*>(adds.data()), static_cast<std::streamsize>(n_adds * sizeof(AddRec)));
    in.read(reinterpret_cast<char*>(cancels.data()), static_cast<std::streamsize>(n_cancels * 8));
    if (!in) {
        std::fprintf(stderr, "truncated stream file\n");
        return 2;
    }

    const std::vector<double> overhead = timer_overhead();

    std::string per_pass = "[";
    std::vector<double> first_add, first_cancel;
    std::size_t fills_total = 0, resting_after_adds = 0, cancelled = 0, far_levels = 0;
    for (int pass = 0; pass < passes; ++pass) {
        ee::Config cfg;
        cfg.order_capacity = std::max<std::size_t>(n_adds, 1024);
        cfg.level_capacity = std::max<std::size_t>(n_adds, 1024);
        cfg.far_capacity = std::max<std::size_t>(n_adds / 4, 1024);
        cfg.window_ticks = 1u << window_log2;
        cfg.prefault = true;
        ee::OrderBook book(cfg);
        std::size_t fills = 0;
        auto on_fill = [&](const ee::FillEvent&) { ++fills; };
        std::vector<double> add_us(n_adds), cancel_us(n_cancels);
        for (std::size_t i = 0; i < n_adds; ++i) {
            const AddRec& r = adds[i];
            const auto t0 = Clock::now();
            book.add_limit(r.id, r.side ? ee::Side::Ask : ee::Side::Bid, r.price, r.qty, on_fill);
            const auto t1 = Clock::now();
            add_us[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        const std::size_t resting = book.size();
        const std::size_t far = book.far_levels(ee::Side::Bid) + book.far_levels(ee::Side::Ask);
        std::size_t ok = 0;
        for (std::size_t i = 0; i < n_cancels; ++i) {
            const auto t0 = Clock::now();
            const bool found = book.cancel(cancels[i]).found;
            const auto t1 = Clock::now();
            cancel_us[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
            ok += found;
        }
        per_pass += (pass ? "," : "") + std::string("{\"add_order\":") + stats_json(add_us) + ",\"cancel_order\":" +
                    stats_json(cancel_us) + "}";
        if (pass == 0) {
            first_add = add_us;
            first_cancel = cancel_us;
            fills_total = fills;
            resting_after_adds = resting;
            cancelled = ok;
            far_levels = far;
        }
    }
    per_pass += "]";

    std::FILE* out = std::fopen(argv[2], "w");
    if (!out) {
        std::fprintf(stderr, "cannot write %s\n", argv[2]);
        return 2;
    }
    std::fprintf(out,
                 "{\"orders\":%llu,\"cancels\":%llu,\"passes\":%d,\"window_ticks\":%u,\"fills\":%zu,"
                 "\"resting_after_adds\":%zu,\"cancels_found\":%zu,\"far_levels_after_adds\":%zu,"
                 "\"timer\":\"std::chrono::steady_clock\",\"timer_overhead_us\":{\"p50\":%.6f,\"p99\":%.6f},"
                 "\"add_order\":%s,\"cancel_order\":%s,\"per_pass\":%s,\"add_order_us\":%s,\"cancel_order_us\":%s}\n",
                 static_cast<unsigned long long>(n_adds), static_cast<unsigned long long>(n_cancels), passes,
                 1u << window_log2, fills_total, resting_after_adds, cancelled, far_levels,
                 percentile(overhead, 50), percentile(overhead, 99), stats_json(first_add).c_str(),
                 stats_json(first_cancel).c_str(), per_pass.c_str(), array_json(first_add).c_str(),
                 array_json(first_cancel).c_str());
    std::fclose(out);
    std::printf("bench_native: %llu adds, %llu cancels, %d passes, %zu fills -> %s\n",
                static_cast<unsigned long long>(n_adds), static_cast<unsigned long long>(n_cancels), passes, fills_total, argv[2]);
    return 0;
}
