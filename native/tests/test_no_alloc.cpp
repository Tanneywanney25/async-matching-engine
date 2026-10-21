// Proves the hot path never touches the heap: global operator new/delete are replaced with
// counting versions, a book is built with enough capacity, and a 200,000-operation stream of
// limit orders (near the mid and far from it), market sweeps and cancels must run with zero
// allocations and zero deallocations. A control check confirms the counter sees allocations.

#include "ee/order_book.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <random>
#include <vector>

namespace {
std::atomic<long> g_news{0};
std::atomic<long> g_deletes{0};
}  // namespace

// Kept out of line: if GCC inlines a replacement operator new it sees the malloc inside and
// reports every matching delete as mismatched (-Wmismatched-new-delete), a false positive.
#if defined(_MSC_VER)
#define EE_NOINLINE __declspec(noinline)
#else
#define EE_NOINLINE [[gnu::noinline]]
#endif

EE_NOINLINE void* operator new(std::size_t n) {
    ++g_news;
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
EE_NOINLINE void* operator new[](std::size_t n) {
    ++g_news;
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
EE_NOINLINE void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    ++g_news;
    return std::malloc(n ? n : 1);
}
EE_NOINLINE void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    ++g_news;
    return std::malloc(n ? n : 1);
}
EE_NOINLINE void operator delete(void* p) noexcept {
    if (p) ++g_deletes;
    std::free(p);
}
EE_NOINLINE void operator delete[](void* p) noexcept {
    if (p) ++g_deletes;
    std::free(p);
}
EE_NOINLINE void operator delete(void* p, std::size_t) noexcept {
    if (p) ++g_deletes;
    std::free(p);
}
EE_NOINLINE void operator delete[](void* p, std::size_t) noexcept {
    if (p) ++g_deletes;
    std::free(p);
}

int main() {
    {  // control: the counter does see an allocation
        const long before = g_news.load();
        auto* v = new std::vector<int>(100);
        delete v;
        if (g_news.load() - before < 1) {
            std::fprintf(stderr, "counting allocator is not active\n");
            return 1;
        }
    }
    constexpr int kOps = 200000;
    ee::Config cfg;
    cfg.order_capacity = 1 << 17;  // more than can rest at once in this stream
    cfg.level_capacity = 1 << 16;
    cfg.far_capacity = 1 << 12;
    ee::OrderBook book(cfg);
    std::mt19937_64 rng(7);
    // Draw the whole stream first so that generating it is outside the counted region.
    struct Op { int kind; ee::OrderId id; ee::Side side; ee::Price price; ee::Qty qty; };
    std::vector<Op> ops;
    ops.reserve(kOps);
    std::vector<ee::OrderId> live;
    live.reserve(kOps);
    ee::Price mid = 6'400'000;
    for (int i = 0; i < kOps; ++i) {
        if (i % 1000 == 0) mid += static_cast<ee::Price>(rng() % 2001) - 1000;
        const unsigned r = rng() % 100;
        const ee::Side s = (rng() & 1) ? ee::Side::Bid : ee::Side::Ask;
        if (r < 70 || live.empty()) {
            const ee::Price off = (rng() % 50 == 0) ? static_cast<ee::Price>(rng() % 600000) - 300000  // beyond the window
                                                    : static_cast<ee::Price>(rng() % 201) - 100;
            ops.push_back({0, i + 1, s, mid + off, 1 + static_cast<ee::Qty>(rng() % 200000000)});
            live.push_back(i + 1);
        } else if (r < 80) {
            ops.push_back({1, 0, s, 0, 1 + static_cast<ee::Qty>(rng() % 400000000)});
        } else {
            ops.push_back({2, live[rng() % live.size()], s, 0, 0});
        }
    }
    long fills = 0;
    auto on_fill = [&](const ee::FillEvent&) { ++fills; };
    const long news0 = g_news.load(), deletes0 = g_deletes.load();
    for (const Op& op : ops) {
        if (op.kind == 0) book.add_limit(op.id, op.side, op.price, op.qty, on_fill);
        else if (op.kind == 1) book.match_market(op.side, op.qty, on_fill);
        else book.cancel(op.id);
    }
    const long news = g_news.load() - news0, deletes = g_deletes.load() - deletes0;
    std::printf("test_no_alloc: %d operations, %ld fills, %zu resting, %zu far bid levels: %ld allocations, %ld deallocations\n",
                kOps, fills, book.size(), book.far_levels(ee::Side::Bid), news, deletes);
    if (news != 0 || deletes != 0) {
        std::fprintf(stderr, "FAIL: the hot path allocated\n");
        return 1;
    }
    if (fills == 0 || book.far_levels(ee::Side::Bid) + book.far_levels(ee::Side::Ask) == 0) {
        std::fprintf(stderr, "FAIL: the stream did not exercise matching and the far-price fallback\n");
        return 1;
    }
    return 0;
}
