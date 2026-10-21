// Native tests for ee::OrderBook: building blocks, matching rules, and a randomized
// differential test against a deliberately simple reference book (std::map + std::deque).
// Run through ctest; exits non-zero on the first failed check.

#include "ee/order_book.hpp"

#include <cstdio>
#include <deque>
#include <map>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

int g_checks = 0;

#define CHECK(cond)                                                                   \
    do {                                                                              \
        ++g_checks;                                                                   \
        if (!(cond)) {                                                                \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

using ee::Side;
using ee::Status;

struct Fill {
    ee::OrderId maker;
    ee::Price price;
    ee::Qty qty;
    ee::Qty maker_remaining;
    bool operator==(const Fill&) const = default;
};

std::vector<Fill> add(ee::OrderBook& b, ee::OrderId id, Side s, ee::Price p, ee::Qty q, ee::AddResult* out = nullptr) {
    std::vector<Fill> fills;
    auto r = b.add_limit(id, s, p, q, [&](const ee::FillEvent& f) { fills.push_back({f.maker_id, f.price, f.qty, f.maker_remaining}); });
    if (out) *out = r;
    return fills;
}

std::vector<Fill> market(ee::OrderBook& b, Side s, ee::Qty q) {
    std::vector<Fill> fills;
    b.match_market(s, q, [&](const ee::FillEvent& f) { fills.push_back({f.maker_id, f.price, f.qty, f.maker_remaining}); });
    return fills;
}

std::vector<std::pair<ee::Price, ee::Qty>> depth(const ee::OrderBook& b, Side s, std::size_t n) {
    std::vector<std::pair<ee::Price, ee::Qty>> out;
    b.depth(s, n, [&](ee::Price p, ee::Qty q) { out.emplace_back(p, q); });
    return out;
}

// ---------------------------------------------------------------- building blocks

void test_bitmap() {
    std::mt19937_64 rng(1);
    for (std::uint32_t bits : {64u, 4096u, 1u << 14}) {
        ee::Bitmap bm(bits);
        std::set<std::uint32_t> ref;
        for (int step = 0; step < 20000; ++step) {
            const std::uint32_t i = rng() % bits;
            if (rng() & 1) { bm.set(i); ref.insert(i); }
            else { bm.clear(i); ref.erase(i); }
            const std::uint32_t q = rng() % bits;
            auto up = ref.lower_bound(q);
            CHECK(bm.next(q) == (up == ref.end() ? -1 : std::int64_t(*up)));
            auto dn = ref.upper_bound(q);
            CHECK(bm.prev(q) == (dn == ref.begin() ? -1 : std::int64_t(*std::prev(dn))));
        }
    }
}

void test_id_map() {
    std::mt19937_64 rng(2);
    std::vector<ee::OrderNode> nodes(3000);
    ee::IdMap map(16);  // tiny on purpose: forces rehashes and long probe runs
    std::unordered_map<ee::OrderId, ee::OrderNode*> ref;
    for (int step = 0; step < 50000; ++step) {
        const ee::OrderId id = static_cast<ee::OrderId>(rng() % 4000) - 1000;  // negative ids too
        if (rng() % 3 && !ref.count(id) && ref.size() < nodes.size()) {
            ee::OrderNode* n = &nodes[ref.size()];
            // reuse a node slot not currently referenced
            for (auto& cand : nodes) {
                bool used = false;
                for (auto& kv : ref) used |= kv.second == &cand;
                if (!used) { n = &cand; break; }
            }
            n->id = id;
            map.insert(n);
            ref[id] = n;
        } else if (ref.count(id)) {
            map.erase(id);
            ref.erase(id);
        }
        CHECK(map.size() == ref.size());
        const ee::OrderId q = static_cast<ee::OrderId>(rng() % 4000) - 1000;
        auto it = ref.find(q);
        CHECK(map.find(q) == (it == ref.end() ? nullptr : it->second));
        if (step > 2000) break;  // the node-reuse scan above is quadratic; 2,000 steps cover it
    }
}

// ---------------------------------------------------------------- matching rules

void test_rules() {
    ee::OrderBook b;
    ee::AddResult r{};
    CHECK(add(b, 1, Side::Bid, 100, 5, &r).empty() && r.status == Status::Open && r.rested);
    CHECK(add(b, 2, Side::Bid, 100, 3).empty());
    CHECK(add(b, 3, Side::Bid, 99, 4).empty());
    CHECK(b.best_bid() == 100 && !b.best_ask());
    CHECK((depth(b, Side::Bid, 5) == std::vector<std::pair<ee::Price, ee::Qty>>{{100, 8}, {99, 4}}));
    // price-time priority: order 1 then order 2 at 100, then 99; partial fill of order 3
    auto f = add(b, 4, Side::Ask, 99, 10, &r);
    CHECK((f == std::vector<Fill>{{1, 100, 5, 0}, {2, 100, 3, 0}, {3, 99, 2, 2}}));
    CHECK(r.status == Status::Filled && r.remaining == 0 && !r.rested);
    CHECK(b.find(3)->status == Status::Partial && b.find(3)->qty == 2 && !b.find(1));
    // limit stops at its price and rests the remainder as PARTIAL
    f = add(b, 5, Side::Ask, 99, 5, &r);
    CHECK((f == std::vector<Fill>{{3, 99, 2, 0}}) && r.status == Status::Partial && r.remaining == 3);
    CHECK(b.best_ask() == 99 && !b.best_bid() && b.size() == 1);
    // duplicate resting id is rejected; validation errors come first
    add(b, 5, Side::Bid, 10, 1, &r);
    CHECK(r.error == ee::AddError::DuplicateId);
    add(b, 5, Side::Bid, 10, 0, &r);
    CHECK(r.error == ee::AddError::NonPositiveQuantity);
    add(b, 6, Side::Bid, 0, 1, &r);
    CHECK(r.error == ee::AddError::NonPositivePrice);
    // cancel
    CHECK(b.cancel(5).found && b.cancel(5).found == false && b.size() == 0 && !b.best_ask());
    // market sweep drops the remainder
    add(b, 7, Side::Ask, 101, 1);
    add(b, 8, Side::Ask, 105, 2);
    f = market(b, Side::Bid, 10);
    CHECK((f == std::vector<Fill>{{7, 101, 1, 0}, {8, 105, 2, 0}}) && b.size() == 0);
    // a filled id may be reused
    add(b, 7, Side::Bid, 50, 1, &r);
    CHECK(r.error == ee::AddError::None && b.size() == 1);
}

void test_far_prices_and_recentre() {
    ee::Config cfg;
    cfg.window_ticks = 64;
    ee::OrderBook b(cfg);
    // first level centres the window on 1000; 5000 and 10 land in the fallback
    add(b, 1, Side::Bid, 1000, 1);
    add(b, 2, Side::Bid, 5000, 1);
    add(b, 3, Side::Bid, 10, 1);
    add(b, 4, Side::Bid, 1001, 1);
    CHECK(b.best_bid() == 5000 && b.far_levels(Side::Bid) == 2);
    CHECK((depth(b, Side::Bid, 10) == std::vector<std::pair<ee::Price, ee::Qty>>{{5000, 1}, {1001, 1}, {1000, 1}, {10, 1}}));
    b.cancel(2);
    CHECK(b.best_bid() == 1001);
    b.cancel(4);
    b.cancel(1);  // window empties: it re-centres on the fallback's best, 10
    CHECK(b.best_bid() == 10 && b.far_levels(Side::Bid) == 0 && b.levels(Side::Bid) == 1);
    add(b, 5, Side::Ask, 12, 3);
    auto f = add(b, 6, Side::Bid, 15, 4);
    CHECK((f == std::vector<Fill>{{5, 12, 3, 0}}) && b.best_bid() == 15);
}

// ---------------------------------------------------------------- differential test

struct RefBook {
    struct O { ee::OrderId id; ee::Qty qty; };
    std::map<ee::Price, std::deque<O>, std::greater<>> bids;
    std::map<ee::Price, std::deque<O>> asks;
    std::unordered_map<ee::OrderId, std::pair<Side, ee::Price>> live;

    template <typename M>
    void consume(M& opp, Side taker, ee::Qty& rem, bool limited, ee::Price limit, std::vector<Fill>& out) {
        while (rem > 0 && !opp.empty()) {
            auto it = opp.begin();
            if (limited && (taker == Side::Bid ? it->first > limit : it->first < limit)) break;
            auto& q = it->second;
            while (!q.empty() && rem > 0) {
                const ee::Qty t = std::min(rem, q.front().qty);
                q.front().qty -= t;
                rem -= t;
                out.push_back({q.front().id, it->first, t, q.front().qty});
                if (q.front().qty == 0) { live.erase(q.front().id); q.pop_front(); }
            }
            if (q.empty()) opp.erase(it);
        }
    }

    std::vector<Fill> add(ee::OrderId id, Side s, ee::Price p, ee::Qty q, bool& rejected) {
        std::vector<Fill> out;
        rejected = live.count(id) != 0;
        if (rejected) return out;
        ee::Qty rem = q;
        if (s == Side::Bid) consume(asks, s, rem, true, p, out);
        else consume(bids, s, rem, true, p, out);
        if (rem > 0) {
            if (s == Side::Bid) bids[p].push_back({id, rem});
            else asks[p].push_back({id, rem});
            live[id] = {s, p};
        }
        return out;
    }

    std::vector<Fill> market(Side s, ee::Qty q) {
        std::vector<Fill> out;
        ee::Qty rem = q;
        if (s == Side::Bid) consume(asks, s, rem, false, 0, out);
        else consume(bids, s, rem, false, 0, out);
        return out;
    }

    bool cancel(ee::OrderId id) {
        auto it = live.find(id);
        if (it == live.end()) return false;
        auto erase_from = [&](auto& side) {
            auto& q = side[it->second.second];
            for (auto o = q.begin(); o != q.end(); ++o)
                if (o->id == id) { q.erase(o); break; }
            if (q.empty()) side.erase(it->second.second);
        };
        if (it->second.first == Side::Bid) erase_from(bids);
        else erase_from(asks);
        live.erase(it);
        return true;
    }

    template <typename M>
    static std::vector<std::pair<ee::Price, ee::Qty>> depth_of(const M& m, std::size_t n) {
        std::vector<std::pair<ee::Price, ee::Qty>> out;
        for (auto& [p, q] : m) {
            if (out.size() == n) break;
            ee::Qty t = 0;
            for (auto& o : q) t += o.qty;
            out.emplace_back(p, t);
        }
        return out;
    }
};

void test_differential(std::uint32_t window, int streams) {
    for (int seed = 0; seed < streams; ++seed) {
        std::mt19937_64 rng(1000 + seed);
        ee::Config cfg;
        cfg.window_ticks = window;
        cfg.order_capacity = 64;  // small: pools, the id map and the fallback all grow during the run
        cfg.level_capacity = 64;
        cfg.far_capacity = 4;
        ee::OrderBook b(cfg);
        RefBook ref;
        ee::Price mid = 100000;
        ee::OrderId next_id = 1;
        std::vector<ee::OrderId> ids;
        for (int step = 0; step < 600; ++step) {
            if (rng() % 50 == 0) mid += static_cast<ee::Price>(rng() % 400) - 200;  // the market moves
            const unsigned r = rng() % 100;
            const Side s = (rng() & 1) ? Side::Bid : Side::Ask;
            const ee::Qty q = 1 + static_cast<ee::Qty>(rng() % 9);
            if (r < 55 || ids.empty()) {
                const ee::Price p = mid + (rng() % 20 == 0 ? static_cast<ee::Price>(rng() % 2000) - 1000
                                                           : static_cast<ee::Price>(rng() % 25) - 12);
                const ee::OrderId id = (r < 50 || ids.empty()) ? next_id++ : ids[rng() % ids.size()];
                ids.push_back(id);
                bool rejected = false;
                auto want = ref.add(id, s, p, q, rejected);
                ee::AddResult got_r{};
                auto got = add(b, id, s, p, q, &got_r);
                CHECK((got_r.error == ee::AddError::DuplicateId) == rejected);
                CHECK(got == want);
            } else if (r < 75) {
                CHECK(market(b, s, q * 3) == ref.market(s, q * 3));
            } else {
                const ee::OrderId id = r < 95 ? ids[rng() % ids.size()] : 999999;
                CHECK(b.cancel(id).found == ref.cancel(id));
            }
            CHECK(b.size() == ref.live.size());
            CHECK(b.best_bid() == (ref.bids.empty() ? std::nullopt : std::optional<ee::Price>(ref.bids.begin()->first)));
            CHECK(b.best_ask() == (ref.asks.empty() ? std::nullopt : std::optional<ee::Price>(ref.asks.begin()->first)));
            CHECK(depth(b, Side::Bid, 8) == RefBook::depth_of(ref.bids, 8));
            CHECK(depth(b, Side::Ask, 8) == RefBook::depth_of(ref.asks, 8));
        }
    }
}

}  // namespace

int main() {
    test_bitmap();
    test_id_map();
    test_rules();
    test_far_prices_and_recentre();
    test_differential(64, 300);       // tiny window: most levels in the fallback, constant re-centring
    test_differential(1u << 12, 300);  // normal: nearly everything in the window
    std::printf("test_order_book: %d checks passed\n", g_checks);
    return 0;
}
