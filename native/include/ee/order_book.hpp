// ee/order_book.hpp -- C++20 limit order book core with price-time priority.
//
// Same matching semantics as the Python reference engine (src/exchange_engine/orderbook.py):
// an incoming limit order walks the opposite side while it crosses, fills makers in FIFO
// order at each level's price, rests any remainder (status OPEN if untouched, PARTIAL if
// partly filled) and is FILLED otherwise; a market order walks without a price limit and
// drops any remainder; a cancel removes a resting order and reports what was still open.
// Prices are integer ticks and quantities integer lots: Decimal is converted at the Python
// boundary (exchange_engine.native_book), never in here.
//
// Layout, chosen so that the steady-state hot path never touches the heap:
//   * price levels per side: a window of `window_ticks` slots indexed by (price - base), with a
//     two-level bitmap of non-empty slots so the next best level is a few word scans away;
//     prices outside the window go to an ordered fallback (a sorted vector with reserved
//     capacity). An empty window re-centres on the next level that arrives;
//   * orders: intrusive doubly linked FIFO list per level, nodes from a preallocated pool
//     with a free list; levels come from a second pool the same way;
//   * order id -> node: open addressing with linear probing and backward-shift deletion
//     (no tombstones), so cancel is constant time on average;
//   * fills are reported through a callback, so the core never builds a container per call.
// Pools, the hash table and the fallback grow (allocating) only when a configured capacity is
// exceeded; tests/test_order_book.cpp proves zero allocations for a stream within capacity.

#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <vector>

namespace ee {

using Price = std::int64_t;    // ticks
using Qty = std::int64_t;      // lots
using OrderId = std::int64_t;

enum class Side : std::uint8_t { Bid = 0, Ask = 1 };
enum class Status : std::uint8_t { Open = 0, Partial = 1, Filled = 2, Cancelled = 3 };
enum class AddError : std::uint8_t { None = 0, NonPositiveQuantity, NonPositivePrice, DuplicateId };

constexpr Side opposite(Side s) noexcept { return s == Side::Bid ? Side::Ask : Side::Bid; }

struct Level;

struct OrderNode {
    OrderId id;
    Price price;
    Qty qty;  // remaining open quantity
    Side side;
    Status status;
    OrderNode* prev;
    OrderNode* next;  // also the free-list link
    Level* level;
};

struct Level {
    Price price;
    Qty total;
    std::uint32_t count;
    OrderNode* head;
    OrderNode* tail;
    Level* next_free;
};

/// One execution against a resting order (the taker is the order being added or matched).
struct FillEvent {
    OrderId maker_id;
    Price price;
    Qty qty;
    Qty maker_remaining;
};

struct AddResult {
    AddError error;
    Status status;   // the incoming order's status afterwards
    Qty remaining;   // its open quantity afterwards (0 when filled)
    bool rested;
};

struct CancelResult {
    bool found;
    Qty remaining;
};

struct Config {
    std::size_t order_capacity = 1 << 12;
    std::size_t level_capacity = 1 << 12;
    std::size_t far_capacity = 1 << 10;
    std::uint32_t window_ticks = 1u << 18;  // power of two
    bool prefault = false;  // touch the window arrays at construction (benchmarks), instead of on first use
};

// --------------------------------------------------------------------------- //
// Fixed-capacity object pool with a free list; grows by whole chunks only past capacity.
// --------------------------------------------------------------------------- //

template <typename T, T* T::*Link>
class Pool {
public:
    explicit Pool(std::size_t capacity) { grow(std::max<std::size_t>(capacity, 64)); }

    T* get() {
        if (!free_) grow(capacity_);  // doubles; allocates, but only beyond the configured capacity
        T* t = free_;
        free_ = t->*Link;
        ++live_;
        return t;
    }

    void put(T* t) noexcept {
        t->*Link = free_;
        free_ = t;
        --live_;
    }

    std::size_t live() const noexcept { return live_; }
    std::size_t capacity() const noexcept { return capacity_; }

private:
    void grow(std::size_t n) {
        chunks_.push_back(std::make_unique<T[]>(n));
        T* c = chunks_.back().get();
        for (std::size_t i = n; i-- > 0;) {
            c[i].*Link = free_;
            free_ = &c[i];
        }
        capacity_ += n;
    }

    std::vector<std::unique_ptr<T[]>> chunks_;
    T* free_ = nullptr;
    std::size_t capacity_ = 0;
    std::size_t live_ = 0;
};

// --------------------------------------------------------------------------- //
// Order id -> node: open addressing, linear probing, backward-shift deletion.
// --------------------------------------------------------------------------- //

class IdMap {
public:
    explicit IdMap(std::size_t expected) { rehash(std::bit_ceil(std::max<std::size_t>(16, expected * 2))); }

    OrderNode* find(OrderId id) const noexcept {
        for (std::size_t i = slot(id);; i = (i + 1) & mask_) {
            OrderNode* n = nodes_[i];
            if (!n) return nullptr;
            if (n->id == id) return n;
        }
    }

    void insert(OrderNode* node) {
        if ((size_ + 1) * 2 > nodes_.size()) rehash(nodes_.size() * 2);  // load factor stays <= 1/2
        std::size_t i = slot(node->id);
        while (nodes_[i]) i = (i + 1) & mask_;
        nodes_[i] = node;
        ++size_;
    }

    void erase(OrderId id) noexcept {
        std::size_t i = slot(id);
        while (nodes_[i] && nodes_[i]->id != id) i = (i + 1) & mask_;
        if (!nodes_[i]) return;
        // Backward shift: pull later entries of the same probe run into the hole.
        std::size_t hole = i;
        for (std::size_t j = (i + 1) & mask_; nodes_[j]; j = (j + 1) & mask_) {
            const std::size_t home = slot(nodes_[j]->id);
            const bool movable = hole <= j ? (home <= hole || home > j) : (home <= hole && home > j);
            if (movable) {
                nodes_[hole] = nodes_[j];
                hole = j;
            }
        }
        nodes_[hole] = nullptr;
        --size_;
    }

    std::size_t size() const noexcept { return size_; }

private:
    std::size_t slot(OrderId id) const noexcept {
        return static_cast<std::size_t>((static_cast<std::uint64_t>(id) * 0x9E3779B97F4A7C15ull) >> shift_);
    }

    void rehash(std::size_t capacity) {
        std::vector<OrderNode*> old;
        old.swap(nodes_);
        nodes_.assign(capacity, nullptr);
        mask_ = capacity - 1;
        shift_ = 64 - std::countr_zero(capacity);
        size_ = 0;
        for (OrderNode* n : old)
            if (n) insert(n);
    }

    std::vector<OrderNode*> nodes_;
    std::size_t mask_ = 0;
    int shift_ = 64;
    std::size_t size_ = 0;
};

// --------------------------------------------------------------------------- //
// Two-level bitmap over the window slots.
// --------------------------------------------------------------------------- //

class Bitmap {
public:
    explicit Bitmap(std::uint32_t bits)
        : words_((bits + 63) / 64), l0_(static_cast<std::uint64_t*>(std::calloc(words_, 8))),
          l1_(static_cast<std::uint64_t*>(std::calloc((words_ + 63) / 64, 8))) {
        if (!l0_ || !l1_) throw std::bad_alloc();
    }
    ~Bitmap() {
        std::free(l0_);
        std::free(l1_);
    }
    Bitmap(const Bitmap&) = delete;
    Bitmap& operator=(const Bitmap&) = delete;

    void set(std::uint32_t i) noexcept {
        l0_[i >> 6] |= 1ull << (i & 63);
        l1_[i >> 12] |= 1ull << ((i >> 6) & 63);
    }

    void clear(std::uint32_t i) noexcept {
        std::uint64_t& w = l0_[i >> 6];
        w &= ~(1ull << (i & 63));
        if (!w) l1_[i >> 12] &= ~(1ull << ((i >> 6) & 63));
    }

    void reset() noexcept {
        std::memset(l0_, 0, words_ * 8);
        std::memset(l1_, 0, ((words_ + 63) / 64) * 8);
    }

    /// Highest set bit <= i, or -1.
    std::int64_t prev(std::uint32_t i) const noexcept {
        std::uint32_t w = i >> 6;
        std::uint64_t bits = l0_[w] & (~0ull >> (63 - (i & 63)));
        if (bits) return (std::int64_t(w) << 6) + 63 - std::countl_zero(bits);
        if (w == 0) return -1;
        std::uint32_t wi = w - 1;  // search the summary for the highest non-empty word < w
        std::uint32_t s = wi >> 6;
        std::uint64_t sb = l1_[s] & (~0ull >> (63 - (wi & 63)));
        while (!sb) {
            if (s == 0) return -1;
            sb = l1_[--s];
        }
        const std::uint32_t word = (s << 6) + 63 - std::countl_zero(sb);
        return (std::int64_t(word) << 6) + 63 - std::countl_zero(l0_[word]);
    }

    /// Lowest set bit >= i, or -1.
    std::int64_t next(std::uint32_t i) const noexcept {
        std::uint32_t w = i >> 6;
        if (w >= words_) return -1;
        std::uint64_t bits = l0_[w] & (~0ull << (i & 63));
        if (bits) return (std::int64_t(w) << 6) + std::countr_zero(bits);
        std::uint32_t wi = w + 1;
        if (wi >= words_) return -1;
        std::uint32_t s = wi >> 6;
        const std::uint32_t last = (words_ - 1) >> 6;
        std::uint64_t sb = l1_[s] & (~0ull << (wi & 63));
        while (!sb) {
            if (s == last) return -1;
            sb = l1_[++s];
        }
        const std::uint32_t word = (s << 6) + std::countr_zero(sb);
        return (std::int64_t(word) << 6) + std::countr_zero(l0_[word]);
    }

private:
    std::uint32_t words_;
    std::uint64_t* l0_;
    std::uint64_t* l1_;
};

// --------------------------------------------------------------------------- //
// One side's price levels: the tick window plus the ordered fallback.
// --------------------------------------------------------------------------- //

class SideLevels {
public:
    SideLevels(Side side, std::uint32_t window_ticks, std::size_t far_capacity, bool prefault)
        : side_(side), w_(window_ticks), slots_(static_cast<Level**>(std::calloc(window_ticks, sizeof(Level*)))),
          bits_(window_ticks) {
        if (!slots_) throw std::bad_alloc();
        assert(std::has_single_bit(window_ticks));
        far_.reserve(far_capacity);
        if (prefault) clear();  // calloc'd pages are mapped lazily; writing zeros faults them in now
    }
    ~SideLevels() { std::free(slots_); }
    SideLevels(const SideLevels&) = delete;
    SideLevels& operator=(const SideLevels&) = delete;

    bool better(Price a, Price b) const noexcept { return side_ == Side::Bid ? a > b : a < b; }

    Level* best() const noexcept { return best_; }
    bool empty() const noexcept { return best_ == nullptr; }

    Level* find(Price p) const noexcept {
        if (in_window(p)) return slots_[p - base_];
        auto it = far_lower_bound(p);
        return (it != far_.end() && (*it)->price == p) ? *it : nullptr;
    }

    void insert(Level* lvl) {
        const Price p = lvl->price;
        if (!in_window(p) && window_count_ == 0) recentre(p);
        if (in_window(p)) {
            slots_[p - base_] = lvl;
            bits_.set(static_cast<std::uint32_t>(p - base_));
            ++window_count_;
        } else {
            far_.insert(far_lower_bound(p), lvl);
        }
        if (!best_ || better(p, best_->price)) best_ = lvl;
    }

    void erase(Level* lvl) noexcept {
        const Price p = lvl->price;
        if (in_window(p)) {
            slots_[p - base_] = nullptr;
            bits_.clear(static_cast<std::uint32_t>(p - base_));
            --window_count_;
        } else {
            far_.erase(far_lower_bound(p));
        }
        if (best_ == lvl) best_ = scan_best(p);
        if (window_count_ == 0 && !far_.empty()) recentre(best_->price);
    }

    /// Visit up to n levels best first: f(const Level&).
    template <typename F>
    void visit(std::size_t n, F&& f) const {
        if (!best_) return;
        // Merge the window (scanned through the bitmap) with the fallback, best first.
        std::size_t fi = 0;  // index into far_ in best-first order
        const std::size_t nf = far_.size();
        auto far_at = [&](std::size_t k) -> Level* { return side_ == Side::Bid ? far_[nf - 1 - k] : far_[k]; };
        std::int64_t wi = window_count_ ? first_window_index() : -1;
        while (n > 0 && (wi >= 0 || fi < nf)) {
            Level* a = wi >= 0 ? slots_[wi] : nullptr;
            Level* b = fi < nf ? far_at(fi) : nullptr;
            Level* take = (a && (!b || better(a->price, b->price))) ? a : b;
            f(*take);
            --n;
            if (take == a) wi = next_window_index(wi);
            else ++fi;
        }
    }

    std::size_t levels() const noexcept { return window_count_ + far_.size(); }
    std::size_t far_levels() const noexcept { return far_.size(); }

    void clear() noexcept {
        bits_.reset();
        std::memset(slots_, 0, w_ * sizeof(Level*));
        far_.clear();
        window_count_ = 0;
        best_ = nullptr;
    }

private:
    bool in_window(Price p) const noexcept { return p >= base_ && p - base_ < static_cast<Price>(w_); }

    // far_ is sorted ascending by price.
    std::vector<Level*>::iterator far_lower_bound(Price p) {
        return std::lower_bound(far_.begin(), far_.end(), p, [](const Level* l, Price v) { return l->price < v; });
    }
    std::vector<Level*>::const_iterator far_lower_bound(Price p) const {
        return std::lower_bound(far_.begin(), far_.end(), p, [](const Level* l, Price v) { return l->price < v; });
    }

    std::int64_t first_window_index() const noexcept {
        return side_ == Side::Bid ? bits_.prev(w_ - 1) : bits_.next(0);
    }
    std::int64_t next_window_index(std::int64_t i) const noexcept {
        if (side_ == Side::Bid) return i == 0 ? -1 : bits_.prev(static_cast<std::uint32_t>(i - 1));
        return i + 1 >= static_cast<std::int64_t>(w_) ? -1 : bits_.next(static_cast<std::uint32_t>(i + 1));
    }

    /// The best remaining level after the level at `gone` was removed.
    Level* scan_best(Price gone) const noexcept {
        Level* win = nullptr;
        if (window_count_) {
            std::int64_t i;
            if (in_window(gone)) {
                const auto g = static_cast<std::uint32_t>(gone - base_);
                i = side_ == Side::Bid ? bits_.prev(g) : bits_.next(g);  // the slot at g is already clear
                if (i < 0) i = first_window_index();
            } else {
                i = first_window_index();
            }
            win = i >= 0 ? slots_[i] : nullptr;
        }
        Level* far = far_.empty() ? nullptr : (side_ == Side::Bid ? far_.back() : far_.front());
        if (!win) return far;
        if (!far) return win;
        return better(win->price, far->price) ? win : far;
    }

    /// Move the window so that it is centred on `p`, migrating fallback levels into it.
    void recentre(Price p) {
        base_ = p - static_cast<Price>(w_ / 2);
        auto lo = far_lower_bound(base_);
        auto hi = far_lower_bound(base_ + static_cast<Price>(w_));
        for (auto it = lo; it != hi; ++it) {
            slots_[(*it)->price - base_] = *it;
            bits_.set(static_cast<std::uint32_t>((*it)->price - base_));
            ++window_count_;
        }
        far_.erase(lo, hi);
    }

    Side side_;
    std::uint32_t w_;
    Price base_ = 0;
    Level** slots_;
    Bitmap bits_;
    std::size_t window_count_ = 0;
    std::vector<Level*> far_;
    Level* best_ = nullptr;
};

// --------------------------------------------------------------------------- //
// The book
// --------------------------------------------------------------------------- //

class OrderBook {
public:
    explicit OrderBook(const Config& cfg = Config{})
        : orders_(cfg.order_capacity), levels_(cfg.level_capacity), ids_(cfg.order_capacity),
          bids_(Side::Bid, cfg.window_ticks, cfg.far_capacity, cfg.prefault),
          asks_(Side::Ask, cfg.window_ticks, cfg.far_capacity, cfg.prefault) {}

    /// Add a limit order: match while it crosses, rest any remainder. Calls on_fill per fill.
    template <typename OnFill>
    AddResult add_limit(OrderId id, Side side, Price price, Qty qty, OnFill&& on_fill) {
        if (qty <= 0) return {AddError::NonPositiveQuantity, Status::Open, qty, false};
        if (price <= 0) return {AddError::NonPositivePrice, Status::Open, qty, false};
        if (ids_.find(id)) return {AddError::DuplicateId, Status::Open, qty, false};
        const Qty remaining = consume(side, qty, true, price, on_fill);
        if (remaining == 0) return {AddError::None, Status::Filled, 0, false};
        const Status st = remaining < qty ? Status::Partial : Status::Open;
        rest(id, side, price, remaining, st);
        return {AddError::None, st, remaining, true};
    }

    /// Match a market order; any unfilled remainder is dropped. Returns that remainder.
    template <typename OnFill>
    Qty match_market(Side side, Qty qty, OnFill&& on_fill) {
        if (qty <= 0) return qty;
        return consume(side, qty, false, 0, on_fill);
    }

    CancelResult cancel(OrderId id) noexcept {
        OrderNode* n = ids_.find(id);
        if (!n) return {false, 0};
        const Qty remaining = n->qty;
        Level* lvl = n->level;
        lvl->total -= remaining;
        unlink(lvl, n);
        ids_.erase(id);
        n->status = Status::Cancelled;
        orders_.put(n);
        if (!lvl->head) drop_level(n->side, lvl);
        return {true, remaining};
    }

    std::optional<Price> best_bid() const noexcept { return bids_.empty() ? std::nullopt : std::optional<Price>(bids_.best()->price); }
    std::optional<Price> best_ask() const noexcept { return asks_.empty() ? std::nullopt : std::optional<Price>(asks_.best()->price); }

    /// Visit up to n aggregated levels of `side`, best first: f(price, total quantity).
    template <typename F>
    void depth(Side side, std::size_t n, F&& f) const {
        book(side).visit(n, [&](const Level& l) { f(l.price, l.total); });
    }

    const OrderNode* find(OrderId id) const noexcept { return ids_.find(id); }
    std::size_t size() const noexcept { return ids_.size(); }
    std::size_t levels(Side side) const noexcept { return book(side).levels(); }
    std::size_t far_levels(Side side) const noexcept { return book(side).far_levels(); }
    std::size_t order_capacity() const noexcept { return orders_.capacity(); }

private:
    using OrderPool = Pool<OrderNode, &OrderNode::next>;
    using LevelPool = Pool<Level, &Level::next_free>;

    SideLevels& book(Side s) noexcept { return s == Side::Bid ? bids_ : asks_; }
    const SideLevels& book(Side s) const noexcept { return s == Side::Bid ? bids_ : asks_; }

    template <typename OnFill>
    Qty consume(Side taker, Qty qty, bool limited, Price limit, OnFill& on_fill) {
        SideLevels& opp = book(opposite(taker));
        Qty remaining = qty;
        while (remaining > 0 && !opp.empty()) {
            Level* lvl = opp.best();
            if (limited && (taker == Side::Bid ? lvl->price > limit : lvl->price < limit)) break;
            while (lvl->head && remaining > 0) {
                OrderNode* m = lvl->head;
                const Qty traded = std::min(remaining, m->qty);
                m->qty -= traded;
                remaining -= traded;
                lvl->total -= traded;
                on_fill(FillEvent{m->id, lvl->price, traded, m->qty});
                if (m->qty == 0) {
                    m->status = Status::Filled;
                    unlink(lvl, m);
                    ids_.erase(m->id);
                    orders_.put(m);
                } else {
                    m->status = Status::Partial;
                }
            }
            if (!lvl->head) drop_level(opposite(taker), lvl);
        }
        return remaining;
    }

    void rest(OrderId id, Side side, Price price, Qty qty, Status st) {
        SideLevels& sl = book(side);
        Level* lvl = sl.find(price);
        if (!lvl) {
            lvl = levels_.get();
            *lvl = Level{price, 0, 0, nullptr, nullptr, nullptr};
            sl.insert(lvl);
        }
        OrderNode* n = orders_.get();
        *n = OrderNode{id, price, qty, side, st, lvl->tail, nullptr, lvl};
        if (lvl->tail) lvl->tail->next = n;
        else lvl->head = n;
        lvl->tail = n;
        lvl->total += qty;
        ++lvl->count;
        ids_.insert(n);
    }

    static void unlink(Level* lvl, OrderNode* n) noexcept {
        if (n->prev) n->prev->next = n->next;
        else lvl->head = n->next;
        if (n->next) n->next->prev = n->prev;
        else lvl->tail = n->prev;
        --lvl->count;
    }

    void drop_level(Side side, Level* lvl) noexcept {
        book(side).erase(lvl);
        levels_.put(lvl);
    }

    OrderPool orders_;
    LevelPool levels_;
    IdMap ids_;
    SideLevels bids_;
    SideLevels asks_;
};

}  // namespace ee
