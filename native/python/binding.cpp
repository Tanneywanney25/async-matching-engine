// pybind11 binding for ee::OrderBook: integers in, integers out (ticks, lots, order ids).
// Decimal conversion and the Order/Fill objects live in exchange_engine.native_book, which
// gives this core the same interface as exchange_engine.orderbook.OrderBook.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ee/order_book.hpp"

namespace py = pybind11;

namespace {

ee::Side to_side(int s) {
    if (s != 0 && s != 1) throw py::value_error("side must be 0 (bid) or 1 (ask)");
    return s == 0 ? ee::Side::Bid : ee::Side::Ask;
}

class Book {
public:
    Book(std::uint32_t window_ticks, std::size_t order_capacity, bool prefault)
        : book_(config(window_ticks, order_capacity, prefault)) {}

    // (status, remaining lots, [(maker id, price ticks, lots, maker remaining lots), ...])
    py::tuple add_limit(ee::OrderId id, int side, ee::Price price, ee::Qty qty) {
        py::list fills;
        const ee::AddResult r = book_.add_limit(id, to_side(side), price, qty, [&](const ee::FillEvent& f) {
            fills.append(py::make_tuple(f.maker_id, f.price, f.qty, f.maker_remaining));
        });
        switch (r.error) {
            case ee::AddError::None: break;
            case ee::AddError::NonPositiveQuantity: throw py::value_error("order quantity must be positive");
            case ee::AddError::NonPositivePrice: throw py::value_error("limit price must be positive");
            case ee::AddError::DuplicateId: throw py::value_error("duplicate resting order id: " + std::to_string(id));
        }
        return py::make_tuple(static_cast<int>(r.status), r.remaining, fills);
    }

    // Timing entry point for the binding benchmark: the same call, but fills are only counted,
    // so what is measured is the core plus the Python call, without building result objects.
    int add_limit_count(ee::OrderId id, int side, ee::Price price, ee::Qty qty) {
        int n = 0;
        const ee::AddResult r = book_.add_limit(id, to_side(side), price, qty, [&](const ee::FillEvent&) { ++n; });
        if (r.error != ee::AddError::None) throw py::value_error("add_limit rejected the order");
        return n;
    }

    py::list match_market(int side, ee::Qty qty) {
        if (qty <= 0) throw py::value_error("market order quantity must be positive");
        py::list fills;
        book_.match_market(to_side(side), qty, [&](const ee::FillEvent& f) {
            fills.append(py::make_tuple(f.maker_id, f.price, f.qty, f.maker_remaining));
        });
        return fills;
    }

    bool cancel(ee::OrderId id) { return book_.cancel(id).found; }

    std::optional<ee::Price> best_bid() const { return book_.best_bid(); }
    std::optional<ee::Price> best_ask() const { return book_.best_ask(); }

    std::vector<std::pair<ee::Price, ee::Qty>> depth(int side, std::size_t n) const {
        std::vector<std::pair<ee::Price, ee::Qty>> out;
        out.reserve(n);
        book_.depth(to_side(side), n, [&](ee::Price p, ee::Qty q) { out.emplace_back(p, q); });
        return out;
    }

    bool contains(ee::OrderId id) const { return book_.find(id) != nullptr; }
    std::size_t size() const { return book_.size(); }
    std::size_t far_levels(int side) const { return book_.far_levels(to_side(side)); }

private:
    static ee::Config config(std::uint32_t window_ticks, std::size_t order_capacity, bool prefault) {
        if (window_ticks == 0 || (window_ticks & (window_ticks - 1)) != 0)
            throw py::value_error("window_ticks must be a power of two");
        ee::Config cfg;
        cfg.window_ticks = window_ticks;
        cfg.order_capacity = order_capacity;
        cfg.level_capacity = order_capacity;
        cfg.far_capacity = std::max<std::size_t>(order_capacity / 4, 64);
        cfg.prefault = prefault;
        return cfg;
    }

    ee::OrderBook book_;
};

}  // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "C++20 order book core (ee::OrderBook); see exchange_engine.native_book for the Decimal interface";
    m.attr("STATUS_OPEN") = static_cast<int>(ee::Status::Open);
    m.attr("STATUS_PARTIAL") = static_cast<int>(ee::Status::Partial);
    m.attr("STATUS_FILLED") = static_cast<int>(ee::Status::Filled);
    py::class_<Book>(m, "Book")
        .def(py::init<std::uint32_t, std::size_t, bool>(), py::arg("window_ticks") = 1u << 18,
             py::arg("order_capacity") = 4096, py::arg("prefault") = false)
        .def("add_limit", &Book::add_limit, py::arg("order_id"), py::arg("side"), py::arg("price"), py::arg("qty"))
        .def("add_limit_count", &Book::add_limit_count, py::arg("order_id"), py::arg("side"), py::arg("price"), py::arg("qty"))
        .def("match_market", &Book::match_market, py::arg("side"), py::arg("qty"))
        .def("cancel", &Book::cancel, py::arg("order_id"))
        .def("best_bid", &Book::best_bid)
        .def("best_ask", &Book::best_ask)
        .def("depth", &Book::depth, py::arg("side"), py::arg("n"))
        .def("contains", &Book::contains, py::arg("order_id"))
        .def("far_levels", &Book::far_levels, py::arg("side"))
        .def("__len__", &Book::size);
}
