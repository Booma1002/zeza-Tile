#include "header/Engine.hpp"
using namespace bm;

std::atomic_uint64_t watching;
void see(Jade& t, std::string msg =""){
    std::cout << std::format("\n\n{:x}) {}:-\n",
             watching.load(std::memory_order::relaxed), msg);
    watching.fetch_add(1, std::memory_order_relaxed);
    t.display(2, 6);
}

int main(){
    LOG_INFO("[Engine] Initiating Bare Metal (BM) Ignition Sequence...");
    watching.store(0, std::memory_order_relaxed);
    Jade::set_seed(42);
    bm::Jade Ar = Jade::arange(DType::FLOAT64, Slice(111, 10, -5));
    see(Ar, "arange");
    auto a = Jade::randint(DType::FLOAT64, -3, 0,   4,4);
    see(a, "a");
    auto b = Jade::rand(DType::FLOAT64, 4,4);
    see(b, "b");
    auto c = Jade::randn(DType::FLOAT64, 4,4);
    see(c, "c");
    c.seed(15);
    c = Jade::array(DType::FLOAT64, 2,4)
            = {1, 2, 3, 4,
               4, 3, 2, 1};
    see(c, "d");
}