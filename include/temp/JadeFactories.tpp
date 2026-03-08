#include "header/Jade.hpp"
#include <random>
namespace bm{
    template<typename T>
    void Jade::set_seed(T s) {
        if constexpr (std::is_same_v<T, std::nullptr_t>) {
            globalSeed = std::nullopt;
        }
        else {
            globalSeed = static_cast<uint64_t>(s);
        }
    }

    template<typename T>
    void Jade::seed(T s){
        if constexpr (std::is_same_v<T, std::nullptr_t>) {
            localSeed = std::nullopt;
        }
        else {
            localSeed = static_cast<uint64_t>(s);
        }
    }

    inline uint64_t Jade::getSeed() {
        if (localSeed.has_value())
            return localSeed.value();
        if (globalSeed.has_value())
            return globalSeed.value();
        std::random_device rd;
        return rd();

    }

    template<typename... Dims>
    Jade::ArrayBuilder Jade::array(DType dType, const Dims... dims) {
        uint64_t nelm = 1;
        ((nelm *= dims), ...);
        return ArrayBuilder(dType, {static_cast<uint64_t>(dims)...}, nelm);
    }

    template<typename... Dims>
    Jade Jade::zeros(DType dType, const Dims... dims) {
        return Jade(dType, 0.0, dims...);
    }

    template<typename... Dims>
    Jade Jade::ones(DType dType, const Dims... dims) {
        return Jade(dType, 1.0, dims...);
    }

    template<typename... Dims>
    Jade Jade::randn(DType dType, const Dims... dims) {
        uint64_t shape[] = {static_cast<uint64_t>(dims)...};
        Jade out(dType, 0.0, shape, sizeof...(dims));

        std::mt19937 gen(getSeed());
        std::normal_distribution<double> dist(0.0, 1.0);

        auto* ptr = static_cast<double*>(out.data_ptr());
        for(uint64_t i = 0; i < out.get_size(); ++i) {
            ptr[i] = dist(gen);
        }
        return out;
    }

    template<typename... Dims>
    Jade Jade::rand(DType dType, const Dims... dims) {
        uint64_t shape[] = {static_cast<uint64_t>(dims)...};
        Jade out(dType, 0.0, shape, sizeof...(dims));

        std::mt19937 gen(getSeed());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        auto * ptr = static_cast<double*>(out.data_ptr());
        for(uint64_t i = 0; i < out.get_size(); ++i) {
            ptr[i] = dist(gen);
        }
        return out;
    }

    template<typename... Dims>
    Jade Jade::randint(DType dType, double lo, double hi, const Dims... dims) {
        uint64_t shape[] = {static_cast<uint64_t>(dims)...};
        Jade out(dType, 0.0, shape, sizeof...(dims));
        if(lo > hi) std::swap(hi, lo);
        std::mt19937 gen(getSeed());
        std::uniform_real_distribution<double> dist(lo, hi);

        auto* ptr = static_cast<double*>(out.data_ptr());
        for(uint64_t i = 0; i < out.get_size(); ++i) {
            ptr[i] = dist(gen);
        }
        return out;
    }



}