#include "header/Jade.hpp"
#include "header/Dispatcher.hpp"
using namespace bm;
std::optional<uint64_t> Jade::globalSeed = std::nullopt;
std::optional<uint64_t> Jade::localSeed = std::nullopt;

Jade Jade::ArrayBuilder::operator=(std::initializer_list<double> data) {
    if (data.size() != nelm) {
        std::string msg = std::format("[ArrayBuilder] Brace mismatch:"
                                      " Expected {:x}, Got {:x}.", nelm, data.size());
        LOG_ERR(msg);
        throw std::invalid_argument(msg);
    }
    Jade out(dtype, 0.0, shape.data(), shape.size());
    auto* ptr = static_cast<double*>(out.data_ptr());
    uint64_t i = 0;
    for (auto& val : data) ptr[i++] = val;
    return out;
}

Jade Jade::arange(DType dtype, Slice range) {
    long long start = range.start;
    long long stop = range.stop;
    long long step = range.step;

    if (step == 0) {
        std::string msg = "[JadeFactory] Arange step cannot be zero.";
        LOG_ERR(msg);
        throw std::invalid_argument(msg);
    }
    uint64_t len = 0;
    if (step > 0 && stop > start) len = (stop - start + step - 1) / step;
    else if (step < 0 && stop < start) len = (start - stop - step - 1) / -step;

    Jade output(dtype, 0.0, len);
    if (len == 0) return output;
    auto arg_start = static_cast<double>(start);
    auto arg_step  = static_cast<double>(step);

    Dispatcher::execute_unary(OpCode::ARANGE, output, output, arg_start, arg_step);

    return output;
}


Jade Jade::zeros_like(const Jade& other) {
    Jade output(other.dtype, 0.0f, other.shape.get(), other.ndims);
    return output;
}

Jade Jade::ones_like(const Jade& other) {
    Jade output(other.dtype, 1.0f, other.shape.get(), other.ndims);
    return output;
}

Jade Jade::full_like(const Jade& other, const double val){
    Jade output(other.dtype, val, other.shape.get(), other.ndims);
    return output;
}
