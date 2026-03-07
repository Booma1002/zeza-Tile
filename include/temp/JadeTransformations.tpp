#include "header/Jade.hpp"
#include "header/Dispatcher.hpp"
namespace bm {
    template<typename... Dims>
    Jade &Jade::reshape(Dims... dims) {
        uint64_t sz = 1;
        ((sz *= dims), ...); // check size match
        uint64_t n = sizeof...(Dims);
        if (get_size() != sz) {
            LOG_ERR("[Jade] Cannot reshape Jade into the given dims.");
            throw ShapeMismatchException("Cannot reshape Jade into the given dims.");
        }
        std::string repr1 = repr();
        ndims = n;
        init_metadata(dims...); // initialize jade
        std::string msg;
        msg+= std::format("Reshaped Jade from {} Into {}", repr1 ,repr());
        LOG_INFO(msg);
        return *this;
    }


    template<typename... Dims>
    Jade Jade::zeros(DType dType, const Dims... dims){
        uint64_t shape_array[] = {static_cast<uint64_t>(dims)...};
        Jade output(dtype, 0.0, shape_array, sizeof...(dims));
        return output;
    }

    template<typename... Dims>
    Jade Jade::ones(DType dType, const Dims... dims){
        uint64_t shape_array[] = {static_cast<uint64_t>(dims)...};
        Jade output(dtype, 1.0, shape_array, sizeof...(dims));
        return output;
    }

    Jade Jade::arange(DType dType, Slice range){
        int n=0;
        long long l = range.start;
        long long r = range.stop;
        long long jmp = range.step;
        uint64_t size = 1;
        if (jmp > 0) {
            if (l < 0) l += size;
            if (r < 0) r += size;
            if (l < 0) l = 0;
            if (r > static_cast<long long>(size)) r = size;
            if (l > r) l = r;
        } else {
            std::string msg = "Negative stride arrays not yet fully implemented :)";
            LOG_ERR(msg);
            throw MemoryException(msg);
        }
        uint64_t len = 0;
        if (jmp > 0) len = (r - l + jmp - 1) / jmp;
        Jade output(dtype, 0.0f, len, 1);
        Jade input = output;
        Dispatcher::execute_unary(OpCode::ARANGE, output, input ,range);
        return output;
    }

    template<typename... Dims>
    Jade Jade::array(DType dType, const Dims... dims){

    }

    template<typename... Dims>
    Jade Jade::rand(DType dType, const Dims... dims){

    }

    template<typename... Dims>
    Jade Jade::randn(DType dType, const Dims... dims){

    }

    template<typename... Dims>
    Jade Jade::randint(DType dType, const Dims... dims){

    }
}