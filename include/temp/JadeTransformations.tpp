#include "header/Jade.hpp"
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
}