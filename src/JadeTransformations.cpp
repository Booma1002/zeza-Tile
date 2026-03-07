#include "header/Jade.hpp"
#include "header/Dispatcher.hpp"
using namespace bm;
Jade  Jade::transpose() {
    Jade newJade(*this);
    array arr = std::make_unique<uint64_t[]>(ndims);
    array strd = std::make_unique<uint64_t[]>(ndims);
    std::memcpy(arr.get(), shape.get(), ndims * sizeof(uint64_t));
    std::memcpy(strd.get(), strides.get(), ndims * sizeof(uint64_t));
    reverse(arr.get(), ndims);
    reverse(strd.get(), ndims);
    std::memcpy(newJade.shape.get(), arr.get(), ndims * sizeof(uint64_t));
    std::memcpy(newJade.strides.get(), strd.get(), ndims * sizeof(uint64_t));
//        std::cout << "Transposed " << this->repr() <<" Into " << newJade.repr() << std::endl;
    return newJade;
}

Jade Jade::zeros_like(const Jade& other) {
    Jade output(other.dtype, 0.0f, other.shape.get(), other.ndims);
    return output;
}

Jade Jade::fill_like(const Jade& other, const double val){
    Jade output(other.dtype, val, other.shape.get(), other.ndims);
    return output;
}

Jade Jade::pad(double fill_val, const uint64_t* pads) const {
    // TODO : UNFINISHED;
    // TODO : NEED TESTING;
    auto new_shape = std::make_unique<uint64_t[]>(ndims);
    for(size_t i=0; i<ndims; ++i)
        new_shape[i] = shape[i] + pads[i*2] + pads[i*2+1];
    Jade output(this->dtype, fill_val, new_shape.get(), ndims);
    Jade view (output);
    for(size_t i=0; i<ndims; ++i) {
        view.offset += pads[i*2] * output.strides[i];
        view.shape[i] = this-> shape[i];
    }

    view.copy_from(*this);
    return output;
}

void Jade::copy_from(const Jade& other) {
    Dispatcher::execute_unary(OpCode::COPY, *this, other);
}

Jade Jade::copy(){
    Jade view = Jade(*this);
    Dispatcher::execute_unary(OpCode::COPY, view, *this);
    return view;
}


Jade& Jade::flatten(){};



