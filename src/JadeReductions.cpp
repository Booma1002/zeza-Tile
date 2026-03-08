#include "header/Jade.hpp"
using namespace bm;


Jade Jade::std(const Jade& input){
    Jade view(input.dtype, 0.0f, input.shape.get(), input.ndims);
    Dispatcher::execute_unary(OpCode::STD, view, input);
    return view;
}
Jade Jade::var(const Jade& input){
    Jade view(input.dtype, 0.0f, input.shape.get(), input.ndims);
    Dispatcher::execute_unary(OpCode::VAR, view, input);
    return view;
}
Jade Jade::max(const Jade& input) {
    Jade output(input.dtype, 0.0, 1ULL);
    Dispatcher::execute_unary(OpCode::MAX, output, input);
    return output;
}

Jade Jade::min(const Jade& input) {
    Jade output(input.dtype, 0.0, 1ULL);
    Dispatcher::execute_unary(OpCode::MIN, output, input);
    return output;
}

Jade Jade::mean(const Jade& input) {
    Jade output(input.dtype, 0.0, 1ULL);
    Dispatcher::execute_unary(OpCode::MEAN, output, input);
    return output;
}

Jade Jade::dot(const Jade& other) const {
    Jade output(this->dtype, 0.0, 1ULL);
    Dispatcher::execute_binary(OpCode::DOT, output, *this, other);
    return output;
}

Jade Jade::argmax(const Jade& input){
    Jade view(input.dtype, 0.0f, input.shape.get(), input.ndims);
    Dispatcher::execute_variadic(OpCode::ARGMAX, view, input);
    return view;
}
Jade Jade::argmin(const Jade& input){
    Jade view(input.dtype, 0.0f, input.shape.get(), input.ndims);
    Dispatcher::execute_variadic(OpCode::ARGMIN, view, input);
    return view;
}

