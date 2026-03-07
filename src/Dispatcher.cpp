#include "header/Dispatcher.hpp"
using namespace bm;

void Dispatcher::execute_binary(OpCode op, Jade& out, const Jade& a, const Jade& b) {
    // TODO: check if A, B are on the same device
    Device target_device = Device::CPU;
    JadeReactor opr;
    if (op == OpCode::MATMUL) opr = JadeReactor::operate_matmul(out, a, b);
    else opr = JadeReactor::operate_binary(out, a, b);
    Kernel kernel_func = Registry::get().lookup(op, target_device);
    kernel_func(opr);
    std::string msg = std::format("[Dispatcher] Executed Binary Operation: OpCode={:#x}" ,static_cast<int>(op));
    LOG_DEBUG(msg);
}

void Dispatcher::execute_unary(OpCode op, Jade& out, const Jade& a, const double left, const double right) {
    Device target_device = Device::CPU;
    JadeReactor opr = JadeReactor::operate_unary(out, a, left, right);
    Kernel kernel_func = Registry::get().lookup(op, target_device);
    kernel_func(opr);
    std::string msg = std::format("[Dispatcher] Executed Unary Operation: OpCode={:#x}" ,static_cast<int>(op));
    LOG_DEBUG(msg);
}

void Dispatcher::execute_scalar(OpCode op, Jade& out, const double a) {
    Device target_device = Device::CPU;
    JadeReactor opr = JadeReactor::operate_scalar(out, a);
    Kernel kernel_func = Registry::get().lookup(op, target_device);
    kernel_func(opr);
    std::string msg = std::format("[Dispatcher] Executed Scalar Operation: OpCode={:#x}" ,static_cast<int>(op));
    LOG_DEBUG(msg);
}

