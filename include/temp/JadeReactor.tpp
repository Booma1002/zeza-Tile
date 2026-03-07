#pragma once
#include "header/Jade.hpp"
namespace bm {
    template<auto MemberFunc, typename T, typename Ret, typename... Args>
    struct Binder<MemberFunc, Ret (T::*)(Args...)> {
        static void bind(JadeReactor *op, OperatorMethod id) {
            op->template create_thunk<T, Args..., MemberFunc>(id);
        }
    };

    template<auto MemberFunc, typename T, typename Ret, typename... Args>
    struct Binder<MemberFunc, Ret (T::*)(Args...) const> {
        static void bind(JadeReactor *op, OperatorMethod id) {
            op->template create_thunk<T, Args..., MemberFunc>(id);
        }
    };

    template<auto MemberFunc, typename T, typename Ret, typename... Args>
    struct Binder<MemberFunc, Ret (T::*)(Args...) noexcept> {
        static void bind(JadeReactor *op, OperatorMethod id) {
            op->template create_thunk<T, Args..., MemberFunc>(id);
        }
    };

    template<auto MemberFunc, typename T, typename Ret, typename... Args>
    struct Binder<MemberFunc, Ret (T::*)(Args...) const noexcept> {
        static void bind(JadeReactor *op, OperatorMethod id) {
            op->template create_thunk<T, Args..., MemberFunc>(id);
        }
    };


    template<typename... Args>
    void JadeReactor::call(OperatorMethod id, Args... args) {
        int idx = static_cast<int>(id);
        if (idx >= MAX_OP_METHODS || idx < 0)
            throw OperatorException("[Operator]>> Method ID out of bounds.");
        if (!methods[idx])
            throw OperatorException("[Operator]>> Call To Undefined OperatorMethod ID (Not Bound).");

        using TypedFunc = void (*)(void *, Args...);
        auto func = reinterpret_cast<TypedFunc>(methods[idx]);
        func(bound_obj, args...);

        if (bound_obj) phys[0] = static_cast<Jade *>(bound_obj)->memory->template data<float>();
    }

    template<typename T, typename... Args, auto MemberPtr>
    void JadeReactor::create_thunk(OperatorMethod id) {
        int idx = static_cast<int>(id);
        if (idx >= MAX_OP_METHODS || idx < 0)
            throw OperatorException("[Operator]>> Method ID out of bounds.");

        auto thunk = [](void *ctx, Args... args) {
            T *instance = static_cast<T *>(ctx);
            (instance->*MemberPtr)(args...);
        };
        methods[idx] = reinterpret_cast<GenericFunc>(+thunk);
    }

    constexpr bool JadeReactor::has(OperatorMethod id) const {
        return methods[static_cast<int>(id)] != nullptr;
    }

    template<typename Func>
    void JadeReactor::bind(OperatorMethod id, Func &&f) {
        int idx = static_cast<int>(id);
        if (idx >= MAX_OP_METHODS || idx < 0)
            throw OperatorException("[Operator]>> Method ID out of bounds.");

        // Safe conversion of stateless lambda to function pointer
        auto ptr = reinterpret_cast<GenericFunc>(+f);
        methods[idx] = ptr;
    }

    template<auto MemberFunc>
    void JadeReactor::bind_private(OperatorMethod id) {
        Binder<MemberFunc>::bind(this, id);
    }
}