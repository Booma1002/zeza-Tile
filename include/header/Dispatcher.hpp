#pragma once
#include "Registry.hpp"
#include "Vein.hpp"
namespace bm {
////////////////////////////////////////////////////////////////////////
/////////////////***************************************////////////////
/////////////////**  Dispatcher Class Initialization  **////////////////
/////////////////***************************************////////////////
////////////////////////////////////////////////////////////////////////
    ;

    class Jade;

    struct Dispatcher {
/////////////////////////////////////////////////////////////
/////////////////****************************////////////////
/////////////////**  Dispatcher Executors  **////////////////
/////////////////****************************////////////////
/////////////////////////////////////////////////////////////
        /**
         *
         * @param op
         * @param out
         * @param a
         * @param b
         */

        template<typename... Args>
        static void execute_binary(OpCode op, Jade &out, const Jade &a, const Jade &b, Args&... args);

        /**
         *
         * @param op
         * @param out
         * @param a
         */

        template<typename... Args>
        static void execute_unary(OpCode op, Jade &out, const Jade &a, Args&... args);


        template<typename... Args>
        static void execute_scalar(OpCode op, Jade &out, double a, Args&... args);

        template<typename... Args>
        static void execute_reduction_unary(OpCode op, Jade& out, const Jade& a, Args&... args);

        template<typename... Args>
        static void execute_reduction_binary(OpCode op, Jade& out, const Jade& a, const Jade& b, Args&... args);

        template<typename... Args>
        static void execute_backward_binary(OpCode fwd_op, Jade a, Jade b, Vein* out_vein);

        template<typename... Args>
        static void execute_backward_unary(OpCode fwd_op, Jade a, Vein* out_vein);

        template<typename... Args>
        static void execute_backward_reduction_unary(OpCode fwd_op, Jade a, Vein* out_vein);

        template<typename... Args>
        static void execute_backward_reduction_binary(OpCode fwd_op, Jade a, Jade b, Vein* out_vein);
    };

}// namespace bm

#include "temp/Dispatcher.tpp"
