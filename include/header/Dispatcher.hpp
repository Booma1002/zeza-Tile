#pragma once
#include "Registry.hpp"
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
        static void execute_binary(OpCode op, Jade &out, const Jade &a, const Jade &b);

        /**
         *
         * @param op
         * @param out
         * @param a
         */
        static void
        execute_unary(OpCode op, Jade &out, const Jade &a, const double left = 0.f, const double right = 0.f);

        static void execute_scalar(OpCode op, Jade &out, double a);
    };

}// namespace bm

#include "temp/Dispatcher.tpp"
