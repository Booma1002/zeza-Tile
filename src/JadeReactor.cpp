#include "header/JadeReactor.hpp"
using namespace bm;

void JadeReactor::merge_dims() {
    if (ndims <= 1) return;
    for (int cur = ndims - 1; cur > 0; --cur) {
        int mother = cur - 1;
        bool can_do_collapse = true;
        for (int _ = 0; _ < RE_MAX_REACTANTS; ++_) {
            if (strides[_][mother] != shape[cur] * strides[_][cur]) {
                can_do_collapse = false;
                break;
            }
        }
        if (can_do_collapse) {
            // mother copies current metadata:
            shape[mother] *= shape[cur];
            for (int _ = 0; _ < RE_MAX_REACTANTS; ++_)
                strides[_][mother] = strides[_][cur];

            // current tracker copies its daughter metadata
            for (int inward = cur; inward < ndims - 1; ++inward) {
                int daughter = inward + 1;

                shape[inward] = shape[daughter];
                for (int _ = 0; _ < RE_MAX_REACTANTS; ++_)
                    strides[_][inward] = strides[_][daughter];
            }
            ndims--;
        }
    }
}



