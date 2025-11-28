#include "DataAugmentation.h"

bool isInputShapeBHWN(const std::vector<size_t> &inputShape, const size_t N) {
    if (inputShape.size() != 4) {
        return false;
    }

    return inputShape[3] == N;
}
