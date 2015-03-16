#pragma once

#include "../Core/Vector.h"

namespace MLx
{
    // Example is a numeric data row that can be used directly by learners
    class Example final {
    public:
        size_t Id;
        const UREF<Vector> Features;
        const float Label;
        const float Weight;
        const UREF<string> Name;

        Example(Vector* features, float label, float weight = 1, UREF<string> name = nullptr);
    };
}