/*
Authors: Kenneth Tran <one@kentran.net>
License: BSD 3 clause
 */

#pragma once

#include "../Core/Vector.h"

namespace MLx
{
    //Example is a numeric data row that can be used directly by learners
    class Example final {
        UREF<Vector> features_;
        UREF<string> name_;
    public:
        size_t Id; //ToDo
        const float Label;
        const float Weight;

        Example(Vector* features, float label, float weight = 1, UREF<string> name = nullptr);
        const Vector& Features() const;
        const string& Name() const;
    };
}