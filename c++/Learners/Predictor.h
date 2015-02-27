/*
Authors: Kenneth Tran <one@kentran.net>
License: BSD 3 clause
 */

#pragma once

#include "../Core/Vector.h"

namespace MLx {
    template<class TOut>
    class PredictorBase {
    public:
        const size_t Dimension;
        virtual TOut Predict(const Vector &features) const = 0;
        virtual TOut Predict(const FloatVec &features) const = 0;
    protected:
        PredictorBase(int dimension) : Dimension(dimension) {}

    };

    class RegressorBase : public PredictorBase<float> {
    protected:
        RegressorBase(int dimension) : PredictorBase(dimension) {}
    };

    class BinaryClassifierBase : public PredictorBase<float> {
    public:
        virtual float PredictProbability(const Vector &features, float &score) const = 0;
    protected:
        BinaryClassifierBase(int dimension) : PredictorBase(dimension) {}
    };
}
