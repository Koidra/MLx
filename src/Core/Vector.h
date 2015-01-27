#pragma once

#include "Commons.h"

namespace MLx {
    class DenseVector;
    class SparseVector;

    class Vector {
    public:
        size_t Length; // read-only
        virtual float Dot(const DenseVector& rhs) = 0;

    protected:
        Vector(size_t length);
    };

    class DenseVector final : public Vector {
    friend class SparseVector;

    public:
        DenseVector(Vec& values);
        float Dot(const DenseVector& rhs) override;

    private:
        Vec values_;
    };

    class SparseVector final : public Vector {
    public:
        SparseVector(size_t length, IntVec& indices, Vec& values);
        float Dot(const DenseVector& rhs) override;

    private:
        IntVec indices_;
        Vec values_;

    };
}