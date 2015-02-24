#pragma once

#include "Commons.h"

namespace MLx {
    class DenseVector;
    class SparseVector;

    typedef std::pair<IntVec&,FloatVec&> EnumeratePair;

    class Vector {
    public:
        size_t Length; // read-only
        virtual EnumeratePair AsEnumeratePair() const = 0;
        virtual float Dot(const DenseVector& rhs) = 0;

    protected:
        IntVec indices_;
        FloatVec values_;
        Vector(size_t length);
    };

    class DenseVector final : public Vector {
    public:
        DenseVector(FloatVec & values);
        EnumeratePair AsEnumeratePair() const override;
        float Dot(const DenseVector& rhs) override;
    };

    class SparseVector final : public Vector {
    public:
        SparseVector(size_t length, IntVec& indices, FloatVec & values);
        EnumeratePair AsEnumeratePair() const override;
        float Dot(const DenseVector& rhs) override;
    };

    class BinaryVector final : public Vector {
    public:
        BinaryVector(size_t length, IntVec &indices);
        EnumeratePair AsEnumeratePair() const override;
        float Dot(const DenseVector& rhs) override;
    };
}