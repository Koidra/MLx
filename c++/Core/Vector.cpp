/*
Authors: Kenneth Tran <one@kentran.net>
License: BSD 3 clause
 */

#include <numeric>
#include <smmintrin.h>
#include <Python/Python.h>
#include "cblas.h"
#include "Vector.h"

//ToDo: make loading more efficient
#define LOAD4(pi,ps) _mm_setr_ps(ps[pi[0]], ps[pi[1]], ps[pi[2]], ps[pi[3]])

namespace MLx {
    using namespace std;
    using namespace Contracts;

    Vector::Vector(size_t length) : Length(length) {}

    DenseVector::DenseVector(FloatVec &values)
            : Vector::Vector(values.size()), values_(move(values)) {}

    EnumeratePair DenseVector::AsEnumeratePair() const {
        if (indices_.empty()) {
            indices_.resize(Length);
            iota(indices_.begin(), indices_.end(), 0);
        }
        assert(indices_.size() == Length && values_.size() == Length);
        return EnumeratePair(indices_, values_);
    }

    float DenseVector::Dot(const DenseVector& rhs) {
        return cblas_sdot(Length, &values_[0], 1, &rhs.values_[0], 1);
    }

    void CheckIndices(IntVec &indices, int dimension) {
        int lastIndex = -1;
        for (int i : indices) {
            CheckArg(lastIndex < i && i < dimension, "Sparse index out of range");
            lastIndex = i;
        }
    }

    SparseVector::SparseVector(size_t length, IntVec &indices, FloatVec &values)
            : Vector::Vector(length), indices_(move(indices)), values_(move(values))
    {
        CheckIndices(indices_, length);
    }

    EnumeratePair SparseVector::AsEnumeratePair() const {
        return EnumeratePair(indices_, values_);
    }

    //For sparse BLAS, we don't know any good free library so we just implement using SSE instructions
    float SparseVector::Dot(const DenseVector& rhs) {
        throw domain_error("Implement aligned vector allocator");
        const size_t* piLim = &*indices_.end(); // upper bound for pi

        // Process 4 non-zero elements at a time
        const size_t* pi = indices_.data();
        const float* pv = values_.data();
        const float* ps = rhs.values_.data();
        __m128 res;
        for (; pi <= piLim - 4; pi += 4, pv += 4)
        {
            __m128 x = _mm_load_ps(pv);
            __m128 y = _mm_setr_ps(ps[pi[0]], ps[pi[1]], ps[pi[2]], ps[pi[3]]);
            res = _mm_add_ps(res, _mm_mul_ps(x, y));
        }
        res = _mm_hadd_ps(res, res);
        float sum = _mm_cvtss_f32(_mm_hadd_ps(res, res));


        for (; pi < piLim; ++pi, ++pv) // process the remaining elements, one by one
            sum += (*pv) * ps[*pi];

        return sum;
    }

    BinaryVector::BinaryVector(size_t length, IntVec &indices)
            : Vector(length), indices_(move(indices))
    {
        CheckIndices(indices_, length);
    }

    EnumeratePair BinaryVector::AsEnumeratePair() const {
        if (values_.empty())
            values_ = FloatVec(values_.size(), 1);
        return EnumeratePair(indices_, values_);
    }

    float BinaryVector::Dot(const DenseVector& rhs) {
        const size_t* piLim = &*indices_.end(); // upper bound for pi

        // Process 4 non-zero elements at a time
        const size_t* pi = indices_.data();
        const float* ps = rhs.values_.data();
        __m128 res;
        for (; pi <= piLim - 4; pi += 4)
        {
            __m128 x = _mm_setr_ps(ps[pi[0]], ps[pi[1]], ps[pi[2]], ps[pi[3]]);
            res = _mm_add_ps(res, x);
        }
        res = _mm_hadd_ps(res, res);
        float sum = _mm_cvtss_f32(_mm_hadd_ps(res, res));

        for (; pi < piLim; ++pi) // process the remaining elements, one by one
            sum += ps[*pi];

        return sum;
    }
}