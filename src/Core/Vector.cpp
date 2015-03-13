#include <smmintrin.h>
#include "cblas.h"
#include "Vector.h"

//ToDo: make loading more efficient
#define LOAD4(pi,ps) _mm_setr_ps(ps[pi[0]], ps[pi[1]], ps[pi[2]], ps[pi[3]])

namespace MLx {
    using namespace std;

    Vector::Vector(size_t length) : Length(length) {}

    DenseVector::DenseVector(Vec &values) : Vector::Vector(values.size()), values_(move(values)) {}

    float DenseVector::Dot(const DenseVector& rhs) {
        return cblas_sdot(Length, &values_[0], 1, &rhs.values_[0], 1);
    }

    SparseVector::SparseVector(size_t length, IntVec &indices, Vec &values)
            : Vector::Vector(length), indices_(move(indices)), values_(move(values)) {}

    //For sparse BLAS, we don't know any good free library so we just implement using SSE instructions
    float SparseVector::Dot(const DenseVector& rhs) {
        const int* piLim = &*indices_.end(); // upper bound for pi

        // Process 4 non-zero elements at a time
        const int* pi = indices_.data();
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
            sum = (*pv) * ps[*pi];

        return sum;
    }
}