/*
Authors: Kenneth Tran <one@kentran.net>
License: BSD 3 clause
 */

#pragma once
#include "Examples.h"

namespace MLx {
    struct Histograms {
        // The following historgrams have size = number of features
        // REVIEW: double is expensive (especially when using SSE/AVX).
        // For better perf, we need to have special cases, such as:
        //  - The weight and labels are integers => no need to use double for computing histograms
        const DoubleVec PositiveCounts;
        const DoubleVec PositiveSums;
        const double Count;
        const double Sum;

        Histograms(DoubleVec& positiveCounts, DoubleVec& positiveSums, double count, double sum)
                : PositiveCounts(move(positiveCounts)), PositiveSums(move(positiveSums)), Count(count), Sum(sum) {}
    };

    ///Implementation notes: histograms computation is the bottleneck of growing a decision tree, so we want enable fast computation of histograms
    ///Which decides our data structure
    /// * Features are stored in columns so that we can compute the histograms in parallel.
    ///   It also leverages SSE for the computation of each column's histogram
    /// * Features are limited to binary, to make the code simpler and efficient. There is a constructor that takes a generic dataset,
    ///   which may include binary/numeric/categorical features, and transform to binary features.
    /// * Histograms are precomputed
    class TreeExamples final : public Examples {
    public:
        TreeExamples(const string& filename);
        TreeExamples(const Examples& other);
        const std::vector<BinaryVector>&FeatureColumns() const;
        const FloatVec& Labels() const;
        const FloatVec& Weights() const;
        const Histograms& Histograms() const;
        void Save(const string& filename) const override;

    protected:
        std::vector<BinaryVector> featureColumns_; //size(feature columns) = number of features
        FloatVec labels_; //size(labels) = number of rows
        FloatVec weights_; //size(weights) = number of rows
        Histograms histograms_;
        class NoState;
    };
}

