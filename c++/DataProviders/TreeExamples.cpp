/*
Authors: Kenneth Tran <one@kentran.net>
License: BSD 3 clause
 */

#include "TreeExamples.h"

namespace MLx {
    using namespace std;
    using namespace Contracts;

    class TreeExamples::NoState final : public Examples::State {
        void FailNotImplemented() {
            Fail<domain_error>("TreeExamples doesn't support iterating over rows.");
        }
    public:
        void Reset() override {
            FailNotImplemented();
        }
        bool MoveNext() override {
            FailNotImplemented();
        }
        const Example* Current() const override {
            FailNotImplemented();
        }
    };

    TreeExamples::TreeExamples(const string &filename) {
        //ToDo: read data
        // - Initialize schema_
        // - Initialize data_
        state_ = UREF<RowState>(new RowState());
    }

    TreeExamples::TreeExamples(const Examples &other) {
        schema_ = other.schema_;
        auto otherAsTreeExamples = dynamic_cast<TreeExamples*>(&other);
        if (otherAsTreeExamples != nullptr) {
            features_ = move(otherAsTreeExamples->features_);
            labels_ = move(otherAsTreeExamples->labels_);
            weights_ = move(otherAsTreeExamples->weights_);
        }
        else {
            cout << "Converting data format from row-based data to column-based ... ";
            size_t dimension = schema_->Dimension();
            positiveCounts_.resize(dimension);
            positiveSums_.resize(dimension);
            negativeCounts_.resize(dimension);
            negativeSums_.resize(dimension);
            vector<IntVec> featureColumns(dimension);

            size_t n = 0;
            for (Example& example : other) {
                float label = example.Label;
                labels_.push_back(example.Label);
                float weight = example.Weight;
                count_ += weight;
                weights_.push_back(example.Weight);
                double weightedLabel = weight * label;
                sum_ += weightedLabel;

                EnumeratePair ivPair = example.Features().AsEnumeratePair();
                IntVec& indices = ivPair.first;
                FloatVec& values = ivPair.second;
                for (int i = 0; i < indices.size(); ++i) {
                    CheckArg(values[i] == 1, "The source dataset contains non-binary features.");
                    size_t c = indices[i];
                    featureColumns[c].Add(n);
                    positiveCounts_[c] += weight;
                    positiveSums_[c] += weightedLabel;
                }
                ++n;
            }

            featureColumns_.resize(dimension);
            for (int c = 0; c < dimension; ++c)
                featureColumns_[c] = BinaryVector(n, featureColumns[c]);
            cout << "done" << endl;
        }
        state_ = UREF<NoState>(new NoState());
    }

    const vector<BinaryVector>& TreeExamples::FeatureColumns() const {
        return featureColumns_;
    }

    const FloatVec& TreeExamples::Labels() const {
        return labels_;
    }

    const FloatVec& TreeExamples::Weights() const {
        return weights_;
    }

    const Histograms& TreeExamples::Histograms() const {
        return histograms_;
    }

    //ToDo: implement and serialize header
    void TreeExamples::Save(string& filename) const {
        char* buffer = new char[schema_->Dimension()]
        unsigned long long a[99999];
        FILE* pFile = fopen(filename.c_str(), "wb");
        for (unsigned long long j = 0; j < 1024; ++j) {
            //Some calculations to fill a[]
            fwrite(a, 1, size*sizeof(unsigned long long), pFile);
        }
        fclose(pFile);
    }
}