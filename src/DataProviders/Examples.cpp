#include "Examples.h"

namespace MLx {
    using namespace std;
    using namespace Contracts;
    using namespace Utils;

    DataSchema::DataSchema(StrVec& featureNames) : FeatureNames(move(featureNames)) {
    }

    size_t DataSchema::GetDimension() {
        return (FeatureNames).size();
    }

    bool Examples::IsSparse() {
        return isSparse_;
    }

    REF<DataSchema> Examples::GetSchema() {
        return schema_;
    }

    Example *ExamplesIterator::operator->() {
        return operator*();
    }
}