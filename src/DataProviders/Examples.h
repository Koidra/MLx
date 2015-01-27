#pragma once
#include "Example.h"

namespace MLx {
    class DataSchema {
    public:
        StrVec FeatureNames;
        DataSchema(StrVec& featureNames);
        size_t GetDimension();
    };

    class Examples;
    class ExamplesIterator;

    class Examples {
    public:
        virtual ~Examples() {};
        virtual ExamplesIterator* begin() = 0;
        virtual ExamplesIterator* end() = 0;
        bool IsSparse();
        REF<DataSchema> GetSchema();

    protected:
        bool isSparse_;
        REF<DataSchema> schema_;
    };

    class ExamplesIterator {
    public:
        virtual ~ExamplesIterator() {};
        virtual Example* operator*() = 0;
        virtual Example* operator->() = 0;
        virtual ExamplesIterator* operator++() = 0;
    };
}