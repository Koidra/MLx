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
    class ExampleIterator;
    class ExamplesReadState;
    class InMemoryExamples;

    class Examples {
    friend class InMemoryExamples;
    public:
        bool IsSparse();
        DataSchema* GetSchema();
        ExampleIterator begin() const;
        ExampleIterator end() const;

    protected:
        bool isSparse_;
        UREF<DataSchema> schema_;
        UREF<ExamplesReadState> state_;
    };

    struct ExampleIterator {
        ExampleIterator(ExamplesReadState *state);
        const Example& operator*() const;
        const Example* operator->() const;
        ExampleIterator operator++();
        bool operator!=(const ExampleIterator& other);
    private:
        ExamplesReadState *state_;
    };

    class ExamplesReadState {
    public:
        virtual void Reset() = 0;
        virtual bool MoveNext() = 0;
        virtual const Example* Current() const = 0;
    };

    class InMemoryExamples final : public Examples {
    public:
        //This ctor takes ownership of the data
        InMemoryExamples(Examples &src);
        bool IsEmpty();
        size_t  Size();
    private:
        std::vector<Example> data_;
        class State final : public ExamplesReadState {
        public:
            State(std::vector<Example> &data);
            void Reset() override;
            bool MoveNext() override;
            const Example* Current() const override;
            const std::vector<Example>* data_;
            std::vector<Example>::const_iterator iterator_;
        };
    };

    class StreamingExamples : public Examples {
    public:
        REF<InMemoryExamples> Cache();
    protected:
        REF<InMemoryExamples> cached_;
    };
}