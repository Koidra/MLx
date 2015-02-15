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

    class Examples {
    public:
        bool IsSparse();
        DataSchema* GetSchema();
        ExampleIterator begin() const;
        ExampleIterator end() const;

    protected:
        REF<DataSchema> schema_;
        bool isSparse_;
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

    class ShuffleExamples : public Examples {
    public:
        virtual size_t Size() = 0;
    };

    class InMemoryExamples final : public ShuffleExamples {
    public:
        InMemoryExamples(REF<DataSchema> schema, bool isSparse);
        InMemoryExamples(REF<DataSchema> schema, bool isSparse, std::vector<Example> &data);
        size_t  Size() override;
        void Add(UREF<Example> example); //this takes ownership of Example embedded in the parameter
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
        explicit operator InMemoryExamples();
    protected:
        std::vector<Example> cache_;
        class StreamingLoaderState : public ExamplesReadState {
        public:
            void Cache(std::vector<Example> &cache);
        protected:
            UREF<Example> current_;
        };
    };
}