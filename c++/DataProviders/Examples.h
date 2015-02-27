/*
Authors: Kenneth Tran <one@kentran.net>
License: BSD 3 clause
 */

#pragma once
#include "Example.h"

namespace MLx {
    class DataSchema {
    public:
        StrVec FeatureNames;
        DataSchema(StrVec& featureNames);
        size_t Dimension() const;
    };

    class Examples {
    protected:
        //The main reason we don't combine Examples and its State into one is to satisfy that begin() is const
        class State {
        public:
            virtual ~State() {}
            virtual void Reset() = 0;
            virtual bool MoveNext() = 0;
            virtual const Example* Current() const = 0;
        };

        REF<DataSchema> schema_;
        bool isSparse_;
        UREF<State> state_;

    public:
        struct Iterator {
            Iterator(const State *state);
            const Example& operator*() const;
            const Example* operator->() const;
            Iterator operator++();
            bool operator!=(const Iterator & other);
        private:
            State *state_;
        };

        bool IsSparse();
        const DataSchema* Schema() const;
        Iterator begin() const;
        Iterator end() const;
        virtual void Serialize(const string& filename) const = 0;
    };

    class RandomAccessExamples : public Examples {
    public:
        virtual size_t Size() = 0;
    };

    class InMemoryExamples : public RandomAccessExamples {
    public:
        InMemoryExamples(REF<DataSchema> schema, bool isSparse, std::vector<Example> &data);
        size_t  Size() override;
        void Add(UREF<Example> example); //this takes ownership of Example embedded in the parameter
    protected:
        std::vector<Example> data_;
        class State;
    };

    class StreamingExamples : public Examples {
    public:
        explicit operator InMemoryExamples();
    protected:
        class State : public Examples::State {
        public:
            std::vector<Example>& Cache();
            const Example* Current() const override;
            virtual ~State();
        protected:
            Example* current_;
            std::vector<Example> cache_;
            State() {}
            State(State &src) = delete;
        };
    };
}