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

    DataSchema* Examples::GetSchema() {
        return &*schema_;
    }

    ExampleIterator Examples::begin() const {
        state_->Reset();
        return ExampleIterator(&*state_);
    }

    ExampleIterator NullIter = ExampleIterator(nullptr);
    ExampleIterator Examples::end() const {
        return NullIter;
    }

    ExampleIterator::ExampleIterator(ExamplesReadState *state) : state_(state) {}

    const Example& ExampleIterator::operator*() const {
        return *(state_->Current());
    }

    const Example* ExampleIterator::operator->() const {
        return state_->Current();
    }

    ExampleIterator ExampleIterator::operator++() {
        return  state_->MoveNext() ? ExampleIterator(state_) : NullIter;
    }

    bool ExampleIterator::operator!=(const ExampleIterator &other) {
        return other.state_ != state_;
    }

    InMemoryExamples::InMemoryExamples(REF<DataSchema> schema, bool isSparse)
    {
        schema_ = schema;
        isSparse_ = isSparse;
        state_ = UREF<ExamplesReadState>(new InMemoryExamples::State(data_));
    }

    InMemoryExamples::InMemoryExamples(REF<DataSchema> schema, bool isSparse, std::vector<Example> &data)
            : InMemoryExamples(schema, isSparse)
    {
        data_ = move(data);
    }

    size_t InMemoryExamples::Size() {
        return data_.size();
    }

    void InMemoryExamples::Add(UREF<Example> example) {
        data_.push_back(move(*example));
    }

    InMemoryExamples::State::State(std::vector<Example> &data)
            : data_(&data), iterator_(data.begin()) {}

    void InMemoryExamples::State::Reset() {
        iterator_ = data_->begin();
    }

    bool InMemoryExamples::State::MoveNext() {
        return ++iterator_ != data_->end();
    }

    const Example* InMemoryExamples::State::Current() const {
        return &(*iterator_);
    }

    StreamingExamples::operator InMemoryExamples() {
        if (cache_.empty())
            ((StreamingExamples::StreamingLoaderState&)(*state_)).Cache(cache_);
        return  InMemoryExamples(schema_, isSparse_, cache_);
    }

    void StreamingExamples::StreamingLoaderState::Cache(vector<Example> &cache) {
        cache.clear();
        Reset();
        do{
            cache.push_back(move(*current_));
        } while (MoveNext());
    }
}